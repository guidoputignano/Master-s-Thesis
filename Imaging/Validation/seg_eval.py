#!/usr/bin/env python3
"""Ground-truth validation of the imaging pipeline: cells, nuclei and holes.

The notebooks in ``Imaging/Segmentation`` write label masks for whole-cell
territories (``Segmented/<cond>/Cell_merged_conservative``) and nuclei
(``Segmented/<cond>/Nuclei``), and binary masks for intercellular gaps
(``Segmented/<cond>/Holes*``). This module scores those outputs, or any
baseline's, against expert annotations. It reports the error on the
quantities the model is calibrated on (cell area, aspect ratio, orientation,
multinucleation, gap area fraction), not only overlap scores.

Library:  ``from seg_eval import evaluate_cells, evaluate_holes, ...``
CLI:      ``python seg_eval.py {cells,holes,nuclei,voronoi} --help``

Conventions
-----------
* Label images: 0 is background, each object a unique positive integer.
* ROI scoping (annotating a tile of a field): the annotator draws, in full,
  every cell whose centroid lies inside the ROI. A predicted object is in
  scope when its centroid lies inside the ROI. GT objects are matched against
  *all* predictions, so a GT cell is not penalised when the centroid of its
  predicted twin falls just outside the ROI.
* Matching: one-to-one assignment maximising total IoU among pairs with
  IoU >= t. For t >= 0.5 this equals the unique greedy match.
* Split/merge (majority-overlap convention): a GT cell is *split* when at
  least two predicted objects each have more than half of their area inside
  it; a predicted object is a *merge* when at least two GT cells each have
  more than half of their area inside it.
* Orientation: flow runs along the image x axis (L2R). ``axial_deg`` is the
  major-axis angle to the x axis in [0, 180); ``misalign_deg`` is the acute
  angle to the flow axis in [0, 90].
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from skimage.measure import label as connected_components
from skimage.measure import regionprops

DEFAULT_THRESHOLDS = tuple(np.round(np.arange(0.5, 0.96, 0.05), 2))

# [Pressure]_[Series]_[Date]_[Mag]_[FlowDir]_[ImagingType]_[Sequence]
# (data/previous/renaming_procedure.md), e.g. 1.4Pa_A1_20dec21_20xA_L2RA_FlatA_seq003
KEY_RE = re.compile(
    r"(\d+(?:\.\d+)?Pa(?:-\d+(?:\.\d+)?Pa)?_[^_]+_\d{2}[a-z]{3}\d{2}_[^_]+_[^_]+_[^_]+_seq\d{3})")


# ----------------------------------------------------------------------------
# File naming, units, I/O
# ----------------------------------------------------------------------------

def sample_key(name):
    """Acquisition key shared by every file derived from one field, or None."""
    m = KEY_RE.search(os.path.basename(str(name)))
    return m.group(1) if m else None


def condition_of(key):
    """'1.4Pa_A1_20dec21_20xA_L2RA_FlatA_seq003' -> '1.4Pa_A1_20x'."""
    parts = key.split('_')
    mag = re.match(r'\d+x', parts[3])
    return f"{parts[0]}_{parts[1]}_{mag.group(0) if mag else parts[3]}"


def parse_px_um(spec):
    """Pixel size lookup from '0.65' or '20x=0.65,40x=0.325'; None means pixels.

    There is deliberately no default: the repository uses three different
    x20 pixel sizes (0.325, 0.429 and 650/1024 um), so the value must come
    from the OME metadata of the acquisitions.
    """
    if spec is None:
        return None
    if '=' not in spec:
        value = float(spec)
        return lambda key: value
    table = {k.strip(): float(v) for k, v in (kv.split('=') for kv in spec.split(','))}

    def lookup(key):
        mag = condition_of(key).split('_')[-1]
        if mag not in table:
            raise KeyError(f"no pixel size given for magnification {mag!r} ({key})")
        return table[mag]
    return lookup


def read_mask(path):
    import tifffile
    a = np.squeeze(tifffile.imread(path))
    if a.ndim != 2:
        raise ValueError(f"{path}: expected a 2-D mask, got shape {a.shape}")
    return a


def read_points(path):
    """Nucleus clicks as an (N, 2) array of (row, col).

    Accepts napari's points CSV (``index,axis-0,axis-1``), a ``y,x`` CSV or
    any CSV whose first two numeric columns are (row, col).
    """
    df = pd.read_csv(path)
    for cols in (('axis-0', 'axis-1'), ('y', 'x'), ('row', 'col')):
        if all(c in df.columns for c in cols):
            return df[list(cols)].to_numpy(float)
    num = df.select_dtypes('number').drop(columns=['index'], errors='ignore')
    return num.iloc[:, :2].to_numpy(float)


# ----------------------------------------------------------------------------
# Core instance-level machinery
# ----------------------------------------------------------------------------

def overlap(gt, pred):
    """Contingency table between the non-zero labels of two label images.

    Returns ``(gt_ids, pred_ids, inter, gt_area, pred_area)`` where
    ``inter[i, j]`` is the pixel overlap of ``gt_ids[i]`` and ``pred_ids[j]``.
    """
    gt, pred = np.asarray(gt), np.asarray(pred)
    if gt.shape != pred.shape:
        raise ValueError(f"shape mismatch: gt {gt.shape} vs pred {pred.shape}")
    g_ids, g_inv = np.unique(gt, return_inverse=True)
    p_ids, p_inv = np.unique(pred, return_inverse=True)
    pair = g_inv.ravel().astype(np.int64) * len(p_ids) + p_inv.ravel()
    table = np.bincount(pair, minlength=len(g_ids) * len(p_ids)).reshape(len(g_ids), len(p_ids))
    kg, kp = g_ids != 0, p_ids != 0
    return (g_ids[kg], p_ids[kp], table[np.ix_(kg, kp)],
            table.sum(1)[kg], table.sum(0)[kp])


def iou_matrix(inter, gt_area, pred_area):
    union = gt_area[:, None] + pred_area[None, :] - inter
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(union > 0, inter / np.maximum(union, 1), 0.0)


def match(iou, thr):
    """Row/col indices of the IoU-maximising one-to-one matching with IoU >= thr."""
    if iou.size == 0:
        return np.empty(0, int), np.empty(0, int)
    w = np.where(iou >= thr, iou, 0.0)
    r, c = linear_sum_assignment(w, maximize=True)
    keep = w[r, c] > 0
    return r[keep], c[keep]


def centroids(labels, ids):
    if len(ids) == 0:
        return np.zeros((0, 2))
    return np.asarray(ndimage.center_of_mass(labels > 0, labels, ids), float).reshape(-1, 2)


def in_roi(points, roi):
    """roi = (y0, x0, h, w); points = (N, 2) array of (row, col)."""
    if roi is None:
        return np.ones(len(points), bool)
    y0, x0, h, w = roi
    y, x = points[:, 0], points[:, 1]
    return (y >= y0) & (y < y0 + h) & (x >= x0) & (x < x0 + w)


def border_ids(labels):
    edge = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    return set(np.unique(edge[edge > 0]).tolist())


def split_merge(inter, gt_area, pred_area):
    """Majority-overlap split/merge flags: (split_gt[i], merge_pred[j], n_gt_in_pred[j])."""
    pred_inside_gt = inter * 2 > pred_area[None, :]
    gt_inside_pred = inter * 2 > gt_area[:, None]
    n_gt_in_pred = gt_inside_pred.sum(0)
    return pred_inside_gt.sum(1) >= 2, n_gt_in_pred >= 2, n_gt_in_pred


def detection_counts(iou, pred_scope, thr):
    r, c = match(iou, thr)
    tp = len(r)
    fp = int(pred_scope.sum() - pred_scope[c].sum())
    fn = iou.shape[0] - tp
    return tp, fp, fn, iou[r, c]


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else float('nan')
    r = tp / (tp + fn) if tp + fn else float('nan')
    f = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float('nan')
    return p, r, f


# ----------------------------------------------------------------------------
# Morphometrics and agreement
# ----------------------------------------------------------------------------

def morphology(labels, px_um=None):
    """Per-object area, aspect ratio, orientation and circularity."""
    scale = px_um ** 2 if px_um else 1.0
    edge = border_ids(labels)
    rows = []
    for rp in regionprops(labels):
        major, minor = rp.axis_major_length, rp.axis_minor_length
        axial = (90.0 + math.degrees(rp.orientation)) % 180.0
        rows.append(dict(
            label=rp.label, area=rp.area * scale, area_px=rp.area,
            aspect_ratio=major / minor if minor > 0 else np.nan,
            axial_deg=axial, misalign_deg=min(axial, 180.0 - axial),
            circularity=4 * math.pi * rp.area / rp.perimeter ** 2 if rp.perimeter > 0 else np.nan,
            centroid_y=rp.centroid[0], centroid_x=rp.centroid[1],
            touches_border=rp.label in edge))
    cols = ['label', 'area', 'area_px', 'aspect_ratio', 'axial_deg', 'misalign_deg',
            'circularity', 'centroid_y', 'centroid_x', 'touches_border']
    return pd.DataFrame(rows, columns=cols).set_index('label')


def axial_diff(a, b):
    """Smallest difference between axial angles (degrees, period 180)."""
    d = np.abs(np.asarray(a, float) - np.asarray(b, float)) % 180.0
    return np.minimum(d, 180.0 - d)


def agreement(gt_values, pred_values):
    """Bland-Altman bias/limits, relative bias, median APE and Lin's CCC."""
    x, y = np.asarray(gt_values, float), np.asarray(pred_values, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 2:
        return dict(n=len(x))
    d = y - x
    sd = d.std(ddof=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = d / x
    rel = rel[np.isfinite(rel)]
    cov = ((x - x.mean()) * (y - y.mean())).mean()
    den = x.var() + y.var() + (x.mean() - y.mean()) ** 2
    ccc = 2 * cov / den if den > 0 else 1.0
    return dict(n=len(x), bias=d.mean(), loa_low=d.mean() - 1.96 * sd,
                loa_high=d.mean() + 1.96 * sd, rel_bias=rel.mean() if len(rel) else np.nan,
                median_ape=np.median(np.abs(rel)) if len(rel) else np.nan, ccc=ccc)


def nuclei_per_object(labels, points=None, nuclei=None):
    """Nuclei count per object from click points or from a nuclear label image."""
    if points is None and nuclei is None:
        return None
    if points is None:
        ids = np.unique(nuclei)
        points = centroids(nuclei, ids[ids > 0])
    pts = np.rint(np.asarray(points, float)).astype(int).reshape(-1, 2)
    h, w = labels.shape
    pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < h) & (pts[:, 1] >= 0) & (pts[:, 1] < w)]
    hit = labels[pts[:, 0], pts[:, 1]]
    ids, counts = np.unique(hit[hit > 0], return_counts=True)
    return dict(zip(ids.tolist(), counts.tolist()))


# ----------------------------------------------------------------------------
# Evaluations
# ----------------------------------------------------------------------------

def evaluate_cells(gt, pred, px_um=None, roi=None, thresholds=DEFAULT_THRESHOLDS,
                   gt_points=None, pred_nuclei=None, min_ar_orient=1.3):
    """Score whole-cell territories.

    Returns ``(summary, pairs)``: a dict of detection, split/merge and
    reported-quantity metrics for this image/ROI, and a DataFrame of the
    IoU >= 0.5 matched pairs with GT and predicted morphometrics.

    ``gt_points`` (nucleus clicks) and ``pred_nuclei`` (the pipeline's nuclear
    label image) enable the multinucleation comparison.
    """
    gt = np.asarray(gt)
    pred = np.asarray(pred)
    g_ids, p_ids, inter, ga, pa = overlap(gt, pred)
    iou = iou_matrix(inter, ga, pa)
    p_scope = in_roi(centroids(pred, p_ids), roi)

    s = dict(n_gt=len(g_ids), n_pred=int(p_scope.sum()))
    ap = []
    for t in thresholds:
        tp, fp, fn, ious = detection_counts(iou, p_scope, t)
        ap.append(tp / (tp + fp + fn) if tp + fp + fn else np.nan)
        if math.isclose(t, 0.5) or math.isclose(t, 0.75):
            k = f"{int(round(t * 100))}"
            s[f'tp@{k}'], s[f'fp@{k}'], s[f'fn@{k}'] = tp, fp, fn
            s[f'precision@{k}'], s[f'recall@{k}'], s[f'f1@{k}'] = prf(tp, fp, fn)
            if k == '50':
                s['mean_iou_matched'] = float(ious.mean()) if len(ious) else np.nan
                s['sum_iou_matched'] = float(ious.sum())
                denom = tp + 0.5 * fp + 0.5 * fn
                s['pq'] = float(ious.sum() / denom) if denom else np.nan
    s['ap50_95'] = float(np.nanmean(ap)) if len(ap) else np.nan

    split_gt, merge_pred, n_in = split_merge(inter, ga, pa)
    s['n_split_gt'] = int(split_gt.sum())
    s['n_merge_pred'] = int((merge_pred & p_scope).sum())
    s['n_gt_in_merges'] = int(n_in[merge_pred & p_scope].sum())

    # Reported quantities over *all* in-scope, interior objects: this is what
    # the analysis notebooks average, merges and spurious objects included.
    # Sums and counts (not means) are stored so groups pool correctly.
    mg, mp_all = morphology(gt, px_um), morphology(pred, px_um)
    mp = mp_all.loc[p_ids[p_scope]]
    ng = nuclei_per_object(gt, points=gt_points)
    npred = nuclei_per_object(pred, nuclei=pred_nuclei)
    for tag, m, nuc in (('gt', mg, ng), ('pred', mp, npred)):
        inner = m[~m.touches_border]
        ar = inner.aspect_ratio[np.isfinite(inner.aspect_ratio)]
        elong = inner.misalign_deg[inner.aspect_ratio >= min_ar_orient]
        s.update({f'{tag}_n_interior': len(inner), f'{tag}_sum_area': inner.area.sum(),
                  f'{tag}_median_area': inner.area.median(),
                  f'{tag}_n_ar': len(ar), f'{tag}_sum_ar': ar.sum(),
                  f'{tag}_n_elongated': len(elong), f'{tag}_sum_misalign': elong.sum()})
        if nuc is not None:
            counts = np.array([nuc.get(i, 0) for i in inner.index])
            s[f'{tag}_n_multinucleated'] = int((counts >= 2).sum())
            s[f'{tag}_n_no_nucleus'] = int((counts == 0).sum())

    r, c = match(iou, 0.5)
    pairs = pd.DataFrame({'gt_label': g_ids[r], 'pred_label': p_ids[c], 'iou': iou[r, c]})
    if len(pairs):
        a = mg.loc[pairs.gt_label].reset_index(drop=True).add_prefix('gt_')
        b = mp_all.loc[pairs.pred_label].reset_index(drop=True).add_prefix('pred_')
        pairs = pd.concat([pairs, a, b], axis=1)
        pairs['orient_err_deg'] = axial_diff(pairs.gt_axial_deg, pairs.pred_axial_deg)
        pairs['either_border'] = pairs.gt_touches_border | pairs.pred_touches_border
        if ng is not None:
            pairs['gt_n_nuclei'] = [ng.get(i, 0) for i in pairs.gt_label]
        if npred is not None:
            pairs['pred_n_nuclei'] = [npred.get(i, 0) for i in pairs.pred_label]
    return s, pairs


def evaluate_holes(gt, pred, px_um=None, roi=None, min_area_px=0, iou_thr=0.3):
    """Score an intercellular-gap mask: pixels, gap area fraction and objects."""
    g, p = np.asarray(gt) > 0, np.asarray(pred) > 0
    if g.shape != p.shape:
        raise ValueError(f"shape mismatch: gt {g.shape} vs pred {p.shape}")
    if roi is not None:
        y0, x0, h, w = roi
        g, p = g[y0:y0 + h, x0:x0 + w], p[y0:y0 + h, x0:x0 + w]

    def objects(mask):
        lab = connected_components(mask, connectivity=2)
        if min_area_px > 1 and lab.max():
            sizes = np.bincount(lab.ravel())
            small = sizes < min_area_px
            small[0] = False
            lab[small[lab]] = 0
            lab = connected_components(lab > 0, connectivity=2)
        return lab

    gl, pl = objects(g), objects(p)
    gf, pf = gl > 0, pl > 0
    tp, fp, fn = int((gf & pf).sum()), int((~gf & pf).sum()), int((gf & ~pf).sum())
    both_empty = not gf.any() and not pf.any()
    s = dict(n_px=gf.size, gt_area_frac=gf.mean(), pred_area_frac=pf.mean(),
             pred_area_frac_unfiltered=p.mean(),
             dice=1.0 if both_empty else 2 * tp / (2 * tp + fp + fn),
             iou=1.0 if both_empty else tp / (tp + fp + fn),
             px_tp=tp, px_fp=fp, px_fn=fn,
             gt_present=bool(gf.any()), pred_present=bool(pf.any()))
    s['area_frac_err'] = s['pred_area_frac'] - s['gt_area_frac']
    if px_um:
        s['gt_area_um2'] = gf.sum() * px_um ** 2
        s['pred_area_um2'] = pf.sum() * px_um ** 2

    g_ids, p_ids, inter, ga, pa = overlap(gl, pl)
    otp, ofp, ofn, _ = detection_counts(iou_matrix(inter, ga, pa), np.ones(len(p_ids), bool), iou_thr)
    s.update(obj_n_gt=len(g_ids), obj_n_pred=len(p_ids), obj_tp=otp, obj_fp=ofp, obj_fn=ofn)
    s['obj_precision'], s['obj_recall'], s['obj_f1'] = prf(otp, ofp, ofn)
    return s


def evaluate_nuclei_points(points, pred, roi=None, tol_px=3):
    """Nuclear detection from one GT click per nucleus.

    A click inside (or within ``tol_px`` of) a predicted nucleus hits it. Each
    predicted nucleus counts once: extra clicks in the same object are misses
    (two nuclei segmented as one). In-scope predictions without a click are
    false positives, which is where over-split nuclei show up.
    """
    pred = np.asarray(pred)
    pts = np.rint(np.asarray(points, float)).astype(int).reshape(-1, 2)
    h, w = pred.shape
    pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < h) & (pts[:, 1] >= 0) & (pts[:, 1] < w)]
    hit = pred[pts[:, 0], pts[:, 1]].astype(np.int64)
    for k in np.flatnonzero(hit == 0):
        y, x = pts[k]
        win = pred[max(0, y - tol_px):y + tol_px + 1, max(0, x - tol_px):x + tol_px + 1]
        yy, xx = np.nonzero(win)
        if len(yy):
            d = (yy + max(0, y - tol_px) - y) ** 2 + (xx + max(0, x - tol_px) - x) ** 2
            j = int(np.argmin(d))
            if d[j] <= tol_px ** 2:
                hit[k] = win[yy[j], xx[j]]
    ids = np.unique(pred)
    ids = ids[ids > 0]
    scope = in_roi(centroids(pred, ids), roi)
    hit_ids, counts = np.unique(hit[hit > 0], return_counts=True)
    tp = len(hit_ids)
    fp = int(scope.sum() - np.isin(ids[scope], hit_ids).sum())
    fn = len(pts) - tp
    p, r, f = prf(tp, fp, fn)
    return dict(n_gt=len(pts), n_pred=int(scope.sum()), tp=tp, fp=fp, fn=fn,
                precision=p, recall=r, f1=f, n_pred_with_2plus_clicks=int((counts >= 2).sum()))


def voronoi_from_seeds(seeds, mask=None):
    """Nearest-nucleus tessellation: a null model any junction-aware method should beat."""
    seeds = np.asarray(seeds)
    if not seeds.any():
        return np.zeros_like(seeds)
    _, (iy, ix) = ndimage.distance_transform_edt(seeds == 0, return_indices=True)
    out = seeds[iy, ix]
    if mask is not None:
        out = np.where(mask, out, 0)
    return out


# ----------------------------------------------------------------------------
# Aggregation with cluster bootstrap (resampling images/ROIs, not cells)
# ----------------------------------------------------------------------------

def cluster_bootstrap(rows, stat, n_boot=2000, seed=0, ci=0.95):
    """Point estimate and percentile CI of ``stat(rows)``, resampling rows.

    ``rows`` holds one row per image/ROI (the independent unit), so cells of
    the same field are never treated as independent.
    """
    rows = rows.reset_index(drop=True)
    est = stat(rows)
    if len(rows) < 2:
        return est, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = [stat(rows.iloc[rng.integers(0, len(rows), len(rows))]) for _ in range(n_boot)]
    lo, hi = np.nanpercentile(boots, [50 * (1 - ci), 50 * (1 + ci)])
    return est, lo, hi


def pooled_f1(k):
    return lambda d: 2 * d[f'tp@{k}'].sum() / max(1, 2 * d[f'tp@{k}'].sum() + d[f'fp@{k}'].sum() + d[f'fn@{k}'].sum())


def ratio_of_sums(num, den):
    return lambda d: d[num].sum() / d[den].sum() if d[den].sum() else np.nan


def fmt_ci(est, lo, hi, pct=False, digits=3):
    f = (lambda v: f"{100 * v:.1f}%") if pct else (lambda v: f"{v:.{digits}f}")
    if not np.isfinite(est):
        return 'n/a'
    return f"{f(est)} [{f(lo)}, {f(hi)}]" if np.isfinite(lo) else f(est)


def _mean(tag, q, n):
    return lambda x: x[f'{tag}_sum_{q}'].sum() / x[f'{tag}_n_{n}'].sum() if x[f'{tag}_n_{n}'].sum() else np.nan


def _rel(q, n):
    return lambda x: _mean('pred', q, n)(x) / _mean('gt', q, n)(x) - 1


def summarise_cells(per_roi, pairs, n_boot=2000):
    quality = ['| group | ROIs | GT cells | F1@0.5 | F1@0.75 | AP@0.5:0.95 | PQ | '
               'GT cells split | GT cells in merges |', '|---|---|---|---|---|---|---|---|---|']
    reported = ['| group | cells GT / pred | mean area GT / pred (rel. err.) | mean AR GT / pred (rel. err.) | '
                'mean misalignment GT / pred (deg) | multinucleated GT / pred |', '|---|---|---|---|---|---|']
    out = {}
    for name, d in [('ALL', per_roi)] + list(per_roi.groupby('condition')):
        b = lambda f, **kw: cluster_bootstrap(d, f, n_boot, **kw)  # noqa: E731
        f50, f75 = b(pooled_f1(50)), b(pooled_f1(75))
        apm = b(lambda x: x.ap50_95.mean())
        pq = b(lambda x: x.sum_iou_matched.sum() / max(1e-9, (x['tp@50'] + 0.5 * x['fp@50'] + 0.5 * x['fn@50']).sum()))
        spl, mrg = b(ratio_of_sums('n_split_gt', 'n_gt')), b(ratio_of_sums('n_gt_in_merges', 'n_gt'))
        area_err, ar_err = b(_rel('area', 'interior')), b(_rel('ar', 'ar'))
        quality.append(f"| {name} | {len(d)} | {int(d.n_gt.sum())} | {fmt_ci(*f50)} | {fmt_ci(*f75)} | "
                       f"{fmt_ci(*apm)} | {fmt_ci(*pq)} | {fmt_ci(*spl, pct=True)} | {fmt_ci(*mrg, pct=True)} |")
        mis = (_mean('gt', 'misalign', 'elongated')(d), _mean('pred', 'misalign', 'elongated')(d))
        multi = 'n/a'
        if 'gt_n_multinucleated' in d and 'pred_n_multinucleated' in d:
            multi = (f"{100 * ratio_of_sums('gt_n_multinucleated', 'gt_n_interior')(d):.1f}% / "
                     f"{100 * ratio_of_sums('pred_n_multinucleated', 'pred_n_interior')(d):.1f}%")
        reported.append(f"| {name} | {int(d.gt_n_interior.sum())} / {int(d.pred_n_interior.sum())} | "
                        f"{_mean('gt', 'area', 'interior')(d):.4g} / {_mean('pred', 'area', 'interior')(d):.4g} "
                        f"({fmt_ci(*area_err, pct=True)}) | {_mean('gt', 'ar', 'ar')(d):.3f} / "
                        f"{_mean('pred', 'ar', 'ar')(d):.3f} ({fmt_ci(*ar_err, pct=True)}) | "
                        f"{mis[0]:.1f} / {mis[1]:.1f} | {multi} |")
        out[name] = dict(f1_50=f50, f1_75=f75, ap50_95=apm, pq=pq, split_rate=spl, merge_rate=mrg,
                         mean_area_rel_err=area_err, mean_ar_rel_err=ar_err)
    text = ('Segmentation quality (cluster-bootstrap 95% CI):\n\n' + '\n'.join(quality) +
            '\n\nReported quantities, all interior cells in scope (merges and spurious objects included):\n\n'
            + '\n'.join(reported))
    if len(pairs):
        ok = pairs[~pairs.either_border]
        text += '\n\nPer-cell agreement on IoU>=0.5 matches (interior cells):\n\n'
        text += '| quantity | n | bias | 95% limits of agreement | median abs. rel. error | CCC |\n|---|---|---|---|---|---|\n'
        for q in ('area', 'aspect_ratio'):
            a = agreement(ok[f'gt_{q}'], ok[f'pred_{q}'])
            if a.get('n', 0) >= 2:
                text += (f"| {q} | {a['n']} | {a['bias']:.3g} | [{a['loa_low']:.3g}, {a['loa_high']:.3g}] | "
                         f"{100 * a['median_ape']:.1f}% | {a['ccc']:.3f} |\n")
        el = ok[ok.gt_aspect_ratio >= 1.3]
        if len(el):
            text += (f"| orientation (GT AR>=1.3) | {len(el)} | median axial error "
                     f"{el.orient_err_deg.median():.1f} deg | 90th pct {el.orient_err_deg.quantile(0.9):.1f} deg | | |\n")
        if 'gt_n_nuclei' in ok and 'pred_n_nuclei' in ok:
            cm = pd.crosstab(ok.gt_n_nuclei >= 2, ok.pred_n_nuclei >= 2,
                             rownames=['GT multinucleated'], colnames=['pred multinucleated'])
            text += '\nMultinucleation on matched cells:\n\n' + cm.to_string() + '\n'
    return text, out


def summarise_holes(per_img, n_boot=2000):
    lines = ['| group | images | GT gap fraction | pred gap fraction | mean abs error (pp) | '
             'pooled Dice | object F1 | presence agreement |', '|---|---|---|---|---|---|---|---|']
    out = {}
    for name, d in [('ALL', per_img)] + list(per_img.groupby('condition')):
        gtf = cluster_bootstrap(d, lambda x: x.gt_area_frac.mean(), n_boot)
        prf_ = cluster_bootstrap(d, lambda x: x.pred_area_frac.mean(), n_boot)
        mae = cluster_bootstrap(d, lambda x: (x.area_frac_err.abs() * 100).mean(), n_boot)
        dice = cluster_bootstrap(d, lambda x: 2 * x.px_tp.sum() / max(1, 2 * x.px_tp.sum() + x.px_fp.sum() + x.px_fn.sum()), n_boot)
        of1 = cluster_bootstrap(d, lambda x: 2 * x.obj_tp.sum() / max(1, 2 * x.obj_tp.sum() + x.obj_fp.sum() + x.obj_fn.sum()), n_boot)
        pres = (d.gt_present == d.pred_present).mean()
        lines.append(f"| {name} | {len(d)} | {fmt_ci(*gtf, pct=True)} | {fmt_ci(*prf_, pct=True)} | "
                     f"{fmt_ci(*mae, digits=2)} | {fmt_ci(*dice)} | {fmt_ci(*of1)} | {100 * pres:.0f}% |")
        out[name] = dict(gt_frac=gtf, pred_frac=prf_, mae_pp=mae, dice=dice, obj_f1=of1, presence=pres)
    return '\n'.join(lines), out


def summarise_nuclei(per_img, n_boot=2000):
    lines = ['| group | images | GT nuclei | precision | recall | F1 | objects with 2+ clicks |',
             '|---|---|---|---|---|---|---|']
    out = {}
    for name, d in [('ALL', per_img)] + list(per_img.groupby('condition')):
        p = cluster_bootstrap(d, ratio_of_sums('tp', 'n_pred'), n_boot)
        r = cluster_bootstrap(d, ratio_of_sums('tp', 'n_gt'), n_boot)
        f = cluster_bootstrap(d, lambda x: 2 * x.tp.sum() / max(1, 2 * x.tp.sum() + x.fp.sum() + x.fn.sum()), n_boot)
        lines.append(f"| {name} | {len(d)} | {int(d.n_gt.sum())} | {fmt_ci(*p)} | {fmt_ci(*r)} | "
                     f"{fmt_ci(*f)} | {int(d.n_pred_with_2plus_clicks.sum())} |")
        out[name] = dict(precision=p, recall=r, f1=f)
    return '\n'.join(lines), out


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def index_by_key(directory, pattern):
    """Map sample key -> file; refuses ambiguous folders instead of guessing."""
    found = {}
    for path in sorted(glob.glob(os.path.join(directory, pattern))):
        key = sample_key(path)
        if key is None:
            continue
        if key in found:
            raise SystemExit(f"two files for {key} in {directory} ({os.path.basename(found[key])}, "
                             f"{os.path.basename(path)}); narrow --pred-glob / --gt-glob")
        found[key] = path
    return found


def find_roi_file(directory, roi_id, pattern):
    """File named ``<roi_id>`` + separator + anything that matches ``pattern``.

    The separator check keeps ``..._r01`` from also picking up ``..._r010``.
    """
    import fnmatch
    for path in sorted(glob.glob(os.path.join(directory, f"{glob.escape(roi_id)}*"))):
        name = os.path.basename(path)
        rest = name[len(roi_id):]
        if rest[:1] in ('.', '_', '-') and fnmatch.fnmatch(name, pattern):
            return path
    return None


def work_items(args):
    """(roi_id, key, roi, gt_path) tuples from a manifest or from GT file names."""
    if args.manifest:
        man = pd.read_csv(args.manifest)
        if 'task' in man:
            # Nucleus clicks are made on the cell ROIs unless a manifest has its own rows.
            wanted = {'cells': ['cells'], 'nuclei': ['nuclei', 'cells'], 'holes': ['holes']}[args.cmd]
            task = next((t for t in wanted if t in set(man.task)), None)
            if task is None:
                raise SystemExit(f"manifest has no rows with task {wanted[0]!r}")
            man = man[man.task == task]
        items = []
        for row in man.itertuples():
            path = find_roi_file(args.gt, row.roi_id, args.gt_glob)
            if path:
                items.append((row.roi_id, row.key, (int(row.y0), int(row.x0), int(row.h), int(row.w)), path))
        return items
    return [(key, key, None, path) for key, path in index_by_key(args.gt, args.gt_glob).items()]


def parse_methods(args):
    """[(name, folder, glob)] from repeated --method NAME=DIR[::GLOB], or from --pred."""
    if args.method:
        out = []
        for spec in args.method:
            name, _, rest = spec.partition('=')
            folder, _, pattern = rest.partition('::')
            if not name or not folder:
                raise SystemExit(f"bad --method {spec!r}; use NAME=DIR or NAME=DIR::GLOB")
            out.append((name, folder, pattern or args.pred_glob))
        return out
    if not args.pred:
        raise SystemExit("give --pred DIR or one or more --method NAME=DIR[::GLOB]")
    return [(None, args.pred, args.pred_glob)]


def evaluate_method(args, items, pred_dir, pred_glob, px):
    """Score one method's outputs on every GT item; returns (per_item, pairs)."""
    pred_files = index_by_key(pred_dir, pred_glob)
    nuc_files = index_by_key(args.pred_nuclei, args.pred_nuclei_glob) if getattr(args, 'pred_nuclei', None) else {}
    rows, all_pairs, missing = [], [], []
    for roi_id, key, roi, gt_path in items:
        if key not in pred_files:
            missing.append(key)
            continue
        pred = read_mask(pred_files[key])
        base = dict(roi_id=roi_id, key=key, condition=condition_of(key),
                    gt_file=os.path.basename(gt_path), pred_file=os.path.basename(pred_files[key]))
        um = px(key) if px else None
        if args.cmd == 'cells':
            points = None
            if args.gt_nuclei:
                hit = find_roi_file(args.gt_nuclei, roi_id, '*.csv')
                points = read_points(hit) if hit else None
            nuc = read_mask(nuc_files[key]) if key in nuc_files else None
            s, pairs = evaluate_cells(read_mask(gt_path), pred, um, roi, gt_points=points, pred_nuclei=nuc)
            if len(pairs):
                all_pairs.append(pairs.assign(roi_id=roi_id, condition=base['condition']))
        elif args.cmd == 'holes':
            min_px = args.min_area_px
            if args.min_area_um2 is not None:
                if um is None:
                    raise SystemExit("--min-area-um2 needs --px-um")
                min_px = int(math.ceil(args.min_area_um2 / um ** 2))
            s = evaluate_holes(read_mask(gt_path), pred, um, roi, min_px, args.iou)
        else:
            s = evaluate_nuclei_points(read_points(gt_path), pred, roi, args.tol_px)
        rows.append({**base, **s})
    if missing:
        print(f"warning: {pred_dir}: no prediction for {len(missing)} item(s): {missing[:5]}", file=sys.stderr)
    pairs = pd.concat(all_pairs, ignore_index=True) if all_pairs else pd.DataFrame()
    return pd.DataFrame(rows), pairs


# Headline columns of the multi-method leaderboard: (label, summary key, as percent)
LEADERBOARD = {
    'cells': [('F1@0.5', 'f1_50', False), ('F1@0.75', 'f1_75', False), ('AP@0.5:0.95', 'ap50_95', False),
              ('PQ', 'pq', False), ('GT split', 'split_rate', True), ('GT in merges', 'merge_rate', True),
              ('mean-area error', 'mean_area_rel_err', True), ('mean-AR error', 'mean_ar_rel_err', True)],
    'holes': [('gap-fraction MAE (pp)', 'mae_pp', False), ('pooled Dice', 'dice', False),
              ('object F1', 'obj_f1', False)],
    'nuclei': [('precision', 'precision', False), ('recall', 'recall', False), ('F1', 'f1', False)],
}


def run(args):
    px = parse_px_um(args.px_um)
    if px is None and args.cmd in ('cells', 'holes'):
        print("warning: --px-um not given; areas are reported in pixels", file=sys.stderr)
    items = work_items(args)
    if not items:
        raise SystemExit("no ground-truth files found (check --gt, --gt-glob, --manifest)")
    methods = parse_methods(args)
    summarise = {'cells': lambda per, pairs: summarise_cells(per, pairs, args.n_boot),
                 'holes': lambda per, pairs: summarise_holes(per, args.n_boot),
                 'nuclei': lambda per, pairs: summarise_nuclei(per, args.n_boot)}[args.cmd]
    units = 'um^2' if px else 'px'
    board = []
    for name, pred_dir, pred_glob in methods:
        per, pairs = evaluate_method(args, items, pred_dir, pred_glob, px)
        if per.empty:
            print(f"warning: nothing evaluated for {name or pred_dir}", file=sys.stderr)
            continue
        out = os.path.join(args.out, name) if name else args.out
        os.makedirs(out, exist_ok=True)
        per.to_csv(os.path.join(out, f'{args.cmd}_per_item.csv'), index=False)
        if args.cmd == 'cells':
            pairs.to_csv(os.path.join(out, 'cells_matched_pairs.csv'), index=False)
        text, summary = summarise(per, pairs)
        header = (f"# {args.cmd} validation{f': {name}' if name else ''}\n\nGT: `{args.gt}`  \n"
                  f"Prediction: `{pred_dir}` (`{pred_glob}`)  \nArea units: {units}. "
                  f"95% CIs from {args.n_boot} bootstrap resamples of images/ROIs.\n\n")
        with open(os.path.join(out, f'{args.cmd}_summary.md'), 'w') as f:
            f.write(header + text + '\n')
        with open(os.path.join(out, f'{args.cmd}_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=float)
        print(header + text + '\n')
        board.append((name or os.path.basename(os.path.normpath(pred_dir)), len(per), summary))
    if not board:
        raise SystemExit("nothing evaluated")
    if len(methods) > 1:
        cols = LEADERBOARD[args.cmd]
        groups = sorted({g for _, _, s in board for g in s}, key=lambda g: (g != 'ALL', g))
        lines = [f"# {args.cmd}: method comparison on the same ground truth\n",
                 f"Area units: {units}. Point estimate [95% cluster-bootstrap CI]; every method is scored "
                 "on exactly the same GT items.\n"]
        for g in groups:
            lines += [f"\n## {g}\n", '| method | items | ' + ' | '.join(c[0] for c in cols) + ' |',
                      '|---' * (len(cols) + 2) + '|']
            for name, n, s in board:
                if g in s:
                    lines.append(f"| {name} | {n} | " + ' | '.join(
                        fmt_ci(*s[g][k], pct=pct) for _, k, pct in cols) + ' |')
        with open(os.path.join(args.out, f'{args.cmd}_leaderboard.md'), 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print('\n'.join(lines))


def run_voronoi(args):
    os.makedirs(args.out, exist_ok=True)
    masks = index_by_key(args.mask_dir, args.mask_glob) if args.mask_dir else {}
    n = 0
    for key, path in index_by_key(args.nuclei, args.nuclei_glob).items():
        import tifffile
        mask = read_mask(masks[key]) > 0 if key in masks else None
        tifffile.imwrite(os.path.join(args.out, f"{key}_voronoi.tif"),
                         voronoi_from_seeds(read_mask(path), mask).astype(np.uint32))
        n += 1
    print(f"wrote {n} Voronoi baseline mask(s) to {args.out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name, gt_glob, helptext in (
            ('cells', '*.tif', 'whole-cell instance masks'),
            ('holes', '*.tif', 'binary gap masks'),
            ('nuclei', '*.csv', 'nucleus click CSVs vs predicted nuclear masks')):
        p = sub.add_parser(name, help=helptext)
        p.add_argument('--gt', required=True, help='folder of ground-truth files')
        p.add_argument('--gt-glob', default=gt_glob,
                       help='suffix pattern of GT files after the roi_id/key (default %(default)s)')
        p.add_argument('--pred', help='folder of pipeline (or baseline) outputs')
        p.add_argument('--pred-glob', default='*.tif', help='file pattern inside --pred (default %(default)s)')
        p.add_argument('--method', action='append', metavar='NAME=DIR[::GLOB]',
                       help='repeat to score several methods on the same GT and write a leaderboard')
        p.add_argument('--manifest', help='ROI manifest from sample_rois.py')
        p.add_argument('--px-um', help="pixel size, e.g. 0.65 or '20x=0.65,40x=0.325' (from OME metadata)")
        p.add_argument('--out', required=True)
        p.add_argument('--n-boot', type=int, default=2000)
        if name == 'cells':
            p.add_argument('--gt-nuclei', help='folder of nucleus click CSVs (<roi_id>*.csv) for multinucleation')
            p.add_argument('--pred-nuclei', help="folder of the pipeline's nuclear masks (Segmented/<cond>/Nuclei)")
            p.add_argument('--pred-nuclei-glob', default='*.tif')
        if name == 'holes':
            p.add_argument('--min-area-um2', type=float, help='drop gap objects smaller than this (GT and pred)')
            p.add_argument('--min-area-px', type=int, default=0)
            p.add_argument('--iou', type=float, default=0.3, help='IoU for gap object matching (default %(default)s)')
        if name == 'nuclei':
            p.add_argument('--tol-px', type=int, default=3, help='click-to-mask tolerance in pixels')
    v = sub.add_parser('voronoi', help='write the nearest-nucleus (Voronoi) baseline for comparison')
    v.add_argument('--nuclei', required=True, help='folder of nuclear label masks (seeds)')
    v.add_argument('--nuclei-glob', default='*.tif')
    v.add_argument('--mask-dir', help='optional folder of foreground masks (e.g. not-hole) to clip to')
    v.add_argument('--mask-glob', default='*.tif')
    v.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    if args.cmd == 'voronoi':
        run_voronoi(args)
    else:
        run(args)


if __name__ == '__main__':
    main()

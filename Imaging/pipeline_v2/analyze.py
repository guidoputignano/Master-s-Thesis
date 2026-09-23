#!/usr/bin/env python3
"""Compare the original pipeline (v1) with v2 on every field, and estimate senescence.

Reads the data repository layout (Projection/, Segmented/, Senescence/) plus the v2
masks written by segment_v2.py. Writes, to --out:

* cells_v1.csv, cells_v2.csv: per-cell features, computed identically for both;
* fields.csv: per-field summaries (density, gaps, alignment order);
* summary.md: per-condition comparison, v1 vs v2 agreement, senescence mixture fits.

No annotation is involved; the expert verdicts are scored separately (verdict_session.py).
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage
from skimage.measure import label as cc_label
from skimage.morphology import disk, opening
from skimage.segmentation import find_boundaries

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'Validation'))
sys.path.insert(0, HERE)
import seg_eval as se  # noqa: E402
import features as ft  # noqa: E402
import senescence as sn  # noqa: E402
from diagnostics import md_table  # noqa: E402

CONDS = ['Static-x20', 'Static-x40', '1.4Pa-x20', '1.4Pa-x40']
V1_HOLES = {'Static-x20': ('Holes', '*regional_segmented.tif'), 'Static-x40': ('Holes_algo', '*regional_segmented.tif'),
            '1.4Pa-x20': ('Holes_masks', '*segmented_dilated.tif'),   # includes seq016's '_manual_' mask
            '1.4Pa-x40': ('Holes_masks', '*regional_segmented.tif')}
# 'Nuclei' is the filtered subset of 'Nuclei_raw' and the seeds' source, except in 1.4Pa-x40,
# where 'Nuclei' is a copy of 'Nuclei_raw' and the seeds came from 'Nuclei_filtered'.
V1_NUCLEI = {'Static-x20': 'Nuclei', 'Static-x40': 'Nuclei', '1.4Pa-x20': 'Nuclei', '1.4Pa-x40': 'Nuclei_filtered'}
DUPLICATE = re.compile(r'\s?\(\d+\)')   # 'seq001 (1).tif', 'tophat(1).tif': copies of a field
GAP_MIN_UM2 = 10.0      # smallest gap counted (both methods, for the comparison)
GAP_OPEN_UM = 1.0       # opening radius that removes the thin unlabelled lines between cells
GAP_DARK_PCT = 1.0      # gap pixels are darker than all but this % of cell interiors (per field)
GAP_SMOOTH_UM = 1.0     # Gaussian smoothing of the VE-cadherin projection before the darkness test
CORE_UM = 2.0           # cell interior = farther than this from any cell boundary


def index(folder, pattern='*.tif'):
    d = {}
    for p in sorted(glob.glob(os.path.join(folder, pattern))):
        k = se.sample_key(p)
        if k and not DUPLICATE.search(os.path.basename(p)) and k not in d:
            d[k] = p
    return d


def gaps_from_cells(cells, um):
    """v2 gaps: pixels no cell covers, opened to drop inter-cell lines, above a minimum area."""
    free = cells == 0
    r = max(1, int(round(GAP_OPEN_UM / um)))
    free = opening(free, disk(r))
    lab = cc_label(free, connectivity=2)
    if lab.max():
        sizes = np.bincount(lab.ravel()) * um ** 2
        keep = sizes >= GAP_MIN_UM2
        keep[0] = False
        free = keep[lab]
    return free


def interior_level(img, cells, um):
    """Median intensity of each cell's interior (away from the junctions)."""
    core = cells * (ndimage.distance_transform_edt(~find_boundaries(cells, mode='thick')) >= CORE_UM / um)
    ids = np.unique(core)
    ids = ids[ids > 0]
    return np.asarray(ndimage.median(img, labels=core, index=ids)) if len(ids) else np.array([])


def dark_gaps(cells, nuclei, cad, um, pct=GAP_DARK_PCT):
    """v2 gaps: darker than cell cytoplasm, not inside a nucleated cell, no nucleus inside.

    ``cad`` is the VE-cadherin projection without top-hat (the top-hat removes the
    diffuse cytoplasmic signal that separates a cell interior from bare substrate).
    The threshold is the ``pct`` percentile of the per-cell interior medians of the
    same field. Returns the gap mask and the threshold.
    """
    smooth = ndimage.gaussian_filter(np.asarray(cad, float), GAP_SMOOTH_UM / um)
    level = interior_level(smooth, cells, um)
    if not len(level):
        return np.zeros(cells.shape, bool), np.nan
    thr = float(np.percentile(level, pct))
    owner, _, _ = ft.nuclei_assignment(cells, nuclei)
    nucleated = np.isin(cells, list(set(owner.values())))
    cand = opening((smooth < thr) & ~nucleated, disk(max(1, int(round(GAP_OPEN_UM / um)))))
    lab = cc_label(cand, connectivity=2)
    if not lab.max():
        return cand, thr
    area = np.bincount(lab.ravel()) * um ** 2
    nuc = np.bincount(lab.ravel(), weights=(nuclei > 0).ravel(), minlength=len(area)) * um ** 2
    keep = (area >= GAP_MIN_UM2) & (nuc <= 0.05 * area)
    keep[0] = False
    return keep[lab], thr


def orphan_fraction(cells, nuclei):
    """Share of nuclei with no cell covering more than half of them."""
    owner, n_ids, _ = ft.nuclei_assignment(cells, nuclei)
    return 1 - len(owner) / len(n_ids) if len(n_ids) else np.nan


def nematic(df, min_ar=1.3):
    d = df[~df.touches_border & (df.aspect_ratio >= min_ar)]
    if len(d) < 5:
        return np.nan, np.nan
    z = np.exp(2j * np.deg2rad(d.axial_deg.to_numpy())).mean()
    return abs(z), (np.rad2deg(np.angle(z)) / 2) % 180


def run(root, v2_root, out, conds, limit=None):
    os.makedirs(out, exist_ok=True)
    all1, all2, fields, agree = [], [], [], []
    for cond in conds:
        v1c = index(f'{root}/Segmented/{cond}/Cell_merged_conservative')
        v1n = index(f'{root}/Segmented/{cond}/{V1_NUCLEI[cond]}')
        hd, hp = V1_HOLES[cond]
        v1h = index(f'{root}/Segmented/{cond}/{hd}', hp)
        v2c = index(f'{v2_root}/{cond}', '*_v2_cells.tif')
        v2n = index(f'{v2_root}/{cond}', '*_v2_nuclei.tif')
        cad = index(f'{root}/Projection/{cond}/Cadherins/background')
        keys = sorted(set(v1c) & set(v2c) & set(v2n) & set(cad))[:limit]
        os.makedirs(f'{out}/v2_gaps/{cond}', exist_ok=True)
        for k in keys:
            um = ft.pixel_um(k)
            c1, c2 = tifffile.imread(v1c[k]), tifffile.imread(v2c[k])
            n1 = tifffile.imread(v1n[k]) if k in v1n else None
            n2 = tifffile.imread(v2n[k])
            img = tifffile.imread(cad[k])
            h1 = tifffile.imread(v1h[k]) > 0 if k in v1h else np.zeros_like(c1, bool)
            h2, thr = dark_gaps(c2, n2, img, um)
            tifffile.imwrite(f'{out}/v2_gaps/{cond}/{k}_v2_gaps.tif', h2.astype(np.uint8), compression='zlib')
            sens = {f'v2_gap_frac_p{p:g}': dark_gaps(c2, n2, img, um, p)[0].mean() for p in (0.5, 5.0)}
            f1 = ft.cell_features(c1, k, n1, h1)
            f2 = ft.cell_features(c2, k, n2, h2)
            f1['method'], f2['method'] = 'v1', 'v2'
            all1.append(f1)
            all2.append(f2)
            g, p, inter, ga, pa = se.overlap(c1, c2)
            iou = se.iou_matrix(inter, ga, pa)
            r, c = se.match(iou, 0.5)
            spl, mrg, _ = se.split_merge(inter, ga, pa)      # 'gt' = v1 here
            fov = (c1.shape[0] * um / 1000) * (c1.shape[1] * um / 1000)
            s1, a1 = nematic(f1)
            s2, a2 = nematic(f2)
            fields.append(dict(key=k, condition=se.condition_of(k), folder=cond, fov_mm2=fov,
                               v1_cells=len(f1), v2_cells=len(f2),
                               v1_density=len(f1) / fov, v2_density=len(f2) / fov,
                               v1_gap_frac=h1.mean(), v2_gap_frac=h2.mean(), **sens, v2_gap_threshold=thr,
                               v2_uncovered_frac=gaps_from_cells(c2, um).mean(),
                               gap_iou=(h1 & h2).sum() / max(1, (h1 | h2).sum()),
                               v1_orphan_nuclei=orphan_fraction(c1, n2), v2_orphan_nuclei=orphan_fraction(c2, n2),
                               v1_S=s1, v2_S=s2, v1_axis=a1, v2_axis=a2))
            agree.append(dict(key=k, condition=se.condition_of(k), v1=len(g), v2=len(p), matched=len(r),
                              median_iou=float(np.median(iou[r, c])) if len(r) else np.nan,
                              v1_split_by_v2=int(spl.sum()), v2_merging_v1=int(mrg.sum())))
            print(f"{cond} {k}: v1 {len(f1)} v2 {len(f2)} matched {len(r)}", flush=True)
    cells1, cells2 = pd.concat(all1, ignore_index=True), pd.concat(all2, ignore_index=True)
    cells1.to_csv(f'{out}/cells_v1.csv', index=False)
    cells2.to_csv(f'{out}/cells_v2.csv', index=False)
    fdf, adf = pd.DataFrame(fields), pd.DataFrame(agree)
    fdf.to_csv(f'{out}/fields.csv', index=False)
    adf.to_csv(f'{out}/agreement.csv', index=False)
    return cells1, cells2, fdf, adf


def describe(cells, fields, tag):
    rows = []
    for cond, d in cells.groupby('condition'):
        inner = d[~d.touches_border]
        f = fields[fields.condition == cond]
        rows.append(dict(method=tag, condition=cond, fields=d.key.nunique(), cells=len(d),
                         density=f[f'{tag}_density'].mean(), border_pct=100 * d.touches_border.mean(),
                         area_median=inner.area_um2.median(), AR_median=inner.aspect_ratio.median(),
                         misalign_mean=inner.misalign_deg.mean(), nematic_S=f[f'{tag}_S'].median(),
                         no_nucleus_pct=100 * (inner.n_nuclei == 0).mean(),
                         multinucleated_pct=100 * (inner.n_nuclei >= 2).mean(),
                         nucleus_median=inner.largest_nuc_um2.median(),
                         gap_pct=100 * f[f'{tag}_gap_frac'].mean(),
                         orphan_nuclei_pct=100 * f[f'{tag}_orphan_nuclei'].mean()))
    return pd.DataFrame(rows)


def magnification_table(cells, fields, mixture):
    """40x / 20x ratios of per-condition summaries, per method (same slides, two objectives)."""
    rows = []
    for method, d in cells.groupby('method'):
        inner = d[~d.touches_border]
        free = mixture[(mixture.method == method) & (mixture.model == 'free ratio')].set_index('group')
        for series in sorted({c.rsplit('_', 1)[0] for c in d.condition}):
            c20, c40 = f'{series}_20x', f'{series}_40x'
            if c20 not in set(d.condition) or c40 not in set(d.condition):
                continue
            def stat(c):
                x, f = inner[inner.condition == c], fields[fields.condition == c]
                return dict(area_median=x.area_um2.median(), nucleus_median=x.largest_nuc_um2.median(),
                            density=f[f'{method}_density'].mean(), multinucleated_pct=100 * (x.n_nuclei >= 2).mean(),
                            enlarged_frac=free.frac_enlarged.get(c, np.nan), AR_mean=x.aspect_ratio.mean())
            a, b = stat(c20), stat(c40)
            rows.append(dict(method=method, series=series, **{f'{k}_20x': a[k] for k in a},
                             **{f'{k}_ratio': b[k] / a[k] if a[k] else np.nan for k in a}))
    return pd.DataFrame(rows)


def _summarise(task):
    d, ratio, equal, tag, name = task
    return sn.summarise(d, fixed_log_ratio=ratio, equal_var=equal).assign(method=tag, model=name)


def mixture_cells(cells):
    """Cells entering the mixture: interior, at least one nucleus, above 50 um^2."""
    return cells[~cells.touches_border & (cells.n_nuclei >= 1) & (cells.area_um2 > 50)]


def with_posterior(cells):
    """Mixture cells with P(enlarged) from a free two-component fit per condition."""
    d = mixture_cells(cells).copy()
    d['posterior'] = np.nan
    for _, g in d.groupby('condition'):
        x = np.log(g.area_um2.to_numpy())
        d.loc[g.index, 'posterior'] = sn.posterior(x, sn.fit(x)['params'])
    return d


def chala_table(post):
    rows = []
    for (method, cond), d in post.groupby(['method', 'condition']):
        big = d.posterior > 0.5
        for name, g in (('all', d), ('normal', d[~big]), ('enlarged', d[big])):
            rows.append(dict(method=method, condition=cond, group=name, cells=len(g), pct=100 * len(g) / len(d),
                             area_median=g.area_um2.median(), AR_mean=g.aspect_ratio.mean(),
                             AR_sd=g.aspect_ratio.std(), misalign_mean=g.misalign_deg.mean(),
                             misalign_sd=g.misalign_deg.std(), nucleus_median=g.largest_nuc_um2.median(),
                             multinucleated_pct=100 * (g.n_nuclei >= 2).mean()))
    return pd.DataFrame(rows)


def reported_classes(root, conds):
    """Rule-based calls as reported (Senescence_Results), keyed like the masks."""
    from diagnostics import load_classes
    paths = [p for c in conds for p in glob.glob(
        f'{root}/Senescence/{c}/Senescence_Results/cell_classification_rule_based_full.csv')]
    if not paths:
        return None
    df = load_classes(paths)
    return df.assign(reported_senescent=df.cell_type.eq('Senescent'))[['key', 'label', 'reported_senescent']]


def cohen_kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--root', required=True, help='data repository (Projection/, Segmented/, Senescence/)')
    ap.add_argument('--v2', required=True, help='folder with <cond>/<key>_v2_cells.tif and _v2_nuclei.tif')
    ap.add_argument('--out', required=True)
    ap.add_argument('--conditions', nargs='+', default=CONDS)
    ap.add_argument('--limit', type=int, help='fields per condition (quick look)')
    args = ap.parse_args(argv)
    c1, c2, fdf, adf = run(args.root, args.v2, args.out, args.conditions, args.limit)
    lines = ['# v1 vs v2 on the same fields\n']
    tab = pd.concat([describe(c1, fdf, 'v1'), describe(c2, fdf, 'v2')]).sort_values(['condition', 'method'])
    lines += ['## Per condition (interior cells unless noted)\n', md_table(tab.round(2).set_index('method'))]
    a = adf.groupby('condition')[['v1', 'v2', 'matched', 'v1_split_by_v2', 'v2_merging_v1']].sum()
    a['matched_pct_of_v1'] = 100 * a.matched / a.v1
    lines += ['\n## Agreement (v1 cells matched by a v2 cell at IoU >= 0.5)\n', a.round(1).to_string()]
    models = (('free ratio', None, True), ('Chala ratio 2.27', np.log(sn.CHALA_RATIO), True),
              ('free ratio, unequal variances (not used)', None, False))
    tasks = [(g[['key', 'condition', 'area_um2']], ratio, equal, tag, name)
             for tag, cells in (('v1', c1), ('v2', c2)) for _, g in mixture_cells(cells).groupby('condition')
             for name, ratio, equal in models]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        sens = list(pool.map(_summarise, tasks))
    order = {name: i for i, (name, _, _) in enumerate(models)}
    sdf = pd.concat(sens).sort_values(['method', 'model', 'group'], key=lambda c: c.map(order) if c.name == 'model' else c)
    sdf.to_csv(f'{args.out}/senescence_mixture.csv', index=False)
    lines += ['\n## Senescence: two-component log-normal mixture on interior cells with >= 1 nucleus\n',
              'Shifted-population model (shared variance). The unequal-variance fit is shown to document '
              'why it is not used: it tends to a narrow core plus a broad component.\n',
              sdf.round(3).to_string(index=False)]
    mag = magnification_table(pd.concat([c1, c2], ignore_index=True), fdf, sdf)
    if len(mag):
        lines += ['\n## Magnification consistency: the same slides at 20x and 40x (ratio 40x / 20x; 1 = consistent)\n',
                  md_table(mag.round(3).set_index('method'))]
    post = pd.concat([with_posterior(c1), with_posterior(c2)], ignore_index=True)
    post.to_csv(f'{args.out}/cells_posterior.csv', index=False)
    lines += ['\n## Enlarged (posterior > 0.5) vs normal cells, next to Chala et al. (2021)\n',
              'Chala (HUVEC, 16 h at 1.4 Pa): AR 1.9 +- 0.67 static / 2.3 +- 0.78 flow (control), '
              '1.9 +- 0.65 / 2.0 +- 0.72 (TNF-alpha); misalignment 49 +- 25 -> 20 +- 14 deg (control), '
              '42 +- 26 -> 45 +- 27 deg (TNF-alpha); TNF-alpha / control mean area 2.27; 446-590 cells/mm^2.\n',
              md_table(chala_table(post).round(2).set_index('method'))]
    rep = reported_classes(args.root, args.conditions)
    if rep is not None:
        j = post[post.method == 'v1'].merge(rep, on=['key', 'label'], how='inner')
        rows = []
        for cond, d in j.groupby('condition'):
            a, b = d.reported_senescent.to_numpy(), (d.posterior > 0.5).to_numpy()
            rows.append(dict(condition=cond, cells=len(d), reported_pct=100 * a.mean(), mixture_pct=100 * b.mean(),
                             both=int((a & b).sum()), reported_only=int((a & ~b).sum()),
                             mixture_only=int((~a & b).sum()), kappa=cohen_kappa(a, b)))
        lines += ['\n## v1 cells: reported rule-based call vs mixture posterior (same cells, interior, >= 1 nucleus)\n',
                  md_table(pd.DataFrame(rows).round(3).set_index('condition'))]
    text = '\n'.join(lines) + '\n'
    open(f'{args.out}/summary.md', 'w').write(text)
    print(text)


if __name__ == '__main__':
    main()

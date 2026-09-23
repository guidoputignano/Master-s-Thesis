#!/usr/bin/env python3
"""Blind yes/no session comparing v1 and v2: cells, enlarged ("senescent") cells, gaps.

The expert answers one question per crop, without knowing which method drew the
outline. Three blocks:

* ``cell``: interior cells of each method, stratified by whether the other method
  finds the same cell (IoU >= 0.5). "Is the outlined region exactly one whole cell?"
  The stratified yes-rate estimates each method's cell precision;
* ``enlarged``: v1 cells reported as senescent (any rule) and v2 cells in the
  enlarged mixture component (posterior > 0.5). Same question: an enlarged call is
  only meaningful if the region is one cell;
* ``gap``: connected gap components (>= 10 um^2) of the v1 hole masks and of the v2
  gaps written by analyze.py. "Is the outlined area a gap in the monolayer (no cell
  covers it)?"

  python build_verdicts.py build --root data-mt --v2 v2_masks --analysis v2_analysis --out session
  python build_verdicts.py score --session session/index.html --code 'VS147:YYN...'

``key.csv`` keeps the hidden metadata (method, condition, stratum and its population
share). ``score`` reports yes-rates with Wilson intervals and, for the cell block, the
stratified precision (Jeffreys posterior median and 95 % interval).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import tifffile
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage.measure import label as cc_label
from skimage.segmentation import find_boundaries

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'Validation'))
sys.path.insert(0, HERE)
import seg_eval as se  # noqa: E402
import features as ft  # noqa: E402
import analyze as an  # noqa: E402
import segment_v2 as sg  # noqa: E402
from audit_gallery import normalise, wilson  # noqa: E402
from diagnostics import load_classes  # noqa: E402
import verdict_session as vs  # noqa: E402

Q_CELL = 'Is the outlined region exactly one whole cell (not part of a cell, not two or more cells)?'
Q_GAP = 'Is the outlined area a gap in the monolayer (no cell covers it)?'
N_CELL = 4          # per method, condition and stratum (matched / unmatched)
N_ENLARGED = 6      # per method and condition
N_GAP = 5           # per method and condition


def projections(root, cond):
    """The images v2 segmented: {key: {'Cadherins': path, 'Nuclei': path}}."""
    return {k: {'Cadherins': c, 'Nuclei': n} for k, (c, n) in sg.inputs(root, cond).items()}


def composite(cad, nuc):
    m, n = normalise(cad), normalise(nuc)
    return np.clip(np.stack([m, n, np.maximum(m, n)], -1), 0, 1)


def render(rgb, mask, um, pad_um=12.0, min_half_um=20.0, size=420, bar_um=10.0):
    """Plain crop (with a scale bar) | the same crop with the object outlined in yellow."""
    ys, xs = np.nonzero(mask)
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    half = max((ys.max() - ys.min()) // 2 + round(pad_um / um), (xs.max() - xs.min()) // 2 + round(pad_um / um),
               round(min_half_um / um))
    y0, y1 = max(0, cy - half), min(mask.shape[0], cy + half + 1)
    x0, x1 = max(0, cx - half), min(mask.shape[1], cx + half + 1)
    crop, m = rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    scale = size / max(crop.shape[:2])
    wh = (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale)))
    im = Image.fromarray((crop * 255).astype(np.uint8)).resize(wh, Image.BILINEAR)
    mm = np.array(Image.fromarray(m.astype(np.uint8) * 255).resize(wh, Image.NEAREST)) > 0
    over = np.array(im)
    over[ndimage.binary_dilation(find_boundaries(mm, mode='inner'))] = (255, 215, 0)
    left = im.copy()
    bar = bar_um / um * scale
    ImageDraw.Draw(left).rectangle([10, wh[1] - 16, 10 + bar, wh[1] - 10], fill=(255, 255, 255))
    canvas = Image.new('RGB', (2 * size + 8, size), (24, 24, 24))
    canvas.paste(left, (0, 0))
    canvas.paste(Image.fromarray(over), (size + 8, 0))
    return canvas


def v2_gaps(analysis, cond, key):
    return tifffile.imread(os.path.join(analysis, 'v2_gaps', cond, f'{key}_v2_gaps.tif')) > 0


def matched_labels(a, b):
    """Labels of ``a`` and of ``b`` that have a one-to-one partner at IoU >= 0.5."""
    ga, pb, inter, aa, ab = se.overlap(a, b)
    r, c = se.match(se.iou_matrix(inter, aa, ab), 0.5)
    return set(ga[r].tolist()), set(pb[c].tolist())


def v2_enlarged(cells2):
    post = an.with_posterior(cells2)       # same mixture fit as the analysis summary
    return post[post.posterior > 0.5]


def pick(df, n, rng):
    return df if len(df) <= n else df.iloc[rng.choice(len(df), n, replace=False)]


def build(args):
    rng = np.random.default_rng(args.seed)
    cells1 = pd.read_csv(os.path.join(args.analysis, 'cells_v1.csv'))
    cells2 = pd.read_csv(os.path.join(args.analysis, 'cells_v2.csv'))
    calls = load_classes(sorted(glob.glob(os.path.join(
        args.root, 'Senescence', '*', 'Senescence_Results', 'cell_classification_rule_based_full.csv'))))
    calls = calls[calls.cell_type == 'Senescent'][['key', 'label', 'rule_based_classification_granular']]
    enlarged2 = v2_enlarged(cells2)

    # Pass 1: per-field agreement and gap components -> candidate tables.
    cand, gaps = [], []
    for cond in args.conditions:
        v1c = an.index(f'{args.root}/Segmented/{cond}/Cell_merged_conservative')
        hd, hp = an.V1_HOLES[cond]
        v1h = an.index(f'{args.root}/Segmented/{cond}/{hd}', hp)
        v2c = an.index(f'{args.v2}/{cond}', '*_v2_cells.tif')
        for k in sorted(set(v1c) & set(v2c)):
            um = ft.pixel_um(k)
            c1, c2 = tifffile.imread(v1c[k]), tifffile.imread(v2c[k])
            m1, m2 = matched_labels(c1, c2)
            for tag, m in (('v1', m1), ('v2', m2)):
                cand.append(pd.DataFrame({'key': k, 'folder': cond, 'method': tag, 'matched': sorted(m)}))
            h1 = tifffile.imread(v1h[k]) > 0 if k in v1h else np.zeros(c1.shape, bool)
            for tag, h in (('v1', h1), ('v2', v2_gaps(args.analysis, cond, k))):
                lab = cc_label(h, connectivity=2)
                area = np.bincount(lab.ravel()) * um ** 2
                for comp in np.nonzero(area >= an.GAP_MIN_UM2)[0]:
                    if comp:
                        gaps.append(dict(key=k, folder=cond, method=tag, component=int(comp), area_um2=area[comp]))
    matched = pd.concat(cand, ignore_index=True)
    gaps = pd.DataFrame(gaps)
    folder_of = matched.drop_duplicates('key').set_index('key').folder

    items = []
    for tag, cells in (('v1', cells1), ('v2', cells2)):
        inner = cells[~cells.touches_border & cells.key.isin(folder_of.index)].copy()
        mset = set(map(tuple, matched[matched.method == tag][['key', 'matched']].to_numpy().tolist()))
        inner['stratum'] = ['matched' if (k, lab) in mset else 'unmatched'
                            for k, lab in zip(inner.key, inner.label)]
        for cond, d in inner.groupby(inner.key.map(folder_of)):
            for stratum, s in d.groupby('stratum'):
                for _, r in pick(s, N_CELL, rng).iterrows():
                    items.append(dict(block='cell', method=tag, folder=cond, key=r.key, label=int(r.label),
                                      stratum=stratum, stratum_share=len(s) / len(d), stratum_n=len(s),
                                      area_um2=r.area_um2, question=Q_CELL))
    sen1 = calls.merge(cells1[~cells1.touches_border][['key', 'label', 'area_um2']], on=['key', 'label'])
    for tag, d in (('v1', sen1), ('v2', enlarged2)):
        d = d[d.key.isin(folder_of.index)]
        for cond, g in d.groupby(d.key.map(folder_of)):
            for _, r in pick(g, N_ENLARGED, rng).iterrows():
                items.append(dict(block='enlarged', method=tag, folder=cond, key=r.key, label=int(r.label),
                                  stratum=r.get('rule_based_classification_granular', 'posterior>0.5'),
                                  area_um2=r.area_um2, question=Q_CELL))
    for (tag, cond), g in gaps.groupby(['method', 'folder']):
        for _, r in pick(g, N_GAP, rng).iterrows():
            items.append(dict(block='gap', method=tag, folder=cond, key=r.key, label=int(r.component),
                              stratum='component', area_um2=r.area_um2, question=Q_GAP))

    # Pass 2: render, one field at a time.
    todo = pd.DataFrame(items)
    rendered = []
    for (cond, k), g in todo.groupby(['folder', 'key']):
        proj = projections(args.root, cond)[k]
        rgb = composite(tifffile.imread(proj['Cadherins']), tifffile.imread(proj['Nuclei']))
        um = ft.pixel_um(k)
        c1 = tifffile.imread(an.index(f'{args.root}/Segmented/{cond}/Cell_merged_conservative')[k])
        c2 = tifffile.imread(an.index(f'{args.v2}/{cond}', '*_v2_cells.tif')[k])
        hd, hp = an.V1_HOLES[cond]
        hp1 = an.index(f'{args.root}/Segmented/{cond}/{hd}', hp)
        gap_lab = {'v1': cc_label(tifffile.imread(hp1[k]) > 0, connectivity=2) if k in hp1 else None,
                   'v2': cc_label(v2_gaps(args.analysis, cond, k), connectivity=2)}
        for _, r in g.iterrows():
            if r.block == 'gap':
                mask = gap_lab[r.method] == r.label
            else:
                mask = (c1 if r.method == 'v1' else c2) == r.label
            it = r.to_dict()
            it['id'] = f"{r.block}:{r.method}:{k}:{r.label}"
            it['image'] = render(rgb, mask, um)
            rendered.append(it)
    page = vs.write_session(rendered, args.out, title='Segmentation verdicts', seed=args.seed, embed=True)
    print(f"{len(rendered)} items -> {page}")
    print(todo.groupby(['block', 'method']).size().to_string())


def score(args):
    key = vs.read_key(args.session)
    ver = vs.decode(args.code, len(key)) if args.code else pd.read_csv(args.verdicts, dtype=str).fillna('')
    d = key.merge(ver.rename(columns={'id': 'verdict_id'}), on='verdict_id')
    d = d[d.answer.isin(['yes', 'no'])].assign(yes=lambda x: x.answer.eq('yes'))
    print(f"{len(d)} yes/no answers ({(ver.answer == 'unsure').sum()} unsure, "
          f"{(~ver.answer.isin(['yes', 'no', 'unsure'])).sum()} unanswered)\n")
    for by in (['block', 'method'], ['block', 'method', 'stratum'], ['block', 'method', 'folder']):
        rows = []
        for g_key, g in d.groupby(by):
            p, lo, hi = wilson(int(g.yes.sum()), len(g))
            rows.append(dict(zip(by, g_key), n=len(g), yes_rate=p, lo=lo, hi=hi))
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.2f}") + '\n')
    print('\nCell precision: each stratum weighted by its share of the method\'s interior cells; posterior')
    print('median and 95 % interval from Jeffreys Beta draws per stratum (strata without answers dropped):')
    for by, label in ((('stratum',), 'agreement strata, conditions pooled'),
                      (('folder', 'stratum'), 'condition x agreement strata (sensitivity)')):
        for method, est, lo, hi, used in cell_precision(key, d, by=by):
            print(f"  {method}: {est:.3f} [{lo:.3f}, {hi:.3f}]  ({label}; {used} strata)")
    print('\nGap area confirmed:')
    for method, est, lo, hi, n in gap_area_precision(key, d):
        print(f"  {method}: {est:.2f} [{lo:.2f}, {hi:.2f}]  (n={n})")


def cell_precision(key, d, block='cell', by=('stratum',), n_draw=20000, seed=0):
    """Stratified precision per method: sum over strata of W * p, W the stratum's share of
    the method's interior cells (population counts from the key), p ~ Beta(k + 1/2, n - k + 1/2)
    (Jeffreys). Returns the posterior median and 95 % interval; strata without yes/no answers
    are dropped and the weights renormalised."""
    rng = np.random.default_rng(seed)
    by = list(by)
    out = []
    for method, g in d[d.block == block].groupby('method'):
        pop = (key[(key.block == block) & (key.method == method)]
               .drop_duplicates(['folder', 'stratum']).groupby(by).stratum_n.sum())
        cells = g.groupby(by).yes.agg(['sum', 'count'])
        cells = cells[cells['count'] > 0]
        w = pop.reindex(cells.index).astype(float).to_numpy()
        w = w / w.sum()
        k, n = cells['sum'].to_numpy(float), cells['count'].to_numpy(float)
        draws = rng.beta(k + 0.5, n - k + 0.5, size=(n_draw, len(k))) @ w
        lo, med, hi = np.percentile(draws, [2.5, 50, 97.5])
        out.append((method, float(med), float(lo), float(hi), len(k)))
    return out


def gap_area_precision(key, d, n_draw=20000, seed=0):
    """Share of gap area confirmed. Uniform samples: answers weighted by component area.
    Area-proportional samples drawn with replacement ('mult' = times drawn): the yes-rate
    within a folder estimates that folder's confirmed area share, and folders are combined
    with weights equal to their total gap area ('folder_area')."""
    rng = np.random.default_rng(seed)
    out = []
    for method, g in d[d.block == 'gap'].groupby('method'):
        if 'sampling' in g and g.sampling.eq('pps').all():
            mult = g['mult'].fillna(1).astype(float) if 'mult' in g else pd.Series(1.0, index=g.index)
            parts = []
            for folder, x in g.assign(mult=mult).groupby('folder'):
                k, n = float((x.yes * x.mult).sum()), float(x.mult.sum())
                parts.append((float(x.folder_area.iloc[0]), k, n))
            w = np.array([a for a, _, _ in parts])
            w = w / w.sum()
            k = np.array([p[1] for p in parts])
            n = np.array([p[2] for p in parts])
            draws = rng.beta(k + 0.5, n - k + 0.5, size=(n_draw, len(k))) @ w
            lo, est, hi = np.percentile(draws, [2.5, 50, 97.5])
            out.append((method, float(est), float(lo), float(hi), int(n.sum())))
        else:
            est = float((g.yes * g.area_um2).sum() / g.area_um2.sum())
            out.append((method, est, np.nan, np.nan, len(g)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('--root', required=True)
    b.add_argument('--v2', required=True)
    b.add_argument('--analysis', required=True, help='output folder of analyze.py (cells_v1.csv, cells_v2.csv)')
    b.add_argument('--out', required=True)
    b.add_argument('--conditions', nargs='+', default=an.CONDS)
    b.add_argument('--seed', type=int, default=2026)
    s = sub.add_parser('score')
    s.add_argument('--session', required=True, help='session folder, or its index.html')
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument('--verdicts')
    g.add_argument('--code')
    args = ap.parse_args(argv)
    build(args) if args.cmd == 'build' else score(args)


if __name__ == '__main__':
    main()

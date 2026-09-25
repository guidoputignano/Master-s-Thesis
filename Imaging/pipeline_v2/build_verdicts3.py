#!/usr/bin/env python3
"""Third blind session: the two masks no review has checked yet.

Rounds 1 and 2 (build_verdicts.py, build_verdicts2.py) confirmed the cells, the enlarged
cells and the seed gaps. Two masks behind reported numbers were never shown:

* ``gap``: the grown v2.1 gaps (refine_v2.py grows each seed over the connected area
  below the 5th percentile). The question is whether all of the outlined area is bare
  substrate, so an overshoot into cytoplasm counts as "no". Components are drawn with
  replacement, with probability proportional to area, within each shear stress; the
  multiplicity-weighted yes-rate estimates the share of that shear stress's grown gap
  area that is real.
* ``multinucleated``: v1 cells with two or more nuclei. Round 2 confirmed that each is a
  single cell; here the question is whether the marked nuclei are that many separate,
  whole nuclei, with no other nucleus in the cell. It checks the nuclei count behind the
  multinucleated fractions (3.7 % static, 5.7 % under flow).

Only the clear fields are used (the analysis folder of the quality-filtered run), and the
fields of each shear stress are pooled. Every candidate within 15 um of an object shown in
round 1 or 2 is left out. Views as in round 2: 1 junctions, 2 haze, 3 Golgi.

  python build_verdicts3.py build --root data-mt --v2r v2r2_masks --analysis v2r2_analysis_clean \
      --previous session1/key.csv v2_masks v2_analysis --previous session2/key.csv v2r_masks v2r_analysis \
      --out session3
  python build_verdicts3.py score --session session3/index.html --code 'VS40:...'
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from scipy import ndimage
from skimage.measure import label as cc_label
from skimage.segmentation import find_boundaries

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'Validation'))
sys.path.insert(0, HERE)
import analyze as an  # noqa: E402
import build_verdicts as bv  # noqa: E402
import build_verdicts2 as bv2  # noqa: E402
import features as ft  # noqa: E402
import verdict_session as vs  # noqa: E402
from audit_gallery import wilson  # noqa: E402

N_GAP, N_MULTI = 10, 10               # per shear stress
Q_GAP = 'Is all of the outlined area bare substrate (no cell covers any part of it)?'
NUC_RGB = (255, 40, 40)               # nuclei outlines (red, against cyan nuclei); the cell is yellow
NOTE = ('Round 3: grown gaps and multinucleated cells. Press <kbd>1</kbd> <kbd>2</kbd> <kbd>3</kbd> '
        'to switch views: junctions, haze (cytoplasm grey, bare substrate black), Golgi (yellow). '
        'For a gap, answer N if any part of the outline covers cytoplasm. For a multinucleated cell, '
        'the nuclei are outlined in red: answer N if a marked nucleus is two nuclei or half of one, '
        'or if the cell holds another nucleus. Use U freely when you cannot tell.')


def shear_of(folder):
    return 'Static' if folder.startswith('Static') else '1.4 Pa'


def question_multi(n):
    return (f'The cell (yellow) holds {n} marked nuclei (red). Are they {n} separate, whole nuclei, '
            'with no other nucleus in the cell?')


def previous_zone(root, rounds):
    """{key: mask within 15 um of any object shown in the earlier rounds}.

    ``rounds``: (key.csv, v2 masks folder, v2 gaps analysis folder) per round, the folders
    that round showed (build_verdicts2.round1_zone checks the gap areas against the key)."""
    zones = {}
    for key_csv, masks, gaps in rounds:
        for k, z in bv2.round1_zone(root, key_csv, masks, gaps).items():
            zones[k] = zones[k] | z if k in zones else z
    return zones


def render_multi(rgb, cell_mask, nuc_labels, um, size=380):
    """bv.render with each nucleus of the cell outlined in red over the right panel.

    ``nuc_labels`` is a label image of the cell's nuclei (0 elsewhere): boundaries are drawn
    between labels too, so two touching nuclei show as two outlines."""
    canvas = bv.render(rgb, cell_mask, um, size=size)
    # repeat bv.render's window so the nuclei outlines land on the same pixels
    ys, xs = np.nonzero(cell_mask)
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    pad, half_min = 12.0 * ft.SCALE, 20.0 * ft.SCALE          # 18 and 30 um, as in the rounds
    half = max((ys.max() - ys.min()) // 2 + round(pad / um), (xs.max() - xs.min()) // 2 + round(pad / um),
               round(half_min / um))
    side = min(2 * half + 1, cell_mask.shape[0], cell_mask.shape[1])
    y0 = int(np.clip(cy - half, 0, cell_mask.shape[0] - side))
    x0 = int(np.clip(cx - half, 0, cell_mask.shape[1] - side))
    m = nuc_labels[y0:y0 + side, x0:x0 + side]
    scale = size / side
    wh = (max(1, round(side * scale)), max(1, round(side * scale)))
    idx_y = np.minimum((np.arange(wh[1]) / scale).astype(int), side - 1)     # nearest-neighbour resize
    idx_x = np.minimum((np.arange(wh[0]) / scale).astype(int), side - 1)
    mm = m[idx_y][:, idx_x]
    right = np.array(canvas)[:, size + 8:size + 8 + wh[0]]
    right[:wh[1]][ndimage.binary_dilation(find_boundaries(mm, mode='inner'))] = NUC_RGB
    out = np.array(canvas)
    out[:, size + 8:size + 8 + wh[0]] = right
    return Image.fromarray(out)


def candidates(args, zones):
    """Grown v2.1 gap components and multinucleated v1 cells of the clear fields."""
    cells = an.mixture_cells(pd.read_csv(os.path.join(args.analysis, 'cells_v1.csv')))
    gaps, multi = [], []
    for cond in args.conditions:
        keys = sorted(an.index(f'{args.analysis}/v2_gaps/{cond}', '*_v2_gaps.tif'))
        v1c = an.index(f'{args.root}/Segmented/{cond}/Cell_merged_conservative')
        for k in keys:
            um = ft.pixel_um(k)
            zone = zones.get(k)
            lab = cc_label(bv.v2_gaps(args.analysis, cond, k), connectivity=2)
            area = np.bincount(lab.ravel()) * um ** 2
            near = set(np.unique(lab[zone]).tolist()) if zone is not None else set()
            for comp in np.nonzero(area >= an.GAP_MIN_UM2)[0]:
                if comp and comp not in near:
                    gaps.append(dict(key=k, folder=cond, shear=shear_of(cond), label=int(comp),
                                     area_um2=float(area[comp])))
            c1 = tifffile.imread(v1c[k])
            near_c = set(np.unique(c1[zone]).tolist()) if zone is not None else set()
            d = cells[(cells.key == k) & (cells.n_nuclei >= 2)]
            for _, r in d.iterrows():
                if int(r.label) not in near_c:
                    multi.append(dict(key=k, folder=cond, shear=shear_of(cond), label=int(r.label),
                                      area_um2=float(r.area_um2), nuclei=int(r.n_nuclei)))
    return pd.DataFrame(gaps), pd.DataFrame(multi), cells


def build(args):
    rng = np.random.default_rng(args.seed)
    zones = previous_zone(args.root, args.previous)
    gaps, multi, cells = candidates(args, zones)
    items = []
    for shear, g in gaps.groupby('shear'):
        shear_area = g.area_um2.sum()           # the estimand: grown gap area outside the earlier zones
        draws = rng.choice(len(g), N_GAP, replace=True, p=g.area_um2.to_numpy() / shear_area)
        for i, mult in zip(*np.unique(draws, return_counts=True)):
            r = g.iloc[i]
            items.append(dict(block='gap', method='v2.1 grown', folder=r.folder, shear=shear, key=r.key,
                              label=int(r.label), stratum='component', sampling='pps', mult=int(mult),
                              shear_area=shear_area, area_um2=r.area_um2, question=Q_GAP, view0='haze'))
    for shear, g in multi.groupby('shear'):
        n_all = int(((cells.condition.str.startswith('0Pa') if shear == 'Static'
                      else cells.condition.str.startswith('1.4Pa')) & (cells.n_nuclei >= 2)).sum())
        for _, r in bv.pick(g, N_MULTI, rng).iterrows():
            items.append(dict(block='multinucleated', method='v1', folder=r.folder, shear=shear, key=r.key,
                              label=int(r.label), stratum='multi', nuclei=int(r.nuclei), stratum_n=n_all,
                              area_um2=r.area_um2, question=question_multi(int(r.nuclei)), view0='junctions'))
    todo = pd.DataFrame(items)
    rendered = []
    order = ['junctions', 'haze', 'golgi']
    for (cond, k), g in todo.groupby(['folder', 'key']):
        views = bv2.view_images(args.root, cond, k)
        names = [n for n in order if n in views]
        um = ft.pixel_um(k)
        gap_lab = cc_label(bv.v2_gaps(args.analysis, cond, k), connectivity=2)
        c1 = tifffile.imread(an.index(f'{args.root}/Segmented/{cond}/Cell_merged_conservative')[k])
        n1 = tifffile.imread(an.index(f'{args.root}/Segmented/{cond}/{an.V1_NUCLEI[cond]}')[k])
        owner, _, _ = ft.nuclei_assignment(c1, n1)
        for _, r in g.iterrows():
            it = r.to_dict()
            it['id'] = f"{r.block}:{r.method}:{k}:{r.label}"
            if r.block == 'gap':
                mask = gap_lab == r.label
                it['images'] = [(n, bv.render(views[n], mask, um, size=380)) for n in names]
            else:
                mask = c1 == r.label
                nuc = np.where(np.isin(n1, [n for n, c in owner.items() if c == r.label]), n1, 0)
                it['images'] = [(n, render_multi(views[n], mask, nuc, um)) for n in names]
            rendered.append(it)
    page = vs.write_session(rendered, args.out, title='Segmentation verdicts, round 3', seed=args.seed,
                            embed=True, note=NOTE, quality=80)
    print(f"{len(rendered)} items -> {page}")
    print(todo.groupby(['block', 'shear']).size().to_string())
    print(f"candidates: {len(gaps)} gap components, {len(multi)} multinucleated cells")


def score_table(key, answers, n_draw=200_000, seed=0):
    """Rows per block and shear stress (and pooled): yes-rate or confirmed area share.

    Gaps: the multiplicity-weighted yes-rate per shear stress, Jeffreys-Beta interval, and a
    pooled value weighted by each shear stress's gap area. Multinucleated cells: the yes-rate
    per shear stress (Wilson interval) and a pooled value weighted by the number of
    multinucleated cells. Unsure answers are left out."""
    rng = np.random.default_rng(seed)
    d = key.merge(answers.rename(columns={'id': 'verdict_id'}), on='verdict_id')
    d = d[d.answer.isin(['yes', 'no'])].assign(yes=lambda x: x.answer.eq('yes').astype(float))
    rows = []
    g = d[d.block == 'gap']
    if len(g):
        parts = []
        for shear, x in g.groupby('shear'):
            k, n = float((x.yes * x['mult']).sum()), float(x['mult'].sum())
            draws = rng.beta(k + 0.5, n - k + 0.5, n_draw)
            lo, est, hi = np.percentile(draws, [2.5, 50, 97.5])
            rows.append(dict(block='gap', shear=shear, n=int(n), estimate=est, lo=lo, hi=hi))
            parts.append((float(x.shear_area.iloc[0]), draws))
        w = np.array([a for a, _ in parts])
        pooled = np.stack([dr for _, dr in parts], 1) @ (w / w.sum())
        lo, est, hi = np.percentile(pooled, [2.5, 50, 97.5])
        rows.append(dict(block='gap', shear='pooled by area', n=int(g['mult'].sum()), estimate=est, lo=lo, hi=hi))
    m = d[d.block == 'multinucleated']
    if len(m):
        parts = []
        for shear, x in m.groupby('shear'):
            p, lo, hi = wilson(int(x.yes.sum()), len(x))
            rows.append(dict(block='multinucleated', shear=shear, n=len(x), estimate=p, lo=lo, hi=hi))
            parts.append((float(x.stratum_n.iloc[0]), p))
        w = np.array([a for a, _ in parts])
        rows.append(dict(block='multinucleated', shear='pooled by count', n=len(m),
                         estimate=float(np.dot(w / w.sum(), [p for _, p in parts])), lo=np.nan, hi=np.nan))
    return pd.DataFrame(rows)


def score(args):
    key = vs.read_key(args.session)
    ans = vs.decode(args.code, len(key)) if args.code else pd.read_csv(args.verdicts, dtype=str).fillna('')
    print(f"{ans.answer.isin(['yes', 'no']).sum()} yes/no answers, {(ans.answer == 'unsure').sum()} unsure\n")
    print(score_table(key, ans).to_string(index=False, float_format=lambda v: f"{v:.2f}"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('--root', required=True)
    b.add_argument('--v2r', required=True, help='refine_v2.py output (not read; recorded for provenance)')
    b.add_argument('--analysis', required=True, help='analyze.py output of the clear fields (grown gaps)')
    b.add_argument('--previous', nargs=3, action='append', default=[], metavar=('KEY', 'V2_MASKS', 'V2_GAPS'),
                   help='an earlier round: its key.csv, the v2 masks and the gaps analysis it showed')
    b.add_argument('--out', required=True)
    b.add_argument('--conditions', nargs='+', default=an.CONDS)
    b.add_argument('--seed', type=int, default=2028)
    s = sub.add_parser('score')
    s.add_argument('--session', required=True)
    s.add_argument('--code')
    s.add_argument('--verdicts')
    a = ap.parse_args(argv)
    build(a) if a.cmd == 'build' else score(a)


if __name__ == '__main__':
    main()

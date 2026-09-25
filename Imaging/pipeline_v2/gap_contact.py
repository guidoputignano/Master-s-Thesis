#!/usr/bin/env python3
"""Do gaps sit next to enlarged-nucleus cells more than their size explains?

For each interior v1 cell (at least one Cellpose nucleus, area > 115 um^2, largest nucleus
>= 57 um^2): does it lie within 1.5 um of a v2.1 (grown) gap? Enlarged-nucleus cells are
about three times larger, so they touch more gaps by geometry alone. The reference is the
same gap mask moved to random positions in the same field (circular shifts): the shifted
masks keep the gap shapes and the cell geometry and break any link to the nuclei. The
p-value is the share of shifted draws whose enlarged share of gap-touching cells is at
least the observed one.

Writes gap_contact.pkl (per field: counts and the null draws) and prints, per shear stress
x experiment, the enlarged share of gap-touching cells against the geometric expectation.

  python gap_contact.py --root data-mt --masks v2r2_masks --features features/features_v1.csv --out gap_contact
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'Validation'))
sys.path.insert(0, HERE)
import analyze as an  # noqa: E402
import features as ft  # noqa: E402

ENLARGED_UM2 = 95.7 * ft.AREA     # 219.7 um2: static antimode of the largest Cellpose nucleus, pooled (by_shear.py)
TOUCH_UM = 1.0 * ft.SCALE          # 1.5 um


def contacts(cells, near, enlarged, normal):
    """Numbers of enlarged and normal cells with at least one pixel in ``near``."""
    n = len(enlarged)
    t = np.bincount(cells[near & (cells > 0)], minlength=n) > 0
    return int((t & enlarged).sum()), int((t & normal).sum())


def field_null(cells, gaps, labels, is_enlarged, um, n_shifts, rng):
    """Observed contacts and ``n_shifts`` circularly shifted draws for one field."""
    near = ndimage.binary_dilation(gaps, iterations=max(1, int(round(TOUCH_UM / um))))
    n = int(cells.max()) + 1
    enl, norm = np.zeros(n, bool), np.zeros(n, bool)
    enl[labels[is_enlarged]] = True
    norm[labels[~is_enlarged]] = True
    obs = contacts(cells, near, enl, norm)
    null = np.array([contacts(cells, np.roll(near, (rng.integers(0, cells.shape[0]), rng.integers(0, cells.shape[1])),
                                              axis=(0, 1)), enl, norm) for _ in range(n_shifts)])
    return obs, null, int(enl.sum()), int(norm.sum())


def summarise(r, n_shifts):
    rows = []
    r = r.assign(shear=np.where(r.condition.str.startswith('0Pa'), 'Static', '1.4 Pa'),
                 date=r.key.str.split('_').str[2])
    for name, sel in (('Static', r.shear == 'Static'),
                      ('1.4 Pa, 19dec21', (r.shear == '1.4 Pa') & (r.date == '19dec21')),
                      ('1.4 Pa, 20dec21', (r.shear == '1.4 Pa') & (r.date == '20dec21'))):
        g = r[sel & r.n_enl.notna()]
        tot = np.sum(np.stack(g.null.to_numpy()), axis=0)
        oe, on = g.obs_enl.sum(), g.obs_norm.sum()
        share = oe / max(oe + on, 1)
        share_null = tot[:, 0] / np.maximum(tot.sum(1), 1)
        rows.append(dict(slides=name, fields=len(g), touching_enlarged=int(oe), touching_normal=int(on),
                         enlarged_share=share, expected=float(np.median(share_null)),
                         p=(np.sum(share_null >= share) + 1) / (n_shifts + 1)))
    out = pd.DataFrame(rows)
    for sh, g in r.groupby('shear'):
        rho = spearmanr(g.enl_frac, g.gap_frac)
        print(f'per field, {sh}: gap fraction vs enlarged fraction rho = {rho.correlation:.2f} (p = {rho.pvalue:.2f})')
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--root', required=True, help='data repository (Segmented/<c>/Cell_merged_conservative)')
    ap.add_argument('--masks', required=True, help='refine_v2.py output: <cond>/<key>_v2_gaps.tif')
    ap.add_argument('--features', required=True, help='features_v1.csv from cell_features.py')
    ap.add_argument('--out', required=True)
    ap.add_argument('--threshold', type=float, default=ENLARGED_UM2, help='enlarged nucleus, um^2')
    ap.add_argument('--shifts', type=int, default=200)
    ap.add_argument('--seed', type=int, default=7)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    d = pd.read_csv(a.features)
    d = d[~d.touches_border.astype(bool) & (d.cp_n_nuclei >= 1) & (d.area_um2 > 50 * ft.AREA)
          & (d.cp_nuc_largest_um2 >= 25 * ft.AREA)].copy()
    d['enlarged'] = d.cp_nuc_largest_um2 > a.threshold
    res = []
    for (cond, key), g in d.groupby(['folder', 'key']):
        C = tifffile.imread(an.index(f'{a.root}/Segmented/{cond}/Cell_merged_conservative')[key])
        G = tifffile.imread(an.index(f'{a.masks}/{cond}', '*_v2_gaps.tif')[key]) > 0
        row = dict(condition=g.condition.iloc[0], key=key, gap_frac=G.mean(), enl_frac=g.enlarged.mean())
        if G.any():
            obs, null, ne, nn = field_null(C, G, g.label.to_numpy(), g.enlarged.to_numpy(), ft.pixel_um(key), a.shifts, rng)
            row.update(n_enl=ne, n_norm=nn, obs_enl=obs[0], obs_norm=obs[1], null=null)
        res.append(row)
    r = pd.DataFrame(res)
    r.to_pickle(f'{a.out}/gap_contact.pkl')
    s = summarise(r, a.shifts)
    s.to_csv(f'{a.out}/gap_contact.csv', index=False)
    print(s.round(3).to_string(index=False))


if __name__ == '__main__':
    main()

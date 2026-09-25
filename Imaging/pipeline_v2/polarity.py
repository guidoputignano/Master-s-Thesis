#!/usr/bin/env python3
"""Nucleus-to-Golgi polarity: which way the flow ran, without the file names.

Under laminar flow, endothelial cells move the Golgi upstream of the nucleus. Chala et al.
(2021) found this counterflow polarisation in control and in TNF-alpha-senescent monolayers
alike, although the senescent cells did not align. Averaged over the cells of a slide, the
side on which the Golgi sits therefore gives the flow direction, even where the cell bodies
did not align.

* **Per cell.** The Golgi is the part of the cell brighter than a per-field Otsu threshold
  on the log Golgi top-hat projection (pixels inside cells). The polarity vector runs from
  the centroid of the cell's largest nucleus (a nucleus belongs to the cell covering more
  than half of it) to the intensity-weighted Golgi centroid. Kept: interior cells over
  115 um^2 with a nucleus and a vector longer than 1.5 um.
* **Per group** (condition x experiment, and per field): the mean resultant length R of the
  unit vectors (0 = no preferred side, 1 = every Golgi on the same side), its direction in
  degrees counter-clockwise from the image +x axis (image y points down, so 180 = left)
  and the Rayleigh p-value. An offset between the Golgi and nuclear channels would add the
  same vector to every cell, so the group statistics are repeated after subtracting the
  mean vector of the static slide of the same experiment and magnification.

Writes polarity_cells.csv (per cell: keep it private), polarity_fields.csv and polarity.csv.

  python polarity.py --root data-mt --v2 v2r_masks --exclude-fields quality.csv --out polarity
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import tifffile
from skimage.filters import threshold_otsu

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'Validation'))
sys.path.insert(0, HERE)
import seg_eval as se  # noqa: E402
import features as ft  # noqa: E402
import analyze as an  # noqa: E402

MIN_CELL_UM2 = 50.0 * ft.AREA     # 115 um2 (features.SCALE)
MIN_VECTOR_UM = 1.0 * ft.SCALE     # 1.5 um


def golgi_vectors(cells, nuclei, golgi, um):
    """Per interior cell: label and nucleus -> Golgi vector in um (x to the right, y down)."""
    cells, nuclei = np.asarray(cells), np.asarray(nuclei)
    g = np.asarray(golgi, float)
    empty = pd.DataFrame(columns=['label', 'dx_um', 'dy_um'])
    inside = cells > 0
    vals = g[inside]
    vals = vals[vals > 0]
    if len(vals) < 100:
        return empty
    thr = np.exp(threshold_otsu(np.log(vals)))
    lab = np.where((g > thr) & inside, cells, 0).ravel()
    n = int(cells.max()) + 1
    yy, xx = np.indices(cells.shape)
    w = np.bincount(lab, weights=g.ravel(), minlength=n)
    gy = np.bincount(lab, weights=(g * yy).ravel(), minlength=n)
    gx = np.bincount(lab, weights=(g * xx).ravel(), minlength=n)
    m = int(nuclei.max()) + 1
    cnt = np.maximum(np.bincount(nuclei.ravel(), minlength=m), 1)
    ny = np.bincount(nuclei.ravel(), weights=yy.ravel(), minlength=m) / cnt
    nx = np.bincount(nuclei.ravel(), weights=xx.ravel(), minlength=m) / cnt
    owner, n_ids, n_area = ft.nuclei_assignment(cells, nuclei)
    largest = {}
    for nid, a in zip(n_ids.tolist(), n_area.tolist()):
        c = owner.get(nid)
        if c is not None and a > largest.get(c, (0, 0))[0]:
            largest[c] = (a, nid)
    area = np.bincount(cells.ravel(), minlength=n) * um ** 2
    edge = se.border_ids(cells)
    rows = [(c, (gx[c] / w[c] - nx[nid]) * um, (gy[c] / w[c] - ny[nid]) * um)
            for c, (_, nid) in sorted(largest.items())
            if c not in edge and w[c] > 0 and area[c] > MIN_CELL_UM2]
    if not rows:
        return empty
    v = pd.DataFrame(rows, columns=['label', 'dx_um', 'dy_um'])
    return v[np.hypot(v.dx_um, v.dy_um) > MIN_VECTOR_UM].reset_index(drop=True)


def rayleigh_p(n, r):
    """Rayleigh test of uniform direction (Zar 1999, eq. 27.4); r = mean resultant length."""
    if n == 0:
        return np.nan
    rn = n * r
    return float(min(1.0, np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - rn ** 2)) - (1 + 2 * n))))


def mean_direction(dx, dy):
    """(n, R, direction in degrees counter-clockwise from +x with y up, Rayleigh p)."""
    dx, dy = np.asarray(dx, float), np.asarray(dy, float)
    if dx.size == 0:
        return 0, np.nan, np.nan, np.nan
    z = np.exp(1j * np.arctan2(-dy, dx)).mean()
    return dx.size, float(abs(z)), float(np.rad2deg(np.angle(z)) % 360), rayleigh_p(dx.size, abs(z))


def summarise(cells, by=('condition', 'date')):
    """Polarity per group, with and without the static slide's mean vector removed."""
    rows = []
    for _, g in cells.groupby(list(by)):
        row = {c: g[c].iloc[0] for c in dict.fromkeys([*by, 'condition', 'date'])}
        mag = row['condition'].split('_')[-1]
        ref = cells[cells.condition.str.startswith('0Pa') & cells.condition.str.endswith(mag)
                    & (cells.date == row['date'])]
        ox, oy = (ref.dx_um.mean(), ref.dy_um.mean()) if len(ref) else (0.0, 0.0)
        n, r, ang, p = mean_direction(g.dx_um, g.dy_um)
        _, rc, angc, pc = mean_direction(g.dx_um - ox, g.dy_um - oy)
        row.update(cells=n, R=r, R_random=0.886 / np.sqrt(n), direction_deg=ang, rayleigh_p=p,
                   golgi_left_share=float((g.dx_um < 0).mean()), offset_x_um=ox, offset_y_um=oy,
                   R_corrected=rc, direction_corrected_deg=angc, rayleigh_p_corrected=pc)
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--root', required=True, help='data repository (Projection/<c>/Golgi, Segmented/<c>/...)')
    ap.add_argument('--v2', required=True, help='segment_v2.py or refine_v2.py output (Cellpose nuclei)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--exclude-fields', help='quality.py table; fields marked exclude are left out')
    ap.add_argument('--conditions', nargs='+', default=an.CONDS)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    skip = an.excluded_fields(args.exclude_fields)
    parts = []
    for cond in args.conditions:
        cells = an.index(f'{args.root}/Segmented/{cond}/Cell_merged_conservative')
        nuclei = an.index(f'{args.v2}/{cond}', '*_v2_nuclei.tif')
        golgi = an.index(f'{args.root}/Projection/{cond}/Golgi/tophat')
        for k in sorted((set(cells) & set(nuclei) & set(golgi)) - skip):
            v = golgi_vectors(tifffile.imread(cells[k]), tifffile.imread(nuclei[k]),
                              tifffile.imread(golgi[k]), ft.pixel_um(k))
            v.insert(0, 'key', k)
            v.insert(1, 'condition', se.condition_of(k))
            v.insert(2, 'date', an.date_of(k))
            parts.append(v)
            print(f'{cond} {k}: {len(v)} cells', flush=True)
    cells = pd.concat(parts, ignore_index=True)
    cells.to_csv(f'{args.out}/polarity_cells.csv', index=False)
    summarise(cells, by=('key',)).to_csv(f'{args.out}/polarity_fields.csv', index=False)
    groups = summarise(cells)
    groups.to_csv(f'{args.out}/polarity.csv', index=False)
    print(groups.to_string(index=False, float_format=lambda x: f'{x:.3g}'))


if __name__ == '__main__':
    main()

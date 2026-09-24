#!/usr/bin/env python3
"""Per-cell senescence features for v1 and v2.1 cells on the clear fields.

Adds to the per-cell tables of ``analyze.py`` (cells_v1.csv, cells_v2.csv):

* Cellpose nuclei per cell (a nucleus belongs to the cell covering more than half of it):
  count, total and largest area, background-subtracted DAPI of the projection (field
  normalised), shape of the largest nucleus. v1's own nuclear masks are inflated at
  0.43 um/px, so every nuclear feature uses the Cellpose nuclei, for both segmentations.
* Golgi: area, fragments and intensity above a per-field Otsu threshold on the log Golgi
  top-hat, and the nucleus-to-Golgi vector.
* Junction channel (beta-catenin, folder "Cadherins"): mean at the cell boundary and in the
  interior, field normalised (columns ``vecad_*``, named before the stain was known to be
  beta-catenin); number of neighbours.
* The cell-area mixture posterior of ``analyze.py`` (cells_posterior.csv).

Writes features_v1.csv and features_v2.1.csv (per cell: keep them private).

  python cell_features.py --root data-mt --masks v2r2_masks --analysis v2r2_analysis_clean \\
      --quality quality.csv --out features
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.measure import label as cc_label, regionprops_table
from skimage.segmentation import find_boundaries

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'Validation'))
sys.path.insert(0, HERE)
import analyze as an  # noqa: E402
import features as ft  # noqa: E402
import segment_v2 as sg  # noqa: E402

CORE_UM = 2.0           # cell interior = farther than this from any cell boundary
MIN_FRAGMENT_UM2 = 1.0  # smallest Golgi fragment counted


def per_label_sum(lab, img, n):
    return np.bincount(lab.ravel(), weights=np.asarray(img, float).ravel(), minlength=n)


def neighbours(cells):
    """Number of touching cells (8-connectivity) per label."""
    n = int(cells.max()) + 1
    pairs = set()
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        h, w = cells.shape
        a = cells[:h - dy, max(0, -dx):w - max(0, dx)]
        b = cells[dy:, max(0, dx):w + min(0, dx)]
        m = (a != b) & (a > 0) & (b > 0)
        pairs |= set(zip(np.minimum(a[m], b[m]).tolist(), np.maximum(a[m], b[m]).tolist()))
    nb = np.zeros(n, int)
    for i, j in pairs:
        nb[i] += 1
        nb[j] += 1
    return nb


def field_features(cells, nuclei, nuc_img, golgi, junction, um):
    """Per-cell features of one field (images as arrays, ``um`` = pixel size)."""
    C, N = np.asarray(cells), np.asarray(nuclei)
    nuc, G, V = (np.asarray(a, float) for a in (nuc_img, golgi, junction))
    a2 = um ** 2
    nC = int(C.max()) + 1
    owner, _, _ = ft.nuclei_assignment(C, N)
    outside = (N == 0) & (C > 0)
    bg = np.median(nuc[outside]) if outside.any() else np.median(nuc)
    props = pd.DataFrame(regionprops_table(N, intensity_image=nuc, properties=(
        'label', 'area', 'centroid', 'major_axis_length', 'minor_axis_length', 'solidity', 'intensity_mean')))
    props['area_um2'] = props.area * a2
    props['dapi_int'] = (props.intensity_mean - bg) * props.area
    props['nuc_ar'] = props.major_axis_length / props.minor_axis_length.clip(lower=1e-6)
    props['cell'] = props.label.map(owner)
    props = props.dropna(subset=['cell'])
    props['cell'] = props.cell.astype(int)
    pos = props.dapi_int > 0
    props['dapi_int_n'] = props.dapi_int / (np.median(props.dapi_int[pos]) if pos.any() else 1.0)
    props['dapi_mean_n'] = (props.intensity_mean - bg) / np.median(props.intensity_mean - bg)
    g = props.sort_values('area', ascending=False).groupby('cell')
    nuc_agg = pd.DataFrame({'cp_n_nuclei': g.size(), 'cp_nuc_total_um2': g.area_um2.sum(),
                            'cp_nuc_largest_um2': g.area_um2.first(), 'dapi_total_n': g.dapi_int_n.sum(),
                            'dapi_mean_largest_n': g.dapi_mean_n.first(), 'nuc_ar_largest': g.nuc_ar.first(),
                            'nuc_solidity_largest': g.solidity.first(),
                            'nuc_cy': g['centroid-0'].first(), 'nuc_cx': g['centroid-1'].first()})
    inside = C > 0
    gv = G[inside]
    gv = gv[gv > 0]
    thr = np.exp(threshold_otsu(np.log(gv))) if len(gv) > 100 else np.inf
    gm = (G > thr) & inside
    glab = np.where(gm, C, 0)
    area_g = np.bincount(C[gm].ravel(), minlength=nC) * a2
    int_g = per_label_sum(glab, G, nC)
    yy, xx = np.indices(C.shape)
    wy, wx = per_label_sum(glab, G * yy, nC), per_label_sum(glab, G * xx, nC)
    frag = cc_label(gm, connectivity=2)
    fr = pd.DataFrame({'frag': frag[gm], 'cell': C[gm]}).drop_duplicates()
    fsize = np.bincount(frag.ravel()) * a2
    nfrag = fr[fsize[fr.frag.to_numpy()] >= MIN_FRAGMENT_UM2].groupby('cell').size()
    bd = find_boundaries(C, mode='inner') & inside
    core = (ndimage.distance_transform_edt(~find_boundaries(C, mode='thick')) * um > CORE_UM) & inside
    jn, cn = np.bincount(C[bd], minlength=nC), np.bincount(C[core], minlength=nC)
    jmean = np.where(jn > 0, np.bincount(C[bd], weights=V[bd], minlength=nC) / np.maximum(jn, 1), np.nan)
    cmean = np.where(cn > 0, np.bincount(C[core], weights=V[core], minlength=nC) / np.maximum(cn, 1), np.nan)
    labs = np.arange(1, nC)
    out = pd.DataFrame({'label': labs, 'golgi_area_um2': area_g[labs], 'golgi_int': int_g[labs],
                        'golgi_cy': np.where(int_g[labs] > 0, wy[labs] / np.maximum(int_g[labs], 1e-9), np.nan),
                        'golgi_cx': np.where(int_g[labs] > 0, wx[labs] / np.maximum(int_g[labs], 1e-9), np.nan),
                        'golgi_fragments': nfrag.reindex(labs).fillna(0).astype(int).to_numpy(),
                        'vecad_junction_n': jmean[labs] / np.nanmedian(jmean[1:]),
                        'vecad_interior_n': cmean[labs] / np.nanmedian(cmean[1:]),
                        'neighbours': neighbours(C)[labs]})
    pos = out.golgi_int > 0
    out['golgi_int_n'] = out.golgi_int / np.nanmedian(out.golgi_int[pos]) if pos.any() else np.nan
    out = out.merge(nuc_agg, left_on='label', right_index=True, how='left')
    out['cp_n_nuclei'] = out.cp_n_nuclei.fillna(0).astype(int)
    out['golgi_dx_um'] = (out.golgi_cx - out.nuc_cx) * um       # image axes: x to the right, y down
    out['golgi_dy_um'] = (out.golgi_cy - out.nuc_cy) * um
    return out.drop(columns=['golgi_int'])


def run(root, masks, analysis, quality, out):
    os.makedirs(out, exist_ok=True)
    skip = an.excluded_fields(quality)
    folder = pd.read_csv(f'{analysis}/fields.csv').set_index('key').folder
    post = pd.read_csv(f'{analysis}/cells_posterior.csv')
    for method, fn, post_tag in (('v1', 'cells_v1.csv', 'v1'), ('v2.1', 'cells_v2.csv', 'v2')):
        base = pd.read_csv(f'{analysis}/{fn}')
        rows = []
        for key in sorted(set(base.key) - skip):
            cond = folder[key]
            if method == 'v1':
                C = tifffile.imread(an.index(f'{root}/Segmented/{cond}/Cell_merged_conservative')[key])
            else:
                C = tifffile.imread(an.index(f'{masks}/{cond}', '*_v2_cells.tif')[key])
            N = tifffile.imread(an.index(f'{masks}/{cond}', '*_v2_nuclei.tif')[key])
            _, nuc_path = sg.inputs(root, cond)[key]
            G = tifffile.imread(an.index(f'{root}/Projection/{cond}/Golgi/tophat')[key])
            V = tifffile.imread(an.index(f'{root}/Projection/{cond}/Cadherins/background')[key])
            f = field_features(C, N, tifffile.imread(nuc_path), G, V, ft.pixel_um(key))
            f['key'], f['folder'] = key, cond
            rows.append(f)
            print(method, cond, key, len(f), flush=True)
        d = base.merge(pd.concat(rows, ignore_index=True), on=['key', 'label'], how='left')
        p = post[post.method == post_tag][['key', 'label', 'posterior']]
        d = d.merge(p, on=['key', 'label'], how='left')
        d['method'], d['date'] = method, d.key.str.split('_').str[2]
        d.to_csv(f'{out}/features_{method}.csv', index=False)
        print(method, 'cells', len(d), 'fields', d.key.nunique())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--root', required=True, help='data repository (Projection/, Segmented/)')
    ap.add_argument('--masks', required=True, help='refine_v2.py output: <cond>/<key>_v2_cells.tif, _v2_nuclei.tif')
    ap.add_argument('--analysis', required=True, help='analyze.py output on the clear fields')
    ap.add_argument('--quality', help='quality.py table; fields marked exclude are left out')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    run(a.root, a.masks, a.analysis, a.quality, a.out)


if __name__ == '__main__':
    main()

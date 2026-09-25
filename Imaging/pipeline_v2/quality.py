#!/usr/bin/env python3
"""Field quality: which projections are clear enough to analyse, and which are repeats.

Two checks. Neither uses a segmentation.

* **Junction clarity.** The β-catenin top-hat projection (the image both segmentations
  use) is filtered for bright ridges at the junction scale: -lambda_min of the Hessian at
  sigma = 0.91 um, scale-normalised by sigma^2. The field's score is the 95th percentile of
  that ridge strength divided by the image noise (robust MAD of the Laplacian residual,
  Immerkaer 1996). Defocus and haze lower it, because the junction line gets wider and
  fainter against the noise. A field is **low quality** when its log score lies more than
  3 robust SDs (1.4826 x MAD) below the median of the fields of the same condition and
  magnification. Comparing within a condition keeps flow-induced junction changes out of
  the rule.
* **Repeated fields.** Two fields of the same condition and magnification that image the
  same area: phase correlation of the β-catenin projections, overlap >= 50 % with
  correlation >= 0.9 inside the overlap. One of the pair is kept (the higher score), so
  no cell is counted twice.

The nuclear contrast (nucleus over a ring 2.3-4.5 um outside it, median over nuclei) is
reported as a diagnostic, not used by the rule: haze and defocus lower it too, while
flow-induced junction changes do not.

Writes ``quality.csv``: key, folder, date, ridge_snr, z, nuclear_contrast, nuclear_z,
low_quality, duplicate_of, exclude. ``analyze.py --exclude-fields quality.csv`` and
``build_verdicts.py score --exclude-fields quality.csv`` then leave the excluded fields out.

  python quality.py --root data-mt --v2 v2r_masks --out quality.csv
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.registration import phase_cross_correlation
from skimage.transform import resize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'Validation'))
sys.path.insert(0, HERE)
import features as ft  # noqa: E402
import analyze as an  # noqa: E402
import segment_v2 as sg  # noqa: E402

RIDGE_SIGMA_UM = 0.6 * ft.SCALE     # 0.91 um junction half-width scale (features.SCALE)
RIDGE_PCT = 95.0         # junction pixels are a few % of a field
Z_CUT = -3.0             # robust z below which a field is low quality
DUP_OVERLAP = 0.5
DUP_CORR = 0.9
DUP_SIZE = 512           # fields are compared at this size
LAPLACIAN = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], float)


def noise_sigma(img):
    """Robust noise SD: MAD of the Laplacian residual (its SD is 6 sigma for white noise)."""
    r = ndimage.convolve(np.asarray(img, float), LAPLACIAN, mode='reflect')
    return float(1.4826 * np.median(np.abs(r - np.median(r))) / 6.0)


def ridge_snr(cad, um, sigma_um=RIDGE_SIGMA_UM, pct=RIDGE_PCT):
    """High percentile of bright-ridge strength at the junction scale, over the noise SD."""
    img = np.asarray(cad, float)
    s = sigma_um / um
    h = hessian_matrix(img, sigma=s, order='rc', use_gaussian_derivatives=True)
    lam = hessian_matrix_eigvals(h)[1]                  # the smaller eigenvalue: across a bright ridge
    ridge = np.clip(-lam, 0, None) * s ** 2
    return float(np.percentile(ridge, pct) / max(noise_sigma(img), 1e-12))


def nuclear_contrast(nuc, nuclei, um):
    """Median over nuclei of (inside - ring) / ring, ring 1.5-3 um outside the nucleus."""
    img = np.asarray(nuc, float)
    inner = ndimage.binary_erosion(nuclei > 0, iterations=max(1, round(1.0 * ft.SCALE / um))) & (nuclei > 0)
    dist, (iy, ix) = ndimage.distance_transform_edt(nuclei == 0, return_indices=True)
    ring = (dist * um > 1.5 * ft.SCALE) & (dist * um <= 3.0 * ft.SCALE)      # 2.3-4.5 um
    near = nuclei[iy, ix]
    i_in = pd.Series(img[inner]).groupby(nuclei[inner]).median()
    i_out = pd.Series(img[ring]).groupby(near[ring]).median()
    d = pd.concat([i_in.rename('i'), i_out.rename('o')], axis=1).dropna()
    d = d[d.o > 0]
    return float(np.median((d.i - d.o) / d.o)) if len(d) else np.nan


def robust_z(values):
    """(log x - median) / (1.4826 MAD) within the group."""
    lv = np.log(np.asarray(values, float))
    med = np.median(lv)
    mad = 1.4826 * np.median(np.abs(lv - med))
    return (lv - med) / mad if mad > 0 else np.zeros_like(lv)


def overlap(a, b):
    """(overlap fraction, correlation inside the overlap) after the best translation."""
    shift = phase_cross_correlation(a, b, normalization=None)[0]
    dy, dx = (int(round(v)) for v in shift)
    h, w = a.shape
    ya, yb = slice(max(0, dy), min(h, h + dy)), slice(max(0, -dy), min(h, h - dy))
    xa, xb = slice(max(0, dx), min(w, w + dx)), slice(max(0, -dx), min(w, w - dx))
    pa, pb = a[ya, xa].ravel(), b[yb, xb].ravel()
    frac = pa.size / a.size
    if pa.size < 100 or pa.std() == 0 or pb.std() == 0:
        return frac, 0.0
    return frac, float(np.corrcoef(pa, pb)[0, 1])


def repeated_fields(images, size=DUP_SIZE, min_overlap=DUP_OVERLAP, min_corr=DUP_CORR):
    """Pairs (key_a, key_b, overlap, corr) of fields that show the same area."""
    small = {k: resize(np.asarray(v, float), (size, size), anti_aliasing=True) for k, v in images.items()}
    out = []
    for a, b in itertools.combinations(sorted(small), 2):
        frac, r = overlap(small[a], small[b])
        if frac >= min_overlap and r >= min_corr:
            out.append((a, b, frac, r))
    return out


def assess(rows, pairs, z_cut=Z_CUT):
    """Add z, low_quality, duplicate_of and exclude to per-field rows (one condition folder)."""
    q = pd.DataFrame(rows)
    q['z'] = robust_z(q.ridge_snr)
    q['nuclear_z'] = robust_z(q.nuclear_contrast) if q.nuclear_contrast.notna().all() else np.nan
    q['low_quality'] = q.z < z_cut
    q['duplicate_of'] = ''
    score = q.set_index('key').ridge_snr
    for a, b, _, _ in pairs:
        drop, keep = (a, b) if score[a] < score[b] else (b, a)
        q.loc[q.key == drop, 'duplicate_of'] = keep
    q['exclude'] = q.low_quality | q.duplicate_of.ne('')
    return q


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--root', required=True, help='data repository (Projection/<cond>/...)')
    ap.add_argument('--v2', help='segment_v2.py or refine_v2.py output, for the nuclear-contrast diagnostic')
    ap.add_argument('--out', required=True, help='quality.csv')
    ap.add_argument('--conditions', nargs='+', default=sg.CONDS)
    args = ap.parse_args(argv)
    tables, pairs_all = [], []
    for cond in args.conditions:
        imgs = sg.inputs(args.root, cond)
        raw = an.index(f'{args.root}/Projection/{cond}/Cadherins/background')
        nuc_masks = an.index(f'{args.v2}/{cond}', '*_v2_nuclei.tif') if args.v2 else {}
        rows = []
        for k in sorted(imgs):
            um = ft.pixel_um(k)
            cad_p, nuc_p = imgs[k]
            nc = (nuclear_contrast(tifffile.imread(nuc_p), tifffile.imread(nuc_masks[k]), um)
                  if k in nuc_masks else np.nan)
            rows.append(dict(key=k, folder=cond, date=an.date_of(k),
                             ridge_snr=ridge_snr(tifffile.imread(cad_p), um), nuclear_contrast=nc))
            print(f"{cond} {k}: ridge SNR {rows[-1]['ridge_snr']:.2f}", flush=True)
        pairs = repeated_fields({k: tifffile.imread(raw[k]) for k in sorted(imgs) if k in raw})
        pairs_all += [(cond, *p) for p in pairs]
        tables.append(assess(rows, pairs))
    q = pd.concat(tables, ignore_index=True)
    q.to_csv(args.out, index=False)
    print('\nRepeated fields:')
    for cond, a, b, frac, r in pairs_all:
        print(f'  {cond}: {a} / {b}: overlap {frac:.2f}, r = {r:.3f}')
    print('\nExcluded:')
    print(q[q.exclude][['folder', 'key', 'ridge_snr', 'z', 'nuclear_z', 'low_quality', 'duplicate_of']]
          .to_string(index=False, float_format=lambda v: f'{v:.2f}'))


if __name__ == '__main__':
    main()

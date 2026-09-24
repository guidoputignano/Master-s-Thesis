#!/usr/bin/env python3
"""Segmentation v2: whole cells and nuclei with Cellpose, one setting for every condition.

Whole cells come from Cellpose ``cyto3`` on two channels: the β-catenin top-hat
projection (cytoplasm channel) and the nuclear projection (nucleus channel). Nuclei
come from the Cellpose ``nuclei`` model on the nuclear projection. The expected
diameters are fixed in micrometres and converted with the calibration in
``features.PIXEL_UM``, so 20x and 40x fields are processed at the same physical scale.

There are no per-condition or per-field parameters, no seed merging and no hole
threshold. Gaps are derived afterwards from the cells, nuclei and β-catenin
intensity (``analyze.dark_gaps``).

Inputs follow the data repository layout::

  <root>/Projection/<cond>/Cadherins/tophat/*.tif
  <root>/Projection/<cond>/Nuclei/tophat/*.tif        (20x)
  <root>/Projection/<cond>/Nuclei/background/*.tif    (40x)

The nuclear top-hat was made with a structuring element sized for 20x; at 40x it
hollows the nuclei into speckle, so 40x uses the projection without top-hat (as the
original notebooks did).

Outputs: ``<out>/<cond>/<key>_v2_cells.tif`` and ``<key>_v2_nuclei.tif`` (uint32 label
images, zlib) plus ``<out>/run_log.jsonl``. Fields already done are skipped.

  python segment_v2.py --root /path/to/data-mt --out v2_masks
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np
import tifffile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'Validation'))
sys.path.insert(0, HERE)
import seg_eval as se  # noqa: E402
import features as ft  # noqa: E402

CONDS = ['Static-x20', '1.4Pa-x20', 'Static-x40', '1.4Pa-x40']
CELL_DIAMETER_UM = 25.7     # 60 px at 20x, 120 px at 40x
NUC_DIAMETER_UM = 11.2      # 26 px at 20x, 52 px at 40x
NUCLEAR_PROJECTION = {'20x': 'tophat', '40x': 'background'}
DUPLICATE = re.compile(r'\s?\(\d+\)')


def magnification(key):
    return se.condition_of(key).split('_')[-1]


def diameters_px(key):
    um = ft.pixel_um(key)
    return round(CELL_DIAMETER_UM / um), round(NUC_DIAMETER_UM / um)


def inputs(root, cond):
    """{key: (β-catenin path, nuclear path)} for one condition."""
    def idx(sub, kind):
        d = {}
        for p in sorted(glob.glob(os.path.join(root, 'Projection', cond, sub, kind, '*.tif'))):
            k = se.sample_key(p)
            if k and not DUPLICATE.search(os.path.basename(p)) and k not in d:
                d[k] = p
        return d
    cad = idx('Cadherins', 'tophat')
    nuc = {k: p for kind in sorted(set(NUCLEAR_PROJECTION.values())) for k, p in idx('Nuclei', kind).items()
           if NUCLEAR_PROJECTION[magnification(k)] == kind}
    return {k: (cad[k], nuc[k]) for k in sorted(set(cad) & set(nuc))}


def segment(cad, nuc, key, cyto, nucm):
    """Label images (cells, nuclei) for one field."""
    d_cell, d_nuc = diameters_px(key)
    cells, _, _ = cyto.eval(np.stack([cad, nuc], -1).astype(np.float32), channels=[1, 2], diameter=d_cell)
    nuclei, _, _ = nucm.eval(nuc.astype(np.float32), channels=[0, 0], diameter=d_nuc)
    return cells.astype(np.uint32), nuclei.astype(np.uint32)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--root', required=True, help='data repository with Projection/<cond>/...')
    ap.add_argument('--out', required=True)
    ap.add_argument('--conditions', nargs='+', default=CONDS)
    ap.add_argument('--gpu', action='store_true')
    args = ap.parse_args(argv)

    import cellpose
    from cellpose import models
    cyto = models.CellposeModel(model_type='cyto3', gpu=args.gpu)
    nucm = models.CellposeModel(model_type='nuclei', gpu=args.gpu)
    version = getattr(cellpose, 'version', None) or getattr(cellpose, '__version__', 'unknown')
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, 'run_log.jsonl'), 'a')
    for cond in args.conditions:
        os.makedirs(os.path.join(args.out, cond), exist_ok=True)
        for key, (cad_p, nuc_p) in inputs(args.root, cond).items():
            out_c = os.path.join(args.out, cond, f'{key}_v2_cells.tif')
            out_n = os.path.join(args.out, cond, f'{key}_v2_nuclei.tif')
            if os.path.exists(out_c) and os.path.exists(out_n):
                continue
            t = time.time()
            cells, nuclei = segment(tifffile.imread(cad_p), tifffile.imread(nuc_p), key, cyto, nucm)
            tifffile.imwrite(out_c, cells, compression='zlib')
            tifffile.imwrite(out_n, nuclei, compression='zlib')
            rec = dict(cond=cond, key=key, cells=int(cells.max()), nuclei=int(nuclei.max()),
                       diameters_px=diameters_px(key), nuclear_projection=NUCLEAR_PROJECTION[magnification(key)],
                       cellpose=str(version), sec=round(time.time() - t, 1))
            log.write(json.dumps(rec) + '\n')
            log.flush()
            print(json.dumps(rec), flush=True)


if __name__ == '__main__':
    main()

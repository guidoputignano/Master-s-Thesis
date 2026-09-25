#!/usr/bin/env python3
"""Segmentation v2.1: Cellpose cells grown to their junctions.

The blind expert review of v2 found two error types. When junctions are faint,
Cellpose often outlines only the part of a cell around the nucleus ("too small") and
leaves the rest unassigned. It also produces nucleus-free fragments. v2.1 keeps
Cellpose's separation of neighbouring cells, but lets each cell grow over the
β-catenin landscape until it meets a junction, as the original watershed did.

* markers: Cellpose cells that contain a nucleus; nucleus-free Cellpose cells at least
  as large as the field's median nucleated cell (real cells whose nucleus was missed);
  and nuclei that no Cellpose cell covers (missed cells). Smaller nucleus-free
  fragments are not markers, so their pixels go to the neighbour whose basin they
  belong to;
* landscape: β-catenin top-hat smoothed with sigma = 1.5 um, so junctions are ridges;
* mask: every pixel except the gaps. Gap seeds are ``analyze.dark_gaps`` on the Cellpose
  cells, with the nuclear-stain test. Pixels within about 1.5 um of a detected nucleus (a
  dilation by round(1.5 um / pixel) steps: 2 px at 20x, 5 px at 40x) are never gap, so every
  nucleus can seed a cell. After the second review, each seed is grown over the connected
  pixels that pass the same test at the 5th percentile (the reviewer saw gaps extending
  beyond their outline), and enclosed specks under 23 um^2 without nuclear signal are
  filled. After the third review, a grown gap that surrounds a nucleus is removed (it is a
  faint cell) and nucleus-free voids under 115 um^2 inside a gap are filled
  (``analyze.nucleus_rule``).

Reads the segment_v2.py output and writes, per field, to ``--out/<cond>/``:
``<key>_v2_cells.tif`` (grown cells), ``<key>_v2_nuclei.tif`` (copied),
``<key>_v2_gaps.tif`` (the grown gaps) and ``<key>_v2_gaps_sens.tif`` (value 2: threshold at
the 0.5th percentile, 4: at the 5th, 8: the 1st-percentile seeds before growth, the masks the
second review showed). ``analyze.py --v2 <out>`` then compares v1 with v2.1 and uses these
gap masks.

  python refine_v2.py --root /path/to/data-mt --v2 v2_masks --out v2r_masks
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np
import tifffile
from scipy import ndimage
from skimage.segmentation import relabel_sequential, watershed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'Validation'))
sys.path.insert(0, HERE)
import features as ft  # noqa: E402
import analyze as an  # noqa: E402
import segment_v2 as sg  # noqa: E402

LANDSCAPE_SIGMA_UM = 1.0 * ft.SCALE     # 1.5 um (1 um at the recorded pixel size, features.py)
CANDIDATE_MIN_UM2 = an.GAP_MIN_UM2


def markers(cells, nuclei):
    """Marker image and a summary of what became a marker."""
    owner, n_ids, _ = ft.nuclei_assignment(cells, nuclei)
    area = np.bincount(cells.ravel())
    nucleated = np.array(sorted(set(owner.values())), dtype=np.int64)
    med = np.median(area[nucleated]) if len(nucleated) else 0
    ids = np.arange(len(area))
    no_nuc = np.setdiff1d(ids[(ids > 0) & (area > 0)], nucleated)
    kept_free = no_nuc[area[no_nuc] >= med]
    keep = np.zeros(len(area), bool)
    keep[nucleated] = True
    keep[kept_free] = True
    m = np.where(keep[cells], cells, 0).astype(np.int64)
    orphans = [n for n in n_ids.tolist() if n not in owner]
    nxt = int(cells.max()) + 1
    added = 0
    for n in orphans:
        pix = (nuclei == n) & (m == 0)
        if pix.any():
            m[pix] = nxt
            nxt += 1
            added += 1
    return m, dict(nucleated=int(len(nucleated)), nucleus_free_kept=int(len(kept_free)),
                   fragments_dropped=int(len(no_nuc) - len(kept_free)), orphan_markers=added)


def refine(cells, nuclei, cad_tophat, cad_raw, nuc_img, um, grow=True):
    # A detected nucleus (plus about 1.5 um) is never gap, so every nucleus can get a cell.
    nuc_zone = ndimage.binary_dilation(nuclei > 0, iterations=max(1, int(round(1.0 * ft.SCALE / um))))

    def gaps_at(pct, min_um2=CANDIDATE_MIN_UM2):
        g, t = an.dark_gaps(cells, nuclei, cad_raw, um, pct, nuc_img, min_um2)
        return an.area_filter(g & ~nuc_zone, um, min_um2), t

    seeds, thr = gaps_at(an.GAP_DARK_PCT)
    gaps = seeds
    if grow:
        nuclear = (nuclei > 0) | an.nuclear_signal(nuc_img, nuclei)
        gaps = an.grow_gaps(seeds, gaps_at(an.GAP_GROW_PCT, 0.0)[0], nuclear, um)
        gaps = an.nucleus_rule(gaps, nuclei, nuclear, um)      # third review (see analyze.py)
    sens = (seeds * 8).astype(np.uint8)
    for bit, pct in ((2, 0.5), (4, 5.0)):
        sens |= (gaps_at(pct)[0] * bit).astype(np.uint8)
    m, info = markers(cells, nuclei)
    m[gaps] = 0
    land = ndimage.gaussian_filter(np.asarray(cad_tophat, float), LANDSCAPE_SIGMA_UM / um)
    grown = watershed(land, markers=m, mask=~gaps)
    grown, _, _ = relabel_sequential(grown)
    info.update(threshold=thr, cells_in=int(len(np.unique(cells)) - 1), cells_out=int(grown.max()),
                covered_in=float((cells > 0).mean()), covered_out=float((grown > 0).mean()),
                gap_seeds=float(seeds.mean()), gaps=float(gaps.mean()))
    return grown.astype(np.uint32), gaps, sens, info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--root', required=True, help='data repository (Projection/<cond>/...)')
    ap.add_argument('--v2', required=True, help='segment_v2.py output')
    ap.add_argument('--out', required=True)
    ap.add_argument('--conditions', nargs='+', default=sg.CONDS)
    ap.add_argument('--no-grow', action='store_true', help='gaps = the 1st-percentile seeds (as in the second review)')
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, 'refine_log.jsonl'), 'a')
    for cond in args.conditions:
        os.makedirs(os.path.join(args.out, cond), exist_ok=True)
        imgs = sg.inputs(args.root, cond)
        raw = an.index(f'{args.root}/Projection/{cond}/Cadherins/background')
        v2c = an.index(f'{args.v2}/{cond}', '*_v2_cells.tif')
        v2n = an.index(f'{args.v2}/{cond}', '*_v2_nuclei.tif')
        for k in sorted(set(imgs) & set(raw) & set(v2c) & set(v2n)):
            um = ft.pixel_um(k)
            cells, nuclei = tifffile.imread(v2c[k]), tifffile.imread(v2n[k])
            cad_p, nuc_p = imgs[k]
            grown, gaps, sens, info = refine(cells, nuclei, tifffile.imread(cad_p), tifffile.imread(raw[k]),
                                             tifffile.imread(nuc_p), um, grow=not args.no_grow)
            base = os.path.join(args.out, cond, k)
            tifffile.imwrite(f'{base}_v2_cells.tif', grown, compression='zlib')
            tifffile.imwrite(f'{base}_v2_gaps.tif', gaps.astype(np.uint8), compression='zlib')
            tifffile.imwrite(f'{base}_v2_gaps_sens.tif', sens, compression='zlib')
            shutil.copyfile(v2n[k], f'{base}_v2_nuclei.tif')
            rec = dict(cond=cond, key=k, **info)
            log.write(json.dumps(rec) + '\n')
            log.flush()
            print(json.dumps(rec), flush=True)


if __name__ == '__main__':
    main()

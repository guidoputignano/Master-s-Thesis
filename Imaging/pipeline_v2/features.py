"""Per-cell features in physical units, computed identically for any segmentation.

Used to compare the original pipeline (v1) with v2 on equal terms and as input
to the senescence model. One row per cell: area, shape, orientation, border
contact, nuclei (a nucleus belongs to the cell covering more than half of it,
the same rule as the original Senescence notebooks) and gap contact.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.measure import regionprops

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Validation'))
import seg_eval as se  # noqa: E402

# Calibration. The .nd2 files record 0.429 um/px: the Nikon Ti zoom changer at position 1.5
# (x a 1.01 relay = 1.515) with the 20x objective, 13 / (20 * 1.515) for the 13 um pixels of the
# Andor iXon 888. Every A1 file records this same optics state, including the 23 files taken with
# the 40x objective, so the recorded state was stale. The stage sets the scale: two overlapping
# fields (1.4 Pa 19dec21 seq013 and seq016) lie 102.8 um apart on the stage and 158 px apart in all
# three channels, 0.650 um/px (Imaging/Validation/stage_calibration.py), the camera pixel through
# the 20x objective alone; 2019 files of the same microscope and camera, recorded at zoom position
# 1.0, pass the same check. The 40x files are half (their nuclei are four times larger in pixels).
# calibration.csv holds the per-file record: recorded optics against the pixel size used.
RECORDED_PIXEL_UM = {'20x': 0.429, '40x': 0.2145}     # used by the analysis until the stage check
PIXEL_UM = {'20x': 0.650, '40x': 0.325}
# The pipeline's lengths were chosen, and its outputs reviewed, at the recorded pixel size. Each
# um constant is therefore its original value times SCALE (areas SCALE ** 2): every pixel
# operation, mask and verdict stays as it was, and the constant is stated in true units.
SCALE = PIXEL_UM['20x'] / RECORDED_PIXEL_UM['20x']     # 1.5152
AREA = SCALE ** 2                                      # 2.2957

_CALIBRATION = None


def calibration():
    """Per-file calibration table (calibration.csv): field key -> pixel size used, with provenance."""
    global _CALIBRATION
    if _CALIBRATION is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calibration.csv')
        _CALIBRATION = pd.read_csv(path).set_index('key') if os.path.exists(path) else pd.DataFrame()
    return _CALIBRATION


def pixel_um(key):
    """True pixel size of a field: its row in calibration.csv, else its magnification."""
    cal = calibration()
    if len(cal) and key in cal.index:
        return float(cal.loc[key, 'px_um'])
    return PIXEL_UM[se.condition_of(key).split('_')[-1]]


def nuclei_assignment(cells, nuclei):
    """Map nucleus label -> cell label (> 50 % of the nucleus inside the cell)."""
    n_ids, c_ids, inter, n_area, _ = se.overlap(nuclei, cells)
    if inter.size == 0:
        return {}, n_ids, n_area
    j = inter.argmax(1)
    best = inter[np.arange(len(n_ids)), j]
    ok = best * 2 > n_area
    return dict(zip(n_ids[ok].tolist(), c_ids[j[ok]].tolist())), n_ids, n_area


def cell_features(cells, key, nuclei=None, holes=None, adjacency_px=2):
    """DataFrame of per-cell features for one field (``key`` gives the magnification)."""
    um = pixel_um(key)
    edge = se.border_ids(cells)
    rows = []
    for rp in regionprops(cells):
        axial = (90.0 + math.degrees(rp.orientation)) % 180.0
        minor = rp.axis_minor_length
        rows.append(dict(
            label=rp.label, area_um2=rp.area * um ** 2, perimeter_um=rp.perimeter * um,
            aspect_ratio=rp.axis_major_length / minor if minor > 0 else np.nan,
            axial_deg=axial, misalign_deg=min(axial, 180 - axial),
            circularity=4 * math.pi * rp.area / rp.perimeter ** 2 if rp.perimeter > 0 else np.nan,
            solidity=rp.solidity, touches_border=rp.label in edge,
            centroid_y=rp.centroid[0], centroid_x=rp.centroid[1]))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.insert(0, 'key', key)
    df.insert(1, 'condition', se.condition_of(key))
    if nuclei is not None:
        owner, n_ids, n_area = nuclei_assignment(cells, nuclei)
        per_cell = {}
        for n, a in zip(n_ids.tolist(), n_area.tolist()):
            if n in owner:
                per_cell.setdefault(owner[n], []).append(a * um ** 2)
        df['n_nuclei'] = [len(per_cell.get(l, [])) for l in df.label]
        df['nuc_area_um2'] = [sum(per_cell.get(l, [])) or np.nan for l in df.label]
        df['largest_nuc_um2'] = [max(per_cell[l]) if l in per_cell else np.nan for l in df.label]
    if holes is not None:
        h = np.asarray(holes) > 0
        near = ndimage.binary_dilation(h, iterations=adjacency_px) if h.any() else h
        adj = set(np.unique(cells[near & (cells > 0)]).tolist())
        df['hole_adjacent'] = df.label.isin(adj)
    return df

"""Synthetic check for Imaging/Validation/stage_calibration.py: two overlapping crops of one texture,
a known shift and stage displacement, and the pixel size recovered from them."""
import os
import sys

import numpy as np
import pytest

pytest.importorskip('skimage')
from scipy import ndimage as ndi  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Imaging', 'Validation'))
import stage_calibration as sc  # noqa: E402


@pytest.mark.parametrize('px', [0.429, 0.650])
def test_pixel_size_from_stage_and_shift(px):
    rng = np.random.default_rng(0)
    canvas = ndi.gaussian_filter(rng.random((700, 700)), 2.0)
    canvas += np.linspace(0, 3, 700)[None, :] * canvas.std()      # illumination gradient, removed by the filter
    dy, dx = 37, -22
    a = canvas[150:406, 150:406][None]
    b = canvas[150 + dy:406 + dy, 150 + dx:406 + dx][None]
    stage_a, stage_b = (1000.0, 2000.0), (1000.0 + dx * px, 2000.0 - dy * px)
    r = sc.calibrate(stage_a, stage_b, a, b)[0]
    assert r['shift_px'] == pytest.approx(np.hypot(dy, dx), rel=0.01)
    assert r['um_per_px'] == pytest.approx(px, rel=0.01)

#!/usr/bin/env python3
"""Pixel size from the microscope stage, independent of the recorded calibration.

Two fields that overlap: their stage positions (um, from the .nd2 frame metadata) and the image
shift between them (px, phase correlation of high-pass filtered projections, per channel). The pixel
size is |stage displacement| / |image shift|; the angle between the two vectors gives the camera's
orientation on the stage, and all channels must agree.

A1 (Chala, 22 Dec 2021, Plan Apo 20x/0.75, Andor DU-888). The files record 0.429 um/px, a 1.515x
zoom. The one overlapping pair with a displacement, 1.4 Pa 19dec21 seq013 and seq016, is 102.8 um
apart on the stage and 158 px apart in all three channels: 0.650 um/px, the 13 um camera pixel
through the 20x objective without the zoom. The fields are 666 x 666 um, not 439 x 439 um, and
the 40x files (recorded as 20x) are 0.325 um/px, not 0.2145. On files whose calibration is right,
the check returns the recorded pixel size within 0.4 % (tests/test_stage_calibration.py covers the
arithmetic on synthetic shifts).

  python stage_calibration.py a.nd2 b.nd2
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy import ndimage as ndi


def read(path):
    """Stage xy (um) of the first frame, per-channel maximum projection (C, Y, X), recorded um/px,
    channel names."""
    import nd2
    with nd2.ND2File(path) as f:
        pos = f.frame_metadata(0).channels[0].position.stagePositionUm
        names = [c.channel.name for c in f.metadata.channels]
        a = np.asarray(f.asarray())
        dims = list(f.sizes)
        um = f.voxel_size().x
    if 'Z' in dims:
        a = a.max(axis=dims.index('Z'))
    if a.ndim == 2:
        a = a[None]
    return np.array(pos[:2], float), a.astype(np.float32), float(um), names


def highpass(img, sigma=25.0, alpha=0.25):
    """Remove illumination and vignetting, then taper the borders (Tukey window): both would otherwise
    pull the phase-correlation peak to zero shift."""
    from scipy.signal.windows import tukey
    a = np.asarray(img, np.float64)
    a = a - ndi.gaussian_filter(a, sigma)
    a = a * np.outer(tukey(a.shape[0], alpha), tukey(a.shape[1], alpha))
    return a / (a.std() + 1e-12)


def calibrate(stage_a, stage_b, img_a, img_b, upsample=20):
    """Pixel size (um/px) and the angle (deg) between the stage vector and the image shift.

    img_a, img_b: (C, Y, X). Returns one dict per channel."""
    from skimage.registration import phase_cross_correlation
    d = np.asarray(stage_b, float) - np.asarray(stage_a, float)
    dist = float(np.hypot(*d))
    out = []
    for c in range(img_a.shape[0]):
        s, err, _ = phase_cross_correlation(highpass(img_a[c]), highpass(img_b[c]), upsample_factor=upsample,
                                            normalization='phase')
        mag = float(np.hypot(*s))
        ang = np.degrees(np.arctan2(s[0], s[1]) - np.arctan2(-d[1], d[0]))
        out.append(dict(channel=c, shift_y_px=float(s[0]), shift_x_px=float(s[1]), shift_px=mag,
                        stage_um=dist, um_per_px=dist / mag if mag > 0 else np.nan,
                        angle_deg=float((ang + 180) % 360 - 180), error=float(err)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('a')
    ap.add_argument('b')
    a = ap.parse_args(argv)
    pa, ia, um_a, names = read(a.a)
    pb, ib, um_b, _ = read(a.b)
    print(f'stage: {pa.round(2)} -> {pb.round(2)} um; recorded pixel {um_a:.4f} / {um_b:.4f} um')
    for r in calibrate(pa, pb, ia, ib):
        print(f"{names[r['channel']]}: shift ({r['shift_y_px']:.2f}, {r['shift_x_px']:.2f}) px = {r['shift_px']:.2f} px "
              f"for {r['stage_um']:.2f} um -> {r['um_per_px']:.4f} um/px; angle {r['angle_deg']:+.1f} deg")


if __name__ == '__main__':
    main()

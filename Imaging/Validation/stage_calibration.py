#!/usr/bin/env python3
"""Pixel size from the microscope stage, independent of the recorded calibration.

Two fields that overlap: their stage positions (um, from the .nd2 frame metadata) and the image
shift between them (px, per channel, on high-pass filtered projections). The pixel size is
|stage displacement| / |image shift|; the angle between the two vectors gives the camera's
orientation on the stage, and all channels must agree.

The shift comes from the masked normalised cross-correlation over every shift that keeps at least
4 % of the field in common (whole pixels). It finds thin edge strips and shifts beyond half the
field, where phase correlation wraps around (on the 40x pair below it returns 0.450 um/px at
42 deg). ``--phase`` gives the tapered phase correlation with subpixel precision instead, for pairs
that share most of the field.

A1 (Chala, 22 Dec 2021, Plan Apo 20x/0.75, Andor DU-888). The files record 0.429 um/px, a 1.515x
zoom. Seven pairs of field positions overlap with a displacement, 103-668 um apart on the stage,
and all give the camera's 13 um pixel through the objective without the zoom, in all three
channels:

  * 20x, 0.649-0.653 um/px: 1.4 Pa 19dec21 seq013/seq016 (102.8 um = 158 px, 83 % shared) and
    five 20dec21 flow pairs 537-668 um apart (seq010/011, 012/013, 002/003, 003/004, 002/017;
    seq010again and seq014 repeat seq010 and seq013 and give the same), which share a strip of
    6-19 % of the field. At 0.429 um/px a field is 439 um wide, and those five could not overlap;
  * 40x (recorded as 20x), 0.328 um/px: Static 19dec21 40x-003/004, 208.8 um = 637 px; at
    0.2145 um/px the stage would put them 973 px apart, where the images do not match.

The camera sits at 179 deg to the stage in every pair. On files whose calibration is right, the
check returns the recorded pixel size within 0.4 % (tests/test_stage_calibration.py covers the
arithmetic on synthetic shifts).

  python stage_calibration.py a.nd2 b.nd2 [--phase]
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


def highpass(img, sigma=25.0, alpha=0.25, taper=True):
    """Remove illumination and vignetting, then taper the borders (Tukey window): both would otherwise
    pull the phase-correlation peak to zero shift."""
    from scipy.signal.windows import tukey
    a = np.asarray(img, np.float64)
    a = a - ndi.gaussian_filter(a, sigma)
    if taper:
        a = a * np.outer(tukey(a.shape[0], alpha), tukey(a.shape[1], alpha))
    return a / (a.std() + 1e-12)


def calibrate(stage_a, stage_b, img_a, img_b, method='masked', upsample=20, overlap_ratio=0.04):
    """Pixel size (um/px) and the angle (deg) between the stage vector and the image shift.

    img_a, img_b: (C, Y, X). ``method``: 'masked', the normalised cross-correlation over every shift
    that keeps at least ``overlap_ratio`` of the field in common (whole pixels; error = 1 - the
    correlation inside the overlap), or 'phase', tapered phase correlation (subpixel; shifts beyond
    half the field wrap around). Returns one dict per channel."""
    from skimage.registration import phase_cross_correlation
    d = np.asarray(stage_b, float) - np.asarray(stage_a, float)
    dist = float(np.hypot(*d))
    out = []
    for c in range(img_a.shape[0]):
        if method == 'masked':
            a, b = highpass(img_a[c], taper=False), highpass(img_b[c], taper=False)
            m = np.ones(a.shape, bool)
            s = np.asarray(phase_cross_correlation(a, b, reference_mask=m, moving_mask=m,
                                                   overlap_ratio=overlap_ratio)[0], float)
            err = 1.0 - overlap_correlation(a, b, s)
        else:
            s, err, _ = phase_cross_correlation(highpass(img_a[c]), highpass(img_b[c]), upsample_factor=upsample,
                                                normalization='phase')
        mag = float(np.hypot(*s))
        ang = np.degrees(np.arctan2(s[0], s[1]) - np.arctan2(-d[1], d[0]))
        out.append(dict(channel=c, shift_y_px=float(s[0]), shift_x_px=float(s[1]), shift_px=mag,
                        stage_um=dist, um_per_px=dist / mag if mag > 0 else np.nan,
                        angle_deg=float((ang + 180) % 360 - 180), error=float(err)))
    return out


def overlap_correlation(a, b, shift):
    """Pearson correlation of a and b where they overlap once b is moved by ``shift`` (y, x) onto a."""
    dy, dx = (int(round(v)) for v in shift)
    h, w = a.shape
    if abs(dy) >= h or abs(dx) >= w:
        return float('nan')
    pa = a[max(0, dy):min(h, h + dy), max(0, dx):min(w, w + dx)].ravel()
    pb = b[max(0, -dy):min(h, h - dy), max(0, -dx):min(w, w - dx)].ravel()
    return float(np.corrcoef(pa, pb)[0, 1]) if pa.size > 1 else float('nan')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('a')
    ap.add_argument('b')
    ap.add_argument('--phase', action='store_true', help='tapered phase correlation, subpixel (large overlaps only)')
    a = ap.parse_args(argv)
    pa, ia, um_a, names = read(a.a)
    pb, ib, um_b, _ = read(a.b)
    print(f'stage: {pa.round(2)} -> {pb.round(2)} um; recorded pixel {um_a:.4f} / {um_b:.4f} um')
    for r in calibrate(pa, pb, ia, ib, method='phase' if a.phase else 'masked'):
        print(f"{names[r['channel']]}: shift ({r['shift_y_px']:.2f}, {r['shift_x_px']:.2f}) px = {r['shift_px']:.2f} px "
              f"for {r['stage_um']:.2f} um -> {r['um_per_px']:.4f} um/px; angle {r['angle_deg']:+.1f} deg")


if __name__ == '__main__':
    main()

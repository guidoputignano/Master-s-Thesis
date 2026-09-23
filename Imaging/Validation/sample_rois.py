#!/usr/bin/env python3
"""Seeded, stratified choice of what to annotate (writes manifest.csv).

Picking annotation regions by eye tends to favour the easy, clean parts of a
field. This script fixes the choice in advance:

* ``cells`` rows: square ROIs, at most one per field, spread over as many
  fields as possible, balanced per condition between 'gap' regions (pipeline
  hole fraction >= --gap-min) and 'confluent' regions (<= --confluent-max)
  when hole masks are given, because gap-adjacent territories are where the
  pipeline is weakest;
* ``holes`` rows: whole fields (the gap fraction is a field-level quantity),
  deliberately including fields the pipeline declared hole-free.

The annotation order is shuffled and each row gets a neutral ``blind_id``,
so conditions are not annotated in blocks.

Example:
  python sample_rois.py --images Projected/*/Cadherins/tophat \
      --holes Segmented/*/Holes_masks Segmented/Static-x20/Holes --holes-glob '*_segmented.tif' \
      --rois-per-condition 4 --fields-per-condition 4 --size 400 --out manifest.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seg_eval as se  # noqa: E402


def image_shape(path):
    import tifffile
    with tifffile.TiffFile(path) as tf:
        shape = tf.pages[0].shape
    return shape[-2:] if len(shape) > 2 else shape


def window_fractions(mask, size, n, rng, margin):
    """Hole fraction of ``n`` random size x size windows (integral image)."""
    h, w = mask.shape
    ii = np.pad(np.cumsum(np.cumsum(mask.astype(np.int64), 0), 1), ((1, 0), (1, 0)))
    ys = rng.integers(margin, max(margin + 1, h - size - margin + 1), n)
    xs = rng.integers(margin, max(margin + 1, w - size - margin + 1), n)
    s = ii[ys + size, xs + size] - ii[ys, xs + size] - ii[ys + size, xs] + ii[ys, xs]
    return ys, xs, s / float(size * size)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split('\n\n', 1)[1])
    ap.add_argument('--images', nargs='+', required=True, help='folder(s) with one image per field')
    ap.add_argument('--images-glob', default='*.tif')
    ap.add_argument('--holes', nargs='+', help='folder(s) of pipeline hole masks, for gap stratification')
    ap.add_argument('--holes-glob', default='*.tif')
    ap.add_argument('--rois-per-condition', type=int, default=4)
    ap.add_argument('--fields-per-condition', type=int, default=4)
    ap.add_argument('--size', type=int, default=400, help='ROI side in pixels (x40: use about twice the x20 size)')
    ap.add_argument('--margin', type=int, default=32, help='keep ROIs this far from the field edge')
    ap.add_argument('--gap-min', type=float, default=0.03)
    ap.add_argument('--confluent-max', type=float, default=0.005)
    ap.add_argument('--conditions', nargs='+', help='restrict to these conditions, e.g. 0Pa_A1_20x')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)

    images, holes = {}, {}
    for f in args.images:
        images.update(se.index_by_key(f, args.images_glob))
    for f in args.holes or []:
        holes.update(se.index_by_key(f, args.holes_glob))
    if not images:
        raise SystemExit('no images found')
    by_cond = {}
    for key in sorted(images):
        by_cond.setdefault(se.condition_of(key), []).append(key)
    if args.conditions:
        by_cond = {c: k for c, k in by_cond.items() if c in args.conditions}

    rows = []
    for cond, keys in sorted(by_cond.items()):
        order = list(rng.permutation(keys))
        # --- whole fields for hole annotation (include pipeline-empty fields)
        fields = order[:args.fields_per_condition]
        if holes:
            empty = [k for k in order if k in holes and not se.read_mask(holes[k]).any()]
            if empty and not set(fields) & set(empty):
                fields = fields[:-1] + [empty[0]]
        for k in fields:
            h, w = image_shape(images[k])
            frac = se.read_mask(holes[k]).astype(bool).mean() if k in holes else np.nan
            rows.append(dict(task='holes', key=k, condition=cond, y0=0, x0=0, h=h, w=w,
                             stratum='field', pred_hole_frac=frac, file=images[k]))
        # --- cell ROIs, one per field, alternating gap / confluent strata
        want = ['gap', 'confluent'] if holes else ['random']
        taken = 0
        for i, k in enumerate(order):
            if taken >= args.rois_per_condition:
                break
            h, w = image_shape(images[k])
            if min(h, w) < args.size + 2 * args.margin:
                raise SystemExit(f"--size {args.size} does not fit {k} ({h}x{w})")
            target = want[taken % len(want)]
            if k in holes:
                ys, xs, fr = window_fractions(se.read_mask(holes[k]) > 0, args.size, 400, rng, args.margin)
                ok = fr >= args.gap_min if target == 'gap' else fr <= args.confluent_max
                if not ok.any():
                    continue            # this field has no such region; try the next field
                j = int(rng.choice(np.flatnonzero(ok)))
                y0, x0, frac = int(ys[j]), int(xs[j]), float(fr[j])
            else:
                if holes:               # stratified run, but no mask for this field
                    continue
                y0 = int(rng.integers(args.margin, h - args.size - args.margin + 1))
                x0 = int(rng.integers(args.margin, w - args.size - args.margin + 1))
                frac, target = np.nan, 'random'
            rows.append(dict(task='cells', key=k, condition=cond, y0=y0, x0=x0, h=args.size, w=args.size,
                             stratum=target, pred_hole_frac=frac, file=images[k]))
            taken += 1
        if taken < args.rois_per_condition:
            print(f"warning: {cond}: only {taken} cell ROI(s) found", file=sys.stderr)

    man = pd.DataFrame(rows)
    man['roi_id'] = [f"{r.key}__{'f' if r.task == 'holes' else 'r'}{i:02d}" for i, r in enumerate(man.itertuples())]
    man = man.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    man['blind_id'] = [f"A{i + 1:03d}" for i in range(len(man))]
    cols = ['blind_id', 'roi_id', 'task', 'key', 'condition', 'stratum', 'y0', 'x0', 'h', 'w', 'pred_hole_frac', 'file']
    man[cols].to_csv(args.out, index=False)
    print(man.groupby(['task', 'condition', 'stratum']).size().to_string())
    print(f"\nwrote {len(man)} rows to {args.out}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Ground-truth-free diagnostics of the pipeline outputs.

Runs on the masks the notebooks already wrote and flags the failure modes
that bias the reported numbers, before any annotation is done:

* coverage: cells, holes, cells overlapping holes, pixels assigned to neither
  (cells and holes are defined by two different thresholds);
* cells touching the field border (truncated, but kept in the statistics);
* degenerate objects (tiny areas, circularity > 1);
* nuclei per cell (0 or >= 2 nuclei feed the 'Polynucleated' rule);
* enrichment of 'Senescent' calls next to gaps and at the border: if large
  territories are gap leakage rather than biology, senescent calls pile up
  there;
* the senescent fraction per condition under the original, condition-specific
  gates versus one harmonised rule set (same physical thresholds everywhere).

Example (one condition; repeat or point at several folders):
  python diagnostics.py \
      --cells  Segmented/Static-x20/Cell_merged_conservative \
      --nuclei Segmented/Static-x20/Nuclei --nuclei-glob '*_filtered_mask.tif' \
      --holes  Segmented/Static-x20/Holes  --holes-glob '*_segmented.tif' \
      --classes Analysis/Static-x20/Senescence_Results/cell_classification_rule_based_full.csv \
      --px-um '20x=0.65,40x=0.325' --out diag/Static-x20
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seg_eval as se  # noqa: E402


def image_diagnostics(cells, holes=None, nuclei=None, px_um=None, adjacency_px=2, tiny_px=50):
    """Per-image summary dict and a per-cell DataFrame (index = cell label)."""
    cells = np.asarray(cells)
    m = se.morphology(cells, px_um)
    fg = cells > 0
    d = dict(n_cells=len(m), cell_cover=fg.mean(),
             frac_border=m.touches_border.mean() if len(m) else np.nan,
             n_tiny=int((m.area_px < tiny_px).sum()),
             n_circularity_gt1=int((m.circularity > 1.0).sum()))
    m['hole_adjacent'] = False
    if holes is not None:
        h = np.asarray(holes) > 0
        d.update(hole_frac=h.mean(), cells_on_holes_frac=(fg & h).mean(),
                 unassigned_frac=(~fg & ~h).mean())
        near = ndimage.binary_dilation(h, iterations=adjacency_px) if h.any() else h
        adj = np.unique(cells[near & fg])
        m['hole_adjacent'] = m.index.isin(adj)
        d['frac_hole_adjacent'] = m.hole_adjacent.mean() if len(m) else np.nan
        inner = m[~m.touches_border]
        a, b = inner.area[inner.hole_adjacent], inner.area[~inner.hole_adjacent]
        d['median_area_ratio_adjacent'] = a.median() / b.median() if len(a) and len(b) else np.nan
    else:
        d['unassigned_frac'] = (~fg).mean()
    if nuclei is not None:
        counts = se.nuclei_per_object(cells, nuclei=nuclei)
        m['n_nuclei'] = [counts.get(i, 0) for i in m.index]
        d.update(frac_no_nucleus=(m.n_nuclei == 0).mean(), frac_2plus_nuclei=(m.n_nuclei >= 2).mean())
    return d, m


def md_table(df):
    """DataFrame -> markdown table without the optional `tabulate` dependency."""
    head = [df.index.name or ''] + [str(c) for c in df.columns]
    rows = [[str(i)] + [f"{v:g}" if isinstance(v, float) else str(v) for v in r] for i, r in zip(df.index, df.values)]
    return '\n'.join(['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)] + ['| ' + ' | '.join(r) + ' |' for r in rows])


def load_classes(paths):
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        uid = df['cell_id_unique'].astype(str).str.rsplit('_', n=1)
        df['key'] = [se.sample_key(u[0]) for u in uid]
        df['label'] = [int(float(u[1])) for u in uid]
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None


def harmonised_gates(df, px_lookup, very_large_um2, poly=True, low_ratio=0.1):
    """Re-gate cells with one physical rule set for every condition.

    Uses the per-cell features the notebooks saved; ``cell_area`` there is in
    native pixels, converted here with the same pixel-size table as the masks.
    """
    area_um2 = df.cell_area * df.key.map(lambda k: px_lookup(k) ** 2)
    sen = area_um2 > very_large_um2
    if poly and 'nuclei_count' in df:
        sen |= df.nuclei_count > 1
    if low_ratio is not None and 'nucleus_to_cell_area_ratio' in df:
        sen |= df.nucleus_to_cell_area_ratio < low_ratio
    return sen


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split('\n\n', 2)[2])
    ap.add_argument('--cells', required=True, nargs='+', help='folder(s) of cell label masks')
    ap.add_argument('--cells-glob', default='*.tif')
    ap.add_argument('--holes', nargs='+', help='folder(s) of binary hole masks')
    ap.add_argument('--holes-glob', default='*.tif')
    ap.add_argument('--nuclei', nargs='+', help='folder(s) of nuclear label masks')
    ap.add_argument('--nuclei-glob', default='*.tif')
    ap.add_argument('--classes', nargs='+', help='cell_classification_rule_based_full.csv file(s)')
    ap.add_argument('--px-um', help="e.g. '20x=0.65,40x=0.325' (needed for harmonised gates)")
    ap.add_argument('--very-large-um2', type=float,
                    help='single physical area gate for the harmonised re-classification')
    ap.add_argument('--adjacency-px', type=int, default=2)
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)

    def index(folders, pattern):
        found = {}
        for f in folders or []:
            found.update(se.index_by_key(f, pattern))
        return found

    cells, holes, nucs = index(args.cells, args.cells_glob), index(args.holes, args.holes_glob), \
        index(args.nuclei, args.nuclei_glob)
    px = se.parse_px_um(args.px_um)
    classes = load_classes(args.classes) if args.classes else None
    if not cells:
        raise SystemExit('no cell masks found')

    rows, per_cell = [], []
    for key, path in sorted(cells.items()):
        c = se.read_mask(path)
        h = se.read_mask(holes[key]) if key in holes else None
        n = se.read_mask(nucs[key]) if key in nucs else None
        d, m = image_diagnostics(c, h, n, px(key) if px else None, args.adjacency_px)
        rows.append(dict(key=key, condition=se.condition_of(key), has_holes=h is not None, **d))
        per_cell.append(m.reset_index().assign(key=key, condition=se.condition_of(key)))
    per = pd.DataFrame(rows)
    cells_df = pd.concat(per_cell, ignore_index=True)

    lines = ['# Pipeline diagnostics (no ground truth)\n',
             f"{len(per)} images, {len(cells_df)} cells. Areas in {'um^2' if px else 'px'}.\n",
             '| condition | images | cells/image | on border | tiny (<50 px) | circularity > 1 | '
             'cells over holes | unassigned px | hole-adjacent cells | area ratio adjacent/other |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for cond, d in per.groupby('condition'):
        g = lambda c: d[c].mean() if c in d and d[c].notna().any() else np.nan  # noqa: E731
        lines.append(f"| {cond} | {len(d)} | {d.n_cells.mean():.0f} | {100 * g('frac_border'):.1f}% | "
                     f"{int(d.n_tiny.sum())} | {int(d.n_circularity_gt1.sum())} | "
                     f"{100 * g('cells_on_holes_frac'):.2f}% | {100 * g('unassigned_frac'):.2f}% | "
                     f"{100 * g('frac_hole_adjacent'):.1f}% | {g('median_area_ratio_adjacent'):.2f} |")
    if 'frac_no_nucleus' in per:
        lines += ['\n| condition | cells with 0 nuclei | cells with >= 2 nuclei |', '|---|---|---|']
        for cond, d in per.groupby('condition'):
            lines.append(f"| {cond} | {100 * d.frac_no_nucleus.mean():.1f}% | {100 * d.frac_2plus_nuclei.mean():.1f}% |")

    if classes is not None:
        merged = cells_df.merge(classes, on=['key', 'label'], how='inner', suffixes=('', '_csv'))
        lost = len(classes) - len(merged)
        lines += [f"\n## Senescence calls vs. position ({len(merged)} cells joined; {lost} CSV rows without a mask)\n",
                  'If large territories were real senescent cells, their rate would not depend on '
                  'touching a gap or the field border.\n',
                  '| condition | senescent (as reported) | rate next to gaps | rate elsewhere | '
                  'rate on border | rate interior |', '|---|---|---|---|---|---|']
        merged['sen'] = merged.cell_type.eq('Senescent')
        for cond, d in merged.groupby('condition'):
            rate = lambda s: f"{100 * s.mean():.1f}% (n={len(s)})" if len(s) else 'n/a'  # noqa: E731
            lines.append(f"| {cond} | {rate(d.sen)} | {rate(d.sen[d.hole_adjacent])} | "
                         f"{rate(d.sen[~d.hole_adjacent])} | {rate(d.sen[d.touches_border])} | "
                         f"{rate(d.sen[~d.touches_border])} |")
        if 'rule_based_classification_granular' in merged:
            tab = pd.crosstab(merged.condition, merged.rule_based_classification_granular, normalize='index') * 100
            lines += ['\nReported rule mix (% of cells):\n', md_table(tab.round(1))]
        if px and args.very_large_um2:
            merged['sen_harmonised'] = harmonised_gates(merged, px, args.very_large_um2).to_numpy()
            clean = merged[~merged.touches_border & ~merged.hole_adjacent]
            lines += ["\n## Sensitivity of the senescent fraction to analysis choices\n",
                      f"Harmonised gates: area > {args.very_large_um2:g} um^2, > 1 nucleus, "
                      'nucleus/cell area < 0.1 (same everywhere; no dataset-normalised score).\n',
                      '| condition | as reported | harmonised | harmonised, interior and not next to gaps |',
                      '|---|---|---|---|']
            for cond, d in merged.groupby('condition'):
                c = clean[clean.condition == cond]
                lines.append(f"| {cond} | {100 * d.sen.mean():.1f}% | {100 * d.sen_harmonised.mean():.1f}% | "
                             f"{100 * c.sen_harmonised.mean():.1f}% (n={len(c)}) |")
        cells_df = cells_df.merge(classes[['key', 'label', 'cell_type']], on=['key', 'label'], how='left')

    os.makedirs(args.out, exist_ok=True)
    per.to_csv(os.path.join(args.out, 'diagnostics_per_image.csv'), index=False)
    cells_df.to_csv(os.path.join(args.out, 'diagnostics_per_cell.csv'), index=False)
    text = '\n'.join(lines) + '\n'
    with open(os.path.join(args.out, 'diagnostics.md'), 'w') as f:
        f.write(text)
    print(text)


if __name__ == '__main__':
    main()

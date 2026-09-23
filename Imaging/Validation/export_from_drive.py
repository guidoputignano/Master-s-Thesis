#!/usr/bin/env python3
"""Collect exactly the Drive files the validation needs, one zip per condition.

The folder layout is not documented in one place; it is reconstructed from the
paths hard-coded in the Imaging notebooks (see ITEMS below for which notebook
writes each folder). Run it in Colab with Drive mounted, or locally on a copy:

  python export_from_drive.py --check            # only report what exists / is missing
  python export_from_drive.py --part masks       # masks + per-cell tables (small; enough for diagnostics)
  python export_from_drive.py --part all         # + projected images (for the audit and annotation)
  python export_from_drive.py --root <Thesis folder> --out <destination>

Zips keep the Drive-relative paths (Segmented/<cond>/..., Projected/<cond>/...),
so the commands in README.md work unchanged after unzipping into one folder.
Pixel sizes are read from one original .nd2 per acquisition series and
magnification (the TIF_Converted files carry no calibration); the stacks
themselves are not copied. Zips are split to stay under GitHub's 100 MB limit.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seg_eval as se  # noqa: E402

ROOT = '/content/drive/MyDrive/knowledge/University/Master/Thesis'
CONDITIONS = ['Static-x20', 'Static-x40', 'flow3-x20', '1.4Pa-x20', '1.4Pa-x40']

ALL = None
PA14 = ('1.4Pa-x20', '1.4Pa-x40')
# (part, folder, pattern, conditions it exists for, written by, why it is needed)
ITEMS = [
    ('masks', 'Segmented/{c}/Cell_merged_conservative', '*.tif', ALL, 'Segmentation/<c>/Cells.ipynb',
     'cell territories behind every reported morphometric and class'),
    ('masks', 'Segmented/{c}/Cell', '*_cell_mask.tif', ALL, 'Segmentation/<c>/Cells.ipynb',
     'territories before the enclave merge (stage comparison)'),
    ('masks', 'Segmented/{c}/Nuclei', '*.tif', ALL, 'Segmentation/<c>/Nuclei.ipynb',
     'nuclei the classifier counts (Polynucleated rule)'),
    ('masks', 'Segmented/{c}/Nuclei_filtered', '*.tif', PA14, 'Segmentation/1.4Pa-*/Nuclei.ipynb',
     'nuclei the 1.4 Pa seeds were built from (compare with Nuclei/)'),
    ('masks', 'Segmented/{c}/Seed', '*_segmented_cells.tif', ('Static-x20', 'Static-x40', 'flow3-x20'),
     'Segmentation/<c>/Seed.ipynb', 'merged nuclear seeds used by the watershed'),
    ('masks', 'Segmented/{c}/Seed_or', '*_segmented_cells.tif', ('1.4Pa-x20',),
     'Segmentation/1.4Pa-x20/Seed.ipynb (cell 0)', 'merged nuclear seeds used by the watershed'),
    ('masks', 'Segmented/{c}/Seed_gol', '*_segmented_cells.tif', ('1.4Pa-x40',),
     'Segmentation/1.4Pa-x40/Seed_new.ipynb', 'merged nuclear seeds used by the watershed'),
    ('masks', 'Segmented/{c}/Holes', '*_segmented*.tif', ALL, 'Segmentation/<c>/Holes.ipynb',
     'hole masks (1.4Pa-x20 also has the dilated version)'),
    ('masks', 'Segmented/{c}/Holes_masks', '*.tif', PA14, 'Segmentation/1.4Pa-*/Holes.ipynb (copy step)',
     'hole masks the 1.4 Pa watershed excluded'),
    ('masks', 'Analysis/{c}/Senescence_Results', '*.csv', ALL, 'Analysis/<c>/Senescence.ipynb',
     'per-cell features and rule-based classes'),
    ('images', 'Projected/{c}/Cadherins/tophat', '*.tif', ALL, 'Projection/<c>.ipynb',
     'VE-cadherin used by the membrane mask and watershed'),
    ('images', 'Projected/{c}/Cadherins/background', '*.tif', ALL, 'Projection/<c>.ipynb',
     'VE-cadherin used by the hole threshold'),
    ('images', 'Projected/{c}/Nuclei/tophat', '*.tif', ('Static-x20', '1.4Pa-x20'), 'Projection/<c>.ipynb',
     'nuclear channel the nuclei were segmented from'),
    ('images', 'Projected/{c}/Nuclei/background', '*.tif', ('Static-x40', 'flow3-x20', '1.4Pa-x40'),
     'Projection/<c>.ipynb', 'nuclear channel the nuclei were segmented from'),
    ('images', 'Projected/{c}/Golgi/tophat', '*.tif', ('Static-x20', 'Static-x40', '1.4Pa-x20', '1.4Pa-x40'),
     'Projection/<c>.ipynb', 'Golgi channel, context for annotation'),
]
COMMON = [
    ('masks', 'Analysis', 'combined_cell_data_adjusted.csv', 'Analysis/Comparisons.ipynb', 'pooled per-cell table'),
    ('masks', 'Analysis', 'descriptive_stats_by_pressure_cell_type_adjusted.csv', 'Analysis/Comparisons.ipynb',
     'the summary statistics as reported'),
    ('masks', 'Analysis/Holes', '*.csv', 'Analysis/Holes.ipynb', 'hole statistics as reported'),
]


def matches(root, folder, pattern):
    return sorted(p for p in glob.glob(os.path.join(root, folder, pattern)) if os.path.isfile(p))


def pixel_sizes(root):
    """Pixel size per (series, magnification), e.g. A1_20x, A1_40x, U_20x.

    Read from the original .nd2 files in 'Renamed Data' with the `nd2` package
    (pip install nd2). The files in TIF_Converted were written by 2Tiff.ipynb
    with a bare `imwrite(path, data)`, so they carry no calibration; they are
    only used if they happen to contain OME metadata.
    """
    found = {}
    try:
        import nd2
    except ImportError:
        nd2 = None
    for path in sorted(glob.glob(os.path.join(root, 'Renamed Data', '**', '*.nd2'), recursive=True)):
        key = se.sample_key(path)
        group = se.condition_of(key).split('_', 1)[1] if key else None
        if nd2 is None or group is None or group in found:
            continue
        try:
            with nd2.ND2File(path) as f:
                vs = f.voxel_size()
                row = dict(group=group, file=os.path.relpath(path, root), px_x_um=vs.x, px_y_um=vs.y,
                           z_step_um=vs.z, sizes=dict(f.sizes))
                try:
                    m = f.metadata.channels[0].microscope
                    row.update(objective=m.objectiveName, magnification=m.objectiveMagnification,
                               na=m.objectiveNumericalAperture)
                except Exception:
                    pass
            named = re.search(r'(\d+)x$', group)
            if named and row.get('magnification') and float(row['magnification']) != float(named.group(1)):
                # The A1 "40x" files record the 20x objective and its pixel size, although the
                # images are sampled twice as finely (nuclei four times larger in pixels).
                row['note'] = (f"metadata objective {row['magnification']:g}x does not match the file name "
                               f"({named.group(1)}x): do not use this pixel size")
                print(f"warning: {path}: {row['note']}", file=sys.stderr)
            found[group] = row
        except Exception as e:
            print(f"warning: could not read {path}: {e}", file=sys.stderr)
    import tifffile
    for path in sorted(glob.glob(os.path.join(root, 'TIF_Converted', '**', '*.tif*'), recursive=True)):
        key = se.sample_key(path)
        group = se.condition_of(key).split('_', 1)[1] if key else None
        if group is None or group in found:
            continue
        try:
            with tifffile.TiffFile(path) as tf:
                xml = tf.ome_metadata
        except Exception:
            continue
        px = re.search(r'PhysicalSizeX="([0-9.eE+-]+)"', xml or '')
        if px:
            py = re.search(r'PhysicalSizeY="([0-9.eE+-]+)"', xml)
            found[group] = dict(group=group, file=os.path.relpath(path, root), px_x_um=float(px.group(1)),
                                px_y_um=float(py.group(1)) if py else float(px.group(1)), source='OME-XML')
    return found, nd2 is not None


def duplicate_count(files):
    """Files that are the same product of the same field, e.g. 'x (1).tif' next to 'x.tif'."""
    seen, dup = set(), 0
    for p in files:
        key = se.sample_key(p) or ''
        rest = re.sub(r'\s*\(\d+\)', '', os.path.basename(p).replace(key, ''))
        dup += (key, rest) in seen
        seen.add((key, rest))
    return dup


class SplitZip:
    """Zip writer that starts a new part before a file would exceed the size limit."""

    def __init__(self, stem, limit_mb):
        self.stem, self.limit, self.n, self.z, self.paths = stem, limit_mb * 1e6, 0, None, []
        self._next()

    def _next(self):
        if self.z:
            self.z.close()
        self.n += 1
        path = f"{self.stem}.zip" if self.n == 1 else f"{self.stem}_part{self.n}.zip"
        self.paths.append(path)
        self.z = zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6)

    def write(self, src, arcname):
        if self.z.fp.tell() > 0 and self.z.fp.tell() + os.path.getsize(src) * 0.9 > self.limit:
            self._next()
        self.z.write(src, arcname)

    def writestr(self, arcname, data):
        self.z.writestr(arcname, data)

    def close(self):
        self.z.close()
        if len(self.paths) > 1 and os.path.exists(self.paths[0]):
            first = f"{self.stem}_part1.zip"
            os.replace(self.paths[0], first)
            self.paths[0] = first
        return self.paths


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split('\n\n', 1)[1])
    ap.add_argument('--root', default=ROOT, help='the Thesis folder (default: %(default)s)')
    ap.add_argument('--out', help='destination folder (default: <root>/validation_export)')
    ap.add_argument('--part', choices=['masks', 'images', 'all'], default='masks')
    ap.add_argument('--conditions', nargs='+', default=CONDITIONS)
    ap.add_argument('--check', action='store_true', help='report only, write nothing')
    ap.add_argument('--split-mb', type=float, default=95, help="max zip size (GitHub's per-file limit is 100 MB)")
    args = ap.parse_args(argv)
    if not os.path.isdir(args.root):
        raise SystemExit(f"{args.root} not found (in Colab: from google.colab import drive; drive.mount('/content/drive'))")
    out = args.out or os.path.join(args.root, 'validation_export')
    parts = {'masks', 'images'} if args.part == 'all' else {args.part}

    report = ['| condition | folder | files | MB | duplicates | needed for |', '|---|---|---|---|---|---|']
    bundles = {}
    for c in args.conditions:
        for part, folder, pattern, only, _, why in ITEMS:
            if part not in parts or (only and c not in only):
                continue
            f = folder.format(c=c)
            files = matches(args.root, f, pattern)
            dup = duplicate_count(files)
            mb = sum(os.path.getsize(p) for p in files) / 1e6
            report.append(f"| {c} | {f} | {len(files) or '**missing/empty**'} | {mb:.0f} | {dup or ''} | {why} |")
            bundles.setdefault(c, []).extend(files)
    for part, folder, pattern, _, why in COMMON:
        if part in parts:
            files = matches(args.root, folder, pattern)
            report.append(f"| all | {folder}/{pattern} | {len(files) or '**missing**'} | "
                          f"{sum(os.path.getsize(p) for p in files) / 1e6:.0f} | | {why} |")
            bundles.setdefault('common', []).extend(files)
    sizes, have_nd2 = pixel_sizes(args.root)
    report += ['', '| series_magnification | pixel x (um) | pixel y (um) | z step (um) | objective | read from | note |',
               '|---|---|---|---|---|---|---|']
    for g, r in sorted(sizes.items()):
        report.append(f"| {g} | {r['px_x_um']:.4f} | {r['px_y_um']:.4f} | {r.get('z_step_um', float('nan')):.3f} | "
                      f"{r.get('objective', '')} {r.get('magnification', '')} NA {r.get('na', '')} | {r['file']} | "
                      f"{r.get('note', '')} |")
    if not sizes:
        report.append('| none found | | | | | ' + ('no readable .nd2 in Renamed Data' if have_nd2 else
                                                 'run `pip install nd2` so the .nd2 files can be read') + ' | |')
    text = '\n'.join(report)
    print(text)
    if args.check:
        return

    os.makedirs(out, exist_ok=True)
    tag = 'all' if args.part == 'all' else args.part
    bundles.setdefault('common', [])
    for name, files in bundles.items():
        if not files and name != 'common':
            continue
        z = SplitZip(os.path.join(out, f"{name}_{tag}"), args.split_mb)
        for p in files:
            z.write(p, os.path.relpath(p, args.root))
        if name == 'common':
            z.writestr('metadata/pixel_sizes.json', json.dumps(sizes, indent=2, default=str))
            z.writestr('metadata/export_report.md', text + '\n')
        for path in z.close():
            mb = os.path.getsize(path) / 1e6
            print(f"wrote {path} ({mb:.0f} MB)" + ("  <-- a single file exceeds --split-mb" if mb > 100 else ''))


if __name__ == '__main__':
    main()

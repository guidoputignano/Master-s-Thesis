#!/usr/bin/env python3
"""Export what the A1 .nd2 files hold beyond the projected TIFFs: metadata, sharpest planes, DAPI sums.

The projected TIFFs lost the z-stacks and all calibration. This script goes back to the
original .nd2 files in 'Renamed Data' and writes, per field:

  * one metadata row: objective, recorded and corrected pixel size, z step and number of
    planes, channel names, exposure, stage X/Y/Z and acquisition time;
  * the sharpest plane of every channel (focus by variance of the Laplacian), for nuclear
    texture and shape;
  * the sum of the DAPI planes, for DNA content (integrated DAPI per nucleus);
  * optionally (--stacks) the full z-stack.

The A1 experiment only; flow3 (series U) is not used, its conditions being unknown. The
"40x" files record the 20x objective and its 0.429 um pixel, although they are sampled
twice as finely: their corrected pixel is 0.2145 um (Imaging/pipeline_v2/features.py).

Run it in Colab with Drive mounted:

  !pip install nd2 tifffile
  from google.colab import drive; drive.mount('/content/drive')
  !python export_nd2.py --check                  # list the files and their metadata, write nothing
  !python export_nd2.py                          # metadata + planes, zipped
  !python export_nd2.py --stacks --keys seq006   # also the z-stacks of fields whose key contains seq006

Output in --out (default <root>/nd2_export): nd2_metadata.csv, nd2_focus.csv (sharpness of
every plane), and nd2_planes*.zip (plus nd2_stacks*.zip). Zips are split below 95 MB,
GitHub's per-file limit being 100 MB. Images are zlib-compressed OME-TIFFs that carry the
corrected pixel size.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import os
import re
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seg_eval as se  # noqa: E402
from export_from_drive import ROOT, SplitZip  # noqa: E402

# The A1 experiment only (see export_from_drive.CONDITIONS); keys look like 1.4Pa_A1_20dec21_40x_...
CONDITIONS = ['0Pa_A1_20x', '0Pa_A1_40x', '1.4Pa_A1_20x', '1.4Pa_A1_40x']
CORRECT_PX_UM = {'20x': 0.429, '40x': 0.2145}   # as Imaging/pipeline_v2/features.PIXEL_UM
DAPI_RE = re.compile(r'dapi|hoechst|405|nuc', re.I)
META_FIELDS = ['key', 'condition', 'file', 'objective', 'recorded_magnification', 'na',
               'recorded_px_um', 'px_um', 'px_note', 'z_step_um', 'n_z', 'n_c', 'height', 'width',
               'dtype', 'channels', 'exposure_ms', 'stage_x_um', 'stage_y_um', 'stage_z_um',
               'z_first_um', 'z_last_um', 'acquired', 'dapi_channel', 'error']


def find_nd2(root, conditions=CONDITIONS, keys=None):
    """(path, key, condition) for every A1 .nd2 in 'Renamed Data', in key order."""
    out = []
    for path in glob.glob(os.path.join(root, 'Renamed Data', '**', '*.nd2'), recursive=True):
        key = se.sample_key(path)
        if key is None:
            continue
        cond = se.condition_of(key)
        if cond not in conditions or (keys and not any(k in key for k in keys)):
            continue
        out.append((path, key, cond))
    return sorted(out, key=lambda t: t[1])


def _get(fn, default=None):
    try:
        v = fn()
        return default if v is None else v
    except Exception:
        return default


def _julian_to_iso(jdn):
    """Nikon stores times as a Julian day number with the fraction of the day."""
    if not jdn:
        return None
    seconds = round((float(jdn) - 2440587.5) * 86400.0)       # UTC, to the nearest second
    return (dt.datetime(1970, 1, 1) + dt.timedelta(seconds=seconds)).isoformat()


def read_metadata(f, path, key, cond, root):
    """One metadata row; every field is read defensively, so a missing entry leaves a blank."""
    sizes = dict(_get(lambda: f.sizes, {}))
    vs = _get(f.voxel_size)
    ch = _get(lambda: f.metadata.channels, [])
    mic = _get(lambda: ch[0].microscope)
    names = [_get(lambda c=c: c.channel.name, f'C{i}') for i, c in enumerate(ch)] or \
        [f'C{i}' for i in range(sizes.get('C', 1))]
    mag_name = cond.split('_')[-1]
    rec_mag = _get(lambda: float(mic.objectiveMagnification))
    rec_px = _get(lambda: float(vs.x))
    px, note = CORRECT_PX_UM[mag_name], ''
    if rec_mag and float(mag_name[:-1]) != rec_mag:
        note = f'metadata records {rec_mag:g}x; pixel corrected to {px} um from the file name ({mag_name})'
    elif rec_px and abs(rec_px - px) > 0.005:
        note = f'recorded pixel {rec_px:.4f} um differs from the calibration {px} um'
    pos = _get(lambda: f.frame_metadata(0).channels[0].position.stagePositionUm)
    n_z = sizes.get('Z', 1)
    z_last = _get(lambda: f.frame_metadata(_seq_index(f, n_z - 1)).channels[0].position.stagePositionUm[2])
    text = _get(lambda: f.text_info, {})
    exposure = None
    m = re.search(r'Exposure:\s*([\d.]+)\s*ms', str(text.get('capturing', '')))
    if m:
        exposure = float(m.group(1))
    acquired = _get(lambda: _julian_to_iso(f.frame_metadata(0).channels[0].time.absoluteJulianDayNumber)) \
        or text.get('date')
    dapi = next((i for i, n in enumerate(names) if DAPI_RE.search(str(n))), None)
    return dict(key=key, condition=cond, file=os.path.relpath(path, root),
                objective=_get(lambda: mic.objectiveName), recorded_magnification=rec_mag,
                na=_get(lambda: float(mic.objectiveNumericalAperture)), recorded_px_um=rec_px, px_um=px,
                px_note=note, z_step_um=_get(lambda: float(vs.z)), n_z=n_z, n_c=sizes.get('C', 1),
                height=sizes.get('Y'), width=sizes.get('X'), dtype=str(_get(lambda: f.dtype, '')),
                channels='|'.join(map(str, names)), exposure_ms=exposure,
                stage_x_um=pos[0] if pos else None, stage_y_um=pos[1] if pos else None,
                stage_z_um=pos[2] if pos else None, z_first_um=pos[2] if pos else None, z_last_um=z_last,
                acquired=acquired, dapi_channel=dapi, error='')


def _seq_index(f, z):
    """Sequence (frame) index of plane z at the first position of any other loop."""
    for i, idx in enumerate(_get(lambda: f.loop_indices, ()) or ()):
        if idx.get('Z', 0) == z and all(v == 0 for k, v in idx.items() if k != 'Z'):
            return i
    return z


def zcyx(f):
    """The image as (Z, C, Y, X); other loops (T, P) are reduced to their first index."""
    a = np.asarray(f.asarray())
    axes = list(dict(f.sizes))
    for ax in [a_ for a_ in axes if a_ not in 'ZCYX']:
        a = np.take(a, 0, axis=axes.index(ax))
        axes.remove(ax)
    for ax in 'ZC':
        if ax not in axes:
            a = a[np.newaxis]
            axes.insert(0, ax)
    return np.transpose(a, [axes.index(ax) for ax in 'ZCYX'])


def sharpness(plane):
    """Variance of the Laplacian of a lightly smoothed plane (higher = sharper)."""
    from scipy import ndimage as ndi
    return float(ndi.laplace(ndi.gaussian_filter(plane.astype(np.float32), 1.0)).var())


def best_planes(stack):
    """Sharpest z index per channel and the full sharpness profile, for a (Z, C, Y, X) stack."""
    prof = np.array([[sharpness(stack[z, c]) for c in range(stack.shape[1])] for z in range(stack.shape[0])])
    return prof.argmax(0), prof


def write_tiff(path, arr, px_um, axes):
    import tifffile
    md = {'axes': axes, 'PhysicalSizeX': px_um, 'PhysicalSizeXUnit': 'µm',
          'PhysicalSizeY': px_um, 'PhysicalSizeYUnit': 'µm'}
    tifffile.imwrite(path, arr, ome=True, metadata=md, compression='zlib')


def export(root, out, conditions=CONDITIONS, keys=None, stacks=False, dapi_channel=None,
           split_mb=95.0, check=False):
    import nd2
    files = find_nd2(root, conditions, keys)
    if not files:
        raise SystemExit(f"no A1 .nd2 files under {os.path.join(root, 'Renamed Data')}")
    rows, focus = [], []
    zp = zs = None
    tmp = tempfile.mkdtemp()
    if not check:
        os.makedirs(out, exist_ok=True)
        zp = SplitZip(os.path.join(out, 'nd2_planes'), split_mb)
        zs = SplitZip(os.path.join(out, 'nd2_stacks'), split_mb) if stacks else None
    for path, key, cond in files:
        try:
            with nd2.ND2File(path) as f:
                row = read_metadata(f, path, key, cond, root)
                if not check:
                    stack = zcyx(f)
                    best, prof = best_planes(stack)
                    names = row['channels'].split('|')
                    dapi = dapi_channel if dapi_channel is not None else row['dapi_channel']
                    for c, z in enumerate(best):
                        focus += [dict(key=key, channel=names[c] if c < len(names) else f'C{c}', z=zi,
                                       sharpness=float(prof[zi, c]), best=int(zi == z))
                                  for zi in range(prof.shape[0])]
                        name = re.sub(r'\W+', '', names[c] if c < len(names) else f'C{c}') or f'C{c}'
                        p = os.path.join(tmp, f'{key}_best_{name}.ome.tif')
                        write_tiff(p, stack[z, c], row['px_um'], 'YX')
                        zp.write(p, f'nd2_planes/{cond}/{os.path.basename(p)}')
                        os.remove(p)
                    row['best_z'] = '|'.join(map(str, best.tolist()))
                    if dapi is not None:
                        p = os.path.join(tmp, f'{key}_dapi_sum.ome.tif')
                        write_tiff(p, stack[:, dapi].astype(np.uint32).sum(0, dtype=np.uint32),
                                   row['px_um'], 'YX')
                        zp.write(p, f'nd2_planes/{cond}/{os.path.basename(p)}')
                        os.remove(p)
                    if zs is not None:
                        p = os.path.join(tmp, f'{key}_stack.ome.tif')
                        write_tiff(p, stack, row['px_um'], 'ZCYX')
                        zs.write(p, f'nd2_stacks/{cond}/{os.path.basename(p)}')
                        os.remove(p)
        except Exception as e:
            row = dict(key=key, condition=cond, file=os.path.relpath(path, root), error=repr(e))
            print(f'warning: {path}: {e!r}', file=sys.stderr)
        rows.append(row)
        print(f"{key}: {row.get('n_z')} planes x {row.get('n_c')} channels, "
              f"px {row.get('px_um')} um{'  (' + row['px_note'] + ')' if row.get('px_note') else ''}"
              f"{', stage ' + format(row['stage_x_um'], '.0f') + ' / ' + format(row['stage_y_um'], '.0f') + ' um' if row.get('stage_x_um') is not None else ''}",
              flush=True)
    if check:
        return rows, [], []
    fields = META_FIELDS + ['best_z']
    for name, data, cols in (('nd2_metadata.csv', rows, fields),
                             ('nd2_focus.csv', focus, ['key', 'channel', 'z', 'sharpness', 'best'])):
        with open(os.path.join(out, name), 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
            w.writeheader()
            w.writerows(data)
        zp.write(os.path.join(out, name), f'nd2_planes/{name}')
    with open(os.path.join(out, 'nd2_metadata.json'), 'w') as fh:
        json.dump(rows, fh, indent=1, default=str)
    planes = zp.close()
    stack_zips = zs.close() if zs is not None else []
    for p in planes + stack_zips:
        print(f'wrote {p} ({os.path.getsize(p) / 1e6:.0f} MB)')
    return rows, planes, stack_zips


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split('\n\n', 1)[1])
    ap.add_argument('--root', default=ROOT, help='the Thesis folder (default: %(default)s)')
    ap.add_argument('--out', help='destination folder (default: <root>/nd2_export)')
    ap.add_argument('--conditions', nargs='+', default=CONDITIONS)
    ap.add_argument('--keys', nargs='+', help='only fields whose key contains one of these strings')
    ap.add_argument('--stacks', action='store_true', help='also write the full z-stacks')
    ap.add_argument('--dapi-channel', type=int, help='DAPI channel index, if the channel names do not say')
    ap.add_argument('--split-mb', type=float, default=95, help="max zip size (GitHub's per-file limit is 100 MB)")
    ap.add_argument('--check', action='store_true', help='list the files and their metadata, write nothing')
    args = ap.parse_args(argv)
    if not os.path.isdir(args.root):
        raise SystemExit(f"{args.root} not found (in Colab: from google.colab import drive; drive.mount('/content/drive'))")
    export(args.root, args.out or os.path.join(args.root, 'nd2_export'), args.conditions, args.keys,
           args.stacks, args.dapi_channel, args.split_mb, args.check)


if __name__ == '__main__':
    main()

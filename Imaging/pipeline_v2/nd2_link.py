#!/usr/bin/env python3
"""Link the original .nd2 acquisitions to the analysed fields, and measure DNA content.

The .nd2 files in the data repository (``Original/``) carry Nafsika Chala's file names
(``H_P3-2-1.4Pa_A_70c-30T_19-20.12.21-001.nd2``), not the field keys of the projections
(``1.4Pa_A1_19dec21_20xA_L2RA_FlatA_seq001``). Three steps:

* ``meta``: per file, the acquisition metadata (objective, pixel size, z-step, channels,
  stage position, time). Files are read from the Git LFS object store, so the repository
  does not need them checked out (``git lfs fetch`` is enough).
* ``map``: each field key to its file, by image content (the DAPI maximum projection
  against the projected nuclei of the field, both reduced to 256 x 256).
* ``dna``: per Cellpose nucleus, the DAPI signal summed over the z-stack minus the local
  background ring (1.5-4.5 um outside the nucleus), and a DNA index: that sum divided by the
  median of the field's normal-size nuclei (20th-60th area percentile, below the enlarged
  cut-off), so 1 is the field's typical 2N nucleus and 2 is doubled DNA (4N).
* ``calibration``: the per-file calibration table, ``calibration.csv`` next to this script:
  each field key with its file, the recorded optics (objective, zoom, pixel size) and the
  pixel size used, with its source.

Pixel size: every A1 file records the same stale optics state (20x objective, zoom 1.515,
0.429 um/px), also the files taken with the stronger objective. The pixel size used comes from
the stage calibration, 0.650 um at 20x and 0.325 um at 40x (``features.PIXEL_UM``), not from the
file.

  python nd2_link.py meta --repo data-mt --out nd2
  python nd2_link.py map  --repo data-mt --out nd2
  python nd2_link.py dna  --repo data-mt --masks v2r2_masks --quality quality.csv --out nd2
  python nd2_link.py calibration --repo data-mt --out nd2       # -> calibration.csv next to this script
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'Validation'))
sys.path.insert(0, HERE)
import features as ft  # noqa: E402

NAME = re.compile(r'H_P3-2-(Static|1\.4Pa)_A_70c-30T_(\d\d)-\d\d\.12\.21(_40x)?-(\d{3})(again)?\.nd2')
FOLDER = {('0Pa', '20x'): 'Static-x20', ('0Pa', '40x'): 'Static-x40', ('1.4Pa', '20x'): '1.4Pa-x20',
          ('1.4Pa', '40x'): '1.4Pa-x40'}
DAPI = 'WF 395'
ENLARGED_UM2 = 95.7 * ft.AREA     # 219.7 um2: the static antimode of by_shear.py (95.7 at the recorded pixel)
BINS = [0, 0.6, 1.4, 1.7, 2.6, 99]
CLASSES = ['<0.6', '2N', 'S', '4N', '>2.6']


def lfs_files(repo):
    """(path in the repository, path of the LFS object) of every .nd2."""
    out = subprocess.run(['git', 'lfs', 'ls-files', '-l'], cwd=repo, capture_output=True, text=True, check=True).stdout
    for line in out.strip().splitlines():
        oid, _, path = line.split(' ', 2)
        if path.endswith('.nd2'):
            yield path, os.path.join(repo, '.git', 'lfs', 'objects', oid[:2], oid[2:4], oid)


def meta(repo):
    import nd2
    rows = []
    for path, obj in lfs_files(repo):
        m = NAME.search(os.path.basename(path))
        if not m:
            continue
        r = dict(path=path, obj=obj, shear='1.4Pa' if m.group(1) == '1.4Pa' else '0Pa', date=m.group(2) + 'dec21',
                 named40=bool(m.group(3)), seq=int(m.group(4)), again=bool(m.group(5)))
        with nd2.ND2File(obj) as f:
            r['sizes'] = json.dumps(dict(f.sizes))
            vs = f.voxel_size()
            r.update(recorded_px_um=vs.x, z_um=vs.z)                  # the recorded calibration (stale)
            ch = f.metadata.channels
            r['channels'] = '|'.join(c.channel.name for c in ch)
            mic = ch[0].microscope
            r.update(objective=mic.objectiveName, mag=mic.objectiveMagnification, na=mic.objectiveNumericalAperture,
                     zoom=mic.zoomMagnification)
            pos = f.frame_metadata(0).channels[0]
            r.update(x_um=pos.position.stagePositionUm[0], y_um=pos.position.stagePositionUm[1],
                     z0_um=pos.position.stagePositionUm[2], jdn=pos.time.absoluteJulianDayNumber)
        rows.append(r)
    d = pd.DataFrame(rows).sort_values(['shear', 'date', 'named40', 'seq', 'again'])
    d['acquired'] = pd.to_datetime((d.jdn - 2440587.5) * 86400, unit='s').dt.round('s')
    return d


def _small(a):
    from skimage.transform import resize
    a = resize(np.asarray(a, np.float32), (256, 256), anti_aliasing=True)
    a = a - a.mean()
    return a / (a.std() + 1e-9)


def mapping(repo, m):
    """Field key -> .nd2 file with the most similar DAPI projection (same shear, date, objective)."""
    import nd2
    import tifffile
    cache = {}
    for _, r in m.iterrows():
        with nd2.ND2File(r.obj) as f:
            names = [c.channel.name for c in f.metadata.channels]
            cache[r.path] = _small(f.asarray()[:, names.index(DAPI)].max(0))
    rows = []
    for p in sorted(glob.glob(f'{repo}/Projection/*/Nuclei/*/*.tif')):
        hit = re.search(r'denoised_(.+?)_Nuclei', os.path.basename(p))
        if not hit:
            continue
        key = hit.group(1)
        shear, _, date, mag = key.split('_')[:4]
        img = _small(tifffile.imread(p))
        cand = m[(m.shear == shear) & (m.date == date) & (m.named40 == mag.startswith('40x'))]
        if not len(cand):
            continue
        corr = {r.path: float((img * cache[r.path]).mean()) for _, r in cand.iterrows()}
        best = max(corr, key=corr.get)
        rows.append(dict(key=key, nd2_best=os.path.basename(best), corr_best=corr[best]))
    return pd.DataFrame(rows).drop_duplicates('key')


def dna_sums(stack, nuclei, um):
    """Per nucleus label: summed DAPI (z-summed stack, nucleus grown by 1 um) minus the median
    of a background ring 1-3 um outside, times the area; and the nucleus area in um^2."""
    from scipy import ndimage as ndi
    from skimage.measure import regionprops
    st = np.asarray(stack, np.float32)
    st = st - np.percentile(st, 5, axis=(1, 2), keepdims=True)
    s = st.sum(0)
    r_in, r_bg = int(round(1.0 * ft.SCALE / um)), int(round(3.0 * ft.SCALE / um))     # 1.5 and 4.5 um
    grown = ndi.grey_dilation(nuclei, size=(2 * r_in + 1, 2 * r_in + 1))
    grown[nuclei > 0] = nuclei[nuclei > 0]
    ring = ndi.grey_dilation(nuclei, size=(2 * r_bg + 1, 2 * r_bg + 1))
    ring[grown > 0] = 0
    n = int(nuclei.max())
    bg = ndi.median(s, ring, index=np.arange(1, n + 1)) if n else []
    counts = np.bincount(nuclei.ravel(), minlength=n + 1)
    rows = []
    for rp in regionprops(grown, intensity_image=s):
        b = bg[rp.label - 1] if np.isfinite(bg[rp.label - 1]) else 0.0
        rows.append(dict(label=rp.label, dna_raw=float(rp.image_intensity[rp.image].sum() - b * rp.area),
                         area_um2=float(counts[rp.label]) * um * um))
    return pd.DataFrame(rows)


def dna_index(d, enlarged_um2=ENLARGED_UM2):
    """Normalise each field by its normal-size nuclei; classify 2N / S / 4N."""
    d = d[d.area_um2 >= 25 * ft.AREA].copy()          # 57 um2
    d['enlarged'] = d.area_um2 > enlarged_um2

    def norm(g):
        ref = g[(~g.enlarged) & g.area_um2.between(g.area_um2.quantile(0.2), g.area_um2.quantile(0.6))]
        return g.assign(dna=g.dna_raw / ref.dna_raw.median())
    d = pd.concat([norm(g) for _, g in d.groupby('key')], ignore_index=True)
    d['cls'] = pd.cut(d.dna, BINS, labels=CLASSES)
    return d


def dna(repo, masks, m, k, skip):
    import nd2
    import tifffile
    rows = []
    for _, r in k.iterrows():
        key = r.key
        if key in skip:
            continue
        shear, _, _, magtag = key.split('_')[:4]
        mag = '40x' if magtag.startswith('40x') else '20x'
        nuc_file = f'{masks}/{FOLDER[(shear, mag)]}/{key}_v2_nuclei.tif'
        if not os.path.exists(nuc_file):
            continue
        obj = m[m.path.str.endswith(r.nd2_best)].obj.iloc[0]
        with nd2.ND2File(obj) as f:
            names = [c.channel.name for c in f.metadata.channels]
            stack = f.asarray()[:, names.index(DAPI)]
        t = dna_sums(stack, tifffile.imread(nuc_file), ft.pixel_um(key))
        t.insert(0, 'key', key)
        t['shear'], t['date'], t['mag'] = shear, key.split('_')[2], mag
        rows.append(t)
        print(key, len(t), flush=True)
    return dna_index(pd.concat(rows, ignore_index=True))


def calibration(m, k):
    """Per-file calibration: field key, file, recorded optics and the pixel size used (stage)."""
    m = m.assign(nd2=m.path.map(os.path.basename))
    d = k[['key', 'nd2_best']].rename(columns={'nd2_best': 'nd2'}).merge(m, on='nd2', how='left')
    mag = d.key.map(lambda key: '40x' if key.split('_')[3].startswith('40x') else '20x')
    rec = d['recorded_px_um'] if 'recorded_px_um' in d else d['px_um']
    return pd.DataFrame(dict(
        key=d.key, nd2=d.nd2, recorded_objective=d.objective, recorded_zoom=d.zoom, recorded_px_um=rec,
        magnification=mag, px_um=mag.map(ft.PIXEL_UM),
        source=np.where(mag == '20x',
                        'stage: 6 overlapping position pairs 103-668 um apart give 0.649-0.653 um/px, '
                        'e.g. 1.4Pa 19dec21 seq013/seq016, 102.8 um = 158 px (stage_calibration.py)',
                        'half the 20x pixel (objective ratio; nuclei 3.8-4.0x larger in pixels); '
                        'stage: Static 19dec21 40x-003/004, 208.8 um = 637 px, 0.328 um/px'))).sort_values('key')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('step', choices=['meta', 'map', 'dna', 'calibration'])
    ap.add_argument('--repo', required=True, help='data repository with Original/*.nd2 in Git LFS')
    ap.add_argument('--out', required=True)
    ap.add_argument('--masks', help='refine_v2.py output (<cond>/<key>_v2_nuclei.tif), for dna')
    ap.add_argument('--quality', help='quality.py table; fields marked exclude are left out (dna)')
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    if a.step == 'meta':
        d = meta(a.repo)
        d.to_csv(f'{a.out}/nd2_meta.csv', index=False)
        print(d[['path', 'objective', 'zoom', 'recorded_px_um', 'z_um', 'x_um', 'y_um', 'acquired']].to_string(index=False))
    elif a.step == 'map':
        d = mapping(a.repo, pd.read_csv(f'{a.out}/nd2_meta.csv'))
        d.to_csv(f'{a.out}/key_map.csv', index=False)
        print(len(d), 'keys mapped; lowest correlation', round(d.corr_best.min(), 3))
    elif a.step == 'calibration':
        d = calibration(pd.read_csv(f'{a.out}/nd2_meta.csv'), pd.read_csv(f'{a.out}/key_map.csv'))
        d.to_csv(os.path.join(HERE, 'calibration.csv'), index=False)
        print(d.groupby(['magnification', 'recorded_zoom', 'recorded_px_um', 'px_um']).size().to_string())
    else:
        import analyze as an
        d = dna(a.repo, a.masks, pd.read_csv(f'{a.out}/nd2_meta.csv'), pd.read_csv(f'{a.out}/key_map.csv'),
                an.excluded_fields(a.quality))
        d.to_csv(f'{a.out}/dna_index.csv', index=False)
        print(pd.crosstab([d.shear, d.enlarged], d.cls, normalize='index').round(3).to_string())


if __name__ == '__main__':
    main()

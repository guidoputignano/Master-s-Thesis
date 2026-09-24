#!/usr/bin/env python3
"""Second blind session: v1 vs v2.1, new objects, three views per crop.

Round 1 (build_verdicts.py) showed that v1 and v2 cells are both mostly correct, and
that v2's errors were cells cut short around the nucleus. This round tests v2.1 on
objects never shown before. Every candidate (cell or gap) whose mask comes within
10 um of any round-1 object's mask is excluded, and no object is shown twice.

* ``cell``: interior cells of v1 and v2.1, stratified by agreement (as in round 1);
* ``multinucleated``: cells with at least two nuclei, per method. The methods disagree
  twofold on this fraction, and v1's "Poly" calls failed in round 1;
* ``enlarged``: v2.1 cells in the enlarged mixture component;
* ``gap``: gap components drawn with replacement, with probability proportional to area,
  within each method x condition. The multiplicity-weighted yes-rate estimates the share
  of that condition's gap area that is real, and conditions are combined by gap area.
  Tiny components (which the expert could not judge) are rarely shown.

Views (number keys on the page): 1 junctions (β-catenin top-hat + nuclei), 2 haze
(β-catenin without top-hat, low range stretched: cytoplasm is grey, bare substrate is
black), 3 Golgi (one Golgi per cell helps separate one cell from two).

  python build_verdicts2.py --root data-mt --v2r v2r_masks --analysis v2r_analysis \
      --round1 session1/key.csv --round1-v2 v2_masks --round1-v2-gaps v2_analysis --out session2
  python build_verdicts.py score --session session2/index.html --code 'VS...'
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage
from skimage.measure import label as cc_label

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'Validation'))
sys.path.insert(0, HERE)
import features as ft  # noqa: E402
import analyze as an  # noqa: E402
import build_verdicts as bv  # noqa: E402
import segment_v2 as sg  # noqa: E402
from audit_gallery import normalise  # noqa: E402
import verdict_session as vs  # noqa: E402

N_CELL, N_MULTI, N_ENLARGED, N_GAP = 5, 4, 4, 3
EXCLUDE_UM = 10.0
Q_MULTI = bv.Q_CELL
NOTE = ('Round 2. Press <kbd>1</kbd> <kbd>2</kbd> <kbd>3</kbd> to switch views: junctions, haze '
        '(cytoplasm grey, bare substrate black), Golgi (yellow). A gap is bare substrate: black in '
        'the haze view, with no nucleus. Use U freely when you cannot tell.')


def view_images(root, cond, key):
    """{name: RGB float image} for one field."""
    cad_p, nuc_p = sg.inputs(root, cond)[key]
    cad, nuc = tifffile.imread(cad_p), tifffile.imread(nuc_p)
    raw = tifffile.imread(an.index(f'{root}/Projection/{cond}/Cadherins/background')[key]).astype(float)
    golgi_p = an.index(f'{root}/Projection/{cond}/Golgi/tophat').get(key)
    n = normalise(nuc)
    views = {'junctions': bv.composite(cad, nuc)}
    h = np.clip((raw - np.percentile(raw, 0.5)) / (np.percentile(raw, 60) - np.percentile(raw, 0.5) + 1e-9), 0, 1) ** 0.7
    views['haze'] = np.clip(np.stack([h, np.maximum(h, 0.6 * n), h], -1), 0, 1)
    if golgi_p:
        g = normalise(tifffile.imread(golgi_p))
        m = 0.45 * normalise(cad)
        views['golgi'] = np.clip(np.stack([np.maximum(g, m), np.maximum(g, n), m + n * 0.8], -1), 0, 1)
    return views


def round1_zone(root, round1, v2_masks, v2_gap_dir):
    """{key: boolean mask} covering every round-1 object dilated by EXCLUDE_UM."""
    k = pd.read_csv(round1)
    zones = {}
    for (cond, key), g in k.groupby(['folder', 'key']):
        um = ft.pixel_um(key)
        c1 = tifffile.imread(an.index(f'{root}/Segmented/{cond}/Cell_merged_conservative')[key])
        z = np.zeros(c1.shape, bool)
        lazy = {}

        def get(name):
            if name not in lazy:
                if name == 'v2cells':
                    lazy[name] = tifffile.imread(an.index(f'{v2_masks}/{cond}', '*_v2_cells.tif')[key])
                elif name == 'v1gaps':
                    hd, hp = an.V1_HOLES[cond]
                    lazy[name] = cc_label(tifffile.imread(an.index(f'{root}/Segmented/{cond}/{hd}', hp)[key]) > 0,
                                          connectivity=2)
                else:
                    lazy[name] = cc_label(bv.v2_gaps(v2_gap_dir, cond, key), connectivity=2)
            return lazy[name]
        for _, r in g.iterrows():
            if r.block == 'gap':
                lab = get('v1gaps' if r.method == 'v1' else 'v2gaps')
            else:
                lab = c1 if r.method == 'v1' else get('v2cells')
            obj = lab == r.label
            if r.block == 'gap' and abs(obj.sum() * um ** 2 - r.area_um2) > 1e-6 * max(1.0, r.area_um2):
                raise ValueError(f'{key} gap {r.label}: area differs from round 1 '
                                 '(pass the analysis folder whose gaps round 1 showed)')
            z |= obj
        zones[key] = ndimage.distance_transform_edt(~z) * um <= EXCLUDE_UM      # a true 10 um disk
    return zones


def excluded_labels(zones, key, labels):
    """Labels of a label image that touch the round-1 exclusion zone of that field."""
    if key not in zones:
        return set()
    return set(np.unique(labels[zones[key]]).tolist()) - {0}


def item_mask(r, root, v2r, analysis, cache):
    """Mask of one round-2 item (cells: v1 or v2.1 labels; gaps: v1 or v2.1 components)."""
    cond, key = r['folder'], r['key']
    tag = (r['block'] == 'gap', r['method'], key)
    if tag not in cache:
        if r['block'] == 'gap':
            if r['method'] == 'v1':
                hd, hp = an.V1_HOLES[cond]
                cache[tag] = cc_label(tifffile.imread(an.index(f'{root}/Segmented/{cond}/{hd}', hp)[key]) > 0,
                                      connectivity=2)
            else:
                cache[tag] = cc_label(bv.v2_gaps(analysis, cond, key), connectivity=2)
        elif r['method'] == 'v1':
            cache[tag] = tifffile.imread(an.index(f'{root}/Segmented/{cond}/Cell_merged_conservative')[key])
        else:
            cache[tag] = tifffile.imread(an.index(f'{v2r}/{cond}', '*_v2_cells.tif')[key])
    return cache[tag] == r['label']


def cross_method_dedupe(items, args, rng, max_tries=50):
    """Replace v2.1 items that show the same object as a v1 item (either lies more than half
    inside the other), drawing again from the same block, condition and stratum."""
    cache, pools = {}, getattr(args, 'pools', {})
    v1 = [it for it in items if it['method'] == 'v1' and it['block'] != 'gap']
    shown = {(it['method'], it['block'] == 'gap', it['key'], it['label']) for it in items}
    out, replaced = [], 0
    for it in items:
        if it['method'] == 'v1' or it['block'] == 'gap':     # gaps keep their area-proportional draw
            out.append(it)
            continue
        cand, tries = it, 0
        while True:
            m = item_mask(cand, args.root, args.v2r, args.analysis, cache)
            clash = False
            for o in v1:
                if o['key'] != cand['key']:
                    continue
                om = item_mask(o, args.root, args.v2r, args.analysis, cache)
                inter = (m & om).sum()
                if inter * 2 > m.sum() or inter * 2 > om.sum():
                    clash = True
                    break
            if not clash:
                break
            pool = pools.get((cand['block'], cand['folder'], cand['stratum']), [])
            pool = [p for p in pool if ('v2.1', p['block'] == 'gap', p['key'], p['label']) not in shown]
            tries += 1
            if not pool or tries > max_tries:
                cand = None
                break
            new = dict(pool[rng.integers(len(pool))])
            shown.add(('v2.1', new['block'] == 'gap', new['key'], new['label']))
            cand, replaced = new, replaced + 1
        if cand is not None:
            out.append(cand)
    print(f'cross-method duplicates replaced: {replaced}; items dropped: {len(items) - len(out)}')
    return out


def build(args):
    rng = np.random.default_rng(args.seed)
    cells1 = pd.read_csv(os.path.join(args.analysis, 'cells_v1.csv'))
    cells2 = pd.read_csv(os.path.join(args.analysis, 'cells_v2.csv'))
    zones = round1_zone(args.root, args.round1, args.round1_v2, args.round1_v2_gaps)
    post2 = an.with_posterior(cells2)

    cand, gaps, excluded = [], [], set()
    for cond in args.conditions:
        v1c = an.index(f'{args.root}/Segmented/{cond}/Cell_merged_conservative')
        hd, hp = an.V1_HOLES[cond]
        v1h = an.index(f'{args.root}/Segmented/{cond}/{hd}', hp)
        v2c = an.index(f'{args.v2r}/{cond}', '*_v2_cells.tif')
        for k in sorted(set(v1c) & set(v2c)):
            um = ft.pixel_um(k)
            c1, c2 = tifffile.imread(v1c[k]), tifffile.imread(v2c[k])
            m1, m2 = bv.matched_labels(c1, c2)
            for tag, m, lab in (('v1', m1, c1), ('v2.1', m2, c2)):
                cand.append(pd.DataFrame({'key': k, 'folder': cond, 'method': tag, 'matched': sorted(m)}))
                excluded |= {('cell', tag, k, lb) for lb in excluded_labels(zones, k, lab)}
            h1 = tifffile.imread(v1h[k]) > 0 if k in v1h else np.zeros(c1.shape, bool)
            for tag, h in (('v1', h1), ('v2.1', bv.v2_gaps(args.analysis, cond, k))):
                lab = cc_label(h, connectivity=2)
                excluded |= {('gap', tag, k, lb) for lb in excluded_labels(zones, k, lab)}
                area = np.bincount(lab.ravel()) * um ** 2
                for comp in np.nonzero(area >= an.GAP_MIN_UM2)[0]:
                    if comp:
                        gaps.append(dict(key=k, folder=cond, method=tag, component=int(comp), area_um2=area[comp]))
    matched = pd.concat(cand, ignore_index=True)
    gaps = pd.DataFrame(gaps)
    folder_of = matched.drop_duplicates('key').set_index('key').folder

    def fresh(df, tag, block='cell', col='label'):
        return df[[(block, tag, k, int(lb)) not in excluded for k, lb in zip(df.key, df[col])]]

    items, shown = [], set()          # shown: (method, key, label), so no cell appears twice
    pools = {}                        # v2.1 candidates for cross-method replacement
    args.pools = pools

    def cell_item(block, tag, cond, r, stratum, **extra):
        return dict(block=block, method=tag, folder=cond, key=r.key, label=int(r.label), stratum=stratum,
                    area_um2=r.area_um2, question=bv.Q_CELL, view0='junctions', **extra)
    for tag, cells in (('v1', cells1), ('v2.1', cells2)):
        inner = fresh(cells[~cells.touches_border & cells.key.isin(folder_of.index)].copy(), tag)
        mset = set(map(tuple, matched[matched.method == tag][['key', 'matched']].to_numpy().tolist()))
        inner['stratum'] = ['matched' if (k, lab) in mset else 'unmatched' for k, lab in zip(inner.key, inner.label)]
        for cond, d in inner.groupby(inner.key.map(folder_of)):
            for stratum, s in d.groupby('stratum'):
                extra = dict(stratum_share=len(s) / len(d), stratum_n=len(s))
                if tag == 'v2.1':
                    pools[('cell', cond, stratum)] = [cell_item('cell', tag, cond, r, stratum, **extra)
                                                      for _, r in s.iterrows()]
                for _, r in bv.pick(s, N_CELL, rng).iterrows():
                    shown.add((tag, r.key, int(r.label)))
                    items.append(cell_item('cell', tag, cond, r, stratum, **extra))
            multi = d[d.n_nuclei >= 2]
            n_multi = len(multi)
            multi = multi[[(tag, k, int(lb)) not in shown for k, lb in zip(multi.key, multi.label)]]
            if tag == 'v2.1':
                pools[('multinucleated', cond, 'multi')] = [
                    cell_item('multinucleated', tag, cond, r, 'multi', nuclei=int(r.n_nuclei), stratum_n=n_multi)
                    for _, r in multi.iterrows()]
            for _, r in bv.pick(multi, N_MULTI, rng).iterrows():
                shown.add((tag, r.key, int(r.label)))
                items.append(cell_item('multinucleated', tag, cond, r, 'multi', nuclei=int(r.n_nuclei),
                                       stratum_n=n_multi))
    big = fresh(post2[(post2.posterior > 0.5) & post2.key.isin(folder_of.index)], 'v2.1')
    big = big[[('v2.1', k, int(lb)) not in shown for k, lb in zip(big.key, big.label)]]
    for cond, g in big.groupby(big.key.map(folder_of)):
        pools[('enlarged', cond, 'posterior>0.5')] = [cell_item('enlarged', 'v2.1', cond, r, 'posterior>0.5')
                                                     for _, r in g.iterrows()]
        for _, r in bv.pick(g, N_ENLARGED, rng).iterrows():
            shown.add(('v2.1', r.key, int(r.label)))
            items.append(cell_item('enlarged', 'v2.1', cond, r, 'posterior>0.5'))
    for (tag, cond), g in gaps.groupby(['method', 'folder']):
        g = fresh(g, tag, 'gap', 'component')
        if not len(g):
            continue
        folder_area = g.area_um2.sum()             # the estimand: gap area outside the round-1 zone
        draws = rng.choice(len(g), N_GAP, replace=True, p=g.area_um2.to_numpy() / g.area_um2.sum())
        for i, mult in zip(*np.unique(draws, return_counts=True)):
            r = g.iloc[i]
            items.append(dict(block='gap', method=tag, folder=cond, key=r.key, label=int(r.component),
                              stratum='component', sampling='pps', mult=int(mult), folder_area=folder_area,
                              area_um2=r.area_um2, question=bv.Q_GAP, view0='haze'))
    print(f"excluded near round-1 objects: {sum(1 for e in excluded if e[0] == 'cell')} cells, "
          f"{sum(1 for e in excluded if e[0] == 'gap')} gap components")

    items = cross_method_dedupe(items, args, rng)
    todo = pd.DataFrame(items)
    rendered = []
    order = ['junctions', 'haze', 'golgi']
    for (cond, k), g in todo.groupby(['folder', 'key']):
        views = view_images(args.root, cond, k)
        um = ft.pixel_um(k)
        c1 = tifffile.imread(an.index(f'{args.root}/Segmented/{cond}/Cell_merged_conservative')[k])
        c2 = tifffile.imread(an.index(f'{args.v2r}/{cond}', '*_v2_cells.tif')[k])
        hd, hp = an.V1_HOLES[cond]
        hp1 = an.index(f'{args.root}/Segmented/{cond}/{hd}', hp)
        gap_lab = {'v1': cc_label(tifffile.imread(hp1[k]) > 0, connectivity=2) if k in hp1 else None,
                   'v2.1': cc_label(bv.v2_gaps(args.analysis, cond, k), connectivity=2)}
        for _, r in g.iterrows():
            mask = gap_lab[r.method] == r.label if r.block == 'gap' else (c1 if r.method == 'v1' else c2) == r.label
            names = [n for n in order if n in views]        # fixed order: keys 1-3 always mean the same view
            it = r.to_dict()
            it['id'] = f"{r.block}:{r.method}:{k}:{r.label}"
            it['images'] = [(n, bv.render(views[n], mask, um, size=380)) for n in names if n in views]
            rendered.append(it)
    page = vs.write_session(rendered, args.out, title='Segmentation verdicts, round 2', seed=args.seed,
                            embed=True, note=NOTE, quality=80)
    print(f"{len(rendered)} items -> {page}")
    print(todo.groupby(['block', 'method']).size().to_string())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--root', required=True)
    ap.add_argument('--v2r', required=True, help='refine_v2.py output')
    ap.add_argument('--analysis', required=True, help='analyze.py output for v1 vs v2.1')
    ap.add_argument('--round1', required=True, help='key.csv of round 1')
    ap.add_argument('--round1-v2', required=True, help='segment_v2.py masks shown in round 1')
    ap.add_argument('--round1-v2-gaps', required=True, help='analysis folder whose v2_gaps/ were shown in round 1')
    ap.add_argument('--out', required=True)
    ap.add_argument('--conditions', nargs='+', default=an.CONDS)
    ap.add_argument('--seed', type=int, default=2027)
    build(ap.parse_args(argv))


if __name__ == '__main__':
    main()

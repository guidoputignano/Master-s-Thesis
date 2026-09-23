#!/usr/bin/env python3
"""Blind error audit of predicted cells: the fastest route to honest numbers.

Instead of annotating whole fields, an expert judges a random, stratified
sample of the cells the pipeline produced (a few seconds each), blind to the
predicted class. This estimates, with confidence intervals:

* how many predicted cells are correct single cells versus merges, splits,
  gap leakage or non-cells, per predicted class (e.g. each senescence rule);
* how many cells called 'Senescent' look senescent to the expert, and how many
  cells called 'Non-senescent' do;
* the senescent fraction corrected for those errors (stratified estimator).

It cannot measure missed cells (use nucleus clicks for recall) or boundary
accuracy (use the small full ground truth); it complements them.

  build:  python audit_gallery.py build --cells Segmented/Static-x20/Cell_merged_conservative \
              --membrane Projected/Static-x20/Cadherins/tophat --nuclei-img Projected/Static-x20/Nuclei/tophat \
              --classes Analysis/Static-x20/Senescence_Results/cell_classification_rule_based_full.csv \
              --per-stratum 40 --out audit/Static-x20
          then open audit/Static-x20/index.html in a browser, judge, press "Download CSV"
  score:  python audit_gallery.py score --audit audit/Static-x20 --verdicts audit_verdicts.csv
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seg_eval as se  # noqa: E402

SEG_CHOICES = ['correct', 'merged', 'split', 'leaks', 'not_a_cell', 'unsure']
PHENO_CHOICES = ['normal', 'senescent_like', 'multinucleated', 'unsure']


def normalise(img, lo=1, hi=99.8):
    img = np.asarray(img, float)
    a, b = np.percentile(img, [lo, hi])
    return np.clip((img - a) / (b - a), 0, 1) if b > a else np.zeros_like(img)


def composite(membrane, nuclei=None):
    """VE-cadherin in magenta, nuclei in cyan (same colours as the paper figure)."""
    m = normalise(membrane)
    n = normalise(nuclei) if nuclei is not None else np.zeros_like(m)
    return np.clip(np.stack([m, n, np.maximum(m, n)], -1), 0, 1)


def render_cell(rgb, labels, label, pad=30, size=300):
    """Side-by-side crop: raw composite | composite with the cell outline (yellow)
    and neighbouring territory boundaries (grey)."""
    from PIL import Image
    from skimage.segmentation import find_boundaries
    ys, xs = np.nonzero(labels == label)
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    half = max(ys.max() - ys.min(), xs.max() - xs.min()) // 2 + pad
    y0, y1 = max(0, cy - half), min(labels.shape[0], cy + half + 1)
    x0, x1 = max(0, cx - half), min(labels.shape[1], cx + half + 1)
    crop, lab = rgb[y0:y1, x0:x1], labels[y0:y1, x0:x1]
    over = crop.copy()
    over[find_boundaries(lab, mode='inner') & (lab != label)] = (0.55, 0.55, 0.55)
    from scipy import ndimage
    over[ndimage.binary_dilation(find_boundaries(lab == label, mode='inner'))] = (1.0, 0.85, 0.0)
    tiles = []
    for part in (crop, over):
        im = Image.fromarray((part * 255).astype(np.uint8))
        scale = size / max(im.size)
        tiles.append(im.resize((max(1, round(im.size[0] * scale)), max(1, round(im.size[1] * scale))),
                               Image.NEAREST if scale >= 1 else Image.BILINEAR))
    canvas = Image.new('RGB', (2 * size + 8, size), (24, 24, 24))
    canvas.paste(tiles[0], (0, 0))
    canvas.paste(tiles[1], (size + 8, 0))
    return canvas


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def build(args):
    rng = np.random.default_rng(args.seed)
    cells = {}
    for f in args.cells:
        cells.update(se.index_by_key(f, args.cells_glob))
    mem = {}
    for f in args.membrane:
        mem.update(se.index_by_key(f, args.membrane_glob))
    nuc = {}
    for f in args.nuclei_img or []:
        nuc.update(se.index_by_key(f, args.nuclei_glob))
    keys = sorted(set(cells) & set(mem))
    if not keys:
        raise SystemExit('no field has both a cell mask and a membrane image')

    # population of predicted cells (label, key, stratum)
    pop = []
    for k in keys:
        ids = np.unique(se.read_mask(cells[k]))
        pop.append(pd.DataFrame({'key': k, 'label': ids[ids > 0]}))
    pop = pd.concat(pop, ignore_index=True)
    pop['condition'] = pop.key.map(se.condition_of)
    if args.classes:
        import diagnostics as dg
        cls = dg.load_classes(args.classes).drop_duplicates(['key', 'label'])
        col = 'rule_based_classification_granular' if args.stratify == 'rule' else 'cell_type'
        pop = pop.merge(cls[['key', 'label', 'cell_type'] + ([col] if col != 'cell_type' else [])],
                        on=['key', 'label'], how='left')
        pop['stratum'] = pop[col].fillna('unclassified')
    else:
        pop['stratum'] = 'all'
        pop['cell_type'] = np.nan
    if args.by_condition:
        pop['stratum'] = pop.condition + ' | ' + pop.stratum
    pop['stratum_population'] = pop.groupby('stratum').label.transform('size')

    picks = []
    for s, d in pop.groupby('stratum'):
        picks.append(d.iloc[rng.permutation(len(d))[:args.per_stratum]])
    sample = pd.concat(picks).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    sample['audit_id'] = [f"C{i + 1:04d}" for i in range(len(sample))]

    os.makedirs(os.path.join(args.out, 'crops'), exist_ok=True)
    for k, d in sample.groupby('key'):
        labels = se.read_mask(cells[k])
        rgb = composite(se.read_mask(mem[k]), se.read_mask(nuc[k]) if k in nuc else None)
        for r in d.itertuples():
            render_cell(rgb, labels, r.label, args.pad, args.size).save(
                os.path.join(args.out, 'crops', f"{r.audit_id}.png"))
    sample.to_csv(os.path.join(args.out, 'audit_key.csv'), index=False)   # not shown to the auditor
    pop.groupby('stratum').size().rename('n').to_csv(os.path.join(args.out, 'strata_population.csv'))
    with open(os.path.join(args.out, 'index.html'), 'w') as f:
        f.write(HTML.replace('__IDS__', json.dumps(sample.audit_id.tolist()))
                    .replace('__SEG__', json.dumps(SEG_CHOICES)).replace('__PHENO__', json.dumps(PHENO_CHOICES))
                    .replace('__STORE__', json.dumps('audit:' + os.path.abspath(args.out))))
    print(sample.groupby('stratum').size().to_string())
    print(f"\n{len(sample)} crops in {args.out}/crops; open {args.out}/index.html")


def score(args):
    key = pd.read_csv(os.path.join(args.audit, 'audit_key.csv'))
    ver = pd.read_csv(args.verdicts)
    d = key.merge(ver, on='audit_id', how='inner')
    d = d[d.seg.notna()]
    if d.empty:
        raise SystemExit('no verdicts matched audit_key.csv')
    lines = [f"# Error audit: {len(d)} of {len(key)} sampled cells judged\n",
             '| stratum | cells in data | judged | correct single cell | merged | split | leaks | not a cell | '
             'looks senescent (incl. multinucleated) |', '|---|---|---|---|---|---|---|---|---|']
    est = []
    for s, g in d.groupby('stratum'):
        n = len(g)
        fmt = lambda k: "{:.0%} [{:.0%}, {:.0%}]".format(*wilson(k, n))  # noqa: E731
        sen = g.pheno.isin(['senescent_like', 'multinucleated'])
        ok = g.seg.eq('correct')
        lines.append(f"| {s} | {int(g.stratum_population.iloc[0])} | {n} | {fmt(ok.sum())} | "
                     f"{fmt(g.seg.eq('merged').sum())} | {fmt(g.seg.eq('split').sum())} | "
                     f"{fmt(g.seg.eq('leaks').sum())} | {fmt(g.seg.eq('not_a_cell').sum())} | {fmt(sen.sum())} |")
        est.append((s, int(g.stratum_population.iloc[0]), (ok & sen).mean(), ok.mean()))
    e = pd.DataFrame(est, columns=['stratum', 'N', 'p_true_senescent_cell', 'p_correct'])
    total = e.N.sum()
    lines += ['', "Stratified estimate of the senescent fraction among real, correctly segmented cells:",
              f"sum_s N_s * p(correct and senescent-looking) / sum_s N_s * p(correct) = "
              f"{(e.N * e.p_true_senescent_cell).sum() / max(1e-9, (e.N * e.p_correct).sum()):.1%} "
              f"(over {total} predicted cells)."]
    if key.cell_type.notna().any():
        rep = key.drop_duplicates('stratum')
        reported = (rep.stratum_population * rep.cell_type.eq('Senescent')).sum() / rep.stratum_population.sum()
        lines.append(f"Reported by the pipeline for the same strata: {reported:.1%}.")
    text = '\n'.join(lines) + '\n'
    with open(os.path.join(args.audit, 'audit_summary.md'), 'w') as f:
        f.write(text)
    print(text)


HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Cell audit</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#111;--fg:#eee;--mut:#999;--acc:#ffd23f;--card:#1c1c1c}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.4 system-ui,sans-serif}
main{max-width:960px;margin:0 auto;padding:16px}
img{width:100%;height:auto;background:#000;border-radius:6px}
.row{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}
button{background:var(--card);color:var(--fg);border:1px solid #333;border-radius:6px;padding:6px 10px;cursor:pointer}
button.on{border-color:var(--acc);color:var(--acc)}
.mut{color:var(--mut)} kbd{border:1px solid #444;border-radius:3px;padding:0 4px;font-size:12px}
input{width:100%;box-sizing:border-box;background:var(--card);color:var(--fg);border:1px solid #333;border-radius:6px;padding:6px}
</style></head><body><main>
<p class="mut">Left: raw (VE-cadherin magenta, nuclei cyan). Right: the outlined object (yellow) and neighbouring
territories (grey). Judge the <b>yellow</b> object only. Keys: segmentation <kbd>1</kbd>-<kbd>6</kbd>,
phenotype <kbd>q</kbd> <kbd>w</kbd> <kbd>e</kbd> <kbd>r</kbd>, <kbd>&larr;</kbd>/<kbd>&rarr;</kbd> to move. Progress is kept in this browser.</p>
<h3 id="title"></h3><img id="img" alt="cell crop">
<div><b>Segmentation</b><div class="row" id="seg"></div></div>
<div><b>Phenotype</b> <span class="mut">(ignore segmentation errors; judge the cell underneath)</span><div class="row" id="pheno"></div></div>
<input id="note" placeholder="optional note">
<div class="row"><button id="prev">&larr; previous</button><button id="next">next &rarr;</button>
<button id="dl">Download CSV</button><span id="done" class="mut"></span></div>
</main><script>
const IDS=__IDS__, SEG=__SEG__, PHENO=__PHENO__, KEY=__STORE__;
let st={}; try{st=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
let i=0; const $=id=>document.getElementById(id);
function save(){try{localStorage.setItem(KEY,JSON.stringify(st))}catch(e){}}
function buttons(el,opts,field,keys){el.innerHTML='';opts.forEach((o,k)=>{const b=document.createElement('button');
 b.textContent=(keys[k]||'')+' '+o.replace(/_/g,' ');b.onclick=()=>set(field,o);b.dataset.v=o;el.appendChild(b)})}
function set(field,v){const id=IDS[i];st[id]=st[id]||{};st[id][field]=v;save();show();
 if(st[id].seg&&st[id].pheno&&i<IDS.length-1){i++;show()}}
function show(){const id=IDS[i],r=st[id]||{};$('title').textContent=id+'  ('+(i+1)+' / '+IDS.length+')';
 $('img').src='crops/'+id+'.png';for(const [el,f] of [['seg','seg'],['pheno','pheno']])
 for(const b of $(el).children)b.classList.toggle('on',b.dataset.v===r[f]);$('note').value=r.note||'';
 $('done').textContent=Object.values(st).filter(x=>x.seg&&x.pheno).length+' judged'}
function csv(){const rows=[['audit_id','seg','pheno','note']];for(const id of IDS){const r=st[id]||{};
 rows.push([id,r.seg||'',r.pheno||'',(r.note||'').replace(/"/g,'""')])}
 const text=rows.map(r=>r.map(x=>'"'+x+'"').join(',')).join('\n');
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([text],{type:'text/csv'}));
 a.download='audit_verdicts.csv';a.click()}
buttons($('seg'),SEG,'seg',['1','2','3','4','5','6']);buttons($('pheno'),PHENO,'pheno',['q','w','e','r']);
$('prev').onclick=()=>{if(i>0){i--;show()}};$('next').onclick=()=>{if(i<IDS.length-1){i++;show()}};$('dl').onclick=csv;
$('note').oninput=e=>{const id=IDS[i];st[id]=st[id]||{};st[id].note=e.target.value;save()};
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;const s='123456'.indexOf(e.key),p='qwer'.indexOf(e.key);
 if(s>=0)set('seg',SEG[s]);else if(p>=0)set('pheno',PHENO[p]);else if(e.key==='ArrowRight')$('next').click();
 else if(e.key==='ArrowLeft')$('prev').click()});
let first=IDS.findIndex(id=>!(st[id]&&st[id].seg&&st[id].pheno));i=first<0?0:first;show();
</script></body></html>
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split('\n\n', 1)[1])
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build', help='sample cells and write crops + index.html')
    b.add_argument('--cells', nargs='+', required=True)
    b.add_argument('--cells-glob', default='*.tif')
    b.add_argument('--membrane', nargs='+', required=True, help='VE-cadherin projections')
    b.add_argument('--membrane-glob', default='*.tif')
    b.add_argument('--nuclei-img', nargs='+', help='nuclear-channel projections')
    b.add_argument('--nuclei-glob', default='*.tif')
    b.add_argument('--classes', nargs='+', help='cell_classification_rule_based_full.csv file(s)')
    b.add_argument('--stratify', choices=['rule', 'cell_type'], default='rule')
    b.add_argument('--by-condition', action='store_true', help='stratify by condition as well')
    b.add_argument('--per-stratum', type=int, default=40)
    b.add_argument('--pad', type=int, default=30)
    b.add_argument('--size', type=int, default=300)
    b.add_argument('--seed', type=int, default=0)
    b.add_argument('--out', required=True)
    s = sub.add_parser('score', help='summarise verdicts with confidence intervals')
    s.add_argument('--audit', required=True, help='folder written by build')
    s.add_argument('--verdicts', required=True, help='CSV downloaded from index.html')
    args = ap.parse_args(argv)
    build(args) if args.cmd == 'build' else score(args)


if __name__ == '__main__':
    main()

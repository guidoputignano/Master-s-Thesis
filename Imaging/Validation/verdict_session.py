#!/usr/bin/env python3
"""Blind yes/no verdict sessions: the expert answers one question per crop.

Build a session from a list of items (each: an id, a question, a PIL image and any
hidden metadata), open ``index.html`` in a browser, answer with Y / N / U (unsure),
then either download the CSV or copy the short answer code. The page shows only the
crop and the question. Method names, predicted classes and conditions stay in
``key.csv`` for scoring:

  python verdict_session.py score --session DIR --verdicts verdicts.csv --by block,method
  python verdict_session.py score --session DIR/index.html --code 'VS164:YYNU...' --by block,method

reports the yes-rate per group with Wilson 95% intervals (unsure answers excluded
and counted separately).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import zlib

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_gallery import wilson  # noqa: E402

LETTERS = {'yes': 'Y', 'no': 'N', 'unsure': 'U'}


def _data_uri(image, quality=88):
    buf = io.BytesIO()
    image.convert('RGB').save(buf, format='JPEG', quality=quality)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()


def write_session(items, out_dir, title='Verdict session', seed=0, embed=False, note='', quality=88):
    """``items``: dicts with 'id', 'question', 'image' (PIL) and hidden metadata. Shuffled.

    Instead of 'image', an item may carry 'images': a list of (view name, PIL image);
    the expert switches views with the number keys. With ``embed`` the crops are inlined
    and a compressed copy of ``key.csv`` is stored in the page (never displayed), so
    ``index.html`` alone can be answered and scored. ``note`` is shown above the crops.
    """
    os.makedirs(os.path.join(out_dir, 'crops'), exist_ok=True)
    order = np.random.default_rng(seed).permutation(len(items))
    items = [items[i] for i in order]
    meta, page_items = [], []
    for k, it in enumerate(items):
        vid = f"V{k + 1:04d}"
        views = it['images'] if 'images' in it else [('image', it['image'])]
        srcs = []
        for j, (_, im) in enumerate(views):
            if embed:
                srcs.append(_data_uri(im, quality))
            else:
                im.save(os.path.join(out_dir, 'crops', f"{vid}_{j + 1}.png"))
                srcs.append(f'crops/{vid}_{j + 1}.png')
        meta.append({'verdict_id': vid, **{a: b for a, b in it.items() if a not in ('image', 'images')}})
        names = [n for n, _ in views]
        page_items.append({'id': vid, 'q': it['question'], 'srcs': srcs, 'views': names,
                           'v0': names.index(it['view0']) if it.get('view0') in names else 0})
    key_csv = pd.DataFrame(meta).to_csv(index=False)
    with open(os.path.join(out_dir, 'key.csv'), 'w') as f:
        f.write(key_csv)
    packed = base64.b64encode(zlib.compress(key_csv.encode(), 9)).decode() if embed else ''
    page = (HTML.replace('__TITLE__', title).replace('__NOTE__', note).replace('__ITEMS__', json.dumps(page_items))
            .replace('__STORE__', json.dumps(f"verdicts:{title}:{len(meta)}:{zlib.crc32(key_csv.encode()):08x}"))
            .replace('__KEYZ__', json.dumps(packed)))
    with open(os.path.join(out_dir, 'index.html'), 'w') as f:
        f.write(page)
    return os.path.join(out_dir, 'index.html')


def decode(code, n=None):
    """'VS<n>:<letters>' -> DataFrame(id, answer); '-' means not answered."""
    m = re.fullmatch(r'\s*VS(\d+):([YNU-]+)\s*', code.replace('\n', '').replace(' ', ''))
    if not m or int(m.group(1)) != len(m.group(2)):
        raise ValueError('not a verdict code (expected VS<n>:<n letters Y/N/U/->)')
    if n is not None and int(m.group(1)) != n:
        raise ValueError(f'code has {m.group(1)} answers, the session has {n} items')
    inv = {v: k for k, v in LETTERS.items()}
    return pd.DataFrame([{'id': f"V{i + 1:04d}", 'answer': inv.get(c, '')} for i, c in enumerate(m.group(2))])


def read_key(session):
    """The hidden key: ``key.csv`` in a session folder, or the copy inside ``index.html``."""
    if os.path.isdir(session):
        return pd.read_csv(os.path.join(session, 'key.csv'))
    m = re.search(r'const KEYZ=("[^"]*");', open(session).read())
    if not m or not json.loads(m.group(1)):
        raise ValueError(f'{session}: no embedded key')
    return pd.read_csv(io.StringIO(zlib.decompress(base64.b64decode(json.loads(m.group(1)))).decode()))


def score(session, verdicts=None, by=('block',), code=None):
    key = read_key(session)
    ver = decode(code, len(key)) if code is not None else pd.read_csv(verdicts, dtype=str).fillna('')
    ver = ver.rename(columns={'id': 'verdict_id'})
    d = key.merge(ver, on='verdict_id', how='inner')
    d = d[d.answer.isin(['yes', 'no', 'unsure'])]
    rows = []
    for g, x in d.groupby(list(by)):
        yn = x[x.answer != 'unsure']
        p, lo, hi = wilson(int((yn.answer == 'yes').sum()), len(yn))
        rows.append(dict(zip(by, g if isinstance(g, tuple) else (g,)), answered=len(yn),
                         unsure=int((x.answer == 'unsure').sum()), yes_rate=p, ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(rows)


HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>__TITLE__</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#111;--fg:#eee;--mut:#9a9a9a;--acc:#ffd23f;--card:#1c1c1c;--yes:#3ecf8e;--no:#ff6b6b}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif}
main{max-width:980px;margin:0 auto;padding:16px}
img{width:100%;height:auto;background:#000;border-radius:6px}
h2{font-size:19px;margin:10px 0}
.row{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0;align-items:center}
button{background:var(--card);color:var(--fg);border:1px solid #333;border-radius:6px;padding:8px 14px;cursor:pointer;font-size:15px}
button.on{border-color:var(--acc);color:var(--acc)}
#y.on{border-color:var(--yes);color:var(--yes)} #n.on{border-color:var(--no);color:var(--no)}
.mut{color:var(--mut)} kbd{border:1px solid #444;border-radius:3px;padding:0 5px;font-size:12px}
input{flex:1;min-width:200px;background:var(--card);color:var(--fg);border:1px solid #333;border-radius:6px;padding:7px}
textarea{width:100%;box-sizing:border-box;background:var(--card);color:var(--fg);border:1px solid #333;border-radius:6px;padding:7px;font:13px monospace}
</style></head><body><main>
<p class="mut">Answer the question for the <b>outlined</b> object (right panel; the left panel is the same crop without outline;
magenta β-catenin, cyan nuclei, bar 10&nbsp;&micro;m). <kbd>Y</kbd> yes &middot; <kbd>N</kbd> no &middot; <kbd>U</kbd> unsure &middot;
<kbd>&larr;</kbd>/<kbd>&rarr;</kbd> move. Answers are saved in this browser. At the end press <b>Copy answer code</b> and paste it
into the chat, or <b>Download CSV</b>.</p><p>__NOTE__</p>
<div class="row"><b id="pos"></b><span id="done" class="mut"></span></div>
<h2 id="q"></h2><div class="row" id="views"></div><img id="img" alt="crop">
<div class="row"><button id="y">Y &middot; yes</button><button id="n">N &middot; no</button><button id="u">U &middot; unsure</button>
<input id="note" placeholder="optional note (CSV only)"></div>
<div class="row"><button id="prev">&larr; previous</button><button id="next">next &rarr;</button>
<button id="cp">Copy answer code</button><button id="dl">Download CSV</button></div>
<textarea id="code" rows="3" readonly placeholder="answer code appears here"></textarea>
</main><script>
const ITEMS=__ITEMS__, KEY=__STORE__;
const KEYZ=__KEYZ__; let st={}; try{st=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
let i=0, v=0; const $=id=>document.getElementById(id); const L={yes:'Y',no:'N',unsure:'U'};
function save(){try{localStorage.setItem(KEY,JSON.stringify(st))}catch(e){}}
function go(j){i=j;v=ITEMS[i].v0||0;show()}
function answer(a){if(document.activeElement)document.activeElement.blur();const id=ITEMS[i].id;st[id]=st[id]||{};st[id].answer=a;save();if(i<ITEMS.length-1)go(i+1);else show()}
function code(){return 'VS'+ITEMS.length+':'+ITEMS.map(it=>L[(st[it.id]||{}).answer]||'-').join('')}
function show(){const it=ITEMS[i],r=st[it.id]||{};$('pos').textContent=(i+1)+' / '+ITEMS.length;
 $('q').textContent=it.q;if(v>=it.srcs.length)v=0;$('img').src=it.srcs[v];
 $('views').innerHTML=it.views.length>1?it.views.map((n,j)=>'<button class="'+(j===v?'on':'')+'" onclick="v='+j+';show()"><kbd>'+(j+1)+'</kbd> '+n+'</button>').join(''):'';for(const [b,a] of [['y','yes'],['n','no'],['u','unsure']])
 $(b).classList.toggle('on',r.answer===a);$('note').value=r.note||'';
 $('done').textContent=' · '+Object.values(st).filter(x=>x.answer).length+' answered';$('code').value=code()}
function csv(){const rows=[['id','answer','note']];for(const it of ITEMS){const r=st[it.id]||{};
 rows.push([it.id,r.answer||'',(r.note||'').replace(/"/g,'""')])}
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([rows.map(r=>r.map(x=>'"'+x+'"').join(',')).join('\n')],{type:'text/csv'}));
 a.download='verdicts.csv';a.click()}
$('y').onclick=()=>answer('yes');$('n').onclick=()=>answer('no');$('u').onclick=()=>answer('unsure');
$('prev').onclick=()=>{if(i>0)go(i-1)};$('next').onclick=()=>{if(i<ITEMS.length-1)go(i+1)};$('dl').onclick=csv;
$('cp').onclick=()=>{const t=$('code');t.value=code();t.select();try{navigator.clipboard.writeText(t.value)}catch(e){try{document.execCommand('copy')}catch(e2){}}};
$('note').oninput=e=>{const id=ITEMS[i].id;st[id]=st[id]||{};st[id].note=e.target.value;save()};
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT'||e.target.tagName==='TEXTAREA')return;
 if(e.repeat||e.key===' '||e.key==='Enter'){e.preventDefault();return}const k=e.key.toLowerCase();
 if(k==='y')answer('yes');else if(k==='n')answer('no');else if(k==='u')answer('unsure');
 else if(/^[1-9]$/.test(e.key)&&+e.key<=ITEMS[i].srcs.length){v=+e.key-1;show()}
 else if(e.key==='ArrowRight')$('next').click();else if(e.key==='ArrowLeft')$('prev').click()});
let f=ITEMS.findIndex(it=>!(st[it.id]&&st[it.id].answer));go(f<0?0:f);
</script></body></html>
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('score', help='yes-rate per group with Wilson intervals')
    s.add_argument('--session', required=True, help='session folder, or its index.html')
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument('--verdicts', help='verdicts.csv downloaded from the page')
    g.add_argument('--code', help="answer code copied from the page ('VS<n>:...')")
    s.add_argument('--by', default='block', help='comma-separated key.csv columns to group by')
    args = ap.parse_args(argv)
    t = score(args.session, args.verdicts, args.by.split(','), args.code)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == '__main__':
    main()

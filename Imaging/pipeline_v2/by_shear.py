#!/usr/bin/env python3
"""Every imaging result of IMAGING_VALIDATION, per shear stress (static, 1.4 Pa).

The clear fields of each condition are pooled. Each field is measured in um at its own pixel
size (0.429 um, or 0.2145 um for the files whose metadata record the wrong objective), so
pooling needs no rescaling. ``--group folder`` keeps the four image folders apart instead;
it reproduces the per-folder tables of the fifth version of the report, as a check.

Reads the tables written by the other scripts (per cell: keep them private):

* ``--analysis``: analyze.py on the v2.1 masks (cells_v1.csv, cells_v2.csv = v2.1, fields.csv,
  agreement.csv); ``--analysis-v2``: analyze.py on the v2 masks (Cellpose alone);
* ``--features``: cell_features.py (features_v1.csv, features_v2.1.csv);
* ``--polarity``: polarity.py (polarity_cells.csv); ``--dna``: nd2_link.py (dna_index.csv);
* ``--root``: the data repository, for the reported rule-based calls (Senescence/).

Prints the tables and writes them to ``--out``.

  python by_shear.py --analysis v2r2_analysis_clean --analysis-v2 v2_analysis_clean \\
      --features features --polarity polarity/polarity_cells.csv --dna nd2/dna_index.csv \\
      --quality quality.csv --root data-mt --out by_shear
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'Validation'))
sys.path.insert(0, HERE)
import analyze as an  # noqa: E402
import nuclear_mixture as nm  # noqa: E402
import polarity as po  # noqa: E402
import senescence as sn  # noqa: E402

FOLDERS = ['0Pa_A1_20x', '0Pa_A1_40x', '1.4Pa_A1_20x', '1.4Pa_A1_40x']
SHEARS = ['Static', '1.4 Pa']
MIX_6H, MIX_16H, STEFOPOULOS_30, CONTROLS_6H = 29.9, 27.6, 31.0, 23.4    # expected misalignment (deg)
REPORT_FOLDERS = ['Static-x20', 'Static-x40', '1.4Pa-x20', '1.4Pa-x40']


class Grouping:
    def __init__(self, how):
        self.how = how
        self.names = FOLDERS if how == 'folder' else SHEARS

    def label(self, df):
        df = df.copy()
        df['grp'] = df.condition if self.how == 'folder' else np.where(
            df.condition.astype(str).str.startswith('0Pa'), 'Static', '1.4 Pa')
        if 'date' not in df:
            df['date'] = df.key.str.split('_').str[2]
        return df

    def static_of(self, g):
        return g.replace('1.4Pa', '0Pa') if self.how == 'folder' else 'Static'

    @staticmethod
    def is_static(g):
        return g.startswith('0Pa') or g == 'Static'


def table(rows):
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ segmentation (section 4)
def segmentation(G, cells, fields, agreement):
    rows = []
    for m in ('v1', 'v2', 'v2.1'):
        c, f = G.label(cells[m]), G.label(fields[m])
        col = 'v1_density' if m == 'v1' else 'v2_density'
        for g in G.names:
            d, fg = c[c.grp == g], f[f.grp == g]
            i = d[~d.touches_border]
            r = dict(grp=g, method=m, fields=d.key.nunique(), cells=len(d), density=fg[col].mean(),
                     area_median=i.area_um2.median(), AR_mean=i.aspect_ratio.mean(), misalign=i.misalign_deg.mean(),
                     multinucleated_pct=100 * (i.n_nuclei >= 2).mean(), no_nucleus_pct=100 * (i.n_nuclei == 0).mean(),
                     orphan_nuclei_pct=100 * fg[f'{"v1" if m == "v1" else "v2"}_orphan_nuclei'].mean(),
                     unassigned_pct=100 * fg.v2_uncovered_frac.mean() if m != 'v1' else np.nan)
            if m != 'v1':
                a = G.label(agreement[m])
                a = a[a.grp == g]
                r.update(matched_pct=100 * a.matched.sum() / a.v1.sum(), median_iou=a.median_iou.median(),
                         split_any_pct=100 * a.v1_split_by_v2.sum() / a.v1.sum(),
                         split_nucleated_pct=100 * a.v1_split_nucleated.sum() / a.v1.sum(),
                         merge_nucleated_pct=100 * a.v2_merging_nucleated.sum() / a.v2.sum())
            rows.append(r)
    return table(rows)


# ------------------------------------------------------------------ gaps (section 5)
def gaps(G, fields):
    f21, f2 = G.label(fields['v2.1']), G.label(fields['v2'])
    rows = []
    for g in G.names:
        a, b = f21[f21.grp == g], f2[f2.grp == g]
        area = a.v1_gap_frac * a.fov_mm2
        ok = a.v1_gap_nuclear_frac.notna()
        both = a[(a.v1_gap_frac > 0.001) & (a.v2_gap_frac > 0.001)]
        rows.append(dict(grp=g, fields=len(a), v1_holes=100 * a.v1_gap_frac.mean(), v2=100 * b.v2_gap_frac.mean(),
                         v21=100 * a.v2_gap_frac.mean(), v21_seeds=100 * a.v2_gap_frac_seeds.mean(),
                         v21_min25=100 * a.v2_gap_frac_min25.mean(), v21_min50=100 * a.v2_gap_frac_min50.mean(),
                         v21_p05=100 * a['v2_gap_frac_p0.5'].mean(), v21_p5=100 * a.v2_gap_frac_p5.mean(),
                         v1_stain_field=100 * a.v1_gap_nuclear_frac.mean(),
                         v1_stain_area=100 * (a.v1_gap_nuclear_frac * area)[ok].sum() / area[ok].sum(),
                         v1_intensity=a.v1_gap_rel_intensity.median(), iou=both.gap_iou.median(), iou_fields=len(both)))
    by_date = []
    for date in ('19dec21', '20dec21'):
        d = f21[f21.date == date]
        st, fl = d[d.condition.str.startswith('0Pa')], d[d.condition.str.startswith('1.4Pa')]
        r = dict(date=date, fields_static=len(st), fields_flow=len(fl))
        for col in ('v2_gap_frac', 'v2_gap_frac_seeds', 'v2_gap_frac_p0.5', 'v2_gap_frac_p5',
                    'v2_gap_frac_min25', 'v2_gap_frac_min50'):
            r[f'{col}_static'], r[f'{col}_flow'] = 100 * st[col].mean(), 100 * fl[col].mean()
            r[f'{col}_p'] = stats.mannwhitneyu(fl[col], st[col], alternative='greater').pvalue
        r['flow_median'], r['static_median'] = 100 * fl.v2_gap_frac.median(), 100 * st.v2_gap_frac.median()
        by_date.append(r)
    return table(rows), table(by_date)


# ------------------------------------------------------------------ cell area (section 6)
def _area_task(a):
    m, name, ratio, d, n_boot = a
    return sn.summarise(d, group='grp', fixed_log_ratio=ratio, n_boot=n_boot).assign(method=m, model=name)


def area_mixture(G, cells, n_boot, root, quality):
    mc = {m: G.label(an.mixture_cells(cells[m])) for m in ('v1', 'v2', 'v2.1')}
    models = (('free', None), ('ratio 2.27', np.log(sn.CHALA_RATIO)))
    tasks = [(m, name, ratio, mc[m][['key', 'grp', 'area_um2']], n_boot) for m in mc for name, ratio in models]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        fits = pd.concat(list(pool.map(_area_task, tasks)), ignore_index=True)
    by_date, compare, post = [], [], {}
    for m, d in mc.items():
        d = d.copy()
        d['p'] = np.nan
        for g in G.names:
            s = d[d.grp == g]
            x = np.log(s.area_um2.to_numpy())
            d.loc[s.index, 'p'] = sn.posterior(x, sn.fit(x)['params'])
            for date in ('19dec21', '20dec21'):
                f = sn.fit(np.log(s[s.date == date].area_um2.to_numpy()))
                by_date.append(dict(method=m, grp=g, date=date, frac=100 * f['frac_enlarged'],
                                    delta_bic=f['bic1'] - f['bic2']))
        post[m] = d
        for g in G.names:
            s = d[d.grp == g]
            e = s.p > 0.5
            if e.sum() < 3:
                continue
            compare.append(dict(method=m, grp=g, n_enlarged=int(e.sum()),
                                nucleus_enlarged=s[e].largest_nuc_um2.median(), nucleus_normal=s[~e].largest_nuc_um2.median(),
                                multi_enlarged=100 * (s[e].n_nuclei >= 2).mean(), multi_normal=100 * (s[~e].n_nuclei >= 2).mean(),
                                one_nucleus_enlarged=100 * (s[e].n_nuclei == 1).mean(),
                                AR_enlarged=s[e].aspect_ratio.mean(), AR_normal=s[~e].aspect_ratio.mean(),
                                AR_p=stats.mannwhitneyu(s[e].aspect_ratio, s[~e].aspect_ratio).pvalue,
                                misalign_enlarged=s[e].misalign_deg.mean(), misalign_normal=s[~e].misalign_deg.mean(),
                                area_normal=s[~e].area_um2.median(), area_enlarged=s[e].area_um2.median()))
    rules = kappa = None
    if root:
        from diagnostics import load_classes
        skip = an.excluded_fields(quality)
        parts, cls = [], []
        for folder in REPORT_FOLDERS:
            path = f'{root}/Senescence/{folder}/Senescence_Results/cell_classification_rule_based_full.csv'
            df = pd.read_csv(path)
            df['key'] = df.sample_id.astype(str)
            df = df[~df.key.isin(skip)].copy()
            df['condition'] = df.key.str.split('_').str[0] + '_A1_' + ('20x' if folder.endswith('x20') else '40x')
            parts.append(df)
            c = load_classes([path])
            cls.append(c.assign(reported=c.cell_type.eq('Senescent'))[['key', 'label', 'reported']])
        rep = G.label(pd.concat(parts, ignore_index=True))
        rows = []
        for g in G.names:
            r = rep[rep.grp == g]
            vc = r.rule_based_classification_granular.value_counts()
            rows.append(dict(grp=g, cells=len(r), **{k: 100 * vc.get(f'Rule_Sen_{k}', 0) / len(r) for k in
                                                    ('Poly', 'VeryLarge', 'LowCirc', 'LowNucRatio', 'HighScore')},
                             total=100 * (r.cell_type == 'Senescent').mean()))
        rules = table(rows)
        j = post['v1'].merge(pd.concat(cls, ignore_index=True), on=['key', 'label'], how='inner')
        kappa = table([dict(grp=g, cells=int((j.grp == g).sum()),
                            kappa=an.cohen_kappa(j[j.grp == g].reported.to_numpy(bool), (j[j.grp == g].p > 0.5).to_numpy()))
                       for g in G.names])
    return fits, table(by_date), table(compare), rules, kappa


# ------------------------------------------------------------------ nuclear size (section 6)
def nuclear_cells(features):
    d = features
    return d[~d.touches_border.astype(bool) & (d.cp_n_nuclei >= 1) & (d.area_um2 > 50) & (d.cp_nuc_largest_um2 >= 25)]


def _nuc_task(a):
    tag, x, keys, seed, n_boot = a
    f = nm.fit(x)
    rng = np.random.default_rng(seed)
    uf = np.unique(keys)
    idx = {k: np.flatnonzero(keys == k) for k in uf}
    draws = [nm.fit(x[np.concatenate([idx[k] for k in rng.choice(uf, len(uf), replace=True)])])['frac_enlarged']
             for _ in range(n_boot)]
    return tag, f, np.array(draws)


def antimode(params):
    mu1, mu2, cov, w2 = params
    grid = np.linspace(mu1[0], mu2[0], 2000)
    return float(np.exp(grid[np.argmin(nm.mixture_logpdf(grid, params))]))


def nuclear(G, features, n_boot):
    data = {seg: G.label(nuclear_cells(features[seg])) for seg in ('v1', 'v2.1')}
    tasks, seed = [], 0
    for seg, d in data.items():
        for g in G.names:
            for date in ('pooled', '19dec21', '20dec21'):
                s = d[d.grp == g] if date == 'pooled' else d[(d.grp == g) & (d.date == date)]
                seed += 1
                tasks.append(((seg, g, date), np.log(s.cp_nuc_largest_um2.to_numpy()), s.key.to_numpy(), seed,
                              n_boot if (seg == 'v1' and date == 'pooled') else 0))
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        res = {tag: (f, dr) for tag, f, dr in pool.map(_nuc_task, tasks)}
    rows = []
    for (seg, g, date), (f, dr) in res.items():
        lo, hi = np.percentile(dr, [2.5, 97.5]) if len(dr) else (np.nan, np.nan)
        rows.append(dict(segmentation=seg, grp=g, date=date, cells=f['n'], frac=100 * f['frac_enlarged'],
                         lo=100 * lo, hi=100 * hi, ratio=f['ratio'], delta_bic=f['bic1'] - f['bic2'],
                         normal_median=f['median_small']))
    fits = table(rows)
    # flow minus static, independent field bootstraps (v1)
    diffs = []
    for g in G.names:
        if G.is_static(g):
            continue
        s0 = G.static_of(g)
        fs, ds = res[('v1', s0, 'pooled')]
        ff, df = res[('v1', g, 'pooled')]
        k = min(len(ds), len(df))
        lo, hi = np.percentile(100 * (df[:k] - ds[:k]), [2.5, 97.5]) if k else (np.nan, np.nan)
        diffs.append(dict(grp=g, static=s0, diff=100 * (ff['frac_enlarged'] - fs['frac_enlarged']), lo=lo, hi=hi))
    # count above the static antimode, and the other hallmarks against that count (v1 cells)
    d = data['v1']
    hall = []
    for g in G.names:
        thr = antimode(res[('v1', G.static_of(g), 'pooled')][0]['params'])
        s = d[d.grp == g]
        big = s.cp_nuc_largest_um2 > thr
        multi_normal = (s.cp_n_nuclei >= 2) & ~big
        nn, ne = s[~big], s[big]
        r = dict(grp=g, antimode_um2=thr, above_pct=100 * big.mean(), multi_normal_pct=100 * multi_normal.mean(),
                 with_multi_pct=100 * (big | multi_normal).mean(),
                 golgi_area_ratio=ne.golgi_area_um2.median() / nn.golgi_area_um2.median(),
                 golgi_share_normal=(nn.golgi_area_um2 / nn.area_um2).median(),
                 golgi_fragments_enlarged=ne.golgi_fragments.median(), golgi_fragments_normal=nn.golgi_fragments.median(),
                 auc_nuclear_ar=roc_auc_score(big, s.nuc_ar_largest), auc_solidity=roc_auc_score(big, s.nuc_solidity_largest),
                 auc_dapi_projection=roc_auc_score(big, s.dapi_mean_largest_n),
                 normal_nucleus_median=nn.cp_nuc_largest_um2.median())
        if not G.is_static(g):
            s0 = d[d.grp == G.static_of(g)]
            k = r['normal_nucleus_median'] / s0[s0.cp_nuc_largest_um2 <= thr].cp_nuc_largest_um2.median()
            b2 = s.cp_nuc_largest_um2 > thr * k
            r.update(normal_nucleus_ratio=k, scaled_cutoff_pct=100 * b2.mean(),
                     scaled_with_multi_pct=100 * (b2 | ((s.cp_n_nuclei >= 2) & ~b2)).mean())
        hall.append(r)
    return fits, table(diffs), table(hall)


# ------------------------------------------------------------------ alignment (section 7)
def alignment(G, features):
    out = []
    for seg in ('v2.1', 'v1'):
        d = G.label(features[seg][~features[seg].touches_border.astype(bool)])
        ok = (d.cp_n_nuclei >= 1) & (d.area_um2 > 50) & (d.cp_nuc_largest_um2 >= 25)
        d['p_sen'] = np.nan
        for g in G.names:
            m = ok & (d.grp == g)
            x = np.log(d.loc[m, 'cp_nuc_largest_um2'].to_numpy())
            d.loc[m, 'p_sen'] = nm.posterior(x, nm.fit(x)['params'])
        d['normal'] = (d.p_sen < 0.5) & (d.cp_n_nuclei == 1)
        for g in G.names:
            s = d[d.grp == g]
            pf = s.groupby('key').misalign_deg.mean()
            ph = s[s.normal].groupby('key').misalign_deg.mean()
            r = dict(segmentation=seg, grp=g, cells=len(s), fields=len(pf), AR=s.aspect_ratio.mean(),
                     AR_sd=s.aspect_ratio.std(), misalign=s.misalign_deg.mean(), misalign_sd=s.misalign_deg.std(),
                     per_field=pf.mean(), per_field_sd=pf.std(), normal_per_field=ph.mean(), normal_per_field_sd=ph.std(),
                     order_all=np.cos(2 * np.radians(s.misalign_deg)).mean(),
                     order_normal=np.cos(2 * np.radians(s[s.normal].misalign_deg)).mean())
            if not G.is_static(g):
                r.update(p_vs_mix_6h=stats.ttest_1samp(pf, MIX_6H).pvalue, p_vs_mix_16h=stats.ttest_1samp(pf, MIX_16H).pvalue,
                         p_vs_stefopoulos=stats.ttest_1samp(pf, STEFOPOULOS_30).pvalue,
                         p_normal_vs_controls_6h=stats.ttest_1samp(ph.dropna(), CONTROLS_6H).pvalue)
            out.append(r)
        st = d[d.condition.str.startswith('0Pa')].groupby('key').misalign_deg.mean()
        fl = d[d.condition.str.startswith('1.4Pa')].groupby('key').misalign_deg.mean()
        nem = []
        for k, s in d.groupby('key'):
            z = s[s.aspect_ratio >= 1.3]
            if len(z) >= 5:
                v = np.exp(2j * np.deg2rad(z.axial_deg.to_numpy())).mean()
                nem.append(dict(flow=s.condition.iloc[0].startswith('1.4Pa'), S=abs(v), axis=np.angle(v) / 2))
        nem = table(nem)
        ax = nem[nem.flow].axis.to_numpy()
        R = abs(np.exp(2j * ax).mean())
        out.append(dict(segmentation=seg, grp='static vs flow', per_field_static=st.mean(), per_field_flow=fl.mean(),
                        p_static_vs_flow=stats.ttest_ind(fl, st, equal_var=False).pvalue,
                        nematic_static=nem[~nem.flow].S.mean(), nematic_flow=nem[nem.flow].S.mean(),
                        p_nematic=stats.mannwhitneyu(nem[nem.flow].S, nem[~nem.flow].S, alternative='greater').pvalue,
                        axes_rayleigh_p=float(np.exp(-len(ax) * R ** 2))))
    # density against order, per field (v1 cells; border cells count half)
    a = G.label(features['v1'])
    um = np.where(a.condition.str.endswith('20x'), 0.429, 0.2145)
    a['fov'] = (1024 * um * 1e-3) ** 2
    dens = a.assign(w=np.where(a.touches_border.astype(bool), 0.5, 1.0)).groupby(['grp', 'key']).agg(
        w=('w', 'sum'), fov=('fov', 'first'))
    dens['density'] = dens.w / dens.fov
    i = a[~a.touches_border.astype(bool)]
    S_ = i.assign(S=np.cos(2 * np.radians(i.misalign_deg))).groupby(['grp', 'key']).S.mean()
    j = dens.join(S_).reset_index()
    for g in G.names:
        s = j[j.grp == g]
        rho = stats.spearmanr(s.density, s.S)
        out.append(dict(segmentation='v1', grp=g, density_min=s.density.min(), density_max=s.density.max(),
                        rho_order_density=rho.correlation, p_order_density=rho.pvalue))
    return table(out)


# ------------------------------------------------------------------ polarity (section 7)
def polarity(G, cells, quality):
    c = cells[~cells.key.isin(an.excluded_fields(quality))].copy()
    c['pixel'] = c.condition.str[-3:]
    off = c[c.condition.str.startswith('0Pa')].groupby(['pixel', 'date'])[['dx_um', 'dy_um']].mean()
    c = c.join(off, on=['pixel', 'date'], rsuffix='_off')
    c['cx'], c['cy'] = c.dx_um - c.dx_um_off, c.dy_um - c.dy_um_off
    c = G.label(c)
    rows = []
    for (g, date), s in c.groupby(['grp', 'date']):
        n, r, ang, p = po.mean_direction(s.dx_um, s.dy_um)
        _, rc, angc, pc = po.mean_direction(s.cx, s.cy)
        rows.append(dict(grp=g, date=date, cells=n, R=r, direction=ang, p=p, golgi_left=(s.dx_um < 0).mean(),
                         R_corrected=rc, direction_corrected=angc, p_corrected=pc))
    fields = []
    for k, s in c[c.condition.str.startswith('1.4Pa')].groupby('key'):
        n, rc, angc, pc = po.mean_direction(s.cx, s.cy)
        side = 'none' if pc > 0.05 else ('left' if 135 < angc < 225 else 'right' if (angc < 45 or angc > 315)
                                         else 'up' if angc <= 135 else 'down')
        fields.append(dict(key=k, cells=n, R=rc, direction=angc, p=pc, side=side, left_on_average=s.cx.mean() < 0))
    return table(rows), table(fields)


# ------------------------------------------------------------------ DNA content (section 6)
def dna(dna_index):
    d = dna_index.copy()
    d['grp'] = np.where(d.shear == '0Pa', 'Static', '1.4 Pa')
    rows = []
    for g, s in d.groupby('grp'):
        e = s[s.enlarged]
        n = s[~s.enlarged]
        rows.append(dict(grp=g, nuclei=len(s), enlarged_pct=100 * s.enlarged.mean(),
                         enlarged_4N=100 * (e.cls == '4N').mean(), enlarged_2N=100 * (e.cls == '2N').mean(),
                         enlarged_S=100 * (e.cls == 'S').mean(), normal_4N=100 * (n.cls == '4N').mean(),
                         dna_enlarged=e.dna.median(), dna_normal=n.dna.median(),
                         **{f'enlarged_4N_{date}': 100 * (e[e.date == date].cls == '4N').mean() for date in ('19dec21', '20dec21')}))
    f = d.groupby(['key', 'grp', 'date']).agg(f4N=('cls', lambda x: (x == '4N').mean())).reset_index()
    tests = [dict(date='both', static=100 * f[f.grp == 'Static'].f4N.mean(), flow=100 * f[f.grp == '1.4 Pa'].f4N.mean(),
                  p=stats.mannwhitneyu(f[f.grp == 'Static'].f4N, f[f.grp == '1.4 Pa'].f4N).pvalue)]
    for date, s in f.groupby('date'):
        tests.append(dict(date=date, static=100 * s[s.grp == 'Static'].f4N.mean(), flow=100 * s[s.grp == '1.4 Pa'].f4N.mean(),
                          p=stats.mannwhitneyu(s[s.grp == 'Static'].f4N, s[s.grp == '1.4 Pa'].f4N).pvalue))
    return table(rows), table(tests)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--analysis', required=True, help='analyze.py output on the v2.1 masks (clear fields)')
    ap.add_argument('--analysis-v2', required=True, help='analyze.py output on the v2 masks (clear fields)')
    ap.add_argument('--features', required=True, help='cell_features.py output')
    ap.add_argument('--polarity', help='polarity_cells.csv from polarity.py')
    ap.add_argument('--dna', help='dna_index.csv from nd2_link.py')
    ap.add_argument('--quality', help='quality.py table (fields marked exclude are left out)')
    ap.add_argument('--root', help='data repository, for the reported rule-based calls')
    ap.add_argument('--group', choices=['shear', 'folder'], default='shear')
    ap.add_argument('--boot', type=int, default=300)
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    G = Grouping(a.group)
    cells = {'v1': pd.read_csv(f'{a.analysis}/cells_v1.csv'), 'v2': pd.read_csv(f'{a.analysis_v2}/cells_v2.csv'),
             'v2.1': pd.read_csv(f'{a.analysis}/cells_v2.csv')}
    fields = {'v1': pd.read_csv(f'{a.analysis}/fields.csv'), 'v2': pd.read_csv(f'{a.analysis_v2}/fields.csv'),
              'v2.1': pd.read_csv(f'{a.analysis}/fields.csv')}
    agreement = {'v2': pd.read_csv(f'{a.analysis_v2}/agreement.csv'), 'v2.1': pd.read_csv(f'{a.analysis}/agreement.csv')}
    features = {s: pd.read_csv(f'{a.features}/features_{s}.csv') for s in ('v1', 'v2.1')}
    out = {'segmentation': segmentation(G, cells, fields, agreement)}
    out['gaps'], out['gaps_by_experiment'] = gaps(G, fields)
    (out['area_mixture'], out['area_by_experiment'], out['enlarged_vs_normal'],
     out['reported_rules'], out['reported_kappa']) = area_mixture(G, cells, a.boot, a.root, a.quality)
    out['nuclear'], out['nuclear_flow_minus_static'], out['hallmarks'] = nuclear(G, features, a.boot)
    out['alignment'] = alignment(G, features)
    if a.polarity:
        out['polarity'], out['polarity_fields'] = polarity(G, pd.read_csv(a.polarity), a.quality)
    if a.dna and a.group == 'shear':
        out['dna'], out['dna_tests'] = dna(pd.read_csv(a.dna))
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 40)
    with open(f'{a.out}/summary.txt', 'w') as fh:
        for name, t in out.items():
            if t is None:
                continue
            t.to_csv(f'{a.out}/{name}.csv', index=False)
            txt = f'\n== {name}\n{t.round(4).to_string(index=False)}\n'
            fh.write(txt)
            print(txt)


if __name__ == '__main__':
    main()

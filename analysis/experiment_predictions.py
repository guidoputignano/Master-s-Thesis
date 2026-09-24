"""
Predictions for the proposed conditioning experiment, written down before any lab work.

Design: HUVEC monolayers with 0 % and 30 % senescent (TNF-alpha-treated) cells; flow
started as a step (fast start) or as a linear ramp; two shear levels, 1.4 Pa (where the
shape plateau was measured) and the top of the justified range (``--top``); slides fixed
at 1, 6 and 24 h, with static slides of each senescent share as controls.

For every condition the script predicts, from the reported model, the population mean
acute angle to the flow and the mean aspect ratio, as absolute values and as the change
from the static slide of the same senescent share (the change is what the imaging
pipeline measures without its own offset; in A1 it reads static aspect ratios about 0.1
higher than the hand-drawn outlines of the reference data). Two bands are given:

* ``model``: the reported model over its parameter ranges (PARAMS: the adaptation
  constant from the 6-8 h plateau in the same chamber, the plateau at 1.4 Pa, the
  activation shear that carries the targets above 1.4 Pa, the senescent cells);
* ``literature``: the same, with the adaptation constant widened to the slower kinetics
  other HUVEC studies report (PARAMS['T_lit_h']). Dense monolayers, as in A1 (no
  alignment after 6 h), can be slower still; the design should therefore control and
  report the density.

Gap area is not a state of the model. Its band (GAP) comes from the A1 slides (the static
gap fraction and the flow/static ratio at 6 h) and from the direction the literature gives
(a transient barrier weakening after a step onset that recovers within hours; more
permeable monolayers with senescent cells). These are bounds, not model predictions; the
experiment is what would calibrate them.

    python -m analysis.experiment_predictions predict [--top 4] [--ramp-min 30]
    python -m analysis.experiment_predictions compare --measured fields.csv [--top 4]

``compare`` reads one row per imaged field (columns: senescent_share, onset [step, ramp,
static], shear_pa, time_h, angle_deg, aspect_ratio, gap_pct; slide and field optional),
reports each condition's mean with a 95 % interval against the bands, and fits the
adaptation constant and the plateau from the measured time course.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'results', 'experiment_predictions')

REF_PA = 1.4                      # shear at which the plateau was measured
TIMES_H = (1.0, 6.0, 24.0)
SHARES = (0.0, 0.30)
ONSETS = ('step', 'ramp')

# (low, nominal, high). Sources in docs/shear_range_and_limits.md and research/2026-09-24_limits_checked.md.
PARAMS = {
    'theta_stat_deg': (45.0, 45.0, 45.0),     # random orientation
    'rho_stat': (1.5, 1.9, 2.0),              # static: 1.9 (Chala 2021), 1.5 (low passage, Exarchos 2022), 2.0 in A1 images
    'theta_ref_deg': (11.0, 20.0, 22.0),      # plateau at 1.4 Pa: 20 deg at 16 h (Chala 2021), ~21 deg at 8 h
                                              # (Stefopoulos 2022), 11 deg at 16 h (low passage, Exarchos 2022); same chamber
    'rho_ref': (2.2, 2.3, 3.3),               # plateau at 1.4 Pa: 2.3 (Chala 2021), 3.3 (Exarchos 2022)
    'tau_act_pa': (0.3, 0.5, 0.7),            # activation shear of the gate that carries the targets above 1.4 Pa
    'T_model_h': (2.0, 3.0, 4.0),             # plateau after 6-8 h at 1.4 Pa, same chamber (Stefopoulos 2022)
    'T_lit_h': (2.0, 3.0, 12.0),              # other HUVEC studies: 45 to 15 deg by about 24 h (DeStefano 2017),
                                              # alignment complete after 36 h (Giantsos-Adams 2013)
    'theta_sen_deg': (44.0, 45.0, 47.0),      # senescent cells stay random: 44 deg (17 PD, Exarchos 2022), 47 deg (Stefopoulos 2022)
    'rho_sen': (1.76, 2.0, 2.1),              # 1.76 (17 PD under flow, Exarchos 2022), 2.0 (TNF-alpha, Chala 2021)
    'theta_floor_deg': (11.0, 11.0, 11.0),    # no plateau below the lowest measured in this chamber (11 deg)
}

# Alternative hypotheses for the response above 1.4 Pa, which the experiment tells apart.
HYPOTHESES = {
    'model': 'reported model: parallel alignment improving up to 5 Pa (Robotti 2014, same chamber)',
    'edge_2Pa': 'HUVEC aligned only up to about 2 Pa, perpendicular above (Baeyens 2015; crossover 2.7 Pa)',
    'long_confluent': 'long-confluent monolayers, as measured after 16 h in the same chamber (Robotti 2014)',
}
ROBOTTI_LONG_CONFLUENT = ([0.0, 1.4, 4.0, 5.0, 6.0, 8.0], [45.0, 38.0, 36.5, 21.5, 43.5, 39.5])
THETA_PERP_DEG = 70.0
FIELD_SD_DEG = 7.1   # spread of the mean angle between fields, A1 flow fields (IMAGING_VALIDATION, section 7)
A1_FLOW_6H = dict(share=0.30, shear_pa=1.4, time_h=6.0, angle=41.6, sd=7.1, fields=33)   # A1, dense monolayers

# Gap area fraction (% of the field): bounds (low, high), not model predictions.
GAP = {
    'static_30': (0.5, 1.5),         # A1 static slides, 30 % senescent: 0.67 and 1.19 % (two experiments)
    'young_ratio': (1.0, 3.5),       # G(30 %)/G(0 %): senescent monolayers 2.5x, mixtures 3.5x more permeable (Krouwer 2012)
    'flow_ratio': {                  # G(flow)/G(static) of the same senescent share
        (1.0, 'step'): (1.0, 2.0),   # transient weakening after a step onset (permeability up to about 2x)
        (1.0, 'ramp'): (1.0, 2.0),   # a ramp is expected to remove it (signalling data), not yet shown for barrier
        (6.0, 'step'): (1.0, 3.8),   # A1 at 1.4 Pa, 6 h: 1.0 on one experiment, 3.8 on the other
        (6.0, 'ramp'): (1.0, 3.8),
        (24.0, 0.0): (0.5, 1.0),     # young monolayers tighten over a day
        (24.0, 0.3): (0.5, 3.8),     # mixtures: tightening, or the excess of the gap-rich A1 slide
    },
}


def _gate(tau, tau_act):
    tau = np.asarray(tau, float)
    return np.where(tau > tau_act, 1.0 - np.exp(-(tau - tau_act) / tau_act), 0.0)


def _smoothstep(tau, band_top, crossover):
    if tau <= band_top:
        return 0.0
    x = min(1.0, (tau - band_top) / (2.0 * crossover - 2.0 * band_top))
    return x * x * (3.0 - 2.0 * x)


def targets(tau, p, hypothesis='model'):
    """Healthy-cell targets (angle deg, aspect ratio) at shear ``tau`` for parameters ``p``
    (dict of arrays): the gated static -> plateau interpolation of the model, kept above the
    floor where the gate extrapolates beyond 1.4 Pa; under ``edge_2Pa`` the angle turns
    perpendicular above 2 Pa, and under ``long_confluent`` it follows the long-confluent
    monolayers of the same chamber (aspect ratio as in the model, not reported there)."""
    r = _gate(tau, p['tau_act_pa']) / _gate(REF_PA, p['tau_act_pa'])
    th = p['theta_stat_deg'] + (p['theta_ref_deg'] - p['theta_stat_deg']) * r
    th = np.where(np.asarray(tau) > REF_PA, np.maximum(th, p['theta_floor_deg']), th)
    rh = p['rho_stat'] + (p['rho_ref'] - p['rho_stat']) * r
    if hypothesis == 'edge_2Pa':
        w = _smoothstep(float(tau), 2.0, 2.7)
        th = th + (THETA_PERP_DEG - th) * w
    elif hypothesis == 'long_confluent':
        x, y = ROBOTTI_LONG_CONFLUENT
        lc = float(np.interp(tau, x, y))
        th = np.full_like(np.asarray(th, float), lc) if float(tau) > 0 else th
    return th, rh


def shear_protocol(t_h, top, onset, ramp_h):
    t_h = np.asarray(t_h, float)
    if onset == 'static':
        return np.zeros_like(t_h)
    if onset == 'step':
        return np.full_like(t_h, top)
    return top * np.clip(t_h / ramp_h, 0.0, 1.0)


def healthy_course(times_h, top, onset, ramp_h, p, T, dt_h=1.0 / 60, hypothesis='model'):
    """First-order relaxation of the healthy-cell angle and aspect ratio toward the
    targets of the applied shear, from the static state; returns arrays (len(times), n)."""
    n = np.size(T)
    th = np.broadcast_to(p['theta_stat_deg'], (n,)).astype(float).copy()
    rh = np.broadcast_to(p['rho_stat'], (n,)).astype(float).copy()
    out_t, out_r, t = [], [], 0.0
    for t_end in times_h:
        while t < t_end - 1e-9:
            h = min(dt_h, t_end - t)
            tau = float(shear_protocol(t + 0.5 * h, top, onset, ramp_h))
            ts, rs = targets(tau, p, hypothesis)
            a = 1.0 - np.exp(-h / T)
            th += (ts - th) * a
            rh += (rs - rh) * a
            t += h
        out_t.append(th.copy())
        out_r.append(rh.copy())
    return np.array(out_t), np.array(out_r)


def sample(n, seed=0, T_key='T_model_h', nominal=False):
    """Parameter draws: uniform over each (low, high) range, log-uniform for T."""
    rng = np.random.default_rng(seed)
    p = {}
    for k, (lo, mid, hi) in PARAMS.items():
        if nominal or lo == hi:
            p[k] = np.full(n, mid)
        else:
            p[k] = rng.uniform(lo, hi, n)
    lo, mid, hi = PARAMS[T_key]
    T = np.full(n, mid) if nominal else np.exp(rng.uniform(np.log(lo), np.log(hi), n))
    return p, T


def population(th_h, rh_h, share, p):
    return ((1 - share) * th_h + share * p['theta_sen_deg'],
            (1 - share) * rh_h + share * p['rho_sen'])


def predict(top, ramp_min=30.0, n=4000, seed=0, hypotheses=('model',)):
    """Table of predictions, one row per (hypothesis, senescent share, onset, shear, time)."""
    ramp_h = ramp_min / 60.0
    rows = []
    conds = [(s, 'static', 0.0) for s in SHARES]
    conds += [(s, o, tau) for s in SHARES for o in ONSETS for tau in sorted({REF_PA, float(top)})]
    for hyp in hypotheses:
        for share, onset, tau in conds:
            res = {}
            for band, key in (('nominal', 'T_model_h'), ('model', 'T_model_h'), ('literature', 'T_lit_h')):
                p, T = sample(1 if band == 'nominal' else n, seed, key, nominal=band == 'nominal')
                th, rh = healthy_course(TIMES_H, tau, onset, ramp_h, p, T, hypothesis=hyp)
                st_th, st_rh = healthy_course(TIMES_H, 0.0, 'static', ramp_h, p, T, hypothesis=hyp)
                pa, pr = population(th, rh, share, p)
                sa, sr = population(st_th, st_rh, share, p)
                res[band] = dict(angle=pa, ar=pr, d_angle=pa - sa, d_ar=pr - sr)
            for i, t in enumerate(TIMES_H):
                row = dict(hypothesis=hyp, senescent_share=share, onset=onset, shear_pa=tau, time_h=t)
                for q in ('angle', 'ar', 'd_angle', 'd_ar'):
                    row[q] = float(res['nominal'][q][i][0])
                    for band in ('model', 'literature'):
                        lo, hi = np.percentile(res[band][q][i], [2.5, 97.5])
                        row[f'{q}_{band}_lo'], row[f'{q}_{band}_hi'] = float(lo), float(hi)
                row['gap_lo'], row['gap_hi'] = gap_bounds(share, onset, t)
                rows.append(row)
    return pd.DataFrame(rows)


def gap_bounds(share, onset, t):
    s_lo, s_hi = GAP['static_30']
    if share < 0.15:
        r_lo, r_hi = GAP['young_ratio']
        s_lo, s_hi = s_lo / r_hi, s_hi / r_lo
    if onset == 'static':
        return s_lo, s_hi
    key = (t, onset) if t < 12 else (t, 0.0 if share < 0.15 else 0.3)
    f_lo, f_hi = GAP['flow_ratio'][key]
    return s_lo * f_lo, s_hi * f_hi


def contrast(top, ramp_min=30.0, n=4000, seed=0, hypotheses=tuple(HYPOTHESES)):
    """Change from 1.4 Pa to ``top`` at each time, per senescent share and onset. Both shears
    use the same parameter draws, so the band is that of the paired difference, which is
    narrower than the two bands side by side; this contrast is what tells the hypotheses apart."""
    ramp_h = ramp_min / 60.0
    rows = []
    for hyp in hypotheses:
        for share in SHARES:
            for onset in ONSETS:
                res = {}
                for band, key in (('nominal', 'T_model_h'), ('model', 'T_model_h'), ('literature', 'T_lit_h')):
                    p, T = sample(1 if band == 'nominal' else n, seed, key, nominal=band == 'nominal')
                    a_hi, r_hi = population(*healthy_course(TIMES_H, top, onset, ramp_h, p, T, hypothesis=hyp), share, p)
                    a_lo, r_lo = population(*healthy_course(TIMES_H, REF_PA, onset, ramp_h, p, T, hypothesis=hyp), share, p)
                    res[band] = dict(d_angle=a_hi - a_lo, d_ar=r_hi - r_lo)
                for i, t in enumerate(TIMES_H):
                    row = dict(hypothesis=hyp, senescent_share=share, onset=onset, shear_pa=float(top), time_h=t)
                    for q in ('d_angle', 'd_ar'):
                        row[q] = float(res['nominal'][q][i][0])
                        for band in ('model', 'literature'):
                            lo, hi = np.percentile(res[band][q][i], [2.5, 97.5])
                            row[f'{q}_{band}_lo'], row[f'{q}_{band}_hi'] = float(lo), float(hi)
                    row['fields_per_arm'] = fields_needed(row['d_angle'])
                    rows.append(row)
    return pd.DataFrame(rows)


def fields_needed(delta_deg, sd_deg=FIELD_SD_DEG, alpha=0.05, power=0.80):
    """Imaged fields per arm to detect a difference ``delta_deg`` of the mean angle (two-sided,
    two-sample, normal approximation) with the A1 spread between fields. Fields of one slide are
    not independent, so this is a lower bound; slides or chambers are the experimental unit."""
    from scipy import stats
    if abs(delta_deg) < 1e-6:
        return float('inf')
    z = stats.norm.ppf(1.0 - alpha / 2.0) + stats.norm.ppf(power)
    return int(np.ceil(2.0 * (z * sd_deg / delta_deg) ** 2))


# ---------------------------------------------------------------- comparison
def _ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (float(x.mean()) if len(x) else np.nan), np.nan, np.nan
    from scipy import stats
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    h = stats.t.ppf(0.975, len(x) - 1) * se
    return float(m), float(m - h), float(m + h)


def compare(measured, top, ramp_min=30.0):
    """Measured condition means (95 % interval over fields) against the predicted bands of the
    reported model, and the hypothesis whose nominal prediction lies closest."""
    pred_all = predict(top, ramp_min, hypotheses=tuple(HYPOTHESES))
    pred = pred_all[pred_all.hypothesis == 'model']
    m = measured.copy()
    m['onset'] = m['onset'].str.lower()
    m.loc[m['shear_pa'] <= 0, 'onset'] = 'static'
    keys = ['senescent_share', 'onset', 'shear_pa', 'time_h']
    static = (m[m.onset == 'static'].groupby(['senescent_share', 'time_h'])
              [['angle_deg', 'aspect_ratio']].mean())
    rows = []
    for k, g in m.groupby(keys):
        share, onset, tau, t = k
        sel = lambda d: d[np.isclose(d.senescent_share, share) & (d.onset == onset)
                          & np.isclose(d.shear_pa, tau) & np.isclose(d.time_h, t)]
        pr = sel(pred)
        if pr.empty:
            continue
        pr = pr.iloc[0]
        row = dict(zip(keys, k), fields=len(g))
        for col, q in (('angle_deg', 'angle'), ('aspect_ratio', 'ar')):
            if col not in g:
                continue
            mean, lo, hi = _ci(g[col])
            row[f'{q}'], row[f'{q}_lo'], row[f'{q}_hi'] = mean, lo, hi
            row[f'{q}_pred'] = pr[q]
            row[f'{q}_verdict'] = _verdict(lo, hi, pr[f'{q}_model_lo'], pr[f'{q}_model_hi'],
                                           pr[f'{q}_literature_lo'], pr[f'{q}_literature_hi'])
            if onset != 'static' and (share, t) in static.index:
                d = g[col] - static.loc[(share, t), col]
                dm, dlo, dhi = _ci(d)
                row[f'd_{q}'], row[f'd_{q}_verdict'] = dm, _verdict(
                    dlo, dhi, pr[f'd_{q}_model_lo'], pr[f'd_{q}_model_hi'],
                    pr[f'd_{q}_literature_lo'], pr[f'd_{q}_literature_hi'])
        if 'angle_deg' in g and onset != 'static':
            alt = sel(pred_all).set_index('hypothesis').angle
            row['closest_hypothesis'] = str((alt - row['angle']).abs().idxmin())
        if 'gap_pct' in g:
            mean, lo, hi = _ci(g['gap_pct'])
            row.update(gap=mean, gap_lo_ci=lo, gap_hi_ci=hi, gap_bounds=f"{pr.gap_lo:.2f}-{pr.gap_hi:.2f}",
                       gap_verdict=_verdict(lo, hi, pr.gap_lo, pr.gap_hi, pr.gap_lo, pr.gap_hi))
        rows.append(row)
    return pd.DataFrame(rows), fit_kinetics(m)


def _verdict(lo, hi, m_lo, m_hi, l_lo, l_hi):
    if not np.isfinite(lo):
        return 'too few fields'
    if hi >= m_lo and lo <= m_hi:
        return 'inside model band'
    if hi >= l_lo and lo <= l_hi:
        return 'inside literature band only'
    return 'above' if lo > l_hi else 'below'


def fit_kinetics(m):
    """Adaptation constant and plateau angle per (share, onset, shear) from the time course of
    the field means: theta(t) = theta_inf + (theta_0 - theta_inf) exp(-t/T), theta_0 taken
    from the static slides. Needs at least three time points."""
    from scipy.optimize import least_squares
    out = []
    st = m[m.onset == 'static'].groupby('senescent_share').angle_deg.mean()
    for (share, onset, tau), g in m[m.onset != 'static'].groupby(['senescent_share', 'onset', 'shear_pa']):
        tm = g.groupby('time_h').angle_deg.mean()
        if len(tm) < 3 or share not in st.index:
            continue
        th0 = float(st.loc[share])
        t, y = tm.index.to_numpy(float), tm.to_numpy(float)

        def res(x):
            T, inf = np.exp(x[0]), x[1]
            return inf + (th0 - inf) * np.exp(-t / T) - y
        fit = least_squares(res, x0=[np.log(3.0), float(np.clip(y[-1], 0.5, 89.5))],
                            bounds=([np.log(0.1), 0.0], [np.log(200.0), 90.0]))
        out.append(dict(senescent_share=share, onset=onset, shear_pa=tau, T_h=float(np.exp(fit.x[0])),
                        plateau_deg=float(fit.x[1]), static_deg=th0, points=len(tm)))
    return pd.DataFrame(out)


# ---------------------------------------------------------------- outputs
def figure(pred, path_noext, top):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    blue, orange = '#2a78d6', '#eb6834'
    ink, muted = '#0b0b0b', '#52514e'
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                         'axes.edgecolor': muted, 'axes.linewidth': 0.6, 'xtick.color': muted,
                         'ytick.color': muted, 'axes.labelcolor': ink, 'pdf.fonttype': 42, 'savefig.dpi': 300})
    alt = pred[pred.hypothesis != 'model'] if 'hypothesis' in pred else pred.iloc[0:0]
    pred = pred[pred.hypothesis == 'model'] if 'hypothesis' in pred else pred
    taus = sorted(pred[pred.onset != 'static'].shear_pa.unique())
    fig, axes = plt.subplots(2, len(taus), figsize=(17.5 / 2.54, 11 / 2.54), sharex=True, sharey='row',
                             squeeze=False)
    t = np.array(TIMES_H)
    for j, tau in enumerate(taus):
        for i, (q, lab) in enumerate((('angle', 'mean angle to the flow (deg)'), ('ar', 'mean aspect ratio'))):
            ax = axes[i, j]
            for share, c in ((0.0, blue), (0.30, orange)):
                d = pred[np.isclose(pred.senescent_share, share) & (pred.onset == 'step') & np.isclose(pred.shear_pa, tau)]
                ax.fill_between(t, d[f'{q}_literature_lo'], d[f'{q}_literature_hi'], color=c, alpha=0.12, lw=0)
                ax.fill_between(t, d[f'{q}_model_lo'], d[f'{q}_model_hi'], color=c, alpha=0.30, lw=0)
                ax.plot(t, d[q], color=c, lw=1.5, marker='o', ms=3)
                dy = 0.0 if q == 'angle' else (0.07 if share == 0.0 else -0.07)
                ax.text(24.8, d[q].iloc[-1] + dy, f'{int(share * 100)} %', color=ink, va='center', fontsize=7)
            if q == 'angle' and np.isclose(tau, A1_FLOW_6H['shear_pa']):
                ax.errorbar([A1_FLOW_6H['time_h']], [A1_FLOW_6H['angle']], yerr=[A1_FLOW_6H['sd']], color=ink,
                            marker='D', ms=3.5, lw=0.8, capsize=2, zorder=5)
                ax.text(A1_FLOW_6H['time_h'] * 1.12, A1_FLOW_6H['angle'] + 4.5, 'A1, 30 %', color=ink, fontsize=6.5)
            if q == 'angle' and len(alt):
                # the alternative hypotheses, young monolayers only; they differ from the model in angle
                for hyp, ls, name in (('long_confluent', '--', 'long-confluent'), ('edge_2Pa', ':', 'band edge 2 Pa')):
                    d = alt[(alt.hypothesis == hyp) & np.isclose(alt.senescent_share, 0.0) & (alt.onset == 'step')
                            & np.isclose(alt.shear_pa, tau)]
                    m = pred[np.isclose(pred.senescent_share, 0.0) & (pred.onset == 'step') & np.isclose(pred.shear_pa, tau)]
                    if d.empty or np.allclose(d.angle.to_numpy(), m.angle.to_numpy(), atol=0.5):
                        continue
                    ax.plot(t, d.angle, color=muted, lw=1.0, ls=ls)
                    ax.text(24.8, d.angle.iloc[-1], f'{name}, 0 %', color=muted, va='center', fontsize=6.5)
            ax.set_xscale('log')
            ax.set_xticks(t)
            ax.set_xticklabels([f'{x:g}' for x in t])
            ax.set_xlim(0.8, 60)
            ax.spines[['top', 'right']].set_visible(False)
            if j == 0:
                ax.set_ylabel(lab)
            if i == 0:
                ax.set_title(f'{tau:g} Pa, step onset', fontsize=8, loc='left')
            if i == 1:
                ax.set_xlabel('time after flow onset (h)')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{path_noext}.{ext}', bbox_inches='tight')
    plt.close(fig)


def markdown(pred, top, ramp_min, con=None):
    base = pred[pred.hypothesis == 'model']
    alt = {h: pred[pred.hypothesis == h] for h in HYPOTHESES if h != 'model'}
    lines = [f'# Predictions for the proposed experiment (top shear {top:g} Pa, ramp {ramp_min:g} min)', '',
             'Reported model: nominal value [model band; literature band], 95 % of parameter draws. '
             'The last two columns give the nominal angle under the alternative hypotheses: '
             + '; '.join(f'**{h}**, {HYPOTHESES[h]}' for h in alt) + '. '
             'Gap area: bounds from A1 and the literature, not model predictions.', '',
             '| Senescent | Onset | Shear (Pa) | Time (h) | Angle (deg) | Aspect ratio | Change of angle vs static (deg) '
             '| Gap area (%) | ' + ' | '.join(f'Angle, {h}' for h in alt) + ' |',
             '|---|---|---|---|---|---|---|---|' + '---|' * len(alt)]
    for _, r in base.iterrows():
        def band(q, fmt):
            return (f"{r[q]:{fmt}} [{r[f'{q}_model_lo']:{fmt}}–{r[f'{q}_model_hi']:{fmt}}; "
                    f"{r[f'{q}_literature_lo']:{fmt}}–{r[f'{q}_literature_hi']:{fmt}}]")
        others = []
        for h, d in alt.items():
            v = d[np.isclose(d.senescent_share, r.senescent_share) & (d.onset == r.onset)
                  & np.isclose(d.shear_pa, r.shear_pa) & np.isclose(d.time_h, r.time_h)].angle
            others.append(f'{v.iloc[0]:.1f}' if len(v) else '—')
        lines.append(f"| {int(r.senescent_share * 100)} % | {r.onset} | {r.shear_pa:g} | {r.time_h:g} | "
                     f"{band('angle', '.1f')} | {band('ar', '.2f')} | "
                     f"{'—' if r.onset == 'static' else band('d_angle', '.1f')} | {r.gap_lo:.2f}–{r.gap_hi:.2f} | "
                     + ' | '.join(others) + ' |')
    if con is not None:
        lines += ['', f'## Change from 1.4 Pa to {top:g} Pa (same cells, same time)', '',
                  'Reported model: nominal value [model band; literature band] of the paired difference. '
                  f'Fields per arm: imaged fields needed to detect the nominal change of angle (80 % power, '
                  f'two-sided 5 %, spread between fields {FIELD_SD_DEG:g} deg as in A1); a lower bound, since '
                  'fields of one slide are not independent.', '',
                  '| Senescent | Onset | Time (h) | Change of angle (deg) | Change of aspect ratio | Fields per arm | '
                  + ' | '.join(f'Change of angle, {h}' for h in alt) + ' |',
                  '|---|---|---|---|---|---|' + '---|' * len(alt)]
        cb = con[con.hypothesis == 'model']
        for _, r in cb.iterrows():
            def band(q, fmt):
                return (f"{r[q]:{fmt}} [{r[f'{q}_model_lo']:{fmt}}–{r[f'{q}_model_hi']:{fmt}}; "
                        f"{r[f'{q}_literature_lo']:{fmt}}–{r[f'{q}_literature_hi']:{fmt}}]")
            others = []
            for h in alt:
                v = con[(con.hypothesis == h) & np.isclose(con.senescent_share, r.senescent_share)
                        & (con.onset == r.onset) & np.isclose(con.time_h, r.time_h)].d_angle
                others.append(f'{v.iloc[0]:+.1f}' if len(v) else '—')
            lines.append(f"| {int(r.senescent_share * 100)} % | {r.onset} | {r.time_h:g} | {band('d_angle', '+.1f')} | "
                         f"{band('d_ar', '+.2f')} | {r.fields_per_arm:g} | " + ' | '.join(others) + ' |')
    return '\n'.join(lines) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('predict', 'compare'):
        s = sub.add_parser(name)
        s.add_argument('--top', type=float, default=4.0, help='top of the justified shear range (Pa)')
        s.add_argument('--ramp-min', type=float, default=30.0, help='duration of the linear ramp (min)')
        s.add_argument('--out', default=OUT)
        if name == 'compare':
            s.add_argument('--measured', required=True, help='one row per field (see the module docstring)')
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    if a.cmd == 'predict':
        pred = predict(a.top, a.ramp_min, hypotheses=tuple(HYPOTHESES))
        con = contrast(a.top, a.ramp_min)
        pred.to_csv(os.path.join(a.out, 'predictions.csv'), index=False)
        con.to_csv(os.path.join(a.out, 'contrast.csv'), index=False)
        with open(os.path.join(a.out, 'predictions.md'), 'w') as f:
            f.write(markdown(pred, a.top, a.ramp_min, con))
        with open(os.path.join(a.out, 'parameters.json'), 'w') as f:
            json.dump({'PARAMS': PARAMS, 'HYPOTHESES': HYPOTHESES, 'GAP': {k: (v if not isinstance(v, dict) else {str(kk): vv for kk, vv in v.items()})
                                                 for k, v in GAP.items()},
                       'top_pa': a.top, 'ramp_min': a.ramp_min}, f, indent=1)
        figure(pred, os.path.join(a.out, 'fig_predictions'), a.top)
        print(markdown(pred, a.top, a.ramp_min, con))
    else:
        table, kin = compare(pd.read_csv(a.measured), a.top, a.ramp_min)
        table.to_csv(os.path.join(a.out, 'comparison.csv'), index=False)
        kin.to_csv(os.path.join(a.out, 'kinetics_fit.csv'), index=False)
        with pd.option_context('display.width', 200, 'display.max_columns', 40):
            print(table.to_string(index=False, float_format=lambda v: f'{v:.2f}'))
            print(kin.to_string(index=False, float_format=lambda v: f'{v:.2f}'))


if __name__ == '__main__':
    main()

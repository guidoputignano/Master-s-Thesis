"""Senescence estimate grounded in the A1 design and Chala et al. (2021).

Chala et al. (Nano Lett. 21:4911) report TNF-alpha-treated HUVECs 2.3x larger than
controls on average (2354 vs 5335 um^2), with log-normal area distributions. The A1
slides mix 70 % control and 30 % TNF-alpha-treated cells. Absolute thresholds from
that paper (e.g. > 5000 um^2) do not transfer: these monolayers are ~2.7x denser.
The scale-free prediction does transfer. For interior cells, log(area) should be a
two-component mixture:

* the larger component's median is about 2.3x the smaller one's;
* it holds at most ~30 % of cells (less if controls kept dividing).

"Larger by a factor" means the same distribution shape shifted in log area, so the
two components share one variance. That constraint matters: with free variances, the
likelihood is often highest for a narrow core plus a broad component (weight 0.7-0.8,
ratio 1.2-1.9), which only describes skewness, not a second population.

``fit`` estimates the shifted-population model by EM from several starting points,
either with a free ratio or with the ratio fixed at Chala's 2.27. It compares one
versus two components by BIC. With equal variances the ratio of medians equals the
ratio of means, which is the quantity Chala reports. ``summarise`` adds field-level
bootstrap confidence intervals. Each cell's posterior probability of belonging to
the enlarged component replaces hand-set gates and k-means (which always returns two
clusters, whether or not two populations exist).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CHALA_RATIO = 5335 / 2354          # 2.27x, TNF-alpha vs control mean area
DESIGN_TNF_FRACTION = 0.30          # A1 = 70 % control + 30 % TNF-alpha-treated


def _norm_logpdf(x, mu, var):
    return -0.5 * (np.log(2 * np.pi * var) + (x - mu) ** 2 / var)


STARTS = [(q0, q1, w) for q0, q1 in ((0.4, 0.9), (0.3, 0.8), (0.5, 0.95), (0.2, 0.7)) for w in (0.1, 0.25, 0.5)]
WARM_STARTS = [(0.4, 0.9, 0.25), (0.2, 0.7, 0.5)]      # added to a warm start (bootstrap)


def _em(x, mu1, mu2, var1, var2, w2, fixed_log_ratio, equal_var, n_iter, tol):
    prev = ll = -np.inf
    for _ in range(n_iter):
        l1 = np.log(1 - w2) + _norm_logpdf(x, mu1, var1)
        l2 = np.log(w2) + _norm_logpdf(x, mu2, var2)
        m = np.maximum(l1, l2)
        ll = float((m + np.log(np.exp(l1 - m) + np.exp(l2 - m))).sum())
        r2 = np.exp(l2 - m) / (np.exp(l1 - m) + np.exp(l2 - m))
        r1 = 1 - r2
        w2 = float(np.clip(r2.mean(), 1e-6, 1 - 1e-6))
        if fixed_log_ratio is None:
            mu1, mu2 = (r1 * x).sum() / r1.sum(), (r2 * x).sum() / r2.sum()
        else:
            a1, a2 = r1 / var1, r2 / var2
            mu1 = ((a1 * x).sum() + (a2 * (x - fixed_log_ratio)).sum()) / (a1.sum() + a2.sum())
            mu2 = mu1 + fixed_log_ratio
        if equal_var:
            var1 = var2 = max(((r1 * (x - mu1) ** 2).sum() + (r2 * (x - mu2) ** 2).sum()) / len(x), 1e-6)
        else:
            var1 = max((r1 * (x - mu1) ** 2).sum() / r1.sum(), 1e-6)
            var2 = max((r2 * (x - mu2) ** 2).sum() / r2.sum(), 1e-6)
        if ll - prev < tol:
            break
        prev = ll
    return ll, mu1, mu2, var1, var2, w2


def fit(x, fixed_log_ratio=None, equal_var=True, n_iter=1000, tol=1e-9, init=None):
    """Two-component 1-D Gaussian mixture on ``x`` (log areas) by EM, best of several starts.

    With ``fixed_log_ratio`` the component means differ by exactly that amount; with
    ``equal_var`` (default) the components share one variance. ``init`` (the ``params``
    of an earlier fit) adds a warm start and reduces the grid to two more starts, which
    is what the bootstrap uses. Returns a dict with the enlarged fraction, median ratio,
    BIC for one and two components and the parameters.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return dict(n=n)
    starts = []
    for q0, q1, w in (STARTS if init is None else WARM_STARTS):
        mu1, mu2 = np.quantile(x, [q0, q1])
        starts.append((mu1, mu1 + fixed_log_ratio if fixed_log_ratio is not None else mu2,
                       x.var() / 2, x.var() / 2, w))
    if init is not None:
        mu1, var1, mu2, var2, w2 = init
        starts.insert(0, (mu1, mu2, var1, var2, w2))
    best = None
    for mu1, mu2, var1, var2, w in starts:
        res = _em(x, mu1, mu2, var1, var2, w, fixed_log_ratio, equal_var, n_iter, tol)
        if best is None or res[0] > best[0] + 1e-9:
            best = res
    ll, mu1, mu2, var1, var2, w2 = best
    if mu2 < mu1:                     # keep component 2 = enlarged
        mu1, mu2, var1, var2, w2 = mu2, mu1, var2, var1, 1 - w2
    ll1 = float(_norm_logpdf(x, x.mean(), x.var()).sum())
    k2 = 5 - (fixed_log_ratio is not None) - bool(equal_var)
    return dict(n=n, frac_enlarged=w2, median_ratio=float(np.exp(mu2 - mu1)),
                mean_ratio=float(np.exp(mu2 + var2 / 2 - mu1 - var1 / 2)),
                median_small=float(np.exp(mu1)), median_large=float(np.exp(mu2)),
                sd_small=float(np.sqrt(var1)), sd_large=float(np.sqrt(var2)),
                loglik=ll, bic1=-2 * ll1 + 2 * np.log(n), bic2=-2 * ll + k2 * np.log(n),
                params=(mu1, var1, mu2, var2, w2))


def posterior(x, params):
    """P(enlarged component | x) for each value of ``x``."""
    mu1, var1, mu2, var2, w2 = params
    l1 = np.log(1 - w2) + _norm_logpdf(x, mu1, var1)
    l2 = np.log(w2) + _norm_logpdf(x, mu2, var2)
    return 1 / (1 + np.exp(l1 - l2))


def summarise(cells, group='condition', value='area_um2', field='key', n_boot=300, seed=0,
              fixed_log_ratio=None, equal_var=True):
    """Per-group mixture fit with field-level bootstrap CIs. ``cells`` needs ``value``, ``field``."""
    rng = np.random.default_rng(seed)
    out = []
    for g, d in cells.groupby(group):
        x = np.log(d[value].to_numpy())
        f = fit(x, fixed_log_ratio, equal_var)
        fields = d[field].unique()
        by_field = {k: np.log(v[value].to_numpy()) for k, v in d.groupby(field)}
        boots = []
        for _ in range(n_boot):
            pick = rng.choice(fields, len(fields), replace=True)
            b = fit(np.concatenate([by_field[k] for k in pick]), fixed_log_ratio, equal_var, init=f['params'])
            boots.append((b.get('frac_enlarged', np.nan), b.get('median_ratio', np.nan)))
        boots = np.array(boots)
        lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
        out.append(dict(group=g, n_cells=f['n'], n_fields=len(fields), delta_bic=f['bic1'] - f['bic2'],
                        frac_enlarged=f['frac_enlarged'], frac_lo=lo[0], frac_hi=hi[0],
                        median_ratio=f['median_ratio'], ratio_lo=lo[1], ratio_hi=hi[1],
                        median_small=f['median_small'], median_large=f['median_large'],
                        sd_small=f['sd_small'], sd_large=f['sd_large']))
    return pd.DataFrame(out)

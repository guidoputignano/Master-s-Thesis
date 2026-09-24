"""Nuclear-marker mixture for senescent (TNF-alpha-enlarged) cells.

Used by by_shear.py on the largest Cellpose nucleus of each interior cell (IMAGING_VALIDATION
section 6). Adapted from senescence.py (which it does not import):
the same shifted log-normal model (two components that share one (co)variance, so the
enlarged component is the normal one shifted in log space), the same minority constraint
(enlarged weight <= max_frac, best unconstrained fit kept as ``*_any``), BIC against one
component, EM from a grid of starting points. Generalised here to d dimensions with a
shared full covariance, so that the 1-D fit on log nuclear area and the 2-D fit on
(log nuclear area, log DAPI content) are the same code. For d=1 it reproduces
senescence.fit(equal_var=True, fixed_log_ratio=None).

Starting points are always the full grid (also in the bootstrap), so bootstrap CIs are not
pulled towards the point estimate by a warm start.

Change against senescence.py (found while checking degenerate bootstrap draws): senescence.py
discards EM solutions whose enlarged weight exceeds max_frac. When the data's enlarged share is
above max_frac (static 20dec21 fields hold 20-55 % large-nucleus cells), every start is discarded
except a degenerate one (two coincident components). ``fit`` therefore adds, when the best
constrained solution is degenerate (median ratio < 1.2) or missing, EM runs whose weight is
clipped at max_frac in the M-step (the constrained maximiser of the concave weight term), from
the unconstrained optimum and from the grid. The result is the constrained MLE on the boundary;
``fallback`` in the output records that this happened.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

# (q0, q1, w): means at quantiles q0 / q1 of each dimension, starting weight w of the enlarged
# component. Same grid as senescence.STARTS.
STARTS = ([(q0, q1, w) for q0, q1 in ((0.4, 0.9), (0.3, 0.8), (0.5, 0.95), (0.2, 0.7)) for w in (0.1, 0.25, 0.5)]
          + [(0.02, 0.55, 0.97), (0.05, 0.6, 0.9)])


def _logpdf(x, mu, icov, logdet):
    """x (n,d), mu (d,), shared covariance given by inverse and log-determinant."""
    z = x - mu
    d = x.shape[1]
    if d == 1:
        return -0.5 * (np.log(2 * np.pi) + logdet + icov[0, 0] * z[:, 0] ** 2)
    return -0.5 * (d * np.log(2 * np.pi) + logdet + ((z @ icov) * z).sum(1))


def _inv(cov):
    cov = np.atleast_2d(cov)
    sign, logdet = np.linalg.slogdet(cov)
    return np.linalg.inv(cov), logdet


def _em(x, mu1, mu2, cov, w2, n_iter=2000, tol=1e-9, w_max=1 - 1e-6):
    n, d = x.shape
    prev = -np.inf
    ll = -np.inf
    for _ in range(n_iter):
        icov, logdet = _inv(cov)
        l1 = np.log(1 - w2) + _logpdf(x, mu1, icov, logdet)
        l2 = np.log(w2) + _logpdf(x, mu2, icov, logdet)
        m = np.maximum(l1, l2)
        s = np.exp(l1 - m) + np.exp(l2 - m)
        ll = float((m + np.log(s)).sum())
        r2 = np.exp(l2 - m) / s
        r1 = 1 - r2
        w2 = float(np.clip(r2.mean(), 1e-6, w_max))
        mu1 = (r1[:, None] * x).sum(0) / max(r1.sum(), 1e-12)
        mu2 = (r2[:, None] * x).sum(0) / max(r2.sum(), 1e-12)
        z1, z2 = x - mu1, x - mu2
        cov = ((r1[:, None] * z1).T @ z1 + (r2[:, None] * z2).T @ z2) / n
        cov = cov + np.eye(d) * 1e-8
        if ll - prev < tol:
            break
        prev = ll
    return ll, mu1, mu2, cov, w2


def one_component_loglik(x):
    x = np.atleast_2d(np.asarray(x, float).T).T if np.ndim(x) == 1 else np.asarray(x, float)
    mu = x.mean(0)
    cov = np.atleast_2d(np.cov(x.T, bias=True))
    icov, logdet = _inv(cov)
    return float(_logpdf(x, mu, icov, logdet).sum()), mu, cov


def fit(x, max_frac=0.5, orient_dim=0):
    """Two-component shared-covariance Gaussian mixture on x ((n,) or (n,d), already log-transformed).

    Component 2 ('enlarged') is the one with the larger mean in ``orient_dim`` and must hold at most
    ``max_frac`` of the cells. Returns weight, ratios, BIC one vs two components, parameters.
    """
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[:, None]
    x = x[np.isfinite(x).all(1)]
    n, d = x.shape
    if n < 10:
        return dict(n=n)
    cov0 = np.atleast_2d(np.cov(x.T, bias=True))
    starts = []
    for q0, q1, w in STARTS:
        starts.append((np.quantile(x, q0, axis=0), np.quantile(x, q1, axis=0), cov0 / 2, w))
    if d > 1:     # hard splits on each dimension and on the sum, to reach splits along either axis
        for s in list(range(d)) + ['sum']:
            v = x.sum(1) if s == 'sum' else x[:, s]
            for q in (0.6, 0.75, 0.9):
                hi = v > np.quantile(v, q)
                starts.append((x[~hi].mean(0), x[hi].mean(0), cov0 / 2, hi.mean()))
    best = best_any = None
    for mu1, mu2, cov, w in starts:
        ll_, m1, m2, c, w2 = _em(x, mu1, mu2, cov, w)
        if m2[orient_dim] < m1[orient_dim]:
            m1, m2, w2 = m2, m1, 1 - w2
        res = (ll_, m1, m2, c, w2)
        if best_any is None or res[0] > best_any[0] + 1e-9:
            best_any = res
        if w2 <= max_frac and (best is None or res[0] > best[0] + 1e-9):
            best = res
    constrained_found = best is not None
    fallback = False
    if best is None or np.exp(best[2][orient_dim] - best[1][orient_dim]) < 1.2:
        fb_starts = [(best_any[1], best_any[2], best_any[3], min(best_any[4], max_frac))] + \
                    [(m1, m2, c, min(w, max_frac)) for m1, m2, c, w in starts[:12]]
        for mu1, mu2, cov, w in fb_starts:
            ll_, m1, m2, c, w2 = _em(x, mu1, mu2, cov, w, w_max=max_frac)
            if m2[orient_dim] < m1[orient_dim]:
                m1, m2, w2 = m2, m1, 1 - w2
            if w2 <= max_frac + 1e-9 and (best is None or ll_ > best[0] + 1e-9):
                best = (ll_, m1, m2, c, w2)
                fallback = True
        constrained_found = best is not None
    if best is None:
        best = best_any
    ll, mu1, mu2, cov, w2 = best
    ll1, _, _ = one_component_loglik(x)
    k1 = d + d * (d + 1) // 2
    k2 = 2 * d + d * (d + 1) // 2 + 1
    sd = np.sqrt(np.diag(cov))
    out = dict(n=n, d=d, frac_enlarged=float(w2),
               ratio=float(np.exp(mu2[0] - mu1[0])),           # median ratio in dimension 0
               median_small=float(np.exp(mu1[0])), median_large=float(np.exp(mu2[0])), sd=float(sd[0]),
               separation=float(np.sqrt((mu2 - mu1) @ np.linalg.solve(cov, mu2 - mu1))),   # Mahalanobis
               loglik=ll, loglik1=ll1, bic1=-2 * ll1 + k1 * np.log(n), bic2=-2 * ll + k2 * np.log(n),
               bic2_any=-2 * best_any[0] + k2 * np.log(n), frac_any=float(best_any[4]),
               constrained_found=constrained_found, fallback=fallback, params=(mu1, mu2, cov, w2))
    if d > 1:
        out.update(ratio_dim1=float(np.exp(mu2[1] - mu1[1])), median_small_dim1=float(np.exp(mu1[1])),
                   median_large_dim1=float(np.exp(mu2[1])), corr=float(cov[0, 1] / (sd[0] * sd[1])))
    return out


def posterior(x, params):
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[:, None]
    mu1, mu2, cov, w2 = params
    icov, logdet = _inv(cov)
    l1 = np.log(1 - w2) + _logpdf(x, mu1, icov, logdet)
    l2 = np.log(w2) + _logpdf(x, mu2, icov, logdet)
    return 1 / (1 + np.exp(np.clip(l1 - l2, -700, 700)))


def mixture_logpdf(x, params):
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[:, None]
    mu1, mu2, cov, w2 = params
    icov, logdet = _inv(cov)
    l1 = np.log(1 - w2) + _logpdf(x, mu1, icov, logdet)
    l2 = np.log(w2) + _logpdf(x, mu2, icov, logdet)
    return np.logaddexp(l1, l2)


def skewnorm_fit(x):
    """Unimodal skewed alternative (3 parameters, 1-D). Returns (loglik, params)."""
    x = np.asarray(x, float)
    p = stats.skewnorm.fit(x)
    # refine from a few starts; scipy's fit can stall
    best = (float(stats.skewnorm.logpdf(x, *p).sum()), p)
    for a0 in (2.0, 5.0, 10.0):
        try:
            q = stats.skewnorm.fit(x, a0, loc=np.quantile(x, 0.2), scale=x.std() * 1.3)
            ll = float(stats.skewnorm.logpdf(x, *q).sum())
            if ll > best[0]:
                best = (ll, q)
        except Exception:
            pass
    return best


def n_modes(params, lo, hi, n=4000):
    """Number of local maxima of the fitted 1-D mixture density on [lo, hi]."""
    g = np.linspace(lo, hi, n)
    f = np.exp(mixture_logpdf(g, params))
    return int(((f[1:-1] > f[:-2]) & (f[1:-1] > f[2:])).sum())

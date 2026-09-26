"""Calibration of the monolayer-state model and its ensemble of accepted parameter sets.

Least squares on the training observations (weighted by their sigma) with weak Gaussian priors
(centred on the initial values, sd half the allowed range) that only keep unidentified parameters
from drifting to their bounds. Multi-start from Latin-hypercube points. The ensemble is a
random-walk Metropolis sample of exp(-chi2/2 - prior) started at the best fit, thinned; it carries
the parameter uncertainty into every prediction.
"""
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from . import model as M
from . import observations as O

PRIOR_SD = 0.5 * (M.HI - M.LO)
PRIOR_MEAN = M.X0.copy()
# Informative prior from studies outside the calibration set: HUVEC align along the flow at 1-2 Pa
# and misalign above (gradient chamber, crossover 2.7 Pa; Baeyens et al., eLife 2015), and flat
# monolayers in the reference chamber still aligned at 5 Pa (Robotti et al. 2014).
INFORMATIVE = {"tau_x": (3.0, 0.75)}
for _k, (_m, _s) in INFORMATIVE.items():
    PRIOR_MEAN[M.NAMES.index(_k)] = _m
    PRIOR_SD[M.NAMES.index(_k)] = _s


class Problem:
    def __init__(self, role="train", exclude=(), only=None):
        obs = O.observations(role) if only is None else [o for o in O.OBS if o.protocol in only]
        self.obs = [o for o in obs if o.protocol not in exclude]
        self.protocols = O.protocols_for(self.obs)
        self.index = {p.name: i for i, p in enumerate(self.protocols)}
        self.duration = max(p.duration for p in self.protocols)
        t, sh = M.shear_array(self.protocols, self.duration)
        self.t, self.shear = t, sh
        self.pp1 = None
        if any(p.pp1_from is not None for p in self.protocols):
            self.pp1 = np.stack([(t >= p.pp1_from) if p.pp1_from is not None else np.zeros_like(t, bool)
                                 for p in self.protocols], axis=1)
        self.surfaces = [M.SURFACES[p.surface] for p in self.protocols]
        self.k = np.array([min(int(round(o.time / M.DT)), len(t) - 1) for o in self.obs])
        self.j = np.array([self.index[o.protocol] for o in self.obs])
        self.q = [o.quantity for o in self.obs]
        self.y = np.array([o.value for o in self.obs])
        self.s = np.array([o.sigma for o in self.obs])

    def predict(self, theta):
        out = M.simulate(self.shear, theta, self.surfaces, self.pp1)
        return np.array([out[q][k, j] for q, k, j in zip(self.q, self.k, self.j)])

    def residuals(self, theta, prior=True):
        r = (self.predict(theta) - self.y) / self.s
        if prior:
            r = np.concatenate([r, (np.asarray(theta) - PRIOR_MEAN) / PRIOR_SD])
        return r

    def chi2(self, theta):
        r = (self.predict(theta) - self.y) / self.s
        return float(r @ r)


def _one(args):
    x0, max_nfev, exclude = (args + ((),))[:3] if len(args) == 2 else args
    pb = Problem("train", exclude=exclude)
    res = least_squares(pb.residuals, x0, bounds=(M.LO, M.HI), x_scale="jac", max_nfev=max_nfev,
                        diff_step=1e-3)
    return res.x, pb.chi2(res.x)


def fit(n_starts=8, seed=0, max_nfev=500, processes=4, verbose=True, exclude=(), x_first=None):
    """Multi-start least squares (Latin-hypercube starts in the middle 60 % of each range)."""
    from multiprocessing import Pool
    rng = np.random.default_rng(seed)
    starts = [M.X0.copy() if x_first is None else np.asarray(x_first, float)]
    perm = np.array([rng.permutation(n_starts - 1) for _ in M.X0]).T
    for i in range(n_starts - 1):
        u = (perm[i] + rng.random(len(M.X0))) / (n_starts - 1)
        starts.append(M.LO + (0.2 + 0.6 * u) * (M.HI - M.LO))
    t0 = time.time()
    with Pool(processes) as pool:
        results = pool.map(_one, [(x, max_nfev, tuple(exclude)) for x in starts])
    if verbose:
        for i, (_, c2) in enumerate(results):
            print(f"start {i}: chi2 {c2:.1f}", flush=True)
        print(f"{len(starts)} starts in {time.time() - t0:.0f} s", flush=True)
    best = min(results, key=lambda r: r[1])
    return best[0], best[1], Problem("train", exclude=exclude)


def log_post(theta, pb, temperature=1.0):
    """Log posterior with the likelihood tempered by `temperature` (the reduced chi2 of the best
    fit), which widens the errors so that the misfit between experiments is carried as uncertainty."""
    if np.any(theta < M.LO) or np.any(theta > M.HI):
        return -np.inf
    r = (pb.predict(theta) - pb.y) / pb.s
    pr = (np.asarray(theta) - PRIOR_MEAN) / PRIOR_SD
    return -0.5 * float(r @ r) / temperature - 0.5 * float(pr @ pr)


def ensemble(theta_best, pb, n_steps=40000, thin=100, seed=1, scale=0.02, temperature=None, verbose=True):
    """Random-walk Metropolis in parameter space scaled by the prior range; adaptive step size.
    Returns the thinned samples after a burn-in of a fifth of the chain."""
    rng = np.random.default_rng(seed)
    if temperature is None:
        temperature = max(pb.chi2(theta_best) / (len(pb.y) - len(M.NAMES)), 1.0)
    x = np.array(theta_best, float)
    lp = log_post(x, pb, temperature)
    width = (M.HI - M.LO) * scale
    kept, acc = [], 0
    for i in range(1, n_steps + 1):
        y = x + rng.normal(size=x.size) * width
        ly = log_post(y, pb, temperature)
        if np.log(rng.random()) < ly - lp:
            x, lp, acc = y, ly, acc + 1
        if i % 500 == 0:                          # keep acceptance near 0.25
            rate = acc / 500
            width *= np.exp(2.0 * (rate - 0.25))
            acc = 0
            if verbose and i % 5000 == 0:
                print(f"step {i}: acceptance {rate:.2f}, chi2 {pb.chi2(x):.1f}", flush=True)
        if i > n_steps // 5 and i % thin == 0:
            kept.append(x.copy())
    return np.array(kept), temperature


def held_out(samples, theta_best):
    """Predictions for the test observations: best fit, median and 90 % interval of the ensemble."""
    pt = Problem("test")
    ens = np.array([pt.predict(th) for th in samples])
    rows = []
    for j, o in enumerate(pt.obs):
        lo, med, hi = np.percentile(ens[:, j], [5, 50, 95])
        rows.append(dict(protocol=o.protocol, quantity=o.quantity, observed=o.value, sigma=o.sigma,
                         best=float(pt.predict(theta_best)[j]), median=float(med), p5=float(lo), p95=float(hi)))
    return rows


def leave_out(groups, theta_best, n_starts=8):
    """Refit without each group of protocols and predict it: a check that the mechanisms, not the
    fit, carry the behaviour of that experiment."""
    rows = []
    for name, protos in groups.items():
        th, c2, _ = fit(n_starts=n_starts, max_nfev=2500, verbose=False, exclude=protos, x_first=theta_best)
        pt = Problem("train", only=protos)
        pred = pt.predict(th)
        full = pt.predict(theta_best)
        for o, pv, fv in zip(pt.obs, pred, full):
            rows.append(dict(group=name, protocol=o.protocol, time=o.time, quantity=o.quantity,
                             observed=o.value, sigma=o.sigma, left_out=float(pv), full_fit=float(fv)))
    return rows


def _fixed(args):
    """Least squares with one parameter held at a value (profile likelihood)."""
    name, value, x0, max_nfev, seed = args
    pb = Problem("train")
    i = M.NAMES.index(name)
    free = np.array([k for k in range(len(M.NAMES)) if k != i])
    rng = np.random.default_rng(seed)
    best = None
    for start in [x0[free]] + [M.LO[free] + (0.2 + 0.6 * rng.random(free.size)) * (M.HI[free] - M.LO[free])
                               for _ in range(3)]:
        def res(x):
            th = np.array(x0, float); th[free] = x; th[i] = value
            return pb.residuals(th, prior=True)
        r = least_squares(res, start, bounds=(M.LO[free], M.HI[free]), x_scale="jac", max_nfev=max_nfev,
                          diff_step=1e-3)
        th = np.array(x0, float); th[free] = r.x; th[i] = value
        c2 = pb.chi2(th)
        if best is None or c2 < best[1]:
            best = (th, c2)
    return best


def source_of(protocol):
    """Data source of a protocol name, for splitting the fit by study."""
    if protocol.startswith("stef22"):
        return "Stefopoulos 2022"
    if protocol.startswith("rob14") and ("par" in protocol or "perp" in protocol):
        return "Robotti 2014, gratings"
    if protocol.startswith("rob14"):
        return "Robotti 2014, flat and static"
    return "Wu 2021"


def profile(theta_best, name="tau_x", values=(2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0), max_nfev=4000,
            processes=3, seed=11):
    """Profile of the data chi2 over one parameter: refit all others with it held at each value
    (best fit and three random starts). Returns, per value, the refitted parameters, the chi2 and
    its split by data source."""
    from multiprocessing import Pool
    with Pool(processes) as pool:
        fits = pool.map(_fixed, [(name, v, theta_best, max_nfev, seed + k) for k, v in enumerate(values)])
    pb = Problem("train")
    src = np.array([source_of(o.protocol if isinstance(o.protocol, str) else o.protocol.name)
                    for o in O.observations("train")])
    rows = []
    for v, (th, c2) in zip(values, fits):
        r = (pb.predict(th) - pb.y) / pb.s
        rows.append(dict(value=float(v), chi2=float(c2), theta=th.tolist(),
                         by_source={g: float(np.sum(r[src == g] ** 2)) for g in sorted(set(src))}))
    return dict(parameter=name, rows=rows, n_obs={g: int(np.sum(src == g)) for g in sorted(set(src))})


def scenario_ensemble(theta_best, name="tau_x", values=(2.2, 3.0, 4.0), n_per=16, n_steps=12000, seed=3):
    """Parameter sets for design under a scenario of a weakly identified parameter: for each value,
    refit the others with it held fixed, then sample around that fit (tempered as the main
    ensemble) with it still fixed. Used so that designs hold wherever the switch lies."""
    from multiprocessing import Pool
    with Pool(min(4, len(values))) as pool:
        fits = pool.map(_fixed, [(name, v, theta_best, 2500, seed + k) for k, v in enumerate(values)])
    pb = Problem("train")
    i = M.NAMES.index(name)
    out = []
    for (th, c2), v in zip(fits, values):
        temp = max(c2 / (len(pb.y) - len(M.NAMES) + 1), 1.0)
        rng = np.random.default_rng(seed)
        x = th.copy()
        lp = log_post(x, pb, temp)
        width = (M.HI - M.LO) * 0.01
        width[i] = 0.0
        kept = []
        for step in range(1, n_steps + 1):
            y = x + rng.normal(size=x.size) * width
            ly = log_post(y, pb, temp)
            if np.log(rng.random()) < ly - lp:
                x, lp = y, ly
            if step > n_steps // 4 and step % ((3 * n_steps // 4) // n_per) == 0:
                kept.append(x.copy())
        out.append(dict(value=v, chi2=c2, samples=np.array(kept[:n_per])))
    return out


def save(path, theta_best, chi2, samples, pb, temperature=1.0, tests=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = [dict(protocol=o.protocol, time=o.time, quantity=o.quantity, value=o.value, sigma=o.sigma,
                  fit=float(f)) for o, f in zip(pb.obs, pb.predict(theta_best))]
    json.dump(dict(names=M.NAMES, best=list(map(float, theta_best)), chi2=chi2, n_obs=len(pb.y),
                   n_params=len(M.NAMES), temperature=temperature, informative_priors=INFORMATIVE,
                   samples=samples.tolist(), fit_table=table, held_out=tests or []), open(path, "w"), indent=1)


def load(path):
    d = json.load(open(path))
    return np.array(d["best"]), np.array(d["samples"]), d


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "results/conditioning/calibration.json"
    theta, c2, pb = fit(n_starts=16, max_nfev=2500)
    samples, temp = ensemble(theta, pb)
    save(out, theta, c2, samples, pb, temp, held_out(samples, theta))
    print("saved", out)

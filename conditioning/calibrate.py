"""Calibration of the monolayer-state model and its ensemble of accepted parameter sets.

Least squares on the training observations (weighted by their sigma) with weak Gaussian priors
(centred on the initial values, sd half the allowed range); several parameters still end at a
bound, which the paper reports. Multi-start: the initial values and Latin-hypercube points. The
ensemble is an adaptive random-walk Metropolis sample of the tempered posterior
exp(-chi2/(2T) - prior) started at the best fit, thinned; it carries the parameter uncertainty into
every prediction. Scenario ensembles hold one weakly identified parameter (the switch shear) at a
value and sample the others the same way.
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
# Prior on the switch shear: a modelling choice, not a measurement. It spans the upper edge of the
# along-flow band of HUVEC in a gradient chamber (aligned at about 1-2 Pa, orientation crossing 45 deg
# near 2 Pa after 16 h; Baeyens et al., eLife 2015, not in the calibration set) and higher values.
# The lower bound of tau_x (2 Pa, model.PARAMS) is set from the same study. The 2.7 Pa of our first
# draft was the zero crossing of the nematic analogy of Stefopoulos et al. 2022 mapped linearly onto
# shear, not a value of Baeyens et al.
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


def _chain(args):
    """One random-walk Metropolis chain of the tempered posterior (a Pool worker).

    Gaussian proposals; a proposal outside the parameter bounds is rejected. During the burn-in (a
    quarter of the steps) the proposal scale is adapted every 500 steps toward an acceptance of 0.25,
    and at a quarter, half and three quarters of the burn-in its covariance is re-estimated from the
    second half of the chain so far (adaptive Metropolis), blended with cov0 at the first update; both
    are frozen before any sample is kept. Parameters listed in `fixed` stay at their starting values.
    Returns the n_keep thinned samples and the acceptance rate after the burn-in."""
    x0, cov0, fixed, temperature, n_steps, n_keep, seed = args
    pb = Problem("train")
    rng = np.random.default_rng(seed)
    free = np.setdiff1d(np.arange(len(x0)), np.asarray(fixed, int))
    jitter = np.diag((1e-4 * (M.HI - M.LO)[free]) ** 2)

    def chol(c):
        return np.linalg.cholesky(np.asarray(c)[np.ix_(free, free)] + jitter)

    L = chol(cov0)
    scale = 2.38 / np.sqrt(free.size)
    x = np.array(x0, float)
    lp = log_post(x, pb, temperature)
    burn = n_steps // 4
    updates = {burn // 4: 0.5, burn // 2: 0.0, 3 * burn // 4: 0.0}   # step: weight of cov0
    thin = (n_steps - burn) // n_keep
    hist, kept, acc, acc_after = [], [], 0, 0
    for i in range(1, n_steps + 1):
        y = x.copy()
        y[free] += scale * (L @ rng.normal(size=free.size))
        ly = log_post(y, pb, temperature)
        if np.log(rng.random()) < ly - lp:
            x, lp = y, ly
            acc += 1
            acc_after += i > burn
        if i <= burn:
            hist.append(x.copy())
            if i % 500 == 0:
                scale *= np.exp(2.0 * (acc / 500 - 0.25))
                acc = 0
            if i in updates:
                h = np.array(hist[len(hist) // 2:])
                w0 = updates[i]
                L = chol(w0 * np.asarray(cov0) + (1 - w0) * np.cov(h.T))
                scale = 2.38 / np.sqrt(free.size)
        elif (i - burn) % thin == 0 and len(kept) < n_keep:
            kept.append(x.copy())
    return np.array(kept), acc_after / (n_steps - burn)


def split_rhat(chains):
    """Split-R-hat per parameter for chains (n_chains, n_samples, n_params) (Gelman et al.)."""
    c = np.asarray(chains, float)
    h = c.shape[1] // 2
    c = np.concatenate([c[:, :h], c[:, h:2 * h]], axis=0)
    m, n = c.shape[0], c.shape[1]
    w = c.var(axis=1, ddof=1).mean(axis=0)
    b = n * c.mean(axis=1).var(axis=0, ddof=1)
    var = (n - 1) / n * w + b / n
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(np.where(w > 0, var / w, 1.0))


def ensemble(theta_best, pb, n_steps=100000, n_keep=320, n_chains=4, seed=1, temperature=None):
    """The main ensemble. n_chains adaptive chains start at the best fit and run in parallel. The
    tempered posterior is multimodal (independent chains settle in different modes, with the switch
    shear near 2 or near 4 Pa), which a random-walk sampler cannot weight; the ensemble is therefore
    the chain with the lowest mean chi2, which describes the uncertainty around the best fit, and the
    weakly identified switch is covered by the scenario ensembles. Returns the samples, the
    temperature, their acceptance rate, and diagnostics of all chains (split-R-hat across chains,
    per-chain mean chi2 and medians)."""
    from multiprocessing import Pool
    if temperature is None:
        temperature = max(pb.chi2(theta_best) / (len(pb.y) - len(M.NAMES)), 1.0)
    cov0 = np.diag((0.02 * (M.HI - M.LO)) ** 2)
    jobs = [(np.asarray(theta_best, float), cov0, (), temperature, n_steps, n_keep, seed + k)
            for k in range(n_chains)]
    with Pool(n_chains) as pool:
        res = pool.map(_chain, jobs)
    chains = np.array([r[0] for r in res])
    mean_chi2 = [float(np.mean([pb.chi2(th) for th in c])) for c in chains]
    best = int(np.argmin(mean_chi2))
    diag = dict(n_chains=n_chains, n_steps=n_steps, chosen=best, mean_chi2=mean_chi2,
                acceptance=[float(r[1]) for r in res],
                split_rhat=dict(zip(M.NAMES, map(float, split_rhat(chains)))),
                chain_medians={n: [float(np.median(c[:, i])) for c in chains] for i, n in enumerate(M.NAMES)})
    return chains[best], temperature, float(res[best][1]), diag


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


# Groups of protocols refitted without and then predicted (a check that the mechanisms, not the
# fit, carry the behaviour of that experiment).
LEAVE_OUT = {
    "gradual drop (Fig. S5)": ("stef22_gradual",),
    "abrupt drop (Fig. 4k,l)": ("stef22_8_to_1.4",),
    "abrupt rise from aligned (Fig. 4g,h)": ("stef22_1.4_to_8",),
    "PP1 during the drop (Fig. 4l)": ("stef22_8_to_1.4_pp1",),
    "drop after 8 Pa, flat (Wu Fig. 5)": ("wu21_fig5_flat",),
    "perpendicular gratings at 10 Pa": ("rob14_coc_perp_10",),
}


def leave_out(groups, theta_best, n_starts=16):
    """Refit without each group of protocols and predict it. The refits start as the main fit does
    (initial values and Latin-hypercube points), not from the full-data best fit."""
    rows = []
    for name, protos in groups.items():
        th, c2, _ = fit(n_starts=n_starts, max_nfev=2500, verbose=False, exclude=protos)
        pt = Problem("train", only=protos)
        pred = pt.predict(th)
        full = pt.predict(theta_best)
        for o, pv, fv in zip(pt.obs, pred, full):
            rows.append(dict(group=name, protocol=o.protocol, time=o.time, quantity=o.quantity,
                             observed=o.value, sigma=o.sigma, left_out=float(pv), full_fit=float(fv),
                             chi2_refit=float(c2)))
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


def profile(theta_best, name="tau_x", values=(1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0),
            max_nfev=4000, processes=3, seed=11):
    """Profile of the data chi2 over one parameter: refit all others with it held at each value
    (best fit and three random starts). Values may lie outside the parameter's bounds (a check of
    what the bound imposes). Returns, per value, the refitted parameters, the chi2 and its split by
    data source."""
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
    i = M.NAMES.index(name)
    return dict(parameter=name, bounds=[float(M.LO[i]), float(M.HI[i])], rows=rows,
                n_obs={g: int(np.sum(src == g)) for g in sorted(set(src))})


def scenario_ensemble(theta_best, samples, name="tau_x", values=(2.2, 3.0, 4.0), n_per=32, n_steps=80000,
                      seed=3):
    """Parameter sets under scenarios of a weakly identified parameter: for each value, refit the
    others with it held fixed, then sample the tempered posterior (temperature: reduced chi2 of that
    refit) with it still fixed, by an adaptive chain whose initial proposal covariance is that of
    the main ensemble `samples`. Returns n_per distinct thinned sets per value; the designs use
    every other one, the rest serve as monolayers the controller does not know."""
    from multiprocessing import Pool
    i = M.NAMES.index(name)
    with Pool(min(4, len(values))) as pool:
        fits = pool.map(_fixed, [(name, v, theta_best, 2500, seed + k) for k, v in enumerate(values)])
    pb = Problem("train")
    cov0 = np.cov(np.asarray(samples, float).T)
    temps = [max(c2 / (len(pb.y) - len(M.NAMES) + 1), 1.0) for _, c2 in fits]
    jobs = [(th, cov0, (i,), T, n_steps, n_per, seed + 100 + k) for k, ((th, _), T) in enumerate(zip(fits, temps))]
    with Pool(min(4, len(values))) as pool:
        chains = pool.map(_chain, jobs)
    return [dict(value=v, chi2=c2, temperature=T, acceptance=acc, samples=smp)
            for v, (_, c2), T, (smp, acc) in zip(values, fits, temps, chains)]


def save(path, theta_best, chi2, samples, pb, temperature=1.0, tests=None, acceptance=None, chains=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = [dict(protocol=o.protocol, time=o.time, quantity=o.quantity, value=o.value, sigma=o.sigma,
                  fit=float(f)) for o, f in zip(pb.obs, pb.predict(theta_best))]
    json.dump(dict(names=M.NAMES, best=list(map(float, theta_best)), chi2=chi2, n_obs=len(pb.y),
                   n_params=len(M.NAMES), temperature=temperature, acceptance=acceptance,
                   chains=chains,
                   informative_priors=INFORMATIVE, samples=samples.tolist(), fit_table=table,
                   held_out=tests or []), open(path, "w"), indent=1)


def load(path):
    d = json.load(open(path))
    return np.array(d["best"]), np.array(d["samples"]), d


if __name__ == "__main__":
    print("run: python -m conditioning.run_all --recalibrate  (calibration, tests, scenarios and designs)")

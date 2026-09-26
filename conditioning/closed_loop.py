"""Model predictive control of conditioning with imaging feedback (shrinking horizon).

The controller carries the calibration ensemble as a weighted belief (a multiple-model, Bayesian
description of what it does not know). Every DECIDE hours it (1) weighs each parameter set by the
likelihood of all live-imaging measurements so far (orientation from DIC or a membrane dye, and cell
density), (2) re-solves the optimal-control problem of design.optimise over the knots not yet applied,
under the weighted ensemble, warm-started from the current plan and from the best of a library of
simple continuations, and (3) applies the next DECIDE hours of the solution. Knots already applied
are never changed. The plant is a parameter set the controller does not know; the open-loop
comparison applies the path designed once from the unweighted ensemble.

Measurement noise follows the imaging: orientation +-3 deg and density +-3 % per time point.
Junction connectivity needs fixed, stained cells, so it is not measured during conditioning.
"""
import numpy as np

from . import design as Dsg
from . import model as M

DECIDE = 1.0           # h between decisions
SIG_ANGLE = 3.0
SIG_R = 0.03
LEVELS = (0.5, 1.4, 2.0, 3.0, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0)   # Pa, intermediate levels in the library
MAXITER = 30           # L-BFGS-B iterations per decision


def free_knots(target, t_now):
    """Knots the controller may still change at t_now: those after it (the knot at t_now ends the
    segment already applied), or all of them before flow starts."""
    tk = np.arange(Dsg.n_knots(target)) * Dsg.KNOT
    return tk > t_now + 1e-9 if t_now > 0 else np.ones(len(tk), bool)


def continuations(current, target, t_now):
    """A library of paths for the remaining knots: go to the target now or after holding the
    current shear for 1-8 h, ramp to it over 1-8 h, or hold an intermediate level for 1-8 h first.
    Returns knot arrays for the remaining conditioning window."""
    tk = np.arange(Dsg.n_knots(target)) * Dsg.KNOT
    rem = free_knots(target, t_now)
    tr = tk[rem] - t_now
    tau = target.tau
    lib = [np.full(tr.shape, tau)]
    for hold in (1, 2, 4, 8):
        lib.append(np.where(tr < hold, current, tau))
    for ramp in (1, 2, 4, 8):
        lib.append(current + (tau - current) * np.minimum(tr / ramp, 1.0))
    for level in LEVELS + (0.5 * tau, 1.5 * tau):
        for hold in (1, 2, 4, 8):
            lib.append(np.where(tr < hold, level, tau))
    return rem, [np.clip(x, target.tau_min, target.tau_max) for x in lib]


def run(target, prior, plant, open_loop_knots, seed=0, verbose=False):
    """Closed loop against a plant parameter set. Returns the applied knots, weight history and
    the plant outcome for closed and open loop."""
    rng = np.random.default_rng(seed)
    nk = Dsg.n_knots(target)
    knots = np.array(open_loop_knots, float)
    logw = np.zeros(len(prior))
    t_grid = Dsg.times(target)
    history = []
    decisions = np.arange(0.0, target.budget - target.hold, DECIDE)
    obs_t, obs_angle, obs_r = [], [], []
    for t_now in decisions:
        if t_now > 0:
            # plant: simulate the applied path up to now, measure with noise
            sh = Dsg.path(knots, target)
            k = int(round(t_now / M.DT))
            pl = M.simulate(sh[:k + 1, None], plant[:, None], [M.SURFACES[target.surface]])
            obs_t.append(k)
            obs_angle.append(pl["angle"][-1, 0] + rng.normal() * SIG_ANGLE)
            obs_r.append(pl["R"][-1, 0] * (1 + rng.normal() * SIG_R))
            # belief: likelihood of all measurements under each parameter set
            ens = M.simulate(np.repeat(sh[:k + 1, None], len(prior), axis=1), prior.T,
                             [M.SURFACES[target.surface]] * len(prior))
            ll = np.zeros(len(prior))
            for kk, a, r in zip(obs_t, obs_angle, obs_r):
                ll += -0.5 * ((ens["angle"][kk] - a) / SIG_ANGLE) ** 2 - 0.5 * ((ens["R"][kk] - r) / SIG_R) ** 2
            logw = ll
        w = np.exp(logw - logw.max())
        w /= w.sum()
        tx = prior[:, M.NAMES.index("tau_x")]
        tx_mean = float(w @ tx)
        # re-solve the optimal-control problem over the knots not yet applied
        current = float(np.interp(t_now, np.arange(nk) * Dsg.KNOT, knots))
        rem, lib = continuations(current, target, t_now)
        cands = []
        for c in lib:
            x = knots.copy()
            x[rem] = c
            cands.append(x)
        obj = Dsg.objective(np.array(cands), target, prior, w)
        warm = cands[int(np.argmax(obj))]
        x_new, _ = Dsg.optimise(target, prior, starts=[knots, warm], maxiter=MAXITER, weights=w, free=rem,
                                fixed=knots)
        j_new, j_keep = Dsg.objective(np.array([x_new, knots]), target, prior, w)
        if j_new > j_keep + 1e-3:          # keep the current plan unless the new one is clearly better
            knots = x_new
        history.append(dict(t=float(t_now), ess=float(1.0 / np.sum(w ** 2)), shear_now=current,
                            tau_x_mean=tx_mean, tau_x_sd=float(np.sqrt(max(w @ (tx - tx_mean) ** 2, 0.0))),
                            observed_angle=float(obs_angle[-1]) if obs_angle else None,
                            knots=knots.tolist()))
        if verbose:
            print(f"t={t_now:4.1f} h  effective sets {history[-1]['ess']:.1f}  shear {current:.2f}", flush=True)
    closed = Dsg.evaluate(Dsg.path(knots, target)[None], target, plant[None])
    opened = Dsg.evaluate(Dsg.path(np.asarray(open_loop_knots), target)[None], target, plant[None])
    return dict(knots=knots, history=history,
                closed={k: float(v[0, 0]) for k, v in closed.items()},
                open={k: float(v[0, 0]) for k, v in opened.items()})

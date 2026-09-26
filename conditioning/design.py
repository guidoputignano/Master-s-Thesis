"""Design of a conditioning path for a target wall shear stress.

The path is shear over time, linear between knots every KNOT hours, starting from static culture.
It must end with `hold` hours at the target; the monolayer then stays at the target for POST hours
(the device in use), and readiness is judged at the end of that period:

    score = retention + connectivity + W_ORDER * (fraction ordered as the target demands)
            - W_DAMAGE * max(0, peak junction damage - D_MAX)

so a path that leaves the monolayer in the wrong ordered state (which would collapse after
implantation) or breaks it on the way scores low. Over an ensemble of parameter sets the design
maximises mean(score) - LAMBDA * sd(score) - MU_TV * (total variation of the path), the last term
a small penalty on changes of shear (as on input moves in predictive control) that removes
oscillations the score cannot tell apart; a path that only rises costs MU_TV * target whatever its
shape (up to the smoothing, 0.05 Pa per knot). Scores are reported without the penalty. Gradients
are finite differences in one batch.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from . import model as M

KNOT = 0.5          # h between knots (the decision interval)
POST = 24.0         # h at the target after conditioning
W_ORDER = 0.5
W_DAMAGE = 4.0
D_MAX = 0.2         # connectivity kept above 0.8 during conditioning
LAMBDA = 1.0
MU_TV = 0.005       # score per Pa of total variation


@dataclass(frozen=True)
class Target:
    tau: float                     # Pa, the shear the device will see
    surface: str = "silicone_flat"
    budget: float = 24.0           # h of conditioning
    hold: float = 2.0              # h at the target at the end of conditioning
    tau_max: float = 12.0          # Pa, pump limit
    tau_min: float = 0.5           # Pa, flow runs from the start (no gain from postponing conditioning)


def n_knots(target):
    return int(round((target.budget - target.hold) / KNOT)) + 1


def times(target):
    return np.arange(0.0, target.budget + POST + 1e-9, M.DT)


def path(knots, target):
    """Shear on the DT grid for knot values (knots[0] is the shear right after flow starts)."""
    knots = np.asarray(knots, float)
    t = times(target)
    tk = np.arange(len(knots)) * KNOT
    tau = np.interp(t, np.append(tk, target.budget - target.hold + 1e-6), np.append(knots, target.tau))
    tau[t >= target.budget - target.hold] = target.tau
    return tau


def variation(knots, target):
    """Total variation of the path in Pa, from static culture to the target (smoothed at 0.05 Pa)."""
    d = np.diff(np.concatenate([[0.0], np.asarray(knots, float), [target.tau]]))
    return float(np.sum(np.sqrt(d ** 2 + 0.05 ** 2) - 0.05))


def matched_fraction(out, target, theta_cols):
    """Fraction of domains ordered in the orientation the target prefers (per column)."""
    p = {k: theta_cols[i] for i, k in enumerate(M.NAMES)}
    pref = M.preference(np.full(theta_cols.shape[1], target.tau), p,
                        bias=M.SURFACES[target.surface].bias)
    return np.where(pref >= 0, out["A"][-1], out["C"][-1])


def evaluate(paths, target, thetas):
    """Scores for paths (n_paths, n_steps) under parameter sets thetas (n_sets, n_params).
    Returns dict of arrays (n_paths, n_sets)."""
    paths = np.atleast_2d(paths)
    thetas = np.atleast_2d(thetas)
    n_p, n_s = len(paths), len(thetas)
    shear = np.repeat(paths, n_s, axis=0).T                     # (steps, n_p * n_s)
    cols = np.tile(thetas, (n_p, 1)).T                          # (params, n_p * n_s)
    surf = [M.SURFACES[target.surface]] * (n_p * n_s)
    out = M.simulate(shear, cols, surf, record_every=1)
    k_end = len(times(target)) - 1
    k_cond = int(round(target.budget / M.DT))
    res = dict(
        retention=out["R"][k_end], ci=out["ci"][k_end], matched=matched_fraction(out, target, cols),
        peak_damage=out["D"][:k_cond + 1].max(axis=0), angle=out["angle"][k_end],
        ci_cond=out["ci"][k_cond], retention_cond=out["R"][k_cond], angle_cond=out["angle"][k_cond])
    res["score"] = (res["retention"] + res["ci"] + W_ORDER * res["matched"]
                    - W_DAMAGE * np.maximum(res["peak_damage"] - D_MAX, 0.0))
    return {k: v.reshape(n_p, n_s) for k, v in res.items()}


def robust(score):
    return score.mean(axis=1) - LAMBDA * score.std(axis=1)


def baselines(target):
    """Reference paths: direct step, the usual start (1 h at 1.4 Pa), a slow 8 h ramp and aligned
    preconditioning (8 h at 1.4 Pa) before the target."""
    nk = n_knots(target)
    tk = np.arange(nk) * KNOT
    tau = target.tau
    ref = {
        "direct": np.full(nk, tau),
        "usual start (1 h at 1.4 Pa)": np.where(tk < 1.0, 1.4, tau),
        "slow ramp (8 h)": np.minimum(tk / 8.0, 1.0) * tau,
        "aligned first (8 h at 1.4 Pa)": np.where(tk < 8.0, 1.4, tau),
    }
    return {k: np.clip(v, 0.0, target.tau_max) for k, v in ref.items()}


def optimise(target, thetas, starts=None, maxiter=60, fd=0.05, verbose=False):
    """Maximise the robust score over knot values in [tau_min, tau_max]."""
    thetas = np.atleast_2d(thetas)
    nk = n_knots(target)
    starts = list(baselines(target).values()) if starts is None else starts

    def f_and_grad(x):
        X = np.vstack([x] + [x + fd * np.eye(nk)[i] for i in range(nk)])
        X = np.clip(X, target.tau_min, target.tau_max)
        sc = robust(evaluate(np.array([path(k, target) for k in X]), target, thetas)["score"])
        sc = sc - MU_TV * np.array([variation(k, target) for k in X])
        return -sc[0], -(sc[1:] - sc[0]) / fd

    best = None
    for x0 in starts:
        res = minimize(f_and_grad, np.asarray(x0, float), jac=True, method="L-BFGS-B",
                       bounds=[(target.tau_min, target.tau_max)] * nk, options=dict(maxiter=maxiter))
        if verbose:
            print(f"  start -> robust score {-res.fun:.3f}", flush=True)
        if best is None or res.fun < best.fun:
            best = res
    x = np.clip(best.x, target.tau_min, target.tau_max)
    return x, float(robust(evaluate(path(x, target)[None], target, thetas)["score"])[0])


def two_level(level, switch, target):
    """Knots of the path that holds `level` until `switch` hours, then the target."""
    tk = np.arange(n_knots(target)) * KNOT
    return np.where(tk < switch - 1e-9, level, target.tau)


def simplify(x, target, thetas, tol=0.01, step=0.1):
    """The simplest path within tol of the optimum x: the best two-level path (a level from the
    start, then the target) if it scores within tol, else x. Returns (knots, robust score, simple)."""
    s_opt = float(robust(evaluate(path(x, target)[None], target, thetas)["score"])[0])
    levels = np.arange(target.tau_min, target.tau_max + 1e-9, step)
    switches = np.arange(0.0, target.budget - target.hold + 1e-9, KNOT)
    cands = [two_level(L, T, target) for T in switches for L in (levels if T > 0 else [target.tau])]
    sc = np.concatenate([robust(evaluate(np.array([path(k, target) for k in cands[i:i + 200]]), target, thetas)["score"])
                         for i in range(0, len(cands), 200)])
    i = int(np.argmax(sc))
    if sc[i] >= s_opt - tol:
        return cands[i], float(sc[i]), True
    return np.asarray(x, float), s_opt, False


def compare(target, thetas, designed=None):
    """Designed path against the baselines: robust score and mean outcomes."""
    ref = baselines(target)
    if designed is not None:
        ref = {"designed": designed, **ref}
    names = list(ref)
    ev = evaluate(np.array([path(ref[n], target) for n in names]), target, thetas)
    rows = []
    for i, n in enumerate(names):
        rows.append(dict(path=n, robust=float(robust(ev["score"][i:i + 1])[0]),
                         **{k: float(ev[k][i].mean()) for k in ("retention", "ci", "matched", "peak_damage",
                                                                "angle", "ci_cond", "retention_cond")},
                         retention_p10=float(np.percentile(ev["retention"][i], 10)),
                         ci_p10=float(np.percentile(ev["ci"][i], 10))))
    return rows

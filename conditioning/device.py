"""One pump, many local shears: conditioning a device in its own flow.

An HVAD inflow cannula at its operating point (Ghodrati et al., Artif Organs 2020, two patients):
18-33 % of its surface below 0.3 Pa, 59-72 % between 0.3 and 9 Pa, 8-10 % above 9 Pa at the tip.
Only these three band areas are reported. We take the means of the two patients (25.5, 65.5 and
9 %), place the lowest band at 0.15 Pa, split the middle band equally over six levels of our choice
(0.5, 1, 2, 3.5, 5.5 and 8 Pa) and put the tip at 11 Pa, beyond the calibrated range (flagged). We
assume that when the device is conditioned in a mock loop every region's shear scales linearly with
the pump flow: tau_i(t) = tau_i(operating) * u(t), with u the flow fraction.

Strategies compared, all ending with the device at its operating flow for POST hours:
- in-device, designed: u(t) chosen by the designer for the area-weighted score;
- in-device, direct: operating flow from the start;
- in-device, slow ramp: u from 0 to 1 over 8 h;
- chamber at 1.4 Pa: uniform conditioning in a chamber, then implantation (each region jumps to its shear).

A region counts as connected when its junction connectivity is at least 0.8 at the end of the 24 h
at operating flow; cells retained and the fraction ordered as the local shear demands are reported
alongside (retention after a day at 8 Pa or more is low whatever the path, in the model).
"""
import numpy as np
from scipy.optimize import minimize

from . import design as Dsg
from . import model as M

REGIONS = np.array([0.15, 0.5, 1.0, 2.0, 3.5, 5.5, 8.0, 11.0])           # Pa at the operating point
WEIGHTS = np.array([0.255] + [0.655 / 6] * 6 + [0.09])
EXTRAPOLATED = REGIONS > 10.0
U_MIN = 0.1        # flow runs from the start


def region_paths(u, tau_op=REGIONS):
    """Local shear (n_steps, n_regions) for a flow-fraction path u (n_steps,)."""
    return np.outer(u, tau_op)


def evaluate_device(u_paths, thetas, surfaces, budget=24.0, hold=2.0, tau_op=REGIONS, weights=WEIGHTS):
    """u_paths: (n_paths, n_steps) flow fractions; surfaces: one surface name per region.
    Returns per path: area-weighted means over regions and parameter sets, and per-region means."""
    thetas = np.atleast_2d(thetas)
    u_paths = np.atleast_2d(u_paths)
    n_p, n_s, n_r = len(u_paths), len(thetas), len(tau_op)
    # columns: path-major, then region, then parameter set
    shear = np.concatenate([np.repeat(region_paths(u, tau_op), n_s, axis=1) for u in u_paths], axis=1)
    cols = np.tile(thetas.T, (1, n_r * n_p))
    surf = [M.SURFACES[s] for s in surfaces for _ in range(n_s)] * n_p
    out = M.simulate(shear, cols, surf)
    k_cond = int(round(budget / M.DT))
    p = {k: cols[i] for i, k in enumerate(M.NAMES)}
    tau_end = np.tile(np.repeat(tau_op, n_s), n_p)
    bias = np.array([s.bias for s in surf])
    pref = M.preference(tau_end, p, bias=bias)
    matched = np.where(pref >= 0, out["A"][-1], out["C"][-1])
    res = dict(retention=out["R"][-1], ci=out["ci"][-1], matched=matched,
               peak_damage=out["D"][:k_cond + 1].max(axis=0))
    res["connected"] = (res["ci"] >= 0.8).astype(float)
    res["score"] = (res["retention"] + res["ci"] + Dsg.W_ORDER * res["matched"]
                    - Dsg.W_DAMAGE * np.maximum(res["peak_damage"] - Dsg.D_MAX, 0.0))
    out_rows = []
    for k in range(n_p):
        sl = slice(k * n_r * n_s, (k + 1) * n_r * n_s)
        per_region = {q: res[q][sl].reshape(n_r, n_s).mean(axis=1) for q in res}
        area = {q: float(np.sum(weights * per_region[q])) for q in res}
        score_sets = np.sum(weights[:, None] * res["score"][sl].reshape(n_r, n_s), axis=0)
        area["robust"] = float(score_sets.mean() - Dsg.LAMBDA * score_sets.std())
        out_rows.append(dict(area=area, per_region=per_region))
    return out_rows


def u_path(knots, budget=24.0, hold=2.0):
    t = np.arange(0.0, budget + Dsg.POST + 1e-9, M.DT)
    tk = np.arange(len(knots)) * Dsg.KNOT
    u = np.interp(t, np.append(tk, budget - hold + 1e-6), np.append(knots, 1.0))
    u[t >= budget - hold] = 1.0
    return u


MU_U = 0.02        # score per unit of total variation of the flow fraction (a separate choice from
                   # design.MU_TV, which is per Pa)


def optimise_device(thetas, surfaces, budget=24.0, hold=2.0, maxiter=40, fd=0.02, starts=None):
    nk = int(round((budget - hold) / Dsg.KNOT)) + 1
    tk = np.arange(nk) * Dsg.KNOT
    starts = starts or [np.ones(nk), np.minimum(tk / 8.0, 1.0), np.full(nk, 0.5)]

    def f_and_grad(x):
        X = np.clip(np.vstack([x] + [x + fd * np.eye(nk)[i] for i in range(nk)]), U_MIN, 1.2)
        rows = evaluate_device(np.array([u_path(k, budget, hold) for k in X]), thetas, surfaces, budget, hold)
        sc = np.array([r["area"]["robust"] for r in rows])
        tv = np.array([np.sum(np.sqrt(np.diff(np.concatenate([[0.0], k, [1.0]])) ** 2 + 0.01 ** 2) - 0.01)
                       for k in X])
        sc = sc - MU_U * tv
        return -sc[0], -(sc[1:] - sc[0]) / fd

    best = None
    for x0 in starts:
        res = minimize(f_and_grad, np.clip(x0, U_MIN, 1.2), jac=True, method="L-BFGS-B",
                       bounds=[(U_MIN, 1.2)] * nk, options=dict(maxiter=maxiter))
        if best is None or res.fun < best.fun:
            best = res
    x = np.clip(best.x, U_MIN, 1.2)
    return x, evaluate_device(u_path(x, budget, hold)[None], thetas, surfaces, budget, hold)[0]["area"]["robust"]


def simplify_device(x, thetas, surfaces, budget=24.0, hold=2.0, tol=0.01, step=0.05):
    """The operating flow from the start if it scores within tol of x, else the best two-level
    flow path (a fraction from the start, then operating flow) if it does, else x.
    Returns (knots, robust score, simple)."""
    nk = len(x)
    tk = np.arange(nk) * Dsg.KNOT
    s_opt = evaluate_device(u_path(x, budget, hold)[None], thetas, surfaces, budget, hold)[0]["area"]["robust"]
    cands = [np.ones(nk)] + [np.where(tk < T - 1e-9, lv, 1.0)
                             for T in np.arange(Dsg.KNOT, budget - hold + 1e-9, Dsg.KNOT)
                             for lv in np.arange(U_MIN, 1.2 + 1e-9, step)]
    sc = []
    for i in range(0, len(cands), 40):
        rows = evaluate_device(np.array([u_path(k, budget, hold) for k in cands[i:i + 40]]), thetas, surfaces,
                               budget, hold)
        sc += [r["area"]["robust"] for r in rows]
    if sc[0] >= s_opt - tol:                                    # cands[0]: operating flow at once
        return cands[0], float(sc[0]), True
    i = int(np.argmax(sc))
    if sc[i] >= s_opt - tol:
        return cands[i], float(sc[i]), True
    return np.asarray(x, float), float(s_opt), False


def chamber_then_implant(thetas, surfaces, tau_c=1.4, budget=24.0, tau_op=REGIONS, weights=WEIGHTS):
    """Uniform conditioning at tau_c in a chamber, then each region jumps to its operating shear."""
    thetas = np.atleast_2d(thetas)
    t = np.arange(0.0, budget + Dsg.POST + 1e-9, M.DT)
    n_s, n_r = len(thetas), len(tau_op)
    shear = np.where(t[:, None] < budget, tau_c, tau_op[None, :])
    shear = np.repeat(shear, n_s, axis=1)
    cols = np.tile(thetas.T, (1, n_r))
    surf = [M.SURFACES[s] for s in surfaces for _ in range(n_s)]
    out = M.simulate(shear, cols, surf)
    p = {k: cols[i] for i, k in enumerate(M.NAMES)}
    pref = M.preference(np.repeat(tau_op, n_s), p, bias=np.array([s.bias for s in surf]))
    matched = np.where(pref >= 0, out["A"][-1], out["C"][-1])
    k_cond = int(round(budget / M.DT))
    res = dict(retention=out["R"][-1], ci=out["ci"][-1], matched=matched,
               peak_damage=out["D"][:k_cond + 1].max(axis=0))
    res["connected"] = (res["ci"] >= 0.8).astype(float)
    per_region = {q: v.reshape(n_r, n_s).mean(axis=1) for q, v in res.items()}
    area = {q: float(np.sum(weights * per_region[q])) for q in res}
    return dict(area=area, per_region=per_region)

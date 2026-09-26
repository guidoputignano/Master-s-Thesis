"""Monolayer-state model for shear conditioning.

State (fractions of the monolayer, then three scalars)
-------------------------------------------------------
N  native isotropic domains (as grown in static culture; junctions intact)
A  domains ordered along the flow
C  domains ordered across the flow
X  frustrated disordered domains, left behind when an ordered domain is forced to the other
   orientation (N + A + C + X = 1)
J  junction destabilisation, 0-1: raised by changes of shear, relaxing in about an hour;
   blocked by the Src inhibitor PP1
D  junction damage, 0-1 (connectivity index = 1 - D)
R  cells retained, relative to the start

Mechanisms, each tied to an observation of the group's chamber
---------------------------------------------------------------
- Shear sets a preferred orientation: along the flow below a switch shear tau_x, across above it
  (Stefopoulos 2022 Fig. 1). Topography adds a bias along its grooves (Robotti 2014).
- Native domains order toward the preferred orientation; ordered neighbours recruit them.
- An ordered domain facing the opposite preference collapses into frustrated disorder. Collapse is
  fast only when junctions are destabilised (abrupt change: Fig. 4); with PP1 or after a gradual
  change the order persists and decays slowly by spreading disorder (Fig. 4k,l PP1; Fig. S5).
- Frustrated domains re-order only slowly (Fig. 4h: still random 6 h after the rise; Fig. S5).
- Collapse through destabilised junctions damages them (abrupt changes: Stefopoulos 2022 Fig. 4,
  Wu 2021 Fig. 5); the slow spontaneous route does not. Gratings along the flow turned cells away
  at 4-8 Pa without breaking junctions, yet broke at 10 Pa, while gratings across the flow held at
  10 Pa (Robotti 2014): damage follows mismatch, not shear alone. Above a limit that depends on the
  material and rises on gratings, shear breaks the junctions of domains not ordered as it demands
  (native, frustrated or opposed), in proportion to the excess; domains ordered as the shear demands
  hold (perpendicular gratings at 10 Pa). Gratings anchor cells: they reduce both the damage and
  the cell loss that a collapse causes. An earlier version, whose limit acted on every domain,
  failed that held-out test. Damage heals where domains are not frustrated.
- Cells are lost while ordered domains collapse (Stefopoulos 2022 Video S5; Wu 2021 Fig. 5), where
  junctions are damaged in proportion to the shear, and above a shear that strips cells even from
  connected monolayers (Robotti 2014 gratings).
- The across-flow response is collective (isolated cells do not show it): where junctions are
  damaged, cells revert to aligning along the flow or the grooves (Robotti 2014, 10 Pa on gratings).

Integration: exponential (conservative) steps of 5 min for many protocols at once.
"""
from dataclasses import dataclass, field

import numpy as np

DT = 1.0 / 12.0            # h
TAU_ACT = 0.5              # Pa, activation of the shear response (fixed)
N_SINGLE = 4.0             # single-cell reversion ~ damage**4: only broken junctions revert (fixed)
M_DRAG = 4.0               # steepness of the drag factor on collapse damage (fixed)
TAU_REF = 8.0              # Pa, reference shear for detachment
K_RELAX_STATIC = 0.02      # 1/h, loss of order without shear (fixed, no data)

# name: (initial value, lower, upper, description)
PARAMS = {
    "tau_x":      (3.0, 2.0, 6.0, "switch shear between along- and across-flow preference (Pa)"),
    "w_x":        (0.8, 0.3, 1.5, "width of the switch (Pa)"),
    "k_ord":      (0.35, 0.03, 2.0, "ordering rate of native domains (1/h)"),
    "rho_c":      (1.5, 0.5, 6.0, "across-flow ordering relative to along-flow"),
    "kappa":      (1.0, 0.0, 6.0, "recruitment by ordered neighbours"),
    "beta_j":     (1.0, 0.0, 6.0, "speed-up of ordering by junction destabilisation"),
    "k_x":        (0.01, 0.0, 0.3, "re-ordering rate of frustrated domains (1/h)"),
    "k_col":      (1.5, 0.05, 12.0, "collapse rate of opposed ordered domains at full destabilisation (1/h)"),
    "j50":        (0.4, 0.1, 0.97, "destabilisation at half the maximal collapse rate"),
    "q_hill":     (8.0, 2.0, 16.0, "steepness of collapse in the destabilisation"),
    "t_p":        (1.0, 0.05, 4.0, "priming time before across-flow ordering (h; polarity first, Fig. 1i)"),
    "tau_d":      (8.0, 3.0, 25.0, "shear at which collapse damage doubles (Pa)"),
    "k_sp":       (0.1, 0.0, 2.0, "spontaneous spreading of disorder in opposed domains (1/h)"),
    "lg_seed":    (-2.0, -5.0, -0.5, "log10 of the seed from which disorder spreads"),
    "tau_j":      (6.0, 0.5, 20.0, "shear change that destabilises junctions by 1 - 1/e (Pa)"),
    "t_j":        (1.0, 0.2, 4.0, "relaxation time of destabilisation (h)"),
    "theta_a":    (18.0, 12.0, 24.0, "mean angle within along-ordered domains (deg)"),
    "theta_c":    (84.0, 78.0, 88.0, "mean angle within across-ordered domains (deg)"),
    "ar_n":       (1.85, 1.75, 1.95, "aspect ratio of native domains"),
    "ar_a":       (3.05, 2.6, 3.4, "aspect ratio of along-ordered domains"),
    "ar_c":       (3.5, 3.0, 3.9, "aspect ratio of across-ordered domains"),
    "ar_x":       (2.05, 1.8, 2.4, "aspect ratio of frustrated domains"),
    "delta_x":    (2.0, 0.0, 12.0, "junction damage per unit of order collapsing through destabilised junctions"),
    "k_heal":     (0.05, 0.0, 0.6, "healing of junction damage (1/h)"),
    "k_lim":      (0.3, 0.0, 4.0, "damage rate per 8 Pa above the limit, on mismatched domains (1/h)"),
    "lim":        (8.0, 4.0, 14.0, "shear above which mismatched domains lose junctions, silicone (Pa)"),
    "lim_coc":    (5.0, 2.5, 9.0, "the same limit on COC (Pa)"),
    "d_grat":     (3.0, 0.0, 8.0, "rise of the limit on gratings (Pa)"),
    "pi_grat":    (0.5, 0.03, 1.0, "damage on gratings relative to flat"),
    "b_topo":     (0.6, 0.0, 1.5, "orientation bias of gratings"),
    "a_grat":     (0.85, 0.6, 1.0, "static fraction ordered by gratings"),
    "f_coc":      (0.3, 0.03, 1.5, "ordering speed on COC relative to silicone"),
    "k_rx":       (0.3, 0.0, 3.0, "cells lost per unit of order collapsing through destabilised junctions"),
    "k_rd":       (0.1, 0.0, 2.0, "cell loss rate from damaged junctions at 8 Pa (1/h)"),
    "k_rt":       (0.03, 0.0, 1.0, "cell loss rate per 8 Pa above tau_r (1/h)"),
    "tau_r":      (5.0, 1.5, 10.0, "shear above which cells are stripped from connected monolayers (Pa)"),
}
NAMES = list(PARAMS)
X0 = np.array([PARAMS[n][0] for n in NAMES])
LO = np.array([PARAMS[n][1] for n in NAMES])
HI = np.array([PARAMS[n][2] for n in NAMES])


@dataclass(frozen=True)
class Surface:
    """How a substrate enters the model."""
    bias: float = 0.0          # +1 gratings along the flow, -1 across, 0 none (times b_topo)
    preorder: float = 0.0      # +1 static order along the flow, -1 across, 0 none (times a_grat)
    order: float = 1.0         # order factor (1 flat; 0.58 breath figures, from Wu 2021 Fig. 4e)
    protect: str = "flat"      # "flat" or "grat" (damage times pi_grat)
    coc: bool = False          # COC substrate: slower ordering of long-confluent monolayers, own limit


SURFACES = {
    "silicone_flat": Surface(),
    "silicone_bf": Surface(order=0.58),
    "coc_flat": Surface(coc=True),
    "coc_par": Surface(bias=1.0, preorder=1.0, protect="grat", coc=True),
    "coc_perp": Surface(bias=-1.0, preorder=-1.0, protect="grat", coc=True),
    # extrapolations used for design only: gratings on silicone (kinetics of silicone, topography
    # of the COC gratings)
    "silicone_par": Surface(bias=1.0, preorder=1.0, protect="grat"),
    "silicone_perp": Surface(bias=-1.0, preorder=-1.0, protect="grat"),
}


def as_dict(theta):
    return dict(zip(NAMES, np.asarray(theta, float)))


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def preference(tau, p, bias=0.0, damage=0.0):
    """Preferred orientation: >0 along the flow, <0 across.

    The across-flow response is collective: isolated endothelial cells do not show it
    (Stefopoulos 2022), so where junctions are damaged cells revert to the single-cell
    preference along the flow."""
    act = tau ** 2 / (tau ** 2 + TAU_ACT ** 2)
    collective = act * np.tanh((p["tau_x"] - tau) / p["w_x"])
    single = damage ** N_SINGLE
    return (1.0 - single) * collective + single * act + bias * p["b_topo"]


KERNEL_ORDER = ("tau_x", "w_x", "k_ord", "rho_c", "kappa", "beta_j", "k_x", "k_col", "j50", "q_hill",
                "t_p", "tau_d", "k_sp", "lg_seed", "tau_j", "t_j", "theta_a", "theta_c", "delta_x",
                "k_heal", "k_lim", "lim", "lim_coc", "d_grat", "pi_grat", "b_topo", "a_grat", "f_coc",
                "k_rx", "k_rd", "k_rt", "tau_r")
USE_KERNEL = True


def _domain_angles(p, f, grat):
    """Mean angle within along- and across-ordered domains; on gratings the spread about the
    grooves is the same whichever way they run, so across-domains mirror along-domains."""
    th_a = 45.0 - f * (45.0 - p["theta_a"])
    th_c = np.where(grat, 90.0 - th_a, 45.0 + f * (p["theta_c"] - 45.0))
    return th_a, th_c


def _observables(rec, p, f, grat):
    out = dict(zip(("N", "A", "C", "X", "J", "D", "R"), rec))
    th_a, th_c = _domain_angles(p, f, grat)
    ar_a = p["ar_n"] + f * (p["ar_a"] - p["ar_n"])
    ar_c = p["ar_n"] + f * (p["ar_c"] - p["ar_n"])
    ar_x = p["ar_n"] + f * (p["ar_x"] - p["ar_n"])
    out["angle"] = out["A"] * th_a + out["C"] * th_c + (out["N"] + out["X"]) * 45.0
    out["ar"] = out["A"] * ar_a + out["C"] * ar_c + out["N"] * p["ar_n"] + out["X"] * ar_x
    out["ci"] = 1.0 - out["D"]
    out["retention"] = out["R"]
    out["s"] = out["A"] * np.cos(2 * np.deg2rad(th_a)) + out["C"] * np.cos(2 * np.deg2rad(th_c))
    return out


def simulate(shear, theta, surfaces, pp1_mask=None, record_every=1, extra_loss=None):
    """Integrate many protocols at once.

    shear      array (n_steps, n_protocols) of wall shear stress (Pa), sampled every DT
    theta      parameter vector (NAMES order), dict, or array (n_params, n_protocols) with one
               parameter set per protocol column
    surfaces   list of Surface, one per protocol
    pp1_mask   optional bool array (n_steps, n_protocols): PP1 present
    returns    dict of arrays (n_records, n_protocols): N, A, C, X, J, D, R, angle, ar, ci, s
    """
    shear = np.asarray(shear, float)
    n_steps, n = shear.shape
    if isinstance(theta, dict):
        p = theta
    else:
        th = np.asarray(theta, float)
        p = as_dict(th) if th.ndim == 1 else {k: th[i] for i, k in enumerate(NAMES)}
    p = {k: np.broadcast_to(np.asarray(v, float), (n,)) for k, v in p.items()}
    bias = np.array([s.bias for s in surfaces])
    f = np.array([s.order for s in surfaces])
    idx = np.arange(n)
    prot = np.where([s.protect == "grat" for s in surfaces], p["pi_grat"], 1.0)
    kfac = np.where([s.coc for s in surfaces], p["f_coc"], 1.0)
    grat = np.array([s.protect == "grat" for s in surfaces])
    lim = (np.where([s.coc for s in surfaces], p["lim_coc"], p["lim"])
           + np.where([s.protect == "grat" for s in surfaces], p["d_grat"], 0.0))
    pre = np.array([s.preorder for s in surfaces]) * p["a_grat"]
    A = np.clip(pre, 0, 1)
    C = np.clip(-pre, 0, 1)
    X = np.zeros(n)
    Nn = 1.0 - A - C
    J = np.zeros(n)
    D = np.zeros(n)
    R = np.ones(n)
    from . import _kernel
    if USE_KERNEL and _kernel.kernel is not None and extra_loss is None:
        Pm = np.array([p[k] for k in NAMES])
        idx = tuple(NAMES.index(k) for k in KERNEL_ORDER)
        pp1 = np.zeros(shear.shape, bool) if pp1_mask is None else np.asarray(pp1_mask, bool)
        rec = _kernel.kernel(np.ascontiguousarray(shear), Pm, idx, bias.astype(float), f.astype(float),
                             np.array([s.protect == "grat" for s in surfaces]),
                             np.array([s.coc for s in surfaces]), np.array([s.preorder for s in surfaces], float),
                             pp1, DT, TAU_ACT, N_SINGLE, M_DRAG, TAU_REF, K_RELAX_STATIC)
        if record_every != 1:
            rec = rec[:, ::record_every]
        return _observables(rec, p, f, grat)
    th_a, th_c = _domain_angles(p, f, grat)
    ar_a = p["ar_n"] + f * (p["ar_a"] - p["ar_n"])
    ar_c = p["ar_n"] + f * (p["ar_c"] - p["ar_n"])
    ar_x = p["ar_n"] + f * (p["ar_x"] - p["ar_n"])
    s_a = np.cos(2 * np.deg2rad(th_a))       # order of an along domain (angle observable proxy)
    s_c = np.cos(2 * np.deg2rad(th_c))
    qh = p["q_hill"]
    q50 = p["j50"] ** qh
    prime = np.zeros(n)
    prev = np.zeros(n)
    rec = {k: [] for k in ("N", "A", "C", "X", "J", "D", "R")}
    for k in range(n_steps):
        tau = shear[k]
        # destabilisation: exact step response to the change of shear, then relaxation
        dtau = np.abs(tau - prev)
        J = 1.0 - (1.0 - J) * np.exp(-dtau / p["tau_j"])
        J = J * np.exp(-DT / p["t_j"])
        if pp1_mask is not None:
            J = np.where(pp1_mask[k], 0.0, J)
        prev = tau
        pref = preference(tau, p, bias, D)
        pp, pm = np.maximum(pref, 0.0), np.maximum(-pref, 0.0)
        hj = J ** qh / (J ** qh + q50)
        prime = prime + (pm - prime) * (1.0 - np.exp(-DT / p["t_p"]))
        boost = (1.0 + p["beta_j"] * J) * (1.0 - D) * kfac
        r_na = p["k_ord"] * pp * (1.0 + p["kappa"] * A) * boost
        r_nc = p["k_ord"] * p["rho_c"] * np.minimum(pm, prime) * (1.0 + p["kappa"] * C) * boost
        r_xa = p["k_x"] * pp * (1.0 + p["kappa"] * A) * kfac
        r_xc = p["k_x"] * pm * (1.0 + p["kappa"] * C) * kfac
        spread = p["k_sp"] * (10.0 ** p["lg_seed"] + X)
        r_ax = pm * (p["k_col"] * hj + spread)
        r_cx = pp * (p["k_col"] * hj + spread)
        w_hj = np.where(p["k_col"] * hj + spread > 0, p["k_col"] * hj / np.maximum(p["k_col"] * hj + spread, 1e-12), 0.0)
        act = tau ** 2 / (tau ** 2 + TAU_ACT ** 2)
        r_rel = K_RELAX_STATIC * (1.0 - act)
        # conservative exponential transfers out of each compartment
        out_n = r_na + r_nc
        fn = 1.0 - np.exp(-out_n * DT)
        with np.errstate(invalid="ignore", divide="ignore"):
            to_a_n = np.where(out_n > 0, Nn * fn * r_na / out_n, 0.0)
            to_c_n = np.where(out_n > 0, Nn * fn * r_nc / out_n, 0.0)
        out_a = r_ax + r_rel
        fa = 1.0 - np.exp(-out_a * DT)
        a_to_x = A * fa * np.where(out_a > 0, r_ax / np.maximum(out_a, 1e-12), 0.0)
        a_to_n = A * fa - a_to_x
        out_c = r_cx + r_rel
        fc = 1.0 - np.exp(-out_c * DT)
        c_to_x = C * fc * np.where(out_c > 0, r_cx / np.maximum(out_c, 1e-12), 0.0)
        c_to_n = C * fc - c_to_x
        out_x = r_xa + r_xc
        fx = 1.0 - np.exp(-out_x * DT)
        with np.errstate(invalid="ignore", divide="ignore"):
            x_to_a = np.where(out_x > 0, X * fx * r_xa / out_x, 0.0)
            x_to_c = np.where(out_x > 0, X * fx * r_xc / out_x, 0.0)
        A = A - a_to_x - a_to_n + to_a_n + x_to_a
        C = C - c_to_x - c_to_n + to_c_n + x_to_c
        X = X + a_to_x + c_to_x - x_to_a - x_to_c
        Nn = np.clip(1.0 - A - C - X, 0.0, 1.0)
        # junction damage: collapsing order, and shear above the surface's limit
        # order lost per hour through destabilised junctions (the damaging route)
        collapse = w_hj * (a_to_x * np.abs(s_a) + c_to_x * np.abs(s_c)) / DT
        drag = 1.0 + (tau / p["tau_d"]) ** M_DRAG
        mismatched = Nn + X + np.where(pref >= 0, C, A)
        over = np.maximum(tau - lim, 0.0) / TAU_REF
        src = prot * p["delta_x"] * f * collapse * drag + p["k_lim"] * over * mismatched
        D = 1.0 - (1.0 - D) * np.exp(-src * DT)
        D = D * np.exp(-p["k_heal"] * (1.0 - X) * DT)
        loss = (prot * p["k_rx"] * collapse + p["k_rd"] * D * tau / TAU_REF
                + p["k_rt"] * np.maximum(tau - p["tau_r"], 0.0) / TAU_REF)
        if extra_loss is not None:
            loss = loss + extra_loss
        R = R * np.exp(-loss * DT)
        if k % record_every == 0:
            for key, v in (("N", Nn), ("A", A), ("C", C), ("X", X), ("J", J), ("D", D), ("R", R)):
                rec[key].append(v.copy())
    return _observables(np.array([rec[k] for k in ("N", "A", "C", "X", "J", "D", "R")]), p, f, grat)


def shear_array(protocols, duration=None):
    """Sample protocols on the DT grid: returns times (h) and shear (n_steps, n)."""
    T = duration or max(p.duration for p in protocols)
    t = np.arange(0.0, T + 1e-9, DT)
    return t, np.stack([p.shear(t) for p in protocols], axis=1)


def run_protocols(protocols, theta, duration=None):
    t, sh = shear_array(protocols, duration)
    pp1 = None
    if any(pr.pp1_from is not None for pr in protocols):
        pp1 = np.stack([(t >= pr.pp1_from) if pr.pp1_from is not None else np.zeros_like(t, bool)
                        for pr in protocols], axis=1)
    out = simulate(sh, theta, [SURFACES[pr.surface] for pr in protocols], pp1)
    out["t"] = t
    out["shear"] = sh
    return out

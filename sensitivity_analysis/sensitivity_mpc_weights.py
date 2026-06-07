"""
Part D - Sensitivity of the MPC cost weights (w_phi : w_rho : w_varphi : w_u).

The receding-horizon controller (main.tex, eq:ocp / eq:stagecost) chooses the
wall-shear protocol tau(t) by minimising

    J = w_phi    * sum_k phi_sen(k)^2
      + w_rho    * sum_k (rho_bar(k)   - 2.3)^2
      + w_varphi * sum_k (varphi_bar(k) - 0.0)^2
      + w_u      * sum_k (tau(k) - tau(k-1))^2

subject to phi_sen(k) <= 0.30 and 0 <= tau(k) <= 2 Pa. The nominal weights are
10 / 1 / 5 / 0.1. Unlike the morphological energy weights (Part C), these enter
the control law directly, so this part quantifies how the *protocol* and the
*closed-loop outcome* depend on them.

For each weight set the receding-horizon loop is run on the reduced prediction
model only (no tessellation rendering): population compartments via solve_ivp
(eq:reduced) and morphology via the closed-form step response (eq:stepsolution).
Recorded outcomes after the conditioning horizon:
    align_final   healthy-cell flow alignment (deg, target 20)
    rho_final     population-mean aspect ratio (target 2.3)
    phisen_final  senescent fraction at the end
    phisen_max    peak senescent fraction (constraint margin to 0.30)
    tau_mean      mean applied shear (Pa)
    tau_tv        total input variation  sum |tau_k - tau_{k-1}|  (smoothness)

Outputs (figures/):
    mpc_weights_outcomes.pdf     outcomes across the swept weight sets
    mpc_weights_sensitivity.pdf  normalised sensitivity indices (+10 %)
and a printed summary table.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from endothelial_simulation.config import SimulationConfig
from endothelial_simulation.control.mpc_controller import (
    RecedingHorizonMPC, flow_alignment_angle, RHO_STAT, THETA_STAT_DEG)

np.random.seed(42)

CM = 1.0 / 2.54
SINGLE = 8.5 * CM
DOUBLE = 17.5 * CM
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

# Nominal MPC cost weights (Source: project spec / main.tex eq:stagecost)
NOMINAL = {"w_phi": 10.0, "w_rho": 1.0, "w_varphi": 5.0, "w_u": 0.1}
N_STEPS = 24            # 24 h conditioning (matches run_mpc.py)


def set_style():
    plt.style.use("default")
    plt.rcParams.update({
        "axes.facecolor": "white", "figure.facecolor": "white",
        "savefig.facecolor": "white", "axes.grid": False,
        "axes.edgecolor": "black", "font.size": 11, "axes.labelsize": 11,
        "axes.titlesize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "axes.linewidth": 0.8, "lines.linewidth": 1.3,
        "savefig.dpi": 600, "pdf.fonttype": 42, "font.family": "sans-serif",
    })


def _initial_state(cfg, N):
    """
    Build the reduced initial state for the conditioning protocol:
    confluent monolayer with phi_sen(0)=0.20 (70/30 stress/telomere), healthy
    cells spread across early division stages, morphology at the static baseline.
    """
    n_total = cfg.initial_cell_count
    phi0 = getattr(cfg, "initial_senescent_fraction", 0.20)
    n_sen = int(round(phi0 * n_total))
    n_str = int(round(n_sen * getattr(cfg, "senescent_stress_fraction", 0.70)))
    n_tel = n_sen - n_str
    n_healthy = n_total - n_sen

    stages = np.arange(N + 1)
    w = np.exp(-0.3 * stages)
    w /= w.sum()
    E = n_healthy * w
    pop = np.concatenate([E, [float(n_tel), float(n_str)]])
    return {"pop": pop, "rho_h": RHO_STAT, "theta_h": np.radians(THETA_STAT_DEG)}


def run_protocol(cfg, weights, n_steps=N_STEPS):
    """Run the receding-horizon MPC on the reduced model; return outcome metrics."""
    mpc = RecedingHorizonMPC(cfg, **weights)
    x = _initial_state(cfg, mpc.N)
    u_prev = 0.0
    taus, phis = [], []
    for _ in range(n_steps):
        u_opt, _ = mpc.solve(x, u_prev)
        tau = float(np.clip(u_opt[0], mpc.tau_min, mpc.tau_max))
        x = mpc.predict_step(x, tau)
        phi_sen, _, _ = mpc.outputs(x)
        taus.append(tau); phis.append(phi_sen)
        u_prev = tau

    phi_sen, rho_bar, _ = mpc.outputs(x)
    taus = np.asarray(taus)
    return {
        "align_final": float(np.degrees(flow_alignment_angle(x["theta_h"]))),
        "rho_final": float(rho_bar),
        "phisen_final": float(phi_sen),
        "phisen_max": float(max(phis)),
        "tau_mean": float(taus.mean()),
        "tau_tv": float(np.abs(np.diff(np.concatenate([[0.0], taus]))).sum()),
    }


# Weight sets: nominal plus factor-of-2 perturbations of each cost weight
WEIGHT_SETS = {
    "nominal":      dict(NOMINAL),
    "w_phi x0.5":   {**NOMINAL, "w_phi": 5.0},
    "w_phi x2":     {**NOMINAL, "w_phi": 20.0},
    "w_rho x0.5":   {**NOMINAL, "w_rho": 0.5},
    "w_rho x2":     {**NOMINAL, "w_rho": 2.0},
    "w_varphi x0.5": {**NOMINAL, "w_varphi": 2.5},
    "w_varphi x2":  {**NOMINAL, "w_varphi": 10.0},
    "w_u x0.5":     {**NOMINAL, "w_u": 0.05},
    "w_u x2":       {**NOMINAL, "w_u": 0.2},
}

# outcomes used for the normalised sensitivity indices
_SENS_OUTPUTS = ("align_final", "phisen_final", "tau_mean")


def normalised_sensitivity(cfg, perturb=0.10):
    """S = (dQ/Q)/(dp/p) at nominal, +perturb on each cost weight."""
    base = run_protocol(cfg, NOMINAL)
    out = {}
    for key in NOMINAL:
        w = dict(NOMINAL)
        w[key] = NOMINAL[key] * (1.0 + perturb)
        pert = run_protocol(cfg, w)
        out[key] = {q: ((pert[q] - base[q]) / base[q]) / perturb if base[q] else 0.0
                    for q in _SENS_OUTPUTS}
    return base, out


def _outcomes_plot(results, fname):
    names = list(results.keys())
    align = [results[n]["align_final"] for n in names]
    rho = [results[n]["rho_final"] for n in names]
    phisen = [results[n]["phisen_final"] for n in names]
    tau = [results[n]["tau_mean"] for n in names]
    x = np.arange(len(names))

    fig, axes = plt.subplots(4, 1, figsize=(DOUBLE, DOUBLE * 0.95), sharex=True)
    specs = [(align, r"$\bar{\varphi}_{\mathrm{healthy}}$ (deg)", 20.0),
             (rho, r"$\bar{\rho}$", 2.3),
             (phisen, r"$\phi_{\mathrm{sen}}$ (final)", 0.30),
             (tau, r"mean $\tau$ (Pa)", None)]
    for ax, (vals, lab, ref) in zip(axes, specs):
        ax.bar(x, vals, color="#4C72B0", width=0.6)
        if ref is not None:
            ax.axhline(ref, ls="--", color="k", lw=1)
        ax.set_ylabel(lab)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(names, rotation=30, ha="right")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, fname)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return path


def _sensitivity_bar(sens, fname):
    keys = list(NOMINAL.keys())
    labels = [r"$w_\phi$", r"$w_\rho$", r"$w_{\bar\varphi}$", r"$w_u$"]
    colors = {"align_final": "#4C72B0", "phisen_final": "#C44E52", "tau_mean": "#55A868"}
    legend = {"align_final": r"$\bar{\varphi}$", "phisen_final": r"$\phi_{\mathrm{sen}}$",
              "tau_mean": r"$\bar{\tau}$"}
    y = np.arange(len(keys))
    h = 0.26
    fig, ax = plt.subplots(figsize=(SINGLE, SINGLE))
    for j, q in enumerate(_SENS_OUTPUTS):
        vals = [sens[k][q] for k in keys]
        ax.barh(y + (1 - j) * h, vals, height=h, color=colors[q], label=legend[q])
    ax.axvline(0.0, color="black", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("Normalised sensitivity index")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, fname)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return path


def run():
    set_style()
    os.makedirs(FIG_DIR, exist_ok=True)
    cfg = SimulationConfig()

    results = {name: run_protocol(cfg, w) for name, w in WEIGHT_SETS.items()}
    base, sens = normalised_sensitivity(cfg, perturb=0.10)
    paths = {
        "outcomes": _outcomes_plot(results, "mpc_weights_outcomes.pdf"),
        "sensitivity": _sensitivity_bar(sens, "mpc_weights_sensitivity.pdf"),
    }
    return {"results": results, "sensitivity": sens, "base": base, "paths": paths}


if __name__ == "__main__":
    s = run()
    print("MPC cost-weight sensitivity complete.\n")
    hdr = f"{'weight set':14s} {'align':>7s} {'rho':>6s} {'phi_f':>7s} {'phi_max':>8s} {'tau_m':>6s} {'tau_tv':>7s}"
    print(hdr)
    for name, r in s["results"].items():
        print(f"{name:14s} {r['align_final']:7.1f} {r['rho_final']:6.3f} "
              f"{r['phisen_final']:7.3f} {r['phisen_max']:8.3f} "
              f"{r['tau_mean']:6.3f} {r['tau_tv']:7.3f}")
    al = [r["align_final"] for r in s["results"].values()]
    tm = [r["tau_mean"] for r in s["results"].values()]
    print(f"\nspread across weight sets:  align range = {max(al)-min(al):.1f} deg, "
          f"mean-tau range = {max(tm)-min(tm):.3f} Pa")
    print("\nnormalised sensitivity (+10%):")
    for k, d in s["sensitivity"].items():
        print(f"  {k:9s}  S[align] = {d['align_final']:+.3f}  "
              f"S[phi_sen] = {d['phisen_final']:+.3f}  S[tau] = {d['tau_mean']:+.3f}")
    for k, v in s["paths"].items():
        print(f"  saved {k}: {v}")

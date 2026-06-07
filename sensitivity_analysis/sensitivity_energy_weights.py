"""
Part C - Sensitivity of the morphological energy weights (w_A : w_rho : w_phi).

The spatial scale of the model (main.tex, eq:argmin) assigns the monolayer
tessellation by minimising a configurational energy whose three modes are
weighted by w_A (area), w_rho (aspect ratio) and w_phi (alignment). Table 1
fixes these to the relative values 1 : 8.5 : 5 and states they are *not
independently identifiable* from the data, with the solution "verified to be
insensitive to their exact magnitude within the morphological range of
interest." This module provides that verification.

For each weight set the pipeline runs the multi-configuration initialiser
(which selects the energy-minimising layout, i.e. where the weights act) and a
6 h adaptation at the flow-adapted shear tau = 1.4 Pa, then records the
converged morphological observables:
    rho_bar   population-mean aspect ratio
    phi_bar   population-mean flow-alignment angle (deg, 0 = aligned)
    area_CV   coefficient of variation of cell areas
    gap_frac  fraction of unassigned pixels (partition quality; should be 0)

Outputs (figures/):
    energy_weights_observables.pdf   observables across the swept weight sets
    energy_weights_sensitivity.pdf   normalised sensitivity indices (+10 %)
and a printed summary table.
"""

import os
import io
import contextlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from endothelial_simulation.config import SimulationConfig
from endothelial_simulation.core.simulator import Simulator

np.random.seed(42)

CM = 1.0 / 2.54
SINGLE = 8.5 * CM
DOUBLE = 17.5 * CM
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

# Nominal relative weights (Source: Table 1, main.tex — w_A:w_rho:w_phi = 1:8.5:5)
NOMINAL = {"area": 1.0, "aspect_ratio": 8.5, "orientation": 5.0}

TAU_INPUT = 1.4        # flow-adapted shear (Pa)
DURATION_MIN = 360     # 6 h
CELL_COUNT = 50        # modest count keeps the sweep cheap but representative
NUM_CONFIGS = 3        # multi-config selection is where the weights act


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


def _build_config():
    cfg = SimulationConfig()
    cfg.enable_temporal_dynamics = True
    cfg.enable_spatial_properties = True
    cfg.enable_population_dynamics = False
    cfg.enable_senescence = False
    cfg.enable_holes = False
    cfg.save_plots = False
    cfg.create_animations = False
    cfg.simulation_duration = DURATION_MIN
    cfg.time_step = 30.0
    cfg.initial_cell_count = CELL_COUNT
    return cfg


def run_once(weights, seed=0):
    """
    Run init (multi-config selection) + 6 h adaptation at tau = 1.4 Pa with the
    given morphological energy weights, and return converged observables.
    """
    np.random.seed(seed)            # same layout draws across weight sets
    cfg = _build_config()
    with contextlib.redirect_stdout(io.StringIO()):
        sim = Simulator(cfg)
        # The morphological energy weights act only when SELECTING among candidate
        # configurations (calculate_biological_energy), not in the distance-based
        # Voronoi geometry. The gradient position optimiser is O(N^2 * pixels) and
        # dominates runtime without affecting what we test, so disable it here; the
        # tessellation remains a gap-free partition from the Poisson seeds.
        sim.grid.optimize_cell_positions = lambda *a, **k: None
        # the swept quantity: morphological energy weights used by eq:argmin scoring
        sim.grid.energy_weights["area"] = weights["area"]
        sim.grid.energy_weights["aspect_ratio"] = weights["aspect_ratio"]
        sim.grid.energy_weights["orientation"] = weights["orientation"]
        sim.set_constant_input(TAU_INPUT)
        sim.initialize_with_multiple_configurations(
            cell_count=CELL_COUNT, num_configurations=NUM_CONFIGS,
            optimization_iterations=0, save_analysis=False)
        sim.run(duration=DURATION_MIN)

        props = sim.grid.get_cell_properties()
        areas = np.asarray(props["areas"], dtype=float)
        rho_bar = float(props["mean_aspect_ratio"])
        phi_bar = float(props["mean_orientation"])     # already folded to [0, 90] deg
        area_cv = float(np.std(areas) / np.mean(areas)) if areas.size and np.mean(areas) > 0 else 0.0
        po = sim.grid.pixel_ownership
        gap_frac = float(np.mean(po == -1))
    return {"rho_bar": rho_bar, "phi_bar": phi_bar,
            "area_CV": area_cv, "gap_frac": gap_frac}


# Weight sets: nominal plus factor-of-2 perturbations of each mode and equal weights
WEIGHT_SETS = {
    "equal (1:1:1)":        {"area": 1.0, "aspect_ratio": 1.0, "orientation": 1.0},
    "nominal (1:8.5:5)":    dict(NOMINAL),
    "w_rho x2":             {"area": 1.0, "aspect_ratio": 17.0, "orientation": 5.0},
    "w_phi x2":             {"area": 1.0, "aspect_ratio": 8.5, "orientation": 10.0},
    "w_A x5":               {"area": 5.0, "aspect_ratio": 8.5, "orientation": 5.0},
}


def normalised_sensitivity(perturb=0.10):
    """S = (dQ/Q)/(dp/p) at nominal, +perturb on each weight, for each observable."""
    base = run_once(NOMINAL, seed=0)
    out = {}
    for key in ("area", "aspect_ratio", "orientation"):
        w = dict(NOMINAL)
        w[key] = NOMINAL[key] * (1.0 + perturb)
        pert = run_once(w, seed=0)
        out[key] = {q: ((pert[q] - base[q]) / base[q]) / perturb if base[q] else 0.0
                    for q in ("rho_bar", "phi_bar", "area_CV")}
    return base, out


def _observables_plot(results, fname):
    names = list(results.keys())
    rho = [results[n]["rho_bar"] for n in names]
    phi = [results[n]["phi_bar"] for n in names]
    cv = [results[n]["area_CV"] for n in names]
    x = np.arange(len(names))

    fig, axes = plt.subplots(3, 1, figsize=(DOUBLE, DOUBLE * 0.75), sharex=True)
    for ax, vals, lab, ref in zip(
            axes, [rho, phi, cv],
            [r"$\bar{\rho}$", r"$\bar{\varphi}$ (deg)", "area CV"],
            [2.3, 20.0, None]):
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
    keys = ["area", "aspect_ratio", "orientation"]
    labels = [r"$w_A$", r"$w_\rho$", r"$w_\varphi$"]
    s_rho = [sens[k]["rho_bar"] for k in keys]
    s_phi = [sens[k]["phi_bar"] for k in keys]
    y = np.arange(len(keys))
    h = 0.38
    fig, ax = plt.subplots(figsize=(SINGLE, SINGLE * 0.85))
    ax.barh(y + h / 2, s_rho, height=h, color="#4C72B0", label=r"$\bar{\rho}$")
    ax.barh(y - h / 2, s_phi, height=h, color="#C44E52", label=r"$\bar{\varphi}$")
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

    results = {name: run_once(w, seed=0) for name, w in WEIGHT_SETS.items()}
    base, sens = normalised_sensitivity(perturb=0.10)

    paths = {
        "observables": _observables_plot(results, "energy_weights_observables.pdf"),
        "sensitivity": _sensitivity_bar(sens, "energy_weights_sensitivity.pdf"),
    }
    return {"results": results, "sensitivity": sens, "base": base, "paths": paths}


if __name__ == "__main__":
    s = run()
    print("Morphological energy-weight sensitivity complete.\n")
    print(f"{'weight set':22s}  {'rho_bar':>8s}  {'phi_bar':>8s}  {'area_CV':>8s}  {'gap':>6s}")
    for name, r in s["results"].items():
        print(f"{name:22s}  {r['rho_bar']:8.3f}  {r['phi_bar']:8.2f}  "
              f"{r['area_CV']:8.3f}  {r['gap_frac']:6.3f}")
    # spread across weight sets (excluding none) — the key insensitivity metric
    rho_vals = [r["rho_bar"] for r in s["results"].values()]
    phi_vals = [r["phi_bar"] for r in s["results"].values()]
    print(f"\nspread across weight sets:  "
          f"rho_bar range = {max(rho_vals)-min(rho_vals):.3f}, "
          f"phi_bar range = {max(phi_vals)-min(phi_vals):.2f} deg")
    print("\nnormalised sensitivity (+10%):")
    for k, d in s["sensitivity"].items():
        print(f"  {k:12s}  S_rho = {d['rho_bar']:+.4f}  S_phi = {d['phi_bar']:+.4f}")
    for k, v in s["paths"].items():
        print(f"  saved {k}: {v}")

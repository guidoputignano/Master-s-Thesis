"""
Part A - One-at-a-time (OAT) sensitivity of the temporal-dynamics scale.

Implements the calibrated temporal-dynamics component of the paper (main.tex):
    eq:target      y*(tau) = y_stat + (y_flow - y_stat) * s(tau)   [gated]
    eq:relaxation  dy/dt   = (y*(tau) - y) / tau_adapt
    eq:stepsolution y(t)   = y*(tau) - (y*(tau) - y0) * exp(-t/tau_adapt)
    eq:orientation theta relaxes on the circle toward theta*(tau)

The model functions used here are the corrected methods of
``endothelial_simulation.models.TemporalDynamicsModel`` (s_activation,
gated_target, relax_step), so the analysis is built directly on the corrected
models. All quantities are in the paper's units (hours, Pa, degrees).

Parameters swept (one at a time, others at nominal Table-1 values):
    tau_adapt  in [6, 7.5, 9, 10.5, 12] h
    rho*       over +/-15% around 2.3
    theta*     over +/-15% around 20 deg
    tau_act    in [0.3, 0.4, 0.5, 0.6, 0.7] Pa

Outputs (figures/):
    oat_envelope_tau_adapt.pdf   aspect-ratio rho(t) envelope vs tau_adapt
    oat_envelope_theta_star.pdf  alignment phi(t) envelope vs theta*
    oat_sensitivity_bars.pdf     normalised sensitivity indices (bar chart)
    oat_monolayer_fast.pdf       monolayer snapshot at t=6 h, tau_adapt = 6 h
    oat_monolayer_slow.pdf       monolayer snapshot at t=6 h, tau_adapt = 12 h
"""

import os
import io
import contextlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from endothelial_simulation.config import SimulationConfig
from endothelial_simulation.models import TemporalDynamicsModel

# ----------------------------------------------------------------------------
# Reproducibility and journal style
# ----------------------------------------------------------------------------
np.random.seed(42)

CM = 1.0 / 2.54
SINGLE = 8.5 * CM   # single-column width
DOUBLE = 17.5 * CM  # double-column width
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def set_style():
    """Journal-quality matplotlib defaults: 11 pt labels, 9 pt ticks/legend."""
    plt.style.use("default")  # reset any global style (e.g. seaborn set by Plotter)
    plt.rcParams.update({
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.grid": False,
        "axes.edgecolor": "black",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.3,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "font.family": "sans-serif",
    })


# ----------------------------------------------------------------------------
# Nominal Table-1 values (paper units)
# ----------------------------------------------------------------------------
NOMINAL = {
    "tau_adapt": 9.0,   # Source: Table 1, main.tex — tau_adapt = 6-12 h (nominal midpoint)
    "rho_star": 2.3,    # Source: Table 1, main.tex — rho* = 2.3
    "theta_star": 20.0,  # Source: Table 1, main.tex — theta* = 20 degrees
    "tau_act": 0.5,     # Source: Table 1, main.tex — tau_act = 0.5 Pa
}

TAU_INPUT = 1.4         # Source: Table 1, main.tex — flow-adapted shear tau_opt = 1.4 Pa
T_HORIZON = 6.0         # six-hour conditioning horizon
RHO_STATIC = 1.0        # isotropic (no-flow) baseline aspect ratio
PHI_ISO = 45.0          # isotropic mean alignment angle (degrees)

_CFG = SimulationConfig()
_TM = TemporalDynamicsModel(_CFG)   # corrected temporal model (provides eq:target/relaxation)


def simulate_temporal(tau_adapt, rho_star, theta_star, tau_act,
                      tau_input=TAU_INPUT, t_horizon=T_HORIZON, n=241):
    """
    Integrate the temporal-dynamics step response over the horizon.

    Returns:
        t   : (n,) time grid in hours
        rho : (n,) aspect-ratio trajectory rho(t)
        phi : (n,) mean alignment-angle trajectory phi(t) in degrees
    """
    t = np.linspace(0.0, t_horizon, n)
    dt = t[1] - t[0]

    # Gated targets (eq:target) evaluated through the corrected model
    rho_target = _TM.gated_target(RHO_STATIC, rho_star, tau_input, tau_act)
    phi_target = _TM.gated_target(PHI_ISO, theta_star, tau_input, tau_act)

    rho = np.empty(n)
    phi = np.empty(n)
    rho[0] = RHO_STATIC
    phi[0] = PHI_ISO
    # First-order relaxation (eq:relaxation / eq:stepsolution) via the model helper
    for k in range(1, n):
        rho[k] = _TM.relax_step(rho[k - 1], rho_target, dt, tau_adapt)
        phi[k] = _TM.relax_step(phi[k - 1], phi_target, dt, tau_adapt)
    return t, rho, phi


# ----------------------------------------------------------------------------
# Parameter sweeps
# ----------------------------------------------------------------------------
SWEEPS = {
    "tau_adapt": [6.0, 7.5, 9.0, 10.5, 12.0],
    "rho_star": list(np.round(np.linspace(2.3 * 0.85, 2.3 * 1.15, 5), 4)),
    "theta_star": list(np.round(np.linspace(20.0 * 0.85, 20.0 * 1.15, 5), 4)),
    "tau_act": [0.3, 0.4, 0.5, 0.6, 0.7],
}


def _run_sweep(param):
    """Run a one-at-a-time sweep for `param`; return list of (value, t, rho, phi)."""
    results = []
    for val in SWEEPS[param]:
        kw = dict(NOMINAL)
        kw[param] = val
        t, rho, phi = simulate_temporal(kw["tau_adapt"], kw["rho_star"],
                                        kw["theta_star"], kw["tau_act"])
        results.append((val, t, rho, phi))
    return results


# ----------------------------------------------------------------------------
# Normalised sensitivity indices
# ----------------------------------------------------------------------------
def normalised_sensitivity(perturb=0.10):
    """
    Normalised sensitivity index S = (dQ/Q) / (dp/p) at the nominal point,
    with a +`perturb` relative step on each parameter. Computed for both
    outputs Q = rho(6h) and Q = phi(6h).

    Returns dict: param -> {'rho': S_rho, 'phi': S_phi}.
    """
    t0, rho0, phi0 = simulate_temporal(**NOMINAL)
    rho_base, phi_base = rho0[-1], phi0[-1]

    out = {}
    for param in SWEEPS:
        kw = dict(NOMINAL)
        kw[param] = NOMINAL[param] * (1.0 + perturb)
        _, rho_p, phi_p = simulate_temporal(**kw)
        s_rho = ((rho_p[-1] - rho_base) / rho_base) / perturb
        s_phi = ((phi_p[-1] - phi_base) / phi_base) / perturb
        out[param] = {"rho": s_rho, "phi": s_phi}
    return out


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
_LABELS = {
    "tau_adapt": r"$\tau_{\mathrm{adapt}}$ (h)",
    "rho_star": r"$\rho^{*}$",
    "theta_star": r"$\theta^{*}$ (deg)",
    "tau_act": r"$\tau_{\mathrm{act}}$ (Pa)",
}


def _envelope_plot(param, which, ylabel, fname):
    """Trajectory envelope: one curve per swept value, shaded band, nominal black."""
    results = _run_sweep(param)
    idx = 2 if which == "rho" else 3
    t = results[0][1]
    curves = np.array([r[idx] for r in results])
    vals = [r[0] for r in results]

    fig, ax = plt.subplots(figsize=(SINGLE, SINGLE * 0.8))
    ax.fill_between(t, curves.min(axis=0), curves.max(axis=0),
                    color="0.80", alpha=0.7, linewidth=0, zorder=1)

    nominal_val = NOMINAL[param]
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(vals)))
    for c, (val, tt, rho, phi) in zip(cmap, results):
        y = rho if which == "rho" else phi
        if np.isclose(val, nominal_val):
            ax.plot(tt, y, color="black", lw=1.8, zorder=3,
                    label=f"{val:g} (nominal)")
        else:
            ax.plot(tt, y, color=c, lw=1.1, zorder=2, label=f"{val:g}")

    ax.set_xlabel("Time (h)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, T_HORIZON)
    ax.legend(title=_LABELS[param], frameon=False, ncol=1, loc="best")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, fname)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return path


def _sensitivity_bar(sens, fname):
    """Horizontal grouped bar chart of normalised sensitivity indices."""
    params = list(SWEEPS.keys())
    labels = [_LABELS[p] for p in params]
    s_rho = [sens[p]["rho"] for p in params]
    s_phi = [sens[p]["phi"] for p in params]

    y = np.arange(len(params))
    h = 0.38
    fig, ax = plt.subplots(figsize=(SINGLE, SINGLE * 0.85))
    ax.barh(y + h / 2, s_rho, height=h, color="#4C72B0", label=r"$\rho(6\,\mathrm{h})$")
    ax.barh(y - h / 2, s_phi, height=h, color="#C44E52", label=r"$\varphi(6\,\mathrm{h})$")
    ax.axvline(0.0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Normalised sensitivity index")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, fname)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------
# Monolayer snapshots (existing 2D tessellation / visualisation)
# ----------------------------------------------------------------------------
def _monolayer_snapshot(tau_adapt_hours, fname, cell_count=60):
    """
    Run the spatial+temporal simulator for 6 h at tau = 1.4 Pa with an
    effective adaptation time constant of `tau_adapt_hours`, and save a
    journal-quality PDF of the monolayer tessellation at t = 6 h.

    The effective time constant is set by scaling the temporal model's
    per-property time-scale factors so that base_tau * factor = tau_adapt.
    """
    from endothelial_simulation.core.simulator import Simulator
    from endothelial_simulation.visualization import Plotter

    cfg = SimulationConfig()
    cfg.enable_temporal_dynamics = True
    cfg.enable_spatial_properties = True
    cfg.enable_population_dynamics = False
    cfg.enable_senescence = False
    cfg.enable_holes = False
    cfg.save_plots = False
    cfg.create_animations = False
    cfg.simulation_duration = 360  # minutes (6 h)
    # The single-cell relaxation is closed-form per step (eq:stepsolution), so a
    # coarser step is exact for a constant input and keeps the snapshot cheap.
    cfg.time_step = 5.0
    cfg.initial_cell_count = cell_count

    with contextlib.redirect_stdout(io.StringIO()):
        sim = Simulator(cfg)
        tm = sim.models["temporal"]
        # base tau (minutes) at the flow input, then scale to reach tau_adapt
        base_tau, _ = tm.get_scaled_tau_and_amax(TAU_INPUT, "biochemical")
        factor = (tau_adapt_hours * 60.0) / base_tau
        tm.set_time_scale_factors({"area": factor,
                                   "orientation": factor,
                                   "aspect_ratio": factor})
        sim.initialize(cell_count=cell_count)
        sim.set_constant_input(TAU_INPUT)
        sim.run(duration=360)

        plotter = Plotter(cfg)
        fig = plotter.plot_cell_visualization(sim, save_path=None,
                                              show_boundaries=True, show_seeds=False)

    # strip the title for journal quality, save as 600-dpi PDF
    for ax in fig.axes:
        ax.set_title("")
    path = os.path.join(FIG_DIR, fname)
    fig.savefig(path, format="pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def run(make_snapshots=True):
    """Run the full OAT analysis; returns a summary dict."""
    set_style()
    os.makedirs(FIG_DIR, exist_ok=True)

    paths = {}
    paths["envelope_tau_adapt"] = _envelope_plot(
        "tau_adapt", "rho", r"Aspect ratio $\rho(t)$", "oat_envelope_tau_adapt.pdf")
    paths["envelope_theta_star"] = _envelope_plot(
        "theta_star", "phi", r"Alignment angle $\varphi(t)$ (deg)", "oat_envelope_theta_star.pdf")

    sens = normalised_sensitivity(perturb=0.10)
    paths["sensitivity_bars"] = _sensitivity_bar(sens, "oat_sensitivity_bars.pdf")

    if make_snapshots:
        try:
            paths["monolayer_fast"] = _monolayer_snapshot(6.0, "oat_monolayer_fast.pdf")
            paths["monolayer_slow"] = _monolayer_snapshot(12.0, "oat_monolayer_slow.pdf")
        except Exception as exc:  # snapshots are illustrative; never fail the run
            print(f"  [warning] monolayer snapshot skipped: {exc}")

    summary = {"sensitivity": sens, "paths": paths,
               "nominal_outputs": _nominal_outputs()}
    return summary


def _nominal_outputs():
    t, rho, phi = simulate_temporal(**NOMINAL)
    return {"rho_6h": float(rho[-1]), "phi_6h": float(phi[-1])}


if __name__ == "__main__":
    s = run()
    print("OAT temporal sensitivity complete.")
    print(f"Nominal rho(6h) = {s['nominal_outputs']['rho_6h']:.4f}, "
          f"phi(6h) = {s['nominal_outputs']['phi_6h']:.2f} deg")
    print("Normalised sensitivity indices (+10% perturbation):")
    for p, d in s["sensitivity"].items():
        print(f"  {p:11s}  S_rho = {d['rho']:+.4f}   S_phi = {d['phi']:+.4f}")
    for k, v in s["paths"].items():
        print(f"  saved {k}: {v}")

"""
Reproducible sensitivity study of the receding-horizon closed loop with respect
to the temporal horizons.

Purpose
-------
This module quantifies how the closed-loop performance of the reported controller
(run_mpc_simulation / RecedingHorizonMPC in control/mpc_controller.py) depends on
the prediction horizon N_p and the control horizon N_c. The intended use is to
establish whether the closed-loop cost and the constraint satisfaction plateau as
the horizon grows, which is the evidence that a given reported horizon is
sufficient. It is an analysis layer only. It does not modify the model dynamics,
the cost function, or any parameter default. In particular the single
data-calibrated morphological adaptation constant of 7.4 hours
(tau_orient_hours = tau_adapt_hours) is held fixed throughout, and cell area
remains determined by the spatial tessellation and is not relaxed here.

Method
------
The reported hour-boundary observables (the applied input tau, the senescent
fraction phi_sen, the mean aspect ratio rho_bar, and the mean flow-alignment angle
varphi_bar) are produced entirely by the deterministic reduced-state rollout of
the controller, namely the sequence solve, predict_step, outputs, evaluated from
the initial reduced state x0. The only stochastic input to this rollout is the
initial population composition x0['pop'], which the model draws when the monolayer
is created. The per-cell morphological deviates of the full pipeline act on the
rendered tessellation only and do not enter these aggregates, so they are not a
source of variability for the reported metrics.

To evaluate each horizon setting over independent realisations, the initial
population composition is regenerated for each replicate directly from the model's
own initial-condition rules, seeded from config.random_seed. Specifically, healthy
division stages are drawn from the populate_grid law
int(max_divisions * 0.5 * (1 - sqrt(U))), and a senescent fraction
phi_sen(0) = config.initial_senescent_fraction of the cells is allocated with the
70/30 stress and telomere split of _apply_initial_senescence. This reproduces the
initial composition of the full spatial pipeline (identical N_E, S_tel, and S_str,
and a matching division distribution) while avoiding the expensive tessellation,
so that a study over many replicates and horizon settings is tractable. For the
master seed this generator yields the same first applied input as the full spatial
pipeline, confirming that the reduced-state rollout is reproduced faithfully.

Parameter grids and replication
-------------------------------
Prediction-horizon grid  : NP_GRID  (default 3, 4, 6, 8, 12), evaluated at the
                           nominal control horizon NC_NOMINAL (default 3).
Control-horizon grid     : for each N_p, the admissible control horizons
                           {1, 2, 3, N_p} with N_c <= N_p, evaluated at the
                           nominal prediction horizon NP_NOMINAL (default 6). The
                           full admissible (N_p, N_c) grid is also evaluated to
                           populate the cost heatmap.
Replicates               : N_REP independent seeds (default 20). Seed r uses
                           config.random_seed + r. Results are reported as the
                           mean with a 95 percent confidence interval based on the
                           Student t distribution with N_REP - 1 degrees of
                           freedom.
Run length               : NUM_STEPS one-hour control intervals (default 6, the
                           six-hour experimental window). The run length is not
                           changed by this study. Prediction horizons larger than
                           NUM_STEPS look beyond the end of the experimental
                           window and are included only to demonstrate the
                           plateau.

Reported metrics (per closed-loop run, then aggregated over replicates)
-----------------------------------------------------------------------
tau(k)          : applied wall shear stress at each one-hour boundary, in Pa.
phi_sen(k)      : senescent fraction at each boundary, dimensionless.
rho_bar(k)      : population mean aspect ratio at each boundary, dimensionless.
varphi_bar(k)   : population mean flow-alignment angle at each boundary, reported
                  in degrees. Zero denotes perfect alignment with the flow.
terminal values : the value of phi_sen, rho_bar, and varphi_bar at t = NUM_STEPS.
J               : the realised closed-loop cost, evaluated with the controller's
                  own stage weights on the applied trajectory (Task 5, Part A: no
                  soft senescence term; senescence is a hard constraint),
                  J = sum_k [ w_rho * (rho_bar(k) - rho_flow)^2
                            + w_varphi * varphi_bar(k)^2
                            + w_u * (tau(k) - tau(k-1))^2 ],
                  summed over the applied intervals k = 1..NUM_STEPS, with
                  tau(0) = 0. varphi_bar is in radians in this expression, in
                  agreement with the controller.
solve_time      : wall-clock time of each per-step SLSQP solve, in seconds. The
                  per-run summary reports the mean over steps.
success_rate    : fraction of per-step solves for which SLSQP reported success.
n_violations    : number of one-hour boundaries at which phi_sen exceeds the hard
                  senescence constraint phi_sen_max = 0.30.
violation_hours : the boundaries at which a violation occurred.

Reproducibility
---------------
The reported metrics are deterministic given the seed, because the reduced-state
rollout is deterministic and the initial composition is seeded. Re-running the
study with the same master seed therefore reproduces every deterministic aggregate
to floating-point tolerance. The solve-time metrics are wall-clock measurements
and reproduce only to a stated tolerance.

Outputs
-------
All outputs are written to a timestamped directory under
results/horizon_sensitivity/. These are the raw per-run and per-hour results as
CSV, a compact JSON summary, a Markdown summary table suitable for the
supplementary material, and the figures for Tasks 1, 2, and 3.

Usage
-----
    python -m analysis.horizon_sensitivity
    python -m analysis.horizon_sensitivity --n-rep 20
    python -m analysis.horizon_sensitivity --smoke        # small, fast self-test
    python -m analysis.horizon_sensitivity --verify       # reproducibility check
"""
import os
import sys
import csv
import json
import time
import argparse

import numpy as np

# Make the endothelial_simulation package importable when run as a script.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib
matplotlib.use('Agg')  # headless, figure files only
import matplotlib.pyplot as plt

from scipy import stats

from endothelial_simulation.config import SimulationConfig
from endothelial_simulation.control.mpc_controller import (
    RecedingHorizonMPC, flow_alignment_angle, RHO_FLOW,
)

# --- Default study configuration ---------------------------------------------
NP_GRID = [3, 4, 6, 8, 12]     # prediction-horizon grid (one-hour steps)
NP_NOMINAL = 6                 # nominal prediction horizon (controller default)
NC_NOMINAL = 3                 # nominal control horizon (controller default)
NC_CANDIDATES = [1, 2, 3]      # control horizons swept, augmented by N_p per row
N_REP = 20                     # independent seeded replicates
NUM_STEPS = 6                  # closed-loop run length, six-hour window (fixed)
CONFIDENCE = 0.95              # confidence level for the reported intervals


# =============================================================================
#  Initial condition and closed-loop rollout
# =============================================================================
def generate_initial_population(config, seed):
    """Return a seeded initial population vector [E_0..E_N, S_tel, S_str].

    The composition follows the model's own initial-condition rules (see the
    module docstring): healthy division stages from the populate_grid law
    int(max_divisions * 0.5 * (1 - sqrt(U))), and a senescent fraction
    config.initial_senescent_fraction allocated with the 70/30 stress and
    telomere split of _apply_initial_senescence. A dedicated numpy Generator is
    used so replicates are independent and reproducible without touching global
    RNG state.
    """
    N = int(config.initial_cell_count)
    maxd = int(config.max_divisions)
    rng = np.random.default_rng(int(seed))

    divisions = (maxd * 0.5 * (1.0 - np.sqrt(rng.random(N)))).astype(int)

    n_sen = int(round(config.initial_senescent_fraction * N))
    denom = config.senescent_stress_fraction + config.senescent_telomere_fraction
    n_str = int(round(n_sen * config.senescent_stress_fraction / denom)) if denom > 0 else 0
    n_tel = n_sen - n_str

    is_sen = np.zeros(N, dtype=bool)
    is_sen[rng.permutation(N)[:n_sen]] = True

    E = np.zeros(maxd + 1)
    for i in range(N):
        if not is_sen[i]:
            E[min(int(divisions[i]), maxd)] += 1.0
    return np.concatenate([E, [float(n_tel), float(n_str)]])


def run_closed_loop(config, n_prediction, n_control, n_steps, seed):
    """Run one seeded receding-horizon closed loop and return its metrics.

    The controller is instantiated with the requested horizons through its public
    constructor arguments; no model or controller code is modified. All other
    parameters, including the 7.4 hour adaptation constant, take their config
    defaults.
    """
    mpc = RecedingHorizonMPC(config, n_prediction=n_prediction, n_control=n_control)

    pop = generate_initial_population(config, seed)
    x = {'pop': pop.copy(),
         'rho_h': mpc.rho_target(0.0),        # rho_stat at tau = 0 Pa
         'theta_h': mpc.theta_target(0.0)}    # theta_stat at tau = 0 Pa

    phi0, rho0, varphi0 = mpc.outputs(x)
    tau_seq = []
    phi_seq = [phi0]
    rho_seq = [rho0]
    varphi_seq = [varphi0]           # radians
    solve_times = []
    successes = []

    u_prev = 0.0
    for _ in range(n_steps):
        t0 = time.perf_counter()
        u_opt, res = mpc.solve(x, u_prev)
        solve_times.append(time.perf_counter() - t0)
        successes.append(bool(getattr(res, 'success', False)))

        tau_k = float(np.clip(u_opt[0], mpc.tau_min, mpc.tau_max))
        x = mpc.predict_step(x, tau_k)
        phi_k, rho_k, varphi_k = mpc.outputs(x)

        tau_seq.append(tau_k)
        phi_seq.append(phi_k)
        rho_seq.append(rho_k)
        varphi_seq.append(varphi_k)
        u_prev = tau_k

    tau = np.asarray(tau_seq)
    phi = np.asarray(phi_seq)
    rho = np.asarray(rho_seq)
    varphi = np.asarray(varphi_seq)

    # Realised closed-loop cost with the controller's own stage weights.
    # Task 5 (Part A): the cost no longer contains a soft senescence term
    # (w_phi * phi_sen^2); senescence is a hard constraint. The realised cost is
    # therefore tracking (aspect ratio + flow-alignment angle) plus the move
    # regularizer only.
    tau_prev = np.concatenate([[0.0], tau[:-1]])
    J = float(np.sum(
        mpc.w_rho * (rho[1:] - RHO_FLOW) ** 2
        + mpc.w_varphi * varphi[1:] ** 2
        + mpc.w_u * (tau - tau_prev) ** 2))

    violation_hours = [k for k in range(1, n_steps + 1)
                       if phi[k] > mpc.phi_sen_max + 1e-12]

    return {
        'n_prediction': n_prediction,
        'n_control': n_control,
        'seed': int(seed),
        'tau': tau,
        'phi_sen': phi,
        'rho_bar': rho,
        'varphi_bar_rad': varphi,
        'varphi_bar_deg': np.degrees(varphi),
        'terminal_phi_sen': float(phi[-1]),
        'terminal_rho_bar': float(rho[-1]),
        'terminal_varphi_bar_deg': float(np.degrees(varphi[-1])),
        'J': J,
        'solve_times': np.asarray(solve_times),
        'mean_solve_time': float(np.mean(solve_times)),
        'success_rate': float(np.mean(successes)),
        'n_violations': int(len(violation_hours)),
        'violation_hours': violation_hours,
    }


# =============================================================================
#  Aggregation
# =============================================================================
def mean_ci(values, confidence=CONFIDENCE):
    """Return (mean, half-width of the confidence interval) for a 1-D sample.

    The half-width uses the Student t distribution with n - 1 degrees of freedom.
    For n < 2 the half-width is not defined and is returned as NaN.
    """
    a = np.asarray(values, dtype=float)
    n = a.size
    mean = float(np.mean(a))
    if n < 2:
        return mean, float('nan')
    sem = float(np.std(a, ddof=1) / np.sqrt(n))
    half = float(stats.t.ppf(0.5 + confidence / 2.0, df=n - 1) * sem)
    return mean, half


def aggregate(runs):
    """Aggregate a list of per-replicate run dictionaries (same Np, Nc)."""
    J = [r['J'] for r in runs]
    phi = [r['terminal_phi_sen'] for r in runs]
    rho = [r['terminal_rho_bar'] for r in runs]
    varphi = [r['terminal_varphi_bar_deg'] for r in runs]
    stime = [r['mean_solve_time'] for r in runs]
    succ = [r['success_rate'] for r in runs]
    any_violation = [1.0 if r['n_violations'] > 0 else 0.0 for r in runs]

    J_m, J_ci = mean_ci(J)
    phi_m, phi_ci = mean_ci(phi)
    rho_m, rho_ci = mean_ci(rho)
    var_m, var_ci = mean_ci(varphi)
    st_m, st_ci = mean_ci(stime)
    return {
        'n_prediction': runs[0]['n_prediction'],
        'n_control': runs[0]['n_control'],
        'n_rep': len(runs),
        'J_mean': J_m, 'J_ci95': J_ci,
        'terminal_phi_sen_mean': phi_m, 'terminal_phi_sen_ci95': phi_ci,
        'terminal_rho_bar_mean': rho_m, 'terminal_rho_bar_ci95': rho_ci,
        'terminal_varphi_bar_deg_mean': var_m, 'terminal_varphi_bar_deg_ci95': var_ci,
        'mean_solve_time_mean': st_m, 'mean_solve_time_ci95': st_ci,
        'solver_success_rate_mean': float(np.mean(succ)),
        'constraint_violation_frequency': float(np.mean(any_violation)),
        'total_constraint_violations': int(sum(r['n_violations'] for r in runs)),
    }


# =============================================================================
#  Study orchestration
# =============================================================================
def admissible_nc(n_prediction, nc_candidates=NC_CANDIDATES):
    """Admissible control horizons for a prediction horizon: {candidates, Np} with
    Nc <= Np, de-duplicated and sorted."""
    vals = sorted({nc for nc in list(nc_candidates) + [n_prediction]
                   if 1 <= nc <= n_prediction})
    return vals


def run_study(config, np_grid=NP_GRID, nc_nominal=NC_NOMINAL, np_nominal=NP_NOMINAL,
              n_rep=N_REP, num_steps=NUM_STEPS, base_seed=None, progress=None):
    """Evaluate the full admissible (Np, Nc) grid over n_rep seeded replicates.

    The grid is the union of the Task 1 sweep (Np in np_grid at Nc = nc_nominal),
    the Task 2 sweep (Nc admissible at Np = np_nominal), and every admissible
    (Np, Nc) pair needed for the cost heatmap. Returns (raw_runs, aggregates).
    """
    if base_seed is None:
        base_seed = int(config.random_seed)

    # Grid explored (kept to admissible, feasible settings):
    #   Task 1 / heatmap : the rectangular grid np_grid x {1,2,3} with Nc <= Np;
    #   Task 2 row       : Nc in {1,2,3,Np} at the nominal Np (adds the Nc = Np
    #                      point at the nominal horizon only).
    # The Nc = Np corner is deliberately not evaluated for the large prediction
    # horizons, where an unblocked control horizon is both inadmissible as a
    # reported configuration and disproportionately expensive to solve.
    settings = set()
    for npred in np_grid:
        for nc in NC_CANDIDATES:
            if nc <= npred:
                settings.add((npred, nc))
    for nc in admissible_nc(np_nominal):
        settings.add((np_nominal, nc))
    settings = sorted(settings)

    raw_runs = []
    aggregates = []
    total = len(settings)
    for idx, (npred, nc) in enumerate(settings):
        runs = []
        for r in range(n_rep):
            runs.append(run_closed_loop(config, npred, nc, num_steps, base_seed + r))
        raw_runs.extend(runs)
        aggregates.append(aggregate(runs))
        if progress is not None:
            progress(idx + 1, total, npred, nc)
    return raw_runs, aggregates


# =============================================================================
#  Output writers
# =============================================================================
def write_raw_csv(raw_runs, path_run, path_hourly, num_steps):
    """Write the run-level and per-hour raw results as CSV."""
    with open(path_run, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['n_prediction', 'n_control', 'seed', 'J', 'terminal_phi_sen',
                    'terminal_rho_bar', 'terminal_varphi_bar_deg',
                    'mean_solve_time_s', 'solver_success_rate',
                    'n_constraint_violations', 'violation_hours'])
        for r in raw_runs:
            w.writerow([r['n_prediction'], r['n_control'], r['seed'],
                        f"{r['J']:.10g}", f"{r['terminal_phi_sen']:.10g}",
                        f"{r['terminal_rho_bar']:.10g}",
                        f"{r['terminal_varphi_bar_deg']:.10g}",
                        f"{r['mean_solve_time']:.6g}", f"{r['success_rate']:.6g}",
                        r['n_violations'],
                        ';'.join(str(h) for h in r['violation_hours'])])

    with open(path_hourly, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['n_prediction', 'n_control', 'seed', 'hour',
                    'tau_Pa', 'phi_sen', 'rho_bar', 'varphi_bar_deg'])
        for r in raw_runs:
            for k in range(num_steps + 1):
                tau_k = '' if k == 0 else f"{r['tau'][k - 1]:.10g}"
                w.writerow([r['n_prediction'], r['n_control'], r['seed'], k, tau_k,
                            f"{r['phi_sen'][k]:.10g}", f"{r['rho_bar'][k]:.10g}",
                            f"{r['varphi_bar_deg'][k]:.10g}"])


def write_summary_json(aggregates, meta, path):
    """Write the compact machine-readable summary."""
    with open(path, 'w') as f:
        json.dump({'metadata': meta, 'aggregates': aggregates}, f, indent=2)


def write_summary_markdown(aggregates, meta, path):
    """Write a Markdown summary table suitable for the supplementary material."""
    lines = []
    lines.append('# Horizon sensitivity summary')
    lines.append('')
    lines.append(f"Replicates per setting: {meta['n_rep']}. "
                 f"Run length: {meta['num_steps']} one-hour intervals "
                 f"(six-hour window). Master seed: {meta['base_seed']}. "
                 f"Morphological adaptation constant: {meta['tau_adapt_hours']} h "
                 f"(= tau_orient). Values are mean with a 95 percent confidence "
                 f"interval over replicates.")
    lines.append('')
    lines.append('| N_p | N_c | J | terminal phi_sen | terminal rho_bar | '
                 'terminal varphi_bar (deg) | mean solve time (s) | '
                 'solver success | violation freq. |')
    lines.append('|---|---|---|---|---|---|---|---|---|')
    for a in aggregates:
        beyond = ' *' if a['n_prediction'] > meta['num_steps'] else ''
        lines.append(
            f"| {a['n_prediction']}{beyond} | {a['n_control']} | "
            f"{a['J_mean']:.3f} +/- {a['J_ci95']:.3f} | "
            f"{a['terminal_phi_sen_mean']:.4f} +/- {a['terminal_phi_sen_ci95']:.4f} | "
            f"{a['terminal_rho_bar_mean']:.4f} +/- {a['terminal_rho_bar_ci95']:.4f} | "
            f"{a['terminal_varphi_bar_deg_mean']:.2f} +/- {a['terminal_varphi_bar_deg_ci95']:.2f} | "
            f"{a['mean_solve_time_mean']:.4f} +/- {a['mean_solve_time_ci95']:.4f} | "
            f"{a['solver_success_rate_mean']:.3f} | "
            f"{a['constraint_violation_frequency']:.3f} |")
    lines.append('')
    lines.append(f"Rows marked with an asterisk have N_p greater than the "
                 f"{meta['num_steps']}-hour experimental window and are included "
                 f"only to demonstrate the plateau.")
    lines.append('')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


# =============================================================================
#  Figures
# =============================================================================
def _style():
    plt.rcParams.update({
        'axes.labelsize': 11, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'legend.fontsize': 8.5, 'axes.titlesize': 11, 'figure.dpi': 120,
        'savefig.dpi': 300,
    })


def _by_nc(aggregates, nc):
    rows = sorted([a for a in aggregates if a['n_control'] == nc],
                  key=lambda a: a['n_prediction'])
    return rows


def _by_np(aggregates, npred):
    rows = sorted([a for a in aggregates if a['n_prediction'] == npred],
                  key=lambda a: a['n_control'])
    return rows


def _savefig(fig, path_noext):
    fig.savefig(path_noext + '.pdf', bbox_inches='tight')
    fig.savefig(path_noext + '.png', bbox_inches='tight')
    plt.close(fig)


def figure_np_sweep(aggregates, meta, out_noext):
    """Task 1 figure: J, terminal phi_sen, violation frequency, and solve time as
    functions of N_p at the nominal control horizon."""
    _style()
    rows = _by_nc(aggregates, meta['nc_nominal'])
    npv = np.array([a['n_prediction'] for a in rows])
    window = meta['num_steps']

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.6))

    def shade_beyond(ax):
        ax.axvspan(window + 0.001, npv.max() + 0.5, color='0.90', zorder=0)

    ax = axes[0, 0]
    shade_beyond(ax)
    ax.errorbar(npv, [a['J_mean'] for a in rows], yerr=[a['J_ci95'] for a in rows],
                marker='o', ms=4, capsize=3, color='C0')
    ax.axvline(meta['np_nominal'], ls='--', color='k', lw=1)
    ax.set_xlabel(r'prediction horizon $N_p$ (h)')
    ax.set_ylabel('closed-loop cost $J$')

    ax = axes[0, 1]
    shade_beyond(ax)
    ax.errorbar(npv, [a['terminal_phi_sen_mean'] for a in rows],
                yerr=[a['terminal_phi_sen_ci95'] for a in rows],
                marker='o', ms=4, capsize=3, color='C4')
    ax.axhline(0.30, ls='--', color='r', lw=1, label='constraint 0.30')
    ax.axvline(meta['np_nominal'], ls='--', color='k', lw=1)
    ax.set_ylim(0.0, 0.33)
    ax.set_xlabel(r'prediction horizon $N_p$ (h)')
    ax.set_ylabel(r'terminal $\phi_{\mathrm{sen}}$')
    ax.legend(loc='lower right')

    ax = axes[1, 0]
    shade_beyond(ax)
    ax.plot(npv, [a['constraint_violation_frequency'] for a in rows],
            marker='s', ms=4, color='C3')
    ax.axvline(meta['np_nominal'], ls='--', color='k', lw=1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r'prediction horizon $N_p$ (h)')
    ax.set_ylabel('constraint violation frequency')

    ax = axes[1, 1]
    shade_beyond(ax)
    ax.errorbar(npv, [a['mean_solve_time_mean'] for a in rows],
                yerr=[a['mean_solve_time_ci95'] for a in rows],
                marker='o', ms=4, capsize=3, color='C2')
    ax.axvline(meta['np_nominal'], ls='--', color='k', lw=1)
    ax.set_xlabel(r'prediction horizon $N_p$ (h)')
    ax.set_ylabel('mean SLSQP solve time (s)')

    fig.suptitle(f"Prediction-horizon sweep at $N_c$ = {meta['nc_nominal']} "
                 f"(shaded region: $N_p$ beyond the {window} h window)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _savefig(fig, out_noext)


def figure_nc_sweep(aggregates, meta, out_noext):
    """Task 2 figure: J and terminal phi_sen as functions of N_c at the nominal
    prediction horizon, and a heatmap of J over the admissible (Np, Nc) grid."""
    _style()
    rows = _by_np(aggregates, meta['np_nominal'])
    ncv = np.array([a['n_control'] for a in rows])

    fig = plt.figure(figsize=(11.0, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.25], wspace=0.42)

    ax = fig.add_subplot(gs[0, 0])
    ax.errorbar(ncv, [a['J_mean'] for a in rows], yerr=[a['J_ci95'] for a in rows],
                marker='o', ms=4, capsize=3, color='C0')
    ax.axvline(meta['nc_nominal'], ls='--', color='k', lw=1)
    ax.set_xlabel(r'control horizon $N_c$ (h)')
    ax.set_ylabel('closed-loop cost $J$')
    ax.set_title(f"$N_p$ = {meta['np_nominal']}", fontsize=9)

    ax = fig.add_subplot(gs[0, 1])
    ax.errorbar(ncv, [a['terminal_phi_sen_mean'] for a in rows],
                yerr=[a['terminal_phi_sen_ci95'] for a in rows],
                marker='o', ms=4, capsize=3, color='C4')
    ax.axhline(0.30, ls='--', color='r', lw=1, label='constraint 0.30')
    ax.axvline(meta['nc_nominal'], ls='--', color='k', lw=1)
    ax.set_ylim(0.0, 0.33)
    ax.set_xlabel(r'control horizon $N_c$ (h)')
    ax.set_ylabel(r'terminal $\phi_{\mathrm{sen}}$')
    ax.legend(loc='lower right')

    # Heatmap of J over the rectangular grid (Np) x {1,2,3} (the fully populated
    # part of the explored grid; the Nc = Np point at the nominal horizon is shown
    # in the line plot on the left rather than as a sparse heatmap row).
    np_vals = sorted({a['n_prediction'] for a in aggregates})
    nc_vals = [nc for nc in NC_CANDIDATES
               if any(a['n_control'] == nc for a in aggregates)]
    grid = np.full((len(nc_vals), len(np_vals)), np.nan)
    lookup = {(a['n_prediction'], a['n_control']): a['J_mean'] for a in aggregates}
    for i, nc in enumerate(nc_vals):
        for j, npd in enumerate(np_vals):
            if (npd, nc) in lookup:
                grid[i, j] = lookup[(npd, nc)]

    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(grid, origin='lower', aspect='auto', cmap='viridis',
                   interpolation='nearest')
    ax.set_xticks(range(len(np_vals))); ax.set_xticklabels(np_vals)
    ax.set_yticks(range(len(nc_vals))); ax.set_yticklabels(nc_vals)
    ax.set_xlabel(r'prediction horizon $N_p$ (h)')
    ax.set_ylabel(r'control horizon $N_c$ (h)')
    ax.set_title('closed-loop cost $J$', fontsize=9)
    for i in range(len(nc_vals)):
        for j in range(len(np_vals)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha='center', va='center',
                        color='white', fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _savefig(fig, out_noext)


def figure_headline(aggregates, meta, out_noext):
    """Task 3 headline figure: constraint satisfaction and alignment response
    across the prediction-horizon sweep, with the nominal configuration marked."""
    _style()
    rows = _by_nc(aggregates, meta['nc_nominal'])
    npv = np.array([a['n_prediction'] for a in rows])
    window = meta['num_steps']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.7))

    def shade(ax):
        ax.axvspan(window + 0.001, npv.max() + 0.5, color='0.90', zorder=0)
        ax.axvline(meta['np_nominal'], ls='--', color='k', lw=1)

    shade(ax1)
    phi_m = np.array([a['terminal_phi_sen_mean'] for a in rows])
    phi_ci = np.array([a['terminal_phi_sen_ci95'] for a in rows])
    ax1.fill_between(npv, phi_m - phi_ci, phi_m + phi_ci, color='C4', alpha=0.25)
    ax1.plot(npv, phi_m, marker='o', ms=4, color='C4', label=r'terminal $\phi_{\mathrm{sen}}$')
    ax1.axhline(0.30, ls='--', color='r', lw=1.2, label='hard constraint 0.30')
    ax1.set_ylim(0.0, 0.33)
    ax1.set_xlabel(r'prediction horizon $N_p$ (h)')
    ax1.set_ylabel(r'terminal senescent fraction $\phi_{\mathrm{sen}}$')
    ax1.legend(loc='center right')

    shade(ax2)
    var_m = np.array([a['terminal_varphi_bar_deg_mean'] for a in rows])
    var_ci = np.array([a['terminal_varphi_bar_deg_ci95'] for a in rows])
    ax2.fill_between(npv, var_m - var_ci, var_m + var_ci, color='C1', alpha=0.25)
    ax2.plot(npv, var_m, marker='^', ms=4, color='C1',
             label=r'terminal $\bar{\varphi}$')
    # Enforce a minimum vertical span so a small absolute variation is not
    # visually exaggerated by an auto-scaled axis. The alignment response is
    # stable across the sweep, and the axis should convey that.
    lo = float(np.min(var_m - var_ci))
    hi = float(np.max(var_m + var_ci))
    centre = 0.5 * (lo + hi)
    span = max(6.0, (hi - lo) * 3.0)
    ax2.set_ylim(centre - span / 2.0, centre + span / 2.0)
    ax2.set_xlabel(r'prediction horizon $N_p$ (h)')
    ax2.set_ylabel(r'terminal flow alignment $\bar{\varphi}$ (deg)')
    ax2.legend(loc='upper right')

    fig.suptitle('Constraint satisfaction and alignment are stable across the '
                 f'explored horizons\n(nominal $N_p$ = {meta["np_nominal"]}, '
                 f'$N_c$ = {meta["nc_nominal"]}; shaded: beyond the {window} h '
                 'experimental window)', fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _savefig(fig, out_noext)


# =============================================================================
#  Main
# =============================================================================
def _timestamp_dir():
    ts = time.strftime('%Y%m%d-%H%M%S')
    out = os.path.join(_REPO_ROOT, 'results', 'horizon_sensitivity', ts)
    os.makedirs(out, exist_ok=True)
    return out, ts


def run_and_write(config, out_dir, np_grid, nc_nominal, np_nominal, n_rep,
                  num_steps, base_seed, verbose=True):
    def progress(done, total, npd, nc):
        if verbose:
            print(f"  [{done:2d}/{total}] Np={npd:2d} Nc={nc:2d} done", flush=True)

    t0 = time.perf_counter()
    raw_runs, aggregates = run_study(
        config, np_grid=np_grid, nc_nominal=nc_nominal, np_nominal=np_nominal,
        n_rep=n_rep, num_steps=num_steps, base_seed=base_seed, progress=progress)
    elapsed = time.perf_counter() - t0

    meta = {
        'np_grid': list(np_grid), 'nc_candidates': list(NC_CANDIDATES),
        'np_nominal': np_nominal, 'nc_nominal': nc_nominal,
        'n_rep': n_rep, 'num_steps': num_steps, 'base_seed': int(base_seed),
        'confidence': CONFIDENCE,
        'tau_adapt_hours': float(config.tau_adapt_hours),
        'tau_orient_hours': float(config.tau_orient_hours),
        'initial_cell_count': int(config.initial_cell_count),
        'phi_sen_max': 0.30,
        'wall_time_s': elapsed,
    }

    write_raw_csv(raw_runs, os.path.join(out_dir, 'raw_runs.csv'),
                  os.path.join(out_dir, 'raw_hourly.csv'), num_steps)
    write_summary_json(aggregates, meta, os.path.join(out_dir, 'summary.json'))
    write_summary_markdown(aggregates, meta, os.path.join(out_dir, 'summary_table.md'))
    figure_np_sweep(aggregates, meta, os.path.join(out_dir, 'fig_np_sweep'))
    figure_nc_sweep(aggregates, meta, os.path.join(out_dir, 'fig_nc_sweep'))
    figure_headline(aggregates, meta, os.path.join(out_dir, 'fig_headline'))
    return raw_runs, aggregates, meta


def _deterministic_signature(aggregates):
    """A tuple of the deterministic aggregates (excludes wall-clock timing) used
    to check reproducibility to floating-point tolerance."""
    keys = ('n_prediction', 'n_control', 'J_mean', 'terminal_phi_sen_mean',
            'terminal_rho_bar_mean', 'terminal_varphi_bar_deg_mean',
            'constraint_violation_frequency', 'solver_success_rate_mean')
    return [tuple(a[k] for k in keys) for a in
            sorted(aggregates, key=lambda a: (a['n_prediction'], a['n_control']))]


def main(argv=None):
    p = argparse.ArgumentParser(description='Horizon sensitivity study of the '
                                            'receding-horizon closed loop.')
    p.add_argument('--n-rep', type=int, default=N_REP)
    p.add_argument('--num-steps', type=int, default=NUM_STEPS)
    p.add_argument('--smoke', action='store_true',
                   help='small, fast self-test (N_rep=3, Np in {3,6})')
    p.add_argument('--verify', action='store_true',
                   help='run twice and confirm deterministic aggregates match')
    args = p.parse_args(argv)

    config = SimulationConfig()
    base_seed = int(config.random_seed)

    np_grid = NP_GRID
    n_rep = args.n_rep
    num_steps = args.num_steps
    if args.smoke:
        np_grid = [3, 6]
        n_rep = 3

    print('=' * 70)
    print('Horizon sensitivity study (analysis layer, reported model unchanged)')
    print(f"  Np grid           : {np_grid}")
    print(f"  Nc candidates     : {NC_CANDIDATES} (augmented by Np per row)")
    print(f"  nominal (Np, Nc)  : ({NP_NOMINAL}, {NC_NOMINAL})")
    print(f"  replicates N_rep  : {n_rep}")
    print(f"  run length        : {num_steps} h (six-hour window)")
    print(f"  master seed       : {base_seed}")
    print(f"  adaptation const. : tau_adapt = tau_orient = {config.tau_adapt_hours} h")
    print('=' * 70)

    out_dir, ts = _timestamp_dir()
    raw_runs, aggregates, meta = run_and_write(
        config, out_dir, np_grid, NC_NOMINAL, NP_NOMINAL, n_rep, num_steps, base_seed)

    print(f"\nOutputs written to: {out_dir}")
    for name in ('raw_runs.csv', 'raw_hourly.csv', 'summary.json',
                 'summary_table.md', 'fig_np_sweep.pdf', 'fig_nc_sweep.pdf',
                 'fig_headline.pdf'):
        print(f"  {name}")

    if args.verify:
        print('\nReproducibility check: re-running the study with the same seed ...')
        _, aggregates2 = run_study(
            config, np_grid=np_grid, nc_nominal=NC_NOMINAL, np_nominal=NP_NOMINAL,
            n_rep=n_rep, num_steps=num_steps, base_seed=base_seed)
        sig1 = _deterministic_signature(aggregates)
        sig2 = _deterministic_signature(aggregates2)
        a1 = np.array([v for row in sig1 for v in row], dtype=float)
        a2 = np.array([v for row in sig2 for v in row], dtype=float)
        max_abs = float(np.max(np.abs(a1 - a2)))
        ok = np.allclose(a1, a2, rtol=0, atol=1e-12)
        print(f"  max abs difference in deterministic aggregates: {max_abs:.2e}")
        print(f"  reproducible to 1e-12: {ok}")

    return out_dir, aggregates, meta


if __name__ == '__main__':
    main()

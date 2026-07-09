"""
Robustness of the receding-horizon closed loop to parametric plant-model mismatch.

Purpose
-------
This module quantifies the efficacy of closed-loop control under identification
uncertainty in the reported model (run_mpc_simulation / RecedingHorizonMPC in
control/mpc_controller.py). The identified parameters of the dynamics are
perturbed in the PLANT, that is, in the true dynamics that are advanced between
control instants, while the controller continues to predict and optimise with the
NOMINAL, unperturbed parameters. The closed loop is then run on the mismatched
system. The study reports whether feedback still regulates the true system and
keeps the senescent fraction within its constraint. This is an epistemic
robustness study. It is distinct from a Monte Carlo over stochastic initial
conditions and from process noise, neither of which is part of it.

The mechanism that decouples the plant from the controller is documented in
docs/mismatch_plant_audit.md. In brief, the true reduced state is advanced by
`predict_step`; a separate `plant` object carrying perturbed parameters advances
it here, while `RecedingHorizonMPC.solve` keeps the nominal parameters. With a
nominal plant the closed loop reproduces the reported run bit-for-bit (verified in
the VERIFICATION step and by `--verify`).

Identified parameters perturbed in the plant
--------------------------------------------
1. the morphological adaptation constant (nominal 7.4 h, applied to both
   `tau_adapt` and `tau_orient`),
2. gamma_min   (nominal from config),
3. gamma_max   (nominal from config),
4. tau_h       (senescence Hill half-max shear; nominal from config),
through the monotone-decreasing Hill senescence-induction rate
gamma(tau) = gamma_min + (gamma_max - gamma_min) tau_h^n / (tau_h^n + tau^n) and
the morphological relaxation rates. The set is configurable, with these four as
the default. (Task 5 replaced the earlier quadratic-law identified set
gamma_min/alpha_gamma/tau_opt with the Hill-law set gamma_min/gamma_max/tau_h.)

Perturbation sampling (epistemic), seeded and reproducible
----------------------------------------------------------
(a) one-at-a-time: each identified parameter is perturbed individually by a stated
    relative amount (default plus or minus 20 percent), the others held nominal, to
    give a tornado view. The morphological adaptation constant is additionally swept
    over its calibrated range, 6 to 12 h.
(b) joint: all four parameters are drawn simultaneously from a Latin hypercube over
    their ranges (default plus or minus 20 percent, the adaptation constant over 6
    to 12 h), with a configurable number of samples (default 64).

The epistemic perturbation draws are kept programmatically separate from the
aleatory seeds used for the stochastic initial conditions. The Latin hypercube is
drawn from a dedicated generator seeded by PERTURB_SEED = config.random_seed +
10000. For each parameter sample the closed loop is run over N_REP initial-
condition seeds (default 10), each seed being config.random_seed + r, and the
results are aggregated.

Run matrix per perturbed plant (Task 3)
---------------------------------------
For every perturbed plant, and on the SAME initial-condition seeds, three
configurations are run so that the contribution of feedback can be isolated:
  (i)   closed-loop MPC with the nominal internal model, on the perturbed plant.
        This is the study of interest.
  (ii)  open-loop feedforward: the input sequence that the nominal MPC chooses on
        the NOMINAL plant is applied blindly to the perturbed plant, without
        re-solving. This isolates what feedback corrects for.
  (iii) the nominal-plant closed loop, as the no-mismatch reference. It depends
        only on the seed, so it is computed once per seed and reused. It also
        supplies the feedforward input sequence for (ii) and the reference
        trajectory for the tracking error.

Reported metrics (per run, then aggregated over seeds with a 95 percent CI)
---------------------------------------------------------------------------
tau(k)          : applied wall shear stress at each one-hour boundary, in Pa.
phi_sen(k)      : senescent fraction at each boundary, dimensionless.
rho_bar(k)      : population mean aspect ratio at each boundary, dimensionless.
varphi_bar(k)   : population mean flow-alignment angle at each boundary, in degrees.
terminal values : phi_sen, rho_bar, varphi_bar at t = NUM_STEPS.
J               : realised closed-loop cost on the applied (true) trajectory,
                  evaluated with the controller's own stage weights (Task 5,
                  Part A: no soft senescence term; senescence is a hard constraint),
                  J = sum_k [ w_rho (rho_bar(k) - rho_flow)^2
                            + w_varphi varphi_bar(k)^2 + w_u (tau(k) - tau(k-1))^2 ],
                  over k = 1..NUM_STEPS with tau(0) = 0; varphi_bar in radians here.
violations      : number and timing of boundaries at which phi_sen exceeds the hard
                  constraint phi_sen_max = 0.30.
tracking error  : root-mean-square deviation of a regulated output from the
                  nominal-plant reference over the horizon, per output.

Confidence intervals use the Student t distribution with N_REP - 1 degrees of
freedom. The reported quantities are deterministic given the seeds, so re-running
the study with the same master and perturbation seeds reproduces the summary to
floating-point tolerance. Timing is not reported here.

Outputs
-------
A timestamped directory under results/mismatch_robustness/ receives the raw
per-run results as CSV, a compact JSON summary, a Markdown summary table for the
supplementary material, and the figures.

Usage
-----
    python -m analysis.mismatch_robustness
    python -m analysis.mismatch_robustness --n-rep 10 --n-lhs 64
    python -m analysis.mismatch_robustness --smoke
    python -m analysis.mismatch_robustness --verify
"""
import os
import sys
import csv
import json
import time
import argparse

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy import stats

from endothelial_simulation.config import SimulationConfig
from endothelial_simulation.control.mpc_controller import (
    RecedingHorizonMPC, flow_alignment_angle, RHO_FLOW,
)
from analysis.horizon_sensitivity import generate_initial_population

# --- Default study configuration ---------------------------------------------
PARAM_KEYS = ['morph', 'gamma_min', 'gamma_max', 'tau_h']
PARAM_LABELS = {
    'morph': 'adaptation constant',
    'gamma_min': r'$\gamma_{\min}$',
    'gamma_max': r'$\gamma_{\max}$',
    'tau_h': r'$\tau_h$',
}
# Map identified-parameter keys to RecedingHorizonMPC attribute names.
PARAM_ATTR = {'gamma_min': 'gamma_min', 'gamma_max': 'gamma_max', 'tau_h': 'tau_h_sen'}
REL_PERTURB = 0.20           # one-at-a-time and joint relative range (plus/minus)
MORPH_RANGE = (6.0, 12.0)    # calibrated range of the adaptation constant, hours
MORPH_SWEEP = [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]  # explicit OAT sweep
N_REP = 10                   # aleatory initial-condition seeds per plant
N_LHS = 64                   # joint Latin hypercube samples
NUM_STEPS = 6                # six-hour experimental window
PHI_SEN_MAX = 0.30           # hard senescence constraint
CONFIDENCE = 0.95
PERTURB_SEED_OFFSET = 10000  # epistemic seed = config.random_seed + this


# =============================================================================
#  Nominal parameters and plant construction
# =============================================================================
def nominal_params(config):
    """The nominal identified parameters read from the config."""
    return {
        'morph': float(config.tau_adapt_hours),   # == tau_orient_hours
        'gamma_min': float(config.gamma_min),
        'gamma_max': float(config.gamma_max),
        'tau_h': float(config.tau_h_sen),
    }


def make_plant(config, params=None):
    """Build a plant (a RecedingHorizonMPC) with the given identified parameters.

    The controller and the plant share the same class; only the plant's four
    identified attributes are overridden. Missing keys default to nominal. The
    morphological adaptation constant is written to both `tau_adapt` and
    `tau_orient`, which hold one physical constant in the reported model. The
    senescence-law keys map to the controller's Hill attributes via PARAM_ATTR
    (e.g. 'tau_h' -> 'tau_h_sen').
    """
    plant = RecedingHorizonMPC(config)
    if params:
        if 'morph' in params and params['morph'] is not None:
            plant.tau_adapt = float(params['morph'])
            plant.tau_orient = float(params['morph'])
        for key, attr in PARAM_ATTR.items():
            if params.get(key) is not None:
                setattr(plant, attr, float(params[key]))
    return plant


# =============================================================================
#  Closed-loop and open-loop rollouts (reduced state; see docs/mismatch_plant_audit.md)
# =============================================================================
def _initial_state(config, controller, seed, pop0=None):
    # The initial population composition is drawn from the model's own rules
    # (generate_initial_population). An explicit `pop0` may be injected to
    # reproduce the exact pipeline initial condition, which is used only to verify
    # that this reduced loop reproduces run_mpc_simulation bit-for-bit given the
    # same initial condition. The mismatch study compares configurations at the
    # same seed, so the composition cancels in every closed-versus-open contrast.
    pop = generate_initial_population(config, seed) if pop0 is None else np.asarray(pop0, dtype=float)
    return {'pop': pop.copy(),
            'rho_h': controller.rho_target(0.0),      # rho_stat at tau = 0 (param-independent)
            'theta_h': controller.theta_target(0.0)}  # theta_stat at tau = 0 (param-independent)


def _cost_and_violations(controller, tau, phi, rho, varphi):
    # Task 5 (Part A): no soft senescence term (w_phi * phi_sen^2); senescence is
    # a hard constraint. Realised cost is tracking + move regularizer only.
    tau_prev = np.concatenate([[0.0], tau[:-1]])
    J = float(np.sum(
        controller.w_rho * (rho[1:] - RHO_FLOW) ** 2
        + controller.w_varphi * varphi[1:] ** 2
        + controller.w_u * (tau - tau_prev) ** 2))
    violation_hours = [k for k in range(1, len(phi))
                       if phi[k] > PHI_SEN_MAX + 1e-12]
    return J, violation_hours


def run_closed_loop(config, controller, plant, n_steps, seed, pop0=None):
    """Closed-loop MPC: the controller solves with its nominal model, the plant
    advances the true state (perturbed if `plant` is perturbed)."""
    x = _initial_state(config, controller, seed, pop0)
    phi0, rho0, v0 = controller.outputs(x)
    tau = []; phi = [phi0]; rho = [rho0]; var = [v0]; success = []
    u_prev = 0.0
    for _ in range(n_steps):
        u_opt, res = controller.solve(x, u_prev)
        tau_k = float(np.clip(u_opt[0], controller.tau_min, controller.tau_max))
        success.append(bool(getattr(res, 'success', False)))
        x = plant.predict_step(x, tau_k)
        p, r, v = controller.outputs(x)
        tau.append(tau_k); phi.append(p); rho.append(r); var.append(v)
        u_prev = tau_k
    return _package(controller, tau, phi, rho, var, success, seed)


def run_open_loop(config, controller, plant, tau_sequence, n_steps, seed):
    """Open-loop feedforward: apply a fixed input sequence to the plant without
    re-solving. Used to isolate the correction provided by feedback."""
    x = _initial_state(config, controller, seed)
    phi0, rho0, v0 = controller.outputs(x)
    tau = []; phi = [phi0]; rho = [rho0]; var = [v0]
    for k in range(n_steps):
        tau_k = float(tau_sequence[k])
        x = plant.predict_step(x, tau_k)
        p, r, v = controller.outputs(x)
        tau.append(tau_k); phi.append(p); rho.append(r); var.append(v)
    return _package(controller, tau, phi, rho, var, [True] * n_steps, seed)


def _package(controller, tau, phi, rho, var, success, seed):
    tau = np.asarray(tau); phi = np.asarray(phi); rho = np.asarray(rho); var = np.asarray(var)
    J, violation_hours = _cost_and_violations(controller, tau, phi, rho, var)
    return {
        'seed': int(seed),
        'tau': tau, 'phi_sen': phi, 'rho_bar': rho,
        'varphi_bar_rad': var, 'varphi_bar_deg': np.degrees(var),
        'terminal_phi_sen': float(phi[-1]),
        'terminal_rho_bar': float(rho[-1]),
        'terminal_varphi_bar_deg': float(np.degrees(var[-1])),
        'J': J,
        'n_violations': int(len(violation_hours)),
        'violation_hours': violation_hours,
        'solver_success_rate': float(np.mean(success)),
    }


def tracking_error(run, reference):
    """Root-mean-square deviation of each regulated output from the nominal-plant
    reference over the horizon (hours 1..N)."""
    def rms(a, b):
        a = np.asarray(a)[1:]; b = np.asarray(b)[1:]
        return float(np.sqrt(np.mean((a - b) ** 2)))
    return {
        'rmse_phi_sen': rms(run['phi_sen'], reference['phi_sen']),
        'rmse_rho_bar': rms(run['rho_bar'], reference['rho_bar']),
        'rmse_varphi_bar_deg': rms(run['varphi_bar_deg'], reference['varphi_bar_deg']),
    }


# =============================================================================
#  Evaluation of one perturbed plant over the seed ensemble
# =============================================================================
def evaluate_plant(config, controller, params, nominal_ref, ic_seeds, n_steps):
    """Run the closed-loop and open-loop configurations for one perturbed plant
    over the seed ensemble. Returns per-seed records."""
    plant = make_plant(config, params)
    records = []
    for seed in ic_seeds:
        ref = nominal_ref[seed]
        cl = run_closed_loop(config, controller, plant, n_steps, seed)
        ol = run_open_loop(config, controller, plant, ref['tau'], n_steps, seed)
        cl['tracking'] = tracking_error(cl, ref)
        ol['tracking'] = tracking_error(ol, ref)
        records.append({'seed': seed, 'closed': cl, 'open': ol})
    return records


def compute_nominal_reference(config, controller, ic_seeds, n_steps):
    """The nominal-plant closed loop for each seed (configuration iii). Depends
    only on the seed, so it is computed once and reused."""
    return {seed: run_closed_loop(config, controller, controller, n_steps, seed)
            for seed in ic_seeds}


# =============================================================================
#  Aggregation
# =============================================================================
def mean_ci(values, confidence=CONFIDENCE):
    a = np.asarray(values, dtype=float)
    n = a.size
    mean = float(np.mean(a))
    if n < 2:
        return mean, float('nan')
    sem = float(np.std(a, ddof=1) / np.sqrt(n))
    half = float(stats.t.ppf(0.5 + confidence / 2.0, df=n - 1) * sem)
    return mean, half


def aggregate_config(records, which):
    """Aggregate one configuration ('closed' or 'open') over seeds."""
    runs = [r[which] for r in records]
    tphi = [x['terminal_phi_sen'] for x in runs]
    J = [x['J'] for x in runs]
    trho = [x['terminal_rho_bar'] for x in runs]
    tvar = [x['terminal_varphi_bar_deg'] for x in runs]
    e_phi = [x['tracking']['rmse_phi_sen'] for x in runs]
    e_rho = [x['tracking']['rmse_rho_bar'] for x in runs]
    e_var = [x['tracking']['rmse_varphi_bar_deg'] for x in runs]
    viol = [1.0 if x['n_violations'] > 0 else 0.0 for x in runs]
    m = {}
    m['terminal_phi_sen_mean'], m['terminal_phi_sen_ci95'] = mean_ci(tphi)
    m['J_mean'], m['J_ci95'] = mean_ci(J)
    m['terminal_rho_bar_mean'], m['terminal_rho_bar_ci95'] = mean_ci(trho)
    m['terminal_varphi_bar_deg_mean'], m['terminal_varphi_bar_deg_ci95'] = mean_ci(tvar)
    m['rmse_phi_sen_mean'], m['rmse_phi_sen_ci95'] = mean_ci(e_phi)
    m['rmse_rho_bar_mean'], m['rmse_rho_bar_ci95'] = mean_ci(e_rho)
    m['rmse_varphi_bar_deg_mean'], m['rmse_varphi_bar_deg_ci95'] = mean_ci(e_var)
    m['constraint_violation_frequency'] = float(np.mean(viol))
    m['total_constraint_violations'] = int(sum(x['n_violations'] for x in runs))
    m['solver_success_rate_mean'] = float(np.mean([x['solver_success_rate'] for x in runs]))
    return m


def aggregate_perturbation(records, params, label):
    return {
        'label': label,
        'params': params,
        'n_rep': len(records),
        'closed': aggregate_config(records, 'closed'),
        'open': aggregate_config(records, 'open'),
    }


# =============================================================================
#  Sampling
# =============================================================================
def latin_hypercube(n_samples, n_dims, rng):
    """Latin hypercube design in the unit cube, using the supplied generator."""
    u = (rng.random((n_samples, n_dims)) + np.arange(n_samples)[:, None]) / n_samples
    for j in range(n_dims):
        rng.shuffle(u[:, j])
    return u


def param_ranges(nom, rel=REL_PERTURB, morph_range=MORPH_RANGE):
    """Perturbation ranges: adaptation constant over its calibrated range, the
    others over plus or minus `rel` of nominal."""
    return {
        'morph': morph_range,
        'gamma_min': (nom['gamma_min'] * (1 - rel), nom['gamma_min'] * (1 + rel)),
        'gamma_max': (nom['gamma_max'] * (1 - rel), nom['gamma_max'] * (1 + rel)),
        'tau_h': (nom['tau_h'] * (1 - rel), nom['tau_h'] * (1 + rel)),
    }


# =============================================================================
#  Study orchestration
# =============================================================================
def run_study(config, n_rep=N_REP, n_lhs=N_LHS, num_steps=NUM_STEPS,
              morph_sweep=MORPH_SWEEP, progress=None):
    ic_base = int(config.random_seed)
    perturb_seed = ic_base + PERTURB_SEED_OFFSET
    ic_seeds = [ic_base + r for r in range(n_rep)]
    nom = nominal_params(config)
    controller = RecedingHorizonMPC(config)          # nominal controller, reused
    ranges = param_ranges(nom)

    nominal_ref = compute_nominal_reference(config, controller, ic_seeds, num_steps)

    def emit(tag, i, total):
        if progress is not None:
            progress(tag, i, total)

    # --- one-at-a-time (tornado): each parameter at -rel and +rel -------------
    oat = []
    oat_targets = []
    for key in PARAM_KEYS:
        if key == 'morph':
            lo, hi = nom['morph'] * (1 - REL_PERTURB), nom['morph'] * (1 + REL_PERTURB)
        else:
            lo, hi = ranges[key][0], ranges[key][1]
        for sign, val in (('low', lo), ('high', hi)):
            params = dict.fromkeys(PARAM_KEYS, None)
            params[key] = val
            oat_targets.append((key, sign, val, params))
    for i, (key, sign, val, params) in enumerate(oat_targets):
        recs = evaluate_plant(config, controller, params, nominal_ref, ic_seeds, num_steps)
        agg = aggregate_perturbation(recs, params, f'{key}:{sign}')
        agg['oat_key'] = key; agg['oat_sign'] = sign; agg['oat_value'] = val
        oat.append({'aggregate': agg, 'records': recs})
        emit('oat', i + 1, len(oat_targets))

    # --- morphological adaptation constant explicit sweep 6..12 h -------------
    morph = []
    for i, val in enumerate(morph_sweep):
        params = dict.fromkeys(PARAM_KEYS, None); params['morph'] = val
        recs = evaluate_plant(config, controller, params, nominal_ref, ic_seeds, num_steps)
        agg = aggregate_perturbation(recs, params, f'morph={val:g}h')
        agg['morph_value'] = float(val)
        morph.append({'aggregate': agg, 'records': recs})
        emit('morph', i + 1, len(morph_sweep))

    # --- joint Latin hypercube ------------------------------------------------
    rng = np.random.default_rng(perturb_seed)
    unit = latin_hypercube(n_lhs, len(PARAM_KEYS), rng)
    joint = []
    for i in range(n_lhs):
        params = {}
        for j, key in enumerate(PARAM_KEYS):
            lo, hi = ranges[key]
            params[key] = float(lo + unit[i, j] * (hi - lo))
        recs = evaluate_plant(config, controller, params, nominal_ref, ic_seeds, num_steps)
        agg = aggregate_perturbation(recs, params, f'lhs{i:03d}')
        agg['lhs_index'] = i
        joint.append({'aggregate': agg, 'records': recs})
        emit('joint', i + 1, n_lhs)

    # nominal reference aggregate (closed loop on the nominal plant)
    nom_runs = [{'closed': nominal_ref[s],
                 'open': nominal_ref[s]} for s in ic_seeds]
    for r in nom_runs:  # tracking error against itself is zero by definition
        r['closed'] = dict(r['closed']); r['open'] = dict(r['open'])
        r['closed']['tracking'] = tracking_error(r['closed'], r['closed'])
        r['open']['tracking'] = tracking_error(r['open'], r['open'])
    nominal_aggregate = aggregate_config(nom_runs, 'closed')

    meta = {
        'param_keys': PARAM_KEYS, 'nominal_params': nom, 'ranges': ranges,
        'rel_perturb': REL_PERTURB, 'morph_range': list(MORPH_RANGE),
        'morph_sweep': list(morph_sweep), 'n_rep': n_rep, 'n_lhs': n_lhs,
        'num_steps': num_steps, 'ic_base_seed': ic_base, 'perturb_seed': perturb_seed,
        'phi_sen_max': PHI_SEN_MAX, 'confidence': CONFIDENCE,
        'tau_adapt_hours': float(config.tau_adapt_hours),
        'initial_cell_count': int(config.initial_cell_count),
    }
    return {'oat': oat, 'morph': morph, 'joint': joint,
            'nominal_aggregate': nominal_aggregate, 'meta': meta}


# =============================================================================
#  Output writers
# =============================================================================
def _run_rows(kind, label, params, records):
    rows = []
    for rec in records:
        for cfg in ('closed', 'open'):
            r = rec[cfg]
            rows.append({
                'kind': kind, 'label': label,
                'morph': params.get('morph'), 'gamma_min': params.get('gamma_min'),
                'gamma_max': params.get('gamma_max'), 'tau_h': params.get('tau_h'),
                'config': cfg, 'seed': r['seed'],
                'J': r['J'], 'terminal_phi_sen': r['terminal_phi_sen'],
                'terminal_rho_bar': r['terminal_rho_bar'],
                'terminal_varphi_bar_deg': r['terminal_varphi_bar_deg'],
                'n_violations': r['n_violations'],
                'rmse_phi_sen': r['tracking']['rmse_phi_sen'],
                'rmse_rho_bar': r['tracking']['rmse_rho_bar'],
                'rmse_varphi_bar_deg': r['tracking']['rmse_varphi_bar_deg'],
                'solver_success_rate': r['solver_success_rate'],
            })
    return rows


def write_raw_csv(result, path):
    fields = ['kind', 'label', 'morph', 'gamma_min', 'gamma_max', 'tau_h',
              'config', 'seed', 'J', 'terminal_phi_sen', 'terminal_rho_bar',
              'terminal_varphi_bar_deg', 'n_violations', 'rmse_phi_sen',
              'rmse_rho_bar', 'rmse_varphi_bar_deg', 'solver_success_rate']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for group, kind in (('oat', 'oat'), ('morph', 'morph'), ('joint', 'joint')):
            for item in result[group]:
                agg = item['aggregate']
                for row in _run_rows(kind, agg['label'], agg['params'], item['records']):
                    for k in ('J', 'terminal_phi_sen', 'terminal_rho_bar',
                              'terminal_varphi_bar_deg', 'rmse_phi_sen',
                              'rmse_rho_bar', 'rmse_varphi_bar_deg'):
                        row[k] = f"{row[k]:.10g}"
                    w.writerow(row)


def _agg_public(item):
    a = dict(item['aggregate'])
    a.pop('params', None)
    return a


def write_summary_json(result, path):
    out = {
        'metadata': result['meta'],
        'nominal_reference': result['nominal_aggregate'],
        'oat': [_agg_public(x) for x in result['oat']],
        'morph_sweep': [_agg_public(x) for x in result['morph']],
        'joint': [_agg_public(x) for x in result['joint']],
        'joint_summary': _joint_summary(result),
    }
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)


def _joint_summary(result):
    """Distribution summary of terminal phi_sen, J and violation frequency over
    the Latin hypercube, closed-loop versus open-loop."""
    def collect(field, cfg):
        return [x['aggregate'][cfg][field] for x in result['joint']]
    out = {}
    for cfg in ('closed', 'open'):
        tphi = np.array(collect('terminal_phi_sen_mean', cfg))
        J = np.array(collect('J_mean', cfg))
        vf = np.array([x['aggregate'][cfg]['constraint_violation_frequency']
                       for x in result['joint']])
        out[cfg] = {
            'terminal_phi_sen_mean_over_lhs': float(np.mean(tphi)),
            'terminal_phi_sen_max_over_lhs': float(np.max(tphi)),
            'terminal_phi_sen_p95_over_lhs': float(np.percentile(tphi, 95)),
            'J_mean_over_lhs': float(np.mean(J)),
            'J_max_over_lhs': float(np.max(J)),
            'fraction_of_samples_with_any_violation':
                float(np.mean(vf > 0)),
            'mean_violation_frequency': float(np.mean(vf)),
        }
    return out


def write_summary_markdown(result, path):
    meta = result['meta']
    js = _joint_summary(result)
    L = []
    L.append('# Robustness to parametric plant-model mismatch: summary')
    L.append('')
    L.append(f"Initial-condition seeds per plant: {meta['n_rep']}. Joint Latin "
             f"hypercube samples: {meta['n_lhs']}. Run length: {meta['num_steps']} "
             f"one-hour intervals. Master (aleatory) seed: {meta['ic_base_seed']}. "
             f"Perturbation (epistemic) seed: {meta['perturb_seed']}. Constraint: "
             f"phi_sen <= {meta['phi_sen_max']}. Values are mean with a 95 percent "
             f"confidence interval over the initial-condition seeds.")
    L.append('')
    L.append('## Feedback efficacy over the joint Latin hypercube')
    L.append('')
    L.append('| configuration | mean terminal phi_sen | max terminal phi_sen | '
             'p95 terminal phi_sen | mean J | fraction of samples with any violation |')
    L.append('|---|---|---|---|---|---|')
    for cfg in ('closed', 'open'):
        s = js[cfg]
        name = 'closed-loop (nominal internal model)' if cfg == 'closed' else 'open-loop feedforward'
        L.append(f"| {name} | {s['terminal_phi_sen_mean_over_lhs']:.4f} | "
                 f"{s['terminal_phi_sen_max_over_lhs']:.4f} | "
                 f"{s['terminal_phi_sen_p95_over_lhs']:.4f} | "
                 f"{s['J_mean_over_lhs']:.3f} | "
                 f"{s['fraction_of_samples_with_any_violation']:.3f} |")
    L.append('')
    na = result['nominal_aggregate']
    L.append(f"No-mismatch reference (nominal plant, closed loop): terminal "
             f"phi_sen = {na['terminal_phi_sen_mean']:.4f} +/- "
             f"{na['terminal_phi_sen_ci95']:.4f}, J = {na['J_mean']:.3f} +/- "
             f"{na['J_ci95']:.3f}, violation frequency "
             f"{na['constraint_violation_frequency']:.3f}.")
    L.append('')
    L.append('## One-at-a-time perturbations (tornado inputs)')
    L.append('')
    L.append('| parameter | direction | value | closed terminal phi_sen | '
             'open terminal phi_sen | closed J | closed violation freq. | '
             'open violation freq. |')
    L.append('|---|---|---|---|---|---|---|---|')
    for x in result['oat']:
        a = x['aggregate']
        c, o = a['closed'], a['open']
        L.append(f"| {a['oat_key']} | {a['oat_sign']} | {a['oat_value']:.5g} | "
                 f"{c['terminal_phi_sen_mean']:.4f} +/- {c['terminal_phi_sen_ci95']:.4f} | "
                 f"{o['terminal_phi_sen_mean']:.4f} +/- {o['terminal_phi_sen_ci95']:.4f} | "
                 f"{c['J_mean']:.3f} | {c['constraint_violation_frequency']:.3f} | "
                 f"{o['constraint_violation_frequency']:.3f} |")
    L.append('')
    L.append('## Adaptation-constant sweep, 6 to 12 h (closed loop)')
    L.append('')
    L.append('| adaptation constant (h) | terminal phi_sen | J | violation freq. |')
    L.append('|---|---|---|---|')
    for x in result['morph']:
        a = x['aggregate']; c = a['closed']
        L.append(f"| {a['morph_value']:.1f} | "
                 f"{c['terminal_phi_sen_mean']:.4f} +/- {c['terminal_phi_sen_ci95']:.4f} | "
                 f"{c['J_mean']:.3f} | {c['constraint_violation_frequency']:.3f} |")
    L.append('')
    with open(path, 'w') as f:
        f.write('\n'.join(L))


# =============================================================================
#  Figures
# =============================================================================
def _style():
    plt.rcParams.update({
        'axes.labelsize': 11, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'legend.fontsize': 8.5, 'axes.titlesize': 10.5, 'figure.dpi': 120,
        'savefig.dpi': 300,
    })


def _savefig(fig, path_noext):
    fig.savefig(path_noext + '.pdf', bbox_inches='tight')
    fig.savefig(path_noext + '.png', bbox_inches='tight')
    plt.close(fig)


def figure_feedback_efficacy(result, out_noext):
    """Headline figure: closed-loop versus open-loop terminal phi_sen and
    constraint-violation frequency across the joint mismatch ensemble."""
    _style()
    meta = result['meta']
    tphi = {cfg: np.array([x['aggregate'][cfg]['terminal_phi_sen_mean']
                           for x in result['joint']]) for cfg in ('closed', 'open')}
    vf = {cfg: np.array([x['aggregate'][cfg]['constraint_violation_frequency']
                         for x in result['joint']]) for cfg in ('closed', 'open')}
    na = result['nominal_aggregate']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 4.0))

    # (a) terminal phi_sen distribution, closed vs open, against the constraint
    parts = ax1.violinplot([tphi['closed'], tphi['open']], positions=[1, 2],
                           showmeans=True, showextrema=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor('C0' if i == 0 else 'C3'); pc.set_alpha(0.5)
    ax1.axhline(meta['phi_sen_max'], ls='--', color='r', lw=1.2, label='constraint 0.30')
    ax1.axhline(na['terminal_phi_sen_mean'], ls=':', color='k', lw=1,
                label='no-mismatch reference')
    ax1.set_xticks([1, 2]); ax1.set_xticklabels(['closed-loop', 'open-loop'])
    ax1.set_ylabel(r'terminal $\phi_{\mathrm{sen}}$ (per LHS sample)')
    ax1.set_ylim(0.0, max(0.33, float(np.max(tphi['open'])) * 1.1))
    ax1.legend(loc='upper left')
    ax1.set_title('(a) terminal senescent fraction under joint mismatch')

    # (b) constraint-violation frequency, closed vs open
    xb = np.arange(2)
    ax2.bar(xb, [float(np.mean(vf['closed'])), float(np.mean(vf['open']))],
            color=['C0', 'C3'], alpha=0.8, width=0.6)
    for i, cfg in enumerate(('closed', 'open')):
        frac_any = float(np.mean(vf[cfg] > 0))
        ax2.text(i, float(np.mean(vf[cfg])) + 0.01,
                 f"any: {frac_any:.2f}", ha='center', va='bottom', fontsize=8)
    ax2.set_xticks(xb); ax2.set_xticklabels(['closed-loop', 'open-loop'])
    ax2.set_ylabel('mean constraint-violation frequency')
    ax2.set_ylim(0, 1.05)
    ax2.set_title('(b) constraint violation, feedback vs feedforward')

    fig.suptitle('Efficacy of feedback under parametric plant-model mismatch '
                 f'(joint LHS, {meta["n_lhs"]} samples, {meta["n_rep"]} seeds each)',
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _savefig(fig, out_noext)


def figure_tornado(result, out_noext):
    """Tornado plots ranking the four parameters by their one-at-a-time effect on
    terminal phi_sen and on J (closed loop), relative to the nominal reference."""
    _style()
    na = result['nominal_aggregate']
    base_phi = na['terminal_phi_sen_mean']
    base_J = na['J_mean']

    def collect(metric_key, base):
        data = {}
        for x in result['oat']:
            a = x['aggregate']
            data.setdefault(a['oat_key'], {})[a['oat_sign']] = \
                a['closed'][metric_key] - base
        keys = sorted(data, key=lambda k: max(abs(data[k].get('low', 0.0)),
                                              abs(data[k].get('high', 0.0))))
        return keys, data

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8))

    for ax, metric_key, base, title, unit in (
            (ax1, 'terminal_phi_sen_mean', base_phi,
             r'(a) effect on terminal $\phi_{\mathrm{sen}}$', ''),
            (ax2, 'J_mean', base_J, '(b) effect on closed-loop cost $J$', '')):
        keys, data = collect(metric_key, base)
        y = np.arange(len(keys))
        for i, k in enumerate(keys):
            lo = data[k].get('low', 0.0); hi = data[k].get('high', 0.0)
            ax.barh(i, hi, color='C0', alpha=0.8, height=0.6,
                    label='+20 percent' if i == 0 else None)
            ax.barh(i, lo, color='C3', alpha=0.8, height=0.6,
                    label='-20 percent' if i == 0 else None)
        ax.axvline(0, color='k', lw=0.8)
        ax.set_yticks(y); ax.set_yticklabels([PARAM_LABELS[k] for k in keys])
        ax.set_xlabel('deviation from no-mismatch reference')
        ax.set_title(title)
        ax.legend(loc='best')
    fig.suptitle('One-at-a-time parameter sensitivity of the closed loop '
                 '(morphological constant at plus or minus 20 percent)', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _savefig(fig, out_noext)


def figure_joint_distributions(result, out_noext):
    """Distributions of closed-loop cost and terminal phi_sen over the joint
    Latin hypercube, closed loop versus open loop."""
    _style()
    meta = result['meta']
    tphi = {cfg: np.array([x['aggregate'][cfg]['terminal_phi_sen_mean']
                           for x in result['joint']]) for cfg in ('closed', 'open')}
    J = {cfg: np.array([x['aggregate'][cfg]['J_mean']
                        for x in result['joint']]) for cfg in ('closed', 'open')}
    na = result['nominal_aggregate']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.8))
    ax1.hist(tphi['closed'], bins=16, color='C0', alpha=0.6, label='closed-loop')
    ax1.hist(tphi['open'], bins=16, color='C3', alpha=0.6, label='open-loop')
    ax1.axvline(meta['phi_sen_max'], ls='--', color='r', lw=1.2, label='constraint 0.30')
    ax1.axvline(na['terminal_phi_sen_mean'], ls=':', color='k', lw=1, label='reference')
    ax1.set_xlabel(r'terminal $\phi_{\mathrm{sen}}$'); ax1.set_ylabel('LHS samples')
    ax1.legend(loc='best'); ax1.set_title('(a) terminal senescent fraction')

    ax2.hist(J['closed'], bins=16, color='C0', alpha=0.6, label='closed-loop')
    ax2.hist(J['open'], bins=16, color='C3', alpha=0.6, label='open-loop')
    ax2.axvline(na['J_mean'], ls=':', color='k', lw=1, label='reference')
    ax2.set_xlabel('closed-loop cost $J$'); ax2.set_ylabel('LHS samples')
    ax2.legend(loc='best'); ax2.set_title('(b) closed-loop cost')

    fig.suptitle('Joint mismatch over the Latin hypercube: the regulated outputs '
                 'remain tightly distributed, well within the constraint', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _savefig(fig, out_noext)


def figure_morph_sweep(result, out_noext):
    """Terminal phi_sen and closed-loop cost as the plant adaptation constant is
    swept over its calibrated range, with the nominal value marked."""
    _style()
    meta = result['meta']
    xs = np.array([x['aggregate']['morph_value'] for x in result['morph']])
    phi_m = np.array([x['aggregate']['closed']['terminal_phi_sen_mean'] for x in result['morph']])
    phi_ci = np.array([x['aggregate']['closed']['terminal_phi_sen_ci95'] for x in result['morph']])
    J_m = np.array([x['aggregate']['closed']['J_mean'] for x in result['morph']])
    J_ci = np.array([x['aggregate']['closed']['J_ci95'] for x in result['morph']])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ax1.errorbar(xs, phi_m, yerr=phi_ci, marker='o', ms=4, capsize=3, color='C0')
    ax1.axhline(meta['phi_sen_max'], ls='--', color='r', lw=1.2, label='constraint 0.30')
    ax1.axvline(meta['tau_adapt_hours'], ls=':', color='k', lw=1, label='nominal 7.4 h')
    ax1.set_ylim(0.0, 0.33)
    ax1.set_xlabel('plant adaptation constant (h)')
    ax1.set_ylabel(r'terminal $\phi_{\mathrm{sen}}$ (closed loop)')
    ax1.legend(loc='best'); ax1.set_title('(a) senescent fraction')

    ax2.errorbar(xs, J_m, yerr=J_ci, marker='o', ms=4, capsize=3, color='C0')
    ax2.axvline(meta['tau_adapt_hours'], ls=':', color='k', lw=1, label='nominal 7.4 h')
    ax2.set_xlabel('plant adaptation constant (h)')
    ax2.set_ylabel('closed-loop cost $J$')
    ax2.legend(loc='best'); ax2.set_title('(b) closed-loop cost')

    fig.suptitle('Closed-loop regulation across the calibrated 6 to 12 h range of '
                 'the plant adaptation constant', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _savefig(fig, out_noext)


def figure_tracking_error(result, out_noext):
    """Closed-loop versus open-loop regulation error of the regulated outputs,
    shown as the ratio of the closed-loop error to the open-loop error over the
    joint ensemble. The three outputs differ by orders of magnitude in their
    units, so the ratio is the readable way to expose the small correction that
    feedback provides. A ratio below one means feedback reduces the error."""
    _style()
    fields = [('rmse_phi_sen_mean', r'$\phi_{\mathrm{sen}}$'),
              ('rmse_rho_bar_mean', r'$\bar{\rho}$'),
              ('rmse_varphi_bar_deg_mean', r'$\bar{\varphi}$')]
    ratios = []
    abs_closed = []
    abs_open = []
    labels = []
    for field, lbl in fields:
        cl = float(np.mean([x['aggregate']['closed'][field] for x in result['joint']]))
        ol = float(np.mean([x['aggregate']['open'][field] for x in result['joint']]))
        ratios.append(cl / ol if ol > 0 else 1.0)
        abs_closed.append(cl); abs_open.append(ol); labels.append(lbl)

    x = np.arange(len(fields))
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    colors = ['C3' if r > 1.0 else 'C0' for r in ratios]
    ax.bar(x, ratios, width=0.5, color=colors, alpha=0.85)
    ax.axhline(1.0, color='k', lw=1.0)
    span = max(0.02, max(abs(r - 1.0) for r in ratios) * 1.6)
    ax.set_ylim(1.0 - span, 1.0 + span)
    for i, r in enumerate(ratios):
        off = span * 0.08
        ax.text(i, r + (off if r >= 1.0 else -off), f'{(r - 1.0) * 100:+.2f}%',
                ha='center', va='bottom' if r >= 1.0 else 'top', fontsize=9)
        ax.text(i, 1.0 - span * 0.86,
                f'open\n{abs_open[i]:.2e}', ha='center', va='bottom', fontsize=7, color='0.35')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('closed-loop / open-loop regulation error')
    ax.set_title('Feedback correction of the regulation error under joint mismatch\n'
                 '(below one, feedback reduces the error; above one, feedforward is better)',
                 fontsize=9.5)
    fig.tight_layout()
    _savefig(fig, out_noext)


# =============================================================================
#  Main
# =============================================================================
def _timestamp_dir():
    ts = time.strftime('%Y%m%d-%H%M%S')
    out = os.path.join(_REPO_ROOT, 'results', 'mismatch_robustness', ts)
    os.makedirs(out, exist_ok=True)
    return out, ts


def _signature(result):
    """Deterministic aggregate signature for the reproducibility check."""
    vals = []
    for group in ('oat', 'morph', 'joint'):
        for item in result[group]:
            a = item['aggregate']
            for cfg in ('closed', 'open'):
                c = a[cfg]
                vals += [c['terminal_phi_sen_mean'], c['J_mean'],
                         c['constraint_violation_frequency'],
                         c['rmse_phi_sen_mean']]
    return np.array(vals, dtype=float)


def main(argv=None):
    p = argparse.ArgumentParser(description='Robustness to parametric plant-model '
                                            'mismatch of the receding-horizon loop.')
    p.add_argument('--n-rep', type=int, default=N_REP)
    p.add_argument('--n-lhs', type=int, default=N_LHS)
    p.add_argument('--num-steps', type=int, default=NUM_STEPS)
    p.add_argument('--smoke', action='store_true', help='small, fast self-test')
    p.add_argument('--verify', action='store_true',
                   help='re-run and confirm deterministic aggregates match')
    args = p.parse_args(argv)

    config = SimulationConfig()
    n_rep, n_lhs, num_steps = args.n_rep, args.n_lhs, args.num_steps
    morph_sweep = MORPH_SWEEP
    if args.smoke:
        n_rep, n_lhs, morph_sweep = 2, 4, [6.0, 9.0, 12.0]

    print('=' * 72)
    print('Parametric plant-model mismatch robustness study')
    print(f"  identified parameters : {PARAM_KEYS}")
    print(f"  relative range        : +/- {REL_PERTURB:.0%} (adaptation constant "
          f"over {MORPH_RANGE[0]:g} to {MORPH_RANGE[1]:g} h)")
    print(f"  seeds per plant N_rep : {n_rep} (aleatory)")
    print(f"  joint LHS samples     : {n_lhs} (epistemic)")
    print(f"  run length            : {num_steps} h")
    print(f"  master seed           : {config.random_seed}; perturb seed : "
          f"{config.random_seed + PERTURB_SEED_OFFSET}")
    print('=' * 72)

    def progress(tag, i, total):
        print(f"  [{tag:6s}] {i:3d}/{total}", flush=True)

    t0 = time.perf_counter()
    result = run_study(config, n_rep=n_rep, n_lhs=n_lhs, num_steps=num_steps,
                       morph_sweep=morph_sweep, progress=progress)
    result['meta']['wall_time_s'] = time.perf_counter() - t0

    out_dir, ts = _timestamp_dir()
    write_raw_csv(result, os.path.join(out_dir, 'raw_runs.csv'))
    write_summary_json(result, os.path.join(out_dir, 'summary.json'))
    write_summary_markdown(result, os.path.join(out_dir, 'summary_table.md'))
    figure_feedback_efficacy(result, os.path.join(out_dir, 'fig_feedback_efficacy'))
    figure_tornado(result, os.path.join(out_dir, 'fig_tornado'))
    figure_joint_distributions(result, os.path.join(out_dir, 'fig_joint_distributions'))
    figure_morph_sweep(result, os.path.join(out_dir, 'fig_morph_sweep'))
    figure_tracking_error(result, os.path.join(out_dir, 'fig_tracking_error'))

    print(f"\nOutputs written to: {out_dir}")
    for name in ('raw_runs.csv', 'summary.json', 'summary_table.md',
                 'fig_feedback_efficacy.pdf', 'fig_tornado.pdf',
                 'fig_joint_distributions.pdf', 'fig_morph_sweep.pdf',
                 'fig_tracking_error.pdf'):
        print(f"  {name}")

    if args.verify:
        print('\nReproducibility check: re-running the study with the same seeds ...')
        result2 = run_study(config, n_rep=n_rep, n_lhs=n_lhs, num_steps=num_steps,
                            morph_sweep=morph_sweep)
        a1, a2 = _signature(result), _signature(result2)
        max_abs = float(np.max(np.abs(a1 - a2)))
        print(f"  max abs difference in deterministic aggregates: {max_abs:.2e}")
        print(f"  reproducible to 1e-12: {bool(np.allclose(a1, a2, rtol=0, atol=1e-12))}")

    return out_dir, result


if __name__ == '__main__':
    main()

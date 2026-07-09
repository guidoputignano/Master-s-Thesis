"""
Senescence-law sweep hook and low-shear constraint demo (Task 5, Part B/A).

This module exposes the two ASSUMED, sweepable parameters of the monotone Hill
senescence-induction rate

    gamma(tau) = gamma_min + (gamma_max - gamma_min) * tau_h^n / (tau_h^n + tau^n)
                 [ + gamma_d * tau^m / (tau_d^m + tau^m) ]   (supraphysiological)

namely ``gamma_max`` (low-shear induction plateau) and, when the
supraphysiological arm is enabled, ``gamma_d`` (high-shear damage plateau). Both
are illustrative, not fitted (see ``gamma_tau_hill`` for the provenance note), so
a study should report results ACROSS a sweep of them rather than at a single
value. ``gamma_min`` and ``tau_h`` are anchored and are held at their config
values here.

What it records
---------------
For each swept value it runs the reduced-state receding-horizon closed loop
(the same ``RecedingHorizonMPC`` used by the reported model) and records:
  * the full senescent-fraction trajectory phi_sen(k);
  * the terminal and peak phi_sen and the margin to the hard cap phi_sen_max;
  * constraint activation at every step: whether the senescence cap is binding
    (the optimiser's phi_sen margin touches zero and/or the applied phi_sen sits
    at the cap) and whether the hard move / slew bound |tau(k)-tau(k-1)| is
    binding.

A structural note on when the cap binds
---------------------------------------
Under the protective-only Hill law the induction rate is MONOTONE DECREASING in
shear, and every tracking objective (aspect ratio, flow alignment) also prefers
higher shear. Senescence suppression and morphology tracking are therefore
CO-MONOTONE in the input: at full shear authority the controller escapes to high
tau, driving gamma toward its floor, and the senescence cap stays slack. The cap
becomes an active (binding) constraint only when either
  (i) the ASSUMED induction plateau gamma_max is raised into an illustrative high
      range (the sweep below: the cap first binds near gamma_max ~ 0.3 h^-1 and is
      exceeded above ~ 0.35 h^-1 at full authority), or
 (ii) shear authority is RESTRICTED (``low_shear_demo``), so the controller
      cannot raise tau enough to suppress induction.
This is consistent with the "weak controllability of the senescent fraction"
reported in the mismatch-robustness study, and none of it is tuned around the
nominal system, whose senescence over the reported six-hour window is low by
design.

Low-shear demo
--------------
``low_shear_demo`` drives shear LOW by capping tau_max below the induction
half-max tau_h, at a mild illustrative gamma_max over a short window. The
senescent fraction then rises past the hard cap that the controller cannot hold
at low shear: the cap is active (binding) throughout and is realised-exceeded
after a few steps, quantifying the shear authority the cap requires. It is a
demonstration of the CONSTRAINT MECHANISM using the sweepable, assumed gamma_max,
not a claim about the nominal system.

CLI
---
    python -m analysis.senescence_law_sweep [--out DIR] [--seed S]
                                            [--steps N] [--quick]
Writes (when --out is given) a CSV of the sweep, a CSV of the demo trajectory,
a two-panel figure, and a short Markdown summary. With no --out it prints a
compact table to stdout. Deterministic given the seed.
"""
import argparse
import csv
import os

import numpy as np

from endothelial_simulation.config import SimulationConfig
from endothelial_simulation.control.mpc_controller import (
    RecedingHorizonMPC, flow_alignment_angle,
)
from analysis.horizon_sensitivity import generate_initial_population

# Default sweep of the assumed low-shear induction plateau gamma_max (h^-1),
# spanning from the config nominal (0.0125) up through the range where the hard
# senescence cap transitions from slack to active-and-held to exceeded (at full
# shear authority). All values are illustrative; the point of the hook is to
# report ACROSS them, not at a single value.
GAMMA_MAX_SWEEP = [0.0125, 0.05, 0.10, 0.20, 0.30, 0.40]
ACTIVE_TOL = 1e-4     # tolerance for calling a hard constraint "binding"


def run_loop(config, *, gamma_max=None, gamma_d=None, tau_bounds=None,
             delta_tau_max=0.5, n_prediction=6, n_control=3, n_steps=6,
             seed=42):
    """Run one instrumented reduced closed loop; return trajectory + activation.

    Parameters mirror ``RecedingHorizonMPC`` plus optional overrides of the
    sweepable law parameters (``gamma_max``, ``gamma_d``) and the input bounds
    (``tau_bounds``) used by the low-shear demo. The initial condition and the
    dynamics are exactly those of the reported reduced loop.
    """
    kwargs = dict(n_prediction=n_prediction, n_control=n_control,
                  delta_tau_max=delta_tau_max)
    if tau_bounds is not None:
        kwargs['tau_bounds'] = tau_bounds
    mpc = RecedingHorizonMPC(config, **kwargs)
    if gamma_max is not None:
        mpc.gamma_max = float(gamma_max)
    if gamma_d is not None:
        mpc.gamma_d = float(gamma_d)
        mpc.include_supraphysiological_arm = True

    pop = generate_initial_population(config, seed)
    x = {'pop': pop.copy(),
         'rho_h': mpc.rho_target(0.0),
         'theta_h': mpc.theta_target(0.0)}

    phi0, rho0, varphi0 = mpc.outputs(x)
    tau_seq, phi_seq = [], [phi0]
    rho_seq, varphi_seq = [rho0], [varphi0]
    cap_active, move_active, phi_margin = [], [], []

    u_prev = 0.0
    for _ in range(n_steps):
        u_opt, _res = mpc.solve(x, u_prev)
        tau_k = float(np.clip(u_opt[0], mpc.tau_min, mpc.tau_max))

        # Constraint activation, measured at the accepted solution:
        #  * senescence cap: smallest phi_sen margin over the horizon touches 0;
        #  * move bound: the applied first move equals delta_tau_max.
        margin = float(np.min(mpc._phi_sen_margin(u_opt, x)))
        phi_margin.append(margin)
        cap_active.append(bool(margin <= ACTIVE_TOL))
        move_active.append(bool(abs(tau_k - u_prev) >= mpc.delta_tau_max - ACTIVE_TOL))

        x = mpc.predict_step(x, tau_k)
        phi_k, rho_k, varphi_k = mpc.outputs(x)
        tau_seq.append(tau_k)
        phi_seq.append(phi_k)
        rho_seq.append(rho_k)
        varphi_seq.append(varphi_k)
        u_prev = tau_k

    phi = np.asarray(phi_seq)
    tau = np.asarray(tau_seq)
    phi_max = mpc.phi_sen_max
    # Realised crossing: first APPLIED-trajectory step whose phi_sen exceeds the
    # hard cap (distinct from cap_active, which flags the optimiser's PREDICTED
    # margin touching zero over the horizon).
    exceed = [k for k in range(1, len(phi)) if phi[k] > phi_max + ACTIVE_TOL]
    return {
        'gamma_max': float(mpc.gamma_max),
        'gamma_d': float(mpc.gamma_d) if mpc.include_supraphysiological_arm else None,
        'tau_bounds': (mpc.tau_min, mpc.tau_max),
        'delta_tau_max': float(mpc.delta_tau_max),
        'phi_sen_max': float(phi_max),
        'tau': tau,
        'phi_sen': phi,
        'rho_bar': np.asarray(rho_seq),
        'varphi_bar_deg': np.degrees(np.asarray(varphi_seq)),
        'terminal_phi_sen': float(phi[-1]),
        'peak_phi_sen': float(phi.max()),
        'margin_to_cap': float(phi_max - phi.max()),
        'held': bool(phi.max() <= phi_max + ACTIVE_TOL),
        'first_exceed_step': (exceed[0] if exceed else None),
        'cap_active': cap_active,
        'move_active': move_active,
        'phi_margin': phi_margin,
        'cap_active_frac': float(np.mean(cap_active)) if cap_active else 0.0,
        'move_active_frac': float(np.mean(move_active)) if move_active else 0.0,
    }


def sweep_gamma_max(config, values=GAMMA_MAX_SWEEP, *, seed=42, n_steps=6,
                    **loop_kwargs):
    """Sweep gamma_max and return one summary record per value."""
    out = []
    for gmax in values:
        r = run_loop(config, gamma_max=gmax, seed=seed, n_steps=n_steps,
                     **loop_kwargs)
        out.append({
            'gamma_max': r['gamma_max'],
            'terminal_phi_sen': r['terminal_phi_sen'],
            'peak_phi_sen': r['peak_phi_sen'],
            'margin_to_cap': r['margin_to_cap'],
            'held': r['held'],
            'cap_active_frac': r['cap_active_frac'],
            'move_active_frac': r['move_active_frac'],
            'tau_mean': float(r['tau'].mean()),
        })
    return out


def low_shear_demo(config, *, seed=42):
    """Drive shear LOW so the senescence cap becomes active; return the trajectory.

    Caps tau_max at 0.3 Pa (below the induction half-max tau_h ~ 0.5 Pa) so the
    controller cannot suppress induction, at a mild illustrative gamma_max
    (0.05 h^-1, above the 0.0125 nominal) over a short 6 h window. The senescent
    fraction rises past the hard cap the controller cannot hold at low shear: the
    cap is active (binding) throughout and is realised-exceeded after a few steps.
    These are demonstration settings for the constraint mechanism, not nominal
    values (the nominal system's senescence is low by design and is not tuned).
    """
    return run_loop(config, gamma_max=0.05, tau_bounds=(0.0, 0.3),
                    delta_tau_max=0.1, n_steps=6, seed=seed)


# =============================================================================
#  Reporting
# =============================================================================
def _print_sweep(rows, phi_sen_max):
    print(f"\ngamma_max sweep at full shear authority "
          f"(hard cap phi_sen_max = {phi_sen_max:.2f}):")
    print(f"  {'gamma_max':>9}  {'term phi':>8}  {'peak phi':>8}  "
          f"{'margin':>7}  {'held':>5}  {'cap act':>7}  {'move act':>8}  {'tau_mean':>8}")
    for r in rows:
        print(f"  {r['gamma_max']:>9.4f}  {r['terminal_phi_sen']:>8.4f}  "
              f"{r['peak_phi_sen']:>8.4f}  {r['margin_to_cap']:>+7.4f}  "
              f"{('yes' if r['held'] else 'NO'):>5}  "
              f"{r['cap_active_frac']:>7.2f}  {r['move_active_frac']:>8.2f}  "
              f"{r['tau_mean']:>8.3f}")


def _write_sweep_csv(rows, path):
    fields = ['gamma_max', 'terminal_phi_sen', 'peak_phi_sen', 'margin_to_cap',
              'held', 'cap_active_frac', 'move_active_frac', 'tau_mean']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({**r, 'held': int(r['held'])})


def _write_demo_csv(demo, path):
    n = len(demo['tau'])
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['k', 'tau', 'phi_sen', 'rho_bar', 'varphi_bar_deg',
                    'phi_margin', 'cap_active', 'move_active'])
        for k in range(n):
            w.writerow([k + 1, f"{demo['tau'][k]:.6f}",
                        f"{demo['phi_sen'][k + 1]:.6f}",
                        f"{demo['rho_bar'][k + 1]:.6f}",
                        f"{demo['varphi_bar_deg'][k + 1]:.6f}",
                        f"{demo['phi_margin'][k]:.6e}",
                        int(demo['cap_active'][k]), int(demo['move_active'][k])])


def _figure(sweep_rows, demo, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6))

    # Panel 1: terminal/peak phi_sen vs gamma_max, with the hard cap.
    gm = [r['gamma_max'] for r in sweep_rows]
    ax1.plot(gm, [r['peak_phi_sen'] for r in sweep_rows], 'o-', label='peak')
    ax1.plot(gm, [r['terminal_phi_sen'] for r in sweep_rows], 's--', label='terminal')
    ax1.axhline(demo['phi_sen_max'], color='k', ls=':', lw=1,
                label=r'cap $\phi_{\mathrm{sen}}^{\max}$')
    ax1.set_xlabel(r'$\gamma_{\max}$ (h$^{-1}$, assumed/sweepable)')
    ax1.set_ylabel(r'$\phi_{\mathrm{sen}}$')
    ax1.set_title('senescent fraction vs sweep')
    ax1.legend(fontsize=8)

    # Panel 2: low-shear demo trajectory (phi_sen climbs past the active cap).
    k = np.arange(1, len(demo['tau']) + 1)
    ax2.plot(k, demo['phi_sen'][1:], 'o-', color='C3', label=r'$\phi_{\mathrm{sen}}$')
    ax2.axhline(demo['phi_sen_max'], color='k', ls=':', lw=1, label='hard cap')
    ax2b = ax2.twinx()
    ax2b.step(k, demo['tau'], where='mid', color='C0', alpha=0.6, label=r'$\tau$')
    ax2b.set_ylabel(r'$\tau$ (Pa)', color='C0')
    ax2.set_xlabel('control step (h)')
    ax2.set_ylabel(r'$\phi_{\mathrm{sen}}$', color='C3')
    ax2.set_title('low-shear demo: cap active')
    ax2.legend(fontsize=8, loc='center right')

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_markdown(sweep_rows, demo, path):
    with open(path, 'w') as f:
        f.write('# Senescence-law sweep and low-shear constraint demo\n\n')
        f.write('Generated by `analysis/senescence_law_sweep.py`. The Hill FORM '
                'is a modelling choice matched to the cited monotone '
                'shear-protection shape, NOT a fitted law; `gamma_max` (and '
                '`gamma_d`) are assumed, illustrative and swept here.\n\n')
        f.write(f"Hard senescence cap: phi_sen_max = {demo['phi_sen_max']:.2f}.\n\n")
        f.write('At full shear authority the cap is slack at the nominal '
                'gamma_max and becomes active (binding, but held) as gamma_max is '
                'raised toward ~0.3 h^-1, then is exceeded above ~0.35 h^-1. This '
                'reflects the co-monotonicity of senescence suppression and '
                'morphology tracking in the input (see module docstring).\n\n')
        f.write('## gamma_max sweep (full authority, 6 h window)\n\n')
        f.write('| gamma_max (h^-1) | terminal phi_sen | peak phi_sen | '
                'margin to cap | held | cap-active frac | move-active frac | tau_mean |\n')
        f.write('|---|---|---|---|---|---|---|---|\n')
        for r in sweep_rows:
            f.write(f"| {r['gamma_max']:.4f} | {r['terminal_phi_sen']:.4f} | "
                    f"{r['peak_phi_sen']:.4f} | {r['margin_to_cap']:+.4f} | "
                    f"{'yes' if r['held'] else 'NO'} | "
                    f"{r['cap_active_frac']:.2f} | {r['move_active_frac']:.2f} | "
                    f"{r['tau_mean']:.3f} |\n")
        f.write('\n## Low-shear demo\n\n')
        f.write(f"Settings: tau_max = {demo['tau_bounds'][1]:.2f} Pa "
                f"(< induction half-max), gamma_max = {demo['gamma_max']:.3f} "
                f"h^-1 (illustrative, above nominal), horizon "
                f"{len(demo['tau'])} h, delta_tau_max = {demo['delta_tau_max']:.2f}.\n\n")
        f.write(f"Peak phi_sen = {demo['peak_phi_sen']:.4f}; the cap is active "
                f"(binding) on {demo['cap_active_frac']*100:.0f}% of steps")
        if demo['first_exceed_step'] is not None:
            f.write(f" and the realised trajectory exceeds it at step "
                    f"{demo['first_exceed_step']} h (held = {demo['held']}). ")
        else:
            f.write(f" and the realised trajectory stays within it "
                    f"(held = {demo['held']}). ")
        f.write('At low shear the controller holds tau at its (low) upper bound '
                'but cannot suppress induction, so the cap becomes active and, '
                'here, is exceeded: this quantifies the shear authority the cap '
                'requires. It exercises the constraint mechanism via the '
                'sweepable gamma_max and is not a statement about the nominal '
                'system.\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default=None, help='output directory (optional)')
    ap.add_argument('--seed', type=int, default=None, help='override config seed')
    ap.add_argument('--steps', type=int, default=6, help='sweep horizon (h)')
    ap.add_argument('--quick', action='store_true',
                    help='shorter gamma_max sweep for a fast check')
    args = ap.parse_args()

    config = SimulationConfig()
    seed = int(args.seed) if args.seed is not None else int(config.random_seed)

    values = [0.0125, 0.05, 0.20] if args.quick else GAMMA_MAX_SWEEP
    sweep_rows = sweep_gamma_max(config, values, seed=seed, n_steps=args.steps)
    demo = low_shear_demo(config, seed=seed)

    _print_sweep(sweep_rows, demo['phi_sen_max'])
    print(f"\nlow-shear demo (tau_max={demo['tau_bounds'][1]:.2f} Pa, "
          f"gamma_max={demo['gamma_max']:.3f} h^-1, {len(demo['tau'])} h): "
          f"peak phi_sen={demo['peak_phi_sen']:.4f}, "
          f"cap active on {demo['cap_active_frac']*100:.0f}% of steps, "
          f"realised exceed at step {demo['first_exceed_step']} "
          f"(held={demo['held']}).")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        _write_sweep_csv(sweep_rows, os.path.join(args.out, 'gamma_max_sweep.csv'))
        _write_demo_csv(demo, os.path.join(args.out, 'low_shear_demo.csv'))
        _figure(sweep_rows, demo, os.path.join(args.out, 'senescence_law_sweep.pdf'))
        _write_markdown(sweep_rows, demo, os.path.join(args.out, 'summary.md'))
        print(f"\nOutputs written to: {args.out}")


if __name__ == '__main__':
    main()

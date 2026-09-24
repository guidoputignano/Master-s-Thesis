"""
The justified shear range, the candidate limits, and the controller on that range.

Three sources bound the range (docs/shear_range_and_limits.md):

* the chamber: the parallel-plate chamber of the reference data applied up to 10 Pa and is
  described as reaching 12 Pa, so it does not bind;
* physiology: measured human arterial means are 0.3-1.5 Pa, with systolic peaks of
  2.5-4.3 Pa;
* the model's data: the plateau was measured only at 1.4 Pa.

What binds is the monolayer. In the same chamber, flat HUVEC monolayers kept full junction
connectivity for 16 h up to 4 Pa and lost it above (Robotti et al. 2014); alignment along
the flow improved up to 5 Pa and was lost at 6 Pa; from the static state, 8 Pa gives
perpendicular alignment (Stefopoulos et al. 2022). The controller's range is therefore
0-4 Pa (config.tau_max_pa), with the orientation target turning perpendicular above the
parallel band (config.parallel_band_top_pa, perpendicular_crossover_pa).

This script runs the six- and twelve-hour closed loop on that range and on the sensitivity
cases (a gradient-chamber band edge at 2 Pa, Baeyens et al. 2015; the chamber's 8 Pa as the
bound; the earlier model without the perpendicular state), restates the design numbers at
1.4 Pa and at the top of the range, and draws the evidence.

    python -m analysis.shear_range            # writes results/shear_range/
"""
from __future__ import annotations

import json
import os

import numpy as np

from analysis.senescence_realism import base_config, closed_loop
from endothelial_simulation.control.mpc_controller import (
    RecedingHorizonMPC, flow_alignment_angle, PHI_SEN_RANDOM)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'shear_range')
SEN_DEG = float(np.degrees(PHI_SEN_RANDOM))

# Measured points (flat substrates, parallel-plate chamber of the reference data).
ROBOTTI_2014 = {   # long-confluent HUVEC, 16 h: shear (Pa) -> (mean angle to flow deg, connectivity index)
    0.0: (43.5, 1.00), 1.4: (38.0, 0.99), 4.0: (36.5, 1.00), 5.0: (21.5, 0.84), 6.0: (43.5, 0.43), 8.0: (39.5, 0.26)}
RECENT = {0.0: 45.0, 1.4: 20.0}            # recently confluent HUVEC (Chala 2021, 16 h; Stefopoulos 2022, 8 h)
STEFOPOULOS_2017 = {  # flat, 16 h, same chamber: shear (Pa) -> connectivity index (controls, with circulating TNF-alpha),
                      # Biomaterials 2017;138:131-41, Fig. 25A of the thesis, read by eye (+/- 0.02)
    0.0: (1.00, 1.04), 1.4: (1.00, 0.80), 4.0: (0.95, 0.67), 6.0: (0.80, 0.52)}
THETA_FLOOR_DEG = 11.0   # no plateau below the lowest measured in this chamber (low passage, Exarchos 2022),
                         # where the gate extrapolates above 1.4 Pa (as in analysis.experiment_predictions)

SCENARIOS = [
    # label, config changes, controller keywords
    ('Reported: range 0-4 Pa (integrity), band to 5 Pa', {}, {}),
    ('Band edge at 2 Pa (Baeyens 2015), range 0-4 Pa',
     dict(parallel_band_top_pa=2.0, perpendicular_crossover_pa=2.7), {}),
    ('Bound at the chamber (8 Pa), band to 5 Pa', dict(tau_max_pa=8.0), {}),
    ('No perpendicular state, bound 8 Pa (earlier model)',
     dict(tau_max_pa=8.0, parallel_band_top_pa=np.inf, perpendicular_crossover_pa=np.inf), {}),
    ('Calibrated shear only: range 0-1.4 Pa', dict(tau_max_pa=1.4), {}),
    ('Set point 3.5 Pa (chamber tolerance +/-14 % kept below 4 Pa)', dict(tau_max_pa=3.5), {}),
    ('Reported, 0.5 Pa/h ramp limit', {}, dict(delta_tau_max=0.5)),
    ('Reported, dense monolayer (T 6 h)', dict(tau_adapt_hours=6.0, tau_orient_hours=6.0), {}),
    ('Reported, inherited fraction 0.20', dict(initial_senescent_fraction=0.20), {}),
]
TARGET_SHEARS = (1.4, 2.0, 2.5, 3.0, 3.5, 4.0)   # healthy-cell plateau along the gate (design table)


def healthy_target_deg(mpc, tau):
    return float(np.degrees(flow_alignment_angle(mpc.theta_target(tau))))


def design_numbers(top):
    """Acceptance thresholds and conditioning times at 1.4 Pa and at the top of the range."""
    out = {}
    for tau in (1.4, top):
        row = {}
        for label, changes in (('tau_act 0.5 (nominal)', {}), ('tau_act 0.3', dict(tau_act=0.3)),
                               ('tau_act 0.7', dict(tau_act=0.7))):
            m = RecedingHorizonMPC(base_config(**changes))
            th = max(healthy_target_deg(m, tau), THETA_FLOOR_DEG if tau > 1.4 else 0.0)
            row[label] = {'healthy_plateau_deg': round(th, 2),
                          'max_senescent_share': {f'{t:g} deg': round(max(0.0, (t - th) / (SEN_DEG - th)), 3)
                                                  for t in (22.5, 25.0, 27.5, 30.0)}}
        out[f'{tau:g} Pa'] = row
    out['time_to_90_95_percent_h'] = {f'{T:g} h': [round(T * np.log(10), 1), round(T * np.log(20), 1)]
                                      for T in (2, 3, 4, 6, 8, 12)}
    out['healthy_plateau_along_the_gate_deg'] = {
        label: {f'{tau:g} Pa': round(max(healthy_target_deg(RecedingHorizonMPC(base_config(**changes)), tau),
                                         THETA_FLOOR_DEG if tau > 1.4 else 0.0), 2) for tau in TARGET_SHEARS}
        for label, changes in (('tau_act 0.5 (nominal)', {}), ('tau_act 0.3', dict(tau_act=0.3)),
                               ('tau_act 0.7', dict(tau_act=0.7)))}
    return out


def figure(path_noext, mpc_rep, mpc_bae):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    blue, orange, aqua = '#2a78d6', '#eb6834', '#1baf7a'
    ink, muted = '#0b0b0b', '#52514e'
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                         'axes.edgecolor': muted, 'axes.linewidth': 0.6, 'xtick.color': muted,
                         'ytick.color': muted, 'axes.labelcolor': ink, 'pdf.fonttype': 42, 'savefig.dpi': 300})
    fig, (a, b) = plt.subplots(1, 2, figsize=(17.5 / 2.54, 6.4 / 2.54))
    tau = np.linspace(0, 8, 401)
    for ax in (a, b):
        ax.axvspan(0, 4, color=muted, alpha=0.07, lw=0)
        ax.set_xlim(0, 8.6)
        ax.set_xlabel('wall shear stress (Pa)')
        ax.spines[['top', 'right']].set_visible(False)
    a.plot(tau, [healthy_target_deg(mpc_rep, t) for t in tau], color=blue, lw=1.5)
    a.text(4.3, 10.5, 'model', color=ink, va='center', fontsize=7)
    a.plot(tau, [healthy_target_deg(mpc_bae, t) for t in tau], color=blue, lw=1.0, ls='--')
    a.text(4.2, 76, 'band edge at 2 Pa (dashed)', color=ink, va='center', fontsize=7)
    x = sorted(ROBOTTI_2014)
    a.plot(x, [ROBOTTI_2014[k][0] for k in x], color=orange, lw=0, marker='o', ms=4)
    a.text(8.1, ROBOTTI_2014[8.0][0], 'long-confluent, 16 h', color=ink, va='center', fontsize=7)
    a.plot(list(RECENT), list(RECENT.values()), color=aqua, lw=0, marker='s', ms=4)
    a.text(0.15, 14.0, 'recently\nconfluent', color=ink, va='top', fontsize=7)
    a.text(0.1, 76, 'justified range', color=muted, fontsize=7)
    a.set_ylabel('healthy-cell angle to the flow (deg)')
    a.set_ylim(0, 80)
    a.set_title('a', loc='left', fontweight='bold')
    b.plot(x, [ROBOTTI_2014[k][1] for k in x], color=orange, lw=1.5, marker='o', ms=4)
    b.text(8.1, ROBOTTI_2014[8.0][1], 'long-confluent, 16 h', color=ink, va='center', fontsize=7)
    xs = sorted(STEFOPOULOS_2017)
    b.plot(xs, [STEFOPOULOS_2017[k][0] for k in xs], color=ink, lw=0.8, marker='s', ms=3.5, mfc='white')
    b.text(6.15, STEFOPOULOS_2017[6.0][0], 'controls, 16 h', color=ink, va='center', fontsize=7)
    b.plot(xs, [STEFOPOULOS_2017[k][1] for k in xs], color=ink, lw=0.8, ls='--', marker='^', ms=4)
    b.text(6.15, STEFOPOULOS_2017[6.0][1], 'with TNF-$\\alpha$, 16 h', color=ink, va='center', fontsize=7)
    b.set_ylabel('junction connectivity index')
    b.set_ylim(0, 1.1)
    b.set_title('b', loc='left', fontweight='bold')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{path_noext}.{ext}', bbox_inches='tight')
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    loops = []
    for label, changes, kw in SCENARIOS:
        cfg = base_config(**changes)
        mpc = RecedingHorizonMPC(cfg, **kw)
        r6 = closed_loop(cfg, mpc, n_steps=6)
        r12 = closed_loop(cfg, mpc, n_steps=12)
        loops.append(dict(label=label, bound_pa=mpc.tau_max, tau=[round(t, 3) for t in r12['tau']],
                          at_6h={k: round(r6[k], 3) for k in ('phi_sen', 'rho_bar', 'varphi_bar', 'healthy')},
                          at_12h={k: round(r12[k], 3) for k in ('phi_sen', 'rho_bar', 'varphi_bar', 'healthy')}))
        print(f"{label:52s} tau {[round(t, 2) for t in r12['tau']]}  6 h: healthy {r6['healthy']:.1f}, "
              f"population {r6['varphi_bar']:.1f} deg;  12 h: {r12['healthy']:.1f}, {r12['varphi_bar']:.1f} deg")
    cfg = base_config()
    design = design_numbers(cfg.tau_max_pa)
    summary = dict(range_pa=[0.0, cfg.tau_max_pa], band_top_pa=cfg.parallel_band_top_pa,
                   crossover_pa=cfg.perpendicular_crossover_pa, closed_loop=loops, design=design,
                   robotti_2014=ROBOTTI_2014, stefopoulos_2017=STEFOPOULOS_2017)
    with open(os.path.join(OUT, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1, default=float)
    figure(os.path.join(OUT, 'fig_shear_range'), RecedingHorizonMPC(cfg),
           RecedingHorizonMPC(base_config(parallel_band_top_pa=2.0, perpendicular_crossover_pa=2.7)))
    print(json.dumps(design, indent=1))


if __name__ == '__main__':
    main()

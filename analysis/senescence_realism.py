"""
Senescence within a conditioning session: the thesis law against the inherited fraction.

The thesis let 1.4 Pa convert 15 % of healthy cells to senescence within 6 h, through
an injury term that was switched on so that the controller would face a trade-off.
No study shows new senescence within 6-24 h under any shear, 1-2 Pa laminar shear
protects over days, and nothing injures endothelial cells in 0-2 Pa (sources in
config.py and docs/senescence_realism.md). The reported model therefore holds the
senescent fraction of the seeded batch constant within a session.

This script compares the two in the six-hour closed loop (master seed), and derives
the design quantities the inherited fraction implies: the plateau alignment of the
population against the senescent share (with the measured mixtures of Stefopoulos
et al. 2022), the largest senescent share compatible with an alignment target, and
the conditioning time as the adaptation constant grows (dense monolayers).

    python -m analysis.senescence_realism           # writes results/senescence_realism/
"""
from __future__ import annotations

import json
import os

import numpy as np

from endothelial_simulation.config import SimulationConfig
from endothelial_simulation.control.mpc_controller import (
    RecedingHorizonMPC, flow_alignment_angle, _gated,
    RHO_STAT, RHO_FLOW, RHO_SEN, THETA_STAT_DEG, THETA_FLOW_DEG, PHI_SEN_RANDOM,
)
from analysis.horizon_sensitivity import generate_initial_population

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'results', 'senescence_realism')
MIXTURE = {0.0: 21.0, 0.3: 31.0, 0.7: 37.0, 1.0: 47.0}   # Stefopoulos et al. 2022, Fig. 5g (deg)
SEN_DEG = float(np.degrees(PHI_SEN_RANDOM))              # senescent cells stay random: 45 deg


def base_config(**changes):
    """The reported run (endothelial_simulation/run_mpc.py), with optional changes."""
    cfg = SimulationConfig().set_full_simulation()
    cfg.enable_holes = False
    cfg.senescent_stress_fraction = 1.0
    cfg.senescent_telomere_fraction = 0.0
    for k, v in changes.items():
        setattr(cfg, k, v)
    return cfg


# The comparison of September 2026 (injury term against inherited senescence) was made with
# the 2 Pa cap of the time; it is kept at that cap so it stays reproducible. The range that
# replaced the cap is in analysis/shear_range.py.
CAP_2PA = dict(tau_bounds=(0.0, 2.0))
SCENARIOS = [
    # label, config changes, controller keyword arguments
    ('Thesis law: injury, induction in session, phi0 0.20, 0.5 Pa/h',
     dict(CONSTANT_SENESCENT_FRACTION=False, INCLUDE_SUPRAPHYSIOLOGICAL_ARM=True,
          initial_senescent_fraction=0.20), dict(delta_tau_max=0.5, **CAP_2PA)),
    ('Reported: inherited fraction 0.30', dict(), dict(CAP_2PA)),
    ('Reported, with a 0.5 Pa/h ramp limit', dict(), dict(delta_tau_max=0.5, **CAP_2PA)),
    ('Reported, dense monolayer (tau 6 h)', dict(tau_adapt_hours=6.0, tau_orient_hours=6.0), dict(CAP_2PA)),
    ('Reported, inherited fraction 0.20', dict(initial_senescent_fraction=0.20), dict(CAP_2PA)),
]


def closed_loop(cfg, mpc, n_steps=6):
    """Six one-hour receding-horizon steps from the static state, master seed."""
    x = {'pop': generate_initial_population(cfg, int(cfg.random_seed)),
         'rho_h': mpc.rho_target(0.0), 'theta_h': mpc.theta_target(0.0)}
    admitted = mpc.admitted(x)
    taus, u_prev, hourly = [], 0.0, []
    for _ in range(n_steps):
        u_opt, _ = mpc.solve(x, u_prev)
        tau_k = float(np.clip(u_opt[0], mpc.tau_min, mpc.tau_max))
        x = mpc.predict_step(x, tau_k)
        phi, rho, varphi = mpc.outputs(x)
        hourly.append(dict(tau=tau_k, phi_sen=phi, rho_bar=rho, varphi_bar=float(np.degrees(varphi)),
                           healthy=float(np.degrees(flow_alignment_angle(x['theta_h'])))))
        taus.append(tau_k)
        u_prev = tau_k
    last = {k: v for k, v in hourly[-1].items() if k != 'tau'}
    return dict(tau=taus, admitted=admitted, hourly=hourly, **last)


_MODEL = None


def _healthy_targets(tau_pa):
    """Healthy-cell plateau (angle deg, aspect ratio) of the reported model at a shear."""
    global _MODEL
    if _MODEL is None:
        _MODEL = RecedingHorizonMPC(base_config())
    return float(np.degrees(flow_alignment_angle(_MODEL.theta_target(tau_pa)))), _MODEL.rho_target(tau_pa)


def top_pa():
    """Top of the justified shear range (config.tau_max_pa, docs/shear_range_and_limits.md)."""
    return float(base_config().tau_max_pa)


def plateau(tau_pa, phi):
    """Population alignment (deg) and aspect ratio at the plateau for a senescent share."""
    th, rh = _healthy_targets(tau_pa)
    return (1 - phi) * th + phi * SEN_DEG, (1 - phi) * rh + phi * RHO_SEN


def population_alignment(t, tau_pa, phi, tau_h):
    """Population alignment (deg) at a constant shear from the static state."""
    th, _ = _healthy_targets(tau_pa)
    healthy = th + (THETA_STAT_DEG - th) * np.exp(-np.asarray(t) / tau_h)
    return (1 - phi) * healthy + phi * SEN_DEG


def design_numbers():
    top = top_pa()
    th_top, _ = _healthy_targets(top)
    out = {
        'plateau': {f'{tau:g} Pa': {f'{p:.1f}': [round(v, 3) for v in plateau(tau, p)]
                                     for p in (0, .1, .2, .3, .4, .5)} for tau in (1.4, top)},
        'mixture_check_1p4Pa': {f'{p:.1f}': [round(plateau(1.4, p)[0], 1), MIXTURE[p]] for p in MIXTURE},
        f'max_senescent_share_{top:g}Pa': {f'{t:g} deg': round((t - th_top) / (SEN_DEG - th_top), 3)
                                           for t in (22.5, 25.0, 27.5)},
        'time_to_90_95_percent_h': {f'{tau:g} h': [round(tau * np.log(10), 1), round(tau * np.log(20), 1)]
                                    for tau in (2, 3, 4, 6, 8)},
    }
    return out


def figure(path_noext):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    blue, orange, aqua = '#2a78d6', '#eb6834', '#1baf7a'          # categorical slots 1-3 (validated)
    ink, muted = '#0b0b0b', '#52514e'
    plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'legend.fontsize': 8, 'axes.edgecolor': muted, 'axes.linewidth': 0.6,
                         'xtick.color': muted, 'ytick.color': muted, 'axes.labelcolor': ink,
                         'pdf.fonttype': 42, 'savefig.dpi': 300})
    fig, (a, b) = plt.subplots(1, 2, figsize=(17.5 / 2.54, 6.4 / 2.54))
    phi = np.linspace(0, 1, 101)
    top = top_pa()
    for tau, c, lab in ((1.4, blue, '1.4 Pa'), (top, orange, f'{top:g} Pa')):
        a.plot(phi * 100, [plateau(tau, p)[0] for p in phi], color=c, lw=1.5, label=f'model, {lab}')
    xs, ys = zip(*MIXTURE.items())
    a.plot(np.array(xs) * 100, ys, 'o', ms=5, mfc='white', mec=ink, mew=1.0, label='measured, 1.4 Pa')
    a.set_xlabel('senescent cells (%)')
    a.set_ylabel('population alignment at plateau (deg)')
    a.set_xlim(-2, 102); a.set_ylim(10, 50)
    a.legend(frameon=False, loc='lower right')
    a.set_title('a', loc='left', fontweight='bold')
    t = np.linspace(0, 16, 161)
    for (p, tau_h, c, lab, dy) in ((0.0, 3.0, blue, r'no senescent cells, $T_{\rm adapt}$ = 3 h', 0.0),
                                   (0.3, 3.0, orange, r'30 % senescent, $T_{\rm adapt}$ = 3 h', -1.3),
                                   (0.3, 6.0, aqua, r'30 % senescent, $T_{\rm adapt}$ = 6 h (dense)', 1.3)):
        y = population_alignment(t, top, p, tau_h)
        b.plot(t, y, color=c, lw=1.5, label=lab)
        b.text(16.3, y[-1] + dy, f'{y[-1]:.0f}°', color=ink, va='center', fontsize=8)
    b.axvline(6, ymax=0.6, color=muted, lw=0.6, ls=':')
    b.text(6.2, 11.2, '6 h', color=muted, fontsize=8)
    b.set_xlabel(f'time at {top:g} Pa (h)')
    b.set_ylabel('population alignment (deg)')
    b.set_xticks([0, 4, 8, 12, 16])
    b.set_xlim(0, 18); b.set_ylim(10, 50)
    b.legend(frameon=False, loc='upper right')
    b.set_title('b', loc='left', fontweight='bold')
    for ax in (a, b):
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{path_noext}.{ext}', bbox_inches='tight')
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    print('Six-hour closed loop (N_p = 6, N_c = 3, master seed):')
    for label, cfg_changes, mpc_kw in SCENARIOS:
        cfg = base_config(**cfg_changes)
        r = closed_loop(cfg, RecedingHorizonMPC(cfg, **mpc_kw))
        rows.append(dict(label=label, **{k: v for k, v in r.items() if k != 'hourly'}, hourly=r['hourly']))
        print(f"  {label:62s} tau = [{', '.join(f'{v:.2f}' for v in r['tau'])}] Pa | phi_sen {r['phi_sen']:.3f} | "
              f"rho_bar {r['rho_bar']:.3f} | healthy {r['healthy']:.1f} deg | population {r['varphi_bar']:.1f} deg | "
              f"admitted {r['admitted']}")
    design = design_numbers()
    print('\nDesign numbers:', json.dumps(design, indent=1))
    with open(os.path.join(OUT, 'summary.json'), 'w') as fh:
        json.dump({'closed_loop': rows, 'design': design}, fh, indent=1)
    figure(os.path.join(OUT, 'fig_design'))
    print(f'\nWrote {OUT}/summary.json and fig_design.pdf/png')


if __name__ == '__main__':
    main()

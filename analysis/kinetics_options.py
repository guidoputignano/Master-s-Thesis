"""
Morphological adaptation kinetics: which time constant and plateau fit the ETH data?

Compares first-order relaxation variants against the measured time course of
control HUVEC monolayers at 1.4 Pa in the ETH parallel-plate bioreactor, checks
the population mean against the measured senescent-mixture series, and shows
what each variant does to the six-hour receding-horizon closed loop (N_p = 6,
N_c = 3, master seed).

Data (control monolayers, 1.4 Pa):
  * t = 0 h   : orientation 49 +/- 25 deg, aspect ratio 1.9 +/- 0.67 (Chala et al., Nano Lett 2021)
  * t = 8 h   : orientation ~21 deg; shape and alignment plateau after ~6-8 h
                (Stefopoulos et al., Adv Sci 2022, Fig. 5g and text)
  * t = 16 h  : orientation 20 +/- 14 deg, aspect ratio 2.3 +/- 0.78 (Chala et al. 2021)
Mixtures at the plateau (Stefopoulos et al. 2022, Fig. 5g): 0, 30, 70 and 100 %
senescent cells give ~21, 31, 37 and 47 deg (0 % at 8 h, the others at 16 h).

Variants (targets at 1.4 Pa, time constant):
  A  thesis        : 7.4 deg and 2.23 at 1.4 Pa (gate toward 0 deg and 2.3), 7.4 h
  B  16 h anchor   : as A, 19.7 h (withdrawn)
  C  plateau       : 20 deg and 2.3 at 1.4 Pa, 2, 3 and 4 h (the reported model; 3 h nominal)

This is an analysis layer: it overrides the target maps and time constants of
RecedingHorizonMPC instances only and changes no model default.

    python -m analysis.kinetics_options
"""
from __future__ import annotations

import numpy as np

from endothelial_simulation.config import SimulationConfig
from endothelial_simulation.control.mpc_controller import (
    RecedingHorizonMPC, flow_alignment_angle, _s_activation,
    RHO_STAT, RHO_FLOW, THETA_STAT_DEG, THETA_FLOW_DEG, PHI_SEN_RANDOM,
)
from analysis.horizon_sensitivity import generate_initial_population

VARIANTS = [
    # name, model, time constant (h)
    ('A thesis (7.4 h toward 0 deg)', 'thesis', 7.4),
    ('B 16 h anchor (19.7 h)', 'thesis', 19.7),
    ('C plateau, 2 h', 'plateau', 2.0),
    ('C plateau, 3 h (reported)', 'plateau', 3.0),
    ('C plateau, 4 h', 'plateau', 4.0),
]
DATA_THETA = {8.0: 21.0, 16.0: 20.0}     # deg, control monolayers at 1.4 Pa
DATA_RHO = {16.0: 2.3}
MIXTURE = {0.0: 21.0, 0.3: 31.0, 0.7: 37.0, 1.0: 47.0}   # senescent fraction -> deg
TIMES = (2.0, 4.0, 6.0, 8.0, 16.0)


def paper_config():
    """The reported run: phi_sen(0) = 0.20, all stress-induced (endothelial_simulation/run_mpc.py)."""
    cfg = SimulationConfig().set_full_simulation()
    cfg.enable_holes = False
    cfg.initial_senescent_fraction = 0.20
    cfg.senescent_stress_fraction = 1.0
    cfg.senescent_telomere_fraction = 0.0
    return cfg


def make_mpc(cfg, model, tau_h):
    """A controller with the given targets and time constant.

    'plateau' is the reported model. 'thesis' restores the earlier targets: the
    gate ran from the static values toward 0 deg and 2.3 without passing through
    the measured values at 1.4 Pa.
    """
    mpc = RecedingHorizonMPC(cfg)
    if model == 'thesis':
        th_s, th_f = np.radians(THETA_STAT_DEG), 0.0
        mpc.theta_target = lambda tau: th_s + (th_f - th_s) * _s_activation(tau, mpc.tau_act)
        mpc.rho_target = lambda tau: RHO_STAT + (RHO_FLOW - RHO_STAT) * _s_activation(tau, mpc.tau_act)
    mpc.tau_adapt = float(tau_h)
    mpc.tau_orient = float(tau_h)
    return mpc


def open_loop(mpc, tau_pa=1.4, times=TIMES):
    """Healthy-cell orientation (deg) and aspect ratio at constant shear, from the static state."""
    out = {}
    th0, rho0 = mpc.theta_target(0.0), mpc.rho_target(0.0)
    for t in times:
        th_t, rho_t = mpc.theta_target(tau_pa), mpc.rho_target(tau_pa)
        diff = ((th_t - th0) + np.pi) % (2 * np.pi) - np.pi
        theta = th0 + diff * (1.0 - np.exp(-t / mpc.tau_orient))
        rho = rho_t - (rho_t - rho0) * np.exp(-t / mpc.tau_adapt)
        out[t] = (float(np.degrees(flow_alignment_angle(theta))), float(rho))
    return out


def mixture_plateau(mpc, tau_pa=1.4):
    """Population-mean alignment (deg) at the 1.4 Pa plateau for each senescent fraction.

    Healthy cells sit at their target, senescent cells stay randomly oriented, and
    the population value is the cell-weighted mean (as in RecedingHorizonMPC.outputs).
    """
    healthy = float(np.degrees(flow_alignment_angle(mpc.theta_target(tau_pa))))
    return {f: (1 - f) * healthy + f * np.degrees(PHI_SEN_RANDOM) for f in MIXTURE}


def closed_loop(cfg, mpc, n_steps=6):
    """The six-hour receding-horizon run from the static state, master seed."""
    x = {'pop': generate_initial_population(cfg, int(cfg.random_seed)),
         'rho_h': mpc.rho_target(0.0), 'theta_h': mpc.theta_target(0.0)}
    taus, u_prev = [], 0.0
    for _ in range(n_steps):
        u_opt, _ = mpc.solve(x, u_prev)
        tau_k = float(np.clip(u_opt[0], mpc.tau_min, mpc.tau_max))
        x = mpc.predict_step(x, tau_k)
        taus.append(tau_k)
        u_prev = tau_k
    phi, rho, varphi = mpc.outputs(x)
    return dict(tau=taus, phi_sen=phi, rho_bar=rho, varphi_bar=float(np.degrees(varphi)),
                healthy_align=float(np.degrees(flow_alignment_angle(x['theta_h']))),
                rho_h=float(x['rho_h']))


def main():
    cfg = paper_config()
    print('Open loop at 1.4 Pa from the static state (healthy cells):')
    print('  variant                          ' + '  '.join(f'theta({t:g} h)' for t in TIMES)
          + '  rho(6 h)  rho(16 h)')
    for name, model, tau_h in VARIANTS:
        ol = open_loop(make_mpc(cfg, model, tau_h))
        print(f'  {name:32s} ' + '  '.join(f'{ol[t][0]:10.1f}' for t in TIMES)
              + f'  {ol[6.0][1]:8.2f}  {ol[16.0][1]:8.2f}')
    print('  data (Chala 2021, Stefopoulos 2022): theta(8 h) ~ 21, theta(16 h) = 20, rho(16 h) = 2.3')

    print('\nPopulation mean at the 1.4 Pa plateau, by senescent fraction (deg):')
    print('  fraction senescent        ' + '  '.join(f'{f:5.0%}' for f in MIXTURE))
    for label, model in (('A thesis targets', 'thesis'), ('C plateau targets', 'plateau')):
        mp = mixture_plateau(make_mpc(cfg, model, 3.0))
        print(f'  {label:25s} ' + '  '.join(f'{mp[f]:5.1f}' for f in MIXTURE))
    print('  data (Stefopoulos 2022)   ' + '  '.join(f'{v:5.1f}' for v in MIXTURE.values()))

    print('\nSix-hour closed loop (N_p = 6, N_c = 3, master seed):')
    for name, model, tau_h in VARIANTS:
        r = closed_loop(cfg, make_mpc(cfg, model, tau_h))
        taus = ', '.join(f'{v:.2f}' for v in r['tau'])
        print(f"  {name:32s} tau = [{taus}] Pa | phi_sen(6 h) = {r['phi_sen']:.3f} | "
              f"rho_bar = {r['rho_bar']:.3f} | healthy alignment {r['healthy_align']:.1f} deg | "
              f"population mean {r['varphi_bar']:.1f} deg")


if __name__ == '__main__':
    main()

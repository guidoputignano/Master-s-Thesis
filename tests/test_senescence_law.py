"""
test_senescence_law.py — the Task 5 refactor of the senescence law and MPC.

Covers:
  * the monotone-decreasing Hill induction rate `gamma_tau_hill` (shape, limits,
    half-max, optional supraphysiological arm);
  * that the per-cell population update and the reduced control kernel use the
    IDENTICAL gamma;
  * that the three structure flags toggle the population structure as specified
    (replicative ladder + S_tel, contact-inhibition density factor);
  * the MPC Part-A changes: no soft senescence weight `w_phi`, a hard move/slew
    bound that binds, senescence kept only as a hard constraint, deterministic
    multi-start solve, and that the controller actuates out of the gate dead zone.
"""
import numpy as np

from endothelial_simulation.config import SimulationConfig
from endothelial_simulation.models.population_dynamics import (
    PopulationDynamicsModel, population_reduced_rhs, gamma_tau_hill,
)
from endothelial_simulation.control.mpc_controller import (
    RecedingHorizonMPC, RHO_FLOW,
)
from analysis.horizon_sensitivity import generate_initial_population

TOL = 1e-12


# ---------------------------------------------------------------------------
# Hill induction rate
# ---------------------------------------------------------------------------
def test_hill_shape_and_limits():
    gmin, gmax, tau_h, n = 0.00278, 0.0125, 0.5, 2
    tau = np.linspace(0.0, 8.0, 400)
    g = gamma_tau_hill(tau, gmin, gmax, tau_h, n)

    # endpoints: gamma(0) = gamma_max, gamma(large) -> gamma_min
    assert np.isclose(g[0], gmax, atol=TOL)
    assert g[-1] > gmin and np.isclose(g[-1], gmin, atol=1e-3)
    # strictly monotone decreasing
    assert np.all(np.diff(g) < 0)
    # half-max at tau_h: gamma = gamma_min + (gamma_max - gamma_min)/2
    assert np.isclose(gamma_tau_hill(tau_h, gmin, gmax, tau_h, n),
                      gmin + 0.5 * (gmax - gmin), atol=TOL)


def test_hill_supraphysiological_arm():
    gmin, gmax, tau_h, n = 0.00278, 0.0125, 0.5, 2
    gd, tau_d, m = 0.02, 6.0, 2
    # Without the damage arm the rate keeps falling; with it, the rate rises
    # again at high shear (interior minimum), and the damage adds gd/2 at tau_d.
    base_hi = gamma_tau_hill(10.0, gmin, gmax, tau_h, n)
    dmg_hi = gamma_tau_hill(10.0, gmin, gmax, tau_h, n, gamma_d=gd, tau_d=tau_d, m=m)
    assert dmg_hi > base_hi
    added = (gamma_tau_hill(tau_d, gmin, gmax, tau_h, n, gamma_d=gd, tau_d=tau_d, m=m)
             - gamma_tau_hill(tau_d, gmin, gmax, tau_h, n))
    assert np.isclose(added, gd / 2.0, atol=TOL)


# ---------------------------------------------------------------------------
# One gamma everywhere: per-cell update == reduced kernel
# ---------------------------------------------------------------------------
def test_gamma_unified_update_and_kernel():
    cfg = SimulationConfig()
    m = PopulationDynamicsModel(cfg)
    N = cfg.max_divisions
    pop = np.zeros(N + 3)
    pop[0] = 120.0
    pop[N + 1] = 15.0

    for tau in (0.0, 0.25, 0.5, 1.4, 3.0):
        g_ref = gamma_tau_hill(tau, cfg.gamma_min, cfg.gamma_max,
                               cfg.tau_h_sen, cfg.n_hill)
        # per-cell update path
        assert np.isclose(m.calculate_shear_stress_effect(tau), g_ref, atol=TOL)
        # reduced kernel: dS_str = gamma * N_E (xi removed)
        d = population_reduced_rhs(
            pop, tau, cfg.proliferation_rate, m.K,
            cfg.gamma_min, cfg.gamma_max, cfg.tau_h_sen, cfg.n_hill, N,
            include_replicative=True, model_growth=False)
        N_E = float(pop[:N + 1].sum())
        assert np.isclose(d[N + 2] / N_E, g_ref, atol=TOL)


# ---------------------------------------------------------------------------
# Structure flags
# ---------------------------------------------------------------------------
def _rhs(pop, tau, cfg, **flags):
    N = cfg.max_divisions
    K = cfg.carrying_capacity
    return population_reduced_rhs(
        pop, tau, cfg.proliferation_rate, K,
        cfg.gamma_min, cfg.gamma_max, cfg.tau_h_sen, cfg.n_hill, N, **flags)


def test_replicative_flag_toggles_ladder_and_stel():
    cfg = SimulationConfig()
    N = cfg.max_divisions
    pop = np.zeros(N + 3)
    pop[:N + 1] = 10.0                       # cells spread across the ladder
    tau = 1.4

    d_on = _rhs(pop, tau, cfg, include_replicative=True, model_growth=False)
    d_off = _rhs(pop, tau, cfg, include_replicative=False, model_growth=False)

    # replicative ON: terminal division feeds S_tel (dS_tel > 0)
    assert d_on[N + 1] > 0.0
    # replicative OFF: no division, so no telomere senescence, only stress loss
    assert d_off[N + 1] == 0.0
    assert np.allclose(d_off[:N + 1], -gamma_tau_hill(
        tau, cfg.gamma_min, cfg.gamma_max, cfg.tau_h_sen, cfg.n_hill) * pop[:N + 1])
    # both conserve stress senescence flux gamma * N_E
    N_E = float(pop[:N + 1].sum())
    g = gamma_tau_hill(tau, cfg.gamma_min, cfg.gamma_max, cfg.tau_h_sen, cfg.n_hill)
    assert np.isclose(d_on[N + 2], g * N_E, atol=TOL)
    assert np.isclose(d_off[N + 2], g * N_E, atol=TOL)


def test_growth_flag_toggles_density_factor():
    cfg = SimulationConfig()
    N = cfg.max_divisions
    pop = np.zeros(N + 3)
    pop[0] = 200.0                            # appreciable N_E so g < 1 clearly
    tau = 1.4

    d_g1 = _rhs(pop, tau, cfg, include_replicative=True, model_growth=False)
    d_gK = _rhs(pop, tau, cfg, include_replicative=True, model_growth=True)

    # With growth modelled, g = 1/(1+N_E/K) < 1 slows division out of E_0, so the
    # division-driven part of dE[0] is smaller in magnitude than with g == 1.
    g = 1.0 / (1.0 + 200.0 / cfg.carrying_capacity)
    assert g < 1.0
    # dE0 = -r*g*E0 - gamma*E0 ; the division piece scales exactly by g.
    gamma = gamma_tau_hill(tau, cfg.gamma_min, cfg.gamma_max, cfg.tau_h_sen, cfg.n_hill)
    div_g1 = -(d_g1[0] + gamma * pop[0])      # = r*1*E0
    div_gK = -(d_gK[0] + gamma * pop[0])      # = r*g*E0
    assert np.isclose(div_gK, g * div_g1, atol=1e-9)


# ---------------------------------------------------------------------------
# MPC Part A: cost, constraints, move bound, determinism, actuation
# ---------------------------------------------------------------------------
def test_mpc_has_no_wphi_and_has_move_bound():
    mpc = RecedingHorizonMPC(SimulationConfig())
    assert not hasattr(mpc, 'w_phi')
    for a in ('w_rho', 'w_varphi', 'w_u', 'delta_tau_max'):
        assert hasattr(mpc, a)
    # constraint helpers present: hard senescence cap AND hard move bound
    assert hasattr(mpc, '_phi_sen_margin') and hasattr(mpc, '_move_margin')


def test_cost_excludes_senescence():
    """The cost must contain ONLY the tracking (rho, varphi) and move terms; no
    phi_sen term. Recompute J from the rollout with just those three weights and
    require it to equal cost() exactly."""
    mpc = RecedingHorizonMPC(SimulationConfig())
    N = mpc.N
    pop = np.zeros(N + 3)
    pop[0] = 900.0
    pop[N + 2] = 100.0                        # nonzero senescent fraction
    x0 = {'pop': pop, 'rho_h': 1.9, 'theta_h': np.radians(30)}
    u = np.array([1.0, 1.2, 1.4])
    u_prev = 0.9

    traj = mpc._rollout(u, x0)
    prev = u_prev
    j_expected = 0.0
    for tau, (phi_sen, rho_bar, varphi_bar) in traj:
        j_expected += (mpc.w_rho * (rho_bar - RHO_FLOW) ** 2
                       + mpc.w_varphi * varphi_bar ** 2
                       + mpc.w_u * (tau - prev) ** 2)
        prev = tau
        assert phi_sen > 0.0                  # senescence is present in the state
    assert np.isclose(mpc.cost(u, x0, u_prev), j_expected, atol=TOL)


def test_move_bound_binds_and_is_deterministic():
    cfg = SimulationConfig()
    N = cfg.max_divisions
    pop = np.zeros(N + 3)
    pop[0] = 170.0
    pop[N + 1] = 10.0
    x0 = {'pop': pop, 'rho_h': 1.9, 'theta_h': np.radians(45)}

    # Tight move bound: from u_prev = 0 the tracking objective wants to climb, so
    # the first move saturates the bound exactly.
    mpc = RecedingHorizonMPC(cfg, delta_tau_max=0.2)
    u1, _ = mpc.solve(x0, u_prev=0.0)
    u2, _ = mpc.solve(x0, u_prev=0.0)
    assert np.allclose(u1, u2, atol=TOL)                      # deterministic
    assert abs(u1[0] - 0.0) <= mpc.delta_tau_max + 1e-6       # bound respected
    assert np.isclose(abs(u1[0] - 0.0), mpc.delta_tau_max, atol=1e-3)  # binds


def test_controller_actuates_out_of_gate_dead_zone():
    """With w_phi removed the objective is flat below tau_act; the multi-start
    solve must still climb rather than stall at u_prev = 0."""
    cfg = SimulationConfig()
    mpc = RecedingHorizonMPC(cfg)
    pop = generate_initial_population(cfg, seed=42)
    x = {'pop': pop, 'rho_h': mpc.rho_target(0.0), 'theta_h': mpc.theta_target(0.0)}
    u_prev = 0.0
    taus = []
    for _ in range(4):
        u, _res = mpc.solve(x, u_prev)
        tau_k = float(np.clip(u[0], mpc.tau_min, mpc.tau_max))
        x = mpc.predict_step(x, tau_k)
        taus.append(tau_k)
        # every applied move respects the (default) hard slew bound
        assert abs(tau_k - u_prev) <= mpc.delta_tau_max + 1e-6
        u_prev = tau_k
    # the controller drives shear up (does not stall at zero)
    assert taus[-1] > 0.5
    assert taus[-1] >= taus[0]

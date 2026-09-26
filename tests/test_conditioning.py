"""Tests of the target-specific conditioning package (model, observations, design)."""
import numpy as np
import pytest

from conditioning import design as Dsg
from conditioning import model as M
from conditioning import observations as O


def run(segments, surface="silicone_flat", theta=None, pp1_from=None):
    pr = O.Protocol("t", tuple(segments), surface, "test", pp1_from=pp1_from)
    return M.run_protocols([pr], M.X0 if theta is None else theta)


def test_fractions_are_conserved_and_bounded():
    out = run([(8, 1.4, 1.4), (6, 8, 8), (10, 0.5, 6)])
    tot = out["N"] + out["A"] + out["C"] + out["X"]
    assert np.allclose(tot, 1.0, atol=1e-9)
    for k in ("N", "A", "C", "X", "J", "D", "R"):
        assert out[k].min() >= -1e-12 and out[k].max() <= 1 + 1e-12


def test_static_monolayer_stays_isotropic_and_intact():
    out = run([(24, 0.0, 0.0)])
    assert abs(out["angle"][-1, 0] - 45.0) < 1e-6
    assert out["ci"][-1, 0] == pytest.approx(1.0)
    assert out["R"][-1, 0] == pytest.approx(1.0)


def test_low_shear_aligns_along_and_high_shear_across():
    lo, hi = run([(12, 1.4, 1.4)]), run([(12, 8.0, 8.0)])
    assert lo["angle"][-1, 0] < 35.0
    assert hi["angle"][-1, 0] > 60.0


def test_pp1_blocks_the_collapse_after_an_abrupt_drop():
    free = run([(8, 8, 8), (6, 1.4, 1.4)])
    pp1 = run([(8, 8, 8), (6, 1.4, 1.4)], pp1_from=7.0)
    assert pp1["J"][-1, 0] == 0.0
    assert pp1["angle"][-1, 0] > free["angle"][-1, 0]


def test_destabilisation_follows_the_size_of_a_step_not_a_slow_ramp():
    step = run([(1, 0.0, 0.0), (4, 6.0, 6.0)])
    ramp = run([(1, 0.0, 0.0), (4, 0.0, 6.0)])
    assert step["J"].max() > 2 * ramp["J"].max()


def test_per_column_parameters_match_single_runs():
    prots = O.protocols_for(O.observations("train"))[:4]
    t, sh = M.shear_array(prots)
    th2 = np.repeat(M.X0[:, None], len(prots), axis=1)
    th2[M.NAMES.index("tau_x"), 1] = 5.0
    out = M.simulate(sh, th2, [M.SURFACES[p.surface] for p in prots])
    alt = M.X0.copy()
    alt[M.NAMES.index("tau_x")] = 5.0
    single = M.simulate(sh[:, 1:2], alt, [M.SURFACES[prots[1].surface]])
    assert np.allclose(out["angle"][:, 1], single["angle"][:, 0])


def test_observations_have_protocols_and_roles():
    for o in O.OBS:
        assert o.protocol in O.P
        assert o.role in ("train", "test")
        assert o.sigma > 0
    assert any(o.role == "test" for o in O.OBS)


def test_protocol_shear_is_piecewise_linear():
    pr = O.P["stef22_gradual"]
    assert pr.shear([0.0, 7.9, 8.1, 10.5, 13.0]).tolist() == [8.0, 8.0, 6.0, 4.0, 1.4]


def test_design_path_ends_at_the_target_and_scores_baselines():
    tg = Dsg.Target(tau=8.0, budget=12.0)
    base = Dsg.baselines(tg)
    p = Dsg.path(base["slow ramp (8 h)"], tg)
    t = Dsg.times(tg)
    assert np.all(p[t >= tg.budget - tg.hold] == 8.0)
    rows = Dsg.compare(tg, M.X0[None])
    assert {r["path"] for r in rows} == set(base)
    for r in rows:
        assert 0.0 <= r["retention"] <= 1.0 and 0.0 <= r["ci"] <= 1.0


def test_compiled_kernel_matches_numpy():
    prots = O.protocols_for(O.observations())
    t, sh = M.shear_array(prots)
    pp1 = np.stack([(t >= p.pp1_from) if p.pp1_from is not None else np.zeros_like(t, bool) for p in prots], 1)
    surf = [M.SURFACES[p.surface] for p in prots]
    rng = np.random.default_rng(0)
    theta = M.LO + rng.random(len(M.X0)) * (M.HI - M.LO)
    M.USE_KERNEL = False
    ref = M.simulate(sh, theta, surf, pp1)
    M.USE_KERNEL = True
    fast = M.simulate(sh, theta, surf, pp1)
    for k in ("angle", "ar", "ci", "R", "J"):
        assert np.allclose(ref[k], fast[k], atol=1e-9), k


def test_variation_penalises_only_reversals():
    tg = Dsg.Target(tau=8.0, budget=8.0)
    nk = Dsg.n_knots(tg)
    rising = Dsg.variation(np.linspace(1.0, 7.0, nk), tg)
    step = Dsg.variation(np.r_[np.full(nk - 1, 4.3), 4.3], tg)
    zigzag = Dsg.variation(np.where(np.arange(nk) % 2, 3.0, 5.0), tg)
    assert rising == pytest.approx(step, abs=0.05 * nk)     # equal up to the smoothing, 0.05 Pa per knot
    assert zigzag > rising + 10.0


def test_two_level_path_and_simplify_never_worsen_the_design():
    tg = Dsg.Target(tau=8.0, budget=4.0)
    k = Dsg.two_level(4.3, 1.5, tg)
    assert k[:3].tolist() == [4.3, 4.3, 4.3] and k[3:].tolist() == [8.0] * (len(k) - 3)
    thetas = M.X0[None]
    x = np.full(Dsg.n_knots(tg), 8.0)
    s0 = Dsg.robust(Dsg.evaluate(Dsg.path(x, tg)[None], tg, thetas)["score"])[0]
    x2, s2, simple = Dsg.simplify(x, tg, thetas, step=0.5)
    assert s2 >= s0 - 0.01
    assert x2.min() >= tg.tau_min and x2.max() <= tg.tau_max


def test_closed_loop_library_respects_the_bounds():
    from conditioning import closed_loop as CL
    tg = Dsg.Target(tau=8.0, budget=8.0)
    rem, lib = CL.continuations(4.3, tg, 2.0)
    assert rem.sum() == Dsg.n_knots(tg) - 4
    for c in lib:
        assert len(c) == rem.sum()
        assert c.min() >= tg.tau_min - 1e-12 and c.max() <= tg.tau_max + 1e-12

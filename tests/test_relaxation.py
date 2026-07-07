"""
test_relaxation.py — the reported first-order adaptation kernels.

`relax_step` (aspect ratio / area, tau_adapt) and `orientation_step`
(orientation, tau_orient, on the circle) are the closed-form step response the
paper path uses between control instants. These tests pin them to the analytic
first-order relaxation

    y(t + dt) = y* - (y* - y0) * exp(-dt / tau)

(with shortest-arc wrapping for orientation) to floating-point tolerance.
"""
import numpy as np

from endothelial_simulation.config import SimulationConfig
from endothelial_simulation.models.temporal_dynamics import TemporalDynamicsModel

TOL = 1e-12


def _model():
    return TemporalDynamicsModel(SimulationConfig())


def _closed_form(y0, y_star, dt, tau):
    """Analytic first-order relaxation."""
    return y_star - (y_star - y0) * np.exp(-dt / tau)


def _wrap(psi):
    """Shortest-arc wrap to (-pi, pi]."""
    return ((psi + np.pi) % (2 * np.pi)) - np.pi


# ---------------------------------------------------------------------------
# relax_step (aspect ratio / area) — tau_adapt
# ---------------------------------------------------------------------------
def test_relax_step_matches_closed_form():
    tm = _model()
    tau_adapt = 9.0  # h — Table 1 nominal (aspect ratio / area)
    cases = [
        (1.9, 2.3, 1.0),    # static -> flow aspect ratio, 1 h
        (2.3, 1.9, 0.25),   # flow -> static, 15 min
        (1.0, 2.6, 3.7),    # arbitrary
        (2.0, 2.0, 1.0),    # already at target -> unchanged
    ]
    for y0, y_star, dt in cases:
        got = tm.relax_step(y0, y_star, dt, tau_adapt=tau_adapt)
        assert abs(got - _closed_form(y0, y_star, dt, tau_adapt)) < TOL


def test_relax_step_default_tau_adapt_is_config_value():
    """With no tau_adapt argument, relax_step must use config.tau_adapt_hours."""
    cfg = SimulationConfig()
    tm = TemporalDynamicsModel(cfg)
    y0, y_star, dt = 1.9, 2.3, 2.0
    got = tm.relax_step(y0, y_star, dt)  # default tau_adapt
    assert abs(got - _closed_form(y0, y_star, dt, cfg.tau_adapt_hours)) < TOL


def test_relax_step_limits():
    tm = _model()
    y0, y_star, tau = 1.9, 2.3, 9.0
    # dt = 0 -> no change
    assert abs(tm.relax_step(y0, y_star, 0.0, tau_adapt=tau) - y0) < TOL
    # dt -> large -> reaches target
    assert abs(tm.relax_step(y0, y_star, 1e6, tau_adapt=tau) - y_star) < 1e-9


# ---------------------------------------------------------------------------
# orientation_step — tau_orient, shortest-arc wrapping on the circle
# ---------------------------------------------------------------------------
def test_orientation_step_matches_closed_form_no_wrap():
    tm = _model()
    tau = 7.4  # h — orientation
    th0, th_star, dt = np.radians(45.0), np.radians(0.0), 1.0
    diff = _wrap(th_star - th0)
    expected = th0 + diff * (1.0 - np.exp(-dt / tau))
    got = tm.orientation_step(th0, th_star, dt, tau_adapt=tau)
    assert abs(got - expected) < TOL


def test_orientation_step_shortest_arc_wrap():
    """Target across the +/-pi seam must relax the SHORT way."""
    tm = _model()
    tau = 7.4
    th0 = np.radians(170.0)
    th_star = np.radians(-170.0)   # == +190 deg; shortest arc is +20 deg
    dt = 1.0
    diff = _wrap(th_star - th0)
    assert abs(diff - np.radians(20.0)) < 1e-9       # short arc, not -340 deg
    expected = th0 + diff * (1.0 - np.exp(-dt / tau))
    got = tm.orientation_step(th0, th_star, dt, tau_adapt=tau)
    assert abs(got - expected) < TOL
    # moved toward the target by |diff|*(1-exp), i.e. a small positive step
    assert got > th0


def test_orientation_step_limits():
    tm = _model()
    th0, th_star, tau = np.radians(45.0), np.radians(0.0), 7.4
    assert abs(tm.orientation_step(th0, th_star, 0.0, tau_adapt=tau) - th0) < TOL
    got = tm.orientation_step(th0, th_star, 1e6, tau_adapt=tau)
    assert abs(_wrap(got - th_star)) < 1e-9


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print(f"All {len(fns)} relaxation tests passed.")
    sys.exit(0)

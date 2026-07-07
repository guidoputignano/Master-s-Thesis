"""
test_reproducibility.py — deterministic seeding of the reported (paper) path.

Two runs of the receding-horizon MPC closed loop with the same
``config.random_seed`` must produce identical logged outputs at the hour
boundaries: the applied input sequence ``tau`` and the regulated outputs
``phi_sen``, ``rho_bar`` and ``varphi_bar`` (plus the healthy-cell alignment
diagnostic). This locks in Task 1's deterministic seeding of both `random` and
`numpy.random`, including the per-cell heterogeneity deviates (z_theta, z_rho).
"""
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest

from endothelial_simulation.config import SimulationConfig
from endothelial_simulation.core.simulator import Simulator
from endothelial_simulation.control.mpc_controller import run_mpc_simulation

# Hour-boundary log channels the paper reports.
LOG_KEYS = ('tau', 'phi_sen', 'rho_bar', 'varphi_bar', 'healthy_align')


def _make_config(seed):
    cfg = SimulationConfig().set_full_simulation()
    cfg.random_seed = seed
    cfg.enable_holes = False
    cfg.create_animations = False
    cfg.initial_cell_count = 12      # small & fast; determinism is size-independent
    cfg.initial_senescent_fraction = 0.20
    cfg.senescent_stress_fraction = 0.70
    cfg.senescent_telomere_fraction = 0.30
    return cfg


def _run_paper_path(seed, out_dir):
    cfg = _make_config(seed)
    sim = Simulator(cfg)
    sim.set_constant_input(0.0)
    sim.initialize()
    return run_mpc_simulation(
        sim, cfg, n_control_steps=2, output_dir=str(out_dir),
        render_minutes=(0, 30),
    )


def test_same_seed_reproduces_logged_outputs(tmp_path):
    res1 = _run_paper_path(42, tmp_path / "run1")
    res2 = _run_paper_path(42, tmp_path / "run2")

    # The seed is recorded in the run summary.
    assert res1['seed'] == 42 and res2['seed'] == 42

    for key in LOG_KEYS:
        a = np.asarray(res1['log'][key], dtype=float)
        b = np.asarray(res2['log'][key], dtype=float)
        assert a.shape == b.shape, f"log['{key}'] shape differs"
        # Identical seed + identical code path => bit-for-bit identical.
        assert np.array_equal(a, b), (
            f"log['{key}'] differs between two same-seed runs:\n {a}\n {b}"
        )


def test_different_seed_changes_outputs(tmp_path):
    """Sanity check that the seed actually drives the stochastic layout: a
    different seed should perturb at least one reported channel (otherwise the
    'identical' test above would be vacuous)."""
    res_a = _run_paper_path(42, tmp_path / "a")
    res_b = _run_paper_path(1234, tmp_path / "b")
    differs = any(
        not np.array_equal(np.asarray(res_a['log'][k], dtype=float),
                           np.asarray(res_b['log'][k], dtype=float))
        for k in LOG_KEYS
    )
    assert differs, "changing random_seed did not change any reported output"


if __name__ == "__main__":
    import tempfile
    import pathlib
    with tempfile.TemporaryDirectory() as d:
        root = pathlib.Path(d)
        test_same_seed_reproduces_logged_outputs(root / "same")
        print("PASS test_same_seed_reproduces_logged_outputs")
        test_different_seed_changes_outputs(root / "diff")
        print("PASS test_different_seed_changes_outputs")
    print("All reproducibility tests passed.")

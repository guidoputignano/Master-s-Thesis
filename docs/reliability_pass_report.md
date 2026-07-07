# Reliability & reproducibility pass — final report

Companion to `docs/code_audit.md` (the Task 0 call-graph audit). This report
records what changed and states the numeric facts explicitly. No biological
parameter value was altered silently; the one place where an in-effect value had
to be chosen (the duplicated `tau_base`/`lambda_scale`) is called out below.

## Audit conclusions (recap)

* **Live paper `run_mpc_simulation`:** `control/mpc_controller.py:run_mpc_simulation`
  (driving `RecedingHorizonMPC`), invoked by `run_mpc.py` →
  `python -m endothelial_simulation.run_mpc`. The `main.py:run_mpc_simulation`
  copy is the **legacy** CLI driver (path B, `EndothelialMPCController`),
  invoked by `python -m endothelial_simulation.main`.
* **Not reachable from the paper path (A):** `calculate_A_max`, `calculate_tau`,
  `update_cell_responses`, `EndothelialMPCController`, the per-cell
  `stress_exposure_time` accumulator. `_initialize_cell_properties_for_pressure`
  **is** reachable from path A via the `core/simulator.py` copy; the
  `event_driven_simulator.py` copy is dead.
* **`EventDrivenSimulator`** (`management/event_driven_simulator.py`) is dead
  code — never imported or instantiated.

## Explicit facts requested

| Item | Value |
|---|---|
| RNG seed now in use | **42** (`config.random_seed`, default) |
| `tau_orient` (orientation) | **7.4 h** (`config.tau_orient_hours`; `20 = 45·exp(-6/tau) ⇒ tau = 6/ln(45/20) ≈ 7.4`) |
| `tau_adapt` (aspect ratio) | **9.0 h** at the time of this pass; **since unified to 7.4 h** (= `tau_orient`) and the "area" role corrected — see `docs/tau_adapt_unification.md` |
| Was Task 2's method on the paper path? | **No.** `event_driven_simulator._initialize_cell_properties_for_pressure` is in the dead `EventDrivenSimulator`. The fix is correctness hygiene and does **not** affect reported results. (The paper path uses the correct `core/simulator.py` copy.) |

## Symbols quarantined vs retained

**Quarantined** (kept importable + functional, annotated `DEPRECATED` with a
one-shot `DeprecationWarning` on entry — *not* moved/deleted, because several are
still wired into the non-MPC CLI and the sensitivity scripts):

* `EndothelialMPCController` (class) — warns on construction.
* `TemporalDynamicsModel.calculate_A_max`, `.calculate_tau`, `.update_cell_responses`.
* Per-cell `stress_exposure_time` clock — `Cell.update_and_check_all_senescence`,
  `.update_stress_and_check_senescence`, `.update_stress_exposure` warn on entry;
  the attribute is annotated.
* Config `tau_base`, `lambda_scale`, `known_pressures`, `known_A_max`,
  `initial_response` — de-duplicated and relabelled as legacy (path B) config.
* `EventDrivenSimulator` — module-level `DEPRECATED / DEAD CODE` notice.

**Retained untouched** (the reported model, per the task's "do not touch" list):
`RecedingHorizonMPC`, `relax_step`, `orientation_step`, `rho_target`,
`theta_target`, the activation gate (`_s_activation` / `_gated` /
`s_activation` / `gated_target`), `population_reduced_rhs`,
`_reconcile_senescence`.

Verified: the paper path triggers **zero** legacy `DeprecationWarning`s; the
legacy CLI path (path B) still runs and emits each of the five deprecations
exactly once.

## Files changed — per-task summary

| File | Tasks | Change |
|---|---|---|
| `config.py` | 1, 4 | Add `random_seed = 42` (documented master seed); add seed to `get_summary()` / `describe()`. De-duplicate the twice-assigned `tau_base` (→ **60.0**), `lambda_scale` (→ **0.3**), `known_pressures`, `known_A_max`, `initial_response`, keeping the values already in effect and relabelling them legacy (path B). |
| `core/simulator.py` | 1, 3 | Replace `time.time_ns()` seeding in `Simulator.__init__` with `config.random_seed`; seed `random` + `numpy`; print the seed once. Rewrite `get_safe_final_statistics`: remove the all-zeros masking and the silent `1e6` energy cap; narrow to `(ArithmeticError, ValueError, OverflowError)`; report **NaN + a `RuntimeWarning`** (never 0) on numerical failure or a non-finite result. |
| `control/mpc_controller.py` | 1, 3, 4, 5 | Add `import random`; reseed both RNGs from `config.random_seed` before the per-cell `_z_theta`/`_z_rho` draws in `run_mpc_simulation`; record the seed in the return dict. Remove the module-level `warnings.filterwarnings('ignore')` that masked numerical/deprecation warnings. Deprecate `EndothelialMPCController` (docstring + `DeprecationWarning`). Expand `RecedingHorizonMPC` / `predict_step` docstrings to document the reported dynamics (gated static→flow targets; fixed adaptation constants — since unified to a single `tau_adapt = tau_orient = 7.4 h`, see `docs/tau_adapt_unification.md`; senescence via the reduced population ODEs, not a per-cell stress clock). |
| `models/temporal_dynamics.py` | 4 | Add one-shot legacy-deprecation helper; annotate `calculate_A_max`, `calculate_tau`, `update_cell_responses` (`DEPRECATED` docstrings + `DeprecationWarning`). `relax_step` / `orientation_step` / the gate untouched. |
| `core/cell.py` | 4 | Add one-shot legacy-deprecation helper; annotate the `stress_exposure_time` attribute and the three clock-driving methods. |
| `management/event_driven_simulator.py` | 1, 2, 4 | Replace `time.time_ns()` seeding with `config.random_seed`. **Task 2:** fix `calculate_target_area(pressure, …)` → `current_pressure` (undefined variable). Add module-level `DEPRECATED / DEAD CODE` notice. |
| `docs/code_audit.md` | 0 | New — the call-graph audit. |
| `docs/reliability_pass_report.md` | — | New — this report. |
| `tests/test_relaxation.py` | verify | New — pins `relax_step` / `orientation_step` to the closed-form `y(t+dt) = y* − (y* − y0)·exp(−dt/tau)` (with shortest-arc wrapping) to 1e-12. |
| `tests/test_reproducibility.py` | verify | New — two same-seed paper-path runs give identical `tau`, `phi_sen`, `rho_bar`, `varphi_bar`, `healthy_align` at the hour boundaries; a different seed perturbs them. |
| `tests/conftest.py` | verify | New — path setup + headless matplotlib. |

## Verification performed

* Paper path (`run_mpc_simulation`) smoke-tested on a short horizon: completes
  and writes the 3 summary PDFs, the animation, the dashboard, and the frame
  PDFs without error.
* `tests/test_relaxation.py` (6) and `tests/test_reproducibility.py` (2) pass.
* Path-A purity: the reported path triggers **0** legacy `DeprecationWarning`s.
* Legacy path B (`EndothelialMPCController` + `Simulator.step`) still runs and
  warns once per legacy symbol.
* Every changed module byte-compiles and imports.

## Output-affecting changes — flagged for your attention

* **`get_safe_final_statistics`** no longer caps `biological_energy` at `1e6`.
  For a normal run the energy is small and finite (≈ 60–140 in smoke tests), so
  this changes nothing in practice; it only affects the (pathological) overflow
  case, which now reports **NaN + a warning** instead of a capped/`0.0` value.
  This is a *reported statistic* (console printout), not a manuscript figure.
* **`tau_base` / `lambda_scale` de-duplication** keeps the values that were
  already winning (`60.0` / `0.3`), so the legacy CLI's `calculate_tau` output is
  unchanged. These do not touch the paper path.
* No biological parameter on the paper path was modified.

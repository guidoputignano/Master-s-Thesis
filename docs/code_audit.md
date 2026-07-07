# Call-graph audit — `endothelial_simulation`

Status: **Task 0 (read-only audit).** No code was modified to produce this
document. All subsequent destructive steps (Tasks 2 and 4) are conditional on
the conclusions below.

Terminology (from the task brief):

* **Path A — the PAPER path.** `run_mpc_simulation(...)` in
  `control/mpc_controller.py`, driving `RecedingHorizonMPC`. Morphological
  targets come from the gated static→flow interpolation; senescence is governed
  by the reduced population ODEs (`population_reduced_rhs`) with
  `_reconcile_senescence`.
* **Path B — the LEGACY path.** `EndothelialMPCController`,
  `TemporalDynamicsModel.update_cell_responses`, `calculate_A_max`,
  `calculate_tau` (`tau = tau_base * A_max**lambda_scale`), and the per-cell
  `stress_exposure_time` clock.

There is also a **third category** worth naming explicitly because it affects
Task 4: the **non-MPC CLI simulation modes** (single-step, multi-step, protocol,
constant) reached from `main.py` via `Simulator.run() → Simulator.step()`. These
are neither "the paper" nor "path B" but they *do* exercise several path-B
symbols, and they must keep working (the general constraint requires
`python -m endothelial_simulation.main` to stay functional).

---

## 0.1 Which `run_mpc_simulation` is live for the paper

There are **two** distinct functions named `run_mpc_simulation`, with different
signatures. They do **not** shadow each other — each caller imports/defines its
own — so both are reachable, but from different entry points.

| Definition | Signature | Drives | Entry point | Verdict |
|---|---|---|---|---|
| `control/mpc_controller.py:1083` | `run_mpc_simulation(simulator, config, n_control_steps=6, output_dir=None, render_minutes=(0,15,30,45))` | `RecedingHorizonMPC` | `run_mpc.py:main()` → `python -m endothelial_simulation.run_mpc` | **LIVE / PAPER (path A)** |
| `main.py:190` | `run_mpc_simulation(config, mpc_response_target, mpc_orientation_target, duration=None)` | `EndothelialMPCController` | `main.py:main()` → `python -m endothelial_simulation.main` (default `--mpc-control=True`) | **LEGACY (path B)** |

Evidence:

* `run_mpc.py` module docstring: *"Single entry point for the endothelial
  mechanoadaptation MPC study (main.tex)."* It imports the paper function
  explicitly: `from endothelial_simulation.control.mpc_controller import
  run_mpc_simulation` (`run_mpc.py:22`) and calls it at `run_mpc.py:58`.
* `main.py` defines its *own* local `run_mpc_simulation` (`main.py:190`) and
  imports only `EndothelialMPCController` from the control module
  (`main.py:20`). Its `main()` dispatch calls the local function at
  `main.py:586`. With no CLI flags, `--mpc-control` defaults to `True`
  (`main.py:417`) and the dispatch chain (`main.py:539–599`) selects the
  `elif args.mpc_control:` branch — so the bare CLI runs **path B**.

**Conclusion.** The `control/mpc_controller.py` definition is the one that
produces the manuscript results. The `main.py` definition is the legacy CLI
driver. The `main.py` copy is *not* dead (the CLI reaches it) but it is *not*
the paper path either.

---

## 0.2 Reachability: path A vs. path B

### Path A entry chain (`run_mpc.py`)

```
run_mpc.main()
 └─ Simulator(config)                       # core/simulator.py
     └─ models = {temporal: TemporalDynamicsModel,
                  population: PopulationDynamicsModel,
                  spatial: SpatialPropertiesModel}
 └─ simulator.set_constant_input(0.0)
 └─ simulator.initialize()
     ├─ grid.populate_grid(...)
     ├─ _apply_initial_senescence()          # np.random.permutation, reset_senescence
     ├─ _initialize_cell_properties_for_pressure()   # core/simulator.py copy (correct)
     │    └─ spatial.calculate_target_area / _aspect_ratio / _orientation
     └─ grid.adapt_cell_properties(); _record_state()
 └─ control.mpc_controller.run_mpc_simulation(simulator, config, ...)
     ├─ RecedingHorizonMPC(config)
     │    └─ rho_target, theta_target, rho_std, theta_std,
     │       cell_rho_target, cell_theta_target, predict_step, outputs,
     │       _expand, _rollout, cost, _phi_sen_margin, solve
     │       (module helpers: flow_alignment_angle, _s_activation, _gated)
     ├─ per-cell heterogeneity draws: c._z_theta, c._z_rho (np.random.randn)
     ├─ _measure_state → PopulationDynamicsModel.update_from_cells(dt=0)   # reads compartments only
     ├─ predict_step → solve_ivp(population_reduced_rhs, ...)              # authoritative senescence
     ├─ temporal.relax_step, temporal.orientation_step                    # TemporalDynamicsModel
     ├─ _reconcile_senescence → spatial.calculate_target_area / _orientation
     ├─ grid._update_voronoi_tessellation(preserve_temporal_dynamics=True)
     └─ rendering: _apply_plot_style, _class_array, _ownership_rgb,
        _render_frame, _build_animation, _summary_plots, _build_dashboard
```

Crucially, **path A never calls `Simulator.step()` or `Simulator.run()`.** It
advances the cells by hand (closed-form `relax_step`/`orientation_step`) and
advances the population compartments by hand (`predict_step` + direct writes to
`models['population'].state` + `_reconcile_senescence`). Everything reached only
through `Simulator.step()` is therefore *not* on path A.

### Answers for the six queried symbols

| Symbol | Reachable from Path A? | Where it *is* reached | Notes |
|---|:---:|---|---|
| `calculate_A_max` | **NO** | `TemporalDynamicsModel.model` / `get_scaled_tau_and_amax` / `update_cell_responses`; `EndothelialMPCController.predict_future_state` (`mpc_controller.py:296`) | Path B + non-MPC CLI (via `spatial.update_cell_properties → get_scaled_tau_and_amax`) |
| `calculate_tau` | **NO** | same as `calculate_A_max` (`mpc_controller.py:297`) | `tau = tau_base * A_max**lambda_scale` |
| `update_cell_responses` | **NO** | `Simulator.step()` (`simulator.py:661`); `EventDrivenSimulator.step()` (dead) | Reached by non-MPC CLI modes via `run()→step()`; never by path A |
| `EndothelialMPCController` | **NO** | `main.py:210` (legacy `run_mpc_simulation`); exported by `control/__init__.py` | Path B only |
| per-cell `stress_exposure_time` accumulator | **NO** (as an accumulator) | Initialised to `0.0` in `Cell.__init__` (reached); incremented only in `update_and_check_all_senescence`/`update_stress_and_check_senescence` (`cell.py:198,655,779`), all under `Simulator.step()` | On path A it stays `0.0` and is never read for any decision — senescence there is driven entirely by the population ODEs + `_reconcile_senescence` |
| `_initialize_cell_properties_for_pressure` | **YES** (core copy) / **NO** (legacy copy) | `core/simulator.py:350` reached via `initialize()`; `management/event_driven_simulator.py:455` never reached | The two copies diverge — see Task 2 below |

### Path-B-only symbols (candidates for Task 4 quarantine)

Reached **only** through path B (`EndothelialMPCController` / `main.py`
legacy driver), i.e. never by path A and never by the non-MPC CLI modes:

* `EndothelialMPCController` and its private helpers
  (`_extract_senescence_rate`, `_extract_hole_dynamics`,
  `_extract_orientation_dynamics`, `predict_future_state`, `calculate_cost`,
  `optimize_control`, `_fallback_control`, `control_step`,
  `get_constraint_status`).
* `main.py:run_mpc_simulation` (the legacy driver itself).

### Symbols that are legacy in *spirit* but still wired into the non-MPC CLI

These belong to the old per-cell response + stress-clock model (path B), are
**not** on path A, but are **not dead** — they are still exercised by
`python -m endothelial_simulation.main` in single-step/multi-step/protocol/
constant mode (via `Simulator.run() → step()`), and some by the
`sensitivity_analysis/` scripts:

* `TemporalDynamicsModel.update_cell_responses` — `Simulator.step():661`.
* `TemporalDynamicsModel.calculate_A_max`, `calculate_tau`,
  `get_scaled_tau_and_amax`, `get_tau_and_amax`, `model`, `simulate*`,
  `fit_parameters` — reached via `spatial.update_cell_properties`
  (`spatial_properties.py:288–290`) inside `Simulator.step()`, via
  `EndothelialMPCController` (path B), via `transition_controller.py:216`, and
  via `sensitivity_analysis/sensitivity_oat.py:271`.
* per-cell `stress_exposure_time` clock — `Simulator.step()` senescence check.

**Implication for Task 4.** Quarantining these must **not** break the CLI or the
sensitivity scripts. The safe mechanism is therefore the *annotate-in-place*
option (a `DEPRECATED:` docstring + `warnings.warn(..., DeprecationWarning)` on
entry) rather than *move-to-`legacy/`* (which would break `import` sites). A
one-shot `DeprecationWarning` keeps the code importable and functional while
making its status unambiguous. Note that `mpc_controller.py:13` currently calls
`warnings.filterwarnings('ignore')`, which would suppress such warnings if left
in place — this must be scoped or removed for the deprecation notices to be
visible.

### Dead code found (not on any live path)

* **`management/event_driven_simulator.py` — the entire `EventDrivenSimulator`
  class.** Grep confirms it is never imported or instantiated anywhere in the
  package, the CLI, or the analysis scripts. It is a stale, near-duplicate copy
  of `core/simulator.py`. Its `_initialize_cell_properties_for_pressure` carries
  the Task 2 bug.

---

## 0.3 `tau_adapt_hours` and `tau_orient_hours`

* **`tau_adapt_hours`** — defined once at `config.py:105` as `9.0` (h), the
  Table-1 nominal midpoint for aspect-ratio/area adaptation. Grep shows **no
  override anywhere** — neither `run_mpc.build_config()` nor `main.py` nor the
  sensitivity scripts reassign it. `RecedingHorizonMPC.__init__` reads it into
  `self.tau_adapt` (`mpc_controller.py:675`) and uses it for the `rho_h`
  (aspect-ratio) relaxation in `predict_step`. **Value in effect: 9.0 h.**
* **`tau_orient_hours`** — defined once at `config.py:109` as `7.4` (h).
  `RecedingHorizonMPC` reads it via `getattr(config, 'tau_orient_hours', 7.4)`
  into `self.tau_orient` (`mpc_controller.py:679`) and uses it for the `theta_h`
  (orientation) relaxation. **Value in effect: 7.4 h.** Calibration recorded in
  the code: `20 = 45·exp(-6/tau) ⇒ tau = 6/ln(45/20) ≈ 7.4 h`.

Consistency check inside the paper loop (`run_mpc_simulation`):

* aspect ratio: `temporal.relax_step(rho0c, tgt_rho, th)` is called **without**
  a `tau_adapt` argument, so it defaults to `TemporalDynamicsModel.tau_adapt_hours
  = config.tau_adapt_hours = 9.0 h` (`temporal_dynamics.py:113–115`).
* orientation: `temporal.orientation_step(th0c, tgt_theta, th, mpc.tau_orient)`
  passes `7.4 h` explicitly.
* the authoritative reduced state (`predict_step`) uses `self.tau_adapt = 9.0 h`
  for `rho_h` and `self.tau_orient = 7.4 h` for `theta_h`.

So orientation relaxes at **7.4 h** and aspect ratio/area at **9.0 h**
everywhere on path A — consistent.

### Duplicated / overwritten config assignments (Task 4 target)

`SimulationConfig.__init__` assigns several fields **twice**; the second
assignment wins:

| Field | First (line) | Second (line) | Value in effect | Consumed by |
|---|---|---|---|---|
| `tau_base` | `1.0` (137) | `60.0` (147) | **60.0** | `calculate_tau` (path B / non-MPC CLI) |
| `lambda_scale` | `0.5` (138) | `0.3` (148) | **0.3** | `calculate_tau` (path B / non-MPC CLI) |
| `known_pressures` | `[0.0, 1.4]` (127) | `[0.0, 1.4]` (141) | `[0.0, 1.4]` | `TemporalDynamicsModel.__init__` (P_values) |
| `known_A_max` | `{0.0:1.0, 1.4:2.5}` (128) | `{0.0:1.0, 1.4:2.5}` (142) | `{0.0:1.0, 1.4:2.5}` | `TemporalDynamicsModel.__init__` (A_max_map) |
| `initial_response` | `1.0` (134) | `1.0` (146) | `1.0` | `TemporalDynamicsModel.__init__` (y0) |

`TemporalDynamicsModel` *is* constructed on path A (as `models['temporal']`), so
its `__init__` reads these values — but the quantities it derives from them
(`slope`, `intercept`, `tau_base`, `lambda_scale`) are used only by
`calculate_A_max`/`calculate_tau`, which are **not** on path A. Path A uses only
`relax_step`/`orientation_step`, which depend on `tau_adapt_hours` /
`tau_orient_hours`, not on these fields. So none of `tau_base`, `lambda_scale`,
`known_pressures`, `known_A_max` feed the reported model; they are legacy config.

> **Caution (silent-output risk).** `tau_base` and `lambda_scale` currently
> take the *second* value (`60.0`, `0.3`) because of assignment order. Whichever
> value we keep when de-duplicating must be the one currently in effect
> (`60.0` / `0.3`), otherwise the non-MPC CLI modes' `calculate_tau` output
> changes silently. Removing the *first* (shadowed) assignment preserves current
> behaviour; removing the second would not. This is flagged for an explicit
> numeric decision.

---

## Summary of conclusions (gates for later tasks)

1. **Live paper function:** `control/mpc_controller.py:run_mpc_simulation`
   (via `run_mpc.py`). The `main.py` copy is the legacy CLI driver.
2. **Not reachable from path A:** `calculate_A_max`, `calculate_tau`,
   `update_cell_responses`, `EndothelialMPCController`, the per-cell
   `stress_exposure_time` accumulator. `_initialize_cell_properties_for_pressure`
   *is* reachable from path A via the `core/simulator.py` copy; the
   `event_driven_simulator.py` copy is dead.
3. **Task 2** target (`event_driven_simulator.py`) is **legacy-only / dead**
   (`EventDrivenSimulator` is never instantiated) → fix as correctness hygiene;
   it does **not** affect reported results.
4. **Task 4** candidates are legacy in spirit but several are still wired into
   the non-MPC CLI and the sensitivity scripts → quarantine by *annotate +
   `DeprecationWarning`*, not by moving/deleting, so the CLI keeps working.
   De-duplicate the config fields keeping the values currently in effect.
5. **`tau_adapt` = 9.0 h (default, never overridden); `tau_orient` = 7.4 h.**

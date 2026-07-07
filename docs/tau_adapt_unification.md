# Unification of the morphological adaptation time constant (tau_adapt = 7.4 h)

This records a **deliberate, author-approved parameter change on the reported
(paper) path** and the accompanying documentation correction. It is NOT a silent
change: it alters the reported aspect-ratio dynamics. The effect is quantified
below.

## Decision

Unify the morphological adaptation onto a single, data-calibrated time constant:
set the **aspect-ratio** adaptation constant `tau_adapt_hours` equal to the
**orientation** constant, **7.4 h** (was 9.0 h, the Table-1 6–12 h midpoint).

**Rationale.** Orientation and aspect ratio are driven by the same cytoskeletal
remodelling. Only the orientation constant is calibrated against imaging
(`theta(6 h) = 20 deg`, giving `6/ln(45/20) ≈ 7.4 h`); there is no independent
aspect-ratio timecourse to justify a distinct value. A single constant is
therefore the parsimonious and defensible choice.

`tau_adapt_hours` and `tau_orient_hours` are **kept as separate config fields**
(both now 7.4 h) so the planned sensitivity study can still sweep them
independently — but they represent **one physical constant**.

## What the constant governs (documentation correction)

- `tau_adapt` governs the **aspect-ratio** relaxation only.
- Cell **AREA** is determined by the Voronoi tessellation (spatial model) and is
  **not** relaxed with a temporal constant on the reported path. The previous
  "aspect ratio and area" wording was wrong and has been removed from the code.
- The reported morphological dynamics are now a **single** first-order constant
  of 7.4 h for both orientation and aspect ratio (superseding the previous
  "7.4 h orientation, 9 h aspect ratio/area" description).

## Quantified effect (Task 3)

Paper path (`run_mpc_simulation` / `RecedingHorizonMPC`) at the fixed master seed
`random_seed = 42`, the paper config (179 confluent cells, `phi_sen(0)=0.20`,
holes off), 24 one-hour control steps. Single variable changed: `tau_adapt_hours`
9.0 → 7.4. (The reported hour-boundary log is produced entirely by the
deterministic reduced-state rollout `_measure_state → solve → predict_step →
outputs`; this was verified to reproduce the full simulator-driven run exactly,
so the comparison isolates `tau_adapt`.)

| Quantity | Before (9.0 h) | After (7.4 h) | Δ |
|---|---|---|---|
| mean aspect ratio `rho_bar`, terminal (t = 24 h) | 2.1910 | 2.1996 | **+0.0086** |
| `rho_bar` max abs. difference over the horizon | — | — | **0.0214** at **t = 8 h** |
| applied input `tau(k)`, max abs. difference | — | — | **0.00104 Pa** |
| senescent fraction `phi_sen`, terminal | 0.2238 | 0.2238 | +2.6e-05 |
| `phi_sen` maximum over the horizon | 0.2238 | 0.2238 | ~0 |
| senescence constraint `phi_sen ≤ 0.30` engaged? | no | no | unchanged |

`rho_bar` trajectory (hour : before → after = Δ), abbreviated:

```
 0 : 1.9201 -> 1.9201 = +0.0000
 4 : 2.0278 -> 2.0454 = +0.0176
 8 : 2.0961 -> 2.1175 = +0.0214   (peak Δ)
12 : 2.1386 -> 2.1580 = +0.0194
24 : 2.1910 -> 2.1996 = +0.0086   (terminal)
```

**Interpretation.** With the shorter time constant the aspect ratio relaxes
slightly faster, so `rho_bar` is marginally higher at every hour: the difference
grows to a peak of **+0.021** in the mid-transient (≈ hour 8) and settles to
**+0.009** at 24 h — a **< 1 % relative** shift toward the same flow plateau
(`rho* = 2.3`). The applied wall-shear-stress sequence `tau(k)` is essentially
unchanged (max |Δ| ≈ 0.001 Pa), the senescent fraction is unchanged to ~1e-5, and
the `phi_sen ≤ 0.30` senescence constraint is not engaged in either case.

**No qualitative conclusion changes.** The effect is confined to a small upward
shift in the aspect-ratio trajectory; the control decisions, the senescence
outcome, and the convergence to the flow-adapted plateau are all unchanged.

## Manuscript implications (action required outside the code)

- **Table 1** must now report a **single** morphological adaptation constant of
  **7.4 h** (for both orientation and aspect ratio), not two separate constants
  (7.4 h and 9 h).
- The **"area" role** previously attributed to `tau_adapt` must be **removed**
  from the manuscript methods text: area is set by the tessellation, not by a
  temporal relaxation constant.
- The reported `rho_bar` trajectory shifts up by < 0.022 (see table); if the
  manuscript quotes specific `rho_bar` values they should be regenerated from the
  7.4 h run.

## Files changed

| File | Change |
|---|---|
| `config.py` | `tau_adapt_hours` 9.0 → **7.4** (unit + rationale comment; notes it equals `tau_orient_hours`; both kept as separate fields for sensitivity sweeps). |
| `control/mpc_controller.py` | `RecedingHorizonMPC` / `predict_step` docstrings rewritten to a **single 7.4 h** morphological constant; `self.tau_adapt` inline comment now says aspect-ratio only, area set by tessellation. |
| `run_mpc.py` | conditioning comment updated (single 7.4 h constant). |
| `models/temporal_dynamics.py` | legacy-deprecation docstring updated (single 7.4 h constant). |
| `models/parameters.py` | unused parallel `ModelParameters.tau_adapt_hours` aligned to 7.4 h with corrected comment (no output effect; the reported config is `SimulationConfig`). |

## Verification

- `tests/test_reproducibility.py` (same-seed self-consistency, now at 7.4 h) and
  `tests/test_relaxation.py` (closed-form relaxation for an arbitrary constant;
  the default-constant test reads `config.tau_adapt_hours` dynamically, so **no
  golden value was tied to 9.0 h** — nothing to regenerate) — **8 passed**.
- Paper-path smoke test completes and writes all figures at `tau_adapt = 7.4 h`.

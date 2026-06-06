# Model audit: implementation vs. published paper

This document records every discrepancy found between the equations and parameter
values in `main.tex` (the published paper) and the implementations in
`endothelial_simulation/models/`. It is the basis for the corrections applied in
Step 2. Each finding lists the paper reference, the offending code, and the fix.

Ground-truth references (all from `main.tex`):

* Temporal dynamics: `eq:target`, `eq:relaxation`, `eq:stepsolution`, `eq:orientation`, Table 1.
* Population dynamics: `eq:density`, `eq:gamma_quad`, `eq:Ei`, `eq:Stel`, `eq:Sstress`, `eq:reduced`, Table 1.
* Spatial model: `eq:localenergy`, `eq:argmin`, Table 1.

Relevant Table 1 values:

| Symbol | Value | Unit |
|---|---|---|
| `A_E*` (healthy target area) | 2354 | μm² |
| `A_S` (senescent area) | 5000–8600 | μm² |
| `ρ*` (adapted aspect ratio) | 2.3 | – |
| `θ*` (adapted orientation) | 20 | deg |
| `τ_act` (activation threshold) | 0.5 | Pa |
| `w_A : w_ρ : w_φ` | 1 : 8.5 : 5 | – |
| `γ_min` (injury minimum) | 0.00278 | h⁻¹ |
| `α_γ` (curvature) | 0.00497 | Pa⁻² h⁻¹ |
| `τ_opt` (protective shear) | 1.4 | Pa |
| `r` (division rate) | 0.02–0.03 | h⁻¹ |
| `ξ` (per-stage senescence scaling) | 0.05 | per stage |
| `N` (Hayflick limit) | 15–18 | PD |
| `K` (density midpoint) | 5–6 × 10⁴ | cells cm⁻² |
| `φ_sen^max` (senescence constraint) | 30 | % |
| `τ_adapt` (adaptation time constant) | 6–12 | h |

---

## 1. `temporal_dynamics.py`

**Correct as-is**

* `model()` implements `eq:relaxation` `dy/dt = (A_max − y)/τ` — correct in form.
* `update_cell_responses()` implements `eq:stepsolution`
  `y(t) = A_max − (A_max − y₀) e^{−dt/τ}` — correct in form.

**Discrepancies**

1.1 **Missing activation gate `s(τ)` (`eq:target`).** The paper defines the target as
the *gated* interpolation
`y*(τ) = y_stat + (y_flow − y_stat)·s(τ)`, with `s(τ)=0` for `τ ≤ τ_act` and
`s(τ)→1` for `τ ≫ τ_act` (`τ_act = 0.5 Pa`, Table 1). The code computes the target
(`calculate_A_max`) by lookup / linear interpolation with **no activation
threshold**, so morphology is not held isotropic below `τ_act`.
*Fix:* add `s_activation(tau)` and `gated_target(y_stat, y_flow, tau)` implementing
`eq:target`; expose `τ_act` from config. Existing signatures preserved (additive).

1.2 **Adaptation time constant.** The calibrated/reduced instance uses a single
constant `τ_adapt ∈ [6,12] h` (paper text after `eq:stepsolution` and Table 1). The
code uses the *general-form* power law `τ = τ_base · A_max^{λ}` only, with no
explicit `τ_adapt`. The `time_scale_factors['orientation'] = 0.1` is an arbitrary
speed-up not present in the paper.
*Fix:* expose `tau_adapt_hours` from config and add `relax_step()` /
`orientation_step()` helpers that relax with a single `τ_adapt` (`eq:relaxation`,
`eq:orientation`). The general-form power law is retained for backward
compatibility but is no longer the only path.

1.3 **Angular relaxation / shortest-arc wrapping (`eq:orientation`)** is absent from
the temporal model (the generic scalar `model()` does not wrap). Added as the new
`orientation_step()` helper using the wrapping operator
`⟨ψ⟩ = ((ψ+π) mod 2π) − π`.

1.4 **Units.** Code works in *minutes* (`tau_base = 60`); the paper is in *hours*.
The new helpers operate in hours (paper units). Documented; old minute-based path
left intact so the simulator is unaffected.

## 2. `parameters.py` (`ModelParameters`)

This class is standalone (only instantiated at module bottom; not imported by the
simulator), so it is corrected freely to the paper.

2.1 **`calculate_optimal_aspect_ratio` is wrong.** Implements `2.0 + 0.2·τ` (linear,
unbounded). Paper: the adapted aspect ratio is the gated plateau `ρ* = 2.3`
(`eq:target`, Table 1).
*Fix:* gated interpolation toward `ρ* = 2.3` with `τ_act`.

2.2 **`calculate_shear_stress_effect` is wrong (form, scale, and values).**
Implements a piecewise-linear law with breakpoints at `τ = 10, 20` and ad-hoc
coefficients. Paper `eq:gamma_quad`: U-shaped quadratic
`γ_τ(τ) = γ_min + α_γ·(τ − τ_opt)²` with
`γ_min = 0.00278 h⁻¹`, `α_γ = 0.00497 Pa⁻² h⁻¹`, `τ_opt = 1.4 Pa`.
*Fix:* replace with `eq:gamma_quad`.

2.3 **Parameter values.** `division_rate = 0`, `carrying_capacity = 200`,
`max_divisions = 15` (this one is within paper range), `a_max_map = {15,25,45}`
(pressure points in mmHg-like units, not the Pa shear of the paper, not Table 1).
*Fix:* set `r`, `K`, `N`, `γ_min`, `α_γ`, `τ_opt`, `ξ`, `τ_act`, `ρ*`, `θ*`,
`τ_adapt` to Table-1 values with `# Source:` comments.

## 3. `population_dynamics.py`

3.1 **`calculate_shear_stress_effect` is wrong (form & scale).** Piecewise-linear in
`τ` with breakpoints at `τ = 1, 2`. Paper `eq:gamma_quad`: U-shaped quadratic
(see 2.2).
*Fix:* replace with `eq:gamma_quad` using config parameters.

3.2 **`calculate_density_factor` is wrong (functional form & argument).** Implements
the logistic `1/(1 + exp(10·(N/K − 0.7)))` evaluated on **total** cells. Paper
`eq:density`: `g(N_E) = 1/(1 + N_E/K)`, evaluated on the **healthy** count `N_E`.
*Fix:* replace with `eq:density`; feed `N_E` (= sum of `E_i`) in `update()`.

3.3 **Extra `division_capacity` factor in the balance.** `eq:Ei` has division terms
`2·r·g·E_{i−1} − r·g·E_i` only; the code multiplies each by a `division_capacity(i)`
ramp that is not in the paper.
*Fix:* remove the factor from the rate (method kept for signature compatibility).

3.4 **Age-scaled death / SASP sensitivity factors** (`1 + 0.03·i`, `1 + 0.08·i`) are
not in the paper. They multiply `d_E` and senolytic toxicity, which are zero in the
present study (`eq:reduced`), so they are inert; left in place but noted.

3.5 **Stress-senescence age scaling `(1 + 0.05·i)` matches `ξ = 0.05`** ✓. Made
configurable (`self.xi = config.xi`) so the global analysis can vary it; default
value unchanged.

3.6 **Reduced regime.** With `d_E = d_S = γ_S = α = 0` (Table 1, present study),
`eq:Ei/Stel/Sstress` collapse to `eq:reduced`. A new `reduced_rhs(state, tau)`
method returns the exact `eq:reduced` derivatives for continuous integration
(used by the global sensitivity analysis). Additive.

## 4. `spatial_properties.py`

4.1 **Energy functional** (`eq:localenergy`, `eq:argmin`) lives in
`core/grid.py` and is **already correct**: `e^A=(A/Â−1)²`, `e^ρ=(ρ/ρ̂−1)²`,
`e^φ=((φ−φ̂)/(π/2))²`, with weights `1 : 8.5 : 5` (Table 1). Per instructions, the
core/grid/tessellation code is **not modified**.

4.2 **Adapted orientation target.** `control_params['orientation_mean'][1.4] = 22.0°`;
paper `θ* = 20°` (Table 1).
*Fix:* set the flow-adapted orientation target to `20.0°`.

4.3 **Aspect-ratio target** `control_params['aspect_ratio'][1.4] = 2.3` already
matches `ρ* = 2.3` ✓.

4.4 **Area conversion inconsistency (noted, not changed).** The pixel area targets
(`area = 3216 px`, commented "2155 μm²") do not reconcile with the stated pixel
spacing (0.429 μm/px → 0.184 μm²/px) nor with `A_E* = 2354 μm²`. This lives in the
pixel/tessellation domain that the visualisation depends on; changing it risks the
viz, so it is left intact and flagged here only.

## 5. `config.py`

5.1 **Missing Table-1 parameters.** No `γ_min`, `α_γ`, `τ_opt`, `ξ`, `τ_act`, `ρ*`,
`θ*`, `τ_adapt`. Added (additive, with `# Source:` comments).

5.2 **`max_divisions = 7`** vs paper `N = 15–18 PD`.
*Fix:* set to 15.

5.3 **`proliferation_rate = 0`**; paper keeps division active at `r ≈ 0.02–0.03 h⁻¹`
in the reduced regime.
*Fix:* set nominal `r = 0.025 h⁻¹`.

5.4 **`carrying_capacity = 200`** (cell count) vs paper `K ≈ 5–6 × 10⁴ cells cm⁻²`.
*Fix:* set to `5.5e4` (paper units). The population component is near-inert over the
6 h horizon, so simulator runs are unaffected in practice.

---

### Summary of corrected files

| File | Change |
|---|---|
| `config.py` | add Table-1 parameters; fix `N`, `r`, `K` |
| `parameters.py` | gated `ρ*`; `eq:gamma_quad`; Table-1 values |
| `population_dynamics.py` | `eq:gamma_quad`; `eq:density` on `N_E`; drop non-paper division factor; configurable `ξ`; add `reduced_rhs` (`eq:reduced`) |
| `temporal_dynamics.py` | add `s_activation`/`gated_target` (`eq:target`); `relax_step`/`orientation_step` (`eq:relaxation`, `eq:orientation`) with single `τ_adapt` |
| `spatial_properties.py` | adapted orientation target `22° → 20°` (`θ*`) |
| `core/grid.py` | **unchanged** (energy already matches `eq:localenergy`/`eq:argmin`) |

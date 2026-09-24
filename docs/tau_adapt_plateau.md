# Adaptation constant 3 h, with the targets through the 1.4 Pa plateau

The morphological adaptation constant is now **3 h** (was 7.4 h), and the gated
targets now pass through the measured plateau at 1.4 Pa: **20°** for healthy cells
and **aspect ratio 2.3**. In the reported six-hour run the mean aspect ratio rises
from 2.03 to 2.12, the alignment barely moves (healthy cells 29.4° → 28.9°), and
the senescent fraction drops from 0.272 to 0.270. Every study keeps its conclusion.
This is an author-approved change on the reported (paper) path; the manuscript text
is updated separately (plan item A3).

## Why

Nafsika Chala's group measured control HUVEC monolayers at 1.4 Pa in the ETH
parallel-plate bioreactor:

- static: orientation 49 ± 25°, aspect ratio 1.9 ± 0.67 (Chala et al., Nano Lett 2021);
- 16 h: 20 ± 14°, 2.3 ± 0.78 (Chala et al. 2021);
- 8 h: ≈21° (Stefopoulos et al., Adv Sci 2022, Fig. 5g). The same paper states that
  "coherent cell polarity was accomplished within ≈2–4 h" and that shape change and
  alignment reach "a plateau after ≈6–8 h".

So 20° and 2.3 are a plateau, reached by 6–8 h. A plateau at 6–8 h means 86–95 % of
the change by then, so the time constant is 2–4 h; 3 h is the middle. No fitted time
course has been published. The numbers behind Figure 1c–d of the 2022 paper, or
Chala's thesis (ETH Diss. 28991), would allow a fit.

The old model placed the 16 h values at 6 h: 7.4 h = 6/ln(45/20), relaxing toward
0°. Because the gate is s(1.4 Pa) = 0.83, its targets at 1.4 Pa were 7.4° and 2.23,
so at a constant 1.4 Pa it kept aligning past the measured plateau (11.8° at 16 h).
A 19.7 h variant, which put 20° at 16 h on the way to 0°, was considered and
dropped: it gives 32.5° at 8 h against ≈21° measured.

## What changed in the model

- **Targets** (`control/mpc_controller.py`, `_gated`):
  y\*(τ) = y_stat + (y_flow − y_stat) · s(τ) / s(1.4 Pa), with θ_flow = 20° and
  ρ_flow = 2.3 as measured at 1.4 Pa. So y\*(1.4 Pa) is the measured value for any
  τ_act. Above 1.4 Pa the targets keep rising along the same gate (at 2 Pa: 16.5°
  and 2.36); below τ_act = 0.5 Pa they stay static.
- **One constant, 3 h,** for orientation and aspect ratio (`config.py`:
  `tau_adapt_hours = tau_orient_hours = 3.0`).
- **Unchanged:** the cost still tracks ρ̄ → 2.3 and φ̄ → 0°. Senescent cells keep
  AR 2.0 and random orientation (45°). The senescence law is unchanged.

## Checks against the data

Healthy cells from the static state, constant 1.4 Pa (`analysis/kinetics_options.py`):

| Time | Measured | Old model (7.4 h toward 0°) | New model (3 h, plateau) |
|---|---|---|---|
| 6 h | — | 24.1°, AR 2.09 | 23.4°, AR 2.25 |
| 8 h | ≈21° | 20.2° | 21.7° |
| 16 h | 20°, AR 2.3 | 11.8°, AR 2.20 | 20.1°, AR 2.30 |

Population mean at the 1.4 Pa plateau against the measured senescent mixtures
(Stefopoulos et al. 2022, Fig. 5g; healthy cells at their target, senescent cells at
45°, the cell-weighted mean as in `RecedingHorizonMPC.outputs`):

| Senescent fraction | 0 % | 30 % | 70 % | 100 % |
|---|---|---|---|---|
| Measured | 21° | 31° | 37° | 47° |
| Old targets | 7.4° | 18.7° | 33.7° | 45.0° |
| New targets | 20.0° | 27.5° | 37.5° | 45.0° |

## Effect on the reported run

`run_mpc_simulation`, 6 × 1 h steps, N_p = 6, N_c = 3, φ_sen(0) = 0.20, master seed
42 (logs in `results/paper_run/20260924-3h/`):

| Quantity | 7.4 h | 3 h |
|---|---|---|
| τ(k), Pa | 0.50, 1.00, 1.49, 1.35, 1.11, 0.90 | 0.50, 1.00, 1.43, 1.27, 1.06, 0.88 |
| φ_sen at 6 h | 0.272 | 0.270 |
| ρ̄ at 6 h | 2.028 | 2.115 |
| healthy-cell alignment at 6 h | 29.4° | 28.9° |
| population alignment φ̄ at 6 h | 33.7° | 33.2° |

Hourly (0 → 6 h), old → new:

```
rho_bar   1.920 1.921 1.947 1.978 2.003 2.020 2.028
          1.920 1.921 1.989 2.058 2.100 2.117 2.115
healthy   45.0  45.0  41.4  36.9  33.3  30.8  29.4 deg
          45.0  45.0  39.6  34.0  30.4  28.8  28.9 deg
```

**Interpretation.** With 3 h the cells follow the shear almost at once. The
alignment at 6 h no longer lags the target: it sits at the target for the shear the
controller ends on (29.0° at 0.88 Pa). What limits the adaptation in the six-hour
window is now the senescence cap alone. The controller lowers the shear after 2 h
because the injury arm of the senescence law raises φ_sen toward 0.30. The paper's
second reason, "because T_adapt ≈ 7.4 h is comparable to the window length", no
longer holds.

## Effect on the studies

Each study was run twice, on the old code (git worktree at `e1688f3`) and on the new
code, with the paper's settings.

- **Mismatch** (`analysis.mismatch_robustness --n-rep 5 --n-lhs 24`, the paper's
  settings). The rerun on the old code reproduces the stored paper run
  (`results/mismatch_robustness/20260724-090500`) to within 10⁻⁶. The new run is in
  `results/mismatch_robustness/20260924-073004`.

  | Quantity | 7.4 h | 3 h |
  |---|---|---|
  | nominal φ_sen at 6 h | 0.2720 | 0.2695 |
  | one-at-a-time ±20 %, closed-loop φ_sen range | 0.2705–0.2735 | 0.2679–0.2712 |
  | joint LHS, closed loop: mean / p95 / max φ_sen | 0.2721 / 0.2743 / 0.2753 | 0.2695 / 0.2719 / 0.2729 |
  | joint LHS, open loop: max φ_sen | 0.2773 | 0.2749 |
  | adaptation-constant sweep | 6–12 h: 0.2719–0.2722 | 2–8 h: 0.2692–0.2701 |
  | runs violating φ_sen ≤ 0.30 | 0 | 0 |

  The conclusions hold. Feedback still lowers the worst case (0.275 → 0.273 with the
  new model), and the adaptation constant still changes only the cost.
- **Horizon** (`analysis.horizon_sensitivity`, 20 replicates, not in the paper): at
  N_p = 6, N_c = 3, J 14.34 → 13.26, ρ̄ 2.028 → 2.116, φ̄ 33.7° → 33.2°, φ_sen
  0.272 → 0.270. For N_c ≥ 2, ρ̄ rises by 0.04–0.11 and φ̄ changes by −1.1 to
  +0.8°. The N_c = 1 rows never leave 0 Pa in either version (J = 19.36): their one
  move is held for the whole horizon and capped at 0.5 Pa, where the gate is still
  closed. Results in `results/horizon_sensitivity/20260924-065642` (7.4 h) and
  `20260924-070441` (3 h).
- **One-at-a-time** (`sensitivity_oat`, paper figure `oat_sensitivity_bars.pdf`). This
  study used a different model before: static aspect ratio 1.0 instead of 1.9, 9 h
  instead of 7.4 h, and 20° as an ungated asymptote. It now uses the reported
  model's targets and 3 h, swept over 2–4 h. Nominal outputs at 6 h: ρ 1.53 → 2.25,
  φ 34.8° → 23.4°. Normalised indices (ρ, φ): T_adapt (−0.23, 0.19) → (−0.05, 0.29);
  ρ\* (0.61, 0) → (0.89, 0); θ\* (0, 0.23) → (0, 0.74); τ_act (−0.20, 0.17) → (0, 0).
  τ_act no longer matters at 1.4 Pa, because the targets there are the measured
  values.
- **MPC cost weights** (`sensitivity_mpc_weights`): nominal alignment 29.5° → 29.0°,
  ρ̄ 2.028 → 2.114, φ_sen 0.272 → 0.270. Under factor-of-two changes of each weight
  the outcomes move by at most 0.09°, 0.001 and 0.0002, and every normalised index
  stays below 0.01. The paper's statement holds.
- **Energy weights** (`sensitivity_energy_weights`): identical before and after. It
  runs the legacy per-cell simulator, not the reported model. It also gives identical
  outputs for all five weight sets, because the weights only pick one of three
  candidate layouts and every set picks the same one. The paper's "varied by <0.1 %
  and … a few degrees" overstates this: the study shows no variation, by
  construction. This is independent of the adaptation constant.
- **Sobol** (`sensitivity_sobol`): not rerun. It integrates the senescence ODE only,
  which does not involve the adaptation constant.

## Manuscript changes needed (A3)

- **Methods, lines 473, 504–522.** T_adapt = 3 h, from the 6–8 h plateau of
  Stefopoulos et al. 2022. θ\* = 20° and ρ\* = 2.3 are the plateau at 1.4 Pa (Chala et
  al. 2021, 16 h). Add the normalised gate. Remove "≈20° at six hours",
  "6/ln(45/20) ≈ 7.4 h", "continues toward the parallel target 0°" and the L-BFGS-B
  fit claim: that fit belongs to the legacy response model, not these targets.
- **`fig:model` caption (732–735), Table 1 (820–821), line 1479.** T_adapt 3 h,
  "set from the published plateau time"; θ\* = 20° at 1.4 Pa.
- **Results (1122–1145) and the `fig:mpc-results` caption.** Peak ≈1.4 Pa (was
  ≈1.5), then ≈0.9 Pa; ρ̄ ≈ 2.12 (was 2.03); healthy alignment ≈29°; φ_sen ≈ 0.27.
  Remove the T_adapt reason for partial adaptation.
- **Sensitivity (1205) and mismatch (1262–1320).** The 2–8 h sweep (was 6–12 h),
  the one-at-a-time range [0.268, 0.271] (was [0.270, 0.274]), and the joint worst
  cases 0.273 closed-loop and 0.275 open-loop (were 0.275 and 0.277).
- **Figures to replace:**
  - `mpc_tau_trajectory`, `mpc_phi_sen` and `mpc_morphology`, from
    `results/paper_run/20260924-3h/`;
  - the four snapshots, from its `frames/` (t = 0, 2 and 4 h are `mpc_k00_t00`,
    `mpc_k02_t00` and `mpc_k04_t00`; t = 6 h was presumably the last frame);
  - `oat_sensitivity_bars` and `mpc_weights_sensitivity`, from
    `sensitivity_analysis/figures/`;
  - `mismatch_tornado` and `mismatch_feedback`, from
    `results/mismatch_robustness/20260924-073004/` (`fig_tornado`,
    `fig_feedback_efficacy`).

`endothelial_simulation/figures/` still holds the older 24-step demonstration run of
`python -m endothelial_simulation.run_mpc`. The paper does not use it, and it was not
regenerated.

## Files changed

| File | Change |
|---|---|
| `config.py`, `models/parameters.py` | `tau_adapt_hours = tau_orient_hours = 3.0`; `theta_star = 20` (informational); sources in the comments |
| `control/mpc_controller.py` | `THETA_FLOW_DEG = 20`, `TAU_FLOW_PA = 1.4`, normalised `_gated`, docstrings |
| `models/temporal_dynamics.py` | `gated_target` calls the reported `_gated`; fallback constant 3 h |
| `analysis/mismatch_robustness.py` | range 2–4 h, sweep 2–8 h, labels |
| `analysis/horizon_sensitivity.py`, `run_mpc.py` | docstrings |
| `sensitivity_analysis/sensitivity_oat.py` | the reported model's baselines, targets and 3 h; sweep 2–4 h |
| `analysis/kinetics_options.py` | new: old and new kinetics against the data and in the closed loop |
| `tests/test_relaxation.py` | targets through the plateau at 1.4 Pa; plateau by 6–8 h |

## Verification

- `pytest tests`: 66 passed. This includes three new tests: the targets pass through
  20° and 2.3 at 1.4 Pa for τ_act = 0.3–0.7 Pa; the temporal model uses the same
  targets; healthy cells reach 21.7° by 8 h and 20.0° with AR 2.30 by 16 h.
- The old-code reruns reproduce the paper's reported run, and its stored mismatch
  study to within 10⁻⁶.

# Robustness to parametric plant-model mismatch: report

Companion to `analysis/mismatch_robustness.py` and the audit note
`docs/mismatch_plant_audit.md`. This report states the control-theoretic result.
It does not draw biological conclusions.

> **STALE / PRE-REFACTOR NOTE (Task 5).** The numbers in this report were produced
> BEFORE the Task 5 refactor of the model and therefore describe the pre-refactor
> reported model, in which (i) the senescence-induction rate was the symmetric
> quadratic `gamma(tau) = gamma_min + alpha_gamma (tau - tau_opt)^2` and (ii) the
> MPC cost contained a soft senescence term `w_phi * phi_sen^2`. The reported
> model now uses a monotone-decreasing **Hill** law
> (`gamma_min`/`gamma_max`/`tau_h`) and enforces senescence as a **hard
> constraint** (no `w_phi`); a single hard move/slew bound was also added. The
> identified-parameter set of the study was accordingly changed from
> `{morph, gamma_min, alpha_gamma, tau_opt}` to `{morph, gamma_min, gamma_max,
> tau_h}`. `analysis/mismatch_robustness.py` was updated to run against the new
> model, but the figures, CSVs and the specific quantities quoted below have NOT
> been regenerated. Re-run the study to refresh them before quoting post-refactor
> numbers. The Task-0 mechanism conclusion (where and how the plant is advanced)
> is structural and still holds.

## Task 0 audit conclusion

The reported closed loop advances an authoritative reduced state
`x = {pop, rho_h, theta_h}` between control instants. The advance is performed
only by `RecedingHorizonMPC.predict_step`, at two sites in `run_mpc_simulation`
(the sub-hour dashboard points and the one-hour boundary). Inside `predict_step`
the population compartments are integrated by `population_reduced_rhs`, which
reads `gamma_min`, `alpha_gamma` and `tau_opt` through the senescence-induction
rate, and the morphological observables relax at rates set by `tau_adapt` (aspect
ratio) and `tau_orient` (orientation), which together hold the single
morphological adaptation constant. In the original code the same
`RecedingHorizonMPC` instance performs both the plant advance and the
controller's internal prediction inside `solve`, so the plant and the controller
shared all four identified parameters. There was no separate plant simulation.

## Task 1 decoupling and the zero-perturbation identity

`run_mpc_simulation` was given an optional `plant` argument used only for the two
authoritative `predict_step` calls. When it is None the plant is bound to the
nominal controller, so the reported run is unchanged. This was verified two ways.

1. Zero-perturbation identity. `run_mpc_simulation(plant=None)` and
   `run_mpc_simulation(plant=RecedingHorizonMPC(config))` (a nominal plant)
   produce bit-for-bit identical logs for `tau`, `phi_sen`, `rho_bar`,
   `varphi_bar` and `healthy_align` at the master seed. This proves the nominal
   path is untouched by the mechanism.
2. Dynamics fidelity of the study tool. The reduced-state closed loop used by the
   study reproduces `run_mpc_simulation` bit-for-bit when given the same initial
   condition, confirming that it uses the same `solve` and `predict_step`. The
   study draws the initial population composition from the model's own rules
   (`generate_initial_population`), which reproduces the pipeline composition;
   because every configuration is compared at the same seed, the composition
   cancels in every closed-versus-open contrast.

## Study configuration

Identified parameters perturbed in the plant: the morphological adaptation
constant (nominal 7.4 h, applied to `tau_adapt` and `tau_orient`), `gamma_min`,
`alpha_gamma`, `tau_opt`. One-at-a-time perturbations of plus or minus 20 percent
(the adaptation constant also swept explicitly over 6 to 12 h), and a joint Latin
hypercube of 64 samples over the same ranges. Ten initial-condition seeds per
plant. The epistemic perturbation draws use a dedicated generator seeded at
`config.random_seed + 10000`, kept separate from the aleatory seeds
`config.random_seed + r`. Run length six one-hour intervals, the reported window.
Three configurations on the same plant and seeds: closed-loop MPC with the nominal
internal model, open-loop feedforward (the nominal-plant input applied blindly),
and the nominal-plant closed loop as the no-mismatch reference.

## Which parameters the closed loop is most and least robust to

Sensitivity of the terminal senescent fraction to a plus or minus 20 percent
one-at-a-time perturbation, measured as the swing about the no-mismatch reference
(`terminal phi_sen = 0.2085`):

| parameter | terminal phi_sen swing | rank |
|---|---|---|
| `tau_opt` | 0.0147 (0.2177 at -20 percent to 0.2030 at +20 percent) | most sensitive |
| `gamma_min` | 0.0056 | second |
| `alpha_gamma` | 0.0028 | third |
| adaptation constant | 0.0006 | least sensitive |

The closed loop is therefore least robust, for the senescent fraction, to
`tau_opt`, the location of the minimum of the U-shaped senescence-induction rate,
and most robust to the morphological adaptation constant. For the closed-loop
cost the ranking is inverted: the cost is dominated by the adaptation constant
(a swing of about 2.2 in J across 6 to 12 h, rising monotonically with a slower
plant adaptation), while `gamma_min`, `alpha_gamma` and `tau_opt` move it by less
than 0.2. In every one-at-a-time case the terminal senescent fraction remained
between 0.203 and 0.218, far inside the 0.30 constraint.

## Constraint-violation frequency, closed-loop versus open-loop

Over the joint Latin hypercube of 64 mismatched plants, ten seeds each:

| configuration | mean terminal phi_sen | max terminal phi_sen | p95 terminal phi_sen | fraction of samples with any violation |
|---|---|---|---|---|
| closed-loop (nominal internal model) | 0.2094 | 0.2210 | 0.2189 | 0.000 |
| open-loop feedforward | 0.2092 | 0.2210 | 0.000 (max 0.2210) | 0.000 |

The senescence constraint `phi_sen <= 0.30` was satisfied in every run of every
configuration. The constraint-violation frequency is zero for both the closed
loop and the open-loop feedforward under all tested mismatches.

## The efficacy-of-feedback result

The closed loop regulates the true system within its senescence constraint under
all tested parametric mismatch, with a large margin (the worst terminal senescent
fraction over 64 joint mismatches was 0.221, against the bound of 0.30). This is
the primary result: closed-loop control is robust to plus or minus 20 percent
identification error in the four parameters.

The comparison with open-loop feedforward, however, shows that at this mismatch
level the robustness is not attributable to strong active feedback correction. The
closed-loop control action diverges from the nominal input by at most about 0.004
Pa for a senescence-parameter mismatch and about 0.036 Pa for the adaptation-
constant mismatch, because the optimal input is largely set by driving the
morphology toward its flow-adapted targets and is near-saturated at the upper
bound. Consequently the open-loop feedforward achieves the same zero violation
frequency and nearly the same terminal senescent fraction. The measurable effect
of feedback is a small reduction of the morphological regulation error under joint
mismatch, `rho_bar` by 0.41 percent and `varphi_bar` by 0.34 percent relative to
open loop, at a marginal increase of the senescent-fraction tracking error of 0.14
percent. The senescent fraction is weakly controllable through the input over this
range, so feedback does not correct it appreciably.

For context, the senescent fraction accumulates slowly. Over the six-hour reported
window it rises only from 0.20 to about 0.21, so the constraint is not binding. A
supplementary integration over longer horizons at the master seed shows the
mismatch pushing the terminal senescent fraction toward the bound, reaching about
0.29 at 48 h for the worst `tau_opt` perturbation, while the closed-loop and open-
loop trajectories remain nearly identical at every horizon. The weak
controllability of the senescent fraction therefore persists beyond the reported
window. This is reported as an observation and left for the author to pursue.

## Reproducibility

The reported aggregates are deterministic given the aleatory and epistemic seeds.
The reproducibility self-check (`--verify`) reports a maximum absolute difference
of 0.00e+00 between two runs. A full re-run of the study reproduces the committed
summary to floating-point tolerance.

## Outputs and figures

Directory: `results/mismatch_robustness/20260708-072745/`.

- `raw_runs.csv`, `summary.json`, `summary_table.md`.
- `fig_feedback_efficacy`: terminal senescent fraction and constraint-violation
  frequency, closed loop versus open loop, over the joint mismatch ensemble.
- `fig_tornado`: one-at-a-time sensitivity of the terminal senescent fraction and
  of the cost to the four parameters.
- `fig_joint_distributions`: distributions of the terminal senescent fraction and
  the cost over the Latin hypercube.
- `fig_morph_sweep`: closed-loop regulation across the 6 to 12 h range of the
  plant adaptation constant.
- `fig_tracking_error`: the ratio of closed-loop to open-loop regulation error,
  exposing the small correction that feedback provides.

## Control-theoretic summary

Under parametric plant-model mismatch of plus or minus 20 percent in the four
identified parameters, the receding-horizon closed loop keeps the true system
within its senescence constraint in every case, with a wide margin. The system is
least robust to `tau_opt` for the senescent fraction and to the adaptation
constant for the cost. At this mismatch level the senescence constraint is
satisfied equally by the open-loop feedforward, and the closed-loop control action
departs only marginally from the nominal input, so the robustness follows from the
constraint margin and the near-saturated optimal input rather than from strong
feedback correction. Feedback provides a small improvement of the morphological
regulation. The senescent fraction is weakly controllable through the applied
input over the ranges and horizons examined.

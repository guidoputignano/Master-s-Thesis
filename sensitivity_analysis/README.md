# Sensitivity & robustness analysis

Sensitivity and robustness study of the hierarchical endothelial-mechanoadaptation
model, built on the **corrected** model implementations in
`endothelial_simulation/models/` (see [`audit.md`](audit.md) for the corrections
made to bring the code in line with the published paper, `main.tex`).

## Contents

| File | Purpose |
|---|---|
| `audit.md` | Audit of model code vs. paper; basis for the corrections in `endothelial_simulation/models/` |
| `sensitivity_oat.py` | Part A — one-at-a-time sensitivity of the temporal-dynamics scale |
| `sensitivity_sobol.py` | Part B — global Sobol sensitivity of the reduced population ODE |
| `sensitivity_energy_weights.py` | Part C — sensitivity of the morphological energy weights `w_A:w_rho:w_phi` |
| `sensitivity_mpc_weights.py` | Part D — sensitivity of the MPC cost weights `w_phi:w_rho:w_varphi:w_u` |
| `run_all.py` | Runs Parts A–D and prints a summary table |
| `figures/` | All output figures (600-dpi PDF) |

## Running

```bash
# from the repository root
python -m sensitivity_analysis.run_all      # both parts + summary table
python -m sensitivity_analysis.sensitivity_oat
python -m sensitivity_analysis.sensitivity_sobol
```

Requires Python ≥ 3.10 with `numpy`, `scipy`, `matplotlib`, `SALib`.
A fixed seed (`numpy.random.seed(42)`) is used throughout.

## Part A — temporal dynamics (OAT)

Implements `eq:target`, `eq:relaxation`, `eq:stepsolution`, `eq:orientation`.
Sweeps `tau_adapt`, `rho*`, `theta*`, `tau_act` one at a time around their
Table-1 nominal values, records `rho(t)` and the mean alignment angle `phi(t)`
over the 6 h horizon, and computes normalised sensitivity indices
(+10 % perturbation).

Figures: `oat_envelope_tau_adapt.pdf`, `oat_envelope_theta_star.pdf`,
`oat_sensitivity_bars.pdf`, and two monolayer snapshots at `t = 6 h`
(`oat_monolayer_fast.pdf`, `oat_monolayer_slow.pdf`) produced with the existing
Voronoi/tessellation visualisation for the fastest (6 h) and slowest (12 h)
adaptation.

## Part B — population dynamics (global Sobol)

Integrates the reduced population ODE (`eq:reduced`) at `tau = tau_opt = 1.4 Pa`
over 6 h and records `phi_sen(6h)` (`eq:senfraction`). Uncertain parameters
(`gamma_min`, `alpha_gamma`, `r`, `K`, `xi`) are sampled over their Table-1
ranges with a Saltelli design (`N = 2000` base samples). First-order Sobol
indices are computed with SALib (manual Saltelli fallback included).

Figures: `sobol_first_order_bars.pdf`, `sobol_phisen_hist.pdf` (with the 30 % MPC
constraint), and scatter plots vs. the two most influential parameters.

**Note.** Because the integration is performed exactly at the protective optimum
`tau_opt = 1.4 Pa`, the quadratic curvature `alpha_gamma (tau - tau_opt)^2`
vanishes, so `alpha_gamma` has (correctly) negligible first-order influence; the
senescent fraction is governed by the injury minimum `gamma_min`. Across the full
sampled range `phi_sen(6h)` stays far below the 30 % limit, i.e. the constraint is
robustly satisfied at the protective shear.

## Part C — morphological energy weights (spatial scale)

Table 1 fixes the configurational-energy weights of `eq:argmin` to the relative
values `w_A : w_rho : w_phi = 1 : 8.5 : 5` and states they are *not independently
identifiable* and that the solution is *insensitive to their exact magnitude
within the morphological range of interest*. This part provides that
verification. For several weight sets (equal, nominal, and factor-of-2
perturbations of each mode) it runs the multi-configuration initialiser — where
the weights act, by selecting the energy-minimising layout — followed by a 6 h
adaptation at `tau = 1.4 Pa`, and records the converged population-mean aspect
ratio `rho_bar`, mean alignment `phi_bar`, area coefficient of variation, and the
partition gap fraction.

Figures: `energy_weights_observables.pdf` (observables across the weight sets) and
`energy_weights_sensitivity.pdf` (normalised +10 % sensitivity indices).

The weights are kept fixed (as relative regularisers) rather than fitted, since
they are non-identifiable; the analysis confirms the converged morphology is
effectively independent of their exact magnitudes.

## Part D — MPC cost weights (control law)

The receding-horizon controller (`eq:ocp`, `eq:stagecost`) minimises
`J = w_phi*phi_sen^2 + w_rho*(rho_bar-2.3)^2 + w_varphi*varphi_bar^2 + w_u*dtau^2`
with nominal weights `10 : 1 : 5 : 0.1`. These weights enter the control law
directly (unlike the Part-C morphological energy weights), so this part
quantifies how the optimised protocol and the closed-loop outcome respond to
them. For each weight set (nominal and factor-of-2 perturbations of each weight)
the receding-horizon loop is run for 24 h on the *reduced* prediction model
(population via `solve_ivp`, morphology via the closed-form step response — no
tessellation rendering), recording the final healthy alignment, mean aspect
ratio, final/peak senescent fraction, mean shear and total input variation, plus
normalised +10 % sensitivity indices.

Figures: `mpc_weights_outcomes.pdf` and `mpc_weights_sensitivity.pdf`.

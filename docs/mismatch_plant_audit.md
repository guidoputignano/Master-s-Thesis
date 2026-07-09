# Task 0 audit: the plant advance in `run_mpc_simulation`

Status: read-only audit. No code was modified to produce this note. The
mismatch-robustness mechanism (Task 1) and the study (Tasks 2 to 5) depend on the
conclusions recorded here.

> **POST-REFACTOR NOTE (Task 5).** The STRUCTURAL conclusions below — where the
> true state is advanced (the two `predict_step` sites), that the plant and the
> controller share the same object/parameters unless a distinct plant is
> injected, and that per-cell rendering does not feed back — are unchanged by the
> Task 5 refactor. Only the NAMES of the identified senescence parameters changed:
> the senescence-induction rate `population_reduced_rhs` now reads is the
> monotone Hill law in `gamma_min` / `gamma_max` / `tau_h` (not the quadratic
> `gamma_min` / `alpha_gamma` / `tau_opt` written below), and the `xi` argument
> was removed. Read the `alpha_gamma` / `tau_opt` / `xi` mentions below as the
> pre-refactor parameterisation of the same senescence channel.

## Where the true plant state is advanced

The reported closed loop keeps an authoritative reduced state
`x_state = {pop, rho_h, theta_h}`, where `pop` are the population compartments
`[E_0, ..., E_N, S_tel, S_str]`, `rho_h` is the healthy mean aspect ratio, and
`theta_h` is the healthy mean orientation. The reported hour-boundary observables
(`phi_sen`, `rho_bar`, `varphi_bar`) are computed from this state by
`RecedingHorizonMPC.outputs`.

The true state is advanced between control instants at exactly two call sites in
`run_mpc_simulation`, both invoking `predict_step`:

1. `control/mpc_controller.py:1284`, `x_sub = mpc.predict_step(x0, tau_k, dt=th)`.
   This evaluates the true state at the sub-hour points `th` in `render_minutes`
   and feeds the dashboard sub-frame time series `ts`.
2. `control/mpc_controller.py:1314`, `x_state = mpc.predict_step(x0, tau_k)`.
   This advances the true state across the full one-hour interval. Its result
   feeds the reported `log`, is written back into the population model state
   (`pm.state['E'|'S_tel'|'S_stress']`), and drives `_reconcile_senescence`,
   which maps the compartment counts onto individual cells.

The next iteration takes `x0 = x_state`, that is, the state propagated by
`predict_step`. It is not re-measured from the tessellation, so the per-cell
rendering never feeds back into the authoritative state.

## How each channel is advanced, and which identified parameters it reads

`predict_step` (`control/mpc_controller.py:808`) performs two coupled updates.

(i) Population compartments (senescence). The compartments are integrated over the
interval by `scipy.integrate.solve_ivp` (RK45) applied to `population_reduced_rhs`:

```
population_reduced_rhs(y, tau, self.r, self.K, self.xi,
                       self.gamma_min, self.alpha_gamma, self.tau_opt, self.N)
```

This reads three of the four identified parameters, `self.gamma_min`,
`self.alpha_gamma`, and `self.tau_opt`, through the senescence-induction rate
`gamma(tau) = gamma_min + alpha_gamma (tau - tau_opt)^2`. The remaining arguments
`self.r`, `self.K`, `self.xi`, `self.N` are not part of the identified set for
this study.

(ii) Morphological observables. The aspect ratio and orientation relax by the
closed-form first-order step response:

```
rho_h_new   = rho_t - (rho_t - rho_h) * exp(-dt / self.tau_adapt)
theta_h_new = theta_h + diff * (1 - exp(-dt / self.tau_orient))
```

This reads the fourth identified parameter, the morphological adaptation constant,
which appears as `self.tau_adapt` for the aspect ratio and `self.tau_orient` for
the orientation. In the reported model these two attributes hold one physical
constant of 7.4 h. The morphological targets `rho_t = self.rho_target(tau)` and
`th_t = self.theta_target(tau)` depend on `tau_act` and the static and flow
plateau values, none of which are in the identified set, so the targets are
invariant under the perturbations considered here. Only the relaxation rates are
affected.

## Do the plant advance and the controller share these parameters?

Yes, completely. The controller solves the optimal control problem with
`mpc.solve` (`:1262`), which calls `self._rollout` (`:867`), which calls
`self.predict_step` (`:872`). Both the plant advance (the two sites above) and the
controller's internal prediction therefore call `predict_step` on the same single
`RecedingHorizonMPC` instance `mpc`, and consequently read the same
`self.gamma_min`, `self.alpha_gamma`, `self.tau_opt`, `self.tau_adapt`, and
`self.tau_orient`. There is no separate plant simulation in the current code. The
true dynamics that are advanced and the controller's internal model are one and
the same object with one and the same parameters.

## Consequence for Task 1

To introduce parametric plant-model mismatch it is sufficient, and minimal, to
route the two authoritative advance calls (`:1284` and `:1314`) through a separate
`plant` object that carries perturbed values of the four identified parameters,
while leaving `mpc.solve` on the nominal instance. When no distinct plant is
supplied the plant reference is bound to `mpc`, so the two calls remain literally
`mpc.predict_step` and the reported run is unchanged.

## Note on the per-cell rendering

The per-cell morphology updates inside the interval
(`temporal.relax_step`, `temporal.orientation_step` at `:1295`, `:1296`, `:1307`,
`:1308`) drive the rendered Voronoi frames only. They read the nominal
morphological constant (`temporal.tau_adapt_hours` from the config, and
`mpc.tau_orient`). They are a visualisation layer and do not enter the reported
reduced-state observables, so they are outside the plant advance defined above and
are left on the nominal constant. The reported observables are governed entirely
by the authoritative `predict_step` advance and therefore entirely by the plant
once the mechanism of Task 1 is in place.

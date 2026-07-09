# STALE — pre-Task-5-refactor results

The artifacts under this directory (figures, `raw_runs.csv`, `raw_hourly.csv`,
`summary.json`, `summary_table.md`) were generated BEFORE the Task 5 refactor and
describe the **pre-refactor** reported model:

- senescence-induction rate = symmetric quadratic
  `gamma(tau) = gamma_min + alpha_gamma (tau - tau_opt)^2` (now a monotone Hill law);
- MPC cost included a soft senescence term `w_phi * phi_sen^2` (senescence is now a
  **hard constraint**; `w_phi` removed);
- no explicit move/slew bound (a single hard bound `|tau(k)-tau(k-1)| <= delta_tau_max`
  was added).

`analysis/horizon_sensitivity.py` was updated to run against the new model (the
realised cost `J` no longer contains the `w_phi` term), but these outputs were NOT
regenerated. Re-run:

    python -m analysis.horizon_sensitivity --out results/horizon_sensitivity

to refresh them before quoting post-refactor numbers.

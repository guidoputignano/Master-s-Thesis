# STALE — pre-Task-5-refactor results

The artifacts under this directory (figures, `raw_runs.csv`, `summary.json`,
`summary_table.md`) were generated BEFORE the Task 5 refactor and describe the
**pre-refactor** reported model:

- senescence-induction rate = symmetric quadratic
  `gamma(tau) = gamma_min + alpha_gamma (tau - tau_opt)^2` (now a monotone Hill law
  `gamma_min`/`gamma_max`/`tau_h`);
- MPC cost included a soft senescence term `w_phi * phi_sen^2` (senescence is now a
  **hard constraint**; `w_phi` removed);
- identified-parameter set was `{morph, gamma_min, alpha_gamma, tau_opt}` (now
  `{morph, gamma_min, gamma_max, tau_h}`).

`analysis/mismatch_robustness.py` was updated to run against the new model, but
these outputs were NOT regenerated. See `docs/mismatch_robustness_report.md` for
the full note. Re-run the study to refresh them before quoting post-refactor
numbers.

**Current run (24 September 2026):** `20260924-073004`, with the 3 h adaptation
constant and the paper's settings (`--n-rep 5 --n-lhs 24`). `20260724-090500` is the
same study on the previous 7.4 h model, as reported in the paper. See
`docs/tau_adapt_plateau.md`.

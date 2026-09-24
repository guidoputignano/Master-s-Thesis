# Senescence within a conditioning session: inherited, not induced

The reported model now holds the senescent fraction of the seeded batch constant
during a conditioning session. It no longer has an injury term, and the 30 % cap
is an admission check on the batch. With realistic biology, shear in 0–2 Pa only
helps, so the controller goes to 2 Pa within the first hour and holds it. At 6 h
the healthy cells are at 20.4° and the population at 27.8° (30 % senescent cells).
The trade-off in the thesis model came from an injury term that was added for
that purpose. The senescent share of the seeded batch, not the protocol, sets
how far the monolayer can align.

## Why

The thesis law, γ(τ) = γ_min + (γ_max − γ_min)·τ_h²/(τ_h² + τ²) + γ_d·τ²/(τ_d² + τ²),
converted 15 % of healthy cells to senescence in 6 h at 1.4 Pa (0.027 per hour),
2.2× faster than without flow. The config recorded the reason for the injury arm:
without it "the optimiser just saturates the shear ceiling and the phi_sen<=0.30
constraint is never active". The literature does not support either part:

| Question | Finding | Source |
|---|---|---|
| New senescence within hours? | None shown within 6–24 h under any shear. Disturbed flow needs days in vitro and weeks in vivo; TNF-α 3–6 days | Warboys et al., ATVB 2014; Kandhaya-Pillai et al., Aging 2017 |
| Does laminar 1–2 Pa cause or prevent it? | Prevents it, over ≥ 36 h (SA-β-gal 50 % → 18 % in an H₂O₂ model) | Korean J Exerc Physiol 2021; Warboys et al. 2014 |
| Growth arrest under laminar shear | p21 rises and cells stop dividing at 0.5–3 Pa: the protective response, not senescence | Akimoto et al., Circ Res 2000 |
| Injury in 0–2 Pa? | None: 7.5 Pa for 24 h is cytoprotective; cells stay attached at 10 Pa for 24 h; acute erosion needs about 38 Pa | White et al., J Cell Physiol 2011; Am J Physiol Cell Physiol 2012 (PMC3330730); Fry, Circ Res 1968 |
| Does a slow ramp matter? | A sudden onset triggers proliferation and inflammatory genes; a 30 s ramp avoids it | White et al., Circulation 2001; Bao et al., ATVB 1999 |

So within a session the senescent fraction is a property of the seeded cells. It
depends on passage, donor and, in A1, the TNF-α pre-treatment. In A1 the design
was 30 %, and the images agree: 30 % [25–35] of static cells have enlarged nuclei,
and 72 % of those nuclei (77 % under flow) carry doubled DNA, as senescent HUVEC
do.

## What changed

| Setting | Before | After |
|---|---|---|
| `CONSTANT_SENESCENT_FRACTION` | (not present: the ODE ran) | `True`: population held within the session |
| `INCLUDE_SUPRAPHYSIOLOGICAL_ARM` | `True` (γ_d = 0.05 per hour, τ_d = 1.5 Pa) | `False` |
| φ_sen(0) | 0.20 | 0.30, the A1 design |
| φ_sen ≤ 0.30 | Hard constraint over the horizon | Admission check on φ_sen(0), `RecedingHorizonMPC.admitted` |
| Move bound | 0.5 Pa per hour | The whole band (a 30 s ramp is below the 1 h step) |

The thesis model stays available: set `CONSTANT_SENESCENT_FRACTION = False` and
`INCLUDE_SUPRAPHYSIOLOGICAL_ARM = True`, and pass `delta_tau_max=0.5`.

## Effect on the reported run

Six one-hour steps, N_p = 6, N_c = 3, master seed (`python -m analysis.senescence_realism`,
results in `results/senescence_realism/`):

| Scenario | τ(k), Pa | φ_sen at 6 h | ρ̄ at 6 h | Healthy | Population |
|---|---|---|---|---|---|
| Thesis law (injury, induction, φ0 0.20, 0.5 Pa/h) | 0.50, 1.00, 1.43, 1.27, 1.06, 0.88 | 0.270 | 2.115 | 28.9° | 33.2° |
| **Reported (inherited 0.30)** | 1.78, 2.00, 2.00, 2.00, 2.00, 2.00 | 0.302 | 2.205 | 20.4° | 27.8° |
| Reported, 0.5 Pa/h ramp limit | 0.50, 1.00, 1.50, 2.00, 2.00, 2.00 | 0.302 | 2.177 | 22.9° | 29.6° |
| Reported, dense monolayer (τ = 6 h) | 1.72, 2.00, 2.00, 2.00, 2.00, 2.00 | 0.302 | 2.130 | 27.1° | 32.5° |
| Reported, inherited 0.20 | 1.81, 2.00, 2.00, 2.00, 2.00, 2.00 | 0.201 | 2.234 | 20.4° | 25.4° |

The first move stops short of 2 Pa because of the move penalty w_u.

## Design quantities (`fig_design`)

- **Plateau against senescent share.** At the plateau the population alignment is
  (1 − φ)·θ_h + φ·45°, with θ_h = 20° at 1.4 Pa and 16.5° at 2 Pa. At 1.4 Pa this
  gives 20 / 27.5 / 37.5 / 45° for 0 / 30 / 70 / 100 %; Stefopoulos et al. 2022
  measured 21 / 31 / 37 / 47°.
- **Admission threshold.** At 2 Pa, a 22.5° target admits at most 21 % senescent
  cells, 25° at most 30 %, and 27.5° at most 39 %.
- **Duration.** The time to 90 % / 95 % of the change is 6.9 / 9.0 h for τ = 3 h,
  13.8 / 18 h for τ = 6 h and 18.4 / 24 h for τ = 8 h. Dense monolayers, such as A1,
  need longer.
- **The 2 Pa target is extrapolated.** The plateau was measured at 1.4 Pa only; the
  2 Pa targets come from the shear gate s(τ) = 1 − exp(−(τ − τ_act)/τ_act), with
  τ_act = 0.5 Pa. For τ_act from 0.3 to 0.7 Pa the healthy-cell target at 2 Pa runs
  from 19.4° to 11.6°, and the share a 25° population target admits from 22 % to
  40 %. Without the extrapolation, at the measured 1.4 Pa, a 25° target admits
  20 %. Measuring the plateau at 2 Pa would remove this uncertainty.

## The paper run (12 h)

`python -m endothelial_simulation.run_mpc --steps 12 --out results/paper_run/20260924-realistic`
runs the reported model for twice the A1 flow (seed 42, φ_sen(0) = 0.302, 54 of
179 cells). The input is 1.78 Pa in the first hour, then 2 Pa. The healthy cells
reach 20.4° and aspect ratio 2.29 at 6 h, and 17.1° and 2.35 at 12 h (targets
16.5° and 2.36). The population reaches 27.8° and 2.21 at 6 h, and 25.5° and 2.24
at 12 h. The folder holds the frames, the animation, the dashboard, the four
summary figures and `log.json`; the paper's Figs. 5–6 come from it.

## What this leaves open

- **Gaps.** Senescent cells loosen junctions under flow and damage their young
  neighbours' junctions (Exarchos et al. 2022; Krouwer et al. 2012). Our 20 Dec flow
  slide had more gaps, next to enlarged cells more often than their size explains.
  No study gives gap area against senescent share or against the protocol, so no
  gap state is modelled. It would need gap or permeability time courses for 0 / 30 /
  70 % mixtures under flow.
- **Multi-day conditioning up to device shear.** A VAD cannula tip exceeds 9 Pa, and
  six days of graded conditioning cut cell loss 130-fold on grafts (Ott & Ballermann
  1995). Senescence and adhesion then evolve, and the controller would matter. The
  ODE and the injury arm remain in the code for such a study, but with magnitudes
  that would first need calibrating.
- **Studies that become trivial.** With the input at the top of the band, the horizon,
  cost-weight and mismatch studies change nothing. The Sobol study of the senescence
  ODE no longer describes the session.

## Verification

- `pytest tests`: 67 passed at this change, 72 with the imaging tests added since.
  A new test checks that the population is held, that the cap is an admission check
  (a 50 % batch fails it), and that the controller reaches 2 Pa. The seed-sensitivity test now runs the evolving-population mode,
  because with an inherited fraction the logged outputs no longer depend on the
  layout seed.

# Results of the conditioning package

Written by `python -m conditioning.run_all` (see `conditioning/run_all.py`). Every number and figure of
the paper *Target-Specific Shear Conditioning of Endothelial Monolayers for Blood-Contacting Devices*
comes from these files; none of them holds images or per-cell data.

| File | Content | Paper |
|---|---|---|
| `calibration.json` | best fit, chi2, fit table (every published value with its prediction), tempered ensemble (320 sets), held-out and leave-one-out tests, scenario ensembles with the switch shear held at 2.2, 3 and 4 Pa (`scenarios`) and at 5.5 Pa (`scenarios_extra`, controller only) | Figs. 3-4, Tables S1-S2 |
| `profile_tau_x.json` | chi2 profile over the switch shear (2-6 Pa), all other parameters refitted, split by data source | Fig. S1 |
| `experiment.json` | the designed arm B (8 Pa, 8 h budget), trajectories of every path (median and 5-95 %), registered differences B minus A | Fig. 7, `EXPERIMENT_8PA.md` |
| `conditioning_map.json` | designed path and reference paths for 11 targets on flat silicone, breath figures and gratings across the flow | Fig. 5 |
| `device.json` | HVAD inflow cannula: designed pump-flow profile, per-region and area-weighted outcomes for four strategies, flat surface and gratings above 3 Pa | Fig. 6 |
| `closed_loop.json` | arm B in open loop, in open loop designed up to 5.5 Pa, and in closed loop, against three monolayers (switch at 2.2, 3 and 5.5 Pa) | Fig. 8 |
| `calibration_v1_substrate_limits.json`, `model_v1_substrate_limits.py.txt` | the first model version (a shear limit on every domain) and its failed prediction for gratings across the flow | Fig. 4a |

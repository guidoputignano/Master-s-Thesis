# Results of the conditioning package

Written by `python -m conditioning.run_all --recalibrate` (see `conditioning/run_all.py`). Every
number and figure of the paper *Target-Specific Shear Conditioning of Endothelial Monolayers for
Blood-Contacting Devices* comes from these files; none of them holds images or per-cell data. All
design, device and closed-loop results are simulations of the model.

| File | Content | Paper |
|---|---|---|
| `calibration.json` | best fit, chi2, fit table (every published value with its prediction), ensemble (320 sets: the chain, of four started at the best fit, with the lowest mean chi2; `chains` holds the diagnostics of all four, which settle in different modes), held-out predictions (breath figures), leave-out refits (`leave_out`, six groups defined in `calibrate.LEAVE_OUT`), scenario ensembles with the switch shear held at 2.2, 3 and 4 Pa (`scenarios`) and at 5.5 Pa (`scenarios_extra`), 32 distinct draws each: designs and the controller's belief use every other draw, the closed loop draws its simulated monolayers from the rest | Figs. 3-4, Tables S1-S2 |
| `profile_tau_x.json` | chi2 profile over the switch shear (1.5-6 Pa; the two values below the 2 Pa bound show what the bound imposes), all other parameters refitted, split by data source | Fig. S1 |
| `experiment.json` | the designed arm B (8 Pa, 8 h budget), trajectories of every path (median and 5-95 %), simulated differences B minus A | Fig. 7, `EXPERIMENT_8PA.md` |
| `conditioning_map.json` | designed path and reference paths for 11 targets on flat silicone, breath figures and gratings across the flow | Fig. 5 |
| `device.json` | HVAD inflow cannula: designed pump-flow profile, per-region and area-weighted outcomes for four strategies, flat surface and gratings above 3 Pa | Fig. 6 |
| `closed_loop.json` | arm B in open loop, in open loop designed up to 5.5 Pa, and in closed loop, against four simulated monolayers per switch scenario (2.2, 3 and 5.5 Pa; not in the controller's belief; own noise seeds), and for 2.2 and 3 Pa also with a tenfold smaller move penalty | Fig. 8 |
| `calibration_v1_substrate_limits.json`, `model_v1_substrate_limits.py.txt` | the first model version (a shear limit on every domain) and its failed prediction for gratings across the flow, archived as run (not rerun by `run_all`) | Fig. 4a |

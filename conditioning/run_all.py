"""Reproduce every conditioning result used in the paper.

    python -m conditioning.run_all [--out results/conditioning] [--recalibrate] [--quick]
                                   [--parts experiment map device profile loop]

Writes JSON files to --out: calibration.json (if --recalibrate or missing), experiment.json,
conditioning_map.json, device.json, profile_tau_x.json, closed_loop.json (parts can run in
separate processes).
Designs use 32 sets of the calibration ensemble (seed 0) and 16 sets from each scenario of the
switch shear (2.2, 3 and 4 Pa, refitted with the switch held fixed), so that they hold wherever
the switch lies. Each design is reported as its simplest near-optimal form: the best two-level
path (one level from the start, then the target) when it scores within 0.01 of the optimum.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import calibrate as C
from . import closed_loop as CL
from . import design as D
from . import device as DV
from . import model as M


def subset(samples, n=64, seed=0):
    rng = np.random.default_rng(seed)
    return samples[rng.choice(len(samples), min(n, len(samples)), replace=False)]


def trajectories(paths, surface, thetas, t_end):
    """Ensemble percentiles (5, 50, 95) of angle, ar, ci, retention along each path."""
    t = np.arange(0.0, t_end + 1e-9, M.DT)
    out = {}
    for name, sh in paths.items():
        sh = sh[:len(t)]
        res = M.simulate(np.repeat(sh[:, None], len(thetas), axis=1), thetas.T,
                         [M.SURFACES[surface]] * len(thetas))
        out[name] = dict(shear=sh.tolist(), **{
            q: np.percentile(res[q], [5, 50, 95], axis=1).tolist() for q in ("angle", "ar", "ci", "retention")})
    return dict(t=t.tolist(), paths=out)


def experiment(thetas, t_end=20.0):
    """The single experiment: a slow ramp to 8 Pa against the onset the model designs, both until
    20 h; the other known paths for reference. Flat silicone and breath figures."""
    tg = D.Target(tau=8.0, surface="silicone_flat", budget=8.0, hold=2.0)
    x_opt, s_opt = D.optimise(tg, thetas, maxiter=80)
    x, score, simple = D.simplify(x_opt, tg, thetas)
    t = np.arange(0.0, t_end + 1e-9, M.DT)

    def seg(knots_or_fn):
        return knots_or_fn(t)
    designed = D.path(x, tg)[:len(t)]
    paths = {
        "A: slow ramp (8 h)": np.minimum(t / 8.0, 1.0) * 8.0,
        "B: model's onset": designed,
        "direct": np.full_like(t, 8.0),
        "usual start (1 h at 1.4 Pa)": np.where(t < 1.0, 1.4, 8.0),
        "aligned first (8 h at 1.4 Pa)": np.where(t < 8.0, 1.4, 8.0),
    }
    res = {s: trajectories(paths, s, thetas, t_end) for s in ("silicone_flat", "silicone_bf")}
    # paired difference B - A at the end, per parameter set (the registered prediction)
    diffs = {}
    for s in ("silicone_flat", "silicone_bf"):
        ends = {}
        for name in ("A: slow ramp (8 h)", "B: model's onset"):
            r = M.simulate(np.repeat(paths[name][:, None], len(thetas), axis=1), thetas.T,
                           [M.SURFACES[s]] * len(thetas))
            ends[name] = {q: r[q][-1] for q in ("angle", "ci", "retention", "ar")}
        diffs[s] = {q: np.percentile(ends["B: model's onset"][q] - ends["A: slow ramp (8 h)"][q], [5, 50, 95]).tolist()
                    for q in ("angle", "ci", "retention", "ar")}
    return dict(designed_knots=x.tolist(), knot_h=D.KNOT, robust_score=score, simple=simple,
                optimum_knots=x_opt.tolist(), optimum_score=s_opt, results=res, difference_B_minus_A=diffs)


def _map_one(args):
    s, tau, thetas, budget = args
    tg = D.Target(tau=tau, surface=s, budget=budget, hold=2.0)
    t0 = time.time()
    x_opt, s_opt = D.optimise(tg, thetas, maxiter=60)
    x, score, simple = D.simplify(x_opt, tg, thetas)
    comp = D.compare(tg, thetas, designed=x)
    print(f"  map {s:14s} {tau:5.1f} Pa: designed {comp[0]['robust']:.2f} "
          f"(best baseline {max(r['robust'] for r in comp[1:]):.2f}) {time.time() - t0:.0f} s", flush=True)
    return dict(surface=s, target=tau, knots=x.tolist(), simple=simple, optimum_knots=x_opt.tolist(),
                optimum_score=s_opt, comparison=comp)


def conditioning_map(thetas, targets, surfaces, budget=12.0, processes=3):
    from multiprocessing import Pool
    jobs = [(s, tau, thetas, budget) for s in surfaces for tau in targets]
    with Pool(processes) as pool:
        rows = pool.map(_map_one, jobs, chunksize=1)
    return dict(budget=budget, knot_h=D.KNOT, post_h=D.POST, rows=rows)


def device_case(thetas, budget=12.0):
    flat = ["silicone_flat"] * len(DV.REGIONS)
    graded = ["silicone_perp" if tau > 3.0 else "silicone_flat" for tau in DV.REGIONS]
    out = {}
    for label, surfs in (("flat everywhere", flat), ("gratings across the flow above 3 Pa", graded)):
        x_opt, s_opt = DV.optimise_device(thetas, surfs, budget=budget)
        x, score, simple = DV.simplify_device(x_opt, thetas, surfs, budget=budget)
        nk = len(x)
        tk = np.arange(nk) * D.KNOT
        strategies = {
            "in-device, designed": DV.u_path(x, budget),
            "in-device, direct": DV.u_path(np.ones(nk), budget),
            "in-device, slow ramp (8 h)": DV.u_path(np.minimum(tk / 8.0, 1.0), budget),
        }
        rows = DV.evaluate_device(np.array(list(strategies.values())), thetas, surfs, budget)
        res = {k: dict(area=r["area"], per_region={q: v.tolist() for q, v in r["per_region"].items()})
               for k, r in zip(strategies, rows)}
        ch = DV.chamber_then_implant(thetas, surfs, 1.4, budget)
        res["chamber at 1.4 Pa, then implanted"] = dict(area=ch["area"],
                                                        per_region={q: v.tolist() for q, v in ch["per_region"].items()})
        out[label] = dict(surfaces=surfs, designed_u=x.tolist(), simple=simple, optimum_u=x_opt.tolist(),
                          optimum_score=s_opt, strategies=res)
        for k, r in res.items():
            a = r["area"]
            print(f"  device {label} | {k}: connected {a['connected']:.2f} matched {a['matched']:.2f} "
                  f"retention {a['retention']:.2f} robust {a.get('robust', float('nan')):.2f}", flush=True)
    return dict(regions_pa=DV.REGIONS.tolist(), weights=DV.WEIGHTS.tolist(),
                extrapolated=DV.EXTRAPOLATED.tolist(), budget=budget, cases=out)


def closed_loop_case(thetas, extra, scenarios, t_end=20.0, seed=0):
    """Arm B of the experiment (8 Pa, 8 h budget) against monolayers the controller does not know:
    the median parameter set of each switch scenario (2.2 and 3 Pa, inside the range the designs
    cover; 5.5 Pa, the second minimum of the profile, which the prior disfavours). Three strategies:
    the open-loop design robust over the 80 design sets (arm B), an open-loop design robust also
    over the 16 sets refitted with the switch at 5.5 Pa, and closed loop from arm B with a belief
    over all 96 sets."""
    tg = D.Target(tau=8.0, surface="silicone_flat", budget=8.0, hold=2.0)
    x_open, _, _ = D.simplify(D.optimise(tg, thetas, maxiter=80)[0], tg, thetas)
    prior = np.vstack([thetas, extra])
    x_wide, _, _ = D.simplify(D.optimise(tg, prior, maxiter=80)[0], tg, prior)
    t = np.arange(0.0, t_end + 1e-9, M.DT)
    plants = {}
    for label, sets in scenarios:
        pl = np.median(sets, axis=0)
        run = CL.run(tg, prior, pl, x_open, seed=seed)
        wide = D.evaluate(D.path(x_wide, tg)[None], tg, pl[None])
        traj = {}
        for name, kn in (("open", x_open), ("open, wide", x_wide), ("closed", run["knots"])):
            sh = D.path(kn, tg)[:len(t)]
            r = M.simulate(sh[:, None], pl[:, None], [M.SURFACES[tg.surface]])
            traj[name] = dict(shear=sh[::6].tolist(), **{q: r[q][::6, 0].tolist() for q in ("angle", "ci", "retention")})
        plants[label] = dict(plant=pl.tolist(), history=run["history"], closed=run["closed"], open=run["open"],
                             open_wide={k: float(v[0, 0]) for k, v in wide.items()},
                             closed_knots=run["knots"].tolist(), trajectories=traj)
        print(f"  closed loop, {label}: open matched {run['open']['matched']:.2f} R {run['open']['retention']:.3f}"
              f" | wide matched {plants[label]['open_wide']['matched']:.2f} R {plants[label]['open_wide']['retention']:.3f}"
              f" | closed matched {run['closed']['matched']:.2f} R {run['closed']['retention']:.3f}", flush=True)
    return dict(target=tg.tau, budget=tg.budget, open_knots=x_open.tolist(), wide_knots=x_wide.tolist(),
                t_traj=t[::6].tolist(), prior_sets=len(prior), plants=plants)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/conditioning")
    ap.add_argument("--recalibrate", action="store_true")
    ap.add_argument("--quick", action="store_true", help="fewer targets and surfaces")
    ap.add_argument("--parts", nargs="+", default=["experiment", "map", "device", "profile", "loop"])
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cal = out / "calibration.json"
    if a.recalibrate or not cal.exists():
        theta, c2, pb = C.fit(n_starts=16, max_nfev=2500)
        samples, temp = C.ensemble(theta, pb, n_steps=60000, thin=150)
        C.save(cal, theta, c2, samples, pb, temp, C.held_out(samples, theta))
    theta, samples, d = C.load(cal)
    if "scenarios" not in d:
        sc = C.scenario_ensemble(theta)
        d["scenarios"] = [dict(value=x["value"], chi2=x["chi2"], samples=x["samples"].tolist()) for x in sc]
        json.dump(d, open(cal, "w"), indent=1)
    if "scenarios_extra" not in d and "loop" in a.parts:
        # the controller's belief also carries a switch outside the range the data allow
        sc = C.scenario_ensemble(theta, values=(5.5,))
        d["scenarios_extra"] = [dict(value=x["value"], chi2=x["chi2"], samples=x["samples"].tolist()) for x in sc]
        json.dump(d, open(cal, "w"), indent=1)
    thetas = np.vstack([subset(samples, 32)] + [np.array(x["samples"]) for x in d["scenarios"]])
    t0 = time.time()
    if "experiment" in a.parts:
        json.dump(experiment(thetas), open(out / "experiment.json", "w"))
        print(f"experiment done ({time.time() - t0:.0f} s)", flush=True)
    if "map" in a.parts:
        targets = [1.0, 2.0, 3.0, 5.0, 8.0] if a.quick else [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
        surfaces = ["silicone_flat"] if a.quick else ["silicone_flat", "silicone_bf", "silicone_perp"]
        json.dump(conditioning_map(thetas, targets, surfaces), open(out / "conditioning_map.json", "w"))
        print(f"map done ({time.time() - t0:.0f} s)", flush=True)
    if "device" in a.parts:
        json.dump(device_case(thetas), open(out / "device.json", "w"))
        print(f"device done ({time.time() - t0:.0f} s)", flush=True)
    if "profile" in a.parts:
        json.dump(C.profile(theta), open(out / "profile_tau_x.json", "w"), indent=1)
        print(f"profile done ({time.time() - t0:.0f} s)", flush=True)
    if "loop" in a.parts:
        extra = np.vstack([np.array(x["samples"]) for x in d["scenarios_extra"]])
        scen = [(f"switch at {x['value']:g} Pa", np.array(x["samples"])) for x in d["scenarios"] if x["value"] in (2.2, 3.0)]
        scen += [(f"switch at {x['value']:g} Pa (outside the design range)", np.array(x["samples"]))
                 for x in d["scenarios_extra"]]
        json.dump(closed_loop_case(thetas, extra, scen), open(out / "closed_loop.json", "w"))
        print(f"closed loop done ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()

"""Reproduce every conditioning result used in the paper (all of them simulations).

    python -m conditioning.run_all [--out results/conditioning] [--recalibrate] [--quick]
                                   [--parts experiment map device profile loop]

Writes JSON files to --out: calibration.json (fit, ensemble, held-out predictions, leave-out
refits and scenario ensembles; recomputed with --recalibrate or when missing), experiment.json,
conditioning_map.json, device.json, profile_tau_x.json, closed_loop.json (parts can run in
separate processes). The first model version (calibration_v1_substrate_limits.json) is archived as
it was run, with its code as text; it is not rerun here.
Designs use 32 sets of the calibration ensemble (seed 0) and 16 sets from each scenario of the
switch shear (2.2, 3 and 4 Pa, sampled with the switch held fixed; every other set of each chain),
so that they hold wherever the switch lies. Each design is reported as its simplest near-optimal
form: the direct step, or the best two-level path (one level from the start, then the target),
when it scores within 0.01 of the optimum.
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


def scenario_design(x):
    """Every other set of a scenario chain: the sets used for design and in the controller's belief."""
    return np.array(x["samples"])[::2]


def scenario_held(x):
    """The other sets of a scenario chain: monolayers the controller does not know."""
    return np.array(x["samples"])[1::2]


def design_thetas(d, samples):
    """The 80 design sets: 32 of the main ensemble (seed 0) and 16 of each switch scenario."""
    return np.vstack([subset(samples, 32)] + [scenario_design(x) for x in d["scenarios"]])


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
        "1 h at 1.4 Pa first": np.where(t < 1.0, 1.4, 8.0),
        "aligned first (8 h at 1.4 Pa)": np.where(t < 8.0, 1.4, 8.0),
    }
    res = {s: trajectories(paths, s, thetas, t_end) for s in ("silicone_flat", "silicone_bf")}
    # paired difference B - A at the end, per parameter set (the predicted difference)
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
    s, tau, thetas, budget = args[:4]
    x_prev = args[4] if len(args) > 4 else None
    tg = D.Target(tau=tau, surface=s, budget=budget, hold=2.0)
    t0 = time.time()
    if x_prev is None:
        x_opt, s_opt = D.optimise(tg, thetas, maxiter=60)
    else:       # reuse a stored optimum; only the simplification and the comparison are redone
        x_opt = np.asarray(x_prev, float)
        s_opt = float(D.robust(D.evaluate(D.path(x_opt, tg)[None], tg, thetas)["score"])[0])
    x, score, simple = D.simplify(x_opt, tg, thetas)
    comp = D.compare(tg, thetas, designed=x)
    print(f"  map {s:14s} {tau:5.1f} Pa: designed {comp[0]['robust']:.2f} "
          f"(best baseline {max(r['robust'] for r in comp[1:]):.2f}) {time.time() - t0:.0f} s", flush=True)
    return dict(surface=s, target=tau, knots=x.tolist(), simple=simple, optimum_knots=x_opt.tolist(),
                optimum_score=s_opt, comparison=comp)


def conditioning_map(thetas, targets, surfaces, budget=12.0, processes=3, previous=None):
    """Designs for every target and surface; with `previous` (a stored map), its optima are reused."""
    from multiprocessing import Pool
    prev = {} if previous is None else {(r["surface"], r["target"]): r["optimum_knots"] for r in previous["rows"]}
    jobs = [(s, tau, thetas, budget, prev.get((s, tau))) for s in surfaces for tau in targets]
    with Pool(processes) as pool:
        rows = pool.map(_map_one, jobs, chunksize=1)
    return dict(budget=budget, knot_h=D.KNOT, post_h=D.POST, rows=rows)


def device_case(thetas, budget=12.0, previous=None):
    """With `previous` (a stored device.json), its optima are reused: the operating flow at once is
    taken if it scores within 0.01 of the optimum, else the stored simplified path is kept."""
    flat = ["silicone_flat"] * len(DV.REGIONS)
    graded = ["silicone_perp" if tau > 3.0 else "silicone_flat" for tau in DV.REGIONS]
    out = {}
    for label, surfs in (("flat everywhere", flat), ("gratings across the flow above 3 Pa", graded)):
        prev = None if previous is None else previous["cases"].get(label)
        if prev is None:
            x_opt, s_opt = DV.optimise_device(thetas, surfs, budget=budget)
            x, score, simple = DV.simplify_device(x_opt, thetas, surfs, budget=budget)
        else:
            x_opt, s_opt = np.array(prev["optimum_u"]), prev["optimum_score"]
            ones = np.ones(len(x_opt))
            s_direct = DV.evaluate_device(DV.u_path(ones, budget)[None], thetas, surfs, budget)[0]["area"]["robust"]
            if s_direct >= s_opt - 0.01:
                x, score, simple = ones, s_direct, True
            else:
                x, simple = np.array(prev["designed_u"]), prev["simple"]
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


def _loop_one(args):
    tg, prior, plant, x_open, seed, mu = args
    return CL.run(tg, prior, plant, x_open, seed=seed, mu=mu)


def closed_loop_case(thetas, extra, scenarios, t_end=20.0, n_plants=4, processes=4):
    """Arm B of the experiment (8 Pa, 8 h budget) against monolayers the controller does not know.
    `scenarios`: (label, held-out sets, switch value). Per scenario (2.2 and 3 Pa, inside the range
    the designs cover; 5.5 Pa, the second minimum of the profile, which the prior disfavours), the
    plants are n_plants parameter sets drawn from the same scenario chain but not among the
    controller's sets, each with its own measurement-noise seed. Three strategies: the open-loop
    design robust over the 80 design sets (arm B), an open-loop design robust also over the 16 sets
    with the switch at 5.5 Pa, and closed loop from arm B with a belief over all 96 sets (uniform
    prior weights). For the 2.2 and 3 Pa scenarios the closed loop is also run with a move penalty
    ten times smaller."""
    from multiprocessing import Pool
    tg = D.Target(tau=8.0, surface="silicone_flat", budget=8.0, hold=2.0)
    x_open, _, _ = D.simplify(D.optimise(tg, thetas, maxiter=80)[0], tg, thetas)
    prior = np.vstack([thetas, extra])
    x_wide, _, _ = D.simplify(D.optimise(tg, prior, maxiter=80)[0], tg, prior)
    t = np.arange(0.0, t_end + 1e-9, M.DT)
    jobs, keys = [], []
    for label, held, value in scenarios:
        for j in range(min(n_plants, len(held))):
            jobs.append((tg, prior, held[j], x_open, j, None))
            keys.append((label, j, "closed"))
            if value in (2.2, 3.0):
                jobs.append((tg, prior, held[j], x_open, j, D.MU_TV / 10))
                keys.append((label, j, "mu/10"))
    with Pool(processes) as pool:
        res = dict(zip(keys, pool.map(_loop_one, jobs, chunksize=1)))
    plants = {}
    for label, held, value in scenarios:
        reps = []
        for j in range(min(n_plants, len(held))):
            pl = np.asarray(held[j], float)
            run = res[(label, j, "closed")]
            wide = D.evaluate(D.path(x_wide, tg)[None], tg, pl[None])
            rep = dict(plant=pl.tolist(), seed=j, tau_x=float(pl[M.NAMES.index("tau_x")]), history=run["history"],
                       open=run["open"], open_wide={k: float(v[0, 0]) for k, v in wide.items()},
                       closed=run["closed"], closed_knots=run["knots"].tolist())
            if (label, j, "mu/10") in res:
                rep["closed_mu_tenth"] = res[(label, j, "mu/10")]["closed"]
                rep["closed_mu_tenth_knots"] = res[(label, j, "mu/10")]["knots"].tolist()
            if j == 0:
                traj = {}
                for name, kn in (("open", x_open), ("open, wide", x_wide), ("closed", run["knots"])):
                    sh = D.path(kn, tg)[:len(t)]
                    r = M.simulate(sh[:, None], pl[:, None], [M.SURFACES[tg.surface]])
                    traj[name] = dict(shear=sh[::6].tolist(),
                                      **{q: r[q][::6, 0].tolist() for q in ("angle", "ci", "retention")})
                rep["trajectories"] = traj
            reps.append(rep)
            extra_txt = (f" | mu/10 matched {rep['closed_mu_tenth']['matched']:.2f} "
                         f"R {rep['closed_mu_tenth']['retention']:.3f}" if "closed_mu_tenth" in rep else "")
            print(f"  closed loop, {label}, plant {j} (tau_x {rep['tau_x']:.2f}): open matched "
                  f"{rep['open']['matched']:.2f} R {rep['open']['retention']:.3f} | wide matched "
                  f"{rep['open_wide']['matched']:.2f} R {rep['open_wide']['retention']:.3f} | closed matched "
                  f"{rep['closed']['matched']:.2f} R {rep['closed']['retention']:.3f}{extra_txt}", flush=True)
        first = reps[0]
        plants[label] = dict(value=value, replicates=reps,
                             **{k: first[k] for k in ("plant", "history", "open", "open_wide", "closed",
                                                      "closed_knots", "trajectories")})
    n_main = len(thetas) - 16 * 3
    return dict(target=tg.tau, budget=tg.budget, open_knots=x_open.tolist(), wide_knots=x_wide.tolist(),
                t_traj=t[::6].tolist(), prior_sets=len(prior), mu=D.MU_TV,
                prior_weight={"main ensemble": n_main / len(prior), "each switch scenario": 16 / len(prior)},
                noise=dict(angle_deg=CL.SIG_ANGLE, density=CL.SIG_R, likelihood="same as simulated noise"),
                plants=plants)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/conditioning")
    ap.add_argument("--recalibrate", action="store_true")
    ap.add_argument("--quick", action="store_true", help="fewer targets and surfaces")
    ap.add_argument("--parts", nargs="+", default=["experiment", "map", "device", "profile", "loop"])
    ap.add_argument("--reuse-optima", action="store_true",
                    help="map and device: reuse the stored optima (redo the simplification only)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cal = out / "calibration.json"
    if a.recalibrate or not cal.exists():
        theta, c2, pb = C.fit(n_starts=16, max_nfev=2500)
        samples, temp, acc, diag = C.ensemble(theta, pb)
        C.save(cal, theta, c2, samples, pb, temp, C.held_out(samples, theta), acceptance=acc, chains=diag)
        print(f"calibration: chi2 {c2:.1f}, temperature {temp:.2f}; chains: mean chi2 "
              f"{[round(x, 1) for x in diag['mean_chi2']]}, switch medians {[round(x, 2) for x in diag['chain_medians']['tau_x']]}, "
              f"chosen {diag['chosen']} (acceptance {acc:.2f})", flush=True)
    theta, samples, d = C.load(cal)
    if "leave_out" not in d:
        d["leave_out"] = C.leave_out(C.LEAVE_OUT, theta)
        json.dump(d, open(cal, "w"), indent=1)
        print("leave-out refits done", flush=True)

    def stored(sc):
        return [dict(value=x["value"], chi2=x["chi2"], temperature=x["temperature"], acceptance=x["acceptance"],
                     samples=x["samples"].tolist()) for x in sc]
    if "scenarios" not in d:
        d["scenarios"] = stored(C.scenario_ensemble(theta, samples))
        json.dump(d, open(cal, "w"), indent=1)
    if "scenarios_extra" not in d:
        # the controller's belief also carries a switch outside the range the designs cover
        d["scenarios_extra"] = stored(C.scenario_ensemble(theta, samples, values=(5.5,)))
        json.dump(d, open(cal, "w"), indent=1)
    for x in d["scenarios"] + d["scenarios_extra"]:
        sm = np.array(x["samples"])
        print(f"scenario {x['value']:g} Pa: {len(np.unique(sm.round(12), axis=0))} distinct of {len(sm)}, "
              f"acceptance {x.get('acceptance', float('nan')):.2f}", flush=True)

    thetas = design_thetas(d, samples)
    t0 = time.time()
    if "experiment" in a.parts:
        json.dump(experiment(thetas), open(out / "experiment.json", "w"))
        print(f"experiment done ({time.time() - t0:.0f} s)", flush=True)
    if "map" in a.parts:
        targets = [1.0, 2.0, 3.0, 5.0, 8.0] if a.quick else [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
        surfaces = ["silicone_flat"] if a.quick else ["silicone_flat", "silicone_bf", "silicone_perp"]
        previous = json.load(open(out / "conditioning_map.json")) if a.reuse_optima else None
        json.dump(conditioning_map(thetas, targets, surfaces, previous=previous),
                  open(out / "conditioning_map.json", "w"))
        print(f"map done ({time.time() - t0:.0f} s)", flush=True)
    if "device" in a.parts:
        previous = json.load(open(out / "device.json")) if a.reuse_optima else None
        json.dump(device_case(thetas, previous=previous), open(out / "device.json", "w"))
        print(f"device done ({time.time() - t0:.0f} s)", flush=True)
    if "profile" in a.parts:
        json.dump(C.profile(theta), open(out / "profile_tau_x.json", "w"), indent=1)
        print(f"profile done ({time.time() - t0:.0f} s)", flush=True)
    if "loop" in a.parts:
        extra = np.vstack([scenario_design(x) for x in d["scenarios_extra"]])
        scen = [(f"switch at {x['value']:g} Pa", scenario_held(x), x["value"]) for x in d["scenarios"]
                if x["value"] in (2.2, 3.0)]
        scen += [(f"switch at {x['value']:g} Pa (outside the design range)", scenario_held(x), x["value"])
                 for x in d["scenarios_extra"]]
        json.dump(closed_loop_case(thetas, extra, scen), open(out / "closed_loop.json", "w"))
        print(f"closed loop done ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()

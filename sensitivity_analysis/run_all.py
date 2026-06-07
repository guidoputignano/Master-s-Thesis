"""
Run the full sensitivity and robustness analysis and print a summary table.

  Part A: one-at-a-time sensitivity of the temporal-dynamics scale  (sensitivity_oat)
  Part B: global Sobol sensitivity of the population scale          (sensitivity_sobol)
  Part C: sensitivity of the morphological energy weights          (sensitivity_energy_weights)
  Part D: sensitivity of the MPC cost weights                      (sensitivity_mpc_weights)

All figures are written to sensitivity_analysis/figures/ as 600-dpi PDFs.
Usage:
    python -m sensitivity_analysis.run_all
"""

import os

import numpy as np

from sensitivity_analysis import (sensitivity_oat, sensitivity_sobol,
                                  sensitivity_energy_weights, sensitivity_mpc_weights)

np.random.seed(42)


def _hr(width=72):
    print("-" * width)


def main(make_snapshots=True, n_base=sensitivity_sobol.N_BASE):
    print("=" * 72)
    print("Endothelial mechanoadaptation — sensitivity & robustness analysis")
    print("=" * 72)

    # ---- Part A ------------------------------------------------------------
    print("\n[Part A] One-at-a-time sensitivity (temporal dynamics) ...")
    oat = sensitivity_oat.run(make_snapshots=make_snapshots)

    print("\nTemporal dynamics — nominal outputs at t = 6 h:")
    print(f"    rho(6h)     = {oat['nominal_outputs']['rho_6h']:.4f}")
    print(f"    phi(6h)     = {oat['nominal_outputs']['phi_6h']:.2f} deg")

    print("\nNormalised sensitivity indices (+10% perturbation):")
    print(f"    {'parameter':<12}{'S[rho(6h)]':>14}{'S[phi(6h)]':>14}")
    _hr(40)
    for p, d in oat["sensitivity"].items():
        print(f"    {p:<12}{d['rho']:>14.4f}{d['phi']:>14.4f}")

    # ---- Part B ------------------------------------------------------------
    print("\n[Part B] Global Sobol sensitivity (population dynamics) ...")
    sob = sensitivity_sobol.run(n_base=n_base)

    print(f"\nReduced population ODE integrated at tau = {sensitivity_sobol.TAU} Pa "
          f"over 6 h")
    print(f"Sobol backend: {sob['backend']}   model evaluations: {sob['n_eval']}")
    print("\nFirst-order Sobol indices  S1  (output = phi_sen at t = 6 h):")
    print(f"    {'parameter':<12}{'S1':>12}")
    _hr(26)
    for name, val in sorted(sob["S1"].items(), key=lambda kv: -abs(kv[1])):
        print(f"    {name:<12}{val:>12.4f}")

    print("\nRobustness of the senescence constraint (phi_sen <= 0.30):")
    print(f"    mean phi_sen(6h)              = {sob['phisen_mean']:.4f}")
    print(f"    max  phi_sen(6h)              = {sob['phisen_max']:.4f}")
    print(f"    samples satisfying constraint = {sob['frac_satisfy']*100:.1f}%")
    print(f"    two most influential params  = {sob['top2']}")

    # ---- Part C ------------------------------------------------------------
    print("\n[Part C] Morphological energy-weight sensitivity (spatial scale) ...")
    ew = sensitivity_energy_weights.run()
    rho_vals = [r["rho_bar"] for r in ew["results"].values()]
    phi_vals = [r["phi_bar"] for r in ew["results"].values()]
    print("\nConverged morphology across weight sets (tau = 1.4 Pa, 6 h):")
    print(f"    {'weight set':<20}{'rho_bar':>10}{'phi_bar':>10}{'gap':>8}")
    _hr(48)
    for name, r in ew["results"].items():
        print(f"    {name:<20}{r['rho_bar']:>10.3f}{r['phi_bar']:>10.2f}{r['gap_frac']:>8.3f}")
    print(f"\n    spread: rho_bar range = {max(rho_vals)-min(rho_vals):.3f}, "
          f"phi_bar range = {max(phi_vals)-min(phi_vals):.2f} deg")
    print("\nNormalised sensitivity indices (+10% on each weight):")
    print(f"    {'weight':<14}{'S[rho_bar]':>12}{'S[phi_bar]':>12}")
    _hr(40)
    for k, d in ew["sensitivity"].items():
        print(f"    {k:<14}{d['rho_bar']:>12.4f}{d['phi_bar']:>12.4f}")

    # ---- Part D ------------------------------------------------------------
    print("\n[Part D] MPC cost-weight sensitivity (control law) ...")
    mw = sensitivity_mpc_weights.run()
    al = [r["align_final"] for r in mw["results"].values()]
    tm = [r["tau_mean"] for r in mw["results"].values()]
    print("\nClosed-loop outcomes across MPC cost-weight sets (24 h):")
    print(f"    {'weight set':<14}{'align':>8}{'rho':>7}{'phi_f':>8}{'tau_m':>8}")
    _hr(46)
    for name, r in mw["results"].items():
        print(f"    {name:<14}{r['align_final']:>8.1f}{r['rho_final']:>7.3f}"
              f"{r['phisen_final']:>8.3f}{r['tau_mean']:>8.3f}")
    print(f"\n    spread: align range = {max(al)-min(al):.1f} deg, "
          f"mean-tau range = {max(tm)-min(tm):.3f} Pa")
    print("\nNormalised sensitivity indices (+10% on each cost weight):")
    print(f"    {'weight':<10}{'S[align]':>10}{'S[phi_sen]':>12}{'S[tau]':>9}")
    _hr(42)
    for k, d in mw["sensitivity"].items():
        print(f"    {k:<10}{d['align_final']:>10.3f}{d['phisen_final']:>12.3f}{d['tau_mean']:>9.3f}")

    # ---- figure inventory --------------------------------------------------
    fig_dir = sensitivity_oat.FIG_DIR
    print("\nFigures written to:", fig_dir)
    for f in sorted(os.listdir(fig_dir)):
        if f.endswith(".pdf"):
            print("    -", f)
    print("=" * 72)
    return {"oat": oat, "sobol": sob, "energy_weights": ew, "mpc_weights": mw}


if __name__ == "__main__":
    main()

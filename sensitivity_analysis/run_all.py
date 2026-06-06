"""
Run the full sensitivity and robustness analysis and print a summary table.

  Part A: one-at-a-time sensitivity of the temporal-dynamics scale  (sensitivity_oat)
  Part B: global Sobol sensitivity of the population scale          (sensitivity_sobol)

All figures are written to sensitivity_analysis/figures/ as 600-dpi PDFs.
Usage:
    python -m sensitivity_analysis.run_all
"""

import os

import numpy as np

from sensitivity_analysis import sensitivity_oat, sensitivity_sobol

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

    # ---- figure inventory --------------------------------------------------
    fig_dir = sensitivity_oat.FIG_DIR
    print("\nFigures written to:", fig_dir)
    for f in sorted(os.listdir(fig_dir)):
        if f.endswith(".pdf"):
            print("    -", f)
    print("=" * 72)
    return {"oat": oat, "sobol": sob}


if __name__ == "__main__":
    main()

"""
Part B - Global (Sobol) sensitivity of the reduced population-dynamics model.

Integrates the reduced population ODE (eq:reduced, main.tex) at the protective
shear tau = tau_opt = 1.4 Pa over the six-hour horizon and records the senescent
fraction phi_sen(6h) = (S_tel + S_str) / N_tot (eq:senfraction). The right-hand
side is provided by the corrected
``endothelial_simulation.models.PopulationDynamicsModel.reduced_rhs``, which
implements

    dE_i/dt   = 2 r g(N_E) E_{i-1} - r g(N_E) E_i - gamma_tau(tau)(1+xi i) E_i
    dS_tel/dt = r g(N_E) E_N
    dS_str/dt = sum_i gamma_tau(tau)(1+xi i) E_i

with g(N_E) = 1/(1 + N_E/K)               (eq:density)
and gamma_tau(tau) = gamma_min + alpha_gamma (tau - tau_opt)^2  (eq:gamma_quad).

Uncertain parameters and ranges (Table 1, main.tex):
    gamma_min   : +/-30% around 0.00278 h^-1
    alpha_gamma : +/-30% around 0.00497 Pa^-2 h^-1
    r           : uniform [0.02, 0.03] h^-1
    K           : uniform [5e4, 6e4] cells/cm^2
    xi          : uniform [0.025, 0.075] per stage

Sampling: N = 2000 base samples via the Saltelli design (the sampling required
to estimate Sobol indices with SALib). First-order Sobol indices are computed
with SALib; if SALib is unavailable a manual Saltelli (Sobol-Jansen) estimator
is used as a fallback.

Outputs (figures/):
    sobol_first_order_bars.pdf   first-order Sobol indices
    sobol_phisen_hist.pdf        histogram of phi_sen(6h) with 30% constraint
    sobol_scatter_<p1>.pdf       phi_sen vs most influential parameter
    sobol_scatter_<p2>.pdf       phi_sen vs second most influential parameter
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from endothelial_simulation.config import SimulationConfig
from endothelial_simulation.models import PopulationDynamicsModel

# ----------------------------------------------------------------------------
# Reproducibility and journal style
# ----------------------------------------------------------------------------
np.random.seed(42)

CM = 1.0 / 2.54
SINGLE = 8.5 * CM
DOUBLE = 17.5 * CM
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

TAU = 1.4               # Source: Table 1, main.tex — tau_opt = 1.4 Pa (integration shear)
T_HORIZON = 6.0         # six-hour horizon
N_BASE = 2000           # base sample size
PHI_SEN_MAX = 0.30      # Source: Table 1, main.tex — phi_sen^max = 30% (MPC constraint)
N0 = 2.0e4              # initial healthy density (cells/cm^2), all at division stage 0


def set_style():
    plt.style.use("default")  # reset any global style (e.g. seaborn set by Plotter)
    plt.rcParams.update({
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.grid": False,
        "axes.edgecolor": "black",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.3,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "font.family": "sans-serif",
    })


# ----------------------------------------------------------------------------
# Problem definition (Table 1 ranges)
# ----------------------------------------------------------------------------
PROBLEM = {
    "num_vars": 5,
    "names": ["gamma_min", "alpha_gamma", "r", "K", "xi"],
    "bounds": [
        [0.00278 * 0.70, 0.00278 * 1.30],  # gamma_min +/-30%   (Table 1)
        [0.00497 * 0.70, 0.00497 * 1.30],  # alpha_gamma +/-30% (Table 1)
        [0.02, 0.03],                       # r          (Table 1)
        [5.0e4, 6.0e4],                     # K          (Table 1)
        [0.025, 0.075],                     # xi         (Table 1)
    ],
}

_PRETTY = {
    "gamma_min": r"$\gamma_{\min}$",
    "alpha_gamma": r"$\alpha_{\gamma}$",
    "r": r"$r$",
    "K": r"$K$",
    "xi": r"$\xi$",
}

# A single reusable model instance; per-sample parameters are overwritten.
_CFG = SimulationConfig()
_CFG.max_divisions = 15  # Source: Table 1, main.tex — N (Hayflick limit) = 15-18 PD
_MODEL = PopulationDynamicsModel(_CFG)
_NSTAGE = _MODEL.max_divisions + 1


def evaluate(sample):
    """
    Integrate the reduced population ODE for one parameter sample and return
    phi_sen(6h).

    sample order: [gamma_min, alpha_gamma, r, K, xi]
    """
    gamma_min, alpha_gamma, r, K, xi = sample
    _MODEL.gamma_min = gamma_min
    _MODEL.alpha_gamma = alpha_gamma
    _MODEL.r = r
    _MODEL.K = K
    _MODEL.xi = xi

    y0 = np.zeros(_NSTAGE + 2)
    y0[0] = N0  # all healthy cells start at division stage 0

    sol = solve_ivp(lambda t, y: _MODEL.reduced_rhs(y, TAU),
                    (0.0, T_HORIZON), y0, method="RK45",
                    t_eval=[T_HORIZON], rtol=1e-6, atol=1e-8)
    y = sol.y[:, -1]
    E = y[:_NSTAGE]
    S_tel = y[_NSTAGE]
    S_str = y[_NSTAGE + 1]
    N_tot = E.sum() + S_tel + S_str
    return (S_tel + S_str) / N_tot if N_tot > 0 else 0.0


# ----------------------------------------------------------------------------
# Sampling + Sobol analysis (SALib, with manual Saltelli fallback)
# ----------------------------------------------------------------------------
def _sample_and_analyze(n_base):
    """Return (param_values, Y, S1, S1_conf, backend_str)."""
    try:
        try:
            from SALib.sample import sobol as sobol_sampler  # SALib >= 1.4
            param_values = sobol_sampler.sample(PROBLEM, n_base, calc_second_order=False)
        except Exception:
            from SALib.sample import saltelli  # older SALib
            param_values = saltelli.sample(PROBLEM, n_base, calc_second_order=False)
        from SALib.analyze import sobol as sobol_analyze
        Y = np.array([evaluate(x) for x in param_values])
        Si = sobol_analyze.analyze(PROBLEM, Y, calc_second_order=False,
                                   print_to_console=False)
        return param_values, Y, np.asarray(Si["S1"]), np.asarray(Si["S1_conf"]), "SALib"
    except Exception as exc:
        print(f"  [info] SALib unavailable ({exc}); using manual Saltelli estimator.")
        return _manual_saltelli(n_base)


def _manual_saltelli(n_base):
    """
    Manual first-order Sobol estimator (Saltelli design, Sobol 2007 estimator).

    Builds A, B and the D matrices AB_i; estimates
        S1_i = (1/N) sum f(B)*(f(AB_i) - f(A)) / Var(f).
    Returns the same tuple layout as the SALib branch.
    """
    D = PROBLEM["num_vars"]
    bounds = np.array(PROBLEM["bounds"])

    def scale(u):
        return bounds[:, 0] + u * (bounds[:, 1] - bounds[:, 0])

    U = np.random.rand(n_base, 2 * D)
    A = scale(U[:, :D])
    B = scale(U[:, D:])

    fA = np.array([evaluate(x) for x in A])
    fB = np.array([evaluate(x) for x in B])
    varY = np.var(np.concatenate([fA, fB]), ddof=1)

    S1 = np.zeros(D)
    for i in range(D):
        AB = A.copy()
        AB[:, i] = B[:, i]
        fAB = np.array([evaluate(x) for x in AB])
        S1[i] = np.mean(fB * (fAB - fA)) / varY

    # assemble a combined design matrix and Y for downstream plotting
    param_values = np.vstack([A, B])
    Y = np.concatenate([fA, fB])
    return param_values, Y, S1, np.zeros(D), "manual-Saltelli"


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def _bar_figure(S1, S1_conf, fname):
    names = PROBLEM["names"]
    labels = [_PRETTY[n] for n in names]
    order = np.argsort(S1)
    y = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(SINGLE, SINGLE * 0.85))
    xerr = S1_conf[order] if np.any(S1_conf) else None
    ax.barh(y, S1[order], xerr=xerr, color="#4C72B0",
            error_kw=dict(ecolor="0.3", lw=0.8, capsize=2))
    ax.set_yticks(y)
    ax.set_yticklabels([labels[i] for i in order])
    ax.set_xlabel("First-order Sobol index $S_1$")
    ax.axvline(0.0, color="black", lw=0.8)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, fname)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return path


def _hist_figure(Y, fname):
    frac_ok = float(np.mean(Y <= PHI_SEN_MAX))
    fig, ax = plt.subplots(figsize=(SINGLE, SINGLE * 0.8))
    ax.hist(Y, bins=40, color="#55A868", edgecolor="white", linewidth=0.3)
    ax.axvline(PHI_SEN_MAX, color="#C44E52", ls="--", lw=1.4)
    ax.set_xlabel(r"Senescent fraction $\phi_{\mathrm{sen}}(6\,\mathrm{h})$")
    ax.set_ylabel("Count")
    ax.annotate(f"{frac_ok*100:.1f}% satisfy\n"
                r"$\phi_{\mathrm{sen}}\leq 0.30$",
                xy=(0.97, 0.95), xycoords="axes fraction",
                ha="right", va="top",
                bbox=dict(boxstyle="round", fc="white", ec="0.7", lw=0.6))
    fig.tight_layout()
    path = os.path.join(FIG_DIR, fname)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return path, frac_ok


def _scatter_figure(param_values, Y, name, fname):
    j = PROBLEM["names"].index(name)
    fig, ax = plt.subplots(figsize=(SINGLE, SINGLE * 0.8))
    ax.scatter(param_values[:, j], Y, s=3, alpha=0.25, color="#4C72B0",
               edgecolors="none", rasterized=True)
    ax.axhline(PHI_SEN_MAX, color="#C44E52", ls="--", lw=1.2)
    ax.set_xlabel(_PRETTY[name])
    ax.set_ylabel(r"$\phi_{\mathrm{sen}}(6\,\mathrm{h})$")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, fname)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def run(n_base=N_BASE):
    set_style()
    os.makedirs(FIG_DIR, exist_ok=True)

    param_values, Y, S1, S1_conf, backend = _sample_and_analyze(n_base)

    paths = {}
    paths["bars"] = _bar_figure(S1, S1_conf, "sobol_first_order_bars.pdf")
    paths["hist"], frac_ok = _hist_figure(Y, "sobol_phisen_hist.pdf")

    # two most influential parameters (by |S1|)
    order = np.argsort(np.abs(S1))[::-1]
    top2 = [PROBLEM["names"][order[0]], PROBLEM["names"][order[1]]]
    paths["scatter_1"] = _scatter_figure(param_values, Y, top2[0],
                                         f"sobol_scatter_{top2[0]}.pdf")
    paths["scatter_2"] = _scatter_figure(param_values, Y, top2[1],
                                         f"sobol_scatter_{top2[1]}.pdf")

    summary = {
        "backend": backend,
        "n_eval": int(len(Y)),
        "S1": {PROBLEM["names"][i]: float(S1[i]) for i in range(len(S1))},
        "phisen_mean": float(np.mean(Y)),
        "phisen_max": float(np.max(Y)),
        "frac_satisfy": frac_ok,
        "top2": top2,
        "paths": paths,
    }
    return summary


if __name__ == "__main__":
    s = run()
    print("Global (Sobol) population sensitivity complete.")
    print(f"Backend: {s['backend']}, model evaluations: {s['n_eval']}")
    print("First-order Sobol indices:")
    for name, val in sorted(s["S1"].items(), key=lambda kv: -abs(kv[1])):
        print(f"  {name:11s}  S1 = {val:+.4f}")
    print(f"phi_sen(6h): mean = {s['phisen_mean']:.4f}, max = {s['phisen_max']:.4f}")
    print(f"Fraction satisfying phi_sen <= 0.30: {s['frac_satisfy']*100:.1f}%")
    print(f"Two most influential: {s['top2']}")
    for k, v in s["paths"].items():
        print(f"  saved {k}: {v}")

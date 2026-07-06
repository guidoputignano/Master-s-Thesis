"""
Monolayer KPIs — final quantification step for the endothelial simulation.

This module summarises the state of the cell monolayer at a single timestep and,
via :func:`print_before_after` / :func:`save_kpi_figure`, compares the first and
last timestep of a run.

Per-cell state (see ``endothelial_simulation/core/cell.py``) is:
    - actual_orientation  : orientation angle in radians (axial, defined mod pi,
                            obtained from PCA of the territory -> ``arctan2``).
    - actual_aspect_ratio : aspect ratio (>= 1.0).
    - is_senescent        : senescence flag (bool).
    - actual_area         : cell area as the number of territory pixels.
    - centroid            : (x, y) centroid of the territory.

These are recorded per-timestep by ``Simulator._record_frame`` (frame['cells'])
and are also live on ``simulator.grid.cells``.

A note on ``area``/coverage
---------------------------
``actual_area`` is the cell's **Voronoi territory** (number of pixels the cell
owns in the space-filling tessellation), not a separately measured occupied
area. Because the tessellation tiles the whole domain (with holes disabled the
hole-distance threshold is +inf, so no pixel is left unassigned — see
``Grid._update_voronoi_tessellation``), ``sum(area) ~= domain_area`` and coverage
is always ~1. We therefore report **gap fraction from the modelled gaps** (the
domain pixels that belong to no cell, i.e. holes): ``gap = 1 - coverage``. Both
numbers are returned, and ``coverage_is_territory`` flags which definition was
used.
"""

import numpy as np


# Constraint on the senescent fraction (Source: Table 1, main.tex -> phi_sen^max).
PHI_SEN_MAX = 0.30


def _wrap_to_pm90(angle_deg):
    """Wrap an angle in degrees to the half-open interval (-90, 90]."""
    wrapped = (angle_deg + 90.0) % 180.0 - 90.0
    # ((x + 90) % 180) - 90 lands in [-90, 90); move the -90 endpoint to +90 so
    # the interval is (-90, 90] as specified.
    if np.isclose(wrapped, -90.0):
        wrapped = 90.0
    return wrapped


def compute_metrics(orientations, aspect_ratios, senescent_flags, areas,
                    domain_area, *, orientation_in_degrees=False,
                    phi_sen_max=PHI_SEN_MAX):
    """
    Compute monolayer KPIs for a single timestep.

    Parameters
    ----------
    orientations : array-like
        Per-cell orientation angles. Radians by default; pass
        ``orientation_in_degrees=True`` if they are in degrees. Orientation is
        axial (defined mod pi): circular statistics are used, never an
        arithmetic mean of angles.
    aspect_ratios : array-like
        Per-cell aspect ratios (>= 1).
    senescent_flags : array-like
        Per-cell senescence flags (truthy == senescent).
    areas : array-like
        Per-cell occupied area (here: Voronoi territory pixel counts).
    domain_area : float
        Total area of the simulation domain, in the same units as ``areas``.
    orientation_in_degrees : bool, optional
        Set True if ``orientations`` are already in degrees.
    phi_sen_max : float, optional
        Upper bound used for the senescent-fraction constraint check.

    Returns
    -------
    dict
        Keys:
            n_cells               : number of cells.
            alignment_S           : nematic order parameter S in [0, 1].
            deviation_from_flow    : |director| in degrees, in [0, 90]. Smaller
                                    means better aligned with the flow (x-axis).
            director_deg          : signed director angle in (-90, 90].
            aspect_ratio_mean     : mean aspect ratio.
            aspect_ratio_std      : std of aspect ratio.
            senescent_fraction    : mean(flag).
            senescent_ok          : bool, senescent_fraction <= phi_sen_max.
            phi_sen_max           : the constraint value used.
            coverage              : sum(area) / domain_area (~1 for a tiling).
            gap_fraction          : 1 - coverage (from the modelled gaps/holes).
            coverage_is_territory : True (area is the Voronoi territory).
    """
    theta = np.asarray(orientations, dtype=float).ravel()
    ar = np.asarray(aspect_ratios, dtype=float).ravel()
    flags = np.asarray(senescent_flags).ravel().astype(bool)
    area = np.asarray(areas, dtype=float).ravel()

    if orientation_in_degrees:
        theta = np.radians(theta)

    n_cells = int(theta.size)

    if n_cells == 0:
        return {
            'n_cells': 0,
            'alignment_S': float('nan'),
            'deviation_from_flow': float('nan'),
            'director_deg': float('nan'),
            'aspect_ratio_mean': float('nan'),
            'aspect_ratio_std': float('nan'),
            'senescent_fraction': float('nan'),
            'senescent_ok': True,
            'phi_sen_max': float(phi_sen_max),
            'coverage': float('nan'),
            'gap_fraction': float('nan'),
            'coverage_is_territory': True,
        }

    # --- Alignment: nematic order parameter via circular statistics ---
    # Axial angles are doubled so that theta and theta+pi are identified.
    z = np.mean(np.exp(2j * theta))
    S = float(np.abs(z))

    # --- Mean deviation from flow (flow is along x, i.e. theta = 0) ---
    director_deg = np.degrees(0.5 * np.angle(z))      # 0.5*angle(z) is in (-90, 90]
    director_deg = _wrap_to_pm90(director_deg)
    deviation_from_flow = float(abs(director_deg))

    # --- Aspect ratio ---
    ar_mean = float(np.mean(ar))
    ar_std = float(np.std(ar))

    # --- Senescent fraction + constraint check ---
    senescent_fraction = float(np.mean(flags))
    senescent_ok = bool(senescent_fraction <= phi_sen_max)

    # --- Coverage / gap fraction ---
    coverage = float(np.sum(area) / domain_area) if domain_area else float('nan')
    gap_fraction = float(max(0.0, 1.0 - coverage)) if domain_area else float('nan')

    return {
        'n_cells': n_cells,
        'alignment_S': S,
        'deviation_from_flow': deviation_from_flow,
        'director_deg': float(director_deg),
        'aspect_ratio_mean': ar_mean,
        'aspect_ratio_std': ar_std,
        'senescent_fraction': senescent_fraction,
        'senescent_ok': senescent_ok,
        'phi_sen_max': float(phi_sen_max),
        'coverage': coverage,
        'gap_fraction': gap_fraction,
        'coverage_is_territory': True,
    }


# ---------------------------------------------------------------------------
# Table specification shared by the text and figure renderers.
# Each row: (metric label, dict key, formatter).
# ---------------------------------------------------------------------------
def _fmt2(v):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.2f}"


def _fmt1(v):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.1f}"


def _fmt_ok(v):
    return "OK" if v else "OVER"


_ROWS = [
    ("Nematic order S",         'alignment_S',         _fmt2),
    ("Deviation from flow (deg)", 'deviation_from_flow', _fmt1),
    ("Aspect ratio (mean)",     'aspect_ratio_mean',   _fmt2),
    ("Aspect ratio (std)",      'aspect_ratio_std',    _fmt2),
    ("Senescent fraction",      'senescent_fraction',  _fmt2),
    ("Senescent <= 0.30",       'senescent_ok',        _fmt_ok),
    ("Gap fraction",            'gap_fraction',        _fmt2),
]


def print_before_after(m0, m1):
    """
    Print a plain-text table (metric | initial | final) to stdout.

    Numbers are shown to 2 decimals, angles to 1 decimal.
    """
    rows = [(label, fmt(m0.get(key)), fmt(m1.get(key))) for label, key, fmt in _ROWS]

    metric_w = max(len("Metric"), *(len(r[0]) for r in rows))
    init_w = max(len("Initial"), *(len(r[1]) for r in rows))
    final_w = max(len("Final"), *(len(r[2]) for r in rows))

    header = f"{'Metric':<{metric_w}}  {'Initial':>{init_w}}  {'Final':>{final_w}}"
    sep = "-" * len(header)

    print(sep)
    print("Monolayer KPIs: first vs last timestep")
    print(sep)
    print(header)
    print(sep)
    for label, init, final in rows:
        print(f"{label:<{metric_w}}  {init:>{init_w}}  {final:>{final_w}}")
    print(sep)


def save_kpi_figure(m0, m1, path):
    """
    Save a slide-ready PNG table comparing the initial and final timestep.

    The table has a dark header row and three columns: Metric, Initial, Final.
    Returns the path written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [(label, fmt(m0.get(key)), fmt(m1.get(key))) for label, key, fmt in _ROWS]

    n = len(rows)
    fig_h = 0.6 + 0.5 * (n + 1)
    fig, ax = plt.subplots(figsize=(7.5, fig_h))
    ax.axis("off")

    header_color = "#1f2a44"   # dark header
    header_text = "#ffffff"
    row_colors = ["#f4f6fa", "#e8ecf3"]
    edge_color = "#c9d1e0"

    col_labels = ["Metric", "Initial", "Final"]
    cell_text = [[label, init, final] for label, init, final in rows]

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.5, 0.25, 0.25],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.0, 1.6)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(edge_color)
        cell.set_linewidth(0.8)
        if r == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color=header_text, fontweight="bold")
        else:
            cell.set_facecolor(row_colors[(r - 1) % 2])
            # Left-align the metric-name column for readability.
            if c == 0:
                cell.set_text_props(ha="left")
                cell.PAD = 0.04

    ax.set_title("Monolayer KPIs — first vs last timestep",
                 fontsize=14, fontweight="bold", pad=12)

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Convenience extractors for wiring the KPIs into a live simulation.
# ---------------------------------------------------------------------------
def metrics_from_cells(cells, domain_area, phi_sen_max=PHI_SEN_MAX):
    """
    Compute metrics from a collection of live ``Cell`` objects.

    ``cells`` may be a dict (grid.cells) or any iterable of Cell instances.
    """
    cell_iter = cells.values() if hasattr(cells, "values") else cells
    orientations, aspect_ratios, flags, areas = [], [], [], []
    for cell in cell_iter:
        orientations.append(cell.actual_orientation)
        aspect_ratios.append(cell.actual_aspect_ratio)
        flags.append(cell.is_senescent)
        areas.append(cell.actual_area)
    return compute_metrics(orientations, aspect_ratios, flags, areas,
                           domain_area, phi_sen_max=phi_sen_max)


def metrics_from_frame(frame, domain_area, phi_sen_max=PHI_SEN_MAX):
    """
    Compute metrics from one entry of ``Simulator.frame_data``.

    Each frame stores per-cell 'orientation' (radians), 'aspect_ratio', 'area'
    and 'is_senescent'.
    """
    cells = frame['cells']
    orientations = [c['orientation'] for c in cells]
    aspect_ratios = [c['aspect_ratio'] for c in cells]
    flags = [c['is_senescent'] for c in cells]
    areas = [c['area'] for c in cells]
    return compute_metrics(orientations, aspect_ratios, flags, areas,
                           domain_area, phi_sen_max=phi_sen_max)


def summarize_simulation(simulator, figure_path=None):
    """
    Final quantification step: summarise the monolayer at the first and last
    recorded timestep, print the table and (optionally) save the figure.

    Uses recorded frames when available (they carry per-cell senescence flags
    and raw orientation); falls back to the live grid for the final state.

    Returns ``(m0, m1)``.
    """
    grid = simulator.grid
    domain_area = grid.comp_width * grid.comp_height
    phi_sen_max = getattr(simulator.config, 'phi_sen_max', PHI_SEN_MAX)

    frames = getattr(simulator, 'frame_data', None)
    if frames and len(frames) >= 2:
        m0 = metrics_from_frame(frames[0], domain_area, phi_sen_max)
        m1 = metrics_from_frame(frames[-1], domain_area, phi_sen_max)
    elif frames and len(frames) == 1:
        m0 = metrics_from_frame(frames[0], domain_area, phi_sen_max)
        m1 = metrics_from_cells(grid.cells, domain_area, phi_sen_max)
    else:
        # No frames recorded: only the final (live) state is available.
        m1 = metrics_from_cells(grid.cells, domain_area, phi_sen_max)
        m0 = m1

    print_before_after(m0, m1)

    if figure_path is None:
        import os
        save_dir = getattr(simulator.config, 'plot_directory', '.')
        os.makedirs(save_dir, exist_ok=True)
        figure_path = os.path.join(save_dir, "monolayer_kpis.png")

    save_kpi_figure(m0, m1, figure_path)
    print(f"KPI figure saved to: {figure_path}")
    return m0, m1


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _self_test():
    """
    Build synthetic monolayers (broad angle spread + round cells at t0;
    flow-aligned + elongated cells at the final step), run both renderers and
    confirm the PNG was written.
    """
    import os

    rng = np.random.default_rng(0)
    n = 300
    domain_area = 256 * 256  # matches comp_width * comp_height for a 1024 grid

    # t0: broad orientation spread (near-uniform over the axial range) and
    # round cells (aspect ratio ~1). A modest senescent fraction below 0.30.
    theta0 = rng.uniform(-np.pi / 2, np.pi / 2, size=n)
    ar0 = 1.0 + np.abs(rng.normal(0.0, 0.05, size=n))
    sen0 = rng.random(n) < 0.20
    area0 = np.full(n, domain_area / n)  # perfect tiling -> coverage ~1

    # final: flow-aligned (theta ~ 0, small spread) and elongated cells.
    thetaf = rng.normal(0.0, np.radians(8.0), size=n)
    arf = rng.normal(3.2, 0.4, size=n)
    senf = rng.random(n) < 0.20
    # Introduce a small modelled gap so gap fraction is visibly non-zero.
    areaf = np.full(n, 0.97 * domain_area / n)

    m0 = compute_metrics(theta0, ar0, sen0, area0, domain_area)
    m1 = compute_metrics(thetaf, arf, senf, areaf, domain_area)

    print_before_after(m0, m1)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "monolayer_kpis_selftest.png")
    save_kpi_figure(m0, m1, out_path)

    if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
        print(f"\n[self-test] PNG written OK: {out_path} "
              f"({os.path.getsize(out_path)} bytes)")
    else:
        raise SystemExit(f"[self-test] FAILED: PNG not written at {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monolayer KPI quantification.")
    parser.add_argument("--self-test", action="store_true",
                        help="Build synthetic arrays, print the table and save the figure.")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
    else:
        parser.print_help()

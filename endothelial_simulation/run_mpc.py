"""
Single entry point for the endothelial mechanoadaptation MPC study (main.tex).

Pipeline:
  1. Initialise the simulator with the corrected confluent cell count and the
     Table-1 parameters (areas in physical um^2, gap-free Voronoi tessellation).
  2. Set the initial senescent composition phi_sen(0) = 0.30 (the A1 design),
     all stress-induced (handled inside Simulator.initialize()).
  3. Run the receding-horizon MPC (run_mpc_simulation), by default for 12
     one-hour steps: twice the 6 h A1 flow, long enough to show the approach to
     the plateau (95 % of the change after 9 h with the 3 h constant).
  4. Save the tessellation frames, the assembled animation, the summary figures
     and the log (log.json) under endothelial_simulation/figures/ or --out.

Usage:
    python -m endothelial_simulation.run_mpc [--steps 12] [--out DIR]
"""
import argparse
import json
import os
import matplotlib
matplotlib.use('Agg')  # headless rendering

from endothelial_simulation.config import SimulationConfig
from endothelial_simulation.core.simulator import Simulator
from endothelial_simulation.control.mpc_controller import run_mpc_simulation


def build_config():
    """Full simulation, holes disabled, Table-1 parameters (see config.py)."""
    config = SimulationConfig().set_full_simulation()
    config.enable_holes = False          # Source: spec — holes off for the MPC run
    config.create_animations = False     # MPC frames are rendered by run_mpc_simulation
    # Initial senescent composition: the A1 design (70 % control, 30 % TNF-alpha),
    # entirely stress-induced (telomere senescence removed, main.tex Sec 2.3).
    config.initial_senescent_fraction = 0.30
    config.senescent_stress_fraction = 1.0
    config.senescent_telomere_fraction = 0.0
    return config


def main(n_control_steps=12, output_dir=None):
    # Each step is a 1 h receding-horizon decision; with the single morphological
    # adaptation constant tau_adapt = tau_orient = 3 h, 12 h reach the plateau.
    config = build_config()

    print("=" * 70)
    print("Endothelial mechanoadaptation — receding-horizon MPC")
    print(f"  confluent cell count : {config.initial_cell_count}")
    print(f"  phi_sen(0)           : {config.initial_senescent_fraction} "
          f"(all stress-induced)")
    print(f"  pixel scale          : {config.pixel_scale_um:.4f} um/px")
    print("=" * 70)

    # 1-2. Initialise simulator (sets cell count, areas, phi_sen(0))
    simulator = Simulator(config)
    simulator.set_constant_input(0.0)    # static baseline initial condition (0 Pa)
    simulator.initialize()

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'figures')

    # 3-4. Run MPC and write frames / animation / summary plots
    results = run_mpc_simulation(
        simulator, config,
        n_control_steps=n_control_steps,
        output_dir=output_dir,
    )

    with open(os.path.join(output_dir, 'log.json'), 'w') as fh:
        json.dump({'log': results['log'], 'admitted': results['admitted'], 'seed': results['seed'],
                   'phi_sen0': config.initial_senescent_fraction}, fh, indent=1)

    print("\n✅ MPC run complete.")
    print(f"   frames    : {len(results['frames'])} PDFs in {output_dir}/frames")
    print(f"   animation : {results['animation']}")
    print(f"   dashboard : {results['dashboard']}")
    print(f"   summaries : mpc_tau_trajectory.pdf, mpc_aspect_ratio.pdf, mpc_alignment.pdf, "
          f"mpc_phi_sen.pdf, log.json")
    return results


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--steps', type=int, default=12, help='one-hour control steps')
    ap.add_argument('--out', help='output folder (default endothelial_simulation/figures)')
    a = ap.parse_args()
    main(a.steps, a.out)

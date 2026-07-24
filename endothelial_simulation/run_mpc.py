"""
Single entry point for the endothelial mechanoadaptation MPC study (main.tex).

Pipeline:
  1. Initialise the simulator with the corrected confluent cell count and the
     Table-1 parameters (areas in physical um^2, gap-free Voronoi tessellation).
  2. Set the initial senescent composition phi_sen(0) = 0.20 with a 70/30
     stress/telomere split (handled inside Simulator.initialize()).
  3. Run the receding-horizon MPC (run_mpc_simulation) for 6 control steps.
  4. Save the 24 tessellation frames, the assembled animation, and the three
     summary figures under endothelial_simulation/figures/.

Usage:
    python -m endothelial_simulation.run_mpc
"""
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
    # Initial senescent composition: telomere senescence removed (main.tex Sec 2.3),
    # so the initial pool is entirely stress-induced.
    config.initial_senescent_fraction = 0.20
    config.senescent_stress_fraction = 1.0
    config.senescent_telomere_fraction = 0.0
    return config


def main(n_control_steps=24):
    # ~24 h conditioning so the morphology converges to the flow-adapted plateau
    # (single morphological adaptation constant tau_adapt = tau_orient = 7.4 h).
    # Each step is a 1 h receding-horizon decision.
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

    output_dir = os.path.join(os.path.dirname(__file__), 'figures')

    # 3-4. Run MPC and write frames / animation / summary plots
    results = run_mpc_simulation(
        simulator, config,
        n_control_steps=n_control_steps,
        output_dir=output_dir,
    )

    print("\n✅ MPC run complete.")
    print(f"   frames    : {len(results['frames'])} PDFs in {output_dir}/frames")
    print(f"   animation : {results['animation']}")
    print(f"   dashboard : {results['dashboard']}")
    print(f"   summaries : mpc_tau_trajectory.pdf, mpc_phi_sen.pdf, mpc_morphology.pdf")
    return results


if __name__ == '__main__':
    main()

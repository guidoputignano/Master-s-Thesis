# Endothelial Cell Simulation Framework

This repository contains a sophisticated, agent-based computational model for simulating the behavior of endothelial cell monolayers. The framework is designed to study cellular mechanotransduction, population dynamics, and spatial organization in response to various stimuli. It incorporates advanced features like event-driven simulation, model predictive control, and comprehensive analysis tools.

## Key Features

- **Agent-Based Modeling**: Each endothelial cell is an individual agent with its own state and properties.
- **Event-Driven Architecture**: The simulation progresses based on discrete events, allowing for efficient and flexible modeling of complex biological processes.
- **Advanced Control Systems**: Includes a receding-horizon controller (`control/mpc_controller.py`). It advances its own model rather than a measured state, so it is an open-loop protocol optimiser; it is kept for the thesis record. The paper uses the `conditioning/` package at the repository root instead.
- **Rich Biophysical Models**: Incorporates detailed models for population dynamics (proliferation, senescence), spatial properties (cell shape, orientation), and temporal dynamics.
- **Simulation Management**: Features robust configuration management and modules for specialized simulation scenarios like optimal stopping problems.
- **Comprehensive Visualization**: Extensive tools for generating plots, animations, and composite videos to analyze simulation results, including cell distributions and energy metrics.

## Project Structure

The project is organized into the following modules:

```
endothelial_simulation/
├── main.py                    # Main script to run simulations
├── config.py                  # Global configuration settings
├── control/                   # Control systems for the simulation
│   └── mpc_controller.py      # Model Predictive Control (MPC)
├── core/                      # Core simulation engine and components
│   ├── cell.py                # Defines the state and behavior of a single cell
│   ├── event_system.py        # Manages the scheduling and execution of events
│   ├── grid.py                # Manages the spatial grid and cell neighbors
│   ├── holes.py               # Logic for detecting and managing gaps in the monolayer
│   └── simulator.py           # The main simulation loop
├── management/                # High-level simulation management and logic
│   ├── configuration_manager.py # Manages loading and saving of simulation configs
│   ├── event_driven_simulator.py # Implements the event-driven simulation logic
│   ├── optimal_stopping.py    # Framework for optimal stopping problems
│   └── transition_controller.py # Manages state transitions
├── models/                    # Mathematical models for cell behavior
│   ├── parameters.py          # Defines model parameters
│   ├── population_dynamics.py # Models cell proliferation, senescence, and death
│   ├── spatial_properties.py  # Models cell morphology and orientation
│   └── temporal_dynamics.py   # Models time-dependent cellular responses
└── visualization/             # Tools for data analysis and visualization
    ├── analysis.py            # General analysis scripts
    ├── animations.py          # Creates animations from simulation frames
    ├── cell_distributions.py  # Plots for cell state distributions
    ├── composite_video.py     # Tools to combine multiple plots into a single video
    ├── energy_analysis.py     # Analysis of system-level energy functions
    └── plotters.py            # Core plotting functions
```

## How to Run a Simulation

1.  **Configure**: Adjust parameters in `config.py` or create a new configuration file.
2.  **Execute**: Run the main simulation script from the root directory:
    ```bash
    python -m endothelial_simulation.main
    ```

## Dependencies

This project relies on several scientific computing and visualization libraries. Key dependencies include:

- `numpy`
- `scipy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `scikit-image`
- `opencv-python`
- `ffmpeg-python`
- `moviepy`
- `h5py`
- `pyyaml`
- `plotly`
- `numba`

Install all required packages using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

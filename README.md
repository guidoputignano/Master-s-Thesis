# Master's Thesis: A Computational Framework for Simulating Endothelial Cell Dynamics

This repository contains the complete body of work for the Master's Thesis by Guido Putignano. The project focuses on the design, implementation, and application of a multi-faceted computational framework to model, simulate, and analyze the complex behavior of endothelial cell monolayers. It integrates experimental data, advanced image processing, mathematical modeling, and a sophisticated agent-based simulation to investigate cellular responses to mechanical stress, senescence, and other stimuli.

---

## Target-specific conditioning (paper, 2026)

The package [`conditioning/`](./conditioning) holds the model and the protocol designer of the paper
*Target-Specific Shear Conditioning of Endothelial Monolayers for Blood-Contacting Devices*:

- `observations.py`: every published value used (Stefopoulos 2022, Robotti 2014, Wu 2021, same chamber),
  with its source, panel, how it was read and its role (train or held out);
- `model.py` and `_kernel.py`: the monolayer-state model (domains ordered along or across the flow,
  frustrated domains, junction destabilisation, damage, cell retention), in NumPy and compiled with numba;
- `calibrate.py`: least squares with priors, tempered ensemble, scenario ensembles with the switch shear
  held fixed, the profile of the switch shear split by data source, leave-one-out refits;
- `design.py`: protocol design for a target shear, robust over the ensemble (mean minus standard deviation
  of a readiness score, with a small penalty on changes of shear), reported as the simplest near-optimal
  path (the best two-level path within 0.01 of the optimum);
- `device.py`: one pump flow for a whole device (HVAD inflow cannula, eight regions);
- `closed_loop.py`: receding-horizon design with imaging feedback (orientation and density), with a belief
  that also covers a switch shear outside the range the designs assume;
- `run_all.py`: reproduces every result, written to `results/conditioning/`:

  ```
  python -m conditioning.run_all --parts experiment map device profile loop
  ```

  (parts can run in separate processes; the map takes about 20 min on four cores).

Tests: `python -m pytest tests/test_conditioning.py`. The earlier receding-horizon controller in
`endothelial_simulation/control/mpc_controller.py` (single relaxation toward shear-dependent targets,
senescent-cell ladder, tessellation) is kept for the thesis record and is not used by the paper.

##  Workflow Overview

The project follows a structured workflow, where each component builds upon the last:

1.  **Data Acquisition**: Utilizes experimental data from real-world microscopy.
2.  **Image Processing**: Raw image data is processed through a pipeline to extract quantitative metrics.
3.  **Mathematical Modeling**: Insights from the processed data and scientific literature are used to develop mathematical models of cell behavior.
4.  **Agent-Based Simulation**: The models are integrated into a powerful simulation framework to run complex *in silico* experiments.

## Key Components

This repository is organized into four primary components, each located in its own directory.

### 1. 🗂️ Data (`/data`)

This directory contains information regarding the experimental data used to inform and validate the models. The primary dataset was provided by **Costanza Giampietro**.

> For detailed information on the dataset, its source, associated publications, and access restrictions, please see the dedicated **[data/Costanza/README.md](./data/Costanza/README.md)**.

### 2. 🖼️ Imaging Pipeline (`/Imaging`)

A collection of Jupyter notebooks designed for the processing and analysis of raw microscopy images. This pipeline handles the crucial steps of converting 3D image stacks into 2D projections, segmenting cells, and performing quantitative analysis.

> The pipeline is optimized for Google Colab to leverage GPU acceleration. For a full explanation of the workflow, see the **[Imaging/README.md](./Imaging/README.md)**.

### 3. 📈 Mathematical Models (`/models`)

This directory houses a series of Python scripts that implement various mathematical models developed during the research. These include:
- Deterministic and stochastic models of cell population dynamics.
- Models for cellular senescence and the effects of senolytic treatments.
- Formulations for temporal dynamics in response to shear stress.

These models form the theoretical foundation for the main simulation framework.

### 4. 🔬 Endothelial Simulation Framework (`/endothelial_simulation`)

The core of this thesis project. It is a powerful, agent-based simulation environment built in Python. It is designed to be modular and extensible, allowing for the simulation of complex, emergent behaviors in endothelial cell monolayers.

Key features include an event-driven architecture, Model Predictive Control (MPC) capabilities, and extensive visualization tools.

> For a comprehensive guide to the framework's architecture, features, and usage instructions, please refer to the **[endothelial_simulation/README.md](./endothelial_simulation/README.md)**.

## 📂 Repository Structure

Here is a breakdown of the key directories and their purpose within the project:

| Directory | Description |
| :--- | :--- |
| **`/data`** | Contains the datasets used for the project, with documentation on origin and access. |
| **`/docs`** | Holds supplementary documentation and figures generated for the thesis manuscript. |
| **`/endothelial_simulation`** | The primary simulation framework, a self-contained Python module for running agent-based models. See the [detailed README](./endothelial_simulation/README.md). |
| **`/Imaging`** | The complete pipeline for processing and analyzing microscopy images. See the [detailed README](./Imaging/README.md). |
| **`/models`** | A collection of Python scripts implementing the core mathematical models of cell dynamics (population, senescence, temporal response). |
| **`/notebooks`** | Jupyter notebooks for exploratory data analysis, model prototyping, and generating specific plots. |
| `requirements.txt` | A list of all Python packages required to run the project. |

---

## Getting Started

### Prerequisites

- Python 3.8+
- `pip` for package management

### Installation

Clone the repository and install the required dependencies from the root directory:

```bash
git clone <repository-url>
cd Master-s-Thesis
pip install -r requirements.txt
```

### Running the Simulation

The main simulation can be executed by running the `main.py` script within the `endothelial_simulation` module. Ensure you have configured your desired parameters in `endothelial_simulation/config.py`.

```bash
python -m endothelial_simulation.main
```

For more detailed instructions, please consult the simulation's README file.

## Contact
For questions contact guido.putignano@bioergotech.org

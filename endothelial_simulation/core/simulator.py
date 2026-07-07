"""
Complete event-driven simulator implementation with all missing functionality restored.
This version includes all the original features while using the event-driven system.
"""
import numpy as np
import time
import os
import random
import warnings
from typing import Dict, List, Optional
import matplotlib.pyplot as plt

from endothelial_simulation.core.cell import Cell
from endothelial_simulation.core.grid import Grid
from endothelial_simulation.models.temporal_dynamics import TemporalDynamicsModel
from endothelial_simulation.models.population_dynamics import PopulationDynamicsModel
from endothelial_simulation.models.spatial_properties import SpatialPropertiesModel
from endothelial_simulation.visualization import Plotter
from endothelial_simulation.visualization.animations import create_detailed_cell_animation, create_metrics_animation


class Simulator:
    """
    Complete event-driven simulator with all original functionality restored.
    """

    def __init__(self, config):
        """Initialize the simulator with configuration parameters."""
        # === REPRODUCIBILITY ===
        # Seed both RNGs from the deterministic master seed (config.random_seed,
        # default 42) rather than the wall clock, so every run is reproducible.
        # `random` and `numpy as np` are already imported at module top.
        self.random_seed = int(getattr(config, 'random_seed', 42))  # dimensionless
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        print(f"🎲 RNG seed (master): {self.random_seed}")

        self.config = config

        # Create grid and disable force-based optimization
        self.grid = Grid(
            width=config.grid_size[0],
            height=config.grid_size[1],
            config=config
        )

        # Disable continuous biological adaptation
        self.grid.biological_optimization_enabled = False
        self.grid.continuous_adaptation_disabled = True

        # Enable energy tracking if biological optimization was enabled
        if getattr(config, 'biological_optimization_enabled', True):
            print("🔋 Enabling automatic energy tracking for biological optimization...")
            self.grid.enable_energy_tracking()
            self.energy_tracking_enabled = True
        else:
            self.energy_tracking_enabled = False

        # Initialize model components
        self.models = {}

        if config.enable_temporal_dynamics:
            self.models['temporal'] = TemporalDynamicsModel(config)

        if config.enable_population_dynamics:
            self.models['population'] = PopulationDynamicsModel(config)

        if config.enable_spatial_properties:
            temporal_model = self.models.get('temporal', None)
            self.models['spatial'] = SpatialPropertiesModel(config, temporal_model)

        # Initialize event-driven components
        from endothelial_simulation.core.event_system import EventDetector
        from endothelial_simulation.management.configuration_manager import ConfigurationManager
        from endothelial_simulation.management.transition_controller import TransitionController

        self.event_detector = EventDetector(config)
        self.configuration_manager = ConfigurationManager(self.grid, config)
        self.transition_controller = TransitionController(
            self.grid,
            temporal_model=self.models.get('temporal', None)
        )

        # Give configuration manager access to current conditions
        self.configuration_manager.simulator = self

        # Simulation state
        self.time = 0.0
        self.step_count = 0
        self.history = []

        # Input pattern information
        self.input_pattern = {
            'type': 'constant',
            'value': 0.0,
            'params': {}
        }

        # Event processing
        self.pending_events = []
        self.event_history = []
        self.last_reconfiguration_time = 0.0

        # Configuration tracking
        self.configuration_history = []
        self.current_configuration_id = 0

        # Animation settings
        self.record_frames = config.create_animations
        self.frame_data = []
        self.record_interval = 1

        # Mosaic-specific parameters (kept for compatibility)
        self.tessellation_update_interval = 1
        self.target_update_interval = 1
        self.position_optimization_interval = 20
        self.last_tessellation_update = 0
        self.last_position_optimization = 0

        # Add biological ID tracking
        self.biological_id_counter = 0
        self.biological_id_map = {}  # Maps grid_cell_id -> biological_id
        self.cell_properties_map = {}  # Maps biological_id -> persistent properties

        self.mpc_controller = None

    # =============================================================================
    # INITIALIZATION METHODS
    # =============================================================================

    def get_or_create_biological_id(self, grid_cell_id, cell_position, cell_divisions):
        """Get or create a persistent biological ID for a cell."""

        # Check if we already have a mapping for this grid cell
        if grid_cell_id in self.biological_id_map:
            return self.biological_id_map[grid_cell_id]

        # Try to find existing biological ID based on position and properties
        position_key = (round(cell_position[0], 1), round(cell_position[1], 1))

        for bio_id, props in self.cell_properties_map.items():
            stored_pos = props.get('original_position')
            stored_div = props.get('divisions')

            if (stored_pos and stored_div is not None and
                    abs(stored_pos[0] - cell_position[0]) < 50 and  # 50 pixel tolerance
                    abs(stored_pos[1] - cell_position[1]) < 50 and
                    stored_div == cell_divisions):
                # Found matching cell, reuse biological ID
                self.biological_id_map[grid_cell_id] = bio_id
                return bio_id

        # Create new biological ID
        self.biological_id_counter += 1
        bio_id = f"bio_{self.biological_id_counter:05d}"

        # Store mappings
        self.biological_id_map[grid_cell_id] = bio_id
        self.cell_properties_map[bio_id] = {
            'original_position': cell_position,
            'divisions': cell_divisions,
            'creation_time': self.time
        }

        return bio_id

    def initialize(self, cell_count=None):
        """Initialize with standard single configuration."""
        if cell_count is None:
            cell_count = self.config.initial_cell_count

        print(f"🔄 Initializing event-driven simulation with {cell_count} cells...")

        # Calculate base area per cell
        total_area = self.grid.width * self.grid.height
        base_area_per_cell = total_area / cell_count

        def area_distribution():
            return np.random.uniform(base_area_per_cell * 0.7, base_area_per_cell * 1.3)

        # Populate grid with initial cells
        self.grid.populate_grid(cell_count, area_distribution=area_distribution)

        # Apply the prescribed initial senescent composition (phi_sen(0) = 0.20, 70/30 split)
        self._apply_initial_senescence()

        # Initialize cell properties for current pressure
        self._initialize_cell_properties_for_pressure()

        # Initial adaptation
        self.grid.adapt_cell_properties()

        # Record initial state
        self._record_state()

        # Record initial energy state if tracking is enabled
        if self.energy_tracking_enabled:
            print("🔋 Recording initial energy state...")
            self.grid.record_energy_state(self.step_count, label="initialization")

        print(f"✅ Initialized with energy: {self.grid.calculate_biological_energy():.4f}")

    def initialize_with_multiple_configurations(self, cell_count=None, num_configurations=10,
                                                optimization_iterations=3, save_analysis=True):
        """
        Initialize by testing multiple configurations and selecting the best one.

        Parameters:
            cell_count: Number of cells to create (default: from config)
            num_configurations: Number of configurations to test
            optimization_iterations: Optimization steps per configuration
            save_analysis: Whether to save detailed analysis

        Returns:
            Dictionary with configuration selection results
        """
        if cell_count is None:
            cell_count = self.config.initial_cell_count

        print(f"🚀 Initializing simulation with multi-configuration selection:")
        print(f"   Target cells: {cell_count}")
        print(f"   Configurations to test: {num_configurations}")
        print(f"   Optimization iterations per config: {optimization_iterations}")

        # Calculate base area per cell for distribution
        total_area = self.grid.width * self.grid.height
        base_area_per_cell = total_area / cell_count

        def area_distribution():
            return np.random.uniform(base_area_per_cell * 0.7, base_area_per_cell * 1.3)

        def division_distribution():
            max_div = self.config.max_divisions
            r = np.random.random()
            return int(max_div * 0.5 * (1 - np.sqrt(r)))

        # Generate and test multiple configurations
        config_results = self.grid.generate_multiple_initial_configurations(
            cell_count=cell_count,
            num_configurations=num_configurations,
            division_distribution=division_distribution,
            area_distribution=area_distribution,
            optimization_iterations=optimization_iterations,
            verbose=True
        )

        # Apply the prescribed initial senescent composition (phi_sen(0) = 0.20, 70/30 split)
        self._apply_initial_senescence()

        # Initialize cell properties for current pressure
        self._initialize_cell_properties_for_pressure()

        # Final adaptation
        self.grid.adapt_cell_properties()

        # Record initial state
        self._record_state()

        # Record initial energy state if tracking is enabled
        if self.energy_tracking_enabled:
            print("🔋 Recording initial energy state...")
            self.grid.record_energy_state(self.step_count, label="initialization_best_config")

        # Save analysis if requested
        if save_analysis and hasattr(self.grid, 'save_configuration_analysis'):
            self.grid.save_configuration_analysis(config_results)

        print(f"\n✅ Initialization complete with best configuration selected!")
        print(f"   Final energy: {config_results['best_config']['energy']:.4f}")
        print(f"   Energy improvement: {config_results['energy_improvement']:.4f}")

        return config_results

    def initialize_smart(self, cell_count=None, **kwargs):
        """
        Smart initialization that automatically chooses between single and multi-configuration.
        """
        if cell_count is None:
            cell_count = self.config.initial_cell_count

        # Use multi-configuration for larger simulations or if requested
        use_multi_config = (
                cell_count >= 50 or  # Large simulations benefit more
                kwargs.get('force_multi_config', False) or
                getattr(self.config, 'use_multi_config_init', False)
        )

        if use_multi_config:
            # Set reasonable defaults based on simulation size
            default_configs = min(20, max(5, cell_count // 10))
            kwargs.setdefault('num_configurations', default_configs)
            kwargs.setdefault('optimization_iterations', 3)

            return self.initialize_with_multiple_configurations(cell_count, **kwargs)
        else:
            print(f"🚀 Using standard initialization for {cell_count} cells")
            return self.initialize(cell_count)

    def _apply_initial_senescence(self, fraction=None, stress_fraction=None,
                                  telomere_fraction=None):
        """
        Set the initial senescent composition of the monolayer.

        Marks a fraction phi_sen(0) of the cells as senescent, split into
        stress-induced (S_str) and telomere-induced (S_tel) compartments.

        Defaults (Source: project knowledge / Table 1, main.tex):
            phi_sen(0) = 0.20  (passage-6 HUVEC, PDL 15)
            70% of the senescent cells are stress-induced (S_str)
            30% of the senescent cells are telomere-induced (S_tel)
        """
        if fraction is None:
            fraction = getattr(self.config, 'initial_senescent_fraction', 0.0)
        if stress_fraction is None:
            stress_fraction = getattr(self.config, 'senescent_stress_fraction', 0.70)
        if telomere_fraction is None:
            telomere_fraction = getattr(self.config, 'senescent_telomere_fraction', 0.30)

        cells = list(self.grid.cells.values())
        n_total = len(cells)
        if n_total == 0 or fraction <= 0.0:
            return

        # Start from an all-healthy monolayer so the composition is exact.
        # reset_senescence() is the only sanctioned way to clear the otherwise
        # irreversible senescent state, and is used here for initial setup only.
        for cell in cells:
            cell.reset_senescence()

        n_sen = int(round(fraction * n_total))
        # Split the senescent pool 70/30 (stress/telomere) by default.
        denom = stress_fraction + telomere_fraction
        n_stress = int(round(n_sen * stress_fraction / denom)) if denom > 0 else 0
        n_tel = n_sen - n_stress

        order = np.random.permutation(n_total)
        selected = order[:n_sen]

        for k, idx in enumerate(selected):
            cell = cells[idx]
            cell.is_senescent = True
            if k < n_stress:
                cell.senescence_cause = 'stress'           # Source: 70% of senescent = S_str
            else:
                cell.senescence_cause = 'telomere'         # Source: 30% of senescent = S_tel
                # Telomere-induced cells have exhausted their replicative capacity.
                cell.divisions = self.config.max_divisions
                cell.telomere_length = 0.0

        achieved = n_sen / n_total if n_total else 0.0
        print(f"🧬 Initial senescence set: phi_sen(0)={achieved:.3f} "
              f"({n_sen}/{n_total} cells: {n_stress} stress, {n_tel} telomere)")

    def _initialize_cell_properties_for_pressure(self):
        """Initialize cell properties for current pressure."""
        current_pressure = self.input_pattern.get('value', 0.0)

        if 'spatial' not in self.models:
            return

        spatial_model = self.models['spatial']

        for cell_id, cell in self.grid.cells.items():
            old_target = cell.target_area
            target_area = spatial_model.calculate_target_area(
                current_pressure, cell.is_senescent, cell.senescence_cause
            )
            target_aspect_ratio = spatial_model.calculate_target_aspect_ratio(
                current_pressure, cell.is_senescent
            )
            target_orientation = spatial_model.calculate_target_orientation(
                current_pressure, cell.is_senescent
            )

            cell.target_area = target_area
            # Per-cell debug print disabled (kept only warnings / end-of-step summaries).
            # print(f"🐛 PRESSURE_UPDATE: Cell {cell_id}: {old_target} → {target_area}")
            cell.target_aspect_ratio = target_aspect_ratio
            cell.target_orientation = target_orientation
            cell.actual_aspect_ratio = target_aspect_ratio
            cell.actual_orientation = target_orientation

        self.grid._update_voronoi_tessellation()

    # =============================================================================
    # INPUT PATTERN METHODS
    # =============================================================================

    def set_constant_input(self, value):
        """Set a constant input value for MPC control."""
        self.input_pattern = {
            'type': 'constant',
            'value': value,
            'params': {'value': value}
        }

    # Add this method inside the Simulator class, after the existing methods
    def debug_quick_aspect_ratio_check(self):
        """
        Quick diagnostic of current aspect ratio state.
        """
        current_pressure = self.input_pattern.get('value', 0.0)

        print(f"\n🚀 QUICK ASPECT RATIO DIAGNOSIS")
        print(f"Current pressure: {current_pressure}")

        # Check first few cells
        for i, (cell_id, cell) in enumerate(list(self.grid.cells.items())[:3]):
            print(f"\nCell {cell_id}:")
            print(f"  Is senescent: {cell.is_senescent}")
            print(f"  Target AR: {getattr(cell, 'target_aspect_ratio', 'NOT SET')}")
            print(f"  Actual AR: {getattr(cell, 'actual_aspect_ratio', 'NOT SET')}")

            # What SHOULD it be?
            if 'spatial' in self.models:
                expected = self.models['spatial'].calculate_target_aspect_ratio(
                    current_pressure, cell.is_senescent
                )
                print(f"  Expected AR: {expected}")

                if hasattr(cell, 'target_aspect_ratio'):
                    match_target = abs(cell.target_aspect_ratio - expected) < 0.001
                    print(f"  Match target: {match_target}")
                else:
                    print(f"  Match target: NO TARGET SET")

            if i >= 2:  # Only show first 3 cells
                break
        print()

    def set_step_input(self, initial_value, final_value, step_time):
        """Set step input pattern."""
        self.input_pattern = {
            'type': 'step',
            'value': initial_value,
            'params': {
                'initial_value': initial_value,
                'final_value': final_value,
                'step_time': step_time
            }
        }

    def set_ramp_input(self, initial_value, final_value, ramp_start_time, ramp_end_time):
        """Set a ramp input pattern."""
        self.input_pattern = {
            'type': 'ramp',
            'value': initial_value,
            'params': {
                'initial_value': initial_value,
                'final_value': final_value,
                'ramp_start_time': ramp_start_time,
                'ramp_end_time': ramp_end_time
            }
        }

    def set_oscillatory_input(self, base_value, amplitude, frequency, phase=0):
        """Set an oscillatory input pattern."""
        self.input_pattern = {
            'type': 'oscillatory',
            'value': base_value,
            'params': {
                'base_value': base_value,
                'amplitude': amplitude,
                'frequency': frequency,
                'phase': phase
            }
        }

    def set_multi_step_input(self, step_schedule):
        """
        Set a multi-step input pattern with multiple step changes.

        Parameters:
            step_schedule: List of (time, value) tuples defining the schedule
        """
        self.input_pattern = {
            'type': 'multi_step',
            'value': step_schedule[0][1] if step_schedule else 0.0,
            'params': {
                'schedule': step_schedule
            }
        }

    def set_protocol_input(self, protocol_name, **kwargs):
        """
        Set a predefined protocol input pattern.

        Parameters:
            protocol_name: Name of the protocol to use
            **kwargs: Protocol-specific parameters
        """
        protocols = {
            'acute_stress': [(0, 0), (30, 2.0), (90, 0)],
            'chronic_stress': [(0, 0), (60, 1.0), (360, 1.0), (420, 0)],
            'stepwise_increase': [(0, 0), (60, 0.5), (120, 1.0), (180, 1.5), (240, 2.0)],
            'oscillatory_low': [(0, 0), (30, 1.0), (90, 0), (120, 1.0), (180, 0)],
            'high_stress_brief': [(0, 0), (45, 3.0), (75, 0)]
        }

        if protocol_name not in protocols:
            raise ValueError(f"Unknown protocol: {protocol_name}")

        # Get base schedule
        schedule = protocols[protocol_name].copy()

        # Apply scaling if requested
        scale_time = kwargs.get('scale_time', 1.0)
        scale_stress = kwargs.get('scale_stress', 1.0)
        max_stress = kwargs.get('max_stress', None)

        if scale_time != 1.0 or scale_stress != 1.0 or max_stress is not None:
            scaled_schedule = []
            for time_point, stress_value in schedule:
                new_time = time_point * scale_time
                new_stress = stress_value * scale_stress

                if max_stress is not None and new_stress > max_stress:
                    new_stress = max_stress

                scaled_schedule.append((new_time, new_stress))

            schedule = scaled_schedule

        self.set_multi_step_input(schedule)

    def update_input_value(self):
        """Update the current input value based on the input pattern and current time."""
        pattern_type = self.input_pattern['type']
        params = self.input_pattern['params']

        if pattern_type == 'constant':
            self.input_pattern['value'] = params.get('value', 0.0)

        elif pattern_type == 'step':
            if self.time < params['step_time']:
                self.input_pattern['value'] = params['initial_value']
            else:
                self.input_pattern['value'] = params['final_value']

        elif pattern_type == 'multi_step':
            schedule = params['schedule']
            current_value = schedule[0][1]  # Default to first value

            for i, (step_time, step_value) in enumerate(schedule):
                if self.time >= step_time:
                    current_value = step_value
                else:
                    break

            if self.input_pattern['value'] != current_value:
                old_value = self.input_pattern['value']
                self.input_pattern['value'] = current_value
                print(f"Step change at t={self.time:.1f}min: {old_value:.2f} → {current_value:.2f} Pa")

        elif pattern_type == 'ramp':
            if self.time < params['ramp_start_time']:
                self.input_pattern['value'] = params['initial_value']
            elif self.time > params['ramp_end_time']:
                self.input_pattern['value'] = params['final_value']
            else:
                progress = (self.time - params['ramp_start_time']) / (
                        params['ramp_end_time'] - params['ramp_start_time'])
                self.input_pattern['value'] = params['initial_value'] + progress * (
                        params['final_value'] - params['initial_value'])

        elif pattern_type == 'oscillatory':
            omega = 2 * np.pi * params['frequency']
            self.input_pattern['value'] = params['base_value'] + params['amplitude'] * np.sin(
                omega * self.time + params['phase'])

        return self.input_pattern['value']

    def _get_current_input(self):
        """Get current input value (for compatibility)."""
        return self.update_input_value()

    # =============================================================================
    # SIMULATION EXECUTION METHODS
    # =============================================================================

    def _convert_healthy_to_senescent(self, cause, preferred_divisions=None):
        """
        Convert an existing healthy cell to senescent instead of creating a new one.
        Maintains the same cell ID and just changes the state.
        """
        # Find healthy cells to convert
        healthy_cells = []
        for grid_cell_id, cell in self.grid.cells.items():
            if not cell.is_senescent:
                healthy_cells.append((grid_cell_id, cell))

        if not healthy_cells:
            print(f"Warning: No healthy cells available to convert to senescent ({cause})")
            return

        # Choose which cell to convert
        target_cell_id, target_cell = None, None

        if preferred_divisions is not None:
            # Try to find a cell with matching division count
            for grid_cell_id, cell in healthy_cells:
                if cell.divisions == preferred_divisions:
                    target_cell_id, target_cell = grid_cell_id, cell
                    break

        # If no specific cell found, convert the first available healthy cell
        if target_cell is None:
            target_cell_id, target_cell = healthy_cells[0]

        # Convert the cell (same ID, just change state)
        old_biological_id = getattr(target_cell, 'biological_id', 'unknown')

        # --- CRITICAL FIX ---
        # We must provide the spatial_model and current pressure to immediately update the cell's orientation.
        spatial_model = self.models.get('spatial')
        current_pressure = self.update_input_value() # Get the most up-to-date pressure

        target_cell.induce_senescence(
            cause,
            spatial_model=spatial_model,
            pressure=current_pressure
        )
        # --- END FIX ---

        print(f"🔄 Converted cell {old_biological_id} from healthy to senescent ({cause})")

    def step(self, dt=None):
        """Modified step method to include deterministic senescence checking"""
        if dt is None:
            dt = self.config.time_step

        self.time += dt
        self.step_count += 1
        current_input = self.update_input_value()

        # Detect events (pressure changes, senescence, holes)
        events = self.event_detector.detect_events(self)

        # Process any detected events
        for event in events:
            self._process_event(event)

        # Update active transitions
        if self.transition_controller.is_transitioning():
            self.transition_controller.update_transition(self.time, dt)

        # Apply shear stress (existing code)
        self._apply_shear_stress(current_input, dt)

        # SINGLE DETERMINISTIC SENESCENCE CHECK
        senescent_count = 0
        spatial_model = self.models.get('spatial')
        current_pressure = self.update_input_value()

        for grid_cell_id, cell in self.grid.cells.items():
            if cell.update_and_check_all_senescence(dt, self.config, spatial_model=spatial_model, pressure=current_pressure):
                senescent_count += 1

        if senescent_count > 0:
            print(f"📊 {senescent_count} cells became senescent (Time: {self.time:.1f})")

        # Update models (population dynamics, temporal dynamics)
        if 'temporal' in self.models and self.config.enable_temporal_dynamics:
            model = self.models['temporal']
            model.update_cell_responses(self.grid.cells, current_input, dt)

        # Update spatial properties if in transition
        if 'spatial' in self.models and self.config.enable_spatial_properties:
            model = self.models['spatial']
            # Set transition mode if transitioning
            model._in_transition_mode = self.transition_controller.is_transitioning()
            all_cells_dynamics_info = []
            for cell in self.grid.cells.values():
                dynamics_result = model.update_cell_properties(cell, current_input, dt, self.grid.cells)
                all_cells_dynamics_info.append({'cell_id': cell.cell_id, 'dynamics': dynamics_result})

        # Update population dynamics
        if 'population' in self.models and self.config.enable_population_dynamics:
            model = self.models['population']
            stem_cell_rate = self.config.stem_cell_rate if self.config.enable_stem_cells else 0
            model.update_from_cells(self.grid.cells, dt, current_input, stem_cell_rate)
            actions = model.synchronize_cells(self.grid.cells)
            self._execute_population_actions(actions)

        # Update hole system
        if hasattr(self.grid, 'update_holes'):
            self.grid.update_holes(dt)

        """ 
        # --- FIX: Periodically update adaptation, tessellation, and positions ---
        if self.step_count % self.tessellation_update_interval == 0:
            self.grid.adapt_cell_properties()
            self.grid._update_voronoi_tessellation()
            
        if self.step_count % self.position_optimization_interval == 0:
            self.grid.optimize_cell_positions(iterations=1)
        # --- END FIX ---
        """
        if self.step_count % self.target_update_interval == 0:
            # Update ONLY targets (no tessellation)
            if 'spatial' in self.models:
                current_input = self.input_pattern.get('value', 0.0)
                spatial_model = self.models['spatial']

                for cell in self.grid.cells.values():
                    # Use the existing methods to calculate targets
                    target_area = spatial_model.calculate_target_area(
                        current_input, cell.is_senescent, cell.senescence_cause
                    )
                    target_orientation = spatial_model.calculate_target_orientation(
                        current_input, cell.is_senescent
                    )
                    target_aspect_ratio = spatial_model.calculate_target_aspect_ratio(
                        current_input, cell.is_senescent
                    )

                    # Update targets
                    cell.target_area = target_area
                    cell.target_orientation = target_orientation
                    cell.target_aspect_ratio = target_aspect_ratio

        if self.step_count % self.tessellation_update_interval == 0:
            # Less frequent tessellation updates
            self.grid._update_voronoi_tessellation(preserve_temporal_dynamics=True)

        if self.step_count % self.position_optimization_interval == 30:
            self.grid.optimize_cell_positions(iterations=1)

        # Record frames for animation
        if self.record_frames and self.step_count % self.record_interval == 0:
            self._record_frame()

        # Record state
        self._record_state(all_cells_dynamics_info)

        # THIS IS THE IMPORTANT PART - RETURN THE RIGHT KEYS
        return {
            'time': self.time,
            'step_count': self.step_count,
            'cell_count': len(self.grid.cells),
            'input_value': current_input,
            'transitioning': self.transition_controller.is_transitioning()
        }

    def get_stress_statistics(self):

            if not self.grid.cells:
                return {}

            stress_data = []
            at_risk_count = 0
            will_senesce_count = 0

            for cell in self.grid.cells.values():
                if not cell.is_senescent:
                    status = cell.get_stress_status(self.config)  # Pass config parameter
                    stress_data.append(status)

                    if status['at_risk']:
                        at_risk_count += 1
                    if status['will_senesce']:
                        will_senesce_count += 1

            if not stress_data:
                return {}

            final_stresses = [s['final_stress'] for s in stress_data]
            stress_ratios = [s['stress_ratio'] for s in stress_data]

            return {
                'total_healthy_cells': len(stress_data),
                'cells_at_risk': at_risk_count,
                'cells_will_senesce': will_senesce_count,
                'avg_final_stress': np.mean(final_stresses),
                'max_final_stress': np.max(final_stresses),
                'avg_stress_ratio': np.mean(stress_ratios),
                'max_stress_ratio': np.max(stress_ratios)
            }



    # Example usage in your main simulation loop:
    def run_simulation_with_monitoring():
        """Example of how to monitor the deterministic senescence"""

        simulator = Simulator(config)

        # Your existing simulation setup...
        simulator.initialize(cell_count=config.initial_cell_count)

        for step in range(num_steps):
            step_result = simulator.step()  # Use your existing step method

            # Monitor stress status every 10 steps
            if step % 10 == 0:
                stress_stats = simulator.get_stress_statistics()
                if stress_stats:
                    print(f"Step {step}: {stress_stats['cells_at_risk']} cells at risk, "
                          f"{stress_stats['cells_will_senesce']} will senesce")
                    print(f"  Max stress ratio: {stress_stats['max_stress_ratio']:.2f}")

    # Or use your existing run method:
    def monitor_during_run():
        """Monitor stress during your existing run() method"""
        simulator = Simulator(config)
        simulator.initialize(cell_count=config.initial_cell_count)

        # Set up your input pattern (step, constant, etc.)
        simulator.set_step_input(initial_value=0, final_value=2.0, step_time=30)

        # Run simulation (this calls step() internally)
        results = simulator.run(duration=120)  # 2 hours

        # Check final stress statistics
        final_stats = simulator.get_stress_statistics()
        print(f"Final simulation stats: {final_stats}")

    def run(self, duration=None):
        """
        Run simulation for specified duration.

        Parameters:
            duration: Duration to run in simulation time units (default: from config)

        Returns:
            Dictionary with simulation results
        """
        if duration is None:
            duration = self.config.simulation_duration

        # Calculate number of steps
        dt = self.config.time_step
        num_steps = int(duration / dt)
        target_time = self.time + duration

        print(f"🚀 Running event-driven simulation for {duration} minutes ({num_steps} steps)...")
        start_time = time.time()

        # --- FRAME RECORDING FIX ---
        # Record initial state if animation is enabled
        if self.record_frames and not self.frame_data:
            print("🔴 Recording initial frame for animation.")
            self._record_frame()
        # --- END FIX ---

        # Run steps
        for i in range(num_steps):
            if self.time >= target_time:
                break

            step_info = self.step(dt)

            # Print progress periodically
            if (i + 1) % 100 == 0 or i == num_steps - 1:
                progress = min((i + 1) / num_steps * 100, 100)
                elapsed = time.time() - start_time
                estimated_total = elapsed / (i + 1) * num_steps if i > 0 else elapsed
                remaining = max(0, estimated_total - elapsed)

                # Get current grid statistics
                grid_stats = self.grid.get_grid_statistics()
                packing_eff = grid_stats.get('packing_efficiency', 0)

                print(f"Progress: {progress:.1f}% (Step {i + 1}/{num_steps}), "
                      f"Time: {elapsed:.1f}s, Remaining: {remaining:.1f}s, "
                      f"Cells: {step_info['cell_count']}, "
                      f"Packing: {packing_eff:.2f}, "
                      f"Transitioning: {step_info['transitioning']}")

        # --- FRAME RECORDING FIX ---
        # Record final state if animation is enabled
        if self.record_frames:
            print("🔴 Recording final frame for animation.")
            self._record_frame()
        # --- END FIX ---

        end_time = time.time()
        total_time = end_time - start_time

        print(f"✅ Event-driven simulation completed in {total_time:.1f} seconds")

        # Return results
        return {
            'duration': duration,
            'steps': num_steps,
            'final_time': self.time,
            'execution_time': total_time,
            'history': self.history,
            'time_points': [state['time'] for state in self.history],
            'animations_created': self.record_frames and len(self.frame_data) > 0,
            'final_grid_stats': self.grid.get_grid_statistics(),
            'configuration_history': getattr(self, 'configuration_history', []),
            'event_history': getattr(self, 'event_history', [])
        }

    # =============================================================================
    # EVENT-DRIVEN LOGIC
    # =============================================================================

    def _process_event(self, event):
        """Process a detected event by triggering reconfiguration."""
        print(f"🔍 Processing event: {event.event_type.name} at t={self.time / 60:.1f}h")

        # Check minimum interval between reconfigurations
        time_since_last = self.time - self.last_reconfiguration_time
        min_interval = getattr(self.config, 'min_reconfiguration_interval', 30.0)

        if time_since_last < min_interval:
            print(f"   ⏳ Skipping - too soon (last: {time_since_last:.1f}min ago)")
            return

        # Generate reconfiguration using the correct method
        try:
            print(f"   🔄 Triggering reconfiguration...")

            reconfiguration_result = self.configuration_manager.generate_reconfiguration(
                event=event,
                num_configurations=getattr(self.config, 'multi_config_count', 5),
                optimization_iterations=3
            )

            # Check if reconfiguration provides meaningful improvement
            energy_improvement = reconfiguration_result.get('energy_improvement', 0)

            if energy_improvement > 0.001:  # Meaningful improvement threshold
                # Start transition to new configuration
                self.transition_controller.start_transition(
                    reconfiguration_result=reconfiguration_result,
                    current_time=self.time
                )

                self.last_reconfiguration_time = self.time

                # Record event
                event_record = {
                    'time': self.time,
                    'event': event.event_type.name,
                    'pressure': self.update_input_value(),
                    'energy_improvement': energy_improvement,
                    'cell_count': len(self.grid.cells)
                }

                self.configuration_history.append(event_record)
                self.event_history.append(event_record)

                print(f"   ✅ Reconfiguration started (ΔE: {energy_improvement:.4f})")
            else:
                print(f"   ❌ No beneficial reconfiguration found (ΔE: {energy_improvement:.4f})")

        except Exception as e:
            print(f"   ⚠️  Reconfiguration failed: {e}")
            # Continue simulation even if reconfiguration fails

    # =============================================================================
    # UTILITY METHODS
    # =============================================================================

    def _apply_shear_stress(self, shear_stress, duration):
        """Apply shear stress to all cells."""
        if hasattr(self.grid, 'apply_shear_stress_field'):
            def shear_stress_function(x, y):
                return shear_stress
            self.grid.apply_shear_stress_field(shear_stress_function, duration)

    def _execute_population_actions(self, actions):
        """
        Execute population actions from PopulationDynamicsModel.
        FIXED: Convert existing cells to senescent instead of creating new ones.
        """
        if not actions or not isinstance(actions, dict):
            return

        # Process births (only for healthy cells now)
        births = actions.get('births', [])
        healthy_births = [b for b in births if b['type'] == 'healthy']
        senescent_births = [b for b in births if b['type'] == 'senescent']

        # Add healthy cells normally
        for birth in healthy_births:
            self._add_healthy_cell(birth['divisions'])

        # FIXED: Convert existing healthy cells to senescent instead of adding new ones
        for birth in senescent_births:
            self._convert_healthy_to_senescent(birth['cause'], birth.get('divisions', None))

        # Process deaths (remove cells)
        deaths = actions.get('deaths', [])
        for death in deaths:
            if death['type'] == 'healthy':
                self._remove_healthy_cells(death['divisions'], death['count'])
            elif death['type'] == 'senescent':
                self._remove_senescent_cells(death['cause'], death['count'])

        # Log population changes
        if births or deaths:
            total_deaths = sum(d.get('count', 1) for d in deaths)
            senescent_conversions = len(senescent_births)
            healthy_births_count = len(healthy_births)

            print(f"  Population: +{healthy_births_count} healthy births, "
                  f"{senescent_conversions} conversions to senescent, -{total_deaths} deaths")


    def _add_healthy_cell(self, divisions):
        """Add a healthy cell with specified division count."""
        try:
            if hasattr(self.grid, 'add_cell'):
                self.grid.add_cell(divisions=divisions, is_senescent=False)
            else:
                self._create_cell_manually(divisions=divisions, is_senescent=False)
        except Exception as e:
            print(f"Warning: Could not add healthy cell: {e}")


    def _remove_healthy_cells(self, divisions, count):
        """Remove specified number of healthy cells with given division count."""
        if count <= 0:
            return

        # Find matching cells
        candidates = [
            cell_id for cell_id, cell in self.grid.cells.items()
            if not cell.is_senescent and getattr(cell, 'divisions', 0) == divisions
        ]

        # Remove up to 'count' cells
        removed = 0
        for cell_id in candidates:
            if removed >= count:
                break
            if self.grid.remove_cell(cell_id):
                removed += 1

    def _remove_senescent_cells(self, cause, count):
        """Remove specified number of senescent cells with given cause."""
        if count <= 0:
            return

        # Find matching cells
        candidates = [
            cell_id for cell_id, cell in self.grid.cells.items()
            if cell.is_senescent and getattr(cell, 'senescence_cause', None) == cause
        ]

        # Remove up to 'count' cells
        removed = 0
        for cell_id in candidates:
            if removed >= count:
                break
            if self.grid.remove_cell(cell_id):
                removed += 1

    def _create_cell_manually(self, **cell_properties):
        """
        Fallback method to create a cell manually if grid.add_cell doesn't exist.
        """
        try:
            from endothelial_simulation.core.cell import Cell
            import uuid

            # Generate unique cell ID
            cell_id = str(uuid.uuid4())

            # Create cell with basic properties
            cell = Cell(
                cell_id=cell_id,
                position=(
                    np.random.uniform(0, self.grid.width),
                    np.random.uniform(0, self.grid.height)
                ),
                **cell_properties
            )

            # Add to grid
            self.grid.cells[cell_id] = cell

            # Initialize spatial properties if model exists
            if 'spatial' in self.models:
                current_pressure = self.update_input_value()
                spatial_model = self.models['spatial']

                cell.target_area = spatial_model.calculate_target_area(
                    current_pressure, cell.is_senescent, getattr(cell, 'senescence_cause', None)
                )
                cell.target_aspect_ratio = spatial_model.calculate_target_aspect_ratio(
                    current_pressure, cell.is_senescent
                )
                cell.target_orientation = spatial_model.calculate_target_orientation(
                    current_pressure, cell.is_senescent
                )

            # Update tessellation
            if hasattr(self.grid, '_update_voronoi_tessellation'):
                self.grid._update_voronoi_tessellation()

        except Exception as e:
            print(f"Warning: Manual cell creation failed: {e}")

    # =============================================================================
    # STATE RECORDING AND ANALYSIS
    # =============================================================================

    def _record_state(self, all_cells_dynamics_info=None):
        """Record current simulation state."""
        # Get cell properties
        cell_properties = self.grid.get_cell_properties()

        # Build basic state
        state = {
            'time': self.time,
            'step_count': self.step_count,
            'cell_count': len(self.grid.cells),
            'input_value': self.update_input_value(),
            'biological_energy': self.grid.calculate_biological_energy(),
            'is_transitioning': self.transition_controller.is_transitioning() if hasattr(self, 'transition_controller') else False,
            'cell_dynamics_info': all_cells_dynamics_info # Add the new dynamics info
        }

        # Add cell properties
        state['cell_properties'] = cell_properties

        # Add population dynamics if enabled
        if 'population' in self.models:
            pop_model = self.models['population']
            totals = pop_model.calculate_total_cells()
            avg_div = pop_model.calculate_average_division_age()
            tel_len = pop_model.calculate_telomere_length()

            state.update({
                'healthy_cells': totals['E_total'],
                'senescent_tel': totals['S_tel'],
                'senescent_stress': totals['S_stress'],
                'avg_division_age': avg_div,
                'telomere_length': tel_len
            })

        # Add spatial properties if enabled
        if 'spatial' in self.models:
            spatial_model = self.models['spatial']
            collective_props = spatial_model.calculate_collective_properties(
                self.grid.cells, self.update_input_value()
            )
            state.update(collective_props)

            alignment = spatial_model.calculate_alignment_index(self.grid.cells)
            shape_index = spatial_model.calculate_shape_index(self.grid.cells)
            packing_quality = spatial_model.calculate_packing_quality(self.grid.cells)

            state.update({
                'alignment_index': alignment,
                'shape_index': shape_index,
                'packing_quality': packing_quality,
                'confluency': self.grid.calculate_confluency()
            })

        # Calculate adaptation metrics
        if len(self.grid.cells) > 0:
            target_areas = cell_properties['target_areas']
            target_ars = cell_properties['target_aspect_ratios']
            actual_areas = cell_properties['areas']
            actual_ars = cell_properties['aspect_ratios']

            # Mean adaptation errors
            area_adaptation_error = np.mean([abs(t - a) / max(t, 1) for t, a in zip(target_areas, actual_areas)])
            ar_adaptation_error = np.mean([abs(t - a) / max(t, 1) for t, a in zip(target_ars, actual_ars)])

            state.update({
                'mean_target_area': np.mean(target_areas),
                'std_target_area': np.std(target_areas),
                'mean_target_aspect_ratio': np.mean(target_ars),
                'std_target_aspect_ratio': np.std(target_ars),
                'area_adaptation_error': area_adaptation_error,
                'ar_adaptation_error': ar_adaptation_error,
            })

        self.history.append(state)

    # Replace your current senescence debugging with this enhanced version

    def _record_frame(self):
        """Record frame data with detailed senescence bias debugging."""
        cells_data = []

        # Detailed analysis of cell properties
        healthy_props = []
        senescent_props = []

        for grid_cell_id, cell in self.grid.cells.items():
            biological_id = self.get_or_create_biological_id(
                grid_cell_id,
                cell.position,
                cell.divisions
            )

            cell.biological_id = biological_id

            # Collect detailed properties
            bio_id_num = int(biological_id.replace('bio_', ''))

            cell_props = {
                'bio_id': biological_id,
                'bio_id_num': bio_id_num,
                'divisions': cell.divisions,
                'age': cell.age,
                'stress_resistance': getattr(cell, 'stress_resistance', 1.0),
                'local_shear_stress': cell.local_shear_stress,
                'stress_exposure_time': getattr(cell, 'stress_exposure_time', 0.0),
                'creation_time': self.cell_properties_map[biological_id].get('creation_time', self.time),
                'target_area': getattr(cell, 'target_area', 0),
                'is_senescent': cell.is_senescent,
                'senescence_cause': cell.senescence_cause
            }

            if cell.is_senescent:
                senescent_props.append(cell_props)
            else:
                healthy_props.append(cell_props)

            # Calculate senescence probability for debugging
            if hasattr(cell, 'calculate_senescence_probability'):
                try:
                    sen_prob = cell.calculate_senescence_probability(self.config)
                    stress_prob = sen_prob.get('stress', 0)
                    tel_prob = sen_prob.get('telomere', 0)

                    if stress_prob > 0.1 or tel_prob > 0.1:  # High probability
                        print(f"⚠️  {biological_id} (#{bio_id_num:02d}) HIGH SENESCENCE RISK:")
                        print(f"     Stress prob: {stress_prob:.3f}, Tel prob: {tel_prob:.3f}")
                        print(f"     Divisions: {cell.divisions}, Age: {cell.age:.1f}")
                        print(f"     Stress resistance: {cell_props['stress_resistance']:.2f}")
                        print(f"     Local shear: {cell.local_shear_stress:.2f}")
                        print(f"     Creation time: {cell_props['creation_time']:.1f}")
                except Exception as e:
                    print(f"❌ Error calculating senescence for {biological_id}: {e}")

            # Standard frame data
            cells_data.append({
                'cell_id': biological_id,
                'position': cell.position,
                'orientation': cell.actual_orientation,
                'aspect_ratio': cell.actual_aspect_ratio,
                'area': cell.actual_area,
                'is_senescent': cell.is_senescent,
                'senescence_cause': cell.senescence_cause,
                'territory': self.grid.get_display_territories().get(grid_cell_id, [])
            })

        # DETAILED COMPARISON ANALYSIS
        if senescent_props and healthy_props:
            print(f"\n🔬 DETAILED SENESCENCE BIAS ANALYSIS:")
            print(f"=" * 50)

            # Compare averages
            sen_avg_div = np.mean([c['divisions'] for c in senescent_props])
            healthy_avg_div = np.mean([c['divisions'] for c in healthy_props])

            sen_avg_age = np.mean([c['age'] for c in senescent_props])
            healthy_avg_age = np.mean([c['age'] for c in healthy_props])

            sen_avg_stress_res = np.mean([c['stress_resistance'] for c in senescent_props])
            healthy_avg_stress_res = np.mean([c['stress_resistance'] for c in healthy_props])

            sen_avg_creation = np.mean([c['creation_time'] for c in senescent_props])
            healthy_avg_creation = np.mean([c['creation_time'] for c in healthy_props])

            print(f"SENESCENT CELLS (n={len(senescent_props)}):")
            print(f"  Avg divisions: {sen_avg_div:.2f}")
            print(f"  Avg age: {sen_avg_age:.2f}")
            print(f"  Avg stress resistance: {sen_avg_stress_res:.2f}")
            print(f"  Avg creation time: {sen_avg_creation:.1f}")

            print(f"\nHEALTHY CELLS (n={len(healthy_props)}):")
            print(f"  Avg divisions: {healthy_avg_div:.2f}")
            print(f"  Avg age: {healthy_avg_age:.2f}")
            print(f"  Avg stress resistance: {healthy_avg_stress_res:.2f}")
            print(f"  Avg creation time: {healthy_avg_creation:.1f}")

            # Identify the smoking gun
            print(f"\n🎯 KEY DIFFERENCES:")

            div_diff = sen_avg_div - healthy_avg_div
            if abs(div_diff) > 1:
                print(f"  📊 DIVISIONS: Senescent cells have {div_diff:+.2f} more divisions!")

            age_diff = sen_avg_age - healthy_avg_age
            if abs(age_diff) > 10:
                print(f"  ⏰ AGE: Senescent cells are {age_diff:+.2f} time units older!")

            stress_res_diff = sen_avg_stress_res - healthy_avg_stress_res
            if abs(stress_res_diff) > 0.5:
                print(f"  🛡️  STRESS RESISTANCE: Senescent cells have {stress_res_diff:+.2f} different resistance!")

            creation_diff = sen_avg_creation - healthy_avg_creation
            if abs(creation_diff) > 30:
                print(f"  🕰️  CREATION TIME: Senescent cells created {creation_diff:+.1f} time units later!")

            print(f"=" * 50)

        # Store frame data
        cell_properties = self.grid.get_cell_properties()

        frame_info = {
            'time': self.time,
            'input_value': self.input_pattern['value'],
            'cell_count': len(self.grid.cells),
            'cells': cells_data,
            'cell_properties': cell_properties,
            'transitioning': self.transition_controller.is_transitioning()
        }

        self.frame_data.append(frame_info)

    # =============================================================================
    # RESULTS AND ANALYSIS METHODS
    # =============================================================================

    def save_results(self, filename):
        """Save simulation results to file."""
        # Ensure directory exists
        save_dir = self.config.plot_directory
        os.makedirs(save_dir, exist_ok=True)

        # Create full path
        if not filename.endswith('.npz'):
            filename += '.npz'
        filepath = os.path.join(save_dir, filename)

        # Prepare data for saving
        save_data = {
            'history': np.array(self.history, dtype=object),
            'time_points': np.array([state['time'] for state in self.history]),
            'configuration_history': np.array(getattr(self, 'configuration_history', []), dtype=object),
            'event_history': np.array(getattr(self, 'event_history', []), dtype=object),
            'final_stats': self.grid.get_grid_statistics(),
            'config_params': {
                'duration': self.time,
                'cell_count': len(self.grid.cells),
                'grid_size': self.config.grid_size,
                'time_step': self.config.time_step
            }
        }

        # Add frame data if available
        if self.frame_data:
            save_data['frame_data'] = np.array(self.frame_data, dtype=object)

        # Save to file
        np.savez_compressed(filepath, **save_data)
        print(f"💾 Results saved to: {filepath}")

        return filepath

    # Add this function to your simulator class or as a standalone function

    def test_cell_id_consistency(simulator):
        """
        Test function to verify cell ID consistency after your changes.
        Run this after a simulation to check if biological IDs are working.
        """
        if not simulator.frame_data or len(simulator.frame_data) < 2:
            print("❌ Need at least 2 frames to test consistency")
            return False

        print("🧪 Testing cell ID consistency...")

        # Track cells across first few frames
        consistency_issues = 0
        transitions_found = 0

        for frame_idx in range(1, min(len(simulator.frame_data), 6)):  # Check first 5 transitions
            prev_frame = simulator.frame_data[frame_idx - 1]
            current_frame = simulator.frame_data[frame_idx]

            # Create lookup for current frame cells
            current_cells = {cell['cell_id']: cell for cell in current_frame['cells']}

            # Check each cell from previous frame
            for prev_cell in prev_frame['cells']:
                cell_id = prev_cell['cell_id']

                if cell_id in current_cells:
                    current_cell = current_cells[cell_id]

                    # Check for senescence transitions
                    if not prev_cell['is_senescent'] and current_cell['is_senescent']:
                        transitions_found += 1
                        print(
                            f"  ✅ Cell {cell_id} transitioned to senescent (cause: {current_cell['senescence_cause']})")

                    # Check for impossible reversals
                    if prev_cell['is_senescent'] and not current_cell['is_senescent']:
                        consistency_issues += 1
                        print(f"  ❌ Cell {cell_id} reverted from senescent to healthy (impossible!)")

                    # Check for senescence cause changes
                    if (prev_cell['is_senescent'] and current_cell['is_senescent'] and
                            prev_cell['senescence_cause'] != current_cell['senescence_cause']):
                        consistency_issues += 1
                        print(f"  ❌ Cell {cell_id} changed senescence cause (impossible!)")

        # Summary
        print(f"\n📊 Test Results:")
        print(f"   Frames tested: {min(len(simulator.frame_data), 6)}")
        print(f"   Senescence transitions found: {transitions_found}")
        print(f"   Consistency issues: {consistency_issues}")

        if consistency_issues == 0:
            print("✅ Cell ID consistency test PASSED!")
            return True
        else:
            print("❌ Cell ID consistency test FAILED!")
            return False

    # Also add this simple check function
    def check_biological_ids(simulator):
        """Quick check to see if cells have biological IDs."""
        if not simulator.grid.cells:
            print("❌ No cells in grid")
            return

        cells_with_bio_id = 0
        total_cells = len(simulator.grid.cells)

        for cell in simulator.grid.cells.values():
            if hasattr(cell, 'biological_id'):
                cells_with_bio_id += 1

        print(f"📊 Biological ID Status:")
        print(f"   Total cells: {total_cells}")
        print(f"   Cells with biological ID: {cells_with_bio_id}")
        print(f"   Coverage: {cells_with_bio_id / total_cells * 100:.1f}%")

        if cells_with_bio_id == total_cells:
            print("✅ All cells have biological IDs!")
        else:
            print(f"⚠️  {total_cells - cells_with_bio_id} cells missing biological IDs")

        # Show a few example IDs
        example_ids = []
        for cell in list(simulator.grid.cells.values())[:3]:
            if hasattr(cell, 'biological_id'):
                example_ids.append(cell.biological_id)

        if example_ids:
            print(f"   Example biological IDs: {example_ids}")

    # Test your animation works
    def test_animation_with_biological_ids(plotter, simulator):
        """Test that the animation shows consistent cell IDs."""
        if not simulator.frame_data:
            print("❌ No frame data for animation")
            return

        print("🎬 Testing animation with biological IDs...")

        # Check first frame
        first_frame = simulator.frame_data[0]
        print(f"   First frame has {len(first_frame['cells'])} cells")

        # Show sample cell data
        if first_frame['cells']:
            sample_cell = first_frame['cells'][0]
            print(f"   Sample cell data: {sample_cell}")

        # Try creating animation (this will use your fixed _record_frame method)
        try:
            animation = plotter.create_mosaic_animation(
                simulator,
                save_path="test_biological_ids.mp4",
                fps=1,
                max_frames=5
            )
            if animation:
                print("✅ Animation created successfully with biological IDs!")
            else:
                print("❌ Animation creation failed")
        except Exception as e:
            print(f"❌ Animation error: {e}")
            print("   Check that your _record_frame changes are correct")

    def get_safe_final_statistics(self):
        """
        Get final statistics, guarding against numerical failures.

        Reproducibility/reporting contract: a numerical failure is reported as an
        explicit NaN accompanied by a warning — NEVER as a masked value of 0
        (which is indistinguishable from a legitimate zero result). Exception
        handling is narrowed to the numerical failure modes; genuinely structural
        errors (e.g. a missing grid) are allowed to surface to the caller rather
        than being swallowed into an all-zeros dictionary.

        Returns:
            Dictionary of final statistics. Numeric fields are NaN if their
            computation failed (with a warning logged); count fields are 0 only
            when the count is genuinely zero.
        """
        stats = {}

        # Basic cell counts (pure iteration, not a numerical computation).
        total_cells = len(self.grid.cells)
        stats['total_cells'] = total_cells

        senescent_count = 0
        healthy_count = 0
        for cell in self.grid.cells.values():
            if getattr(cell, 'is_senescent', False):
                senescent_count += 1
            else:
                healthy_count += 1
        stats['healthy_cells'] = healthy_count
        stats['senescent_cells'] = senescent_count

        # Biological energy (dimensionless model energy). Report the true finite
        # value; on a numerical failure or a non-finite result, report NaN + a
        # warning — not 0.0, and with no silent 1e6 cap that would hide overflow.
        try:
            biological_energy = float(self.grid.calculate_biological_energy())
        except (ArithmeticError, ValueError, OverflowError) as e:
            warnings.warn(
                f"biological_energy computation failed numerically ({e}); "
                "reporting NaN, not 0.", RuntimeWarning)
            biological_energy = float('nan')
        else:
            if not np.isfinite(biological_energy):
                warnings.warn(
                    f"biological_energy is non-finite ({biological_energy}); "
                    "reporting NaN, not 0.", RuntimeWarning)
                biological_energy = float('nan')
        stats['biological_energy'] = biological_energy

        # Packing efficiency = cells / grid area (dimensionless, clipped to 1).
        try:
            grid_area = self.grid.width * self.grid.height
            if grid_area <= 0:
                warnings.warn(
                    f"grid area is non-positive ({grid_area}); packing efficiency "
                    "is undefined -> NaN, not 0.", RuntimeWarning)
                packing_efficiency = float('nan')
            else:
                packing_efficiency = min(total_cells / grid_area, 1.0)
        except (ArithmeticError, ValueError, OverflowError) as e:
            warnings.warn(
                f"packing_efficiency computation failed numerically ({e}); "
                "reporting NaN, not 0.", RuntimeWarning)
            packing_efficiency = float('nan')
        stats['packing_efficiency'] = packing_efficiency

        # Event-driven bookkeeping (list lengths — genuine counts, never masked).
        stats['reconfigurations_count'] = len(getattr(self, 'configuration_history', []))
        stats['events_count'] = len(getattr(self, 'event_history', []))

        return stats

    def get_best_config_parameters(self, save_excel=False, excel_path=None):
        """
        Show parameters for the best configuration.
        """
        if not hasattr(self, '_config_results') or not self._config_results:
            print("❌ No configuration results available.")
            return None

        config_results = self._config_results
        best_config = config_results['best_config']


        # Extract parameters from best configuration
        params = {
            'config_id': best_config['config_idx'],
            'energy': best_config['energy'],
            'cell_count': len(best_config['cell_data']),
            'cells': []
        }

        # Check that all cells from config exist in grid
        missing_cells = [cell_id for cell_id in best_config['cell_data'].keys()
                         if cell_id not in self.grid.cells]

        if missing_cells:
            print(f"⚠️ Warning: {len(missing_cells)} cells missing from grid: {missing_cells[:5]}...")

        # Get actual values from current grid state
        for cell_id, cell_data in best_config['cell_data'].items():
            if cell_id in self.grid.cells:  # Only process existing cells
                actual_cell = self.grid.cells[cell_id]
                cell_params = {
                    'cell_id': cell_id,
                    'target_area': cell_data['target_area'],
                    'area': actual_cell.actual_area,
                    'target_aspect_ratio': cell_data['target_aspect_ratio'],
                    'aspect_ratio': actual_cell.actual_aspect_ratio,
                    'target_orientation_degrees': np.degrees(cell_data['target_orientation']),
                    'orientation_degrees': np.degrees(actual_cell.actual_orientation),
                    'divisions': cell_data.get('divisions', 0),
                    'is_senescent': cell_data.get('is_senescent', False)
                }
                params['cells'].append(cell_params)

        print(f"📊 Processed {len(params['cells'])} cells with actual tessellation data")

        # Calculate averages
        areas = [cell['area'] for cell in params['cells']]
        ars = [cell['aspect_ratio'] for cell in params['cells']]
        orientations = [cell['orientation_degrees'] for cell in params['cells']]

        params['averages'] = {
            'area': np.mean(areas),
            'aspect_ratio': np.mean(ars),
            'orientation_degrees': np.mean(orientations)
        }

        # Save to Excel if requested
        if save_excel:
            try:
                import pandas as pd

                if excel_path is None:
                    excel_path = os.path.join(self.config.plot_directory, 'best_config_parameters.xlsx')

                # Create DataFrame
                df = pd.DataFrame(params['cells'])

                # Save to Excel
                with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Cell_Parameters', index=False)

                    # Add summary sheet
                    summary_df = pd.DataFrame([params['averages']], index=['Average'])
                    summary_df.to_excel(writer, sheet_name='Summary')

                print(f"📊 Parameters saved to Excel: {excel_path}")

            except ImportError:
                print("⚠️  pandas not available - Excel export skipped")

        return params

    # =============================================================================
    # ANIMATION AND VISUALIZATION METHODS
    # =============================================================================

    def _create_animations(self):
        """Create animations for the simulation."""
        if not self.frame_data:
            print("⚠️  No frame data available for animation")
            return

        print(f"🎬 Creating animations from {len(self.frame_data)} frames...")

        try:
            # Create plotter
            plotter = Plotter(self.config)

            # Create detailed cell animation
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            animation_path = os.path.join(self.config.plot_directory, f"cell_animation_{timestamp}.mp4")

            ani = create_detailed_cell_animation(
                plotter, self.frame_data, self,
                save_path=animation_path,
                fps=10, dpi=100
            )

            if ani:
                print(f"✅ Cell animation saved to: {animation_path}")

            # Create metrics animation if history is available
            if self.history:
                metrics_path = os.path.join(self.config.plot_directory, f"metrics_animation_{timestamp}.mp4")
                metrics_ani = create_metrics_animation(
                    plotter, self.history,
                    save_path=metrics_path,
                    fps=10
                )

                if metrics_ani:
                    print(f"✅ Metrics animation saved to: {metrics_path}")

        except Exception as e:
            print(f"⚠️  Animation creation failed: {e}")

    def plot_energy_evolution(self, save_path=None):
        """Plot energy evolution if tracking is enabled."""
        if self.energy_tracking_enabled and hasattr(self.grid, 'plot_energy_evolution'):
            return self.grid.plot_energy_evolution(save_path)
        else:
            print("Energy tracking not enabled")
            return None

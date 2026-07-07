"""
Event-driven simulator that uses configuration transitions instead of continuous forces.
Integrates EventDetector, ConfigurationManager, and TransitionController.

DEPRECATED / DEAD CODE: the ``EventDrivenSimulator`` class below is a stale,
near-duplicate copy of ``core/simulator.py:Simulator``. The call-graph audit
(docs/code_audit.md) confirms it is never imported or instantiated anywhere in
the package, the CLI, or the analysis scripts, and it is NOT used by
run_mpc_simulation / the reported model. It is retained only for reference; do
not build new work on it (use ``core.simulator.Simulator``).
"""
import numpy as np
import time
import os
from typing import Dict, List, Optional

#Import existing classes (these would be from your existing modules)
from endothelial_simulation.core.grid import Grid
from endothelial_simulation.models.temporal_dynamics import TemporalDynamicsModel
from endothelial_simulation.models.population_dynamics import PopulationDynamicsModel
from endothelial_simulation.models.spatial_properties import SpatialPropertiesModel

# Import our new components
from endothelial_simulation.core.event_system import EventDetector, ConfigurationEvent, EventType
from .configuration_manager import ConfigurationManager
from .transition_controller import TransitionController


class EventDrivenSimulator:
    """
    Main simulator class using event-driven configuration changes.
    Replaces continuous force-based adaptation with discrete transitions.
    """
    
    def __init__(self, config):
        """Initialize the event-driven simulator."""
        # === REPRODUCIBILITY ===
        # Seed both RNGs from the deterministic master seed (config.random_seed,
        # default 42) rather than the wall clock. (Legacy class — see the
        # module-level DEPRECATED notice; retained only for reference.)
        import random
        self.random_seed = int(getattr(config, 'random_seed', 42))  # dimensionless
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        self.config = config
        
        # Create grid (remove force-based optimization)
        self.grid = Grid(
            width=config.grid_size[0],
            height=config.grid_size[1],
            config=config
        )
        
        # Disable continuous biological adaptation in grid
        self.grid.biological_optimization_enabled = False
        
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
        self.event_detector = EventDetector(config)
        self.configuration_manager = ConfigurationManager(self.grid, config)
        self.transition_controller = TransitionController(
            self.grid, 
            temporal_model=self.models.get('temporal', None)
        )
        
        # Give configuration manager access to simulator for current conditions
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
        self.min_reconfiguration_interval = 5.0  # Minimum 30 minutes between reconfigurations
        
        # Configuration tracking
        self.configuration_history = []
        self.current_configuration_id = 0
        
    def initialize(self, cell_count=None):
        """
        Initialize with optimal configuration selection.
        Uses multi-configuration approach for best starting state.
        """
        if cell_count is None:
            cell_count = self.config.initial_cell_count
        
        print("🚀 Initializing event-driven simulator with optimal configuration...")
        
        # Use existing multi-configuration initialization
        config_results = self.initialize_with_multiple_configurations(
            cell_count=cell_count,
            num_configurations=getattr(self.config, 'multi_config_count', 10),
            optimization_iterations=getattr(self.config, 'multi_config_optimization_steps', 3)
        )
        
        # Store configuration results
        self._config_results = config_results
        
        # Initialize cell properties for current pressure
        self._initialize_cell_properties_for_pressure()
        
        # Initialize event detector state
        self.event_detector.last_pressure = self.input_pattern['value']
        self.event_detector.last_hole_count = len(self.grid.hole_manager.holes) if self.grid.hole_manager else 0
        cell_counts = self.grid.count_cells_by_type()
        self.event_detector.last_senescent_count = cell_counts['telomere_senescent'] + cell_counts['stress_senescent']
        self.event_detector.last_cell_count = cell_counts['total']
        
        # Record initial state
        self._record_state()
        
        print("✅ Event-driven initialization complete")
        
        return config_results
    
    def step(self):
        """
        Advance simulation by one time step with event-driven logic.
        """
        dt = self.config.time_step
        
        # Update input value
        current_input = self.update_input_value()
        
        # 1. Detect events that might trigger reconfiguration
        events = self.event_detector.detect_events(self)
        
        # 2. Process events and trigger reconfigurations if needed
        for event in events:
            self._process_event(event)
        
        # 3. Update ongoing transition (if any)
        transition_complete = self.transition_controller.update_transition(self.time, dt)
        
        # 4. Apply shear stress to cells
        self._apply_shear_stress(current_input, dt)
        
        # 5. Update models (population dynamics, but not continuous spatial adaptation)
        self._update_models_event_driven(current_input, dt)
        
        # 6. Update hole system
        self.grid.update_holes(dt)
        
        # 7. Minimal tessellation maintenance (no force-based optimization)
        #if self.step_count % 10 == 0:  # Every 10 steps
        #    self.grid._update_voronoi_tessellation()
        self.grid._update_voronoi_tessellation()
        # Update time and step count
        self.time += dt
        self.step_count += 1
        
        # Record state periodically
        if self.step_count % self.config.plot_interval == 0:
            self._record_state()
        
        return {
            'time': self.time,
            'step_count': self.step_count,
            'input_value': current_input,
            'cell_count': len(self.grid.cells),
            'transitioning': self.transition_controller.is_transitioning(),
            'transition_progress': self.transition_controller.get_transition_progress(),
            'events_processed': len(events)
        }
    
    def _process_event(self, event: 'ConfigurationEvent'):
        """
        Process an event and decide whether to trigger reconfiguration.
        """
        print(f"📅 Processing event: {event}")
        
        # Add to event history
        self.event_history.append(event)
        
        # Check if we should trigger reconfiguration
        should_reconfigure = self._should_trigger_reconfiguration(event)
        
        if should_reconfigure:
            self._trigger_reconfiguration(event)
        else:
            print(f"   Event not significant enough for reconfiguration")
    
    def _should_trigger_reconfiguration(self, event: 'ConfigurationEvent') -> bool:
        """
        Determine if an event should trigger reconfiguration.
        """
        # Don't reconfigure if already transitioning
        if self.transition_controller.is_transitioning():
            print(f"   Skipping reconfiguration - transition in progress")
            return False
        
        # Don't reconfigure too frequently
        time_since_last = self.time - self.last_reconfiguration_time
        if time_since_last < self.min_reconfiguration_interval:
            print(f"   Too soon since last reconfiguration ({time_since_last:.1f} < {self.min_reconfiguration_interval:.1f} min)")
            return False
        
        # Event-specific criteria
        if event.event_type.value == 'pressure_change':
            pressure_change = abs(event.data['pressure_change'])
            return pressure_change >= 0.2  # Significant pressure change
        
        elif event.event_type.value in ['hole_created', 'hole_filled']:
            return True  # Always reconfigure for holes
        
        elif event.event_type.value == 'senescence_event':
            new_senescent = event.data['new_senescent_cells']
            return new_senescent >= 3  # Multiple cells became senescent
        
        elif event.event_type.value in ['division_event', 'death_event']:
            cell_change = abs(event.data['cell_change'])
            return cell_change >= 5  # Significant population change
        
        return False
    
    def _trigger_reconfiguration(self, event: 'ConfigurationEvent'):
        """
        Trigger a reconfiguration in response to an event.
        """
        print(f"🔄 Triggering reconfiguration for {event.event_type.value}")
        
        # Generate new configuration candidates
        reconfiguration_result = self.configuration_manager.generate_reconfiguration(
            event=event,
            num_configurations=self.config.multi_config_count,
            optimization_iterations=3
        )
        
        # Check if reconfiguration is beneficial
        energy_improvement = reconfiguration_result['energy_improvement']
        
        if energy_improvement > 0.001:  # Meaningful improvement
            print(f"✅ Beneficial reconfiguration found (ΔE = {energy_improvement:.4f})")
            
            # Start transition to new configuration
            self.transition_controller.start_transition(reconfiguration_result, self.time)
            
            # Update reconfiguration timing
            self.last_reconfiguration_time = self.time
            
            # Store configuration info
            self.configuration_history.append({
                'id': self.current_configuration_id,
                'time': self.time,
                'event': event,
                'result': reconfiguration_result
            })
            self.current_configuration_id += 1
            
        else:
            print(f"❌ No beneficial reconfiguration found (ΔE = {energy_improvement:.4f})")
    
    def _update_models_event_driven(self, current_input, dt):
        """
        Update models with event-driven approach (no continuous spatial adaptation).
        """
        # Update temporal dynamics (biochemical responses)
        if 'temporal' in self.models and self.config.enable_temporal_dynamics:
            model = self.models['temporal']
            model.update_cell_responses(self.grid.cells, current_input, dt)
        
        # Update spatial properties ONLY during transitions
        if 'spatial' in self.models and self.config.enable_spatial_properties:
            if self.transition_controller.is_transitioning():
                # During transition, spatial targets are managed by TransitionController
                pass
            else:
                # Outside transitions, only update if targets need recalculation
                # (This happens during reconfiguration generation)
                pass
        
        # Update population dynamics (unchanged)
        if 'population' in self.models and self.config.enable_population_dynamics:
            model = self.models['population']
            stem_cell_rate = 10 if self.config.enable_stem_cells else 0
            model.update_from_cells(self.grid.cells, dt, current_input, stem_cell_rate)
            actions = model.synchronize_cells(self.grid.cells)
            self._execute_population_actions(actions)
    
    def run(self, duration=None):
        """
        Run the event-driven simulation.
        """
        if duration is None:
            duration = self.config.simulation_duration
        
        num_steps = int(duration / self.config.time_step)
        
        print(f"🚀 Running event-driven simulation for {duration} time units ({num_steps} steps)...")
        start_time = time.time()
        
        # Run simulation steps
        for i in range(num_steps):
            step_info = self.step()
            
            # Print progress periodically
            if (i + 1) % 100 == 0 or i == num_steps - 1:
                progress = (i + 1) / num_steps * 100
                elapsed = time.time() - start_time
                estimated_total = elapsed / (i + 1) * num_steps
                remaining = estimated_total - elapsed
                
                transition_info = ""
                if step_info['transitioning']:
                    transition_info = f", Transition: {step_info['transition_progress']:.1%}"
                
                print(f"Progress: {progress:.1f}% (Step {i + 1}/{num_steps}), "
                      f"Time: {elapsed:.1f}s, Remaining: {remaining:.1f}s, "
                      f"Cells: {step_info['cell_count']}, "
                      f"Events: {step_info['events_processed']}{transition_info}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"✅ Event-driven simulation completed in {total_time:.1f} seconds")
        print(f"📊 Events processed: {len(self.event_history)}")
        print(f"🔄 Reconfigurations: {len(self.configuration_history)}")
        
        return {
            'duration': duration,
            'steps': num_steps,
            'final_time': self.time,
            'execution_time': total_time,
            'history': self.history,
            'event_history': self.event_history,
            'configuration_history': self.configuration_history,
            'final_grid_stats': self.grid.get_grid_statistics()
        }
    
    def get_simulation_summary(self) -> Dict:
        """Get a comprehensive summary of the event-driven simulation."""
        return {
            'total_events': len(self.event_history),
            'reconfigurations': len(self.configuration_history),
            'current_transition': self.transition_controller.get_transition_info(),
            'event_types': {
                event_type.value: len([e for e in self.event_history if e.event_type == event_type])
                for event_type in EventType
            },
            'final_cell_count': len(self.grid.cells),
            'final_energy': self.grid.calculate_biological_energy(),
            'simulation_time': self.time
        }
    
    # Keep existing methods for compatibility
    def update_input_value(self):
        """Update the current input value (unchanged from original)."""
        pattern_type = self.input_pattern['type']
        params = self.input_pattern['params']

        if pattern_type == 'constant':
            self.input_pattern['value'] = params['value']

        elif pattern_type == 'step':
            if self.time < params['step_time']:
                self.input_pattern['value'] = params['initial_value']
            else:
                self.input_pattern['value'] = params['final_value']

        elif pattern_type == 'multi_step':
            schedule = params['schedule']
            current_value = schedule[0][1]

            for i, (step_time, step_value) in enumerate(schedule):
                if self.time >= step_time:
                    current_value = step_value
                else:
                    break

            if self.input_pattern['value'] != current_value:
                old_value = self.input_pattern['value']
                self.input_pattern['value'] = current_value
                print(f"Step change at t={self.time:.1f}min: {old_value:.2f} → {current_value:.2f} Pa")

        return self.input_pattern['value']
    
    def set_step_input(self, initial_value, final_value, step_time):
        """Set a step input pattern (unchanged)."""
        self.input_pattern = {
            'type': 'step',
            'value': initial_value,
            'params': {
                'initial_value': initial_value,
                'final_value': final_value,
                'step_time': step_time
            }
        }
    
    def _apply_shear_stress(self, shear_stress, duration):
        """Apply shear stress to all cells (unchanged)."""
        def shear_stress_function(x, y):
            return shear_stress
        self.grid.apply_shear_stress_field(shear_stress_function, duration)
    
    def _record_state(self):
        """Record simulation state (enhanced with event-driven info)."""
        # Use existing state recording but add event-driven information
        # [Keep the existing _record_state implementation but add:]
        
        state = {
            'time': self.time,
            'step_count': self.step_count,
            'input_value': self.input_pattern['value'],
            'cells': len(self.grid.cells),
            'transitioning': self.transition_controller.is_transitioning(),
            'transition_progress': self.transition_controller.get_transition_progress(),
            'events_count': len(self.event_history),
            'reconfigurations_count': len(self.configuration_history)
        }
        
        # Add all the existing state recording logic here...
        # [This would include the full _record_state method from the original simulator]
        
        self.history.append(state)
    
    # Delegate other methods to maintain compatibility
    def initialize_with_multiple_configurations(self, cell_count, num_configurations, optimization_iterations):
        """Use the original grid method for initial configuration."""
        # Calculate base area per cell
        total_area = self.grid.width * self.grid.height
        base_area_per_cell = total_area / cell_count

        def area_distribution():
            return np.random.uniform(base_area_per_cell * 0.7, base_area_per_cell * 1.3)

        def division_distribution():
            max_div = self.config.max_divisions
            r = np.random.random()
            return int(max_div * (1 - np.sqrt(r)))

        return self.grid.generate_multiple_initial_configurations(
            cell_count=cell_count,
            num_configurations=num_configurations,
            division_distribution=division_distribution,
            area_distribution=area_distribution,
            optimization_iterations=optimization_iterations,
            verbose=True
        )
    
    def _initialize_cell_properties_for_pressure(self):
        """Initialize cell properties for current pressure (unchanged)."""
        current_pressure = self.input_pattern.get('value', 0.0)
        
        if 'spatial' not in self.models:
            return
        
        spatial_model = self.models['spatial']
        
        for cell_id, cell in self.grid.cells.items():
            target_area = spatial_model.calculate_target_area(
                current_pressure, cell.is_senescent, cell.senescence_cause  # was undefined `pressure`; the local is current_pressure
            )
            target_aspect_ratio = spatial_model.calculate_target_aspect_ratio(
                current_pressure, cell.is_senescent
            )
            target_orientation = spatial_model.calculate_target_orientation(
                current_pressure, cell.is_senescent
            )
            
            cell.target_area = target_area
            cell.target_aspect_ratio = target_aspect_ratio
            cell.target_orientation = target_orientation
            cell.actual_aspect_ratio = target_aspect_ratio
            cell.actual_orientation = target_orientation
        
        self.grid._update_voronoi_tessellation()
    
    def _execute_population_actions(self, actions):
        """Execute population actions (unchanged from original)."""
        # Keep the existing implementation
        pass
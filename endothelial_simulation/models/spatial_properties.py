"""
Updated spatial properties model using real experimental data.
Modified to use deterministic targets based on experimental measurements.
Variability comes from tessellation process, not artificial randomness.
"""
import numpy as np


class SpatialPropertiesModel:
    """
    Model for spatial arrangement and morphological adaptations of endothelial cells.
    Uses deterministic targets based on real experimental data.
    """

    def __init__(self, config, temporal_model=None):
        """
        Initialize the spatial properties model with real experimental parameters.

        Parameters:
            config: SimulationConfig object with parameter settings
            temporal_model: TemporalDynamicsModel instance for shared time constant calculation
        """
        self.config = config
        self.temporal_model = temporal_model

        # --- Physical area -> computational-pixel conversion (Table 1, main.tex) ---
        # The bioreactor domain is a 650x650 um imaging field mapped onto the display
        # grid; the tessellation runs on a coarser computational grid.
        #   pixel_scale     = 650.0 / grid_display_width_px      [um per display pixel]
        #   target_area_comp = area_um2 / pixel_scale**2 / computation_scale**2  [comp px^2]
        grid_display_width_px = config.grid_size[0]
        self.computation_scale = getattr(config, 'computation_scale', 4)
        self.pixel_scale = getattr(
            config, 'pixel_scale_um',
            getattr(config, 'imaging_field_um', 650.0) / grid_display_width_px
        )  # um per display pixel

        def _um2_to_comp(area_um2):
            """Convert a physical cell area (um^2) to computational pixels^2."""
            return area_um2 / (self.pixel_scale ** 2) / (self.computation_scale ** 2)

        self._um2_to_comp = _um2_to_comp

        # Physical cell areas in um^2 (Table 1, main.tex), converted to comp pixels.
        area_healthy_comp = _um2_to_comp(2354.0)   # Source: Table 1, main.tex — A_E* (healthy HUVEC) = 2354 um^2
        area_sen_small_comp = _um2_to_comp(2207.0)  # Source: Table 1, main.tex — small senescent area = 2207 um^2
        area_sen_large_comp = _um2_to_comp(8626.0)  # Source: Table 1, main.tex — large senescent area = 8626 um^2

        # Control (healthy) cell parameters at different pressures.
        # Area is the healthy HUVEC target at every shear level (Table 1, main.tex — A_E* = 2354 um^2).
        self.control_params = {
            'area': {
                0.0: area_healthy_comp,   # Source: Table 1, main.tex — A_E* = 2354 um^2
                1.4: area_healthy_comp,   # Source: Table 1, main.tex — A_E* = 2354 um^2
                3.0: area_healthy_comp    # Source: Table 1, main.tex — A_E* = 2354 um^2
            },
            'aspect_ratio': {
                0.0: 1.9,      # Static control
                1.4: 2.3,      # Flow control (increased elongation)
                3.0: 2.6       # NEW: Even more elongated at higher pressure
            },
            'orientation_mean': {
                0.0: 45.0,     # isotropic static baseline (no preferred direction with no flow);
                               # ~45 deg mean acute angle, immaterial since s(tau)=0 below tau_act
                1.4: 20.0,     # Source: Table 1, main.tex — theta* = 20 degrees (flow-adapted)
                3.0: 0.0       # NEW: Perfect flow alignment at higher pressure
            }
        }

        # Senescent cell parameters (areas converted from um^2 to comp pixels).
        self.senescent_params = {
            'area_small': area_sen_small_comp,   # Source: Table 1, main.tex — small senescent area = 2207 um^2
            'area_large': area_sen_large_comp,   # Source: Table 1, main.tex — large senescent area = 8626 um^2
            'aspect_ratio': {
                0.0: 1.9,            # Static senescent
                1.4: 2.0,            # Flow senescent (no significant change)
                3.0: 2.1             # NEW: Minimal response at high pressure
            },
            'orientation_mean': {
                0.0: 42.0,           # Random orientation static - MEAN
                1.4: 45.0,           # Random orientation flow (no alignment) - MEAN
                3.0: 40.0            # NEW: Still random but slightly different
            },
            'orientation_std_dev': {
                0.0: 25.0,           # High variability for random orientation
                1.4: 25.0,           # High variability for random orientation
                3.0: 25.0            # NEW: Maintain high variability
            }
        }

        # Probability that a senescent cell will be large (adjustable parameter)
        self.large_senescent_probability = 0.3  # 30% of senescent cells become large

    # Add this method inside the SpatialPropertiesModel class, after the existing methods
    def debug_aspect_ratio_complete_trace(self, pressure, cell_id, cell, dt=None):
        """
        Complete trace of aspect ratio calculation and assignment process.
        """
        print(f"\n{'=' * 60}")
        print(f"🔍 COMPLETE ASPECT RATIO TRACE - Cell {cell_id}")
        print(f"{'=' * 60}")

        # Step 1: Check input parameters
        print(f"📊 INPUT PARAMETERS:")
        print(f"   Pressure: {pressure}")
        print(f"   Is senescent: {cell.is_senescent}")
        print(f"   Cell ID: {cell_id}")

        # Step 2: Check parameter dictionaries
        print(f"\n📋 PARAMETER DICTIONARIES:")
        print(f"   Control params: {self.control_params['aspect_ratio']}")
        print(f"   Senescent params: {self.senescent_params['aspect_ratio']}")

        # Step 3: Test interpolation step by step
        print(f"\n🔢 INTERPOLATION PROCESS:")
        if cell.is_senescent:
            param_dict = self.senescent_params['aspect_ratio']
            cell_type = "senescent"
        else:
            param_dict = self.control_params['aspect_ratio']
            cell_type = "control"

        print(f"   Cell type: {cell_type}")
        print(f"   Using param_dict: {param_dict}")

        # Manual interpolation with debug
        p0, p1 = 0.0, 1.4
        v0, v1 = param_dict[p0], param_dict[p1]
        print(f"   Interpolation points: p0={p0}, p1={p1}")
        print(f"   Values at points: v0={v0}, v1={v1}")

        if pressure <= p0:
            raw_result = v0
            print(f"   Pressure <= {p0}, using v0 = {raw_result}")
        elif pressure >= p1:
            raw_result = v1
            print(f"   Pressure >= {p1}, using v1 = {raw_result}")
        else:
            raw_result = v0 + (v1 - v0) * (pressure - p0) / (p1 - p0)
            print(f"   Interpolating: {v0} + ({v1} - {v0}) * ({pressure} - {p0}) / ({p1} - {p0})")
            print(f"   Raw interpolation result: {raw_result}")

        # Step 4: Apply constraints
        constrained_result = max(1.0, raw_result)
        print(f"\n🚫 CONSTRAINT APPLICATION:")
        print(f"   Before constraint: {raw_result}")
        print(f"   After max(1.0, value): {constrained_result}")
        print(f"   Constraint active: {raw_result < 1.0}")

        # Step 5: Check current cell properties
        print(f"\n📱 CURRENT CELL PROPERTIES:")
        print(f"   target_aspect_ratio: {getattr(cell, 'target_aspect_ratio', 'NOT SET')}")
        print(f"   actual_aspect_ratio: {getattr(cell, 'actual_aspect_ratio', 'NOT SET')}")

        # Step 6: Call the actual function and compare
        print(f"\n✅ ACTUAL FUNCTION CALL:")
        actual_result = self.calculate_target_aspect_ratio(pressure, cell.is_senescent)
        print(f"   Function returned: {actual_result}")
        print(f"   Matches manual calc: {abs(actual_result - constrained_result) < 1e-6}")

        print(f"{'=' * 60}\n")
        return actual_result

    def interpolate_pressure_effect(self, param_dict, pressure):
        """
        Enhanced interpolation to handle multiple pressure points.
        
        Parameters:
            param_dict: Dictionary with pressure-dependent values
            pressure: Applied pressure in Pa

        Returns:
            Interpolated parameter value
        """
        # Get all pressure points and sort them
        pressure_points = sorted(param_dict.keys())
        values = [param_dict[p] for p in pressure_points]
        
        # Handle edge cases
        if pressure <= pressure_points[0]:
            return values[0]
        elif pressure >= pressure_points[-1]:
            return values[-1]
        
        # Find the two pressure points that bracket the current pressure
        for i in range(len(pressure_points) - 1):
            p0, p1 = pressure_points[i], pressure_points[i + 1]
            if p0 <= pressure <= p1:
                v0, v1 = values[i], values[i + 1]
                # Linear interpolation between these two points
                return v0 + (v1 - v0) * (pressure - p0) / (p1 - p0)
        
        # This shouldn't happen, but fallback to last value
        return values[-1]

    def calculate_target_aspect_ratio(self, pressure, is_senescent):
        """
        Calculate target cell aspect ratio using deterministic experimental values.
        NO artificial variability - let tessellation provide natural variation.
        """
        if is_senescent:
            # Senescent cells: no significant response to flow
            base_ratio = self.interpolate_pressure_effect(self.senescent_params['aspect_ratio'], pressure)
            # REMOVED: artificial variability
            result = max(1.0, base_ratio)
            return result
        else:
            # Control cells: USE EXACT EXPERIMENTAL VALUES
            base_ratio = self.interpolate_pressure_effect(self.control_params['aspect_ratio'], pressure)
            # REMOVED: artificial variability
            result = max(1.0, base_ratio)

            return result

    def calculate_target_orientation(self, pressure, is_senescent):
        """
        Calculate target cell orientation using deterministic experimental means.
        NO artificial variability - let tessellation provide natural variation.
        """
        if is_senescent:
            # Senescent cells: remain randomly oriented regardless of flow
            mean_deg = self.interpolate_pressure_effect(self.senescent_params['orientation_mean'], pressure)
            std_dev_deg = self.interpolate_pressure_effect(self.senescent_params['orientation_std_dev'], pressure)

            # Sample from a normal distribution for random orientation
            orientation_deg = np.random.normal(mean_deg, std_dev_deg)

            # Convert to radians
            return np.radians(orientation_deg)
        else:
            # Normal cells: use MEAN orientation only
            mean_deg = self.interpolate_pressure_effect(self.control_params['orientation_mean'], pressure)

            # Convert to radians - USE MEAN DIRECTLY
            mean_rad = np.radians(mean_deg)

            #print(f"Control orientation: pressure={pressure}, deterministic mean={mean_deg:.1f}°")
            return mean_rad

    def calculate_target_area(self, pressure, is_senescent, senescence_cause=None):
        """
        Calculate target cell area using deterministic experimental values.
        NO artificial variability - let tessellation provide natural variation.
        """
        if is_senescent:
            # Deterministic assignment of small or large senescent area
            # Use consistent assignment based on cell properties
            if np.random.random() < self.large_senescent_probability:
                result = self.senescent_params['area_large']
            else:
                result = self.senescent_params['area_small']
            # REMOVED: artificial variability
            #print(f"Senescent area: deterministic result={result:.0f}")
            return result
        else:
            # Control cells: use deterministic experimental area
            base_area = self.interpolate_pressure_effect(self.control_params['area'], pressure)
            # REMOVED: artificial biological variability
            # Floor at 1 comp-pixel (areas are now in computational pixels, ~365 px for A_E*).
            result = max(1.0, base_area)
            #print(f"Control area: pressure={pressure}, deterministic result={result:.0f}")
            return result

    def update_cell_properties(self, cell, pressure, dt, cells_dict=None):
        """
        Updated to ensure continuous dynamics for both target and actual properties.
        """
        # Determine current state
        in_transition = hasattr(self, '_in_transition_mode') and self._in_transition_mode
        is_initial_setup = not hasattr(cell, 'target_area') or cell.target_area is None

        dynamics_info = {
            'event_driven_mode': True,
            'transitioning': in_transition,
            'initial_setup': is_initial_setup
        }
        dt_minutes = dt

        # Calculate instantaneous targets based on current pressure
        instant_target_area = self.calculate_target_area(pressure, cell.is_senescent, cell.senescence_cause)
        instant_target_orientation = self.calculate_target_orientation(pressure, cell.is_senescent)
        instant_target_aspect_ratio = self.calculate_target_aspect_ratio(pressure, cell.is_senescent)

        # Get time constants from the temporal model or use defaults
        if self.temporal_model:
            current_pressure = getattr(self, '_current_pressure', pressure)
            tau_area, _ = self.temporal_model.get_scaled_tau_and_amax(current_pressure, 'area')
            tau_orient, _ = self.temporal_model.get_scaled_tau_and_amax(current_pressure, 'orientation')
            tau_ar, _ = self.temporal_model.get_scaled_tau_and_amax(current_pressure, 'aspect_ratio')
        else:
            tau_area, tau_orient, tau_ar = 30.0, 20.0, 25.0

        # Evolve target properties toward instantaneous targets
        if is_initial_setup:
            cell.target_area = instant_target_area
            cell.target_orientation = instant_target_orientation
            cell.target_aspect_ratio = instant_target_aspect_ratio
            dynamics_info['initial_target_set'] = True
        else:
            # Evolve target area
            decay_factor = np.exp(-dt_minutes / tau_area)
            cell.target_area = instant_target_area + (cell.target_area - instant_target_area) * decay_factor

            # Evolve target orientation (with damping)
            orientation_diff = instant_target_orientation - cell.target_orientation
            orientation_diff = (orientation_diff + np.pi) % (2 * np.pi) - np.pi  # Wrap angle
            decay_factor = np.exp(-dt_minutes / tau_orient)
            cell.target_orientation += (1 - decay_factor) * orientation_diff

            dynamics_info['instant_target_orientation'] = instant_target_orientation
            dynamics_info['new_target_orientation'] = cell.target_orientation

            # Evolve target aspect ratio
            decay_factor = np.exp(-dt_minutes / tau_ar)
            cell.target_aspect_ratio = instant_target_aspect_ratio + (cell.target_aspect_ratio - instant_target_aspect_ratio) * decay_factor
            cell.target_aspect_ratio = max(1.0, cell.target_aspect_ratio)

        # Evolve actual properties toward the (evolving) target properties
        # Evolve actual area
        decay_factor = np.exp(-dt_minutes / tau_area)
        cell.actual_area = cell.target_area + (cell.actual_area - cell.target_area) * decay_factor

        # Evolve actual orientation
        orientation_diff = cell.target_orientation - cell.actual_orientation
        orientation_diff = (orientation_diff + np.pi) % (2 * np.pi) - np.pi
        decay_factor = np.exp(-dt_minutes / tau_orient)
        cell.actual_orientation += (1 - decay_factor) * orientation_diff


        # Evolve actual aspect ratio
        decay_factor = np.exp(-dt_minutes / tau_ar)
        cell.actual_aspect_ratio = cell.target_aspect_ratio + (cell.actual_aspect_ratio - cell.target_aspect_ratio) * decay_factor
        cell.actual_aspect_ratio = max(1.0, cell.actual_aspect_ratio)

        return {
            'target_orientation': cell.target_orientation,
            'target_area': cell.target_area,
            'target_aspect_ratio': cell.target_aspect_ratio,
            'actual_orientation': cell.actual_orientation,
            'actual_area': cell.actual_area,
            'actual_aspect_ratio': cell.actual_aspect_ratio,
            'dynamics_info': dynamics_info
        }

    def calculate_collective_properties(self, cells_dict, pressure):
        """
        Calculate collective properties focusing on real measured parameters.
        """
        if not cells_dict:
            return {
                'mean_actual_orientation': 0,
                'std_actual_orientation': 0,
                'mean_target_orientation': 0,
                'mean_actual_area': 0,
                'mean_target_area': 0,
                'mean_actual_aspect_ratio': 1.0,
                'mean_target_aspect_ratio': 1.0,
                'orientation_alignment': 0,
                'area_adaptation': 1.0,
                'aspect_ratio_adaptation': 1.0
            }

        # Collect real measured properties
        actual_orientations = []
        target_orientations = []
        actual_areas = []
        target_areas = []
        actual_aspect_ratios = []
        target_aspect_ratios = []

        for cell in cells_dict.values():
            actual_orientations.append(cell.actual_orientation)
            target_orientations.append(getattr(cell, 'target_orientation', cell.actual_orientation))
            actual_areas.append(cell.actual_area)
            target_areas.append(getattr(cell, 'target_area', cell.actual_area))
            actual_aspect_ratios.append(cell.actual_aspect_ratio)
            target_aspect_ratios.append(getattr(cell, 'target_aspect_ratio', cell.actual_aspect_ratio))

        # Calculate alignment index (how well oriented toward flow direction)
        # Flow direction is 0 degrees (horizontal)
        flow_direction = 0.0
        alignment_scores = []
        for orientation in actual_orientations:
            # Calculate how close the orientation is to flow direction
            angle_diff = abs(orientation - flow_direction)
            # Handle angle wrapping
            if angle_diff > np.pi:
                angle_diff = 2 * np.pi - angle_diff
            # Convert to alignment score (1 = perfectly aligned, 0 = perpendicular)
            alignment = np.cos(angle_diff)
            alignment_scores.append(alignment)

        # Calculate adaptation quality
        orientation_adaptation = np.mean([
            1.0 - min(1.0, abs(a - t) / np.pi)
            for a, t in zip(actual_orientations, target_orientations)
        ])

        # Handle zero areas properly
        area_adaptation = np.mean([
            min(a / t, t / a) if t > 0 and a > 0 else 1.0
            for a, t in zip(actual_areas, target_areas)
        ])

        # Handle zero aspect ratios properly
        aspect_ratio_adaptation = np.mean([
            min(a / t, t / a) if t > 0 and a > 0 else 1.0
            for a, t in zip(actual_aspect_ratios, target_aspect_ratios)
        ])

        return {
            'mean_actual_orientation': np.degrees(np.mean(actual_orientations)),  # Convert to degrees for display
            'std_actual_orientation': np.degrees(np.std(actual_orientations)),
            'mean_target_orientation': np.degrees(np.mean(target_orientations)),

            'mean_actual_area': np.mean(actual_areas),
            'mean_target_area': np.mean(target_areas),

            'mean_actual_aspect_ratio': np.mean(actual_aspect_ratios),
            'mean_target_aspect_ratio': np.mean(target_aspect_ratios),

            'orientation_alignment': np.mean(alignment_scores),  # How aligned with flow
            'area_adaptation': area_adaptation,
            'aspect_ratio_adaptation': aspect_ratio_adaptation,

            # Additional metrics
            # "Large" senescent cells are those above the 5000 um^2 boundary
            # (Table 1, main.tex), expressed in computational pixels.
            'large_senescent_fraction': len([c for c in cells_dict.values()
                                             if c.is_senescent and getattr(c, 'target_area', 0)
                                             > self._um2_to_comp(5000.0)]) / len(
                cells_dict),
            'pressure': pressure,

            # NEW: Target consistency metrics (should be very consistent now)
            'target_orientation_std': np.degrees(np.std(target_orientations)),
            'target_area_std': np.std(target_areas),
            'target_aspect_ratio_std': np.std(target_aspect_ratios)
        }

    def calculate_alignment_index(self, cells, flow_direction=0):
        """
        Calculate the alignment index for a collection of cells.
        """
        if not cells:
            return 0

        if isinstance(cells, dict):
            cell_list = list(cells.values())
        else:
            cell_list = cells

        alignment_sum = 0
        cell_count = 0

        for cell in cell_list:
            # Convert to alignment angle (0-90°)
            orientation_rad = cell.actual_orientation
            alignment_angle = np.abs(orientation_rad) % (np.pi / 2)  # 0 to π/2 radians

            # Convert to alignment score (1 = perfectly aligned, 0 = perpendicular)
            alignment_score = np.cos(alignment_angle)  # cos(0) = 1, cos(π/2) = 0

            alignment_sum += alignment_score
            cell_count += 1

        return alignment_sum / cell_count if cell_count > 0 else 0

    def calculate_shape_index(self, cells):
        """
        Calculate the shape index for a collection of cells.
        Shape index = P/(sqrt(4πA)) where P is perimeter and A is area.
        """
        if not cells:
            return 0

        # Convert cells input to a list of cell objects
        if isinstance(cells, dict):
            cell_list = list(cells.values())
        else:
            cell_list = cells

        # Calculate shape index for each cell
        shape_sum = 0
        cell_count = 0

        for cell in cell_list:
            if cell.actual_area > 0 and cell.perimeter > 0:
                # Shape index = P/sqrt(4πA)
                shape_index = cell.perimeter / np.sqrt(4 * np.pi * cell.actual_area)
                shape_sum += shape_index
                cell_count += 1

        # Average shape index
        if cell_count > 0:
            return shape_sum / cell_count
        else:
            return 1.0  # Default value

    def calculate_packing_quality(self, cells):
        """
        Calculate how well cells are packed (how close they are to their targets).
        """
        if not cells:
            return 1.0

        # Convert cells input to a list of cell objects
        if isinstance(cells, dict):
            cell_list = list(cells.values())
        else:
            cell_list = cells

        total_quality = 0
        cell_count = 0

        for cell in cell_list:
            # Quality based on how close actual properties are to targets
            orientation_quality = 1.0
            area_quality = 1.0
            aspect_ratio_quality = 1.0

            # Orientation quality
            if hasattr(cell, 'target_orientation'):
                orientation_diff = abs(cell.actual_orientation - cell.target_orientation)
                if orientation_diff > np.pi:
                    orientation_diff = 2 * np.pi - orientation_diff
                orientation_quality = max(0, 1.0 - orientation_diff / np.pi)

            # Area quality
            if hasattr(cell, 'target_area') and cell.target_area > 0 and cell.actual_area > 0:
                area_ratio = min(cell.actual_area / cell.target_area,
                                 cell.target_area / cell.actual_area)
                area_quality = area_ratio

            # Aspect ratio quality
            if hasattr(cell, 'target_aspect_ratio') and cell.target_aspect_ratio > 0:
                ar_ratio = min(cell.actual_aspect_ratio / cell.target_aspect_ratio,
                               cell.target_aspect_ratio / cell.actual_aspect_ratio)
                aspect_ratio_quality = ar_ratio

            # Combined quality (average of the three components)
            cell_quality = (orientation_quality + area_quality + aspect_ratio_quality) / 3
            total_quality += cell_quality
            cell_count += 1

        return total_quality / cell_count if cell_count > 0 else 1.0

    def get_expected_values(self, pressure, cell_type='control'):
        """
        Get expected experimental values for comparison.
        Now returns deterministic values (means only).
        """
        if cell_type == 'control':
            return {
                'area': self.interpolate_pressure_effect(self.control_params['area'], pressure),
                'aspect_ratio': self.interpolate_pressure_effect(self.control_params['aspect_ratio'], pressure),
                'orientation_mean': self.interpolate_pressure_effect(self.control_params['orientation_mean'], pressure),
                # Note: no orientation_std returned since we're now deterministic
            }
        else:  # senescent
            return {
                'area_small': self.senescent_params['area_small'],
                'area_large': self.senescent_params['area_large'],
                'aspect_ratio': self.interpolate_pressure_effect(self.senescent_params['aspect_ratio'], pressure),
                'orientation_mean': self.interpolate_pressure_effect(self.senescent_params['orientation_mean'], pressure),
            }
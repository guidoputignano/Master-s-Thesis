"""
Core module for the Cell class that represents a single endothelial cell with territory management.
Optimized version for better performance.
"""
import numpy as np
from endothelial_simulation import config


class Cell:
    """
    Class representing a single endothelial cell in the simulation with territory-based properties.
    Optimized for performance.
    """

    def __init__(self, cell_id, position=(0, 0), divisions=0, is_senescent=False, senescence_cause=None, target_area=100.0):
        """
        Initialize a cell with its properties.

        Parameters:
            cell_id: Unique identifier for the cell
            position: (x, y) coordinates of the cell center (seed point)
            divisions: Number of divisions the cell has undergone
            is_senescent: Boolean indicating if the cell is senescent
            senescence_cause: 'telomere' or 'stress' indicating the cause of senescence
            target_area: Target area the cell wants to achieve
        """
        # Basic cell properties
        self.cell_id = cell_id
        self.biological_id = f"bio_{hash((position[0], position[1], cell_id)) % 100000}"
        self.position = position
        self.divisions = divisions
        self.is_senescent = is_senescent
        self.senescence_cause = senescence_cause

        # Territory and morphology properties
        self.target_area = target_area  # Desired area from biological parameters
        # Per-cell debug print disabled (kept only warnings / end-of-step summaries).
        # print(f"🐛 CELL_INIT: Cell {cell_id} target_area = {self.target_area}")
        self.actual_area = 0.0  # Actual area assigned in the mosaic
        self.territory_pixels = []  # List of (x, y) pixel coordinates owned by this cell
        self.boundary_points = []  # Boundary of the cell territory
        self.centroid = position  # Actual centroid of the territory (may differ from seed)

        # Orientation properties
        self.target_orientation = 0.0  # Target orientation from flow/senescence
        self.actual_orientation = 0.0  # Actual orientation of the cell territory
        self.orientation_variability = 0.1  # How much orientation can vary (in radians)

        # Shape adaptation properties
        self.target_aspect_ratio = 1.0  # Target aspect ratio from biological parameters
        self.actual_aspect_ratio = 1.0  # Actual aspect ratio from territory shape
        self.shape_flexibility = 0.3  # How much the cell can deviate from target shape (0-1)

        # Computed properties from territory
        self.perimeter = 0.0

        # Cell state properties
        self.age = 0.0
        self.adhesion_strength = 1.0
        self.response = 1.0

        # Mechanical properties
        self.local_shear_stress = 0.0


        # Growth and adaptation properties
        self.growth_pressure = 0.0  # Pressure to expand beyond current territory
        self.compression_ratio = 1.0  # How compressed the cell is compared to target size

        # Senescent growth
        # Probabilistic senescent growth properties
        self.senescent_growth_factor = 1.0  # Current size multiplier (starts at 1.0)
        self.max_senescent_growth = 4.0  # Maximum size (4x normal)
        self.growth_probability_base = 0.15  # 15% chance per hour to grow
        self.growth_increment = 0.05  # 5% size increase when growth occurs

        # Added temporal
        self.target_orientation = 0.0  # Will be set by spatial model
        self.target_aspect_ratio = 1.0  # Will be set by spatial model
        self.target_area = target_area  # Already exists, but ensure it's set

        # Optional: For monitoring temporal dynamics
        self.last_dynamics_info = {}

        # TELOMERE SYSTEM - Simple and configurable
        self.telomere_length = np.random.normal(100, 20)  # Mean=100, std=20
        self.telomere_length = max(10, self.telomere_length)  # Minimum viable length
        self.telomere_loss_per_division = 14.3  # 100/7 divisions

        # STRESS SYSTEM - Hardcoded for simplicity
        base_resistance = 0.5
        variability = np.random.normal(1.0, 0.2)
        self.cellular_resistance = base_resistance * max(0.5, variability)

        # Stress accumulation tracking
        self.accumulated_stress = 0.0
        self.stress_exposure_time = 0.0

    def calculate_final_stress(self, config, exposure_time=None):
        """
        Calculate final stress based on biologically calibrated stress factor and exposure time.
        Uses your existing _calculate_stress_factor method for proper stress conversion.

        Args:
            config: Simulation configuration
            exposure_time: Time exposed to stress (hours). If None, uses self.stress_exposure_time

        Returns:
            final_stress: Cumulative stress value
        """
        if exposure_time is None:
            exposure_time = self.stress_exposure_time

        # Use your existing calibrated stress factor conversion
        stress_factor = self._calculate_stress_factor(config)

        # Option 1: Linear accumulation model
        final_stress = stress_factor * exposure_time

        return final_stress

    def get_senescence_status(self, config):
        """Get comprehensive senescence risk status."""
        if self.is_senescent:
            return {
                'is_senescent': True,
                'cause': self.senescence_cause,
                'telomere_length': self.telomere_length,
                'stress_ratio': 0
            }

        # Calculate risk factors
        telomere_ratio = self.telomere_length / max(1, self.telomere_loss_per_division)
        divisions_left = int(telomere_ratio)

        stress_ratio = 0
        if self.stress_exposure_time > 0:
            final_stress = self.calculate_final_stress(config)
            stress_ratio = final_stress / self.cellular_resistance

        return {
            'is_senescent': False,
            'telomere_length': self.telomere_length,
            'divisions_left': divisions_left,
            'telomere_at_risk': divisions_left <= 1,
            'stress_ratio': stress_ratio,
            'stress_at_risk': stress_ratio > 0.8,
            'will_senesce_soon': divisions_left <= 1 or stress_ratio >= 1.0
        }

    def update_stress_and_check_senescence(self, dt_hours, config, spatial_model=None, pressure=None):
        """
        Update stress exposure and check for deterministic senescence.
        Call this method each time step.

        Args:
            dt_hours: Time step in hours
            config: Simulation configuration
            spatial_model (SpatialPropertiesModel, optional): The spatial model for property updates.
            pressure (float, optional): The current pressure for property updates.

        Returns:
            bool: True if senescence was triggered
        """
        if self.is_senescent:
            return False

        # Update exposure time if under stress
        if self.local_shear_stress > 0:
            self.stress_exposure_time += dt_hours

            # Calculate current final stress using calibrated stress factor
            final_stress = self.calculate_final_stress(config)

            # Deterministic senescence check
            if final_stress > self.cellular_resistance:
                # CRITICAL FIX: Pass along spatial_model and pressure
                self.induce_senescence(
                    "stress_threshold_exceeded",
                    spatial_model=spatial_model,
                    pressure=pressure
                )
                print(f"🔴 Deterministic senescence triggered!")
                print(f"   Final stress: {final_stress:.6f}")
                print(f"   Cellular resistance: {self.cellular_resistance:.6f}")
                print(f"   Raw shear stress: {self.local_shear_stress:.2f} Pa")
                print(f"   Stress factor: {self._calculate_stress_factor(config):.6f}")
                print(f"   Exposure time: {self.stress_exposure_time:.2f} hours")
                return True

        return False

    def assign_territory(self, pixel_list):
        """
        Assign a list of pixels to this cell's territory.
        Now enforces expansion limits to prevent unrealistic over-expansion.

        Parameters:
            pixel_list: List of (x, y) tuples representing pixels owned by this cell
        """
        # Calculate base target area (without senescent growth factor)
        if self.is_senescent:
            base_target_area = self.target_area / max(1.0, self.senescent_growth_factor)
            max_allowed_area = base_target_area * 4.0  # 400% expansion limit
        else:
            base_target_area = self.target_area
            max_allowed_area = base_target_area * 1.2  # 120% expansion limit

        # Enforce expansion limits
        requested_area = len(pixel_list)
        if requested_area > max_allowed_area:
            # Cap the territory to maximum allowed area
            # Keep pixels closest to cell center/position
            if pixel_list:
                pixels_array = np.array(pixel_list)
                distances = np.linalg.norm(pixels_array - np.array(self.position), axis=1)
                sorted_indices = np.argsort(distances)
                max_pixels = int(max_allowed_area)
                pixel_list = [pixel_list[i] for i in sorted_indices[:max_pixels]]

            print(f"⚠️  Cell {self.cell_id} expansion limited: {requested_area:.0f} → {len(pixel_list):.0f} pixels "
                  f"({'senescent' if self.is_senescent else 'healthy'} limit: {max_allowed_area:.0f})")

        # Assign territory
        self.territory_pixels = pixel_list
        self.actual_area = len(pixel_list)

        if pixel_list:
            # Calculate centroid efficiently
            pixels_array = np.array(pixel_list)
            self.centroid = np.mean(pixels_array, axis=0)

            # Calculate boundary points (optimized)
            self._calculate_boundary_fast()

            # Calculate geometric properties (optimized)
            self._calculate_geometry_fast()

            # Update compression ratio (based on current target, which includes senescent growth)
            self.compression_ratio = self.actual_area / max(1, self.target_area)

            # Calculate growth pressure (higher when compressed)
            if self.compression_ratio < 1.0:
                self.growth_pressure = (1.0 - self.compression_ratio) * 2.0
            else:
                self.growth_pressure = 0.0

    def update_senescent_growth(self, dt_hours):
        """
        Probabilistic growth for senescent cells.
        Now enforces expansion limits based on actual area.

        Parameters:
            dt_hours: Time step in hours

        Returns:
            Boolean indicating if growth occurred
        """
        if not self.is_senescent:
            return False

        # Check expansion limits BEFORE growth factor limits
        base_target_area = self.target_area / max(1.0, self.senescent_growth_factor)
        max_allowed_area = base_target_area * 4.0  # 300% expansion limit

        # If actual area already exceeds expansion limit, prevent further growth
        if self.actual_area >= max_allowed_area:
            # Optionally adjust target_area down to match actual constraint
            constrained_target = min(self.target_area, max_allowed_area)
            if constrained_target < self.target_area:
                self.target_area = constrained_target
                # Recalculate growth factor based on constrained target
                self.senescent_growth_factor = self.target_area / base_target_area
            return False

        # Can't grow beyond maximum growth factor
        if self.senescent_growth_factor >= self.max_senescent_growth:
            return False

        # Calculate growth probability for this time step
        growth_prob = self.growth_probability_base * dt_hours

        # Factors that influence growth probability:
        # 1. Mechanical stress increases growth probability
        stress_factor = 1.0 + 0.1 * self.local_shear_stress

        # 2. Compression increases growth probability (crowded cells try to expand)
        compression_factor = max(1.0, 2.0 - self.compression_ratio)

        # 3. Growth becomes less likely as cell approaches maximum size
        size_factor = (self.max_senescent_growth - self.senescent_growth_factor) / \
                      (self.max_senescent_growth - 1.0)

        # 4. NEW: Growth becomes less likely as actual area approaches expansion limit
        area_factor = max(0.1, (max_allowed_area - self.actual_area) / (max_allowed_area * 0.5))

        # Combined probability
        final_prob = growth_prob * stress_factor * compression_factor * size_factor * area_factor
        final_prob = min(0.3 * dt_hours, final_prob)  # Cap at 30% per hour

        # Random growth check
        if np.random.random() < final_prob:
            # Calculate proposed growth
            growth_increase = self.growth_increment * np.random.uniform(0.8, 1.2)  # Add variability
            proposed_growth_factor = min(self.max_senescent_growth,
                                         self.senescent_growth_factor + growth_increase)

            # Calculate proposed target area
            proposed_target_area = base_target_area * proposed_growth_factor

            # Ensure proposed target doesn't exceed expansion limit
            if proposed_target_area <= max_allowed_area:
                self.senescent_growth_factor = proposed_growth_factor
                self.target_area = proposed_target_area
                return True
            else:
                # Growth would exceed limit - cap at maximum allowed
                self.target_area = max_allowed_area
                self.senescent_growth_factor = max_allowed_area / base_target_area
                return True

        return False

    def check_expansion_compliance(self):
        """
        Check if cell is within expansion limits and return compliance info.

        Returns:
            Dict with compliance information
        """
        if self.is_senescent:
            base_target_area = self.target_area / max(1.0, self.senescent_growth_factor)
            max_allowed_area = base_target_area * 4.0  # 400% limit
            expansion_type = "senescent"
            limit_multiplier = 3.0
        else:
            base_target_area = self.target_area
            max_allowed_area = base_target_area * 1.2  # 120% limit
            expansion_type = "healthy"
            limit_multiplier = 1.2

        expansion_ratio = self.actual_area / base_target_area if base_target_area > 0 else 0
        is_compliant = self.actual_area <= max_allowed_area

        return {
            'is_compliant': is_compliant,
            'expansion_type': expansion_type,
            'limit_multiplier': limit_multiplier,
            'base_target_area': base_target_area,
            'max_allowed_area': max_allowed_area,
            'actual_area': self.actual_area,
            'expansion_ratio': expansion_ratio,
            'overage': max(0, self.actual_area - max_allowed_area)
        }

    def _calculate_boundary_fast(self):
        """Calculate the boundary points of the cell territory - optimized version."""
        if not self.territory_pixels:
            self.boundary_points = []
            self.perimeter = 0
            return

        # For large territories, sample boundary points
        if len(self.territory_pixels) > 1000:
            # Use convex hull for large territories
            try:
                from scipy.spatial import ConvexHull
                points = np.array(self.territory_pixels)
                hull = ConvexHull(points)
                self.boundary_points = points[hull.vertices].tolist()
                self.perimeter = len(self.boundary_points)
                return
            except:
                pass

        # For smaller territories, find actual boundary
        pixels_set = set(self.territory_pixels)
        boundary = []

        # Sample pixels for boundary detection if too many
        pixels_to_check = self.territory_pixels
        if len(pixels_to_check) > 500:
            # Randomly sample pixels for boundary detection
            sample_size = min(500, len(pixels_to_check))
            pixels_to_check = np.random.choice(len(pixels_to_check), sample_size, replace=False)
            pixels_to_check = [self.territory_pixels[i] for i in pixels_to_check]

        # Check each pixel for boundary
        for x, y in pixels_to_check:
            # Check 4-connected neighbors (faster than 8-connected)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (x + dx, y + dy)
                if neighbor not in pixels_set:
                    boundary.append((x, y))
                    break

        self.boundary_points = boundary
        self.perimeter = len(boundary)

    def _calculate_geometry_fast(self):
        """Calculate geometric properties from the territory - real parameters only."""
        if not self.territory_pixels:
            return

        pixels_array = np.array(self.territory_pixels)

        # For very large territories, sample points for PCA
        if len(pixels_array) > 1000:
            sample_size = min(1000, len(pixels_array))
            indices = np.random.choice(len(pixels_array), sample_size, replace=False)
            sample_pixels = pixels_array[indices]
        else:
            sample_pixels = pixels_array

        # Calculate actual orientation using principal component analysis
        if len(sample_pixels) > 1:
            # Center the points
            centered = sample_pixels - self.centroid

            # Calculate covariance matrix efficiently
            if len(centered) > 1:
                try:
                    cov_matrix = np.cov(centered, rowvar=False)
                    eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)

                    # Sort by eigenvalue (largest first)
                    idx = np.argsort(eigenvals)[::-1]
                    eigenvals = eigenvals[idx]
                    eigenvecs = eigenvecs[:, idx]

                    # Principal axis gives orientation
                    principal_axis = eigenvecs[:, 0]
                    self.actual_orientation = np.arctan2(principal_axis[1], principal_axis[0])

                    # Aspect ratio from eigenvalues
                    if eigenvals[1] > 0:
                        self.actual_aspect_ratio = np.sqrt(max(eigenvals[0] / eigenvals[1], 1.0))
                    else:
                        self.actual_aspect_ratio = 1.0

                except:
                    # Fallback to simple calculations
                    self._calculate_geometry_simple()
            else:
                self._calculate_geometry_simple()
        else:
            self._calculate_geometry_simple()


    def _calculate_geometry_simple(self):
        """Simple fallback geometry calculation - real parameters only."""
        self.actual_orientation = self.target_orientation
        self.actual_aspect_ratio = max(1.0, self.target_aspect_ratio)

    def adapt_to_constraints(self, available_space_factor=1.0):
        """
        Adapt cell properties based on space constraints.

        Parameters:
            available_space_factor: Factor indicating how much space is available (0-1)
        """
        # Adjust target area based on available space
        adjusted_target_area = self.target_area * available_space_factor

        # If compressed, try to maintain shape as much as possible
        if self.actual_area < adjusted_target_area:
            # Cell is compressed - increase growth pressure
            self.growth_pressure = (adjusted_target_area - self.actual_area) / adjusted_target_area
        else:
            # Cell has enough or more space
            self.growth_pressure = 0.0

        # Adapt orientation gradually towards target
        orientation_diff = self.target_orientation - self.actual_orientation

        # Handle angle wrapping
        while orientation_diff > np.pi:
            orientation_diff -= 2 * np.pi
        while orientation_diff < -np.pi:
            orientation_diff += 2 * np.pi

        # Add some variability around target orientation
        variability = np.random.normal(0, self.orientation_variability)

        # Gradually adjust actual orientation
        adaptation_rate = 0.1
        self.actual_orientation += adaptation_rate * orientation_diff

    def update_target_properties(self, target_orientation, target_aspect_ratio, target_area):
        """
        ADD this method to your existing Cell class.
        Update target properties based on biological parameters.

        Parameters:
            target_orientation: Target orientation in radians
            target_aspect_ratio: Target aspect ratio
            target_area: Target area in pixels
        """
        self.target_orientation = target_orientation
        self.target_aspect_ratio = target_aspect_ratio
        self.target_area = target_area

    def get_shape_deviation(self):
        """
        Calculate how much the cell deviates from its target shape.

        Returns:
            Dictionary with deviation metrics
        """
        orientation_deviation = abs(self.actual_orientation - self.target_orientation)
        if orientation_deviation > np.pi:
            orientation_deviation = 2 * np.pi - orientation_deviation

        # Protect against division by zero
        if self.target_aspect_ratio > 0:
            aspect_ratio_deviation = abs(self.actual_aspect_ratio - self.target_aspect_ratio) / self.target_aspect_ratio
        else:
            aspect_ratio_deviation = 0

        if self.target_area > 0:
            area_deviation = abs(self.actual_area - self.target_area) / self.target_area
        else:
            area_deviation = 0

        return {
            'orientation_deviation': orientation_deviation,
            'aspect_ratio_deviation': aspect_ratio_deviation,
            'area_deviation': area_deviation,
            'total_deviation': (orientation_deviation + aspect_ratio_deviation + area_deviation) / 3
        }

    def get_territory_info(self):
        """
        Get information about the cell's territory.

        Returns:
            Dictionary with territory information
        """
        return {
            'territory_size': len(self.territory_pixels),
            'actual_area': self.actual_area,
            'target_area': self.target_area,
            'compression_ratio': self.compression_ratio,
            'growth_pressure': self.growth_pressure,
            'centroid': self.centroid,
            'boundary_length': len(self.boundary_points),
            'compactness': self.compactness
        }

    # Keep existing methods for compatibility
    def update_shape(self, orientation, aspect_ratio, area, eccentricity=None, circularity=None):
        """Update target shape properties (for compatibility)."""
        self.target_orientation = orientation
        self.target_aspect_ratio = aspect_ratio
        self.target_area = area
        if eccentricity is not None:
            self.eccentricity = eccentricity
        if circularity is not None:
            self.circularity = circularity

    def update_response(self, new_response):
        """Update the cell's temporal response value."""
        self.response = new_response

    def update_position(self, new_position):
        """Update the cell's seed position."""
        self.position = new_position

    def increment_age(self, time_step):
        """Increment the cell's age by the given time step."""
        self.age += time_step

    def divide(self):
        """Perform cell division with telomere shortening."""
        if self.is_senescent:
            return False

        # Shorten telomeres with each division
        self.telomere_length -= self.telomere_loss_per_division
        self.divisions += 1
        self.age = 0.0

        return True

    def check_deterministic_senescence(self, config):
        """
        Check for all deterministic senescence triggers.

        Returns:
            str or None: Senescence cause if triggered, None otherwise
        """
        if self.is_senescent:
            return None

        # 1. TELOMERE SENESCENCE - Simple threshold check
        if self.telomere_length <= 0:
            return "telomere_exhaustion"

        # 2. STRESS SENESCENCE - Existing system
        if self.stress_exposure_time > 0:
            final_stress = self.calculate_final_stress(config)
            if final_stress > self.cellular_resistance:
                return "stress_threshold_exceeded"

        return None

    def update_and_check_all_senescence(self, dt_hours, config, spatial_model=None, pressure=None):
        """
        Update stress exposure and check all senescence mechanisms.

        Args:
            dt_hours: Time step in hours
            config: Simulation configuration
            spatial_model (SpatialPropertiesModel, optional): The spatial model, needed for property updates.
            pressure (float, optional): The current pressure, needed for property updates.

        Returns:
            bool: True if senescence was triggered
        """
        if self.is_senescent:
            return False

        # Update stress exposure time
        if self.local_shear_stress > 0:
            self.stress_exposure_time += dt_hours

        # Check for senescence
        senescence_cause = self.check_deterministic_senescence(config)

        if senescence_cause:
            # CRITICAL FIX: Pass spatial_model and pressure to induce_senescence
            self.induce_senescence(senescence_cause, spatial_model=spatial_model, pressure=pressure)
            print(f"🔴 {senescence_cause}: Telomere={self.telomere_length:.1f}, Divisions={self.divisions}")
            return True

        return False

    def induce_senescence(self, cause, spatial_model=None, pressure=None):
        """
        Induce senescence and immediately update orientation to reflect the new state.
        This is a critical fix to ensure senescent cells behave correctly right after conversion.

        Args:
            cause (str): The reason for senescence (e.g., 'stress', 'telomere_exhaustion').
            spatial_model (SpatialPropertiesModel, optional): The spatial model needed to recalculate orientation.
            pressure (float, optional): The current pressure, required for recalculation.
        """
        if self.is_senescent:
            return  # Already senescent

        original_biological_id = getattr(self, 'biological_id', 'unknown')
        print(f"🔄 Converting cell {original_biological_id} to senescent state (cause: {cause})...")

        # --- CORE STATE CHANGE ---
        self.is_senescent = True
        self.senescence_cause = cause

        # --- IMMEDIATE PROPERTY UPDATE (CRITICAL FIX) ---
        if spatial_model and pressure is not None:
            print(f"   recalculating target orientation for new senescent cell at pressure={pressure:.2f} Pa.")
            
            # Recalculate the target orientation using senescent parameters
            self.target_orientation = spatial_model.calculate_target_orientation(
                pressure=pressure,
                is_senescent=True  # Now it's senescent
            )
            
            # To ensure the change takes effect immediately, we can also set the actual_orientation.
            # This avoids any delay or gradual adaptation period.
            self.actual_orientation = self.target_orientation
            
            print(f"  New target orientation: {np.degrees(self.target_orientation):.1f}° (randomized for senescence)")
        else:
            print("  ⚠️ spatial_model or pressure not provided. Orientation will update on the next cycle.")

        # Optional: Stop proliferation
        if hasattr(self, 'max_divisions'):
            self.divisions = self.max_divisions

        print(f"✅ Cell {original_biological_id} is now senescent.")


    def apply_shear_stress(self, shear_stress):
        """Apply shear stress to the cell for the given duration."""
        self.local_shear_stress = shear_stress


    def calculate_senescence_probability(self, config):
        """
        Modified method - now only handles telomere-induced senescence.
        Stress-induced senescence is now deterministic.
        """
        if self.is_senescent:
            return {'telomere': 0.0, 'stress': 0.0}

        # Telomere-induced senescence probability (keep existing logic)
        tel_prob = 0.0
        if self.divisions >= config.max_divisions:
            tel_prob = 1.0
        elif self.divisions > 0.7 * config.max_divisions:
            tel_prob = ((self.divisions - 0.7 * config.max_divisions) /
                        (0.3 * config.max_divisions)) * 0.5

        # Stress-induced senescence is now deterministic (handled separately)
        # Return 0 for stress probability since it's no longer probabilistic
        stress_prob = 0.0

        return {'telomere': tel_prob, 'stress': stress_prob}

    def get_stress_status(self, config):
        """
        Get current stress status for monitoring.

        Args:
            config: Simulation configuration

        Returns:
            dict: Current stress information
        """
        if self.stress_exposure_time > 0:
            final_stress = self.calculate_final_stress(config)
            stress_ratio = final_stress / self.cellular_resistance
        else:
            final_stress = 0.0
            stress_ratio = 0.0

        return {
            'final_stress': final_stress,
            'cellular_resistance': self.cellular_resistance,
            'stress_ratio': stress_ratio,
            'exposure_time': self.stress_exposure_time,
            'current_shear_stress': self.local_shear_stress,
            'stress_factor': self._calculate_stress_factor(config) if self.local_shear_stress > 0 else 0.0,
            'at_risk': stress_ratio > 0.8,  # Warning threshold
            'will_senesce': stress_ratio >= 1.0
        }

    def _calculate_stress_factor(self, config):
        """Calculate stress factor based on shear stress magnitude."""
        tau = self.local_shear_stress
        if tau <= 0:
            return (0.00278 / (16.5*5))
        else:
            return ((0.00278 + tau * 0.00992) / (16.5*5))

    def update_stress_exposure(self, dt_hours):
        """Update stress exposure time for any stress above 0 Pa."""
        if self.local_shear_stress > 0:
            self.stress_exposure_time += dt_hours

    def get_state_dict(self):
        """Get cell state dictionary with real parameters only."""
        state_dict = {
            'cell_id': self.cell_id,
            'position': self.position,
            'centroid': self.centroid,
            'divisions': self.divisions,
            'is_senescent': self.is_senescent,
            'senescence_cause': self.senescence_cause,
            'target_orientation': getattr(self, 'target_orientation', self.actual_orientation),
            'actual_orientation': self.actual_orientation,
            'target_aspect_ratio': getattr(self, 'target_aspect_ratio', self.actual_aspect_ratio),
            'actual_aspect_ratio': self.actual_aspect_ratio,
            'target_area': getattr(self, 'target_area', self.actual_area),
            'actual_area': self.actual_area,
            'territory_size': len(self.territory_pixels),
            'perimeter': self.perimeter,
            # Removed: eccentricity, circularity, compactness (fake parameters)
            'compression_ratio': self.compression_ratio,
            'growth_pressure': self.growth_pressure,
            'age': self.age,
            'adhesion_strength': self.adhesion_strength,
            'response': self.response,
            'local_shear_stress': self.local_shear_stress,
            'senescent_growth_factor': self.senescent_growth_factor,
            'is_enlarged_senescent': self.senescent_growth_factor > 1.5
        }
        return state_dict
"""
Enhanced Grid module with biological energy minimization that integrates with existing temporal dynamics.
"""
import os
import time

import numpy as np
from scipy.spatial.distance import cdist
from .cell import Cell
from .holes import Hole, HoleManager
import random
from typing import List, Tuple, Dict, Optional


class Grid:
    """
    Enhanced Grid class that couples tessellation with biological target properties
    while preserving existing temporal dynamics.
    """

    def __init__(self, width, height, config):
        """Initialize the biological grid with gradient-based energy optimization."""
        self.width = width
        self.height = height
        self.config = config

        # Computational grid (reduced resolution for performance)
        self.computation_scale = 4
        self.comp_width = width // self.computation_scale
        self.comp_height = height // self.computation_scale

        print(f"Grid initialized: Display {width}x{height}, Computation {self.comp_width}x{self.comp_height}")

        # Core data structures
        self.cells = {}
        self.pixel_ownership = np.full((self.comp_height, self.comp_width), -1, dtype=int)
        self.pixel_coords = self._create_pixel_coordinate_grid()
        self.next_cell_id = 0
        self.cell_seeds = {}
        self.territory_map = {}

        # Biological optimization parameters
        self.energy_weights = {
            'area': 1.0,  # Weight for area deviation
            'aspect_ratio': 8.5,  # Weight for aspect ratio deviation
            'orientation': 5.0,  # Weight for orientation deviation
            'overlap': 2.0,  # Weight for preventing overlap
            'boundary': 0.1  # Weight for staying within bounds
        }

        # Keep original grid parameters for compatibility
        self.global_pressure = 1.0
        self.adaptation_rate = 0.1

        # Initialize step counter
        self._adaptation_step_counter = 0

        # Hole management system
        self.hole_manager = HoleManager(self) if getattr(config, 'enable_holes', True) else None
        self.holes_enabled = getattr(config, 'enable_holes', True)

        if self.holes_enabled:
            print(f"🕳️  Hole management enabled (max {getattr(config, 'max_holes', 5)} holes)")

        # Disable continuous biological adaptation
        self.biological_optimization_enabled = False
        self.continuous_adaptation_disabled = True


    def add_cell(self, position=None, divisions=0, is_senescent=False, senescence_cause=None, target_area=100.0):
        """Add a new cell with biological properties."""
        if position is None:
            margin = 20
            position = (
                np.random.uniform(margin, self.width - margin),
                np.random.uniform(margin, self.height - margin)
            )

        cell_id = self.next_cell_id
        self.next_cell_id += 1

        comp_target_area = target_area / (self.computation_scale ** 2)
        cell = Cell(cell_id, position, divisions, is_senescent, senescence_cause, comp_target_area)

        # Add to data structures
        self.cells[cell_id] = cell
        self.cell_seeds[cell_id] = position

        return cell

    def calculate_biological_energy(self):
        """
        Calculate the total biological energy of the system.
        Lower energy means better fit to target properties.
        """
        total_energy = 0.0

        for cell in self.cells.values():
            if cell.is_senescent:
                continue

            if not hasattr(cell, 'target_area') or cell.target_area is None:
                continue

            # Area deviation energy
            if cell.actual_area > 0 and cell.target_area > 0:
                area_ratio = cell.actual_area / cell.target_area
                area_energy = (area_ratio - 1.0) ** 2
                total_energy += self.energy_weights['area'] * area_energy

            # Aspect ratio deviation energy
            if hasattr(cell, 'target_aspect_ratio') and cell.target_aspect_ratio > 0:
                ar_ratio = cell.actual_aspect_ratio / cell.target_aspect_ratio
                ar_energy = (ar_ratio - 1.0) ** 2
                total_energy += self.energy_weights['aspect_ratio'] * ar_energy

            # Orientation deviation energy
            if hasattr(cell, 'target_orientation'):
                target_align = self.to_alignment_angle(cell.target_orientation)
                actual_align = self.to_alignment_angle(cell.actual_orientation)
                angle_diff = abs(target_align - actual_align)
                orientation_energy = (angle_diff / (np.pi/2)) ** 2
                total_energy += self.energy_weights['orientation'] * orientation_energy

        return total_energy

    def update_biological_adaptation(self):
        """
        DISABLED for event-driven system.
        In event-driven mode, adaptations happen during transitions only.
        """
        self._adaptation_step_counter += 1
        pass



    def is_event_driven_mode(self):
        """Check if grid is in event-driven mode."""
        return (hasattr(self.config, 'use_event_driven_system') and
                self.config.use_event_driven_system) or self.continuous_adaptation_disabled

    def preserve_current_configuration(self):
        """
        Preserve the current configuration for transition purposes.
        """
        return {
            'cells': self.cells.copy(),
            'seeds': self.cell_seeds.copy(),
            'territories': self.territory_map.copy(),
            'energy': self.calculate_biological_energy(),
            'timestamp': time.time()
        }

    def apply_configuration(self, config_data):
        """
        Apply a stored configuration to the grid.
        """
        self.cells = config_data['cells']
        self.cell_seeds = config_data['seeds']
        self.territory_map = config_data['territories']
        self._update_voronoi_tessellation()

    def to_alignment_angle(self, angle_rad):
        """Convert any angle to [0, π/2] flow alignment angle"""
        return np.abs(angle_rad) % (np.pi / 2)

    def _update_voronoi_tessellation(self, preserve_temporal_dynamics=False):
        """
        MODIFIED: Enhanced Weighted Voronoi Tessellation with Hole-Based Pressure Relief

        This replaces your existing _update_voronoi_tessellation method with the enhanced version
        that follows your mathematical formulation from Step 4.
        """

        if not self.cells:
            self.pixel_ownership.fill(-1)
            self.territory_map.clear()
            return

        # Control parameters from mathematical formulation
        alpha = 0.2  # Area control strength
        beta = 1.0  # Orientation influence strength (increased for stronger visual impact)
        gamma = 0.4  # Aspect ratio control strength
        # Space-filling partition (eq:argmin, main.tex): the tessellation must tile the
        # whole domain with no gaps. With holes disabled for the MPC run we never want a
        # pixel left unassigned, so the hole-creation distance threshold is set to +inf.
        # Lower this only if biological holes are explicitly required.
        # Source: Table 1, main.tex — phi_sen^max constraint run with config.enable_holes = False
        tau_hole = np.inf if not getattr(self.config, 'enable_holes', False) else 15.0

        seed_points = []
        cell_ids = []
        cell_objects = []

        if preserve_temporal_dynamics:
            temporal_props = {
                cell_id: {
                    'actual_orientation': cell.actual_orientation,
                    'actual_aspect_ratio': cell.actual_aspect_ratio
                }
                for cell_id, cell in self.cells.items()
            }

        # Handle existing holes
        if self.holes_enabled and self.hole_manager:
            if hasattr(self, '_create_pixel_coordinate_grid_with_holes'):
                self.pixel_coords = self._create_pixel_coordinate_grid_with_holes()
            else:
                # Fallback: use regular pixel coordinate grid
                if not hasattr(self, 'pixel_coords') or self.pixel_coords is None:
                    self.pixel_coords = self._create_pixel_coordinate_grid()

            hole_pixels = self.hole_manager.get_hole_pixels()
            for x, y in hole_pixels:
                if 0 <= x < self.comp_width and 0 <= y < self.comp_height:
                    self.pixel_ownership[y, x] = -999
        else:
            # Ensure pixel_coords exists
            if not hasattr(self, 'pixel_coords') or self.pixel_coords is None:
                self.pixel_coords = self._create_pixel_coordinate_grid()

        # Collect cell data
        for cell_id, display_pos in self.cell_seeds.items():
            if hasattr(self, '_display_to_comp_coords'):
                comp_pos = self._display_to_comp_coords(display_pos[0], display_pos[1])
            else:
                # Fallback: assume simple scaling
                comp_pos = (display_pos[0] / self.computation_scale, display_pos[1] / self.computation_scale)

            seed_points.append(comp_pos)
            cell_ids.append(cell_id)
            cell_objects.append(self.cells[cell_id])

        if len(seed_points) < 1:
            return

        if len(seed_points) == 1:
            # Single cell gets all non-hole pixels
            cell_id = cell_ids[0]
            all_pixels = [(x, y) for x in range(self.comp_width)
                          for y in range(self.comp_height)
                          if not (self.holes_enabled and self.hole_manager and
                                  self.hole_manager.is_point_in_hole(x, y))]
            self._assign_territory_to_cell(cell_id, all_pixels)
            return

        seed_array = np.array(seed_points)

        # Step 1: Calculate standard Euclidean distances
        euclidean_distances = cdist(self.pixel_coords, seed_array)

        # Step 2: Get current actual areas for area adjustment
        current_actual_areas = []
        for cell_obj in cell_objects:
            if hasattr(cell_obj, 'actual_area') and cell_obj.actual_area > 0:
                current_actual_areas.append(cell_obj.actual_area)
            else:
                current_territory_size = len(getattr(cell_obj, 'territory_pixels', []))
                current_actual_areas.append(max(1, current_territory_size))

        # Step 3: Calculate enhanced distances using mathematical formulation
        enhanced_distances = np.copy(euclidean_distances)

        for i, (cell_obj, seed_pos, actual_area) in enumerate(zip(cell_objects, seed_points, current_actual_areas)):
            # MODIFIED: Use actual properties for tessellation if preserving temporal dynamics
            if preserve_temporal_dynamics:
                orientation_for_tess = getattr(cell_obj, 'actual_orientation', 0.0)
                aspect_ratio_for_tess = getattr(cell_obj, 'actual_aspect_ratio', 1.0)
            else:
                orientation_for_tess = getattr(cell_obj, 'target_orientation', 0.0)
                aspect_ratio_for_tess = getattr(cell_obj, 'target_aspect_ratio', 1.0)

            target_area = getattr(cell_obj, 'target_area', actual_area)
            target_orientation = orientation_for_tess
            target_aspect_ratio = aspect_ratio_for_tess

            # Ensure target_area is not zero to prevent division by zero
            target_area = max(1.0, target_area)

            # Step 3a: Area Control Adjustment (Equation 16)
            area_ratio = actual_area / target_area
            area_adjustment = (area_ratio - 1.0) * np.sqrt(target_area) * 0.5

            # Step 3b: Orientation Influence Adjustment (Equation 17)
            pixel_vectors = self.pixel_coords - seed_pos
            pixel_angles = np.arctan2(pixel_vectors[:, 1], pixel_vectors[:, 0])
            angle_diffs = np.abs(pixel_angles - target_orientation)
            # Handle angle wrapping for flow direction
            angle_diffs = np.minimum(angle_diffs, np.abs(angle_diffs - np.pi))
            normalized_angle_diffs = np.minimum(angle_diffs / (np.pi / 2), 1.0)
            pixel_distances = np.linalg.norm(pixel_vectors, axis=1)
            orientation_adjustments = normalized_angle_diffs * pixel_distances * 2.0

            # Step 3c: Aspect Ratio Influence Adjustment
            if aspect_ratio_for_tess > 1.0:
                # Calculate pixel positions relative to seed in oriented coordinate system
                pixel_vectors = self.pixel_coords - seed_pos

                # Rotate vectors to align with target orientation
                cos_theta = np.cos(-target_orientation)  # Negative for inverse rotation
                sin_theta = np.sin(-target_orientation)

                # Rotate pixel vectors to cell's coordinate system
                rotated_x = pixel_vectors[:, 0] * cos_theta - pixel_vectors[:, 1] * sin_theta
                rotated_y = pixel_vectors[:, 0] * sin_theta + pixel_vectors[:, 1] * cos_theta

                # Apply aspect ratio scaling: compress in perpendicular direction
                # This makes elongated cells "prefer" pixels along their major axis
                compression_factor = 1.0 / target_aspect_ratio
                scaled_y = rotated_y * compression_factor

                # Calculate modified distance in scaled space
                scaled_distances = np.sqrt(rotated_x ** 2 + scaled_y ** 2)
                original_distances = np.linalg.norm(pixel_vectors, axis=1)

                # Aspect ratio adjustment: difference between scaled and original distance
                aspect_ratio_adjustments = (scaled_distances - original_distances) * 2.0
            else:
                aspect_ratio_adjustments = np.zeros(len(pixel_vectors))

            # Step 3d: Apply complete enhanced distance formula (MODIFIED)
            enhanced_distances[:, i] = (euclidean_distances[:, i] +
                                        alpha * area_adjustment +
                                        beta * orientation_adjustments +
                                        gamma * aspect_ratio_adjustments)  # NEW TERM

        # Step 4: Find minimum distances and implement hole creation (Equations 18-19)
        min_distances = np.min(enhanced_distances, axis=1)
        nearest_cell_indices = np.argmin(enhanced_distances, axis=1)
        hole_mask = min_distances > tau_hole

        # Step 5: Territory assignment with hole preservation
        self.territory_map.clear()
        self.pixel_ownership.fill(-1)

        newly_created_holes = []

        # Assign territories
        for i, cell_id in enumerate(cell_ids):
            cell_pixel_mask = (nearest_cell_indices == i) & ~hole_mask
            cell_pixel_indices = np.where(cell_pixel_mask)[0]
            pixels = [(int(self.pixel_coords[idx][0]), int(self.pixel_coords[idx][1]))
                      for idx in cell_pixel_indices]
            self._assign_territory_to_cell(cell_id, pixels)

        # Handle newly created holes - mark pixels but let existing hole manager handle creation
        if self.holes_enabled and self.hole_manager:
            hole_pixel_indices = np.where(hole_mask)[0]
            for idx in hole_pixel_indices:
                x, y = int(self.pixel_coords[idx][0]), int(self.pixel_coords[idx][1])
                if 0 <= x < self.comp_width and 0 <= y < self.comp_height:
                    self.pixel_ownership[y, x] = -999
                    newly_created_holes.append((x, y))

            # Let the existing biological hole management system handle hole creation
            # The pressure relief pixels are marked, and the hole manager will create holes as needed
            if newly_created_holes:
                print(f"🕳️  Marked {len(newly_created_holes)} pixels for pressure relief")

        # Step 6: Enforce biological limits and update cell properties
        if hasattr(self, 'enforce_biological_limits'):
            self.enforce_biological_limits()

        # Update actual cell properties from territories (Step 7 from formulation)
        if hasattr(self, '_update_actual_properties_from_territories'):
            self._update_actual_properties_from_territories()
        else:
            # Fallback: update properties manually if helper method not yet added
            for cell_id, cell in self.cells.items():
                if cell_id in self.territory_map and self.territory_map[cell_id]:
                    cell.actual_area = len(self.territory_map[cell_id])

        if preserve_temporal_dynamics:
            for cell_id, props in temporal_props.items():
                if cell_id in self.cells:
                    self.cells[cell_id].actual_orientation = props['actual_orientation']
                    self.cells[cell_id].actual_aspect_ratio = props['actual_aspect_ratio']

    def _update_actual_properties_from_territories(self):
        """
        ADDED: Update actual cell properties based on territories.
        This implements Step 7 from your mathematical formulation.
        """
        for cell_id, cell in self.cells.items():
            if cell_id not in self.territory_map:
                continue

            territory_pixels = self.territory_map[cell_id]
            if not territory_pixels:
                continue

            # Calculate actual area (Equation 25)
            cell.actual_area = len(territory_pixels)

            # Calculate actual aspect ratio and orientation using PCA (Equations 26-27)
            pixels_array = np.array(territory_pixels)

            if len(pixels_array) > 1:
                centroid = np.mean(pixels_array, axis=0)
                centered = pixels_array - centroid

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
                        cell.actual_orientation = np.arctan2(principal_axis[1], principal_axis[0])

                        # Aspect ratio from eigenvalues
                        if eigenvals[1] > 0:
                            cell.actual_aspect_ratio = np.sqrt(max(eigenvals[0] / eigenvals[1], 1.0))
                        else:
                            cell.actual_aspect_ratio = 1.0
                    except:
                        # Fallback to target values
                        cell.actual_aspect_ratio = getattr(cell, 'target_aspect_ratio', 1.0)
                        cell.actual_orientation = getattr(cell, 'target_orientation', 0.0)
                else:
                    cell.actual_aspect_ratio = getattr(cell, 'target_aspect_ratio', 1.0)
                    cell.actual_orientation = getattr(cell, 'target_orientation', 0.0)
            else:
                cell.actual_aspect_ratio = getattr(cell, 'target_aspect_ratio', 1.0)
                cell.actual_orientation = getattr(cell, 'target_orientation', 0.0)

    # Include all the other necessary methods from the previous response
    def _angle_difference(self, target, actual):
        """Calculate angle difference considering 0°/180° equivalence for flow direction."""
        target_norm = target % (2 * np.pi)
        actual_norm = actual % (2 * np.pi)
        diff1 = target_norm - actual_norm
        diff2 = (target_norm + np.pi) % (2 * np.pi) - actual_norm
        diff1 = ((diff1 + np.pi) % (2 * np.pi)) - np.pi
        diff2 = ((diff2 + np.pi) % (2 * np.pi)) - np.pi
        if abs(diff1) <= abs(diff2):
            return diff1
        else:
            return diff2

    def _display_to_comp_coords(self, x, y):
        """Convert display coordinates to computational coordinates."""
        return (x / self.computation_scale, y / self.computation_scale)

    def _comp_to_display_coords(self, x, y):
        """Convert computational coordinates to display coordinates."""
        return (x * self.computation_scale, y * self.computation_scale)

    def _create_pixel_coordinate_grid(self):
        """Create a grid of pixel coordinates for computation."""
        y_coords, x_coords = np.mgrid[0:self.comp_height, 0:self.comp_width]
        return np.column_stack([x_coords.ravel(), y_coords.ravel()])

    def _assign_territory_to_cell(self, cell_id, pixels):
        """Assign a territory to a cell."""
        self.territory_map[cell_id] = pixels

        for x, y in pixels:
            if 0 <= x < self.comp_width and 0 <= y < self.comp_height:
                self.pixel_ownership[y, x] = cell_id

        if cell_id in self.cells:
            self.cells[cell_id].assign_territory(pixels)

            # Update territory_map with the limited territory
            self.territory_map[cell_id] = self.cells[cell_id].territory_pixels

    # Add all the missing methods from your original Grid class
    def adapt_cell_properties(self):
        """Adapt cell properties based on space constraints and targets."""
        if not self.cells:
            return

        total_target_area = sum(cell.target_area for cell in self.cells.values())
        total_available_area = self.comp_width * self.comp_height
        self.global_pressure = total_target_area / total_available_area if total_available_area > 0 else 1.0

        for cell in self.cells.values():
            local_space_factor = min(1.0, 1.0 / self.global_pressure)
            cell.adapt_to_constraints(local_space_factor)

    def remove_cell(self, cell_id):
        """Remove a cell."""
        if cell_id not in self.cells:
            return False

        del self.cells[cell_id]
        if cell_id in self.cell_seeds:
            del self.cell_seeds[cell_id]
        if cell_id in self.territory_map:
            del self.territory_map[cell_id]

        return True

    def apply_shear_stress_field(self, shear_stress_function, duration):
        """Apply a shear stress field to all cells."""
        for cell in self.cells.values():
            if cell.centroid is not None:
                display_centroid = self._comp_to_display_coords(cell.centroid[0], cell.centroid[1])
                x, y = display_centroid
            else:
                x, y = cell.position
            shear_stress = shear_stress_function(x, y)
            cell.apply_shear_stress(shear_stress)

    def calculate_confluency(self):
        """Calculate the confluency."""
        return 1.0

    def calculate_packing_efficiency(self):
        """Calculate packing efficiency."""
        if not self.cells:
            return 1.0

        total_deviation = 0
        for cell in self.cells.values():
            if cell.target_area > 0:
                deviation = abs(cell.actual_area - cell.target_area) / cell.target_area
                total_deviation += deviation

        average_deviation = total_deviation / len(self.cells)
        return max(0, 1.0 - average_deviation)

    def optimize_cell_positions(self, iterations=3):
        """
        MODIFIED: Enhanced position optimization following Step 5 from your formulation.

        This replaces your existing optimize_cell_positions method with the enhanced version
        that implements biological energy minimization (Equations 20-23).
        """

        # Energy function weights from mathematical formulation
        w_area = 1.0
        w_AR = 1.0
        w_orient = 1.0

        for iteration in range(iterations):
            position_updates = {}

            for cell_id, cell in self.cells.items():
                if cell_id not in self.cell_seeds:
                    continue

                current_pos = np.array(self.cell_seeds[cell_id])

                # Step 5a: Lloyd's algorithm component (Equation 21)
                if cell_id in self.territory_map and self.territory_map[cell_id]:
                    territory_pixels = np.array(self.territory_map[cell_id])
                    centroid = np.mean(territory_pixels, axis=0)

                    # Convert to display coordinates (with fallback)
                    if hasattr(self, '_comp_to_display_coords'):
                        centroid_display = self._comp_to_display_coords(centroid[0], centroid[1])
                    else:
                        # Fallback: assume 1:1 scaling
                        centroid_display = (centroid[0] * self.computation_scale, centroid[1] * self.computation_scale)

                    # Lloyd's update with smoothing factor α = 0.3
                    lloyd_pos = 0.7 * current_pos + 0.3 * np.array(centroid_display)
                else:
                    lloyd_pos = current_pos

                # Step 5b: Energy minimization component (Equation 22)
                delta = 2.0  # Small perturbation for gradient estimation

                # Calculate current biological energy
                if hasattr(self, '_calculate_biological_energy_for_cell'):
                    current_energy = self._calculate_biological_energy_for_cell(cell)
                else:
                    # Fallback to simple energy calculation
                    current_energy = self._simple_energy_calculation(cell)

                # Test small movements and find energy-minimizing direction
                best_pos = lloyd_pos
                best_energy = current_energy

                for dx, dy in [(delta, 0), (-delta, 0), (0, delta), (0, -delta)]:
                    test_pos = lloyd_pos + np.array([dx, dy])

                    # Check bounds
                    if (test_pos[0] < 20 or test_pos[0] > self.width - 20 or
                            test_pos[1] < 20 or test_pos[1] > self.height - 20):
                        continue

                    # Temporarily update position and estimate energy change
                    old_pos = self.cell_seeds[cell_id]
                    self.cell_seeds[cell_id] = tuple(test_pos)

                    # Quick energy estimation (without full tessellation)
                    if hasattr(self, '_estimate_energy_after_position_change'):
                        test_energy = self._estimate_energy_after_position_change(cell_id, test_pos)
                    else:
                        # Fallback energy calculation
                        test_energy = self._simple_energy_calculation(cell)

                    if test_energy < best_energy:
                        best_energy = test_energy
                        best_pos = test_pos

                    # Restore position
                    self.cell_seeds[cell_id] = old_pos

                position_updates[cell_id] = best_pos

            # Apply position updates
            movement_applied = False
            for cell_id, new_pos in position_updates.items():
                old_pos = np.array(self.cell_seeds[cell_id])
                movement = np.linalg.norm(new_pos - old_pos)

                if movement > 0.5:  # Apply if movement is significant
                    self.cell_seeds[cell_id] = tuple(new_pos)
                    self.cells[cell_id].update_position(tuple(new_pos))
                    movement_applied = True

            if movement_applied:
                # Update tessellation after position changes
                self._update_voronoi_tessellation()
            else:
                break  # Converged

    def _estimate_energy_after_position_change(self, cell_id, new_pos):
        """
        ADDED: Quick energy estimation without full tessellation.
        Used during position optimization for efficiency.
        """

        cell = self.cells[cell_id]

        # Simple distance-based territory estimation
        if cell_id not in self.territory_map or not self.territory_map[cell_id]:
            return self._calculate_biological_energy_for_cell(cell)

        # Estimate how territory would change with new position
        current_territory = self.territory_map[cell_id]
        comp_pos = self._display_to_comp_coords(new_pos[0], new_pos[1])

        # Rough estimation: keep pixels that are still closest to this cell
        estimated_territory = []
        for pixel in current_territory:
            pixel_pos = np.array(pixel)
            dist_to_new_pos = np.linalg.norm(pixel_pos - comp_pos)

            # Check if still closest (simplified)
            is_still_closest = True
            for other_cell_id, other_display_pos in self.cell_seeds.items():
                if other_cell_id == cell_id:
                    continue
                other_comp_pos = self._display_to_comp_coords(other_display_pos[0], other_display_pos[1])
                dist_to_other = np.linalg.norm(pixel_pos - other_comp_pos)

                if dist_to_other < dist_to_new_pos:
                    is_still_closest = False
                    break

            if is_still_closest:
                estimated_territory.append(pixel)

        # Update cell's estimated properties
        estimated_area = len(estimated_territory)

        # Calculate energy with estimated area
        target_area = getattr(cell, 'target_area', estimated_area)
        target_AR = getattr(cell, 'target_aspect_ratio', 1.0)
        target_orient = getattr(cell, 'target_orientation', 0.0)

        # Use current AR and orientation (position change doesn't immediately affect these)
        actual_AR = getattr(cell, 'actual_aspect_ratio', target_AR)
        actual_orient = getattr(cell, 'actual_orientation', target_orient)

        # Calculate energy
        area_error = (estimated_area / max(1.0, target_area) - 1.0) ** 2
        AR_error = (actual_AR / max(1.0, target_AR) - 1.0) ** 2

        orient_diff = np.abs(actual_orient - target_orient)
        orient_diff = min(orient_diff, np.abs(orient_diff - np.pi))
        orient_error = (orient_diff / (np.pi / 2)) ** 2

        return area_error + AR_error + orient_error

    def _calculate_biological_energy_for_cell(self, cell):
        """
        ADDED: Calculate biological energy for a single cell.
        Implements Equation (23) from your mathematical formulation.
        """

        # Get target and actual properties
        target_area = getattr(cell, 'target_area', getattr(cell, 'actual_area', 100))
        target_AR = getattr(cell, 'target_aspect_ratio', 1.0)
        target_orient = getattr(cell, 'target_orientation', 0.0)

        actual_area = getattr(cell, 'actual_area', target_area)
        actual_AR = getattr(cell, 'actual_aspect_ratio', target_AR)
        actual_orient = getattr(cell, 'actual_orientation', target_orient)

        # Ensure no division by zero
        target_area = max(1.0, target_area)
        target_AR = max(1.0, target_AR)

        # Calculate energy components (Equation 23)
        area_error = (actual_area / target_area - 1.0) ** 2
        AR_error = (actual_AR / target_AR - 1.0) ** 2

        # Orientation error (considering 0°/180° equivalence)
        orient_diff = np.abs(actual_orient - target_orient)
        orient_diff = min(orient_diff, np.abs(orient_diff - np.pi))
        orient_error = (orient_diff / (np.pi / 2)) ** 2

        # Weights from mathematical formulation
        w_area = 1.0
        w_AR = 1.0
        w_orient = 1.0

        total_energy = w_area * area_error + w_AR * AR_error + w_orient * orient_error
        return total_energy

    def get_biological_fitness(self):
        """Get current biological fitness (0-1, where 1 is perfect)."""
        if not self.cells:
            return 1.0

        total_fitness = 0.0
        cell_count = 0

        for cell in self.cells.values():
            if not hasattr(cell, 'target_orientation') or cell.is_senescent:
                continue

            cell_fitness = 0.0
            component_count = 0

            # Orientation fitness
            if hasattr(cell, 'actual_orientation'):
                orientation_error = abs(self._angle_difference(cell.target_orientation, cell.actual_orientation))
                orientation_fitness = max(0, 1.0 - orientation_error / np.pi)
                cell_fitness += orientation_fitness
                component_count += 1

            # Aspect ratio fitness
            if hasattr(cell, 'target_aspect_ratio') and hasattr(cell, 'actual_aspect_ratio'):
                if cell.target_aspect_ratio > 0:
                    ar_ratio = min(cell.actual_aspect_ratio / cell.target_aspect_ratio,
                                  cell.target_aspect_ratio / cell.actual_aspect_ratio)
                    cell_fitness += ar_ratio
                    component_count += 1

            # Area fitness
            if hasattr(cell, 'target_area') and cell.actual_area > 0:
                if cell.target_area > 0:
                    area_ratio = min(cell.actual_area / cell.target_area,
                                   cell.target_area / cell.actual_area)
                    cell_fitness += area_ratio
                    component_count += 1

            if component_count > 0:
                total_fitness += cell_fitness / component_count
                cell_count += 1

        return total_fitness / cell_count if cell_count > 0 else 1.0

    def count_cells_by_type(self):
        """Count cells by type."""
        normal_count = 0
        tel_sen_count = 0
        stress_sen_count = 0

        for cell in self.cells.values():
            if not cell.is_senescent:
                normal_count += 1
            elif cell.senescence_cause == 'telomere':
                tel_sen_count += 1
            elif cell.senescence_cause == 'stress':
                stress_sen_count += 1

        return {
            'normal': normal_count,
            'telomere_senescent': tel_sen_count,
            'stress_senescent': stress_sen_count,
            'total': len(self.cells)
        }

    def get_grid_statistics(self):
        """Get comprehensive statistics about the grid state."""
        if not self.cells:
            return {}

        cell_counts = self.count_cells_by_type()
        biological_energy = self.calculate_biological_energy()
        biological_fitness = self.get_biological_fitness()

        target_areas = [cell.target_area * (self.computation_scale ** 2) for cell in self.cells.values()]
        actual_areas = [cell.actual_area * (self.computation_scale ** 2) for cell in self.cells.values()]
        compression_ratios = [cell.compression_ratio for cell in self.cells.values()]

        aspect_ratios = [cell.actual_aspect_ratio for cell in self.cells.values()]
        orientations = [cell.actual_orientation for cell in self.cells.values()]

        # Add hole statistics
        hole_stats = self.get_hole_statistics()

        return {
            'cell_counts': cell_counts,
            'global_pressure': self.global_pressure,
            'packing_efficiency': self.calculate_packing_efficiency(),
            'biological_fitness': biological_fitness,
            'biological_energy': biological_energy,
            'hole_statistics': hole_stats,
            'area_stats': {
                'mean_target_area': np.mean(target_areas),
                'mean_actual_area': np.mean(actual_areas),
                'mean_compression_ratio': np.mean(compression_ratios),
                'std_compression_ratio': np.std(compression_ratios)
            },
            'shape_stats': {
                'mean_aspect_ratio': np.mean(aspect_ratios),
                'std_aspect_ratio': np.std(aspect_ratios),
                'mean_orientation': np.mean(orientations),
                'std_orientation': np.std(orientations)
            }
        }

    def get_display_territories(self):
        """Get cell territories scaled up to display resolution."""
        display_territories = {}

        for cell_id, comp_pixels in self.territory_map.items():
            display_pixels = []
            for comp_x, comp_y in comp_pixels:
                for dx in range(self.computation_scale):
                    for dy in range(self.computation_scale):
                        display_x = comp_x * self.computation_scale + dx
                        display_y = comp_y * self.computation_scale + dy
                        if 0 <= display_x < self.width and 0 <= display_y < self.height:
                            display_pixels.append((display_x, display_y))
            display_territories[cell_id] = display_pixels

        return display_territories

    def enforce_biological_limits(self):
        """Clamp cell territories to biological limits and create holes."""

        for cell in self.cells.values():
            max_area = cell.target_area * (2.0 if not cell.is_senescent else 3.0)

            # Per-cell debug print disabled (kept only warnings / end-of-step summaries).
            # print(
            #     f"🐛 Cell {cell.cell_id}: actual={cell.actual_area} max={max_area} exceeds={cell.actual_area > max_area}")

            if cell.actual_area > max_area:
                # Keep only the pixels closest to cell center
                distances = [(pixel, self._distance_to_cell_center(pixel, cell))
                             for pixel in cell.territory_pixels]
                distances.sort(key=lambda x: x[1])  # Sort by distance

                # Keep only the closest pixels up to max_area
                pixels_to_keep = int(max_area)
                cell.territory_pixels = [pixel for pixel, dist in distances[:pixels_to_keep]]
                cell.actual_area = len(cell.territory_pixels)

                self.territory_map[cell.cell_id] = cell.territory_pixels

                # Mark excess pixels as holes/unassigned
                excess_pixels = [pixel for pixel, dist in distances[pixels_to_keep:]]
                for pixel in excess_pixels:
                    if 0 <= pixel[0] < self.comp_width and 0 <= pixel[1] < self.comp_height:
                        self.pixel_ownership[pixel[1], pixel[0]] = -1

    def _distance_to_cell_center(self, pixel, cell):
        """Calculate distance from pixel to cell center."""
        return np.sqrt((pixel[0] - cell.position[0]) ** 2 + (pixel[1] - cell.position[1]) ** 2)

    def populate_grid(self, count, division_distribution=None, area_distribution=None):
        """Populate the grid with initial cells."""
        created_cells = []

        if division_distribution is None:
            max_div = self.config.max_divisions
            def division_distribution():
                # Start cells at 0-50% of max divisions (avoid immediate senescence)
                r = np.random.random()
                return int(max_div * 0.5 * (1 - np.sqrt(r)))

        if area_distribution is None:
            def area_distribution():
                base_area = (self.width * self.height) / count
                return np.random.uniform(base_area * 0.7, base_area * 1.3)

        positions = self._generate_poisson_disk_samples(count)

        print(f"Creating {count} cells...")

        for i, position in enumerate(positions):
            if i % 50 == 0:
                print(f"Created {i}/{count} cells")

            divisions = division_distribution()
            target_area = area_distribution()
            # Per-cell debug print disabled (kept only warnings / end-of-step summaries).
            # print(f"🐛 INITIAL: Cell will be created with target_area = {target_area}")

            is_senescent = False
            senescence_cause = None

            if divisions >= self.config.max_divisions:
                is_senescent = True
                senescence_cause = 'telomere'


            cell = self.add_cell(position, divisions, is_senescent, senescence_cause, target_area)
            created_cells.append(cell)

        print("Computing initial tessellation...")
        self._update_voronoi_tessellation()

        print("Optimizing initial positions...")
        self.optimize_cell_positions(iterations=2)

        print(f"Grid populated with {len(created_cells)} cells")
        return created_cells

    def _generate_poisson_disk_samples(self, count):
        """Generate positions using Poisson disk sampling."""
        area = self.width * self.height
        min_dist = np.sqrt(area / count) * 0.5

        positions = []
        attempts = 0
        max_attempts = count * 50

        while len(positions) < count and attempts < max_attempts:
            x = np.random.uniform(min_dist, self.width - min_dist)
            y = np.random.uniform(min_dist, self.height - min_dist)
            pos = (x, y)

            valid = True
            for existing_pos in positions:
                dist = np.sqrt((x - existing_pos[0])**2 + (y - existing_pos[1])**2)
                if dist < min_dist:
                    valid = False
                    break

            if valid:
                positions.append(pos)

            attempts += 1

        while len(positions) < count:
            x = np.random.uniform(20, self.width - 20)
            y = np.random.uniform(20, self.height - 20)
            positions.append((x, y))

        return positions

    def add_controlled_variability(self):
        """Add controlled variability to cell orientations and positions."""
        for cell in self.cells.values():
            if cell.territory_pixels:
                max_displacement = 8
                current_pos = self.cell_seeds[cell.cell_id]

                displacement = (
                    np.random.uniform(-max_displacement, max_displacement),
                    np.random.uniform(-max_displacement, max_displacement)
                )

                new_pos = (
                    max(0, min(self.width - 1, current_pos[0] + displacement[0])),
                    max(0, min(self.height - 1, current_pos[1] + displacement[1]))
                )

                if cell.centroid is not None:
                    display_centroid = self._comp_to_display_coords(cell.centroid[0], cell.centroid[1])
                    centroid_dist = np.sqrt((new_pos[0] - display_centroid[0])**2 + (new_pos[1] - display_centroid[1])**2)
                    if centroid_dist < max_displacement * 2:
                        self.cell_seeds[cell.cell_id] = new_pos
                        cell.update_position(new_pos)

    def enable_energy_tracking(self):
        """
        Enable comprehensive energy tracking for your existing Grid.
        Call this method after creating your grid to start tracking energy.
        """
        import numpy as np

        # Initialize energy tracking
        self.energy_tracker = {
            'history': [],
            'detailed_history': [],
            'optimization_iterations': [],
            'weights': self.energy_weights.copy()
        }

        # Store original methods
        self._original_update_biological_adaptation = self.update_biological_adaptation

        # Replace with tracking versions
        self.update_biological_adaptation = self._tracked_update_biological_adaptation

        print("✅ Energy tracking enabled for Grid")

    def get_detailed_energy_breakdown(self):
        """
        Get detailed energy breakdown by component and cell.

        Returns:
            Dictionary with comprehensive energy information
        """
        import numpy as np

        breakdown = {
            'total_energy': 0.0,
            'area_energy': 0.0,
            'aspect_ratio_energy': 0.0,
            'orientation_energy': 0.0,
            'per_cell_energies': [],
            'energy_stats': {}
        }

        if not self.cells:
            return breakdown

        area_energies = []
        ar_energies = []
        orientation_energies = []

        for cell_id, cell in self.cells.items():
            cell_energy = {'cell_id': cell_id, 'is_senescent': cell.is_senescent}

            # Area energy
            if hasattr(cell, 'target_area') and cell.target_area > 0 and cell.actual_area > 0:
                area_ratio = cell.actual_area / cell.target_area
                area_energy = (area_ratio - 1.0) ** 2 * self.energy_weights['area']
                cell_energy['area_energy'] = area_energy
                area_energies.append(area_energy)
            else:
                cell_energy['area_energy'] = 0.0

            # Aspect ratio energy
            if hasattr(cell, 'target_aspect_ratio') and cell.target_aspect_ratio > 0:
                ar_ratio = cell.actual_aspect_ratio / cell.target_aspect_ratio
                ar_energy = (ar_ratio - 1.0) ** 2 * self.energy_weights['aspect_ratio']
                cell_energy['ar_energy'] = ar_energy
                ar_energies.append(ar_energy)
            else:
                cell_energy['ar_energy'] = 0.0

            # Orientation energy
            if hasattr(cell, 'target_orientation'):
                target_align = self.to_alignment_angle(cell.target_orientation)
                actual_align = self.to_alignment_angle(cell.actual_orientation)
                angle_diff = abs(target_align - actual_align)
                orientation_energy = (angle_diff / (np.pi / 2)) ** 2 * self.energy_weights['orientation']
                cell_energy['orientation_energy'] = orientation_energy
                orientation_energies.append(orientation_energy)
            else:
                cell_energy['orientation_energy'] = 0.0

            # Total cell energy
            cell_energy['total_energy'] = (cell_energy['area_energy'] +
                                           cell_energy['ar_energy'] +
                                           cell_energy['orientation_energy'])

            breakdown['per_cell_energies'].append(cell_energy)

        # Aggregate energies
        breakdown['area_energy'] = sum(area_energies)
        breakdown['aspect_ratio_energy'] = sum(ar_energies)
        breakdown['orientation_energy'] = sum(orientation_energies)
        breakdown['total_energy'] = (breakdown['area_energy'] +
                                     breakdown['aspect_ratio_energy'] +
                                     breakdown['orientation_energy'])

        # Statistics
        breakdown['energy_stats'] = {
            'mean_area_energy': np.mean(area_energies) if area_energies else 0,
            'mean_ar_energy': np.mean(ar_energies) if ar_energies else 0,
            'mean_orientation_energy': np.mean(orientation_energies) if orientation_energies else 0,
            'max_area_energy': np.max(area_energies) if area_energies else 0,
            'max_ar_energy': np.max(ar_energies) if ar_energies else 0,
            'max_orientation_energy': np.max(orientation_energies) if orientation_energies else 0,
            'cell_count': len(self.cells)
        }

        return breakdown

    def record_energy_state(self, time_step=None, label=""):
        """
        Record current energy state.

        Parameters:
            time_step: Current time step
            label: Optional label for this energy recording
        """
        if not hasattr(self, 'energy_tracker'):
            return

        breakdown = self.get_detailed_energy_breakdown()
        breakdown['time_step'] = time_step
        breakdown['label'] = label
        breakdown['timestamp'] = len(self.energy_tracker['history'])

        self.energy_tracker['history'].append(breakdown)

    def _tracked_update_biological_adaptation(self):
        """Enhanced biological adaptation with energy tracking."""
        # Record energy before adaptation
        if hasattr(self, 'energy_tracker'):
            self.record_energy_state(label="before_adaptation")

        # Run original adaptation
        result = self._original_update_biological_adaptation()

        # Record energy after adaptation
        if hasattr(self, 'energy_tracker'):
            self.record_energy_state(label="after_adaptation")

        return result

    def _tracked_run_adaptive_optimization(self, intensity, initial_energy):
        """Enhanced adaptive optimization with detailed energy tracking."""
        if hasattr(self, 'energy_tracker'):
            # Record optimization start
            breakdown = self.get_detailed_energy_breakdown()
            breakdown['optimization_intensity'] = intensity
            breakdown['iteration'] = 0
            breakdown['phase'] = 'start'
            self.energy_tracker['optimization_iterations'].append(breakdown)

        # Run original optimization
        result = self._original_run_adaptive_optimization(intensity, initial_energy)

        if hasattr(self, 'energy_tracker'):
            # Record optimization end
            breakdown = self.get_detailed_energy_breakdown()
            breakdown['optimization_intensity'] = intensity
            breakdown['iteration'] = 1
            breakdown['phase'] = 'end'
            self.energy_tracker['optimization_iterations'].append(breakdown)

        return result

    def get_energy_summary(self):
        """
        Get a summary of current energy state.

        Returns:
            Dictionary with energy summary
        """
        if not hasattr(self, 'energy_tracker'):
            return {"error": "Energy tracking not enabled. Call enable_energy_tracking() first."}

        breakdown = self.get_detailed_energy_breakdown()

        summary = {
            'current_total_energy': breakdown['total_energy'],
            'current_area_energy': breakdown['area_energy'],
            'current_ar_energy': breakdown['aspect_ratio_energy'],
            'current_orientation_energy': breakdown['orientation_energy'],
            'energy_per_cell': breakdown['total_energy'] / max(1, len(self.cells)),
            'cell_count': len(self.cells),
            'recordings_count': len(self.energy_tracker['history']) if hasattr(self, 'energy_tracker') else 0,
            'optimization_iterations': len(self.energy_tracker['optimization_iterations']) if hasattr(self,
                                                                                                      'energy_tracker') else 0
        }

        # Add component breakdown percentages
        total = breakdown['total_energy']
        if total > 0:
            summary['component_percentages'] = {
                'area_percent': (breakdown['area_energy'] / total) * 100,
                'ar_percent': (breakdown['aspect_ratio_energy'] / total) * 100,
                'orientation_percent': (breakdown['orientation_energy'] / total) * 100
            }
        else:
            summary['component_percentages'] = {'area_percent': 0, 'ar_percent': 0, 'orientation_percent': 0}

        # Energy trends
        if hasattr(self, 'energy_tracker') and len(self.energy_tracker['history']) > 1:
            recent_energies = [h['total_energy'] for h in self.energy_tracker['history'][-5:]]
            summary['recent_trend'] = 'decreasing' if recent_energies[-1] < recent_energies[0] else 'increasing'
            summary['energy_change'] = recent_energies[-1] - recent_energies[0]

        return summary

    def plot_energy_evolution(self, save_path=None):
        """
        Create a simple energy evolution plot.

        Parameters:
            save_path: Path to save the plot

        Returns:
            matplotlib.Figure
        """
        if not hasattr(self, 'energy_tracker') or not self.energy_tracker['history']:
            print("No energy history available. Enable tracking and run simulation first.")
            return None

        import matplotlib.pyplot as plt

        # Extract data
        history = self.energy_tracker['history']
        timestamps = [h['timestamp'] for h in history]
        total_energies = [h['total_energy'] for h in history]
        area_energies = [h['area_energy'] for h in history]
        ar_energies = [h['aspect_ratio_energy'] for h in history]
        orientation_energies = [h['orientation_energy'] for h in history]

        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Total energy
        ax1.plot(timestamps, total_energies, 'k-', linewidth=2, label='Total Energy')
        ax1.set_ylabel('Total Energy')
        ax1.set_title('Energy Evolution Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Component energies
        ax2.plot(timestamps, area_energies, 'b-', linewidth=2, label='Area')
        ax2.plot(timestamps, ar_energies, 'r-', linewidth=2, label='Aspect Ratio')
        ax2.plot(timestamps, orientation_energies, 'g-', linewidth=2, label='Orientation')
        ax2.set_xlabel('Recording Number')
        ax2.set_ylabel('Component Energy')
        ax2.set_title('Energy by Component')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Energy evolution plot saved to: {save_path}")

        return fig

    def print_energy_report(self):
        """Print a comprehensive energy report."""
        import numpy as np

        if not hasattr(self, 'energy_tracker'):
            print("❌ Energy tracking not enabled. Call grid.enable_energy_tracking() first.")
            return

        summary = self.get_energy_summary()

        print("\n" + "=" * 50)
        print("ENERGY ANALYSIS REPORT")
        print("=" * 50)
        print(f"Total Energy: {summary['current_total_energy']:.4f}")
        print(f"Energy per Cell: {summary['energy_per_cell']:.4f}")
        print(f"Cell Count: {summary['cell_count']}")
        print(f"Recordings: {summary['recordings_count']}")
        print(f"Optimization Iterations: {summary['optimization_iterations']}")

        print(f"\nComponent Breakdown:")
        print(
            f"  Area Energy: {summary['current_area_energy']:.4f} ({summary['component_percentages']['area_percent']:.1f}%)")
        print(
            f"  Aspect Ratio Energy: {summary['current_ar_energy']:.4f} ({summary['component_percentages']['ar_percent']:.1f}%)")
        print(
            f"  Orientation Energy: {summary['current_orientation_energy']:.4f} ({summary['component_percentages']['orientation_percent']:.1f}%)")

        if 'recent_trend' in summary:
            print(f"\nRecent Trend: {summary['recent_trend']} (Δ{summary['energy_change']:+.4f})")

        # Find highest energy cells
        breakdown = self.get_detailed_energy_breakdown()
        if breakdown['per_cell_energies']:
            sorted_cells = sorted(breakdown['per_cell_energies'],
                                  key=lambda x: x['total_energy'], reverse=True)
            print(f"\nTop 3 Highest Energy Cells:")
            for i, cell in enumerate(sorted_cells[:3]):
                print(f"  {i + 1}. Cell {cell['cell_id']}: {cell['total_energy']:.4f} "
                      f"(A:{cell['area_energy']:.3f}, AR:{cell['ar_energy']:.3f}, O:{cell['orientation_energy']:.3f}) "
                      f"{'[SEN]' if cell['is_senescent'] else '[HEALTHY]'}")

        print("=" * 50)
        return

    def update_holes(self, dt):
            """Update hole system for one timestep."""
            if self.holes_enabled and self.hole_manager:
                self.hole_manager.update(dt)

    def get_hole_statistics(self):
            """Get current hole statistics."""
            if self.holes_enabled and self.hole_manager:
                return self.hole_manager.get_hole_statistics()
            return {
                'hole_count': 0,
                'total_hole_area': 0,
                'average_hole_size': 0,
                'hole_area_fraction': 0,
                'holes': []
            }

    def _create_pixel_coordinate_grid_with_holes(self):
        """Create pixel coordinate grid excluding hole areas."""
        # Get all pixel coordinates
        y_coords, x_coords = np.mgrid[0:self.comp_height, 0:self.comp_width]
        all_coords = np.column_stack([x_coords.ravel(), y_coords.ravel()])

        if not self.holes_enabled or not self.hole_manager or not self.hole_manager.holes:
            return all_coords

        # Filter out hole pixels
        valid_coords = []
        for coord in all_coords:
            x, y = coord
            if not self.hole_manager.is_point_in_hole(x, y):
                valid_coords.append(coord)

        return np.array(valid_coords) if valid_coords else all_coords

    def generate_multiple_initial_configurations(self, cell_count, num_configurations=10,
                                                 division_distribution=None, area_distribution=None,
                                                 optimization_iterations=3, verbose=True):
        """
        Generate multiple initial configurations and select the one with lowest energy.

        Parameters:
            cell_count: Number of cells to create
            num_configurations: Number of different initial configurations to try
            division_distribution: Function to generate division counts
            area_distribution: Function to generate target areas
            optimization_iterations: Number of optimization steps for each configuration
            verbose: Whether to print progress information

        Returns:
            Dictionary with best configuration info and all results
        """
        if verbose:
            print(f"🔬 Generating {num_configurations} initial configurations...")

        # Store original state
        original_cells = self.cells.copy()
        original_seeds = self.cell_seeds.copy()
        original_territories = self.territory_map.copy()

        configurations = []

        for config_idx in range(num_configurations):
            if verbose and config_idx % max(1, num_configurations // 5) == 0:
                print(f"   Testing configuration {config_idx + 1}/{num_configurations}...")

            # Clear current state
            self.cells.clear()
            self.cell_seeds.clear()
            self.territory_map.clear()
            self.pixel_ownership.fill(-1)
            self.next_cell_id = 0

            # Generate new configuration
            created_cells = self.populate_grid(
                cell_count,
                division_distribution=division_distribution,
                area_distribution=area_distribution
            )

            # Run optimization to settle the configuration
            for _ in range(optimization_iterations):
                self.update_biological_adaptation()
                self.optimize_cell_positions(iterations=1)

            # Calculate energy
            energy = self.calculate_biological_energy()
            fitness = self.get_biological_fitness()

            # Get detailed statistics
            grid_stats = self.get_grid_statistics()

            # Store configuration data
            config_data = {
                'config_idx': config_idx,
                'energy': energy,
                'fitness': fitness,
                'cell_count': len(self.cells),
                'packing_efficiency': grid_stats.get('packing_efficiency', 0),
                'biological_fitness': grid_stats.get('biological_fitness', 0),
                'global_pressure': grid_stats.get('global_pressure', 1.0),
                # Store cell positions and properties for reconstruction
                'cell_data': {
                    cell_id: {
                        'position': cell.position,
                        'divisions': cell.divisions,
                        'is_senescent': cell.is_senescent,
                        'senescence_cause': cell.senescence_cause,
                        'target_area': cell.target_area,
                        'target_orientation': getattr(cell, 'target_orientation', 0.0),
                        'target_aspect_ratio': getattr(cell, 'target_aspect_ratio', 1.0)
                    }
                    for cell_id, cell in self.cells.items()
                },
                'grid_stats': grid_stats
            }

            configurations.append(config_data)

        # Find best configuration (lowest energy)
        best_config = min(configurations, key=lambda x: x['energy'])
        best_idx = best_config['config_idx']

        if verbose:
            print(f"\n✅ Best configuration found:")
            print(f"   Configuration #{best_idx + 1}")
            print(f"   Energy: {best_config['energy']:.4f}")
            print(f"   Fitness: {best_config['fitness']:.4f}")
            print(f"   Packing efficiency: {best_config['packing_efficiency']:.3f}")

            # Show energy distribution
            energies = [c['energy'] for c in configurations]
            print(f"\n📊 Energy distribution across {num_configurations} configurations:")
            print(f"   Best: {min(energies):.4f}")
            print(f"   Worst: {max(energies):.4f}")
            print(f"   Mean: {np.mean(energies):.4f}")
            print(f"   Std: {np.std(energies):.4f}")
            print(f"   Improvement: {((max(energies) - min(energies)) / max(energies) * 100):.1f}%")

        # Reconstruct best configuration
        self._reconstruct_configuration(best_config['cell_data'])

        return {
            'best_config': best_config,
            'all_configurations': configurations,
            'energy_improvement': (max(c['energy'] for c in configurations) - best_config['energy']),
            'selected_idx': best_idx
        }

    def get_cell_properties(self):
        """
        Get comprehensive cell properties for all cells in the grid.

        Returns:
            Dictionary containing various cell property arrays and statistics
        """
        if not self.cells:
            return {
                'areas': [],
                'aspect_ratios': [],
                'orientations': [],
                'target_areas': [],
                'target_aspect_ratios': [],
                'target_orientations': [],
                'mean_area': 0.0,
                'std_area': 0.0,
                'mean_aspect_ratio': 1.0,
                'std_aspect_ratio': 0.0,
                'mean_orientation': 0.0,
                'std_orientation': 0.0,
                'senescent_count': 0,
                'healthy_count': 0
            }

        # Collect properties from all cells
        areas = []
        aspect_ratios = []
        orientations_deg = []
        target_areas = []
        target_aspect_ratios = []
        target_orientations_deg = []
        senescent_count = 0
        healthy_count = 0

        for cell in self.cells.values():
            # Basic counts
            if cell.is_senescent:
                senescent_count += 1
            else:
                healthy_count += 1

            # Actual properties
            areas.append(cell.actual_area)
            aspect_ratios.append(cell.actual_aspect_ratio)

            # Convert orientation to degrees and normalize
            orientation_deg = np.degrees(cell.actual_orientation) % 180
            if orientation_deg > 90:
                orientation_deg = 180 - orientation_deg
            orientations_deg.append(orientation_deg)

            # Target properties
            target_areas.append(getattr(cell, 'target_area', cell.actual_area))
            target_aspect_ratios.append(getattr(cell, 'target_aspect_ratio', cell.actual_aspect_ratio))

            target_orientation_deg = np.degrees(getattr(cell, 'target_orientation', cell.actual_orientation)) % 180
            if target_orientation_deg > 90:
                target_orientation_deg = 180 - target_orientation_deg
            target_orientations_deg.append(target_orientation_deg)

        # Convert to numpy arrays for statistics
        areas = np.array(areas)
        aspect_ratios = np.array(aspect_ratios)
        orientations_deg = np.array(orientations_deg)
        target_areas = np.array(target_areas)
        target_aspect_ratios = np.array(target_aspect_ratios)
        target_orientations_deg = np.array(target_orientations_deg)

        return {
            # Raw data arrays
            'areas': areas.tolist(),
            'aspect_ratios': aspect_ratios.tolist(),
            'orientations': orientations_deg.tolist(),
            'target_areas': target_areas.tolist(),
            'target_aspect_ratios': target_aspect_ratios.tolist(),
            'target_orientations': target_orientations_deg.tolist(),

            # Statistics
            'mean_area': float(np.mean(areas)),
            'std_area': float(np.std(areas)),
            'mean_aspect_ratio': float(np.mean(aspect_ratios)),
            'std_aspect_ratio': float(np.std(aspect_ratios)),
            'mean_orientation': float(np.mean(orientations_deg)),
            'std_orientation': float(np.std(orientations_deg)),

            # Counts
            'senescent_count': senescent_count,
            'healthy_count': healthy_count,
            'total_count': len(self.cells)
        }
    def _reconstruct_configuration(self, cell_data):
        """
        Reconstruct a specific configuration from stored cell data.

        Parameters:
            cell_data: Dictionary with cell information
        """
        # Clear current state
        self.cells.clear()
        self.cell_seeds.clear()
        self.territory_map.clear()
        self.pixel_ownership.fill(-1)
        self.next_cell_id = 0

        # Recreate cells
        for cell_id, data in cell_data.items():
            # Create cell
            cell = self.add_cell(
                position=data['position'],
                divisions=data['divisions'],
                is_senescent=data['is_senescent'],
                senescence_cause=data['senescence_cause'],
                target_area=data['target_area']
            )

            # Set target properties
            cell.target_orientation = data['target_orientation']
            cell.target_aspect_ratio = data['target_aspect_ratio']
            cell.target_area = data['target_area']

        # Update tessellation
        self._update_voronoi_tessellation()

        # Run brief optimization to settle
        self.optimize_cell_positions(iterations=5)



    def save_configuration_analysis(self, configurations_data, filename=None):
        """
        Save detailed analysis of configuration comparison with cell parameters.
        ENHANCED: Now includes average area, aspect ratio, and orientation for each config.

        Parameters:
            configurations_data: Result from generate_multiple_initial_configurations
            filename: Optional filename for saving
        """
        import pandas as pd
        import matplotlib.pyplot as plt
        import os

        os.makedirs(self.config.plot_directory, exist_ok=True)

        if filename is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"configuration_analysis_{timestamp}"

        configurations = configurations_data['all_configurations']

        # Create DataFrame for analysis with ENHANCED cell parameters
        df_data = []
        for config in configurations:
            # Original metrics
            row = {
                'config_idx': config['config_idx'],
                'energy': config['energy'],
                'fitness': config['fitness'],
                'packing_efficiency': config['packing_efficiency'],
                'biological_fitness': config['biological_fitness'],
                'global_pressure': config['global_pressure'],
                'cell_count': config['cell_count']
            }

            # NEW: Calculate average cell parameters for this configuration
            cell_data = config['cell_data']
            areas = []
            aspect_ratios = []
            orientations_deg = []

            for cell_id, cell_props in cell_data.items():
                # Extract cell parameters
                area = cell_props.get('target_area', 0)
                ar = cell_props.get('target_aspect_ratio', 1.0)
                orientation_rad = cell_props.get('target_orientation', 0.0)

                # Convert to display units and flow alignment angle
                display_area = area * (self.computation_scale ** 2)
                orientation_deg = np.degrees(orientation_rad) % 180
                if orientation_deg > 90:
                    orientation_deg = 180 - orientation_deg

                areas.append(display_area)
                aspect_ratios.append(ar)
                orientations_deg.append(orientation_deg)

            # Add average cell parameters to the row
            row.update({
                'avg_area_pixels': np.mean(areas) if areas else 0,
                'avg_aspect_ratio': np.mean(aspect_ratios) if aspect_ratios else 1.0,
                'avg_orientation_degrees': np.mean(orientations_deg) if orientations_deg else 0.0,
                'is_selected': config['config_idx'] == configurations_data['selected_idx']
            })

            df_data.append(row)

        df = pd.DataFrame(df_data)

        # Sort by energy (best first) for easy reading
        df = df.sort_values('energy').reset_index(drop=True)

        # Save CSV
        csv_path = os.path.join(self.config.plot_directory, f"{filename}.csv")
        df.to_csv(csv_path, index=False)

        # Print summary showing ALL configurations like the user requested
        print(f"\n🎯 ALL CONFIGURATION PARAMETERS")
        print("=" * 60)

        for _, row in df.iterrows():
            status = "⭐ SELECTED" if row['is_selected'] else f"Config #{int(row['config_idx']) + 1}"
            print(f"{status}")
            print("-" * 25)
            print(f"Area: {row['avg_area_pixels']:.1f} pixels²")
            print(f"Aspect Ratio: {row['avg_aspect_ratio']:.2f}")
            print(f"Orientation: {row['avg_orientation_degrees']:.1f}° (flow alignment)")
            print(f"Energy: {row['energy']:.4f}")
            print(f"Fitness: {row['fitness']:.3f}")
            print()

        print("=" * 60)
        print(f"📊 Complete analysis saved to: {csv_path}")

        # Create the existing plots (unchanged)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Energy distribution
        axes[0, 0].hist(df['energy'], bins=min(10, len(configurations) // 2), alpha=0.7, color='blue')
        axes[0, 0].axvline(df[df['is_selected']]['energy'].iloc[0], color='red', linestyle='--',
                           label=f"Selected: {df[df['is_selected']]['energy'].iloc[0]:.4f}")
        axes[0, 0].set_xlabel('Energy')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Energy Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Energy vs Fitness
        selected_row = df[df['is_selected']].iloc[0]
        axes[0, 1].scatter(df['energy'], df['fitness'], alpha=0.7)
        axes[0, 1].scatter(selected_row['energy'], selected_row['fitness'],
                           color='red', s=100, label='Selected')
        axes[0, 1].set_xlabel('Energy')
        axes[0, 1].set_ylabel('Fitness')
        axes[0, 1].set_title('Energy vs Fitness')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # NEW: Aspect Ratio vs Orientation
        axes[1, 0].scatter(df['avg_aspect_ratio'], df['avg_orientation_degrees'], alpha=0.7)
        axes[1, 0].scatter(selected_row['avg_aspect_ratio'], selected_row['avg_orientation_degrees'],
                           color='red', s=100, label='Selected')
        axes[1, 0].set_xlabel('Average Aspect Ratio')
        axes[1, 0].set_ylabel('Average Orientation (degrees)')
        axes[1, 0].set_title('Aspect Ratio vs Orientation')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Configuration ranking
        df_sorted = df.sort_values('energy')
        axes[1, 1].plot(range(len(df_sorted)), df_sorted['energy'], 'b-o', alpha=0.7)
        axes[1, 1].set_xlabel('Configuration Rank')
        axes[1, 1].set_ylabel('Energy')
        axes[1, 1].set_title('Configuration Ranking by Energy')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(self.config.plot_directory, f"{filename}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"📊 Configuration plot saved: {plot_path}")

        return csv_path, plot_path



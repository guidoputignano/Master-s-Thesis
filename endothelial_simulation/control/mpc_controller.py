import random
import numpy as np
from typing import Dict, Tuple, Optional, List
from scipy.optimize import minimize
import warnings
from ..models.temporal_dynamics import TemporalDynamicsModel
from ..models.population_dynamics import PopulationDynamicsModel


# NOTE: the previous module-level `warnings.filterwarnings('ignore')` was removed
# deliberately. It suppressed *all* warnings process-wide (both entry points
# import this module), which masked numerical warnings (numpy overflow/invalid,
# scipy convergence) and the DeprecationWarnings this module now raises for the
# legacy path. Surfacing those is the point of the reliability pass; if a
# specific benign warning becomes noisy, silence that category narrowly instead.


class EndothelialMPCController:
    """
    DEPRECATED (legacy path B): NOT used by run_mpc_simulation / the reported
    model. This is the legacy event-driven MPC controller reached only from
    `main.py`'s local run_mpc_simulation (i.e. `python -m endothelial_simulation.main`).
    It relies on the deprecated per-cell response model (calculate_A_max /
    calculate_tau via predict_future_state). The manuscript results are produced
    by `RecedingHorizonMPC` + `run_mpc_simulation` below. Retained (with a
    DeprecationWarning) so the CLI keeps working; do not build new work on it.

    Enhanced MPC Controller with soft constraints and predictive capabilities.

    Features:
    - Soft constraint for senescent fraction ≤ 0.30
    - Soft constraint for hole area ≤ 5% of total area
    - Predictive hole prevention
    - Cell density constraints (min/max cells)
    - Rate limitation (0.7 Pa)
    - Spatial boundary constraints
    """

    def __init__(self, simulator, config):
        warnings.warn(
            "EndothelialMPCController is DEPRECATED: not used by "
            "run_mpc_simulation / the reported model (path A). It is the legacy "
            "MPC controller (path B), retained only for the `main.py` CLI. Use "
            "RecedingHorizonMPC / run_mpc_simulation for the reported model.",
            DeprecationWarning, stacklevel=2)
        self.simulator = simulator
        self.config = config

        # Control parameters

        self.control_horizon = 20  # steps
        self.dt = 1.0  # minute

        # Constraint parameters
        self.senescence_threshold = 0.30  # 30% senescent fraction hard limit
        self.constraint_tolerance = 0.01  # Tolerance for constraints
        self.hole_area_threshold = 0.05  # 5% hole area limit
        self.rate_limit = 1.4 / 60  # Pa/min
        self.shear_stress_limits = (0.0, 4.0)  # Pa

        # Soft constraint weights (penalty scaling)
        self.weights = {
            'tracking': 100.0,  # Response tracking
            'holes': 0.0,  # Soft hole penalty
            'cell_density': 0.0,  # Cell density penalty
            'rate_limit': 0.0,  # Rate limit penalty
            'control_effort': 0.0,  # Control effort penalty
            'hole_prediction': 0.0,  # Predictive hole prevention
            'flow_alignment': 10000.0,  # Flow alignment penalty
            'senescence': 0.0, # Senescence penalty
        }

        # Spatial parameters
        self.average_cell_area = 30.0  # pixels^2
        self.average_expansion_factor = 1.1
        self.baseline_shear = 1.0

        self.temporal_model = TemporalDynamicsModel(self.config)
        self.population_model = PopulationDynamicsModel(self.config)

        # State history for prediction
        self.state_history = []
        self.max_history_length = 10
        self.history = []

        print("🎯 Enhanced MPC Controller initialized with soft constraints")
        print(f"   Senescence threshold: {self.senescence_threshold:.1%}")
        print(f"   Hole area threshold: {self.hole_area_threshold:.1%}")
        print(f"   Rate limit: {self.rate_limit} Pa/min")


    def set_targets(self, targets: Dict):
        """Set control targets."""
        self.targets = targets

    def get_current_state(self) -> Dict:
        """Get comprehensive current state including orientations."""
        cells = self.simulator.grid.cells
        if not cells:
            return {}

        # Basic cell properties
        responses = [getattr(cell, 'response', 1.0) for cell in cells.values()]
        senescent_count = sum(1 for cell in cells.values() if cell.is_senescent)
        senescence_fraction = senescent_count / len(cells)

        # NEW: Add orientation data
        orientations = [cell.actual_orientation for cell in cells.values()]
        target_orientation = self.targets.get('orientation', 0.0)

        # Calculate alignment metrics
        alignment_errors = []
        for orientation in orientations:
            aligned_angle = self.simulator.grid.to_alignment_angle(orientation)
            target_aligned = self.simulator.grid.to_alignment_angle(target_orientation)
            alignment_errors.append(abs(aligned_angle - target_aligned))

        # Hole information (keep existing)
        hole_manager = getattr(self.simulator.grid, 'hole_manager', None)
        hole_count = len(hole_manager.holes) if hole_manager else 0

        # Calculate hole area fraction (keep existing)
        total_area = self.simulator.grid.comp_width * self.simulator.grid.comp_height
        hole_area = sum(hole.get_area() for hole in hole_manager.holes.values()) if hole_manager else 0
        hole_area_fraction = hole_area / total_area if total_area > 0 else 0

        # Cell density constraints (keep existing)
        available_area = total_area - hole_area
        minimum_cells = available_area / (self.average_cell_area * self.average_expansion_factor)
        maximum_cells = 1.5 * minimum_cells

        # Current shear stress (keep existing)
        current_shear = self.simulator.input_pattern.get('value', 0.0)

        # Calculate biological status for hole prediction (keep existing)
        unfillable_area = 0
        if hole_manager:
            biological_status = hole_manager.get_biological_status()
            unfillable_area = biological_status.get('unfillable_area', 0)

        state = {
            'responses': np.array(responses),
            'senescence_fraction': senescence_fraction,
            'hole_count': hole_count,
            'hole_area_fraction': hole_area_fraction,
            'current_shear': current_shear,
            'cell_count': len(cells),
            'minimum_cells': minimum_cells,
            'maximum_cells': maximum_cells,
            'total_area': total_area,
            'available_area': available_area,
            'unfillable_area': unfillable_area,
            'time': getattr(self.simulator, 'time', 0.0),
            # NEW: Add orientation data
            'orientations': np.array(orientations),
            'mean_alignment_error': np.mean(alignment_errors) if alignment_errors else 0.0,
            'alignment_variance': np.var(alignment_errors) if alignment_errors else 0.0,
        }

        # Update state history (keep existing)
        self.state_history.append(state)
        if len(self.state_history) > self.max_history_length:
            self.state_history.pop(0)

        return state

    def _extract_senescence_rate(self, current_state: Dict, shear_stress: float) -> float:
        """Extract actual senescence rate from PopulationDynamicsModel."""
        try:
            # Initialize population model
            pop_model = self.population_model

            # Set current state from cells
            current_cells = self.simulator.grid.cells
            if not current_cells:
                return 0.0

            # Get current population state
            pop_model.update_from_cells(current_cells, dt=0, tau=shear_stress)
            initial_state = pop_model.state.copy()

            # Predict one time step forward
            dt_hours = self.dt / 60.0  # Convert minutes to hours
            predicted_state = pop_model.update(dt_hours, tau=shear_stress)

            # Calculate senescence rate
            initial_senescent = initial_state['S_tel'] + initial_state['S_stress']
            initial_total = sum(initial_state['E']) + initial_senescent

            predicted_senescent = predicted_state['S_tel'] + predicted_state['S_stress']
            predicted_total = sum(predicted_state['E']) + predicted_senescent

            if initial_total > 0 and predicted_total > 0:
                initial_fraction = initial_senescent / initial_total
                predicted_fraction = predicted_senescent / predicted_total
                senescence_rate = (predicted_fraction - initial_fraction) / self.dt
            else:
                senescence_rate = 0.0

            return max(0.0, senescence_rate)  # Ensure non-negative

        except Exception as e:
            print(f"⚠️ Senescence rate extraction failed: {e}")
            # Fallback to simple model
            return 0.001 * max(0, shear_stress - 2.0)

    def _extract_hole_dynamics(self, current_state: Dict, senescence_fraction: float) -> Dict:
        """Extract hole formation probability from BiologicalHoleManager."""
        try:
            hole_manager = self.simulator.grid.hole_manager
            if not hole_manager:
                return {'creation_prob': 0.0, 'filling_prob': 0.0}

            # Use actual biological decision logic
            unfillable_area = hole_manager._calculate_unfillable_area()

            # Deterministic hole creation
            if unfillable_area > 0:
                creation_probability = 1.0
            elif senescence_fraction >= hole_manager.senescence_threshold:
                # Use actual probabilistic model
                base_prob = hole_manager.hole_creation_probability_base
                creation_probability = base_prob * (1.0 + 2.0 * senescence_fraction)
                creation_probability = min(0.8, creation_probability)
            else:
                creation_probability = 0.0

            # Hole filling probability
            if unfillable_area <= 0 and senescence_fraction < hole_manager.senescence_threshold:
                filling_probability = 0.3
            else:
                filling_probability = 0.0

            return {
                'creation_prob': creation_probability,
                'filling_prob': filling_probability,
                'unfillable_area': unfillable_area
            }
        except Exception as e:
            print(f"⚠️ Hole dynamics extraction failed: {e}")
            # Fallback to simple model
            hole_risk = max(0, senescence_fraction - self.senescence_threshold) * 0.1
            return {'creation_prob': hole_risk, 'filling_prob': 0.0, 'unfillable_area': 0}

    def _extract_orientation_dynamics(self, current_state: Dict, shear_stress: float) -> np.array:
        """Extract orientation dynamics from SpatialPropertiesModel."""
        try:
            current_orientations = current_state.get('orientations', [])
            if len(current_orientations) == 0:
                return np.array([])

            # Use the spatial model from the simulator
            spatial_model = self.simulator.models.get('spatial')
            if not spatial_model:
                raise ValueError("SpatialPropertiesModel not found in simulator")

            # Get the target orientation from the spatial model
            target_orientation = spatial_model.calculate_target_orientation(shear_stress, is_senescent=False)

            # Get time constant for orientation
            tau_orient, _ = self.temporal_model.get_scaled_tau_and_amax(shear_stress, 'orientation')

            # Apply first-order dynamics
            predicted_orientations = []
            for current_orientation in current_orientations:
                orientation_diff = target_orientation - current_orientation
                orientation_diff = (orientation_diff + np.pi) % (2 * np.pi) - np.pi  # Wrap angle

                decay_factor = np.exp(-self.dt / tau_orient)
                new_orientation = current_orientation + (1 - decay_factor) * orientation_diff
                predicted_orientations.append(new_orientation)

            return np.array(predicted_orientations)

        except Exception as e:
            print(f"⚠️ Orientation dynamics extraction failed: {e}")
            return current_state.get('orientations', np.array([]))

    def predict_future_state(self, current_state: Dict, control_sequence: List[float]) -> List[Dict]:
        """Enhanced prediction using extracted dynamics from existing models."""
        predictions = []
        state = current_state.copy()

        for i, u in enumerate(control_sequence):
            # 1. EXTRACT ACTUAL SENESCENCE DYNAMICS
            senescence_rate = self._extract_senescence_rate(state, u)
            new_senescence = min(1.0, state['senescence_fraction'] + senescence_rate * self.dt)

            # 2. EXTRACT ACTUAL HOLE FORMATION DYNAMICS
            hole_dynamics = self._extract_hole_dynamics(state, new_senescence)

            # Predict hole changes
            current_hole_count = state['hole_count']
            expected_new_holes = hole_dynamics['creation_prob'] * self.dt
            expected_filled_holes = hole_dynamics['filling_prob'] * current_hole_count * self.dt

            new_hole_count = max(0, current_hole_count + expected_new_holes - expected_filled_holes)
            new_hole_count = min(new_hole_count, getattr(self.simulator.grid.hole_manager, 'max_holes', 5))

            # Estimate hole area
            avg_hole_area = state['hole_area_fraction'] / current_hole_count if current_hole_count > 0 else 0.01
            new_hole_area = new_hole_count * avg_hole_area

            # 3. EXTRACT ACTUAL FLOW ALIGNMENT DYNAMICS
            predicted_orientations = self._extract_orientation_dynamics(state, u)

            # 4. EXTRACT RESPONSE DYNAMICS (existing temporal model)
            temporal_model = self.temporal_model
            current_responses = state.get('responses', [])

            if len(current_responses) > 0:
                A_max = temporal_model.calculate_A_max(u)
                tau = temporal_model.calculate_tau(A_max)

                # Apply dy/dt = (A_max - y) / tau for each cell
                new_responses = []
                for response in current_responses:
                    # Analytical solution
                    decay_factor = np.exp(-self.dt / tau)
                    new_response = A_max - (A_max - response) * decay_factor
                    new_responses.append(new_response)
            else:
                new_responses = []

            # Calculate alignment metrics for predicted orientations
            target_orientation = self.targets.get('orientation', 0.0)
            alignment_errors = []
            if len(predicted_orientations) > 0:
                for orientation in predicted_orientations:
                    aligned_angle = self.simulator.grid.to_alignment_angle(orientation)
                    target_aligned = self.simulator.grid.to_alignment_angle(target_orientation)
                    alignment_errors.append(abs(aligned_angle - target_aligned))

            # 5. UPDATE PREDICTED STATE
            predicted_state = state.copy()
            predicted_state.update({
                'senescence_fraction': new_senescence,
                'hole_count': new_hole_count,
                'hole_area_fraction': new_hole_area,
                'orientations': predicted_orientations,
                'mean_alignment_error': np.mean(alignment_errors) if alignment_errors else 0.0,
                'responses': np.array(new_responses),
                'current_shear': u,
                'time': state['time'] + (i + 1) * self.dt,
            })

            predictions.append(predicted_state)
            state = predicted_state

        return predictions

    def calculate_cost(self, control_sequence: List[float], current_state: Dict) -> float:
        """Calculate total cost function with soft constraints."""
        total_cost = 0.0

        # Predict future states
        predictions = self.predict_future_state(current_state, control_sequence)

        for i, (u, predicted_state) in enumerate(zip(control_sequence, predictions)):
            step_cost = 0.0

            # 1. TRACKING COST
            target_response = self.targets.get('response', 2.0)
            if len(predicted_state['responses']) > 0:
                response_error = target_response - np.mean(predicted_state['responses'])
                step_cost += self.weights['tracking'] * response_error ** 2

            # FLOW ALIGNMENT COST
            target_orientation = self.targets.get('orientation', 0.0)
            if 'mean_alignment_error' in predicted_state:
                alignment_error = predicted_state['mean_alignment_error']
                step_cost += self.weights['flow_alignment'] * alignment_error ** 2

            # 2. HOLE AREA SOFT CONSTRAINT (5% threshold)
            hole_violation = max(0, predicted_state['hole_area_fraction'] - self.hole_area_threshold)
            if hole_violation > 0:
                # Quadratic penalty with scaling
                penalty = (hole_violation / self.hole_area_threshold) ** 2
                step_cost += self.weights['holes'] * penalty

            # 3. PREDICTIVE HOLE PREVENTION
            if predicted_state['unfillable_area'] > 0:
                # Penalize conditions that lead to unfillable area
                penalty = predicted_state['unfillable_area'] / predicted_state['total_area']
                step_cost += self.weights['hole_prediction'] * penalty ** 2

            # 4. CELL DENSITY CONSTRAINTS
            cell_count = predicted_state['cell_count']
            min_cells = predicted_state['minimum_cells']
            max_cells = predicted_state['maximum_cells']

            if cell_count < min_cells:
                violation = (min_cells - cell_count) / min_cells
                step_cost += self.weights['cell_density'] * violation ** 2
            elif cell_count > max_cells:
                violation = (cell_count - max_cells) / max_cells
                step_cost += self.weights['cell_density'] * violation ** 2

            # Senescence SOFT CONSTRAINT
            senescence_violation = max(0, predicted_state['senescence_fraction'] - self.senescence_threshold)
            if senescence_violation > 0:
                # Quadratic penalty with scaling
                penalty = (senescence_violation / self.senescence_threshold) ** 2
                step_cost += self.weights['senescence'] * penalty

            # 5. CONTROL EFFORT
            step_cost += self.weights['control_effort'] * u ** 2

            # 6. RATE LIMIT CONSTRAINT
            if i == 0:  # Only check first control action
                rate_change = abs(u - current_state['current_shear'])
                if rate_change > self.rate_limit:
                    violation = rate_change - self.rate_limit
                    step_cost += self.weights['rate_limit'] * violation ** 2

            # Weight by prediction horizon (recent predictions more important)
            time_weight = 0.9 ** i
            total_cost += time_weight * step_cost

        return total_cost

    def optimize_control(self, current_state: Dict) -> Tuple[float, Dict]:
        """Optimize control action using pure MPC approach with hard constraints for senescence."""

        # Set up optimization problem
        def objective(control_sequence):
            return self.calculate_cost(control_sequence, current_state)

        # Initial guess (maintain current control)
        x0 = [current_state['current_shear']] * self.control_horizon

        # Bounds for control variables
        bounds = [(self.shear_stress_limits[0], self.shear_stress_limits[1])] * self.control_horizon

        # --- CONSTRAINTS ---
        constraints = []

        # 2. Rate Limit Constraints (Linearized)
        for i in range(self.control_horizon):
            if i == 0:
                def rate_constraint_upper(x, i=i):
                    return self.rate_limit - (x[i] - current_state['current_shear'])
                def rate_constraint_lower(x, i=i):
                    return self.rate_limit + (x[i] - current_state['current_shear'])
                constraints.append({'type': 'ineq', 'fun': rate_constraint_upper})
                constraints.append({'type': 'ineq', 'fun': rate_constraint_lower})
            else:
                def rate_constraint_upper(x, i=i):
                    return self.rate_limit - (x[i] - x[i-1])
                def rate_constraint_lower(x, i=i):
                    return self.rate_limit + (x[i] - x[i-1])
                constraints.append({'type': 'ineq', 'fun': rate_constraint_upper})
                constraints.append({'type': 'ineq', 'fun': rate_constraint_lower})


        # Solve optimization
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-6, 'disp': False}
            )

            if result.success:
                optimal_control = result.x[0]
                cost = result.fun
                print(f"✅ Optimization successful. Cost: {cost:.2f}")
                print(f"   Optimal control sequence: {result.x}")
            else:
                # Detailed error logging for infeasible optimization
                print(f"⚠️ Optimization failed: {result.message}")
                if 'inequality constraints incompatible' in result.message:
                    print("   Error: Infeasible solution. The senescence constraint may be too strict.")
                optimal_control = self._fallback_control(current_state)
                cost = float('inf')

        except Exception as e:
            print(f"⚠️ Optimization failed unexpectedly: {e}")
            optimal_control = self._fallback_control(current_state)
            cost = float('inf')

        return optimal_control, {
            'optimal_shear': optimal_control,
            'cost': cost,
            'current_state': current_state
        }

    def _fallback_control(self, current_state: Dict) -> float:
        """Conservative fallback control to minimize stress and prevent constraint violations."""
        current_shear = current_state['current_shear']

        # Reduce shear stress to a baseline level to minimize senescence
        fallback_shear = self.baseline_shear * 0.8  # Reduce to 80% of baseline

        # Ensure the fallback is within the rate limit
        rate_change = fallback_shear - current_shear
        if abs(rate_change) > self.rate_limit:
            if rate_change > 0:
                optimal_shear = current_shear + self.rate_limit
            else:
                optimal_shear = current_shear - self.rate_limit
        else:
            optimal_shear = fallback_shear

        # Clip to absolute limits
        optimal_shear = np.clip(optimal_shear, self.shear_stress_limits[0], self.shear_stress_limits[1])

        print(f"🛡️ Fallback control activated. Setting shear to {optimal_shear:.2f} Pa.")
        return optimal_shear

    def control_step(self, targets: Optional[Dict] = None) -> Tuple[float, Dict]:
        """Main control step function with early stopping for senescence."""
        if targets is not None:
            self.set_targets(targets)

        # Get current state
        current_state = self.get_current_state()
        if not current_state:
            return self.baseline_shear, {'error': 'No state available'}

        # --- EARLY STOPPING --- #
        if current_state['senescence_fraction'] > self.senescence_threshold:
            print("🛑 Early stopping: Senescence exceeds hard limit. Halting simulation.")
            # Signal to stop the simulation
            return np.nan, {'status': 'STOP', 'reason': 'Senescence limit exceeded'}

        # Optimize control
        optimal_shear, control_info = self.optimize_control(current_state)

        # Add constraint status to info
        control_info.update({
            'constraints': {
                'senescence_fraction': current_state['senescence_fraction'],
                'senescence_violation': max(0, current_state['senescence_fraction'] - self.senescence_threshold),
                'hole_area_fraction': current_state['hole_area_fraction'],
                'hole_violation': max(0, current_state['hole_area_fraction'] - self.hole_area_threshold),
                'cell_count': current_state['cell_count'],
                'cell_density_ok': (current_state['minimum_cells'] <= current_state['cell_count'] <= current_state[
                    'maximum_cells']),
                'rate_limit_ok': abs(optimal_shear - current_state['current_shear']) <= self.rate_limit
            },
            'targets': self.targets.copy()
        })

        self.history.append({
            'time': self.simulator.time,
            'shear_stress': optimal_shear,
            'target': self.targets.get('response', 0),
            'actual': np.mean(current_state.get('responses', [0])),
            'control_signal': optimal_shear # Or any other relevant control signal
        })

        return optimal_shear, control_info

    def get_constraint_status(self) -> Dict:
        """Get detailed constraint status for monitoring."""
        if not self.state_history:
            return {}

        current_state = self.state_history[-1]

        return {
            'senescence': {
                'current': current_state['senescence_fraction'],
                'threshold': self.senescence_threshold,
                'violation': max(0, current_state['senescence_fraction'] - self.senescence_threshold),
                'status': 'OK' if current_state['senescence_fraction'] <= self.senescence_threshold else 'VIOLATED'
            },
            'hole_area': {
                'current': current_state['hole_area_fraction'],
                'threshold': self.hole_area_threshold,
                'violation': max(0, current_state['hole_area_fraction'] - self.hole_area_threshold),
                'status': 'OK' if current_state['hole_area_fraction'] <= self.hole_area_threshold else 'VIOLATED'
            },
            'cell_density': {
                'current': current_state['cell_count'],
                'minimum': current_state['minimum_cells'],
                'maximum': current_state['maximum_cells'],
                'status': 'OK' if current_state['minimum_cells'] <= current_state['cell_count'] <= current_state[
                    'maximum_cells'] else 'VIOLATED'
            },
            'rate_limit': {
                'limit': self.rate_limit,
                'status': 'OK'  # Checked dynamically during control
            }
        }


# =============================================================================
#  Receding-horizon MPC for endothelial mechanoadaptation (main.tex, sec. 2.4)
#
#  Implements run_mpc_simulation(): the control-oriented MPC posed directly on
#  the hierarchical model. The single-cell observables propagate by the closed-
#  form step response (eq:stepsolution / eq:orientation) and the population
#  compartments by the reduced ODEs (eq:reduced), integrated with
#  scipy.integrate.solve_ivp (RK45). At each control instant a constrained
#  SLSQP problem is solved; only the first move is applied (receding horizon).
# =============================================================================
import os
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

from ..models.population_dynamics import population_reduced_rhs


# ----- morphological targets (Table 1, main.tex) -----------------------------
RHO_STAT = 1.9          # Source: imaging — rho_stat = 1.9 (static aspect ratio)
RHO_FLOW = 2.3          # Source: Table 1, main.tex — rho_flow (rho*) = 2.3
RHO_STAT_STD = 0.67     # Source: imaging — aspect ratio spread, static (1.9 +/- 0.67)
RHO_FLOW_STD = 0.78     # Source: imaging — aspect ratio spread, flow (2.3 +/- 0.78)
THETA_STAT_DEG = 45.0   # static orientation baseline AND initial condition: isotropic (no
                        # preferred direction with no flow), ~45 deg mean acute angle; the
                        # relaxation start point theta(0) for theta(t)=theta* +
                        # (theta_stat-theta*) exp(-t/tau_orient)
THETA_FLOW_DEG = 0.0    # orientation target theta* = 0 deg (PARALLEL / perfect flow
                        # alignment). Re-calibrated: parallel is the optimal plateau and
                        # 20 deg is the transient reached at t=6 h (was 20 deg, an
                        # asymptote). See tau_orient (tau = 6/ln(45/20) ~ 7.4 h).
THETA_STAT_STD_DEG = 25.0  # Source: imaging — orientation spread, static (49 +/- 25 deg)
THETA_FLOW_STD_DEG = 14.0  # Source: imaging — orientation spread, flow (20 +/- 14 deg)
RHO_SEN = 2.0           # senescent aspect ratio (no flow response)
PHI_SEN_RANDOM = np.pi / 4.0  # mean acute alignment of randomly oriented senescent cells


def flow_alignment_angle(theta):
    """
    Acute flow-alignment angle phi (eq:alignment, main.tex):
        phi = min(|theta mod pi|, pi - |theta mod pi|)  in [0, pi/2]
    phi = 0 denotes perfect alignment with the flow (theta = 0).
    """
    a = np.abs(theta) % np.pi
    return np.minimum(a, np.pi - a)


def _s_activation(tau, tau_act):
    """Monotone activation gate s(tau) of eq:target (main.tex)."""
    if tau <= tau_act:
        return 0.0
    return 1.0 - np.exp(-(tau - tau_act) / tau_act)


def _gated(y_stat, y_flow, tau, tau_act):
    """Gated interpolation target y*(tau) = y_stat + (y_flow - y_stat) s(tau)."""
    return y_stat + (y_flow - y_stat) * _s_activation(tau, tau_act)


class RecedingHorizonMPC:
    """
    Receding-horizon MPC controller for endothelial mechanoadaptation
    (main.tex, sec. 2.4, eq:ocp). This class + ``run_mpc_simulation`` below are
    THE reported model (path A); the docstring is written to serve directly as
    the basis for the methods section.

    Optimal control problem
    -----------------------
    Decision variable:  tau(k) in [0, 2] Pa on a 1 h control grid.
    Prediction horizon: N_p steps;  control horizon: N_c steps (blocked input).
    Cost (per spec):
        J = w_phi    * sum_k phi_sen(k)^2
          + w_rho    * sum_k (rho_bar(k)   - 2.3)^2
          + w_varphi * sum_k (varphi_bar(k) - 0.0)^2
          + w_u      * sum_k (tau(k) - tau(k-1))^2
    Hard constraints:  phi_sen(k) <= 0.30 ;  0 <= tau(k) <= 2.
    Solved by constrained SLSQP; only the first move is applied (receding
    horizon).

    Prediction state
    ----------------
    x = {pop, rho_h, theta_h}, where
      * pop = [E_0, ..., E_N, S_tel, S_str] are the reduced population
        compartments (healthy cells by division stage + telomere- and
        stress-senescent counts);
      * rho_h is the healthy-population mean aspect ratio;
      * theta_h is the healthy-population mean orientation (radians).

    Morphological targets — gated static -> flow interpolation
    ----------------------------------------------------------
    Targets are NOT free set-points: each is the flow-driven interpolation of a
    measured static baseline toward a measured flow plateau, opened by the
    monotone activation gate s(tau) (eq:target):

        s(tau) = 0                                for tau <= tau_act
        s(tau) = 1 - exp(-(tau - tau_act)/tau_act) for tau >  tau_act
        y*(tau) = y_stat + (y_flow - y_stat) * s(tau)

    with tau_act = config.tau_act (Table 1; 0.5 Pa). Applied to:
      * aspect ratio:  rho_stat = 1.9  ->  rho_flow = 2.3  (rho_target)
      * orientation:   theta_stat = 45 deg -> theta_flow = 0 deg (parallel /
        perfect flow alignment) (theta_target)
    Below tau_act the monolayer stays isotropic (s = 0, targets = static).
    Per-cell heterogeneity is added as target = mean + z * std(tau), z fixed per
    cell, with the experimental spread itself gated static -> flow.

    A SINGLE morphological adaptation time constant (this is the reported dynamics)
    ------------------------------------------------------------------------------
    Between control instants the healthy morphology relaxes toward its target by
    the closed-form first-order step response (eq:stepsolution)
        y(t+dt) = y* - (y* - y0) * exp(-dt / tau),
    with ONE data-calibrated, INPUT-INDEPENDENT constant tau = 7.4 h governing
    BOTH morphological channels (there is no tau = f(A_max) scaling here — that is
    the deprecated legacy model). Orientation and aspect ratio are driven by the
    same cytoskeletal remodelling; only the orientation channel is calibrated
    against imaging, and that single calibration fixes the shared constant:
      * ORIENTATION (config.tau_orient_hours = 7.4 h): theta relaxes on the circle
        using the shortest-arc wrap <psi> = ((psi + pi) mod 2pi) - pi. Calibrated
        so theta(6 h) = 20 deg matches the reference imaging:
        20 = 45*exp(-6/tau) => tau = 6/ln(45/20) ~ 7.4 h.
      * ASPECT RATIO (config.tau_adapt_hours = 7.4 h): rho relaxes toward
        rho_target(tau) with the SAME constant (previously 9.0 h; unified onto the
        single calibrated value — there is no independent aspect-ratio timecourse
        to justify a distinct value). The two config fields are kept separate,
        both 7.4 h, only so a sensitivity study can perturb them independently;
        physically they are one constant.
    Cell AREA is NOT relaxed with a temporal constant on the reported path: it is
    fixed by the Voronoi tessellation (spatial model). tau_adapt therefore governs
    the ASPECT RATIO only.

    Senescence — reduced population ODEs (NOT a per-cell stress clock)
    -----------------------------------------------------------------
    The senescent fraction is governed by the reduced population ODE system
    ``population_reduced_rhs`` (eq:reduced), integrated over each control
    interval with scipy.integrate.solve_ivp (RK45):
        dE_i/dt   = 2 r g(N_E) E_{i-1} - r g(N_E) E_i - gamma_tau(tau)(1+xi i) E_i
        dS_tel/dt = r g(N_E) E_N
        dS_str/dt = sum_i gamma_tau(tau)(1+xi i) E_i
        g(N_E)    = 1/(1 + N_E/K)                            (contact inhibition)
        gamma_tau = gamma_min + alpha_gamma (tau - tau_opt)^2 (U-shaped induction)
    phi_sen = (S_tel + S_str) / (N_E + S_tel + S_str). The per-cell
    ``stress_exposure_time`` clock of the legacy model plays NO role here; in the
    closed loop the ODE compartment counts are mapped back onto individual cells
    by ``_reconcile_senescence``.

    Regulated outputs (population means; see ``outputs``)
    ----------------------------------------------------
    phi_sen; rho_bar (senescent cells carry the non-adapting RHO_SEN = 2.0);
    varphi_bar = mean acute flow-alignment angle (senescent cells randomly
    oriented at PHI_SEN_RANDOM = pi/4).
    """

    def __init__(self, config,
                 dt_h=1.0, n_prediction=6, n_control=3,
                 tau_bounds=(0.0, 2.0), phi_sen_max=0.30,
                 w_phi=10.0, w_rho=1.0, w_varphi=5.0, w_u=0.1):
        self.config = config
        self.dt_h = dt_h
        self.Np = n_prediction
        self.Nc = n_control
        self.tau_min, self.tau_max = tau_bounds
        self.phi_sen_max = phi_sen_max
        self.w_phi, self.w_rho, self.w_varphi, self.w_u = w_phi, w_rho, w_varphi, w_u

        # Model parameters (Table 1, main.tex) read from config
        self.N = config.max_divisions
        self.r = config.proliferation_rate
        # Contact inhibition g(N_E)=1/(1+N_E/K) needs K and N_E in the same units.
        # Table 1 gives K as a density (cells/cm^2); the simulation tracks counts,
        # so convert to a count-based capacity over the imaging-field area.
        # area = (650 um)^2 = (0.065 cm)^2 = 4.225e-3 cm^2.
        area_cm2 = (getattr(config, 'imaging_field_um', 650.0) / 1.0e4) ** 2
        self.K = config.carrying_capacity * area_cm2   # Source: Table 1, K=5.5e4 cells/cm^2
        self.xi = config.xi
        self.gamma_min = config.gamma_min
        self.alpha_gamma = config.alpha_gamma
        self.tau_opt = config.tau_opt
        self.tau_act = config.tau_act
        # ASPECT-RATIO (rho) relaxation constant only; 7.4 h, equal to tau_orient
        # (one physical morphological constant). Cell AREA is fixed by the Voronoi
        # tessellation, NOT relaxed with this constant on the reported path.
        self.tau_adapt = config.tau_adapt_hours   # h — aspect-ratio relaxation (= tau_orient = 7.4 h)
        # Orientation time constant: theta relaxes from theta_stat=45 deg toward
        # the parallel target theta*=0 deg, calibrated so theta(6 h)=20 deg matches the
        # reference imaging:  20 = 45*exp(-6/tau)  ->  tau = 6/ln(45/20) ~ 7.4 h.
        # tau_adapt is set equal to this single calibrated value.
        self.tau_orient = getattr(config, 'tau_orient_hours', 7.4)
        self.theta_stat = np.radians(THETA_STAT_DEG)
        self.theta_flow = np.radians(THETA_FLOW_DEG)

    # ---- target maps (population mean) --------------------------------------
    def rho_target(self, tau):
        return _gated(RHO_STAT, RHO_FLOW, tau, self.tau_act)

    def theta_target(self, tau):
        return _gated(self.theta_stat, self.theta_flow, tau, self.tau_act)

    # ---- per-cell target spread (experimental +/- std, gated by flow) -------
    def rho_std(self, tau):
        """Aspect-ratio spread, interpolating static->flow with the activation gate."""
        return _gated(RHO_STAT_STD, RHO_FLOW_STD, tau, self.tau_act)

    def theta_std(self, tau):
        """Orientation spread (rad), interpolating static->flow with the gate."""
        return _gated(np.radians(THETA_STAT_STD_DEG),
                      np.radians(THETA_FLOW_STD_DEG), tau, self.tau_act)

    def cell_rho_target(self, tau, z):
        """Per-cell aspect-ratio target = mean + z * std (z fixed per cell)."""
        return max(1.0, self.rho_target(tau) + z * self.rho_std(tau))

    def cell_theta_target(self, tau, z):
        """Per-cell orientation target = mean + z * std (z fixed per cell)."""
        return self.theta_target(tau) + z * self.theta_std(tau)

    # ---- one-step prediction (closed-form morphology + RK45 population) ------
    def predict_step(self, state, tau, dt=None):
        """
        Advance the reduced prediction state by one control interval at ``tau``.

        Three coupled updates, each per the reported model:
          1. Population compartments (senescence): integrate ``population_reduced_rhs``
             (eq:reduced) over [0, dt] with solve_ivp/RK45.
          2. Aspect ratio rho_h: closed-form relaxation toward rho_target(tau)
             with the FIXED tau_adapt (7.4 h = tau_orient).
          3. Orientation theta_h: closed-form relaxation on the circle
             (shortest-arc wrap) toward theta_target(tau) with the FIXED
             tau_orient (7.4 h).
        A single morphological constant (7.4 h) governs both channels; cell area
        is set by the tessellation, not relaxed here. No input-dependent
        time-constant scaling is used (that is the deprecated legacy model).
        """
        dt = self.dt_h if dt is None else dt
        pop, rho_h, theta_h = state['pop'], state['rho_h'], state['theta_h']

        # Population compartments: integrate eq:reduced with solve_ivp / RK45.
        sol = solve_ivp(
            lambda t, y: population_reduced_rhs(
                y, tau, self.r, self.K, self.xi,
                self.gamma_min, self.alpha_gamma, self.tau_opt, self.N),
            (0.0, dt), pop, method='RK45', rtol=1e-6, atol=1e-9)
        pop_new = np.clip(sol.y[:, -1], 0.0, None)

        # Single-cell observables: closed-form step response (eq:stepsolution).
        rho_t = self.rho_target(tau)
        rho_h_new = rho_t - (rho_t - rho_h) * np.exp(-dt / self.tau_adapt)

        # Orientation relaxes on the circle (eq:orientation) with its own time constant.
        th_t = self.theta_target(tau)
        diff = ((th_t - theta_h) + np.pi) % (2 * np.pi) - np.pi
        theta_h_new = theta_h + diff * (1.0 - np.exp(-dt / self.tau_orient))

        return {'pop': pop_new, 'rho_h': rho_h_new, 'theta_h': theta_h_new}

    def outputs(self, state):
        """Population-mean regulated outputs (phi_sen, rho_bar, varphi_bar)."""
        pop = state['pop']
        N_E = float(pop[:self.N + 1].sum())
        S = float(pop[self.N + 1] + pop[self.N + 2])
        N_tot = N_E + S
        if N_tot <= 0:
            return 0.0, state['rho_h'], flow_alignment_angle(state['theta_h'])
        phi_sen = S / N_tot
        # senescent fraction carries distinct (non-adapting) morphology
        rho_bar = (N_E * state['rho_h'] + S * RHO_SEN) / N_tot
        varphi_bar = (N_E * flow_alignment_angle(state['theta_h'])
                      + S * PHI_SEN_RANDOM) / N_tot
        return phi_sen, rho_bar, varphi_bar

    # ---- cost and constraints ----------------------------------------------
    def _expand(self, u):
        """Block the control horizon: hold the last move to the prediction end."""
        u = list(u)
        return np.array(u + [u[-1]] * (self.Np - self.Nc))

    def _rollout(self, u_seq, x0):
        u_full = self._expand(u_seq)
        state = {'pop': x0['pop'].copy(), 'rho_h': x0['rho_h'], 'theta_h': x0['theta_h']}
        traj = []
        for tau in u_full:
            state = self.predict_step(state, tau)
            traj.append((tau, self.outputs(state)))
        return traj

    def cost(self, u_seq, x0, u_prev):
        traj = self._rollout(u_seq, x0)   # list of (tau, (phi_sen, rho_bar, varphi_bar))
        J = 0.0
        prev = u_prev
        for tau, (phi_sen, rho_bar, varphi_bar) in traj:
            J += (self.w_phi * phi_sen ** 2
                  + self.w_rho * (rho_bar - RHO_FLOW) ** 2
                  + self.w_varphi * (varphi_bar - 0.0) ** 2
                  + self.w_u * (tau - prev) ** 2)
            prev = tau
        return J

    def _phi_sen_margin(self, u_seq, x0):
        """Constraint vector: phi_sen_max - phi_sen(k) >= 0 over the horizon."""
        traj = self._rollout(u_seq, x0)
        return np.array([self.phi_sen_max - out[0] for _, out in traj])

    def solve(self, x0, u_prev):
        """Solve the OCP (eq:ocp) and return the full optimal control horizon."""
        u0 = np.full(self.Nc, np.clip(u_prev, self.tau_min, self.tau_max))
        bounds = [(self.tau_min, self.tau_max)] * self.Nc
        constraints = [{
            'type': 'ineq',
            'fun': lambda u, x0=x0: self._phi_sen_margin(u, x0)
        }]
        res = minimize(self.cost, u0, args=(x0, u_prev), method='SLSQP',
                       bounds=bounds, constraints=constraints,
                       options={'maxiter': 200, 'ftol': 1e-8})
        u_opt = res.x if res.success else u0
        return np.asarray(u_opt), res


# =============================================================================
#  Visualisation helpers
# =============================================================================
def _apply_plot_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        'axes.labelsize': 11, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'legend.fontsize': 9, 'savefig.dpi': 300, 'figure.dpi': 120,
        'axes.titlesize': 11,
    })


def _class_array(grid):
    """Map the tessellation pixel ownership to a cell-class image.

    0 = healthy, 1 = stress-senescent, 2 = telomere-senescent, NaN = gap/hole.
    """
    po = grid.pixel_ownership
    out = np.full(po.shape, np.nan)
    for cid, cell in grid.cells.items():
        if not cell.is_senescent:
            val = 0
        elif cell.senescence_cause == 'stress':
            val = 1
        else:
            val = 2
        out[po == cid] = val
    return out


# class fill colours: healthy / stress-senescent / telomere-senescent
_CLASS_RGB = {0: (0.49, 0.79, 0.49), 1: (0.255, 0.41, 0.88), 2: (0.86, 0.08, 0.24)}


def _ownership_rgb(grid):
    """Render the tessellation as an RGB image: cells filled by class with black
    cell boundaries drawn so individual territories remain visible."""
    po = grid.pixel_ownership
    cls = _class_array(grid)
    rgb = np.ones((po.shape[0], po.shape[1], 3))      # white background (gaps)
    for v, col in _CLASS_RGB.items():
        rgb[cls == v] = col
    # cell boundaries: a pixel whose 4-neighbour belongs to a different owner
    bnd = np.zeros(po.shape, dtype=bool)
    diff_v = po[:-1, :] != po[1:, :]
    bnd[:-1, :] |= diff_v; bnd[1:, :] |= diff_v
    diff_h = po[:, :-1] != po[:, 1:]
    bnd[:, :-1] |= diff_h; bnd[:, 1:] |= diff_h
    rgb[bnd] = (0.0, 0.0, 0.0)
    return rgb


def _render_frame(grid, tau, t_h, phi_sen, path):
    """Render the current tessellation to a PDF frame and return the RGB image."""
    import matplotlib.pyplot as plt

    rgb = _ownership_rgb(grid)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(rgb, interpolation='nearest', origin='lower')
    ax.set_xlabel(r'$x$ (computational pixels)')
    ax.set_ylabel(r'$y$ (computational pixels)')
    # annotate (no title, per style): a small text box for time / input
    ax.text(0.02, 0.98,
            rf'$t={t_h:.2f}$ h    $\tau={tau:.2f}$ Pa    $\phi_{{sen}}={phi_sen:.2f}$',
            transform=ax.transAxes, va='top', ha='left', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    return rgb


def _build_animation(frames, out_dir):
    """Assemble the frame class-arrays into an MP4 (ffmpeg) or GIF (Pillow)."""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xticks([]); ax.set_yticks([])
    im = ax.imshow(frames[0]['arr'], interpolation='nearest', origin='lower')
    txt = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top', ha='left',
                  fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def update(i):
        f = frames[i]
        im.set_array(f['arr'])
        txt.set_text(rf"$t={f['t_h']:.2f}$ h  $\tau={f['tau']:.2f}$ Pa  "
                     rf"$\phi_{{sen}}={f['phi_sen']:.2f}$")
        return im, txt

    anim = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)

    mp4 = os.path.join(out_dir, 'mpc_animation.mp4')
    gif = os.path.join(out_dir, 'mpc_animation.gif')
    try:
        if animation.writers.is_available('ffmpeg'):
            anim.save(mp4, writer=animation.FFMpegWriter(fps=4))
            plt.close(fig)
            return mp4
    except Exception as e:
        print(f"⚠️  ffmpeg animation failed ({e}); falling back to GIF.")
    anim.save(gif, writer=animation.PillowWriter(fps=4))
    plt.close(fig)
    return gif


def _summary_plots(log, out_dir):
    """Produce the three summary figures (tau, phi_sen, morphology)."""
    import matplotlib.pyplot as plt

    t = np.asarray(log['t_h'])          # hour boundaries 0..n
    tau = np.asarray(log['tau'])        # applied input per step (len n)
    phi = np.asarray(log['phi_sen'])    # len n+1
    rho = np.asarray(log['rho_bar'])    # len n+1
    varphi_deg = np.degrees(np.asarray(log['varphi_bar']))

    # tau trajectory (step function); tau[k] held on [t_k, t_{k+1})
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.step(np.append(t[:-1], t[-1]), np.append(tau, tau[-1]), where='post')
    ax.set_xlabel('time (h)')
    ax.set_ylabel(r'wall shear stress $\tau$ (Pa)')
    ax.set_ylim(-0.05, 2.05)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'mpc_tau_trajectory.pdf'),
                                    format='pdf', bbox_inches='tight'); plt.close(fig)

    # phi_sen with constraint line
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(t, phi, marker='o', ms=3)
    ax.axhline(0.30, ls='--', color='k', lw=1)
    ax.set_xlabel('time (h)')
    ax.set_ylabel(r'senescent fraction $\phi_{\mathrm{sen}}$')
    ax.set_ylim(0.0, 0.35)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'mpc_phi_sen.pdf'),
                                    format='pdf', bbox_inches='tight'); plt.close(fig)

    # morphology: rho_bar and alignment (deg) on twin axes
    halign_deg = np.degrees(np.asarray(log['healthy_align']))
    fig, ax1 = plt.subplots(figsize=(5, 3.2))
    l1, = ax1.plot(t, rho, marker='o', ms=3, color='C0', label=r'$\bar{\rho}$')
    ax1.axhline(RHO_FLOW, ls=':', color='C0', lw=1)
    ax1.set_xlabel('time (h)')
    ax1.set_ylabel(r'mean aspect ratio $\bar{\rho}$', color='C0')
    ax1.tick_params(axis='y', labelcolor='C0')
    ax2 = ax1.twinx()
    # population-mean alignment (main.tex phi_bar over all cells) and the
    # healthy-cell alignment diagnostic (converges to the 20 deg flow target)
    l2, = ax2.plot(t, varphi_deg, marker='s', ms=3, color='C3',
                   label=r'$\bar{\varphi}$ (all cells)')
    l3, = ax2.plot(t, halign_deg, marker='^', ms=3, color='C1', ls='--',
                   label=r'$\bar{\varphi}_{\mathrm{healthy}}$')
    ax2.axhline(THETA_FLOW_DEG, ls=':', color='C1', lw=1)
    ax2.set_ylabel(r'flow alignment (deg)', color='C3')
    ax2.tick_params(axis='y', labelcolor='C3')
    ax1.legend(handles=[l1, l2, l3], loc='best')
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'mpc_morphology.pdf'),
                                    format='pdf', bbox_inches='tight'); plt.close(fig)


def _build_dashboard(frames, ts, mpc, out_dir):
    """Composite animation: tessellation (left) + synced time series (right)."""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    t = np.asarray(ts['t_h'])
    tau = np.asarray(ts['tau'])
    phi = np.asarray(ts['phi_sen'])
    rho = np.asarray(ts['rho_bar'])
    varphi_deg = np.degrees(np.asarray(ts['varphi_bar']))
    halign_deg = np.degrees(np.asarray(ts['healthy_align']))
    tmax = max(1.0, float(t[-1]))

    fig = plt.figure(figsize=(11, 5))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.05, 1.0], hspace=0.45, wspace=0.28)
    ax_img = fig.add_subplot(gs[:, 0])
    ax_tau = fig.add_subplot(gs[0, 1])
    ax_phi = fig.add_subplot(gs[1, 1])
    ax_mor = fig.add_subplot(gs[2, 1])

    ax_img.set_xticks([]); ax_img.set_yticks([])
    ax_img.set_xlabel('tessellation (healthy / stress-sen / telomere-sen)')
    im = ax_img.imshow(frames[0]['arr'], interpolation='nearest', origin='lower')
    txt = ax_img.text(0.02, 0.98, '', transform=ax_img.transAxes, va='top', ha='left',
                      fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # static reference lines / full faint trajectories
    for ax in (ax_tau, ax_phi, ax_mor):
        ax.set_xlim(0, tmax)
    ax_tau.set_ylim(-0.05, 2.05); ax_tau.set_ylabel(r'$\tau$ (Pa)')
    ax_phi.set_ylim(0, 0.35); ax_phi.set_ylabel(r'$\phi_{\mathrm{sen}}$')
    ax_phi.axhline(0.30, ls='--', color='k', lw=1)
    ax_mor.set_ylabel('align (deg)'); ax_mor.set_xlabel('time (h)')
    ax_mor.axhline(THETA_FLOW_DEG, ls=':', color='C1', lw=1)
    ax_mor.set_ylim(0, max(55, float(np.nanmax(varphi_deg)) + 5))
    ax_tau_r = ax_mor.twinx(); ax_tau_r.set_ylabel(r'$\bar{\rho}$', color='C0')
    ax_tau_r.set_ylim(1.8, 2.4); ax_tau_r.tick_params(axis='y', labelcolor='C0')

    (ln_tau,) = ax_tau.plot([], [], color='C2', drawstyle='steps-post')
    (ln_phi,) = ax_phi.plot([], [], color='C4', marker='', lw=1.5)
    (ln_va,) = ax_mor.plot([], [], color='C3', lw=1.5, label=r'$\bar{\varphi}$ all')
    (ln_ha,) = ax_mor.plot([], [], color='C1', ls='--', lw=1.5, label=r'$\bar{\varphi}$ healthy')
    (ln_rho,) = ax_tau_r.plot([], [], color='C0', lw=1.5, label=r'$\bar{\rho}$')
    ax_mor.legend(loc='upper right', fontsize=7)

    def update(i):
        f = frames[i]
        n = f['snap']                      # number of ts points up to this frame
        im.set_array(f['arr'])
        txt.set_text(rf"$t={f['t_h']:.2f}$ h   $\tau={f['tau']:.2f}$ Pa   "
                     rf"$\phi_{{sen}}={f['phi_sen']:.2f}$")
        ln_tau.set_data(t[:n], tau[:n])
        ln_phi.set_data(t[:n], phi[:n])
        ln_va.set_data(t[:n], varphi_deg[:n])
        ln_ha.set_data(t[:n], halign_deg[:n])
        ln_rho.set_data(t[:n], rho[:n])
        return im, txt, ln_tau, ln_phi, ln_va, ln_ha, ln_rho

    anim = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)
    mp4 = os.path.join(out_dir, 'mpc_dashboard.mp4')
    gif = os.path.join(out_dir, 'mpc_dashboard.gif')
    try:
        if animation.writers.is_available('ffmpeg'):
            anim.save(mp4, writer=animation.FFMpegWriter(fps=6))
            plt.close(fig)
            return mp4
    except Exception as e:
        print(f"⚠️  ffmpeg dashboard failed ({e}); falling back to GIF.")
    anim.save(gif, writer=animation.PillowWriter(fps=6))
    plt.close(fig)
    return gif


# =============================================================================
#  Closed-loop driver
# =============================================================================
def _measure_state(simulator, mpc):
    """Read the reduced MPC state x0 from the simulator (population + morphology)."""
    cells = simulator.grid.cells
    pop_model = simulator.models['population']
    tau_now = simulator.input_pattern.get('value', 0.0)
    # sync the population compartments from the actual cells
    pop_model.update_from_cells(cells, dt=0.0, tau=tau_now)
    E = np.asarray(pop_model.state['E'], dtype=float)
    pop = np.concatenate([E, [pop_model.state['S_tel'], pop_model.state['S_stress']]])

    healthy = [c for c in cells.values() if not c.is_senescent]
    if healthy:
        rho_h = float(np.mean([c.actual_aspect_ratio for c in healthy]))
        # circular mean of orientation
        s = np.mean([np.sin(2 * c.actual_orientation) for c in healthy])
        co = np.mean([np.cos(2 * c.actual_orientation) for c in healthy])
        theta_h = 0.5 * np.arctan2(s, co)
    else:
        rho_h, theta_h = RHO_STAT, np.radians(THETA_STAT_DEG)
    return {'pop': pop, 'rho_h': rho_h, 'theta_h': theta_h}


def _reconcile_senescence(simulator, pop):
    """Convert healthy cells to senescent so cell counts match population state."""
    N = simulator.config.max_divisions
    want_str = int(round(pop[N + 2]))
    want_tel = int(round(pop[N + 1]))
    cells = list(simulator.grid.cells.values())
    have_str = sum(1 for c in cells if c.is_senescent and c.senescence_cause == 'stress')
    have_tel = sum(1 for c in cells if c.is_senescent and c.senescence_cause == 'telomere')
    spatial = simulator.models.get('spatial')
    pressure = simulator.input_pattern.get('value', 0.0)
    healthy = [c for c in cells if not c.is_senescent]
    np.random.shuffle(healthy)

    def convert(cell, cause):
        cell.is_senescent = True
        cell.senescence_cause = cause
        if cause == 'telomere':
            cell.divisions = N
        if spatial is not None:
            cell.target_area = spatial.calculate_target_area(pressure, True, cause)
            cell.target_orientation = spatial.calculate_target_orientation(pressure, True)
            cell.actual_orientation = cell.target_orientation

    for _ in range(max(0, want_str - have_str)):
        if healthy:
            convert(healthy.pop(), 'stress')
    for _ in range(max(0, want_tel - have_tel)):
        if healthy:
            convert(healthy.pop(), 'telomere')


def run_mpc_simulation(simulator, config, n_control_steps=6, output_dir=None,
                       render_minutes=(0, 15, 30, 45)):
    """
    Run the receding-horizon MPC closed loop on `simulator` (main.tex, eq:ocp).

    For each of `n_control_steps` 1 h intervals the OCP is solved, the first move
    tau is applied, and the closed-form step response is evaluated at the
    `render_minutes` sub-points (default 0/15/30/45) to produce smooth frames of
    the re-rendered Voronoi tessellation. After the run the frames are assembled
    into an animation and three summary figures are written.

    Returns a dict with the control/output log and the list of frame paths.
    """
    _apply_plot_style()
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
    frames_dir = os.path.join(output_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)

    mpc = RecedingHorizonMPC(config)
    temporal = simulator.models['temporal']

    # === REPRODUCIBILITY ===
    # Reseed both RNGs from the master seed (config.random_seed, default 42)
    # before drawing the per-cell heterogeneity deviates, so those standardized
    # normal deviates (z_theta, z_rho) are reproducible independently of how many
    # random draws initialisation consumed. Seed is recorded in the return value.
    seed = int(getattr(config, 'random_seed', 42))  # dimensionless
    random.seed(seed)
    np.random.seed(seed)
    print(f"🎲 MPC heterogeneity RNG seed: {seed}")

    # Per-cell heterogeneity: fix a standardised normal deviate per cell so each
    # cell tracks its own target = mean + z * std (spread tightens under flow).
    for c in simulator.grid.cells.values():
        c._z_theta = float(np.random.randn())
        c._z_rho = float(np.random.randn())

    # step-resolution log (hour boundaries) and sub-frame time series (for the dashboard)
    log = {'t_h': [0.0], 'tau': [], 'phi_sen': [], 'rho_bar': [], 'varphi_bar': [],
           'healthy_align': []}
    ts = {'t_h': [], 'tau': [], 'phi_sen': [], 'rho_bar': [], 'varphi_bar': [],
          'healthy_align': []}
    frames = []

    # --- authoritative reduced state (re-estimated from the model, not from a
    #     live PCA measurement of the tessellation; main.tex, sec. 2.4) ---------
    tau0 = simulator.input_pattern.get('value', 0.0)
    x_state = _measure_state(simulator, mpc)          # pop compartments from the cells
    # morphology initial condition = static baseline targets at the initial input
    x_state['rho_h'] = mpc.rho_target(tau0)           # rho_stat at tau0 = 0 Pa
    x_state['theta_h'] = mpc.theta_target(tau0)       # theta_stat at tau0 = 0 Pa

    phi0, rho0, varphi0 = mpc.outputs(x_state)
    log['phi_sen'].append(phi0)
    log['rho_bar'].append(rho0)
    log['varphi_bar'].append(varphi0)
    log['healthy_align'].append(flow_alignment_angle(x_state['theta_h']))
    print(f"▶ MPC start: phi_sen={phi0:.3f}, rho_bar={rho0:.3f}, "
          f"varphi_bar={np.degrees(varphi0):.1f} deg, "
          f"healthy_align={np.degrees(flow_alignment_angle(x_state['theta_h'])):.1f} deg")

    u_prev = tau0

    for k in range(n_control_steps):
        x0 = x_state
        u_opt, res = mpc.solve(x0, u_prev)
        tau_k = float(np.clip(u_opt[0], mpc.tau_min, mpc.tau_max))
        status = 'ok' if getattr(res, 'success', False) else 'fallback'

        simulator.set_constant_input(tau_k)

        # capture per-cell start-of-interval morphology and per-cell targets
        cells = simulator.grid.cells
        start = {}
        for cid, c in cells.items():
            if c.is_senescent:
                tgt_rho = RHO_SEN
                tgt_theta = c.actual_orientation  # senescent: no flow response
            else:
                tgt_rho = mpc.cell_rho_target(tau_k, getattr(c, '_z_rho', 0.0))
                tgt_theta = mpc.cell_theta_target(tau_k, getattr(c, '_z_theta', 0.0))
            start[cid] = (c.actual_aspect_ratio, c.actual_orientation, tgt_rho, tgt_theta)

        # intermediate frames via the closed-form step response (eq:stepsolution)
        for m in render_minutes:
            th = m / 60.0
            # reduced sub-state (for synced dashboard time series)
            x_sub = mpc.predict_step(x0, tau_k, dt=th)
            phi_s, rho_s, varphi_s = mpc.outputs(x_sub)
            halign_s = flow_alignment_angle(x_sub['theta_h'])
            t_abs = k + th
            ts['t_h'].append(t_abs); ts['tau'].append(tau_k)
            ts['phi_sen'].append(phi_s); ts['rho_bar'].append(rho_s)
            ts['varphi_bar'].append(varphi_s); ts['healthy_align'].append(halign_s)

            # update actual cell morphology toward each cell's own target
            for cid, c in cells.items():
                rho0c, th0c, tgt_rho, tgt_theta = start[cid]
                c.actual_aspect_ratio = temporal.relax_step(rho0c, tgt_rho, th)
                c.actual_orientation = temporal.orientation_step(th0c, tgt_theta, th,
                                                                 mpc.tau_orient)
            simulator.grid._update_voronoi_tessellation(preserve_temporal_dynamics=True)
            path = os.path.join(frames_dir, f"mpc_k{k:02d}_t{m:02d}.pdf")
            arr = _render_frame(simulator.grid, tau_k, t_abs, phi_s, path)
            frames.append({'arr': arr.copy(), 't_h': t_abs, 'tau': tau_k,
                           'phi_sen': phi_s, 'snap': len(ts['t_h']), 'path': path})

        # advance the cells (for visualisation only) to the end of the interval
        for cid, c in cells.items():
            rho0c, th0c, tgt_rho, tgt_theta = start[cid]
            c.actual_aspect_ratio = temporal.relax_step(rho0c, tgt_rho, mpc.dt_h)
            c.actual_orientation = temporal.orientation_step(th0c, tgt_theta, mpc.dt_h,
                                                             mpc.tau_orient)
            c.target_aspect_ratio = tgt_rho
            c.target_orientation = tgt_theta

        # advance the authoritative reduced state by 1 h (eq:flowmap)
        x_state = mpc.predict_step(x0, tau_k)
        pop_end = x_state['pop']
        pm = simulator.models['population']
        pm.state['E'] = list(pop_end[:mpc.N + 1])
        pm.state['S_tel'] = float(pop_end[mpc.N + 1])
        pm.state['S_stress'] = float(pop_end[mpc.N + 2])
        _reconcile_senescence(simulator, pop_end)
        simulator.grid._update_voronoi_tessellation(preserve_temporal_dynamics=True)
        simulator.time += 60.0  # minutes

        # log model-propagated outputs at the hour boundary
        phi_m, rho_m, varphi_m = mpc.outputs(x_state)
        halign_m = flow_alignment_angle(x_state['theta_h'])
        log['t_h'].append(k + 1.0)
        log['tau'].append(tau_k)
        log['phi_sen'].append(phi_m)
        log['rho_bar'].append(rho_m)
        log['varphi_bar'].append(varphi_m)
        log['healthy_align'].append(halign_m)
        u_prev = tau_k
        print(f"  step {k:02d}: tau={tau_k:.3f} Pa  (SLSQP {status}, J={res.fun:.3f})  "
              f"phi_sen={phi_m:.3f}  rho_bar={rho_m:.3f}  "
              f"healthy_align={np.degrees(halign_m):.1f} deg")

    # assemble outputs
    print(f"🎞️  Assembling tessellation animation from {len(frames)} frames ...")
    anim_path = _build_animation(frames, output_dir)
    print(f"   animation -> {anim_path}")
    print(f"🖥️  Assembling composite dashboard ...")
    dash_path = _build_dashboard(frames, ts, mpc, output_dir)
    print(f"   dashboard -> {dash_path}")
    _summary_plots(log, output_dir)
    print(f"📈 Summary figures written to {output_dir}")

    return {'log': log, 'ts': ts, 'frames': [f['path'] for f in frames],
            'animation': anim_path, 'dashboard': dash_path, 'output_dir': output_dir,
            'seed': seed}   # master RNG seed used for this run (reproducibility)

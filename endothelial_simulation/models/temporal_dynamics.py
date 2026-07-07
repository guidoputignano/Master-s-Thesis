"""
Temporal dynamics model for endothelial cell adaptation to mechanical stimuli.
Enhanced with scaled time constants for different biological properties.
"""
import numpy as np
import warnings
from scipy.integrate import odeint
from scipy.optimize import minimize


# --- Legacy (path B) deprecation plumbing -------------------------------------
# DEPRECATED symbols: the per-cell "response" model (calculate_A_max,
# calculate_tau with tau = tau_base * A_max**lambda_scale, update_cell_responses,
# and the get_*tau_and_amax wrappers built on them) is NOT used by
# run_mpc_simulation / RecedingHorizonMPC (the reported model, path A). The
# reported dynamics use relax_step / orientation_step + the reduced population
# ODEs instead. These functions survive only for the legacy CLI simulation modes
# (Simulator.step) and the sensitivity scripts, so they are kept importable and
# functional but emit a one-shot DeprecationWarning on entry. This leaves a
# single, unambiguous representation of the reported model.
_LEGACY_WARNED = set()


def _warn_legacy_once(name, detail=""):
    """Emit a one-shot DeprecationWarning for a legacy (path-B) symbol.

    One-shot (guarded by a module-level set) so per-step / per-cell callers such
    as Simulator.step do not flood the log.
    """
    if name not in _LEGACY_WARNED:
        _LEGACY_WARNED.add(name)
        warnings.warn(
            f"{name} is DEPRECATED: not used by run_mpc_simulation / the reported "
            f"model (path A); retained only for the legacy path (path B). {detail}",
            DeprecationWarning, stacklevel=2)


class TemporalDynamicsModel:
    """
    Model for time-dependent adaptation of endothelial cells to mechanical stimuli.

    This model captures how cellular responses evolve following pressure changes,
    implementing the first-order differential equation described in the thesis.
    Enhanced with scaled time constants for different biological properties.
    """

    def __init__(self, config):
        """
        Initialize the temporal dynamics model with configuration parameters.

        Parameters:
            config: SimulationConfig object with parameter settings
        """
        self.config = config

        # Known pressure values and their corresponding Amax values
        self.P_values = np.array(config.known_pressures)
        self.A_max_map = config.known_A_max

        # Initial response value
        self.y0 = config.initial_response

        # Time constant parameters
        self.tau_base = config.tau_base
        self.lambda_scale = config.lambda_scale

        # Paper (Table 1, main.tex) temporal-dynamics parameters
        self.tau_act = getattr(config, 'tau_act', 0.5)              # Source: Table 1, main.tex — tau_act = 0.5 Pa
        self.tau_adapt_hours = getattr(config, 'tau_adapt_hours', 9.0)  # Source: Table 1, main.tex — tau_adapt = 6-12 h

        # Calculate linear model parameters for A_max
        P_known = np.array(list(self.A_max_map.keys()))
        A_max_known = np.array(list(self.A_max_map.values()))
        self.slope, self.intercept = np.polyfit(P_known, A_max_known, 1)

        # NEW: Biological time scale factors for different properties
        # Start with all factors = 1.0 (unified approach)
        # You can modify these later for realistic scaling
        self.time_scale_factors = {
            'biochemical': 1.0,      # Base - your fitted data
            'area': 1.0,             # Cell volume/area changes
            'orientation': 0.1,      # Cell orientation/alignment (reduced for faster adaptation)
            'aspect_ratio': 1.0,     # Cell elongation/shape
        }

    def s_activation(self, tau, tau_act=None):
        """
        Monotone activation gate s(tau) of eq:target (main.tex):

            s(tau) = 0                              for tau <= tau_act
            s(tau) = 1 - exp(-(tau - tau_act)/tau_act)  for tau >  tau_act   -> 1 as tau >> tau_act

        Cells reorganize only above the activation shear stress tau_act; below it
        the morphology remains isotropic.

        Parameters:
            tau: wall shear stress (Pa)
            tau_act: activation threshold (Pa); defaults to Table-1 value (0.5 Pa)

        Returns:
            Activation in [0, 1).
        """
        if tau_act is None:
            tau_act = self.tau_act  # Source: Table 1, main.tex — tau_act = 0.5 Pa
        if tau <= tau_act:
            return 0.0
        return 1.0 - np.exp(-(tau - tau_act) / tau_act)

    def gated_target(self, y_stat, y_flow, tau, tau_act=None):
        """
        Gated interpolation target y*(tau) of eq:target (main.tex):

            y*(tau) = y_stat + (y_flow - y_stat) * s(tau)

        Parameters:
            y_stat: static (no-flow) baseline value
            y_flow: flow-adapted plateau value
            tau:    wall shear stress (Pa)
            tau_act: activation threshold (Pa)

        Returns:
            Stimulus-dependent target value.
        """
        return y_stat + (y_flow - y_stat) * self.s_activation(tau, tau_act)

    def relax_step(self, y, y_target, dt, tau_adapt=None):
        """
        Closed-form first-order relaxation step (eq:relaxation / eq:stepsolution):

            y(t+dt) = y_target - (y_target - y) * exp(-dt / tau_adapt)

        Parameters:
            y:         current value
            y_target:  stimulus-dependent target y*(tau)
            dt:        time step (hours, paper units)
            tau_adapt: adaptation time constant (hours); defaults to Table-1 nominal

        Returns:
            Updated value after dt.
        """
        if tau_adapt is None:
            tau_adapt = self.tau_adapt_hours  # Source: Table 1, main.tex — tau_adapt = 6-12 h
        return y_target - (y_target - y) * np.exp(-dt / tau_adapt)

    def orientation_step(self, theta, theta_target, dt, tau_adapt=None):
        """
        Angular first-order relaxation on the circle (eq:orientation, main.tex)
        using the shortest-arc wrapping operator
            <psi> = ((psi + pi) mod 2*pi) - pi.

        Parameters:
            theta:        current orientation (radians)
            theta_target: target orientation theta*(tau) (radians)
            dt:           time step (hours)
            tau_adapt:    adaptation time constant (hours)

        Returns:
            Updated orientation (radians).
        """
        if tau_adapt is None:
            tau_adapt = self.tau_adapt_hours
        # shortest-arc difference
        diff = ((theta_target - theta) + np.pi) % (2 * np.pi) - np.pi
        return theta + diff * (1.0 - np.exp(-dt / tau_adapt))

    def calculate_A_max(self, P):
        """
        DEPRECATED (legacy path B): not used by run_mpc_simulation / the reported
        model. Retained only for the legacy CLI simulation modes and sensitivity
        scripts.

        Calculate the maximum response (steady-state value) for a given pressure.

        Uses a hybrid approach with direct lookup for known pressures and linear
        interpolation for unknown pressures.

        Parameters:
            P: Pressure value (Pa)

        Returns:
            Maximum attainable response at the given pressure
        """
        _warn_legacy_once("TemporalDynamicsModel.calculate_A_max",
                          "The reported model uses gated static->flow targets, not A_max.")
        # For known pressure values, use the original A_max from the map
        if P in self.A_max_map:
            return self.A_max_map[P]
        else:
            # For unknown P values, use linear interpolation/extrapolation
            # Ensure A_max is non-negative (minimum value of 1)
            return max(1, self.slope * P + self.intercept)

    def calculate_tau(self, A_max):
        """
        DEPRECATED (legacy path B): not used by run_mpc_simulation / the reported
        model. The reported dynamics use a single fixed morphological adaptation
        time constant (7.4 h; tau_orient = tau_adapt), not tau = tau_base * A_max**lambda_scale.
        Retained only for the legacy CLI simulation modes and sensitivity scripts.

        Calculate the time constant based on A_max.

        Time constant scales with A_max following a power law relationship.

        Parameters:
            A_max: Maximum attainable response

        Returns:
            Time constant (tau) value
        """
        _warn_legacy_once("TemporalDynamicsModel.calculate_tau",
                          "tau = tau_base * A_max**lambda_scale (legacy power law).")
        # Reference value is 1.0
        return self.tau_base * (A_max ** self.lambda_scale)

    def get_tau_and_amax(self, pressure):
        """
        Get time constant and A_max for given pressure.
        This method provides backward compatibility.

        Parameters:
            pressure: Pressure value in Pa

        Returns:
            tuple: (tau, A_max) using biochemical scaling (base)
        """
        return self.get_scaled_tau_and_amax(pressure, 'biochemical')

    def get_scaled_tau_and_amax(self, pressure, property_type='biochemical'):
        """
        Get time constant scaled for specific biological property.

        Parameters:
            pressure: Applied pressure (Pa)
            property_type: Type of property ('biochemical', 'area', 'orientation', 'aspect_ratio')

        Returns:
            tuple: (scaled_tau, A_max)
        """
        # Base calculation using your experimental data
        A_max = self.calculate_A_max(pressure)
        base_tau = self.calculate_tau(A_max)

        # Apply biological scaling
        scale_factor = self.time_scale_factors.get(property_type, 1.0)
        scaled_tau = base_tau * scale_factor

        return scaled_tau, A_max

    def model(self, y, t, P):
        """
        First-order differential equation model for cellular response.

        dy/dt = (A_max - y) / tau

        Parameters:
            y: Current response value
            t: Time
            P: Pressure value (Pa)

        Returns:
            Rate of change of response (dy/dt)
        """
        # Calculate A_max for this pressure
        A_max = self.calculate_A_max(P)

        # Calculate tau based on A_max
        tau = self.calculate_tau(A_max)

        # Differential equation: dy/dt = (A_max - y) / tau
        dydt = (A_max - y) / tau

        return dydt

    def simulate(self, P, y0=None, t_span=(0, 8), t_points=100):
        """
        Simulate the cellular response to a constant pressure over time.

        Parameters:
            P: Pressure value (Pa)
            y0: Initial response value, uses default if None
            t_span: Time range (start, end) in arbitrary units
            t_points: Number of time points to evaluate

        Returns:
            t: Time points
            y: Response values at each time point
        """
        # Use default initial value if none provided
        if y0 is None:
            y0 = self.y0

        # Create time points for evaluation
        t = np.linspace(t_span[0], t_span[1], t_points)

        # Solve the ODE
        solution = odeint(self.model, y0, t, args=(P,))

        # Flatten the solution array
        y = solution.flatten()

        return t, y

    def simulate_step_response(self, P_initial, P_final, t_step, y0=None, t_span=(0, 12), t_points=100):
        """
        Simulate the cellular response to a step change in pressure.

        Parameters:
            P_initial: Initial pressure value (Pa)
            P_final: Final pressure value after step (Pa)
            t_step: Time at which the step occurs
            y0: Initial response value, uses default if None
            t_span: Time range (start, end) in arbitrary units
            t_points: Number of time points to evaluate

        Returns:
            t: Time points
            y: Response values at each time point
            P: Pressure at each time point
        """
        # Use default initial value if none provided
        if y0 is None:
            y0 = self.y0

        # Create time points for evaluation
        t = np.linspace(t_span[0], t_span[1], t_points)

        # Initialize arrays for solution and pressure
        y = np.zeros_like(t)
        P = np.zeros_like(t)

        # First phase with P_initial
        t1_indices = t <= t_step
        t1 = t[t1_indices]
        P[t1_indices] = P_initial

        if len(t1) > 0:
            sol1 = odeint(self.model, y0, t1, args=(P_initial,))
            y[t1_indices] = sol1.flatten()

        # Second phase with P_final
        t2_indices = t > t_step
        t2 = t[t2_indices]
        P[t2_indices] = P_final

        if len(t2) > 0:
            # Start second phase from end of first phase
            y0_2 = y[t1_indices][-1] if len(t1) > 0 else y0

            # Adjust time to start from 0 for ODE solver
            t2_adjusted = t2 - t_step

            sol2 = odeint(self.model, y0_2, t2_adjusted, args=(P_final,))
            y[t2_indices] = sol2.flatten()

        return t, y, P

    def simulate_ramp_response(self, P_initial, P_final, t_ramp_start, t_ramp_end,
                               y0=None, t_span=(0, 16), t_points=100):
        """
        Simulate the cellular response to a ramp change in pressure.

        Parameters:
            P_initial: Initial pressure value (Pa)
            P_final: Final pressure value after ramp (Pa)
            t_ramp_start: Time at which the ramp begins
            t_ramp_end: Time at which the ramp ends
            y0: Initial response value, uses default if None
            t_span: Time range (start, end) in arbitrary units
            t_points: Number of time points to evaluate

        Returns:
            t: Time points
            y: Response values at each time point
            P: Pressure at each time point
        """
        # Use default initial value if none provided
        if y0 is None:
            y0 = self.y0

        # Create time points for evaluation
        t = np.linspace(t_span[0], t_span[1], t_points)

        # Initialize array for solution
        y = np.zeros_like(t)

        # Calculate pressure at each time point
        P = np.zeros_like(t)

        # Phase 1: Initial pressure
        mask1 = t <= t_ramp_start
        P[mask1] = P_initial

        # Phase 2: Ramp
        mask2 = (t > t_ramp_start) & (t <= t_ramp_end)
        ramp_duration = t_ramp_end - t_ramp_start
        P[mask2] = P_initial + (P_final - P_initial) * (t[mask2] - t_ramp_start) / ramp_duration

        # Phase 3: Final pressure
        mask3 = t > t_ramp_end
        P[mask3] = P_final

        # Solve the ODE numerically for each small time step
        y[0] = y0

        for i in range(1, len(t)):
            # Use the previous value as initial condition
            y0_i = y[i - 1]

            # Time interval for this step
            t_i = np.array([t[i - 1], t[i]])

            # Average pressure during this interval
            P_avg = (P[i - 1] + P[i]) / 2

            # Solve for this small interval
            sol_i = odeint(self.model, y0_i, t_i, args=(P_avg,))

            # Store the result
            y[i] = sol_i[1]

        return t, y, P

    def update_cell_responses(self, cells, P, dt):
        """
        DEPRECATED (legacy path B): not used by run_mpc_simulation / the reported
        model. Only Simulator.step() (the non-MPC CLI simulation modes) calls this
        per-cell scalar-response update; the reported model tracks morphology
        (aspect ratio / orientation) + the population ODEs instead.

        Update the response values of all cells based on the current pressure.

        Parameters:
            cells: Dictionary of Cell objects
            P: Current pressure value (Pa)
            dt: Time step

        Returns:
            Dictionary mapping cell_id to new response value
        """
        _warn_legacy_once("TemporalDynamicsModel.update_cell_responses",
                          "The reported model does not use a per-cell scalar response.")
        updated_responses = {}

        for cell_id, cell in cells.items():
            # Current response
            y0 = cell.response

            # Calculate parameters
            A_max = self.calculate_A_max(P)
            tau = self.calculate_tau(A_max)

            # Analytical solution for one time step
            y_new = A_max - (A_max - y0) * np.exp(-dt / tau)

            # Store new response
            updated_responses[cell_id] = y_new

            # Update cell response
            cell.update_response(y_new)

        return updated_responses

    def fit_parameters(self, experimental_data, initial_params=None, bounds=None):
        """
        Fit model parameters to experimental data.

        Parameters:
            experimental_data: Dictionary mapping pressure values to time and response data
            initial_params: Dictionary of initial parameter values
            bounds: Dictionary of parameter bounds (min, max)

        Returns:
            optimized_params: Dictionary of optimized parameter values
        """
        if initial_params is None:
            initial_params = {'tau_base': self.tau_base, 'lambda_scale': self.lambda_scale}

        if bounds is None:
            bounds = {'tau_base': (0.1, 5.0), 'lambda_scale': (0.1, 2.0)}

        # Extract parameters and bounds for optimizer
        param_names = ['tau_base', 'lambda_scale']
        initial_values = [initial_params[name] for name in param_names]
        param_bounds = [(bounds[name][0], bounds[name][1]) for name in param_names]

        # Define objective function for minimization
        def objective_function(params):
            # Set model parameters
            tau_base, lambda_scale = params
            tau_base_original = self.tau_base
            lambda_scale_original = self.lambda_scale

            self.tau_base = tau_base
            self.lambda_scale = lambda_scale

            # Calculate error for each pressure condition
            total_error = 0

            for P, data in experimental_data.items():
                # Get experimental data for this pressure
                t_exp = data['t']
                y_exp = data['y']

                # Simulate model response
                _, y_model = self.simulate(P, t_span=(t_exp[0], t_exp[-1]), t_points=len(t_exp))

                # Calculate sum of squared errors
                error = np.sum((y_model - y_exp) ** 2)
                total_error += error

            # Restore original parameters
            self.tau_base = tau_base_original
            self.lambda_scale = lambda_scale_original

            return total_error

        # Run optimization
        result = minimize(
            objective_function,
            initial_values,
            method='L-BFGS-B',
            bounds=param_bounds
        )

        # Extract optimized parameters
        optimized_values = result.x
        optimized_params = {name: value for name, value in zip(param_names, optimized_values)}

        # Update model parameters
        self.tau_base = optimized_params['tau_base']
        self.lambda_scale = optimized_params['lambda_scale']

        # Update config parameters
        self.config.tau_base = self.tau_base
        self.config.lambda_scale = self.lambda_scale

        return optimized_params

    def get_parameters(self):
        """
        Get current model parameters.

        Returns:
            Dictionary of parameter values
        """
        return {
            'tau_base': self.tau_base,
            'lambda_scale': self.lambda_scale,
            'slope': self.slope,
            'intercept': self.intercept,
            'time_scale_factors': self.time_scale_factors
        }

    def set_time_scale_factors(self, factors):
        """
        Set time scale factors for different properties.

        Parameters:
            factors: Dictionary with keys 'biochemical', 'area', 'orientation', 'aspect_ratio'
        """
        for key, value in factors.items():
            if key in self.time_scale_factors:
                self.time_scale_factors[key] = value
            else:
                print(f"Warning: Unknown property type '{key}' for time scaling")
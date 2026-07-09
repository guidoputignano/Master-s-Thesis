"""
Population dynamics model for endothelial cell mechanotransduction.
"""
import numpy as np


def gamma_tau_hill(tau, gamma_min, gamma_max, tau_h, n,
                   gamma_d=0.0, tau_d=None, m=2):
    """
    Monotone-decreasing, bounded Hill senescence-induction rate.

        gamma(tau) = gamma_min + (gamma_max - gamma_min) * tau_h^n / (tau_h^n + tau^n)
                     [ + gamma_d * tau^m / (tau_d^m + tau^m) ]   (optional damage arm)

    Low shear (atheroprone) gives high induction (-> gamma_max as tau -> 0); high
    shear (atheroprotective) gives low induction (-> gamma_min as tau -> inf).

    IMPORTANT (provenance): the Hill FORM is a modelling choice matched to the
    cited monotone shear-protection shape (KLF2/P2X4 flow signalling; atheroprone
    vs atheroprotective wall-shear boundary). It is NOT an equation fitted to data
    in any single source. `gamma_min` and `tau_h` are anchored to cited
    values/thresholds; `gamma_max` (and the optional `gamma_d`) are assumed,
    illustrative and sweepable — not fitted or measured.

    Pure-numpy helper (no object access) so the MPC optimiser can call it inside
    the ODE right-hand side without Python/attribute overhead.

    Parameters:
        tau:       wall shear stress (Pa), scalar or array
        gamma_min: high-shear floor plateau rate (h^-1)   [tau -> inf]
        gamma_max: low-shear plateau rate (h^-1)          [tau -> 0]
        tau_h:     half-max shear of the protective Hill (Pa)
        n:         protective Hill exponent (shape constant)
        gamma_d:   optional supraphysiological damage amplitude (h^-1); 0 disables it
        tau_d:     damage half-max shear (Pa); used only when gamma_d > 0
        m:         damage Hill exponent

    Returns:
        senescence-induction rate gamma(tau) (h^-1), same shape as tau.
    """
    tau_h_n = tau_h ** n
    gamma = gamma_min + (gamma_max - gamma_min) * tau_h_n / (tau_h_n + tau ** n)
    if gamma_d:
        gamma = gamma + gamma_d * tau ** m / (tau_d ** m + tau ** m)
    return gamma


def population_reduced_rhs(state, tau, r, K, gamma_min, gamma_max, tau_h, n, N,
                           include_replicative=True, model_growth=False,
                           gamma_d=0.0, tau_d=None, m=2):
    """
    Standalone right-hand side of the reduced population ODE (eq:reduced, main.tex),
    after the Task 5 refactor (Hill induction, xi removed, structure flags).

    This is the control-oriented kernel used by the MPC optimiser: it takes only
    numpy arrays / scalars (no Grid, no model object) so scipy.integrate.solve_ivp
    can call it with minimal overhead. The senescence-induction rate is the SAME
    Hill law (`gamma_tau_hill`) used by the per-cell population update.

    Replicative arm ON (default), growth-to-confluence per `model_growth`:
        dE_i/dt   = 2 r g E_{i-1} - r g E_i - gamma(tau) E_i
        dS_tel/dt = r g E_N
        dS_str/dt = sum_i gamma(tau) E_i = gamma(tau) N_E
        g         = 1/(1 + N_E/K)  if model_growth else 1   (contact inhibition)

    Replicative arm OFF: no division and no telomere senescence; the healthy pool
    N_E is depleted only by stress-induced senescence:
        dE_i/dt   = -gamma(tau) E_i,   dS_tel/dt = 0,   dS_str/dt = gamma(tau) N_E

    Parameters:
        state: array [E_0, ..., E_N, S_tel, S_str]  (length N + 3)
        tau:   wall shear stress (Pa)
        r, K, N: proliferation rate, carrying capacity (count units), Hayflick N
        gamma_min, gamma_max, tau_h, n, gamma_d, tau_d, m: Hill-law parameters
        include_replicative: keep the E ladder + S_tel (True) or collapse (False)
        model_growth: keep the density factor g(N_E) (True) or set g == 1 (False)

    Returns:
        numpy array of derivatives, same layout as `state`.
    """
    state = np.asarray(state, dtype=float)
    E = state[:N + 1]
    N_E = float(E.sum())

    # Contact-inhibition density factor. When growth-to-confluence is not
    # modelled, g == 1 (proliferation is density-independent); see config flag.
    g = 1.0 / (1.0 + N_E / K) if model_growth else 1.0

    # Senescence-induction rate (identical Hill law to the per-cell update).
    gamma = gamma_tau_hill(tau, gamma_min, gamma_max, tau_h, n,
                           gamma_d=gamma_d, tau_d=tau_d, m=m)

    dstate = np.empty_like(state)
    if include_replicative:
        dE = np.empty(N + 1)
        # division in: 2 r g E_{i-1}; division out: r g E_i; senescence loss: gamma E_i
        dE[0] = -r * g * E[0] - gamma * E[0]
        dE[1:] = 2.0 * r * g * E[:-1] - r * g * E[1:] - gamma * E[1:]
        dstate[:N + 1] = dE
        dstate[N + 1] = r * g * E[N]        # terminal division -> S_tel
        dstate[N + 2] = gamma * N_E         # stress-induced senescence (xi=0)
    else:
        # Single healthy pool: no division, no telomere senescence.
        dstate[:N + 1] = -gamma * E
        dstate[N + 1] = 0.0                 # S_tel inert (replicative arm disabled)
        dstate[N + 2] = gamma * N_E         # stress-induced senescence
    return dstate


class PopulationDynamicsModel:
    """
    Model for endothelial cell population dynamics under mechanical stimuli.

    This model tracks the evolution of different cell populations (healthy cells at different
    division stages, telomere-induced senescent cells, and stress-induced senescent cells)
    in response to mechanical stimuli and other factors.
    """

    def __init__(self, config):
        """
        Initialize the population dynamics model with configuration parameters.

        Parameters:
            config: SimulationConfig object with parameter settings
        """
        self.config = config

        # Model parameters
        self.max_divisions = config.max_divisions
        self.r = config.proliferation_rate
        self.K = config.carrying_capacity

        # Model-structure flags (Task 5); default to the reported structure.
        self.include_replicative_arm = getattr(config, 'INCLUDE_REPLICATIVE_ARM', True)
        self.model_growth_to_confluence = getattr(config, 'MODEL_GROWTH_TO_CONFLUENCE', False)
        self.include_supraphysiological_arm = getattr(config, 'INCLUDE_SUPRAPHYSIOLOGICAL_ARM', False)

        # Senescence-induction rate gamma(tau): monotone-decreasing bounded Hill.
        # The Hill FORM is a modelling choice matched to the cited shear-protection
        # shape, not a fitted law (see gamma_tau_hill / config for provenance).
        self.gamma_min = getattr(config, 'gamma_min', 0.00278)   # h^-1 [anchored]
        self.gamma_max = getattr(config, 'gamma_max', 0.0125)    # h^-1 [assumed, sweepable]
        self.tau_h_sen = getattr(config, 'tau_h_sen', 0.5)       # Pa   [anchored]
        self.n_hill = getattr(config, 'n_hill', 2)               # -    [fixed shape]
        self.gamma_d = getattr(config, 'gamma_d', 0.0125)        # h^-1 [assumed, sweepable]
        self.tau_d = getattr(config, 'tau_d', 7.0)               # Pa   [assumed]
        self.m_hill = getattr(config, 'm_hill', 2)               # -    [fixed shape]

        # Initialize population state
        self.initialize_state()

    def initialize_state(self, initial_distribution=None):
        """
        Initialize the population state.

        Parameters:
            initial_distribution: Optional dictionary with initial population counts
        """
        # State variables:
        # E[i] for i=0..max_divisions: Healthy cells at division stage i
        # S_tel: Telomere-induced senescent cells
        # S_stress: Stress-induced senescent cells

        # Create state dictionary with zeros
        self.state = {
            'E': [0] * (self.max_divisions + 1),
            'S_tel': 0,
            'S_stress': 0
        }

        # Set initial values if provided
        if initial_distribution is not None:
            if 'E' in initial_distribution:
                # Handle array or list
                if isinstance(initial_distribution['E'], (list, np.ndarray)):
                    for i in range(min(len(initial_distribution['E']), self.max_divisions + 1)):
                        self.state['E'][i] = initial_distribution['E'][i]
                # Handle dictionary with indices
                elif isinstance(initial_distribution['E'], dict):
                    for i, count in initial_distribution['E'].items():
                        if 0 <= i <= self.max_divisions:
                            self.state['E'][i] = count

            if 'S_tel' in initial_distribution:
                self.state['S_tel'] = initial_distribution['S_tel']

            if 'S_stress' in initial_distribution:
                self.state['S_stress'] = initial_distribution['S_stress']

    def calculate_total_cells(self):
        """
        Calculate the total number of cells.

        Returns:
            Dictionary with totals for different cell categories
        """
        E_total = sum(self.state['E'])
        S_tel = self.state['S_tel']
        S_stress = self.state['S_stress']
        S_total = S_tel + S_stress
        total_cells = E_total + S_total

        return {
            'E_total': E_total,
            'S_tel': S_tel,
            'S_stress': S_stress,
            'S_total': S_total,
            'total': total_cells
        }

    def calculate_average_division_age(self):
        """
        Calculate the average division age of healthy cells.

        Returns:
            Average division age
        """
        E_total = sum(self.state['E'])

        if E_total > 0:
            weighted_sum = sum(i * self.state['E'][i] for i in range(self.max_divisions + 1))
            return weighted_sum / E_total
        else:
            return np.nan

    def calculate_telomere_length(self):
        """
        Calculate the average telomere length based on division age.

        Returns:
            Average telomere length
        """
        avg_division_age = self.calculate_average_division_age()

        if np.isnan(avg_division_age):
            return np.nan

        max_telomere = 100
        min_telomere = 20

        # Normalize division age to 0-1 range
        normalized_age = min(max(0, avg_division_age / self.max_divisions), 1)

        # Calculate telomere length (decreases with division age)
        telomere_length = max_telomere - (max_telomere - min_telomere) * normalized_age

        return telomere_length

    def calculate_shear_stress_effect(self, tau):
        """
        Senescence-induction rate gamma(tau) (h^-1): monotone-decreasing Hill.

        Delegates to the SAME `gamma_tau_hill` used by the reduced kernel and the
        MPC rollout, so the per-cell update and the control-oriented prediction
        share one induction law. The supraphysiological damage arm is included
        only when INCLUDE_SUPRAPHYSIOLOGICAL_ARM is set.

        Parameters:
            tau: Shear stress value (Pa)

        Returns:
            Senescence induction rate gamma(tau) (h^-1)
        """
        gamma_d = self.gamma_d if self.include_supraphysiological_arm else 0.0
        return gamma_tau_hill(tau, self.gamma_min, self.gamma_max,
                              self.tau_h_sen, self.n_hill,
                              gamma_d=gamma_d, tau_d=self.tau_d, m=self.m_hill)

    # Task 5 (Part C): the senolytic response curves and the stem-cell input
    # distribution have been removed with the rest of the treatment machinery
    # (calculate_senolytic_effect, calculate_senolytic_toxicity,
    # calculate_stem_cell_distribution). The reported six-hour regime is
    # treatment-free, so these terms were dead and are deleted rather than
    # zero-weighted.

    def calculate_density_factor(self, N_E):
        """
        Calculate density-dependent (contact-inhibition) factor for cell division.

        Parameters:
            N_E: Healthy cell count/density (sum of E_i), per eq:density

        Returns:
            Density factor g(N_E) in (0, 1]
        """
        # eq:density, main.tex — contact inhibition through the density factor
        #   g(N_E) = 1 / (1 + N_E / K)
        # Source: Table 1, main.tex — K = 5-6e4 cells/cm^2
        return 1.0 / (1.0 + N_E / self.K)

    def reduced_rhs(self, state, tau):
        """
        Right-hand side of the reduced population ODE (eq:reduced, main.tex) in
        the parallel-plate, six-hour, treatment-free regime (Task 5 form: Hill
        induction, xi removed, structure flags):

            dE_i/dt   = 2 r g E_{i-1} - r g E_i - gamma(tau) E_i   (replicative arm)
            dS_tel/dt = r g E_N
            dS_str/dt = sum_i gamma(tau) E_i = gamma(tau) N_E

        with g == 1 unless MODEL_GROWTH_TO_CONFLUENCE, and the ladder + S_tel
        present only when INCLUDE_REPLICATIVE_ARM. Units are the paper's: rates in
        h^-1, tau in Pa.

        Parameters:
            state: array-like [E_0, ..., E_N, S_tel, S_str] (length N+3)
            tau:   wall shear stress (Pa)

        Returns:
            numpy array of time derivatives, same layout as `state`.
        """
        # Delegate to the standalone numpy-only kernel (see population_reduced_rhs).
        gamma_d = self.gamma_d if self.include_supraphysiological_arm else 0.0
        return population_reduced_rhs(
            state, tau,
            r=self.r, K=self.K,
            gamma_min=self.gamma_min, gamma_max=self.gamma_max,
            tau_h=self.tau_h_sen, n=self.n_hill, N=self.max_divisions,
            include_replicative=self.include_replicative_arm,
            model_growth=self.model_growth_to_confluence,
            gamma_d=gamma_d, tau_d=self.tau_d, m=self.m_hill,
        )

    def calculate_division_capacity(self, division_stage):
        """
        Calculate division capacity based on division stage.

        Parameters:
            division_stage: Number of divisions completed

        Returns:
            Division capacity factor (0-1)
        """
        if division_stage <= 0.7 * self.max_divisions:
            return 1.0
        else:
            # Linear reduction in division capacity as cells approach max divisions
            return 1.0 - 0.5 * ((division_stage - 0.7 * self.max_divisions) /
                                (0.3 * self.max_divisions))

    def update(self, dt, tau, stem_cell_rate=None):
        """
        Advance the population state by one explicit-Euler step (legacy /
        per-cell-sync path).

        Task 5 (Part C) pruned this to the treatment-free reduced law: no death,
        no SASP, no senolytics, no stem-cell input, no per-stage age modulation
        and no xi stage weighting. The senescence-induction rate is the SAME Hill
        law used by the reduced kernel and the MPC rollout
        (calculate_shear_stress_effect -> gamma_tau_hill). Division terms are
        active only when the replicative arm is enabled; the contact-inhibition
        density factor is active only when growth-to-confluence is modelled.

            dE_i/dt   = 2 r g E_{i-1} - r g E_i - gamma(tau) E_i   (replicative arm)
            dS_tel/dt = r g E_N
            dS_str/dt = sum_i gamma(tau) E_i

        Parameters:
            dt: time step (h). dt == 0 performs a pure state<-cell-count sync.
            tau: shear stress value (Pa).
            stem_cell_rate: accepted for API compatibility; ignored (no stem input).

        Returns:
            Updated state dict.
        """
        # Get current state variables
        E = self.state['E'].copy()
        S_tel = self.state['S_tel']
        S_stress = self.state['S_stress']

        E_total = sum(E)  # N_E (healthy count) used by the density factor
        # Density factor g(N_E) only when growth-to-confluence is modelled; else 1.
        g = (self.calculate_density_factor(E_total)
             if self.model_growth_to_confluence else 1.0)
        gamma_tau = self.calculate_shear_stress_effect(tau)  # shared Hill law

        # Initialize arrays for new state
        E_new = [0] * (self.max_divisions + 1)
        S_tel_new = 0.0
        S_stress_new = 0.0

        # Update each division stage (xi removed: stage-independent senescence).
        for i in range(self.max_divisions + 1):
            stress_senescence_term = gamma_tau * E[i]

            # Division terms (eq:Ei): active only with the replicative arm.
            if self.include_replicative_arm:
                division_out = self.r * g * E[i]
                division_in = 0.0 if i == 0 else 2 * self.r * g * E[i - 1]
            else:
                division_out = 0.0
                division_in = 0.0

            if i == self.max_divisions:
                # Terminal stage: divisions lead to telomere-induced senescence.
                E_new[i] = E[i] + dt * (division_in - division_out - stress_senescence_term)
                S_tel_new += dt * division_out
            else:
                E_new[i] = E[i] + dt * (division_in - division_out - stress_senescence_term)

            # Stress-induced senescence accumulation.
            S_stress_new += dt * stress_senescence_term

        # Carry senescent compartments forward (no death, no senolytic clearance).
        S_tel_new += S_tel
        S_stress_new += S_stress

        # Ensure non-negative values
        E_new = [max(0, e) for e in E_new]
        S_tel_new = max(0, S_tel_new)
        S_stress_new = max(0, S_stress_new)

        # Update state
        self.state['E'] = E_new
        self.state['S_tel'] = S_tel_new
        self.state['S_stress'] = S_stress_new

        return self.state

    def update_from_cells(self, cells, dt, tau, stem_cell_rate=None):
        """
        Update the population state based on a collection of individual cells.

        Parameters:
            cells: Dictionary of Cell objects
            dt: Time step
            tau: Shear stress value (Pa)
            stem_cell_rate: Optional stem cell input rate, uses 0 if None

        Returns:
            Updated state
        """
        # Create divisions counter
        divisions_count = [0] * (self.max_divisions + 1)
        S_tel_count = 0
        S_stress_count = 0

        # Count cells by division stage and senescence status
        for cell in cells.values():
            if cell.is_senescent:
                if cell.senescence_cause == 'telomere':
                    S_tel_count += 1
                elif cell.senescence_cause == 'stress':
                    S_stress_count += 1
            else:
                div = min(cell.divisions, self.max_divisions)
                divisions_count[div] += 1

        # Set current state to cell counts
        self.state['E'] = divisions_count
        self.state['S_tel'] = S_tel_count
        self.state['S_stress'] = S_stress_count

        # Update state
        return self.update(dt, tau, stem_cell_rate)

    def synchronize_cells(self, cells):
        """
        Synchronize individual cells with the population state.

        Parameters:
            cells: Dictionary of Cell objects

        Returns:
            Actions dictionary with births, deaths, and senescence events
        """
        # Get current state
        E = self.state['E'].copy()
        S_tel = self.state['S_tel']
        S_stress = self.state['S_stress']

        # Count cells by division stage and senescence status
        divisions_count = [0] * (self.max_divisions + 1)
        S_tel_count = 0
        S_stress_count = 0

        for cell in cells.values():
            if cell.is_senescent:
                if cell.senescence_cause == 'telomere':
                    S_tel_count += 1
                elif cell.senescence_cause == 'stress':
                    S_stress_count += 1
            else:
                div = min(cell.divisions, self.max_divisions)
                divisions_count[div] += 1

        # Calculate differences
        E_diff = [int(round(E[i] - divisions_count[i])) for i in range(self.max_divisions + 1)]
        S_tel_diff = int(round(S_tel - S_tel_count))
        S_stress_diff = int(round(S_stress - S_stress_count))

        # Initialize actions
        actions = {
            'births': [],
            'deaths': [],
            'senescence': []
        }

        # Handle differences
        # Birth actions (positive E_diff)
        for i in range(self.max_divisions + 1):
            if E_diff[i] > 0:
                for _ in range(E_diff[i]):
                    actions['births'].append({
                        'type': 'healthy',
                        'divisions': i
                    })

        # Handle telomere-induced senescence
        if S_tel_diff > 0:
            for _ in range(S_tel_diff):
                actions['births'].append({
                    'type': 'senescent',
                    'cause': 'telomere'
                })

        # Handle stress-induced senescence
        if S_stress_diff > 0:
            for _ in range(S_stress_diff):
                actions['births'].append({
                    'type': 'senescent',
                    'cause': 'stress'
                })

        # Death actions (negative diffs)
        for i in range(self.max_divisions + 1):
            if E_diff[i] < 0:
                actions['deaths'].append({
                    'type': 'healthy',
                    'divisions': i,
                    'count': -E_diff[i]
                })

        if S_tel_diff < 0:
            actions['deaths'].append({
                'type': 'senescent',
                'cause': 'telomere',
                'count': -S_tel_diff
            })

        if S_stress_diff < 0:
            actions['deaths'].append({
                'type': 'senescent',
                'cause': 'stress',
                'count': -S_stress_diff
            })

        return actions
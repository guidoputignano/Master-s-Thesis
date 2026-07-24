"""
Simplified configuration for event-driven endothelial cell simulation.
"""
import os
import time


class SimulationConfig:
    """Clean, simplified configuration for event-driven system."""

    def __init__(self):
        # === REPRODUCIBILITY ===
        # Master RNG seed for the whole simulation (dimensionless integer).
        # Seeds both the stdlib `random` module and numpy's global RNG — in
        # Simulator.__init__ and again before the MPC per-cell heterogeneity
        # draws — so every stochastic draw (cell layout, initial senescence
        # assignment, per-cell z-deviates) is reproducible across runs. Replaces
        # the previous wall-clock (time.time_ns()) seeding.
        self.random_seed = 42

        # === CORE SIMULATION PARAMETERS ===
        self.simulation_duration = 360  # minutes (6 hours)
        self.time_step = 1.0  # minutes
        self.time_unit = "minutes"
        self.grid_size = (1024, 1024)  # pixels (display resolution)

        # === IMAGING FIELD / PIXEL SCALE (Table 1, main.tex) ===
        # The bioreactor domain corresponds to a 650x650 um imaging field at 20x.
        self.imaging_field_um = 650.0          # Source: Table 1, main.tex — imaging field = 650 um (20x)
        self.computation_scale = 4             # comp pixels per display pixel (per axis); see Grid
        # micrometres per display pixel
        self.pixel_scale_um = self.imaging_field_um / self.grid_size[0]  # = 650/1024 um/px

        # Confluent cell count derived from the healthy HUVEC area (A_E* = 2354 um^2):
        #   target_cell_count = round(650 * 650 / 2354) ~ 180
        # Source: Table 1, main.tex — A_E* = 2354 um^2 ; imaging field 650x650 um
        self.initial_cell_count = round(
            self.imaging_field_um * self.imaging_field_um / 2354.0
        )  # = 180 cells

        # === INITIAL SENESCENCE COMPOSITION (passage 6 HUVEC, PDL 15) ===
        # Source: NMR senescence literature (project knowledge) — phi_sen(0) = 0.20 at PDL 15
        self.initial_senescent_fraction = 0.20   # phi_sen(0) = 20%
        # Of the senescent fraction: 70% stress-induced (S_str), 30% telomere-induced (S_tel)
        self.senescent_stress_fraction = 0.70    # Source: project spec — S_str share of senescent
        self.senescent_telomere_fraction = 0.30  # Source: project spec — S_tel share of senescent

        # === EVENT-DRIVEN SYSTEM (MAIN FEATURE) ===
        self.use_event_driven_system = False  # Always enabled
        self.biological_optimization_enabled = False  # Always disabled

        # Event detection sensitivity
        self.pressure_change_threshold = 0.1  # Pa
        self.cell_count_change_threshold = 5  # cells
        self.senescence_threshold_change = 0.05  # 5%

        # Reconfiguration timing
        self.min_reconfiguration_interval = 5.0  # minutes

        # Transition parameters
        self.max_compression_ratio = 0.7  # cells can compress to 70%
        self.transition_completion_threshold = 0.95
        self.trajectory_checkpoint_interval = 20.0  # minutes

        # Multi-configuration system
        self.multi_config_count = 10
        self.multi_config_optimization_steps = 3

        # === SIMULATION COMPONENTS ===
        self.enable_temporal_dynamics = True
        self.enable_spatial_properties = True
        self.enable_population_dynamics = True

        # Population features
        self.enable_senescence = True
        self.enable_senolytics = False
        self.enable_stem_cells = False

        # Hole system
        self.enable_holes = False # True
        self.max_holes = 5
        self.hole_creation_probability_base = 0.02
        self.hole_creation_threshold_senescence = 0.30

        # === VISUALIZATION ===
        self.plot_interval = 10  # steps
        self.save_plots = True
        self.create_animations = True
        self.plot_directory = self._generate_plot_directory()

        # === DEBUG OPTIONS ===
        self.debug_events = False
        self.debug_transitions = False

        # === OPTIMAL STOPPING SYSTEM ===
        self.enable_optimal_stopping = False  # True Master switch
        self.stopping_criteria = {
            'max_senescence': 0.25,  # Stop if senescence exceeds 25%
            'response_stability_window': 15,  # Time steps to evaluate stability
            'response_stability_threshold': 0.05,  # Max std dev for a stable response
            'min_simulation_time': 60  # Minimum time before stopping can occur
        }

        # === TEMPORAL DYNAMICS ===
        self.tau_area_minutes = 45.0
        self.tau_orientation_minutes = 1.0
        self.tau_aspect_ratio_minutes = 50.0

        # === PAPER (Table 1, main.tex) GROUND-TRUTH PARAMETERS ===
        # Spatial / temporal morphological targets
        self.tau_act = 0.5            # Source: Table 1, main.tex — tau_act = 0.5 Pa
        self.rho_star = 2.3           # Source: Table 1, main.tex — rho* = 2.3 (-)
        self.theta_star = 0.0         # orientation target theta* = 0 deg (parallel / perfect alignment); 20 deg is now the t=6 h transient
        # ASPECT-RATIO adaptation time constant (hours). Set equal to
        # tau_orient_hours (7.4 h): orientation and aspect ratio are driven by the
        # same cytoskeletal remodelling, and only the orientation constant is
        # calibrated against imaging (theta(6 h)=20 deg => 6/ln(45/20) ~ 7.4 h).
        # There is no independent aspect-ratio timecourse to justify a distinct
        # value, so a single, data-calibrated morphological constant is the
        # parsimonious choice (was 9.0 h, the Table-1 6-12 h midpoint). NOTE: this
        # constant governs ASPECT RATIO only — cell AREA is fixed by the Voronoi
        # tessellation and is not relaxed with a temporal constant on the paper path.
        self.tau_adapt_hours = 7.4    # hours — aspect-ratio adaptation; equals tau_orient_hours (one physical constant)
        # Orientation adaptation time constant (hours): theta relaxes from
        # theta_stat=45 deg toward theta*=0 deg, calibrated so theta(6 h)=20 deg
        # matches the reference imaging (Chala/Nafsika):
        #   20 = 45*exp(-6/tau)  ->  tau = 6/ln(45/20) ~ 7.4 h
        # Kept as a SEPARATE field from tau_adapt_hours (both = 7.4 h, representing
        # one physical constant) so the planned sensitivity study can still sweep
        # orientation and aspect-ratio constants independently.
        self.tau_orient_hours = 7.4   # hours — orientation adaptation time constant (see above)
        self.target_area_healthy_um2 = 2354.0   # Source: Table 1, main.tex — A_E* = 2354 um^2
        # === MODEL-STRUCTURE FLAGS (Task 5 refactor) ===
        # Select which arms of the senescence / population model are active.
        # The defaults are the reported structure except where a comment marks a
        # deliberate change of simulation output.
        self.INCLUDE_REPLICATIVE_ARM = True
        #   True : keep the replicative ladder E_0..E_N and the telomere-
        #          senescence compartment S_tel (terminal division -> S_tel).
        #   False: collapse the ladder to a single healthy pool N_E and drop
        #          S_tel (no replicative/telomere senescence; stress arm only).
        self.MODEL_GROWTH_TO_CONFLUENCE = False
        #   True : keep the contact-inhibition density factor g(N_E)=1/(1+N_E/K).
        #   False: drop g and K (set g == 1); proliferation is density-independent
        #          at rate r. NOTE: dropping g CHANGES simulation output relative
        #          to g-on (proliferation is no longer slowed by density); r is
        #          left at its Table-1 value (the folded constant defaults to 1).
        self.INCLUDE_SUPRAPHYSIOLOGICAL_ARM = True
        #   True : add the high-shear damage term gamma_d*tau^m/(tau_d^m+tau^m)
        #          to the induction rate (see gamma_d/tau_d/m_hill below).
        #   False: no supraphysiological damage arm.
        # NOW ON (author decision): the protective-only monotone Hill makes the
        # control problem trivial (higher shear improves morphology AND lowers
        # senescence, so the optimiser just saturates the shear ceiling and the
        # phi_sen<=0.30 constraint is never active). Re-introducing the VAD-relevant
        # supraphysiological injury arm makes senescence rise again toward the
        # ceiling, so the controller must trade morphology against the senescence
        # limit and the constraint becomes active. See main.tex Sec 2.3 / 3.4.

        # === SENESCENCE-INDUCTION RATE gamma(tau): monotone-decreasing Hill ===
        # Replaces the earlier symmetric quadratic (eq:gamma_quad). Low shear
        # (atheroprone) -> high induction; high shear (atheroprotective) -> low:
        #
        #     gamma(tau) = gamma_min + (gamma_max - gamma_min)
        #                              * tau_h^n / (tau_h^n + tau^n)
        #
        # IMPORTANT (provenance): the Hill FORM is a modelling choice chosen to
        # match the cited monotone shear-protection shape (KLF2/P2X4 flow
        # signalling; atheroprone vs atheroprotective wall-shear boundary). It is
        # NOT an equation fitted to data in any single source. Each parameter is
        # tagged [anchored] (traceable to a cited value/threshold) or [assumed]
        # (illustrative, sweepable); assumed values are NOT fitted or measured.
        self.gamma_min = 0.00278   # h^-1 [anchored] high-shear floor plateau (Table 1, main.tex)
        self.gamma_max = 0.0125    # h^-1 [assumed]  low-shear (tau->0) plateau; sweepable.
        #                                 Illustrative value; equals the old quadratic rate at
        #                                 tau=0 (gamma_min + alpha_gamma*tau_opt^2 = 0.00278 +
        #                                 0.00497*1.4^2 ~ 0.0125). Not a fitted quantity.
        self.tau_h_sen = 0.5       # Pa   [anchored] half-max shear of the protective Hill.
        #                                 Atheroprotective / KLF2-P2X4 boundary (~0.5 Pa). Kept
        #                                 SEPARATE from tau_act (also 0.5 Pa): same physical
        #                                 threshold but a different equation, so a future study
        #                                 can sweep them independently (cf. tau_adapt/tau_orient).
        self.n_hill = 2            # -    [fixed]    protective Hill exponent (shape constant,
        #                                 plausible 2-4); fixed at 2, NOT fitted.
        # Supraphysiological (high-shear) damage arm (ON, see INCLUDE_ flag above):
        # gamma(tau) += gamma_d * tau^m / (tau_d^m + tau^m). Rises with shear, so
        # senescence is minimised at moderate laminar shear and grows again toward
        # the VAD-relevant supraphysiological ceiling.
        self.gamma_d = 0.05        # h^-1 [assumed]  high-shear damage plateau; illustrative,
        #                                 sweepable (like gamma_max); NOT fitted. Chosen so the
        #                                 phi_sen<=0.30 constraint is active over the 6 h window.
        self.tau_d = 1.5           # Pa   [assumed]  damage half-max shear, just above the ~1.4 Pa
        #                                 physiological optimum (injury onset in the achievable
        #                                 [0,2] Pa VAD band). Sweepable; NOT fitted.
        self.m_hill = 2            # -    [fixed]    damage Hill exponent (plausible 2-4); fixed at 2.

        self.phi_sen_max = 0.30    # -    Source: Table 1, main.tex — phi_sen^max = 30% of population

        # === TELOMERE SENESCENCE PARAMETERS ===
        self.max_divisions = 16  # Source: Table 1, main.tex — N (Hayflick limit, HUVEC) = 16 (midpoint of [15,18] PD)
        self.initial_telomere_mean = 100  # Average starting telomere length
        self.initial_telomere_std = 20  # Variability in starting length

        # Calculated automatically to match max_divisions
        self.telomere_loss_per_division = self.initial_telomere_mean / self.max_divisions

        # === LEGACY TEMPORAL-RESPONSE PARAMETERS (path B only) ===
        # DEPRECATED: consumed only by the legacy per-cell response model
        # (TemporalDynamicsModel.calculate_A_max / calculate_tau /
        # update_cell_responses), which is NOT used by run_mpc_simulation / the
        # reported model. The reported dynamics use the gated static->flow targets
        # and the single fixed morphological adaptation constant defined above
        # (tau_orient_hours = tau_adapt_hours = 7.4 h).
        #
        # These five fields were previously assigned twice in __init__; the
        # values kept here are the ones that were already in effect (the second,
        # winning assignment: tau_base = 60.0, lambda_scale = 0.3), so removing
        # the shadowed duplicates changes no simulation output. Retained in this
        # config (rather than relocated to a separate module) because
        # TemporalDynamicsModel.__init__ reads them at construction and the legacy
        # CLI must keep working.
        self.known_pressures = [0.0, 1.4]   # Pa — pressures with a measured A_max (legacy)
        self.known_A_max = {
            0.0: 1.0,   # A_max at 0 Pa (baseline) — legacy per-cell response
            1.4: 2.5    # A_max at 1.4 Pa — legacy per-cell response
        }
        self.initial_response = 1.0   # legacy per-cell response initial value
        self.tau_base = 60.0          # minutes — legacy response time-constant base (value in effect)
        self.lambda_scale = 0.3       # legacy response tau power-law exponent (value in effect)

        # === POPULATION DYNAMICS PARAMETERS ===
        self.proliferation_rate = 0.025  # Source: Table 1, main.tex — r = 0.02-0.03 h^-1 (nominal 0.025)
        self.carrying_capacity = 5.5e4   # Source: Table 1, main.tex — K = 5-6e4 cells/cm^2 (nominal 5.5e4)
        # Task 5 population pruning (Part C): the treatment-free six-hour reported
        # regime sets death rates, SASP induction, the senolytic block and the
        # stem-cell input to zero. These terms are REMOVED (not zero-weighted).
        # The reduced law is
        #     dE_i/dt   = 2 r g E_{i-1} - r g E_i - gamma(tau) E_i   (replicative arm)
        #     dS_tel/dt = r g E_N
        #     dS_str/dt = sum_i gamma(tau) E_i    (xi=0 -> gamma(tau) N_E)
        # with g == 1 unless MODEL_GROWTH_TO_CONFLUENCE is True.

        # Deterministic senescence parameters (per-cell resistance distribution;
        # used by the spatial/visual layer and reserved for the optional Part D
        # low-shear dose rule).
        self.base_cellular_resistance = 0.5  # Base resistance threshold (adjusted for stress_factor scale)
        self.resistance_variability = 0.2  # Cell-to-cell variability


    def _generate_plot_directory(self):
        """Generate timestamped plot directory."""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        return f"results_event_driven_{timestamp}"

    # === SIMPLE CONFIGURATION METHODS ===

    def set_temporal_only(self):
        """Focus only on temporal dynamics."""
        self.enable_temporal_dynamics = True
        self.enable_spatial_properties = False
        self.enable_population_dynamics = False
        self.enable_senescence = False
        return self

    def set_spatial_only(self):
        """Focus only on spatial properties."""
        self.enable_temporal_dynamics = False
        self.enable_spatial_properties = True
        self.enable_population_dynamics = False
        self.enable_senescence = False
        return self

    def set_population_only(self):
        """Focus only on population dynamics."""
        self.enable_temporal_dynamics = False
        self.enable_spatial_properties = False
        self.enable_population_dynamics = True
        self.enable_senescence = True
        return self


    def set_full_simulation(self):
        """Enable all simulation components."""
        self.enable_temporal_dynamics = True
        self.enable_spatial_properties = True
        self.enable_population_dynamics = True
        self.enable_senescence = True
        return self

    def set_minimal(self):
        """Minimal simulation - basic tessellation only."""
        self.enable_temporal_dynamics = False
        self.enable_spatial_properties = False
        self.enable_population_dynamics = False
        self.enable_senescence = False
        self.enable_holes = False
        return self

    # === EVENT-DRIVEN CONFIGURATION ===

    def set_event_sensitivity(self, pressure=0.1, cells=5, senescence=0.05):
        """Set event detection sensitivity."""
        self.pressure_change_threshold = pressure
        self.cell_count_change_threshold = cells
        self.senescence_threshold_change = senescence
        return self

    def set_transition_params(self, compression=0.7, completion=0.95, interval=20.0):
        """Set transition parameters."""
        self.max_compression_ratio = compression
        self.transition_completion_threshold = completion
        self.trajectory_checkpoint_interval = interval
        return self

    def enable_debug(self, events=True, transitions=True):
        """Enable debug output."""
        self.debug_events = events
        self.debug_transitions = transitions
        return self

    # === QUICK SETUPS ===

    def quick_test(self):
        """Quick test configuration - small, fast."""
        self.initial_cell_count = 20
        self.simulation_duration = 120  # 2 hours
        self.multi_config_count = 5
        self.enable_holes = False
        self.create_animations = True
        return self

    def research_quality(self):
        """High-quality research configuration."""
        self.initial_cell_count = 100
        self.simulation_duration = 720  # 12 hours
        self.multi_config_count = 20
        self.multi_config_optimization_steps = 5
        self.create_animations = True
        return self

    def get_summary(self):
        """Get configuration summary."""
        return {
            'mode': 'event-driven',
            'seed': self.random_seed,   # master RNG seed (reproducibility)
            'components': [
                name for name, enabled in [
                    ('temporal', self.enable_temporal_dynamics),
                    ('spatial', self.enable_spatial_properties),
                    ('population', self.enable_population_dynamics),
                    ('senescence', self.enable_senescence),
                    ('holes', self.enable_holes)
                ] if enabled
            ],
            'duration': f"{self.simulation_duration} min ({self.simulation_duration/60:.1f}h)",
            'cells': self.initial_cell_count,
            'pressure_threshold': self.pressure_change_threshold,
            'multi_configs': self.multi_config_count,
            'debug': self.debug_events or self.debug_transitions
        }

    def describe(self):
        """Generate description of configuration."""
        summary = self.get_summary()

        desc = f"""
🧬 Event-Driven Endothelial Cell Simulation Configuration

📊 Simulation Settings:
   Duration: {summary['duration']}
   Initial cells: {summary['cells']}
   Grid size: {self.grid_size[0]}×{self.grid_size[1]}
   Time step: {self.time_step} min
   RNG seed: {summary['seed']}

🔄 Event-Driven System:
   Pressure threshold: {self.pressure_change_threshold} Pa
   Min reconfig interval: {self.min_reconfiguration_interval} min
   Max compression: {self.max_compression_ratio}
   Multi-configurations: {self.multi_config_count}

🧪 Enabled Components: {', '.join(summary['components'])}

🐛 Debug: {'ON' if summary['debug'] else 'OFF'}
"""
        return desc.strip()


# === SIMPLE CONFIGURATION CREATION FUNCTIONS ===

def create_full_config():
    """Create full event-driven configuration."""
    return SimulationConfig().set_full_simulation()

def create_temporal_only_config():
    """Create temporal-only configuration."""
    return SimulationConfig().set_temporal_only()

def create_spatial_only_config():
    """Create spatial-only configuration."""
    return SimulationConfig().set_spatial_only()

def create_population_only_config():
    """Create population-only configuration."""
    return SimulationConfig().set_population_only()

def create_minimal_config():
    """Create minimal configuration."""
    return SimulationConfig().set_minimal()

def enable_all(self):
    """Enable all simulation components (backward compatibility method)."""
    return self.set_full_simulation()

def create_test_config():
    """Create quick test configuration."""
    return SimulationConfig().set_full_simulation().quick_test()

def create_research_config():
    """Create research-quality configuration."""
    return SimulationConfig().set_full_simulation().research_quality()

import numpy as np
from scipy.integrate import solve_ivp
import logging

class MyoFE:
    """Class representing a myocardial fiber reorientation and stress model.

    Attributes:
        n (int): Number of computational nodes.
        region_labels (tuple of int): Labels defining available tissue regions (Healthy, Hyper, Hypo, Fibrosis).
        region_assignments (np.ndarray): Node-specific region assignments.
        E_values (np.ndarray): Elastic modulus values.
        k1_params (np.ndarray): Stress adjustment parameters.
        fiber_angles (np.ndarray): Current fiber orientation angles.
        kappa (float): Reorientation sensitivity parameter.
        base_stress (float): Baseline myocardial stress.
        cardiac_period (float): Duration of one cardiac cycle.
        t_total (float): Total simulation time.
        C (float): Windkessel compliance.
        R (float): Windkessel resistance.
        P_env (float): Peripheral pressure.
        Q_in (np.ndarray): Input flow measurements.
    """
    
    def __init__(
        self,
        num_nodes: int = 100,
        region_labels: tuple[int, int, int, int] = (0, 1, 2, 3),
        base_E: float = 1.5,
        fib_E: float = 4.0,
        k1_hyper: float = 0.2,
        k1_hypo: float = -0.15,
        kappa: float = 0.05,
        base_stress: float = 1e4,
        cardiac_period: float = 0.8,
        cycles: int = 200,
    ):
        """Initialize the myocardial simulation parameters.

        Args:
            num_nodes (int): Number of nodes in the model.
            region_labels (tuple[int, int, int, int]): Regional identifiers in order.
            base_E (float): Base elastic modulus (Pa).
            fib_E (float): Fibrotic region modulus multiplier.
            k1_hyper (float): Stress adjustment for hyperactive regions.
            k1_hypo (float): Stress adjustment for hypotonic regions.
            kappa (float): Rotation sensitivity parameter.
            base_stress (float): Baseline stress value (Pa).
            cardiac_period (float): Duration of a cardiac cycle (s).
            cycles (int): Number of cycles to simulate.
        """
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Input validation
        if len(region_labels) != 4:
            raise ValueError("region_labels must contain exactly 4 elements indicating [Healthy, Hyper, Hypo, Fibrosis]")
        if cycles < 1:
            raise ValueError("cycles must be a positive integer")

        self.n = num_nodes
        self.region_labels = region_labels
        self.region_assignments = np.random.choice(
            region_labels,
            size=num_nodes,
            p=[0.75, 0.1, 0.1, 0.05]
        )
        
        # Material properties
        self.E_values = base_E * np.ones(num_nodes)
        fib_label = region_labels[3]  # Fibrosis is last entry
        fib_indices = self.region_assignments == fib_label
        self.E_values[fib_indices] *= fib_E

        # Stress modifiers
        self.k1_params = np.zeros(num_nodes)
        self.k1_params[self.region_assignments == region_labels[1]] = k1_hyper
        self.k1_params[self.region_assignments == region_labels[2]] = k1_hypo

        # Structural properties
        self.fiber_angles = np.random.uniform(0, 2*np.pi, num_nodes)
        self.kappa = kappa
        self.base_stress = base_stress

        # Simulation parameters
        self.cardiac_period = cardiac_period
        self.cycles = cycles
        self.t_total = cycles * cardiac_period
        self.C = 0.0003
        self.R = 1000
        self.P_env = 100
        self.Q_in = np.zeros(cycles)

    def _compute_activation_stress(self) -> np.ndarray:
        """Calculate tissue contractility contributions.

        Returns:
            np.ndarray: Active stress values (Pa).
        """
        active = self.base_stress * (1 + self.k1_params)
        fib_label = self.region_labels[3]
        fib_masks = self.region_assignments == fib_label
        active[fib_masks] *= 0.8
        return active

    def _calculate_stress_tensor(self, current_angle: np.ndarray, active: np.ndarray) -> np.ndarray:
        """Compute combined active-passive tissue stress.

        Args:
            current_angle (np.ndarray): Current fiber angles.
            active (np.ndarray): Activation stresses.

        Returns:
            np.ndarray: Total stress values.
        """
        passive = self.E_values * np.sin(2*(current_angle - self.fiber_angles))
        return active + passive

    def _determine_principal_directions(self, stress_values: np.ndarray) -> np.ndarray:
        """Determine principal stress directions.

        Args:
            stress_values (np.ndarray): Computed stress values.

        Returns:
            np.ndarray: angualr directions (rad).
        """
        return np.arctan2(stress_values, np.ones_like(stress_values)) % (2*np.pi)

    def _windkessel_equation(self, t: float, y: np.ndarray) -> np.ndarray:
        """Windkessel model differential equation.

        Args:
            t (float): Time (s).
            y (np.ndarray): [Volume (L)] - Volume in ventricular chamber.

        Returns:
            np.ndarray: Rate of volume change (L/s).
        """
        V = y[0]
        Q_out = (V/self.C - self.P_env)/self.R
        dVdt = self.input_flow(t) - Q_out
        return np.array([dVdt])

    def input_flow(self, t: float) -> float:
        """Pulsatile flow input function.

        Args:
            t (float): Current time (s).

        Returns:
            float: Instantaneous flow (L/s).
        """
        t_scaled = np.mod(t, self.cardiac_period)
        return 3.5 * (1 + np.sin(2*np.pi*t_scaled/self.cardiac_period))

    def _run_hemodynamic_cycle(self, simulation_time: float) -> float:
        """Simulate hemodynamic dynamics over a single cycle.

        Args:
            simulation_time (float): Duration to simulate.

        Returns:
            float: End volume value.
        """        
        sol = solve_ivp(
            self._windkessel_equation,
            (0, simulation_time),
            [0.0],
            t_eval=[simulation_time]
        )
        return sol.y[0][-1]

    def reorient_fibers(self, dt: float):
        """Update fiber orientations using stress direction.

        Args:
            dt (float): Simulation time step.
        """
        active = self._compute_activation_stress()
        total_stress = self._calculate_stress_tensor(self.fiber_angles, active)
        principle_dir = self._determine_principal_directions(total_stress)
        
        delta = (principle_dir - self.fiber_angles) * (dt/self.kappa)
        self.fiber_angles += delta
        self.fiber_angles %= (2*np.pi)

    def execute(self) -> tuple[np.ndarray, np.ndarray]:
        """Run the full simulation with hemodynamic and structural coupling.

        Returns:
            tuple: Simulation outputs (flow data, final angles).
        """
        time_steps = np.linspace(0, self.t_total, int(self.t_total/0.01))
        for t in time_steps:
            current_cycle = int(t // self.cardiac_period)
            
            # Update hemodynamics at new cycle start
            if np.isclose(t % self.cardiac_period, 0, atol=1e-8):
                self.Q_in[current_cycle] = self._run_hemodynamic_cycle(self.cardiac_period)
            
            self.reorient_fibers(0.01)
            
        return self.Q_in, self.fiber_angles

def main() -> None:
    """Entry point for simulation execution."""
    model = MyoFE(
        num_nodes=100,
        region_labels=(0, 1, 2, 3),
        fib_E=5.0,
        cycles=3
    )
    
    flow_data, final_angles = model.execute()
    
    # Postprocessing
    valid_volumes = flow_data[flow_data > 0]
    avg_stroke_vol = np.mean(valid_volumes)
    std_dev = np.std(final_angles)
    
    print(f"Average Stroke Volume: {avg_stroke_vol:.2f}")
    print(f"Fiber Angle Disarray: {std_dev:.2f} rad")

if __name__ == "__main__":
    main()
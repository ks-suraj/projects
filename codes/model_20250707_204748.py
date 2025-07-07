# Generated on 2025-07-07 20:47:48
# Model implementation below

import numpy as np
from scipy.integrate import solve_ivp

class MyoFE:
    def __init__(self, num_nodes=100, region_mask=(0, 0, 1, 2), base_E=1.5, 
                 fib_E=4.0, k1_hyper=0.2, k1_hypo=-0.15, kappa=0.05, 
                 base_stress=1e4, cardiac_period=0.8, cycles=200):
        self.n = num_nodes
        self.region_mask = np.random.choice(region_mask, size=num_nodes, p=[0.75, 0.1, 0.1, 0.05]) # Healthy, Hyper, Hypo, Fibrosis
        self.E_values = base_E * np.ones(self.n)
        self.E_values[self.region_mask == 2] *= fib_E  # Fibrotic region
        self.k1_params = np.zeros(self.n)
        self.k1_params[self.region_mask == 1] = k1_hyper
        self.k1_params[self.region_mask == 3] = k1_hypo
        
        self.fiber_angles = np.random.uniform(0, 2*np.pi, self.n)
        self.kappa = kappa
        self.base_stress = base_stress
        self.cardiac_period = cardiac_period
        self.t_total = cycles * cardiac_period
        # Windkessel parameters
        self.C = 0.0003  # compliance
        self.R = 1000    # resistance
        self.P_env = 100 # Peripheral pressure
        self.Q_in = np.zeros(cycles) # To store flow data

    def _compute_activate_stress(self):
        active = self.base_stress * (1 + self.k1_params)
        active[self.region_mask == 2] *= 0.8  # Reduced contraction for fibrotic regions
        return active
    
    def _stress_tensor(self, current_angle, active):
        passive = self.E_values * np.sin(2*(current_angle - self.fiber_angles)) 
        # Simplified anisotropic stress component
        return active + passive
    
    def _principal_stress_direction(self, stress_values):
        theta = np.arctan2(stress_values, 1)  # Assuming simplified principal direction
        return theta % (2*np.pi)
    
    def _windkessel(self, t, y):
        V, = y
        Q_out = (V/self.C - self.P_env)/self.R
        dVdt = self.input_flow(t) - Q_out/self.C
        return dVdt
    
    def input_flow(self, t):
        t_scaled = t % self.cardiac_period
        return 3.5 * (1 + np.sin(2*np.pi*t_scaled/self.cardiac_period))
    
    def _hemodynamic_model(self, t_final):
        sol = solve_ivp(lambda t, y: self._windkessel(t, y), [0, t_final], [0])
        return sol.y[0][-1]
    
    def fiber_reorient(self, dt):
        active = self._compute_activate_stress()
        total_stress = self._stress_tensor(self.fiber_angles, active)
        principle_dir = self._principal_stress_direction(total_stress)
        delta_theta = (principle_dir - self.fiber_angles) * (1/self.kappa)*dt
        self.fiber_angles += delta_theta
        self.fiber_angles %= (2*np.pi)
        
    def run_simulation(self):
        time_steps = np.linspace(0, self.t_total, int(self.t_total/0.01))
        for t_current in time_steps:
            cycle = int(t_current//self.cardiac_period)
            if t_current % self.cardiac_period == 0:
                self.Q_in[cycle] = self._hemodynamic_model(self.cardiac_period)
            self.fiber_reorient(0.01)
        return self.Q_in, self.fiber_angles

def main():
    model = MyoFE(num_nodes=100, region_mask=(0,1,2,3), fib_E=5.0, cycles=3)
    stroke_volume, final_angles = model.run_simulation()
    # Output results for analysis
    print("Stroke Volume:", np.mean(stroke_volume[stroke_volume > 0]))
    print("Average fiber disarray:", np.std(final_angles))

if __name__ == "__main__":
    main()
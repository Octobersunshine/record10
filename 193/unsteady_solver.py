import numpy as np
from airfoil_geometry import AirfoilGeometry
from vlm_solver import VortexLatticeMethod


class UnsteadySolver:
    def __init__(self, n_panels=80, n_points=50):
        self.geom = AirfoilGeometry(n_points=n_points)
        self.vlm = VortexLatticeMethod(n_panels=n_panels)
        self.n_panels = n_panels
        
    def get_collocation_and_vortex_points(self, x, y):
        n = len(x) - 1
        
        x_c = np.zeros(n)
        y_c = np.zeros(n)
        x_v = np.zeros(n)
        y_v = np.zeros(n)
        
        for i in range(n):
            x_c[i] = 0.75 * x[i] + 0.25 * x[i+1]
            y_c[i] = 0.75 * y[i] + 0.25 * y[i+1]
            x_v[i] = 0.25 * x[i] + 0.75 * x[i+1]
            y_v[i] = 0.25 * y[i] + 0.75 * y[i+1]
        
        return x_c, y_c, x_v, y_v
    
    def compute_panel_velocities(self, x_prev, y_prev, x_curr, y_curr, dt):
        n = len(x_prev) - 1
        velocities = np.zeros((n, 2))
        
        for i in range(n):
            x_mid_prev = 0.5 * (x_prev[i] + x_prev[i+1])
            y_mid_prev = 0.5 * (y_prev[i] + y_prev[i+1])
            x_mid_curr = 0.5 * (x_curr[i] + x_curr[i+1])
            y_mid_curr = 0.5 * (y_curr[i] + y_curr[i+1])
            
            velocities[i, 0] = (x_mid_curr - x_mid_prev) / dt
            velocities[i, 1] = (y_mid_curr - y_mid_prev) / dt
        
        return velocities
    
    def get_deformed_airfoil(self, time, config):
        x_u, y_u, x_l, y_l = self.geom.generate_naca4(config.get('naca', '0012'))
        
        if 'camber_amplitude' in config:
            camber_amp = config['camber_amplitude']
            camber_freq = config.get('camber_frequency', 2.0)
            x_u, y_u = self.geom.apply_camber_deformation(x_u, y_u, camber_amp, camber_freq, time)
            x_l, y_l = self.geom.apply_camber_deformation(x_l, y_l, camber_amp, camber_freq, time)
        
        alpha_mean = config.get('alpha_mean', 0.0)
        alpha_amp = config.get('alpha_amplitude', 0.0)
        pitch_freq = config.get('pitch_frequency', 0.0)
        pivot = config.get('pitch_pivot', 0.25)
        
        x_u, y_u, alpha = self.geom.apply_pitch_motion(
            x_u, y_u, alpha_mean, alpha_amp, pitch_freq, time, pivot
        )
        x_l, y_l, _ = self.geom.apply_pitch_motion(
            x_l, y_l, alpha_mean, alpha_amp, pitch_freq, time, pivot
        )
        
        return x_u, y_u, x_l, y_l, alpha
    
    def solve_unsteady(self, config):
        V_inf = config.get('V_inf', 10.0)
        c = config.get('chord', 1.0)
        rho = config.get('rho', 1.225)
        t_final = config.get('t_final', 2.0)
        dt = config.get('dt', 0.01)
        
        n_steps = int(t_final / dt)
        
        times = np.linspace(0, t_final, n_steps)
        Cl_history = np.zeros(n_steps)
        Cd_history = np.zeros(n_steps)
        Cm_history = np.zeros(n_steps)
        alpha_history = np.zeros(n_steps)
        
        self.vlm.reset_wake()
        gamma_prev = None
        
        x_prev, y_prev = None, None
        
        for step, t in enumerate(times):
            x_u, y_u, x_l, y_l, alpha = self.get_deformed_airfoil(t, config)
            
            x, y, x_mid, y_mid, length, nx, ny, tx, ty = self.geom.get_panel_geometry(
                x_u, y_u, x_l, y_l
            )
            
            x_c, y_c, x_v, y_v = self.get_collocation_and_vortex_points(x, y)
            
            panel_velocities = None
            if x_prev is not None:
                panel_velocities = self.compute_panel_velocities(x_prev, y_prev, x, y, dt)
            
            gamma = self.vlm.solve(
                x_c, y_c, x_v, y_v, nx, ny,
                V_inf, alpha, panel_velocities
            )
            
            Cl = self.vlm.compute_lift_coefficient(V_inf)
            
            Cp = self.vlm.compute_pressure_coefficient(
                x_c, y_c, x_v, y_v, V_inf, alpha, panel_velocities
            )
            
            force_x, force_y = self.vlm.compute_forces_from_pressure(
                Cp, nx, ny, length, V_inf, rho
            )
            
            qS = 0.5 * rho * V_inf**2 * c
            Cd = (force_x * np.cos(alpha) + force_y * np.sin(alpha)) / qS
            
            Cm = self.vlm.compute_moment_coefficient(
                Cp, x_c, y_c, nx, ny, length, V_inf, c, 
                config.get('pitch_pivot', 0.25), rho
            )
            
            Cl_history[step] = Cl
            Cd_history[step] = Cd
            Cm_history[step] = Cm
            alpha_history[step] = np.degrees(alpha)
            
            te_idx = np.argmax(x)
            x_te = x[te_idx]
            y_te = y[te_idx]
            self.vlm.shed_wake(x_te, y_te, dt, V_inf, alpha)
            self.vlm.convect_wake(dt, V_inf, alpha, convection_factor=0.9)
            
            gamma_prev = gamma.copy()
            x_prev, y_prev = x.copy(), y.copy()
            
        results = {
            'times': times,
            'Cl': Cl_history,
            'Cd': Cd_history,
            'Cm': Cm_history,
            'alpha': alpha_history,
            'wake_vortices': self.vlm.wake_vortices,
            'wake_strengths': self.vlm.wake_strengths
        }
        
        return results

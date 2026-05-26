import numpy as np


class VortexLatticeMethod:
    def __init__(self, n_panels=80):
        self.n_panels = n_panels
        self.gamma = None
        self.wake_vortices = []
        self.wake_strengths = []
        
    def vortex_2d(self, xv, yv, x, y, gamma=1.0):
        dx = x - xv
        dy = y - yv
        r2 = dx**2 + dy**2 + 1e-10
        
        u = -gamma * dy / (2 * np.pi * r2)
        v = gamma * dx / (2 * np.pi * r2)
        
        return u, v
    
    def build_influence_matrix(self, x_c, y_c, x_v, y_v, nx, ny):
        n = len(x_c)
        self.n_panels = n
        A = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                u, v = self.vortex_2d(x_v[j], y_v[j], x_c[i], y_c[i])
                A[i, j] = u * nx[i] + v * ny[i]
        
        return A
    
    def build_wake_influence(self, x_c, y_c, nx, ny):
        n = len(x_c)
        wake_influence = np.zeros(n)
        
        for wake_idx, (xw, yw, strength) in enumerate(zip(self.wake_vortices[::2], 
                                                          self.wake_vortices[1::2], 
                                                          self.wake_strengths)):
            for i in range(n):
                u, v = self.vortex_2d(xw, yw, x_c[i], y_c[i], strength)
                wake_influence[i] += u * nx[i] + v * ny[i]
        
        return wake_influence
    
    def compute_induced_velocity(self, x, y, gamma, x_v, y_v):
        u = np.zeros_like(x)
        v = np.zeros_like(x)
        
        for j in range(len(gamma)):
            u_j, v_j = self.vortex_2d(x_v[j], y_v[j], x, y, gamma[j])
            u += u_j
            v += v_j
        
        for wake_idx, (xw, yw, strength) in enumerate(zip(self.wake_vortices[::2], 
                                                          self.wake_vortices[1::2], 
                                                          self.wake_strengths)):
            u_w, v_w = self.vortex_2d(xw, yw, x, y, strength)
            u += u_w
            v += v_w
        
        return u, v
    
    def solve(self, x_c, y_c, x_v, y_v, nx, ny, 
              V_inf, alpha, panel_velocities=None):
        n = len(x_c)
        self.n_panels = n
        
        A = self.build_influence_matrix(x_c, y_c, x_v, y_v, nx, ny)
        
        U_inf = V_inf * np.cos(alpha)
        W_inf = V_inf * np.sin(alpha)
        
        RHS = -(U_inf * nx + W_inf * ny)
        
        if panel_velocities is not None:
            RHS -= panel_velocities[:, 0] * nx + panel_velocities[:, 1] * ny
        
        wake_influence = self.build_wake_influence(x_c, y_c, nx, ny)
        RHS -= wake_influence
        
        self.gamma = np.linalg.solve(A, RHS)
        
        return self.gamma
    
    def shed_wake(self, x_te, y_te, dt, V_inf, alpha):
        if self.gamma is not None:
            gamma_te = self.gamma[-1]
            
            x_wake = x_te + V_inf * np.cos(alpha) * dt
            y_wake = y_te + V_inf * np.sin(alpha) * dt
            
            self.wake_vortices.extend([x_wake, y_wake])
            self.wake_strengths.append(-gamma_te)
    
    def convect_wake(self, dt, V_inf, alpha, convection_factor=1.0):
        if len(self.wake_vortices) == 0:
            return
        
        wake_array = np.array(self.wake_vortices).reshape(-1, 2)
        
        u_conv = V_inf * np.cos(alpha) * convection_factor
        v_conv = V_inf * np.sin(alpha) * convection_factor
        
        wake_array[:, 0] += u_conv * dt
        wake_array[:, 1] += v_conv * dt
        
        self.wake_vortices = wake_array.flatten().tolist()
    
    def compute_lift_coefficient(self, V_inf):
        if self.gamma is None:
            return 0.0
        total_gamma = np.sum(self.gamma)
        Cl = total_gamma / (0.5 * V_inf)
        return Cl
    
    def compute_pressure_coefficient(self, x_c, y_c, x_v, y_v, 
                                     V_inf, alpha, panel_velocities=None):
        u, v = self.compute_induced_velocity(x_c, y_c, self.gamma, x_v, y_v)
        
        U_inf = V_inf * np.cos(alpha)
        W_inf = V_inf * np.sin(alpha)
        
        u_total = u + U_inf
        v_total = v + W_inf
        
        if panel_velocities is not None:
            u_total -= panel_velocities[:, 0]
            v_total -= panel_velocities[:, 1]
        
        V_mag = np.sqrt(u_total**2 + v_total**2)
        Cp = 1.0 - (V_mag / V_inf)**2
        
        return Cp
    
    def compute_forces_from_pressure(self, Cp, nx, ny, length, V_inf, rho=1.225):
        n = len(Cp)
        force_x = 0.0
        force_y = 0.0
        
        q = 0.5 * rho * V_inf**2
        
        for i in range(n):
            force_mag = q * Cp[i] * length[i]
            force_x += force_mag * nx[i]
            force_y += force_mag * ny[i]
        
        return force_x, force_y
    
    def compute_moment_coefficient(self, Cp, x_c, y_c, nx, ny, length, 
                                   V_inf, c=1.0, pivot=0.25, rho=1.225):
        n = len(Cp)
        Cm = 0.0
        
        q = 0.5 * rho * V_inf**2
        qS = q * c
        
        for i in range(n):
            moment_arm_x = x_c[i] - pivot
            moment_arm_y = y_c[i]
            
            force_mag = q * Cp[i] * length[i]
            local_force_x = force_mag * nx[i]
            local_force_y = force_mag * ny[i]
            
            Cm += (local_force_y * moment_arm_x - local_force_x * moment_arm_y) / (qS * c)
        
        return Cm
    
    def reset_wake(self):
        self.wake_vortices = []
        self.wake_strengths = []
        self.gamma = None

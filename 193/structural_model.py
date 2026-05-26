import numpy as np


class EulerBernoulliBeam:
    def __init__(self, n_points=50, chord=1.0, thickness=0.12, 
                 E=70e9, rho=2700, root_pinned=True):
        self.n_points = n_points
        self.chord = chord
        self.thickness = thickness
        self.E = E
        self.rho = rho
        self.root_pinned = root_pinned
        
        self.x = np.linspace(0, chord, n_points)
        self.dx = self.x[1] - self.x[0]
        
        self.I = np.zeros(n_points)
        self.A = np.zeros(n_points)
        self.m = np.zeros(n_points)
        
        self._update_properties()
        
        self.w = np.zeros(n_points)
        self.w_dot = np.zeros(n_points)
        self.w_ddot = np.zeros(n_points)
        
        self.theta = np.zeros(n_points)
        
        self.force_aero = np.zeros(n_points)
        self.force_internal = np.zeros(n_points)
        
        self.K = None
        self.M = None
        
    def _update_properties(self):
        t_le = self.thickness * 1.0
        t_te = self.thickness * 0.01
        t = t_le - (t_le - t_te) * self.x / self.chord
        
        self.I = self._calculate_moment_of_inertia(t)
        self.A = self._calculate_area(t)
        self.m = self.rho * self.A * self.dx
        
    def _calculate_moment_of_inertia(self, t):
        I = t**3 / 12.0
        return I
        
    def _calculate_area(self, t):
        A = t * 1.0
        return A
        
    def assemble_stiffness_matrix(self):
        n = self.n_points
        K = np.zeros((n, n))
        
        for i in range(2, n - 2):
            coeff = self.E * self.I[i] / self.dx**4
            K[i, i-2] = coeff
            K[i, i-1] = -4 * coeff
            K[i, i] = 6 * coeff
            K[i, i+1] = -4 * coeff
            K[i, i+2] = coeff
        
        if self.root_pinned:
            K[0, 0] = 1.0
            K[1, 1] = 1.0
            
            i = 2
            if n > 2:
                coeff = self.E * self.I[i] / self.dx**4
                K[i, i] = 6 * coeff
                if n > 3:
                    K[i, i+1] = -4 * coeff
                if n > 4:
                    K[i, i+2] = coeff
                    
            i = n - 2
            if n > 2:
                coeff = self.E * self.I[i] / self.dx**4
                if n > 4:
                    K[i, i-2] = coeff
                if n > 3:
                    K[i, i-1] = -4 * coeff
                K[i, i] = 6 * coeff
                
            i = n - 1
            if n > 2:
                coeff = self.E * self.I[i] / self.dx**4
                if n > 4:
                    K[i, i-2] = coeff
                if n > 3:
                    K[i, i-1] = -4 * coeff
                K[i, i] = 6 * coeff
        else:
            i = 0
            coeff = self.E * self.I[i] / self.dx**4
            K[i, i] = 7 * coeff
            K[i, i+1] = -4 * coeff
            K[i, i+2] = coeff
            
            i = 1
            coeff = self.E * self.I[i] / self.dx**4
            K[i, i-1] = -4 * coeff
            K[i, i] = 6 * coeff
            K[i, i+1] = -4 * coeff
            K[i, i+2] = coeff
            
            i = n - 2
            coeff = self.E * self.I[i] / self.dx**4
            K[i, i-2] = coeff
            K[i, i-1] = -4 * coeff
            K[i, i] = 6 * coeff
            
            i = n - 1
            coeff = self.E * self.I[i] / self.dx**4
            K[i, i-2] = coeff
            K[i, i-1] = -4 * coeff
            K[i, i] = 6 * coeff
        
        self.K = K
        return K
        
    def assemble_mass_matrix(self):
        n = self.n_points
        M = np.zeros((n, n))
        
        for i in range(n):
            M[i, i] = self.m[i]
        
        self.M = M
        return M
        
    def compute_deflection(self, forces):
        if self.K is None:
            self.assemble_stiffness_matrix()
        
        F = forces.copy()
        n = self.n_points
        
        if self.root_pinned:
            F[0] = 0.0
            F[1] = 0.0
            
            n_solve = n - 2
            w_sol = np.linalg.solve(self.K[2:n, 2:n], F[2:n])
            self.w[2:n] = w_sol
            self.w[0] = 0.0
            self.w[1] = 0.0
        else:
            self.w = np.linalg.solve(self.K, F)
        
        self._compute_slopes()
        
        return self.w.copy()
        
    def _compute_slopes(self):
        for i in range(1, self.n_points - 1):
            self.theta[i] = (self.w[i+1] - self.w[i-1]) / (2 * self.dx)
        
        self.theta[0] = (self.w[1] - self.w[0]) / self.dx
        self.theta[-1] = (self.w[-1] - self.w[-2]) / self.dx
        
    def compute_strain_energy(self):
        if self.K is None:
            self.assemble_stiffness_matrix()
        
        U = 0.5 * self.w @ self.K @ self.w
        return U
        
    def compute_kinetic_energy(self):
        if self.M is None:
            self.assemble_mass_matrix()
        
        T = 0.5 * self.w_dot @ self.M @ self.w_dot
        return T
        
    def get_camber_line(self, w=None):
        if w is None:
            w = self.w
            
        y_c = w.copy()
        return self.x.copy(), y_c
        
    def update_properties_deformed(self):
        ds = np.sqrt(1 + self.theta**2) * self.dx
        self._update_properties()
        
    def reset(self):
        self.w = np.zeros(self.n_points)
        self.w_dot = np.zeros(self.n_points)
        self.w_ddot = np.zeros(self.n_points)
        self.theta = np.zeros(self.n_points)
        self.force_aero = np.zeros(self.n_points)
        self.force_internal = np.zeros(self.n_points)


class SimpleSpringModel:
    def __init__(self, n_points=50, chord=1.0, stiffness=1000.0, 
                 damping=10.0, mass_per_length=10.0):
        self.n_points = n_points
        self.chord = chord
        self.stiffness = stiffness
        self.damping = damping
        self.mass_per_length = mass_per_length
        
        self.x = np.linspace(0, chord, n_points)
        self.dx = self.x[1] - self.x[0]
        
        self.w = np.zeros(n_points)
        self.w_dot = np.zeros(n_points)
        
        self.k = np.ones(n_points) * stiffness
        self.c = np.ones(n_points) * damping
        self.m = np.ones(n_points) * mass_per_length * self.dx
        
        self._apply_root_constraint()
        
    def _apply_root_constraint(self):
        self.k[0] = 1e10
        self.k[1] = 1e10
        self.m[0] = 1e10
        self.m[1] = 1e10
        
    def compute_deflection(self, forces):
        for i in range(2, self.n_points):
            F_net = forces[i] - self.k[i] * self.w[i] - self.c[i] * self.w_dot[i]
            self.w[i] += F_net / (self.k[i] + 1e-10) * 0.1
            
        self.w[0] = 0.0
        self.w[1] = 0.0
        
        return self.w.copy()
        
    def compute_velocity(self, forces, dt):
        for i in range(2, self.n_points):
            F_net = forces[i] - self.k[i] * self.w[i] - self.c[i] * self.w_dot[i]
            self.w_dot[i] += F_net / self.m[i] * dt
            
        self.w_dot[0] = 0.0
        self.w_dot[1] = 0.0
        
        return self.w_dot.copy()
        
    def update_position(self, dt):
        for i in range(2, self.n_points):
            self.w[i] += self.w_dot[i] * dt
            
        self.w[0] = 0.0
        self.w[1] = 0.0
        
    def get_camber_line(self):
        return self.x.copy(), self.w.copy()
        
    def reset(self):
        self.w = np.zeros(self.n_points)
        self.w_dot = np.zeros(self.n_points)


class StructuralDynamics:
    def __init__(self, model_type='spring', **kwargs):
        if model_type == 'beam':
            self.model = EulerBernoulliBeam(**kwargs)
        else:
            self.model = SimpleSpringModel(**kwargs)
            
        self.n_points = self.model.n_points
        self.x = self.model.x.copy()
        
    def compute_deformation(self, forces, dt=0.01, n_iter=10):
        for _ in range(n_iter):
            if isinstance(self.model, EulerBernoulliBeam):
                self.model.compute_deflection(forces)
            else:
                self.model.compute_deflection(forces)
                
        return self.get_camber_line()
        
    def time_step(self, forces, dt):
        if isinstance(self.model, SimpleSpringModel):
            self.model.compute_velocity(forces, dt)
            self.model.update_position(dt)
            
        return self.get_camber_line()
        
    def get_camber_line(self):
        return self.model.get_camber_line()
        
    def reset(self):
        self.model.reset()
        
    def get_slopes(self):
        if hasattr(self.model, 'theta'):
            return self.model.theta.copy()
        else:
            theta = np.zeros_like(self.x)
            w = self.model.w
            dx = self.x[1] - self.x[0]
            for i in range(1, len(self.x) - 1):
                theta[i] = (w[i+1] - w[i-1]) / (2 * dx)
            theta[0] = (w[1] - w[0]) / dx
            theta[-1] = (w[-1] - w[-2]) / dx
            return theta


def test_structural():
    print("Testing Structural Model")
    print("=" * 50)
    
    struct = StructuralDynamics(model_type='beam', n_points=50, chord=1.0, 
                                 thickness=0.12, E=70e9, rho=2700)
    
    forces = np.sin(np.pi * struct.x) * 1000.0
    
    x, y = struct.compute_deformation(forces)
    
    print(f"Max deflection: {np.max(np.abs(y)):.6f} m")
    print(f"Tip deflection: {y[-1]:.6f} m")
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y * 1000, 'b-', linewidth=2)
    plt.xlabel('x (m)')
    plt.ylabel('Deflection (mm)')
    plt.title('Beam Deflection')
    plt.grid(True)
    plt.savefig('beam_deflection.png', dpi=150)
    plt.close()
    print("Saved beam_deflection.png")
    
    return struct


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    test_structural()

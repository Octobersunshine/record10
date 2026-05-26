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
        
        self.K = None
        self.M = None
        
    def _update_properties(self):
        t_le = self.thickness * 1.0
        t_te = self.thickness * 0.01
        t = t_le - (t_le - t_te) * self.x / self.chord
        
        self.I = t**3 / 12.0
        self.A = t * 1.0
        self.m = self.rho * self.A * self.dx
        
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
                K[i, i+1] = -4 * coeff
                if n > 4:
                    K[i, i+2] = coeff
                    
            i = n - 2
            if n > 2:
                coeff = self.E * self.I[i] / self.dx**4
                if n > 4:
                    K[i, i-2] = coeff
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


def test():
    beam = EulerBernoulliBeam(n_points=20, chord=1.0, thickness=0.12, E=70e9)
    beam.assemble_stiffness_matrix()
    forces = np.sin(np.pi * beam.x) * 1000.0
    beam.compute_deflection(forces)
    print(f"Max deflection: {np.max(np.abs(beam.w)):.6f} m")
    print("Test PASSED")


if __name__ == "__main__":
    test()

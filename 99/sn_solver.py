import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


class DiscreteOrdinatesSolver:
    def __init__(self, geometry, cross_sections, quadrature, boundary_conditions,
                 use_positivity_fix=True, positivity_method='set_to_zero'):
        self.geometry = geometry
        self.cross_sections = cross_sections
        self.quadrature = quadrature
        self.bc = boundary_conditions
        
        self.N = geometry['N']
        self.L = geometry['L']
        self.dx = self.L / self.N
        self.x = np.linspace(self.dx/2, self.L - self.dx/2, self.N)
        
        self.M = len(quadrature['mu'])
        self.mu = quadrature['mu']
        self.w = quadrature['w']
        
        self.sigma_t = cross_sections['sigma_t']
        self.sigma_s = cross_sections['sigma_s']
        self.sigma_a = self.sigma_t - self.sigma_s
        self.Q = cross_sections.get('Q', np.zeros(self.N))
        
        self.phi = np.zeros((self.N,))
        self.psi = np.zeros((self.N, self.M))
        
        self.bc_type = boundary_conditions.get('type', 'incident')
        
        self.use_positivity_fix = use_positivity_fix
        self.positivity_method = positivity_method
        self.negative_psi_count = 0
        self.negative_phi_count = 0
        
    def source_iteration(self, tol=1e-8, max_iter=1000, use_dsa=False, dsa_inner_iter=1):
        phi_old = np.ones(self.N)
        converged = False
        self.negative_psi_count = 0
        self.negative_phi_count = 0
        
        for it in range(max_iter):
            self._sweep(phi_old)
            
            phi_new = np.zeros(self.N)
            for i in range(self.N):
                phi_new[i] = np.sum(self.w * self.psi[i, :])
            
            if self.use_positivity_fix:
                phi_new = self._enforce_positivity(phi_new)
            
            if use_dsa and it % dsa_inner_iter == 0:
                phi_new = self._diffusion_synthetic_acceleration(phi_old, phi_new)
                if self.use_positivity_fix:
                    phi_new = self._enforce_positivity(phi_new)
            
            res = np.linalg.norm(phi_new - phi_old) / (np.linalg.norm(phi_new) + 1e-10)
            
            if res < tol:
                converged = True
                break
                
            phi_old = phi_new.copy()
        
        self.phi = phi_new
        
        if self.use_positivity_fix and (self.negative_psi_count > 0 or self.negative_phi_count > 0):
            print(f"  Positivity fix applied: {self.negative_psi_count} angular, {self.negative_phi_count} scalar")
        
        return converged, it + 1, res
    
    def _sweep(self, phi):
        Q_total = np.zeros((self.N, self.M))
        for i in range(self.N):
            scatter = 0.5 * self.sigma_s[i] * phi[i]
            for m in range(self.M):
                Q_total[i, m] = scatter + 0.5 * self.Q[i]
        
        for m in range(self.M):
            mu = self.mu[m]
            if mu > 0:
                if self.bc_type == 'reflective':
                    psi_edge = self._get_reflective_boundary(m, 'left')
                else:
                    psi_edge = self.bc.get('left', 0.0)
                for i in range(self.N):
                    denom = mu + self.sigma_t[i] * self.dx
                    self.psi[i, m] = (mu * psi_edge + Q_total[i, m] * self.dx) / denom
                    psi_edge = 2 * self.psi[i, m] - psi_edge
            else:
                if self.bc_type == 'reflective':
                    psi_edge = self._get_reflective_boundary(m, 'right')
                else:
                    psi_edge = self.bc.get('right', 0.0)
                for i in range(self.N - 1, -1, -1):
                    denom = -mu + self.sigma_t[i] * self.dx
                    self.psi[i, m] = (-mu * psi_edge + Q_total[i, m] * self.dx) / denom
                    psi_edge = 2 * self.psi[i, m] - psi_edge
        
        if self.use_positivity_fix:
            self._enforce_angular_positivity()
    
    def _get_reflective_boundary(self, m, side):
        mu = self.mu[m]
        m_mirror = np.argmin(np.abs(self.mu + mu))
        
        if side == 'left':
            if m_mirror >= self.M:
                m_mirror = self.M - 1
            if self.psi[0, m_mirror] > 0:
                return self.psi[0, m_mirror]
            else:
                return 0.0
        else:
            if m_mirror >= self.M:
                m_mirror = self.M - 1
            if self.psi[-1, m_mirror] > 0:
                return self.psi[-1, m_mirror]
            else:
                return 0.0
    
    def _enforce_angular_positivity(self):
        if self.positivity_method == 'set_to_zero':
            negative_mask = self.psi < 0
            if np.any(negative_mask):
                self.negative_psi_count += np.sum(negative_mask)
                self.psi[negative_mask] = 0.0
        elif self.positivity_method == 'scale':
            for i in range(self.N):
                for m in range(self.M):
                    if self.psi[i, m] < 0:
                        self.psi[i, m] = 0.0
                        self.negative_psi_count += 1
    
    def _enforce_positivity(self, phi):
        negative_mask = phi < 0
        if np.any(negative_mask):
            self.negative_phi_count += np.sum(negative_mask)
            if self.positivity_method == 'set_to_zero':
                phi[negative_mask] = 0.0
            elif self.positivity_method == 'scale':
                min_phi = np.min(phi)
                if min_phi < 0:
                    phi = phi - min_phi + 1e-10
        return phi
    
    def _diffusion_synthetic_acceleration(self, phi_old, phi_new):
        D = 1.0 / (3.0 * self.sigma_t)
        sigma_a = self.sigma_a
        
        main_diag = np.zeros(self.N)
        off_diag = np.zeros(self.N - 1)
        
        for i in range(self.N):
            D_left = D[i] if i == 0 else 0.5 * (D[i] + D[i-1])
            D_right = D[i] if i == self.N - 1 else 0.5 * (D[i] + D[i+1])
            
            main_diag[i] = (D_left + D_right) / self.dx**2 + sigma_a[i]
            if i > 0:
                off_diag[i-1] = -D_left / self.dx**2
            if i < self.N - 1:
                off_diag[i] = -D_right / self.dx**2
        
        if self.bc_type == 'vacuum' or self.bc_type == 'incident':
            main_diag[0] += 2 * D[0] / (self.dx**2)
            main_diag[-1] += 2 * D[-1] / (self.dx**2)
        
        A = diags([off_diag, main_diag, off_diag], [-1, 0, 1], format='csr')
        b = self.sigma_a * (phi_new - phi_old) + (phi_new - phi_old) * self.sigma_s
        
        try:
            correction = spsolve(A, b)
            phi_corrected = phi_new - correction
            phi_corrected = np.maximum(phi_corrected, 0.0)
        except:
            phi_corrected = phi_new
        
        return phi_corrected
    
    def solve(self, method='source_iteration', **kwargs):
        if method == 'source_iteration':
            return self.source_iteration(**kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def get_flux(self):
        return self.phi
    
    def get_angular_flux(self):
        return self.psi
    
    def get_current(self):
        J = np.zeros(self.N)
        for i in range(self.N):
            J[i] = np.sum(self.w * self.mu * self.psi[i, :])
        return J
    
    def get_reaction_rates(self):
        absorption_rate = self.sigma_a * self.phi
        fission_rate = np.zeros_like(self.phi)
        if 'nu_sigma_f' in self.cross_sections:
            fission_rate = self.cross_sections['nu_sigma_f'] * self.phi
        return absorption_rate, fission_rate
    
    def check_negative_flux(self):
        neg_psi = np.sum(self.psi < 0)
        neg_phi = np.sum(self.phi < 0)
        min_psi = np.min(self.psi)
        min_phi = np.min(self.phi)
        
        return {
            'negative_angular_cells': neg_psi,
            'negative_scalar_cells': neg_phi,
            'min_angular_flux': min_psi,
            'min_scalar_flux': min_phi,
            'has_negative': (neg_psi > 0 or neg_phi > 0)
        }

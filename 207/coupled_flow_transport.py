import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve


class SoilPropertyModel:
    def __init__(self, model_type='van_genuchten'):
        self.model_type = model_type
        self.params = {
            'theta_r': 0.08,
            'theta_s': 0.43,
            'alpha': 3.5,
            'n': 1.6,
            'Ks': 0.0005,
            'l': 0.5
        }
    
    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value
    
    def effective_saturation(self, h):
        if self.model_type == 'van_genuchten':
            return self._van_genuchten_Se(h)
        return 0.5
    
    def moisture_content(self, h):
        Se = self.effective_saturation(h)
        return self.params['theta_r'] + Se * (self.params['theta_s'] - self.params['theta_r'])
    
    def hydraulic_conductivity(self, h):
        Se = self.effective_saturation(h)
        Ks = self.params['Ks']
        
        if self.model_type == 'van_genuchten':
            return self._mualem_K(Se, Ks)
        return Ks * Se
    
    def specific_moisture_capacity(self, h):
        if self.model_type == 'van_genuchten':
            return self._van_genuchten_C(h)
        return 0.001
    
    def _van_genuchten_Se(self, h):
        if h >= 0:
            return 1.0
        
        alpha = self.params['alpha']
        n = self.params['n']
        m = 1 - 1/n
        
        x = (alpha * abs(h)) ** n
        Se = (1 + x) ** (-m)
        return np.clip(Se, 1e-10, 1.0)
    
    def _van_genuchten_C(self, h):
        if h >= 0:
            return 1e-10
        
        alpha = self.params['alpha']
        n = self.params['n']
        m = 1 - 1/n
        theta_r = self.params['theta_r']
        theta_s = self.params['theta_s']
        
        x = (alpha * abs(h)) ** n
        dSe_dh = alpha * m * n * (alpha * abs(h)) ** (n - 1)
        dSe_dh *= (1 + x) ** (-m - 1) / h
        
        C = dSe_dh * (theta_s - theta_r)
        return max(abs(C), 1e-10)
    
    def _mualem_K(self, Se, Ks):
        l = self.params['l']
        n = self.params['n']
        m = 1 - 1/n
        
        term1 = Se ** l
        term2 = (1 - (1 - Se ** (1/m)) ** m) ** 2
        
        return Ks * term1 * term2


class UnsaturatedFlowSolver:
    def __init__(self, nx, ny, dx, dy, dt=1.0):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.dt = dt
        
        self.h = np.ones((ny, nx)) * -1.0
        self.theta = np.ones((ny, nx)) * 0.2
        
        self.soil_model = SoilPropertyModel()
        
        self.bc_dirichlet = {}
        self.bc_flux = {}
        
        self.source_terms = np.zeros((ny, nx))
    
    def set_soil_params(self, **kwargs):
        self.soil_model.set_params(**kwargs)
        self._update_properties()
    
    def _update_properties(self):
        for j in range(self.ny):
            for i in range(self.nx):
                self.theta[j, i] = self.soil_model.moisture_content(self.h[j, i])
    
    def set_initial_head(self, h0):
        if isinstance(h0, (int, float)):
            self.h = np.ones((self.ny, self.nx)) * h0
        else:
            self.h = h0.copy()
        self._update_properties()
    
    def set_initial_moisture(self, theta0):
        if isinstance(theta0, (int, float)):
            self.theta = np.ones((self.ny, self.nx)) * theta0
        else:
            self.theta = theta0.copy()
    
    def set_dirichlet_bc(self, i, j, value):
        self.bc_dirichlet[(i, j)] = value
    
    def set_flux_bc(self, i, j, value):
        self.bc_flux[(i, j)] = value
    
    def set_neumann_bc(self, i, j, value):
        self.bc_flux[(i, j)] = value
    
    def set_source(self, i, j, value):
        self.source_terms[j, i] = value
    
    def _node_idx(self, i, j):
        return j * self.nx + i
    
    def _harmonic_mean_K(self, h1, h2):
        K1 = self.soil_model.hydraulic_conductivity(h1)
        K2 = self.soil_model.hydraulic_conductivity(h2)
        if K1 <= 0 or K2 <= 0:
            return 0.0
        return 2 * K1 * K2 / (K1 + K2)
    
    def assemble_matrix_rhs(self, h_old, theta_old):
        n = self.nx * self.ny
        A = lil_matrix((n, n))
        b = np.zeros(n)
        
        for j in range(self.ny):
            for i in range(self.nx):
                idx = self._node_idx(i, j)
                
                if (i, j) in self.bc_dirichlet:
                    A[idx, idx] = 1.0
                    b[idx] = self.bc_dirichlet[(i, j)]
                    continue
                
                K_center = self.soil_model.hydraulic_conductivity(h_old[j, i])
                C = self.soil_model.specific_moisture_capacity(h_old[j, i])
                
                coeff = C / self.dt
                
                if i > 0:
                    K_left = self._harmonic_mean_K(h_old[j, i], h_old[j, i-1])
                    val = K_left / (self.dx ** 2)
                    A[idx, self._node_idx(i-1, j)] = -val
                    coeff += val
                else:
                    if (i, j) in self.bc_flux:
                        b[idx] += self.bc_flux[(i, j)] / self.dx
                
                if i < self.nx - 1:
                    K_right = self._harmonic_mean_K(h_old[j, i], h_old[j, i+1])
                    val = K_right / (self.dx ** 2)
                    A[idx, self._node_idx(i+1, j)] = -val
                    coeff += val
                else:
                    if (i, j) in self.bc_flux:
                        b[idx] -= self.bc_flux[(i, j)] / self.dx
                
                if j > 0:
                    K_bottom = self._harmonic_mean_K(h_old[j, i], h_old[j-1, i])
                    val = K_bottom / (self.dy ** 2)
                    A[idx, self._node_idx(i, j-1)] = -val
                    coeff += val
                else:
                    if (i, j) in self.bc_flux:
                        b[idx] += self.bc_flux[(i, j)] / self.dy
                
                if j < self.ny - 1:
                    K_top = self._harmonic_mean_K(h_old[j, i], h_old[j+1, i])
                    val = K_top / (self.dy ** 2)
                    A[idx, self._node_idx(i, j+1)] = -val
                    coeff += val
                else:
                    if (i, j) in self.bc_flux:
                        b[idx] -= self.bc_flux[(i, j)] / self.dy
                
                A[idx, idx] = coeff
                
                b[idx] = C / self.dt * h_old[j, i]
                b[idx] += self.source_terms[j, i]
        
        return csr_matrix(A), b
    
    def solve_step(self, max_iter=20, tol=1e-6):
        h_old = self.h.copy()
        theta_old = self.theta.copy()
        
        for iteration in range(max_iter):
            A, b = self.assemble_matrix_rhs(h_old, theta_old)
            
            h_new = spsolve(A, b)
            h_new = h_new.reshape((self.ny, self.nx))
            
            max_change = np.max(np.abs(h_new - h_old))
            
            h_old = h_new.copy()
            self._update_moisture(h_old)
            
            if max_change < tol:
                break
        
        self.h = h_old
        self._update_properties()
        
        return self.h
    
    def _update_moisture(self, h):
        for j in range(self.ny):
            for i in range(self.nx):
                self.theta[j, i] = self.soil_model.moisture_content(h[j, i])
    
    def compute_velocity(self):
        u = np.zeros((self.ny, self.nx))
        v = np.zeros((self.ny, self.nx))
        
        for j in range(self.ny):
            for i in range(self.nx):
                K = self.soil_model.hydraulic_conductivity(self.h[j, i])
                
                if 0 < i < self.nx - 1:
                    u[j, i] = -K * (self.h[j, i+1] - self.h[j, i-1]) / (2 * self.dx)
                elif i == 0:
                    u[j, i] = -K * (self.h[j, i+1] - self.h[j, i]) / self.dx
                else:
                    u[j, i] = -K * (self.h[j, i] - self.h[j, i-1]) / self.dx
                
                if 0 < j < self.ny - 1:
                    v[j, i] = -K * (self.h[j+1, i] - self.h[j-1, i]) / (2 * self.dy)
                elif j == 0:
                    v[j, i] = -K * (self.h[j+1, i] - self.h[j, i]) / self.dy
                else:
                    v[j, i] = -K * (self.h[j, i] - self.h[j-1, i]) / self.dy
        
        return u, v
    
    def compute_darcy_velocity(self):
        u, v = self.compute_velocity()
        return u / self.theta, v / self.theta


class SoluteTransportSolver:
    def __init__(self, nx, ny, dx, dy, dt=0.1):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.dt = dt
        
        self.C = np.zeros((ny, nx))
        
        self.DL = 1.0
        self.DT = 0.1
        
        self.rho_b = 1.6
        self.Kd = 0.0
        self.lambda_decay = 0.0
        
        self.bc_concentration = {}
        self.bc_mass_flux = {}
        
        self.source_terms = np.zeros((ny, nx))
    
    def set_params(self, DL=1.0, DT=0.1, rho_b=1.6, Kd=0.0, lambda_decay=0.0):
        self.DL = DL
        self.DT = DT
        self.rho_b = rho_b
        self.Kd = Kd
        self.lambda_decay = lambda_decay
    
    def set_initial_concentration(self, C0):
        if isinstance(C0, (int, float)):
            self.C = np.ones((self.ny, self.nx)) * C0
        else:
            self.C = C0.copy()
    
    def set_dirichlet_bc(self, i, j, value):
        self.bc_concentration[(i, j)] = value
    
    def set_flux_bc(self, i, j, value):
        self.bc_mass_flux[(i, j)] = value
    
    def set_source(self, i, j, value):
        self.source_terms[j, i] = value
    
    def _node_idx(self, i, j):
        return j * self.nx + i
    
    def _get_retardation_factor(self):
        porosity = 0.3
        R = 1 + (self.rho_b * self.Kd) / porosity
        return R
    
    def assemble_matrix_rhs(self, C_old, u, v, theta):
        n = self.nx * self.ny
        A = lil_matrix((n, n))
        b = np.zeros(n)
        
        R = self._get_retardation_factor()
        porosity = 0.3
        
        for j in range(self.ny):
            for i in range(self.nx):
                idx = self._node_idx(i, j)
                
                if (i, j) in self.bc_concentration:
                    A[idx, idx] = 1.0
                    b[idx] = self.bc_concentration[(i, j)]
                    continue
                
                D_eff = self._effective_dispersion(u[j, i], v[j, i], theta[j, i])
                
                coeff = theta[j, i] / (R * self.dt)
                
                if i > 0:
                    D_left = 0.5 * (D_eff + self._effective_dispersion(
                        u[j, i-1], v[j, i-1], theta[j, i-1]))
                    adv = u[j, i] * theta[j, i] / (2 * self.dx)
                    val = D_left / (self.dx ** 2) - adv
                    A[idx, self._node_idx(i-1, j)] = -val
                    coeff += val
                else:
                    if (i, j) in self.bc_mass_flux:
                        b[idx] += self.bc_mass_flux[(i, j)] / (R * self.dx)
                
                if i < self.nx - 1:
                    D_right = 0.5 * (D_eff + self._effective_dispersion(
                        u[j, i+1], v[j, i+1], theta[j, i+1]))
                    adv = u[j, i] * theta[j, i] / (2 * self.dx)
                    val = D_right / (self.dx ** 2) + adv
                    A[idx, self._node_idx(i+1, j)] = -val
                    coeff += val
                else:
                    if (i, j) in self.bc_mass_flux:
                        b[idx] -= self.bc_mass_flux[(i, j)] / (R * self.dx)
                
                if j > 0:
                    D_bottom = 0.5 * (D_eff + self._effective_dispersion(
                        u[j-1, i], v[j-1, i], theta[j-1, i]))
                    adv = v[j, i] * theta[j, i] / (2 * self.dy)
                    val = D_bottom / (self.dy ** 2) - adv
                    A[idx, self._node_idx(i, j-1)] = -val
                    coeff += val
                else:
                    if (i, j) in self.bc_mass_flux:
                        b[idx] += self.bc_mass_flux[(i, j)] / (R * self.dy)
                
                if j < self.ny - 1:
                    D_top = 0.5 * (D_eff + self._effective_dispersion(
                        u[j+1, i], v[j+1, i], theta[j+1, i]))
                    adv = v[j, i] * theta[j, i] / (2 * self.dy)
                    val = D_top / (self.dy ** 2) + adv
                    A[idx, self._node_idx(i, j+1)] = -val
                    coeff += val
                else:
                    if (i, j) in self.bc_mass_flux:
                        b[idx] -= self.bc_mass_flux[(i, j)] / (R * self.dy)
                
                A[idx, idx] = coeff + self.lambda_decay * theta[j, i] / R
                
                b[idx] = theta[j, i] / (R * self.dt) * C_old[j, i]
                b[idx] += self.source_terms[j, i]
        
        return csr_matrix(A), b
    
    def _effective_dispersion(self, u, v, theta):
        velocity_mag = np.sqrt(u**2 + v**2)
        
        D = np.zeros((2, 2))
        if velocity_mag > 1e-15:
            alpha_L = self.DL
            alpha_T = self.DT
            
            D[0, 0] = alpha_L * u**2 / velocity_mag + alpha_T * v**2 / velocity_mag
            D[0, 1] = (alpha_L - alpha_T) * u * v / velocity_mag
            D[1, 0] = D[0, 1]
            D[1, 1] = alpha_L * v**2 / velocity_mag + alpha_T * u**2 / velocity_mag
        
        Dm = 1e-9
        
        D_eff = D[0, 0] + Dm
        return max(D_eff, 1e-10)
    
    def solve_step(self, u, v, theta):
        C_old = self.C.copy()
        
        A, b = self.assemble_matrix_rhs(C_old, u, v, theta)
        
        C_new = spsolve(A, b)
        self.C = C_new.reshape((self.ny, self.nx))
        self.C = np.clip(self.C, 0, None)
        
        return self.C


class CoupledFlowTransportModel:
    def __init__(self, nx, ny, dx, dy, dt_flow=1.0, dt_transport=0.1):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        
        self.flow_solver = UnsaturatedFlowSolver(nx, ny, dx, dy, dt_flow)
        self.transport_solver = SoluteTransportSolver(nx, ny, dx, dy, dt_transport)
        
        self.time = 0.0
        self.output_times = []
        self.output_concentrations = []
        self.output_heads = []
    
    def set_soil_properties(self, **kwargs):
        self.flow_solver.set_soil_params(**kwargs)
    
    def set_transport_params(self, **kwargs):
        self.transport_solver.set_params(**kwargs)
    
    def set_initial_conditions(self, h0=None, C0=None, theta0=None):
        if h0 is not None:
            self.flow_solver.set_initial_head(h0)
        if theta0 is not None:
            self.flow_solver.set_initial_moisture(theta0)
        if C0 is not None:
            self.transport_solver.set_initial_concentration(C0)
    
    def set_boundary_conditions(self, flow_bc=None, transport_bc=None):
        if flow_bc is not None:
            for bc_type, i, j, value in flow_bc:
                if bc_type == 'dirichlet':
                    self.flow_solver.set_dirichlet_bc(i, j, value)
                elif bc_type == 'flux':
                    self.flow_solver.set_flux_bc(i, j, value)
        
        if transport_bc is not None:
            for bc_type, i, j, value in transport_bc:
                if bc_type == 'dirichlet':
                    self.transport_solver.set_dirichlet_bc(i, j, value)
                elif bc_type == 'flux':
                    self.transport_solver.set_flux_bc(i, j, value)
    
    def set_sources(self, flow_sources=None, transport_sources=None):
        if flow_sources is not None:
            for i, j, value in flow_sources:
                self.flow_solver.set_source(i, j, value)
        
        if transport_sources is not None:
            for i, j, value in transport_sources:
                self.transport_solver.set_source(i, j, value)
    
    def simulate(self, total_time, output_interval=None):
        n_steps_flow = int(total_time / self.flow_solver.dt)
        flow_steps_per_transport = max(1, int(self.flow_solver.dt / self.transport_solver.dt))
        
        if output_interval is None:
            output_interval = total_time / 10
        
        next_output_time = output_interval
        
        self._save_output(0.0)
        
        for step in range(n_steps_flow):
            self.flow_solver.solve_step()
            
            for t_step in range(flow_steps_per_transport):
                u, v = self.flow_solver.compute_darcy_velocity()
                self.transport_solver.solve_step(u, v, self.flow_solver.theta)
            
            self.time = (step + 1) * self.flow_solver.dt
            
            if self.time >= next_output_time:
                self._save_output(self.time)
                next_output_time += output_interval
        
        self._save_output(self.time)
        
        return self.flow_solver.h, self.transport_solver.C
    
    def _save_output(self, time):
        self.output_times.append(time)
        self.output_concentrations.append(self.transport_solver.C.copy())
        self.output_heads.append(self.flow_solver.h.copy())
    
    def get_concentration_history(self):
        return self.output_times, self.output_concentrations
    
    def get_head_history(self):
        return self.output_times, self.output_heads
    
    def compute_mass_balance(self):
        total_mass = np.sum(self.transport_solver.C * self.flow_solver.theta) * self.dx * self.dy
        
        u, v = self.flow_solver.compute_velocity()
        flux_mass = 0.0
        
        for j in range(self.ny):
            flux_mass += self.transport_solver.C[j, 0] * u[j, 0] * self.dy
            flux_mass -= self.transport_solver.C[j, -1] * u[j, -1] * self.dy
        
        for i in range(self.nx):
            flux_mass += self.transport_solver.C[0, i] * v[0, i] * self.dx
            flux_mass -= self.transport_solver.C[-1, i] * v[-1, i] * self.dx
        
        return total_mass, flux_mass

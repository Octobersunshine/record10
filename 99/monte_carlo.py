import numpy as np


class MonteCarloNeutron:
    def __init__(self, x, mu, weight=1.0):
        self.x = x
        self.mu = mu
        self.weight = weight
        self.alive = True


class MonteCarloSolver:
    def __init__(self, geometry, cross_sections, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.geometry = geometry
        self.cross_sections = cross_sections
        
        self.N = geometry['N']
        self.L = geometry['L']
        self.dx = self.L / self.N
        self.x_edges = np.linspace(0, self.L, self.N + 1)
        
        self.sigma_t = cross_sections['sigma_t']
        self.sigma_s = cross_sections['sigma_s']
        self.sigma_a = self.sigma_t - self.sigma_s
        self.Q = cross_sections.get('Q', np.zeros(self.N))
        
        self.phi = np.zeros(self.N)
        self.phi_sq = np.zeros(self.N)
        self.tally_count = np.zeros(self.N)
        
        self.angular_surface_flux_left = np.zeros(self.N + 1)
        self.angular_surface_flux_right = np.zeros(self.N + 1)
        
        self.left_leakage = 0.0
        self.right_leakage = 0.0
        self.absorbed = 0.0
        
        self.boundary_current_left = 0.0
        self.boundary_current_right = 0.0
        
        self.incident_sources_left = []
        self.incident_sources_right = []
        
    def _get_cell_index(self, x):
        if x <= 0:
            return 0
        if x >= self.L:
            return self.N - 1
        idx = int(np.floor(x / self.dx))
        return np.clip(idx, 0, self.N - 1)
    
    def _sample_distance(self, sigma_t):
        return -np.log(np.random.random() + 1e-10) / sigma_t
    
    def _sample_scattering(self):
        return 2.0 * np.random.random() - 1.0
    
    def _sample_isotropic_source(self):
        return 2.0 * np.random.random() - 1.0
    
    def set_incident_source(self, side, particles):
        if side == 'left':
            self.incident_sources_left = particles
        else:
            self.incident_sources_right = particles
    
    def run_source_simulation(self, n_particles=10000, source_type='uniform'):
        self.phi = np.zeros(self.N)
        self.phi_sq = np.zeros(self.N)
        self.tally_count = np.zeros(self.N)
        self.left_leakage = 0.0
        self.right_leakage = 0.0
        self.absorbed = 0.0
        self.boundary_current_left = 0.0
        self.boundary_current_right = 0.0
        
        total_particles = n_particles
        particle_list = []
        
        if source_type == 'uniform' and np.sum(self.Q) > 0:
            q_total = np.sum(self.Q)
            if q_total > 0:
                q_prob = self.Q / q_total
                for _ in range(n_particles):
                    cell = np.random.choice(self.N, p=q_prob)
                    x = self.x_edges[cell] + np.random.random() * self.dx
                    mu = self._sample_isotropic_source()
                    weight = q_total * self.dx / n_particles
                    particle_list.append(MonteCarloNeutron(x, mu, weight))
        
        if source_type == 'left_boundary':
            for _ in range(n_particles):
                x = 0.0
                mu = np.sqrt(np.random.random())
                weight = 1.0 / n_particles
                particle_list.append(MonteCarloNeutron(x, mu, weight))
        
        for src in self.incident_sources_left:
            particle_list.append(MonteCarloNeutron(0.0, src['mu'], src['weight']))
        
        for neutron in particle_list:
            self._track_neutron(neutron)
        
        self.phi /= self.dx
        
        valid = self.tally_count > 0
        self.phi_var = np.zeros(self.N)
        self.phi_var[valid] = (self.phi_sq[valid]/self.dx**2 - self.phi[valid]**2) / np.maximum(self.tally_count[valid] - 1, 1)
        
        return self.phi
    
    def _track_neutron(self, neutron):
        while neutron.alive:
            cell = self._get_cell_index(neutron.x)
            sigma_t = self.sigma_t[cell]
            
            dist_to_collision = self._sample_distance(sigma_t)
            
            if neutron.mu > 0:
                dist_to_boundary = (self.x_edges[cell + 1] - neutron.x) / neutron.mu
            else:
                dist_to_boundary = (self.x_edges[cell] - neutron.x) / neutron.mu
            
            if dist_to_collision < abs(dist_to_boundary):
                neutron.x += neutron.mu * dist_to_collision
                
                path_length = dist_to_collision
                cell_new = self._get_cell_index(neutron.x)
                self.phi[cell_new] += neutron.weight * path_length
                self.phi_sq[cell_new] += (neutron.weight * path_length) ** 2
                self.tally_count[cell_new] += 1
                
                if np.random.random() < self.sigma_s[cell_new] / sigma_t:
                    neutron.mu = self._sample_scattering()
                else:
                    self.absorbed += neutron.weight
                    neutron.alive = False
            else:
                neutron.x += neutron.mu * abs(dist_to_boundary)
                
                path_length = abs(dist_to_boundary)
                self.phi[cell] += neutron.weight * path_length
                self.phi_sq[cell] += (neutron.weight * path_length) ** 2
                self.tally_count[cell] += 1
                
                if neutron.x <= 0:
                    self.left_leakage += neutron.weight
                    if neutron.mu < 0:
                        self.boundary_current_left += neutron.weight * abs(neutron.mu)
                    neutron.alive = False
                elif neutron.x >= self.L:
                    self.right_leakage += neutron.weight
                    if neutron.mu > 0:
                        self.boundary_current_right += neutron.weight * neutron.mu
                    neutron.alive = False
    
    def run_fixed_source(self, n_particles=10000):
        return self.run_source_simulation(n_particles, 'uniform')
    
    def run_incident_source(self, n_particles=10000):
        return self.run_source_simulation(n_particles, 'left_boundary')
    
    def get_flux(self):
        return self.phi
    
    def get_flux_uncertainty(self):
        return np.sqrt(self.phi_var / np.maximum(self.tally_count, 1))
    
    def get_leakage(self):
        return self.left_leakage, self.right_leakage
    
    def get_boundary_currents(self):
        return self.boundary_current_left, self.boundary_current_right


class MCSNCoupling:
    def __init__(self, full_geometry, cross_sections, mc_regions, n_particles=20000):
        self.full_geometry = full_geometry
        self.cross_sections = cross_sections
        self.mc_regions = mc_regions
        self.n_particles = n_particles
        
        self.N = full_geometry['N']
        self.L = full_geometry['L']
        self.dx = self.L / self.N
        self.x = np.linspace(self.dx/2, self.L - self.dx/2, self.N)
        
        self.mc_mask = np.zeros(self.N, dtype=bool)
        for region in mc_regions:
            start, end = region
            start_idx = int(start / self.dx)
            end_idx = int(end / self.dx)
            self.mc_mask[start_idx:end_idx] = True
        
        self.sn_mask = ~self.mc_mask
        
        self.phi = np.zeros(self.N)
        self.phi_mc = np.zeros(self.N)
        self.phi_sn = np.zeros(self.N)
        
        self.mc_statistics = []
        
    def get_region_geometry(self, start_idx, end_idx):
        n_cells = end_idx - start_idx
        return {
            'N': n_cells,
            'L': n_cells * self.dx
        }
    
    def get_region_cross_sections(self, start_idx, end_idx):
        return {
            'sigma_t': self.cross_sections['sigma_t'][start_idx:end_idx],
            'sigma_s': self.cross_sections['sigma_s'][start_idx:end_idx],
            'Q': self.cross_sections.get('Q', np.zeros(self.N))[start_idx:end_idx]
        }
    
    def solve_coupled(self, sn_solver_class, quadrature, boundary_conditions,
                      use_positivity_fix=True, max_iter=5, tol=1e-2):
        phi_old = np.ones(self.N)
        
        print(f"\nMC-SN Coupled Solver")
        print(f"=" * 50)
        print(f"Total cells: {self.N}")
        print(f"SN cells: {np.sum(self.sn_mask)}")
        print(f"MC cells: {np.sum(self.mc_mask)}")
        print(f"MC particles per region: {self.n_particles}")
        
        for iteration in range(max_iter):
            print(f"\nIteration {iteration + 1}/{max_iter}")
            
            self._solve_sn_regions(sn_solver_class, quadrature, boundary_conditions,
                                 use_positivity_fix, phi_old)
            
            self._solve_mc_regions(phi_old)
            
            self.phi = self.phi_sn.copy()
            self.phi[self.mc_mask] = self.phi_mc[self.mc_mask]
            
            res = np.linalg.norm(self.phi - phi_old) / (np.linalg.norm(self.phi) + 1e-10)
            print(f"  Residual: {res:.4e}")
            
            if res < tol:
                print(f"Converged!")
                break
            
            phi_old = self.phi.copy()
        
        return self.phi
    
    def _solve_sn_regions(self, sn_solver_class, quadrature, boundary_conditions,
                          use_positivity_fix, phi_guess):
        sn_regions = self._get_contiguous_regions(self.sn_mask)
        
        self.phi_sn = np.zeros(self.N)
        
        for start_idx, end_idx in sn_regions:
            if end_idx - start_idx < 2:
                continue
            
            region_geo = self.get_region_geometry(start_idx, end_idx)
            region_xs = self.get_region_cross_sections(start_idx, end_idx)
            
            region_bc = self._get_region_boundary_conditions(
                start_idx, end_idx, boundary_conditions, phi_guess
            )
            
            solver = sn_solver_class(
                region_geo, region_xs, quadrature, region_bc,
                use_positivity_fix=use_positivity_fix
            )
            solver.solve(tol=1e-6, max_iter=500)
            
            self.phi_sn[start_idx:end_idx] = solver.get_flux()
        
        if np.any(self.phi_sn > 0):
            print(f"  SN regions solved: max flux = {np.max(self.phi_sn[self.phi_sn > 0]):.4e}")
    
    def _solve_mc_regions(self, phi_guess):
        mc_regions = self._get_contiguous_regions(self.mc_mask)
        
        self.phi_mc = np.zeros(self.N)
        
        for start_idx, end_idx in mc_regions:
            if end_idx - start_idx < 2:
                continue
            
            region_geo = self.get_region_geometry(start_idx, end_idx)
            region_xs = self.get_region_cross_sections(start_idx, end_idx)
            
            mc_solver = MonteCarloSolver(region_geo, region_xs, seed=42)
            
            left_source = []
            if start_idx > 0 and self.sn_mask[start_idx - 1]:
                boundary_flux = phi_guess[start_idx - 1]
                n_source = int(self.n_particles * 0.1)
                for _ in range(n_source):
                    mu = np.sqrt(np.random.random())
                    left_source.append({'mu': mu, 'weight': boundary_flux / n_source})
            
            mc_solver.set_incident_source('left', left_source)
            
            mc_solver.run_fixed_source(n_particles=self.n_particles)
            
            self.phi_mc[start_idx:end_idx] = mc_solver.get_flux()
            
            stats = {
                'region': (start_idx, end_idx),
                'max_flux': np.max(mc_solver.get_flux()),
                'leakage': mc_solver.get_leakage(),
                'mean_uncertainty': np.mean(mc_solver.get_flux_uncertainty())
            }
            self.mc_statistics.append(stats)
            
            print(f"  MC region [{start_idx}:{end_idx}]: max flux = {stats['max_flux']:.4e}, "
                  f"mean sigma = {stats['mean_uncertainty']:.2e}")
    
    def _get_contiguous_regions(self, mask):
        regions = []
        in_region = False
        start = 0
        
        for i in range(len(mask)):
            if mask[i] and not in_region:
                start = i
                in_region = True
            elif not mask[i] and in_region:
                regions.append((start, i))
                in_region = False
        
        if in_region:
            regions.append((start, len(mask)))
        
        return regions
    
    def _get_region_boundary_conditions(self, start_idx, end_idx, global_bc, phi_guess):
        region_bc = {'type': 'vacuum'}
        
        if start_idx == 0:
            region_bc['left'] = global_bc.get('left', 0.0)
        else:
            avg_phi = phi_guess[start_idx] if start_idx < len(phi_guess) else 0.1
            region_bc['left'] = max(avg_phi * 0.25, 1e-6)
        
        if end_idx == self.N:
            region_bc['right'] = global_bc.get('right', 0.0)
        else:
            avg_phi = phi_guess[end_idx - 1] if end_idx - 1 < len(phi_guess) else 0.1
            region_bc['right'] = max(avg_phi * 0.25, 1e-6)
        
        return region_bc
    
    def get_flux(self):
        return self.phi
    
    def get_mc_mask(self):
        return self.mc_mask
    
    def get_mc_statistics(self):
        return self.mc_statistics


class DeepPenetrationBenchmark:
    def __init__(self, L=20.0, sigma_t=1.0, sigma_s_ratio=0.9, N=200):
        self.L = L
        self.sigma_t = sigma_t
        self.sigma_s_ratio = sigma_s_ratio
        self.N = N
        self.dx = L / N
        
    def get_geometry(self):
        return {'N': self.N, 'L': self.L}
    
    def get_cross_sections(self):
        sigma_t = self.sigma_t * np.ones(self.N)
        sigma_s = self.sigma_t * self.sigma_s_ratio * np.ones(self.N)
        Q = np.zeros(self.N)
        Q[0] = 10.0 / self.dx
        
        return {
            'sigma_t': sigma_t,
            'sigma_s': sigma_s,
            'Q': Q
        }
    
    def get_mc_regions_deep(self, threshold_x=10.0):
        return [(threshold_x, self.L)]
    
    def get_mc_regions_alternating(self):
        regions = []
        for i in range(2, int(self.L/2)):
            if i % 2 == 0:
                regions.append((i*2, i*2 + 2))
        return regions


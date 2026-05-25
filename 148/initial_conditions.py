import numpy as np
from cosmology import Cosmology

class InitialConditions:
    def __init__(self, npart, box_size, z_init=50.0, seed=42, cosmo=None):
        self.npart = npart
        self.box_size = box_size
        self.z_init = z_init
        self.seed = seed
        self.cosmo = cosmo if cosmo is not None else Cosmology()
        
        np.random.seed(seed)
        
    def generate_white_noise(self, ng):
        delta_k = np.random.normal(0.0, 1.0, (ng, ng, ng)) + \
                  1j * np.random.normal(0.0, 1.0, (ng, ng, ng))
        
        delta_k[0, 0, 0] = 0.0
        
        for i in range(ng):
            for j in range(ng):
                for k in range(ng):
                    if i + j + k == 0:
                        continue
                    if (ng - i) % ng == i and (ng - j) % ng == j and (ng - k) % ng == k:
                        delta_k[i, j, k] = np.real(delta_k[i, j, k])
                    elif (ng - i) % ng == i and (ng - j) % ng == j:
                        delta_k[i, j, k] = np.conj(delta_k[i, j, ng - k])
                    elif (ng - i) % ng == i and (ng - k) % ng == k:
                        delta_k[i, j, k] = np.conj(delta_k[i, ng - j, k])
                    elif (ng - j) % ng == j and (ng - k) % ng == k:
                        delta_k[i, j, k] = np.conj(delta_k[ng - i, j, k])
        
        return delta_k
    
    def generate_density_field(self, ng):
        kf = 2.0 * np.pi / self.box_size
        kx = np.fft.fftfreq(ng, d=1.0/ng) * kf
        ky = np.fft.fftfreq(ng, d=1.0/ng) * kf
        kz = np.fft.fftfreq(ng, d=1.0/ng) * kf
        
        kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')
        k = np.sqrt(kx**2 + ky**2 + kz**2)
        
        k_flat = k.flatten()
        k_flat[0] = 1e-10
        Pk = self.cosmo.power_spectrum(k_flat, z=self.z_init)
        Pk = Pk.reshape(k.shape)
        Pk[0, 0, 0] = 0.0
        
        delta_k = self.generate_white_noise(ng)
        norm = np.sqrt(Pk * ng**3 / self.box_size**3)
        delta_k = delta_k * norm
        
        delta = np.fft.ifftn(delta_k).real
        
        return delta, delta_k, kx, ky, kz, k
    
    def zeldovich_approximation(self, ng=None):
        if ng is None:
            ng = self.npart
        
        ng_cbrt = int(round(ng**(1/3)))
        if ng_cbrt**3 != ng:
            ng_cbrt = int(round(self.npart**(1/3)))
            ng = ng_cbrt**3
        
        delta, delta_k, kx, ky, kz, k = self.generate_density_field(ng_cbrt)
        
        k2 = k**2
        k2[0, 0, 0] = 1.0
        
        phi_k = -delta_k / k2
        
        vx_k = 1j * kx * phi_k
        vy_k = 1j * ky * phi_k
        vz_k = 1j * kz * phi_k
        
        dx = np.fft.ifftn(vx_k).real
        dy = np.fft.ifftn(vy_k).real
        dz = np.fft.ifftn(vz_k).real
        
        a_init = 1.0 / (1.0 + self.z_init)
        D_init = self.cosmo.D(self.z_init)
        f_init = self.cosmo.f(self.z_init)
        
        cell_size = self.box_size / ng_cbrt
        grid_pos = np.arange(ng_cbrt) * cell_size + 0.5 * cell_size
        
        pos_grid = np.array(np.meshgrid(grid_pos, grid_pos, grid_pos, indexing='ij'))
        pos_grid = pos_grid.reshape(3, -1).T
        
        dx_flat = dx.reshape(-1) * D_init
        dy_flat = dy.reshape(-1) * D_init
        dz_flat = dz.reshape(-1) * D_init
        
        pos = np.copy(pos_grid)
        pos[:, 0] += dx_flat
        pos[:, 1] += dy_flat
        pos[:, 2] += dz_flat
        
        pos = np.mod(pos, self.box_size)
        
        H_init = self.cosmo.H(self.z_init)
        vel_factor = a_init * H_init * f_init * D_init
        
        vx = np.fft.ifftn(vx_k).real * vel_factor
        vy = np.fft.ifftn(vy_k).real * vel_factor
        vz = np.fft.ifftn(vz_k).real * vel_factor
        
        vel = np.column_stack([vx.reshape(-1), vy.reshape(-1), vz.reshape(-1)])
        
        return pos, vel, delta
    
    def generate_particles(self):
        ng_cbrt = int(round(self.npart**(1/3)))
        ng = ng_cbrt**3
        
        if ng != self.npart:
            print(f"Warning: npart={self.npart} is not a perfect cube. Using {ng} particles.")
            self.npart = ng
        
        pos, vel, delta = self.zeldovich_approximation(ng)
        
        return pos, vel, delta

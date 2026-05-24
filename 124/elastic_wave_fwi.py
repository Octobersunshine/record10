import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class ElasticWaveFWI:
    def __init__(self, nx=200, nz=200, dx=10.0, dz=10.0, nt=1000, dt=0.001,
                 free_surface_top=True, anisotropy_type='isotropic',
                 attenuation=False):
        self.nx = nx
        self.nz = nz
        self.dx = dx
        self.dz = dz
        self.nt = nt
        self.dt = dt
        self.free_surface_top = free_surface_top
        self.anisotropy_type = anisotropy_type
        self.attenuation = attenuation
        
        self.vp0 = 3000.0 * np.ones((nz, nx))
        self.vs0 = 1732.0 * np.ones((nz, nx))
        self.rho = 2500.0 * np.ones((nz, nx))
        
        if anisotropy_type in ['VTI', 'TI']:
            self.epsilon = np.zeros((nz, nx))
            self.delta = np.zeros((nz, nx))
            self.gamma = np.zeros((nz, nx))
        
        if attenuation:
            self.Qp = 100.0 * np.ones((nz, nx))
            self.Qs = 50.0 * np.ones((nz, nx))
            self._init_attenuation()
        
        self._update_stiffness_tensor()
        
        self.vx = np.zeros((nz, nx))
        self.vz = np.zeros((nz, nx))
        self.sxx = np.zeros((nz, nx))
        self.szz = np.zeros((nz, nx))
        self.sxz = np.zeros((nz, nx))
        
        self.ux = np.zeros((nz, nx))
        self.uz = np.zeros((nz, nx))
        
        self.src_x = nx // 2
        self.src_z = nz // 4
        
        self.receivers_x = []
        self.receivers_z = []
        self.record_vx = []
        self.record_vz = []
        
        self.t = np.arange(nt) * dt
        self.absorb_width = 25
        
    def _update_stiffness_tensor(self):
        rho = self.rho
        
        if self.anisotropy_type == 'isotropic':
            self.c11 = rho * self.vp0**2
            self.c33 = rho * self.vp0**2
            self.c44 = rho * self.vs0**2
            self.c66 = rho * self.vs0**2
            self.c13 = rho * (self.vp0**2 - 2 * self.vs0**2)
        else:
            vp0_sq = self.vp0**2
            vs0_sq = self.vs0**2
            
            self.c33 = rho * vp0_sq
            self.c44 = rho * vs0_sq
            self.c11 = self.c33 * (1 + 2 * self.epsilon)
            self.c13 = self.c33 * (1 + self.delta) - self.c44
            self.c66 = self.c44 * (1 + 2 * self.gamma)
    
    def _init_attenuation(self):
        self.n_memory = 2
        
        self.mem_sxx = np.zeros((self.n_memory, self.nz, self.nx))
        self.mem_szz = np.zeros((self.n_memory, self.nz, self.nx))
        self.mem_sxz = np.zeros((self.n_memory, self.nz, self.nx))
        
        omega_ref = 2 * np.pi * 15.0
        self.tau_sigma = np.zeros((self.n_memory, self.nz, self.nx))
        self.tau_epsilon = np.zeros((self.n_memory, self.nz, self.nx))
        
        for m in range(self.n_memory):
            tau = 1.0 / (omega_ref * (0.1 ** m))
            self.tau_sigma[m] = tau * np.ones((self.nz, self.nx))
            self.tau_epsilon[m] = tau * np.ones((self.nz, self.nx)) * (1 + 1.0 / (self.Qp + 1e-10))
    
    def set_shale_model(self, v_shale_top=30, v_shale_bottom=80):
        v_back = 3000.0
        v_shale = 3500.0
        
        for z in range(self.nz):
            depth = z * self.dz
            if v_shale_top * 10 <= depth <= v_shale_bottom * 10:
                self.vp0[z, :] = v_shale
                self.vs0[z, :] = v_shale / 1.8
                if self.anisotropy_type in ['VTI', 'TI']:
                    self.epsilon[z, :] = 0.2
                    self.delta[z, :] = 0.1
                    self.gamma[z, :] = 0.15
                if self.attenuation:
                    self.Qp[z, :] = 60.0
                    self.Qs[z, :] = 30.0
            else:
                self.vp0[z, :] = v_back
                self.vs0[z, :] = v_back / 1.732
                if self.anisotropy_type in ['VTI', 'TI']:
                    self.epsilon[z, :] = 0.02
                    self.delta[z, :] = 0.01
                    self.gamma[z, :] = 0.01
        
        self._update_stiffness_tensor()
    
    def set_layered_model(self, layers):
        for z in range(self.nz):
            depth = z * self.dz
            for layer in layers:
                if layer['z_start'] <= depth <= layer['z_end']:
                    self.vp0[z, :] = layer['vp']
                    self.vs0[z, :] = layer.get('vs', layer['vp'] / 1.732)
                    self.rho[z, :] = layer.get('rho', 2500.0)
                    if self.anisotropy_type in ['VTI', 'TI']:
                        self.epsilon[z, :] = layer.get('epsilon', 0.0)
                        self.delta[z, :] = layer.get('delta', 0.0)
                        self.gamma[z, :] = layer.get('gamma', 0.0)
                    if self.attenuation:
                        self.Qp[z, :] = layer.get('Qp', 100.0)
                        self.Qs[z, :] = layer.get('Qs', 50.0)
        self._update_stiffness_tensor()
    
    def add_line_receivers(self, z_depth, x_start=10, x_end=None, dx_receiver=10):
        if x_end is None:
            x_end = self.nx - 10
        
        z_idx = int(z_depth / self.dz)
        x_indices = np.arange(x_start, x_end, int(dx_receiver / self.dx))
        
        self.receivers_x = x_indices
        self.receivers_z = np.full_like(x_indices, z_idx)
        self.record_vx = []
        self.record_vz = []
    
    def ricker_wavelet(self, t, f0=15.0, t0=0.15):
        tau = np.pi * f0 * (t - t0)
        return (1 - 2 * tau**2) * np.exp(-tau**2)
    
    def add_source(self, it, src_type='explosive', amplitude=1e10):
        amp = self.ricker_wavelet(self.t[it]) * amplitude
        if src_type == 'explosive':
            self.sxx[self.src_z, self.src_x] += amp * self.dt
            self.szz[self.src_z, self.src_x] += amp * self.dt
        elif src_type == 'shear_x':
            self.sxz[self.src_z, self.src_x] += amp * self.dt
    
    def update_stress_isotropic(self):
        dx = self.dx
        dz = self.dz
        dt = self.dt
        
        dvx_dx = (self.vx[:, 1:] - self.vx[:, :-1]) / dx
        dvz_dz = (self.vz[1:, :] - self.vz[:-1, :]) / dz
        
        c11 = self.c11
        c33 = self.c33
        c13 = self.c13
        c44 = self.c44
        
        self.sxx[:, 1:-1] += dt * c11[1:-1, 1:-1] * dvx_dx[:, :-1] + dt * c13[1:-1, 1:-1] * dvz_dz[:-1, 1:-1]
        self.szz[1:-1, :] += dt * c13[1:-1, 1:-1] * dvx_dx[1:-1, :] + dt * c33[1:-1, 1:-1] * dvz_dz[:, :-1]
        
        dvx_dz = (self.vx[1:, :] - self.vx[:-1, :]) / dz
        dvz_dx = (self.vz[:, 1:] - self.vz[:, :-1]) / dx
        
        self.sxz[1:-1, 1:-1] += dt * c44[1:-1, 1:-1] * (dvx_dz[:-1, 1:-1] + dvz_dx[1:-1, :-1])
    
    def update_stress_anisotropic(self):
        dx = self.dx
        dz = self.dz
        dt = self.dt
        
        dvx_dx = (self.vx[:, 1:] - self.vx[:, :-1]) / dx
        dvz_dz = (self.vz[1:, :] - self.vz[:-1, :]) / dz
        
        self.sxx[:, 1:-1] += dt * self.c11[1:-1, 1:-1] * dvx_dx[:, :-1] + dt * self.c13[1:-1, 1:-1] * dvz_dz[:-1, 1:-1]
        self.szz[1:-1, :] += dt * self.c13[1:-1, 1:-1] * dvx_dx[1:-1, :] + dt * self.c33[1:-1, 1:-1] * dvz_dz[:, :-1]
        
        dvx_dz = (self.vx[1:, :] - self.vx[:-1, :]) / dz
        dvz_dx = (self.vz[:, 1:] - self.vz[:, :-1]) / dx
        
        self.sxz[1:-1, 1:-1] += dt * self.c44[1:-1, 1:-1] * dvx_dz[:-1, 1:-1] + dt * self.c66[1:-1, 1:-1] * dvz_dx[1:-1, :-1]
    
    def update_stress(self):
        if self.anisotropy_type == 'isotropic':
            self.update_stress_isotropic()
        else:
            self.update_stress_anisotropic()
    
    def apply_attenuation(self):
        if not self.attenuation:
            return
        
        dt = self.dt
        
        for m in range(self.n_memory):
            tau_s = self.tau_sigma[m]
            tau_e = self.tau_epsilon[m]
            
            alpha = np.exp(-dt / tau_s)
            beta = (tau_e / tau_s - 1) * (1 - alpha)
            
            self.mem_sxx[m] = alpha * self.mem_sxx[m] + beta * self.sxx
            self.mem_szz[m] = alpha * self.mem_szz[m] + beta * self.szz
            self.mem_sxz[m] = alpha * self.mem_sxz[m] + beta * self.sxz
            
            self.sxx -= dt / tau_e * self.mem_sxx[m]
            self.szz -= dt / tau_e * self.mem_szz[m]
            self.sxz -= dt / tau_e * self.mem_sxz[m]
    
    def update_velocity(self):
        dx = self.dx
        dz = self.dz
        dt = self.dt
        rho = self.rho
        
        dsxx_dx = (self.sxx[:, 1:] - self.sxx[:, :-1]) / dx
        dsxz_dz = (self.sxz[1:, :] - self.sxz[:-1, :]) / dz
        
        self.vx[:, 1:-1] += dt / rho[1:-1, 1:-1] * (dsxx_dx[:, :-1] + dsxz_dz[:, 1:-1])
        
        dsxz_dx = (self.sxz[:, 1:] - self.sxz[:, :-1]) / dx
        dszz_dz = (self.szz[1:, :] - self.szz[:-1, :]) / dz
        
        self.vz[1:-1, :] += dt / rho[1:-1, 1:-1] * (dsxz_dx[1:-1, :] + dszz_dz[:-1, :])
        
        self.ux += self.vx * dt
        self.uz += self.vz * dt
    
    def record_wavefield(self):
        if len(self.receivers_x) > 0:
            vx_rec = self.vx[self.receivers_z, self.receivers_x]
            vz_rec = self.vz[self.receivers_z, self.receivers_x]
            self.record_vx.append(vx_rec.copy())
            self.record_vz.append(vz_rec.copy())
    
    def apply_free_surface_mirror(self):
        z_surface = 0
        
        self.szz[z_surface, :] = 0.0
        self.sxz[z_surface, :] = 0.0
        
        self.vz[z_surface, :] = -self.vz[z_surface + 1, :]
        self.vx[z_surface, :] = self.vx[z_surface + 1, :]
        self.sxx[z_surface, :] = self.sxx[z_surface + 1, :]
    
    def apply_absorbing_boundaries(self):
        n = self.absorb_width
        coeff = 0.2
        
        for i in range(n):
            damp = 1.0 - coeff * ((n - i) / n)**2
            
            self.vx[-i-1, :] *= damp
            self.vx[:, i] *= damp
            self.vx[:, -i-1] *= damp
            
            self.vz[-i-1, :] *= damp
            self.vz[:, i] *= damp
            self.vz[:, -i-1] *= damp
            
            self.sxx[-i-1, :] *= damp
            self.sxx[:, i] *= damp
            self.sxx[:, -i-1] *= damp
            
            self.szz[-i-1, :] *= damp
            self.szz[:, i] *= damp
            self.szz[:, -i-1] *= damp
            
            self.sxz[-i-1, :] *= damp
            self.sxz[:, i] *= damp
            self.sxz[:, -i-1] *= damp
            
            if not self.free_surface_top:
                self.vx[i, :] *= damp
                self.vz[i, :] *= damp
                self.sxx[i, :] *= damp
                self.szz[i, :] *= damp
                self.sxz[i, :] *= damp
    
    def apply_boundary_conditions(self):
        if self.free_surface_top:
            self.apply_free_surface_mirror()
        self.apply_absorbing_boundaries()
    
    def forward(self, src_type='explosive', plot_interval=20, save_wavefield=False):
        frames = []
        wavefield_data = {'ux': [], 'uz': [], 'vx': [], 'vz': []}
        
        for it in range(self.nt):
            self.add_source(it, src_type)
            self.update_stress()
            self.apply_attenuation()
            self.update_velocity()
            self.apply_boundary_conditions()
            self.record_wavefield()
            
            if it % plot_interval == 0:
                disp_mag = np.sqrt(self.ux**2 + self.uz**2)
                frames.append(disp_mag.copy())
                if save_wavefield:
                    wavefield_data['ux'].append(self.ux.copy())
                    wavefield_data['uz'].append(self.uz.copy())
                    wavefield_data['vx'].append(self.vx.copy())
                    wavefield_data['vz'].append(self.vz.copy())
                if it % 100 == 0:
                    print(f"Time step {it}/{self.nt}, max displacement: {np.max(disp_mag):.6e}")
        
        self.record_vx = np.array(self.record_vx)
        self.record_vz = np.array(self.record_vz)
        
        return frames, wavefield_data
    
    def get_seismic_shot_record(self):
        return self.record_vx, self.record_vz
    
    def misfit(self, observed_vx, observed_vz, norm='L2'):
        if norm == 'L2':
            misfit_vx = 0.5 * np.sum((self.record_vx - observed_vx)**2)
            misfit_vz = 0.5 * np.sum((self.record_vz - observed_vz)**2)
        elif norm == 'L1':
            misfit_vx = np.sum(np.abs(self.record_vx - observed_vx))
            misfit_vz = np.sum(np.abs(self.record_vz - observed_vz))
        else:
            raise ValueError(f"Unknown norm type: {norm}")
        
        return misfit_vx + misfit_vz


def plot_model(sim, save_path=None):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    extent = [0, sim.nx*sim.dx, sim.nz*sim.dz, 0]
    
    im1 = axes[0, 0].imshow(sim.vp0, cmap='jet', aspect='auto', extent=extent)
    axes[0, 0].set_title('Vp (m/s)')
    axes[0, 0].set_xlabel('X (m)')
    axes[0, 0].set_ylabel('Depth Z (m)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(sim.vs0, cmap='jet', aspect='auto', extent=extent)
    axes[0, 1].set_title('Vs (m/s)')
    axes[0, 1].set_xlabel('X (m)')
    axes[0, 1].set_ylabel('Depth Z (m)')
    plt.colorbar(im2, ax=axes[0, 1])
    
    im3 = axes[0, 2].imshow(sim.rho, cmap='jet', aspect='auto', extent=extent)
    axes[0, 2].set_title('Density (kg/m³)')
    axes[0, 2].set_xlabel('X (m)')
    axes[0, 2].set_ylabel('Depth Z (m)')
    plt.colorbar(im3, ax=axes[0, 2])
    
    if sim.anisotropy_type in ['VTI', 'TI'] and not sim.attenuation:
        im4 = axes[1, 0].imshow(sim.epsilon, cmap='hot', aspect='auto', extent=extent)
        axes[1, 0].set_title('Anisotropy ε')
        axes[1, 0].set_xlabel('X (m)')
        axes[1, 0].set_ylabel('Depth Z (m)')
        plt.colorbar(im4, ax=axes[1, 0])
        
        im5 = axes[1, 1].imshow(sim.delta, cmap='hot', aspect='auto', extent=extent)
        axes[1, 1].set_title('Anisotropy δ')
        axes[1, 1].set_xlabel('X (m)')
        axes[1, 1].set_ylabel('Depth Z (m)')
        plt.colorbar(im5, ax=axes[1, 1])
        
        im6 = axes[1, 2].imshow(sim.gamma, cmap='hot', aspect='auto', extent=extent)
        axes[1, 2].set_title('Anisotropy γ')
        axes[1, 2].set_xlabel('X (m)')
        axes[1, 2].set_ylabel('Depth Z (m)')
        plt.colorbar(im6, ax=axes[1, 2])
    
    if sim.attenuation:
        axes[1, 0].clear()
        im4 = axes[1, 0].imshow(sim.Qp, cmap='viridis', aspect='auto', extent=extent)
        axes[1, 0].set_title('Qp')
        axes[1, 0].set_xlabel('X (m)')
        axes[1, 0].set_ylabel('Depth Z (m)')
        plt.colorbar(im4, ax=axes[1, 0])
        
        axes[1, 1].clear()
        im5 = axes[1, 1].imshow(sim.Qs, cmap='viridis', aspect='auto', extent=extent)
        axes[1, 1].set_title('Qs')
        axes[1, 1].set_xlabel('X (m)')
        axes[1, 1].set_ylabel('Depth Z (m)')
        plt.colorbar(im5, ax=axes[1, 1])
        
        axes[1, 2].clear()
        if hasattr(sim, 'epsilon'):
            im6 = axes[1, 2].imshow(sim.epsilon, cmap='hot', aspect='auto', extent=extent)
            axes[1, 2].set_title('Anisotropy ε')
            axes[1, 2].set_xlabel('X (m)')
            axes[1, 2].set_ylabel('Depth Z (m)')
            plt.colorbar(im6, ax=axes[1, 2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_shot_gather(sim, component='vz', save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    if component == 'vx':
        data = sim.record_vx
        title = 'Horizontal Component (Vx)'
    else:
        data = sim.record_vz
        title = 'Vertical Component (Vz)'
    
    vmax = np.max(np.abs(data)) * 0.1
    extent = [0, len(sim.receivers_x) * sim.dx, sim.nt * sim.dt, 0]
    
    im1 = axes[0].imshow(data, cmap='seismic', aspect='auto', 
                         vmin=-vmax, vmax=vmax, extent=extent)
    axes[0].set_title(f'Shot Gather - {title}')
    axes[0].set_xlabel('Receiver Offset (m)')
    axes[0].set_ylabel('Time (s)')
    plt.colorbar(im1, ax=axes[0])
    
    trace_idx = len(sim.receivers_x) // 2
    axes[1].plot(sim.t, data[:, trace_idx])
    axes[1].set_title(f'Seismic Trace at Offset {trace_idx * sim.dx:.0f}m')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_wavefield_snapshot(sim, it, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    disp_mag = np.sqrt(sim.ux**2 + sim.uz**2)
    extent = [0, sim.nx*sim.dx, sim.nz*sim.dz, 0]
    
    vmax_ux = np.max(np.abs(sim.ux)) * 0.1
    vmax_uz = np.max(np.abs(sim.uz)) * 0.1
    
    im1 = axes[0, 0].imshow(sim.ux, cmap='seismic', aspect='auto', 
                            vmin=-vmax_ux, vmax=vmax_ux, extent=extent)
    axes[0, 0].set_title('Horizontal Displacement Ux')
    axes[0, 0].set_xlabel('X (m)')
    axes[0, 0].set_ylabel('Depth Z (m)')
    if sim.free_surface_top:
        axes[0, 0].axhline(y=0, color='k', linestyle='--', linewidth=2, label='Free Surface')
        axes[0, 0].legend()
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(sim.uz, cmap='seismic', aspect='auto',
                            vmin=-vmax_uz, vmax=vmax_uz, extent=extent)
    axes[0, 1].set_title('Vertical Displacement Uz')
    axes[0, 1].set_xlabel('X (m)')
    axes[0, 1].set_ylabel('Depth Z (m)')
    if sim.free_surface_top:
        axes[0, 1].axhline(y=0, color='k', linestyle='--', linewidth=2, label='Free Surface')
        axes[0, 1].legend()
    plt.colorbar(im2, ax=axes[0, 1])
    
    im3 = axes[1, 0].imshow(disp_mag, cmap='viridis', aspect='auto', extent=extent)
    axes[1, 0].set_title('Displacement Magnitude')
    axes[1, 0].set_xlabel('X (m)')
    axes[1, 0].set_ylabel('Depth Z (m)')
    if sim.free_surface_top:
        axes[1, 0].axhline(y=0, color='k', linestyle='--', linewidth=2, label='Free Surface')
        axes[1, 0].legend()
    plt.colorbar(im3, ax=axes[1, 0])
    
    axes[1, 1].plot(sim.t, sim.ricker_wavelet(sim.t))
    axes[1, 1].set_title('Source Wavelet (Ricker)')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Amplitude')
    axes[1, 1].axvline(x=sim.t[min(it, sim.nt-1)], color='r', linestyle='--', label='Current Time')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def test_anisotropic_attenuation_model():
    print("="*80)
    print("Elastic Wave Simulation for FWI: Anisotropy + Attenuation")
    print("="*80)
    
    sim = ElasticWaveFWI(
        nx=256,
        nz=128,
        dx=10.0,
        dz=10.0,
        nt=1500,
        dt=0.001,
        free_surface_top=True,
        anisotropy_type='VTI',
        attenuation=True
    )
    
    layers = [
        {'z_start': 0, 'z_end': 300, 'vp': 2500, 'vs': 1443, 'rho': 2200, 
         'epsilon': 0.05, 'delta': 0.03, 'gamma': 0.04, 'Qp': 80, 'Qs': 40},
        {'z_start': 300, 'z_end': 600, 'vp': 3500, 'vs': 2020, 'rho': 2500, 
         'epsilon': 0.20, 'delta': 0.10, 'gamma': 0.15, 'Qp': 60, 'Qs': 30},
        {'z_start': 600, 'z_end': 10000, 'vp': 4000, 'vs': 2309, 'rho': 2700, 
         'epsilon': 0.02, 'delta': 0.01, 'gamma': 0.01, 'Qp': 100, 'Qs': 50},
    ]
    sim.set_layered_model(layers)
    
    sim.src_z = 15
    sim.src_x = sim.nx // 4
    
    sim.add_line_receivers(z_depth=20, x_start=5, dx_receiver=20)
    
    print(f"\nModel Configuration:")
    print(f"  Grid: {sim.nx} x {sim.nz}")
    print(f"  Grid spacing: dx={sim.dx}m, dz={sim.dz}m")
    print(f"  Time steps: {sim.nt}, dt={sim.dt}s")
    print(f"  Anisotropy type: {sim.anisotropy_type}")
    print(f"  Attenuation: {sim.attenuation}")
    print(f"  Source location: ({sim.src_x}, {sim.src_z})")
    print(f"  Number of receivers: {len(sim.receivers_x)}")
    print()
    
    print("Plotting velocity model...")
    plot_model(sim, save_path='velocity_model.png')
    
    print("\nRunning forward simulation...")
    frames, _ = sim.forward(src_type='explosive', plot_interval=10)
    
    print("\nPlotting wavefield snapshot...")
    plot_wavefield_snapshot(sim, sim.nt-1, save_path='wavefield_snapshot.png')
    
    print("Plotting shot gather...")
    plot_shot_gather(sim, component='vz', save_path='shot_gather_vz.png')
    plot_shot_gather(sim, component='vx', save_path='shot_gather_vx.png')
    
    print("\nVerification:")
    szz_surface = np.abs(sim.szz[0, :])
    sxz_surface = np.abs(sim.sxz[0, :])
    print(f"  Max |σ_zz| at free surface: {np.max(szz_surface):.6e}")
    print(f"  Max |σ_xz| at free surface: {np.max(sxz_surface):.6e}")
    print(f"  Max recorded Vz amplitude: {np.max(np.abs(sim.record_vz)):.6e}")
    
    print("\n" + "="*80)
    print("Simulation complete! Output files:")
    print("  - velocity_model.png: Velocity and anisotropy model")
    print("  - wavefield_snapshot.png: Final wavefield snapshot")
    print("  - shot_gather_vz.png: Vertical component shot gather")
    print("  - shot_gather_vx.png: Horizontal component shot gather")
    print("="*80)


def main():
    test_anisotropic_attenuation_model()


if __name__ == "__main__":
    main()

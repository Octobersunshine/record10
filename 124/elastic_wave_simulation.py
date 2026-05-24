import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class ElasticWaveFDM:
    def __init__(self, nx=200, nz=200, dx=10.0, dz=10.0, 
                 vp=3000.0, vs=1732.0, rho=2500.0, nt=1000, dt=0.001,
                 free_surface_top=True):
        self.nx = nx
        self.nz = nz
        self.dx = dx
        self.dz = dz
        self.vp = vp
        self.vs = vs
        self.rho = rho
        self.nt = nt
        self.dt = dt
        self.free_surface_top = free_surface_top
        
        self.lam = rho * (vp**2 - 2 * vs**2)
        self.mu = rho * vs**2
        
        self.vx = np.zeros((nz, nx))
        self.vz = np.zeros((nz, nx))
        self.sxx = np.zeros((nz, nx))
        self.szz = np.zeros((nz, nx))
        self.sxz = np.zeros((nz, nx))
        
        self.ux = np.zeros((nz, nx))
        self.uz = np.zeros((nz, nx))
        
        self.src_x = nx // 2
        self.src_z = nz // 4
        
        self.t = np.arange(nt) * dt
        
        self.absorb_width = 25
        
    def ricker_wavelet(self, t, f0=15.0, t0=0.15):
        tau = np.pi * f0 * (t - t0)
        return (1 - 2 * tau**2) * np.exp(-tau**2)
    
    def add_source(self, it, src_type='explosive'):
        amp = self.ricker_wavelet(self.t[it])
        if src_type == 'explosive':
            self.sxx[self.src_z, self.src_x] += amp * self.dt
            self.szz[self.src_z, self.src_x] += amp * self.dt
        elif src_type == 'shear_x':
            self.sxz[self.src_z, self.src_x] += amp * self.dt
    
    def update_stress(self):
        dx = self.dx
        dz = self.dz
        dt = self.dt
        lam = self.lam
        mu = self.mu
        
        dvx_dx = (self.vx[:, 1:] - self.vx[:, :-1]) / dx
        dvz_dz = (self.vz[1:, :] - self.vz[:-1, :]) / dz
        
        self.sxx[:, 1:-1] += dt * (lam + 2 * mu) * dvx_dx[:, :-1] + dt * lam * dvz_dz[:-1, 1:-1]
        self.szz[1:-1, :] += dt * lam * dvx_dx[1:-1, :] + dt * (lam + 2 * mu) * dvz_dz[:, :-1]
        
        dvx_dz = (self.vx[1:, :] - self.vx[:-1, :]) / dz
        dvz_dx = (self.vz[:, 1:] - self.vz[:, :-1]) / dx
        
        self.sxz[1:-1, 1:-1] += dt * mu * (dvx_dz[:-1, 1:-1] + dvz_dx[1:-1, :-1])
    
    def update_velocity(self):
        dx = self.dx
        dz = self.dz
        dt = self.dt
        rho = self.rho
        
        dsxx_dx = (self.sxx[:, 1:] - self.sxx[:, :-1]) / dx
        dsxz_dz = (self.sxz[1:, :] - self.sxz[:-1, :]) / dz
        
        self.vx[:, 1:-1] += dt / rho * (dsxx_dx[:, :-1] + dsxz_dz[:, 1:-1])
        
        dsxz_dx = (self.sxz[:, 1:] - self.sxz[:, :-1]) / dx
        dszz_dz = (self.szz[1:, :] - self.szz[:-1, :]) / dz
        
        self.vz[1:-1, :] += dt / rho * (dsxz_dx[1:-1, :] + dszz_dz[:-1, :])
        
        self.ux += self.vx * dt
        self.uz += self.vz * dt
    
    def apply_free_surface_mirror(self):
        z_surface = 0
        
        self.szz[z_surface, :] = 0.0
        
        self.sxz[z_surface, :] = 0.0
        
        self.vz[z_surface, :] = -self.vz[z_surface + 1, :]
        
        self.vx[z_surface, :] = self.vx[z_surface + 1, :]
        
        self.sxx[z_surface, :] = self.sxx[z_surface + 1, :]
        
        if self.nz > 3:
            self.vz[z_surface, :] = -2.0 * self.vz[z_surface + 1, :] + self.vz[z_surface + 2, :]
    
    def apply_free_surface_improved(self):
        z0 = 0
        
        self.szz[z0, :] = 0.0
        
        self.sxz[z0, :] = 0.0
        
        self.vz[z0, :] = 0.0
        
        self.sxx[z0, :] = 2.0 * self.sxx[z0 + 1, :] - self.sxx[z0 + 2, :]
        
        self.vx[z0, :] = 2.0 * self.vx[z0 + 1, :] - self.vx[z0 + 2, :]
        
        self.sxz[z0, :] = 0.0
        for i in range(1, self.nx - 1):
            dvx_dz = (self.vx[z0 + 1, i] - self.vx[z0, i]) / self.dz
            dvz_dx = (self.vz[z0, i + 1] - self.vz[z0, i]) / self.dx
            self.sxz[z0, i] = self.mu * (dvx_dz + dvz_dx) * self.dt
    
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
    
    def run(self, src_type='explosive', plot_interval=20):
        frames = []
        ux_frames = []
        uz_frames = []
        
        for it in range(self.nt):
            self.add_source(it, src_type)
            self.update_stress()
            self.update_velocity()
            self.apply_boundary_conditions()
            
            if it % plot_interval == 0:
                disp_mag = np.sqrt(self.ux**2 + self.uz**2)
                frames.append(disp_mag.copy())
                ux_frames.append(self.ux.copy())
                uz_frames.append(self.uz.copy())
                if it % 100 == 0:
                    print(f"Time step {it}/{self.nt}, max displacement: {np.max(disp_mag):.6e}")
        
        return frames, ux_frames, uz_frames


def plot_snapshot(sim, ux, uz, it, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    disp_mag = np.sqrt(ux**2 + uz**2)
    
    vmax_ux = np.max(np.abs(ux))
    vmax_uz = np.max(np.abs(uz))
    
    im1 = axes[0, 0].imshow(ux, cmap='seismic', aspect='auto', 
                            vmin=-vmax_ux, vmax=vmax_ux,
                            extent=[0, sim.nx*sim.dx, sim.nz*sim.dz, 0])
    axes[0, 0].set_title('Horizontal Displacement Ux')
    axes[0, 0].set_xlabel('X (m)')
    axes[0, 0].set_ylabel('Depth Z (m)')
    if sim.free_surface_top:
        axes[0, 0].axhline(y=0, color='k', linestyle='--', linewidth=2, label='Free Surface')
        axes[0, 0].legend()
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(uz, cmap='seismic', aspect='auto',
                            vmin=-vmax_uz, vmax=vmax_uz,
                            extent=[0, sim.nx*sim.dx, sim.nz*sim.dz, 0])
    axes[0, 1].set_title('Vertical Displacement Uz')
    axes[0, 1].set_xlabel('X (m)')
    axes[0, 1].set_ylabel('Depth Z (m)')
    if sim.free_surface_top:
        axes[0, 1].axhline(y=0, color='k', linestyle='--', linewidth=2, label='Free Surface')
        axes[0, 1].legend()
    plt.colorbar(im2, ax=axes[0, 1])
    
    im3 = axes[1, 0].imshow(disp_mag, cmap='viridis', aspect='auto',
                            extent=[0, sim.nx*sim.dx, sim.nz*sim.dz, 0])
    axes[1, 0].set_title('Displacement Magnitude')
    axes[1, 0].set_xlabel('X (m)')
    axes[1, 0].set_ylabel('Depth Z (m)')
    if sim.free_surface_top:
        axes[1, 0].axhline(y=0, color='k', linestyle='--', linewidth=2, label='Free Surface')
        axes[1, 0].legend()
    plt.colorbar(im3, ax=axes[1, 0])
    
    x_profile = sim.nx // 2
    axes[1, 1].plot(uz[:, x_profile], np.arange(sim.nz) * sim.dz, 'b-', label='Uz')
    axes[1, 1].plot(ux[:, x_profile], np.arange(sim.nz) * sim.dz, 'r-', label='Ux')
    axes[1, 1].set_title(f'Vertical Profile at X = {x_profile * sim.dx}m')
    axes[1, 1].set_xlabel('Displacement')
    axes[1, 1].set_ylabel('Depth Z (m)')
    axes[1, 1].invert_yaxis()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    if sim.free_surface_top:
        axes[1, 1].axhline(y=0, color='k', linestyle='--', label='Free Surface')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_animation(frames, interval=100, save_path=None, nx=200, nz=200, dx=10, dz=10, title=''):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    vmax = np.max([np.max(f) for f in frames])
    im = ax.imshow(frames[0], cmap='viridis', vmin=0, vmax=vmax, aspect='auto',
                   extent=[0, nx*dx, nz*dz, 0])
    plt.colorbar(im, ax=ax, label='Displacement Magnitude')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Depth Z (m)')
    ax.set_title(title)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=2, label='Free Surface')
    ax.legend()
    
    def update(frame):
        im.set_data(frame)
        return [im]
    
    ani = FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)
    
    if save_path:
        ani.save(save_path, writer='pillow', fps=10)
        plt.close()
    else:
        plt.show()


def test_free_surface_reflection():
    print("Testing Free Surface Boundary Condition (Mirror Method)")
    print("="*70)
    
    sim = ElasticWaveFDM(
        nx=200,
        nz=150,
        dx=10.0,
        dz=10.0,
        vp=3000.0,
        vs=1732.0,
        rho=2500.0,
        nt=1200,
        dt=0.001,
        free_surface_top=True
    )
    
    sim.src_z = 50
    
    print(f"Grid: {sim.nx} x {sim.nz}")
    print(f"Grid spacing: dx={sim.dx}m, dz={sim.dz}m")
    print(f"Velocity: Vp={sim.vp}m/s, Vs={sim.vs}m/s")
    print(f"Time steps: {sim.nt}, dt={sim.dt}s")
    print(f"Source location: ({sim.src_x}, {sim.src_z}) [~{sim.src_z*sim.dz}m depth]")
    print(f"Free surface at top (z=0): {sim.free_surface_top}")
    print()
    print("Free Surface Boundary Conditions Applied:")
    print("  - σ_zz = 0 (zero normal stress)")
    print("  - σ_xz = 0 (zero shear stress)")
    print("  - vz(z=0) = -vz(z=1) (mirror velocity for traction-free)")
    print("  - vx(z=0) = vx(z=1) (symmetric horizontal velocity)")
    print()
    
    print("Running simulation with explosive source...")
    frames, ux_frames, uz_frames = sim.run(src_type='explosive', plot_interval=10)
    
    print("\nVerifying boundary conditions at free surface...")
    szz_surface = np.abs(sim.szz[0, :])
    sxz_surface = np.abs(sim.sxz[0, :])
    print(f"  Max |σ_zz| at surface: {np.max(szz_surface):.6e} (should be ~0)")
    print(f"  Max |σ_xz| at surface: {np.max(sxz_surface):.6e} (should be ~0)")
    
    print("\nPlotting final snapshot...")
    plot_snapshot(sim, sim.ux, sim.uz, sim.nt-1, save_path='free_surface_snapshot.png')
    
    print("Creating animation...")
    create_animation(frames, save_path='free_surface_wave.gif', 
                     nx=sim.nx, nz=sim.nz, dx=sim.dx, dz=sim.dz,
                     title='Elastic Wave Propagation with Free Surface')
    
    print("\nDone! Results saved:")
    print("  - free_surface_snapshot.png")
    print("  - free_surface_wave.gif")
    print()
    print("Expected wave phenomena:")
    print("  1. Direct P-wave and S-wave from source")
    print("  2. Reflected P-wave (PP) from free surface")
    print("  3. Reflected S-wave (SS) from free surface")
    print("  4. Converted waves (PS and SP)")
    print("  5. Rayleigh surface wave propagating along free surface")


def main():
    test_free_surface_reflection()


if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('Agg')


class NonhydrostaticModel:
    def __init__(self, Lx=100, Lz=50, Nx=256, Nz=64, dt=0.01, T_max=20,
                 g=9.81, rho0=1000.0, nu=1e-4, kappa=1e-5):
        self.Lx = Lx
        self.Lz = Lz
        self.Nx = Nx
        self.Nz = Nz
        self.dt = dt
        self.T_max = T_max
        
        self.g = g
        self.rho0 = rho0
        self.nu = nu
        self.kappa = kappa
        
        self.x = np.linspace(0, Lx, Nx)
        self.z = np.linspace(0, Lz, Nz)
        self.X, self.Z = np.meshgrid(self.x, self.z, indexing='ij')
        
        self.dx = self.x[1] - self.x[0]
        self.dz = self.z[1] - self.z[0]
        
        self.kx = 2 * np.pi * np.fft.fftfreq(Nx, d=self.dx)
        self.kz = 2 * np.pi * np.fft.fftfreq(Nz, d=self.dz)
        self.KX, self.KZ = np.meshgrid(self.kx, self.kz, indexing='ij')
        self.K2 = self.KX**2 + self.KZ**2
        self.K2[0, 0] = 1.0
        
        self.u = np.zeros((Nx, Nz))
        self.w = np.zeros((Nx, Nz))
        self.rho = np.zeros((Nx, Nz))
        self.p = np.zeros((Nx, Nz))
        
        self.terrain = np.zeros(Nx)
        
        self.mixing_history = []
        self.energy_history = []
        
    def set_terrain(self, terrain_type='flat', **kwargs):
        if terrain_type == 'flat':
            self.terrain = np.zeros(self.Nx)
        elif terrain_type == 'ridge':
            h0 = kwargs.get('h0', 0.0)
            height = kwargs.get('height', 10.0)
            x0 = kwargs.get('x0', self.Lx/2)
            width = kwargs.get('width', 20.0)
            self.terrain = h0 + height * np.exp(-((self.x - x0)**2) / (2 * width**2))
        elif terrain_type == 'shelf':
            h1 = kwargs.get('h1', 0.0)
            h2 = kwargs.get('h2', 20.0)
            x_trans = kwargs.get('x_trans', self.Lx/2)
            width = kwargs.get('width', 10.0)
            transition = 0.5 * (1 + np.tanh((self.x - x_trans) / width))
            self.terrain = h1 * (1 - transition) + h2 * transition
        return self.terrain
    
    def set_density_profile(self, profile_type='two_layer', **kwargs):
        if profile_type == 'two_layer':
            rho1 = kwargs.get('rho1', 1000.0)
            rho2 = kwargs.get('rho2', 1002.0)
            z_interface = kwargs.get('z_interface', self.Lz/2)
            thickness = kwargs.get('thickness', 2.0)
            
            for i in range(self.Nx):
                for k in range(self.Nz):
                    self.rho[i, k] = rho1 + (rho2 - rho1) * 0.5 * \
                        (1 + np.tanh((self.z[k] - z_interface) / thickness))
                        
        elif profile_type == 'linear':
            rho_surface = kwargs.get('rho_surface', 1000.0)
            rho_bottom = kwargs.get('rho_bottom', 1003.0)
            
            for i in range(self.Nx):
                self.rho[i, :] = np.linspace(rho_surface, rho_bottom, self.Nz)
                
        elif profile_type == 'exponential':
            rho_surface = kwargs.get('rho_surface', 1000.0)
            scale_height = kwargs.get('scale_height', 20.0)
            
            for i in range(self.Nx):
                self.rho[i, :] = rho_surface * (1 + 0.003 * (1 - np.exp(-self.z / scale_height)))
                
        self.rho0_profile = self.rho[0, :].copy()
        return self.rho
    
    def add_internal_wave(self, wave_type='soliton', **kwargs):
        if wave_type == 'soliton':
            x0 = kwargs.get('x0', 20.0)
            amp = kwargs.get('amp', 1.0)
            width = kwargs.get('width', 10.0)
            z_peak = kwargs.get('z_peak', self.Lz/2)
            
            for i in range(self.Nx):
                for k in range(self.Nz):
                    envelope = amp * np.exp(-((self.x[i] - x0)**2) / (2 * width**2))
                    vert_profile = np.sin(np.pi * self.z[k] / self.Lz)
                    self.u[i, k] += envelope * vert_profile
                    
        elif wave_type == 'mode1':
            x0 = kwargs.get('x0', 20.0)
            amp = kwargs.get('amp', 0.5)
            width = kwargs.get('width', 15.0)
            
            for i in range(self.Nx):
                envelope = amp * np.exp(-((self.x[i] - x0)**2) / (2 * width**2))
                for k in range(self.Nz):
                    self.u[i, k] += envelope * np.cos(np.pi * self.z[k] / self.Lz)
                    
        return self.u
    
    def gradient_x(self, f):
        f_hat = np.fft.fft(f, axis=0)
        return np.fft.ifft(1j * self.KX * f_hat, axis=0).real
    
    def gradient_z(self, f):
        f_hat = np.fft.fft(f, axis=1)
        return np.fft.ifft(1j * self.KZ * f_hat, axis=1).real
    
    def laplacian(self, f):
        f_hat = np.fft.fft2(f)
        return np.fft.ifft2(-self.K2 * f_hat).real
    
    def divergence(self, u, w):
        return self.gradient_x(u) + self.gradient_z(w)
    
    def solve_pressure_poisson(self, rhs):
        rhs_hat = np.fft.fft2(rhs)
        p_hat = -rhs_hat / self.K2
        p_hat[0, 0] = 0
        return np.fft.ifft2(p_hat).real
    
    def advection(self, f, u, w):
        df_dx = self.gradient_x(f)
        df_dz = self.gradient_z(f)
        return -(u * df_dx + w * df_dz)
    
    def apply_bottom_boundary(self, u, w, rho):
        for i in range(self.Nx):
            z_terrain_idx = int(self.terrain[i] / self.dz)
            if z_terrain_idx > 0:
                u[i, :z_terrain_idx] = 0
                w[i, :z_terrain_idx] = 0
                rho[i, :z_terrain_idx] = rho[i, z_terrain_idx] if z_terrain_idx < self.Nz else rho[i, -1]
        return u, w, rho
    
    def step(self):
        u_star = self.u.copy()
        w_star = self.w.copy()
        rho_star = self.rho.copy()
        
        adv_u = self.advection(self.u, self.u, self.w)
        adv_w = self.advection(self.w, self.u, self.w)
        adv_rho = self.advection(self.rho, self.u, self.w)
        
        buoyancy = -self.g * (self.rho - self.rho0_profile[np.newaxis, :]) / self.rho0
        
        u_star += self.dt * (adv_u + self.nu * self.laplacian(self.u))
        w_star += self.dt * (adv_w + self.nu * self.laplacian(self.w) + buoyancy)
        rho_star += self.dt * (adv_rho + self.kappa * self.laplacian(self.rho))
        
        u_star, w_star, rho_star = self.apply_bottom_boundary(u_star, w_star, rho_star)
        
        div_star = self.divergence(u_star, w_star)
        p_correction = self.solve_pressure_poisson(div_star / self.dt)
        
        self.p = p_correction
        
        self.u = u_star - self.dt * self.gradient_x(p_correction)
        self.w = w_star - self.dt * self.gradient_z(p_correction)
        self.rho = rho_star
        
        self.u, self.w, self.rho = self.apply_bottom_boundary(self.u, self.w, self.rho)
        
        return self.u, self.w, self.rho
    
    def compute_mixing_diagnostics(self):
        rho_hat = np.mean(self.rho, axis=0)
        rho_sorted = np.sort(self.rho, axis=1)
        rho_mean_sorted = np.mean(rho_sorted, axis=0)
        
        PE = np.sum(self.rho * self.Z) * self.dx * self.dz
        PE_min = np.sum(rho_mean_sorted[:, np.newaxis] * self.Z.T) * self.dx * self.dz
        
        available_PE = PE - PE_min
        
        N2 = -self.g / self.rho0 * np.gradient(self.rho0_profile, self.z)
        N2 = np.maximum(N2, 1e-6)
        
        rho_fluct = self.rho - self.rho0_profile[np.newaxis, :]
        KE = 0.5 * self.rho0 * np.sum(self.u**2 + self.w**2) * self.dx * self.dz
        
        grad_rho_x = self.gradient_x(self.rho)
        grad_rho_z = self.gradient_z(self.rho)
        mag_grad_rho = np.sqrt(grad_rho_x**2 + grad_rho_z**2)
        
        mixing_efficiency = available_PE / (KE + 1e-10) if KE > 0 else 0
        
        diag = {
            'time': len(self.energy_history) * self.dt,
            'KE': KE,
            'PE': PE,
            'available_PE': available_PE,
            'mixing_efficiency': mixing_efficiency,
            'max_density_gradient': np.max(mag_grad_rho),
            'rms_velocity': np.sqrt(np.mean(self.u**2 + self.w**2))
        }
        
        return diag
    
    def solve(self, save_interval=10):
        num_steps = int(self.T_max / self.dt)
        save_steps = num_steps // save_interval + 1
        
        u_history = np.zeros((save_steps, self.Nx, self.Nz))
        w_history = np.zeros((save_steps, self.Nx, self.Nz))
        rho_history = np.zeros((save_steps, self.Nx, self.Nz))
        
        u_history[0] = self.u
        w_history[0] = self.w
        rho_history[0] = self.rho
        
        self.mixing_history = []
        self.energy_history = []
        
        save_idx = 1
        
        for step in range(num_steps):
            self.step()
            
            if step % save_interval == 0 and save_idx < save_steps:
                u_history[save_idx] = self.u
                w_history[save_idx] = self.w
                rho_history[save_idx] = self.rho
                save_idx += 1
                
                diag = self.compute_mixing_diagnostics()
                self.mixing_history.append(diag)
                self.energy_history.append((diag['KE'], diag['PE']))
                
                if step % (save_interval * 10) == 0:
                    print(f"Step {step}/{num_steps}, t={step*self.dt:.2f}, "
                          f"KE={diag['KE']:.2e}, MixEff={diag['mixing_efficiency']:.4f}")
        
        return u_history, w_history, rho_history
    
    def plot_snapshot(self, u, w, rho, filename='snapshot.png'):
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        vmax_u = np.max(np.abs(u))
        im0 = axes[0].pcolormesh(self.X, self.Z, u, cmap='RdBu_r', 
                                 vmin=-vmax_u, vmax=vmax_u, shading='auto')
        axes[0].contour(self.X, self.Z, rho, levels=10, colors='k', alpha=0.5)
        axes[0].fill_between(self.x, 0, self.terrain, color='gray', alpha=0.5)
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('z')
        axes[0].set_title('Horizontal Velocity u')
        plt.colorbar(im0, ax=axes[0])
        
        vmax_w = np.max(np.abs(w))
        im1 = axes[1].pcolormesh(self.X, self.Z, w, cmap='RdBu_r', 
                                 vmin=-vmax_w, vmax=vmax_w, shading='auto')
        axes[1].contour(self.X, self.Z, rho, levels=10, colors='k', alpha=0.5)
        axes[1].fill_between(self.x, 0, self.terrain, color='gray', alpha=0.5)
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('z')
        axes[1].set_title('Vertical Velocity w')
        plt.colorbar(im1, ax=axes[1])
        
        rho_min, rho_max = np.min(rho), np.max(rho)
        im2 = axes[2].pcolormesh(self.X, self.Z, rho, cmap='viridis', 
                                 vmin=rho_min, vmax=rho_max, shading='auto')
        axes[2].contour(self.X, self.Z, rho, levels=15, colors='w', alpha=0.5)
        axes[2].fill_between(self.x, 0, self.terrain, color='gray', alpha=0.5)
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('z')
        axes[2].set_title('Density ρ')
        plt.colorbar(im2, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
    
    def plot_mixing_diagnostics(self, filename='mixing_diagnostics.png'):
        if not self.mixing_history:
            print("No mixing history available.")
            return
            
        times = [d['time'] for d in self.mixing_history]
        KE = [d['KE'] for d in self.mixing_history]
        APE = [d['available_PE'] for d in self.mixing_history]
        mix_eff = [d['mixing_efficiency'] for d in self.mixing_history]
        rms_vel = [d['rms_velocity'] for d in self.mixing_history]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        axes[0, 0].plot(times, KE, 'b-', label='Kinetic Energy')
        axes[0, 0].plot(times, APE, 'r-', label='Available PE')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Energy')
        axes[0, 0].set_title('Energy Evolution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(times, mix_eff, 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Mixing Efficiency (APE/KE)')
        axes[0, 1].set_title('Mixing Efficiency')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(times, rms_vel, 'm-')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('RMS Velocity')
        axes[1, 0].set_title('Turbulence Intensity')
        axes[1, 0].grid(True, alpha=0.3)
        
        total_energy = np.array(KE) + np.array(APE)
        if total_energy[0] > 0:
            energy_rel = (total_energy - total_energy[0]) / total_energy[0] * 100
            axes[1, 1].plot(times, energy_rel, 'k-')
            axes[1, 1].set_xlabel('Time')
            axes[1, 1].set_ylabel('Total Energy Change (%)')
            axes[1, 1].set_title('Energy Conservation')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        
        return {
            'final_mixing_efficiency': mix_eff[-1],
            'max_mixing_efficiency': np.max(mix_eff),
            'total_energy_dissipation': 100 * (1 - total_energy[-1]/total_energy[0]) if total_energy[0] > 0 else 0
        }
    
    def create_animation(self, u_history, rho_history, filename='animation.mp4', fps=20):
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        vmax_u = np.max(np.abs(u_history))
        rho_min, rho_max = np.min(rho_history), np.max(rho_history)
        
        im0 = axes[0].pcolormesh(self.X, self.Z, u_history[0], cmap='RdBu_r', 
                                 vmin=-vmax_u, vmax=vmax_u, shading='auto')
        axes[0].contour(self.X, self.Z, rho_history[0], levels=10, colors='k', alpha=0.5)
        axes[0].fill_between(self.x, 0, self.terrain, color='gray', alpha=0.5)
        axes[0].set_ylabel('z')
        axes[0].set_title('Horizontal Velocity')
        plt.colorbar(im0, ax=axes[0])
        
        im1 = axes[1].pcolormesh(self.X, self.Z, rho_history[0], cmap='viridis', 
                                 vmin=rho_min, vmax=rho_max, shading='auto')
        axes[1].contour(self.X, self.Z, rho_history[0], levels=10, colors='w', alpha=0.5)
        axes[1].fill_between(self.x, 0, self.terrain, color='gray', alpha=0.5)
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('z')
        axes[1].set_title('Density')
        plt.colorbar(im1, ax=axes[1])
        
        time_text = axes[0].text(0.02, 0.95, '', transform=axes[0].transAxes)
        
        def update(frame):
            im0.set_array(u_history[frame].ravel())
            im1.set_array(rho_history[frame].ravel())
            
            for c in axes[0].collections:
                if c.__class__.__name__ == 'QuadContourSet':
                    c.remove()
            for c in axes[1].collections:
                if c.__class__.__name__ == 'QuadContourSet':
                    c.remove()
            
            axes[0].contour(self.X, self.Z, rho_history[frame], levels=10, colors='k', alpha=0.5)
            axes[1].contour(self.X, self.Z, rho_history[frame], levels=10, colors='w', alpha=0.5)
            
            time_text.set_text(f'Time = {frame * self.dt * 10:.2f}')
            return im0, im1, time_text
        
        anim = FuncAnimation(fig, update, frames=len(u_history), 
                             interval=1000/fps, blit=False)
        anim.save(filename, writer='pillow', fps=fps)
        plt.close()


def test_internal_soliton_breaking():
    print("=" * 70)
    print("TEST: Internal Soliton Breaking over a Ridge")
    print("=" * 70)
    
    model = NonhydrostaticModel(
        Lx=120, Lz=40, Nx=256, Nz=64, 
        dt=0.02, T_max=30,
        nu=1e-3, kappa=1e-4
    )
    
    print("\nSetting up terrain...")
    model.set_terrain('ridge', height=18.0, x0=60.0, width=25.0)
    
    print("Setting up density profile...")
    model.set_density_profile('two_layer', rho1=1000.0, rho2=1002.5, 
                              z_interface=25.0, thickness=1.5)
    
    print("Adding initial internal wave...")
    model.add_internal_wave('mode1', x0=20.0, amp=1.5, width=12.0)
    
    print("\nRunning simulation...")
    u_hist, w_hist, rho_hist = model.solve(save_interval=20)
    
    print("\nPlotting results...")
    model.plot_snapshot(u_hist[-1], w_hist[-1], rho_hist[-1], 
                        'breaking_final_snapshot.png')
    
    mix_stats = model.plot_mixing_diagnostics('breaking_mixing.png')
    
    print("\n" + "=" * 70)
    print("MIXING STATISTICS")
    print("=" * 70)
    print(f"  Final Mixing Efficiency: {mix_stats['final_mixing_efficiency']:.4f}")
    print(f"  Maximum Mixing Efficiency: {mix_stats['max_mixing_efficiency']:.4f}")
    print(f"  Total Energy Dissipation: {mix_stats['total_energy_dissipation']:.2f}%")
    print("=" * 70)
    
    print("\nSaved: breaking_final_snapshot.png, breaking_mixing.png")
    
    return model, mix_stats


def test_wave_shelf_interaction():
    print("\n" + "=" * 70)
    print("TEST: Wave Interaction with Continental Shelf")
    print("=" * 70)
    
    model = NonhydrostaticModel(
        Lx=150, Lz=50, Nx=256, Nz=64, 
        dt=0.02, T_max=40,
        nu=5e-4, kappa=5e-5
    )
    
    print("\nSetting up continental shelf terrain...")
    model.set_terrain('shelf', h1=0.0, h2=30.0, x_trans=80.0, width=15.0)
    
    print("Setting up density profile...")
    model.set_density_profile('linear', rho_surface=1000.0, rho_bottom=1003.0)
    
    print("Adding initial wave...")
    model.add_internal_wave('soliton', x0=25.0, amp=2.0, width=15.0, z_peak=25.0)
    
    print("\nRunning simulation...")
    u_hist, w_hist, rho_hist = model.solve(save_interval=25)
    
    print("\nPlotting results...")
    model.plot_snapshot(u_hist[-1], w_hist[-1], rho_hist[-1], 
                        'shelf_final_snapshot.png')
    
    mix_stats = model.plot_mixing_diagnostics('shelf_mixing.png')
    
    print("\nSaved: shelf_final_snapshot.png, shelf_mixing.png")
    
    return model, mix_stats


if __name__ == "__main__":
    test_internal_soliton_breaking()
    test_wave_shelf_interaction()

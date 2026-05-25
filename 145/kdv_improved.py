import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('Agg')


class VariableCoefficientKdV:
    def __init__(self, L=50, N=1024, dt=0.001, T_max=10, 
                 equation_type='vkdv', g=9.81):
        self.L = L
        self.N = N
        self.dt = dt
        self.T_max = T_max
        self.equation_type = equation_type
        self.g = g
        
        self.x = np.linspace(-L/2, L/2, N)
        self.dx = self.x[1] - self.x[0]
        
        self.k = 2 * np.pi * np.fft.fftfreq(N, d=L/N)
        self.k2 = self.k ** 2
        self.k3 = self.k ** 3
        
        self.dealias_filter = np.ones(N)
        self.dealias_filter[np.abs(self.k) > 2/3 * np.max(self.k)] = 0
        
        self.energy_history = []
        self.mass_history = []
        
    def compute_diagnostics(self, u, h=None):
        mass = np.sum(u) * self.dx
        energy = 0.5 * np.sum(u**2) * self.dx
        
        if h is not None:
            energy = 0.5 * np.sum(h * u**2) * self.dx
            
        return mass, energy
    
    def terrain_function(self, x, terrain_type='flat', **kwargs):
        if terrain_type == 'flat':
            return np.ones_like(x)
        elif terrain_type == 'shelf':
            h1 = kwargs.get('h1', 1.0)
            h2 = kwargs.get('h2', 0.5)
            x_trans = kwargs.get('x_trans', 0)
            width = kwargs.get('width', 5)
            transition = 0.5 * (1 + np.tanh((x - x_trans) / width))
            return h1 * (1 - transition) + h2 * transition
        elif terrain_type == 'ridge':
            h0 = kwargs.get('h0', 1.0)
            height = kwargs.get('height', 0.3)
            x0 = kwargs.get('x0', 0)
            width = kwargs.get('width', 3)
            return h0 - height * np.exp(-((x - x0) ** 2) / (2 * width ** 2))
        elif terrain_type == 'trench':
            h0 = kwargs.get('h0', 1.0)
            depth = kwargs.get('depth', 0.3)
            x0 = kwargs.get('x0', 0)
            width = kwargs.get('width', 3)
            return h0 + depth * np.exp(-((x - x0) ** 2) / (2 * width ** 2))
        elif terrain_type == 'smooth_ridge':
            h0 = kwargs.get('h0', 1.0)
            height = kwargs.get('height', 0.3)
            x0 = kwargs.get('x0', 0)
            width = kwargs.get('width', 10)
            return h0 - height * (np.cos(np.pi * (x - x0) / width) + 1) / 2 * \
                   (np.abs(x - x0) < width)
        else:
            return np.ones_like(x)
    
    def soliton_solution(self, x, h0=1.0, x0=0, amp=0.1):
        c = np.sqrt(self.g * h0 * (1 + amp / (2 * h0)))
        width = np.sqrt(4 * h0**3 / (3 * amp))
        xi = (x - x0) / width
        return amp * (1 / np.cosh(xi)) ** 2
    
    def gaussian_wave(self, x, x0=0, sigma=2, amp=1):
        return amp * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))
    
    def spatial_derivative(self, u):
        u_hat = np.fft.fft(u)
        return np.fft.ifft(1j * self.k * u_hat).real
    
    def third_derivative(self, u):
        u_hat = np.fft.fft(u)
        return np.fft.ifft(-1j * self.k3 * u_hat).real
    
    def rhs_vkdv(self, u, t, h, h_x, h_xx):
        u_x = self.spatial_derivative(u)
        u_xxx = self.third_derivative(u)
        
        sqrt_h = np.sqrt(h)
        sqrt_h_x = 0.5 * h_x / sqrt_h
        
        term1 = -u * u_x
        term2 = - (self.g * h_x + (sqrt_h_x * u + sqrt_h * u_x)**2 / sqrt_h)
        term3 = - (self.g / 3) * h**2 * u_xxx
        
        rhs = term1 + term2 + term3
        
        rhs_hat = np.fft.fft(rhs)
        rhs_hat *= self.dealias_filter
        rhs = np.fft.ifft(rhs_hat).real
        
        return rhs
    
    def rhs_conservation_vkdv(self, u, t, h, h_x):
        hu = h * u
        hu_x = self.spatial_derivative(hu)
        
        u_x = self.spatial_derivative(u)
        u_xx = self.spatial_derivative(u_x)
        u_xxx = self.spatial_derivative(u_xx)
        
        flux = 0.5 * hu**2 + 0.5 * self.g * h**2 * u - (self.g / 3) * h**3 * u_xx
        flux_x = self.spatial_derivative(flux)
        
        rhs = -flux_x / h
        
        rhs_hat = np.fft.fft(rhs)
        rhs_hat *= self.dealias_filter
        rhs = np.fft.ifft(rhs_hat).real
        
        return rhs
    
    def rhs_boussinesq(self, u, t, h, h_x):
        eta = u
        eta_x = self.spatial_derivative(eta)
        eta_xx = self.spatial_derivative(eta_x)
        eta_xxx = self.spatial_derivative(eta_xx)
        
        c = np.sqrt(self.g * h)
        c2 = c**2
        
        term1 = - (c * eta * eta_x + c2 * eta_x + c * h_x * eta)
        term2 = (c / 6) * (h**2 * eta_xxx)
        term3 = (c / 2) * h * h_x * eta_xx
        
        rhs = term1 + term2 + term3
        
        rhs_hat = np.fft.fft(rhs)
        rhs_hat *= self.dealias_filter
        rhs = np.fft.ifft(rhs_hat).real
        
        return rhs
    
    def rhs_shallow_water(self, u, t, h, h_x):
        eta = u
        eta_x = self.spatial_derivative(eta)
        
        c = np.sqrt(self.g * h)
        flux = c * eta * (1 + eta / (2 * h))
        flux_x = self.spatial_derivative(flux)
        
        rhs = -flux_x / h
        
        rhs_hat = np.fft.fft(rhs)
        rhs_hat *= self.dealias_filter
        rhs = np.fft.ifft(rhs_hat).real
        
        return rhs
    
    def solve(self, u0, terrain_type='flat', **terrain_kwargs):
        h = self.terrain_function(self.x, terrain_type, **terrain_kwargs)
        h_x = np.gradient(h, self.x)
        h_xx = np.gradient(h_x, self.x)
        
        num_steps = int(self.T_max / self.dt)
        u = u0.copy()
        
        u_history = np.zeros((num_steps + 1, self.N))
        u_history[0] = u
        
        self.energy_history = []
        self.mass_history = []
        
        mass, energy = self.compute_diagnostics(u, h)
        self.mass_history.append(mass)
        self.energy_history.append(energy)
        
        for i in range(num_steps):
            t = i * self.dt
            
            if self.equation_type == 'vkdv':
                k1 = self.rhs_conservation_vkdv(u, t, h, h_x)
                k2 = self.rhs_conservation_vkdv(u + 0.5 * self.dt * k1, t + 0.5 * self.dt, h, h_x)
                k3 = self.rhs_conservation_vkdv(u + 0.5 * self.dt * k2, t + 0.5 * self.dt, h, h_x)
                k4 = self.rhs_conservation_vkdv(u + self.dt * k3, t + self.dt, h, h_x)
            elif self.equation_type == 'boussinesq':
                k1 = self.rhs_boussinesq(u, t, h, h_x)
                k2 = self.rhs_boussinesq(u + 0.5 * self.dt * k1, t + 0.5 * self.dt, h, h_x)
                k3 = self.rhs_boussinesq(u + 0.5 * self.dt * k2, t + 0.5 * self.dt, h, h_x)
                k4 = self.rhs_boussinesq(u + self.dt * k3, t + self.dt, h, h_x)
            elif self.equation_type == 'shallow_water':
                k1 = self.rhs_shallow_water(u, t, h, h_x)
                k2 = self.rhs_shallow_water(u + 0.5 * self.dt * k1, t + 0.5 * self.dt, h, h_x)
                k3 = self.rhs_shallow_water(u + 0.5 * self.dt * k2, t + 0.5 * self.dt, h, h_x)
                k4 = self.rhs_shallow_water(u + self.dt * k3, t + self.dt, h, h_x)
            else:
                raise ValueError(f"Unknown equation type: {self.equation_type}")
            
            u = u + (self.dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
            u_history[i + 1] = u
            
            mass, energy = self.compute_diagnostics(u, h)
            self.mass_history.append(mass)
            self.energy_history.append(energy)
        
        return u_history, h
    
    def plot_diagnostics(self, save_path='diagnostics.png'):
        t = np.arange(len(self.energy_history)) * self.dt
        
        energy = np.array(self.energy_history)
        mass = np.array(self.mass_history)
        
        energy_rel = (energy - energy[0]) / energy[0] * 100
        mass_rel = (mass - mass[0]) / mass[0] * 100
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        axes[0].plot(t, energy, 'b-', linewidth=2)
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Energy')
        axes[0].set_title(f'Energy Evolution (Max Rel. Error: {np.max(np.abs(energy_rel)):.4f}%)')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(t, energy_rel, 'r-', linewidth=2)
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Relative Energy Error (%)')
        axes[1].set_title('Relative Energy Conservation Error')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        return {
            'max_energy_error': np.max(np.abs(energy_rel)),
            'final_energy_error': energy_rel[-1]
        }
    
    def plot_evolution(self, u_history, h=None, save_path='evolution.png'):
        t_indices = np.linspace(0, len(u_history) - 1, 5, dtype=int)
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        for idx in t_indices:
            t = idx * self.dt
            axes[0].plot(self.x, u_history[idx], label=f't={t:.2f}', linewidth=1.5)
        
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('Wave Amplitude')
        axes[0].set_title(f'Wave Evolution - {self.equation_type.upper()} Model')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        if h is not None:
            axes[1].plot(self.x, h, 'k-', linewidth=2)
            axes[1].set_xlabel('x')
            axes[1].set_ylabel('Water Depth h(x)')
            axes[1].set_title('Terrain Profile')
            axes[1].grid(True, alpha=0.3)
            axes[1].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
    
    def plot_spacetime(self, u_history, save_path='spacetime.png'):
        t = np.arange(len(u_history)) * self.dt
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        vmax = np.max(np.abs(u_history))
        im = ax.pcolormesh(self.x, t, u_history, cmap='RdBu_r', 
                           shading='auto', vmin=-vmax, vmax=vmax)
        
        ax.set_xlabel('x')
        ax.set_ylabel('Time')
        ax.set_title(f'Spacetime Evolution - {self.equation_type.upper()}')
        plt.colorbar(im, ax=ax, label='Amplitude')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
    
    def create_animation(self, u_history, h=None, save_path='animation.gif', fps=30):
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        vmax = np.max(np.abs(u_history))
        line, = axes[0].plot(self.x, u_history[0], 'b-', linewidth=2)
        axes[0].set_xlim(self.x[0], self.x[-1])
        axes[0].set_ylim(-1.2 * vmax, 1.2 * vmax)
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title(f'Wave Propagation - {self.equation_type.upper()}')
        axes[0].grid(True, alpha=0.3)
        
        if h is not None:
            axes[1].plot(self.x, h, 'k-', linewidth=2)
            axes[1].set_xlabel('x')
            axes[1].set_ylabel('Water Depth h(x)')
            axes[1].set_title('Terrain Profile')
            axes[1].grid(True, alpha=0.3)
            axes[1].invert_yaxis()
        else:
            axes[1].axis('off')
        
        time_text = axes[0].text(0.02, 0.95, '', transform=axes[0].transAxes)
        
        def update(frame):
            line.set_ydata(u_history[frame])
            time_text.set_text(f'Time = {frame * self.dt:.3f}')
            return line, time_text
        
        anim = FuncAnimation(fig, update, frames=len(u_history), 
                             interval=1000/fps, blit=True)
        
        anim.save(save_path, writer='pillow', fps=fps)
        plt.close()


def compare_models():
    print("=" * 70)
    print("Comparing Wave Models on Variable Terrain")
    print("=" * 70)
    
    L = 100
    N = 1024
    dt = 0.0005
    T_max = 4
    
    x = np.linspace(-L/2, L/2, N)
    
    def shelf_terrain(x):
        h1, h2 = 1.0, 0.6
        x_trans, width = 0, 5
        transition = 0.5 * (1 + np.tanh((x - x_trans) / width))
        return h1 * (1 - transition) + h2 * transition
    
    h = shelf_terrain(x)
    h0 = 1.0
    amp = 0.1
    u0 = amp * (1 / np.cosh((x + 30) / np.sqrt(4 * h0**3 / (3 * amp)))) ** 2
    
    results = {}
    
    for eq_type in ['vkdv', 'boussinesq', 'shallow_water']:
        print(f"\nRunning {eq_type.upper()} model...")
        
        solver = VariableCoefficientKdV(L=L, N=N, dt=dt, T_max=T_max, 
                                        equation_type=eq_type)
        
        def terrain_func(x, **kwargs):
            return h
        
        solver.terrain_function = terrain_func
        u_history, _ = solver.solve(u0, terrain_type='flat')
        
        diag = solver.plot_diagnostics(save_path=f'{eq_type}_diagnostics.png')
        solver.plot_evolution(u_history, h, save_path=f'{eq_type}_evolution.png')
        solver.plot_spacetime(u_history, save_path=f'{eq_type}_spacetime.png')
        
        results[eq_type] = {
            'u_history': u_history,
            'energy_error': diag['max_energy_error']
        }
        
        print(f"  Max energy conservation error: {diag['max_energy_error']:.4f}%")
    
    print("\n" + "=" * 70)
    print("Energy Conservation Summary:")
    print("-" * 70)
    for eq_type, res in results.items():
        print(f"  {eq_type.upper():15s}: {res['energy_error']:.6f}% error")
    print("=" * 70)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    t_plot = [0, int(T_max/dt/4), int(T_max/dt/2), int(T_max/dt)]
    
    for i, eq_type in enumerate(['vkdv', 'boussinesq', 'shallow_water']):
        for t_idx in t_plot:
            axes[i].plot(x, results[eq_type]['u_history'][t_idx], 
                        label=f't={t_idx*dt:.2f}')
        axes[i].set_title(f'{eq_type.upper()} - Energy Error: {results[eq_type]["energy_error"]:.4f}%')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylabel('Amplitude')
    
    axes[-1].set_xlabel('x')
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150)
    plt.close()
    
    print("\nSaved comparison figure: model_comparison.png")
    
    return results


if __name__ == "__main__":
    compare_models()

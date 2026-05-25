import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class DropletOscillationBEM:
    def __init__(self, R0=1.0, gamma=0.072, rho=1000.0, N=64, dt=1e-5):
        self.R0 = R0
        self.gamma = gamma
        self.rho = rho
        self.N = N
        self.dt = dt
        
        self.theta = np.linspace(0, 2*np.pi, N, endpoint=False)
        self.dtheta = 2 * np.pi / N
        self.initialize_droplet()
        
        self.rayleigh_freq = self.rayleigh_theoretical_frequency(n=2)
        
    def initialize_droplet(self, amplitude=0.05, mode=2):
        self.r = self.R0 * (1 + amplitude * np.cos(mode * self.theta))
        self.x = self.r * np.cos(self.theta)
        self.y = self.r * np.sin(self.theta)
        self.phi = np.zeros(self.N)
        self.time = 0.0
        self.r_history = []
        self.time_history = []
        self.shape_history = []
        
    def rayleigh_theoretical_frequency(self, n=2):
        return np.sqrt(n * (n - 1) * (n + 2) * self.gamma / (self.rho * self.R0**3))
    
    def compute_geometry(self):
        dx = np.gradient(self.x, self.dtheta)
        dy = np.gradient(self.y, self.dtheta)
        d2x = np.gradient(dx, self.dtheta)
        d2y = np.gradient(dy, self.dtheta)
        
        ds = np.sqrt(dx**2 + dy**2)
        ds = np.where(ds < 1e-10, 1e-10, ds)
        
        nx = dy / ds
        ny = -dx / ds
        
        numerator = dx * d2y - dy * d2x
        denominator = ds**3
        kappa = numerator / denominator
        
        return nx, ny, kappa, ds
    
    def compute_influence_matrices(self, nx, ny, ds):
        G = np.zeros((self.N, self.N))
        H = np.zeros((self.N, self.N))
        
        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    G[i, j] = -self.R0 * np.log(self.R0) / (2 * np.pi) * ds[j]
                    H[i, j] = 0.5
                else:
                    dx_ij = self.x[i] - self.x[j]
                    dy_ij = self.y[i] - self.y[j]
                    r_ij = np.sqrt(dx_ij**2 + dy_ij**2)
                    
                    if r_ij < 1e-10:
                        G[i, j] = 0
                        H[i, j] = 0.5
                        continue
                    
                    G[i, j] = -1.0 / (2 * np.pi) * np.log(r_ij) * ds[j]
                    
                    dot_prod = dx_ij * nx[j] + dy_ij * ny[j]
                    H[i, j] = 1.0 / (2 * np.pi) * dot_prod / (r_ij**2) * ds[j]
        
        return G, H
    
    def solve_bem(self):
        nx, ny, kappa, ds = self.compute_geometry()
        G, H = self.compute_influence_matrices(nx, ny, ds)
        
        pressure_jump = self.gamma * kappa
        
        rhs = np.dot(H, self.phi)
        
        try:
            dphi_dn = np.linalg.solve(G, rhs)
        except np.linalg.LinAlgError:
            dphi_dn = np.linalg.lstsq(G, rhs, rcond=None)[0]
        
        return dphi_dn, pressure_jump, nx, ny
    
    def time_step(self):
        dphi_dn, pressure_jump, nx, ny = self.solve_bem()
        
        v_n = dphi_dn
        
        self.x += v_n * nx * self.dt
        self.y += v_n * ny * self.dt
        
        grad_p = -pressure_jump / self.rho
        self.phi += grad_p * self.dt
        
        self.time += self.dt
        
        current_r = np.sqrt(self.x**2 + self.y**2)
        self.r_history.append(np.mean(current_r))
        self.time_history.append(self.time)
        self.shape_history.append((self.x.copy(), self.y.copy()))
    
    def run_simulation(self, total_time=0.01, save_interval=10):
        n_steps = int(total_time / self.dt)
        for step in range(n_steps):
            self.time_step()
        return np.array(self.time_history), np.array(self.r_history)
    
    def compute_oscillation_frequency(self):
        if len(self.r_history) < 10:
            return None
        
        r_array = np.array(self.r_history)
        t_array = np.array(self.time_history)
        
        r_mean = np.mean(r_array)
        r_centered = r_array - r_mean
        
        try:
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(r_centered, height=0.5*np.max(np.abs(r_centered)))
            
            if len(peaks) >= 2:
                periods = np.diff(t_array[peaks])
                avg_period = np.mean(periods)
                return 2 * np.pi / avg_period
        except ImportError:
            pass
        
        return None
    
    def plot_results(self):
        if len(self.r_history) == 0:
            print("No simulation data available. Run simulation first.")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        theoretical_freq = self.rayleigh_freq
        theoretical_period = 2 * np.pi / theoretical_freq
        
        t = np.array(self.time_history)
        r = np.array(self.r_history)
        
        ax = axes[0, 0]
        ax.plot(t, r, 'b-', linewidth=1.5)
        ax.axhline(y=self.R0, color='r', linestyle='--', label=f'Equilibrium R={self.R0}m')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Mean Radius (m)')
        ax.set_title('Radius vs Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 1]
        r_centered = r - np.mean(r)
        ax.plot(t, r_centered, 'g-', linewidth=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Radius Perturbation (m)')
        ax.set_title('Oscillation Amplitude')
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 0]
        n = len(t)
        if n > 0:
            freqs = np.fft.fftfreq(n, self.dt)
            fft_vals = np.fft.fft(r_centered)
            positive_idx = freqs >= 0
            ax.plot(freqs[positive_idx] * 2 * np.pi, np.abs(fft_vals[positive_idx]), 'b-', linewidth=1.5)
            ax.axvline(x=theoretical_freq, color='r', linestyle='--', 
                       label=f'Rayleigh: {theoretical_freq:.2f} rad/s')
            ax.set_xlabel('Frequency (rad/s)')
            ax.set_ylabel('FFT Magnitude')
            ax.set_title('Frequency Spectrum')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, theoretical_freq * 3)
        
        ax = axes[1, 1]
        x_closed = np.append(self.x, self.x[0])
        y_closed = np.append(self.y, self.y[0])
        ax.plot(x_closed, y_closed, 'b-', linewidth=2, label='Final Shape')
        theta_circle = np.linspace(0, 2*np.pi, 100)
        ax.plot(self.R0 * np.cos(theta_circle), self.R0 * np.sin(theta_circle), 
                'r--', linewidth=1.5, label='Equilibrium')
        ax.set_aspect('equal')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title('Final Droplet Shape')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_animation(self, total_time=0.01, interval=50):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.set_aspect('equal')
        ax1.set_xlim(-1.5*self.R0, 1.5*self.R0)
        ax1.set_ylim(-1.5*self.R0, 1.5*self.R0)
        ax1.set_xlabel('x (m)')
        ax1.set_ylabel('y (m)')
        ax1.set_title('Droplet Shape Evolution')
        
        line, = ax1.plot([], [], 'b-', linewidth=2)
        circle = plt.Circle((0, 0), self.R0, color='r', fill=False, linestyle='--', alpha=0.5)
        ax1.add_patch(circle)
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Mean Radius (m)')
        ax2.set_title('Radius Oscillation')
        ax2.grid(True, alpha=0.3)
        
        radius_line, = ax2.plot([], [], 'g-', linewidth=1.5)
        
        n_steps = int(total_time / self.dt)
        self.r_history = []
        self.time_history = []
        
        def init():
            line.set_data([], [])
            radius_line.set_data([], [])
            return line, radius_line
        
        def update(frame):
            self.time_step()
            
            x_closed = np.append(self.x, self.x[0])
            y_closed = np.append(self.y, self.y[0])
            line.set_data(x_closed, y_closed)
            
            radius_line.set_data(self.time_history, self.r_history)
            if len(self.time_history) > 0:
                ax2.set_xlim(0, max(self.time_history) * 1.1)
                ax2.set_ylim(min(self.r_history) * 0.99, max(self.r_history) * 1.01)
            
            return line, radius_line
        
        anim = FuncAnimation(fig, update, frames=n_steps, init_func=init,
                             interval=interval, blit=True)
        
        plt.tight_layout()
        return anim


def main():
    print("="*60)
    print("Droplet Free Oscillation Simulation using BEM")
    print("Microgravity Environment - Surface Tension Dominated")
    print("="*60)
    
    R0 = 0.01
    gamma = 0.072
    rho = 1000.0
    
    print(f"\nPhysical Parameters:")
    print(f"  Initial Droplet Radius (R0): {R0} m")
    print(f"  Surface Tension (gamma): {gamma} N/m")
    print(f"  Fluid Density (rho): {rho} kg/m^3")
    
    sim = DropletOscillationBEM(R0=R0, gamma=gamma, rho=rho, N=64, dt=1e-6)
    
    theoretical_freq = sim.rayleigh_freq
    theoretical_period = 2 * np.pi / theoretical_freq
    
    print(f"\nRayleigh Theoretical Results (n=2 mode):")
    print(f"  Frequency: {theoretical_freq:.4f} rad/s")
    print(f"  Period: {theoretical_period:.6f} s")
    
    total_time = 5 * theoretical_period
    
    print(f"\nRunning simulation for {total_time:.6f} seconds...")
    print(f"  Time step: {sim.dt} s")
    print(f"  Number of steps: {int(total_time / sim.dt)}")
    print(f"  Boundary elements: {sim.N}")
    
    t, r = sim.run_simulation(total_time=total_time)
    
    computed_freq = sim.compute_oscillation_frequency()
    if computed_freq:
        print(f"\nComputed Oscillation Frequency: {computed_freq:.4f} rad/s")
        print(f"Relative Error: {abs(computed_freq - theoretical_freq) / theoretical_freq * 100:.2f}%")
    
    print("\nGenerating plots...")
    fig = sim.plot_results()
    if fig:
        fig.savefig('droplet_oscillation_results.png', dpi=150, bbox_inches='tight')
        print("Results saved to 'droplet_oscillation_results.png'")
        plt.show()
    
    print("\nSimulation complete!")
    print("="*60)


if __name__ == "__main__":
    main()

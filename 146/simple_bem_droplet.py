import numpy as np
import matplotlib.pyplot as plt


class SimpleBEMDroplet:
    def __init__(self, R0=0.01, gamma=0.072, rho=1000.0, N=32):
        self.R0 = R0
        self.gamma = gamma
        self.rho = rho
        self.N = N
        
        self.theta = np.linspace(0, 2*np.pi, N, endpoint=False)
        self.dtheta = 2*np.pi / N
        
        self.x = R0 * np.cos(self.theta)
        self.y = R0 * np.sin(self.theta)
        
        self.time = 0.0
        self.r_history = []
        self.time_history = []
        
    def rayleigh_frequency(self, n=2):
        return np.sqrt(n * (n-1) * (n+2) * self.gamma / (self.rho * self.R0**3))
    
    def perturb_shape(self, amplitude=0.05, mode=2):
        r = self.R0 * (1 + amplitude * np.cos(mode * self.theta))
        self.x = r * np.cos(self.theta)
        self.y = r * np.sin(self.theta)
    
    def compute_geometry(self):
        dx = np.gradient(self.x, self.dtheta)
        dy = np.gradient(self.y, self.dtheta)
        d2x = np.gradient(dx, self.dtheta)
        d2y = np.gradient(dy, self.dtheta)
        
        ds = np.sqrt(dx**2 + dy**2)
        ds = np.maximum(ds, 1e-10)
        
        nx = dy / ds
        ny = -dx / ds
        
        kappa = (dx * d2y - dy * d2x) / (ds**3)
        
        return nx, ny, kappa, ds
    
    def compute_velocity_from_theory(self, kappa, nx, ny, dt):
        pressure = self.gamma * kappa
        
        mean_pressure = np.mean(pressure)
        dp = pressure - mean_pressure
        
        dp_norm = dp / self.rho
        
        v_n = dp_norm * dt
        
        return v_n
    
    def step(self, dt=1e-6):
        nx, ny, kappa, ds = self.compute_geometry()
        
        v_n = self.compute_velocity_from_theory(kappa, nx, ny, dt)
        
        self.x += v_n * nx * dt
        self.y += v_n * ny * dt
        
        self.time += dt
        
        r_mean = np.mean(np.sqrt(self.x**2 + self.y**2))
        self.r_history.append(r_mean)
        self.time_history.append(self.time)
    
    def run(self, total_time, dt=1e-6):
        n_steps = int(total_time / dt)
        for _ in range(n_steps):
            self.step(dt)
        return np.array(self.time_history), np.array(self.r_history)


def main():
    R0 = 0.01
    gamma = 0.072
    rho = 1000.0
    
    droplet = SimpleBEMDroplet(R0=R0, gamma=gamma, rho=rho, N=64)
    
    omega_theory = droplet.rayleigh_frequency(n=2)
    period_theory = 2 * np.pi / omega_theory
    
    print("="*60)
    print("Simple BEM Droplet Oscillation Simulation")
    print("="*60)
    print(f"\nParameters:")
    print(f"  R0 = {R0} m")
    print(f"  gamma = {gamma} N/m")
    print(f"  rho = {rho} kg/m^3")
    print(f"  N = {droplet.N} boundary elements")
    print(f"\nRayleigh Frequency (n=2): {omega_theory:.4f} rad/s")
    print(f"Rayleigh Period: {period_theory:.6f} s")
    
    droplet.perturb_shape(amplitude=0.05, mode=2)
    
    total_time = 5 * period_theory
    dt = 1e-6
    
    print(f"\nRunning simulation for {total_time:.6f} s...")
    print(f"  dt = {dt} s")
    print(f"  n_steps = {int(total_time / dt)}")
    
    t, r = droplet.run(total_time, dt)
    
    r_centered = r - np.mean(r)
    
    peaks = []
    for i in range(1, len(r_centered)-1):
        if r_centered[i] > r_centered[i-1] and r_centered[i] > r_centered[i+1]:
            peaks.append(i)
    
    if len(peaks) >= 2:
        peak_times = t[peaks]
        periods = np.diff(peak_times)
        avg_period = np.mean(periods)
        omega_computed = 2 * np.pi / avg_period
        
        print(f"\nComputed Frequency: {omega_computed:.4f} rad/s")
        print(f"Computed Period: {avg_period:.6f} s")
        print(f"Relative Error: {abs(omega_computed - omega_theory)/omega_theory*100:.2f}%")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    ax = axes[0, 0]
    ax.plot(t, r, 'b-', linewidth=1.5)
    ax.axhline(y=R0, color='r', linestyle='--', label='Equilibrium')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mean Radius (m)')
    ax.set_title('Radius vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(t, r_centered, 'g-', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Radius Perturbation (m)')
    ax.set_title('Oscillation Amplitude')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    n = len(t)
    if n > 0:
        freqs = np.fft.fftfreq(n, dt)
        fft_vals = np.fft.fft(r_centered)
        positive_idx = freqs >= 0
        omega = freqs[positive_idx] * 2 * np.pi
        ax.plot(omega, np.abs(fft_vals[positive_idx]), 'b-', linewidth=1.5)
        ax.axvline(x=omega_theory, color='r', linestyle='--',
                   label=f'Rayleigh: {omega_theory:.2f} rad/s')
        ax.set_xlabel('Frequency (rad/s)')
        ax.set_ylabel('FFT Magnitude')
        ax.set_title('Frequency Spectrum')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, omega_theory * 3)
    
    ax = axes[1, 1]
    x_closed = np.append(droplet.x, droplet.x[0])
    y_closed = np.append(droplet.y, droplet.y[0])
    ax.plot(x_closed, y_closed, 'b-', linewidth=2, label='Final Shape')
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(R0*np.cos(theta), R0*np.sin(theta), 'r--', 
            linewidth=1.5, label='Equilibrium')
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Final Droplet Shape')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('simple_bem_results.png', dpi=150, bbox_inches='tight')
    print("\nResults saved to 'simple_bem_results.png'")
    
    plt.show()
    print("\nSimulation complete!")
    print("="*60)


if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt


class AnalyticalDropletOscillation:
    def __init__(self, R0=0.01, gamma=0.072, rho=1000.0):
        self.R0 = R0
        self.gamma = gamma
        self.rho = rho
    
    def rayleigh_frequency(self, n=2):
        return np.sqrt(n * (n - 1) * (n + 2) * self.gamma / (self.rho * self.R0**3))
    
    def oscillation_amplitude(self, t, n=2, A0=0.05):
        omega = self.rayleigh_frequency(n)
        return A0 * np.cos(omega * t)
    
    def droplet_shape(self, t, n=2, A0=0.05, theta=None):
        if theta is None:
            theta = np.linspace(0, 2*np.pi, 100)
        
        A = self.oscillation_amplitude(t, n, A0)
        r = self.R0 * (1 + A * np.cos(n * theta))
        
        return r * np.cos(theta), r * np.sin(theta)
    
    def simulate(self, total_time, dt=1e-5, n=2, A0=0.05):
        t = np.arange(0, total_time, dt)
        amplitude = self.oscillation_amplitude(t, n, A0)
        radius = self.R0 * (1 + amplitude)
        return t, radius


def compare_bem_vs_analytical():
    R0 = 0.01
    gamma = 0.072
    rho = 1000.0
    
    analytical = AnalyticalDropletOscillation(R0=R0, gamma=gamma, rho=rho)
    
    omega = analytical.rayleigh_frequency(n=2)
    period = 2 * np.pi / omega
    
    print("="*60)
    print("Analytical Validation of Droplet Oscillation")
    print("="*60)
    print(f"\nPhysical Parameters:")
    print(f"  R0 = {R0} m")
    print(f"  gamma = {gamma} N/m")
    print(f"  rho = {rho} kg/m^3")
    print(f"\nRayleigh Frequency (n=2): {omega:.4f} rad/s")
    print(f"Period: {period:.6f} s")
    
    total_time = 5 * period
    t, r = analytical.simulate(total_time, dt=1e-5, n=2, A0=0.05)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    ax = axes[0, 0]
    ax.plot(t, r, 'b-', linewidth=1.5)
    ax.axhline(y=R0, color='r', linestyle='--', label='Equilibrium')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mean Radius (m)')
    ax.set_title('Analytical Radius vs Time')
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
    n_modes = [2, 3, 4, 5]
    for n in n_modes:
        omega_n = analytical.rayleigh_frequency(n)
        ax.bar(n, omega_n, alpha=0.7, label=f'n={n}: {omega_n:.2f} rad/s')
    ax.set_xlabel('Mode Number n')
    ax.set_ylabel('Frequency (rad/s)')
    ax.set_title('Rayleigh Frequencies for Different Modes')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[1, 1]
    theta = np.linspace(0, 2*np.pi, 100)
    phases = [0, 0.25, 0.5, 0.75]
    colors = ['b', 'g', 'r', 'm']
    for i, phase in enumerate(phases):
        t_phase = phase * period
        x, y = analytical.droplet_shape(t_phase, n=2, A0=0.1, theta=theta)
        ax.plot(x, y, color=colors[i], linewidth=2, 
                label=f't={phase:.2f}T')
    ax.plot(R0*np.cos(theta), R0*np.sin(theta), 'k--', 
            linewidth=1, alpha=0.5, label='Equilibrium')
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Droplet Shape at Different Phases (n=2)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analytical_validation.png', dpi=150, bbox_inches='tight')
    print("\nResults saved to 'analytical_validation.png'")
    
    plt.show()


if __name__ == "__main__":
    compare_bem_vs_analytical()

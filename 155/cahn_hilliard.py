
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import label
import time

class CahnHilliardSolver:
    def __init__(self, N=128, L=32.0, dt=0.01, M=1.0, kappa=0.5, gamma=1.0):
        self.N = N
        self.L = L
        self.dx = L / N
        self.dt = dt
        self.M = M
        self.kappa = kappa
        self.gamma = gamma
        
        self.kx = 2 * np.pi * np.fft.fftfreq(N, d=self.dx)
        self.ky = 2 * np.pi * np.fft.fftfreq(N, d=self.dx)
        self.KX, self.KY = np.meshgrid(self.kx, self.ky)
        self.K2 = self.KX**2 + self.KY**2
        
        self.c = None
        
    def initialize_random(self, c0=0.0, noise=0.1):
        np.random.seed(42)
        self.c = c0 + noise * np.random.randn(self.N, self.N)
        self.c = np.clip(self.c, -1.0, 1.0)
        
    def initialize_spinodal(self, c0=0.0, noise=0.1):
        self.initialize_random(c0, noise)
        
    def step(self):
        c_hat = np.fft.fft2(self.c)
        
        f_prime = -self.gamma * self.c + self.gamma * self.c**3
        f_prime_hat = np.fft.fft2(f_prime)
        
        mu_hat = f_prime_hat - self.kappa * self.K2 * c_hat
        
        prefactor = -self.M * self.K2 * self.dt
        c_hat_new = (c_hat + prefactor * f_prime_hat) / (1.0 + self.M * self.kappa * self.K2 * self.dt)
        
        self.c = np.real(np.fft.ifft2(c_hat_new))
        self.c = np.clip(self.c, -1.0, 1.0)
        
        return self.c
    
    def run_simulation(self, n_steps, save_interval=10):
        history = []
        times = []
        
        for i in range(n_steps):
            self.step()
            if i % save_interval == 0:
                history.append(self.c.copy())
                times.append(i * self.dt)
                if i % (save_interval * 10) == 0:
                    print(f"Step {i}, t = {i*self.dt:.2f}")
        
        return np.array(history), np.array(times)

def compute_domain_size(c, threshold=0.0):
    binary = c > threshold
    labeled, num_features = label(binary)
    
    if num_features == 0:
        return 0.0
    
    sizes = []
    for i in range(1, num_features + 1):
        mask = labeled == i
        sizes.append(np.sum(mask))
    
    if len(sizes) == 0:
        return 0.0
    
    avg_size = np.mean(sizes)
    domain_radius = np.sqrt(avg_size / np.pi)
    
    return domain_radius

def run_full_simulation():
    solver = CahnHilliardSolver(N=128, L=32.0, dt=0.01, M=1.0, kappa=0.5, gamma=1.0)
    solver.initialize_spinodal(c0=0.0, noise=0.1)
    
    n_steps = 5000
    save_interval = 50
    
    print("Starting Cahn-Hilliard simulation...")
    start_time = time.time()
    
    history, times = solver.run_simulation(n_steps, save_interval)
    
    elapsed = time.time() - start_time
    print(f"Simulation completed in {elapsed:.2f} seconds")
    
    return history, times, solver

def plot_snapshots(history, times, num_plots=5):
    indices = np.linspace(0, len(history)-1, num_plots, dtype=int)
    
    fig, axes = plt.subplots(1, num_plots, figsize=(15, 3))
    
    vmin, vmax = -1, 1
    
    for idx, ax in zip(indices, axes):
        im = ax.imshow(history[idx], cmap='seismic', vmin=vmin, vmax=vmax, 
                       extent=[0, 32, 0, 32], origin='lower')
        ax.set_title(f't = {times[idx]:.1f}')
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig('snapshots.png', dpi=150)
    print("Saved snapshots.png")
    plt.show()

def plot_domain_growth(times, history):
    domain_sizes = []
    
    for c in history:
        size = compute_domain_size(c)
        domain_sizes.append(size)
    
    domain_sizes = np.array(domain_sizes)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(times, domain_sizes, 'b-', label='Simulation')
    ax1.set_xlabel('Time t')
    ax1.set_ylabel('Domain size L')
    ax1.set_title('Domain growth (linear scale)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    mask = times > 0.1
    log_t = np.log(times[mask])
    log_L = np.log(domain_sizes[mask])
    
    coeffs = np.polyfit(log_t, log_L, 1)
    slope = coeffs[0]
    
    ax2.loglog(times, domain_sizes, 'b-', label='Simulation')
    ax2.loglog(times[mask], np.exp(coeffs[1]) * times[mask]**slope, 'r--', 
               label=f'Fit: slope = {slope:.3f}')
    ax2.loglog(times[mask], np.exp(coeffs[1]) * times[mask]**(1/3), 'g--', 
               label='L ~ t^{1/3} (theory)')
    ax2.set_xlabel('Time t (log)')
    ax2.set_ylabel('Domain size L (log)')
    ax2.set_title('Domain growth (log-log scale)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('domain_growth.png', dpi=150)
    print(f"Saved domain_growth.png (fitted exponent: {slope:.3f})")
    plt.show()
    
    return domain_sizes

def create_animation(history, times):
    fig, ax = plt.subplots(figsize=(6, 5))
    
    vmin, vmax = -1, 1
    im = ax.imshow(history[0], cmap='seismic', vmin=vmin, vmax=vmax,
                   extent=[0, 32, 0, 32], origin='lower')
    plt.colorbar(im, label='Concentration c')
    title = ax.set_title(f't = {times[0]:.1f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    def update(frame):
        im.set_array(history[frame])
        title.set_text(f't = {times[frame]:.1f}')
        return im, title
    
    ani = FuncAnimation(fig, update, frames=len(history), 
                        interval=50, blit=True)
    
    try:
        ani.save('cahn_hilliard_animation.gif', writer='pillow', fps=10, dpi=100)
        print("Saved cahn_hilliard_animation.gif")
    except Exception as e:
        print(f"Could not save GIF: {e}")
        ani.save('cahn_hilliard_animation.mp4', writer='ffmpeg', fps=10, dpi=100)
        print("Saved cahn_hilliard_animation.mp4")
    
    plt.close()

def main():
    print("=" * 60)
    print("Cahn-Hilliard Equation Simulation")
    print("Spinodal Decomposition of Binary Mixture")
    print("=" * 60)
    
    print("\nParameters:")
    print("  Grid size: 128 x 128")
    print("  Domain size: 32 x 32")
    print("  Time step: dt = 0.01")
    print("  Mobility: M = 1.0")
    print("  Gradient coefficient: kappa = 0.5")
    print("  Nonlinearity: gamma = 1.0")
    print("  Initial condition: c=0 + small random noise")
    
    print("\n" + "=" * 60)
    
    history, times, solver = run_full_simulation()
    
    print("\n" + "=" * 60)
    print("Generating plots...")
    print("=" * 60)
    
    plot_snapshots(history, times)
    
    domain_sizes = plot_domain_growth(times, history)
    
    print("\nCreating animation...")
    create_animation(history, times)
    
    print("\n" + "=" * 60)
    print("Simulation completed successfully!")
    print("Generated files:")
    print("  - snapshots.png: Concentration snapshots at different times")
    print("  - domain_growth.png: Domain size growth analysis")
    print("  - cahn_hilliard_animation.gif: Time evolution animation")
    print("=" * 60)

if __name__ == "__main__":
    main()

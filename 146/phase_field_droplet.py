import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class PhaseFieldDroplet:
    def __init__(self, nx=128, ny=128, Lx=0.04, Ly=0.04, 
                 rho_l=1000.0, rho_g=1.2, mu_l=1e-3, mu_g=1.8e-5,
                 gamma=0.072, epsilon=0.001, W=0.001):
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.dx = Lx / nx
        self.dy = Ly / ny
        
        self.rho_l = rho_l
        self.rho_g = rho_g
        self.mu_l = mu_l
        self.mu_g = mu_g
        self.gamma = gamma
        self.epsilon = epsilon
        self.W = W
        
        self.x = np.linspace(self.dx/2, Lx - self.dx/2, nx)
        self.y = np.linspace(self.dy/2, Ly - self.dy/2, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        
        self.phi = np.zeros((nx, ny))
        self.u = np.zeros((nx, ny))
        self.v = np.zeros((nx, ny))
        self.p = np.zeros((nx, ny))
        
        self.time = 0.0
        self.interface_history = []
        self.time_history = []
        self.volume_history = []
        
    def initialize_single_droplet(self, R0=0.01, x0=None, y0=None, amplitude=0.05, mode=2):
        if x0 is None:
            x0 = self.Lx / 2
        if y0 is None:
            y0 = self.Ly / 2
        
        r = np.sqrt((self.X - x0)**2 + (self.Y - y0)**2)
        
        theta = np.arctan2(self.Y - y0, self.X - x0)
        R_perturbed = R0 * (1 + amplitude * np.cos(mode * theta))
        
        self.phi = 0.5 * (1 - np.tanh((r - R_perturbed) / (np.sqrt(2) * self.epsilon)))
        
        self.u = np.zeros((self.nx, self.ny))
        self.v = np.zeros((self.nx, self.ny))
        self.p = np.zeros((self.nx, self.ny))
        
        self.time = 0.0
        self.interface_history = []
        self.time_history = []
        self.volume_history = []
        
    def initialize_two_droplets(self, R1=0.006, R2=0.006, distance=0.02):
        x1 = self.Lx / 2 - distance / 2
        x2 = self.Lx / 2 + distance / 2
        yc = self.Ly / 2
        
        r1 = np.sqrt((self.X - x1)**2 + (self.Y - yc)**2)
        r2 = np.sqrt((self.X - x2)**2 + (self.Y - yc)**2)
        
        phi1 = 0.5 * (1 - np.tanh((r1 - R1) / (np.sqrt(2) * self.epsilon)))
        phi2 = 0.5 * (1 - np.tanh((r2 - R2) / (np.sqrt(2) * self.epsilon)))
        
        self.phi = np.clip(phi1 + phi2, 0, 1)
        
        self.u = np.zeros((self.nx, self.ny))
        self.v = np.zeros((self.nx, self.ny))
        self.p = np.zeros((self.nx, self.ny))
        
        self.time = 0.0
        self.interface_history = []
        self.time_history = []
        self.volume_history = []
        
    def compute_density(self):
        return self.rho_g + (self.rho_l - self.rho_g) * self.phi
    
    def compute_viscosity(self):
        return self.mu_g + (self.mu_l - self.mu_g) * self.phi
    
    def compute_chemical_potential(self):
        dfdphi = self.W * (4 * self.phi**3 - 6 * self.phi**2 + 2 * self.phi)
        
        lapl_phi = self.laplacian(self.phi)
        
        mu_chem = dfdphi - self.epsilon**2 * lapl_phi
        return mu_chem
    
    def compute_surface_tension_force(self):
        mu_chem = self.compute_chemical_potential()
        
        grad_phi_x, grad_phi_y = self.gradient(self.phi)
        
        fx = -mu_chem * grad_phi_x
        fy = -mu_chem * grad_phi_y
        
        return fx, fy
    
    def gradient(self, f):
        fx = np.zeros_like(f)
        fy = np.zeros_like(f)
        
        fx[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2 * self.dx)
        fy[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2 * self.dy)
        
        fx[0, :] = (f[1, :] - f[0, :]) / self.dx
        fx[-1, :] = (f[-1, :] - f[-2, :]) / self.dx
        fy[:, 0] = (f[:, 1] - f[:, 0]) / self.dy
        fy[:, -1] = (f[:, -1] - f[:, -2]) / self.dy
        
        return fx, fy
    
    def laplacian(self, f):
        lapl = np.zeros_like(f)
        
        lapl[1:-1, 1:-1] = (f[2:, 1:-1] - 2*f[1:-1, 1:-1] + f[:-2, 1:-1]) / self.dx**2 + \
                           (f[1:-1, 2:] - 2*f[1:-1, 1:-1] + f[1:-1, :-2]) / self.dy**2
        
        lapl[0, :] = lapl[1, :]
        lapl[-1, :] = lapl[-2, :]
        lapl[:, 0] = lapl[:, 1]
        lapl[:, -1] = lapl[:, -2]
        
        return lapl
    
    def divergence(self, fx, fy):
        div = np.zeros_like(fx)
        
        div[1:-1, 1:-1] = (fx[2:, 1:-1] - fx[:-2, 1:-1]) / (2 * self.dx) + \
                          (fy[1:-1, 2:] - fy[1:-1, :-2]) / (2 * self.dy)
        
        return div
    
    def solve_cahn_hilliard(self, dt, M=1e-10):
        mu_chem = self.compute_chemical_potential()
        
        grad_mu_x, grad_mu_y = self.gradient(mu_chem)
        
        flux_x = -M * grad_mu_x
        flux_y = -M * grad_mu_y
        
        dphi_dt = -self.divergence(flux_x, flux_y)
        
        self.phi += dt * dphi_dt
        
        self.phi = np.clip(self.phi, 0.0, 1.0)
    
    def solve_navier_stokes(self, dt):
        rho = self.compute_density()
        mu = self.compute_viscosity()
        
        fx_st, fy_st = self.compute_surface_tension_force()
        
        u_star = self.u.copy()
        v_star = self.v.copy()
        
        lapl_u = self.laplacian(self.u)
        lapl_v = self.laplacian(self.v)
        
        u_star += dt * (-self.u * self.gradient(self.u)[0] - self.v * self.gradient(self.u)[1] +
                        (mu / rho) * lapl_u + fx_st / rho)
        v_star += dt * (-self.u * self.gradient(self.v)[0] - self.v * self.gradient(self.v)[1] +
                        (mu / rho) * lapl_v + fy_st / rho)
        
        div_star = self.divergence(u_star, v_star)
        
        for _ in range(50):
            grad_px, grad_py = self.gradient(self.p)
            
            self.u[1:-1, 1:-1] = u_star[1:-1, 1:-1] - dt / rho[1:-1, 1:-1] * grad_px[1:-1, 1:-1]
            self.v[1:-1, 1:-1] = v_star[1:-1, 1:-1] - dt / rho[1:-1, 1:-1] * grad_py[1:-1, 1:-1]
            
            div = self.divergence(self.u, self.v)
            self.p += rho * (div_star - div) * self.dx / dt
            
        self.apply_boundary_conditions()
    
    def apply_boundary_conditions(self):
        self.u[0, :] = 0
        self.u[-1, :] = 0
        self.u[:, 0] = 0
        self.u[:, -1] = 0
        
        self.v[0, :] = 0
        self.v[-1, :] = 0
        self.v[:, 0] = 0
        self.v[:, -1] = 0
    
    def compute_interface_position(self):
        contour_level = 0.5
        interface_points = []
        
        for i in range(self.nx - 1):
            for j in range(self.ny - 1):
                phi_corners = [
                    self.phi[i, j],
                    self.phi[i+1, j],
                    self.phi[i+1, j+1],
                    self.phi[i, j+1]
                ]
                
                if min(phi_corners) <= contour_level <= max(phi_corners):
                    xc = self.x[i] + self.dx / 2
                    yc = self.y[j] + self.dy / 2
                    interface_points.append((xc, yc))
        
        return np.array(interface_points)
    
    def compute_droplet_volume(self):
        return np.sum(self.phi) * self.dx * self.dy
    
    def compute_equivalent_radius(self):
        volume = self.compute_droplet_volume()
        return np.sqrt(volume / np.pi)
    
    def rayleigh_frequency(self, R0=None, n=2):
        if R0 is None:
            R0 = self.compute_equivalent_radius()
        return np.sqrt(n * (n - 1) * (n + 2) * self.gamma / (self.rho_l * R0**3))
    
    def time_step(self, dt=1e-6):
        self.solve_navier_stokes(dt)
        
        self.solve_cahn_hilliard(dt, M=1e-10)
        
        self.time += dt
        
        if len(self.time_history) == 0 or (self.time - self.time_history[-1]) >= dt * 10:
            R_eq = self.compute_equivalent_radius()
            self.time_history.append(self.time)
            self.volume_history.append(R_eq)
    
    def run_simulation(self, total_time, dt=1e-6):
        n_steps = int(total_time / dt)
        
        for step in range(n_steps):
            self.time_step(dt)
            
            if step % 100 == 0:
                print(f"Step {step}/{n_steps}, Time = {self.time:.6f}s")
        
        return np.array(self.time_history), np.array(self.volume_history)
    
    def plot_state(self, title="Phase Field State"):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        ax = axes[0, 0]
        im = ax.contourf(self.X, self.Y, self.phi, levels=20, cmap='viridis')
        ax.contour(self.X, self.Y, self.phi, levels=[0.5], colors='red', linewidths=2)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title('Phase Field (phi)')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax)
        
        ax = axes[0, 1]
        rho = self.compute_density()
        im = ax.contourf(self.X, self.Y, rho, levels=20, cmap='plasma')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title('Density Field')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax)
        
        ax = axes[1, 0]
        speed = np.sqrt(self.u**2 + self.v**2)
        im = ax.contourf(self.X, self.Y, speed, levels=20, cmap='jet')
        ax.streamplot(self.x, self.y, self.u.T, self.v.T, color='white', density=0.5)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title('Velocity Field')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax)
        
        ax = axes[1, 1]
        im = ax.contourf(self.X, self.Y, self.p, levels=20, cmap='coolwarm')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title('Pressure Field')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax)
        
        plt.suptitle(f"{title} (t = {self.time:.6f}s)", fontsize=14)
        plt.tight_layout()
        return fig
    
    def animate(self, total_time, dt=1e-6, interval=50):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        ax1.set_aspect('equal')
        ax1.set_xlabel('x (m)')
        ax1.set_ylabel('y (m)')
        ax1.set_title('Phase Field Evolution')
        
        im1 = ax1.contourf(self.X, self.Y, self.phi, levels=20, cmap='viridis')
        contour1, = ax1.plot([], [], 'r-', linewidth=2)
        plt.colorbar(im1, ax=ax1)
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Equivalent Radius (m)')
        ax2.set_title('Droplet Oscillation')
        ax2.grid(True, alpha=0.3)
        
        line2, = ax2.plot([], [], 'b-', linewidth=1.5)
        
        n_steps = int(total_time / dt)
        self.time_history = []
        self.volume_history = []
        
        def init():
            line2.set_data([], [])
            return contour1, line2
        
        def update(frame):
            self.time_step(dt)
            
            ax1.clear()
            ax1.contourf(self.X, self.Y, self.phi, levels=20, cmap='viridis')
            ax1.contour(self.X, self.Y, self.phi, levels=[0.5], colors='red', linewidths=2)
            ax1.set_xlabel('x (m)')
            ax1.set_ylabel('y (m)')
            ax1.set_title(f'Phase Field (t = {self.time:.6f}s)')
            ax1.set_aspect('equal')
            
            line2.set_data(self.time_history, self.volume_history)
            if len(self.time_history) > 0:
                ax2.set_xlim(0, max(self.time_history) * 1.1)
                ax2.set_ylim(min(self.volume_history) * 0.99, max(self.volume_history) * 1.01)
            
            return contour1, line2
        
        anim = FuncAnimation(fig, update, frames=n_steps//10, init_func=init,
                             interval=interval, blit=False)
        
        plt.tight_layout()
        return anim


def main_single_droplet():
    print("="*60)
    print("Phase Field Simulation of Droplet Oscillation")
    print("(Cahn-Hilliard + Navier-Stokes)")
    print("="*60)
    
    nx, ny = 128, 128
    Lx, Ly = 0.04, 0.04
    R0 = 0.01
    
    print(f"\nDomain: {Lx}m x {Ly}m, Grid: {nx}x{ny}")
    print(f"Droplet Radius: {R0}m")
    print(f"Surface Tension: 0.072 N/m")
    
    sim = PhaseFieldDroplet(nx=nx, ny=ny, Lx=Lx, Ly=Ly,
                           rho_l=1000.0, rho_g=1.2,
                           mu_l=1e-3, mu_g=1.8e-5,
                           gamma=0.072, epsilon=Lx/100)
    
    sim.initialize_single_droplet(R0=R0, amplitude=0.1, mode=2)
    
    omega_theory = sim.rayleigh_frequency(R0=R0, n=2)
    period_theory = 2 * np.pi / omega_theory
    
    print(f"\nRayleigh Frequency (n=2): {omega_theory:.4f} rad/s")
    print(f"Rayleigh Period: {period_theory:.6f} s")
    
    total_time = 3 * period_theory
    dt = 1e-6
    
    print(f"\nRunning simulation for {total_time:.6f}s...")
    print(f"  Time step: {dt}s")
    print(f"  Number of steps: {int(total_time / dt)}")
    
    t, r = sim.run_simulation(total_time, dt=dt)
    
    if len(r) > 10:
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
    
    fig = sim.plot_state("Final Simulation State")
    fig.savefig('phase_field_results.png', dpi=150, bbox_inches='tight')
    print("\nResults saved to 'phase_field_results.png'")
    
    plt.figure(figsize=(10, 5))
    plt.plot(t, r, 'b-', linewidth=1.5)
    plt.axhline(y=R0, color='r', linestyle='--', label='Equilibrium Radius')
    plt.xlabel('Time (s)')
    plt.ylabel('Equivalent Radius (m)')
    plt.title('Droplet Oscillation - Phase Field Method')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('droplet_oscillation_phasefield.png', dpi=150, bbox_inches='tight')
    print("Oscillation plot saved to 'droplet_oscillation_phasefield.png'")
    
    plt.show()
    print("\nSimulation complete!")
    print("="*60)


def main_droplet_merging():
    print("="*60)
    print("Phase Field Simulation of Droplet Merging")
    print("(Demonstrating topological change capability)")
    print("="*60)
    
    nx, ny = 128, 128
    Lx, Ly = 0.04, 0.04
    
    print(f"\nDomain: {Lx}m x {Ly}m, Grid: {nx}x{ny}")
    
    sim = PhaseFieldDroplet(nx=nx, ny=ny, Lx=Lx, Ly=Ly,
                           rho_l=1000.0, rho_g=1.2,
                           mu_l=1e-3, mu_g=1.8e-5,
                           gamma=0.072, epsilon=Lx/100)
    
    sim.initialize_two_droplets(R1=0.006, R2=0.006, distance=0.016)
    
    fig = sim.plot_state("Initial State - Two Droplets")
    fig.savefig('merging_initial.png', dpi=150, bbox_inches='tight')
    
    total_time = 0.005
    dt = 1e-6
    
    print(f"\nRunning merging simulation for {total_time:.6f}s...")
    t, r = sim.run_simulation(total_time, dt=dt)
    
    fig = sim.plot_state("Final State - Merged Droplet")
    fig.savefig('merging_final.png', dpi=150, bbox_inches='tight')
    print("Merging results saved to 'merging_initial.png' and 'merging_final.png'")
    
    plt.figure(figsize=(10, 5))
    plt.plot(t, r, 'b-', linewidth=1.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Equivalent Radius (m)')
    plt.title('Droplet Merging - Volume Evolution')
    plt.grid(True, alpha=0.3)
    plt.savefig('merging_volume.png', dpi=150, bbox_inches='tight')
    
    plt.show()
    print("\nSimulation complete!")
    print("="*60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'merge':
        main_droplet_merging()
    else:
        main_single_droplet()

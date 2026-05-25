import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class LatticeBoltzmannDroplet:
    def __init__(self, nx=128, ny=128, tau=0.8, rho_l=1.0, rho_g=0.001,
                 gamma=0.001, R0=20):
        self.nx = nx
        self.ny = ny
        self.tau = tau
        self.omega = 1.0 / tau
        self.rho_l = rho_l
        self.rho_g = rho_g
        self.gamma = gamma
        self.R0 = R0
        
        self.q = 9
        self.c = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                           [1, 1], [-1, 1], [-1, -1], [1, -1]])
        self.w = np.array([4/9, 1/9, 1/9, 1/9, 1/9,
                           1/36, 1/36, 1/36, 1/36])
        
        self.rho = np.ones((nx, ny)) * rho_g
        self.u = np.zeros((nx, ny, 2))
        self.phi = np.zeros((nx, ny))
        
        self.f = np.zeros((nx, ny, self.q))
        self.g = np.zeros((nx, ny, self.q))
        
        self.time = 0.0
        self.radius_history = []
        self.time_history = []
        
    def initialize_single_droplet(self, x0=None, y0=None, amplitude=0.1, mode=2):
        if x0 is None:
            x0 = self.nx // 2
        if y0 is None:
            y0 = self.ny // 2
        
        X, Y = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing='ij')
        r = np.sqrt((X - x0)**2 + (Y - y0)**2)
        theta = np.arctan2(Y - y0, X - x0)
        
        R_perturbed = self.R0 * (1 + amplitude * np.cos(mode * theta))
        w = 3.0
        self.phi = 0.5 * (1 - np.tanh(2 * (r - R_perturbed) / w))
        
        self.rho = self.rho_g + (self.rho_l - self.rho_g) * self.phi
        
        for k in range(self.q):
            cu = self.u[:, :, 0] * self.c[k, 0] + self.u[:, :, 1] * self.c[k, 1]
            usq = self.u[:, :, 0]**2 + self.u[:, :, 1]**2
            self.f[:, :, k] = self.rho * self.w[k] * (1 + 3*cu + 4.5*cu**2 - 1.5*usq)
            self.g[:, :, k] = self.phi * self.w[k] * (1 + 3*cu + 4.5*cu**2 - 1.5*usq)
        
        self.time = 0.0
        self.radius_history = []
        self.time_history = []
    
    def initialize_two_droplets(self, R1=15, R2=15, distance=40):
        x1 = self.nx // 2 - distance // 2
        x2 = self.nx // 2 + distance // 2
        yc = self.ny // 2
        
        X, Y = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing='ij')
        
        r1 = np.sqrt((X - x1)**2 + (Y - yc)**2)
        r2 = np.sqrt((X - x2)**2 + (Y - yc)**2)
        
        w = 3.0
        phi1 = 0.5 * (1 - np.tanh(2 * (r1 - R1) / w))
        phi2 = 0.5 * (1 - np.tanh(2 * (r2 - R2) / w))
        
        self.phi = np.clip(phi1 + phi2, 0, 1)
        self.rho = self.rho_g + (self.rho_l - self.rho_g) * self.phi
        
        for k in range(self.q):
            cu = self.u[:, :, 0] * self.c[k, 0] + self.u[:, :, 1] * self.c[k, 1]
            usq = self.u[:, :, 0]**2 + self.u[:, :, 1]**2
            self.f[:, :, k] = self.rho * self.w[k] * (1 + 3*cu + 4.5*cu**2 - 1.5*usq)
            self.g[:, :, k] = self.phi * self.w[k] * (1 + 3*cu + 4.5*cu**2 - 1.5*usq)
        
        self.time = 0.0
        self.radius_history = []
        self.time_history = []
    
    def compute_chemical_potential(self):
        A = 0.001
        
        dfdphi = A * 4 * self.phi * (1 - self.phi) * (1 - 2 * self.phi)
        
        lapl_phi = np.zeros_like(self.phi)
        lapl_phi[1:-1, 1:-1] = (self.phi[2:, 1:-1] + self.phi[:-2, 1:-1] + 
                                self.phi[1:-1, 2:] + self.phi[1:-1, :-2] - 4 * self.phi[1:-1, 1:-1])
        
        mu_chem = dfdphi - 0.5 * lapl_phi
        return mu_chem
    
    def compute_surface_tension_force(self):
        mu_chem = self.compute_chemical_potential()
        
        grad_phi_x = np.zeros_like(self.phi)
        grad_phi_y = np.zeros_like(self.phi)
        
        grad_phi_x[1:-1, :] = (self.phi[2:, :] - self.phi[:-2, :]) / 2
        grad_phi_y[:, 1:-1] = (self.phi[:, 2:] - self.phi[:, :-2]) / 2
        
        fx = -mu_chem * grad_phi_x
        fy = -mu_chem * grad_phi_y
        
        return fx, fy
    
    def collide_and_stream(self):
        fx, fy = self.compute_surface_tension_force()
        
        rho_avg = (self.rho_l + self.rho_g) / 2
        tau_phi = 1.0
        
        for k in range(self.q):
            cx, cy = self.c[k]
            
            cu = self.u[:, :, 0] * cx + self.u[:, :, 1] * cy
            usq = self.u[:, :, 0]**2 + self.u[:, :, 1]**2
            
            feq = self.rho * self.w[k] * (1 + 3*cu + 4.5*cu**2 - 1.5*usq)
            
            force_term = self.w[k] * 3 * (fx * cx + fy * cy) * (1 - 0.5 * self.omega)
            
            self.f[:, :, k] += self.omega * (feq - self.f[:, :, k]) + force_term
            
            geq = self.phi * self.w[k] * (1 + 3*cu + 4.5*cu**2 - 1.5*usq)
            self.g[:, :, k] += (1.0 / tau_phi) * (geq - self.g[:, :, k])
        
        for k in range(self.q):
            cx, cy = self.c[k]
            
            self.f[:, :, k] = np.roll(np.roll(self.f[:, :, k], cx, axis=0), cy, axis=1)
            self.g[:, :, k] = np.roll(np.roll(self.g[:, :, k], cx, axis=0), cy, axis=1)
        
        self.apply_boundary_conditions()
        
        self.rho = np.sum(self.f, axis=2)
        self.phi = np.sum(self.g, axis=2)
        self.phi = np.clip(self.phi, 0.0, 1.0)
        
        momentum_x = np.sum(self.f * self.c[np.newaxis, np.newaxis, :, 0], axis=2)
        momentum_y = np.sum(self.f * self.c[np.newaxis, np.newaxis, :, 1], axis=2)
        
        self.u[:, :, 0] = momentum_x / self.rho
        self.u[:, :, 1] = momentum_y / self.rho
    
    def apply_boundary_conditions(self):
        for k in range(self.q):
            self.f[0, :, k] = self.f[1, :, k]
            self.f[-1, :, k] = self.f[-2, :, k]
            self.f[:, 0, k] = self.f[:, 1, k]
            self.f[:, -1, k] = self.f[:, -2, k]
            
            self.g[0, :, k] = self.g[1, :, k]
            self.g[-1, :, k] = self.g[-2, :, k]
            self.g[:, 0, k] = self.g[:, 1, k]
            self.g[:, -1, k] = self.g[:, -2, k]
    
    def compute_equivalent_radius(self):
        volume = np.sum(self.phi)
        return np.sqrt(volume / np.pi)
    
    def rayleigh_frequency(self, R0_lu=None, n=2):
        if R0_lu is None:
            R0_lu = self.compute_equivalent_radius()
        
        return np.sqrt(n * (n - 1) * (n + 2) * self.gamma / (self.rho_l * R0_lu**3))
    
    def time_step(self):
        self.collide_and_stream()
        self.time += 1
        
        if len(self.time_history) == 0 or (self.time - self.time_history[-1]) >= 10:
            R_eq = self.compute_equivalent_radius()
            self.radius_history.append(R_eq)
            self.time_history.append(self.time)
    
    def run_simulation(self, n_steps):
        for step in range(n_steps):
            self.time_step()
            
            if step % 100 == 0:
                print(f"Step {step}/{n_steps}, Time = {self.time}")
        
        return np.array(self.time_history), np.array(self.radius_history)
    
    def plot_state(self, title="LBM State"):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        ax = axes[0, 0]
        im = ax.contourf(self.phi.T, levels=20, cmap='viridis')
        ax.contour(self.phi.T, levels=[0.5], colors='red', linewidths=2)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Phase Field (phi)')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax)
        
        ax = axes[0, 1]
        im = ax.contourf(self.rho.T, levels=20, cmap='plasma')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Density')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax)
        
        ax = axes[1, 0]
        speed = np.sqrt(self.u[:, :, 0]**2 + self.u[:, :, 1]**2)
        im = ax.contourf(speed.T, levels=20, cmap='jet')
        ax.streamplot(np.arange(self.nx), np.arange(self.ny), 
                      self.u[:, :, 0].T, self.u[:, :, 1].T, 
                      color='white', density=0.5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Velocity Field')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax)
        
        ax = axes[1, 1]
        X, Y = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing='ij')
        ax.quiver(X[::4, ::4], Y[::4, ::4], 
                  self.u[::4, ::4, 0].T, self.u[::4, ::4, 1].T)
        ax.contour(self.phi.T, levels=[0.5], colors='red', linewidths=2)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Velocity Vectors')
        ax.set_aspect('equal')
        
        plt.suptitle(f"{title} (t = {self.time})", fontsize=14)
        plt.tight_layout()
        return fig


def main_single_droplet():
    print("="*60)
    print("Lattice Boltzmann + Phase Field Simulation")
    print("of Droplet Oscillation (Shan-Chen model)")
    print("="*60)
    
    nx, ny = 128, 128
    R0 = 25
    
    print(f"\nDomain: {nx} x {ny} lattice units")
    print(f"Droplet Radius: {R0} lattice units")
    
    lbm = LatticeBoltzmannDroplet(nx=nx, ny=ny, tau=0.8, 
                                   rho_l=1.0, rho_g=0.05,
                                   gamma=0.001, R0=R0)
    
    lbm.initialize_single_droplet(amplitude=0.1, mode=2)
    
    omega_theory = lbm.rayleigh_frequency(R0_lu=R0, n=2)
    period_theory = 2 * np.pi / omega_theory
    
    print(f"\nRayleigh Frequency (n=2): {omega_theory:.6f} /LU")
    print(f"Rayleigh Period: {period_theory:.1f} time steps")
    
    n_steps = int(3 * period_theory)
    
    print(f"\nRunning simulation for {n_steps} steps...")
    
    t, r = lbm.run_simulation(n_steps)
    
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
            
            print(f"\nComputed Frequency: {omega_computed:.6f} /LU")
            print(f"Computed Period: {avg_period:.1f} time steps")
            print(f"Relative Error: {abs(omega_computed - omega_theory)/omega_theory*100:.2f}%")
    
    fig = lbm.plot_state("Final State")
    fig.savefig('lbm_phasefield_results.png', dpi=150, bbox_inches='tight')
    print("\nResults saved to 'lbm_phasefield_results.png'")
    
    plt.figure(figsize=(10, 5))
    plt.plot(t, r, 'b-', linewidth=1.5)
    plt.axhline(y=R0, color='r', linestyle='--', label='Equilibrium Radius')
    plt.xlabel('Time (LBM steps)')
    plt.ylabel('Equivalent Radius (LU)')
    plt.title('Droplet Oscillation - LBM + Phase Field')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('lbm_oscillation.png', dpi=150, bbox_inches='tight')
    print("Oscillation plot saved to 'lbm_oscillation.png'")
    
    plt.show()
    print("\nSimulation complete!")
    print("="*60)


def main_droplet_merging():
    print("="*60)
    print("LBM Simulation of Droplet Merging")
    print("(Demonstrating topological change capability)")
    print("="*60)
    
    nx, ny = 128, 128
    
    print(f"\nDomain: {nx} x {ny} lattice units")
    
    lbm = LatticeBoltzmannDroplet(nx=nx, ny=ny, tau=0.8, 
                                   rho_l=1.0, rho_g=0.05,
                                   gamma=0.001)
    
    lbm.initialize_two_droplets(R1=18, R2=18, distance=50)
    
    fig = lbm.plot_state("Initial State - Two Droplets")
    fig.savefig('lbm_merging_initial.png', dpi=150, bbox_inches='tight')
    
    n_steps = 3000
    
    print(f"\nRunning merging simulation for {n_steps} steps...")
    t, r = lbm.run_simulation(n_steps)
    
    fig = lbm.plot_state("Final State - Merged Droplet")
    fig.savefig('lbm_merging_final.png', dpi=150, bbox_inches='tight')
    print("Merging results saved!")
    
    plt.figure(figsize=(10, 5))
    plt.plot(t, r, 'b-', linewidth=1.5)
    plt.xlabel('Time (LBM steps)')
    plt.ylabel('Equivalent Radius (LU)')
    plt.title('Droplet Merging - Volume Evolution')
    plt.grid(True, alpha=0.3)
    plt.savefig('lbm_merging_volume.png', dpi=150, bbox_inches='tight')
    
    plt.show()
    print("\nSimulation complete!")
    print("="*60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'merge':
        main_droplet_merging()
    else:
        main_single_droplet()

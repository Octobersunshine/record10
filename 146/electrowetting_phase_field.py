import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class ElectrowettingDroplet:
    def __init__(self, nx=128, ny=128, Lx=0.04, Ly=0.04,
                 rho_l=1000.0, rho_g=1.2, mu_l=1e-3, mu_g=1.8e-5,
                 gamma=0.072, epsilon=0.001, W=0.001,
                 epsilon_l=80.0, epsilon_g=1.0, epsilon_s=4.0):
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

        self.epsilon_l = epsilon_l
        self.epsilon_g = epsilon_g
        self.epsilon_s = epsilon_s
        self.epsilon_0 = 8.854e-12

        self.x = np.linspace(self.dx/2, Lx - self.dx/2, nx)
        self.y = np.linspace(self.dy/2, Ly - self.dy/2, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

        self.phi = np.zeros((nx, ny))
        self.u = np.zeros((nx, ny))
        self.v = np.zeros((nx, ny))
        self.p = np.zeros((nx, ny))
        self.phi_e = np.zeros((nx, ny))

        self.V = np.zeros(nx)
        self.has_electrode = np.zeros(nx, dtype=bool)
        self.electrode_regions = []

        self.theta_Y = np.pi / 4
        self.theta_A = np.pi / 3
        self.theta_R = np.pi / 6
        self.contact_line_pinning = np.zeros((nx, ny))

        self.time = 0.0
        self.interface_history = []
        self.time_history = []
        self.volume_history = []

    def add_electrode(self, x_start, x_end, y_pos=0):
        i_start = int(x_start / self.dx)
        i_end = int(x_end / self.dx)
        i_start = max(0, min(i_start, self.nx-1))
        i_end = max(0, min(i_end, self.nx-1))
        self.has_electrode[i_start:i_end+1] = True
        self.electrode_regions.append((i_start, i_end))

    def set_electrode_voltage(self, electrode_idx, voltage):
        if electrode_idx < len(self.electrode_regions):
            i_start, i_end = self.electrode_regions[electrode_idx]
            self.V[i_start:i_end+1] = voltage

    def initialize_droplet_on_surface(self, R0=0.008, x0=None, y0=None,
                                       amplitude=0.0, mode=2):
        if x0 is None:
            x0 = self.Lx / 2
        if y0 is None:
            y0 = R0 * np.cos(self.theta_Y)

        r = np.sqrt((self.X - x0)**2 + (self.Y - y0)**2)
        theta = np.arctan2(self.Y - y0, self.X - x0)

        if amplitude > 0:
            R_perturbed = R0 * (1 + amplitude * np.cos(mode * theta))
        else:
            R_perturbed = R0

        self.phi = 0.5 * (1 - np.tanh((r - R_perturbed) / (np.sqrt(2) * self.epsilon)))

        self.phi[self.Y < 0.0005] = 0.0

        self.u = np.zeros((self.nx, self.ny))
        self.v = np.zeros((self.nx, self.ny))
        self.p = np.zeros((self.nx, self.ny))
        self.phi_e = np.zeros((self.nx, self.ny))

        self.time = 0.0
        self.interface_history = []
        self.time_history = []
        self.volume_history = []

    def compute_density(self):
        return self.rho_g + (self.rho_l - self.rho_g) * self.phi

    def compute_viscosity(self):
        return self.mu_g + (self.mu_l - self.mu_g) * self.phi

    def compute_permittivity(self):
        return self.epsilon_0 * (self.epsilon_g +
                                 (self.epsilon_l - self.epsilon_g) * self.phi)

    def compute_chemical_potential(self):
        dfdphi = self.W * (4 * self.phi**3 - 6 * self.phi**2 + 2 * self.phi)
        lapl_phi = self.laplacian(self.phi)
        mu_chem = dfdphi - self.epsilon**2 * lapl_phi
        return mu_chem

    def solve_electric_field(self):
        epsilon = self.compute_permittivity()

        phi_e = np.zeros_like(self.phi_e)

        for i in range(self.nx):
            if self.has_electrode[i]:
                phi_e[i, 0] = self.V[i]

        for _ in range(200):
            phi_e_new = phi_e.copy()

            for i in range(1, self.nx-1):
                for j in range(1, self.ny-1):
                    if self.Y[i, j] < 0.001:
                        if self.has_electrode[i]:
                            phi_e_new[i, j] = self.V[i]
                        else:
                            phi_e_new[i, j] = phi_e[i, j]
                    else:
                        eps_w = 0.5 * (epsilon[i, j] + epsilon[i+1, j])
                        eps_e = 0.5 * (epsilon[i, j] + epsilon[i-1, j])
                        eps_n = 0.5 * (epsilon[i, j] + epsilon[i, j+1])
                        eps_s = 0.5 * (epsilon[i, j] + epsilon[i, j-1])

                        coeff = eps_w + eps_e + eps_n + eps_s
                        phi_e_new[i, j] = (eps_w * phi_e[i+1, j] +
                                           eps_e * phi_e[i-1, j] +
                                           eps_n * phi_e[i, j+1] +
                                           eps_s * phi_e[i, j-1]) / coeff

            phi_e_new[-1, :] = phi_e_new[-2, :]
            phi_e_new[0, :] = phi_e_new[1, :]
            phi_e_new[:, -1] = 0

            max_diff = np.max(np.abs(phi_e_new - phi_e))
            phi_e = phi_e_new

            if max_diff < 1e-6:
                break

        self.phi_e = phi_e

        E_x, E_y = self.gradient(-phi_e)
        return E_x, E_y

    def compute_electrowetting_force(self):
        E_x, E_y = self.solve_electric_field()
        epsilon = self.compute_permittivity()

        grad_phi_x, grad_phi_y = self.gradient(self.phi)

        f_ew_x = 0.5 * (E_x**2 - E_y**2) * (self.epsilon_l - self.epsilon_g) * self.epsilon_0 * grad_phi_x
        f_ew_y = E_x * E_y * (self.epsilon_l - self.epsilon_g) * self.epsilon_0 * grad_phi_y

        return f_ew_x, f_ew_y

    def compute_surface_tension_force(self):
        mu_chem = self.compute_chemical_potential()
        grad_phi_x, grad_phi_y = self.gradient(self.phi)

        fx = -mu_chem * grad_phi_x
        fy = -mu_chem * grad_phi_y

        return fx, fy

    def compute_wetting_force(self):
        grad_phi_x, grad_phi_y = self.gradient(self.phi)

        near_surface = self.Y < 0.002
        f_wall_x = np.zeros_like(self.phi)
        f_wall_y = np.zeros_like(self.phi)

        self.solve_electric_field()
        V_local = np.zeros_like(self.phi)
        for i in range(self.nx):
            if self.has_electrode[i]:
                V_local[i, :] = self.V[i]

        cos_theta_Y = np.cos(self.theta_Y)
        dV = np.max(V_local) - np.min(V_local)
        if dV > 0:
            V_normalized = V_local / dV
            cos_theta_e = cos_theta_Y + 0.5 * self.epsilon_0 * (self.epsilon_l - 1) * V_local**2 / (self.gamma * 1e-6)
            cos_theta_e = np.clip(cos_theta_e, -1, 1)
            theta_e = np.arccos(cos_theta_e)
        else:
            theta_e = self.theta_Y * np.ones_like(self.phi)

        wall_normal = np.array([0, 1])

        f_wall_y[near_surface] = -self.gamma * np.sin(theta_e[near_surface]) * grad_phi_y[near_surface]
        f_wall_x[near_surface] = -self.gamma * (np.cos(theta_e[near_surface]) - 1) * grad_phi_x[near_surface]

        return f_wall_x, f_wall_y

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

        self.phi[:, 0] = 0.0

    def solve_navier_stokes(self, dt):
        rho = self.compute_density()
        mu = self.compute_viscosity()

        fx_st, fy_st = self.compute_surface_tension_force()
        fx_ew, fy_ew = self.compute_electrowetting_force()
        fx_w, fy_w = self.compute_wetting_force()

        fx_total = fx_st + fx_ew + fx_w
        fy_total = fy_st + fy_ew + fy_w

        u_star = self.u.copy()
        v_star = self.v.copy()

        lapl_u = self.laplacian(self.u)
        lapl_v = self.laplacian(self.v)

        u_star += dt * (-self.u * self.gradient(self.u)[0] - self.v * self.gradient(self.u)[1] +
                        (mu / rho) * lapl_u + fx_total / rho)
        v_star += dt * (-self.u * self.gradient(self.v)[0] - self.v * self.gradient(self.v)[1] +
                        (mu / rho) * lapl_v + fy_total / rho)

        div_star = self.divergence(u_star, v_star)

        for _ in range(30):
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

    def compute_contact_angle(self):
        j_surface = 5
        interface_x = []

        for i in range(2, self.nx-2):
            if (self.phi[i, j_surface] - 0.5) * (self.phi[i+1, j_surface] - 0.5) < 0:
                alpha = (0.5 - self.phi[i, j_surface]) / (self.phi[i+1, j_surface] - self.phi[i, j_surface])
                x_intersect = self.x[i] + alpha * self.dx
                interface_x.append(x_intersect)

        if len(interface_x) >= 2:
            x1, x2 = interface_x[0], interface_x[-1]
            dx = x2 - x1

            j1 = min(j_surface + 10, self.ny - 1)
            phi_slice = self.phi[:, j1]
            interface_x2 = []
            for i in range(2, self.nx-2):
                if (phi_slice[i] - 0.5) * (phi_slice[i+1] - 0.5) < 0:
                    alpha = (0.5 - phi_slice[i]) / (phi_slice[i+1] - phi_slice[i])
                    x_intersect = self.x[i] + alpha * self.dx
                    interface_x2.append(x_intersect)

            if len(interface_x2) >= 2:
                dy = (j1 - j_surface) * self.dy
                dx2 = np.mean(interface_x2) - np.mean(interface_x)
                theta = np.arctan2(dy, abs(dx2))
                return theta

        return self.theta_Y

    def compute_droplet_volume(self):
        return np.sum(self.phi) * self.dx * self.dy

    def compute_centroid(self):
        volume = self.compute_droplet_volume()
        cx = np.sum(self.X * self.phi) * self.dx * self.dy / volume
        cy = np.sum(self.Y * self.phi) * self.dx * self.dy / volume
        return cx, cy

    def time_step(self, dt=1e-6):
        self.solve_navier_stokes(dt)
        self.solve_cahn_hilliard(dt, M=1e-10)
        self.time += dt

        if len(self.time_history) == 0 or (self.time - self.time_history[-1]) >= dt * 5:
            cx, cy = self.compute_centroid()
            self.volume_history.append((cx, cy, self.compute_droplet_volume()))
            self.time_history.append(self.time)

    def run_simulation(self, total_time, dt=1e-6, voltage_schedule=None):
        n_steps = int(total_time / dt)

        for step in range(n_steps):
            if voltage_schedule is not None:
                current_time = step * dt
                voltage_schedule(current_time, self)

            self.time_step(dt)

            if step % 100 == 0:
                cx, cy = self.compute_centroid()
                print(f"Step {step}/{n_steps}, Time = {self.time:.6f}s, "
                      f"Centroid = ({cx:.4f}, {cy:.4f})")

        return np.array(self.time_history), self.volume_history

    def plot_state(self, title="Electrowetting State"):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        ax = axes[0, 0]
        im = ax.contourf(self.X, self.Y, self.phi, levels=20, cmap='viridis')
        ax.contour(self.X, self.Y, self.phi, levels=[0.5], colors='red', linewidths=2)
        ax.plot(self.x, np.zeros_like(self.x), 'k-', linewidth=3)
        for i, (i_start, i_end) in enumerate(self.electrode_regions):
            color = 'r' if self.V[i_start] > 0 else 'k'
            ax.plot([self.x[i_start], self.x[i_end]], [0, 0], color=color, linewidth=5)
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

        ax = axes[0, 2]
        speed = np.sqrt(self.u**2 + self.v**2)
        im = ax.contourf(self.X, self.Y, speed, levels=20, cmap='jet')
        ax.streamplot(self.x, self.y, self.u.T, self.v.T, color='white', density=0.5)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title('Velocity Field')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax)

        ax = axes[1, 0]
        im = ax.contourf(self.X, self.Y, self.phi_e, levels=20, cmap='coolwarm')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title('Electric Potential (V)')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax)

        ax = axes[1, 1]
        E_x, E_y = self.gradient(-self.phi_e)
        E_mag = np.sqrt(E_x**2 + E_y**2)
        im = ax.contourf(self.X, self.Y, E_mag, levels=20, cmap='hot')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title('Electric Field Magnitude')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax)

        ax = axes[1, 2]
        im = ax.contourf(self.X, self.Y, self.p, levels=20, cmap='coolwarm')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title('Pressure Field')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax)

        plt.suptitle(f"{title} (t = {self.time:.6f}s)", fontsize=14)
        plt.tight_layout()
        return fig


def demo_droplet_oscillation_with_electrowetting():
    print("="*70)
    print("Electrowetting Simulation - Droplet Oscillation")
    print("(Cahn-Hilliard Phase Field + Electric Field)")
    print("="*70)

    nx, ny = 128, 64
    Lx, Ly = 0.04, 0.02

    print(f"\nDomain: {Lx}m x {Ly}m, Grid: {nx}x{ny}")

    sim = ElectrowettingDroplet(nx=nx, ny=ny, Lx=Lx, Ly=Ly,
                                rho_l=1000.0, rho_g=1.2,
                                mu_l=1e-3, mu_g=1.8e-5,
                                gamma=0.072, epsilon=Lx/200)

    sim.add_electrode(0.0, Lx)
    sim.set_electrode_voltage(0, 0.0)

    sim.initialize_droplet_on_surface(R0=0.006, x0=Lx/2, amplitude=0.1)

    print(f"\nInitial contact angle (Young): {sim.theta_Y * 180/np.pi:.1f} degrees")
    print(f"Advancing angle: {sim.theta_A * 180/np.pi:.1f} degrees")
    print(f"Receding angle: {sim.theta_R * 180/np.pi:.1f} degrees")

    total_time = 0.01
    dt = 1e-6

    print(f"\nRunning simulation for {total_time:.6f}s...")
    t, hist = sim.run_simulation(total_time, dt=dt)

    times = np.array(t)
    cx = np.array([h[0] for h in hist])
    cy = np.array([h[1] for h in hist])
    volume = np.array([h[2] for h in hist])

    fig = sim.plot_state("Final State - Electrowetting")
    fig.savefig('electrowetting_results.png', dpi=150, bbox_inches='tight')
    print("\nResults saved to 'electrowetting_results.png'")

    fig2, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(times, cx, 'b-', linewidth=1.5)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Centroid X (m)')
    axes[0].set_title('Droplet X Position')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times, cy, 'r-', linewidth=1.5)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Centroid Y (m)')
    axes[1].set_title('Droplet Y Position')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('droplet_position.png', dpi=150, bbox_inches='tight')
    print("Position plot saved to 'droplet_position.png'")

    plt.show()
    print("\nSimulation complete!")
    print("="*70)


def demo_electrowetting_actuation():
    print("="*70)
    print("Electrowetting Droplet Actuation - Spatial Fluid Management")
    print("For Space Applications: Fuel Tanks, Thermal Control Systems")
    print("="*70)

    nx, ny = 192, 64
    Lx, Ly = 0.06, 0.02

    print(f"\nDomain: {Lx}m x {Ly}m, Grid: {nx}x{ny}")

    sim = ElectrowettingDroplet(nx=nx, ny=ny, Lx=Lx, Ly=Ly,
                                rho_l=1000.0, rho_g=1.2,
                                mu_l=1e-3, mu_g=1.8e-5,
                                gamma=0.072, epsilon=Lx/300)

    electrode_width = Lx / 6
    for i in range(6):
        sim.add_electrode(i * electrode_width, (i+1) * electrode_width)

    print(f"\nNumber of electrodes: {len(sim.electrode_regions)}")
    print(f"Electrode width: {electrode_width*1000:.1f} mm")

    sim.initialize_droplet_on_surface(R0=0.005, x0=electrode_width*1.5)

    print(f"\nInitial droplet centroid: {electrode_width*1.5:.4f} m")

    def voltage_schedule(t, sim_obj):
        if t < 0.005:
            sim_obj.set_electrode_voltage(1, 20.0)
            sim_obj.set_electrode_voltage(2, 0.0)
        elif t < 0.01:
            sim_obj.set_electrode_voltage(1, 0.0)
            sim_obj.set_electrode_voltage(2, 20.0)
            sim_obj.set_electrode_voltage(3, 0.0)
        elif t < 0.015:
            sim_obj.set_electrode_voltage(2, 0.0)
            sim_obj.set_electrode_voltage(3, 20.0)
            sim_obj.set_electrode_voltage(4, 0.0)
        else:
            sim_obj.set_electrode_voltage(3, 0.0)
            sim_obj.set_electrode_voltage(4, 20.0)

    total_time = 0.02
    dt = 5e-7

    print(f"\nRunning actuation simulation for {total_time:.6f}s...")
    t, hist = sim.run_simulation(total_time, dt=dt, voltage_schedule=voltage_schedule)

    times = np.array(t)
    cx = np.array([h[0] for h in hist])

    fig = sim.plot_state("Final State - Electrowetting Actuation")
    fig.savefig('electrowetting_actuation.png', dpi=150, bbox_inches='tight')
    print("\nActuation results saved to 'electrowetting_actuation.png'")

    plt.figure(figsize=(10, 5))
    plt.plot(times, cx, 'b-', linewidth=1.5)
    for i in range(7):
        plt.axvline(x=i*0.005, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Centroid X (m)')
    plt.title('Droplet Position - Electrowetting Actuation')
    plt.grid(True, alpha=0.3)
    plt.savefig('actuation_position.png', dpi=150, bbox_inches='tight')
    print("Position plot saved to 'actuation_position.png'")

    plt.show()
    print("\nSimulation complete!")
    print("="*70)


def demo_contact_angle_hysteresis():
    print("="*70)
    print("Contact Angle Hysteresis Demonstration")
    print("Advancing/Receding Angles with Pinning Effect")
    print("="*70)

    nx, ny = 128, 64
    Lx, Ly = 0.04, 0.02

    sim = ElectrowettingDroplet(nx=nx, ny=ny, Lx=Lx, Ly=Ly,
                                rho_l=1000.0, rho_g=1.2,
                                mu_l=1e-3, mu_g=1.8e-5,
                                gamma=0.072, epsilon=Lx/200)

    sim.add_electrode(0.0, Lx)

    sim.theta_Y = np.pi / 4
    sim.theta_A = 70 * np.pi / 180
    sim.theta_R = 40 * np.pi / 180

    print(f"\nYoung contact angle: {sim.theta_Y * 180/np.pi:.1f} degrees")
    print(f"Advancing angle (theta_A): {sim.theta_A * 180/np.pi:.1f} degrees")
    print(f"Receding angle (theta_R): {sim.theta_R * 180/np.pi:.1f} degrees")
    print(f"Contact angle hysteresis: {(sim.theta_A - sim.theta_R)*180/np.pi:.1f} degrees")

    sim.initialize_droplet_on_surface(R0=0.007, x0=Lx/2)

    def voltage_schedule(t, sim_obj):
        if t < 0.01:
            V = 50 * t / 0.01
        else:
            V = 50 * (1 - (t - 0.01) / 0.01)
        sim_obj.set_electrode_voltage(0, V)

    total_time = 0.02
    dt = 5e-7

    print(f"\nRunning hysteresis simulation for {total_time:.6f}s...")
    t, hist = sim.run_simulation(total_time, dt=dt, voltage_schedule=voltage_schedule)

    times = np.array(t)
    cx = np.array([h[0] for h in hist])
    cy = np.array([h[1] for h in hist])
    volume = np.array([h[2] for h in hist])

    fig = sim.plot_state("Final State - Contact Angle Hysteresis")
    fig.savefig('hysteresis_results.png', dpi=150, bbox_inches='tight')
    print("\nResults saved to 'hysteresis_results.png'")

    plt.figure(figsize=(10, 5))
    plt.plot(times, volume / volume[0], 'b-', linewidth=1.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Volume')
    plt.title('Droplet Volume - Contact Angle Hysteresis')
    plt.grid(True, alpha=0.3)
    plt.savefig('hysteresis_volume.png', dpi=150, bbox_inches='tight')

    plt.show()
    print("\nSimulation complete!")
    print("="*70)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == 'actuate':
            demo_electrowetting_actuation()
        elif sys.argv[1] == 'hysteresis':
            demo_contact_angle_hysteresis()
        else:
            demo_droplet_oscillation_with_electrowetting()
    else:
        demo_droplet_oscillation_with_electrowetting()

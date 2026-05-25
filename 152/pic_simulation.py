import numpy as np
import matplotlib.pyplot as plt
import time

class Particle:
    def __init__(self, x, v, q, m):
        self.x = x
        self.v = v
        self.q = q
        self.m = m

class PIC1D:
    def __init__(self, N_particles=10000, N_grid=256, L=100.0, dt=0.01, t_max=20.0,
                 B0=0.0, q_over_m=-1.0, n0=1.0, laser_amp=0.0):
        self.N_particles = N_particles
        self.N_grid = N_grid
        self.L = L
        self.dt = dt
        self.t_max = t_max
        self.n_steps = int(t_max / dt)
        
        self.B0 = B0
        self.q_over_m = q_over_m
        self.n0 = n0
        self.laser_amp = laser_amp
        
        self.dx = L / N_grid
        self.x_grid = np.linspace(0, L, N_grid, endpoint=False)
        
        self.particles = []
        self.E_field = np.zeros(N_grid)
        self.B_field = np.zeros(N_grid) + B0
        self.charge_density = np.zeros(N_grid)
        self.current_density = np.zeros(N_grid)
        
        self.phi = np.zeros(N_grid)
        
        self.k = 2 * np.pi * np.fft.fftfreq(N_grid, d=self.dx)
        self.k_sq = self.k**2
        self.k_sq_safe = np.where(self.k_sq == 0, 1e-10, self.k_sq)
        
        self.history = {
            't': [],
            'E_max': [],
            'n_mean': [],
            'kinetic_energy': [],
            'potential_energy': []
        }
    
    def initialize_particles(self, temperature=0.01, perturbation=True):
        np.random.seed(42)
        
        for i in range(self.N_particles):
            x = np.random.uniform(0, self.L)
            v = np.random.normal(0, np.sqrt(temperature), 2)
            
            if perturbation:
                x += 0.1 * self.dx * np.sin(2 * np.pi * x / self.L * 4)
            
            self.particles.append(Particle(x, v, self.q_over_m, 1.0))
        
        self._charge_deposition()
        self._solve_poisson()
    
    def _charge_deposition(self):
        self.charge_density = np.zeros(self.N_grid)
        
        for p in self.particles:
            x0 = p.x / self.dx
            i = int(np.floor(x0)) % self.N_grid
            frac = x0 - i
            
            self.charge_density[i] += p.q * (1 - frac)
            self.charge_density[(i + 1) % self.N_grid] += p.q * frac
        
        self.charge_density = self.charge_density / self.dx + self.n0
    
    def _solve_poisson(self):
        rho_hat = np.fft.fft(self.charge_density - self.n0)
        phi_hat = -rho_hat / self.k_sq_safe
        self.phi = np.fft.ifft(phi_hat).real
        
        E_hat = -1j * self.k * phi_hat
        self.E_field = np.fft.ifft(E_hat).real
    
    def _interpolate_field(self, x, field):
        x0 = x / self.dx
        i = int(np.floor(x0)) % self.N_grid
        frac = x0 - i
        
        return field[i] * (1 - frac) + field[(i + 1) % self.N_grid] * frac
    
    def _update_particles(self):
        omega_c = self.q_over_m * self.B0
        
        for p in self.particles:
            E_local = self._interpolate_field(p.x, self.E_field)
            
            vx, vy = p.v
            
            if self.B0 != 0:
                vx_new = vx + self.q_over_m * E_local * self.dt + omega_c * vy * self.dt
                vy_new = vy - omega_c * vx * self.dt
                
                theta = omega_c * self.dt
                c = np.cos(theta)
                s = np.sin(theta)
                vx_new = vx * c + vy * s + self.q_over_m * E_local * self.dt
                vy_new = -vx * s + vy * c
            else:
                vx_new = vx + self.q_over_m * E_local * self.dt
                vy_new = vy
            
            p.v = np.array([vx_new, vy_new])
            
            p.x += p.v[0] * self.dt
            p.x = p.x % self.L
    
    def _laser_driving(self, t):
        if self.laser_amp > 0:
            laser_profile = self.laser_amp * np.sin(2 * np.pi * t / 2.0)
            self.E_field += laser_profile * np.exp(-(self.x_grid - self.L/2)**2 / 10.0)
    
    def step(self, t):
        self._update_particles()
        self._charge_deposition()
        self._solve_poisson()
        self._laser_driving(t)
    
    def get_kinetic_energy(self):
        ke = 0.0
        for p in self.particles:
            ke += 0.5 * np.sum(p.v**2)
        return ke / self.N_particles
    
    def get_potential_energy(self):
        return 0.5 * np.sum(self.E_field**2) * self.dx
    
    def get_density_profile(self):
        return self.charge_density
    
    def run_simulation(self, plot_interval=10):
        print("=" * 60)
        print("PIC 粒子模拟 - 激光等离子体相互作用")
        print("=" * 60)
        print(f"\n粒子数: {self.N_particles}")
        print(f"网格数: {self.N_grid}")
        print(f"模拟时间: {self.t_max}, 时间步长: {self.dt}")
        print(f"磁场 B0 = {self.B0}")
        print(f"激光振幅: {self.laser_amp}")
        
        self.initialize_particles()
        
        print(f"\n初始动能: {self.get_kinetic_energy():.6f}")
        print(f"初始势能: {self.get_potential_energy():.6f}")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('PIC 粒子模拟 - 1D', fontsize=14, fontweight='bold')
        
        dens_line, = axes[0, 0].plot(self.x_grid, self.charge_density, 'b-', linewidth=2)
        axes[0, 0].set_title('电荷密度', fontsize=12)
        axes[0, 0].set_xlabel('x')
        axes[0, 0].grid(True, alpha=0.3)
        
        E_line, = axes[0, 1].plot(self.x_grid, self.E_field, 'r-', linewidth=2)
        axes[0, 1].set_title('电场 E', fontsize=12)
        axes[0, 1].set_xlabel('x')
        axes[0, 1].grid(True, alpha=0.3)
        
        positions = np.array([p.x for p in self.particles[::10]])
        velocities = np.array([p.v[0] for p in self.particles[::10]])
        phase_scat = axes[1, 0].scatter(positions, velocities, s=1, alpha=0.5)
        axes[1, 0].set_title('相空间 (x, vx)', fontsize=12)
        axes[1, 0].set_xlabel('x')
        axes[1, 0].set_ylabel('vx')
        axes[1, 0].grid(True, alpha=0.3)
        
        t_list = []
        ke_list = []
        pe_list = []
        
        ke_line, = axes[1, 1].plot([], [], 'b-', label='动能')
        pe_line, = axes[1, 1].plot([], [], 'r-', label='势能')
        total_line, = axes[1, 1].plot([], [], 'k--', label='总能量')
        axes[1, 1].set_title('能量演化', fontsize=12)
        axes[1, 1].set_xlabel('t')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        print("\n开始模拟...")
        start_time = time.time()
        
        for step in range(self.n_steps):
            t = step * self.dt
            self.step(t)
            
            if step % plot_interval == 0:
                ke = self.get_kinetic_energy()
                pe = self.get_potential_energy()
                
                t_list.append(t)
                ke_list.append(ke)
                pe_list.append(pe)
                
                dens_line.set_ydata(self.charge_density)
                axes[0, 0].set_ylim(0.5 * np.min(self.charge_density), 1.5 * np.max(self.charge_density))
                
                E_line.set_ydata(self.E_field)
                axes[0, 1].set_ylim(1.1 * np.min(self.E_field), 1.1 * np.max(self.E_field))
                
                positions = np.array([p.x for p in self.particles[::10]])
                velocities = np.array([p.v[0] for p in self.particles[::10]])
                phase_scat.set_offsets(np.column_stack((positions, velocities)))
                axes[1, 0].set_ylim(1.1 * np.min(velocities), 1.1 * np.max(velocities))
                
                ke_line.set_data(t_list, ke_list)
                pe_line.set_data(t_list, pe_list)
                total_line.set_data(t_list, np.array(ke_list) + np.array(pe_list))
                axes[1, 1].set_xlim(0, t + self.dt * plot_interval)
                axes[1, 1].set_ylim(0, 1.2 * (max(ke_list) + max(pe_list)))
                
                fig.suptitle(f'PIC 模拟 - t = {t:.2f}', fontsize=14, fontweight='bold')
                
                plt.pause(0.01)
                
                if step % 50 == 0:
                    elapsed = time.time() - start_time
                    print(f"步骤 {step}/{self.n_steps} | t={t:.2f} | "
                          f"KE={ke:.4f} | PE={pe:.4f} | 耗时={elapsed:.1f}s")
        
        total_time = time.time() - start_time
        print(f"\n模拟完成! 总耗时: {total_time:.2f}秒")
        
        plt.savefig('pic_simulation.png', dpi=150, bbox_inches='tight')
        print("\n结果已保存至: pic_simulation.png")
        
        plt.show()

class ComparisonValidator:
    def __init__(self, L=50.0, N_grid=128, t_max=10.0, dt=0.01):
        self.L = L
        self.N_grid = N_grid
        self.t_max = t_max
        self.dt = dt
    
    def run_fluid_simulation(self, B0=0.0):
        from zakharov_1d import ZakharovSolver1D
        
        solver = ZakharovSolver1D(N=self.N_grid, L=self.L, dt=self.dt, t_max=self.t_max,
                                  dealiasing='2/3')
        solver.set_initial_condition(amp=2.0, width=8.0, noise_amp=0.02)
        
        t_list = [0]
        density_list = [solver.n.copy()]
        intensity_list = [np.abs(solver.E)**2]
        
        for step in range(solver.n_steps):
            solver.step()
            
            if step % 10 == 0:
                t_list.append(step * self.dt)
                density_list.append(solver.n.copy())
                intensity_list.append(np.abs(solver.E)**2)
        
        return t_list, density_list, intensity_list
    
    def run_pic_simulation(self, B0=0.0):
        pic = PIC1D(N_particles=20000, N_grid=self.N_grid, L=self.L, 
                    dt=self.dt * 5, t_max=self.t_max, B0=B0)
        pic.initialize_particles(temperature=0.001)
        
        t_list = [0]
        density_list = [pic.charge_density.copy()]
        
        for step in range(pic.n_steps):
            pic.step(step * pic.dt)
            
            if step % 2 == 0:
                t_list.append(step * pic.dt)
                density_list.append(pic.charge_density.copy())
        
        return t_list, density_list
    
    def compare_models(self):
        print("=" * 70)
        print("流体模型 vs PIC 粒子模拟 对比验证")
        print("=" * 70)
        
        print("\n运行流体模拟 (Zakharov方程)...")
        t_fluid, dens_fluid, intensity_fluid = self.run_fluid_simulation()
        
        print("\n运行PIC粒子模拟...")
        t_pic, dens_pic = self.run_pic_simulation()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('流体模型 vs PIC 粒子模拟 对比', fontsize=16, fontweight='bold')
        
        x_fluid = np.linspace(-self.L/2, self.L/2, self.N_grid)
        x_pic = np.linspace(0, self.L, self.N_grid) - self.L/2
        
        im1 = axes[0, 0].pcolormesh(t_fluid, x_fluid, np.array(dens_fluid).T, cmap='RdBu_r')
        axes[0, 0].set_title('流体模型 - 密度扰动演化', fontsize=12)
        axes[0, 0].set_xlabel('t')
        axes[0, 0].set_ylabel('x')
        plt.colorbar(im1, ax=axes[0, 0])
        
        im2 = axes[0, 1].pcolormesh(t_pic, x_pic, np.array(dens_pic).T - np.mean(dens_pic[0]), cmap='RdBu_r')
        axes[0, 1].set_title('PIC模拟 - 密度扰动演化', fontsize=12)
        axes[0, 1].set_xlabel('t')
        axes[0, 1].set_ylabel('x')
        plt.colorbar(im2, ax=axes[0, 1])
        
        axes[1, 0].plot(x_fluid, dens_fluid[0], 'b--', label='t=0')
        axes[1, 0].plot(x_fluid, dens_fluid[len(dens_fluid)//2], 'g-', label='t=t_max/2')
        axes[1, 0].plot(x_fluid, dens_fluid[-1], 'r-', label='t=t_max')
        axes[1, 0].set_title('流体模型 - 密度剖面', fontsize=12)
        axes[1, 0].set_xlabel('x')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(x_pic, dens_pic[0] - np.mean(dens_pic[0]), 'b--', label='t=0')
        axes[1, 1].plot(x_pic, dens_pic[len(dens_pic)//2] - np.mean(dens_pic[0]), 'g-', label='t=t_max/2')
        axes[1, 1].plot(x_pic, dens_pic[-1] - np.mean(dens_pic[0]), 'r-', label='t=t_max')
        axes[1, 1].set_title('PIC模拟 - 密度剖面', fontsize=12)
        axes[1, 1].set_xlabel('x')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('comparison_validation.png', dpi=150, bbox_inches='tight')
        print("\n对比结果已保存至: comparison_validation.png")
        
        plt.show()

def run_pic_demo():
    pic = PIC1D(N_particles=15000, N_grid=128, L=80.0, dt=0.02, t_max=15.0,
                B0=0.5, laser_amp=0.1)
    pic.run_simulation(plot_interval=5)

def run_comparison():
    validator = ComparisonValidator(L=60.0, N_grid=64, t_max=8.0, dt=0.02)
    validator.compare_models()

if __name__ == "__main__":
    run_pic_demo()

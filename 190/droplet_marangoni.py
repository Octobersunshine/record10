import numpy as np
from scipy.integrate import solve_ivp, odeint
from scipy.interpolate import interp1d, griddata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class DropletWithMarangoni:
    def __init__(self, R0=500e-6, theta0=np.pi/4, T=298.15,
                 N_particles=1000, particle_diameter=100e-9,
                 marangoni_strength=1.0, regularization_method='cutoff'):
        
        self.R0 = R0
        self.theta0 = theta0
        self.T = T
        
        self.D = 2.5e-5
        self.M = 0.01801528
        self.rho = 1000.0
        self.R_g = 8.314
        self.mu = 8.9e-4
        
        self.gamma_T = -0.00015
        
        self.Psat = self._antoine_water(T)
        self.c_sat = (self.Psat * self.M) / (self.R_g * self.T)
        self.c_inf = 0.0
        
        self.regularization_method = regularization_method
        self.theta_min = np.radians(3.0)
        self.delta_m = 1e-9
        
        self.N_particles = N_particles
        self.particle_diameter = particle_diameter
        self.particle_radius = particle_diameter / 2
        self.marangoni_strength = marangoni_strength
        
        self.particles = None
        self.deposited_particles = []
        self.history = []
        
    def _antoine_water(self, T):
        A = 8.07131
        B = 1730.63
        C = 233.426
        T_celsius = T - 273.15
        P_mmHg = 10 ** (A - B / (T_celsius + C))
        return P_mmHg * 133.322
    
    def _g_theta(self, theta):
        theta_clipped = np.maximum(theta, self.theta_min)
        return (0.27 * theta_clipped**2 + 1.30) / (np.sin(theta_clipped) * (1 + np.cos(theta_clipped))**2)
    
    def _volume_from_R_theta(self, R, theta):
        return (np.pi * R**3 / 3) * (1 - np.cos(theta))**2 * (2 + np.cos(theta))
    
    def _height_from_R_theta(self, R, theta):
        return R * (1 - np.cos(theta))
    
    def _dVdt(self, R, theta):
        g = self._g_theta(theta)
        return -np.pi * self.D * (self.c_sat - self.c_inf) * R * g
    
    def initialize_particles(self, distribution='uniform'):
        R = self.R0
        theta = self.theta0
        h = self._height_from_R_theta(R, theta)
        
        self.particles = np.zeros((self.N_particles, 4))
        
        for i in range(self.N_particles):
            if distribution == 'uniform':
                while True:
                    r = np.random.uniform(0, R)
                    phi = np.random.uniform(0, 2 * np.pi)
                    z_max = h * (1 - (r / R)**2)**0.5
                    z = np.random.uniform(0, z_max)
                    
                    if z <= z_max:
                        break
            elif distribution == 'center':
                r = np.random.beta(2, 5) * R
                phi = np.random.uniform(0, 2 * np.pi)
                z_max = h * (1 - (r / R)**2)**0.5
                z = np.random.uniform(0, z_max * 0.5)
            elif distribution == 'edge':
                r = np.random.beta(5, 2) * R
                phi = np.random.uniform(0, 2 * np.pi)
                z_max = h * (1 - (r / R)**2)**0.5
                z = np.random.uniform(0, z_max)
            
            self.particles[i] = [r, phi, z, 0]
        
        return self.particles
    
    def temperature_profile(self, r, R, theta):
        h = self._height_from_R_theta(R, theta)
        h_r = h * np.sqrt(1 - (r / R)**2)
        
        delta_T_surface = 5.0
        
        if R > 1e-10:
            T_surface = self.T - delta_T_surface * (r / R)**2
        else:
            T_surface = self.T
        
        T_bulk = self.T - delta_T_surface * 0.3 * (r / (R + 1e-10))**2
        
        return T_bulk, T_surface
    
    def velocity_capillary(self, r, z, R, theta, dRdt):
        if R < 1e-10 or r > R:
            return np.array([0.0, 0.0])
        
        h = self._height_from_R_theta(R, theta)
        h_r = h * np.sqrt(1 - (r / R)**2)
        
        if h_r < 1e-10:
            return np.array([0.0, 0.0])
        
        v_r = - (dRdt / R) * r * (1 - z / h_r)
        v_z = - (dRdt / R) * z * 0.5
        
        return np.array([v_r, v_z])
    
    def velocity_marangoni(self, r, z, R, theta):
        if R < 1e-10 or r > R:
            return np.array([0.0, 0.0])
        
        h = self._height_from_R_theta(R, theta)
        h_r = h * np.sqrt(1 - (r / R)**2)
        
        if h_r < 1e-10:
            return np.array([0.0, 0.0])
        
        _, T_surface = self.temperature_profile(r, R, theta)
        dTdr = -2 * (T_surface - self.T) * r / (R**2 + 1e-20)
        
        dgamma_dr = self.gamma_T * dTdr
        
        u_surface = - dgamma_dr * h_r / (2 * self.mu)
        
        profile = 2 * (z / h_r) - (z / h_r)**2
        
        v_r = self.marangoni_strength * u_surface * profile
        v_z = - (v_r / (r + 1e-10)) * z * 0.1
        
        return np.array([v_r, v_z])
    
    def velocity_diffusiophoresis(self, r, z, R, theta):
        if R < 1e-10 or r > R:
            return np.array([0.0, 0.0])
        
        h = self._height_from_R_theta(R, theta)
        h_r = h * np.sqrt(1 - (r / R)**2)
        
        if h_r < 1e-10:
            return np.array([0.0, 0.0])
        
        g = self._g_theta(theta)
        J0 = self.D * (self.c_sat - self.c_inf) * g / R
        
        J_r = J0 * r / R
        
        D_diff = 1e-10
        v_r = D_diff * J_r / (self.c_sat + 1e-10)
        
        return np.array([v_r * 0.1, 0.0])
    
    def total_velocity(self, r, z, R, theta, dRdt, mode='capillary'):
        v_cap = self.velocity_capillary(r, z, R, theta, dRdt)
        
        if mode == 'capillary':
            return v_cap
        elif mode == 'marangoni':
            v_mar = self.velocity_marangoni(r, z, R, theta)
            return v_cap + v_mar
        elif mode == 'diffusiophoresis':
            v_diff = self.velocity_diffusiophoresis(r, z, R, theta)
            return v_cap + v_diff
        else:
            v_mar = self.velocity_marangoni(r, z, R, theta)
            v_diff = self.velocity_diffusiophoresis(r, z, R, theta)
            return v_cap + v_mar + v_diff
    
    def simulate_evaporation_with_particles(self, total_time=None, flow_mode='marangoni',
                                            dt_output=None):
        R = self.R0
        theta = self.theta0
        
        V0 = self._volume_from_R_theta(R, theta)
        dVdt0 = self._dVdt(R, theta)
        if total_time is None:
            total_time = abs(V0 / dVdt0) * 1.2
        
        if dt_output is None:
            dt_output = total_time / 100
        
        self.initialize_particles()
        self.deposited_particles = []
        self.history = []
        
        t = 0.0
        active_mask = np.ones(self.N_particles, dtype=bool)
        
        while t < total_time and R > 1e-9 and theta > self.theta_min:
            dVdt = self._dVdt(R, theta)
            
            dVdR = np.pi * R**2 * (1 - np.cos(theta))**2 * (2 + np.cos(theta))
            dRdt = dVdt / dVdR if abs(dVdR) > 1e-20 else 0.0
            
            dt = min(dt_output, total_time - t)
            
            active_indices = np.where(active_mask)[0]
            
            for idx in active_indices:
                if not active_mask[idx]:
                    continue
                
                r, phi, z, _ = self.particles[idx]
                
                if r >= R - self.particle_radius:
                    self.deposited_particles.append([t, r, phi, z, R, theta])
                    active_mask[idx] = False
                    continue
                
                v_r, v_z = self.total_velocity(r, z, R, theta, dRdt, mode=flow_mode)
                
                D_brownian = self.R_g * self.T / (6 * np.pi * self.mu * self.particle_radius)
                noise = np.sqrt(2 * D_brownian * dt) * np.random.randn(2)
                
                dr = v_r * dt + noise[0]
                dz = v_z * dt + noise[1]
                
                r_new = max(0, r + dr)
                z_new = max(0, z + dz)
                
                ratio = r_new / (R + 1e-10)
                if ratio >= 1.0:
                    ratio = 0.999
                
                h_r = self._height_from_R_theta(R, theta) * np.sqrt(1 - ratio**2)
                if z_new > h_r:
                    z_new = h_r
                
                if r_new >= R - self.particle_radius:
                    self.deposited_particles.append([t, r_new, phi, z_new, R, theta])
                    active_mask[idx] = False
                else:
                    self.particles[idx, 0] = r_new
                    self.particles[idx, 2] = z_new
            
            R = R + dRdt * dt
            
            V = self._volume_from_R_theta(R, theta)
            V_new = V + dVdt * dt
            if V_new > 0:
                theta = np.arccos(1 - np.clip((3 * V_new / (np.pi * R**3) - 2) / (-1), -1, 1))
            else:
                theta = self.theta_min
            
            self.history.append({
                't': t,
                'R': R,
                'theta': theta,
                'V': self._volume_from_R_theta(R, theta),
                'n_active': np.sum(active_mask),
                'n_deposited': len(self.deposited_particles)
            })
            
            t += dt
        
        remaining = np.where(active_mask)[0]
        for idx in remaining:
            r, phi, z, _ = self.particles[idx]
            self.deposited_particles.append([t, r, phi, z, R, theta])
            active_mask[idx] = False
        
        return {
            'history': self.history,
            'deposited_particles': np.array(self.deposited_particles),
            'total_time': t
        }
    
    def plot_deposition_pattern(self, results, filename=None, style='coffee_ring'):
        deposited = results['deposited_particles']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('液滴蒸发颗粒沉积模拟 - DNA芯片图案', fontsize=14, fontweight='bold')
        
        ax1 = axes[0, 0]
        r_dep = deposited[:, 1]
        phi_dep = deposited[:, 2]
        x_dep = r_dep * np.cos(phi_dep)
        y_dep = r_dep * np.sin(phi_dep)
        
        if style == 'coffee_ring':
            ax1.scatter(x_dep * 1e6, y_dep * 1e6, s=5, alpha=0.6, c='blue', edgecolors='none')
        elif style == 'density':
            h, xedges, yedges = np.histogram2d(x_dep * 1e6, y_dep * 1e6, bins=50)
            im = ax1.imshow(h.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                           cmap='viridis', interpolation='gaussian')
            plt.colorbar(im, ax=ax1, label='颗粒数密度')
        
        R_final = deposited[-1, 4]
        circle = Circle((0, 0), R_final * 1e6, fill=False, color='red', linestyle='--', linewidth=2, label='最终接触线')
        ax1.add_patch(circle)
        
        ax1.set_xlabel('x (μm)')
        ax1.set_ylabel('y (μm)')
        ax1.set_title('颗粒沉积图案 (顶视图)')
        ax1.legend()
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        times = deposited[:, 0]
        r_hist, bin_edges = np.histogram(r_dep * 1e6, bins=30, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax2.bar(bin_centers, r_hist, width=np.diff(bin_edges)[0], alpha=0.7, color='green')
        ax2.axvline(x=R_final * 1e6, color='red', linestyle='--', label='最终接触线')
        ax2.set_xlabel('径向位置 (μm)')
        ax2.set_ylabel('概率密度')
        ax2.set_title('颗粒径向分布')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[1, 0]
        hist = results['history']
        t_hist = [h['t'] for h in hist]
        R_hist = [h['R'] for h in hist]
        n_dep_hist = [h['n_deposited'] for h in hist]
        
        ax3_twin = ax3.twinx()
        line1 = ax3.plot(t_hist, np.array(R_hist) * 1e6, 'b-', linewidth=2, label='接触半径')
        line2 = ax3_twin.plot(t_hist, n_dep_hist, 'r-', linewidth=2, label='沉积颗粒数')
        
        ax3.set_xlabel('时间 (s)')
        ax3.set_ylabel('接触半径 (μm)', color='b')
        ax3_twin.set_ylabel('沉积颗粒数', color='r')
        ax3.set_title('蒸发过程动态')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        ax4 = axes[1, 1]
        deposit_time = deposited[:, 0]
        ax4.scatter(deposit_time, r_dep * 1e6, s=10, alpha=0.5, c=deposit_time, cmap='jet')
        ax4.set_xlabel('沉积时间 (s)')
        ax4.set_ylabel('径向位置 (μm)')
        ax4.set_title('颗粒沉积时间分布')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f'沉积图案图已保存至: {filename}')
        
        plt.close()
        
        return fig
    
    def plot_flow_field(self, R, theta, dRdt, flow_mode='marangoni', filename=None):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'速度场对比 - {flow_mode}', fontsize=14, fontweight='bold')
        
        r_grid = np.linspace(-R * 0.99, R * 0.99, 30)
        h = self._height_from_R_theta(R, theta)
        z_grid = np.linspace(0, h * 0.99, 20)
        
        R_mesh, Z_mesh = np.meshgrid(r_grid, z_grid)
        
        Vr = np.zeros_like(R_mesh)
        Vz = np.zeros_like(R_mesh)
        Speed = np.zeros_like(R_mesh)
        
        for i in range(R_mesh.shape[0]):
            for j in range(R_mesh.shape[1]):
                r = abs(R_mesh[i, j])
                z = Z_mesh[i, j]
                
                h_r = h * np.sqrt(1 - (r / R)**2)
                if z <= h_r and r < R:
                    v_r, v_z = self.total_velocity(r, z, R, theta, dRdt, mode=flow_mode)
                    sign = 1 if R_mesh[i, j] >= 0 else -1
                    Vr[i, j] = v_r * sign
                    Vz[i, j] = v_z
                    Speed[i, j] = np.sqrt(v_r**2 + v_z**2)
                else:
                    Vr[i, j] = np.nan
                    Vz[i, j] = np.nan
                    Speed[i, j] = np.nan
        
        ax1 = axes[0]
        strm = ax1.streamplot(R_mesh * 1e6, Z_mesh * 1e6, Vr * 1e6, Vz * 1e6,
                              color=Speed * 1e6, cmap='jet', density=1.5, linewidth=1)
        
        r_profile = np.linspace(0, R, 100)
        h_profile = h * np.sqrt(1 - (r_profile / R)**2)
        ax1.plot(r_profile * 1e6, h_profile * 1e6, 'k-', linewidth=2, label='液滴表面')
        ax1.plot(-r_profile * 1e6, h_profile * 1e6, 'k-', linewidth=2)
        ax1.fill_between(r_profile * 1e6, 0, h_profile * 1e6, alpha=0.2, color='blue')
        ax1.fill_between(-r_profile * 1e6, 0, h_profile * 1e6, alpha=0.2, color='blue')
        
        ax1.set_xlabel('径向位置 r (μm)')
        ax1.set_ylabel('高度 z (μm)')
        ax1.set_title('流线图 (速度大小颜色编码)')
        ax1.set_aspect('equal')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.colorbar(strm.lines, ax=ax1, label='速度 (μm/s)')
        
        ax2 = axes[1]
        r_plot = np.linspace(0, R * 0.95, 5)
        z_plot = np.linspace(0, h * 0.95, 5)
        
        for zi in z_plot:
            v_r_list = []
            for ri in r_plot:
                h_r = h * np.sqrt(1 - (ri / R)**2)
                if zi <= h_r:
                    v_r, _ = self.total_velocity(ri, zi, R, theta, dRdt, mode=flow_mode)
                    v_r_list.append(v_r * 1e6)
                else:
                    v_r_list.append(np.nan)
            ax2.plot(r_plot * 1e6, v_r_list, 'o-', label=f'z={zi*1e6:.0f}μm')
        
        ax2.set_xlabel('径向位置 r (μm)')
        ax2.set_ylabel('径向速度 v_r (μm/s)')
        ax2.set_title('不同高度的径向速度分布')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f'速度场图已保存至: {filename}')
        
        plt.close()
        
        return fig
    
    def simulate_dna_chip(self, spot_diameter=200e-6, dna_concentration=1.0,
                          flow_mode='marangoni', filename=None):
        print("=" * 60)
        print("DNA芯片微点沉积模拟")
        print("=" * 60)
        
        R_spot = spot_diameter / 2
        theta_spot = np.radians(45)
        
        self.R0 = R_spot
        self.theta0 = theta_spot
        self.N_particles = int(1000 * dna_concentration)
        
        print(f"\n微点参数:")
        print(f"  直径: {spot_diameter*1e6:.0f} μm")
        print(f"  初始体积: {self._volume_from_R_theta(R_spot, theta_spot)*1e12:.2f} pL")
        print(f"  DNA分子数: {self.N_particles}")
        print(f"  流动模式: {flow_mode}")
        print()
        
        results = self.simulate_evaporation_with_particles(flow_mode=flow_mode)
        
        deposited = results['deposited_particles']
        r_dep = deposited[:, 1]
        
        valid_mask = r_dep < R_spot * 2.0
        if np.sum(valid_mask) > 0:
            r_dep_valid = r_dep[valid_mask]
        else:
            r_dep_valid = r_dep
        
        ring_width = np.std(r_dep_valid)
        mean_r = np.mean(r_dep_valid)
        
        if mean_r > 0 and ring_width > 0:
            coffee_ring_index = mean_r / ring_width
        else:
            coffee_ring_index = 0
        
        uniformity = 1 - (np.max(r_dep_valid) - np.min(r_dep_valid)) / (2 * R_spot) if R_spot > 0 else 0
        
        print(f"沉积结果:")
        print(f"  总沉积颗粒: {len(deposited)}")
        print(f"  平均径向位置: {mean_r*1e6:.1f} μm")
        print(f"  环宽度 (σ): {ring_width*1e6:.1f} μm")
        print(f"  咖啡环指数: {coffee_ring_index:.2f}")
        print(f"  均匀性指数: {max(-1, min(1, uniformity)):.3f}")
        print(f"  蒸发时间: {results['total_time']:.4f} s")
        print()
        
        if filename:
            self.plot_deposition_pattern(results, filename=filename)
        
        return results


def compare_flow_modes():
    print("\n" + "=" * 70)
    print("不同流动模式对比")
    print("=" * 70)
    
    modes = ['capillary', 'marangoni', 'diffusiophoresis']
    mode_names = ['毛细流 (咖啡环)', '马兰戈尼对流', '扩散泳效应']
    
    R0 = 200e-6
    theta0 = np.radians(45)
    
    results_list = []
    
    for mode, name in zip(modes, mode_names):
        print(f"\n--- {name} ---")
        model = DropletWithMarangoni(R0=R0, theta0=theta0, N_particles=500,
                                     marangoni_strength=1.0 if mode == 'marangoni' else 0)
        results = model.simulate_evaporation_with_particles(flow_mode=mode)
        
        deposited = results['deposited_particles']
        r_dep = deposited[:, 1]
        mean_r = np.mean(r_dep)
        std_r = np.std(r_dep)
        
        print(f"  平均径向位置: {mean_r*1e6:.1f} μm")
        print(f"  径向分布宽度: {std_r*1e6:.1f} μm")
        print(f"  咖啡环强度: {mean_r/(std_r+1e-10):.2f}")
        
        results_list.append(results)
        
        model.plot_deposition_pattern(results, filename=f'deposition_{mode}.png')
    
    return results_list


def main():
    print("=" * 70)
    print("微液滴蒸发模拟 - 马兰戈尼效应与颗粒沉积")
    print("=" * 70)
    
    R0 = 300e-6
    theta0 = np.radians(45)
    
    model = DropletWithMarangoni(R0=R0, theta0=theta0, T=298.15,
                                 N_particles=800, particle_diameter=50e-9,
                                 marangoni_strength=1.0)
    
    print(f"\n初始参数:")
    print(f"  接触半径 R0 = {R0*1e6:.0f} μm")
    print(f"  接触角 θ0 = {np.degrees(theta0):.0f}°")
    print(f"  初始体积 V0 = {model._volume_from_R_theta(R0, theta0)*1e12:.2f} pL")
    print(f"  颗粒数 N = {model.N_particles}")
    print(f"  颗粒直径 = {model.particle_diameter*1e9:.0f} nm")
    print()
    
    dRdt_est = model._dVdt(R0, theta0) / (np.pi * R0**2 * (1 - np.cos(theta0))**2 * (2 + np.cos(theta0)))
    
    print("生成速度场图...")
    model.plot_flow_field(R0, theta0, dRdt_est, flow_mode='marangoni',
                          filename='flow_field_marangoni.png')
    
    print("\n运行马兰戈尼效应模拟...")
    results_mar = model.simulate_evaporation_with_particles(flow_mode='marangoni')
    model.plot_deposition_pattern(results_mar, filename='deposition_marangoni.png', style='coffee_ring')
    
    print("\n运行纯毛细流模拟 (咖啡环效应)...")
    model.marangoni_strength = 0.0
    results_cap = model.simulate_evaporation_with_particles(flow_mode='capillary')
    model.plot_deposition_pattern(results_cap, filename='deposition_capillary.png', style='coffee_ring')
    
    print("\n" + "=" * 70)
    print("DNA芯片微点沉积模拟")
    print("=" * 70)
    
    chip_model = DropletWithMarangoni(N_particles=2000, particle_diameter=20e-9,
                                      marangoni_strength=0.3)
    chip_results = chip_model.simulate_dna_chip(spot_diameter=400e-6,
                                                 flow_mode='marangoni',
                                                 filename='dna_chip_spot.png')
    
    print("\n" + "=" * 70)
    print("模拟完成! 生成的图片:")
    print("  - flow_field_marangoni.png")
    print("  - deposition_marangoni.png")
    print("  - deposition_capillary.png")
    print("  - dna_chip_spot.png")
    print("=" * 70)


if __name__ == "__main__":
    main()

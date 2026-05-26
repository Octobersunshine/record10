import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class DropletEvaporation:
    def __init__(self, R0=1e-4, theta0=np.pi/4, T=298.15, P=101325,
                 regularization_method='cutoff', theta_min=np.radians(5.0),
                 delta_m=1e-9):
        self.R0 = R0
        self.theta0 = theta0
        self.T = T
        self.P = P
        
        self.D = 2.5e-5
        self.M = 0.01801528
        self.rho = 1000.0
        self.R_g = 8.314
        
        self.Psat = self._antoine_water(T)
        self.c_sat = (self.Psat * self.M) / (self.R_g * self.T)
        self.c_inf = 0.0
        
        self.regularization_method = regularization_method
        self.theta_min = theta_min
        self.delta_m = delta_m
        
        self._validate_regularization()
    
    def _validate_regularization(self):
        valid_methods = ['none', 'cutoff', 'molecular_cutoff', 'average_flux', 'saturation']
        if self.regularization_method not in valid_methods:
            raise ValueError(f"regularization_method must be one of {valid_methods}")
    
    def _antoine_water(self, T):
        A = 8.07131
        B = 1730.63
        C = 233.426
        T_celsius = T - 273.15
        P_mmHg = 10 ** (A - B / (T_celsius + C))
        return P_mmHg * 133.322
    
    def _g_theta_original(self, theta):
        return (0.27 * theta**2 + 1.30) / (np.sin(theta) * (1 + np.cos(theta))**2)
    
    def _g_theta_cutoff(self, theta):
        theta_clipped = np.maximum(theta, self.theta_min)
        return self._g_theta_original(theta_clipped)
    
    def _g_theta_molecular_cutoff(self, theta, R):
        if theta < self.theta_min:
            h = R * (1 - np.cos(theta))
            h_eff = np.maximum(h, self.delta_m)
            cos_theta_eff = 1 - h_eff / R
            cos_theta_eff = np.clip(cos_theta_eff, -1.0, 1.0)
            theta_eff = np.arccos(cos_theta_eff)
            theta_eff = np.maximum(theta_eff, self.theta_min)
            return self._g_theta_original(theta_eff)
        return self._g_theta_original(theta)
    
    def _g_theta_average_flux(self, theta, R):
        if theta >= self.theta_min:
            return self._g_theta_original(theta)
        
        h = R * (1 - np.cos(theta))
        
        if h < self.delta_m:
            h = self.delta_m
            cos_theta_eff = 1 - h / R
            cos_theta_eff = np.clip(cos_theta_eff, -1.0, 1.0)
            theta_eff = np.arccos(cos_theta_eff)
            return self._g_theta_original(theta_eff)
        
        area = np.pi * R**2
        volume = self._volume_from_R_theta(R, theta)
        avg_flux = volume / (area * h) * self.D * (self.c_sat - self.c_inf)
        
        g_avg = avg_flux * R / (self.D * (self.c_sat - self.c_inf))
        return np.maximum(g_avg, self._g_theta_original(self.theta_min))
    
    def _g_theta_saturation(self, theta):
        if theta >= self.theta_min:
            return self._g_theta_original(theta)
        
        g_min = self._g_theta_original(self.theta_min)
        
        alpha = theta / self.theta_min
        g_theta_small = g_min * (1 + (1 - alpha) * 0.5)
        
        return g_theta_small
    
    def _g_theta(self, theta, R=None):
        if self.regularization_method == 'none':
            return self._g_theta_original(theta)
        elif self.regularization_method == 'cutoff':
            return self._g_theta_cutoff(theta)
        elif self.regularization_method == 'molecular_cutoff':
            return self._g_theta_molecular_cutoff(theta, R if R is not None else self.R0)
        elif self.regularization_method == 'average_flux':
            return self._g_theta_average_flux(theta, R if R is not None else self.R0)
        elif self.regularization_method == 'saturation':
            return self._g_theta_saturation(theta)
        else:
            return self._g_theta_original(theta)
    
    def _volume_from_R_theta(self, R, theta):
        return (np.pi * R**3 / 3) * (1 - np.cos(theta))**2 * (2 + np.cos(theta))
    
    def _height_from_R_theta(self, R, theta):
        return R * (1 - np.cos(theta))
    
    def _R_from_V_theta(self, V, theta):
        numerator = 3 * V
        denominator = np.pi * (1 - np.cos(theta))**2 * (2 + np.cos(theta))
        return (numerator / denominator) ** (1/3)
    
    def dVdt_constant_R(self, R, theta):
        g = self._g_theta(theta, R)
        return -np.pi * self.D * (self.c_sat - self.c_inf) * R * g
    
    def dthetadt_constant_R(self, theta):
        R = self.R0
        dVdt = self.dVdt_constant_R(R, theta)
        
        dVdtheta = (np.pi * R**3 / 3) * (
            2 * (1 - np.cos(theta)) * np.sin(theta) * (2 + np.cos(theta)) -
            (1 - np.cos(theta))**2 * np.sin(theta)
        )
        
        if abs(dVdtheta) < 1e-20:
            return 0.0
        
        return dVdt / dVdtheta
    
    def simulate_constant_contact_radius(self, t_eval=None, max_time=3600):
        theta_terminate = max(self.theta_min * 0.5, 0.001)
        
        def ode(t, y):
            theta = y[0]
            if theta < theta_terminate or theta > np.pi - 0.01:
                return [0.0]
            dtheta = self.dthetadt_constant_R(theta)
            return [float(dtheta)]
        
        t_span = (0, max_time)
        
        if t_eval is None:
            t_eval = np.linspace(0, max_time, 1000)
        
        sol = solve_ivp(ode, t_span, [self.theta0], t_eval=t_eval,
                       method='RK45', rtol=1e-6, atol=1e-8, max_step=max_time/100)
        
        t = sol.t
        theta = sol.y[0]
        R = np.full_like(t, self.R0)
        
        V = self._volume_from_R_theta(R, theta)
        h = self._height_from_R_theta(R, theta)
        
        valid = theta > theta_terminate
        return {
            't': t[valid],
            'R': R[valid],
            'theta': theta[valid],
            'V': V[valid],
            'h': h[valid],
            'mode': 'constant_R',
            'regularization': self.regularization_method
        }
    
    def dRdt_constant_theta(self, R, theta):
        g = self._g_theta(theta, R)
        
        dVdt = -np.pi * self.D * (self.c_sat - self.c_inf) * R * g
        dVdR = np.pi * R**2 * (1 - np.cos(theta))**2 * (2 + np.cos(theta))
        
        if abs(dVdR) < 1e-20:
            return 0.0
        
        return dVdt / dVdR
    
    def simulate_constant_contact_angle(self, t_eval=None, max_time=3600):
        theta = self.theta0
        
        R_terminate = self.delta_m
        
        def ode(t, y):
            R = y[0]
            if R < R_terminate:
                return [0.0]
            dR = self.dRdt_constant_theta(R, theta)
            return [float(dR)]
        
        t_span = (0, max_time)
        
        if t_eval is None:
            t_eval = np.linspace(0, max_time, 1000)
        
        sol = solve_ivp(ode, t_span, [self.R0], t_eval=t_eval,
                       method='RK45', rtol=1e-6, atol=1e-8, max_step=max_time/100)
        
        t = sol.t
        R = sol.y[0]
        theta_arr = np.full_like(t, theta)
        
        V = self._volume_from_R_theta(R, theta)
        h = self._height_from_R_theta(R, theta)
        
        valid = R > R_terminate
        return {
            't': t[valid],
            'R': R[valid],
            'theta': theta_arr[valid],
            'V': V[valid],
            'h': h[valid],
            'mode': 'constant_theta',
            'regularization': self.regularization_method
        }
    
    def plot_results(self, results, filename=None):
        t = results['t']
        R = results['R']
        theta = results['theta']
        V = results['V']
        h = results['h']
        mode = results['mode']
        
        mode_str = '恒定接触半径' if mode == 'constant_R' else '恒定接触角'
        reg_str = results.get('regularization', 'none')
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'微液滴蒸发模拟 - {mode_str}模式 ({reg_str})', fontsize=14, fontweight='bold')
        
        axes[0, 0].plot(t, R * 1e6, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('时间 (s)')
        axes[0, 0].set_ylabel('接触半径 (μm)')
        axes[0, 0].set_title('接触半径随时间变化')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(t, np.degrees(theta), 'r-', linewidth=2)
        axes[0, 1].set_xlabel('时间 (s)')
        axes[0, 1].set_ylabel('接触角 (°)')
        axes[0, 1].set_title('接触角随时间变化')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(t, V * 1e12, 'g-', linewidth=2)
        axes[1, 0].set_xlabel('时间 (s)')
        axes[1, 0].set_ylabel('体积 (pL)')
        axes[1, 0].set_title('体积随时间变化')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(t, h * 1e6, 'm-', linewidth=2)
        axes[1, 1].set_xlabel('时间 (s)')
        axes[1, 1].set_ylabel('高度 (μm)')
        axes[1, 1].set_title('液滴高度随时间变化')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f'图表已保存至: {filename}')
        
        plt.close()
        
        return fig
    
    def compare_modes(self, results1, results2, filename=None):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('两种蒸发模式对比', fontsize=14, fontweight='bold')
        
        for results, label in [(results1, '恒定接触半径'), (results2, '恒定接触角')]:
            t = results['t']
            R = results['R']
            theta = results['theta']
            V = results['V']
            h = results['h']
            
            axes[0, 0].plot(t, R * 1e6, linewidth=2, label=label)
            axes[0, 1].plot(t, np.degrees(theta), linewidth=2, label=label)
            axes[1, 0].plot(t, V * 1e12, linewidth=2, label=label)
            axes[1, 1].plot(t, h * 1e6, linewidth=2, label=label)
        
        axes[0, 0].set_xlabel('时间 (s)')
        axes[0, 0].set_ylabel('接触半径 (μm)')
        axes[0, 0].set_title('接触半径对比')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_xlabel('时间 (s)')
        axes[0, 1].set_ylabel('接触角 (°)')
        axes[0, 1].set_title('接触角对比')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_xlabel('时间 (s)')
        axes[1, 0].set_ylabel('体积 (pL)')
        axes[1, 0].set_title('体积对比')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_xlabel('时间 (s)')
        axes[1, 1].set_ylabel('高度 (μm)')
        axes[1, 1].set_title('液滴高度对比')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f'对比图已保存至: {filename}')
        
        plt.close()
        
        return fig
    
    def compare_regularization_methods(self, methods=None, filename=None):
        if methods is None:
            methods = ['none', 'cutoff', 'molecular_cutoff', 'average_flux', 'saturation']
        
        R0 = self.R0
        theta0 = self.theta0
        T = self.T
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('不同正则化方法对比 (恒定接触半径模式)', fontsize=14, fontweight='bold')
        
        colors = ['black', 'blue', 'red', 'green', 'orange']
        
        for idx, method in enumerate(methods):
            model = DropletEvaporation(R0=R0, theta0=theta0, T=T,
                                       regularization_method=method,
                                       theta_min=np.radians(3.0))
            
            dVdt = model.dVdt_constant_R(R0, theta0)
            V0 = model._volume_from_R_theta(R0, theta0)
            est_time = V0 / abs(dVdt)
            max_time = est_time * 1.5
            
            results = model.simulate_constant_contact_radius(max_time=max_time)
            
            t = results['t']
            theta = np.degrees(results['theta'])
            V = results['V'] * 1e12
            h = results['h'] * 1e6
            
            color = colors[idx % len(colors)]
            
            axes[0, 0].plot(t, theta, color, linewidth=2, label=method)
            axes[0, 1].plot(t, V, color, linewidth=2, label=method)
            axes[1, 0].plot(t, h, color, linewidth=2, label=method)
            
            flux = np.abs(np.gradient(V, t))
            axes[1, 1].plot(t, flux, color, linewidth=2, label=method)
        
        axes[0, 0].set_xlabel('时间 (s)')
        axes[0, 0].set_ylabel('接触角 (°)')
        axes[0, 0].set_title('接触角随时间变化')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_xlabel('时间 (s)')
        axes[0, 1].set_ylabel('体积 (pL)')
        axes[0, 1].set_title('体积随时间变化')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_xlabel('时间 (s)')
        axes[1, 0].set_ylabel('高度 (μm)')
        axes[1, 0].set_title('液滴高度随时间变化')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_xlabel('时间 (s)')
        axes[1, 1].set_ylabel('蒸发速率 (pL/s)')
        axes[1, 1].set_title('蒸发速率随时间变化')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f'正则化方法对比图已保存至: {filename}')
        
        plt.close()
        
        return fig


def main():
    R0 = 1000e-6
    theta0_deg = 60
    theta0 = np.radians(theta0_deg)
    T = 298.15
    
    print("=" * 70)
    print("微液滴蒸发模拟 - 扩散控制模型 (含奇异点修复)")
    print("=" * 70)
    print(f"初始参数:")
    print(f"  初始接触半径 R0 = {R0 * 1e6:.1f} μm")
    print(f"  初始接触角 θ0 = {theta0_deg:.1f}°")
    print(f"  温度 T = {T:.2f} K")
    print()
    
    model = DropletEvaporation(R0=R0, theta0=theta0, T=T,
                               regularization_method='cutoff',
                               theta_min=np.radians(3.0))
    
    V0 = model._volume_from_R_theta(R0, theta0)
    print(f"初始体积 V0 = {V0 * 1e12:.2f} pL")
    print(f"初始高度 h0 = {model._height_from_R_theta(R0, theta0) * 1e6:.2f} μm")
    print()
    
    dVdt = model.dVdt_constant_R(R0, theta0)
    est_time = V0 / abs(dVdt)
    print(f"初始蒸发速率: {abs(dVdt) * 1e12:.2f} pL/s")
    print(f"预计蒸发时间: {est_time:.4f} s")
    print(f"正则化方法: {model.regularization_method}")
    print(f"最小接触角: {np.degrees(model.theta_min):.2f}°")
    print()
    
    max_time = est_time * 1.5
    
    print("运行恒定接触半径模式模拟...")
    results_ccr = model.simulate_constant_contact_radius(max_time=max_time)
    t_ccr = results_ccr['t']
    print(f"  模拟时间: {t_ccr[-1]:.4f} s")
    print(f"  初始接触角: {np.degrees(results_ccr['theta'][0]):.2f}°")
    print(f"  最终接触角: {np.degrees(results_ccr['theta'][-1]):.2f}°")
    print(f"  初始体积: {results_ccr['V'][0] * 1e12:.2f} pL")
    print(f"  最终体积: {results_ccr['V'][-1] * 1e12:.2f} pL")
    print()
    
    print("运行恒定接触角模式模拟...")
    results_cca = model.simulate_constant_contact_angle(max_time=max_time)
    t_cca = results_cca['t']
    print(f"  模拟时间: {t_cca[-1]:.4f} s")
    print(f"  初始接触半径: {results_cca['R'][0] * 1e6:.2f} μm")
    print(f"  最终接触半径: {results_cca['R'][-1] * 1e6:.2f} μm")
    print(f"  初始体积: {results_cca['V'][0] * 1e12:.2f} pL")
    print(f"  最终体积: {results_cca['V'][-1] * 1e12:.2f} pL")
    print()
    
    print("生成结果图表...")
    model.plot_results(results_ccr, filename='evaporation_constant_R.png')
    model.plot_results(results_cca, filename='evaporation_constant_theta.png')
    model.compare_modes(results_ccr, results_cca, filename='evaporation_comparison.png')
    
    print()
    print("生成不同正则化方法的对比图...")
    model.compare_regularization_methods(filename='regularization_comparison.png')
    
    print()
    print("模拟完成!")
    print("生成的图片文件:")
    print("  - evaporation_constant_R.png")
    print("  - evaporation_constant_theta.png")
    print("  - evaporation_comparison.png")
    print("  - regularization_comparison.png")
    print("=" * 70)


if __name__ == "__main__":
    main()

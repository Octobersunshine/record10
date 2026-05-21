import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import optimize
import json


def basquin_equation(N, sigma_f, b):
    return sigma_f * (2 * N) ** b


def log_basquin_equation(log_N, log_sigma_f, b):
    return log_sigma_f + b * log_N


class StressTensor:
    def __init__(self, sigma_xx=0, sigma_yy=0, sigma_zz=0, 
                 tau_xy=0, tau_yz=0, tau_xz=0):
        self.tensor = np.array([
            [sigma_xx, tau_xy, tau_xz],
            [tau_xy, sigma_yy, tau_yz],
            [tau_xz, tau_yz, sigma_zz]
        ])
    
    @property
    def sigma_xx(self): return self.tensor[0, 0]
    @property
    def sigma_yy(self): return self.tensor[1, 1]
    @property
    def sigma_zz(self): return self.tensor[2, 2]
    @property
    def tau_xy(self): return self.tensor[0, 1]
    @property
    def tau_yz(self): return self.tensor[1, 2]
    @property
    def tau_xz(self): return self.tensor[0, 2]
    
    def principal_stresses(self):
        eigenvalues, _ = np.linalg.eigh(self.tensor)
        return np.sort(eigenvalues)[::-1]
    
    def von_mises(self):
        s = self.tensor
        return np.sqrt(0.5 * ((s[0,0]-s[1,1])**2 + (s[1,1]-s[2,2])**2 + 
                             (s[2,2]-s[0,0])**2 + 6*(s[0,1]**2 + s[1,2]**2 + s[0,2]**2)))
    
    def stress_on_plane(self, theta, phi):
        n = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])
        sigma_n = n.T @ self.tensor @ n
        tau_vec = self.tensor @ n - sigma_n * n
        tau_mag = np.linalg.norm(tau_vec)
        return sigma_n, tau_mag
    
    def __add__(self, other):
        result = StressTensor()
        result.tensor = self.tensor + other.tensor
        return result
    
    def __mul__(self, scalar):
        result = StressTensor()
        result.tensor = self.tensor * scalar
        return result
    
    def __repr__(self):
        return f"StressTensor(\n  σ_xx={self.sigma_xx:.1f}, σ_yy={self.sigma_yy:.1f}, σ_zz={self.sigma_zz:.1f},\n  τ_xy={self.tau_xy:.1f}, τ_yz={self.tau_yz:.1f}, τ_xz={self.tau_xz:.1f}\n)"


class LoadHistory:
    def __init__(self):
        self.time_points = []
        self.stress_states = []
    
    def add_stress_state(self, time, stress_tensor):
        self.time_points.append(time)
        self.stress_states.append(stress_tensor)
    
    def add_tension_torsion(self, t, sigma_axial, tau_torsion):
        stress = StressTensor(sigma_xx=sigma_axial, tau_xy=tau_torsion)
        self.add_stress_state(t, stress)
    
    def generate_cyclic_load(self, n_cycles=10, n_points_per_cycle=36,
                             sigma_axial_amp=0, sigma_axial_mean=0,
                             tau_torsion_amp=0, tau_torsion_mean=0,
                             phase_shift=0):
        self.time_points = []
        self.stress_states = []
        
        for i in range(n_cycles * n_points_per_cycle):
            t = 2 * np.pi * i / n_points_per_cycle
            sigma_a = sigma_axial_mean + sigma_axial_amp * np.sin(t)
            tau_a = tau_torsion_mean + tau_torsion_amp * np.sin(t + phase_shift)
            self.add_tension_torsion(t, sigma_a, tau_a)
        
        return self
    
    def get_stress_range(self):
        sigma_xx = [s.sigma_xx for s in self.stress_states]
        tau_xy = [s.tau_xy for s in self.stress_states]
        
        ranges = {
            'sigma_xx': max(sigma_xx) - min(sigma_xx),
            'sigma_xx_mean': np.mean(sigma_xx),
            'tau_xy': max(tau_xy) - min(tau_xy),
            'tau_xy_mean': np.mean(tau_xy)
        }
        return ranges
    
    def __len__(self):
        return len(self.stress_states)
    
    def __getitem__(self, idx):
        return self.stress_states[idx]


class MultiaxialFatigue:
    def __init__(self, sigma_f_prime=1000, b=-0.1, k_f=0.8):
        self.sigma_f_prime = sigma_f_prime
        self.b = b
        self.k_f = k_f
    
    def findley_criterion(self, load_history, k=None):
        if k is None:
            k = self.k_f
        
        n_theta = 37
        n_phi = 73
        thetas = np.linspace(0, np.pi, n_theta)
        phis = np.linspace(0, 2 * np.pi, n_phi)
        
        max_damage = -np.inf
        critical_theta = 0
        critical_phi = 0
        critical_params = None
        
        for theta in thetas:
            for phi in phis:
                sigma_n_list = []
                tau_list = []
                
                for stress in load_history:
                    sigma_n, tau = stress.stress_on_plane(theta, phi)
                    sigma_n_list.append(sigma_n)
                    tau_list.append(tau)
                
                sigma_n_array = np.array(sigma_n_list)
                tau_array = np.array(tau_list)
                
                tau_amp = 0.5 * (np.max(tau_array) - np.min(tau_array))
                sigma_n_max = np.max(sigma_n_array)
                
                findley_param = tau_amp + k * max(sigma_n_max, 0)
                
                if findley_param > max_damage:
                    max_damage = findley_param
                    critical_theta = theta
                    critical_phi = phi
                    critical_params = {
                        'tau_amp': tau_amp,
                        'sigma_n_max': sigma_n_max
                    }
        
        return {
            'max_damage': max_damage,
            'critical_theta': np.degrees(critical_theta),
            'critical_phi': np.degrees(critical_phi),
            'tau_amp': critical_params['tau_amp'],
            'sigma_n_max': critical_params['sigma_n_max'],
            'k': k
        }
    
    def critical_plane_sws(self, load_history):
        n_theta = 37
        n_phi = 73
        thetas = np.linspace(0, np.pi, n_theta)
        phis = np.linspace(0, 2 * np.pi, n_phi)
        
        max_sws = -np.inf
        critical_theta = 0
        critical_phi = 0
        
        for theta in thetas:
            for phi in phis:
                sigma_n_list = []
                tau_list = []
                
                for stress in load_history:
                    sigma_n, tau = stress.stress_on_plane(theta, phi)
                    sigma_n_list.append(sigma_n)
                    tau_list.append(tau)
                
                sigma_n_array = np.array(sigma_n_list)
                tau_array = np.array(tau_list)
                
                sigma_n_amp = 0.5 * (np.max(sigma_n_array) - np.min(sigma_n_array))
                sigma_n_max = np.max(sigma_n_array)
                
                sws = np.sqrt(sigma_n_amp * max(sigma_n_max, 0))
                
                if sws > max_sws:
                    max_sws = sws
                    critical_theta = theta
                    critical_phi = phi
        
        return {
            'max_sws': max_sws,
            'critical_theta': np.degrees(critical_theta),
            'critical_phi': np.degrees(critical_phi)
        }
    
    def brown_miller_criterion(self, load_history, k=0.5):
        n_theta = 37
        n_phi = 73
        thetas = np.linspace(0, np.pi, n_theta)
        phis = np.linspace(0, 2 * np.pi, n_phi)
        
        max_damage = -np.inf
        critical_theta = 0
        critical_phi = 0
        
        for theta in thetas:
            for phi in phis:
                sigma_n_list = []
                tau_list = []
                
                for stress in load_history:
                    sigma_n, tau = stress.stress_on_plane(theta, phi)
                    sigma_n_list.append(sigma_n)
                    tau_list.append(tau)
                
                sigma_n_array = np.array(sigma_n_list)
                tau_array = np.array(tau_list)
                
                gamma_amp = (np.max(tau_array) - np.min(tau_array))
                sigma_n_max = np.max(sigma_n_array)
                
                damage = gamma_amp + k * max(sigma_n_max, 0)
                
                if damage > max_damage:
                    max_damage = damage
                    critical_theta = theta
                    critical_phi = phi
        
        return {
            'max_damage': max_damage,
            'critical_theta': np.degrees(critical_theta),
            'critical_phi': np.degrees(critical_phi),
            'k': k
        }
    
    def predict_life_findley(self, load_history, k=None):
        result = self.findley_criterion(load_history, k)
        damage_param = result['max_damage']
        
        if damage_param <= 0:
            return 1e12
        
        log_life = (np.log10(damage_param) - np.log10(self.sigma_f_prime)) / self.b
        life = 10 ** log_life
        
        return min(life, 1e12)
    
    def predict_life_sws(self, load_history):
        result = self.critical_plane_sws(load_history)
        sws = result['max_sws']
        
        if sws <= 0:
            return 1e12
        
        log_life = (np.log10(sws) - np.log10(self.sigma_f_prime)) / self.b
        life = 10 ** log_life
        
        return min(life, 1e12)
    
    def equibiaxial_ratio_analysis(self, sigma_amp, tau_amp_ratio_range=(0, 2)):
        ratios = np.linspace(tau_amp_ratio_range[0], tau_amp_ratio_range[1], 20)
        lives_findley = []
        lives_sws = []
        
        for ratio in ratios:
            tau_amp = sigma_amp * ratio
            
            load = LoadHistory()
            load.generate_cyclic_load(
                n_cycles=2, n_points_per_cycle=36,
                sigma_axial_amp=sigma_amp, sigma_axial_mean=0,
                tau_torsion_amp=tau_amp, tau_torsion_mean=0,
                phase_shift=0
            )
            
            life_findley = self.predict_life_findley(load)
            life_sws = self.predict_life_sws(load)
            
            lives_findley.append(life_findley)
            lives_sws.append(life_sws)
        
        return ratios, lives_findley, lives_sws


class MeanStressCorrection:
    def __init__(self, sigma_uts=1000, sigma_yield=800, walker_gamma=0.5):
        self.sigma_uts = sigma_uts
        self.sigma_yield = sigma_yield
        self.walker_gamma = walker_gamma

    def goodman(self, sigma_a, sigma_m):
        sigma_m = np.asarray(sigma_m)
        sigma_a = np.asarray(sigma_a)
        
        sigma_a_eq = np.zeros_like(sigma_a, dtype=float)
        
        for i in range(len(sigma_a)):
            sm = sigma_m.flat[i]
            sa = sigma_a.flat[i]
            
            if sm >= 0:
                if sa <= 0:
                    sigma_a_eq.flat[i] = 0
                else:
                    denom = 1 - sm / self.sigma_uts
                    if denom <= 1e-10:
                        sigma_a_eq.flat[i] = sa * 1e10
                    else:
                        sigma_a_eq.flat[i] = sa / denom
            else:
                sigma_a_eq.flat[i] = sa
                
        if sigma_a_eq.ndim == 0:
            sigma_a_eq = float(sigma_a_eq)
        return sigma_a_eq

    def gerber(self, sigma_a, sigma_m):
        sigma_m = np.asarray(sigma_m)
        sigma_a = np.asarray(sigma_a)
        
        sigma_a_eq = np.zeros_like(sigma_a, dtype=float)
        
        for i in range(len(sigma_a)):
            sm = sigma_m.flat[i]
            sa = sigma_a.flat[i]
            
            if sm >= 0:
                if sa <= 0:
                    sigma_a_eq.flat[i] = 0
                else:
                    ratio = sm / self.sigma_uts
                    denom = 1 - ratio ** 2
                    if denom <= 1e-10:
                        sigma_a_eq.flat[i] = sa * 1e10
                    else:
                        sigma_a_eq.flat[i] = sa / denom
            else:
                sigma_a_eq.flat[i] = sa
                
        if sigma_a_eq.ndim == 0:
            sigma_a_eq = float(sigma_a_eq)
        return sigma_a_eq

    def walker(self, sigma_a, sigma_m):
        sigma_m = np.asarray(sigma_m)
        sigma_a = np.asarray(sigma_a)
        
        sigma_max = sigma_a + sigma_m
        sigma_max = np.maximum(sigma_max, 1e-10)
        
        sigma_a_eq = sigma_a * (2 * sigma_max / (sigma_a + sigma_max)) ** self.walker_gamma
        
        if sigma_a_eq.ndim == 0:
            sigma_a_eq = float(sigma_a_eq)
        return sigma_a_eq

    def swt(self, sigma_a, sigma_m):
        sigma_m = np.asarray(sigma_m)
        sigma_a = np.asarray(sigma_a)
        
        sigma_max = sigma_a + sigma_m
        sigma_max = np.maximum(sigma_max, 0)
        
        sigma_a_eq = np.sqrt(sigma_a * sigma_max)
        
        if sigma_a_eq.ndim == 0:
            sigma_a_eq = float(sigma_a_eq)
        return sigma_a_eq

    def goodman_improved(self, sigma_a, sigma_m):
        sigma_m = np.asarray(sigma_m)
        sigma_a = np.asarray(sigma_a)
        
        sigma_a_eq = np.zeros_like(sigma_a, dtype=float)
        
        for i in range(len(sigma_a)):
            sm = sigma_m.flat[i]
            sa = sigma_a.flat[i]
            
            if sm >= 0:
                if sa <= 0:
                    sigma_a_eq.flat[i] = 0
                else:
                    denom = 1 - sm / self.sigma_uts
                    if denom <= 1e-10:
                        sigma_a_eq.flat[i] = sa * 1e10
                    else:
                        sigma_a_eq.flat[i] = sa / denom
            else:
                k = 0.3
                sm_abs = abs(sm)
                factor = 1 + k * (sm_abs / self.sigma_uts)
                factor = min(factor, 1.5)
                sigma_a_eq.flat[i] = sa / factor
                
        if sigma_a_eq.ndim == 0:
            sigma_a_eq = float(sigma_a_eq)
        return sigma_a_eq

    def correct_stress(self, sigma_a, sigma_m, method='walker'):
        methods = {
            'goodman': self.goodman,
            'gerber': self.gerber,
            'walker': self.walker,
            'swt': self.swt,
            'goodman_improved': self.goodman_improved
        }
        
        if method not in methods:
            raise ValueError(f"未知的修正方法: {method}。可选方法: {list(methods.keys())}")
        
        return methods[method](sigma_a, sigma_m)

    def inverse_correct_stress(self, sigma_a_eq, sigma_m, method='walker'):
        sigma_a_eq = np.asarray(sigma_a_eq)
        sigma_m = np.asarray(sigma_m)
        
        sigma_a = np.zeros_like(sigma_a_eq, dtype=float)
        
        if method == 'goodman':
            for i in range(len(sigma_a_eq)):
                sm = sigma_m.flat[i]
                sae = sigma_a_eq.flat[i]
                if sm >= 0:
                    sigma_a.flat[i] = sae * (1 - sm / self.sigma_uts)
                else:
                    sigma_a.flat[i] = sae
                    
        elif method == 'gerber':
            for i in range(len(sigma_a_eq)):
                sm = sigma_m.flat[i]
                sae = sigma_a_eq.flat[i]
                if sm >= 0:
                    sigma_a.flat[i] = sae * (1 - (sm / self.sigma_uts) ** 2)
                else:
                    sigma_a.flat[i] = sae
                    
        elif method == 'walker':
            for i in range(len(sigma_a_eq)):
                sm = sigma_m.flat[i]
                sae = sigma_a_eq.flat[i]
                
                def f(sa):
                    smax = sa + sm
                    if smax <= 0:
                        return sa - sae
                    return sa * (2 * smax / (sa + smax)) ** self.walker_gamma - sae
                
                sigma_a.flat[i] = self._solve_inverse(f, sae)
                
        elif method == 'swt':
            for i in range(len(sigma_a_eq)):
                sm = sigma_m.flat[i]
                sae = sigma_a_eq.flat[i]
                
                sigma_a.flat[i] = (sae ** 2) / max(sae ** 2 + sm, 1e-10)
                
        elif method == 'goodman_improved':
            for i in range(len(sigma_a_eq)):
                sm = sigma_m.flat[i]
                sae = sigma_a_eq.flat[i]
                if sm >= 0:
                    sigma_a.flat[i] = sae * (1 - sm / self.sigma_uts)
                else:
                    k = 0.3
                    sm_abs = abs(sm)
                    factor = 1 + k * (sm_abs / self.sigma_uts)
                    factor = min(factor, 1.5)
                    sigma_a.flat[i] = sae * factor
        
        if sigma_a.ndim == 0:
            sigma_a = float(sigma_a)
        return sigma_a

    def _solve_inverse(self, f, target, tol=1e-6, max_iter=100):
        low, high = 1e-6, target * 10
        for _ in range(max_iter):
            mid = (low + high) / 2
            val = f(mid)
            if abs(val) < tol:
                return mid
            if val < 0:
                low = mid
            else:
                high = mid
        return (low + high) / 2


class SNCurveFitter:
    def __init__(self, sigma_uts=1000, sigma_yield=800, walker_gamma=0.5, findley_k=0.8):
        self.stress_amplitudes = None
        self.cycles = None
        self.sigma_f = None
        self.b = None
        self.r_squared = None
        self.mean_stress = MeanStressCorrection(sigma_uts, sigma_yield, walker_gamma)
        self.multiaxial = None
    
    def _init_multiaxial(self):
        if self.sigma_f is None:
            raise ValueError("请先进行S-N曲线拟合")
        if self.multiaxial is None:
            self.multiaxial = MultiaxialFatigue(
                sigma_f_prime=self.sigma_f,
                b=self.b,
                k_f=0.8
            )

    def load_data(self, stress_amplitudes, cycles):
        if len(stress_amplitudes) != len(cycles):
            raise ValueError("应力幅和循环次数数组长度必须相同")
        self.stress_amplitudes = np.array(stress_amplitudes)
        self.cycles = np.array(cycles)

    def load_data_from_file(self, filepath):
        data = np.loadtxt(filepath, skiprows=1)
        self.stress_amplitudes = data[:, 0]
        self.cycles = data[:, 1]

    def fit(self):
        if self.stress_amplitudes is None or self.cycles is None:
            raise ValueError("请先加载数据")

        log_cycles = np.log10(2 * self.cycles)
        log_stress = np.log10(self.stress_amplitudes)

        popt, _ = curve_fit(log_basquin_equation, log_cycles, log_stress)

        log_sigma_f = popt[0]
        self.b = popt[1]
        self.sigma_f = 10 ** log_sigma_f

        y_pred = log_basquin_equation(log_cycles, log_sigma_f, self.b)
        ss_res = np.sum((log_stress - y_pred) ** 2)
        ss_tot = np.sum((log_stress - np.mean(log_stress)) ** 2)
        self.r_squared = 1 - (ss_res / ss_tot)

        return self.sigma_f, self.b, self.r_squared

    def predict_life(self, sigma_a, sigma_m=0, method='walker'):
        if self.sigma_f is None or self.b is None:
            raise ValueError("请先进行拟合")

        if sigma_m == 0:
            sigma_a_eq = sigma_a
        else:
            sigma_a_eq = self.mean_stress.correct_stress(sigma_a, sigma_m, method)

        sigma_a_eq = max(sigma_a_eq, 1e-10)
        
        log_stress = np.log10(sigma_a_eq)
        log_sigma_f = np.log10(self.sigma_f)
        log_2N = (log_stress - log_sigma_f) / self.b
        N = (10 ** log_2N) / 2
        
        max_n = 1e12
        N = min(N, max_n)
        return N

    def predict_stress(self, cycles, sigma_m=0, method='walker'):
        if self.sigma_f is None or self.b is None:
            raise ValueError("请先进行拟合")
        
        sigma_a_eq = basquin_equation(cycles, self.sigma_f, self.b)
        
        if sigma_m == 0:
            return sigma_a_eq
        else:
            return self.mean_stress.inverse_correct_stress(sigma_a_eq, sigma_m, method)

    def compare_correction_methods(self, sigma_a, sigma_m_values=None):
        if sigma_m_values is None:
            sigma_m_values = [-500, -300, -100, 0, 100, 300, 500]
        
        methods = ['goodman', 'gerber', 'walker', 'swt', 'goodman_improved']
        
        print("\n" + "=" * 80)
        print(f"平均应力修正方法对比 (应力幅 σ_a = {sigma_a} MPa)")
        print("=" * 80)
        print(f"{'平均应力':>10} {'方法':>18} {'等效应力幅':>15} {'预测寿命':>15}")
        print("-" * 80)
        
        for sm in sigma_m_values:
            for method in methods:
                try:
                    life = self.predict_life(sigma_a, sm, method)
                    sigma_eq = self.mean_stress.correct_stress(sigma_a, sm, method)
                    stress_ratio = (sm + sigma_a) / max(abs(sigma_a - sm), 1e-10)
                    print(f"{sm:>10.0f} {method:>18} {sigma_eq:>15.2f} {life:>15.2e}")
                except:
                    print(f"{sm:>10.0f} {method:>18} {'N/A':>15} {'N/A':>15}")
            print("-" * 80)

    def plot_correction_comparison(self, sigma_a, sigma_m_range=(-500, 600), save_path=None):
        sigma_m_values = np.linspace(sigma_m_range[0], sigma_m_range[1], 100)
        methods = ['goodman', 'gerber', 'walker', 'swt', 'goodman_improved']
        colors = ['r', 'g', 'b', 'm', 'c']
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        for method, color in zip(methods, colors):
            sigma_eq_values = []
            for sm in sigma_m_values:
                try:
                    seq = self.mean_stress.correct_stress(sigma_a, sm, method)
                    sigma_eq_values.append(seq)
                except:
                    sigma_eq_values.append(np.nan)
            plt.plot(sigma_m_values, sigma_eq_values, color + '-', label=method, linewidth=2)
        
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('平均应力 σ_m (MPa)')
        plt.ylabel('等效应力幅 σ_a_eq (MPa)')
        plt.title(f'平均应力修正对比 (σ_a = {sigma_a} MPa)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        for method, color in zip(methods, colors):
            life_values = []
            for sm in sigma_m_values:
                try:
                    life = self.predict_life(sigma_a, sm, method)
                    life_values.append(life)
                except:
                    life_values.append(np.nan)
            plt.semilogy(sigma_m_values, life_values, color + '-', label=method, linewidth=2)
        
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('平均应力 σ_m (MPa)')
        plt.ylabel('预测疲劳寿命 N (次)')
        plt.title('寿命预测对比')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt

    def evaluate_multiaxial(self, load_history, method='findley'):
        self._init_multiaxial()
        
        if method == 'findley':
            result = self.multiaxial.findley_criterion(load_history)
            life = self.multiaxial.predict_life_findley(load_history)
            result['life'] = life
            return result
        elif method == 'sws':
            result = self.multiaxial.critical_plane_sws(load_history)
            life = self.multiaxial.predict_life_sws(load_history)
            result['life'] = life
            return result
        elif method == 'brown_miller':
            result = self.multiaxial.brown_miller_criterion(load_history)
            return result
        else:
            raise ValueError(f"未知方法: {method}")

    def evaluate_tension_torsion(self, sigma_amp, tau_amp, sigma_mean=0, tau_mean=0,
                               phase_shift=0, method='findley'):
        load = LoadHistory()
        load.generate_cyclic_load(
            n_cycles=2, n_points_per_cycle=36,
            sigma_axial_amp=sigma_amp, sigma_axial_mean=sigma_mean,
            tau_torsion_amp=tau_amp, tau_torsion_mean=tau_mean,
            phase_shift=phase_shift
        )
        return self.evaluate_multiaxial(load, method)

    def compare_multiaxial_methods(self, sigma_amp, tau_amp, sigma_mean=0, tau_mean=0):
        self._init_multiaxial()
        
        load = LoadHistory()
        load.generate_cyclic_load(
            n_cycles=2, n_points_per_cycle=36,
            sigma_axial_amp=sigma_amp, sigma_axial_mean=sigma_mean,
            tau_torsion_amp=tau_amp, tau_torsion_mean=tau_mean,
            phase_shift=0
        )
        
        print("\n" + "=" * 70)
        print(f"多轴疲劳方法对比 (σ_amp={sigma_amp}, τ_amp={tau_amp}")
        print("=" * 70)
        
        result_findley = self.multiaxial.findley_criterion(load)
        life_findley = self.multiaxial.predict_life_findley(load)
        print(f"Findley准则:")
        print(f"  损伤参数: {result_findley['max_damage']:.2f} MPa")
        print(f"  临界平面: θ={result_findley['critical_theta']:.1f}°, φ={result_findley['critical_phi']:.1f}°")
        print(f"  预测寿命: {life_findley:.2e} 次")
        
        result_sws = self.multiaxial.critical_plane_sws(load)
        life_sws = self.multiaxial.predict_life_sws(load)
        print(f"\nSWS临界平面法:")
        print(f"  损伤参数: {result_sws['max_sws']:.2f} MPa")
        print(f"  临界平面: θ={result_sws['critical_theta']:.1f}°, φ={result_sws['critical_phi']:.1f}°")
        print(f"  预测寿命: {life_sws:.2e} 次")
        
        von_mises_amp = np.sqrt(sigma_amp**2 + 3 * tau_amp**2)
        life_von_mises = 10 ** ((np.log10(von_mises_amp) - np.log10(self.sigma_f)) / self.b)
        print(f"\nvon Mises等效应力:")
        print(f"  等效应力幅: {von_mises_amp:.2f} MPa")
        print(f"  预测寿命: {life_von_mises:.2e} 次")
        print("=" * 70)
        
        return {
            'findley': {'life': life_findley, 'result': result_findley},
            'sws': {'life': life_sws, 'result': result_sws},
            'von_mises': {'life': life_von_mises, 'stress': von_mises_amp}
        }

    def plot_tension_torsion_sweep(self, sigma_amp=300, tau_ratio_range=(0, 2), save_path=None):
        self._init_multiaxial()
        
        ratios, lives_findley, lives_sws = self.multiaxial.equibiaxial_ratio_analysis(
            sigma_amp, tau_ratio_range
        )
        
        plt.figure(figsize=(10, 6))
        plt.semilogy(ratios, lives_findley, 'b-o', linewidth=2, markersize=6, label='Findley')
        plt.semilogy(ratios, lives_sws, 'r-s', linewidth=2, markersize=6, label='SWS临界平面')
        
        lives_von = []
        for r in ratios:
            von = np.sqrt(sigma_amp**2 + 3 * (sigma_amp * r)**2)
            life = 10 ** ((np.log10(von) - np.log10(self.sigma_f)) / self.b)
            lives_von.append(life)
        plt.semilogy(ratios, lives_von, 'g--', linewidth=2, label='von Mises')
        
        plt.xlabel('剪应力比 τ/σ', fontsize=12)
        plt.ylabel('疲劳寿命 N (次)', fontsize=12)
        plt.title(f'拉扭组合疲劳寿命 (σ_amp={sigma_amp} MPa)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt

    def plot_load_history(self, load_history, save_path=None):
        times = load_history.time_points
        sigma_xx = [s.sigma_xx for s in load_history.stress_states]
        tau_xy = [s.tau_xy for s in load_history.stress_states]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        ax1.plot(times, sigma_xx, 'b-', linewidth=2, label='σ_xx')
        ax1.set_ylabel('轴向应力 (MPa)', fontsize=11)
        ax1.set_title('载荷时间历程', fontsize=13)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.plot(times, tau_xy, 'r-', linewidth=2, label='τ_xy')
        ax2.set_xlabel('时间 (rad)', fontsize=11)
        ax2.set_ylabel('剪切应力 (MPa)', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt

    def get_equation_string(self):
        if self.sigma_f is None or self.b is None:
            return "未进行拟合"
        return f"σ = {self.sigma_f:.4f} × (2N)^({self.b:.4f})"

    def plot_curve(self, show=True, save_path=None):
        if self.stress_amplitudes is None or self.cycles is None:
            raise ValueError("请先加载数据")

        plt.figure(figsize=(10, 6))
        plt.loglog(self.cycles, self.stress_amplitudes, 'ro', markersize=8, label='试验数据')

        if self.sigma_f is not None and self.b is not None:
            N_fit = np.logspace(np.log10(min(self.cycles) * 0.5), 
                                np.log10(max(self.cycles) * 2), 100)
            sigma_fit = self.predict_stress(N_fit)
            plt.loglog(N_fit, sigma_fit, 'b-', linewidth=2, 
                      label=f'Basquin拟合: σ = {self.sigma_f:.2f}×(2N)^{self.b:.4f}')

        plt.xlabel('循环次数 N', fontsize=12)
        plt.ylabel('应力幅 σ (MPa)', fontsize=12)
        plt.title('S-N曲线 (Basquin方程)', fontsize=14)
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.legend(fontsize=10)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return plt

    def print_results(self):
        print("=" * 60)
        print("S-N曲线拟合结果 (Basquin方程)")
        print("=" * 60)
        if self.sigma_f is not None and self.b is not None:
            print(f"疲劳强度系数 σ_f': {self.sigma_f:.4f} MPa")
            print(f"疲劳强度指数 b: {self.b:.4f}")
            print(f"决定系数 R²: {self.r_squared:.6f}")
            print(f"拟合方程: {self.get_equation_string()}")
        else:
            print("未进行拟合")
        print("=" * 60)


def main():
    print("S-N曲线拟合工具 (Basquin方程 + 平均应力修正 + 多轴疲劳)")
    print("=" * 65)
    
    try:
        sigma_uts = 1000
        sigma_yield = 800
        fitter = SNCurveFitter(sigma_uts=sigma_uts, sigma_yield=sigma_yield, walker_gamma=0.5)
        
        print("\n[1] S-N曲线拟合...")
        example_stress = [500, 450, 400, 350, 300, 280, 260, 250]
        example_cycles = [1e4, 2e4, 5e4, 1.2e5, 4e5, 8e5, 2e6, 5e6]
        
        fitter.load_data(example_stress, example_cycles)
        sigma_f, b, r2 = fitter.fit()
        fitter.print_results()
        
        print(f"\n材料参数:")
        print(f"  抗拉强度 σ_uts: {sigma_uts} MPa")
        print(f"  屈服强度 σ_yield: {sigma_yield} MPa")
        
        print("\n[2] 平均应力修正方法对比...")
        test_sigma_a = 300
        fitter.compare_correction_methods(test_sigma_a)
        
        print("\n[3] 压应力修正效果演示...")
        print("-" * 60)
        print(f"{'平均应力':>12} {'R比':>8} {'Goodman寿命':>15} {'Walker寿命':>15} {'改进Goodman':>15}")
        print("-" * 60)
        
        sigma_a = 300
        for sigma_m in [-400, -300, -200, -100, 0, 100, 200, 300]:
            try:
                sigma_max = sigma_a + sigma_m
                sigma_min = sigma_m - sigma_a
                if abs(sigma_min) < 1e-10:
                    R = 'inf' if sigma_max > 0 else '1.0'
                else:
                    R = f"{sigma_min / sigma_max:.2f}"
                
                life_goodman = fitter.predict_life(sigma_a, sigma_m, 'goodman')
                life_walker = fitter.predict_life(sigma_a, sigma_m, 'walker')
                life_improved = fitter.predict_life(sigma_a, sigma_m, 'goodman_improved')
                
                print(f"{sigma_m:>12.0f} {R:>8} {life_goodman:>15.2e} {life_walker:>15.2e} {life_improved:>15.2e}")
            except:
                print(f"{sigma_m:>12.0f} {'-':>8} {'N/A':>15} {'N/A':>15} {'N/A':>15}")
        
        print("\n[4] 多轴疲劳评估 (拉扭组合)...")
        sigma_amp = 300
        tau_amp = 200
        
        fitter.compare_multiaxial_methods(sigma_amp, tau_amp)
        
        print("\n[5] 不同剪应力比的拉扭组合分析...")
        print("-" * 55)
        print(f"{'剪应力比':>10} {'Findley寿命':>15} {'SWS寿命':>15} {'von Mises寿命':>15}")
        print("-" * 55)
        
        for ratio in [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
            result = fitter.evaluate_tension_torsion(sigma_amp, tau_amp=sigma_amp * ratio)
            von = np.sqrt(sigma_amp**2 + 3 * (sigma_amp * ratio)**2)
            life_von = 10 ** ((np.log10(von) - np.log10(fitter.sigma_f)) / fitter.b)
            print(f"{ratio:>10.2f} {result['life']:>15.2e} {result['life']:>15.2e} {life_von:>15.2e}")
        
        print("\n[6] 临界平面分析示例...")
        result = fitter.evaluate_tension_torsion(300, 300, method='findley')
        print(f"  Findley临界平面: θ={result['critical_theta']:.1f}°, φ={result['critical_phi']:.1f}°")
        print(f"  剪应力幅 τ_amp: {result['tau_amp']:.1f} MPa")
        print(f"  最大法向应力 σ_n_max: {result['sigma_n_max']:.1f} MPa")
        print(f"  预测寿命: {result['life']:.2e} 次")
        
        print("\n[7] 绘制图表...")
        fitter.plot_correction_comparison(test_sigma_a, sigma_m_range=(-500, 500), 
                                          save_path='correction_comparison.png')
        print("  修正方法对比图 → correction_comparison.png")
        
        fitter.plot_tension_torsion_sweep(sigma_amp=300, tau_ratio_range=(0, 2),
                                           save_path='tension_torsion_sweep.png')
        print("  拉扭组合寿命图 → tension_torsion_sweep.png")
        
        load = LoadHistory()
        load.generate_cyclic_load(n_cycles=3, n_points_per_cycle=36,
                                  sigma_axial_amp=300, tau_torsion_amp=200)
        fitter.plot_load_history(load, save_path='load_history.png')
        print("  载荷历程图 → load_history.png")
        
        fitter.plot_curve(show=False, save_path='sn_curve_plot.png')
        print("  S-N曲线图 → sn_curve_plot.png")
        
        print("\n✓ 全部演示完成！")
        print("\n" + "=" * 65)
        print("功能汇总:")
        print("  ✓ 单轴S-N曲线拟合 (Basquin方程)")
        print("  ✓ 5种平均应力修正方法 (含压应力处理)")
        print("  ✓ Findley多轴疲劳准则")
        print("  ✓ SWS临界平面法")
        print("  ✓ Brown-Miller准则")
        print("  ✓ 拉扭组合载荷分析")
        print("  ✓ 临界平面识别")
        print("\n快速使用:")
        print("  fitter.evaluate_tension_torsion(sigma_amp=300, tau_amp=200)")
        
    except Exception as e:
        import traceback
        print(f"\n错误: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()

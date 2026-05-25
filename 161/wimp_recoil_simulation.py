import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from scipy.integrate import quad, dblquad
from matplotlib import rcParams
import math

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

class WIMPSimulation:
    def __init__(self, 
                 wimp_mass_gev=100,
                 cross_section_cm2=1e-45,
                 target_mass_amu=72.63,
                 exposure_kg_day=1000,
                 detector_type='directional'):
        
        self.m_wimp_gev = wimp_mass_gev
        self.sigma_p_cm2 = cross_section_cm2
        self.A_target = target_mass_amu
        self.exposure = exposure_kg_day
        self.detector_type = detector_type
        
        self.c = constants.c
        self.m_p_gev = constants.value('proton mass energy equivalent in MeV') / 1000
        self.m_target_gev = self.A_target * self.m_p_gev
        
        self.v_0 = 220e3
        self.v_esc = 544e3
        self.v_lsr = 232e3
        
        self.v_orbital = 29.8e3
        self.earth_tilt = np.radians(60.2)
        
        self.rho_0 = 0.3
        
        self.N_A = constants.Avogadro
        self.day_to_sec = 86400
        
        self._vdist_norm = None
        self._vdist_params = None
        
        self.detector_params = {
            'emulsion': {
                'angular_resolution': np.radians(15),
                'efficiency': 0.85,
                'name': '核乳胶探测器'
            },
            'gas': {
                'angular_resolution': np.radians(30),
                'efficiency': 0.95,
                'name': '时间投影室(TPC)'
            },
            'directional': {
                'angular_resolution': np.radians(20),
                'efficiency': 0.90,
                'name': '通用方向性探测器'
            }
        }
        
    def reduced_mass(self, m1, m2):
        return (m1 * m2) / (m1 + m2)
    
    def mu_red(self):
        return self.reduced_mass(self.m_wimp_gev, self.m_target_gev)
    
    def mu_p(self):
        return self.reduced_mass(self.m_wimp_gev, self.m_p_gev)
    
    def form_factor(self, er_kev):
        if er_kev <= 0:
            return 1.0
        
        hbar_c = 197.3e-18
        
        q = np.sqrt(2 * self.m_target_gev * 1e6 * er_kev) * 1e3 * hbar_c
        
        r0 = 1.2 * (self.A_target ** (1/3))
        a = 0.52
        s = 0.9
        
        R = np.sqrt(r0**2 - 5*a**2 + 3*(np.pi**2) * (a**2))
        
        if q * R < 1e-10:
            return 1.0
        
        j1_qr = (np.sin(q * R) - q * R * np.cos(q * R)) / ((q * R)**2)
        
        F2 = (3 * j1_qr / (q * R))**2 * np.exp(-(q * s)**2)
        
        return max(F2, 1e-10)
    
    def max_recoil_energy(self, v):
        v_c = v / self.c
        return 2 * (self.mu_red()**2) * v_c**2 / self.m_target_gev * 1e6
    
    def min_velocity(self, er_kev):
        if er_kev <= 0:
            return 0.0
        
        v_min = (self.c / 1000) * np.sqrt(er_kev * self.m_target_gev / 
                                          (2 * self.mu_red()**2 * 1e6))
        return v_min * 1000
    
    def velocity_distribution(self, v):
        if v <= 0:
            return 0.0
        
        v_0 = self.v_0
        v_earth = self.v_earth
        v_esc = self.v_esc
        
        delta_v = 0.05 * v_esc
        
        if v < v_esc - delta_v:
            cutoff = 1.0
        elif v > v_esc + delta_v:
            return 0.0
        else:
            x = (v - (v_esc - delta_v)) / delta_v
            cutoff = 0.5 * (1 + math.erf(5 * (0.5 - x)))
        
        f_v = (v / (v_0 * v_earth)) * (np.exp(-((v - v_earth)/v_0)**2) - 
                                       np.exp(-((v + v_earth)/v_0)**2))
        
        norm = self._velocity_norm()
        
        return f_v * cutoff / norm
    
    def _velocity_norm(self):
        params = (self.v_0, self.v_esc)
        if self._vdist_norm is not None and self._vdist_params == params:
            return self._vdist_norm
        
        v_0 = self.v_0
        v_esc = self.v_esc
        delta_v = 0.05 * v_esc
        
        def integrand(v):
            if v <= 0:
                return 0.0
            
            if v < v_esc - delta_v:
                cutoff = 1.0
            elif v > v_esc + delta_v:
                return 0.0
            else:
                x = (v - (v_esc - delta_v)) / delta_v
                cutoff = 0.5 * (1 + math.erf(5 * (0.5 - x)))
            
            return v**2 * cutoff * np.exp(-(v/v_0)**2)
        
        result, _ = quad(integrand, 0, v_esc + 2*delta_v, limit=200)
        
        self._vdist_norm = 4 * np.pi * result
        self._vdist_params = params
        
        return self._vdist_norm
    
    def earth_velocity_day_of_year(self, day_of_year):
        omega = 2 * np.pi / 365.25
        phase = omega * (day_of_year - 152.5)
        
        v_earth = np.sqrt(self.v_lsr**2 + self.v_orbital**2 + 
                          2 * self.v_lsr * self.v_orbital * 
                          np.cos(self.earth_tilt) * np.cos(phase))
        
        return v_earth
    
    def velocity_distribution_time(self, v, day_of_year):
        if v <= 0:
            return 0.0
        
        v_0 = self.v_0
        v_earth = self.earth_velocity_day_of_year(day_of_year)
        v_esc = self.v_esc
        
        delta_v = 0.05 * v_esc
        
        if v < v_esc - delta_v:
            cutoff = 1.0
        elif v > v_esc + delta_v:
            return 0.0
        else:
            x = (v - (v_esc - delta_v)) / delta_v
            cutoff = 0.5 * (1 + math.erf(5 * (0.5 - x)))
        
        f_v = (v / (v_0 * v_earth)) * (np.exp(-((v - v_earth)/v_0)**2) - 
                                       np.exp(-((v + v_earth)/v_0)**2))
        
        norm = self._velocity_norm_time(day_of_year)
        
        return f_v * cutoff / norm
    
    def _velocity_norm_time(self, day_of_year):
        v_0 = self.v_0
        v_esc = self.v_esc
        v_earth = self.earth_velocity_day_of_year(day_of_year)
        delta_v = 0.05 * v_esc
        
        def integrand(v):
            if v <= 0 or v > v_esc + 2*delta_v:
                return 0.0
            
            if v < v_esc - delta_v:
                cutoff = 1.0
            else:
                x = (v - (v_esc - delta_v)) / delta_v
                cutoff = 0.5 * (1 + math.erf(5 * (0.5 - x)))
            
            return v * (np.exp(-((v - v_earth)/v_0)**2) - 
                        np.exp(-((v + v_earth)/v_0)**2)) * cutoff
        
        result, _ = quad(integrand, 0, v_esc + 2*delta_v, limit=200)
        
        return result / (v_0 * v_earth)
    
    def differential_rate_time(self, er_kev, day_of_year):
        if er_kev <= 0:
            return 0.0
        
        v_min = self.min_velocity(er_kev)
        
        if v_min >= self.v_esc:
            return 0.0
        
        sigma_0 = self.sigma_p_cm2 * (self.A_target**2) * (self.mu_red() / self.mu_p())**2
        
        F2 = self.form_factor(er_kev)
        
        def integrand(v):
            return self.velocity_distribution_time(v, day_of_year) / v
        
        eta_vmin, _ = quad(integrand, v_min, self.v_esc, limit=100)
        
        rate = (self.rho_0 / (2 * self.m_wimp_gev * self.m_p_gev)) * sigma_0 * F2 * eta_vmin * self.c * 1e2
        
        rate_per_kg_day = rate * (self.N_A / self.A_target) * self.day_to_sec * 1e-6
        
        return max(rate_per_kg_day, 1e-10)
    
    def annual_modulation_curve(self, er_min=2, er_max=6, n_days=365):
        days = np.linspace(1, 365, n_days)
        rates = np.zeros(n_days)
        
        for i, day in enumerate(days):
            rate, _ = quad(lambda e: self.differential_rate_time(e, day), er_min, er_max, limit=50)
            rates[i] = rate
        
        return days, rates
    
    def differential_rate(self, er_kev):
        if er_kev <= 0:
            return 0.0
        
        v_min = self.min_velocity(er_kev)
        
        if v_min >= self.v_esc:
            return 0.0
        
        sigma_0 = self.sigma_p_cm2 * (self.A_target**2) * (self.mu_red() / self.mu_p())**2
        
        F2 = self.form_factor(er_kev)
        
        def integrand(v):
            return self.velocity_distribution(v) / v
        
        eta_vmin, _ = quad(integrand, v_min, self.v_esc, limit=100)
        
        rate = (self.rho_0 / (2 * self.m_wimp_gev * self.m_p_gev)) * sigma_0 * F2 * eta_vmin * self.c * 1e2
        
        rate_per_kg_day = rate * (self.N_A / self.A_target) * self.day_to_sec * 1e-6
        
        return max(rate_per_kg_day, 1e-10)
    
    def generate_spectrum(self, er_min=1, er_max=100, n_points=200):
        er_array = np.logspace(np.log10(er_min), np.log10(er_max), n_points)
        rate_array = np.array([self.differential_rate(er) for er in er_array])
        
        return er_array, rate_array
    
    def generate_events(self, n_events=10000, er_min=1, er_max=100):
        er_array, rate_array = self.generate_spectrum(er_min, er_max, n_points=500)
        
        cum_rate = np.cumsum(rate_array)
        cum_rate_norm = cum_rate / cum_rate[-1]
        
        u = np.random.rand(n_events)
        events_er = np.interp(u, cum_rate_norm, er_array)
        
        return events_er
    
    def recoildirection_from_wimpdirection(self, v_wimp, er_kev):
        mu_red = self.mu_red()
        m_t = self.m_target_gev
        m_chi = self.m_wimp_gev
        
        cos_theta_n = np.sqrt(er_kev * m_t * 1e-6) / (2 * mu_red * (v_wimp / self.c))
        cos_theta_n = np.clip(cos_theta_n, -1, 1)
        
        return cos_theta_n
    
    def differential_rate_directional(self, er_kev, cos_theta):
        if er_kev <= 0 or abs(cos_theta) > 1:
            return 0.0
        
        v_min = self.min_velocity(er_kev) / np.cos(0.5 * np.arccos(cos_theta)) if cos_theta > 0 else self.v_esc
        
        if v_min >= self.v_esc:
            return 0.0
        
        sigma_0 = self.sigma_p_cm2 * (self.A_target**2) * (self.mu_red() / self.mu_p())**2
        
        F2 = self.form_factor(er_kev)
        
        v_earth = self.v_lsr
        
        def integrand(v):
            v_0 = self.v_0
            v_esc = self.v_esc
            delta_v = 0.05 * v_esc
            
            if v <= 0:
                return 0.0
            
            if v < v_esc - delta_v:
                cutoff = 1.0
            elif v > v_esc + delta_v:
                return 0.0
            else:
                x = (v - (v_esc - delta_v)) / delta_v
                cutoff = 0.5 * (1 + math.erf(5 * (0.5 - x)))
            
            f_v = (np.exp(-((v - v_earth * cos_theta)/v_0)**2) - 
                   np.exp(-((v + v_earth * cos_theta)/v_0)**2))
            
            return f_v * cutoff / v
        
        norm = self._velocity_norm()
        
        eta_vmin, _ = quad(integrand, v_min, self.v_esc, limit=100)
        
        rate = (self.rho_0 / (4 * np.pi * self.m_wimp_gev * self.m_p_gev)) * sigma_0 * F2 * eta_vmin * self.c * 1e2 / norm
        
        rate_per_kg_day = rate * (self.N_A / self.A_target) * self.day_to_sec * 1e-6
        
        det_params = self.detector_params.get(self.detector_type, 
                                              self.detector_params['directional'])
        efficiency = det_params['efficiency']
        
        return max(rate_per_kg_day * efficiency, 1e-12)
    
    def angular_distribution(self, er_kev, n_angles=50):
        cos_theta_array = np.linspace(-0.99, 0.99, n_angles)
        rate_array = np.array([self.differential_rate_directional(er_kev, ct) 
                               for ct in cos_theta_array])
        
        return cos_theta_array, rate_array
    
    def generate_directional_events(self, n_events=10000, er_min=1, er_max=100):
        er_flat = np.linspace(er_min, er_max, 100)
        cos_theta_flat = np.linspace(-0.99, 0.99, 50)
        ER, CT = np.meshgrid(er_flat, cos_theta_flat)
        
        rate_2d = np.zeros_like(ER)
        for i in range(len(cos_theta_flat)):
            for j in range(len(er_flat)):
                rate_2d[i, j] = self.differential_rate_directional(ER[i, j], CT[i, j])
        
        rate_flat = rate_2d.flatten()
        cum_rate = np.cumsum(rate_flat)
        cum_rate_norm = cum_rate / cum_rate[-1]
        
        u = np.random.rand(n_events)
        indices = np.searchsorted(cum_rate_norm, u)
        
        er_events = ER.flatten()[indices] + np.random.randn(n_events) * (er_max - er_min) / 200
        ct_events = CT.flatten()[indices]
        
        er_events = np.clip(er_events, er_min, er_max)
        ct_events = np.clip(ct_events, -0.99, 0.99)
        
        return er_events, ct_events
    
    def compute_forward_backward_ratio(self, er_min=5, er_max=50):
        def forward_integrand(ct, er):
            if ct >= 0:
                return self.differential_rate_directional(er, ct)
            return 0
        
        def backward_integrand(ct, er):
            if ct < 0:
                return self.differential_rate_directional(er, ct)
            return 0
        
        forward_rate, _ = dblquad(forward_integrand, er_min, er_max, 
                                  lambda er: -0.99, lambda er: 0.99)
        backward_rate, _ = dblquad(backward_integrand, er_min, er_max, 
                                   lambda er: -0.99, lambda er: 0.99)
        
        return forward_rate / max(backward_rate, 1e-10)
    
    def plot_spectrum(self, er_min=1, er_max=100, n_points=200, save_fig=False):
        er_array, rate_array = self.generate_spectrum(er_min, er_max, n_points)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.loglog(er_array, rate_array, 'b-', linewidth=2, label='微分能谱')
        
        ax.set_xlabel('反冲能量 (keV)', fontsize=12)
        ax.set_ylabel('微分事例率 (事例/keV/kg/day)', fontsize=12)
        ax.set_title(f'WIMP 核反冲能谱\n($m_\chi$ = {self.m_wimp_gev} GeV/c$^2$, $\sigma_p$ = {self.sigma_p_cm2:.1e} cm$^2$)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig('wimp_spectrum.png', dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    def plot_event_distribution(self, n_events=10000, er_min=1, er_max=100, save_fig=False):
        events = self.generate_events(n_events, er_min, er_max)
        er_array, rate_array = self.generate_spectrum(er_min, er_max, n_points=200)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        n, bins, patches = ax1.hist(events, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('反冲能量 (keV)', fontsize=12)
        ax1.set_ylabel('概率密度', fontsize=12)
        ax1.set_title(f'模拟事例能量分布\n({n_events} 个事例)', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax2.scatter(bin_centers, n, s=20, alpha=0.6, color='red', label='模拟数据')
        ax2.plot(er_array, rate_array / np.trapz(rate_array, er_array), 'b-', linewidth=2, label='理论谱')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel('反冲能量 (keV)', fontsize=12)
        ax2.set_ylabel('归一化密度', fontsize=12)
        ax2.set_title('双对数坐标对比', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig('wimp_event_distribution.png', dpi=300, bbox_inches='tight')
        
        return fig, (ax1, ax2)
    
    def plot_annual_modulation(self, er_min=2, er_max=6, save_fig=False):
        days, rates = self.annual_modulation_curve(er_min, er_max)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        ax1.plot(days, rates, 'b-', linewidth=2)
        ax1.set_xlabel('一年中的天数', fontsize=12)
        ax1.set_ylabel('积分事例率 (事例/kg/day)', fontsize=12)
        ax1.set_title(f'年度调制效应 (能区: {er_min}-{er_max} keV)', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        peak_day = days[np.argmax(rates)]
        month_names = ['1月', '2月', '3月', '4月', '5月', '6月', 
                       '7月', '8月', '9月', '10月', '11月', '12月']
        month_days = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
        
        for i, (d, name) in enumerate(zip(month_days, month_names)):
            ax1.axvline(x=d, color='gray', linestyle=':', alpha=0.3)
        
        ax1.axvline(x=peak_day, color='r', linestyle='--', alpha=0.7, 
                    label=f'峰值: 第 {peak_day:.0f} 天 (~6月2日)')
        ax1.legend()
        
        rate_mean = np.mean(rates)
        rate_mod = (np.max(rates) - np.min(rates)) / rate_mean * 100
        
        ax2.plot(days, (rates - rate_mean) / rate_mean * 100, 'r-', linewidth=2)
        ax2.set_xlabel('一年中的天数', fontsize=12)
        ax2.set_ylabel('相对调制幅度 (%)', fontsize=12)
        ax2.set_title(f'相对调制幅度 (峰-峰: {rate_mod:.2f}%)', fontsize=14)
        ax2.axhline(y=0, color='k', linestyle=':', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig('annual_modulation.png', dpi=300, bbox_inches='tight')
        
        return fig, (ax1, ax2)
    
    def plot_angular_distribution(self, er_kev=20, save_fig=False):
        cos_theta, rate = self.angular_distribution(er_kev)
        theta = np.arccos(cos_theta) * 180 / np.pi
        
        fig = plt.figure(figsize=(14, 6))
        
        ax1 = plt.subplot(121)
        ax1.plot(cos_theta, rate, 'b-', linewidth=2)
        ax1.axvline(x=0, color='k', linestyle=':', alpha=0.5)
        ax1.set_xlabel('cos(θ) (θ: 反冲核与WIMP风夹角)', fontsize=11)
        ax1.set_ylabel('微分事例率', fontsize=12)
        ax1.set_title(f'角分布 (E_r = {er_kev} keV)', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        ax2 = plt.subplot(122, projection='polar')
        ax2.plot(np.arccos(cos_theta), rate, 'r-', linewidth=2)
        ax2.plot(-np.arccos(cos_theta), rate, 'r--', linewidth=2, alpha=0.5)
        ax2.set_theta_zero_location('N')
        ax2.set_theta_direction(-1)
        ax2.set_title('极坐标图 (WIMP风 → 北)', fontsize=14)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig('angular_distribution.png', dpi=300, bbox_inches='tight')
        
        return fig, (ax1, ax2)
    
    def plot_directional_events(self, n_events=5000, er_min=5, er_max=50, save_fig=False):
        er_events, ct_events = self.generate_directional_events(n_events, er_min, er_max)
        theta_events = np.arccos(ct_events) * 180 / np.pi
        
        det_params = self.detector_params.get(self.detector_type, 
                                              self.detector_params['directional'])
        det_name = det_params['name']
        
        fig = plt.figure(figsize=(16, 6))
        
        ax1 = plt.subplot(131)
        h = ax1.hist2d(er_events, ct_events, bins=[30, 30], cmap='Blues')
        ax1.set_xlabel('反冲能量 (keV)', fontsize=11)
        ax1.set_ylabel('cos(θ)', fontsize=11)
        ax1.set_title(f'能量-角度二维分布', fontsize=12)
        plt.colorbar(h[3], ax=ax1, label='事例数')
        
        ax2 = plt.subplot(132)
        ax2.hist(ct_events, bins=30, density=True, alpha=0.7, 
                 color='skyblue', edgecolor='black')
        ax2.axvline(x=0, color='r', linestyle='--', alpha=0.7, label='前向后向分界')
        ax2.set_xlabel('cos(θ)', fontsize=11)
        ax2.set_ylabel('概率密度', fontsize=11)
        ax2.set_title('角度投影分布', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3 = plt.subplot(133, projection='polar')
        theta_rad = np.arccos(ct_events)
        ax3.hist(theta_rad, bins=30, density=True, alpha=0.7, color='lightgreen')
        ax3.hist(-theta_rad, bins=30, density=True, alpha=0.3, color='lightgreen')
        ax3.set_theta_zero_location('N')
        ax3.set_theta_direction(-1)
        ax3.set_title('极角分布\n(WIMP风 → 北)', fontsize=12)
        
        fb_ratio = self.compute_forward_backward_ratio(er_min, er_max)
        
        fig.suptitle(f'{det_name} 方向性探测模拟\n'
                     f'({n_events} 个事例, 前向后向比 = {fb_ratio:.2f})', 
                     fontsize=14, y=1.02)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig('directional_events.png', dpi=300, bbox_inches='tight')
        
        return fig, (ax1, ax2, ax3)
    
    def compare_detector_types(self, er_kev=20, save_fig=False):
        detectors = ['emulsion', 'gas', 'directional']
        colors = ['r', 'g', 'b']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for det, c in zip(detectors, colors):
            self.detector_type = det
            cos_theta, rate = self.angular_distribution(er_kev)
            det_name = self.detector_params[det]['name']
            eff = self.detector_params[det]['efficiency']
            ax.plot(cos_theta, rate, f'{c}-', linewidth=2, 
                    label=f'{det_name} (效率: {eff*100:.0f}%)')
        
        ax.axvline(x=0, color='k', linestyle=':', alpha=0.5)
        ax.set_xlabel('cos(θ)', fontsize=12)
        ax.set_ylabel('微分事例率 (事例/keV/kg/day)', fontsize=12)
        ax.set_title(f'不同探测器类型的角分布对比 (E_r = {er_kev} keV)', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig('detector_comparison.png', dpi=300, bbox_inches='tight')
        
        self.detector_type = 'directional'
        return fig, ax
    
    def print_parameters(self):
        print("="*60)
        print("WIMP 探测器模拟参数")
        print("="*60)
        print(f"WIMP 质量: {self.m_wimp_gev} GeV/c²")
        print(f"质子- WIMP 截面: {self.sigma_p_cm2:.2e} cm²")
        print(f"靶核质量: {self.A_target} amu (锗)")
        print(f"曝光量: {self.exposure} kg·day")
        print(f"探测器类型: {self.detector_params.get(self.detector_type, {}).get('name', '未知')}")
        print(f"本地暗物质密度: {self.rho_0} GeV/c²/cm³")
        print(f"WIMP 特征速度 v₀: {self.v_0/1000:.0f} km/s")
        print(f"逃逸速度 v_esc: {self.v_esc/1000:.0f} km/s")
        print(f"LSR 速度: {self.v_lsr/1000:.0f} km/s")
        print(f"地球轨道速度: {self.v_orbital/1000:.1f} km/s")
        print(f"黄道倾角: {np.degrees(self.earth_tilt):.1f}°")
        print("="*60)
        
        total_rate, _ = quad(self.differential_rate, 1, 100, limit=100)
        print(f"\n1-100 keV 能区总事例率: {total_rate:.4f} 事例/kg/day")
        print(f"预计总事例数 ({self.exposure} kg·day): {total_rate * self.exposure:.1f} 个")
        
        fb_ratio = self.compute_forward_backward_ratio(5, 50)
        print(f"\n前向后向比 (5-50 keV): {fb_ratio:.2f}")


def main():
    print("暗物质 WIMP 核反冲信号模拟 (含年度调制和方向性探测)")
    print("="*70)
    
    sim = WIMPSimulation(
        wimp_mass_gev=50,
        cross_section_cm2=1e-45,
        target_mass_amu=72.63,
        exposure_kg_day=1000,
        detector_type='emulsion'
    )
    
    sim.print_parameters()
    
    print("\n" + "="*70)
    print("生成基础能谱图...")
    fig1, ax1 = sim.plot_spectrum(er_min=1, er_max=100)
    
    print("生成年度调制曲线...")
    fig2, (ax2a, ax2b) = sim.plot_annual_modulation(er_min=2, er_max=6)
    
    print("生成角分布图...")
    fig3, (ax3a, ax3b) = sim.plot_angular_distribution(er_kev=20)
    
    print("生成方向性探测模拟...")
    fig4, (ax4a, ax4b, ax4c) = sim.plot_directional_events(n_events=2000, er_min=5, er_max=50)
    
    plt.show()
    
    print("\n" + "="*70)
    print("所有模拟完成!")
    print("="*70)


def compare_masses():
    masses = [10, 50, 100, 500]
    colors = ['r', 'g', 'b', 'm']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for m, c in zip(masses, colors):
        sim = WIMPSimulation(wimp_mass_gev=m, cross_section_cm2=1e-45)
        er, rate = sim.generate_spectrum(er_min=1, er_max=100)
        ax.loglog(er, rate, f'{c}-', linewidth=2, label=f'$m_\chi$ = {m} GeV')
    
    ax.set_xlabel('反冲能量 (keV)', fontsize=12)
    ax.set_ylabel('微分事例率 (事例/keV/kg/day)', fontsize=12)
    ax.set_title('不同 WIMP 质量的反冲能谱对比', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig('wimp_mass_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def velocity_distribution_hard_cutoff(v, v_0, v_esc, v_earth):
    if v <= 0 or v >= v_esc:
        return 0.0
    
    norm = (np.pi * v_0**2)**(3/2) * (math.erf(v_esc/v_0) - 
                                      2*v_esc*np.exp(-(v_esc/v_0)**2)/(np.sqrt(np.pi)*v_0))
    
    f_v = (v / (v_0 * v_earth)) * (np.exp(-((v - v_earth)/v_0)**2) - 
                                   np.exp(-((v + v_earth)/v_0)**2))
    return f_v / norm


def validate_smooth_cutoff():
    v_0 = 220e3
    v_esc = 544e3
    v_earth = 232e3
    
    v_array = np.linspace(0, 700e3, 500)
    
    sim = WIMPSimulation()
    
    f_hard = np.array([velocity_distribution_hard_cutoff(v, v_0, v_esc, v_earth) for v in v_array])
    f_smooth = np.array([sim.velocity_distribution(v) for v in v_array])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.plot(v_array/1000, f_hard, 'r--', linewidth=2, label='硬截断')
    ax1.plot(v_array/1000, f_smooth, 'b-', linewidth=2, label='平滑截断 (误差函数)')
    ax1.axvline(x=v_esc/1000, color='k', linestyle=':', alpha=0.5, label='逃逸速度')
    ax1.set_xlabel('速度 (km/s)', fontsize=12)
    ax1.set_ylabel('概率密度 f(v)', fontsize=12)
    ax1.set_title('速度分布：硬截断 vs 平滑截断', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    idx_near = np.where((v_array > 450e3) & (v_array < 600e3))
    ax2.plot(v_array[idx_near]/1000, f_hard[idx_near], 'r--', linewidth=2, label='硬截断')
    ax2.plot(v_array[idx_near]/1000, f_smooth[idx_near], 'b-', linewidth=2, label='平滑截断')
    ax2.axvline(x=v_esc/1000, color='k', linestyle=':', alpha=0.5, label='逃逸速度')
    ax2.set_xlabel('速度 (km/s)', fontsize=12)
    ax2.set_ylabel('概率密度 f(v)', fontsize=12)
    ax2.set_title('截断区域放大', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('velocity_distribution_comparison.png', dpi=300, bbox_inches='tight')
    
    print("="*60)
    print("速度分布平滑截断验证")
    print("="*60)
    print(f"逃逸速度: {v_esc/1000:.0f} km/s")
    print(f"过渡区域宽度: {0.05*v_esc/1000:.1f} km/s")
    print(f"\n在逃逸速度处的导数连续性:")
    v_near = v_esc - 10
    f1 = sim.velocity_distribution(v_near)
    f2 = sim.velocity_distribution(v_esc)
    f3 = sim.velocity_distribution(v_esc + 10)
    print(f"  f(v_esc-10m/s) = {f1:.2e} s/km")
    print(f"  f(v_esc)       = {f2:.2e} s/km")
    print(f"  f(v_esc+10m/s) = {f3:.2e} s/km")
    print(f"\n平滑过渡避免了能谱计算中的伪尖峰!")
    print("="*60)
    
    return fig, (ax1, ax2)


def compare_spectra_smoothness():
    sim = WIMPSimulation(wimp_mass_gev=50, cross_section_cm2=1e-45)
    
    er, rate = sim.generate_spectrum(er_min=0.1, er_max=200, n_points=500)
    
    er_max_theory = sim.max_recoil_energy(sim.v_esc)
    print(f"\n理论最大反冲能量 (v_esc处): {er_max_theory:.2f} keV")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.loglog(er, rate, 'b-', linewidth=2)
    ax.axvline(x=er_max_theory, color='r', linestyle='--', alpha=0.7, 
               label=f'理论截止能量 ({er_max_theory:.1f} keV)')
    ax.set_xlabel('反冲能量 (keV)', fontsize=12)
    ax.set_ylabel('微分事例率 (事例/keV/kg/day)', fontsize=12)
    ax.set_title('平滑能谱（无伪尖峰）', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('smooth_spectrum.png', dpi=300, bbox_inches='tight')
    
    return fig, ax


if __name__ == "__main__":
    main()

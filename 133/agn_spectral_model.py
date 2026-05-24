"""
AGN综合光谱模型（增强版）
==========================

包含:
1. Novikov-Thorne薄盘热连续谱
2. 相对论性铁Kα线展宽（引力红移+多普勒效应）
3. 盘风/喷流贡献（宽线区、吸收线、外流）
4. 观测数据拟合和黑洞自旋提取

作者: 天体物理计算
日期: 2026
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize, curve_fit
from scipy.stats import norm, chisquare
import matplotlib.pyplot as plt

from novikov_thorne_disk import NovikovThorneDisk


class RelativisticIronLine:
    """
    相对论性铁Kα线模型
    
    基于Laor (1991)模型，考虑：
    - 多普勒红移/蓝移
    - 引力红移
    - 相对论性聚束（beaming）
    - 光行弯曲效应
    
    参数:
        M_BH: 黑洞质量 (M_sun)
        a_star: 自旋参数 (-1 ~ 1)
        inclination: 倾角 (度)
        line_energy: 静止系线能量 (keV)，默认6.4 keV (中性铁)
        r_in: 内发射半径 (R_g)，默认ISCO
        r_out: 外发射半径 (R_g)，默认100 R_g
        emissivity_index: 发射率指数 ε ∝ r^-q
    """
    
    def __init__(self, M_BH=1e8, a_star=0.7, inclination=30.0, 
                 line_energy=6.4, r_in=None, r_out=100.0, 
                 emissivity_index=3.0):
        self.M_BH = M_BH
        self.a_star = a_star
        self.inclination = inclination
        self.line_energy = line_energy
        self.r_out = r_out
        self.emissivity_index = emissivity_index
        
        self.c = 2.99792458e10
        self.G = 6.67430e-8
        self.M_sun = 1.98847e33
        self.eV = 1.602176634e-12
        self.keV = 1e3 * self.eV
        
        self.R_g = self.G * self.M_BH * self.M_sun / self.c**2
        self.inclination_rad = np.radians(inclination)
        
        temp_disk = NovikovThorneDisk(M_BH=M_BH, a_star=a_star)
        self.R_isco = temp_disk.R_isco
        self.R_isco_over_Rg = temp_disk.R_isco_over_Rg
        
        if r_in is None:
            self.r_in = self.R_isco_over_Rg
        else:
            self.r_in = r_in
            
    def _relativistic_factors(self, r, phi):
        """
        计算相对论性因子
        
        参数:
            r: 半径 (R_g单位)
            phi: 方位角
            
        返回:
            g: 能量红移因子 E_obs / E_em
            boost: 聚束因子
        """
        a = self.a_star
        theta = self.inclination_rad
        
        sqrt_r = np.sqrt(r)
        
        Omega = 1.0 / (r**(3.0/2.0) + a)
        beta = r**(-0.5) * (1 + a * r**(-1.5))**(-1) * np.sqrt(1 - 2/r + a**2/r**2)
        
        sin_term = np.sin(theta) * np.sin(phi)
        denom = 1 - beta * sin_term
        
        g = (1 + a * r**(-1.5)) / (sqrt_r * np.sqrt(1 - 3/r + 2*a/r**1.5))
        g = g * (1 - beta * sin_term)
        
        gamma = 1.0 / np.sqrt(1 - beta**2)
        boost = gamma**4 * (1 - beta * sin_term)**4
        
        return g, boost
    
    def emissivity(self, r):
        """发射率剖面 ε ∝ r^-q"""
        return r**(-self.emissivity_index)
    
    def _line_profile_integrand(self, phi, r, energy_grid):
        """
        线轮廓积分的被积函数
        """
        g, boost = self._relativistic_factors(r, phi)
        eps = self.emissivity(r)
        
        observed_energy = self.line_energy * g
        
        return r * eps * boost
    
    def compute_profile(self, energy_grid):
        """
        计算相对论性展宽的线轮廓
        
        参数:
            energy_grid: 能量网格 (keV)
            
        返回:
            line_profile: 归一化的线轮廓
        """
        profile = np.zeros_like(energy_grid)
        
        for i, E_obs in enumerate(energy_grid):
            def integrand(r):
                def phi_integral(phi):
                    g, boost = self._relativistic_factors(r, phi)
                    E_em = self.line_energy * g
                    
                    if abs(E_obs - E_em) < 0.01:
                        eps = self.emissivity(r)
                        return r * eps * boost
                    else:
                        return 0.0
                
                result, _ = quad(phi_integral, 0, 2*np.pi)
                return result
            
            integral, _ = quad(integrand, self.r_in, self.r_out)
            profile[i] = integral
        
        if np.max(profile) > 0:
            profile = profile / np.max(profile)
            
        return profile
    
    def compute_profile_fast(self, energy_grid, num_radii=50, num_phi=100):
        """
        快速计算线轮廓（使用数值网格）
        
        参数:
            energy_grid: 能量网格 (keV)
            num_radii: 径向采样点数
            num_phi: 方位角采样点数
            
        返回:
            line_profile: 归一化的线轮廓
        """
        r_grid = np.logspace(np.log10(self.r_in), np.log10(self.r_out), num_radii)
        phi_grid = np.linspace(0, 2*np.pi, num_phi)
        
        profile = np.zeros_like(energy_grid)
        energy_bins = np.diff(energy_grid)
        energy_bins = np.append(energy_bins, energy_bins[-1])
        
        for r in r_grid:
            for phi in phi_grid:
                g, boost = self._relativistic_factors(r, phi)
                eps = self.emissivity(r)
                
                E_obs = self.line_energy * g
                weight = r * eps * boost
                
                idx = np.searchsorted(energy_grid, E_obs)
                if idx > 0 and idx < len(energy_grid):
                    profile[idx-1] += weight
        
        if np.max(profile) > 0:
            profile = profile / np.max(profile)
            
        return profile


class DiskWindModel:
    """
    盘风/喷流模型
    
    包含:
    - 宽线区 (BLR) 发射线
    - 外流吸收线（蓝移）
    - 部分覆盖吸收体
    """
    
    def __init__(self):
        self.c = 2.99792458e5
        
        self.broad_lines = {
            'Halpha': {'lambda_rest': 6564.6, 'fwhm': 3000, 'strength': 1.0},
            'Hbeta': {'lambda_rest': 4862.7, 'fwhm': 4000, 'strength': 0.5},
            'MgII': {'lambda_rest': 2798.0, 'fwhm': 5000, 'strength': 0.8},
            'CIV': {'lambda_rest': 1549.0, 'fwhm': 6000, 'strength': 0.6},
            'Lyalpha': {'lambda_rest': 1215.7, 'fwhm': 8000, 'strength': 1.2}
        }
        
        self.absorption_lines = {
            'OVI': {'lambda_rest': 1031.9, 'velocity': -5000, 'depth': 0.3},
            'NV': {'lambda_rest': 1242.8, 'velocity': -3000, 'depth': 0.25},
            'CIV_abs': {'lambda_rest': 1548.2, 'velocity': -2000, 'depth': 0.2}
        }
        
    def add_broad_line(self, name, lambda_rest, fwhm_km_s, strength):
        """添加宽发射线"""
        self.broad_lines[name] = {
            'lambda_rest': lambda_rest,
            'fwhm': fwhm_km_s,
            'strength': strength
        }
        
    def add_absorption_line(self, name, lambda_rest, velocity_km_s, depth):
        """添加吸收线"""
        self.absorption_lines[name] = {
            'lambda_rest': lambda_rest,
            'velocity': velocity_km_s,
            'depth': depth
        }
        
    def _gaussian_line(self, wavelength_grid, lambda0, fwhm_km_s, strength):
        """高斯发射线"""
        sigma_km_s = fwhm_km_s / 2.355
        sigma_lambda = lambda0 * sigma_km_s / self.c
        
        return strength * norm.pdf(wavelength_grid, lambda0, sigma_lambda)
    
    def _absorption_feature(self, wavelength_grid, lambda_rest, velocity, depth):
        """蓝移吸收特征"""
        lambda_obs = lambda_rest * (1 + velocity / self.c)
        fwhm = abs(velocity) / 2.0
        sigma_lambda = lambda_rest * (fwhm / 2.355) / self.c
        
        absorption = depth * np.exp(-0.5 * ((wavelength_grid - lambda_obs) / sigma_lambda)**2)
        return 1.0 - absorption
    
    def compute_wind_contribution(self, wavelength_grid):
        """
        计算盘风对光谱的贡献
        
        参数:
            wavelength_grid: 波长网格 (Å)
            
        返回:
            emission_spectrum: 发射线贡献
            absorption_spectrum: 吸收线调制因子
        """
        emission = np.zeros_like(wavelength_grid)
        
        for line_name, line_params in self.broad_lines.items():
            emission += self._gaussian_line(
                wavelength_grid,
                line_params['lambda_rest'],
                line_params['fwhm'],
                line_params['strength']
            )
        
        absorption = np.ones_like(wavelength_grid)
        for line_name, line_params in self.absorption_lines.items():
            absorption *= self._absorption_feature(
                wavelength_grid,
                line_params['lambda_rest'],
                line_params['velocity'],
                line_params['depth']
            )
        
        return emission, absorption


class PowerLawContinuum:
    """
    冕区幂律连续谱（X射线）
    """
    
    def __init__(self, photon_index=1.8, normalization=1.0, cutoff_keV=100.0):
        self.photon_index = photon_index
        self.normalization = normalization
        self.cutoff_keV = cutoff_keV
        
    def compute_spectrum(self, energy_keV):
        """计算幂律谱 dN/dE ∝ E^-Γ * exp(-E/E_cut)"""
        spectrum = self.normalization * energy_keV**(-self.photon_index)
        spectrum *= np.exp(-energy_keV / self.cutoff_keV)
        return spectrum


class ComptonHump:
    """
    康普顿反射峰（~20-30 keV）
    """
    
    def __init__(self, strength=0.3, peak_energy=20.0, width=15.0):
        self.strength = strength
        self.peak_energy = peak_energy
        self.width = width
        
    def compute_spectrum(self, energy_keV):
        """计算康普顿峰"""
        gaussian = np.exp(-0.5 * ((energy_keV - self.peak_energy) / self.width)**2)
        return self.strength * gaussian


class AGNSpectralModel:
    """
    AGN综合光谱模型
    
    组件:
    1. 热连续谱 (Novikov-Thorne盘)
    2. 幂律连续谱 (冕区)
    3. 相对论性铁Kα线
    4. 康普顿反射峰
    5. 盘风发射/吸收线
    """
    
    def __init__(self, M_BH=1e8, M_dot=0.5, a_star=0.7, inclination=30.0):
        self.M_BH = M_BH
        self.M_dot = M_dot
        self.a_star = a_star
        self.inclination = inclination
        
        self.disk = NovikovThorneDisk(M_BH=M_BH, M_dot=M_dot, a_star=a_star, 
                                     inclination=inclination)
        self.iron_line = RelativisticIronLine(M_BH=M_BH, a_star=a_star, 
                                              inclination=inclination)
        self.power_law = PowerLawContinuum()
        self.compton_hump = ComptonHump()
        self.disk_wind = DiskWindModel()
        
        self.iron_line_norm = 0.1
        self.pl_norm = 1.0
        self.reflection_fraction = 0.5
        
    def set_parameters(self, **kwargs):
        """设置模型参数"""
        if 'a_star' in kwargs:
            self.a_star = kwargs['a_star']
            self.disk = NovikovThorneDisk(M_BH=self.M_BH, M_dot=self.M_dot, 
                                         a_star=self.a_star, 
                                         inclination=self.inclination)
            self.iron_line = RelativisticIronLine(M_BH=self.M_BH, 
                                                  a_star=self.a_star, 
                                                  inclination=self.inclination)
        
        if 'M_dot' in kwargs:
            self.M_dot = kwargs['M_dot']
            self.disk.M_dot = self.M_dot
            self.disk.M_dot_cgs = self.M_dot * self.disk.M_sun / self.disk.yr
            
        if 'inclination' in kwargs:
            self.inclination = kwargs['inclination']
            self.disk.inclination = self.inclination
            self.disk.inclination_rad = np.radians(self.inclination)
            self.iron_line = RelativisticIronLine(M_BH=self.M_BH, 
                                                  a_star=self.a_star, 
                                                  inclination=self.inclination)
            
        if 'iron_line_norm' in kwargs:
            self.iron_line_norm = kwargs['iron_line_norm']
        if 'photon_index' in kwargs:
            self.power_law.photon_index = kwargs['photon_index']
        if 'pl_norm' in kwargs:
            self.pl_norm = kwargs['pl_norm']
            
    def compute_xray_spectrum(self, energy_keV):
        """
        计算X射线能谱（2-10 keV）
        
        参数:
            energy_keV: 能量网格 (keV)
            
        返回:
            spectrum: 总谱
            components: 各组件的字典
        """
        disk_nu = energy_keV * 1e3 * self.disk.eV / self.disk.h
        disk_Lnu, _ = self.disk.compute_spectrum(disk_nu)
        
        disk_photons = disk_Lnu * self.disk.h / (energy_keV * 1e3 * self.disk.eV)
        disk_photons = disk_photons / np.max(disk_photons) * 0.1
        
        pl_spectrum = self.power_law.compute_spectrum(energy_keV)
        pl_spectrum = pl_spectrum / np.max(pl_spectrum) * self.pl_norm
        
        iron_profile = self.iron_line.compute_profile_fast(energy_keV)
        iron_spectrum = self.iron_line_norm * iron_profile
        
        compton_spectrum = self.compton_hump.compute_spectrum(energy_keV)
        compton_spectrum = compton_spectrum * self.reflection_fraction
        
        total = pl_spectrum + iron_spectrum + compton_spectrum + disk_photons
        
        components = {
            'power_law': pl_spectrum,
            'iron_line': iron_spectrum,
            'compton_hump': compton_spectrum,
            'disk_thermal': disk_photons
        }
        
        return total, components
    
    def compute_uv_optical_spectrum(self, wavelength_angstrom):
        """
        计算紫外-光学光谱
        
        参数:
            wavelength_angstrom: 波长网格 (Å)
            
        返回:
            spectrum: 总谱
            components: 各组件的字典
        """
        nu = self.disk.c / (wavelength_angstrom * 1e-8)
        disk_Lnu, _ = self.disk.compute_spectrum(nu)
        
        disk_flux = nu * disk_Lnu
        if np.max(disk_flux) > 0:
            disk_flux = disk_flux / np.max(disk_flux)
        
        wind_emission, wind_absorption = self.disk_wind.compute_wind_contribution(
            wavelength_angstrom
        )
        
        total = disk_flux * (1 + wind_emission) * wind_absorption
        
        components = {
            'disk_thermal': disk_flux,
            'wind_emission': wind_emission,
            'wind_absorption': wind_absorption
        }
        
        return total, components


class MockObservation:
    """
    模拟观测数据生成器
    """
    
    def __init__(self, model):
        self.model = model
        self.energy_grid = None
        self.wavelength_grid = None
        self.xray_data = None
        self.xray_error = None
        self.uv_data = None
        self.uv_error = None
        
    def generate_xray_data(self, energy_min=2.0, energy_max=10.0, n_points=100, 
                          noise_level=0.05):
        """
        生成模拟X射线观测数据
        """
        self.energy_grid = np.logspace(np.log10(energy_min), np.log10(energy_max), n_points)
        model_spec, _ = self.model.compute_xray_spectrum(self.energy_grid)
        
        noise = np.random.normal(0, noise_level, size=n_points) * model_spec
        self.xray_data = model_spec * (1 + noise)
        self.xray_error = noise_level * model_spec
        
        return self.energy_grid, self.xray_data, self.xray_error
    
    def generate_uv_data(self, lambda_min=1000, lambda_max=7000, n_points=200,
                        noise_level=0.03):
        """
        生成模拟紫外-光学观测数据
        """
        self.wavelength_grid = np.linspace(lambda_min, lambda_max, n_points)
        model_spec, _ = self.model.compute_uv_optical_spectrum(self.wavelength_grid)
        
        noise = np.random.normal(0, noise_level, size=n_points) * model_spec
        self.uv_data = model_spec * (1 + noise)
        self.uv_error = noise_level * model_spec
        
        return self.wavelength_grid, self.uv_data, self.uv_error


class SpinFitter:
    """
    黑洞自旋拟合器
    
    通过拟合X射线能谱（特别是铁Kα线轮廓）提取黑洞自旋参数
    """
    
    def __init__(self, observation, true_a_star=None):
        self.observation = observation
        self.true_a_star = true_a_star
        self.fit_result = None
        self.chi_square_history = []
        
    def _model_wrapper(self, energy_grid, a_star, iron_norm, pl_norm, photon_index):
        """
        模型包装器用于拟合
        """
        model = self.observation.model
        model.set_parameters(
            a_star=a_star,
            iron_line_norm=iron_norm,
            pl_norm=pl_norm,
            photon_index=photon_index
        )
        
        spectrum, _ = model.compute_xray_spectrum(energy_grid)
        return spectrum
    
    def fit_spin(self, a_guess=0.5, bounds=None):
        """
        拟合自旋参数
        
        参数:
            a_guess: 初始自旋猜测值
            bounds: 参数边界 [(a_min, a_max), ...]
        """
        if bounds is None:
            bounds = [(-0.998, 0.998), (0.01, 0.5), (0.1, 3.0), (1.3, 2.5)]
            
        p0 = [a_guess, 0.1, 1.0, 1.8]
        
        try:
            popt, pcov = curve_fit(
                self._model_wrapper,
                self.observation.energy_grid,
                self.observation.xray_data,
                sigma=self.observation.xray_error,
                p0=p0,
                bounds=list(zip(*bounds)),
                maxfev=1000
            )
            
            perr = np.sqrt(np.diag(pcov))
            
            model_spec = self._model_wrapper(self.observation.energy_grid, *popt)
            chi2 = np.sum(((self.observation.xray_data - model_spec) / 
                          self.observation.xray_error)**2)
            dof = len(self.observation.xray_data) - len(popt)
            red_chi2 = chi2 / dof
            
            self.fit_result = {
                'a_star': popt[0],
                'a_star_error': perr[0],
                'iron_norm': popt[1],
                'iron_norm_error': perr[1],
                'pl_norm': popt[2],
                'pl_norm_error': perr[2],
                'photon_index': popt[3],
                'photon_index_error': perr[3],
                'chi2': chi2,
                'red_chi2': red_chi2,
                'dof': dof
            }
            
            return self.fit_result
            
        except Exception as e:
            print(f"拟合失败: {e}")
            return None
    
    def plot_fit_result(self):
        """
        绘制拟合结果
        """
        if self.fit_result is None:
            print("请先运行拟合")
            return None
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True,
                                gridspec_kw={'height_ratios': [3, 1]})
        
        energy = self.observation.energy_grid
        
        model_spec = self._model_wrapper(
            energy,
            self.fit_result['a_star'],
            self.fit_result['iron_norm'],
            self.fit_result['pl_norm'],
            self.fit_result['photon_index']
        )
        
        _, components = self.observation.model.compute_xray_spectrum(energy)
        
        axes[0].errorbar(energy, self.observation.xray_data, 
                        yerr=self.observation.xray_error,
                        fmt='o', color='black', markersize=4, 
                        label='观测数据', alpha=0.6)
        axes[0].plot(energy, model_spec, 'r-', linewidth=2, label='拟合模型')
        
        for comp_name, comp_spec in components.items():
            axes[0].plot(energy, comp_spec, '--', label=comp_name, alpha=0.7)
        
        axes[0].set_ylabel('相对强度')
        axes[0].set_title('X射线能谱拟合')
        axes[0].legend()
        axes[0].set_xscale('log')
        axes[0].grid(True, alpha=0.3)
        
        residuals = (self.observation.xray_data - model_spec) / self.observation.xray_error
        axes[1].plot(energy, residuals, 'ko', markersize=3)
        axes[1].axhline(0, color='r', linestyle='--')
        axes[1].set_xlabel('能量 (keV)')
        axes[1].set_ylabel('残差 (σ)')
        axes[1].set_xscale('log')
        axes[1].grid(True, alpha=0.3)
        
        info_text = f"自旋 a* = {self.fit_result['a_star']:.3f} ± {self.fit_result['a_star_error']:.3f}"
        if self.true_a_star is not None:
            info_text += f"\n真实值 = {self.true_a_star:.3f}"
        info_text += f"\nχ²/dof = {self.fit_result['red_chi2']:.2f}"
        
        axes[0].text(0.02, 0.98, info_text, transform=axes[0].transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        return fig


def example_xray_fitting():
    """
    示例: X射线能谱拟合和自旋提取
    """
    print("=" * 65)
    print("示例: X射线能谱拟合和黑洞自旋提取")
    print("=" * 65)
    
    true_a_star = 0.7
    print(f"\n模拟观测参数:")
    print(f"  真实自旋 a* = {true_a_star}")
    
    model = AGNSpectralModel(M_BH=1e8, M_dot=0.5, a_star=true_a_star, inclination=30)
    
    obs = MockObservation(model)
    energy, data, error = obs.generate_xray_data(energy_min=4.0, energy_max=8.0, 
                                                  n_points=80, noise_level=0.05)
    
    print(f"\n生成模拟观测数据...")
    print(f"  能量范围: {energy[0]:.1f} - {energy[-1]:.1f} keV")
    print(f"  数据点数: {len(data)}")
    
    fitter = SpinFitter(obs, true_a_star=true_a_star)
    
    print(f"\n开始拟合...")
    result = fitter.fit_spin(a_guess=0.5)
    
    if result:
        print(f"\n拟合结果:")
        print(f"  测量自旋 a* = {result['a_star']:.3f} ± {result['a_star_error']:.3f}")
        print(f"  真实自旋 a* = {true_a_star:.3f}")
        print(f"  偏差: {abs(result['a_star'] - true_a_star):.4f}")
        print(f"  约化χ² = {result['red_chi2']:.2f}")
        
        if abs(result['a_star'] - true_a_star) < 0.1:
            print("  ✓ 自旋测量准确!")
        else:
            print("  ! 自旋测量有偏差")
    
    fig = fitter.plot_fit_result()
    if fig:
        plt.savefig('spin_fitting_result.png', dpi=150, bbox_inches='tight')
        print("\n拟合图已保存为: spin_fitting_result.png")
    
    print("=" * 65)
    return fitter


def example_iron_line_comparison():
    """
    示例: 不同自旋的铁线轮廓比较
    """
    print("\n" + "=" * 65)
    print("示例: 不同自旋参数的铁Kα线轮廓比较")
    print("=" * 65)
    
    energy_grid = np.linspace(4.0, 8.0, 200)
    spins = [0.0, 0.5, 0.9, 0.998]
    colors = ['b', 'g', 'r', 'm']
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for a_star, color in zip(spins, colors):
        line_model = RelativisticIronLine(
            M_BH=1e8, a_star=a_star, inclination=30,
            emissivity_index=3.0
        )
        profile = line_model.compute_profile_fast(energy_grid, num_radii=80, num_phi=200)
        
        ax.plot(energy_grid, profile, color, linewidth=2, 
                label=f'a* = {a_star}')
    
    ax.axvline(6.4, color='k', linestyle='--', alpha=0.5, label='静止系 6.4 keV')
    ax.set_xlabel('观测能量 (keV)')
    ax.set_ylabel('相对强度')
    ax.set_title('相对论性铁Kα线轮廓 vs 黑洞自旋')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('iron_line_spin_comparison.png', dpi=150, bbox_inches='tight')
    print("铁线轮廓比较图已保存为: iron_line_spin_comparison.png")
    
    print("=" * 65)


def example_full_spectrum():
    """
    示例: 完整AGN光谱（光学-紫外-X射线）
    """
    print("\n" + "=" * 65)
    print("示例: 完整AGN多波段光谱")
    print("=" * 65)
    
    model = AGNSpectralModel(M_BH=1e8, M_dot=0.5, a_star=0.7, inclination=30)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    wavelength = np.linspace(1000, 7000, 300)
    uv_spec, uv_comps = model.compute_uv_optical_spectrum(wavelength)
    
    axes[0].plot(wavelength, uv_comps['disk_thermal'], 'b--', label='热连续谱')
    axes[0].plot(wavelength, uv_comps['wind_emission'] * 5, 'g--', label='发射线 (×5)')
    axes[0].plot(wavelength, uv_spec, 'k-', linewidth=2, label='总谱')
    axes[0].set_xlabel('波长 (Å)')
    axes[0].set_ylabel('相对强度')
    axes[0].set_title('紫外-光学光谱')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    energy = np.logspace(0, 2, 200)
    xray_spec, xray_comps = model.compute_xray_spectrum(energy)
    
    axes[1].loglog(energy, xray_comps['power_law'], 'r--', label='幂律连续谱')
    axes[1].loglog(energy, xray_comps['iron_line'], 'g--', label='铁Kα线')
    axes[1].loglog(energy, xray_comps['compton_hump'], 'b--', label='康普顿峰')
    axes[1].loglog(energy, xray_spec, 'k-', linewidth=2, label='总谱')
    axes[1].set_xlabel('能量 (keV)')
    axes[1].set_ylabel('相对强度')
    axes[1].set_title('X射线光谱')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('full_agn_spectrum.png', dpi=150, bbox_inches='tight')
    print("完整光谱图已保存为: full_agn_spectrum.png")
    
    print("=" * 65)


if __name__ == '__main__':
    print("AGN综合光谱模型 v2.0")
    print("=" * 65)
    
    example_iron_line_comparison()
    example_full_spectrum()
    example_xray_fitting()
    
    plt.show()

"""
Novikov-Thorne标准薄盘吸积模型（修正版）
========================================
计算黑洞吸积盘的多波段辐射谱（光学到X射线）

基于Novikov & Thorne (1973) 和 Page & Thorne (1974)的相对论性薄盘模型

修正内容:
- 正确实现ISCO处零力矩边界条件
- 修正辐射通量计算公式
- 确保能量守恒和辐射效率自洽

作者: 天体物理计算
日期: 2026
"""

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt


class NovikovThorneDisk:
    """
    Novikov-Thorne相对论性薄吸积盘模型（修正版）
    
    正确实现ISCO处零力矩边界条件的标准薄盘模型
    
    参数:
        M_BH: 黑洞质量 (单位: 太阳质量 M_sun)
        M_dot: 吸积率 (单位: 太阳质量/年 M_sun/yr)
        a_star: 无量纲自旋参数 (-1 <= a_star <= 1)
                  a_star = 0 为施瓦西黑洞
                  a_star > 0 为顺行吸积
                  a_star < 0 为逆行吸积
        inclination: 倾角 (单位: 度)
        distance: 观测距离 (单位: 秒差距 pc)
    """
    
    def __init__(self, M_BH=1e8, M_dot=1.0, a_star=0.0, inclination=30.0, distance=1e3):
        self.M_BH = M_BH
        self.M_dot = M_dot
        self.a_star = a_star
        self.inclination = inclination
        self.distance = distance
        
        self._setup_constants()
        self._compute_isco()
        self._compute_radiative_efficiency()
        
    def _setup_constants(self):
        """设置物理常数 (CGS单位)"""
        self.G = 6.67430e-8        # 引力常数 [cm^3 g^-1 s^-2]
        self.c = 2.99792458e10     # 光速 [cm s^-1]
        self.h = 6.62607015e-27    # 普朗克常数 [erg s]
        self.k_B = 1.380649e-16    # 玻尔兹曼常数 [erg K^-1]
        self.sigma_SB = 5.670374e-5 # 斯特藩-玻尔兹曼常数 [erg cm^-2 s^-1 K^-4]
        self.M_sun = 1.98847e33    # 太阳质量 [g]
        self.R_sun = 6.957e10      # 太阳半径 [cm]
        self.L_sun = 3.828e33      # 太阳光度 [erg s^-1]
        self.pc = 3.08567758e18    # 秒差距 [cm]
        self.yr = 3.15576e7        # 年 [s]
        self.eV = 1.602176634e-12  # 电子伏特 [erg]
        
        self.M_BH_cgs = self.M_BH * self.M_sun
        self.M_dot_cgs = self.M_dot * self.M_sun / self.yr
        self.inclination_rad = np.radians(self.inclination)
        self.distance_cgs = self.distance * self.pc
        
        self.R_g = self.G * self.M_BH_cgs / self.c**2  # 引力半径
        self.R_s = 2.0 * self.R_g                     # 史瓦西半径
        
    def _compute_isco(self):
        """
        计算最内稳定圆轨道 (ISCO) 半径
        
        根据Bardeen et al. (1972)的精确公式
        """
        a_star = self.a_star
        
        if abs(a_star) > 0.9999:
            a_star = np.sign(a_star) * 0.9999
        
        Z1 = 1 + (1 - a_star**2)**(1.0/3.0) * ((1 + a_star)**(1.0/3.0) + (1 - a_star)**(1.0/3.0))
        Z2 = np.sqrt(3 * a_star**2 + Z1**2)
        
        if a_star >= 0:
            R_isco_over_Rg = 3 + Z2 - np.sqrt((3 - Z1) * (3 + Z1 + 2 * Z2))
        else:
            R_isco_over_Rg = 3 + Z2 + np.sqrt((3 - Z1) * (3 + Z1 + 2 * Z2))
        
        self.R_isco = R_isco_over_Rg * self.R_g
        self.R_isco_over_Rg = R_isco_over_Rg
        
    def _orbital_quantities(self, r_over_Rg):
        """
        计算半径r处的轨道物理量 (几何单位: G=M=c=1)
        
        参数:
            r_over_Rg: 以引力半径为单位的半径 r/R_g
            
        返回:
            Omega_over_cRg: 角速度 (c/R_g单位)
            E: 单位质量比能量 (无量纲)
            L_over_cRg: 单位质量比角动量 (R_g*c单位)
            dOmega_dr_over_cRg2: 角速度径向导数 (c/R_g^2单位)
            dE_dr_over_c2Rg: 能量径向导数
            dL_dr_over_cRg2: 角动量径向导数
        """
        a = self.a_star
        r = float(r_over_Rg)
        
        if r <= self.R_isco_over_Rg + 1e-10:
            r = self.R_isco_over_Rg + 1e-10
        
        sqrt_r = np.sqrt(r)
        
        denominator = r * np.sqrt(r**2 - 3 * r + 2 * a * sqrt_r)
        
        Omega = 1.0 / (r**(3.0/2.0) + a)
        
        E = (r**2 - 2 * r + a * sqrt_r) / denominator
        
        L = sqrt_r * (r**2 - 2 * a * sqrt_r + a**2) / denominator
        
        dOmega_dr = -1.5 * np.sqrt(r) / (r**(3.0/2.0) + a)**2
        
        dr = 1e-6 * r
        r_plus = r + dr
        r_minus = r - dr
        
        if r_minus <= self.R_isco_over_Rg:
            r_minus = self.R_isco_over_Rg + 1e-10
            dr = r - r_minus
        
        sqrt_r_plus = np.sqrt(r_plus)
        sqrt_r_minus = np.sqrt(r_minus)
        
        denom_plus = r_plus * np.sqrt(r_plus**2 - 3 * r_plus + 2 * a * sqrt_r_plus)
        denom_minus = r_minus * np.sqrt(r_minus**2 - 3 * r_minus + 2 * a * sqrt_r_minus)
        
        E_plus = (r_plus**2 - 2 * r_plus + a * sqrt_r_plus) / denom_plus
        E_minus = (r_minus**2 - 2 * r_minus + a * sqrt_r_minus) / denom_minus
        dE_dr = (E_plus - E_minus) / (2 * dr)
        
        L_plus = sqrt_r_plus * (r_plus**2 - 2 * a * sqrt_r_plus + a**2) / denom_plus
        L_minus = sqrt_r_minus * (r_minus**2 - 2 * a * sqrt_r_minus + a**2) / denom_minus
        dL_dr = (L_plus - L_minus) / (2 * dr)
        
        return Omega, E, L, dOmega_dr, dE_dr, dL_dr
        
    def _compute_radiative_efficiency(self):
        """
        计算辐射效率
        
        零力矩边界条件下的辐射效率: eta = 1 - E_isco
        
        其中 E_isco 是ISCO处的比能量，代表单位质量掉入黑洞时携带的能量
        """
        _, E_isco, _, _, _, _ = self._orbital_quantities(self.R_isco_over_Rg)
        self.eta_rad = 1.0 - E_isco
        self.E_isco = E_isco
        self.L_isco = None
        
        _, _, L_isco, _, _, _ = self._orbital_quantities(self.R_isco_over_Rg)
        self.L_isco = L_isco
        
    def flux_profile(self, r):
        """
        计算半径r处的辐射流量 (修正版Novikov-Thorne公式)
        
        正确实现零力矩边界条件: F(R_isco) = 0
        
        公式来自 Page & Thorne (1974), 方程 (12):
        F(r) = - (M_dot c^2 / (4π R_g^2)) * (dΩ/dr) / E^2 * ∫[R_isco, r] (E - Ω L) dL/dr' dr'
        
        参数:
            r: 半径 (cm)
            
        返回:
            F(r): 单位面积辐射功率 [erg cm^-2 s^-1]
        """
        r_over_Rg = r / self.R_g
        
        if r_over_Rg <= self.R_isco_over_Rg + 1e-8:
            return 0.0
            
        def integrand(r_prime):
            """被积函数: (E - Ω L) * dL/dr"""
            Omega, E, L, dOmega_dr, dE_dr, dL_dr = self._orbital_quantities(r_prime)
            return (E - Omega * L) * dL_dr
        
        integral, _ = quad(integrand, self.R_isco_over_Rg, r_over_Rg, epsabs=1e-10, epsrel=1e-8)
        
        Omega_r, E_r, L_r, dOmega_dr_r, _, _ = self._orbital_quantities(r_over_Rg)
        
        prefactor = - self.M_dot_cgs * self.c**2 / (4 * np.pi * self.R_g**2)
        
        F = prefactor * dOmega_dr_r / (E_r**2) * integral
        
        return max(0.0, F)
        
    def flux_profile_analytic(self, r):
        """
        简化的解析形式流量剖面 (数值积分的快速近似)
        
        使用 Page & Thorne (1974) 的积分结果
        """
        r_over_Rg = r / self.R_g
        
        if r_over_Rg <= self.R_isco_over_Rg:
            return 0.0
        
        x = np.sqrt(r_over_Rg)
        x_isco = np.sqrt(self.R_isco_over_Rg)
        
        Omega, E, L, dOmega_dr, _, _ = self._orbital_quantities(r_over_Rg)
        
        integral_part = (L - self.L_isco * E / self.E_isco)
        
        prefactor = - self.M_dot_cgs * self.c**2 / (4 * np.pi * self.R_g**2)
        
        F = prefactor * dOmega_dr / (E**2) * integral_part
        
        return max(0.0, F)
        
    def temperature_profile(self, r):
        """
        计算半径r处的有效温度
        
        参数:
            r: 半径 (cm)
            
        返回:
            T(r): 有效温度 [K]
        """
        F = self.flux_profile(r)
        if F <= 0:
            return 0.0
        return (F / self.sigma_SB)**0.25
        
    def _specific_intensity(self, nu, T):
        """
        普朗克黑体谱
        
        参数:
            nu: 频率 [Hz]
            T: 温度 [K]
            
        返回:
            B_nu: 单位频率比强度 [erg cm^-2 s^-1 Hz^-1 sr^-1]
        """
        if T <= 0:
            return 0.0
        x = self.h * nu / (self.k_B * T)
        if x > 50:
            return 0.0
        if x < 1e-4:
            return (2 * self.k_B * T * nu**2) / self.c**2
        return (2 * self.h * nu**3 / self.c**2) / (np.exp(x) - 1.0)
        
    def _spectrum_integrand(self, r, nu):
        """
        谱积分的被积函数
        
        参数:
            r: 半径 (cm)
            nu: 频率 (Hz)
            
        返回:
            被积函数值
        """
        T = self.temperature_profile(r)
        if T <= 0:
            return 0.0
        return 4 * np.pi * r * self._specific_intensity(nu, T)
        
    def compute_spectrum(self, nu_array, r_out=None):
        """
        计算吸积盘的多波段辐射谱
        
        参数:
            nu_array: 频率数组 [Hz]
            r_out: 外半径 (单位: R_g), 默认1000 R_g
            
        返回:
            L_nu: 单色光度 [erg s^-1 Hz^-1]
            F_nu: 观测流量 [erg cm^-2 s^-1 Hz^-1]
        """
        if r_out is None:
            r_out = 1000.0 * self.R_g
            
        L_nu = np.zeros_like(nu_array)
        cos_i = np.cos(self.inclination_rad)
        
        for i, nu in enumerate(nu_array):
            integral, _ = quad(self._spectrum_integrand, self.R_isco, r_out, 
                              args=(nu,), epsabs=1e-10, epsrel=1e-6)
            L_nu[i] = cos_i * integral
            
        F_nu = L_nu / (4 * np.pi * self.distance_cgs**2)
        
        return L_nu, F_nu
        
    def compute_nuLnu(self, nu_array, r_out=None):
        """
        计算 nu*L_nu 谱 (便于观察光谱能量分布)
        
        参数:
            nu_array: 频率数组 [Hz]
            r_out: 外半径 (单位: R_g)
            
        返回:
            nuL_nu: [erg s^-1]
        """
        L_nu, _ = self.compute_spectrum(nu_array, r_out)
        return nu_array * L_nu
        
    def compute_total_luminosity(self, r_out=None):
        """
        通过积分辐射通量计算总光度
        
        用于验证能量守恒: L_bol = eta * M_dot * c^2
        """
        if r_out is None:
            r_out = 1000.0 * self.R_g
            
        def luminosity_integrand(r):
            return 4 * np.pi * r * self.flux_profile(r) * 2
            
        L_bol, _ = quad(luminosity_integrand, self.R_isco, r_out, 
                       epsabs=1e-5, epsrel=1e-5)
        
        return L_bol
        
    def get_band_fluxes(self, r_out=None):
        """
        计算各波段的流量和光度
        
        返回:
            各波段信息的字典
        """
        bands = {
            '光学 (V)': {'nu_min': 4.0e14, 'nu_max': 7.5e14, 'lambda_mean': 5500},
            '紫外 (UV)': {'nu_min': 7.5e14, 'nu_max': 3e16, 'lambda_mean': 2000},
            '软X射线 (0.1-2 keV)': {'nu_min': 2.42e16, 'nu_max': 4.84e17, 'E_mean': 1.0},
            '硬X射线 (2-10 keV)': {'nu_min': 4.84e17, 'nu_max': 2.42e18, 'E_mean': 5.0}
        }
        
        results = {}
        for band_name, band_info in bands.items():
            nu_array = np.logspace(np.log10(band_info['nu_min']), 
                                   np.log10(band_info['nu_max']), 50)
            L_nu, F_nu = self.compute_spectrum(nu_array, r_out)
            
            L_band = np.trapz(L_nu, nu_array)
            F_band = np.trapz(F_nu, nu_array)
            
            results[band_name] = {
                'Luminosity': L_band,
                'Flux': F_band,
                'L_sun_units': L_band / self.L_sun
            }
            
        return results
        
    def verify_boundary_conditions(self):
        """
        验证零力矩边界条件和能量守恒
        
        返回:
            包含验证结果的字典
        """
        results = {}
        
        r_near_isco = self.R_isco * (1.0 + 1e-6)
        F_near_isco = self.flux_profile(r_near_isco)
        results['F_near_ISCO'] = F_near_isco
        results['F_ISCO_is_zero'] = F_near_isco < 1e-10
        
        L_bol_integrated = self.compute_total_luminosity()
        L_bol_expected = self.eta_rad * self.M_dot_cgs * self.c**2
        
        results['L_bol_integrated'] = L_bol_integrated
        results['L_bol_expected'] = L_bol_expected
        results['energy_conservation_ratio'] = L_bol_integrated / L_bol_expected
        results['energy_conserved'] = abs(L_bol_integrated / L_bol_expected - 1.0) < 0.05
        
        return results
        
    def print_disk_properties(self):
        """打印吸积盘的基本性质"""
        print("=" * 65)
        print("Novikov-Thorne 薄吸积盘模型 (修正版)")
        print("=" * 65)
        print(f"黑洞质量: {self.M_BH:.2e} M_sun")
        print(f"吸积率: {self.M_dot:.4f} M_sun/yr")
        print(f"自旋参数 a*: {self.a_star:.3f}")
        print(f"倾角: {self.inclination:.1f} 度")
        print(f"距离: {self.distance:.1e} pc")
        print()
        print(f"引力半径 R_g: {self.R_g:.4e} cm = {self.R_g/self.R_sun:.4f} R_sun")
        print(f"史瓦西半径 R_s: {self.R_s:.4e} cm = {self.R_s/self.R_sun:.4f} R_sun")
        print(f"ISCO半径: {self.R_isco:.4e} cm = {self.R_isco_over_Rg:.4f} R_g")
        print()
        
        T_isco = self.temperature_profile(self.R_isco * 1.1)
        print(f"ISCO附近 (1.1 R_isco) 温度: {T_isco:.2e} K")
        print()
        
        print(f"辐射效率 eta: {self.eta_rad:.4f}")
        print(f"ISCO处比能量 E_isco: {self.E_isco:.6f}")
        
        L_bol = self.eta_rad * self.M_dot_cgs * self.c**2
        print(f"总光度 L_bol: {L_bol:.4e} erg/s = {L_bol/self.L_sun:.4e} L_sun")
        print(f"爱丁顿光度 L_Edd: {self.eddington_luminosity():.4e} erg/s")
        print(f"爱丁顿比: {L_bol/self.eddington_luminosity():.4f}")
        
        print("\n边界条件验证:")
        bc = self.verify_boundary_conditions()
        print(f"  ISCO附近通量: {bc['F_near_ISCO']:.2e} erg/cm²/s", end="")
        print("  ✓" if bc['F_ISCO_is_zero'] else "  ✗")
        print(f"  能量守恒比: {bc['energy_conservation_ratio']:.4f}", end="")
        print("  ✓" if bc['energy_conserved'] else "  ✗")
        
        print("=" * 65)
        
    def eddington_luminosity(self):
        """计算爱丁顿光度"""
        return 1.26e38 * self.M_BH
        
    def plot_temperature_profile(self, r_min=1.0, r_max=1000.0, ax=None):
        """
        绘制温度剖面图
        
        参数:
            r_min, r_max: 半径范围 (单位: R_g)
        """
        r_array = np.logspace(np.log10(max(r_min, self.R_isco_over_Rg*1.01)), 
                              np.log10(r_max), 200) * self.R_g
        T_array = np.array([self.temperature_profile(r) for r in r_array])
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
        ax.loglog(r_array / self.R_g, T_array, 'b-', linewidth=2)
        ax.axvline(self.R_isco_over_Rg, color='r', linestyle='--', 
                   label=f'ISCO = {self.R_isco_over_Rg:.2f} R_g')
        ax.set_xlabel('半径 (R_g)')
        ax.set_ylabel('有效温度 T (K)')
        ax.set_title('吸积盘温度剖面 (零力矩边界条件)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
        
    def plot_flux_profile(self, r_min=1.0, r_max=1000.0, ax=None):
        """
        绘制辐射流量剖面图
        """
        r_array = np.logspace(np.log10(max(r_min, self.R_isco_over_Rg*1.01)), 
                              np.log10(r_max), 200) * self.R_g
        F_array = np.array([self.flux_profile(r) for r in r_array])
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
        ax.loglog(r_array / self.R_g, F_array, 'r-', linewidth=2)
        ax.axvline(self.R_isco_over_Rg, color='k', linestyle='--', 
                   label=f'ISCO = {self.R_isco_over_Rg:.2f} R_g')
        ax.set_xlabel('半径 (R_g)')
        ax.set_ylabel('辐射通量 F (erg cm$^{-2}$ s$^{-1}$)')
        ax.set_title('辐射流量剖面')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
        
    def plot_spectrum(self, nu_min=1e14, nu_max=1e19, r_out=None, ax=None):
        """
        绘制光谱图 (nu*L_nu vs nu)
        
        参数:
            nu_min, nu_max: 频率范围 [Hz]
        """
        nu_array = np.logspace(np.log10(nu_min), np.log10(nu_max), 200)
        nuL_nu = self.compute_nuLnu(nu_array, r_out)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
        ax.loglog(nu_array, nuL_nu, 'b-', linewidth=2)
        
        wavelength_um = (self.c / nu_array) * 1e4
        ax2 = ax.twiny()
        ax2.loglog(wavelength_um, nuL_nu, alpha=0)
        ax2.set_xlabel('波长 (μm)')
        
        eV_array = nu_array * self.h / self.eV
        ax3 = ax.twiny()
        ax3.spines['top'].set_position(('outward', 40))
        ax3.loglog(eV_array, nuL_nu, alpha=0)
        ax3.set_xlabel('能量 (eV)')
        
        ax.axvspan(4e14, 7.5e14, alpha=0.2, color='r', label='可见光')
        ax.axvspan(7.5e14, 3e16, alpha=0.2, color='m', label='紫外')
        ax.axvspan(2.4e16, 2.4e18, alpha=0.2, color='b', label='X射线')
        
        ax.set_xlabel('频率 ν (Hz)')
        ax.set_ylabel('ν L_ν (erg/s)')
        ax.set_title('Novikov-Thorne 吸积盘辐射谱 (修正版)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        return ax


def example_usage():
    """示例用法"""
    
    print("示例1: 典型的活动星系核 (AGN) 吸积盘")
    print("-" * 65)
    
    disk = NovikovThorneDisk(
        M_BH=1e8,
        M_dot=0.5,
        a_star=0.7,
        inclination=30.0,
        distance=1e6
    )
    
    disk.print_disk_properties()
    
    print("\n各波段辐射功率:")
    band_fluxes = disk.get_band_fluxes()
    for band, data in band_fluxes.items():
        print(f"{band}: {data['L_sun_units']:.4e} L_sun")
    
    print("\n" + "=" * 65)
    print("示例2: 恒星级黑洞 (X射线双星)")
    print("-" * 65)
    
    disk_bh = NovikovThorneDisk(
        M_BH=10.0,
        M_dot=1e-8,
        a_star=0.5,
        inclination=60.0,
        distance=1e3
    )
    
    disk_bh.print_disk_properties()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    disk.plot_temperature_profile(ax=axes[0, 0])
    axes[0, 0].set_title('AGN 吸积盘温度剖面')
    
    disk.plot_flux_profile(ax=axes[0, 1])
    axes[0, 1].set_title('AGN 吸积盘流量剖面')
    
    disk.plot_spectrum(ax=axes[1, 0])
    axes[1, 0].set_title('AGN 吸积盘多波段谱')
    
    spins = [-0.5, 0.0, 0.5, 0.9]
    colors = ['r', 'g', 'b', 'm']
    nu_array = np.logspace(14, 19, 100)
    
    for a_star, color in zip(spins, colors):
        d = NovikovThorneDisk(M_BH=1e8, M_dot=0.5, a_star=a_star)
        nuL_nu = d.compute_nuLnu(nu_array)
        axes[1, 1].loglog(nu_array, nuL_nu, color, 
                          label=f'a* = {a_star}', linewidth=2)
    
    axes[1, 1].set_xlabel('频率 (Hz)')
    axes[1, 1].set_ylabel('ν L_ν (erg/s)')
    axes[1, 1].set_title('不同自旋参数的辐射谱比较')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('novikov_thorne_disk_spectrum.png', dpi=150, bbox_inches='tight')
    print("\n图像已保存为: novikov_thorne_disk_spectrum.png")
    
    return disk, disk_bh


def compare_old_new():
    """比较零力矩边界条件修正前后的差异"""
    print("\n" + "=" * 65)
    print("零力矩边界条件验证")
    print("=" * 65)
    
    disk = NovikovThorneDisk(M_BH=1e8, M_dot=0.5, a_star=0.7)
    
    print(f"\n在ISCO附近的辐射流量:")
    print(f"  1.000001 × R_isco: {disk.flux_profile(disk.R_isco * 1.000001):.2e} erg/cm²/s")
    print(f"  1.01 × R_isco: {disk.flux_profile(disk.R_isco * 1.01):.2e} erg/cm²/s")
    print(f"  1.1 × R_isco: {disk.flux_profile(disk.R_isco * 1.1):.2e} erg/cm²/s")
    
    L_integrated = disk.compute_total_luminosity()
    L_expected = disk.eta_rad * disk.M_dot_cgs * disk.c**2
    
    print(f"\n总光度验证:")
    print(f"  积分流量得到 L_bol = {L_integrated:.4e} erg/s")
    print(f"  辐射效率预测 L_bol = {L_expected:.4e} erg/s")
    print(f"  比值 = {L_integrated / L_expected:.4f}")
    
    if abs(L_integrated / L_expected - 1.0) < 0.05:
        print("  ✓ 能量守恒满足!")
    else:
        print("  ✗ 能量守恒有偏差!")
    
    print("=" * 65)


if __name__ == '__main__':
    print("Novikov-Thorne 薄吸积盘模型 (修正版)")
    print("=" * 65)
    
    disk1, disk2 = example_usage()
    compare_old_new()
    
    plt.show()

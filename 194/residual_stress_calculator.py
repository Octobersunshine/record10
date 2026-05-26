import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from scipy.optimize import minimize, least_squares


@dataclass
class DiffractionPeak:
    """
    中子衍射峰数据类
    """
    two_theta: float
    d_hkl: Optional[float] = None
    wavelength: Optional[float] = None
    hkl: Tuple[int, int, int] = (0, 0, 0)
    phi: float = 0.0
    psi: float = 0.0
    intensity: Optional[float] = None


@dataclass
class ElasticConstants:
    """
    材料弹性常数
    """
    E: float
    nu: float
    crystal_structure: str = "cubic"
    hkl_plane: Optional[Tuple[int, int, int]] = None
    E_hkl: Optional[float] = None
    nu_hkl: Optional[float] = None


@dataclass
class StressMeasurementPoint:
    """
    应力测量点数据（用于应力平衡条件标定d0）
    
    属性:
        depth: 测量点深度 (mm)
        d_hkl: 测量的晶面间距 (Angstrom)
        phi: 测量方位角 (度)
        psi: 测量psi角 (度)
        hkl: 晶面指数
        wavelength: 中子波长 (Angstrom)
        two_theta: 衍射角 (度)，可选
        weight: 测量点权重（用于加权拟合）
    """
    depth: float
    d_hkl: float
    phi: float = 0.0
    psi: float = 90.0
    hkl: Tuple[int, int, int] = (0, 0, 0)
    wavelength: Optional[float] = None
    two_theta: Optional[float] = None
    weight: float = 1.0


@dataclass
class D0CalibrationResult:
    """
    d0标定结果
    
    属性:
        d0: 标定后的无应力晶面间距 (Angstrom)
        method: 使用的标定方法
        residual: 拟合残差
        iterations: 迭代次数
        stress_profile: 标定后的应力分布
        balance_residual: 应力平衡残差
    """
    d0: float
    method: str
    residual: float
    iterations: int
    stress_profile: List[float] = field(default_factory=list)
    balance_residual: Dict[str, float] = field(default_factory=dict)
    convergence_history: List[float] = field(default_factory=list)


@dataclass
class TensorMeasurement:
    """
    三维应力张量测量数据
    
    属性:
        phi: 方位角 (度), 绕法线方向的旋转角
        psi: 倾角 (度), 与法线方向的夹角
        chi: 试样旋转角 (度), 绕测量方向的旋转
        strain: 测量的晶格应变
        strain_error: 应变测量误差 (可选)
        d_hkl: 测量的晶面间距 (Angstrom)
        d0: 无应力晶面间距 (Angstrom)
        hkl: 晶面指数
        wavelength: 中子波长 (Angstrom)
        two_theta: 衍射角 (度)
        weight: 测量点权重
    """
    phi: float
    psi: float
    chi: float = 0.0
    strain: Optional[float] = None
    strain_error: Optional[float] = None
    d_hkl: Optional[float] = None
    d0: Optional[float] = None
    hkl: Tuple[int, int, int] = (0, 0, 0)
    wavelength: Optional[float] = None
    two_theta: Optional[float] = None
    weight: float = 1.0


@dataclass
class StressTensorResult:
    """
    三维应力张量重构结果
    
    属性:
        sigma_xx, sigma_yy, sigma_zz: 正应力分量 (MPa)
        sigma_xy, sigma_xz, sigma_yz: 剪切应力分量 (MPa)
        covariance: 协方差矩阵
        residuals: 拟合残差
        principal_stresses: 主应力 (sigma1, sigma2, sigma3)
        principal_directions: 主应力方向
        von_mises: 冯米塞斯等效应力
        hydrostatic: 静水压力
        error_estimate: 各分量的误差估计
    """
    sigma_xx: float
    sigma_yy: float
    sigma_zz: float
    sigma_xy: float
    sigma_xz: float
    sigma_yz: float
    covariance: Optional[np.ndarray] = None
    residuals: List[float] = field(default_factory=list)
    principal_stresses: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    principal_directions: Optional[np.ndarray] = None
    von_mises: float = 0.0
    hydrostatic: float = 0.0
    error_estimate: Dict[str, float] = field(default_factory=dict)


class ResidualStressCalculator:
    """
    中子衍射残余应力计算器
    
    基本原理：
    1. 布拉格定律：nλ = 2d·sinθ
    2. 晶格应变：ε = (d - d₀) / d₀
    3. 应力转换：σ = f(ε, E, ν, 几何因子)
    """

    def __init__(self, elastic_constants: ElasticConstants):
        self.elastic_constants = elastic_constants

    @staticmethod
    def bragg_law(two_theta: float, wavelength: float, n: int = 1) -> float:
        """
        布拉格定律计算晶面间距d
        
        参数:
            two_theta: 衍射角2θ (度)
            wavelength: 中子波长 (Å)
            n: 衍射级数
            
        返回:
            d: 晶面间距 (Å)
        """
        theta_rad = np.radians(two_theta / 2.0)
        d = n * wavelength / (2.0 * np.sin(theta_rad))
        return d

    @staticmethod
    def bragg_law_inverse(d: float, wavelength: float, n: int = 1) -> float:
        """
        布拉格定律逆计算，从d求2θ
        
        参数:
            d: 晶面间距 (Å)
            wavelength: 中子波长 (Å)
            n: 衍射级数
            
        返回:
            two_theta: 衍射角2θ (度)
        """
        sin_theta = n * wavelength / (2.0 * d)
        sin_theta = np.clip(sin_theta, -1.0, 1.0)
        theta_rad = np.arcsin(sin_theta)
        two_theta = np.degrees(2.0 * theta_rad)
        return two_theta

    @staticmethod
    def calculate_strain(d: float, d0: float) -> float:
        """
        计算晶格应变 ε = (d - d₀) / d₀
        
        参数:
            d: 有应力时的晶面间距 (Å)
            d0: 无应力时的晶面间距 (Å)
            
        返回:
            ε: 晶格应变
        """
        return (d - d0) / d0

    @staticmethod
    def calculate_strain_from_peak_shift(
        two_theta: float, 
        two_theta0: float, 
        wavelength: Optional[float] = None
    ) -> float:
        """
        直接从峰位偏移计算应变
        
        当θ很小时：ε ≈ -cotθ·Δθ（小角度近似）
        精确公式：ε = sinθ₀ / sinθ - 1
        
        参数:
            two_theta: 有应力时的衍射角2θ (度)
            two_theta0: 无应力时的衍射角2θ₀ (度)
            wavelength: 中子波长（可选，用于验证）
            
        返回:
            ε: 晶格应变
        """
        theta = np.radians(two_theta / 2.0)
        theta0 = np.radians(two_theta0 / 2.0)
        
        epsilon = np.sin(theta0) / np.sin(theta) - 1.0
        
        if wavelength is not None:
            d = ResidualStressCalculator.bragg_law(two_theta, wavelength)
            d0 = ResidualStressCalculator.bragg_law(two_theta0, wavelength)
            epsilon_check = ResidualStressCalculator.calculate_strain(d, d0)
            assert np.isclose(epsilon, epsilon_check), "两种计算方法不一致"
        
        return epsilon

    def _get_effective_modulus(self, hkl: Optional[Tuple[int, int, int]] = None) -> Tuple[float, float]:
        """
        获取有效弹性模量（考虑各向异性）
        
        参数:
            hkl: 晶面指数
            
        返回:
            (E_eff, nu_eff): 有效弹性模量和泊松比
        """
        ec = self.elastic_constants
        
        if ec.E_hkl is not None and ec.nu_hkl is not None:
            return ec.E_hkl, ec.nu_hkl
        
        if hkl is not None and ec.crystal_structure == "cubic":
            h, k, l = hkl
            S11 = 1.0 / ec.E
            S12 = -ec.nu / ec.E
            S44 = 2.0 * (1.0 + ec.nu) / ec.E
            
            hkl_sum = h**2 + k**2 + l**2
            if hkl_sum == 0:
                return ec.E, ec.nu
            
            Gamma = (h**2 * k**2 + k**2 * l**2 + l**2 * h**2) / hkl_sum**2
            
            S_hkl = S11 - 2.0 * (S11 - S12 - S44 / 2.0) * Gamma
            E_hkl = 1.0 / S_hkl
            
            nu_hkl = ec.nu
            
            return E_hkl, nu_hkl
        
        return ec.E, ec.nu

    def calculate_stress_uniaxial(
        self,
        strain: float,
        hkl: Optional[Tuple[int, int, int]] = None
    ) -> float:
        """
        单轴应力计算（简单Hooke定律）
        σ = E·ε
        
        参数:
            strain: 晶格应变 ε
            hkl: 晶面指数（用于各向异性修正）
            
        返回:
            σ: 应力 (与E单位相同，通常为MPa)
        """
        E_eff, _ = self._get_effective_modulus(hkl)
        stress = E_eff * strain
        return stress

    def calculate_stress_plane_stress(
        self,
        epsilon_phi: float,
        phi: float,
        psi: float = 90.0,
        epsilon_axial: Optional[float] = None,
        hkl: Optional[Tuple[int, int, int]] = None
    ) -> float:
        """
        平面应力状态下的应力计算
        
        使用sin^2psi法的基本公式：
        sigma_phi = (E / (1 + nu)) * (epsilon_phi / sin^2 psi) - nu/(1-nu)*sigma_axial
        
        当psi=90度（横向测量）时：
        sigma_phi = (E / (1 + nu)) * epsilon_phi - nu/(1-nu)*sigma_axial
        
        参数:
            epsilon_phi: 测量方向的晶格应变
            phi: 样品表面内的方位角 (度)
            psi: 衍射面法线与样品表面法线的夹角 (度)
            epsilon_axial: 轴向应变（如果已知）
            hkl: 晶面指数
            
        返回:
            sigma_phi: 方向phi上的应力
        """
        E_eff, nu_eff = self._get_effective_modulus(hkl)
        psi_rad = np.radians(psi)
        sin_sq_psi = np.sin(psi_rad)**2
        
        if sin_sq_psi < 1e-10:
            if epsilon_axial is not None:
                sigma_axial = self.calculate_stress_uniaxial(epsilon_axial, hkl)
                return sigma_axial * nu_eff / (1 - nu_eff)
            else:
                return 0.0
        
        if epsilon_axial is None:
            sigma_phi = (E_eff / (1.0 + nu_eff)) * (epsilon_phi / sin_sq_psi)
        else:
            sigma_axial = self.calculate_stress_uniaxial(epsilon_axial, hkl)
            sigma_phi = (E_eff / (1.0 + nu_eff)) * (epsilon_phi / sin_sq_psi) \
                        - (nu_eff / (1.0 - nu_eff)) * sigma_axial
        
        return sigma_phi

    def calculate_stress_sin2psi(
        self,
        psi_list: List[float],
        strain_list: List[float],
        hkl: Optional[Tuple[int, int, int]] = None
    ) -> Tuple[float, float, float]:
        """
        sin²ψ法计算残余应力
        
        通过线性拟合 ε vs sin²ψ，斜率M = (1+ν)/E · σ_φ
        截距B = -2ν/E · σ_33（平面应力下σ_33=0）
        
        参数:
            psi_list: ψ角列表 (度)
            strain_list: 对应的应变列表
            hkl: 晶面指数
            
        返回:
            (sigma_phi, slope, intercept): 应力、斜率、截距
        """
        E_eff, nu_eff = self._get_effective_modulus(hkl)
        
        psi_rad = np.radians(psi_list)
        sin_sq_psi = np.sin(psi_rad)**2
        
        coeffs = np.polyfit(sin_sq_psi, strain_list, 1)
        slope = coeffs[0]
        intercept = coeffs[1]
        
        sigma_phi = (E_eff / (1.0 + nu_eff)) * slope
        
        return sigma_phi, slope, intercept

    def calculate_stress_from_peaks(
        self,
        peaks: List[DiffractionPeak],
        d0: float,
        wavelength: Optional[float] = None,
        method: str = "plane_stress",
        **kwargs
    ) -> List[float]:
        """
        从衍射峰数据直接计算应力
        
        参数:
            peaks: 衍射峰列表
            d0: 无应力晶面间距 (Å)
            wavelength: 中子波长 (Å)
            method: 计算方法 ("uniaxial", "plane_stress", "sin2psi")
            
        返回:
            stresses: 应力列表
        """
        stresses = []
        
        for peak in peaks:
            if peak.d_hkl is not None:
                d = peak.d_hkl
            elif wavelength is not None and peak.two_theta is not None:
                d = self.bragg_law(peak.two_theta, wavelength)
            else:
                raise ValueError("必须提供d_hkl或wavelength+two_theta")
            
            strain = self.calculate_strain(d, d0)
            
            if method == "uniaxial":
                stress = self.calculate_stress_uniaxial(strain, peak.hkl)
            elif method == "plane_stress":
                stress = self.calculate_stress_plane_stress(
                    strain, peak.phi, peak.psi, 
                    hkl=peak.hkl, **kwargs
                )
            else:
                raise ValueError(f"未知方法: {method}")
            
            stresses.append(stress)
        
        return stresses

    def _calculate_stress_for_point(
        self,
        point: StressMeasurementPoint,
        d0: float
    ) -> float:
        """
        计算单个测量点的应力
        
        参数:
            point: 应力测量点
            d0: 无应力晶面间距 (Angstrom)
            
        返回:
            应力值 (MPa)
        """
        if point.wavelength is not None and point.two_theta is not None:
            d = self.bragg_law(point.two_theta, point.wavelength)
        else:
            d = point.d_hkl
        
        strain = self.calculate_strain(d, d0)
        stress = self.calculate_stress_plane_stress(
            strain, point.phi, point.psi, hkl=point.hkl
        )
        
        return stress

    def _stress_balance_residual(
        self,
        d0: float,
        points: List[StressMeasurementPoint],
        thickness: Optional[float] = None,
        balance_type: str = "force_moment"
    ) -> Dict[str, float]:
        """
        计算应力平衡条件的残差
        
        参数:
            d0: 无应力晶面间距 (Angstrom)
            points: 测量点列表
            thickness: 试样厚度 (mm)
            balance_type: 平衡类型 ("force", "moment", "force_moment")
            
        返回:
            残差字典
        """
        stresses = []
        depths = []
        weights = []
        
        for point in points:
            stress = self._calculate_stress_for_point(point, d0)
            stresses.append(stress)
            depths.append(point.depth)
            weights.append(point.weight)
        
        stresses = np.array(stresses)
        depths = np.array(depths)
        weights = np.array(weights)
        
        if thickness is not None:
            depths_normalized = depths / thickness
            z_center = depths_normalized - 0.5
        else:
            z_center = depths - np.mean(depths)
        
        residuals = {}
        
        if "force" in balance_type or balance_type == "force_moment":
            weighted_stresses = weights * stresses
            force_residual = np.trapezoid(weighted_stresses, depths)
            if thickness is not None:
                force_residual /= thickness
            residuals["force"] = force_residual
        
        if "moment" in balance_type or balance_type == "force_moment":
            weighted_moments = weights * stresses * z_center
            moment_residual = np.trapezoid(weighted_moments, depths)
            if thickness is not None:
                moment_residual /= thickness ** 2
            residuals["moment"] = moment_residual
        
        return residuals

    def calibrate_d0_force_balance(
        self,
        points: List[StressMeasurementPoint],
        d0_initial: float,
        thickness: Optional[float] = None,
        balance_type: str = "force_moment",
        d0_range: Optional[Tuple[float, float]] = None,
        max_iterations: int = 100,
        tolerance: float = 1e-10,
        **kwargs
    ) -> D0CalibrationResult:
        """
        基于应力平衡条件自洽标定d0
        
        原理：
        对于无外力作用的自由试样，内部残余应力应满足：
        - 合力为零：integral(sigma * dz) = 0
        - 合力矩为零：integral(sigma * z * dz) = 0
        
        通过调整d0使应力分布满足这些条件
        
        参数:
            points: 测量点列表
            d0_initial: 初始d0估计值 (Angstrom)
            thickness: 试样厚度 (mm)
            balance_type: 平衡类型 ("force", "moment", "force_moment")
            d0_range: d0搜索范围 (min, max)
            max_iterations: 最大迭代次数
            tolerance: 收敛容差
            
        返回:
            D0CalibrationResult: 标定结果
        """
        convergence_history = []
        
        def objective_function(d0_val):
            residuals = self._stress_balance_residual(
                d0_val[0], points, thickness, balance_type
            )
            
            residual_sum = sum(r**2 for r in residuals.values())
            convergence_history.append(float(residual_sum))
            
            return residual_sum
        
        if d0_range is None:
            d0_min = d0_initial * 0.99
            d0_max = d0_initial * 1.01
        else:
            d0_min, d0_max = d0_range
        
        result = minimize(
            objective_function,
            x0=[d0_initial],
            method='L-BFGS-B',
            bounds=[(d0_min, d0_max)],
            options={
                'maxiter': max_iterations,
                'ftol': tolerance,
                'gtol': tolerance * 10
            }
        )
        
        d0_calibrated = result.x[0]
        final_residuals = self._stress_balance_residual(
            d0_calibrated, points, thickness, balance_type
        )
        
        stress_profile = []
        for point in points:
            stress = self._calculate_stress_for_point(point, d0_calibrated)
            stress_profile.append(stress)
        
        return D0CalibrationResult(
            d0=d0_calibrated,
            method=f"force_balance_{balance_type}",
            residual=float(result.fun),
            iterations=result.nit,
            stress_profile=stress_profile,
            balance_residual={k: float(v) for k, v in final_residuals.items()},
            convergence_history=convergence_history
        )

    def calibrate_d0_multi_peak(
        self,
        peaks_by_hkl: Dict[Tuple[int, int, int], List[DiffractionPeak]],
        d0_by_hkl: Dict[Tuple[int, int, int], float],
        wavelength: Optional[float] = None,
        fit_type: str = "strain_linear",
        **kwargs
    ) -> Dict[Tuple[int, int, int], D0CalibrationResult]:
        """
        多峰拟合确定各晶面的最佳d0
        
        原理：
        利用多个不同hkl晶面的衍射峰，通过应力-应变关系的线性拟合
        消除系统偏差，确定各晶面的自洽d0值
        
        参数:
            peaks_by_hkl: 按hkl分组的衍射峰列表 {hkl: [peak1, peak2, ...]}
            d0_by_hkl: 各hkl的初始d0估计值 {hkl: d0}
            wavelength: 中子波长 (Angstrom)
            fit_type: 拟合类型 ("strain_linear", "stress_consistency")
            
        返回:
            各hkl的标定结果
        """
        results = {}
        
        for hkl, peaks in peaks_by_hkl.items():
            d0_initial = d0_by_hkl.get(hkl, 2.0)
            
            strains = []
            stresses_consistency = []
            
            for peak in peaks:
                if peak.d_hkl is not None:
                    d = peak.d_hkl
                elif wavelength is not None and peak.two_theta is not None:
                    d = self.bragg_law(peak.two_theta, wavelength)
                else:
                    continue
                
                strain = self.calculate_strain(d, d0_initial)
                strains.append(strain)
                
                stress = self.calculate_stress_plane_stress(
                    strain, peak.phi, peak.psi, hkl=hkl
                )
                stresses_consistency.append(stress)
            
            if len(strains) < 2:
                results[hkl] = D0CalibrationResult(
                    d0=d0_initial,
                    method=f"multi_peak_{fit_type}",
                    residual=0.0,
                    iterations=0,
                    stress_profile=stresses_consistency
                )
                continue
            
            strains = np.array(strains)
            stresses_consistency = np.array(stresses_consistency)
            
            E_eff, nu_eff = self._get_effective_modulus(hkl)
            
            def objective(d0_val):
                adjusted_strains = []
                for p in peaks:
                    if p.d_hkl is not None:
                        d = p.d_hkl
                    elif wavelength is not None and p.two_theta is not None:
                        d = self.bragg_law(p.two_theta, wavelength)
                    else:
                        continue
                    adjusted_strains.append((d - d0_val[0]) / d0_val[0])
                
                if len(adjusted_strains) < 2:
                    return 1e10
                
                adjusted_strains = np.array(adjusted_strains)
                
                if fit_type == "strain_linear":
                    psi_rad = np.radians([p.psi for p in peaks])
                    sin_sq_psi = np.sin(psi_rad) ** 2
                    
                    if np.std(sin_sq_psi) > 1e-10:
                        coeffs = np.polyfit(sin_sq_psi, adjusted_strains, 1)
                        fitted_strains = coeffs[0] * sin_sq_psi + coeffs[1]
                        residual = np.sum((adjusted_strains - fitted_strains) ** 2)
                    else:
                        residual = np.var(adjusted_strains)
                        
                elif fit_type == "stress_consistency":
                    stresses = [
                        self.calculate_stress_plane_stress(
                            s, p.phi, p.psi, hkl=hkl
                        )
                        for s, p in zip(adjusted_strains, peaks)
                    ]
                    residual = np.var(stresses)
                else:
                    residual = np.var(adjusted_strains)
                
                return residual
            
            d0_min = d0_initial * 0.99
            d0_max = d0_initial * 1.01
            
            result = minimize(
                objective,
                x0=[d0_initial],
                method='L-BFGS-B',
                bounds=[(d0_min, d0_max)],
                options={'maxiter': 50}
            )
            
            d0_calibrated = result.x[0]
            
            final_stresses = []
            for peak in peaks:
                if peak.d_hkl is not None:
                    d = peak.d_hkl
                elif wavelength is not None and peak.two_theta is not None:
                    d = self.bragg_law(peak.two_theta, wavelength)
                else:
                    continue
                
                strain = self.calculate_strain(d, d0_calibrated)
                stress = self.calculate_stress_plane_stress(
                    strain, peak.phi, peak.psi, hkl=hkl
                )
                final_stresses.append(stress)
            
            results[hkl] = D0CalibrationResult(
                d0=d0_calibrated,
                method=f"multi_peak_{fit_type}",
                residual=float(result.fun),
                iterations=result.nit,
                stress_profile=final_stresses
            )
        
        return results

    def calculate_stress_depth_profile(
        self,
        points: List[StressMeasurementPoint],
        d0: float,
        thickness: Optional[float] = None
    ) -> Dict:
        """
        计算深度方向的应力分布，并评估应力平衡条件
        
        参数:
            points: 测量点列表
            d0: 无应力晶面间距 (Angstrom)
            thickness: 试样厚度 (mm)
            
        返回:
            包含应力分布和平衡条件评估的字典
        """
        stresses = []
        depths = []
        
        for point in points:
            stress = self._calculate_stress_for_point(point, d0)
            stresses.append(stress)
            depths.append(point.depth)
        
        stresses = np.array(stresses)
        depths = np.array(depths)
        
        if thickness is not None:
            depths_normalized = depths / thickness
            z_center = depths_normalized - 0.5
            dz = thickness
        else:
            z_center = depths - np.mean(depths)
            dz = max(depths) - min(depths) if len(depths) > 1 else 1.0
        
        force_sum = np.trapezoid(stresses, depths) / dz
        moment_sum = np.trapezoid(stresses * z_center, depths) / dz ** 2
        
        return {
            'depths': depths,
            'stresses': stresses,
            'force_sum': float(force_sum),
            'moment_sum': float(moment_sum),
            'thickness': thickness,
            'is_balanced': abs(force_sum) < 1e-3 and abs(moment_sum) < 1e-3
        }

    @staticmethod
    def _strain_transformation_matrix(phi: float, psi: float, chi: float = 0.0) -> np.ndarray:
        """
        构建应变-应力转换矩阵（广义胡克定律）
        
        对于各向同性材料，平面应力状态下：
        epsilon_phi_psi = a1*sigma_xx + a2*sigma_yy + a3*sigma_zz 
                         + a4*sigma_xy + a5*sigma_xz + a6*sigma_yz
        
        参数:
            phi: 方位角 (度)
            psi: 倾角 (度)
            chi: 试样旋转角 (度)
            
        返回:
            6元素的转换系数数组
        """
        phi_rad = np.radians(phi)
        psi_rad = np.radians(psi)
        chi_rad = np.radians(chi)
        
        c_phi = np.cos(phi_rad)
        s_phi = np.sin(phi_rad)
        c_psi = np.cos(psi_rad)
        s_psi = np.sin(psi_rad)
        c_chi = np.cos(chi_rad)
        s_chi = np.sin(chi_rad)
        
        a1 = c_phi**2 * s_psi**2
        a2 = s_phi**2 * s_psi**2
        a3 = c_psi**2
        a4 = 2 * c_phi * s_phi * s_psi**2
        a5 = 2 * c_phi * s_psi * c_psi
        a6 = 2 * s_phi * s_psi * c_psi
        
        return np.array([a1, a2, a3, a4, a5, a6])

    @staticmethod
    def _build_stiffness_tensor(E: float, nu: float) -> np.ndarray:
        """
        构建各向同性材料的刚度张量（工程常数形式）
        
        参数:
            E: 杨氏模量 (MPa)
            nu: 泊松比
            
        返回:
            6x6刚度矩阵 (Voigt notation)
        """
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        
        C = np.zeros((6, 6))
        C[0, 0] = C[1, 1] = C[2, 2] = lam + 2 * mu
        C[0, 1] = C[0, 2] = C[1, 0] = C[1, 2] = C[2, 0] = C[2, 1] = lam
        C[3, 3] = C[4, 4] = C[5, 5] = mu
        
        return C

    @staticmethod
    def _build_compliance_tensor(E: float, nu: float) -> np.ndarray:
        """
        构建各向同性材料的柔度张量
        
        参数:
            E: 杨氏模量 (MPa)
            nu: 泊松比
            
        返回:
            6x6柔度矩阵 (Voigt notation)
        """
        S = np.zeros((6, 6))
        S[0, 0] = S[1, 1] = S[2, 2] = 1.0 / E
        S[0, 1] = S[0, 2] = S[1, 0] = S[1, 2] = S[2, 0] = S[2, 1] = -nu / E
        S[3, 3] = S[4, 4] = S[5, 5] = 2 * (1 + nu) / E
        
        return S

    def reconstruct_stress_tensor(
        self,
        measurements: List[TensorMeasurement],
        d0: Optional[float] = None,
        plane_stress: bool = False,
        constrain_sigma_zz: bool = False,
        sigma_zz_value: float = 0.0,
        hkl: Optional[Tuple[int, int, int]] = None
    ) -> StressTensorResult:
        """
        通过最小二乘法重构三维应力张量
        
        原理：
        利用多个方向的应变测量，通过广义胡克定律重构应力分量。
        
        晶格应变公式（各向同性材料）：
        epsilon_phi_psi = a1*sigma_xx + a2*sigma_yy + a3*sigma_zz 
                         + a4*sigma_xy + a5*sigma_xz + a6*sigma_yz
        
        其中转换系数由测量方向(phi, psi)决定。
        
        对于平面应力假设（sigma_zz = sigma_xz = sigma_yz = 0）：
        只需测量3个方向的应变即可确定sigma_xx, sigma_yy, sigma_xy
        
        参数:
            measurements: 张量测量数据列表
            d0: 无应力晶面间距 (如果测量中未指定)
            plane_stress: 是否假设平面应力状态
            constrain_sigma_zz: 是否约束sigma_zz为固定值
            sigma_zz_value: sigma_zz的约束值 (当constrain_sigma_zz=True时)
            hkl: 晶面指数 (用于各向异性修正)
            
        返回:
            StressTensorResult: 应力张量重构结果
        """
        E_eff, nu_eff = self._get_effective_modulus(hkl)
        
        n_measurements = len(measurements)
        
        if n_measurements < 3 and plane_stress:
            raise ValueError("平面应力状态至少需要3个方向的测量")
        if n_measurements < 6 and not plane_stress:
            raise ValueError("三维应力状态至少需要6个方向的测量")
        
        strains = []
        weights = []
        A_matrix = []
        
        for m in measurements:
            if m.strain is not None:
                strain = m.strain
            elif m.d_hkl is not None and (m.d0 is not None or d0 is not None):
                d0_val = m.d0 if m.d0 is not None else d0
                strain = self.calculate_strain(m.d_hkl, d0_val)
            else:
                continue
            
            strains.append(strain)
            weights.append(m.weight)
            
            a = self._strain_transformation_matrix(m.phi, m.psi, m.chi)
            A_matrix.append(a)
        
        strains = np.array(strains)
        weights = np.array(weights)
        A_matrix = np.array(A_matrix)
        
        if plane_stress:
            # 平面应力: sigma_zz = sigma_xz = sigma_yz = 0
            # epsilon = (1/E)*[(a1-a3*nu)*sigma_xx + (a2-a3*nu)*sigma_yy + a4*sigma_xy]
            a1 = A_matrix[:, 0]
            a2 = A_matrix[:, 1]
            a3 = A_matrix[:, 2]
            a4 = A_matrix[:, 3]
            
            A_reduced = np.column_stack([
                (a1 - nu_eff * a3) / E_eff,
                (a2 - nu_eff * a3) / E_eff,
                a4 / (2 * (1 + nu_eff)) / E_eff
            ])
            
            n_params = 3
        elif constrain_sigma_zz:
            # 约束sigma_zz: sigma_xz = sigma_yz = 0, sigma_zz = sigma_zz_value
            a1 = A_matrix[:, 0]
            a2 = A_matrix[:, 1]
            a3 = A_matrix[:, 2]
            a4 = A_matrix[:, 3]
            
            strains = strains - (a3 - nu_eff * (a1 + a2)) * sigma_zz_value / E_eff
            
            A_reduced = np.column_stack([
                (a1 - nu_eff * (a2 + a3)) / E_eff,
                (a2 - nu_eff * (a1 + a3)) / E_eff,
                a4 / (2 * (1 + nu_eff)) / E_eff
            ])
            
            n_params = 3
        else:
            # 三维应力状态
            a1 = A_matrix[:, 0]
            a2 = A_matrix[:, 1]
            a3 = A_matrix[:, 2]
            a4 = A_matrix[:, 3]
            a5 = A_matrix[:, 4]
            a6 = A_matrix[:, 5]
            
            A_reduced = np.column_stack([
                (a1 - nu_eff * (a2 + a3)) / E_eff,
                (a2 - nu_eff * (a1 + a3)) / E_eff,
                (a3 - nu_eff * (a1 + a2)) / E_eff,
                a4 / (2 * (1 + nu_eff)) / E_eff,
                a5 / (2 * (1 + nu_eff)) / E_eff,
                a6 / (2 * (1 + nu_eff)) / E_eff
            ])
            
            n_params = 6
        
        W = np.diag(np.sqrt(weights)) if len(weights) > 0 else np.eye(len(strains))
        
        try:
            if A_reduced.shape[0] > A_reduced.shape[1]:
                A_weighted = W @ A_reduced
                strains_weighted = W @ strains
                
                sigma_components, residuals_lstsq, rank, s = np.linalg.lstsq(
                    A_weighted, strains_weighted, rcond=None
                )
                
                residuals_full = A_reduced @ sigma_components - strains
                
                try:
                    covariance = np.linalg.inv(A_weighted.T @ A_weighted)
                except np.linalg.LinAlgError:
                    covariance = None
            else:
                sigma_components = np.linalg.solve(A_reduced, strains)
                residuals_full = np.zeros_like(strains)
                covariance = None
        except np.linalg.LinAlgError as e:
            raise ValueError(f"应力张量重构失败: {e}")
        
        if plane_stress:
            sigma_xx, sigma_yy, sigma_xy = sigma_components
            sigma_zz = 0.0
            sigma_xz = 0.0
            sigma_yz = 0.0
        elif constrain_sigma_zz:
            sigma_xx, sigma_yy, sigma_xy = sigma_components
            sigma_zz = sigma_zz_value
            sigma_xz = 0.0
            sigma_yz = 0.0
        else:
            sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_xz, sigma_yz = sigma_components
        
        stress_tensor = np.array([
            [sigma_xx, sigma_xy, sigma_xz],
            [sigma_xy, sigma_yy, sigma_yz],
            [sigma_xz, sigma_yz, sigma_zz]
        ])
        
        eigenvalues, eigenvectors = np.linalg.eigh(stress_tensor)
        idx = np.argsort(eigenvalues)[::-1]
        principal_stresses = tuple(eigenvalues[idx])
        principal_directions = eigenvectors[:, idx]
        
        von_mises = np.sqrt(0.5 * (
            (sigma_xx - sigma_yy)**2 + 
            (sigma_yy - sigma_zz)**2 + 
            (sigma_zz - sigma_xx)**2 + 
            6 * (sigma_xy**2 + sigma_xz**2 + sigma_yz**2)
        ))
        
        hydrostatic = (sigma_xx + sigma_yy + sigma_zz) / 3.0
        
        error_estimate = {}
        if covariance is not None:
            param_names = ['sigma_xx', 'sigma_yy', 'sigma_zz', 'sigma_xy', 'sigma_xz', 'sigma_yz']
            for i, name in enumerate(param_names[:len(sigma_components)]):
                if i < covariance.shape[0]:
                    rmse = np.sqrt(np.mean(residuals_full**2)) if len(residuals_full) > 0 else 0.0
                    error_estimate[name] = np.sqrt(covariance[i, i]) * rmse
        
        return StressTensorResult(
            sigma_xx=float(sigma_xx),
            sigma_yy=float(sigma_yy),
            sigma_zz=float(sigma_zz),
            sigma_xy=float(sigma_xy),
            sigma_xz=float(sigma_xz),
            sigma_yz=float(sigma_yz),
            covariance=covariance,
            residuals=list(residuals_full),
            principal_stresses=principal_stresses,
            principal_directions=principal_directions,
            von_mises=float(von_mises),
            hydrostatic=float(hydrostatic),
            error_estimate=error_estimate
        )

    def reconstruct_stress_depth_profile_3d(
        self,
        measurements_by_depth: Dict[float, List[TensorMeasurement]],
        d0: Optional[float] = None,
        plane_stress: bool = False,
        thickness: Optional[float] = None,
        hkl: Optional[Tuple[int, int, int]] = None
    ) -> Dict:
        """
        重构三维应力深度分布（适用于焊接件、增材制造件）
        
        参数:
            measurements_by_depth: 按深度分组的测量数据 {depth: [measurements]}
            d0: 无应力晶面间距
            plane_stress: 是否假设平面应力状态
            thickness: 试样厚度 (用于应力平衡分析)
            hkl: 晶面指数
            
        返回:
            包含深度应力分布的字典
        """
        depths = sorted(measurements_by_depth.keys())
        stress_tensors = []
        
        for depth in depths:
            measurements = measurements_by_depth[depth]
            tensor_result = self.reconstruct_stress_tensor(
                measurements, d0=d0, plane_stress=plane_stress, hkl=hkl
            )
            stress_tensors.append(tensor_result)
        
        result = {
            'depths': depths,
            'stress_tensors': stress_tensors,
            'sigma_xx_profile': [t.sigma_xx for t in stress_tensors],
            'sigma_yy_profile': [t.sigma_yy for t in stress_tensors],
            'sigma_zz_profile': [t.sigma_zz for t in stress_tensors],
            'sigma_xy_profile': [t.sigma_xy for t in stress_tensors],
            'von_mises_profile': [t.von_mises for t in stress_tensors],
            'hydrostatic_profile': [t.hydrostatic for t in stress_tensors],
        }
        
        if thickness is not None and len(depths) > 1:
            depths_array = np.array(depths)
            
            for comp in ['sigma_xx_profile', 'sigma_yy_profile', 'sigma_zz_profile']:
                profile = np.array(result[comp])
                force = np.trapezoid(profile, depths_array) / thickness
                result[f'{comp}_force'] = float(force)
            
            z_center = depths_array / thickness - 0.5
            for comp in ['sigma_xx_profile', 'sigma_yy_profile', 'sigma_zz_profile']:
                profile = np.array(result[comp])
                moment = np.trapezoid(profile * z_center, depths_array) / thickness**2
                result[f'{comp}_moment'] = float(moment)
        
        return result


def example_usage():
    """
    示例：演示如何使用残余应力计算器
    """
    print("=" * 60)
    print("中子衍射残余应力计算示例")
    print("=" * 60)
    
    # 1. 定义材料弹性常数（以铝合金为例）
    print("\n1. 定义材料弹性常数")
    print("-" * 40)
    al_elastic = ElasticConstants(
        E=70000.0,      # 杨氏模量, MPa
        nu=0.33,        # 泊松比
        crystal_structure="cubic",
        hkl_plane=(3, 1, 1)
    )
    print(f"  杨氏模量 E = {al_elastic.E} MPa")
    print(f"  泊松比 ν = {al_elastic.nu}")
    print(f"  晶体结构: {al_elastic.crystal_structure}")
    
    # 2. 创建计算器
    calculator = ResidualStressCalculator(al_elastic)
    
    # 3. 布拉格定律计算示例
    print("\n2. 布拉格定律计算示例")
    print("-" * 40)
    wavelength = 1.5406  # Angstrom, Cu K-alpha波长（示例）
    two_theta0 = 38.74   # 度, Al(311)无应力峰位
    d0 = calculator.bragg_law(two_theta0, wavelength)
    print(f"  中子波长 lambda = {wavelength} Angstrom")
    print(f"  无应力衍射角 2theta_0 = {two_theta0} deg")
    print(f"  无应力晶面间距 d0 = {d0:.4f} Angstrom")
    
    # 4. 从峰位偏移计算应变
    print("\n3. 峰位偏移与应变计算")
    print("-" * 40)
    delta_2theta = 0.05  # 峰位移, 度
    two_theta_stressed = two_theta0 + delta_2theta
    strain = calculator.calculate_strain_from_peak_shift(two_theta_stressed, two_theta0)
    d_stressed = calculator.bragg_law(two_theta_stressed, wavelength)
    strain_direct = calculator.calculate_strain(d_stressed, d0)
    
    print(f"  有应力衍射角 2theta = {two_theta_stressed} deg")
    print(f"  峰位偏移 delta(2theta) = {delta_2theta} deg")
    print(f"  有应力晶面间距 d = {d_stressed:.4f} Angstrom")
    print(f"  晶面间距变化 (d-d0)/d0 = {strain_direct*1e6:.1f} microstrain")
    print(f"  从峰位计算应变 epsilon = {strain*1e6:.1f} microstrain")
    print(f"  两种方法一致性: {np.isclose(strain, strain_direct)}")
    
    # 5. 单轴应力计算
    print("\n4. 单轴应力计算")
    print("-" * 40)
    stress_uniaxial = calculator.calculate_stress_uniaxial(strain, hkl=(3, 1, 1))
    print(f"  单轴应力 sigma = {stress_uniaxial:.1f} MPa")
    
    # 6. 平面应力计算（sin²ψ法单点）
    print("\n5. 平面应力计算 (psi=90 deg)")
    print("-" * 40)
    stress_plane = calculator.calculate_stress_plane_stress(
        strain, phi=0.0, psi=90.0, hkl=(3, 1, 1)
    )
    print(f"  平面应力 sigma_phi = {stress_plane:.1f} MPa")
    
    # 7. 完整sin²ψ法示例
    print("\n6. sin^2 psi法完整分析")
    print("-" * 40)
    psi_list = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]
    sigma_true = -200.0  # 假设真实应力, MPa
    E_eff, nu_eff = calculator._get_effective_modulus((3, 1, 1))
    slope_true = sigma_true * (1 + nu_eff) / E_eff
    
    strains_simulated = []
    for psi in psi_list:
        psi_rad = np.radians(psi)
        sin_sq_psi = np.sin(psi_rad)**2
        noise = np.random.normal(0, 10e-6)
        eps = slope_true * sin_sq_psi + noise
        strains_simulated.append(eps)
        print(f"  psi = {psi:5.1f} deg, sin^2 psi = {sin_sq_psi:.4f}, epsilon = {eps*1e6:6.1f} microstrain")
    
    sigma_fit, slope, intercept = calculator.calculate_stress_sin2psi(
        psi_list, strains_simulated, hkl=(3, 1, 1)
    )
    print(f"\n  拟合结果:")
    print(f"    斜率 M = {slope*1e6:.2f} microstrain")
    print(f"    截距 B = {intercept*1e6:.2f} microstrain")
    print(f"    拟合应力 sigma_phi = {sigma_fit:.1f} MPa")
    print(f"    真实应力 sigma_true = {sigma_true:.1f} MPa")
    print(f"    相对误差 = {abs(sigma_fit - sigma_true)/abs(sigma_true)*100:.1f}%")
    
    # 8. 从多个衍射峰计算
    print("\n7. 多衍射峰应力计算")
    print("-" * 40)
    peaks = [
        DiffractionPeak(two_theta=38.74 + 0.03, phi=0.0, psi=90.0, hkl=(3, 1, 1)),
        DiffractionPeak(two_theta=38.74 + 0.05, phi=45.0, psi=90.0, hkl=(3, 1, 1)),
        DiffractionPeak(two_theta=38.74 + 0.04, phi=90.0, psi=90.0, hkl=(3, 1, 1)),
    ]
    
    stresses = calculator.calculate_stress_from_peaks(
        peaks, d0, wavelength, method="plane_stress"
    )
    
    for i, (peak, stress) in enumerate(zip(peaks, stresses)):
        print(f"  峰 {i+1}: phi={peak.phi:.0f} deg, 2theta={peak.two_theta:.2f} deg, sigma={stress:.1f} MPa")
    
    return {
        'd0': d0,
        'strain': strain,
        'stress_uniaxial': stress_uniaxial,
        'stress_plane': stress_plane,
        'sigma_fit': sigma_fit,
        'sigma_true': sigma_true,
    }


def example_d0_calibration():
    """
    示例：演示d0自洽标定和多峰拟合
    """
    print("\n" + "=" * 60)
    print("d0自洽标定和多峰拟合示例")
    print("=" * 60)
    
    # 定义材料弹性常数
    al_elastic = ElasticConstants(
        E=70000.0,
        nu=0.33,
        crystal_structure="cubic"
    )
    calculator = ResidualStressCalculator(al_elastic)
    wavelength = 1.5406
    
    # ===== 示例1: 应力平衡条件自洽标定d0 =====
    print("\n1. 应力平衡条件自洽标定d0")
    print("-" * 40)
    
    thickness = 3.0  # 试样厚度, mm
    d0_true = 2.3225  # 真实无应力晶面间距, Angstrom
    
    # 模拟一个平衡的残余应力分布（抛物线型）
    n_points = 11
    depths = np.linspace(0, thickness, n_points)
    z_center = depths / thickness - 0.5
    
    sigma_max = 150.0  # MPa, 最大应力
    sigma_profile = sigma_max * (1 - 4 * z_center**2)
    
    # 确保应力平衡
    E_eff, nu_eff = calculator._get_effective_modulus((3, 1, 1))
    strain_profile = (sigma_profile * (1 + nu_eff)) / E_eff
    
    # 生成模拟的d测量值
    d_measured = d0_true * (1 + strain_profile)
    d0_biased = d0_true * 1.0005  # 有偏差的d0
    
    print(f"  真实d0: {d0_true:.4f} Angstrom")
    print(f"  偏差d0: {d0_biased:.4f} Angstrom (偏差: {(d0_biased-d0_true)/d0_true*1e6:.0f} microstrain)")
    print(f"  试样厚度: {thickness} mm")
    print(f"  最大应力: {sigma_max:.1f} MPa")
    
    # 创建测量点
    points = []
    for i in range(n_points):
        two_theta = calculator.bragg_law_inverse(d_measured[i], wavelength)
        point = StressMeasurementPoint(
            depth=depths[i],
            d_hkl=d_measured[i],
            phi=0.0,
            psi=90.0,
            hkl=(3, 1, 1),
            wavelength=wavelength,
            two_theta=two_theta,
            weight=1.0
        )
        points.append(point)
    
    # 检查使用偏差d0时的应力平衡
    profile_biased = calculator.calculate_stress_depth_profile(
        points, d0_biased, thickness
    )
    print(f"\n  使用偏差d0的应力平衡:")
    print(f"    合力残差: {profile_biased['force_sum']:.6f} MPa")
    print(f"    合力矩残差: {profile_biased['moment_sum']:.6f} MPa")
    print(f"    是否平衡: {profile_biased['is_balanced']}")
    
    # 进行自洽标定
    print(f"\n  开始自洽标定...")
    calib_result = calculator.calibrate_d0_force_balance(
        points,
        d0_initial=d0_biased,
        thickness=thickness,
        balance_type="force_moment",
        d0_range=(d0_true * 0.995, d0_true * 1.005),
        max_iterations=200
    )
    
    print(f"  标定结果:")
    print(f"    初始d0: {d0_biased:.6f} Angstrom")
    print(f"    标定d0: {calib_result.d0:.6f} Angstrom")
    print(f"    真实d0: {d0_true:.6f} Angstrom")
    print(f"    标定误差: {(calib_result.d0 - d0_true)/d0_true*1e6:.1f} microstrain")
    print(f"    迭代次数: {calib_result.iterations}")
    print(f"    最终残差: {calib_result.residual:.2e}")
    print(f"    应力平衡残差: {calib_result.balance_residual}")
    
    # 检查标定后的应力平衡
    profile_calibrated = calculator.calculate_stress_depth_profile(
        points, calib_result.d0, thickness
    )
    print(f"\n  标定后的应力平衡:")
    print(f"    合力残差: {profile_calibrated['force_sum']:.10f} MPa")
    print(f"    合力矩残差: {profile_calibrated['moment_sum']:.10f} MPa")
    print(f"    是否平衡: {profile_calibrated['is_balanced']}")
    
    # 打印应力分布对比
    print(f"\n  应力分布对比 (前5个点):")
    print(f"  {'深度 (mm)':>12} {'真实应力':>10} {'偏差d0计算':>12} {'标定d0计算':>12}")
    print(f"  {'-'*12} {'-'*10} {'-'*12} {'-'*12}")
    for i in range(5):
        d = depths[i]
        s_true = sigma_profile[i]
        s_biased = profile_biased['stresses'][i]
        s_calib = profile_calibrated['stresses'][i]
        print(f"  {d:12.3f} {s_true:10.1f} {s_biased:12.1f} {s_calib:12.1f}")
    
    # ===== 示例2: 多峰拟合确定d0 =====
    print("\n\n2. 多峰拟合确定d0")
    print("-" * 40)
    
    # 定义多个hkl晶面的真实d0
    d0_true_multi = {
        (1, 1, 1): 2.3385,
        (2, 0, 0): 2.0245,
        (2, 2, 0): 1.4315,
        (3, 1, 1): 1.2210
    }
    
    # 假设相同的应力状态（自洽）
    sigma_assumed = 100.0  # MPa
    
    peaks_by_hkl = {}
    d0_by_hkl_biased = {}
    
    for hkl, d0_hkl in d0_true_multi.items():
        E_hkl, nu_hkl = calculator._get_effective_modulus(hkl)
        strain_hkl = sigma_assumed * (1 + nu_hkl) / E_hkl
        
        # 生成多个psi角的测量（不包括psi=0，避免除零问题）
        peaks = []
        for psi in [15, 30, 45, 60, 75, 90]:
            psi_rad = np.radians(psi)
            sin_sq_psi = np.sin(psi_rad)**2
            strain_psi = strain_hkl * sin_sq_psi
            d_psi = d0_hkl * (1 + strain_psi)
            two_theta_psi = calculator.bragg_law_inverse(d_psi, wavelength)
            
            peak = DiffractionPeak(
                two_theta=two_theta_psi,
                d_hkl=d_psi,
                hkl=hkl,
                phi=0.0,
                psi=psi,
                wavelength=wavelength
            )
            peaks.append(peak)
        
        peaks_by_hkl[hkl] = peaks
        d0_by_hkl_biased[hkl] = d0_hkl * 1.0003  # 引入小的偏差
    
    # 多峰拟合
    print(f"  真实d0和偏差d0:")
    for hkl in d0_true_multi:
        d0_true = d0_true_multi[hkl]
        d0_bias = d0_by_hkl_biased[hkl]
        bias = (d0_bias - d0_true) / d0_true * 1e6
        print(f"    hkl={hkl}: 真实d0={d0_true:.4f}, 偏差d0={d0_bias:.4f}, 偏差={bias:.0f} microstrain")
    
    print(f"\n  开始多峰拟合 (strain_linear模式)...")
    multi_results_linear = calculator.calibrate_d0_multi_peak(
        peaks_by_hkl,
        d0_by_hkl_biased,
        wavelength,
        fit_type="strain_linear"
    )
    
    print(f"\n  线性拟合结果:")
    print(f"  {'hkl':>8} {'真实d0':>10} {'初始d0':>10} {'拟合d0':>10} {'误差(ustr)':>10}")
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    
    for hkl, result in multi_results_linear.items():
        d0_true = d0_true_multi[hkl]
        d0_initial = d0_by_hkl_biased[hkl]
        d0_fit = result.d0
        error = (d0_fit - d0_true) / d0_true * 1e6
        print(f"  {str(hkl):>8} {d0_true:10.4f} {d0_initial:10.4f} {d0_fit:10.4f} {error:10.1f}")
    
    # 使用应力一致性模式拟合
    print(f"\n  开始多峰拟合 (stress_consistency模式)...")
    multi_results_consistency = calculator.calibrate_d0_multi_peak(
        peaks_by_hkl,
        d0_by_hkl_biased,
        wavelength,
        fit_type="stress_consistency"
    )
    
    print(f"\n  应力一致性拟合结果:")
    print(f"  {'hkl':>8} {'真实d0':>10} {'初始d0':>10} {'拟合d0':>10} {'误差(ustr)':>10} {'应力方差':>12}")
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")
    
    for hkl, result in multi_results_consistency.items():
        d0_true = d0_true_multi[hkl]
        d0_initial = d0_by_hkl_biased[hkl]
        d0_fit = result.d0
        error = (d0_fit - d0_true) / d0_true * 1e6
        stress_variance = np.var(result.stress_profile) if result.stress_profile else 0
        print(f"  {str(hkl):>8} {d0_true:10.4f} {d0_initial:10.4f} {d0_fit:10.4f} {error:10.1f} {stress_variance:12.2f}")
    
    # 打印收敛历史
    if calib_result.convergence_history:
        print(f"\n  收敛历史 (前10次迭代):")
        for i, val in enumerate(calib_result.convergence_history[:10]):
            print(f"    迭代 {i+1}: 残差 = {val:.6e}")
        if len(calib_result.convergence_history) > 10:
            print(f"    ... (共 {len(calib_result.convergence_history)} 次迭代)")
    
    print("\n" + "=" * 60)
    print("d0标定示例完成")
    print("=" * 60)
    
    return {
        'force_balance_result': calib_result,
        'multi_peak_results_linear': multi_results_linear,
        'multi_peak_results_consistency': multi_results_consistency,
        'profile_biased': profile_biased,
        'profile_calibrated': profile_calibrated
    }


def example_tensor_analysis():
    """
    示例：三维残余应力张量分析
    适用于焊接件、增材制造件的应力分析
    """
    print("\n" + "=" * 60)
    print("三维残余应力张量分析示例")
    print("=" * 60)
    
    # 定义材料弹性常数（钛合金，常用于增材制造）
    ti_elastic = ElasticConstants(
        E=114000.0,
        nu=0.34,
        crystal_structure="hexagonal"
    )
    calculator = ResidualStressCalculator(ti_elastic)
    wavelength = 1.5406
    
    # ===== 示例1: 平面应力状态下的张量重构 =====
    print("\n1. 平面应力状态张量重构 (焊接件典型)")
    print("-" * 40)
    
    # 定义焊接件的真实应力状态（平面应力）
    sigma_xx_true = 200.0  # 纵向应力, MPa (焊接方向)
    sigma_yy_true = 150.0  # 横向应力, MPa
    sigma_xy_true = 30.0   # 剪切应力, MPa
    
    print(f"  真实应力状态:")
    print(f"    sigma_xx = {sigma_xx_true:.1f} MPa")
    print(f"    sigma_yy = {sigma_yy_true:.1f} MPa")
    print(f"    sigma_xy = {sigma_xy_true:.1f} MPa")
    
    # 生成多方向的模拟测量数据
    d0 = 1.5875  # Ti(10-11)面间距
    
    measurements_plane = []
    for phi in [0, 30, 60, 90, 120, 150]:  # 更多方位角
        for psi in [30, 45, 60, 90]:   # 避免psi=0
            phi_rad = np.radians(phi)
            psi_rad = np.radians(psi)
            
            # 计算该方向的应变（使用与重构相同的公式）
            a = calculator._strain_transformation_matrix(phi, psi)
            E_val = ti_elastic.E
            nu_val = ti_elastic.nu
            
            a1, a2, a3, a4, a5, a6 = a
            
            # 平面应力: sigma_zz = sigma_xz = sigma_yz = 0
            strain = ((a1 - nu_val * a3) * sigma_xx_true + 
                     (a2 - nu_val * a3) * sigma_yy_true + 
                     a4 / (2 * (1 + nu_val)) * sigma_xy_true) / E_val
            
            # 添加测量噪声
            strain_noisy = strain + np.random.normal(0, 10e-6)
            
            m = TensorMeasurement(
                phi=phi,
                psi=psi,
                chi=0.0,
                strain=strain_noisy,
                d_hkl=d0 * (1 + strain_noisy),
                d0=d0,
                hkl=(1, 0, 1),
                weight=1.0
            )
            measurements_plane.append(m)
    
    # 重构应力张量（平面应力假设）
    print(f"\n  测量配置:")
    print(f"    方位角范围: 0-150 deg, 步长30 deg")
    print(f"    倾角范围: 30-90 deg, 步长15 deg")
    print(f"    测量点数: {len(measurements_plane)}")
    print(f"    假设: 平面应力状态")
    
    result_plane = calculator.reconstruct_stress_tensor(
        measurements_plane,
        d0=d0,
        plane_stress=True,
        hkl=(1, 0, 1)
    )
    
    print(f"\n  重构结果:")
    print(f"    sigma_xx = {result_plane.sigma_xx:.1f} MPa (真实: {sigma_xx_true:.1f})")
    print(f"    sigma_yy = {result_plane.sigma_yy:.1f} MPa (真实: {sigma_yy_true:.1f})")
    print(f"    sigma_xy = {result_plane.sigma_xy:.1f} MPa (真实: {sigma_xy_true:.1f})")
    print(f"    冯米塞斯应力: {result_plane.von_mises:.1f} MPa")
    print(f"    静水压力: {result_plane.hydrostatic:.1f} MPa")
    print(f"    主应力: {[f'{s:.1f}' for s in result_plane.principal_stresses]}")
    print(f"    拟合残差: {np.mean(np.abs(result_plane.residuals))*1e6:.1f} microstrain")
    
    # ===== 示例2: 三维应力状态（增材制造件典型） =====
    print("\n\n2. 三维应力状态张量重构 (增材制造件典型)")
    print("-" * 40)
    
    # 定义增材制造件的真实三维应力状态
    sigma_xx_3d = 180.0
    sigma_yy_3d = 120.0
    sigma_zz_3d = 80.0
    sigma_xy_3d = 25.0
    sigma_xz_3d = 15.0
    sigma_yz_3d = 10.0
    
    print(f"  真实应力状态:")
    print(f"    sigma_xx = {sigma_xx_3d:.1f} MPa")
    print(f"    sigma_yy = {sigma_yy_3d:.1f} MPa")
    print(f"    sigma_zz = {sigma_zz_3d:.1f} MPa")
    print(f"    sigma_xy = {sigma_xy_3d:.1f} MPa")
    print(f"    sigma_xz = {sigma_xz_3d:.1f} MPa")
    print(f"    sigma_yz = {sigma_yz_3d:.1f} MPa")
    
    # 生成多方向的模拟测量数据（需要更多方向）
    measurements_3d = []
    for phi in [0, 30, 60, 90, 120, 150]:
        for psi in [0, 30, 60, 90]:
            for chi in [0, 45]:
                a = calculator._strain_transformation_matrix(phi, psi, chi)
                E_val = ti_elastic.E
                nu_val = ti_elastic.nu
                
                a1, a2, a3, a4, a5, a6 = a
                
                # 三维应力状态
                strain = ((a1 - nu_val * (a2 + a3)) * sigma_xx_3d + 
                         (a2 - nu_val * (a1 + a3)) * sigma_yy_3d + 
                         (a3 - nu_val * (a1 + a2)) * sigma_zz_3d + 
                         a4 / (2 * (1 + nu_val)) * sigma_xy_3d +
                         a5 / (2 * (1 + nu_val)) * sigma_xz_3d +
                         a6 / (2 * (1 + nu_val)) * sigma_yz_3d) / E_val
                
                strain_noisy = strain + np.random.normal(0, 5e-6)
                
                m = TensorMeasurement(
                    phi=phi,
                    psi=psi,
                    chi=chi,
                    strain=strain_noisy,
                    d_hkl=d0 * (1 + strain_noisy),
                    d0=d0,
                    hkl=(1, 0, 1),
                    weight=1.0
                )
                measurements_3d.append(m)
    
    print(f"\n  测量配置:")
    print(f"    方位角: 6个方向, 倾角: 4个方向, 旋转角: 2个方向")
    print(f"    测量点数: {len(measurements_3d)}")
    print(f"    假设: 三维应力状态")
    
    # 重构三维应力张量
    result_3d = calculator.reconstruct_stress_tensor(
        measurements_3d,
        d0=d0,
        plane_stress=False,
        hkl=(1, 0, 1)
    )
    
    print(f"\n  重构结果:")
    print(f"    sigma_xx = {result_3d.sigma_xx:.1f} MPa (真实: {sigma_xx_3d:.1f})")
    print(f"    sigma_yy = {result_3d.sigma_yy:.1f} MPa (真实: {sigma_yy_3d:.1f})")
    print(f"    sigma_zz = {result_3d.sigma_zz:.1f} MPa (真实: {sigma_zz_3d:.1f})")
    print(f"    sigma_xy = {result_3d.sigma_xy:.1f} MPa (真实: {sigma_xy_3d:.1f})")
    print(f"    sigma_xz = {result_3d.sigma_xz:.1f} MPa (真实: {sigma_xz_3d:.1f})")
    print(f"    sigma_yz = {result_3d.sigma_yz:.1f} MPa (真实: {sigma_yz_3d:.1f})")
    print(f"    冯米塞斯应力: {result_3d.von_mises:.1f} MPa")
    print(f"    主应力: {[f'{s:.1f}' for s in result_3d.principal_stresses]}")
    print(f"    拟合残差: {np.mean(np.abs(result_3d.residuals))*1e6:.1f} microstrain")
    
    # ===== 示例3: 焊接件深度方向应力分布 =====
    print("\n\n3. 焊接件深度方向应力分布分析")
    print("-" * 40)
    
    thickness = 4.0  # 试样厚度, mm
    depths = np.linspace(0, thickness, 9)
    
    # 定义深度方向的应力分布（焊接件典型的M型分布）
    def weld_stress_profile(z, t):
        z_norm = z / t - 0.5
        # 表面拉应力，内部压应力
        sigma_xx = 150 * (1 - 4 * z_norm**2) + 50 * np.cos(2 * np.pi * z_norm)
        sigma_yy = 100 * (1 - 4 * z_norm**2) + 30 * np.cos(2 * np.pi * z_norm)
        sigma_xy = 20 * np.sin(2 * np.pi * z_norm)
        return sigma_xx, sigma_yy, sigma_xy
    
    measurements_by_depth = {}
    
    for depth in depths:
        sigma_xx_z, sigma_yy_z, sigma_xy_z = weld_stress_profile(depth, thickness)
        
        depth_measurements = []
        for phi in [0, 45, 90]:
            psi = 90  # 横向测量
            a = calculator._strain_transformation_matrix(phi, psi)
            E_val = ti_elastic.E
            nu_val = ti_elastic.nu
            
            a1, a2, a3, a4, a5, a6 = a
            
            # 平面应力: sigma_zz = sigma_xz = sigma_yz = 0
            strain = ((a1 - nu_val * a3) * sigma_xx_z + 
                     (a2 - nu_val * a3) * sigma_yy_z + 
                     a4 / (2 * (1 + nu_val)) * sigma_xy_z) / E_val
            
            strain_noisy = strain + np.random.normal(0, 5e-6)
            
            m = TensorMeasurement(
                phi=phi,
                psi=psi,
                chi=0.0,
                strain=strain_noisy,
                d_hkl=d0 * (1 + strain_noisy),
                d0=d0,
                hkl=(1, 0, 1),
                weight=1.0
            )
            depth_measurements.append(m)
        
        measurements_by_depth[depth] = depth_measurements
    
    print(f"  分析配置:")
    print(f"    试样厚度: {thickness} mm")
    print(f"    深度点数: {len(depths)}")
    print(f"    每点测量方向: 3个")
    
    # 重构深度方向应力分布
    depth_profile = calculator.reconstruct_stress_depth_profile_3d(
        measurements_by_depth,
        d0=d0,
        plane_stress=True,
        thickness=thickness,
        hkl=(1, 0, 1)
    )
    
    print(f"\n  深度方向应力分布:")
    print(f"  {'深度 (mm)':>10} {'sigma_xx':>10} {'sigma_yy':>10} {'sigma_xy':>10} {'von_mises':>10}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    
    for i, d in enumerate(depth_profile['depths']):
        s_xx = depth_profile['sigma_xx_profile'][i]
        s_yy = depth_profile['sigma_yy_profile'][i]
        s_xy = depth_profile['sigma_xy_profile'][i]
        vm = depth_profile['von_mises_profile'][i]
        print(f"  {d:10.2f} {s_xx:10.1f} {s_yy:10.1f} {s_xy:10.1f} {vm:10.1f}")
    
    print(f"\n  应力平衡检查:")
    print(f"    sigma_xx 合力: {depth_profile.get('sigma_xx_profile_force', 0):.2f} MPa")
    print(f"    sigma_yy 合力: {depth_profile.get('sigma_yy_profile_force', 0):.2f} MPa")
    print(f"    sigma_xx 合力矩: {depth_profile.get('sigma_xx_profile_moment', 0):.2f} MPa")
    print(f"    sigma_yy 合力矩: {depth_profile.get('sigma_yy_profile_moment', 0):.2f} MPa")
    
    print("\n" + "=" * 60)
    print("三维应力张量分析示例完成")
    print("=" * 60)
    
    return {
        'plane_stress_result': result_plane,
        '3d_stress_result': result_3d,
        'depth_profile': depth_profile
    }


if __name__ == "__main__":
    results = example_usage()
    calibration_results = example_d0_calibration()
    tensor_results = example_tensor_analysis()

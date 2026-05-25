import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D

rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
rcParams['axes.unicode_minus'] = False


class HashinFailureCriteria:
    """Hashin失效准则 - 包含三维应力状态"""
    
    def __init__(self, strength_properties):
        self.Xt = strength_properties['Xt']
        self.Xc = strength_properties['Xc']
        self.Yt = strength_properties['Yt']
        self.Yc = strength_properties['Yc']
        self.Zt = strength_properties.get('Zt', self.Yt)
        self.Zc = strength_properties.get('Zc', self.Yc)
        self.S12 = strength_properties['S12']
        self.S13 = strength_properties.get('S13', self.S12)
        self.S23 = strength_properties.get('S23', self.S12 * 0.6)
        
    def fiber_tension_3d(self, sigma1, tau12, tau13):
        """纤维拉伸失效准则"""
        if sigma1 >= 0:
            return (sigma1 / self.Xt)**2 + (tau12 / self.S12)**2 + (tau13 / self.S13)**2
        else:
            return 0.0
    
    def fiber_compression_3d(self, sigma1):
        """纤维压缩失效准则"""
        if sigma1 < 0:
            return (sigma1 / self.Xc)**2
        else:
            return 0.0
    
    def matrix_tension_3d(self, sigma2, sigma3, tau12, tau13, tau23):
        """基体拉伸失效准则"""
        sigma23 = sigma2 + sigma3
        if sigma23 >= 0:
            return (sigma23 / self.Yt)**2 + (tau12 / self.S12)**2 + (tau13 / self.S13)**2 + (tau23 / self.S23)**2
        else:
            return 0.0
    
    def matrix_compression_3d(self, sigma2, sigma3, tau12, tau13, tau23):
        """基体压缩失效准则"""
        sigma23 = sigma2 + sigma3
        if sigma23 < 0:
            return (sigma23 / (2 * self.S23))**2 + ((self.Yc / (2 * self.S23))**2 - 1) * (sigma23 / self.Yc) + \
                   (tau12 / self.S12)**2 + (tau13 / self.S13)**2 + (tau23 / self.S23)**2
        else:
            return 0.0
    
    def delamination_tension_3d(self, sigma3, tau13, tau23):
        """分层拉伸失效准则"""
        if sigma3 >= 0:
            return (sigma3 / self.Zt)**2 + (tau13 / self.S13)**2 + (tau23 / self.S23)**2
        else:
            return 0.0
    
    def delamination_compression_3d(self, sigma3, tau13, tau23):
        """分层压缩失效准则"""
        if sigma3 < 0:
            return (tau13 / self.S13)**2 + (tau23 / self.S23)**2
        else:
            return 0.0
    
    def check_failure_3d(self, sigma_material):
        """
        检查三维应力状态下的失效
        
        参数:
            sigma_material: 材料主方向应力 [σ1, σ2, σ3, τ23, τ13, τ12]
        """
        sigma1, sigma2, sigma3 = sigma_material[0], sigma_material[1], sigma_material[2]
        tau23, tau13, tau12 = sigma_material[3], sigma_material[4], sigma_material[5]
        
        failure_indices = {
            'fiber_tension': self.fiber_tension_3d(sigma1, tau12, tau13),
            'fiber_compression': self.fiber_compression_3d(sigma1),
            'matrix_tension': self.matrix_tension_3d(sigma2, sigma3, tau12, tau13, tau23),
            'matrix_compression': self.matrix_compression_3d(sigma2, sigma3, tau12, tau13, tau23),
            'delamination_tension': self.delamination_tension_3d(sigma3, tau13, tau23),
            'delamination_compression': self.delamination_compression_3d(sigma3, tau13, tau23)
        }
        
        failure_modes = []
        for mode, index in failure_indices.items():
            if index >= 1.0:
                failure_modes.append(mode)
        
        max_index = max(failure_indices.values())
        is_failed = max_index >= 1.0
        
        return {
            'is_failed': is_failed,
            'max_index': max_index,
            'failure_modes': failure_modes,
            'indices': failure_indices
        }


class DelaminationGrowthModel:
    """分层扩展模型 - 基于内聚力模型和能量释放率"""
    
    def __init__(self, fracture_properties, interface_stiffness=1e15):
        self.GIC = fracture_properties.get('GIC', 200.0)
        self.GIIC = fracture_properties.get('GIIC', 800.0)
        self.GIIIC = fracture_properties.get('GIIIC', 800.0)
        self.interface_stiffness = interface_stiffness
        self.eta = fracture_properties.get('eta', 1.5)
        self.BK_exponent = fracture_properties.get('BK_exponent', 1.5)
        
        self.max_traction_n = np.sqrt(2 * self.GIC * self.interface_stiffness)
        self.max_traction_s = np.sqrt(2 * self.GIIC * self.interface_stiffness)
        
    def calculate_energy_release_rate(self, stresses, delamination_length, h=0.001):
        """
        计算能量释放率分量
        
        参数:
            stresses: 层间应力 [σz, τxz, τyz]
            delamination_length: 分层长度
        """
        sigma_z = stresses[0]
        tau_xz = stresses[1]
        tau_yz = stresses[2]
        
        if delamination_length <= 0:
            return {'GI': 0, 'GII': 0, 'GIII': 0}
        
        GI = max(0, sigma_z**2) * delamination_length / (2 * self.interface_stiffness) if sigma_z > 0 else 0
        GII = tau_xz**2 * delamination_length / (2 * self.interface_stiffness)
        GIII = tau_yz**2 * delamination_length / (2 * self.interface_stiffness)
        
        return {'GI': GI, 'GII': GII, 'GIII': GIII}
    
    def mixed_mode_criterion(self, G, mode='Benzeggagh_Kenane'):
        """
        混合模式失效准则
        
        参数:
            G: 能量释放率字典 {'GI', 'GII', 'GIII'}
            mode: 准则类型
        """
        GI, GII, GIII = G['GI'], G['GII'], G['GIII']
        G_total = GI + GII + GIII
        
        if G_total <= 0:
            return 0.0
        
        if mode == 'Benzeggagh_Kenane':
            G_eq = GII + GIII
            if GI + G_eq > 0:
                B_K = self.BK_exponent
                Gc = self.GIC + (self.GIIC - self.GIC) * (G_eq / (GI + G_eq))**B_K
                return G_total / Gc
            else:
                return 0.0
        elif mode == 'power_law':
            eta = self.eta
            return (GI / self.GIC)**eta + (GII / self.GIIC)**eta + (GIII / self.GIIIC)**eta
        else:
            return G_total / self.GIC
    
    def predict_delamination_growth(self, stresses, current_length, h=0.001):
        """预测分层扩展"""
        G = self.calculate_energy_release_rate(stresses, current_length, h)
        ratio = self.mixed_mode_criterion(G)
        
        if ratio >= 1.0:
            growth_rate = 0.1 * (ratio - 0.9)
            new_length = current_length * (1 + growth_rate)
            return {
                'is_growing': True,
                'growth_ratio': ratio,
                'new_length': new_length,
                'energy_release': G
            }
        else:
            return {
                'is_growing': False,
                'growth_ratio': ratio,
                'new_length': current_length,
                'energy_release': G
            }


class ThreeDStressCalculator:
    """三维层间应力计算器 - 集成Pagano方法"""
    
    def __init__(self, ply_properties, layup, ply_thickness, plate_width=None):
        self.ply_props = ply_properties
        self.layup = layup
        self.t_ply = ply_thickness
        self.n_plies = len(layup)
        self.h_total = self.n_plies * ply_thickness
        self.b = plate_width if plate_width else 20 * self.h_total
        
        self.C_bar = []
        self._initialize_stiffness()
        
    def _initialize_stiffness(self):
        for theta in self.layup:
            C_bar = self._get_3d_stiffness_matrix(theta)
            self.C_bar.append(C_bar)
    
    def _get_3d_stiffness_matrix(self, theta):
        """获取三维刚度矩阵（考虑损伤）"""
        E1 = self.ply_props['E1']
        E2 = self.ply_props['E2']
        E3 = self.ply_props.get('E3', E2)
        nu12 = self.ply_props['nu12']
        nu13 = self.ply_props.get('nu13', nu12)
        nu23 = self.ply_props.get('nu23', 0.3)
        G12 = self.ply_props['G12']
        G13 = self.ply_props['G13']
        G23 = self.ply_props['G23']
        
        nu21 = nu12 * E2 / E1
        nu31 = nu13 * E3 / E1
        nu32 = nu23 * E3 / E2
        
        delta = 1 - nu12*nu21 - nu23*nu32 - nu13*nu31 - 2*nu12*nu23*nu31
        
        C = np.zeros((6, 6))
        C[0, 0] = E1 * (1 - nu23*nu32) / delta
        C[0, 1] = E2 * (nu12 + nu13*nu32) / delta
        C[0, 2] = E3 * (nu13 + nu12*nu23) / delta
        C[1, 0] = C[0, 1]
        C[1, 1] = E2 * (1 - nu13*nu31) / delta
        C[1, 2] = E3 * (nu23 + nu12*nu13) / delta
        C[2, 0] = C[0, 2]
        C[2, 1] = C[1, 2]
        C[2, 2] = E3 * (1 - nu12*nu21) / delta
        C[3, 3] = G23
        C[4, 4] = G13
        C[5, 5] = G12
        
        theta_rad = np.radians(theta)
        c = np.cos(theta_rad)
        s = np.sin(theta_rad)
        
        T = np.zeros((6, 6))
        T[0, 0] = c**2
        T[0, 1] = s**2
        T[0, 5] = 2*s*c
        T[1, 0] = s**2
        T[1, 1] = c**2
        T[1, 5] = -2*s*c
        T[2, 2] = 1
        T[3, 3] = c
        T[3, 4] = -s
        T[4, 3] = s
        T[4, 4] = c
        T[5, 0] = -s*c
        T[5, 1] = s*c
        T[5, 5] = c**2 - s**2
        
        C_bar = T @ C @ T.T
        
        return C_bar
    
    def calculate_pagano_stresses(self, Mx, My, n_terms=15):
        """
        使用Pagano方法计算三维应力场
        
        参数:
            Mx, My: 单位长度弯矩
            n_terms: 傅里叶级数项数
        """
        h = self.h_total
        b = self.b
        
        n_y = 40
        n_z = 40
        y_arr = np.linspace(-b/2, b/2, n_y)
        z_arr = np.linspace(-h/2, h/2, n_z)
        
        sigma_x = np.zeros((n_y, n_z))
        sigma_y = np.zeros((n_y, n_z))
        sigma_z = np.zeros((n_y, n_z))
        tau_xz = np.zeros((n_y, n_z))
        tau_yz = np.zeros((n_y, n_z))
        tau_xy = np.zeros((n_y, n_z))
        
        C11_avg = np.mean([C[0, 0] for C in self.C_bar])
        C22_avg = np.mean([C[1, 1] for C in self.C_bar])
        C12_avg = np.mean([C[0, 1] for C in self.C_bar])
        C55_avg = np.mean([C[4, 4] for C in self.C_bar])
        C44_avg = np.mean([C[3, 3] for C in self.C_bar])
        
        for m in range(1, n_terms + 1):
            alpha_m = m * np.pi / b
            X_m = 2 * np.sin(alpha_m * b / 2) / (alpha_m * b)
            
            for i, y in enumerate(y_arr):
                sin_term = np.sin(alpha_m * y)
                cos_term = np.cos(alpha_m * y)
                
                for j, z in enumerate(z_arr):
                    zi_norm = 2 * z / h
                    f_z = (1 - zi_norm**2)
                    
                    A_m = 12 * Mx * X_m / (C11_avg * h**3)
                    B_m = 12 * My * X_m / (C22_avg * h**3)
                    
                    sigma_x[i, j] += A_m * z * cos_term / m
                    sigma_y[i, j] += B_m * z * cos_term / m
                    tau_xz[i, j] += A_m * C55_avg * alpha_m * f_z * sin_term / (2 * m)
                    tau_yz[i, j] += B_m * C44_avg * alpha_m * f_z * sin_term / (2 * m)
                    sigma_z[i, j] += 0.1 * (A_m + B_m) * f_z * cos_term / m
                    tau_xy[i, j] += 0.5 * (A_m + B_m) * C12_avg / C11_avg * z * sin_term / m
        
        return {
            'y': y_arr,
            'z': z_arr,
            'sigma_x': sigma_x,
            'sigma_y': sigma_y,
            'sigma_z': sigma_z,
            'tau_xz': tau_xz,
            'tau_yz': tau_yz,
            'tau_xy': tau_xy
        }
    
    def get_interlaminar_stresses_at_interface(self, stress_field, interface_idx):
        """获取指定层间界面的应力"""
        z_pos = -self.h_total/2 + (interface_idx + 1) * self.t_ply
        z_idx = np.argmin(np.abs(stress_field['z'] - z_pos))
        
        return {
            'y': stress_field['y'],
            'sigma_z': stress_field['sigma_z'][:, z_idx],
            'tau_xz': stress_field['tau_xz'][:, z_idx],
            'tau_yz': stress_field['tau_yz'][:, z_idx]
        }


class ProgressiveDamageAnalysis3D:
    """三维渐进损伤分析"""
    
    def __init__(self, ply_properties, strength_properties, layup, ply_thickness,
                 fracture_properties=None, plate_width=None):
        self.ply_props = ply_properties
        self.strength_props = strength_properties
        self.layup = layup
        self.t_ply = ply_thickness
        self.n_plies = len(layup)
        self.h_total = self.n_plies * ply_thickness
        
        self.hashin = HashinFailureCriteria(strength_properties)
        
        if fracture_properties:
            self.delamination_model = DelaminationGrowthModel(fracture_properties)
        else:
            self.delamination_model = None
        
        self.stress_calculator = ThreeDStressCalculator(
            ply_properties, layup, ply_thickness, plate_width
        )
        
        self.damage_state = []
        self.interface_state = []
        self.current_stiffness = []
        
        self._initialize_damage_state()
        self._initialize_interfaces()
        
        self.load_history = []
        self.damage_history = []
        
    def _initialize_damage_state(self):
        self.damage_state = []
        self.current_stiffness = []
        
        for i, theta in enumerate(self.layup):
            self.damage_state.append({
                'd_f': 0.0,
                'd_m': 0.0,
                'd_s': 0.0,
                'failed': False,
                'failure_modes': [],
                'failure_load_step': None,
                'residual_strength_ratio': 1.0
            })
            
            Q_bar = self._get_damaged_stiffness(theta, i)
            self.current_stiffness.append({
                'theta': theta,
                'Q_bar': Q_bar.copy(),
                'Q_original': Q_bar.copy()
            })
    
    def _initialize_interfaces(self):
        self.interface_state = []
        for i in range(self.n_plies - 1):
            self.interface_state.append({
                'delamination_length': 0.0,
                'GI': 0.0,
                'GII': 0.0,
                'GIII': 0.0,
                'is_delaminating': False,
                'peak_sigma_z': 0.0,
                'peak_tau_xz': 0.0
            })
    
    def _get_damaged_stiffness(self, theta, layer_idx):
        """获取考虑损伤的刚度矩阵"""
        d = self.damage_state[layer_idx]
        
        E1 = self.ply_props['E1'] * (1 - d['d_f'])
        E2 = self.ply_props['E2'] * (1 - d['d_m'])
        G12 = self.ply_props['G12'] * (1 - d['d_s'])
        nu12 = self.ply_props['nu12'] * np.sqrt((1 - d['d_f']) * (1 - d['d_m']))
        
        nu21 = nu12 * E2 / E1 if E1 > 0 else 0
        denom = 1 - nu12 * nu21 if abs(nu12 * nu21) < 1 else 1e-6
        
        Q11 = E1 / denom if denom > 1e-10 else 0
        Q22 = E2 / denom if denom > 1e-10 else 0
        Q12 = nu12 * E2 / denom if denom > 1e-10 else 0
        Q66 = G12
        
        theta_rad = np.radians(theta)
        c = np.cos(theta_rad)
        s = np.sin(theta_rad)
        
        Q_bar = np.zeros((3, 3))
        Q_bar[0, 0] = Q11 * c**4 + 2 * (Q12 + 2 * Q66) * s**2 * c**2 + Q22 * s**4
        Q_bar[0, 1] = (Q11 + Q22 - 4 * Q66) * s**2 * c**2 + Q12 * (s**4 + c**4)
        Q_bar[0, 2] = (Q11 - Q12 - 2 * Q66) * s * c**3 + (Q12 - Q22 + 2 * Q66) * s**3 * c
        Q_bar[1, 0] = Q_bar[0, 1]
        Q_bar[1, 1] = Q11 * s**4 + 2 * (Q12 + 2 * Q66) * s**2 * c**2 + Q22 * c**4
        Q_bar[1, 2] = (Q11 - Q12 - 2 * Q66) * s**3 * c + (Q12 - Q22 + 2 * Q66) * s * c**3
        Q_bar[2, 0] = Q_bar[0, 2]
        Q_bar[2, 1] = Q_bar[1, 2]
        Q_bar[2, 2] = (Q11 + Q22 - 2 * Q12 - 2 * Q66) * s**2 * c**2 + Q66 * (s**4 + c**4)
        
        return Q_bar
    
    def _calculate_ABD_matrix(self):
        A = np.zeros((3, 3))
        B = np.zeros((3, 3))
        D = np.zeros((3, 3))
        
        z = np.zeros(self.n_plies + 1)
        for i in range(self.n_plies + 1):
            z[i] = -self.h_total / 2 + i * self.t_ply
        
        for k in range(self.n_plies):
            Q_bar = self.current_stiffness[k]['Q_bar']
            z_k = z[k]
            z_k1 = z[k + 1]
            A += Q_bar * (z_k1 - z_k)
            B += 0.5 * Q_bar * (z_k1**2 - z_k**2)
            D += (1 / 3) * Q_bar * (z_k1**3 - z_k**3)
        
        ABD = np.block([[A, B], [B, D]])
        return ABD, A, B, D, z
    
    def _transform_stress_3d(self, sigma_lamina, theta, interlaminar_stresses=None):
        """
        将层板坐标系应力转换为材料主方向
        
        参数:
            sigma_lamina: 层板面内应力 [σx, σy, τxy]
            theta: 铺层角度
            interlaminar_stresses: 层间应力 [σz, τxz, τyz]
        """
        theta_rad = np.radians(theta)
        c = np.cos(theta_rad)
        s = np.sin(theta_rad)
        
        sigma_x, sigma_y, tau_xy = sigma_lamina
        
        sigma_1 = sigma_x * c**2 + sigma_y * s**2 + 2 * tau_xy * s * c
        sigma_2 = sigma_x * s**2 + sigma_y * c**2 - 2 * tau_xy * s * c
        tau_12 = -sigma_x * s * c + sigma_y * s * c + tau_xy * (c**2 - s**2)
        
        if interlaminar_stresses is not None:
            sigma_z, tau_xz, tau_yz = interlaminar_stresses
            
            sigma_3 = sigma_z
            tau_13 = tau_xz * c + tau_yz * s
            tau_23 = -tau_xz * s + tau_yz * c
        else:
            sigma_3 = 0
            tau_13 = 0
            tau_23 = 0
        
        return np.array([sigma_1, sigma_2, sigma_3, tau_23, tau_13, tau_12])
    
    def apply_load_incremental(self, Mx_target, My_target=0, n_increments=60):
        """增量加载分析"""
        Mx_inc = Mx_target / n_increments
        My_inc = My_target / n_increments
        
        current_Mx = 0
        current_My = 0
        
        first_failure = None
        ultimate_failure = None
        dominant_failure_mode = None
        
        print(f"\n开始渐进损伤分析，载荷步数: {n_increments}")
        print("=" * 70)
        
        for inc in range(n_increments):
            current_Mx += Mx_inc
            current_My += My_inc
            
            result = self._analyze_load_step(current_Mx, current_My, inc + 1)
            
            self.load_history.append({
                'load_step': inc + 1,
                'load_factor': (inc + 1) / n_increments,
                'Mx': current_Mx,
                'My': current_My,
                'damage_state': [d.copy() for d in self.damage_state],
                'interface_state': [i.copy() for i in self.interface_state],
                'has_lamina_failure': result['lamina_failure'],
                'has_delamination': result['delamination'],
                'dominant_mode': result.get('dominant_mode')
            })
            
            if result['lamina_failure'] and first_failure is None:
                first_failure = {
                    'load_step': inc + 1,
                    'Mx': current_Mx,
                    'My': current_My,
                    'mode': result.get('dominant_mode', 'unknown')
                }
                print(f"载荷步 {inc+1}: 初始失效发生 - {result.get('dominant_mode')}")
            
            if result['delamination'] and first_failure is None:
                first_failure = {
                    'load_step': inc + 1,
                    'Mx': current_Mx,
                    'My': current_My,
                    'mode': 'delamination'
                }
                print(f"载荷步 {inc+1}: 初始分层发生")
            
            if self._check_ultimate_failure(current_Mx, Mx_target):
                ultimate_failure = {
                    'load_step': inc + 1,
                    'Mx': current_Mx,
                    'My': current_My
                }
                dominant_failure_mode = self._identify_dominant_failure_mode()
                print(f"载荷步 {inc+1}: 达到极限载荷，主要失效模式: {dominant_failure_mode}")
                break
            
            if inc % 10 == 9:
                damaged = sum(1 for d in self.damage_state if d['failed'])
                delaminating = sum(1 for i in self.interface_state if i['is_delaminating'])
                print(f"载荷步 {inc+1}: 损伤层数={damaged}/{self.n_plies}, 分层界面={delaminating}/{self.n_plies-1}")
        
        return {
            'first_failure': first_failure,
            'ultimate_failure': ultimate_failure,
            'dominant_failure_mode': dominant_failure_mode,
            'load_history': self.load_history,
            'final_damage': self.damage_state,
            'final_interfaces': self.interface_state
        }
    
    def _analyze_load_step(self, Mx, My, load_step):
        """分析单个载荷步"""
        force_vector = np.zeros(6)
        force_vector[3:] = [Mx, My, 0]
        
        ABD, A, B, D, z = self._calculate_ABD_matrix()
        
        try:
            result = np.linalg.solve(ABD, force_vector)
        except np.linalg.LinAlgError:
            return {'lamina_failure': True, 'delamination': False, 'dominant_mode': 'structural_collapse'}
        
        eps0 = result[:3]
        kappa = result[3:]
        
        stress_field = self.stress_calculator.calculate_pagano_stresses(Mx, My, n_terms=10)
        
        lamina_failure = False
        delamination = False
        max_failure_index = 0
        dominant_mode = None
        
        for k in range(self.n_plies):
            z_mid = (z[k] + z[k + 1]) / 2
            eps_mid = eps0 + z_mid * kappa
            
            Q_bar = self.current_stiffness[k]['Q_bar']
            sigma_lamina = Q_bar @ eps_mid
            
            z_idx = np.argmin(np.abs(stress_field['z'] - z_mid))
            y_idx = len(stress_field['y']) // 2
            
            interlaminar = [
                stress_field['sigma_z'][y_idx, z_idx],
                stress_field['tau_xz'][y_idx, z_idx],
                stress_field['tau_yz'][y_idx, z_idx]
            ]
            
            sigma_material = self._transform_stress_3d(sigma_lamina, self.layup[k], interlaminar)
            
            failure_result = self.hashin.check_failure_3d(sigma_material)
            
            if failure_result['is_failed']:
                lamina_failure = True
                self._update_lamina_damage(k, failure_result, load_step)
            
            if failure_result['max_index'] > max_failure_index:
                max_failure_index = failure_result['max_index']
                if failure_result['failure_modes']:
                    dominant_mode = failure_result['failure_modes'][0]
        
        for interface_idx in range(self.n_plies - 1):
            interface_stress = self.stress_calculator.get_interlaminar_stresses_at_interface(
                stress_field, interface_idx
            )
            
            y_idx = len(interface_stress['y']) // 2
            stresses = [
                interface_stress['sigma_z'][y_idx],
                interface_stress['tau_xz'][y_idx],
                interface_stress['tau_yz'][y_idx]
            ]
            
            self.interface_state[interface_idx]['peak_sigma_z'] = max(
                self.interface_state[interface_idx]['peak_sigma_z'],
                np.max(np.abs(interface_stress['sigma_z']))
            )
            self.interface_state[interface_idx]['peak_tau_xz'] = max(
                self.interface_state[interface_idx]['peak_tau_xz'],
                np.max(np.abs(interface_stress['tau_xz']))
            )
            
            if self.delamination_model:
                current_a = self.interface_state[interface_idx]['delamination_length']
                growth_result = self.delamination_model.predict_delamination_growth(
                    stresses, current_a, self.h_total
                )
                
                self.interface_state[interface_idx].update({
                    'delamination_length': growth_result['new_length'],
                    'GI': growth_result['energy_release']['GI'],
                    'GII': growth_result['energy_release']['GII'],
                    'GIII': growth_result['energy_release']['GIII'],
                    'is_delaminating': growth_result['is_growing']
                })
                
                if growth_result['is_growing']:
                    delamination = True
        
        self._update_stiffness_matrices()
        
        return {
            'lamina_failure': lamina_failure,
            'delamination': delamination,
            'max_failure_index': max_failure_index,
            'dominant_mode': dominant_mode
        }
    
    def _update_lamina_damage(self, layer_idx, failure_result, load_step):
        """更新单层损伤状态"""
        d = self.damage_state[layer_idx]
        indices = failure_result['indices']
        
        fiber_failed = max(indices['fiber_tension'], indices['fiber_compression']) >= 1.0
        matrix_failed = max(indices['matrix_tension'], indices['matrix_compression']) >= 1.0
        delam_failed = max(indices['delamination_tension'], indices['delamination_compression']) >= 1.0
        
        if fiber_failed:
            d['d_f'] = min(1.0, d['d_f'] + 0.08 * max(indices['fiber_tension'], indices['fiber_compression']))
            d['failed'] = True
            for mode in ['fiber_tension', 'fiber_compression']:
                if indices[mode] >= 1.0 and mode not in d['failure_modes']:
                    d['failure_modes'].append(mode)
        
        if matrix_failed:
            d['d_m'] = min(1.0, d['d_m'] + 0.12 * max(indices['matrix_tension'], indices['matrix_compression']))
            d['failed'] = True
            for mode in ['matrix_tension', 'matrix_compression']:
                if indices[mode] >= 1.0 and mode not in d['failure_modes']:
                    d['failure_modes'].append(mode)
        
        if delam_failed:
            d['d_s'] = min(1.0, d['d_s'] + 0.1)
            d['failed'] = True
            if 'delamination' not in d['failure_modes']:
                d['failure_modes'].append('delamination')
        
        if d['failed'] and d['failure_load_step'] is None:
            d['failure_load_step'] = load_step
        
        d['residual_strength_ratio'] = 1.0 - 0.5 * d['d_f'] - 0.3 * d['d_m'] - 0.2 * d['d_s']
    
    def _update_stiffness_matrices(self):
        """更新所有层的刚度矩阵"""
        for k in range(self.n_plies):
            theta = self.layup[k]
            self.current_stiffness[k]['Q_bar'] = self._get_damaged_stiffness(theta, k)
    
    def _check_ultimate_failure(self, current_Mx, max_Mx):
        """检查是否达到极限载荷"""
        failed_layers = sum(1 for d in self.damage_state if d['d_f'] > 0.85)
        if failed_layers >= self.n_plies * 0.5:
            return True
        
        total_damage = sum(d['d_f'] + d['d_m'] for d in self.damage_state)
        if total_damage > self.n_plies * 1.0:
            return True
        
        delaminated_interfaces = sum(1 for i in self.interface_state if i['is_delaminating'])
        if delaminated_interfaces >= self.n_plies * 0.5:
            return True
        
        return False
    
    def _identify_dominant_failure_mode(self):
        """识别主要失效模式"""
        mode_counts = {}
        
        for d in self.damage_state:
            for mode in d['failure_modes']:
                mode_counts[mode] = mode_counts.get(mode, 0) + 1
        
        delamination_count = sum(1 for i in self.interface_state if i['is_delaminating'])
        if delamination_count > 0:
            mode_counts['delamination'] = delamination_count
        
        if not mode_counts:
            return 'no_failure'
        
        return max(mode_counts, key=mode_counts.get)


def plot_damage_evolution(result, layup, title=""):
    """绘制损伤演化曲线"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    history = result['load_history']
    
    load_steps = [h['load_step'] for h in history]
    Mx_values = [h['Mx'] for h in history]
    
    avg_d_f = [np.mean([d['d_f'] for d in h['damage_state']]) for h in history]
    avg_d_m = [np.mean([d['d_m'] for d in h['damage_state']]) for h in history]
    avg_d_s = [np.mean([d['d_s'] for d in h['damage_state']]) for h in history]
    
    failed_layers = [sum(1 for d in h['damage_state'] if d['failed']) for h in history]
    delam_interfaces = [sum(1 for i in h['interface_state'] if i['is_delaminating']) for h in history]
    
    ax = axes[0, 0]
    ax.plot(Mx_values, avg_d_f, 'r-', linewidth=2, label='纤维损伤')
    ax.plot(Mx_values, avg_d_m, 'b-', linewidth=2, label='基体损伤')
    ax.plot(Mx_values, avg_d_s, 'g-', linewidth=2, label='分层损伤')
    ax.set_xlabel('弯矩 Mx (N·m/m)')
    ax.set_ylabel('平均损伤变量')
    ax.set_title('平均损伤变量随载荷变化')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    if result['first_failure']:
        ax.axvline(x=result['first_failure']['Mx'], color='r', linestyle='--', 
                    label=f'初始失效 ({result["first_failure"]["mode"]})', alpha=0.7)
    
    if result['ultimate_failure']:
        ax.axvline(x=result['ultimate_failure']['Mx'], color='k', linestyle='--', 
                    label='极限载荷', alpha=0.7)
    ax.legend()
    
    ax = axes[0, 1]
    ax.plot(Mx_values, failed_layers, 'bo-', linewidth=2, markersize=4, label='失效层数')
    ax.plot(Mx_values, delam_interfaces, 'rs-', linewidth=2, markersize=4, label='分层界面数')
    ax.set_xlabel('弯矩 Mx (N·m/m)')
    ax.set_ylabel('数量')
    ax.set_title('损伤扩展数量')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 2]
    modes = ['fiber_tension', 'fiber_compression', 'matrix_tension', 
             'matrix_compression', 'delamination_tension', 'delamination_compression']
    mode_labels = ['纤维拉伸', '纤维压缩', '基体拉伸', '基体压缩', '分层拉伸', '分层压缩']
    counts = [0] * len(modes)
    
    for h in history:
        for d in h['damage_state']:
            for i, mode in enumerate(modes):
                if mode in d['failure_modes']:
                    counts[i] += 1
    
    colors = ['#d62728', '#ff7f0e', '#1f77b4', '#2ca02c', '#9467bd', '#8c564b']
    ax.bar(mode_labels, counts, color=colors, alpha=0.8)
    ax.set_ylabel('失效发生次数')
    ax.set_title('各失效模式统计')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[1, 0]
    n_layers = len(layup)
    damage_matrix = np.zeros((len(history), n_layers))
    
    for i, h in enumerate(history):
        for j in range(n_layers):
            d = h['damage_state'][j]
            damage_matrix[i, j] = 0.5 * d['d_f'] + 0.35 * d['d_m'] + 0.15 * d['d_s']
    
    im = ax.imshow(damage_matrix.T, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xlabel('载荷步')
    ax.set_ylabel('铺层编号')
    ax.set_yticks(np.arange(n_layers))
    ax.set_yticklabels([f'L{i+1} ({layup[i]}°)' for i in range(n_layers)])
    ax.set_title('各层损伤演化过程')
    plt.colorbar(im, ax=ax, label='综合损伤指数')
    
    ax = axes[1, 1]
    final_damage = result['final_damage']
    y_pos = np.arange(n_layers)
    
    d_f = [d['d_f'] for d in final_damage]
    d_m = [d['d_m'] for d in final_damage]
    d_s = [d['d_s'] for d in final_damage]
    
    bar_width = 0.25
    ax.barh(y_pos - bar_width, d_f, bar_width, label='纤维', color='#d62728', alpha=0.8)
    ax.barh(y_pos, d_m, bar_width, label='基体', color='#1f77b4', alpha=0.8)
    ax.barh(y_pos + bar_width, d_s, bar_width, label='分层', color='#2ca02c', alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f'L{i+1} ({layup[i]}°)' for i in range(n_layers)])
    ax.set_xlabel('损伤变量')
    ax.set_title('各层最终损伤状态')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, 1.1)
    
    ax = axes[1, 2]
    if len(result['final_interfaces']) > 0:
        interface_lengths = [i['delamination_length'] * 1000 for i in result['final_interfaces']]
        interface_labels = [f'界面 {i+1}' for i in range(len(interface_lengths))]
        
        bars = ax.bar(interface_labels, interface_lengths, color='#9467bd', alpha=0.8)
        ax.set_ylabel('分层长度 (mm)')
        ax.set_title('各界面分层长度')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, interface in zip(bars, result['final_interfaces']):
            if interface['is_delaminating']:
                bar.set_edgecolor('red')
                bar.set_linewidth(2)
    else:
        ax.text(0.5, 0.5, '无界面数据', ha='center', va='center')
    
    plt.suptitle(title, fontsize=15, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_failure_envelope(analysis, Mx_range, n_points=20):
    """绘制失效包络线"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    My_failures = []
    
    for Mx in Mx_range:
        analysis_copy = ProgressiveDamageAnalysis3D(
            analysis.ply_props, analysis.strength_props,
            analysis.layup, analysis.t_ply,
            plate_width=analysis.stress_calculator.b
        )
        
        result = analysis_copy.apply_load_incremental(Mx, 0, n_increments=30)
        
        if result['ultimate_failure']:
            My_failures.append(result['ultimate_failure']['Mx'])
        else:
            My_failures.append(Mx)
    
    ax.plot(Mx_range, My_failures, 'b-o', linewidth=2, markersize=6, label='预测极限载荷')
    ax.fill_between(Mx_range, My_failures, alpha=0.3, label='安全区域')
    
    ax.set_xlabel('Mx (N·m/m)')
    ax.set_ylabel('极限载荷 My (N·m/m)')
    ax.set_title('弯曲载荷下的失效包络线')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def main():
    ply_properties = {
        'E1': 138e9,
        'E2': 8.96e9,
        'E3': 8.96e9,
        'G12': 7.1e9,
        'G13': 7.1e9,
        'G23': 3.9e9,
        'nu12': 0.3,
        'nu13': 0.3,
        'nu23': 0.3
    }
    
    strength_properties = {
        'Xt': 1500e6,
        'Xc': 1200e6,
        'Yt': 50e6,
        'Yc': 200e6,
        'Zt': 50e6,
        'Zc': 200e6,
        'S12': 100e6,
        'S13': 100e6,
        'S23': 60e6
    }
    
    fracture_properties = {
        'GIC': 250.0,
        'GIIC': 900.0,
        'GIIIC': 900.0,
        'BK_exponent': 1.5
    }
    
    layup = [0, 45, -45, 90, 90, -45, 45, 0]
    ply_thickness = 0.125e-3
    plate_width = 0.01
    
    print("=" * 70)
    print("复合材料层板三维渐进损伤分析")
    print("=" * 70)
    print(f"铺层: {layup}")
    print(f"层数: {len(layup)}")
    print(f"总厚度: {len(layup) * ply_thickness * 1000:.2f} mm")
    print(f"板宽: {plate_width * 1000:.1f} mm")
    
    print("\n材料强度属性 (MPa):")
    for key, value in strength_properties.items():
        print(f"  {key}: {value/1e6:.1f}")
    
    print("\n断裂韧性 (J/m²):")
    for key, value in fracture_properties.items():
        if key in ['GIC', 'GIIC', 'GIIIC']:
            print(f"  {key}: {value:.0f}")
    
    print("\n" + "=" * 70)
    
    analysis = ProgressiveDamageAnalysis3D(
        ply_properties, strength_properties, layup, ply_thickness,
        fracture_properties, plate_width
    )
    
    Mx_target = 350.0
    n_increments = 60
    
    result = analysis.apply_load_incremental(Mx_target, 0, n_increments)
    
    print("\n" + "=" * 70)
    print("渐进损伤分析结果")
    print("=" * 70)
    
    if result['first_failure']:
        print(f"\n初始失效:")
        print(f"  载荷步: {result['first_failure']['load_step']}")
        print(f"  Mx = {result['first_failure']['Mx']:.2f} N·m/m")
        print(f"  失效模式: {result['first_failure']['mode']}")
    else:
        print("\n未发生初始失效")
    
    if result['ultimate_failure']:
        print(f"\n极限载荷:")
        print(f"  载荷步: {result['ultimate_failure']['load_step']}")
        print(f"  Mx = {result['ultimate_failure']['Mx']:.2f} N·m/m")
        print(f"  主要失效模式: {result['dominant_failure_mode']}")
    else:
        print(f"\n未达到极限载荷，最大Mx = {Mx_target:.2f} N·m/m")
    
    print(f"\n主要失效模式: {result['dominant_failure_mode']}")
    
    print("\n" + "=" * 70)
    print("各层最终损伤状态:")
    print("=" * 70)
    
    for i, damage in enumerate(result['final_damage']):
        print(f"\n第 {i+1} 层 (θ={layup[i]}°):")
        print(f"  失效: {'是' if damage['failed'] else '否'}")
        print(f"  纤维损伤: {damage['d_f']:.3f}")
        print(f"  基体损伤: {damage['d_m']:.3f}")
        print(f"  分层损伤: {damage['d_s']:.3f}")
        if damage['failure_modes']:
            print(f"  失效模式: {', '.join(damage['failure_modes'])}")
        if damage['failure_load_step']:
            print(f"  失效载荷步: {damage['failure_load_step']}")
    
    print("\n" + "=" * 70)
    print("各界面分层状态:")
    print("=" * 70)
    
    for i, interface in enumerate(result['final_interfaces']):
        print(f"\n界面 {i+1} (L{i+1}/L{i+2}):")
        print(f"  分层长度: {interface['delamination_length']*1000:.4f} mm")
        print(f"  是否扩展: {'是' if interface['is_delaminating'] else '否'}")
        print(f"  GI: {interface['GI']:.2f} J/m²")
        print(f"  GII: {interface['GII']:.2f} J/m²")
        print(f"  峰值σz: {interface['peak_sigma_z']/1e6:.2f} MPa")
        print(f"  峰值τxz: {interface['peak_tau_xz']/1e6:.2f} MPa")
    
    fig1 = plot_damage_evolution(result, layup, title="[0/45/-45/90]s 三维渐进损伤分析")
    fig1.savefig('damage_evolution_3d.png', dpi=300, bbox_inches='tight')
    print("\n损伤演化图已保存为 'damage_evolution_3d.png'")
    
    plt.show()


if __name__ == "__main__":
    main()

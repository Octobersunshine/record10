"""
Advanced Reservoir Simulation Extensions
========================================
扩展功能：
1. 双孔双渗模型（裂缝性油藏）
2. 聚合物驱EOR
3. CO2驱EOR
4. 开发方案优化接口

Dependencies:
    - black_oil_impes.py (base classes)
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from typing import Dict, Tuple, List, Optional, Callable
import warnings
warnings.filterwarnings('ignore')

from black_oil_impes import (
    Grid, RockProperties, BlackOilPVT, RelativePermeability,
    CapillaryPressure, Well, MD_TO_M2, CP_TO_PAS, PER_DAY
)


# ============================================================================
# Part 1: Dual Porosity Dual Permeability (DPDP) Model
# ============================================================================

class DualPorosityRock:
    """
    双孔双渗岩石属性（基质 + 裂缝系统）。
    
    Warren-Root模型：
        - 基质系统：低渗、高孔隙体积，主要储集空间
        - 裂缝系统：高渗、低孔隙体积，主要流动通道
        - 窜流项：基质到裂缝的流体交换
    """
    
    def __init__(self, grid: Grid,
                 permx_matrix: np.ndarray, permy_matrix: np.ndarray,
                 permx_fracture: np.ndarray, permy_fracture: np.ndarray,
                 phi_matrix: np.ndarray, phi_fracture: np.ndarray,
                 sigma: float = 0.1,  # 形状因子 (1/m^2)
                 omega: float = 0.1):  # 储能比
        self.grid = grid
        self.permx_matrix = permx_matrix
        self.permy_matrix = permy_matrix
        self.permx_fracture = permx_fracture
        self.permy_fracture = permy_fracture
        self.phi_matrix = phi_matrix
        self.phi_fracture = phi_fracture
        self.sigma = sigma
        self.omega = omega
        
        # 派生属性
        self.permx_total = permx_matrix + permx_fracture
        self.permy_total = permy_matrix + permy_fracture
        self.phi_total = phi_matrix + phi_fracture
        self.porosity = phi_fracture  # 默认孔隙度（裂缝系统）
        self.permx = permx_fracture  # 兼容接口：默认裂缝渗透率
        self.permy = permy_fracture  # 兼容接口：默认裂缝渗透率
    
    def matrix_fracture_transfer(self, p_matrix: np.ndarray, p_fracture: np.ndarray,
                                  mobility: np.ndarray) -> np.ndarray:
        """
        计算基质到裂缝的窜流量 (Warren-Root模型)。
        
        q_transfer = sigma * k_matrix * mobility / mu * (p_matrix - p_fracture)
        """
        return self.sigma * self.permx_matrix * mobility * (p_matrix - p_fracture)


class DualPorositySolver:
    """
    双孔双渗求解器（裂缝性油藏）。
    
    控制方程（Warren-Root模型）：
        裂缝系统：
            V_f * phi_f * dS_f/dt + div(v_f) + q_matrix->fracture = q_well
        基质系统：
            V_m * phi_m * dS_m/dt - q_matrix->fracture = 0
    """
    
    def __init__(self, grid: Grid, rock: DualPorosityRock,
                 pvt: BlackOilPVT, relperm: RelativePermeability,
                 cap_pres: CapillaryPressure,
                 wells: List[Well] = None,
                 p_init: float = 200.0e5,
                 sw_init: float = 0.20,
                 sg_init: float = 0.0,
                 dt: float = 1.0,
                 t_max: float = 365.0):
        self.grid = grid
        self.rock = rock
        self.pvt = pvt
        self.relperm = relperm
        self.cap_pres = cap_pres
        self.wells = wells if wells else []
        self.p_init = p_init
        self.sw_init = sw_init
        self.sg_init = sg_init
        self.dt = dt
        self.t_max = t_max
        
        n = grid.ncells
        
        # 基质系统状态
        self.p_matrix = np.full(n, p_init)
        self.sw_matrix = np.full(n, sw_init)
        self.sg_matrix = np.full(n, sg_init)
        self.so_matrix = 1.0 - self.sw_matrix - self.sg_matrix
        
        # 裂缝系统状态
        self.p_fracture = np.full(n, p_init)
        self.sw_fracture = np.full(n, sw_init)
        self.sg_fracture = np.full(n, sg_init)
        self.so_fracture = 1.0 - self.sw_fracture - self.sg_fracture
        
        # 历史存储
        self.p_matrix_old = self.p_matrix.copy()
        self.sw_matrix_old = self.sw_matrix.copy()
        self.sg_matrix_old = self.sg_matrix.copy()
        self.p_fracture_old = self.p_fracture.copy()
        self.sw_fracture_old = self.sw_fracture.copy()
        self.sg_fracture_old = self.sg_fracture.copy()
        
        self.volumes = grid.volume()
        self.nsteps = int(np.ceil(t_max / dt))
        self.time = 0.0
        self.time_steps = []
        self.production_data = {w.name: [] for w in self.wells}
        self.mass_balance_errors = []
        self.transfer_rates = []
        
        self._setup_transmissibility()
        for well in self.wells:
            well.compute_well_index(grid, rock)
    
    def _setup_transmissibility(self):
        """计算裂缝和基质系统的传导率。"""
        g = self.grid
        self.trans_x_frac = np.zeros((g.nx - 1, g.ny))
        self.trans_y_frac = np.zeros((g.nx, g.ny - 1))
        self.trans_x_mat = np.zeros((g.nx - 1, g.ny))
        self.trans_y_mat = np.zeros((g.nx, g.ny - 1))
        
        for i in range(g.nx - 1):
            for j in range(g.ny):
                idx1 = g.idx(i, j)
                idx2 = g.idx(i + 1, j)
                k1 = self.rock.permx_fracture[idx1] * MD_TO_M2
                k2 = self.rock.permx_fracture[idx2] * MD_TO_M2
                k_harm = 2 * k1 * k2 / max(k1 + k2, 1e-20)
                self.trans_x_frac[i, j] = k_harm * g.dy * g.dz / g.dx
                
                k1m = self.rock.permx_matrix[idx1] * MD_TO_M2
                k2m = self.rock.permx_matrix[idx2] * MD_TO_M2
                k_harm_m = 2 * k1m * k2m / max(k1m + k2m, 1e-20)
                self.trans_x_mat[i, j] = k_harm_m * g.dy * g.dz / g.dx
        
        for i in range(g.nx):
            for j in range(g.ny - 1):
                idx1 = g.idx(i, j)
                idx2 = g.idx(i, j + 1)
                k1 = self.rock.permy_fracture[idx1] * MD_TO_M2
                k2 = self.rock.permy_fracture[idx2] * MD_TO_M2
                k_harm = 2 * k1 * k2 / max(k1 + k2, 1e-20)
                self.trans_y_frac[i, j] = k_harm * g.dx * g.dz / g.dy
                
                k1m = self.rock.permy_matrix[idx1] * MD_TO_M2
                k2m = self.rock.permy_matrix[idx2] * MD_TO_M2
                k_harm_m = 2 * k1m * k2m / max(k1m + k2m, 1e-20)
                self.trans_y_mat[i, j] = k_harm_m * g.dx * g.dz / g.dy
    
    def _compute_phase_flux(self, p: np.ndarray, sw: np.ndarray, sg: np.ndarray,
                            phase: str, system: str = 'fracture') -> Tuple[np.ndarray, np.ndarray]:
        """计算相通量（基质或裂缝系统）。"""
        g = self.grid
        so = 1.0 - sw - sg
        
        if system == 'fracture':
            perm_mult = 1.0  # 裂缝渗透率倍数
        else:
            perm_mult = 0.1  # 基质渗透率较低
        
        if phase == 'oil':
            kr = self.relperm.kro(sw, sg)
            mu = self.pvt.mu_o(p)
        elif phase == 'water':
            kr = self.relperm.krw(sw)
            mu = self.pvt.mu_w
        else:
            kr = self.relperm.krg(sg)
            mu = self.pvt.mu_g
        
        mobility = kr * perm_mult / np.maximum(mu * CP_TO_PAS, 1e-10)
        
        if system == 'fracture':
            trans_x = self.trans_x_frac
            trans_y = self.trans_y_frac
        else:
            trans_x = self.trans_x_mat
            trans_y = self.trans_y_mat
        
        fx = np.zeros((g.nx - 1, g.ny))
        fy = np.zeros((g.nx, g.ny - 1))
        
        for i in range(g.nx - 1):
            for j in range(g.ny):
                idx1 = g.idx(i, j)
                idx2 = g.idx(i + 1, j)
                lambda_avg = 0.5 * (mobility[idx1] + mobility[idx2])
                fx[i, j] = -trans_x[i, j] * lambda_avg * (p[idx2] - p[idx1])
        
        for i in range(g.nx):
            for j in range(g.ny - 1):
                idx1 = g.idx(i, j)
                idx2 = g.idx(i, j + 1)
                lambda_avg = 0.5 * (mobility[idx1] + mobility[idx2])
                fy[i, j] = -trans_y[i, j] * lambda_avg * (p[idx2] - p[idx1])
        
        return fx, fy
    
    def _compute_matrix_fracture_transfer(self, phase: str) -> np.ndarray:
        """计算基质-裂缝窜流量。"""
        if phase == 'water':
            s_matrix = self.sw_matrix
            s_fracture = self.sw_fracture
            p_matrix = self.p_matrix - self.cap_pres.pcow(self.sw_matrix, self.relperm)
            p_fracture = self.p_fracture - self.cap_pres.pcow(self.sw_fracture, self.relperm)
            kr = self.relperm.krw(0.5 * (self.sw_matrix + self.sw_fracture))
            mu = self.pvt.mu_w
        elif phase == 'gas':
            s_matrix = self.sg_matrix
            s_fracture = self.sg_fracture
            p_matrix = self.p_matrix + self.cap_pres.pcog(self.sg_matrix, self.relperm)
            p_fracture = self.p_fracture + self.cap_pres.pcog(self.sg_fracture, self.relperm)
            kr = self.relperm.krg(0.5 * (self.sg_matrix + self.sg_fracture))
            mu = self.pvt.mu_g
        else:
            s_matrix = self.so_matrix
            s_fracture = self.so_fracture
            p_matrix = self.p_matrix
            p_fracture = self.p_fracture
            sw_avg = 0.5 * (self.sw_matrix + self.sw_fracture)
            sg_avg = 0.5 * (self.sg_matrix + self.sg_fracture)
            kr = self.relperm.kro(sw_avg, sg_avg)
            mu = self.pvt.mu_o(self.p_matrix)
        
        mobility = kr / np.maximum(mu * CP_TO_PAS, 1e-10)
        dp = p_matrix - p_fracture
        
        return self.rock.sigma * self.rock.permx_matrix * MD_TO_M2 * mobility * dp * PER_DAY
    
    def _well_phase_rates(self, p: np.ndarray, sw: np.ndarray, sg: np.ndarray) -> Dict[str, np.ndarray]:
        """计算井的相产量（简化版，只考虑裂缝系统）。"""
        n = self.grid.ncells
        rates = {'oil': np.zeros(n), 'water': np.zeros(n), 'gas': np.zeros(n)}
        
        so = 1.0 - sw - sg
        kro = self.relperm.kro(sw, sg)
        krw = self.relperm.krw(sw)
        krg = self.relperm.krg(sg)
        
        mu_o = self.pvt.mu_o(p)
        mu_w = self.pvt.mu_w
        mu_g = self.pvt.mu_g
        
        for well in self.wells:
            idx = self.grid.idx(well.i, well.j)
            wi = well.well_index
            
            if well.well_type == 'producer':
                bhp = well.target_bhp
                dp = max(p[idx] - bhp, 0)
                
                qo = wi * kro[idx] / np.maximum(mu_o[idx] * CP_TO_PAS, 1e-10) * dp * PER_DAY
                qw = wi * krw[idx] / np.maximum(mu_w * CP_TO_PAS, 1e-10) * dp * PER_DAY
                qg = wi * krg[idx] / np.maximum(mu_g * CP_TO_PAS, 1e-10) * dp * PER_DAY
                
                rates['oil'][idx] = -qo
                rates['water'][idx] = -qw
                rates['gas'][idx] = -qg
            else:
                q_total = well.target_rate / PER_DAY
                if 'water' in well.phases:
                    rates['water'][idx] = q_total
                elif 'gas' in well.phases:
                    rates['gas'][idx] = q_total
        
        return rates
    
    def step(self) -> bool:
        """执行一个时间步（双孔双渗IMPES）。"""
        g = self.grid
        n = g.ncells
        dt = self.dt
        
        # 保存旧状态
        pm_old = self.p_matrix.copy()
        swm_old = self.sw_matrix.copy()
        sgm_old = self.sg_matrix.copy()
        pf_old = self.p_fracture.copy()
        swf_old = self.sw_fracture.copy()
        sgf_old = self.sg_fracture.copy()
        
        # 计算窜流量
        qm_w = self._compute_matrix_fracture_transfer('water')
        qm_o = self._compute_matrix_fracture_transfer('oil')
        qm_g = self._compute_matrix_fracture_transfer('gas')
        
        # ===================== 裂缝系统压力求解 =====================
        mat = lil_matrix((n, n))
        rhs = np.zeros(n)
        
        well_rates = self._well_phase_rates(pf_old, swf_old, sgf_old)
        
        for idx in range(n):
            i, j = g.coord(idx)
            
            acc_coeff = (self.rock.phi_fracture[idx] * self.volumes[idx] / dt)
            mat[idx, idx] += acc_coeff
            
            if i > 0:
                mat[idx, idx] += self.trans_x_frac[i - 1, j]
                mat[idx, g.idx(i - 1, j)] -= self.trans_x_frac[i - 1, j]
            if i < g.nx - 1:
                mat[idx, idx] += self.trans_x_frac[i, j]
                mat[idx, g.idx(i + 1, j)] -= self.trans_x_frac[i, j]
            if j > 0:
                mat[idx, idx] += self.trans_y_frac[i, j - 1]
                mat[idx, g.idx(i, j - 1)] -= self.trans_y_frac[i, j - 1]
            if j < g.ny - 1:
                mat[idx, idx] += self.trans_y_frac[i, j]
                mat[idx, g.idx(i, j + 1)] -= self.trans_y_frac[i, j]
            
            rhs[idx] = acc_coeff * pf_old[idx] - qm_w[idx] + well_rates['water'][idx]
        
        # 参考压力点
        ref_idx = g.idx(g.nx // 2, g.ny // 2)
        mat[ref_idx, ref_idx] += 1e12
        rhs[ref_idx] += 1e12 * pf_old[ref_idx]
        
        mat = mat.tocsr()
        pf_new = spsolve(mat, rhs)
        pf_new = np.clip(pf_new, 1e4, 500e5)
        
        # ===================== 基质系统压力求解 =====================
        mat_m = lil_matrix((n, n))
        rhs_m = np.zeros(n)
        
        for idx in range(n):
            i, j = g.coord(idx)
            acc_coeff = (self.rock.phi_matrix[idx] * self.volumes[idx] / dt)
            mat_m[idx, idx] += acc_coeff
            
            if i > 0:
                mat_m[idx, idx] += self.trans_x_mat[i - 1, j]
                mat_m[idx, g.idx(i - 1, j)] -= self.trans_x_mat[i - 1, j]
            if i < g.nx - 1:
                mat_m[idx, idx] += self.trans_x_mat[i, j]
                mat_m[idx, g.idx(i + 1, j)] -= self.trans_x_mat[i, j]
            if j > 0:
                mat_m[idx, idx] += self.trans_y_mat[i, j - 1]
                mat_m[idx, g.idx(i, j - 1)] -= self.trans_y_mat[i, j - 1]
            if j < g.ny - 1:
                mat_m[idx, idx] += self.trans_y_mat[i, j]
                mat_m[idx, g.idx(i, j + 1)] -= self.trans_y_mat[i, j]
            
            rhs_m[idx] = acc_coeff * pm_old[idx] + qm_w[idx]
        
        mat_m[ref_idx, ref_idx] += 1e12
        rhs_m[ref_idx] += 1e12 * pm_old[ref_idx]
        
        mat_m = mat_m.tocsr()
        pm_new = spsolve(mat_m, rhs_m)
        pm_new = np.clip(pm_new, 1e4, 500e5)
        
        # ===================== 饱和度更新 =====================
        # 裂缝系统
        fx_wf, fy_wf = self._compute_phase_flux(pf_new, self.sw_fracture, self.sg_fracture, 'water', 'fracture')
        swf_new = swf_old.copy()
        for idx in range(n):
            i, j = g.coord(idx)
            div = 0.0
            if i > 0:
                div += fx_wf[i - 1, j]
            if i < g.nx - 1:
                div -= fx_wf[i, j]
            if j > 0:
                div += fy_wf[i, j - 1]
            if j < g.ny - 1:
                div -= fy_wf[i, j]
            
            acc_coeff = self.rock.phi_fracture[idx] * self.volumes[idx] / dt
            swf_new[idx] = swf_old[idx] - (div + qm_w[idx] - well_rates['water'][idx]) / max(acc_coeff, 1e-10)
        
        swf_new = np.clip(swf_new, self.relperm.swc,
                         1 - self.relperm.sor - self.relperm.sgc)
        
        # 基质系统
        fx_wm, fy_wm = self._compute_phase_flux(pm_new, self.sw_matrix, self.sg_matrix, 'water', 'matrix')
        swm_new = swm_old.copy()
        for idx in range(n):
            i, j = g.coord(idx)
            div = 0.0
            if i > 0:
                div += fx_wm[i - 1, j]
            if i < g.nx - 1:
                div -= fx_wm[i, j]
            if j > 0:
                div += fy_wm[i, j - 1]
            if j < g.ny - 1:
                div -= fy_wm[i, j]
            
            acc_coeff = self.rock.phi_matrix[idx] * self.volumes[idx] / dt
            swm_new[idx] = swm_old[idx] - (div - qm_w[idx]) / max(acc_coeff, 1e-10)
        
        swm_new = np.clip(swm_new, self.relperm.swc,
                         1 - self.relperm.sor - self.relperm.sgc)
        
        # 更新状态
        self.p_fracture = pf_new
        self.sw_fracture = swf_new
        self.so_fracture = 1.0 - swf_new - self.sg_fracture
        self.p_matrix = pm_new
        self.sw_matrix = swm_new
        self.so_matrix = 1.0 - swm_new - self.sg_matrix
        
        self.transfer_rates.append(np.mean(np.abs(qm_w)))
        self.time += dt
        self.time_steps.append(self.time)
        
        # 记录产量
        self._record_production(pf_new, swf_new, self.sg_fracture)
        
        return True
    
    def _record_production(self, p: np.ndarray, sw: np.ndarray, sg: np.ndarray):
        """记录井产量数据。"""
        well_rates = self._well_phase_rates(p, sw, sg)
        
        for well in self.wells:
            idx = self.grid.idx(well.i, well.j)
            q_o = -well_rates['oil'][idx]
            q_w = -well_rates['water'][idx]
            q_g = -well_rates['gas'][idx]
            
            self.production_data[well.name].append({
                'time': self.time,
                'oil_rate': q_o,
                'water_rate': q_w,
                'gas_rate': q_g,
                'water_cut': q_w / max(q_o + q_w + q_g, 1e-10),
                'gor': q_g / max(q_o, 1e-10),
            })
    
    def run(self) -> Dict:
        """运行模拟。"""
        print("=" * 60)
        print("  Dual Porosity Dual Permeability Simulator")
        print("  (Fractured Reservoir Model)")
        print("=" * 60)
        print(f"  Grid: {self.grid.nx} x {self.grid.ny} = {self.grid.ncells} cells")
        print(f"  Time step: {self.dt} days")
        print(f"  Total time: {self.t_max} days ({self.nsteps} steps)")
        print(f"  Omega (storage ratio): {self.rock.omega:.4f}")
        print(f"  Sigma (shape factor): {self.rock.sigma:.4f} 1/m2")
        print("=" * 60)
        
        for step in range(self.nsteps):
            success = self.step()
            if not success:
                print(f"  Step {step+1} failed")
                break
            
            if (step + 1) % max(1, self.nsteps // 10) == 0:
                avg_pf = np.mean(self.p_fracture) / 1e5
                avg_pm = np.mean(self.p_matrix) / 1e5
                avg_swf = np.mean(self.sw_fracture)
                avg_swm = np.mean(self.sw_matrix)
                avg_transfer = self.transfer_rates[-1] if len(self.transfer_rates) > 0 else 0
                print(f"  Step {step+1:5d}/{self.nsteps} | "
                      f"Time: {self.time:7.1f} days | "
                      f"P_frac: {avg_pf:6.1f} bar | "
                      f"P_mat: {avg_pm:6.1f} bar | "
                      f"Sw_frac: {avg_swf:.3f} | "
                      f"Sw_mat: {avg_swm:.3f} | "
                      f"Transfer: {avg_transfer:.2e} m3/day")
        
        print("=" * 60)
        print("  Simulation Complete")
        print("=" * 60)
        
        return self.get_results()
    
    def get_results(self) -> Dict:
        return {
            'p_fracture': self.p_fracture.copy(),
            'sw_fracture': self.sw_fracture.copy(),
            'sg_fracture': self.sg_fracture.copy(),
            'so_fracture': self.so_fracture.copy(),
            'p_matrix': self.p_matrix.copy(),
            'sw_matrix': self.sw_matrix.copy(),
            'sg_matrix': self.sg_matrix.copy(),
            'so_matrix': self.so_matrix.copy(),
            'time_steps': np.array(self.time_steps),
            'transfer_rates': np.array(self.transfer_rates),
            'production': self.production_data,
            'grid': self.grid,
        }


# ============================================================================
# Part 2: Polymer Flooding EOR Model
# ============================================================================

class PolymerProperties:
    """
    聚合物驱物理属性。
    
    主要效应：
    1. 增粘效应：提高水相粘度，改善流度比
    2. 吸附效应：聚合物在岩石表面吸附，降低有效浓度
    3. 渗透率降低：滞留聚合物导致绝对渗透率下降
    4. 不可入孔隙体积：聚合物分子无法进入小孔隙
    """
    
    def __init__(self,
                 name: str = 'HPAM',
                 C_max: float = 2000.0,  # 最大注入浓度 (ppm)
                 k_r_max: float = 0.5,  # 最大渗透率降低系数
                 C_ads_max: float = 100.0,  # 最大吸附量 (ug/g)
                 k_ads: float = 0.01,  # 吸附平衡常数 (L/mg)
                 n_ads: float = 1.0,  # 吸附指数
                 mu_pure: float = 1.0,  # 清水粘度 (cP)
                 mu_factor: float = 10.0,  # 最大增粘倍数
                 n_mu: float = 0.5,  # 粘度指数
                 IPV: float = 0.2,  # 不可入孔隙体积
                 perm_reduction: float = 0.3):  # 渗透率降低系数
        self.name = name
        self.C_max = C_max
        self.k_r_max = k_r_max
        self.C_ads_max = C_ads_max
        self.k_ads = k_ads
        self.n_ads = n_ads
        self.mu_pure = mu_pure
        self.mu_factor = mu_factor
        self.n_mu = n_mu
        self.IPV = IPV
        self.perm_reduction = perm_reduction
    
    def viscosity(self, C: np.ndarray, shear_rate: float = 10.0) -> np.ndarray:
        """
        计算聚合物溶液粘度 (cP)。
        幂律模型：mu = mu_water * (1 + k_mu * C^n)
        """
        C_norm = C / max(self.C_max, 1e-10)
        return self.mu_pure * (1.0 + self.mu_factor * np.power(C_norm, self.n_mu))
    
    def adsorption_isotherm(self, C: np.ndarray) -> np.ndarray:
        """
        Langmuir吸附等温线：C_ads = C_ads_max * (k_ads * C)^n / (1 + (k_ads * C)^n)
        """
        kC = self.k_ads * C
        return self.C_ads_max * np.power(kC, self.n_ads) / (1.0 + np.power(kC, self.n_ads))
    
    def permeability_reduction(self, C_ads: np.ndarray) -> np.ndarray:
        """
        计算渗透率降低系数（由于聚合物滞留）。
        k/k0 = 1 - k_r_max * (1 - exp(-alpha * C_ads))
        """
        alpha = 0.01
        return 1.0 - self.k_r_max * (1.0 - np.exp(-alpha * C_ads))
    
    def effective_porosity(self, phi: np.ndarray) -> np.ndarray:
        """计算有效孔隙度（扣除不可入孔隙体积）。"""
        return phi * (1.0 - self.IPV)


class PolymerFloodSolver:
    """
    聚合物驱求解器。
    
    附加输运方程：
        d(phi * C)/dt + d(phi * C_ads)/dt + div(v_w * C) = 0
    
    其中：
        C: 水溶液中聚合物浓度 (ppm)
        C_ads: 吸附聚合物浓度 (ug/g)
    """
    
    def __init__(self, grid: Grid, rock: RockProperties,
                 pvt: BlackOilPVT, relperm: RelativePermeability,
                 cap_pres: CapillaryPressure,
                 polymer: PolymerProperties,
                 wells: List[Well] = None,
                 p_init: float = 200.0e5,
                 sw_init: float = 0.20,
                 sg_init: float = 0.0,
                 C_init: float = 0.0,
                 C_ads_init: float = 0.0,
                 dt: float = 1.0,
                 t_max: float = 365.0):
        self.grid = grid
        self.rock = rock
        self.pvt = pvt
        self.relperm = relperm
        self.cap_pres = cap_pres
        self.polymer = polymer
        self.wells = wells if wells else []
        self.p_init = p_init
        self.sw_init = sw_init
        self.sg_init = sg_init
        self.dt = dt
        self.t_max = t_max
        
        n = grid.ncells
        self.p = np.full(n, p_init)
        self.sw = np.full(n, sw_init)
        self.sg = np.full(n, sg_init)
        self.so = 1.0 - self.sw - self.sg
        
        # 聚合物浓度
        self.C = np.full(n, C_init)  # 水中浓度 (ppm)
        self.C_ads = np.full(n, C_ads_init)  # 吸附浓度 (ug/g)
        
        self.p_old = self.p.copy()
        self.sw_old = self.sw.copy()
        self.sg_old = self.sg.copy()
        self.C_old = self.C.copy()
        self.C_ads_old = self.C_ads.copy()
        
        self.porosity = rock.porosity
        self.volumes = grid.volume()
        self.effective_porosity = polymer.effective_porosity(rock.porosity)
        
        self.nsteps = int(np.ceil(t_max / dt))
        self.time = 0.0
        self.time_steps = []
        self.production_data = {w.name: [] for w in self.wells}
        self.mass_balance_errors = []
        
        self._setup_transmissibility()
        for well in self.wells:
            well.compute_well_index(grid, rock)
    
    def _setup_transmissibility(self):
        g = self.grid
        self.trans_x = np.zeros((g.nx - 1, g.ny))
        self.trans_y = np.zeros((g.nx, g.ny - 1))
        
        for i in range(g.nx - 1):
            for j in range(g.ny):
                idx1 = g.idx(i, j)
                idx2 = g.idx(i + 1, j)
                k1 = self.rock.permx[idx1] * MD_TO_M2
                k2 = self.rock.permx[idx2] * MD_TO_M2
                k_harm = 2 * k1 * k2 / max(k1 + k2, 1e-20)
                self.trans_x[i, j] = k_harm * g.dy * g.dz / g.dx
        
        for i in range(g.nx):
            for j in range(g.ny - 1):
                idx1 = g.idx(i, j)
                idx2 = g.idx(i, j + 1)
                k1 = self.rock.permy[idx1] * MD_TO_M2
                k2 = self.rock.permy[idx2] * MD_TO_M2
                k_harm = 2 * k1 * k2 / max(k1 + k2, 1e-20)
                self.trans_y[i, j] = k_harm * g.dx * g.dz / g.dy
    
    def _compute_water_viscosity(self) -> np.ndarray:
        """计算含聚合物水相粘度。"""
        return self.polymer.viscosity(self.C)
    
    def _well_phase_rates(self, p: np.ndarray, sw: np.ndarray, sg: np.ndarray) -> Dict[str, np.ndarray]:
        n = self.grid.ncells
        rates = {'oil': np.zeros(n), 'water': np.zeros(n), 'gas': np.zeros(n)}
        
        so = 1.0 - sw - sg
        kro = self.relperm.kro(sw, sg)
        krw = self.relperm.krw(sw)
        krg = self.relperm.krg(sg)
        
        mu_o = self.pvt.mu_o(p)
        mu_w = self._compute_water_viscosity()
        mu_g = self.pvt.mu_g
        
        for well in self.wells:
            idx = self.grid.idx(well.i, well.j)
            wi = well.well_index
            
            if well.well_type == 'producer':
                bhp = well.target_bhp
                dp = max(p[idx] - bhp, 0)
                
                qo = wi * kro[idx] / np.maximum(mu_o[idx] * CP_TO_PAS, 1e-10) * dp * PER_DAY
                qw = wi * krw[idx] / np.maximum(mu_w[idx] * CP_TO_PAS, 1e-10) * dp * PER_DAY
                qg = wi * krg[idx] / np.maximum(mu_g * CP_TO_PAS, 1e-10) * dp * PER_DAY
                
                rates['oil'][idx] = -qo
                rates['water'][idx] = -qw
                rates['gas'][idx] = -qg
            else:
                q_total = well.target_rate / PER_DAY
                rates['water'][idx] = q_total
        
        return rates
    
    def _compute_phase_flux(self, p: np.ndarray, sw: np.ndarray, sg: np.ndarray,
                            phase: str) -> Tuple[np.ndarray, np.ndarray]:
        g = self.grid
        so = 1.0 - sw - sg
        
        if phase == 'oil':
            kr = self.relperm.kro(sw, sg)
            mu = self.pvt.mu_o(p)
            perm_factor = np.ones_like(p)
        elif phase == 'water':
            kr = self.relperm.krw(sw)
            mu = self._compute_water_viscosity()
            perm_factor = self.polymer.permeability_reduction(self.C_ads)
        else:
            kr = self.relperm.krg(sg)
            mu = self.pvt.mu_g
            perm_factor = np.ones_like(p)
        
        mobility = kr * perm_factor / np.maximum(mu * CP_TO_PAS, 1e-10)
        
        fx = np.zeros((g.nx - 1, g.ny))
        fy = np.zeros((g.nx, g.ny - 1))
        
        for i in range(g.nx - 1):
            for j in range(g.ny):
                idx1 = g.idx(i, j)
                idx2 = g.idx(i + 1, j)
                lambda_avg = 0.5 * (mobility[idx1] + mobility[idx2])
                fx[i, j] = -self.trans_x[i, j] * lambda_avg * (p[idx2] - p[idx1])
        
        for i in range(g.nx):
            for j in range(g.ny - 1):
                idx1 = g.idx(i, j)
                idx2 = g.idx(i, j + 1)
                lambda_avg = 0.5 * (mobility[idx1] + mobility[idx2])
                fy[i, j] = -self.trans_y[i, j] * lambda_avg * (p[idx2] - p[idx1])
        
        return fx, fy
    
    def _assemble_pressure_matrix(self) -> Tuple[csr_matrix, np.ndarray]:
        g = self.grid
        n = g.ncells
        mat = lil_matrix((n, n))
        rhs = np.zeros(n)
        
        well_rates = self._well_phase_rates(self.p, self.sw, self.sg)
        
        for idx in range(n):
            i, j = g.coord(idx)
            acc_coeff = self.porosity[idx] * self.volumes[idx] / self.dt
            mat[idx, idx] += acc_coeff
            
            if i > 0:
                mat[idx, idx] += self.trans_x[i - 1, j]
                mat[idx, g.idx(i - 1, j)] -= self.trans_x[i - 1, j]
            if i < g.nx - 1:
                mat[idx, idx] += self.trans_x[i, j]
                mat[idx, g.idx(i + 1, j)] -= self.trans_x[i, j]
            if j > 0:
                mat[idx, idx] += self.trans_y[i, j - 1]
                mat[idx, g.idx(i, j - 1)] -= self.trans_y[i, j - 1]
            if j < g.ny - 1:
                mat[idx, idx] += self.trans_y[i, j]
                mat[idx, g.idx(i, j + 1)] -= self.trans_y[i, j]
            
            rhs[idx] = acc_coeff * self.p_old[idx] + well_rates['oil'][idx] + well_rates['water'][idx]
        
        return mat.tocsr(), rhs
    
    def _update_concentration(self, p_new: np.ndarray, sw_new: np.ndarray):
        """更新聚合物浓度（显式上风法）。"""
        g = self.grid
        n = g.ncells
        dt = self.dt
        
        fx_w, fy_w = self._compute_phase_flux(p_new, self.sw, self.sg, 'water')
        
        C_new = self.C.copy()
        C_ads_new = self.C_ads.copy()
        
        for idx in range(n):
            i, j = g.coord(idx)
            
            # 通量散度（上风）
            div = 0.0
            if i > 0:
                flux = fx_w[i - 1, j]
                C_up = self.C[g.idx(i - 1, j)] if flux > 0 else self.C[idx]
                div += flux * C_up
            if i < g.nx - 1:
                flux = fx_w[i, j]
                C_up = self.C[idx] if flux > 0 else self.C[g.idx(i + 1, j)]
                div -= flux * C_up
            if j > 0:
                flux = fy_w[i, j - 1]
                C_up = self.C[g.idx(i, j - 1)] if flux > 0 else self.C[idx]
                div += flux * C_up
            if j < g.ny - 1:
                flux = fy_w[i, j]
                C_up = self.C[idx] if flux > 0 else self.C[g.idx(i, j + 1)]
                div -= flux * C_up
            
            # 吸附平衡
            C_eq = self.polymer.adsorption_isotherm(self.C[idx])
            dC_ads_dt = (C_eq - self.C_ads[idx]) / 1.0
            
            # 质量平衡
            phi_eff = self.effective_porosity[idx]
            vol = self.volumes[idx]
            
            acc_old = phi_eff * self.sw_old[idx] * self.C_old[idx] + phi_eff * self.C_ads_old[idx]
            acc_new = phi_eff * sw_new[idx] * C_new[idx] + phi_eff * C_ads_new[idx]
            
            dC = -(div * dt + (acc_new - acc_old)) / max(phi_eff * sw_new[idx] * vol, 1e-10)
            C_new[idx] += dC
            C_ads_new[idx] = self.polymer.adsorption_isotherm(C_new[idx])
        
        C_new = np.clip(C_new, 0.0, self.polymer.C_max * 2)
        C_ads_new = np.clip(C_ads_new, 0.0, self.polymer.C_ads_max)
        
        return C_new, C_ads_new
    
    def step(self) -> bool:
        dt = self.dt
        
        p_old = self.p.copy()
        sw_old = self.sw.copy()
        C_old = self.C.copy()
        
        # 压力方程（隐式）
        mat, rhs = self._assemble_pressure_matrix()
        ref_idx = self.grid.idx(self.grid.nx // 2, self.grid.ny // 2)
        mat[ref_idx, ref_idx] += 1e12
        rhs[ref_idx] += 1e12 * p_old[ref_idx]
        
        p_new = spsolve(mat, rhs)
        p_new = np.clip(p_new, 1e4, 500e5)
        
        # 饱和度更新（显式）
        fx_w, fy_w = self._compute_phase_flux(p_new, self.sw, self.sg, 'water')
        well_rates = self._well_phase_rates(p_new, self.sw, self.sg)
        sw_new = sw_old.copy()
        
        for idx in range(self.grid.ncells):
            i, j = self.grid.coord(idx)
            div = 0.0
            if i > 0:
                div += fx_w[i - 1, j]
            if i < self.grid.nx - 1:
                div -= fx_w[i, j]
            if j > 0:
                div += fy_w[i, j - 1]
            if j < self.grid.ny - 1:
                div -= fy_w[i, j]
            
            acc_coeff = self.porosity[idx] * self.volumes[idx] / dt
            sw_new[idx] = sw_old[idx] - (div - well_rates['water'][idx]) / max(acc_coeff, 1e-10)
        
        sw_new = np.clip(sw_new, self.relperm.swc,
                         1 - self.relperm.sor - self.relperm.sgc)
        
        # 聚合物浓度更新
        C_new, C_ads_new = self._update_concentration(p_new, sw_new)
        
        # 更新状态
        self.p = p_new
        self.sw = sw_new
        self.so = 1.0 - sw_new - self.sg
        self.C = C_new
        self.C_ads = C_ads_new
        
        self.p_old = p_old
        self.sw_old = sw_old
        self.C_old = C_old
        
        self.time += dt
        self.time_steps.append(self.time)
        
        # 记录产量
        self._record_production(p_new, sw_new, self.sg)
        
        return True
    
    def _record_production(self, p: np.ndarray, sw: np.ndarray, sg: np.ndarray):
        well_rates = self._well_phase_rates(p, sw, sg)
        
        for well in self.wells:
            idx = self.grid.idx(well.i, well.j)
            q_o = -well_rates['oil'][idx]
            q_w = -well_rates['water'][idx]
            q_g = -well_rates['gas'][idx]
            
            self.production_data[well.name].append({
                'time': self.time,
                'oil_rate': q_o,
                'water_rate': q_w,
                'gas_rate': q_g,
                'water_cut': q_w / max(q_o + q_w + q_g, 1e-10),
                'gor': q_g / max(q_o, 1e-10),
                'polymer_concentration': self.C[idx],
            })
    
    def run(self) -> Dict:
        print("=" * 60)
        print(f"  Polymer Flooding EOR Simulator ({self.polymer.name})")
        print("=" * 60)
        print(f"  Grid: {self.grid.nx} x {self.grid.ny} = {self.grid.ncells} cells")
        print(f"  Time step: {self.dt} days")
        print(f"  Total time: {self.t_max} days ({self.nsteps} steps)")
        print(f"  Max viscosity multiplier: {self.polymer.mu_factor:.1f}x")
        print(f"  IPV: {self.polymer.IPV:.2f}")
        print("=" * 60)
        
        for step in range(self.nsteps):
            success = self.step()
            if not success:
                break
            
            if (step + 1) % max(1, self.nsteps // 10) == 0:
                avg_p = np.mean(self.p) / 1e5
                avg_sw = np.mean(self.sw)
                avg_C = np.mean(self.C)
                avg_Cads = np.mean(self.C_ads)
                avg_mu = np.mean(self._compute_water_viscosity())
                print(f"  Step {step+1:5d}/{self.nsteps} | "
                      f"Time: {self.time:7.1f} days | "
                      f"P: {avg_p:6.1f} bar | "
                      f"Sw: {avg_sw:.3f} | "
                      f"C: {avg_C:.0f} ppm | "
                      f"C_ads: {avg_Cads:.1f} ug/g | "
                      f"mu_w: {avg_mu:.2f} cP")
        
        print("=" * 60)
        print("  Simulation Complete")
        print("=" * 60)
        
        return self.get_results()
    
    def get_results(self) -> Dict:
        return {
            'pressure': self.p.copy(),
            'sw': self.sw.copy(),
            'sg': self.sg.copy(),
            'so': self.so.copy(),
            'C_polymer': self.C.copy(),
            'C_adsorbed': self.C_ads.copy(),
            'water_viscosity': self._compute_water_viscosity(),
            'time_steps': np.array(self.time_steps),
            'production': self.production_data,
            'grid': self.grid,
        }


# ============================================================================
# Part 3: CO2 Flooding EOR Model
# ============================================================================

class CO2Properties:
    """
    CO2驱物理属性。
    
    主要效应：
    1. 混相效应：降低原油粘度，改善流度比
    2. 原油膨胀：CO2溶解使原油体积膨胀
    3. 相对渗透率变化：混相后相对渗透率曲线变化
    4. 最小混相压力（MMP）：决定混相程度
    """
    
    def __init__(self,
                 MMP: float = 180.0e5,  # 最小混相压力 (Pa)
                 rho_CO2: float = 0.7,  # CO2密度 (g/cm3)
                 mu_CO2: float = 0.03,  # CO2粘度 (cP)
                 max_oil_swell: float = 1.3,  # 最大原油膨胀系数
                 max_visc_reduction: float = 0.1,  # 最大粘度降低系数
                 solubility_coeff: float = 0.001,  # 溶解度系数 (1/Pa)
                 kr_increase: float = 1.5,  # 混相后相对渗透率增加倍数
                 miscible_efficiency: float = 0.8):  # 混相效率
        self.MMP = MMP
        self.rho_CO2 = rho_CO2
        self.mu_CO2 = mu_CO2
        self.max_oil_swell = max_oil_swell
        self.max_visc_reduction = max_visc_reduction
        self.solubility_coeff = solubility_coeff
        self.kr_increase = kr_increase
        self.miscible_efficiency = miscible_efficiency
    
    def miscibility_factor(self, p: np.ndarray, Sg: np.ndarray) -> np.ndarray:
        """
        计算混相因子（0 = 非混相，1 = 完全混相）。
        基于压力与最小混相压力的比值。
        """
        p_ratio = np.minimum(p / self.MMP, 1.5)
        mf = 1.0 - np.exp(-3.0 * np.maximum(0, p_ratio - 1.0))
        return mf * self.miscible_efficiency
    
    def oil_viscosity(self, p: np.ndarray, mu_o_pure: np.ndarray,
                      miscibility: np.ndarray) -> np.ndarray:
        """计算溶解CO2后的原油粘度。"""
        visc_ratio = 1.0 - miscibility * (1.0 - self.max_visc_reduction)
        return mu_o_pure * visc_ratio
    
    def oil_volume_factor(self, p: np.ndarray, Bo_pure: np.ndarray,
                          miscibility: np.ndarray) -> np.ndarray:
        """计算溶解CO2后的原油体积系数（膨胀效应）。"""
        swell_factor = 1.0 + miscibility * (self.max_oil_swell - 1.0)
        return Bo_pure * swell_factor
    
    def relative_permeability_factor(self, miscibility: np.ndarray) -> np.ndarray:
        """计算混相对相对渗透率的影响。"""
        return 1.0 + miscibility * (self.kr_increase - 1.0)


class CO2FloodSolver:
    """
    CO2驱求解器。
    
    考虑效应：
    - CO2溶解导致原油粘度降低
    - 原油体积膨胀
    - 混相状态对相对渗透率的影响
    - 最小混相压力（MMP）
    """
    
    def __init__(self, grid: Grid, rock: RockProperties,
                 pvt: BlackOilPVT, relperm: RelativePermeability,
                 cap_pres: CapillaryPressure,
                 co2_props: CO2Properties,
                 wells: List[Well] = None,
                 p_init: float = 200.0e5,
                 sw_init: float = 0.20,
                 sg_init: float = 0.0,
                 dt: float = 1.0,
                 t_max: float = 365.0):
        self.grid = grid
        self.rock = rock
        self.pvt = pvt
        self.relperm = relperm
        self.cap_pres = cap_pres
        self.co2_props = co2_props
        self.wells = wells if wells else []
        self.p_init = p_init
        self.sw_init = sw_init
        self.sg_init = sg_init
        self.dt = dt
        self.t_max = t_max
        
        n = grid.ncells
        self.p = np.full(n, p_init)
        self.sw = np.full(n, sw_init)
        self.sg = np.full(n, sg_init)
        self.so = 1.0 - self.sw - self.sg
        
        self.p_old = self.p.copy()
        self.sw_old = self.sw.copy()
        self.sg_old = self.sg.copy()
        
        self.porosity = rock.porosity
        self.volumes = grid.volume()
        self.nsteps = int(np.ceil(t_max / dt))
        self.time = 0.0
        self.time_steps = []
        self.production_data = {w.name: [] for w in self.wells}
        self.mass_balance_errors = []
        self.miscibility_history = []
        
        self._setup_transmissibility()
        for well in self.wells:
            well.compute_well_index(grid, rock)
    
    def _setup_transmissibility(self):
        g = self.grid
        self.trans_x = np.zeros((g.nx - 1, g.ny))
        self.trans_y = np.zeros((g.nx, g.ny - 1))
        
        for i in range(g.nx - 1):
            for j in range(g.ny):
                idx1 = g.idx(i, j)
                idx2 = g.idx(i + 1, j)
                k1 = self.rock.permx[idx1] * MD_TO_M2
                k2 = self.rock.permx[idx2] * MD_TO_M2
                k_harm = 2 * k1 * k2 / max(k1 + k2, 1e-20)
                self.trans_x[i, j] = k_harm * g.dy * g.dz / g.dx
        
        for i in range(g.nx):
            for j in range(g.ny - 1):
                idx1 = g.idx(i, j)
                idx2 = g.idx(i, j + 1)
                k1 = self.rock.permy[idx1] * MD_TO_M2
                k2 = self.rock.permy[idx2] * MD_TO_M2
                k_harm = 2 * k1 * k2 / max(k1 + k2, 1e-20)
                self.trans_y[i, j] = k_harm * g.dx * g.dz / g.dy
    
    def _compute_miscibility(self) -> np.ndarray:
        """计算混相因子。"""
        return self.co2_props.miscibility_factor(self.p, self.sg)
    
    def _well_phase_rates(self, p: np.ndarray, sw: np.ndarray, sg: np.ndarray) -> Dict[str, np.ndarray]:
        n = self.grid.ncells
        rates = {'oil': np.zeros(n), 'water': np.zeros(n), 'gas': np.zeros(n)}
        
        so = 1.0 - sw - sg
        miscibility = self._compute_miscibility()
        kr_factor = self.co2_props.relative_permeability_factor(miscibility)
        
        kro = self.relperm.kro(sw, sg) * kr_factor
        krw = self.relperm.krw(sw)
        krg = self.relperm.krg(sg) * kr_factor
        
        mu_o_pure = self.pvt.mu_o(p)
        mu_o = self.co2_props.oil_viscosity(p, mu_o_pure, miscibility)
        mu_w = self.pvt.mu_w
        mu_g = self.co2_props.mu_CO2
        
        for well in self.wells:
            idx = self.grid.idx(well.i, well.j)
            wi = well.well_index
            
            if well.well_type == 'producer':
                bhp = well.target_bhp
                dp = max(p[idx] - bhp, 0)
                
                qo = wi * kro[idx] / np.maximum(mu_o[idx] * CP_TO_PAS, 1e-10) * dp * PER_DAY
                qw = wi * krw[idx] / np.maximum(mu_w * CP_TO_PAS, 1e-10) * dp * PER_DAY
                qg = wi * krg[idx] / np.maximum(mu_g * CP_TO_PAS, 1e-10) * dp * PER_DAY
                
                rates['oil'][idx] = -qo
                rates['water'][idx] = -qw
                rates['gas'][idx] = -qg
            else:
                q_total = well.target_rate / PER_DAY
                if 'gas' in well.phases:
                    rates['gas'][idx] = q_total
                else:
                    rates['water'][idx] = q_total
        
        return rates
    
    def _compute_phase_flux(self, p: np.ndarray, sw: np.ndarray, sg: np.ndarray,
                            phase: str) -> Tuple[np.ndarray, np.ndarray]:
        g = self.grid
        so = 1.0 - sw - sg
        
        miscibility = self._compute_miscibility()
        kr_factor = self.co2_props.relative_permeability_factor(miscibility)
        
        if phase == 'oil':
            kr = self.relperm.kro(so) * kr_factor
            mu_o_pure = self.pvt.mu_o(p)
            mu = self.co2_props.oil_viscosity(p, mu_o_pure, miscibility)
        elif phase == 'water':
            kr = self.relperm.krw(sw)
            mu = self.pvt.mu_w
        else:
            kr = self.relperm.krg(sg) * kr_factor
            mu = self.co2_props.mu_CO2
        
        mobility = kr / np.maximum(mu * CP_TO_PAS, 1e-10)
        
        fx = np.zeros((g.nx - 1, g.ny))
        fy = np.zeros((g.nx, g.ny - 1))
        
        for i in range(g.nx - 1):
            for j in range(g.ny):
                idx1 = g.idx(i, j)
                idx2 = g.idx(i + 1, j)
                lambda_avg = 0.5 * (mobility[idx1] + mobility[idx2])
                fx[i, j] = -self.trans_x[i, j] * lambda_avg * (p[idx2] - p[idx1])
        
        for i in range(g.nx):
            for j in range(g.ny - 1):
                idx1 = g.idx(i, j)
                idx2 = g.idx(i, j + 1)
                lambda_avg = 0.5 * (mobility[idx1] + mobility[idx2])
                fy[i, j] = -self.trans_y[i, j] * lambda_avg * (p[idx2] - p[idx1])
        
        return fx, fy
    
    def _assemble_pressure_matrix(self) -> Tuple[csr_matrix, np.ndarray]:
        g = self.grid
        n = g.ncells
        mat = lil_matrix((n, n))
        rhs = np.zeros(n)
        
        miscibility = self._compute_miscibility()
        Bo = self.co2_props.oil_volume_factor(self.p, self.pvt.bo(self.p), miscibility)
        well_rates = self._well_phase_rates(self.p, self.sw, self.sg)
        
        for idx in range(n):
            i, j = g.coord(idx)
            acc_coeff = self.porosity[idx] * self.volumes[idx] / self.dt / Bo[idx]
            mat[idx, idx] += acc_coeff
            
            if i > 0:
                mat[idx, idx] += self.trans_x[i - 1, j]
                mat[idx, g.idx(i - 1, j)] -= self.trans_x[i - 1, j]
            if i < g.nx - 1:
                mat[idx, idx] += self.trans_x[i, j]
                mat[idx, g.idx(i + 1, j)] -= self.trans_x[i, j]
            if j > 0:
                mat[idx, idx] += self.trans_y[i, j - 1]
                mat[idx, g.idx(i, j - 1)] -= self.trans_y[i, j - 1]
            if j < g.ny - 1:
                mat[idx, idx] += self.trans_y[i, j]
                mat[idx, g.idx(i, j + 1)] -= self.trans_y[i, j]
            
            rhs[idx] = acc_coeff * self.p_old[idx] + well_rates['oil'][idx] + well_rates['gas'][idx]
        
        return mat.tocsr(), rhs
    
    def step(self) -> bool:
        dt = self.dt
        
        p_old = self.p.copy()
        sw_old = self.sw.copy()
        sg_old = self.sg.copy()
        
        # 压力方程（隐式）
        mat, rhs = self._assemble_pressure_matrix()
        ref_idx = self.grid.idx(self.grid.nx // 2, self.grid.ny // 2)
        mat[ref_idx, ref_idx] += 1e12
        rhs[ref_idx] += 1e12 * p_old[ref_idx]
        
        p_new = spsolve(mat, rhs)
        p_new = np.clip(p_new, 1e4, 500e5)
        
        # 饱和度更新（显式）
        fx_w, fy_w = self._compute_phase_flux(p_new, self.sw, self.sg, 'water')
        fx_g, fy_g = self._compute_phase_flux(p_new, self.sw, self.sg, 'gas')
        well_rates = self._well_phase_rates(p_new, self.sw, self.sg)
        
        sw_new = sw_old.copy()
        sg_new = sg_old.copy()
        
        for idx in range(self.grid.ncells):
            i, j = self.grid.coord(idx)
            
            div_w = 0.0
            if i > 0:
                div_w += fx_w[i - 1, j]
            if i < self.grid.nx - 1:
                div_w -= fx_w[i, j]
            if j > 0:
                div_w += fy_w[i, j - 1]
            if j < self.grid.ny - 1:
                div_w -= fy_w[i, j]
            
            div_g = 0.0
            if i > 0:
                div_g += fx_g[i - 1, j]
            if i < self.grid.nx - 1:
                div_g -= fx_g[i, j]
            if j > 0:
                div_g += fy_g[i, j - 1]
            if j < self.grid.ny - 1:
                div_g -= fy_g[i, j]
            
            acc_coeff = self.porosity[idx] * self.volumes[idx] / dt
            sw_new[idx] = sw_old[idx] - (div_w - well_rates['water'][idx]) / max(acc_coeff, 1e-10)
            sg_new[idx] = sg_old[idx] - (div_g - well_rates['gas'][idx]) / max(acc_coeff, 1e-10)
        
        sw_new = np.clip(sw_new, self.relperm.swc,
                         1 - self.relperm.sor - self.relperm.sgc)
        sg_new = np.clip(sg_new, self.relperm.sgc,
                         1 - self.relperm.sor - self.relperm.swc)
        
        for idx in range(self.grid.ncells):
            if sw_new[idx] + sg_new[idx] > 1 - self.relperm.sor:
                excess = sw_new[idx] + sg_new[idx] - (1 - self.relperm.sor)
                sw_new[idx] -= 0.5 * excess
                sg_new[idx] -= 0.5 * excess
        
        # 更新状态
        self.p = p_new
        self.sw = sw_new
        self.sg = sg_new
        self.so = 1.0 - sw_new - sg_new
        
        self.p_old = p_old
        self.sw_old = sw_old
        self.sg_old = sg_old
        
        miscibility = np.mean(self._compute_miscibility())
        self.miscibility_history.append(miscibility)
        
        self.time += dt
        self.time_steps.append(self.time)
        
        # 记录产量
        self._record_production(p_new, sw_new, sg_new)
        
        return True
    
    def _record_production(self, p: np.ndarray, sw: np.ndarray, sg: np.ndarray):
        well_rates = self._well_phase_rates(p, sw, sg)
        
        for well in self.wells:
            idx = self.grid.idx(well.i, well.j)
            q_o = -well_rates['oil'][idx]
            q_w = -well_rates['water'][idx]
            q_g = -well_rates['gas'][idx]
            
            self.production_data[well.name].append({
                'time': self.time,
                'oil_rate': q_o,
                'water_rate': q_w,
                'gas_rate': q_g,
                'water_cut': q_w / max(q_o + q_w + q_g, 1e-10),
                'gor': q_g / max(q_o, 1e-10),
                'miscibility': self._compute_miscibility()[idx],
            })
    
    def run(self) -> Dict:
        print("=" * 60)
        print("  CO2 Flooding EOR Simulator")
        print("=" * 60)
        print(f"  Grid: {self.grid.nx} x {self.grid.ny} = {self.grid.ncells} cells")
        print(f"  Time step: {self.dt} days")
        print(f"  Total time: {self.t_max} days ({self.nsteps} steps)")
        print(f"  MMP: {self.co2_props.MMP/1e5:.1f} bar")
        print(f"  Max oil swell factor: {self.co2_props.max_oil_swell:.2f}")
        print("=" * 60)
        
        for step in range(self.nsteps):
            success = self.step()
            if not success:
                break
            
            if (step + 1) % max(1, self.nsteps // 10) == 0:
                avg_p = np.mean(self.p) / 1e5
                avg_sw = np.mean(self.sw)
                avg_sg = np.mean(self.sg)
                avg_mf = np.mean(self._compute_miscibility())
                print(f"  Step {step+1:5d}/{self.nsteps} | "
                      f"Time: {self.time:7.1f} days | "
                      f"P: {avg_p:6.1f} bar | "
                      f"Sw: {avg_sw:.3f} | "
                      f"Sg: {avg_sg:.3f} | "
                      f"Miscibility: {avg_mf:.3f}")
        
        print("=" * 60)
        print("  Simulation Complete")
        print("=" * 60)
        
        return self.get_results()
    
    def get_results(self) -> Dict:
        return {
            'pressure': self.p.copy(),
            'sw': self.sw.copy(),
            'sg': self.sg.copy(),
            'so': self.so.copy(),
            'miscibility_factor': self._compute_miscibility(),
            'time_steps': np.array(self.time_steps),
            'miscibility_history': np.array(self.miscibility_history),
            'production': self.production_data,
            'grid': self.grid,
        }


# ============================================================================
# Part 4: Production Optimization Interface
# ============================================================================

class DevelopmentStrategy:
    """开发方案定义。"""
    
    def __init__(self,
                 name: str = 'Base Case',
                 well_positions: List[Tuple[int, int, str, float]] = None,
                 injection_rates: Dict[str, float] = None,
                 production_bhps: Dict[str, float] = None,
                 eor_type: str = None,  # 'polymer' or 'co2'
                 eor_start_time: float = 0.0,
                 eor_concentration: float = 1000.0,
                 simulation_time: float = 365.0 * 5):
        self.name = name
        self.well_positions = well_positions or []
        self.injection_rates = injection_rates or {}
        self.production_bhps = production_bhps or {}
        self.eor_type = eor_type
        self.eor_start_time = eor_start_time
        self.eor_concentration = eor_concentration
        self.simulation_time = simulation_time


class OptimizationResult:
    """优化结果。"""
    
    def __init__(self,
                 strategy: DevelopmentStrategy,
                 cumulative_oil: float,
                 cumulative_water: float,
                 cumulative_gas: float,
                 npv: float = 0.0):
        self.strategy = strategy
        self.cumulative_oil = cumulative_oil
        self.cumulative_water = cumulative_water
        self.cumulative_gas = cumulative_gas
        self.npv = npv
    
    def __repr__(self):
        return (f"OptimizationResult({self.strategy.name}: "
                f"Oil={self.cumulative_oil:.0f} m3, NPV=${self.npv:.0f})")


class FieldOptimizer:
    """
    油田开发方案优化器。
    
    支持优化：
    1. 井位优化
    2. 注入参数优化（速率、浓度）
    3. EOR时机优化
    4. 经济指标评估（NPV）
    """
    
    def __init__(self,
                 grid: Grid, rock: RockProperties,
                 pvt: BlackOilPVT, relperm: RelativePermeability,
                 cap_pres: CapillaryPressure,
                 oil_price: float = 80.0,  # $/bbl
                 water_cost: float = 5.0,  # $/m3
                 gas_price: float = 3.0,  # $/mmBtu
                 discount_rate: float = 0.08):
        self.grid = grid
        self.rock = rock
        self.pvt = pvt
        self.relperm = relperm
        self.cap_pres = cap_pres
        self.oil_price = oil_price
        self.water_cost = water_cost
        self.gas_price = gas_price
        self.discount_rate = discount_rate
        
        self.bbl_to_m3 = 0.158987
    
    def _create_wells(self, strategy: DevelopmentStrategy) -> List[Well]:
        wells = []
        for (i, j, well_type, control_value) in strategy.well_positions:
            name = f"{well_type.upper()}-{len(wells)+1}"
            if well_type == 'producer':
                well = Well(name=name, i=i, j=j, well_type='producer',
                            control_mode='bhp', target_bhp=control_value)
            else:
                phases = ['water']
                if strategy.eor_type == 'co2':
                    phases = ['gas']
                well = Well(name=name, i=i, j=j, well_type='injector',
                            control_mode='rate', target_rate=control_value,
                            phases=phases)
            wells.append(well)
        return wells
    
    def evaluate_strategy(self, strategy: DevelopmentStrategy,
                          solver_type: str = 'impes',
                          eor_props: object = None) -> OptimizationResult:
        """评估一个开发方案。"""
        print(f"\nEvaluating strategy: {strategy.name}")
        
        wells = self._create_wells(strategy)
        
        if solver_type == 'impes':
            from black_oil_impes import IMPESSolver
            solver = IMPESSolver(self.grid, self.rock, self.pvt,
                                 self.relperm, self.cap_pres,
                                 wells=wells, dt=1.0,
                                 t_max=strategy.simulation_time)
        elif solver_type == 'polymer' and eor_props:
            solver = PolymerFloodSolver(self.grid, self.rock, self.pvt,
                                        self.relperm, self.cap_pres,
                                        eor_props, wells=wells, dt=1.0,
                                        t_max=strategy.simulation_time)
        elif solver_type == 'co2' and eor_props:
            solver = CO2FloodSolver(self.grid, self.rock, self.pvt,
                                    self.relperm, self.cap_pres,
                                    eor_props, wells=wells, dt=1.0,
                                    t_max=strategy.simulation_time)
        elif solver_type == 'dual_porosity' and eor_props:
            solver = DualPorositySolver(self.grid, eor_props, self.pvt,
                                        self.relperm, self.cap_pres,
                                        wells=wells, dt=1.0,
                                        t_max=strategy.simulation_time)
        else:
            raise ValueError(f"Unknown solver type: {solver_type}")
        
        results = solver.run()
        
        # 计算累计产量
        cum_oil, cum_water, cum_gas = 0.0, 0.0, 0.0
        for well_name, prod_data in results['production'].items():
            if len(prod_data) > 0:
                t = np.array([d['time'] for d in prod_data])
                dt_arr = np.diff(t, prepend=t[0])
                cum_oil += np.sum(np.array([d['oil_rate'] for d in prod_data]) * dt_arr)
                cum_water += np.sum(np.array([d['water_rate'] for d in prod_data]) * dt_arr)
                cum_gas += np.sum(np.array([d['gas_rate'] for d in prod_data]) * dt_arr)
        
        # 计算NPV
        npv = self._calculate_npv(results, cum_oil, cum_water, cum_gas)
        
        return OptimizationResult(strategy, cum_oil, cum_water, cum_gas, npv)
    
    def _calculate_npv(self, results: Dict, cum_oil: float,
                       cum_water: float, cum_gas: float) -> float:
        """计算净现值。"""
        revenue_oil = cum_oil / self.bbl_to_m3 * self.oil_price
        revenue_gas = cum_gas * self.gas_price
        cost_water = cum_water * self.water_cost
        
        total_time = len(results['time_steps']) if 'time_steps' in results else self.grid.ncells
        discount_factor = 1.0 / (1.0 + self.discount_rate) ** (total_time / 365.0)
        
        npv = (revenue_oil + revenue_gas - cost_water) * discount_factor
        return npv
    
    def optimize_injection_rate(self, base_strategy: DevelopmentStrategy,
                                rate_range: Tuple[float, float],
                                n_points: int = 5) -> List[OptimizationResult]:
        """优化注入速率。"""
        print("\n" + "=" * 60)
        print("  Optimizing Injection Rate")
        print("=" * 60)
        
        results = []
        rates = np.linspace(rate_range[0], rate_range[1], n_points)
        
        for rate in rates:
            new_positions = [
                (i, j, wt, rate if wt == 'injector' else cv)
                for (i, j, wt, cv) in base_strategy.well_positions
            ]
            strategy = DevelopmentStrategy(
                name=f"InjRate_{rate:.0f}m3_day",
                well_positions=new_positions,
                simulation_time=base_strategy.simulation_time
            )
            result = self.evaluate_strategy(strategy)
            results.append(result)
            print(f"  Rate={rate:.0f} m3/day -> Oil={result.cumulative_oil:.0f} m3, "
                  f"NPV=${result.npv:.0f}")
        
        return results
    
    def compare_eor_strategies(self, base_strategy: DevelopmentStrategy,
                                polymer_props: PolymerProperties = None,
                                co2_props: CO2Properties = None) -> Dict[str, OptimizationResult]:
        """对比不同EOR策略。"""
        print("\n" + "=" * 60)
        print("  Comparing EOR Strategies")
        print("=" * 60)
        
        results = {}
        
        # 基础水驱
        strategy_water = DevelopmentStrategy(
            name="Water Flood",
            well_positions=base_strategy.well_positions,
            simulation_time=base_strategy.simulation_time
        )
        results['water_flood'] = self.evaluate_strategy(strategy_water, 'impes')
        
        # 聚合物驱
        if polymer_props:
            strategy_poly = DevelopmentStrategy(
                name="Polymer Flood",
                well_positions=base_strategy.well_positions,
                eor_type='polymer',
                eor_concentration=1500.0,
                simulation_time=base_strategy.simulation_time
            )
            results['polymer_flood'] = self.evaluate_strategy(strategy_poly, 'polymer', polymer_props)
        
        # CO2驱
        if co2_props:
            strategy_co2 = DevelopmentStrategy(
                name="CO2 Flood",
                well_positions=base_strategy.well_positions,
                eor_type='co2',
                simulation_time=base_strategy.simulation_time
            )
            results['co2_flood'] = self.evaluate_strategy(strategy_co2, 'co2', co2_props)
        
        # 打印对比结果
        print("\n" + "=" * 60)
        print("  EOR Strategy Comparison")
        print("=" * 60)
        print(f"  {'Strategy':<15} {'Cum Oil (m3)':<15} {'NPV ($)':<15} {'Increment':<10}")
        print("-" * 60)
        
        base_oil = results['water_flood'].cumulative_oil
        for name, result in results.items():
            increment = (result.cumulative_oil - base_oil) / max(base_oil, 1e-10) * 100
            print(f"  {name:<15} {result.cumulative_oil:<15.0f} "
                  f"{result.npv:<15.0f} {increment:<10.1f}%")
        print("=" * 60)
        
        return results


# ============================================================================
# End of File
# ============================================================================

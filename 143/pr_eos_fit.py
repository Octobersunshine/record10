import numpy as np
from scipy.optimize import minimize, root_scalar
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class PengRobinsonEOS:
    def __init__(self, R: float = 8.314, epsilon: float = 1e-12):
        self.R = R
        self.epsilon = epsilon
        self.params = {}

    def _safe_denominator(self, denom: float, min_val: float = None) -> float:
        if min_val is None:
            min_val = self.epsilon
        if abs(denom) < min_val:
            return np.sign(denom) * min_val if denom != 0 else min_val
        return denom

    def pressure(self, V: float, T: float, a: float, b: float, alpha: float) -> float:
        V_clamped = max(V, b + self.epsilon)
        denom1 = self._safe_denominator(V_clamped - b, self.epsilon)
        denom2 = self._safe_denominator(V_clamped * (V_clamped + b) + b * (V_clamped - b), self.epsilon)
        return self.R * T / denom1 - a * alpha / denom2

    def dPdV(self, V: float, T: float, a: float, b: float, alpha: float) -> float:
        V_clamped = max(V, b + self.epsilon)
        term1_denom = self._safe_denominator((V_clamped - b) ** 2, self.epsilon)
        term1 = -self.R * T / term1_denom
        
        denom2 = V_clamped * (V_clamped + b) + b * (V_clamped - b)
        denom2_safe = self._safe_denominator(denom2 ** 2, self.epsilon)
        term2 = a * alpha * (2 * V_clamped + 2 * b) / denom2_safe
        return term1 + term2

    def d2PdV2(self, V: float, T: float, a: float, b: float, alpha: float) -> float:
        V_clamped = max(V, b + self.epsilon)
        term1_denom = self._safe_denominator((V_clamped - b) ** 3, self.epsilon)
        term1 = 2 * self.R * T / term1_denom
        
        denom2 = V_clamped * (V_clamped + b) + b * (V_clamped - b)
        denom2_safe = self._safe_denominator(denom2 ** 3, self.epsilon)
        numerator = 2 * a * alpha * ((2 * V_clamped + 2 * b) ** 2 - 2 * denom2)
        term2 = numerator / denom2_safe
        return term1 + term2

    def dPdT(self, V: float, T: float, a: float, b: float, alpha: float, 
             Tc: float, kappa: float) -> float:
        V_clamped = max(V, b + self.epsilon)
        denom1 = self._safe_denominator(V_clamped - b, self.epsilon)
        
        Tr = T / Tc if Tc > 0 else 1.0
        sqrt_Tr = np.sqrt(max(Tr, self.epsilon))
        dalpha_dT = -2 * alpha * kappa / (2 * sqrt_Tr * Tc) if sqrt_Tr > 0 else 0
        
        denom2 = self._safe_denominator(V_clamped * (V_clamped + b) + b * (V_clamped - b), self.epsilon)
        return self.R / denom1 - a * dalpha_dT / denom2

    def calculate_alpha(self, T: float, Tc: float, kappa: float) -> float:
        T_clamped = max(T, self.epsilon)
        Tc_clamped = max(Tc, self.epsilon)
        Tr = min(T_clamped / Tc_clamped, 10.0)
        sqrt_Tr = np.sqrt(max(Tr, self.epsilon))
        return (1 + kappa * (1 - sqrt_Tr)) ** 2

    def calculate_kappa(self, omega: float) -> float:
        return 0.37464 + 1.54226 * omega - 0.26992 * omega ** 2

    def heidemann_khalil_conditions(self, a: float, b: float, Tc: float, 
                                     omega: float) -> Tuple[float, float, float]:
        kappa = self.calculate_kappa(omega)
        alpha_c = self.calculate_alpha(Tc, Tc, kappa)
        
        Vc = self._find_critical_volume(a, b, Tc, alpha_c)
        
        dPdV_c = self.dPdV(Vc, Tc, a, b, alpha_c)
        d2PdV2_c = self.d2PdV2(Vc, Tc, a, b, alpha_c)
        
        Pc = self.pressure(Vc, Tc, a, b, alpha_c)
        
        return dPdV_c, d2PdV2_c, Pc

    def _find_critical_volume(self, a: float, b: float, Tc: float, 
                               alpha_c: float) -> float:
        def objective(V):
            return self.dPdV(V, Tc, a, b, alpha_c)
        
        V_low = b * 1.01
        V_high = 10 * b
        
        try:
            result = root_scalar(objective, bracket=[V_low, V_high], method='brentq')
            return result.root
        except:
            return 3 * b

    def solve_volume_damped_newton(self, P: float, T: float, a: float, b: float, 
                                    alpha: float, phase: str = 'vapor',
                                    max_iter: int = 200, tol: float = 1e-12) -> float:
        if phase == 'liquid':
            V = b * 1.01
        elif phase == 'vapor':
            V = max(self.R * T / P * 2.0, b * 10)
        else:
            V = max(self.R * T / P, b * 2)
        
        best_V = V
        best_residual = float('inf')
        
        for i in range(max_iter):
            V_clamped = max(V, b + self.epsilon)
            
            P_calc = self.pressure(V_clamped, T, a, b, alpha)
            dPdV = self.dPdV(V_clamped, T, a, b, alpha)
            
            residual = P_calc - P
            
            if abs(residual) < abs(best_residual):
                best_residual = residual
                best_V = V_clamped
            
            if abs(residual) < tol:
                return V_clamped
            
            if abs(dPdV) < 1e-15:
                dPdV = -1e-10
            
            delta_V = -residual / dPdV
            
            damping_factor = 1.0
            V_new = V_clamped + damping_factor * delta_V
            
            max_step = 0.5 * V_clamped
            while abs(V_new - V_clamped) > max_step or V_new <= b + self.epsilon:
                damping_factor *= 0.5
                V_new = V_clamped + damping_factor * delta_V
                if damping_factor < 1e-4:
                    if V_new <= b + self.epsilon:
                        V_new = max(V_clamped * 0.7, b * 1.01)
                    break
            
            V = V_new
        
        return best_V

    def solve_volume_homotopy(self, P: float, T: float, a: float, b: float, 
                               alpha: float, phase: str = 'vapor') -> float:
        n_steps = 20
        P_ideal = self.R * T / (b * 2) if phase == 'liquid' else self.R * T / (b * 100)
        V_prev = self.R * T / P_ideal
        
        for step in range(n_steps + 1):
            lambda_param = step / n_steps
            P_current = P_ideal * (1 - lambda_param) + P * lambda_param
            
            V_prev = self.solve_volume_damped_newton(
                P_current, T, a, b, alpha, phase=phase,
                max_iter=50, tol=1e-10
            )
        
        return V_prev

    def objective_function(self, params: List[float], V_data: np.ndarray, 
                           P_data: np.ndarray, T_data: np.ndarray,
                           weight_critical: float = 1000.0,
                           weight_pressure: float = 1.0,
                           weight_consistency: float = 100.0) -> float:
        a, b, Tc, omega = params
        
        if a <= 0 or b <= 0 or Tc <= 0 or omega < 0 or omega > 2:
            return 1e20
        
        kappa = self.calculate_kappa(omega)
        
        pressure_error = 0.0
        valid_points = 0
        
        for i in range(len(V_data)):
            if V_data[i] <= b + self.epsilon:
                continue
                
            alpha = self.calculate_alpha(T_data[i], Tc, kappa)
            try:
                P_calc = self.pressure(V_data[i], T_data[i], a, b, alpha)
                if np.isnan(P_calc) or np.isinf(P_calc):
                    pressure_error += 1e10
                else:
                    rel_error = (P_calc - P_data[i]) / P_data[i]
                    pressure_error += rel_error ** 2
                    valid_points += 1
            except:
                pressure_error += 1e10
        
        if valid_points > 0:
            pressure_error /= valid_points
        
        try:
            dPdV_c, d2PdV2_c, Pc_c = self.heidemann_khalil_conditions(a, b, Tc, omega)
            
            Pc_ref = 0.457235 * self.R ** 2 * Tc ** 2 / a if a > 0 else 1e6
            Vc_ref = 3 * b
            
            dPdV_norm = dPdV_c * Vc_ref / max(Pc_ref, self.epsilon)
            d2PdV2_norm = d2PdV2_c * Vc_ref ** 2 / max(Pc_ref, self.epsilon)
            
            critical_error = dPdV_norm ** 2 + d2PdV2_norm ** 2
            
            Pc_theo = 0.457235 * self.R ** 2 * Tc ** 2 / a if a > 0 else 1e10
            b_theo = 0.077796 * self.R * Tc / max(Pc_theo, self.epsilon)
            consistency_error = ((b - b_theo) / max(b_theo, self.epsilon)) ** 2
        except:
            critical_error = 1e10
            consistency_error = 1e10
        
        total_error = (weight_pressure * pressure_error + 
                       weight_critical * critical_error + 
                       weight_consistency * consistency_error)
        
        return total_error

    def fit(self, V_data: np.ndarray, P_data: np.ndarray, 
            T_data: np.ndarray, initial_guess: Dict = None,
            weight_critical: float = 1000.0,
            weight_pressure: float = 1.0,
            weight_consistency: float = 100.0) -> Dict:
        V_data = np.asarray(V_data)
        P_data = np.asarray(P_data)
        T_data = np.asarray(T_data)

        P_max = np.max(P_data)
        T_max = np.max(T_data)
        P_min = np.min(P_data)
        
        if initial_guess is None:
            Pc_guess = P_max * 1.1
            Tc_guess = T_max * 1.05
            a_guess = 0.457235 * self.R ** 2 * Tc_guess ** 2 / Pc_guess
            b_guess = 0.077796 * self.R * Tc_guess / Pc_guess
            omega_guess = 0.1
            initial_guess = [a_guess, b_guess, Tc_guess, omega_guess]
        else:
            initial_guess = [
                initial_guess.get('a', 1.0),
                initial_guess.get('b', 0.001),
                initial_guess.get('Tc', 500.0),
                initial_guess.get('omega', 0.1)
            ]

        bounds = [
            (max(1e-5 * initial_guess[0], 1e-10), 1e3 * initial_guess[0]),
            (max(1e-5 * initial_guess[1], 1e-15), 1e3 * initial_guess[1]),
            (max(T_max * 0.8, 100.0), T_max * 5.0),
            (0.0, 1.5)
        ]

        try:
            result = minimize(
                self.objective_function,
                initial_guess,
                args=(V_data, P_data, T_data, weight_critical, weight_pressure, weight_consistency),
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': 10000, 'ftol': 1e-12, 'gtol': 1e-10, 'eps': 1e-8}
            )
        except Exception as e:
            print(f"优化失败: {e}")
            a_opt, b_opt, Tc_opt, omega_opt = initial_guess
            result = type('Result', (), {'success': False, 'fun': float('inf'), 'message': str(e)})()
        else:
            a_opt, b_opt, Tc_opt, omega_opt = result.x

        kappa_opt = self.calculate_kappa(omega_opt)
        Pc_opt = 0.457235 * self.R ** 2 * Tc_opt ** 2 / a_opt

        try:
            dPdV_c, d2PdV2_c, Pc_calc = self.heidemann_khalil_conditions(a_opt, b_opt, Tc_opt, omega_opt)
        except:
            dPdV_c, d2PdV2_c, Pc_calc = float('nan'), float('nan'), float('nan')

        self.params = {
            'a': a_opt,
            'b': b_opt,
            'Tc': Tc_opt,
            'Pc': Pc_opt,
            'Pc_calc': Pc_calc,
            'omega': omega_opt,
            'kappa': kappa_opt,
            'success': result.success,
            'message': result.message,
            'objective_value': result.fun,
            'dPdV_critical': dPdV_c,
            'd2PdV2_critical': d2PdV2_c
        }

        return self.params

    def predict_pressure(self, V: np.ndarray, T: np.ndarray) -> np.ndarray:
        if not self.params:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        V = np.asarray(V)
        T = np.asarray(T)
        
        a = self.params['a']
        b = self.params['b']
        Tc = self.params['Tc']
        kappa = self.params['kappa']
        
        P_pred = np.zeros_like(V, dtype=float)
        for i in range(len(V)):
            alpha = self.calculate_alpha(T[i], Tc, kappa)
            P_pred[i] = self.pressure(V[i], T[i], a, b, alpha)
        
        return P_pred

    def predict_volume(self, P: np.ndarray, T: np.ndarray, phase: str = 'vapor',
                        use_homotopy: bool = False) -> np.ndarray:
        if not self.params:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        P = np.asarray(P)
        T = np.asarray(T)
        
        a = self.params['a']
        b = self.params['b']
        Tc = self.params['Tc']
        kappa = self.params['kappa']
        
        V_pred = np.zeros_like(P, dtype=float)
        for i in range(len(P)):
            alpha = self.calculate_alpha(T[i], Tc, kappa)
            if use_homotopy:
                V_pred[i] = self.solve_volume_homotopy(P[i], T[i], a, b, alpha, phase)
            else:
                V_pred[i] = self.solve_volume_damped_newton(P[i], T[i], a, b, alpha, phase)
        
        return V_pred

    def print_results(self):
        if not self.params:
            print("Model has not been fitted yet.")
            return
        
        print("=" * 60)
        print("Peng-Robinson状态方程拟合结果（含Heidemann-Khalil稳定化）")
        print("=" * 60)
        print(f"a = {self.params['a']:.6e} Pa·m⁶/mol²")
        print(f"b = {self.params['b']:.6e} m³/mol")
        print(f"Tc = {self.params['Tc']:.2f} K")
        print(f"Pc = {self.params['Pc']:.6e} Pa")
        print(f"Pc (计算值) = {self.params.get('Pc_calc', float('nan')):.6e} Pa")
        print(f"omega = {self.params['omega']:.6f}")
        print(f"kappa = {self.params['kappa']:.6f}")
        print("-" * 60)
        print(f"临界点条件验证:")
        print(f"  dP/dV (临界点) = {self.params.get('dPdV_critical', float('nan')):.6e}")
        print(f"  d²P/dV² (临界点) = {self.params.get('d2PdV2_critical', float('nan')):.6e}")
        print("-" * 60)
        print(f"拟合成功: {self.params['success']}")
        print(f"目标函数值: {self.params['objective_value']:.6e}")
        print(f"优化消息: {self.params['message']}")
        print("=" * 60)


def generate_sample_data(n_points: int = 30, include_near_critical: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    a_true = 0.45724
    b_true = 7.780e-5
    Tc_true = 304.13
    omega_true = 0.225

    eos = PengRobinsonEOS()
    kappa = eos.calculate_kappa(omega_true)

    V_data = []
    P_data = []
    T_data = []

    for T in [273.15, 293.15, 303.15, 313.15, 333.15]:
        if T < Tc_true:
            for V in np.logspace(-4.5, -2.5, 6):
                alpha = eos.calculate_alpha(T, Tc_true, kappa)
                P = eos.pressure(V, T, a_true, b_true, alpha)
                if P > 0:
                    V_data.append(V)
                    P_data.append(P)
                    T_data.append(T)
        else:
            for V in np.logspace(-4, -2, 6):
                alpha = eos.calculate_alpha(T, Tc_true, kappa)
                P = eos.pressure(V, T, a_true, b_true, alpha)
                if P > 0:
                    V_data.append(V)
                    P_data.append(P)
                    T_data.append(T)

    if include_near_critical:
        T_near = Tc_true * 1.02
        for V in np.logspace(-4.2, -3.5, 5):
            alpha = eos.calculate_alpha(T_near, Tc_true, kappa)
            P = eos.pressure(V, T_near, a_true, b_true, alpha)
            if P > 0:
                V_data.append(V)
                P_data.append(P)
                T_data.append(T_near)

    V_data = np.array(V_data)
    P_data = np.array(P_data)
    T_data = np.array(T_data)

    noise = np.random.normal(0, 0.015, len(P_data))
    P_data = P_data * (1 + noise)

    return V_data, P_data, T_data


if __name__ == "__main__":
    np.random.seed(42)
    V_data, P_data, T_data = generate_sample_data(30, include_near_critical=True)

    print("示例PVT数据统计:")
    print(f"  数据点数: {len(V_data)}")
    print(f"  体积范围: {V_data.min():.2e} ~ {V_data.max():.2e} m³/mol")
    print(f"  压力范围: {P_data.min():.2e} ~ {P_data.max():.2e} Pa")
    print(f"  温度范围: {T_data.min():.2f} ~ {T_data.max():.2f} K")
    print()

    pr_eos = PengRobinsonEOS()
    params = pr_eos.fit(V_data, P_data, T_data, 
                        weight_critical=500.0,
                        weight_pressure=1.0,
                        weight_consistency=100.0)
    pr_eos.print_results()

    print("\n拟合验证（前10个数据点）:")
    P_pred = pr_eos.predict_pressure(V_data, T_data)
    for i in range(min(10, len(V_data))):
        rel_error = abs(P_pred[i] - P_data[i]) / P_data[i] * 100
        print(f"点{i+1}: 实验P = {P_data[i]:.2e} Pa, 预测P = {P_pred[i]:.2e} Pa, "
              f"相对误差 = {rel_error:.2f}%")
    
    avg_error = np.mean(np.abs(P_pred - P_data) / P_data) * 100
    print(f"\n平均相对误差: {avg_error:.2f}%")

    print("\n体积求解测试（阻尼牛顿法）:")
    test_P = np.array([1e6, 2e6, 5e6])
    test_T = np.array([300.0, 300.0, 300.0])
    V_liquid = pr_eos.predict_volume(test_P, test_T, phase='liquid')
    V_vapor = pr_eos.predict_volume(test_P, test_T, phase='vapor')
    
    for i in range(len(test_P)):
        print(f"P = {test_P[i]:.1e} Pa, T = {test_T[i]:.1f} K:")
        print(f"  液相体积 = {V_liquid[i]:.6e} m³/mol")
        print(f"  气相体积 = {V_vapor[i]:.6e} m³/mol")

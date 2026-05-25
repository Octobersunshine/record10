import numpy as np
from scipy.optimize import minimize, root_scalar, fsolve
from typing import Tuple, List, Dict, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class CPAEOS:
    def __init__(self, R: float = 8.314, epsilon: float = 1e-12):
        self.R = R
        self.epsilon = epsilon
        self.components = []
        self.params = {}
        self.binary_params = {}
        self.n_components = 0

    def set_components(self, components: List[Union[str, Dict]]):
        self.components = []
        self.n_components = len(components)
        
        for i, comp in enumerate(components):
            if isinstance(comp, str):
                if comp in PURE_COMPONENT_PARAMS:
                    comp_params = PURE_COMPONENT_PARAMS[comp].copy()
                    comp_params['name'] = comp
                    self.components.append(comp_params)
                else:
                    raise ValueError(f"Unknown component: {comp}")
            elif isinstance(comp, dict):
                self.components.append(comp)
            else:
                raise ValueError(f"Invalid component format: {comp}")

    def set_binary_interaction(self, i: int, j: int, k_ij: float, 
                               l_ij: float = 0.0, beta_ij: float = None, 
                               gamma_ij: float = None):
        key = tuple(sorted([i, j]))
        self.binary_params[key] = {
            'k_ij': k_ij,
            'l_ij': l_ij,
            'beta_ij': beta_ij,
            'gamma_ij': gamma_ij
        }

    def _get_binary_param(self, i: int, j: int, param: str, default: float = 0.0) -> float:
        if i == j:
            return 0.0 if param in ['k_ij', 'l_ij'] else None
        key = tuple(sorted([i, j]))
        if key in self.binary_params:
            val = self.binary_params[key].get(param)
            if val is not None:
                return val
        return default

    def calculate_alpha(self, T: float, Tc: float, kappa: float) -> float:
        T_clamped = max(T, self.epsilon)
        Tc_clamped = max(Tc, self.epsilon)
        Tr = min(T_clamped / Tc_clamped, 10.0)
        sqrt_Tr = np.sqrt(max(Tr, self.epsilon))
        return (1 + kappa * (1 - sqrt_Tr)) ** 2

    def calculate_mixture_ab(self, x: np.ndarray, T: float) -> Tuple[float, float]:
        x = np.asarray(x)
        a_mix = 0.0
        b_mix = 0.0
        
        for i in range(self.n_components):
            comp_i = self.components[i]
            alpha_i = self.calculate_alpha(T, comp_i['Tc'], comp_i['kappa'])
            a_i = comp_i['a'] * alpha_i
            b_i = comp_i['b']
            
            for j in range(self.n_components):
                comp_j = self.components[j]
                alpha_j = self.calculate_alpha(T, comp_j['Tc'], comp_j['kappa'])
                a_j = comp_j['a'] * alpha_j
                
                k_ij = self._get_binary_param(i, j, 'k_ij', 0.0)
                a_ij = np.sqrt(a_i * a_j) * (1 - k_ij)
                a_mix += x[i] * x[j] * a_ij
            
            b_mix += x[i] * b_i
        
        return a_mix, b_mix

    def pressure_physical(self, V: float, T: float, a_mix: float, b_mix: float) -> float:
        V_clamped = max(V, b_mix + self.epsilon)
        denom1 = max(V_clamped - b_mix, self.epsilon)
        denom2 = max(V_clamped * (V_clamped + b_mix) + b_mix * (V_clamped - b_mix), self.epsilon)
        return self.R * T / denom1 - a_mix / denom2

    def calculate_association_strength(self, i: int, j: int, T: float, V: float, 
                                        b_mix: float) -> float:
        comp_i = self.components[i]
        comp_j = self.components[j]
        
        if comp_i.get('assoc_sites', 0) == 0 or comp_j.get('assoc_sites', 0) == 0:
            return 0.0
        
        eps_i = comp_i.get('eps_assoc', 0.0)
        eps_j = comp_j.get('eps_assoc', 0.0)
        beta_i = comp_i.get('beta_assoc', 0.0)
        beta_j = comp_j.get('beta_assoc', 0.0)
        
        key = tuple(sorted([i, j]))
        if key in self.binary_params:
            beta_ij = self.binary_params[key].get('beta_ij')
            gamma_ij = self.binary_params[key].get('gamma_ij')
            if beta_ij is not None:
                beta_ij = np.sqrt(beta_i * beta_j) * beta_ij
            else:
                beta_ij = np.sqrt(beta_i * beta_j)
            if gamma_ij is not None:
                eps_ij = (eps_i + eps_j) / 2 * gamma_ij
            else:
                eps_ij = (eps_i + eps_j) / 2
        else:
            beta_ij = np.sqrt(beta_i * beta_j)
            eps_ij = (eps_i + eps_j) / 2
        
        rho = 1.0 / max(V, self.epsilon)
        g = self._hard_sphere_radial_distribution(rho, b_mix)
        
        return g * (np.exp(eps_ij / (self.R * T)) - 1) * beta_ij * b_mix

    def _hard_sphere_radial_distribution(self, rho: float, b_mix: float) -> float:
        eta = rho * b_mix / 4.0
        eta_clamped = min(max(eta, self.epsilon), 0.49)
        return (1 - 0.5 * eta_clamped) / (1 - eta_clamped) ** 3

    def calculate_monmer_fractions(self, x: np.ndarray, T: float, V: float, 
                                    b_mix: float) -> Dict[Tuple[int, str], float]:
        X = {}
        rho = 1.0 / max(V, self.epsilon)
        
        for i in range(self.n_components):
            comp = self.components[i]
            sites = comp.get('assoc_sites_dict', {})
            for site_type in sites:
                X[(i, site_type)] = 1.0
        
        max_iter = 100
        tol = 1e-10
        
        for iteration in range(max_iter):
            X_old = X.copy()
            max_diff = 0.0
            
            for i in range(self.n_components):
                comp_i = self.components[i]
                sites_i = comp_i.get('assoc_sites_dict', {})
                if not sites_i:
                    continue
                
                for site_type_A in sites_i:
                    denom = 1.0
                    for j in range(self.n_components):
                        comp_j = self.components[j]
                        sites_j = comp_j.get('assoc_sites_dict', {})
                        if not sites_j:
                            continue
                        
                        for site_type_B in sites_j:
                            if site_type_A == site_type_B:
                                if i == j:
                                    continue
                            
                            Delta_ijAB = self.calculate_association_strength(i, j, T, V, b_mix)
                            X_jB = X_old.get((j, site_type_B), 1.0)
                            denom += rho * x[j] * X_jB * Delta_ijAB * sites_j.get(site_type_B, 0)
                    
                    X[(i, site_type_A)] = 1.0 / max(denom, self.epsilon)
                    max_diff = max(max_diff, abs(X[(i, site_type_A)] - X_old[(i, site_type_A)]))
            
            if max_diff < tol:
                break
        
        return X

    def pressure_association(self, x: np.ndarray, T: float, V: float, 
                              b_mix: float) -> float:
        X = self.calculate_monmer_fractions(x, T, V, b_mix)
        rho = 1.0 / max(V, self.epsilon)
        
        sum_term = 0.0
        for i in range(self.n_components):
            comp_i = self.components[i]
            sites_i = comp_i.get('assoc_sites_dict', {})
            for site_type in sites_i:
                X_iA = X.get((i, site_type), 1.0)
                sum_term += x[i] * sites_i[site_type] * (np.log(X_iA) + (1 - X_iA) / 2)
        
        return -self.R * T * rho * sum_term

    def pressure(self, x: np.ndarray, T: float, V: float) -> float:
        a_mix, b_mix = self.calculate_mixture_ab(x, T)
        P_physical = self.pressure_physical(V, T, a_mix, b_mix)
        P_assoc = self.pressure_association(x, T, V, b_mix)
        return P_physical + P_assoc

    def ln_phi_physical(self, x: np.ndarray, T: float, V: float, 
                         a_mix: float, b_mix: float) -> np.ndarray:
        x = np.asarray(x)
        rho = 1.0 / max(V, self.epsilon)
        Z = P * V / (self.R * T) if 'P' in locals() else (V / (V - b_mix) - a_mix / (self.R * T * (V + b_mix)))
        
        ln_phi = np.zeros(self.n_components)
        
        for i in range(self.n_components):
            comp_i = self.components[i]
            alpha_i = self.calculate_alpha(T, comp_i['Tc'], comp_i['kappa'])
            a_i = comp_i['a'] * alpha_i
            b_i = comp_i['b']
            
            sum_a = 0.0
            for j in range(self.n_components):
                comp_j = self.components[j]
                alpha_j = self.calculate_alpha(T, comp_j['Tc'], comp_j['kappa'])
                a_j = comp_j['a'] * alpha_j
                k_ij = self._get_binary_param(i, j, 'k_ij', 0.0)
                sum_a += x[j] * np.sqrt(a_i * a_j) * (1 - k_ij)
            
            term1 = b_i / (V - b_mix)
            term2 = -np.log((V - b_mix) / V)
            term3 = -a_mix / (2 * np.sqrt(2) * self.R * T * b_mix) * \
                    (2 * sum_a / a_mix - b_i / b_mix) * \
                    np.log((V + b_mix * (1 + np.sqrt(2))) / (V + b_mix * (1 - np.sqrt(2))))
            
            ln_phi[i] = term1 + term2 + term3
        
        return ln_phi

    def ln_phi_association(self, x: np.ndarray, T: float, V: float, 
                            b_mix: float) -> np.ndarray:
        X = self.calculate_monmer_fractions(x, T, V, b_mix)
        rho = 1.0 / max(V, self.epsilon)
        
        ln_phi_assoc = np.zeros(self.n_components)
        
        for i in range(self.n_components):
            comp_i = self.components[i]
            sites_i = comp_i.get('assoc_sites_dict', {})
            if not sites_i:
                continue
            
            sum_assoc = 0.0
            for site_type_A in sites_i:
                X_iA = X.get((i, site_type_A), 1.0)
                sum_assoc += sites_i[site_type_A] * (np.log(X_iA) + (1 - X_iA) / 2)
                
                sum_delta = 0.0
                for j in range(self.n_components):
                    comp_j = self.components[j]
                    sites_j = comp_j.get('assoc_sites_dict', {})
                    for site_type_B in sites_j:
                        X_jB = X.get((j, site_type_B), 1.0)
                        Delta_ijAB = self.calculate_association_strength(i, j, T, V, b_mix)
                        sum_delta += x[j] * sites_j.get(site_type_B, 0) * X_jB * Delta_ijAB * (1 - X_iA)
                
                sum_assoc -= X_iA / 2 * sum_delta
            
            ln_phi_assoc[i] = sum_assoc
        
        return ln_phi_assoc

    def ln_phi(self, x: np.ndarray, T: float, P: float) -> np.ndarray:
        V = self.solve_volume(x, T, P, phase='liquid')
        a_mix, b_mix = self.calculate_mixture_ab(x, T)
        ln_phi_phys = self.ln_phi_physical(x, T, V, a_mix, b_mix)
        ln_phi_assoc = self.ln_phi_association(x, T, V, b_mix)
        return ln_phi_phys + ln_phi_assoc

    def solve_volume(self, x: np.ndarray, T: float, P: float, phase: str = 'liquid',
                      max_iter: int = 200, tol: float = 1e-12) -> float:
        a_mix, b_mix = self.calculate_mixture_ab(x, T)
        
        if phase == 'liquid':
            V = b_mix * 1.5
        elif phase == 'vapor':
            V = max(self.R * T / P * 2.0, b_mix * 10)
        else:
            V = max(self.R * T / P, b_mix * 2)
        
        best_V = V
        best_residual = float('inf')
        
        for i in range(max_iter):
            V_clamped = max(V, b_mix + self.epsilon)
            
            P_calc = self.pressure(x, T, V_clamped)
            
            dPdV = self._numerical_dPdV(x, T, V_clamped)
            
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
            while abs(V_new - V_clamped) > max_step or V_new <= b_mix + self.epsilon:
                damping_factor *= 0.5
                V_new = V_clamped + damping_factor * delta_V
                if damping_factor < 1e-4:
                    if V_new <= b_mix + self.epsilon:
                        V_new = max(V_clamped * 0.7, b_mix * 1.5)
                    break
            
            V = V_new
        
        return best_V

    def _numerical_dPdV(self, x: np.ndarray, T: float, V: float, h: float = 1e-6) -> float:
        V_plus = V * (1 + h)
        V_minus = V * (1 - h)
        P_plus = self.pressure(x, T, V_plus)
        P_minus = self.pressure(x, T, V_minus)
        return (P_plus - P_minus) / (V_plus - V_minus)

    def bubble_pressure(self, x: np.ndarray, T: float, P_guess: float = None) -> Tuple[float, np.ndarray]:
        x = np.asarray(x)
        
        if P_guess is None:
            P_sat_sum = 0.0
            for i in range(self.n_components):
                comp = self.components[i]
                P_sat = self._estimate_saturation_pressure(i, T)
                P_sat_sum += x[i] * P_sat
            P_guess = P_sat_sum
        
        def objective(P):
            ln_phi_L = self.ln_phi(x, T, P)
            y_est = np.exp(np.log(x) + ln_phi_L - np.min(ln_phi_L))
            y_est = y_est / np.sum(y_est)
            
            ln_phi_V = self.ln_phi(y_est, T, P)
            
            K = np.exp(ln_phi_L - ln_phi_V)
            return np.sum(x * K) - 1.0
        
        try:
            result = root_scalar(objective, bracket=[P_guess * 0.1, P_guess * 10], method='brentq')
            P_bubble = result.root
            
            ln_phi_L = self.ln_phi(x, T, P_bubble)
            y_est = np.exp(np.log(x) + ln_phi_L - np.min(ln_phi_L))
            y_est = y_est / np.sum(y_est)
            ln_phi_V = self.ln_phi(y_est, T, P_bubble)
            K = np.exp(ln_phi_L - ln_phi_V)
            y = x * K
            
            return P_bubble, y
        except:
            return P_guess, x

    def dew_pressure(self, y: np.ndarray, T: float, P_guess: float = None) -> Tuple[float, np.ndarray]:
        y = np.asarray(y)
        
        if P_guess is None:
            P_sat_max = 0.0
            for i in range(self.n_components):
                P_sat = self._estimate_saturation_pressure(i, T)
                P_sat_max = max(P_sat_max, y[i] * P_sat)
            P_guess = P_sat_max * 2
        
        def objective(P):
            ln_phi_V = self.ln_phi(y, T, P)
            x_est = np.exp(np.log(y) + ln_phi_V - np.min(ln_phi_V))
            x_est = x_est / np.sum(x_est)
            ln_phi_L = self.ln_phi(x_est, T, P)
            
            K = np.exp(ln_phi_L - ln_phi_V)
            return np.sum(y / K) - 1.0
        
        try:
            result = root_scalar(objective, bracket=[P_guess * 0.1, P_guess * 10], method='brentq')
            P_dew = result.root
            
            ln_phi_V = self.ln_phi(y, T, P_dew)
            x_est = np.exp(np.log(y) + ln_phi_V - np.min(ln_phi_V))
            x_est = x_est / np.sum(x_est)
            ln_phi_L = self.ln_phi(x_est, T, P_dew)
            K = np.exp(ln_phi_L - ln_phi_V)
            x = y / K
            
            return P_dew, x
        except:
            return P_guess, y

    def _estimate_saturation_pressure(self, comp_idx: int, T: float) -> float:
        comp = self.components[comp_idx]
        Tc = comp['Tc']
        Pc = comp['Pc']
        omega = comp['omega']
        
        if T >= Tc:
            return Pc
        
        Tr = T / Tc
        log_Pr = 5.373 * (1 + omega) * (1 - 1 / Tr)
        return Pc * np.exp(log_Pr)

    def lle_flash(self, z: np.ndarray, T: float, P: float, 
                   x1_guess: np.ndarray = None, x2_guess: np.ndarray = None) -> Tuple[float, np.ndarray, np.ndarray]:
        z = np.asarray(z)
        
        if x1_guess is None:
            x1_guess = z.copy()
            x1_guess[0] = 0.9
            x1_guess = x1_guess / np.sum(x1_guess)
        
        if x2_guess is None:
            x2_guess = z.copy()
            x2_guess[-1] = 0.9
            x2_guess = x2_guess / np.sum(x2_guess)
        
        def objective(vars_):
            n_vars = len(vars_)
            x1 = vars_[:self.n_components]
            x2 = vars_[self.n_components:2*self.n_components]
            beta = vars_[-1]
            
            x1 = np.maximum(x1, 1e-10)
            x2 = np.maximum(x2, 1e-10)
            x1 = x1 / np.sum(x1)
            x2 = x2 / np.sum(x2)
            beta = np.clip(beta, 0.001, 0.999)
            
            ln_gamma1 = self.ln_phi(x1, T, P)
            ln_gamma2 = self.ln_phi(x2, T, P)
            
            K = np.exp(ln_gamma1 - ln_gamma2)
            
            material_balance = K * x1 - x2
            
            sum_constraint1 = np.sum(x1) - 1
            sum_constraint2 = np.sum(x2) - 1
            
            return np.concatenate([material_balance, [sum_constraint1], [sum_constraint2]])
        
        try:
            vars0 = np.concatenate([x1_guess, x2_guess, [0.5]])
            result = fsolve(objective, vars0)
            
            x1 = result[:self.n_components]
            x2 = result[self.n_components:2*self.n_components]
            beta = result[-1]
            
            x1 = np.maximum(x1, 1e-10)
            x2 = np.maximum(x2, 1e-10)
            x1 = x1 / np.sum(x1)
            x2 = x2 / np.sum(x2)
            
            return beta, x1, x2
        except:
            return 0.5, x1_guess, x2_guess


PURE_COMPONENT_PARAMS = {
    'water': {
        'Tc': 647.14,
        'Pc': 2.2064e7,
        'omega': 0.344,
        'a': 1.2277,
        'b': 1.4515e-5,
        'kappa': 0.6736,
        'assoc_sites': 4,
        'assoc_sites_dict': {'H': 2, 'e': 2},
        'eps_assoc': 16655.0,
        'beta_assoc': 0.0692
    },
    'methanol': {
        'Tc': 512.6,
        'Pc': 8.096e6,
        'omega': 0.565,
        'a': 2.5662,
        'b': 3.6843e-5,
        'kappa': 1.0016,
        'assoc_sites': 2,
        'assoc_sites_dict': {'H': 1, 'e': 1},
        'eps_assoc': 13690.0,
        'beta_assoc': 0.0855
    },
    'ethanol': {
        'Tc': 513.9,
        'Pc': 6.148e6,
        'omega': 0.644,
        'a': 3.5702,
        'b': 5.0635e-5,
        'kappa': 1.1174,
        'assoc_sites': 2,
        'assoc_sites_dict': {'H': 1, 'e': 1},
        'eps_assoc': 13390.0,
        'beta_assoc': 0.0797
    },
    '1-propanol': {
        'Tc': 536.8,
        'Pc': 5.175e6,
        'omega': 0.623,
        'a': 4.640,
        'b': 6.440e-5,
        'kappa': 1.15,
        'assoc_sites': 2,
        'assoc_sites_dict': {'H': 1, 'e': 1},
        'eps_assoc': 13000.0,
        'beta_assoc': 0.075
    },
    'methane': {
        'Tc': 190.6,
        'Pc': 4.599e6,
        'omega': 0.011,
        'a': 0.2496,
        'b': 2.6843e-5,
        'kappa': 0.0535,
        'assoc_sites': 0,
        'assoc_sites_dict': {}
    },
    'ethane': {
        'Tc': 305.3,
        'Pc': 4.872e6,
        'omega': 0.099,
        'a': 0.5579,
        'b': 4.5050e-5,
        'kappa': 0.2883,
        'assoc_sites': 0,
        'assoc_sites_dict': {}
    },
    'propane': {
        'Tc': 369.8,
        'Pc': 4.248e6,
        'omega': 0.152,
        'a': 0.9487,
        'b': 6.2690e-5,
        'kappa': 0.3776,
        'assoc_sites': 0,
        'assoc_sites_dict': {}
    },
    'n-butane': {
        'Tc': 425.2,
        'Pc': 3.796e6,
        'omega': 0.193,
        'a': 1.3860,
        'b': 8.0600e-5,
        'kappa': 0.4498,
        'assoc_sites': 0,
        'assoc_sites_dict': {}
    },
    'co2': {
        'Tc': 304.13,
        'Pc': 7.377e6,
        'omega': 0.225,
        'a': 0.4572,
        'b': 2.9680e-5,
        'kappa': 0.3746,
        'assoc_sites': 0,
        'assoc_sites_dict': {}
    },
    'n2': {
        'Tc': 126.2,
        'Pc': 3.390e6,
        'omega': 0.039,
        'a': 0.1361,
        'b': 2.6770e-5,
        'kappa': 0.1350,
        'assoc_sites': 0,
        'assoc_sites_dict': {}
    },
    'benzene': {
        'Tc': 562.2,
        'Pc': 4.898e6,
        'omega': 0.212,
        'a': 1.908,
        'b': 8.293e-5,
        'kappa': 0.3946,
        'assoc_sites': 0,
        'assoc_sites_dict': {}
    }
}


if __name__ == "__main__":
    print("=" * 60)
    print("CPA状态方程示例 - 水-甲醇体系气液平衡计算")
    print("=" * 60)
    
    cpa = CPAEOS()
    
    cpa.set_components(['water', 'methanol'])
    
    cpa.set_binary_interaction(0, 1, k_ij=-0.05)
    
    T = 323.15
    x_water = np.linspace(0.1, 0.9, 5)
    
    print(f"\n温度: {T} K")
    print(f"\n气液平衡计算（泡点压力）:")
    print("-" * 60)
    print(f"{'x_water':>10} {'x_methanol':>12} {'P_bubble':>15} {'y_water':>12} {'y_methanol':>12}")
    print("-" * 60)
    
    for xw in x_water:
        x = np.array([xw, 1 - xw])
        P_bubble, y = cpa.bubble_pressure(x, T)
        print(f"{xw:10.4f} {1-xw:12.4f} {P_bubble/1e5:15.4f} bar {y[0]:12.4f} {y[1]:12.4f}")
    
    print("\n" + "=" * 60)
    print("液液平衡计算示例 - 水-苯体系（估算）")
    print("=" * 60)
    
    cpa2 = CPAEOS()
    cpa2.set_components(['water', 'benzene'])
    cpa2.set_binary_interaction(0, 1, k_ij=0.5)
    
    T_lde = 298.15
    P_lde = 1e5
    z = np.array([0.5, 0.5])
    
    beta, x1, x2 = cpa2.lle_flash(z, T_lde, P_lde)
    
    print(f"\n温度: {T_lde} K, 压力: {P_lde/1e5:.2f} bar")
    print(f"进料组成: 水={z[0]:.4f}, 苯={z[1]:.4f}")
    print(f"\n相1 (水相): 水={x1[0]:.6f}, 苯={x1[1]:.6f}")
    print(f"相2 (有机相): 水={x2[0]:.6f}, 苯={x2[1]:.6f}")
    print(f"相1摩尔分率: {beta:.4f}")
    
    print("\n" + "=" * 60)
    print("示例完成")
    print("=" * 60)

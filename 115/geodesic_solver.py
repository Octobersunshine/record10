import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, root
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Surface:
    """曲面基类"""
    
    def parametric(self, u, v):
        """参数化表示: X(u,v) = (x(u,v), y(u,v), z(u,v))"""
        raise NotImplementedError
    
    def derivatives(self, u, v):
        """计算一阶和二阶偏导数（中心差分）"""
        eps = 1e-6
        
        X = self.parametric(u, v)
        Xu = (self.parametric(u + eps, v) - self.parametric(u - eps, v)) / (2 * eps)
        Xv = (self.parametric(u, v + eps) - self.parametric(u, v - eps)) / (2 * eps)
        
        Xuu = (self.parametric(u + eps, v) - 2 * X + self.parametric(u - eps, v)) / (eps ** 2)
        Xuv = (self.parametric(u + eps, v + eps) - self.parametric(u + eps, v - eps) - 
               self.parametric(u - eps, v + eps) + self.parametric(u - eps, v - eps)) / (4 * eps ** 2)
        Xvv = (self.parametric(u, v + eps) - 2 * X + self.parametric(u, v - eps)) / (eps ** 2)
        
        return X, Xu, Xv, Xuu, Xuv, Xvv
    
    def first_fundamental_form(self, u, v):
        """计算第一基本形式系数 E, F, G"""
        _, Xu, Xv, _, _, _ = self.derivatives(u, v)
        E = np.dot(Xu, Xu)
        F = np.dot(Xu, Xv)
        G = np.dot(Xv, Xv)
        return E, F, G
    
    def christoffel_symbols(self, u, v):
        """计算克里斯托费尔符号 Γ^k_ij"""
        E, F, G = self.first_fundamental_form(u, v)
        eps = 1e-6
        
        Eu = (self.first_fundamental_form(u + eps, v)[0] - self.first_fundamental_form(u - eps, v)[0]) / (2 * eps)
        Ev = (self.first_fundamental_form(u, v + eps)[0] - self.first_fundamental_form(u, v - eps)[0]) / (2 * eps)
        
        Fu = (self.first_fundamental_form(u + eps, v)[1] - self.first_fundamental_form(u - eps, v)[1]) / (2 * eps)
        Fv = (self.first_fundamental_form(u, v + eps)[1] - self.first_fundamental_form(u, v - eps)[1]) / (2 * eps)
        
        Gu = (self.first_fundamental_form(u + eps, v)[2] - self.first_fundamental_form(u - eps, v)[2]) / (2 * eps)
        Gv = (self.first_fundamental_form(u, v + eps)[2] - self.first_fundamental_form(u, v - eps)[2]) / (2 * eps)
        
        det = E * G - F ** 2
        if abs(det) < 1e-12:
            det = 1e-12 * np.sign(det)
        
        Γ1_11 = (G * Eu - 2 * F * Fu + F * Ev) / (2 * det)
        Γ1_12 = (G * Ev - F * Gu) / (2 * det)
        Γ1_22 = (2 * G * Fv - G * Gu - F * Gv) / (2 * det)
        
        Γ2_11 = (2 * E * Fu - E * Ev - F * Eu) / (2 * det)
        Γ2_12 = (E * Gu - F * Ev) / (2 * det)
        Γ2_22 = (E * Gv - 2 * F * Fv + F * Gu) / (2 * det)
        
        return Γ1_11, Γ1_12, Γ1_22, Γ2_11, Γ2_12, Γ2_22
    
    def metric_norm(self, u, v, up, vp):
        """计算切向量的黎曼度量范数: ||(up, vp)||² = E*up² + 2*F*up*vp + G*vp²"""
        E, F, G = self.first_fundamental_form(u, v)
        return E * up * up + 2 * F * up * vp + G * vp * vp
    
    def project_to_unit_speed(self, u, v, up, vp):
        """将切向量投影到单位速度（弧长参数化约束）"""
        norm_sq = self.metric_norm(u, v, up, vp)
        if norm_sq < 1e-12:
            return up, vp
        scale = 1.0 / np.sqrt(norm_sq)
        return up * scale, vp * scale
    
    def geodesic_ode_stable(self, t, y):
        """
        稳定版测地线微分方程，带弧长参数化约束
        y = [u, v, u', v']
        """
        u, v, up, vp = y
        
        Γ1_11, Γ1_12, Γ1_22, Γ2_11, Γ2_12, Γ2_22 = self.christoffel_symbols(u, v)
        
        upp = -(Γ1_11 * up * up + 2 * Γ1_12 * up * vp + Γ1_22 * vp * vp)
        vpp = -(Γ2_11 * up * up + 2 * Γ2_12 * up * vp + Γ2_22 * vp * vp)
        
        return [up, vp, upp, vpp]
    
    def compute_geodesic_rk45(self, u0, v0, du0, dv0, t_span=(0, 10), num_points=1000):
        """
        原始方法：龙格-库塔法求解测地线（保留用于对比）
        """
        y0 = [u0, v0, du0, dv0]
        t_eval = np.linspace(t_span[0], t_span[1], num_points)
        
        solution = solve_ivp(
            self.geodesic_ode_stable, 
            t_span, 
            y0, 
            method='RK45',
            t_eval=t_eval,
            rtol=1e-8,
            atol=1e-10
        )
        
        u_vals = solution.y[0]
        v_vals = solution.y[1]
        
        points = np.array([self.parametric(u, v) for u, v in zip(u_vals, v_vals)])
        
        return {
            't': solution.t,
            'u': u_vals,
            'v': v_vals,
            'points': points,
            'success': solution.success,
            'method': 'RK45'
        }
    
    def compute_geodesic_constrained(self, u0, v0, du0, dv0, t_span=(0, 10), num_points=1000, dt=None):
        """
        约束积分法：显式积分 + 每步速度投影到单位切向量
        这能有效防止刚性发散
        """
        if dt is None:
            dt = (t_span[1] - t_span[0]) / (num_points * 10)
        
        t_total = t_span[1] - t_span[0]
        n_steps = int(np.ceil(t_total / dt))
        
        u = u0
        v = v0
        up, vp = self.project_to_unit_speed(u0, v0, du0, dv0)
        
        u_vals = [u]
        v_vals = [v]
        t_vals = [t_span[0]]
        
        for i in range(n_steps):
            Γ1_11, Γ1_12, Γ1_22, Γ2_11, Γ2_12, Γ2_22 = self.christoffel_symbols(u, v)
            
            upp = -(Γ1_11 * up * up + 2 * Γ1_12 * up * vp + Γ1_22 * vp * vp)
            vpp = -(Γ2_11 * up * up + 2 * Γ2_12 * up * vp + Γ2_22 * vp * vp)
            
            up = up + dt * upp
            vp = vp + dt * vpp
            
            u = u + dt * up
            v = v + dt * vp
            
            up, vp = self.project_to_unit_speed(u, v, up, vp)
            
            u_vals.append(u)
            v_vals.append(v)
            t_vals.append(t_span[0] + (i + 1) * dt)
        
        u_vals = np.array(u_vals)
        v_vals = np.array(v_vals)
        t_vals = np.array(t_vals)
        
        indices = np.linspace(0, len(u_vals) - 1, num_points, dtype=int)
        u_vals = u_vals[indices]
        v_vals = v_vals[indices]
        t_vals = t_vals[indices]
        
        points = np.array([self.parametric(u, v) for u, v in zip(u_vals, v_vals)])
        
        return {
            't': t_vals,
            'u': u_vals,
            'v': v_vals,
            'points': points,
            'success': True,
            'method': 'Constrained_Euler'
        }
    
    def energy_functional(self, params, n_points):
        """
        能量泛函: E = ∫(E(u')² + 2F(u'v') + G(v')²) dt
        params = [u0, u1, ..., uN, v0, v1, ..., vN]
        """
        u_vals = params[:n_points]
        v_vals = params[n_points:]
        
        energy = 0.0
        dt = 1.0 / (n_points - 1)
        
        for i in range(n_points - 1):
            up = (u_vals[i + 1] - u_vals[i]) / dt
            vp = (v_vals[i + 1] - v_vals[i]) / dt
            
            u_mid = (u_vals[i] + u_vals[i + 1]) / 2
            v_mid = (v_vals[i] + v_vals[i + 1]) / 2
            
            E, F, G = self.first_fundamental_form(u_mid, v_mid)
            energy += (E * up * up + 2 * F * up * vp + G * vp * vp) * dt
        
        return energy
    
    def length_functional(self, params, n_points):
        """
        弧长泛函: L = ∫√(E(u')² + 2F(u'v') + G(v')²) dt
        """
        u_vals = params[:n_points]
        v_vals = params[n_points:]
        
        length = 0.0
        dt = 1.0 / (n_points - 1)
        
        for i in range(n_points - 1):
            up = (u_vals[i + 1] - u_vals[i]) / dt
            vp = (v_vals[i + 1] - v_vals[i]) / dt
            
            u_mid = (u_vals[i] + u_vals[i + 1]) / 2
            v_mid = (v_vals[i] + v_vals[i + 1]) / 2
            
            E, F, G = self.first_fundamental_form(u_mid, v_mid)
            metric_norm_sq = E * up * up + 2 * F * up * vp + G * vp * vp
            length += np.sqrt(max(metric_norm_sq, 1e-12)) * dt
        
        return length
    
    def compute_geodesic_variational(self, u_start, v_start, u_end, v_end, n_points=50, max_iter=1000):
        """
        变分法求解两点间测地线（最小化能量泛函）
        使用L-BFGS-B优化器
        """
        u_init = np.linspace(u_start, u_end, n_points)
        v_init = np.linspace(v_start, v_end, n_points)
        params_init = np.concatenate([u_init, v_init])
        
        def objective(params):
            return self.energy_functional(params, n_points)
        
        bounds = []
        for i in range(n_points):
            if i == 0:
                bounds.append((u_start, u_start))
            elif i == n_points - 1:
                bounds.append((u_end, u_end))
            else:
                bounds.append((None, None))
        
        for i in range(n_points):
            if i == 0:
                bounds.append((v_start, v_start))
            elif i == n_points - 1:
                bounds.append((v_end, v_end))
            else:
                bounds.append((None, None))
        
        result = minimize(
            objective,
            params_init,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': max_iter,
                'ftol': 1e-10,
                'gtol': 1e-8
            }
        )
        
        u_vals = result.x[:n_points]
        v_vals = result.x[n_points:]
        points = np.array([self.parametric(u, v) for u, v in zip(u_vals, v_vals)])
        
        t_vals = np.linspace(0, 1, n_points)
        
        return {
            't': t_vals,
            'u': u_vals,
            'v': v_vals,
            'points': points,
            'success': result.success,
            'method': 'Variational',
            'energy': result.fun,
            'message': result.message
        }
    
    def compute_geodesic_shooting(self, u0, v0, u_end, v_end, n_points=100, max_shots=50):
        """
        打靶法：调整初始方向使得测地线到达目标点
        """
        def target_distance(du0_dv0):
            du0, dv0 = du0_dv0
            geo = self.compute_geodesic_constrained(
                u0, v0, du0, dv0,
                t_span=(0, 1),
                num_points=n_points
            )
            u_last = geo['u'][-1]
            v_last = geo['v'][-1]
            return (u_last - u_end)**2 + (v_last - v_end)**2
        
        du_init = (u_end - u0) * 2
        dv_init = (v_end - v0) * 2
        
        result = minimize(
            target_distance,
            [du_init, dv_init],
            method='L-BFGS-B',
            options={'maxiter': max_shots}
        )
        
        du_opt, dv_opt = result.x
        geo = self.compute_geodesic_constrained(
            u0, v0, du_opt, dv_opt,
            t_span=(0, 1),
            num_points=n_points
        )
        
        geo['initial_velocity'] = (du_opt, dv_opt)
        geo['target_error'] = result.fun
        geo['method'] = 'Shooting'
        
        return geo
    
    def compute_geodesic_implicit_midpoint(self, u0, v0, du0, dv0, t_span=(0, 10), num_points=1000, dt=None):
        """
        隐式中点法 + 弧长参数化约束
        无条件稳定，非常适合刚性问题
        """
        if dt is None:
            dt = (t_span[1] - t_span[0]) / num_points
        
        t_total = t_span[1] - t_span[0]
        n_steps = int(np.ceil(t_total / dt))
        
        u = u0
        v = v0
        up, vp = self.project_to_unit_speed(u0, v0, du0, dv0)
        
        u_vals = [u]
        v_vals = [v]
        t_vals = [t_span[0]]
        
        for i in range(n_steps):
            def residual(x):
                u_mid, v_mid, up_mid, vp_mid = x
                
                Γ1_11, Γ1_12, Γ1_22, Γ2_11, Γ2_12, Γ2_22 = self.christoffel_symbols(u_mid, v_mid)
                
                upp_mid = -(Γ1_11 * up_mid * up_mid + 2 * Γ1_12 * up_mid * vp_mid + Γ1_22 * vp_mid * vp_mid)
                vpp_mid = -(Γ2_11 * up_mid * up_mid + 2 * Γ2_12 * up_mid * vp_mid + Γ2_22 * vp_mid * vp_mid)
                
                f1 = u_mid - (u + dt/2 * up)
                f2 = v_mid - (v + dt/2 * vp)
                f3 = up_mid - (up + dt * upp_mid)
                f4 = vp_mid - (vp + dt * vpp_mid)
                
                return [f1, f2, f3, f4]
            
            x0 = [u + dt/2 * up, v + dt/2 * vp, up, vp]
            
            try:
                sol = root(residual, x0, method='hybr', tol=1e-10)
                u_mid, v_mid, up_mid, vp_mid = sol.x
            except:
                u_mid = u + dt/2 * up
                v_mid = v + dt/2 * vp
                Γ1_11, Γ1_12, Γ1_22, Γ2_11, Γ2_12, Γ2_22 = self.christoffel_symbols(u, v)
                upp = -(Γ1_11 * up * up + 2 * Γ1_12 * up * vp + Γ1_22 * vp * vp)
                vpp = -(Γ2_11 * up * up + 2 * Γ2_12 * up * vp + Γ2_22 * vp * vp)
                up_mid = up + dt/2 * upp
                vp_mid = vp + dt/2 * vpp
            
            u = u + dt * up_mid
            v = v + dt * vp_mid
            up = up_mid
            vp = vp_mid
            
            up, vp = self.project_to_unit_speed(u, v, up, vp)
            
            u_vals.append(u)
            v_vals.append(v)
            t_vals.append(t_span[0] + (i + 1) * dt)
        
        u_vals = np.array(u_vals)
        v_vals = np.array(v_vals)
        t_vals = np.array(t_vals)
        
        indices = np.linspace(0, len(u_vals) - 1, num_points, dtype=int)
        u_vals = u_vals[indices]
        v_vals = v_vals[indices]
        t_vals = t_vals[indices]
        
        points = np.array([self.parametric(u, v) for u, v in zip(u_vals, v_vals)])
        
        return {
            't': t_vals,
            'u': u_vals,
            'v': v_vals,
            'points': points,
            'success': True,
            'method': 'Implicit_Midpoint'
        }
    
    def energy_gradient(self, params, n_points):
        """
        能量泛函的解析梯度，用于加速变分法优化
        dE/du_k 和 dE/dv_k
        """
        u_vals = params[:n_points]
        v_vals = params[n_points:]
        dt = 1.0 / (n_points - 1)
        
        grad_u = np.zeros(n_points)
        grad_v = np.zeros(n_points)
        
        eps = 1e-6
        
        for i in range(n_points - 1):
            up = (u_vals[i + 1] - u_vals[i]) / dt
            vp = (v_vals[i + 1] - v_vals[i]) / dt
            
            u_mid = (u_vals[i] + u_vals[i + 1]) / 2
            v_mid = (v_vals[i] + v_vals[i + 1]) / 2
            
            E, F, G = self.first_fundamental_form(u_mid, v_mid)
            
            Eu = (self.first_fundamental_form(u_mid + eps, v_mid)[0] - self.first_fundamental_form(u_mid - eps, v_mid)[0]) / (2 * eps)
            Ev = (self.first_fundamental_form(u_mid, v_mid + eps)[0] - self.first_fundamental_form(u_mid, v_mid - eps)[0]) / (2 * eps)
            
            Fu = (self.first_fundamental_form(u_mid + eps, v_mid)[1] - self.first_fundamental_form(u_mid - eps, v_mid)[1]) / (2 * eps)
            Fv = (self.first_fundamental_form(u_mid, v_mid + eps)[1] - self.first_fundamental_form(u_mid, v_mid - eps)[1]) / (2 * eps)
            
            Gu = (self.first_fundamental_form(u_mid + eps, v_mid)[2] - self.first_fundamental_form(u_mid - eps, v_mid)[2]) / (2 * eps)
            Gv = (self.first_fundamental_form(u_mid, v_mid + eps)[2] - self.first_fundamental_form(u_mid, v_mid - eps)[2]) / (2 * eps)
            
            dE_dumid = 0.5 * (Eu * up * up + 2 * Fu * up * vp + Gu * vp * vp) * dt
            dE_dvmid = 0.5 * (Ev * up * up + 2 * Fv * up * vp + Gv * vp * vp) * dt
            
            dE_dui = (-2 * E * up - 2 * F * vp) / dt * dt
            dE_dui1 = (2 * E * up + 2 * F * vp) / dt * dt
            dE_dvi = (-2 * F * up - 2 * G * vp) / dt * dt
            dE_dvi1 = (2 * F * up + 2 * G * vp) / dt * dt
            
            grad_u[i] += 0.5 * dE_dumid + dE_dui
            grad_u[i + 1] += 0.5 * dE_dumid + dE_dui1
            grad_v[i] += 0.5 * dE_dvmid + dE_dvi
            grad_v[i + 1] += 0.5 * dE_dvmid + dE_dvi1
        
        return np.concatenate([grad_u, grad_v])
    
    def compute_geodesic_variational_fast(self, u_start, v_start, u_end, v_end, n_points=50, max_iter=500):
        """
        快速变分法：使用解析梯度加速优化
        """
        u_init = np.linspace(u_start, u_end, n_points)
        v_init = np.linspace(v_start, v_end, n_points)
        params_init = np.concatenate([u_init, v_init])
        
        def objective(params):
            return self.energy_functional(params, n_points)
        
        def gradient(params):
            return self.energy_gradient(params, n_points)
        
        bounds = []
        for i in range(n_points):
            if i == 0:
                bounds.append((u_start, u_start))
            elif i == n_points - 1:
                bounds.append((u_end, u_end))
            else:
                bounds.append((None, None))
        
        for i in range(n_points):
            if i == 0:
                bounds.append((v_start, v_start))
            elif i == n_points - 1:
                bounds.append((v_end, v_end))
            else:
                bounds.append((None, None))
        
        result = minimize(
            objective,
            params_init,
            jac=gradient,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': max_iter,
                'ftol': 1e-10,
                'gtol': 1e-8
            }
        )
        
        u_vals = result.x[:n_points]
        v_vals = result.x[n_points:]
        points = np.array([self.parametric(u, v) for u, v in zip(u_vals, v_vals)])
        
        t_vals = np.linspace(0, 1, n_points)
        
        return {
            't': t_vals,
            'u': u_vals,
            'v': v_vals,
            'points': points,
            'success': result.success,
            'method': 'Variational_Fast',
            'energy': result.fun,
            'message': result.message,
            'nit': result.nit
        }
    
    def compute_geodesic_adaptive(self, u0, v0, du0, dv0, t_span=(0, 10), num_points=1000, 
                                   dt_init=1e-3, dt_min=1e-6, dt_max=0.1, tol=1e-6):
        """
        自适应步长约束积分法
        根据曲率自动调整步长，提高效率和稳定性
        """
        t = t_span[0]
        u = u0
        v = v0
        up, vp = self.project_to_unit_speed(u0, v0, du0, dv0)
        
        u_vals = [u]
        v_vals = [v]
        t_vals = [t]
        dt_vals = []
        
        dt = dt_init
        
        while t < t_span[1]:
            if t + dt > t_span[1]:
                dt = t_span[1] - t
            
            Γ1_11, Γ1_12, Γ1_22, Γ2_11, Γ2_12, Γ2_22 = self.christoffel_symbols(u, v)
            
            curvature = np.abs(Γ1_11) + np.abs(Γ1_12) + np.abs(Γ1_22) + \
                       np.abs(Γ2_11) + np.abs(Γ2_12) + np.abs(Γ2_22)
            
            dt_opt = min(dt_max, max(dt_min, tol / (curvature + 1e-10)))
            
            if dt > dt_opt * 2:
                dt = dt_opt
                continue
            
            upp = -(Γ1_11 * up * up + 2 * Γ1_12 * up * vp + Γ1_22 * vp * vp)
            vpp = -(Γ2_11 * up * up + 2 * Γ2_12 * up * vp + Γ2_22 * vp * vp)
            
            up_new = up + dt * upp
            vp_new = vp + dt * vpp
            
            u_new = u + dt * up
            v_new = v + dt * vp
            
            up_new, vp_new = self.project_to_unit_speed(u_new, v_new, up_new, vp_new)
            
            t = t + dt
            u = u_new
            v = v_new
            up = up_new
            vp = vp_new
            
            u_vals.append(u)
            v_vals.append(v)
            t_vals.append(t)
            dt_vals.append(dt)
            
            dt = min(dt_max, dt * 1.1)
        
        u_vals = np.array(u_vals)
        v_vals = np.array(v_vals)
        t_vals = np.array(t_vals)
        
        t_interp = np.linspace(t_span[0], t_span[1], num_points)
        u_interp = np.interp(t_interp, t_vals, u_vals)
        v_interp = np.interp(t_interp, t_vals, v_vals)
        
        points = np.array([self.parametric(u, v) for u, v in zip(u_interp, v_interp)])
        
        return {
            't': t_interp,
            'u': u_interp,
            'v': v_interp,
            'points': points,
            'success': True,
            'method': 'Adaptive_Constrained',
            'dt_stats': {
                'mean_dt': np.mean(dt_vals) if dt_vals else 0,
                'min_dt': np.min(dt_vals) if dt_vals else 0,
                'max_dt': np.max(dt_vals) if dt_vals else 0,
                'n_steps': len(dt_vals)
            }
        }


class Sphere(Surface):
    """球面: X(u,v) = R*(sin(u)cos(v), sin(u)sin(v), cos(u))"""
    
    def __init__(self, radius=1.0):
        self.radius = radius
    
    def parametric(self, u, v):
        x = self.radius * np.sin(u) * np.cos(v)
        y = self.radius * np.sin(u) * np.sin(v)
        z = self.radius * np.cos(u)
        return np.array([x, y, z])


class Torus(Surface):
    """环面: X(u,v) = ((R + r*cos(u))*cos(v), (R + r*cos(u))*sin(v), r*sin(u))"""
    
    def __init__(self, major_radius=2.0, minor_radius=1.0):
        self.R = major_radius
        self.r = minor_radius
    
    def parametric(self, u, v):
        x = (self.R + self.r * np.cos(u)) * np.cos(v)
        y = (self.R + self.r * np.cos(u)) * np.sin(v)
        z = self.r * np.sin(u)
        return np.array([x, y, z])


class Cylinder(Surface):
    """圆柱面: X(u,v) = (R*cos(u), R*sin(u), v)"""
    
    def __init__(self, radius=1.0):
        self.radius = radius
    
    def parametric(self, u, v):
        x = self.radius * np.cos(u)
        y = self.radius * np.sin(u)
        z = v
        return np.array([x, y, z])


def visualize_geodesic(surface, geodesic_result, title="测地线", u_range=(0, 2*np.pi), v_range=(0, 2*np.pi)):
    """可视化曲面和测地线"""
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    u_grid = np.linspace(u_range[0], u_range[1], 50)
    v_grid = np.linspace(v_range[0], v_range[1], 50)
    U, V = np.meshgrid(u_grid, v_grid)
    
    X = np.zeros_like(U)
    Y = np.zeros_like(U)
    Z = np.zeros_like(U)
    
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            p = surface.parametric(U[i, j], V[i, j])
            X[i, j], Y[i, j], Z[i, j] = p
    
    ax.plot_surface(X, Y, Z, alpha=0.3, color='lightblue', edgecolor='gray', linewidth=0.5)
    
    points = geodesic_result['points']
    ax.plot(points[:, 0], points[:, 1], points[:, 2], 'r-', linewidth=2.5, label='测地线')
    
    ax.scatter(points[0, 0], points[0, 1], points[0, 2], color='green', s=120, label='起点', zorder=10)
    ax.scatter(points[-1, 0], points[-1, 1], points[-1, 2], color='blue', s=120, label='终点', zorder=10)
    
    method = geodesic_result.get('method', 'Unknown')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"{title}\n方法: {method}")
    ax.legend()
    ax.set_box_aspect([1, 1, 1])
    
    plt.tight_layout()
    return fig, ax


def compute_geodesic_length(geodesic_result):
    """计算测地线的弧长"""
    points = geodesic_result['points']
    diffs = np.diff(points, axis=0)
    lengths = np.linalg.norm(diffs, axis=1)
    total_length = np.sum(lengths)
    return total_length


def check_arc_length_parameterization(surface, geodesic_result):
    """检查弧长参数化: ||X'(t)|| 应该近似恒定"""
    u_vals = geodesic_result['u']
    v_vals = geodesic_result['v']
    t_vals = geodesic_result['t']
    
    speeds = []
    for i in range(len(u_vals) - 1):
        dt = t_vals[i + 1] - t_vals[i]
        up = (u_vals[i + 1] - u_vals[i]) / dt
        vp = (v_vals[i + 1] - v_vals[i]) / dt
        u_mid = (u_vals[i] + u_vals[i + 1]) / 2
        v_mid = (v_vals[i] + v_vals[i + 1]) / 2
        speed = np.sqrt(surface.metric_norm(u_mid, v_mid, up, vp))
        speeds.append(speed)
    
    speeds = np.array(speeds)
    mean_speed = np.mean(speeds)
    std_speed = np.std(speeds)
    
    return {
        'mean_speed': mean_speed,
        'std_speed': std_speed,
        'speed_variation': std_speed / abs(mean_speed) if mean_speed != 0 else float('inf'),
        'speeds': speeds
    }

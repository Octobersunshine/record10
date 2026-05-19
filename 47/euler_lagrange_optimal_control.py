import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Optional, List


class AugmentedLagrangian:
    """
    增广拉格朗日方法：处理不等式约束
    对于约束 g(u) ≤ 0，增广拉格朗日为：
    L_a(u, λ, c) = f(u) + (1/(2c)) * [max(λ + c*g(u), 0)² - λ²]
    """
    def __init__(
        self,
        f: Callable[[np.ndarray], float],
        g_list: List[Callable[[np.ndarray], np.ndarray]],
        c0: float = 1.0,
        c_max: float = 1e6,
        beta: float = 10.0,
        tol: float = 1e-6
    ):
        self.f = f
        self.g_list = g_list
        self.c = c0
        self.c_max = c_max
        self.beta = beta
        self.tol = tol
        self.lambdas = [np.zeros(1) for _ in g_list]
        
    def augmented_lagrangian(self, u: np.ndarray) -> float:
        cost = self.f(u)
        
        for i, g in enumerate(self.g_list):
            g_u = g(u)
            lam = self.lambdas[i]
            term = np.maximum(lam + self.c * g_u, 0)
            cost += (np.sum(term**2) - np.sum(lam**2)) / (2 * self.c)
            
        return cost
    
    def constraint_violation(self, u: np.ndarray) -> float:
        violation = 0.0
        for g in self.g_list:
            g_u = g(u)
            violation += np.sum(np.maximum(g_u, 0)**2)
        return np.sqrt(violation)
    
    def solve(self, u0: np.ndarray, method: str = 'SLSQP') -> Tuple[np.ndarray, float]:
        u = u0.copy()
        iteration = 0
        
        while self.c < self.c_max:
            iteration += 1
            
            result = minimize(
                self.augmented_lagrangian,
                u,
                method=method,
                options={'maxiter': 1000, 'ftol': 1e-8}
            )
            u = result.x
            
            constraint_violation = self.constraint_violation(u)
            if constraint_violation < self.tol:
                print(f"增广拉格朗日收敛于第 {iteration} 次迭代，约束违反 = {constraint_violation:.2e}")
                return u, constraint_violation
            
            for i, g in enumerate(self.g_list):
                self.lambdas[i] = np.maximum(self.lambdas[i] + self.c * g(u), 0)
            
            self.c = min(self.c * self.beta, self.c_max)
            print(f"迭代 {iteration}: c = {self.c:.1e}, 约束违反 = {constraint_violation:.2e}")
        
        return u, self.constraint_violation(u)


class ConstrainedEulerLagrangeSolver:
    """
    带不等式约束的直接法求解器
    泛函: J = Φ(x(T)) + ∫₀ᵀ L(t, x, u) dt
    约束: dx/dt = f(t, x, u)
          u_min ≤ u(t) ≤ u_max
    """
    def __init__(
        self,
        L: Callable[[float, float, float], float],
        f: Callable[[float, float, float], float],
        t_span: Tuple[float, float],
        x0: float,
        phi: Optional[Callable[[float], float]] = None,
        xf: Optional[float] = None,
        u_min: Optional[float] = None,
        u_max: Optional[float] = None,
        n_points: int = 100
    ):
        self.L = L
        self.f = f
        self.phi = phi
        self.t_span = t_span
        self.x0 = x0
        self.xf = xf
        self.u_min = u_min
        self.u_max = u_max
        self.n_points = n_points
        self.t = np.linspace(t_span[0], t_span[1], n_points)
        
    def cost_function(self, u: np.ndarray) -> float:
        dt = self.t[1] - self.t[0]
        x = self.integrate_state(u)
        integral_cost = np.trapz(self.L(self.t, x, u), dx=dt)
        
        if self.phi is not None:
            terminal_cost = self.phi(x[-1])
            return integral_cost + terminal_cost
        return integral_cost
    
    def integrate_state(self, u: np.ndarray) -> np.ndarray:
        x = np.zeros_like(self.t)
        x[0] = self.x0
        dt = self.t[1] - self.t[0]
        
        for i in range(1, len(self.t)):
            x[i] = x[i-1] + self.f(self.t[i-1], x[i-1], u[i-1]) * dt
            
        return x
    
    def solve_with_bounds(self, u_guess: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        if u_guess is None:
            u_guess = np.zeros(self.n_points)
        
        bounds = None
        if self.u_min is not None and self.u_max is not None:
            bounds = [(self.u_min, self.u_max) for _ in range(self.n_points)]
        
        constraints = []
        if self.xf is not None:
            constraints.append({'type': 'eq', 'fun': lambda u: self.integrate_state(u)[-1] - self.xf})
        
        result = minimize(
            self.cost_function,
            u_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'disp': True, 'ftol': 1e-8}
        )
        
        u_opt = result.x
        x_opt = self.integrate_state(u_opt)
        
        return x_opt, u_opt
    
    def solve_augmented_lagrangian(self, u_guess: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, float]:
        if u_guess is None:
            u_guess = np.zeros(self.n_points)
        
        def objective(u):
            return self.cost_function(u)
        
        g_list = []
        if self.u_min is not None:
            g_list.append(lambda u: self.u_min - u)
        if self.u_max is not None:
            g_list.append(lambda u: u - self.u_max)
        
        if self.xf is not None:
            g_list.append(lambda u: np.array([self.integrate_state(u)[-1] - self.xf]))
        
        if len(g_list) == 0:
            return self.solve_with_bounds(u_guess) + (0.0,)
        
        al_solver = AugmentedLagrangian(objective, g_list)
        u_opt, violation = al_solver.solve(u_guess)
        x_opt = self.integrate_state(u_opt)
        
        return x_opt, u_opt, violation


class ConstrainedIndirectSolver:
    """
    带不等式约束的间接法求解器（庞特里亚金最大值原理）
    哈密顿: H(t, x, u, λ) = L(t, x, u) + λ·f(t, x, u)
    约束: u_min ≤ u ≤ u_max
    最优控制: u* = argmin H (在可行域内)
    """
    def __init__(
        self,
        H: Callable[[float, float, float, float], float],
        dH_dx: Callable[[float, float, float, float], float],
        dH_dlam: Callable[[float, float, float, float], float],
        optimal_u_unconstrained: Callable[[float, float, float], float],
        t_span: Tuple[float, float],
        x0: float,
        xf: Optional[float] = None,
        dphi_dx: Optional[Callable[[float], float]] = None,
        u_min: Optional[float] = None,
        u_max: Optional[float] = None,
        n_points: int = 100
    ):
        self.H = H
        self.dH_dx = dH_dx
        self.dH_dlam = dH_dlam
        self.optimal_u_unconstrained = optimal_u_unconstrained
        self.t_span = t_span
        self.x0 = x0
        self.xf = xf
        self.dphi_dx = dphi_dx
        self.u_min = u_min
        self.u_max = u_max
        self.n_points = n_points
        self.t = np.linspace(t_span[0], t_span[1], n_points)
    
    def optimal_u_constrained(self, t: float, x: float, lam: float) -> float:
        u_unconstrained = self.optimal_u_unconstrained(t, x, lam)
        
        if self.u_min is None and self.u_max is None:
            return u_unconstrained
        
        u_opt = u_unconstrained
        if self.u_min is not None:
            u_opt = max(u_opt, self.u_min)
        if self.u_max is not None:
            u_opt = min(u_opt, self.u_max)
        
        return u_opt
    
    def state_costate_equations(self, t: float, y: np.ndarray) -> np.ndarray:
        x, lam = y
        u = self.optimal_u_constrained(t, x, lam)
        dx_dt = self.dH_dlam(t, x, u, lam)
        dlam_dt = -self.dH_dx(t, x, u, lam)
        return [dx_dt, dlam_dt]
    
    def shooting_function(self, lam0: float) -> float:
        y0 = [self.x0, lam0[0]]
        sol = solve_ivp(
            self.state_costate_equations,
            self.t_span,
            y0,
            t_eval=[self.t_span[1]],
            method='RK45',
            rtol=1e-8,
            atol=1e-8
        )
        x_final = sol.y[0, -1]
        lam_final = sol.y[1, -1]
        
        if self.xf is not None:
            return x_final - self.xf
        elif self.dphi_dx is not None:
            return lam_final - self.dphi_dx(x_final)
        else:
            return lam_final
    
    def solve(self, lam0_guess: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        result = minimize(
            lambda lam: self.shooting_function(lam)**2,
            lam0_guess,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'fatol': 1e-8}
        )
        
        lam0_opt = result.x[0]
        y0 = [self.x0, lam0_opt]
        
        sol = solve_ivp(
            self.state_costate_equations,
            self.t_span,
            y0,
            t_eval=self.t,
            method='RK45',
            rtol=1e-8,
            atol=1e-8
        )
        
        x_opt = sol.y[0, :]
        lam_opt = sol.y[1, :]
        u_opt = np.array([self.optimal_u_constrained(t_val, x_val, lam_val) 
                         for t_val, x_val, lam_val in zip(self.t, x_opt, lam_opt)])
        
        return self.t, x_opt, u_opt, lam_opt


def example_bang_bang_control():
    print("=" * 70)
    print("示例1: Bang-Bang控制 (最小时间问题)")
    print("问题: 最小化 J = T，约束: dx/dt = u, |u| ≤ 1")
    print("      x(0) = 0, x(T) = 1")
    print("解析解: u(t) = 1, T = 1")
    print("=" * 70)
    
    def L(t, x, u):
        return 1.0
    
    def f(t, x, u):
        return u
    
    t_span = (0, 1.0)
    x0 = 0.0
    xf = 1.0
    u_min = -1.0
    u_max = 1.0
    
    solver = ConstrainedEulerLagrangeSolver(
        L, f, t_span, x0, xf=xf,
        u_min=u_min, u_max=u_max, n_points=50
    )
    
    x_opt, u_opt, violation = solver.solve_augmented_lagrangian(
        u_guess=np.ones(50) * 0.5
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(solver.t, x_opt, 'b-', linewidth=2)
    axes[0].set_xlabel('t')
    axes[0].set_ylabel('x(t)')
    axes[0].set_title('状态轨迹')
    axes[0].grid(True)
    
    axes[1].plot(solver.t, u_opt, 'b-', linewidth=2, label='最优控制')
    axes[1].axhline(y=u_max, color='r', linestyle='--', label='u_max')
    axes[1].axhline(y=u_min, color='r', linestyle='--', label='u_min')
    axes[1].set_xlabel('t')
    axes[1].set_ylabel('u(t)')
    axes[1].set_title(f'Bang-Bang控制 (约束违反: {violation:.2e})')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('example_bang_bang.png', dpi=150, bbox_inches='tight')
    print(f"\n最终状态: x(1) = {x_opt[-1]:.6f}")
    print(f"控制量约束满足: min(u) = {u_opt.min():.3f}, max(u) = {u_opt.max():.3f}")
    print()


def example_minimum_energy_with_constraint():
    print("=" * 70)
    print("示例2: 带约束的最小能量控制")
    print("问题: J = ∫₀² (1/2)u² dt")
    print("约束: dx/dt = -x + u, |u| ≤ 0.5, x(0) = 1, x(2) = 0")
    print("=" * 70)
    
    def L(t, x, u):
        return 0.5 * u**2
    
    def f(t, x, u):
        return -x + u
    
    t_span = (0, 2.0)
    x0 = 1.0
    xf = 0.0
    u_min = -0.5
    u_max = 0.5
    
    solver = ConstrainedEulerLagrangeSolver(
        L, f, t_span, x0, xf=xf,
        u_min=u_min, u_max=u_max, n_points=100
    )
    
    x_opt, u_opt = solver.solve_with_bounds(
        u_guess=np.zeros(100)
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(solver.t, x_opt, 'b-', linewidth=2)
    axes[0].set_xlabel('t')
    axes[0].set_ylabel('x(t)')
    axes[0].set_title('状态轨迹')
    axes[0].grid(True)
    
    axes[1].plot(solver.t, u_opt, 'b-', linewidth=2, label='最优控制')
    axes[1].axhline(y=u_max, color='r', linestyle='--', label='u_max')
    axes[1].axhline(y=u_min, color='r', linestyle='--', label='u_min')
    axes[1].set_xlabel('t')
    axes[1].set_ylabel('u(t)')
    axes[1].set_title('带约束的最优控制')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('example_constrained_energy.png', dpi=150, bbox_inches='tight')
    print(f"\n最终状态: x(2) = {x_opt[-1]:.6f}")
    print(f"控制量: min = {u_opt.min():.3f}, max = {u_opt.max():.3f}")
    print()


def example_saturated_control():
    print("=" * 70)
    print("示例3: 饱和控制验证（间接法）")
    print("哈密顿: H = (1/2)u² + λ*(-x + u)")
    print("约束: |u| ≤ 0.3, x(0) = 1")
    print("横截条件: λ(T) = 0")
    print("=" * 70)
    
    def H(t, x, u, lam):
        return 0.5 * u**2 + lam * (-x + u)
    
    def dH_dx(t, x, u, lam):
        return -lam
    
    def dH_dlam(t, x, u, lam):
        return -x + u
    
    def optimal_u_unconstrained(t, x, lam):
        return -lam
    
    t_span = (0, 2.0)
    x0 = 1.0
    u_min = -0.3
    u_max = 0.3
    
    solver = ConstrainedIndirectSolver(
        H, dH_dx, dH_dlam, optimal_u_unconstrained,
        t_span, x0, u_min=u_min, u_max=u_max, n_points=100
    )
    
    t, x_opt, u_opt, lam_opt = solver.solve(lam0_guess=0.5)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(t, x_opt, 'b-', linewidth=2)
    axes[0].set_xlabel('t')
    axes[0].set_ylabel('x(t)')
    axes[0].set_title('状态轨迹')
    axes[0].grid(True)
    
    axes[1].plot(t, u_opt, 'b-', linewidth=2, label='最优控制')
    axes[1].axhline(y=u_max, color='r', linestyle='--', label='u_max')
    axes[1].axhline(y=u_min, color='r', linestyle='--', label='u_min')
    axes[1].set_xlabel('t')
    axes[1].set_ylabel('u(t)')
    axes[1].set_title('饱和控制')
    axes[1].legend()
    axes[1].grid(True)
    
    axes[2].plot(t, lam_opt, 'b-', linewidth=2)
    axes[2].set_xlabel('t')
    axes[2].set_ylabel('λ(t)')
    axes[2].set_title('协态变量')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('example_saturated_control.png', dpi=150, bbox_inches='tight')
    print(f"\n最终状态: x(2) = {x_opt[-1]:.6f}")
    print(f"最终协态: λ(2) = {lam_opt[-1]:.6f} (横截条件 λ(T)=0)")
    print(f"控制量饱和时间: {np.sum(np.abs(u_opt) >= 0.299) / len(u_opt) * 100:.1f}%")
    print()


def example_comparison_constrained_vs_unconstrained():
    print("=" * 70)
    print("示例4: 有约束 vs 无约束对比")
    print("问题: 最小能量控制，x(0)=0, x(1)=1")
    print("案例A: 无约束 (解析解 u=1)")
    print("案例B: 有约束 |u| ≤ 0.6")
    print("=" * 70)
    
    def L(t, x, u):
        return 0.5 * u**2
    
    def f(t, x, u):
        return u
    
    t_span = (0, 1.0)
    x0 = 0.0
    xf = 1.0
    n_points = 50
    
    solver_unconstrained = ConstrainedEulerLagrangeSolver(
        L, f, t_span, x0, xf=xf, n_points=n_points
    )
    x_unconstrained, u_unconstrained = solver_unconstrained.solve_with_bounds()
    
    u_min = -0.6
    u_max = 0.6
    solver_constrained = ConstrainedEulerLagrangeSolver(
        L, f, t_span, x0, xf=xf,
        u_min=u_min, u_max=u_max, n_points=n_points
    )
    x_constrained, u_constrained = solver_constrained.solve_with_bounds()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(t_span[0] + np.arange(n_points)/(n_points-1)*(t_span[1]-t_span[0]), 
                 x_unconstrained, 'b-', linewidth=2, label='无约束')
    axes[0].plot(t_span[0] + np.arange(n_points)/(n_points-1)*(t_span[1]-t_span[0]), 
                 x_constrained, 'r-', linewidth=2, label=f'约束 |u|≤{u_max}')
    axes[0].set_xlabel('t')
    axes[0].set_ylabel('x(t)')
    axes[0].set_title('状态轨迹对比')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(t_span[0] + np.arange(n_points)/(n_points-1)*(t_span[1]-t_span[0]), 
                 u_unconstrained, 'b-', linewidth=2, label='无约束')
    axes[1].plot(t_span[0] + np.arange(n_points)/(n_points-1)*(t_span[1]-t_span[0]), 
                 u_constrained, 'r-', linewidth=2, label=f'约束 |u|≤{u_max}')
    axes[1].axhline(y=u_max, color='g', linestyle='--', alpha=0.5)
    axes[1].axhline(y=u_min, color='g', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('t')
    axes[1].set_ylabel('u(t)')
    axes[1].set_title('控制量对比')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('example_constraint_comparison.png', dpi=150, bbox_inches='tight')
    
    energy_unconstrained = np.trapz(0.5 * u_unconstrained**2, dx=1/(n_points-1))
    energy_constrained = np.trapz(0.5 * u_constrained**2, dx=1/(n_points-1))
    
    print(f"\n无约束能量: {energy_unconstrained:.4f}")
    print(f"有约束能量: {energy_constrained:.4f}")
    print(f"能量增加: {(energy_constrained/energy_unconstrained - 1)*100:.1f}%")
    print(f"无约束最终状态: {x_unconstrained[-1]:.4f}")
    print(f"有约束最终状态: {x_constrained[-1]:.4f}")
    print()


if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    example_bang_bang_control()
    example_minimum_energy_with_constraint()
    example_saturated_control()
    example_comparison_constrained_vs_unconstrained()
    
    print("=" * 70)
    print("所有示例计算完成！")
    print("生成的图片文件:")
    print("  - example_bang_bang.png (Bang-Bang控制)")
    print("  - example_constrained_energy.png (带约束的最小能量)")
    print("  - example_saturated_control.png (饱和控制)")
    print("  - example_constraint_comparison.png (约束对比)")
    print("\n实现的约束处理方法:")
    print("1. 边界法 (SLSQP内置)")
    print("2. 增广拉格朗日方法 (Augmented Lagrangian)")
    print("3. 庞特里亚金最大值原理 (饱和控制)")
    print("=" * 70)

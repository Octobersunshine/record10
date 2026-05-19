import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable, Optional


class MomentClosure:
    def __init__(self, species_names: List[str], reaction_propensities: List[Callable],
                 stoichiometries: List[np.ndarray], closure_type: str = 'normal'):
        self.species_names = species_names
        self.num_species = len(species_names)
        self.reaction_propensities = reaction_propensities
        self.stoichiometries = stoichiometries
        self.num_reactions = len(reaction_propensities)
        self.closure_type = closure_type
        
        self.means_history = None
        self.covs_history = None
        self.times = None
    
    def _compute_moment_equations(self, t: float, y: np.ndarray) -> np.ndarray:
        num_species = self.num_species
        means = y[:num_species]
        cov_flat = y[num_species:]
        
        cov = np.zeros((num_species, num_species))
        idx = 0
        for i in range(num_species):
            for j in range(i, num_species):
                cov[i, j] = cov[idx]
                cov[j, i] = cov[idx]
                idx += 1
        
        dmeans_dt = np.zeros(num_species)
        dcov_dt = np.zeros((num_species, num_species))
        
        for r in range(self.num_reactions):
            prop = self.reaction_propensities[r](means, t)
            stoich = self.stoichiometries[r]
            
            mean_jac = self._propensity_jacobian(self.reaction_propensities[r], means, t)
            
            dprop_dt = np.dot(mean_jac, stoich) * prop
            
            dmeans_dt += stoich * prop
            
            for i in range(num_species):
                for j in range(num_species):
                    term1 = stoich[i] * stoich[j] * prop
                    term2 = stoich[i] * np.dot(cov[j, :], mean_jac)
                    term3 = stoich[j] * np.dot(cov[i, :], mean_jac)
                    dcov_dt[i, j] += term1 + term2 + term3
        
        if self.closure_type == 'normal':
            pass
        elif self.closure_type == 'log_normal':
            for i in range(num_species):
                for j in range(num_species):
                    cov_ij = cov[i, j]
                    m_i = means[i]
                    m_j = means[j]
                    if m_i > 0 and m_j > 0:
                        correction = cov_ij * (cov_ij / (m_i * m_j))
                        dcov_dt[i, j] += 0.1 * correction
        elif self.closure_type == 'derivative_matching':
            pass
        
        dcov_flat_dt = []
        for i in range(num_species):
            for j in range(i, num_species):
                dcov_flat_dt.append(dcov_dt[i, j])
        
        return np.concatenate([dmeans_dt, np.array(dcov_flat_dt)])
    
    def _propensity_jacobian(self, propensity: Callable, means: np.ndarray, t: float) -> np.ndarray:
        eps = 1e-6
        jac = np.zeros(self.num_species)
        for i in range(self.num_species):
            means_plus = means.copy()
            means_plus[i] += eps
            means_minus = means.copy()
            means_minus[i] -= eps
            jac[i] = (propensity(means_plus, t) - propensity(means_minus, t)) / (2 * eps)
        return jac
    
    def simulate(self, initial_means: np.ndarray, initial_cov: np.ndarray,
                 t_span: Tuple[float, float], t_eval: Optional[np.ndarray] = None,
                 method: str = 'RK45') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        num_species = self.num_species
        
        initial_cov_flat = []
        for i in range(num_species):
            for j in range(i, num_species):
                initial_cov_flat.append(initial_cov[i, j])
        
        y0 = np.concatenate([initial_means, np.array(initial_cov_flat)])
        
        solution = solve_ivp(self._compute_moment_equations, t_span, y0,
                            method=method, t_eval=t_eval, rtol=1e-6, atol=1e-9)
        
        self.times = solution.t
        self.means_history = solution.y[:num_species, :].T
        
        self.covs_history = np.zeros((len(solution.t), num_species, num_species))
        for t_idx in range(len(solution.t)):
            cov_flat = solution.y[num_species:, t_idx]
            idx = 0
            for i in range(num_species):
                for j in range(i, num_species):
                    self.covs_history[t_idx, i, j] = cov_flat[idx]
                    self.covs_history[t_idx, j, i] = cov_flat[idx]
                    idx += 1
        
        return self.times, self.means_history, self.covs_history
    
    def get_statistics(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.means_history is None or self.covs_history is None:
            raise ValueError("Run simulate first!")
        
        variances = np.array([np.diag(cov) for cov in self.covs_history])
        stds = np.sqrt(variances)
        
        return self.means_history, stds
    
    def plot_results(self, figsize: Tuple[int, int] = (15, 5)):
        if self.means_history is None:
            raise ValueError("Run simulate first!")
        
        means, stds = self.get_statistics()
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        for i, name in enumerate(self.species_names):
            axes[0].plot(self.times, means[:, i], label=f'{name} mean', linewidth=2)
            axes[0].fill_between(self.times,
                                means[:, i] - 2 * stds[:, i],
                                means[:, i] + 2 * stds[:, i],
                                alpha=0.2)
        
        axes[0].set_xlabel('Time', fontsize=12)
        axes[0].set_ylabel('Population', fontsize=12)
        axes[0].set_title(f'Means ± 2σ ({self.closure_type} closure)', fontsize=14)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        for i, name in enumerate(self.species_names):
            axes[1].plot(self.times, stds[:, i] ** 2, label=f'{name} variance', linewidth=2)
        
        axes[1].set_xlabel('Time', fontsize=12)
        axes[1].set_ylabel('Variance', fontsize=12)
        axes[1].set_title('Variances over time', fontsize=14)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'moment_closure_{self.closure_type}.png', dpi=150, bbox_inches='tight')
        print(f"Plot saved to 'moment_closure_{self.closure_type}.png'")


class MomentEstimator:
    def __init__(self, species_names: List[str], reaction_propensities_template: Callable,
                 stoichiometries: List[np.ndarray], param_names: List[str]):
        self.species_names = species_names
        self.reaction_propensities_template = reaction_propensities_template
        self.stoichiometries = stoichiometries
        self.param_names = param_names
        self.num_params = len(param_names)
        
        self.estimated_params = None
        self.optimization_history = []
    
    def _compute_objective(self, params: np.ndarray, target_times: np.ndarray,
                           target_means: np.ndarray, target_covs: Optional[np.ndarray] = None,
                           weights: Optional[np.ndarray] = None) -> float:
        propensities = self.reaction_propensities_template(params)
        
        mc = MomentClosure(self.species_names, propensities, self.stoichiometries,
                          closure_type='normal')
        
        initial_means = target_means[0].copy()
        num_species = len(self.species_names)
        initial_cov = np.eye(num_species) * 0.1
        
        try:
            times, means_history, covs_history = mc.simulate(
                initial_means, initial_cov,
                t_span=(target_times[0], target_times[-1]),
                t_eval=target_times
            )
            
            if weights is None:
                weights = np.ones_like(target_means)
            
            mean_error = np.sum(weights * (means_history - target_means) ** 2)
            
            cov_error = 0.0
            if target_covs is not None:
                for t_idx in range(len(target_times)):
                    cov_error += np.sum((covs_history[t_idx] - target_covs[t_idx]) ** 2)
            
            return mean_error + 0.1 * cov_error
            
        except Exception as e:
            return 1e10
    
    def estimate(self, initial_params: np.ndarray, target_times: np.ndarray,
                 target_means: np.ndarray, target_covs: Optional[np.ndarray] = None,
                 weights: Optional[np.ndarray] = None,
                 bounds: Optional[List[Tuple]] = None,
                 method: str = 'L-BFGS-B') -> np.ndarray:
        
        def objective(params):
            val = self._compute_objective(params, target_times, target_means, target_covs, weights)
            self.optimization_history.append((params.copy(), val))
            return val
        
        result = minimize(objective, initial_params, method=method,
                         bounds=bounds, options={'maxiter': 100, 'disp': True})
        
        self.estimated_params = result.x
        
        return result.x
    
    def get_simulation_with_params(self, params: np.ndarray, initial_means: np.ndarray,
                                   t_span: Tuple[float, float],
                                   t_eval: Optional[np.ndarray] = None):
        propensities = self.reaction_propensities_template(params)
        mc = MomentClosure(self.species_names, propensities, self.stoichiometries,
                          closure_type='normal')
        
        num_species = len(self.species_names)
        initial_cov = np.eye(num_species) * 0.1
        
        return mc.simulate(initial_means, initial_cov, t_span, t_eval)
    
    def plot_estimation_results(self, target_times: np.ndarray, target_means: np.ndarray,
                               figsize: Tuple[int, int] = (12, 6)):
        if self.estimated_params is None:
            raise ValueError("Run estimate first!")
        
        initial_means = target_means[0].copy()
        times, means_history, covs_history = self.get_simulation_with_params(
            self.estimated_params, initial_means,
            t_span=(target_times[0], target_times[-1]),
            t_eval=target_times
        )
        
        stds = np.sqrt(np.array([np.diag(cov) for cov in covs_history]))
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        for i, name in enumerate(self.species_names):
            axes[0].plot(times, means_history[:, i], '--', label=f'{name} fitted', linewidth=2)
            axes[0].plot(target_times, target_means[:, i], 'o', label=f'{name} data', markersize=4)
        
        axes[0].set_xlabel('Time', fontsize=12)
        axes[0].set_ylabel('Population', fontsize=12)
        axes[0].set_title('Fit to Data', fontsize=14)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        history_vals = [h[1] for h in self.optimization_history]
        axes[1].plot(history_vals, 'b-', linewidth=2)
        axes[1].set_xlabel('Iteration', fontsize=12)
        axes[1].set_ylabel('Objective Value', fontsize=12)
        axes[1].set_title('Optimization Progress', fontsize=14)
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('parameter_estimation.png', dpi=150, bbox_inches='tight')
        print("Estimation plot saved to 'parameter_estimation.png'")
        
        print("\nEstimated Parameters:")
        for name, val in zip(self.param_names, self.estimated_params):
            print(f"  {name}: {val:.4f}")


def create_gene_expression_example():
    species_names = ['mRNA', 'Protein']
    
    def propensities_template(params):
        k0, k1, k2, k3 = params
        
        prop0 = lambda m, t: k0 * np.ones_like(m[0])
        prop1 = lambda m, t: k1 * m[0]
        prop2 = lambda m, t: k2 * m[0]
        prop3 = lambda m, t: k3 * m[1]
        
        return [prop0, prop1, prop2, prop3]
    
    stoichiometries = [
        np.array([1, 0]),
        np.array([-1, 0]),
        np.array([0, 1]),
        np.array([0, -1]),
    ]
    
    param_names = ['k0 (transcription)', 'k1 (mRNA deg)', 'k2 (translation)', 'k3 (protein deg)']
    
    return species_names, propensities_template, stoichiometries, param_names


def demo_moment_closure():
    print("=" * 70)
    print("Moment Closure Approximation Demo")
    print("=" * 70)
    
    species_names = ['A', 'B']
    
    propensities = [
        lambda m, t: 0.5 * m[0],
        lambda m, t: 0.1 * m[1],
    ]
    
    stoichiometries = [
        np.array([-1, 1]),
        np.array([0, -1]),
    ]
    
    for closure in ['normal', 'log_normal', 'derivative_matching']:
        print(f"\n{closure.upper()} CLOSURE:")
        print("-" * 50)
        
        mc = MomentClosure(species_names, propensities, stoichiometries, closure_type=closure)
        
        initial_means = np.array([100.0, 0.0])
        initial_cov = np.array([[10.0, 0.0], [0.0, 0.0]])
        
        t_span = (0.0, 50.0)
        t_eval = np.linspace(0, 50, 100)
        
        times, means, covs = mc.simulate(initial_means, initial_cov, t_span, t_eval)
        
        print(f"Final means: A={means[-1, 0]:.2f}, B={means[-1, 1]:.2f}")
        var = np.diag(covs[-1])
        print(f"Final variances: A={var[0]:.2f}, B={var[1]:.2f}")
        
        mc.plot_results()
    
    return mc


def demo_parameter_estimation():
    print("\n" + "=" * 70)
    print("Parameter Estimation using Moment Closure")
    print("=" * 70)
    
    species_names, prop_template, stoichiometries, param_names = create_gene_expression_example()
    
    true_params = np.array([5.0, 0.3, 2.0, 0.1])
    print(f"\nTrue parameters:")
    for name, val in zip(param_names, true_params):
        print(f"  {name}: {val:.4f}")
    
    true_propensities = prop_template(true_params)
    mc_true = MomentClosure(species_names, true_propensities, stoichiometries)
    initial_means = np.array([0.0, 0.0])
    initial_cov = np.eye(2) * 0.1
    target_times = np.linspace(0, 50, 20)
    _, target_means, target_covs = mc_true.simulate(initial_means, initial_cov,
                                                     (0, 50), target_times)
    
    noise_level = 0.05
    target_means_noisy = target_means * (1 + noise_level * np.random.randn(*target_means.shape))
    
    estimator = MomentEstimator(species_names, prop_template, stoichiometries, param_names)
    
    initial_params = np.array([3.0, 0.5, 1.0, 0.2])
    bounds = [(0.1, 20.0), (0.01, 2.0), (0.1, 10.0), (0.01, 1.0)]
    
    print("\nEstimating parameters...")
    estimated = estimator.estimate(initial_params, target_times, target_means_noisy,
                                   bounds=bounds)
    
    print("\nParameter Recovery:")
    for name, true, est in zip(param_names, true_params, estimated):
        rel_error = abs(est - true) / true * 100
        print(f"  {name}: true={true:.4f}, est={est:.4f}, error={rel_error:.1f}%")
    
    estimator.plot_estimation_results(target_times, target_means_noisy)
    
    return estimator


def explain_moment_closure():
    print("\n" + "=" * 70)
    print("Moment Closure: Theory and Methods")
    print("=" * 70)
    
    explanation = """
## 1. 矩方法 (Moment Equations)

对于化学反应系统，种群均值的演化方程：

    d⟨X_i⟩/dt = Σ_j ν_ij ⟨a_j(X)⟩

其中：
- ν_ij 是第j个反应对第i个物种的化学计量
- a_j(X) 是第j个反应的倾向函数
- ⟨·⟩ 表示期望

协方差的演化方程：

    dCov(X_i,X_j)/dt = Σ_k ν_ik ν_jk ⟨a_k(X)⟩
                       + Σ_k ν_ik Cov(X_j, a_k(X))
                       + Σ_k ν_jk Cov(X_i, a_k(X))

## 2. 封闭问题 (Closure Problem)

上述方程不封闭：高阶矩（如协方差）依赖于更高阶的矩。

矩封闭近似通过假设分布形式来"封闭"方程系统。

## 3. 常见封闭方法

### 正态近似 (Normal Closure)
假设种群服从多元正态分布：
- 所有三阶及更高阶累积量为零
- 最简单、最常用的方法

### 对数正态近似 (Log-Normal Closure)
假设种群服从对数正态分布：
- 适合非负种群
- 处理低拷贝数更好

### 导数匹配 (Derivative Matching)
通过匹配分布的导数来确定高阶矩。

## 4. 参数估计

利用矩封闭快速计算矩，然后拟合到数据：

    min_θ Σ_t ||μ(t;θ) - μ_data(t)||² + λ Σ_t ||Σ(t;θ) - Σ_data(t)||²

其中：
- μ(t;θ) 是参数θ下t时刻的均值
- Σ(t;θ) 是协方差矩阵
- λ 是方差匹配的权重

## 5. 优势

1. **速度**：比SSA快几个数量级（ODE vs 随机模拟）
2. **解析梯度**：可以使用基于梯度的优化
3. **不确定性量化**：同时得到均值和方差
4. **参数灵敏度**：容易计算参数对矩的影响
    """
    
    print(explanation)


if __name__ == "__main__":
    demo_moment_closure()
    demo_parameter_estimation()
    explain_moment_closure()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
import time

def compute_loocv_mse(alpha_log, X, y):
    """计算指定log(alpha)下的LOOCV MSE（目标函数，用于最小化）"""
    alpha = np.exp(alpha_log)
    n_samples, n_features = X.shape
    
    ridge = Ridge(alpha=alpha)
    ridge.fit(X, y)
    y_pred = ridge.predict(X)
    
    H = X @ np.linalg.inv(X.T @ X + alpha * np.eye(n_features)) @ X.T
    h_diag = np.diag(H)
    residuals = y - y_pred
    
    loocv_error = np.mean((residuals / (1 - h_diag)) ** 2)
    return loocv_error

def golden_section_search(f, a, b, tol=1e-6, max_iter=50):
    """黄金分割搜索：寻找单峰函数的最小值
    f: 目标函数
    a, b: 搜索区间 [a, b]
    """
    gr = (np.sqrt(5) - 1) / 2
    
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = f(c)
    fd = f(d)
    
    history = [(a, b, c, d, fc, fd)]
    
    for i in range(max_iter):
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = f(d)
        
        history.append((a, b, c, d, fc, fd))
        
        if abs(b - a) < tol:
            break
    
    x_opt = (a + b) / 2
    return x_opt, history

def expected_improvement(x, X_sample, y_sample, gpr, xi=0.01):
    """期望改进采集函数"""
    mu, sigma = gpr.predict(x.reshape(-1, 1), return_std=True)
    mu_sample = gpr.predict(X_sample.reshape(-1, 1))
    
    sigma = sigma.reshape(-1, 1)
    mu_sample_opt = np.min(mu_sample)
    
    with np.errstate(divide='warn'):
        imp = mu_sample_opt - mu - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
    
    return ei[0]

class SimpleGPR:
    """简单的高斯过程回归实现"""
    def __init__(self, length_scale=1.0, noise=1e-6):
        self.length_scale = length_scale
        self.noise = noise
    
    def rbf_kernel(self, x1, x2):
        return np.exp(-0.5 * ((x1 - x2.T) / self.length_scale) ** 2)
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.K = self.rbf_kernel(X, X) + self.noise * np.eye(len(X))
        self.K_inv = np.linalg.inv(self.K)
    
    def predict(self, X, return_std=False):
        K_s = self.rbf_kernel(self.X_train, X)
        mu = K_s.T @ self.K_inv @ self.y_train
        
        if return_std:
            K_ss = self.rbf_kernel(X, X) + self.noise * np.eye(len(X))
            cov = K_ss - K_s.T @ self.K_inv @ K_s
            std = np.sqrt(np.maximum(np.diag(cov), 0))
            return mu.flatten(), std
        return mu.flatten()

def bayesian_optimization(f, bounds, n_init=3, n_iter=20):
    """贝叶斯优化"""
    a, b = bounds
    X_sample = np.random.uniform(a, b, n_init).reshape(-1, 1)
    y_sample = np.array([f(x[0]) for x in X_sample])
    
    history = []
    
    for i in range(n_iter):
        gpr = SimpleGPR(length_scale=0.5)
        gpr.fit(X_sample, y_sample)
        
        candidate_x = np.linspace(a, b, 1000).reshape(-1, 1)
        ei_values = [expected_improvement(x[0], X_sample, y_sample, gpr) for x in candidate_x]
        
        x_next = candidate_x[np.argmax(ei_values)]
        y_next = f(x_next[0])
        
        history.append({
            'iteration': i,
            'X_sample': X_sample.copy(),
            'y_sample': y_sample.copy(),
            'x_next': x_next[0],
            'y_next': y_next,
            'ei_values': np.array(ei_values),
            'candidate_x': candidate_x.flatten()
        })
        
        X_sample = np.vstack([X_sample, x_next])
        y_sample = np.append(y_sample, y_next)
    
    best_idx = np.argmin(y_sample)
    x_opt = X_sample[best_idx][0]
    return x_opt, X_sample, y_sample, history

print("=" * 80)
print("岭回归 - 自适应λ搜索：黄金分割 vs 贝叶斯优化 vs 网格搜索")
print("=" * 80)

np.random.seed(42)
X, y, true_coef = make_regression(
    n_samples=200, n_features=15, n_informative=8,
    noise=15, coef=True, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

n_train = X_train_scaled.shape[0]
n_features = X_train_scaled.shape[1]

def objective(alpha_log):
    return compute_loocv_mse(alpha_log, X_train_scaled, y_train)

search_bounds = [np.log(1e-4), np.log(1e4)]

print("\n" + "=" * 80)
print("方法1: 网格搜索 (100个点)")
print("=" * 80)
start_time = time.time()
alphas_grid = np.logspace(-4, 4, 100)
loocv_grid = []
for alpha in alphas_grid:
    loocv_grid.append(compute_loocv_mse(np.log(alpha), X_train_scaled, y_train))
loocv_grid = np.array(loocv_grid)
best_alpha_grid = alphas_grid[np.argmin(loocv_grid)]
grid_time = time.time() - start_time
grid_evals = len(alphas_grid)
print(f"最佳 λ: {best_alpha_grid:.6f}")
print(f"最小 LOOCV MSE: {np.min(loocv_grid):.4f}")
print(f"计算次数: {grid_evals}")
print(f"耗时: {grid_time:.4f}秒")

print("\n" + "=" * 80)
print("方法2: 黄金分割搜索")
print("=" * 80)
start_time = time.time()
best_log_alpha_gss, gss_history = golden_section_search(
    objective, search_bounds[0], search_bounds[1], tol=1e-8, max_iter=50
)
best_alpha_gss = np.exp(best_log_alpha_gss)
gss_time = time.time() - start_time
gss_evals = len(gss_history) + 2
print(f"最佳 λ: {best_alpha_gss:.6f}")
print(f"最小 LOOCV MSE: {objective(best_log_alpha_gss):.4f}")
print(f"计算次数: {gss_evals}")
print(f"耗时: {gss_time:.4f}秒")
print(f"收敛迭代: {len(gss_history)}")

print("\n" + "=" * 80)
print("方法3: 贝叶斯优化")
print("=" * 80)
start_time = time.time()
best_log_alpha_bo, X_bo, y_bo, bo_history = bayesian_optimization(
    objective, search_bounds, n_init=5, n_iter=15
)
best_alpha_bo = np.exp(best_log_alpha_bo)
bo_time = time.time() - start_time
bo_evals = len(X_bo)
print(f"最佳 λ: {best_alpha_bo:.6f}")
print(f"最小 LOOCV MSE: {objective(best_log_alpha_bo):.4f}")
print(f"计算次数: {bo_evals}")
print(f"耗时: {bo_time:.4f}秒")
print(f"初始采样: 5, 迭代优化: 15")

print("\n" + "=" * 80)
print("方法对比总结")
print("=" * 80)
print(f"{'方法':<15} {'最佳λ':<12} {'LOOCV MSE':<12} {'评估次数':<10} {'耗时(秒)':<10}")
print("-" * 65)
print(f"{'网格搜索':<15} {best_alpha_grid:<12.6f} {np.min(loocv_grid):<12.4f} {grid_evals:<10} {grid_time:<10.4f}")
print(f"{'黄金分割':<15} {best_alpha_gss:<12.6f} {objective(best_log_alpha_gss):<12.4f} {gss_evals:<10} {gss_time:<10.4f}")
print(f"{'贝叶斯优化':<15} {best_alpha_bo:<12.6f} {objective(best_log_alpha_bo):<12.4f} {bo_evals:<10} {bo_time:<10.4f}")

print(f"\n黄金分割比网格搜索快 {(grid_evals/gss_evals):.1f}x")
print(f"贝叶斯优化比网格搜索快 {(grid_evals/bo_evals):.1f}x")

alphas_plot = np.logspace(-4, 4, 200)
loocv_plot = [compute_loocv_mse(np.log(a), X_train_scaled, y_train) for a in alphas_plot]

fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(3, 3)

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(alphas_plot, loocv_plot, 'k-', linewidth=2, alpha=0.7, label='真实目标函数')
ax1.scatter(alphas_grid, loocv_grid, c='blue', s=30, alpha=0.6, label=f'网格搜索 ({grid_evals}次)')
ax1.scatter(best_alpha_grid, np.min(loocv_grid), c='blue', s=200, marker='*', edgecolors='black', zorder=10)
ax1.set_xscale('log')
ax1.set_xlabel('λ', fontsize=12)
ax1.set_ylabel('LOOCV MSE', fontsize=12)
ax1.set_title('网格搜索', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(alphas_plot, loocv_plot, 'k-', linewidth=2, alpha=0.7, label='真实目标函数')
gss_x = [np.exp((h[0]+h[1])/2) for h in gss_history]
gss_y = [objective((h[0]+h[1])/2) for h in gss_history]
ax2.scatter(gss_x, gss_y, c='red', s=50, alpha=0.7, label=f'黄金分割 ({gss_evals}次)')
ax2.plot(gss_x, gss_y, 'r--', linewidth=1.5, alpha=0.5)
ax2.scatter(best_alpha_gss, objective(best_log_alpha_gss), c='red', s=200, marker='*', edgecolors='black', zorder=10)
ax2.set_xscale('log')
ax2.set_xlabel('λ', fontsize=12)
ax2.set_ylabel('LOOCV MSE', fontsize=12)
ax2.set_title('黄金分割搜索', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(alphas_plot, loocv_plot, 'k-', linewidth=2, alpha=0.7, label='真实目标函数')
bo_alphas = np.exp(X_bo.flatten())
ax3.scatter(bo_alphas[:5], y_bo[:5], c='orange', s=60, marker='s', label='初始采样 (5次)', zorder=5)
ax3.scatter(bo_alphas[5:], y_bo[5:], c='green', s=60, marker='^', label='BO迭代 (15次)', zorder=5)
ax3.scatter(best_alpha_bo, objective(best_log_alpha_bo), c='green', s=200, marker='*', edgecolors='black', zorder=10)
ax3.set_xscale('log')
ax3.set_xlabel('λ', fontsize=12)
ax3.set_ylabel('LOOCV MSE', fontsize=12)
ax3.set_title('贝叶斯优化', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

ax4 = fig.add_subplot(gs[1, 0])
iterations = np.arange(1, len(bo_history) + 1)
best_so_far = [np.min(bo_hist['y_sample']) for bo_hist in bo_history]
ax4.plot(iterations, best_so_far, 'go-', linewidth=2, markersize=8)
ax4.axhline(y=np.min(loocv_grid), color='blue', linestyle='--', label='网格搜索最优')
ax4.set_xlabel('BO迭代次数', fontsize=12)
ax4.set_ylabel('当前最佳 LOOCV MSE', fontsize=12)
ax4.set_title('贝叶斯优化收敛曲线', fontsize=14, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

ax5 = fig.add_subplot(gs[1, 1])
if len(bo_history) >= 5:
    bo_vis = bo_history[4]
    gpr = SimpleGPR(length_scale=0.5)
    gpr.fit(bo_vis['X_sample'], bo_vis['y_sample'])
    x_candidates = np.linspace(search_bounds[0], search_bounds[1], 200)
    mu, std = gpr.predict(x_candidates.reshape(-1, 1), return_std=True)
    
    ax5.fill_between(np.exp(x_candidates), mu - 2*std, mu + 2*std, alpha=0.3, color='purple')
    ax5.plot(np.exp(x_candidates), mu, 'purple', linewidth=2, label='GP均值')
    ax5.scatter(np.exp(bo_vis['X_sample'].flatten()), bo_vis['y_sample'], c='black', s=50, zorder=5, label='已评估点')
    ax5.scatter(np.exp(bo_vis['x_next']), bo_vis['y_next'], c='red', s=150, marker='*', label='下一个采样点', zorder=10)
    ax5.plot(np.exp(x_candidates), [compute_loocv_mse(x, X_train_scaled, y_train) for x in x_candidates], 'k--', alpha=0.5, label='真实函数')
    ax5.set_xscale('log')
    ax5.set_xlabel('λ', fontsize=12)
    ax5.set_ylabel('LOOCV MSE', fontsize=12)
    ax5.set_title('贝叶斯优化：GP拟合与不确定性', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

ax6 = fig.add_subplot(gs[1, 2])
if len(bo_history) >= 5:
    bo_vis = bo_history[4]
    ax6.plot(np.exp(bo_vis['candidate_x']), bo_vis['ei_values'], 'green', linewidth=2)
    ax6.fill_between(np.exp(bo_vis['candidate_x']), 0, bo_vis['ei_values'], alpha=0.3, color='green')
    ax6.axvline(x=np.exp(bo_vis['x_next']), color='red', linestyle='--', linewidth=2, label='EI最大点')
    ax6.set_xscale('log')
    ax6.set_xlabel('λ', fontsize=12)
    ax6.set_ylabel('期望改进 (EI)', fontsize=12)
    ax6.set_title('贝叶斯优化：采集函数', fontsize=14, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)

ax7 = fig.add_subplot(gs[2, :])
methods = ['网格搜索', '黄金分割', '贝叶斯优化']
evals = [grid_evals, gss_evals, bo_evals]
colors = ['blue', 'red', 'green']
bars = ax7.bar(methods, evals, color=colors, alpha=0.7, edgecolor='black')
for bar, val in zip(bars, evals):
    height = bar.get_height()
    ax7.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{val}次', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax7.set_ylabel('函数评估次数', fontsize=12)
ax7.set_title('搜索效率对比：评估次数越少越好', fontsize=14, fontweight='bold')
ax7.grid(True, alpha=0.3, axis='y')
for i, (method, ev) in enumerate(zip(methods, evals)):
    speedup = grid_evals / ev
    ax7.text(i, ev/2, f'{speedup:.0f}x\n加速', ha='center', va='center', fontsize=11, fontweight='bold', color='white')

plt.tight_layout()
plt.savefig('ridge_adaptive_search.png', dpi=150, bbox_inches='tight')
print("\n图表已保存为: ridge_adaptive_search.png")

print("\n" + "=" * 80)
print("最终模型验证（测试集性能）")
print("=" * 80)
for name, alpha in zip(['网格搜索', '黄金分割', '贝叶斯优化'], 
                        [best_alpha_grid, best_alpha_gss, best_alpha_bo]):
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    y_pred = ridge.predict(X_test_scaled)
    test_mse = mean_squared_error(y_test, y_pred)
    print(f"{name}选择的λ={alpha:.6f}，测试集MSE: {test_mse:.4f}")

print("\n" + "=" * 80)
print("方法特点总结:")
print("1. 网格搜索: 简单可靠，但效率低，精度受网格密度限制")
print("2. 黄金分割: 高效（O(log n)收敛），但要求目标函数单峰")
print("3. 贝叶斯优化: 智能采样，兼顾探索-利用，适合昂贵的黑盒函数")
print("\n推荐: 如果确定函数单峰，用黄金分割（最快）；否则用贝叶斯优化。")
print("=" * 80)

plt.show()
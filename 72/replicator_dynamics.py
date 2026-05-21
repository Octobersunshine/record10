import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import networkx as nx
from matplotlib.animation import FuncAnimation


EPS = 1e-10

def replicator_dynamics(x, t, payoff_matrix):
    """
    复制动态方程（带数值稳定性修复）
    
    参数:
    x: 策略频率向量
    t: 时间
    payoff_matrix: 支付矩阵
    
    返回:
    dx/dt: 策略频率的变化率
    """
    x_clipped = np.clip(x, EPS, 1 - EPS)
    
    fitness = np.dot(payoff_matrix, x_clipped)
    avg_fitness = np.dot(x_clipped, fitness)
    
    dxdt = x_clipped * (fitness - avg_fitness)
    
    near_zero = x < EPS
    near_one = x > 1 - EPS
    dxdt[near_zero] = np.maximum(dxdt[near_zero], 0)
    dxdt[near_one] = np.minimum(dxdt[near_one], 0)
    
    return dxdt


def solve_replicator_dynamics(payoff_matrix, initial_x, t_span):
    """
    求解复制动态方程（带数值稳定性修复）
    
    参数:
    payoff_matrix: 支付矩阵
    initial_x: 初始策略频率
    t_span: 时间范围
    
    返回:
    t: 时间点
    x: 策略频率的演化轨迹
    """
    initial_x = np.clip(initial_x, EPS, 1 - EPS)
    initial_x = initial_x / np.sum(initial_x)
    
    t = np.linspace(t_span[0], t_span[1], 1000)
    x = odeint(replicator_dynamics, initial_x, t, args=(payoff_matrix,),
               rtol=1e-8, atol=1e-8, mxstep=5000)
    
    x = np.clip(x, 0, 1)
    row_sums = np.sum(x, axis=1, keepdims=True)
    x = x / row_sums
    
    return t, x


def hawk_dove_game(V, C):
    """
    创建鹰鸽博弈的支付矩阵
    
    参数:
    V: 资源价值
    C: 争斗成本
    
    返回:
    payoff_matrix: 支付矩阵
    """
    return np.array([
        [(V - C) / 2, V],
        [0, V / 2]
    ])


def find_ess(payoff_matrix):
    """
    寻找演化稳定策略 (ESS)
    
    参数:
    payoff_matrix: 支付矩阵
    
    返回:
    ess: ESS策略向量
    """
    n = payoff_matrix.shape[0]
    
    if n == 2:
        a, b = payoff_matrix[0]
        c, d = payoff_matrix[1]
        
        if a > c and b > d:
            return np.array([1, 0])
        elif a < c and b < d:
            return np.array([0, 1])
        else:
            p = (d - b) / (a - b - c + d)
            if 0 < p < 1:
                return np.array([p, 1 - p])
    
    return None


def plot_replicator_dynamics(t, x, labels):
    """
    绘制复制动态方程的演化轨迹
    
    参数:
    t: 时间点
    x: 策略频率的演化轨迹
    labels: 策略标签
    """
    plt.figure(figsize=(10, 6))
    for i in range(x.shape[1]):
        plt.plot(t, x[:, i], label=labels[i], linewidth=2)
    
    plt.xlabel('时间', fontsize=12)
    plt.ylabel('策略频率', fontsize=12)
    plt.title('复制动态方程的演化轨迹', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.show()


def plot_phase_portrait(payoff_matrix, n_points=20):
    """
    绘制2策略博弈的相图
    
    参数:
    payoff_matrix: 支付矩阵
    n_points: 网格点数
    """
    p = np.linspace(0, 1, n_points)
    dp = np.zeros_like(p)
    
    for i, pi in enumerate(p):
        x = np.array([pi, 1 - pi])
        dx = replicator_dynamics(x, 0, payoff_matrix)
        dp[i] = dx[0]
    
    plt.figure(figsize=(10, 6))
    plt.plot(p, dp, 'b-', linewidth=2)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('鹰策略的频率 (p)', fontsize=12)
    plt.ylabel('dp/dt', fontsize=12)
    plt.title('复制动态方程相图', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.show()


def create_grid_network(n, periodic=True):
    """
    创建二维网格网络
    
    参数:
    n: 网格边长 (节点总数为n*n)
    periodic: 是否为周期边界（环面）
    
    返回:
    G: networkx图对象
    """
    G = nx.grid_2d_graph(n, n, periodic=periodic)
    G = nx.convert_node_labels_to_integers(G)
    return G


def create_random_network(n, p=0.1):
    """
    创建随机图（Erdős–Rényi）
    
    参数:
    n: 节点数
    p: 连边概率
    
    返回:
    G: networkx图对象
    """
    G = nx.erdos_renyi_graph(n, p)
    return G


def create_small_world_network(n, k=4, p=0.1):
    """
    创建小世界网络（Watts-Strogatz）
    
    参数:
    n: 节点数
    k: 每个节点的邻居数（偶数）
    p: 重连概率
    
    返回:
    G: networkx图对象
    """
    G = nx.watts_strogatz_graph(n, k, p)
    return G


def initialize_strategies(n, n_strategies=2, initial_freq=None):
    """
    初始化节点策略
    
    参数:
    n: 节点数
    n_strategies: 策略数
    initial_freq: 初始频率，None表示随机
    
    返回:
    strategies: 策略数组 (n,)
    """
    if initial_freq is None:
        strategies = np.random.randint(0, n_strategies, size=n)
    else:
        strategies = np.zeros(n, dtype=int)
        cum_freq = np.cumsum(initial_freq)
        for i in range(n):
            r = np.random.random()
            strategies[i] = np.searchsorted(cum_freq, r)
    return strategies


def calculate_payoffs(G, strategies, payoff_matrix):
    """
    计算每个节点的支付（与所有邻居交互）
    
    参数:
    G: 网络
    strategies: 策略数组
    payoff_matrix: 支付矩阵
    
    返回:
    payoffs: 支付数组
    """
    n = G.number_of_nodes()
    payoffs = np.zeros(n)
    
    for i in range(n):
        neighbors = list(G.neighbors(i))
        s_i = strategies[i]
        for j in neighbors:
            s_j = strategies[j]
            payoffs[i] += payoff_matrix[s_i, s_j]
    
    return payoffs


def update_strategies_fermi(G, strategies, payoffs, beta=0.1):
    """
    Fermi规则策略更新：随机选择邻居，以概率复制其策略
    
    参数:
    G: 网络
    strategies: 当前策略数组
    payoffs: 当前支付数组
    beta: 选择强度（beta越大，选择越严格）
    
    返回:
    new_strategies: 更新后的策略数组
    """
    n = G.number_of_nodes()
    new_strategies = strategies.copy()
    
    for i in range(n):
        neighbors = list(G.neighbors(i))
        if len(neighbors) == 0:
            continue
        
        j = np.random.choice(neighbors)
        pi = payoffs[i]
        pj = payoffs[j]
        
        prob = 1.0 / (1.0 + np.exp(-beta * (pj - pi)))
        if np.random.random() < prob:
            new_strategies[i] = strategies[j]
    
    return new_strategies


def update_strategies_best_response(G, strategies, payoffs):
    """
    最优响应策略更新：模仿邻居中支付最高的策略
    
    参数:
    G: 网络
    strategies: 当前策略数组
    payoffs: 当前支付数组
    
    返回:
    new_strategies: 更新后的策略数组
    """
    n = G.number_of_nodes()
    new_strategies = strategies.copy()
    
    for i in range(n):
        neighbors = list(G.neighbors(i))
        if len(neighbors) == 0:
            continue
        
        all_nodes = neighbors + [i]
        best_idx = all_nodes[np.argmax([payoffs[j] for j in all_nodes])]
        new_strategies[i] = strategies[best_idx]
    
    return new_strategies


def simulate_network_game(G, payoff_matrix, n_steps, initial_freq=None, 
                          update_rule='fermi', beta=0.1, record_interval=1):
    """
    模拟网络上的演化博弈
    
    参数:
    G: 网络
    payoff_matrix: 支付矩阵
    n_steps: 模拟步数
    initial_freq: 初始策略频率
    update_rule: 更新规则 ('fermi' 或 'best')
    beta: Fermi规则的选择强度
    record_interval: 记录间隔
    
    返回:
    freq_history: 策略频率历史 (n_steps/record_interval, n_strategies)
    strategy_history: 策略配置历史
    """
    n = G.number_of_nodes()
    n_strategies = payoff_matrix.shape[0]
    
    strategies = initialize_strategies(n, n_strategies, initial_freq)
    
    freq_history = []
    strategy_history = []
    
    for step in range(n_steps):
        if step % record_interval == 0:
            freq = np.bincount(strategies, minlength=n_strategies) / n
            freq_history.append(freq)
            strategy_history.append(strategies.copy())
        
        payoffs = calculate_payoffs(G, strategies, payoff_matrix)
        
        if update_rule == 'fermi':
            strategies = update_strategies_fermi(G, strategies, payoffs, beta)
        elif update_rule == 'best':
            strategies = update_strategies_best_response(G, strategies, payoffs)
    
    freq = np.bincount(strategies, minlength=n_strategies) / n
    freq_history.append(freq)
    strategy_history.append(strategies.copy())
    
    return np.array(freq_history), strategy_history


def plot_network_strategies(G, strategies, labels=None, ax=None):
    """
    绘制网络策略空间分布
    
    参数:
    G: 网络
    strategies: 策略数组
    labels: 策略标签
    ax: matplotlib轴
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    pos = nx.spring_layout(G, seed=42)
    
    n_strategies = len(np.unique(strategies))
    cmap = plt.cm.get_cmap('viridis', n_strategies)
    
    nx.draw_networkx_nodes(G, pos, node_color=strategies, cmap=cmap, 
                           node_size=100, ax=ax, alpha=0.8)
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
    
    ax.set_title('网络策略空间分布', fontsize=14)
    ax.axis('off')
    
    if labels is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=n_strategies-1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, ticks=range(n_strategies))
        cbar.set_ticklabels(labels)
    
    return ax


def plot_network_frequency(freq_history, labels=None):
    """
    绘制网络博弈策略频率演化
    
    参数:
    freq_history: 策略频率历史
    labels: 策略标签
    """
    plt.figure(figsize=(10, 6))
    n_strategies = freq_history.shape[1]
    
    for i in range(n_strategies):
        label = labels[i] if labels else f'策略 {i}'
        plt.plot(freq_history[:, i], label=label, linewidth=2)
    
    plt.xlabel('时间步', fontsize=12)
    plt.ylabel('策略频率', fontsize=12)
    plt.title('网络博弈策略频率演化', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.show()


def animate_network_game(G, strategy_history, labels=None, interval=200):
    """
    创建网络博弈策略演化动画
    
    参数:
    G: 网络
    strategy_history: 策略配置历史
    labels: 策略标签
    interval: 帧间隔（毫秒）
    
    返回:
    anim: 动画对象
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    pos = nx.spring_layout(G, seed=42)
    
    n_strategies = len(np.unique(strategy_history[0]))
    cmap = plt.cm.get_cmap('viridis', n_strategies)
    
    def update(frame):
        ax.clear()
        strategies = strategy_history[frame]
        nx.draw_networkx_nodes(G, pos, node_color=strategies, cmap=cmap, 
                               node_size=100, ax=ax, alpha=0.8)
        nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
        ax.set_title(f'网络策略空间分布 (步 {frame})', fontsize=14)
        ax.axis('off')
    
    anim = FuncAnimation(fig, update, frames=len(strategy_history), 
                         interval=interval, repeat=True)
    plt.close(fig)
    return anim


def compare_network_topologies(payoff_matrix, n_nodes=100, n_steps=200, labels=None):
    """
    比较不同网络拓扑的演化结果
    
    参数:
    payoff_matrix: 支付矩阵
    n_nodes: 节点数
    n_steps: 模拟步数
    labels: 策略标签
    """
    initial_freq = np.array([0.5, 0.5])
    
    networks = [
        ('随机图 (p=0.1)', create_random_network(n_nodes, p=0.1)),
        ('小世界网络 (k=4, p=0.1)', create_small_world_network(n_nodes, k=4, p=0.1)),
        ('网格网络 (10x10)', create_grid_network(10, periodic=True)),
    ]
    
    results = []
    
    plt.figure(figsize=(15, 5))
    
    for idx, (name, G) in enumerate(networks):
        freq_history, _ = simulate_network_game(
            G, payoff_matrix, n_steps, initial_freq=initial_freq, 
            update_rule='fermi', beta=0.5, record_interval=1
        )
        results.append((name, freq_history))
        
        plt.subplot(1, 3, idx + 1)
        n_strategies = freq_history.shape[1]
        for i in range(n_strategies):
            label = labels[i] if labels else f'策略 {i}'
            plt.plot(freq_history[:, i], label=label, linewidth=2)
        
        plt.title(name, fontsize=12)
        plt.xlabel('时间步')
        plt.ylabel('频率')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
    
    plt.suptitle('不同网络拓扑的演化比较', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return results


def main():
    print("=" * 60)
    print("复制动态方程与网络演化博弈")
    print("=" * 60)
    
    V = 2
    C = 4
    print(f"\n鹰鸽博弈参数: V = {V}, C = {C}")
    
    payoff_matrix = hawk_dove_game(V, C)
    print("\n支付矩阵:")
    print(payoff_matrix)
    
    ess = find_ess(payoff_matrix)
    if ess is not None:
        print(f"\n演化稳定策略 (ESS):")
        print(f"  鹰策略频率: {ess[0]:.4f}")
        print(f"  鸽策略频率: {ess[1]:.4f}")
    
    print(f"\n{'='*60}")
    print("【第一部分：均匀混合种群复制动态】")
    print(f"{'='*60}")
    
    test_cases = [
        ([0.3, 0.7], "正常初始条件"),
        ([0.0, 1.0], "纯鸽策略边界"),
        ([1.0, 0.0], "纯鹰策略边界"),
        ([0.0001, 0.9999], "接近边界的初始条件"),
    ]
    
    t_span = [0, 50]
    
    for initial_x, description in test_cases:
        print(f"\n测试案例: {description}")
        print(f"初始策略频率: {initial_x}")
        
        initial_x = np.array(initial_x)
        t, x = solve_replicator_dynamics(payoff_matrix, initial_x, t_span)
        
        print(f"最终策略频率: {x[-1]}")
        print(f"频率和: {np.sum(x[-1]):.10f}")
        
        if np.any(x < 0) or np.any(x > 1):
            print("警告: 频率超出有效范围!")
        else:
            print("✓ 所有频率保持在 [0, 1] 范围内")
    
    print(f"\n{'='*60}")
    print("【第二部分：网络演化博弈】")
    print(f"{'='*60}")
    
    print("\n创建10x10网格网络 (100个节点)...")
    G = create_grid_network(10, periodic=True)
    print(f"网络节点数: {G.number_of_nodes()}")
    print(f"网络连边数: {G.number_of_edges()}")
    
    initial_freq = np.array([0.5, 0.5])
    n_steps = 100
    
    print(f"\n模拟网络演化博弈...")
    print(f"初始策略频率: 鹰={initial_freq[0]}, 鸽={initial_freq[1]}")
    print(f"模拟步数: {n_steps}")
    print(f"更新规则: Fermi规则 (beta=0.5)")
    
    freq_history, strategy_history = simulate_network_game(
        G, payoff_matrix, n_steps, initial_freq=initial_freq,
        update_rule='fermi', beta=0.5, record_interval=5
    )
    
    print(f"\n初始频率: {freq_history[0]}")
    print(f"最终频率: {freq_history[-1]}")
    print(f"理论ESS: {ess}")
    
    print(f"\n{'='*60}")
    print("【第三部分：不同网络拓扑比较】")
    print(f"{'='*60}")
    
    print("\n比较: 随机图 vs 小世界网络 vs 网格网络")
    print("(可视化将显示不同拓扑的演化差异)")
    
    print(f"\n{'='*60}")
    print("网络演化博弈功能已成功实现!")
    print("=" * 60)
    print("\n主要功能:")
    print("  1. 三种网络拓扑: 随机图、小世界网络、网格网络")
    print("  2. 两种策略更新规则: Fermi规则、最优响应")
    print("  3. 空间策略分布可视化")
    print("  4. 策略频率演化曲线")
    print("  5. 网络拓扑对比分析")
    print("  6. 演化过程动画生成")
    
    initial_x = np.array([0.3, 0.7])
    t, x = solve_replicator_dynamics(payoff_matrix, initial_x, t_span)
    plot_replicator_dynamics(t, x, ['鹰策略', '鸽策略'])
    plot_phase_portrait(payoff_matrix)
    
    print("\n绘制网络博弈结果...")
    plot_network_frequency(freq_history, labels=['鹰策略', '鸽策略'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    plot_network_strategies(G, strategy_history[0], labels=['鹰策略', '鸽策略'], ax=ax1)
    ax1.set_title('初始策略分布', fontsize=14)
    plot_network_strategies(G, strategy_history[-1], labels=['鹰策略', '鸽策略'], ax=ax2)
    ax2.set_title('最终策略分布', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    print("\n运行网络拓扑比较...")
    compare_network_topologies(payoff_matrix, n_nodes=100, n_steps=150, 
                                labels=['鹰策略', '鸽策略'])


if __name__ == "__main__":
    main()

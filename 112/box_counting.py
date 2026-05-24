import numpy as np
from PIL import Image
from scipy import stats
import matplotlib.pyplot as plt


def box_counting_dimension(data, is_image=True, box_sizes=None, min_box_size=2, max_box_size=None, 
                            num_sizes=10, use_multigrid=True, num_offsets=10, seed=42):
    """
    使用盒计数法估计分形维数（支持多重网格平均提高鲁棒性）
    
    参数:
        data: 二值图像 (numpy数组, 0为背景, 非0为前景) 或 点集 (Nx2数组)
        is_image: True表示输入是图像, False表示是点集
        box_sizes: 自定义盒子大小列表 (如果为None则自动生成)
        min_box_size: 最小盒子大小
        max_box_size: 最大盒子大小 (如果为None则根据数据自动确定)
        num_sizes: 盒子大小的数量
        use_multigrid: 是否使用多重网格平均（提高鲁棒性，减少网格偏移敏感性）
        num_offsets: 多重网格平均的随机偏移次数（仅当use_multigrid=True时有效）
        seed: 随机种子，保证结果可复现
    
    返回:
        D: 分形维数
        box_sizes: 使用的盒子大小列表
        counts: 每个大小对应的盒子数量（多重网格平均后的值）
        slope, intercept, r_value: 线性回归结果
        counts_std: 每个大小盒子数量的标准差（仅多重网格模式）
        all_counts: 所有偏移的原始计数（仅多重网格模式）
    """
    
    if is_image:
        points = _image_to_points(data)
    else:
        points = np.array(data)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("点集必须是Nx2的二维数组")
    
    if len(points) == 0:
        raise ValueError("没有找到前景像素或点")
    
    if max_box_size is None:
        x_range = points[:, 0].max() - points[:, 0].min()
        y_range = points[:, 1].max() - points[:, 1].min()
        max_box_size = int(max(x_range, y_range) / 2)
    
    if box_sizes is None:
        box_sizes = np.logspace(np.log10(min_box_size), np.log10(max_box_size), num=num_sizes, dtype=int)
        box_sizes = np.unique(box_sizes)
        box_sizes = box_sizes[box_sizes >= min_box_size]
    
    counts = []
    counts_std = []
    all_counts = []
    valid_sizes = []
    
    for size in box_sizes:
        if size <= 0:
            continue
        
        if use_multigrid:
            count_mean, count_std, count_list = _count_boxes_multigrid(
                points, size, num_offsets=num_offsets, seed=seed
            )
            if count_mean > 0:
                counts.append(count_mean)
                counts_std.append(count_std)
                all_counts.append(count_list)
                valid_sizes.append(size)
        else:
            count = _count_boxes(points, size)
            if count > 0:
                counts.append(count)
                valid_sizes.append(size)
    
    if len(counts) < 2:
        raise ValueError("有效的盒子大小数量不足，无法计算分形维数")
    
    log_sizes = np.log(1.0 / np.array(valid_sizes))
    log_counts = np.log(counts)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_sizes, log_counts)
    
    D = slope
    
    if use_multigrid:
        return D, valid_sizes, counts, slope, intercept, r_value, counts_std, all_counts
    else:
        return D, valid_sizes, counts, slope, intercept, r_value, None, None


def _image_to_points(image):
    """将二值图像转换为点集"""
    image = np.array(image)
    if image.ndim == 3:
        image = image.mean(axis=2)
    y_coords, x_coords = np.where(image > 0)
    points = np.column_stack((x_coords, y_coords))
    return points


def _count_boxes(points, box_size, offset_x=0, offset_y=0):
    """统计覆盖点集所需的盒子数量（支持网格偏移）
    
    参数:
        points: 点集 (Nx2数组)
        box_size: 盒子大小
        offset_x, offset_y: 网格偏移量 (0 <= offset < box_size)
    """
    x_min, y_min = points[:, 0].min(), points[:, 1].min()
    
    grid_x = ((points[:, 0] - x_min + offset_x) / box_size).astype(int)
    grid_y = ((points[:, 1] - y_min + offset_y) / box_size).astype(int)
    
    boxes = set(zip(grid_x, grid_y))
    
    return len(boxes)


def _count_boxes_multigrid(points, box_size, num_offsets=10, seed=None):
    """使用多重网格平均统计盒子数量（随机偏移多次取平均）
    
    参数:
        points: 点集 (Nx2数组)
        box_size: 盒子大小
        num_offsets: 随机偏移的次数
        seed: 随机种子
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    counts = []
    for _ in range(num_offsets):
        offset_x = rng.uniform(0, box_size)
        offset_y = rng.uniform(0, box_size)
        count = _count_boxes(points, box_size, offset_x, offset_y)
        counts.append(count)
    
    return np.mean(counts), np.std(counts), counts


def load_binary_image(image_path, threshold=128):
    """加载并二值化图像"""
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    binary = (img_array > threshold).astype(np.uint8) * 255
    return binary


def test_robustness(points, num_trials=20, min_box_size=2, num_sizes=12, num_offsets=10):
    """测试盒计数法的鲁棒性（不同随机偏移下D值的波动）
    
    参数:
        points: 点集
        num_trials: 测试次数
        min_box_size, num_sizes: 盒计数参数
        num_offsets: 多重网格平均的偏移次数
    
    返回:
        results_single: 单次偏移的结果数组
        results_multi: 多重网格平均的结果数组
    """
    results_single = []
    results_multi = []
    
    for i in range(num_trials):
        seed_single = i
        seed_multi = i + 1000
        
        D_single, _, _, _, _, r_single, _, _ = _box_counting_single_offset(
            points, min_box_size=min_box_size, num_sizes=num_sizes, seed=seed_single
        )
        results_single.append(D_single)
        
        D_multi, _, _, _, _, r_multi, _, _ = box_counting_dimension(
            points, is_image=False, min_box_size=min_box_size, 
            num_sizes=num_sizes, use_multigrid=True, num_offsets=num_offsets, seed=seed_multi
        )
        results_multi.append(D_multi)
    
    results_single = np.array(results_single)
    results_multi = np.array(results_multi)
    
    print(f"\n=== 鲁棒性测试结果 (测试次数: {num_trials}) ===")
    print(f"\n单次偏移模式 (网格起始点随机偏移):")
    print(f"  D值范围: [{results_single.min():.4f}, {results_single.max():.4f}]")
    print(f"  D值均值: {results_single.mean():.4f}")
    print(f"  D值标准差: {results_single.std():.4f}")
    print(f"  最大相对波动: {(results_single.max() - results_single.min()) / results_single.mean() * 100:.2f}%")
    
    print(f"\n多重网格平均模式 (偏移次数: {num_offsets}):")
    print(f"  D值范围: [{results_multi.min():.4f}, {results_multi.max():.4f}]")
    print(f"  D值均值: {results_multi.mean():.4f}")
    print(f"  D值标准差: {results_multi.std():.4f}")
    print(f"  最大相对波动: {(results_multi.max() - results_multi.min()) / results_multi.mean() * 100:.2f}%")
    
    improvement = (results_single.std() - results_multi.std()) / results_single.std() * 100
    print(f"\n标准差降低: {improvement:.1f}%")
    
    return results_single, results_multi


def _box_counting_single_offset(points, min_box_size=2, max_box_size=None, num_sizes=10, seed=None):
    """单次偏移的盒计数法（用于鲁棒性对比测试）"""
    rng = np.random.RandomState(seed)
    
    if max_box_size is None:
        x_range = points[:, 0].max() - points[:, 0].min()
        y_range = points[:, 1].max() - points[:, 1].min()
        max_box_size = int(max(x_range, y_range) / 2)
    
    box_sizes = np.logspace(np.log10(min_box_size), np.log10(max_box_size), num=num_sizes, dtype=int)
    box_sizes = np.unique(box_sizes)
    box_sizes = box_sizes[box_sizes >= min_box_size]
    
    offset_x = rng.uniform(0, min_box_size)
    offset_y = rng.uniform(0, min_box_size)
    
    counts = []
    valid_sizes = []
    
    for size in box_sizes:
        if size <= 0:
            continue
        count = _count_boxes(points, size, offset_x * size / min_box_size, offset_y * size / min_box_size)
        if count > 0:
            counts.append(count)
            valid_sizes.append(size)
    
    if len(counts) < 2:
        raise ValueError("有效的盒子大小数量不足")
    
    log_sizes = np.log(1.0 / np.array(valid_sizes))
    log_counts = np.log(counts)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_sizes, log_counts)
    
    D = slope
    
    return D, valid_sizes, counts, slope, intercept, r_value, None, None


def _compute_box_probabilities(points, box_size, offset_x=0, offset_y=0):
    """计算每个盒子中的点概率分布"""
    x_min, y_min = points[:, 0].min(), points[:, 1].min()
    
    grid_x = ((points[:, 0] - x_min + offset_x) / box_size).astype(int)
    grid_y = ((points[:, 1] - y_min + offset_y) / box_size).astype(int)
    
    boxes = {}
    total_points = len(points)
    
    for gx, gy in zip(grid_x, grid_y):
        key = (gx, gy)
        boxes[key] = boxes.get(key, 0) + 1
    
    probabilities = np.array(list(boxes.values())) / total_points
    return probabilities


def generalized_dimension(points, q_values=None, min_box_size=2, max_box_size=None, 
                          num_sizes=10, use_multigrid=True, num_offsets=10, seed=42):
    """计算广义维数谱 D_q (多分形分析)
    
    广义维数定义: D_q = lim(ε→0) [1/(q-1)] * log(Σp_i^q) / log(ε)
    
    参数:
        points: 点集 (Nx2数组)
        q_values: q值列表 (如 [-2, -1, 0, 1, 2, 3, 4])，None则使用默认值
        min_box_size: 最小盒子大小
        max_box_size: 最大盒子大小
        num_sizes: 盒子大小的数量
        use_multigrid: 是否使用多重网格平均
        num_offsets: 随机偏移次数
        seed: 随机种子
    
    返回:
        q_values: 使用的q值列表
        D_q: 对应的广义维数值
        r_values: 每个q值的线性回归R²值
        box_sizes: 盒子大小列表
        all_data: 所有中间计算数据
    """
    points = np.array(points)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("点集必须是Nx2的二维数组")
    
    if q_values is None:
        q_values = np.array([-5, -3, -2, -1, 0, 0.5, 1, 1.5, 2, 3, 5])
    else:
        q_values = np.array(q_values)
    
    if max_box_size is None:
        x_range = points[:, 0].max() - points[:, 0].min()
        y_range = points[:, 1].max() - points[:, 1].min()
        max_box_size = int(max(x_range, y_range) / 2)
    
    box_sizes = np.logspace(np.log10(min_box_size), np.log10(max_box_size), num=num_sizes, dtype=int)
    box_sizes = np.unique(box_sizes)
    box_sizes = box_sizes[box_sizes >= min_box_size]
    
    if use_multigrid:
        rng = np.random.RandomState(seed)
        offsets = [(rng.uniform(0, size), rng.uniform(0, size)) for size in box_sizes for _ in range(num_offsets)]
    else:
        offsets = [(0, 0)] * len(box_sizes)
    
    D_q = []
    r_values = []
    all_data = {}
    
    for q in q_values:
        log_chi_q = []
        log_eps = []
        
        for idx, size in enumerate(box_sizes):
            if size <= 0:
                continue
            
            if use_multigrid:
                chi_q_list = []
                for off_idx in range(num_offsets):
                    ox, oy = offsets[idx * num_offsets + off_idx]
                    probs = _compute_box_probabilities(points, size, ox, oy)
                    
                    if q == 1:
                        probs_nonzero = probs[probs > 0]
                        chi_q = np.exp(-np.sum(probs_nonzero * np.log(probs_nonzero)))
                    else:
                        chi_q = np.sum(probs ** q) ** (1.0 / (q - 1))
                    chi_q_list.append(chi_q)
                chi_q_mean = np.mean(chi_q_list)
            else:
                ox, oy = offsets[idx]
                probs = _compute_box_probabilities(points, size, ox, oy)
                
                if q == 1:
                    probs_nonzero = probs[probs > 0]
                    chi_q_mean = np.exp(-np.sum(probs_nonzero * np.log(probs_nonzero)))
                else:
                    chi_q_mean = np.sum(probs ** q) ** (1.0 / (q - 1))
            
            if chi_q_mean > 0:
                log_chi_q.append(np.log(chi_q_mean))
                log_eps.append(np.log(size))
        
        if len(log_chi_q) < 2:
            D_q.append(np.nan)
            r_values.append(np.nan)
            continue
        
        log_eps = np.array(log_eps)
        log_chi_q = np.array(log_chi_q)
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_eps, log_chi_q)
        D_q.append(slope)
        r_values.append(r_value ** 2)
        
        all_data[f'q_{q}'] = {'log_eps': log_eps, 'log_chi_q': log_chi_q, 
                               'slope': slope, 'intercept': intercept}
    
    return q_values, np.array(D_q), np.array(r_values), box_sizes, all_data


def information_dimension(points, **kwargs):
    """计算信息维数 D_1 (信息维数是广义维数q=1的特例)
    
    信息维数: D_1 = lim(ε→0) [-Σp_i log(p_i)] / log(1/ε)
    """
    q_values, D_q, r_values, box_sizes, all_data = generalized_dimension(
        points, q_values=[1], **kwargs
    )
    return D_q[0], r_values[0], box_sizes, all_data


def correlation_dimension(points, **kwargs):
    """计算关联维数 D_2 (关联维数是广义维数q=2的特例)
    
    关联维数: D_2 = lim(ε→0) log(C(ε)) / log(ε)
    其中 C(ε) 是关联积分
    """
    q_values, D_q, r_values, box_sizes, all_data = generalized_dimension(
        points, q_values=[2], **kwargs
    )
    return D_q[0], r_values[0], box_sizes, all_data


def correlation_integral(points, eps_values=None, num_eps=15, min_eps=0.01, max_eps=None):
    """直接计算关联积分 C(ε) (用于Grassberger-Procaccia算法)
    
    C(ε) = (2/N(N-1)) * Σ_{i<j} θ(ε - |x_i - x_j|)
    其中 θ 是阶跃函数
    """
    points = np.array(points)
    N = len(points)
    
    if max_eps is None:
        dists = np.sqrt(np.sum((points[0] - points[1:]) ** 2, axis=1))
        max_eps = np.max(dists)
    
    if eps_values is None:
        eps_values = np.logspace(np.log10(min_eps), np.log10(max_eps), num=num_eps)
    
    C_eps = np.zeros_like(eps_values, dtype=float)
    
    for k, eps in enumerate(eps_values):
        count = 0
        for i in range(N):
            dists = np.sqrt(np.sum((points[i] - points[i+1:]) ** 2, axis=1))
            count += np.sum(dists < eps)
        C_eps[k] = 2 * count / (N * (N - 1))
    
    return eps_values, C_eps


def correlation_dimension_gp(points, eps_values=None, num_eps=15, min_eps=0.01, max_eps=None):
    """使用Grassberger-Procaccia算法计算关联维数"""
    eps_values, C_eps = correlation_integral(points, eps_values, num_eps, min_eps, max_eps)
    
    valid_idx = C_eps > 0
    log_eps = np.log(eps_values[valid_idx])
    log_C = np.log(C_eps[valid_idx])
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_eps, log_C)
    
    return slope, r_value, eps_values, C_eps


def plot_multifractal_spectrum(q_values, D_q, r_values=None, save_path=None):
    """绘制广义维数谱 D_q"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(q_values, D_q, 'bo-', linewidth=2, markersize=6)
    ax1.set_xlabel('q (矩的阶数)')
    ax1.set_ylabel('D_q (广义维数)')
    ax1.set_title('广义维数谱 D_q')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='q=0 (盒维数)')
    ax1.axvline(x=1, color='g', linestyle='--', alpha=0.5, label='q=1 (信息维数)')
    ax1.axvline(x=2, color='m', linestyle='--', alpha=0.5, label='q=2 (关联维数)')
    ax1.legend()
    
    if r_values is not None:
        ax2.plot(q_values, r_values, 'go-', linewidth=2, markersize=6)
        ax2.set_xlabel('q (矩的阶数)')
        ax2.set_ylabel('R²')
        ax2.set_title('线性回归拟合优度')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0.8, 1.01])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def generate_multifractal_points(num_points=50000, seed=42):
    """生成多分形点集（二项测度）用于测试
    
    使用随机乘子法生成分形测度
    """
    rng = np.random.RandomState(seed)
    
    x, y = 0.0, 0.0
    points = np.zeros((num_points, 2))
    
    p1, p2 = 0.7, 0.3
    
    for i in range(num_points):
        r = rng.random()
        if r < p1 * p1:
            x = x / 2
            y = y / 2
        elif r < p1 * p1 + p1 * p2:
            x = (x + 1) / 2
            y = y / 2
        elif r < p1 * p1 + p1 * p2 + p2 * p1:
            x = x / 2
            y = (y + 1) / 2
        else:
            x = (x + 1) / 2
            y = (y + 1) / 2
        points[i] = [x, y]
    
    return points


def generate_fractal_points(fractal_type='cantor', num_points=10000):
    """生成分形点集用于测试"""
    
    if fractal_type == 'cantor':
        points = []
        x, y = 0.0, 0.0
        for _ in range(num_points):
            r = np.random.random()
            if r < 0.5:
                x = x / 3
                y = y / 3
            else:
                x = (x + 2) / 3
                y = (y + 2) / 3
            points.append([x, y])
        return np.array(points)
    
    elif fractal_type == 'sierpinski':
        points = np.zeros((num_points, 2))
        x, y = 0, 0
        for i in range(num_points):
            r = np.random.randint(0, 3)
            if r == 0:
                x = x / 2
                y = y / 2
            elif r == 1:
                x = (x + 1) / 2
                y = y / 2
            else:
                x = (x + 0.5) / 2
                y = (y + np.sqrt(3) / 2) / 2
            points[i] = [x, y]
        return points
    
    elif fractal_type == 'koch':
        def koch_curve(start, end, depth):
            if depth == 0:
                return [start, end]
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            p1 = (start[0] + dx/3, start[1] + dy/3)
            p2 = (start[0] + dx/2 - dy*np.sqrt(3)/6, start[1] + dy/2 + dx*np.sqrt(3)/6)
            p3 = (start[0] + 2*dx/3, start[1] + 2*dy/3)
            return (koch_curve(start, p1, depth-1)[:-1] + 
                    koch_curve(p1, p2, depth-1)[:-1] + 
                    koch_curve(p2, p3, depth-1)[:-1] + 
                    koch_curve(p3, end, depth-1))
        
        points = koch_curve((0, 0), (1, 0), 5)
        return np.array(points)
    
    else:
        raise ValueError(f"未知的分形类型: {fractal_type}")


def plot_box_counting(box_sizes, counts, D, r_value, save_path=None):
    """绘制盒计数法的log-log图"""
    log_sizes = np.log(1.0 / np.array(box_sizes))
    log_counts = np.log(counts)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.loglog(1.0 / np.array(box_sizes), counts, 'bo-', markersize=6)
    ax1.set_xlabel('1/ε (盒子大小的倒数)')
    ax1.set_ylabel('N(ε) (盒子数量)')
    ax1.set_title('盒计数法 - 双对数坐标')
    ax1.grid(True, which="both", ls="-", alpha=0.3)
    
    ax2.scatter(log_sizes, log_counts, c='blue', s=40, label='数据点')
    x_fit = np.linspace(min(log_sizes), max(log_sizes), 100)
    y_fit = D * x_fit + (log_counts[0] - D * log_sizes[0])
    ax2.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'拟合直线 (D={D:.3f})')
    ax2.set_xlabel('log(1/ε)')
    ax2.set_ylabel('log(N(ε))')
    ax2.set_title(f'线性回归 - 分形维数 D = {D:.3f}\nR² = {r_value**2:.4f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_points(points, title='点集', save_path=None):
    """绘制点集"""
    plt.figure(figsize=(8, 8))
    plt.scatter(points[:, 0], points[:, 1], s=1, c='black', alpha=0.6)
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def main():
    print("=" * 70)
    print("分形维数工具箱 - 盒计数法与多分形分析")
    print("=" * 70)
    
    print("\n1. 单分形测试 - 谢尔宾斯基三角形 (D ≈ 1.585)")
    print("-" * 60)
    sierpinski_points = generate_fractal_points('sierpinski', num_points=50000)
    plot_points(sierpinski_points, '谢尔宾斯基三角形 (单分形)')
    D0, sizes, counts, slope, intercept, r_value, counts_std, all_counts = box_counting_dimension(
        sierpinski_points, is_image=False, min_box_size=2, num_sizes=15,
        use_multigrid=True, num_offsets=10, seed=42
    )
    print(f"盒维数 D_0 = {D0:.4f}")
    print(f"理论维数 ≈ 1.5850")
    print(f"R² = {r_value**2:.4f}")
    plot_box_counting(sizes, counts, D0, r_value)
    
    print("\n2. 多分形分析 - 二项测度多分形点集")
    print("-" * 60)
    print("生成多分形点集...")
    multi_points = generate_multifractal_points(num_points=50000, seed=42)
    plot_points(multi_points, '多分形点集 (二项测度)')
    
    print("\n计算广义维数谱 D_q...")
    q_values, D_q, r_values, box_sizes, all_data = generalized_dimension(
        multi_points, min_box_size=4, num_sizes=10,
        use_multigrid=True, num_offsets=5, seed=42
    )
    print(f"{'q':>6} {'D_q':>10} {'R²':>10}")
    print("-" * 30)
    for q, d, r in zip(q_values, D_q, r_values):
        print(f"{q:>6.1f} {d:>10.4f} {r:>10.4f}")
    plot_multifractal_spectrum(q_values, D_q, r_values)
    
    idx_q0 = np.argmin(np.abs(q_values - 0))
    idx_q1 = np.argmin(np.abs(q_values - 1))
    idx_q2 = np.argmin(np.abs(q_values - 2))
    print(f"\nD_0 (盒维数)   = {D_q[idx_q0]:.4f}")
    print(f"D_1 (信息维数) = {D_q[idx_q1]:.4f}")
    print(f"D_2 (关联维数) = {D_q[idx_q2]:.4f}")
    
    if D_q[idx_q0] > D_q[idx_q1] > D_q[idx_q2]:
        print("✓ 检测到多分形特性: D_0 > D_1 > D_2")
    
    print("\n3. 信息维数与关联维数 (专门函数)")
    print("-" * 60)
    D1, r1, _, _ = information_dimension(
        multi_points, min_box_size=4, num_sizes=10,
        use_multigrid=True, num_offsets=5, seed=42
    )
    print(f"信息维数 D_1 = {D1:.4f}, R² = {r1:.4f}")
    
    D2, r2, _, _ = correlation_dimension(
        multi_points, min_box_size=4, num_sizes=10,
        use_multigrid=True, num_offsets=5, seed=42
    )
    print(f"关联维数 D_2 = {D2:.4f}, R² = {r2:.4f}")
    
    print("\n4. 鲁棒性对比测试")
    print("-" * 60)
    print("对比单次偏移 vs 多重网格平均的稳定性...")
    test_robustness(sierpinski_points, num_trials=20, min_box_size=2, num_sizes=12, num_offsets=10)
    
    print("\n" + "=" * 70)
    print("使用示例:")
    print("=" * 70)
    print("""
# 1. 盒维数 (D_0)
from box_counting import box_counting_dimension, generate_fractal_points
points = generate_fractal_points('sierpinski', num_points=20000)
D0, sizes, counts, _, _, r_value, _, _ = box_counting_dimension(
    points, is_image=False, use_multigrid=True, num_offsets=10
)
print(f"盒维数 D_0 = {D0:.4f}")

# 2. 广义维数谱 D_q (多分形分析，用于湍流、金融等)
from box_counting import generalized_dimension, generate_multifractal_points
multi_points = generate_multifractal_points(num_points=50000)
q_values, D_q, r_values, _, _ = generalized_dimension(
    multi_points, q_values=[-3, -2, -1, 0, 1, 2, 3],
    use_multigrid=True, num_offsets=5
)
for q, d in zip(q_values, D_q):
    print(f"D_{q} = {d:.4f}")

# 3. 信息维数 D_1 (熵的标度律)
from box_counting import information_dimension
D1, r1, _, _ = information_dimension(points, use_multigrid=True)
print(f"信息维数 D_1 = {D1:.4f}")

# 4. 关联维数 D_2 (混沌吸引子、时间序列分析)
from box_counting import correlation_dimension, correlation_dimension_gp
D2, r2, _, _ = correlation_dimension(points, use_multigrid=True)
print(f"关联维数 D_2 (盒计数) = {D2:.4f}")

# Grassberger-Procaccia算法 (更精确但更慢)
D2_gp, r2_gp, _, _ = correlation_dimension_gp(points[::5])
print(f"关联维数 D_2 (GP算法) = {D2_gp:.4f}")
    """)


if __name__ == "__main__":
    main()

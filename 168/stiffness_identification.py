import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


def generate_test_data(K_true=500.0, noise_level=5.0, num_points=50, 
                       outlier_ratio=0.15, outlier_magnitude=30.0):
    """
    生成测试用的力-位移数据，包含离群值（模拟传感器噪声或接触滑动）
    
    参数:
        K_true: 真实刚度值 (N/m)
        noise_level: 高斯噪声水平 (N)
        num_points: 数据点数量
        outlier_ratio: 离群值比例
        outlier_magnitude: 离群值偏移幅度 (N)
    
    返回:
        displacement: 位移数组 (m)
        force: 力数组 (N)
        is_outlier: 布尔数组，标记是否为离群值
    """
    displacement = np.linspace(0.0, 0.02, num_points)
    force_true = K_true * displacement
    
    noise = np.random.normal(0, noise_level, num_points)
    force = force_true + noise
    
    is_outlier = np.zeros(num_points, dtype=bool)
    num_outliers = int(num_points * outlier_ratio)
    if num_outliers > 0:
        outlier_indices = np.random.choice(num_points, num_outliers, replace=False)
        is_outlier[outlier_indices] = True
        outlier_offsets = np.random.uniform(-outlier_magnitude, outlier_magnitude, num_outliers)
        force[outlier_indices] += outlier_offsets
    
    return displacement, force, is_outlier


def load_data_from_file(filename):
    """
    从文件加载力-位移数据
    文件格式: 每行包含位移和力，用空格或逗号分隔
    
    参数:
        filename: 数据文件名
    
    返回:
        displacement: 位移数组 (m)
        force: 力数组 (N)
    """
    try:
        data = np.loadtxt(filename, delimiter=',')
        displacement = data[:, 0]
        force = data[:, 1]
        return displacement, force
    except:
        data = np.loadtxt(filename)
        displacement = data[:, 0]
        force = data[:, 1]
        return displacement, force


def residual(K, displacement, force):
    """
    计算残差: F - K * x
    
    参数:
        K: 刚度值 (N/m)
        displacement: 位移数组 (m)
        force: 力数组 (N)
    
    返回:
        残差数组
    """
    return force - K * displacement


def huber_loss(residual, delta=1.0):
    """
    Huber损失函数
    
    参数:
        residual: 残差数组
        delta: 阈值参数
    
    返回:
        Huber损失值
    """
    abs_res = np.abs(residual)
    return np.where(abs_res <= delta, 
                    0.5 * residual ** 2, 
                    delta * (abs_res - 0.5 * delta))


def identify_stiffness_lstsq(displacement, force):
    """
    使用最小二乘法辨识刚度 K = F / x
    
    参数:
        displacement: 位移数组 (m)
        force: 力数组 (N)
    
    返回:
        K: 辨识得到的刚度值 (N/m)
        K_std: 刚度的标准差
        r_squared: 决定系数 R^2
        inlier_mask: 内点掩码（全为True）
    """
    X = displacement.reshape(-1, 1)
    y = force
    
    K, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    K = K[0]
    
    y_pred = K * displacement
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0
    
    n = len(displacement)
    mse = ss_res / (n - 1)
    K_std = np.sqrt(mse / np.sum(displacement ** 2))
    
    inlier_mask = np.ones_like(displacement, dtype=bool)
    
    return K, K_std, r_squared, inlier_mask


def identify_stiffness_huber(displacement, force, delta=1.5):
    """
    使用Huber损失的鲁棒最小二乘法辨识刚度
    
    参数:
        displacement: 位移数组 (m)
        force: 力数组 (N)
        delta: Huber损失阈值
    
    返回:
        K: 辨识得到的刚度值 (N/m)
        K_std: 刚度的标准差
        r_squared: 决定系数 R^2
        inlier_mask: 内点掩码
    """
    def huber_cost(K):
        res = residual(K[0], displacement, force)
        return huber_loss(res, delta)
    
    K0 = np.array([np.mean(force / (displacement + 1e-10))])
    result = least_squares(huber_cost, K0, method='lm')
    K = result.x[0]
    
    y_pred = K * displacement
    residuals = np.abs(force - y_pred)
    inlier_mask = residuals <= delta * np.std(residuals)
    
    ss_res = np.sum((force - y_pred) ** 2)
    ss_tot = np.sum((force - np.mean(force)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0
    
    n = np.sum(inlier_mask)
    if n > 1:
        mse = ss_res / (n - 1)
        K_std = np.sqrt(mse / np.sum(displacement[inlier_mask] ** 2))
    else:
        K_std = np.nan
    
    return K, K_std, r_squared, inlier_mask


def identify_stiffness_ransac(displacement, force, 
                              min_samples=2, 
                              residual_threshold=5.0,
                              max_trials=1000):
    """
    使用RANSAC算法辨识刚度，自动剔除异常接触点
    
    参数:
        displacement: 位移数组 (m)
        force: 力数组 (N)
        min_samples: 每次迭代最小采样数
        residual_threshold: 残差阈值 (N)，小于此值视为内点
        max_trials: 最大迭代次数
    
    返回:
        K: 辨识得到的刚度值 (N/m)
        K_std: 刚度的标准差
        r_squared: 决定系数 R^2
        inlier_mask: 内点掩码
    """
    n = len(displacement)
    best_K = None
    best_inlier_count = 0
    best_inlier_mask = np.ones(n, dtype=bool)
    
    for _ in range(max_trials):
        sample_indices = np.random.choice(n, min_samples, replace=False)
        x_sample = displacement[sample_indices]
        y_sample = force[sample_indices]
        
        if np.any(x_sample == 0):
            x_sample = x_sample + 1e-10
        
        K_sample = np.mean(y_sample / x_sample)
        
        residuals = np.abs(force - K_sample * displacement)
        inlier_mask = residuals <= residual_threshold
        inlier_count = np.sum(inlier_mask)
        
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_inlier_mask = inlier_mask.copy()
    
    if best_inlier_count >= min_samples:
        x_inliers = displacement[best_inlier_mask]
        y_inliers = force[best_inlier_mask]
        
        X = x_inliers.reshape(-1, 1)
        K, residuals, rank, s = np.linalg.lstsq(X, y_inliers, rcond=None)
        best_K = K[0]
        
        y_pred = best_K * x_inliers
        ss_res = np.sum((y_inliers - y_pred) ** 2)
        ss_tot = np.sum((y_inliers - np.mean(y_inliers)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0
        
        m = best_inlier_count
        mse = ss_res / (m - 1) if m > 1 else 0
        K_std = np.sqrt(mse / np.sum(x_inliers ** 2)) if m > 1 else np.nan
    else:
        best_K = np.mean(force / (displacement + 1e-10))
        K_std = np.nan
        r_squared = 0.0
        best_inlier_mask = np.ones(n, dtype=bool)
    
    return best_K, K_std, r_squared, best_inlier_mask


def identify_stiffness_softl1(displacement, force):
    """
    使用Soft L1损失的鲁棒最小二乘法辨识刚度（scipy内置）
    
    参数:
        displacement: 位移数组 (m)
        force: 力数组 (N)
    
    返回:
        K: 辨识得到的刚度值 (N/m)
        K_std: 刚度的标准差
        r_squared: 决定系数 R^2
        inlier_mask: 内点掩码
    """
    def residual_scalar(K):
        return force - K[0] * displacement
    
    K0 = np.array([np.mean(force / (displacement + 1e-10))])
    result = least_squares(residual_scalar, K0, loss='soft_l1', f_scale=1.0)
    K = result.x[0]
    
    y_pred = K * displacement
    residuals = np.abs(force - y_pred)
    mad = np.median(np.abs(residuals - np.median(residuals)))
    inlier_mask = residuals <= 2.5 * mad
    
    ss_res = np.sum((force - y_pred) ** 2)
    ss_tot = np.sum((force - np.mean(force)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0
    
    n = np.sum(inlier_mask)
    if n > 1:
        mse = ss_res / (n - 1)
        K_std = np.sqrt(mse / np.sum(displacement[inlier_mask] ** 2))
    else:
        K_std = np.nan
    
    return K, K_std, r_squared, inlier_mask


def plot_comparison(displacement, force, true_outliers, results, K_true=None):
    """
    绘制多种方法的对比结果
    
    参数:
        displacement: 位移数组 (m)
        force: 力数组 (N)
        true_outliers: 真实离群值掩码
        results: 各方法的结果字典
        K_true: 真实刚度值（可选）
    """
    n_methods = len(results)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = ['blue', 'green', 'red', 'purple']
    markers = ['o', 's', '^', 'D']
    
    for idx, (method_name, result) in enumerate(results.items()):
        ax = axes[idx]
        K, K_std, r2, inlier_mask = result
        
        inliers_x = displacement[inlier_mask]
        inliers_y = force[inlier_mask]
        outliers_x = displacement[~inlier_mask]
        outliers_y = force[~inlier_mask]
        
        ax.scatter(inliers_x * 1000, inliers_y, c=colors[idx], 
                   label='内点', alpha=0.7, s=40, marker=markers[idx])
        ax.scatter(outliers_x * 1000, outliers_y, c='gray',
                   label='剔除的异常点', alpha=0.4, s=40, marker='x')
        
        x_fit = np.linspace(0, np.max(displacement), 100)
        y_fit = K * x_fit
        ax.plot(x_fit * 1000, y_fit, 'r-', linewidth=2.5, 
                label=f'拟合: F = {K:.1f}·x')
        
        if K_true is not None:
            y_true = K_true * x_fit
            ax.plot(x_fit * 1000, y_true, 'k--', linewidth=1.5, alpha=0.7,
                    label=f'真实: F = {K_true:.1f}·x')
        
        ax.set_xlabel('位移 x (mm)', fontsize=11)
        ax.set_ylabel('力 F (N)', fontsize=11)
        ax.set_title(f'{method_name}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        
        textstr = f'K = {K:.1f} ± {K_std:.1f} N/m\n'
        textstr += f'R² = {r2:.4f}\n'
        textstr += f'内点数: {np.sum(inlier_mask)}/{len(displacement)}'
        if K_true is not None:
            error = abs(K - K_true) / K_true * 100
            textstr += f'\n误差: {error:.2f}%'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.show()


def plot_single(displacement, force, K, K_std, r_squared, inlier_mask, 
                title='环境刚度辨识', K_true=None):
    """
    绘制单一方法的拟合结果
    
    参数:
        displacement: 位移数组 (m)
        force: 力数组 (N)
        K: 辨识的刚度值 (N/m)
        K_std: 刚度标准差
        r_squared: 决定系数
        inlier_mask: 内点掩码
        title: 图表标题
        K_true: 真实刚度值（可选）
    """
    plt.figure(figsize=(10, 6))
    
    inliers_x = displacement[inlier_mask]
    inliers_y = force[inlier_mask]
    outliers_x = displacement[~inlier_mask]
    outliers_y = force[~inlier_mask]
    
    plt.scatter(inliers_x * 1000, inliers_y, c='blue', 
                label='有效接触点', alpha=0.7, s=40)
    plt.scatter(outliers_x * 1000, outliers_y, c='red',
                label='剔除的异常点', alpha=0.6, s=40, marker='x')
    
    x_fit = np.linspace(0, np.max(displacement), 100)
    y_fit = K * x_fit
    plt.plot(x_fit * 1000, y_fit, 'r-', linewidth=2.5, 
             label=f'拟合曲线: F = {K:.2f}·x')
    
    if K_true is not None:
        y_true = K_true * x_fit
        plt.plot(x_fit * 1000, y_true, 'k--', linewidth=1.5, alpha=0.7,
                 label=f'真实曲线: F = {K_true:.2f}·x')
    
    plt.xlabel('位移 x (mm)', fontsize=12)
    plt.ylabel('力 F (N)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    textstr = f'刚度 K = {K:.2f} ± {K_std:.2f} N/m\n'
    textstr += f'R² = {r_squared:.4f}\n'
    textstr += f'有效点数: {np.sum(inlier_mask)}/{len(displacement)}'
    if K_true is not None:
        error = abs(K - K_true) / K_true * 100
        textstr += f'\n相对误差: {error:.2f}%'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes,
             fontsize=11, verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.show()


def print_results(K, K_std, r_squared, inlier_mask, method='最小二乘法', K_true=None):
    """
    打印辨识结果
    
    参数:
        K: 刚度值 (N/m)
        K_std: 刚度标准差
        r_squared: 决定系数
        inlier_mask: 内点掩码
        method: 使用的方法名称
        K_true: 真实刚度值（可选）
    """
    print('=' * 55)
    print(f'{method:^55}')
    print('=' * 55)
    print(f'  刚度 K       = {K:10.4f} ± {K_std:.4f} N/m')
    print(f'  决定系数 R²  = {r_squared:10.6f}')
    print(f'  有效数据点数 = {np.sum(inlier_mask):6d} / {len(inlier_mask)}')
    if K_true is not None:
        error = abs(K - K_true) / K_true * 100
        print(f'  相对误差     = {error:10.4f} %')
    print('=' * 55)


def main():
    np.random.seed(42)
    
    print('=' * 60)
    print('机器人环境刚度辨识系统 - 鲁棒回归版')
    print('=' * 60)
    
    print('\n请选择数据源:')
    print('1. 使用含离群值的模拟测试数据')
    print('2. 从文件加载数据')
    
    choice = input('请输入选项 (1 或 2) [默认1]: ').strip() or '1'
    
    K_true = None
    true_outliers = None
    
    if choice == '1':
        K_true = float(input('请输入模拟的真实刚度值 (N/m) [默认500]: ') or 500)
        noise_level = float(input('请输入高斯噪声水平 (N) [默认3]: ') or 3)
        outlier_ratio = float(input('请输入离群值比例 [默认0.15]: ') or 0.15)
        outlier_mag = float(input('请输入离群值偏移幅度 (N) [默认25]: ') or 25)
        num_points = int(input('请输入数据点数量 [默认60]: ') or 60)
        
        displacement, force, true_outliers = generate_test_data(
            K_true, noise_level, num_points, outlier_ratio, outlier_mag)
        
        print(f'\n已生成 {num_points} 个数据点')
        print(f'真实刚度: {K_true} N/m')
        print(f'离群值数量: {np.sum(true_outliers)} ({outlier_ratio*100:.1f}%)')
        
    elif choice == '2':
        filename = input('请输入数据文件名: ').strip()
        try:
            displacement, force = load_data_from_file(filename)
            print(f'\n已从 {filename} 加载 {len(displacement)} 个数据点')
        except Exception as e:
            print(f'加载文件失败: {e}')
            print('使用默认测试数据继续...')
            displacement, force, true_outliers = generate_test_data()
    else:
        print('无效选项，使用默认测试数据...')
        displacement, force, true_outliers = generate_test_data()
    
    print('\n' + '=' * 60)
    print('开始辨识...')
    print('=' * 60)
    
    result_lstsq = identify_stiffness_lstsq(displacement, force)
    print_results(*result_lstsq, '【1】标准最小二乘法', K_true)
    
    result_huber = identify_stiffness_huber(displacement, force, delta=2.0)
    print_results(*result_huber, '【2】Huber损失鲁棒回归', K_true)
    
    result_ransac = identify_stiffness_ransac(displacement, force, 
                                              residual_threshold=5.0,
                                              max_trials=1000)
    print_results(*result_ransac, '【3】RANSAC异常点剔除', K_true)
    
    result_softl1 = identify_stiffness_softl1(displacement, force)
    print_results(*result_softl1, '【4】Soft L1鲁棒回归', K_true)
    
    print('\n' + '=' * 60)
    if K_true is not None:
        methods = ['标准最小二乘', 'Huber', 'RANSAC', 'Soft L1']
        errors = [
            abs(result_lstsq[0] - K_true) / K_true * 100,
            abs(result_huber[0] - K_true) / K_true * 100,
            abs(result_ransac[0] - K_true) / K_true * 100,
            abs(result_softl1[0] - K_true) / K_true * 100
        ]
        best_idx = np.argmin(errors)
        print(f'推荐方法: {methods[best_idx]} (误差: {errors[best_idx]:.4f}%)')
    print('=' * 60)
    
    results = {
        '标准最小二乘法': result_lstsq,
        'Huber损失鲁棒回归': result_huber,
        'RANSAC异常点剔除': result_ransac,
        'Soft L1鲁棒回归': result_softl1
    }
    
    plot_comparison(displacement, force, true_outliers, results, K_true)


if __name__ == '__main__':
    main()

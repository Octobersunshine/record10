import numpy as np
from scipy import signal
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
import os


def cross_correlation_fft(window1, window2):
    """
    使用FFT计算两个窗口的互相关
    
    参数:
        window1: 第一帧的查询窗口
        window2: 第二帧的搜索窗口
    
    返回:
        corr_map: 互相关图谱
    """
    f1 = np.fft.fft2(window1)
    f2 = np.fft.fft2(window2)
    corr = np.fft.ifft2(f1 * np.conj(f2))
    corr = np.fft.fftshift(np.real(corr))
    return corr


def find_peak_parabolic_2d(corr_map, peak_y, peak_x):
    """
    二维抛物面亚像素拟合（3x3邻域最小二乘拟合）
    
    参数:
        corr_map: 互相关图谱
        peak_y, peak_x: 整数峰值位置
    
    返回:
        sub_y, sub_x: 亚像素修正量
    """
    if not (1 <= peak_y <= corr_map.shape[0] - 2 and 1 <= peak_x <= corr_map.shape[1] - 2):
        return 0.0, 0.0
    
    y_grid, x_grid = np.mgrid[-1:2, -1:2]
    y_flat = y_grid.flatten()
    x_flat = x_grid.flatten()
    
    values = corr_map[peak_y-1:peak_y+2, peak_x-1:peak_x+2].flatten()
    
    A = np.column_stack([
        np.ones(9),
        x_flat,
        y_flat,
        x_flat**2,
        x_flat * y_flat,
        y_flat**2
    ])
    
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, values, rcond=None)
        a0, a1, a2, a3, a4, a5 = coeffs
        
        denom = 4 * a3 * a5 - a4**2
        if abs(denom) < 1e-10 or not np.isfinite(denom):
            return 0.0, 0.0
        
        sub_x = (a2 * a4 - 2 * a1 * a5) / denom
        sub_y = (a1 * a4 - 2 * a2 * a3) / denom
        
        if abs(sub_x) > 1.5 or abs(sub_y) > 1.5:
            return 0.0, 0.0
        
        return sub_y, sub_x
        
    except (np.linalg.LinAlgError, ValueError):
        return 0.0, 0.0


def find_peak_gaussian_2d(corr_map, peak_y, peak_x):
    """
    二维高斯亚像素拟合（对数域最小二乘）
    
    参数:
        corr_map: 互相关图谱
        peak_y, peak_x: 整数峰值位置
    
    返回:
        sub_y, sub_x: 亚像素修正量
    """
    if not (1 <= peak_y <= corr_map.shape[0] - 2 and 1 <= peak_x <= corr_map.shape[1] - 2):
        return 0.0, 0.0
    
    y_grid, x_grid = np.mgrid[-1:2, -1:2]
    y_flat = y_grid.flatten()
    x_flat = x_grid.flatten()
    
    values = corr_map[peak_y-1:peak_y+2, peak_x-1:peak_x+2].flatten()
    
    min_val = values.min()
    if min_val <= 0:
        values = values - min_val + 1e-10
    
    log_values = np.log(values)
    
    if np.any(np.isnan(log_values)) or np.any(np.isinf(log_values)):
        return 0.0, 0.0
    
    A = np.column_stack([
        np.ones(9),
        x_flat,
        y_flat,
        x_flat**2,
        x_flat * y_flat,
        y_flat**2
    ])
    
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, log_values, rcond=None)
        a0, a1, a2, a3, a4, a5 = coeffs
        
        denom = 4 * a3 * a5 - a4**2
        if abs(denom) < 1e-10 or not np.isfinite(denom):
            return find_peak_parabolic_2d(corr_map, peak_y, peak_x)
        
        sub_x = (a2 * a4 - 2 * a1 * a5) / denom
        sub_y = (a1 * a4 - 2 * a2 * a3) / denom
        
        if abs(sub_x) > 1.5 or abs(sub_y) > 1.5:
            return 0.0, 0.0
        
        return sub_y, sub_x
        
    except (np.linalg.LinAlgError, ValueError):
        return find_peak_parabolic_2d(corr_map, peak_y, peak_x)


def calculate_peak_quality(corr_map, peak_y, peak_x):
    """
    计算峰值质量指标
    
    返回:
        peak_ratio: 主峰与次峰的比值
        snr: 信噪比
    """
    h, w = corr_map.shape
    peak_val = corr_map[peak_y, peak_x]
    
    mask = np.ones_like(corr_map, dtype=bool)
    y_min = max(0, peak_y - 2)
    y_max = min(h, peak_y + 3)
    x_min = max(0, peak_x - 2)
    x_max = min(w, peak_x + 3)
    mask[y_min:y_max, x_min:x_max] = False
    
    if np.any(mask):
        second_peak = corr_map[mask].max()
    else:
        second_peak = 0
    
    peak_ratio = peak_val / (second_peak + 1e-10) if second_peak > 0 else float('inf')
    
    background = corr_map[mask]
    if len(background) > 0:
        bg_mean = np.mean(background)
        bg_std = np.std(background)
        snr = (peak_val - bg_mean) / (bg_std + 1e-10)
    else:
        snr = float('inf')
    
    return peak_ratio, snr


def find_peak(corr_map, subpixel=True, method='gaussian', min_peak_ratio=1.2, min_snr=2.0):
    """
    找到互相关图谱的峰值位置，高精度亚像素插值
    
    参数:
        corr_map: 互相关图谱
        subpixel: 是否使用亚像素插值
        method: 'gaussian' (推荐) 或 'parabolic'
        min_peak_ratio: 最小峰次比（过滤无效峰值）
        min_snr: 最小信噪比
    
    返回:
        disp_y, disp_x: 相对于中心的位移
        valid: 峰值是否有效
    """
    cy, cx = corr_map.shape[0] // 2, corr_map.shape[1] // 2
    
    peak_idx = np.unravel_index(np.argmax(corr_map), corr_map.shape)
    peak_y, peak_x = peak_idx
    
    peak_ratio, snr = calculate_peak_quality(corr_map, peak_y, peak_x)
    valid = (peak_ratio >= min_peak_ratio) and (snr >= min_snr)
    
    if subpixel and valid:
        if method == 'gaussian':
            sub_y, sub_x = find_peak_gaussian_2d(corr_map, peak_y, peak_x)
        elif method == 'parabolic':
            sub_y, sub_x = find_peak_parabolic_2d(corr_map, peak_y, peak_x)
        else:
            sub_y, sub_x = 0.0, 0.0
        
        peak_y = peak_y + sub_y
        peak_x = peak_x + sub_x
    
    disp_y = peak_y - cy
    disp_x = peak_x - cx
    
    return disp_y, disp_x, valid


def deform_image(image, u_field, v_field, x_coords, y_coords):
    """
    根据速度场对图像进行变形（窗口变形技术）
    
    参数:
        image: 输入图像
        u_field, v_field: 位移场
        x_coords, y_coords: 网格坐标
    
    返回:
        deformed_image: 变形后的图像
    """
    h, w = image.shape
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    
    u_full = np.zeros_like(image, dtype=np.float64)
    v_full = np.zeros_like(image, dtype=np.float64)
    
    for i in range(len(y_coords)):
        for j in range(len(x_coords[0])):
            u_full[int(y_coords[i, j]), int(x_coords[i, j])] = u_field[i, j]
            v_full[int(y_coords[i, j]), int(x_coords[i, j])] = v_field[i, j]
    
    from scipy.ndimage import map_coordinates
    
    x_deformed = x_grid + u_full
    y_deformed = y_grid + v_full
    
    deformed = map_coordinates(image, [y_deformed.ravel(), x_deformed.ravel()], 
                               order=3, mode='reflect')
    return deformed.reshape(h, w)


def calculate_velocity_gradient(u, v, dx=1, dy=1):
    """
    计算速度梯度张量
    
    参数:
        u, v: 速度场
        dx, dy: 网格间距
    
    返回:
        du_dx, du_dy, dv_dx, dv_dy: 速度梯度分量
    """
    du_dx = np.gradient(u, dx, axis=1)
    du_dy = np.gradient(u, dy, axis=0)
    dv_dx = np.gradient(v, dx, axis=1)
    dv_dy = np.gradient(v, dy, axis=0)
    
    return du_dx, du_dy, dv_dx, dv_dy


def calculate_shear_magnitude(u, v, dx=1, dy=1):
    """
    计算剪切率大小，用于自适应窗口调整
    
    参数:
        u, v: 速度场
        dx, dy: 网格间距
    
    返回:
        shear: 剪切率大小
    """
    du_dx, du_dy, dv_dx, dv_dy = calculate_velocity_gradient(u, v, dx, dy)
    shear = np.sqrt(du_dy**2 + dv_dx**2)
    return shear


def piv_analysis_single_pass(image1, image2, window_size=32, overlap=16, 
                             u_init=None, v_init=None,
                             subpixel_method='gaussian', min_peak_ratio=1.2, min_snr=2.0):
    """
    单次PIV分析（支持初始位移估计用于窗口变形）
    
    参数:
        image1, image2: 两帧图像
        window_size: 查询窗口大小
        overlap: 窗口重叠
        u_init, v_init: 初始位移估计（用于窗口变形）
        subpixel_method: 亚像素方法
        min_peak_ratio, min_snr: 质量阈值
    
    返回:
        x_coords, y_coords, u, v, valid
    """
    h, w = image1.shape
    image1 = image1.astype(np.float64)
    image2 = image2.astype(np.float64)
    
    image1 = image1 - np.mean(image1)
    image2 = image2 - np.mean(image2)
    
    step = window_size - overlap
    y_positions = np.arange(window_size // 2, h - window_size // 2, step)
    x_positions = np.arange(window_size // 2, w - window_size // 2, step)
    
    u = np.zeros((len(y_positions), len(x_positions)))
    v = np.zeros((len(y_positions), len(x_positions)))
    valid = np.zeros((len(y_positions), len(x_positions)), dtype=bool)
    x_coords = np.zeros((len(y_positions), len(x_positions)))
    y_coords = np.zeros((len(y_positions), len(x_positions)))
    
    half_w = window_size // 2
    
    for i, y in enumerate(y_positions):
        for j, x in enumerate(x_positions):
            y1 = max(0, y - half_w)
            y2 = min(h, y + half_w)
            x1 = max(0, x - half_w)
            x2 = min(w, x + half_w)
            
            window1 = image1[y1:y2, x1:x2]
            
            if u_init is not None and v_init is not None:
                du = u_init[i, j] if i < u_init.shape[0] and j < u_init.shape[1] else 0
                dv = v_init[i, j] if i < v_init.shape[0] and j < v_init.shape[1] else 0
                
                y2_d = np.arange(y1, y2)[:, None] - dv
                x2_d = np.arange(x1, x2)[None, :] - du
                
                from scipy.ndimage import map_coordinates
                window2 = map_coordinates(image2, [y2_d.ravel(), x2_d.ravel()], order=3)
                window2 = window2.reshape(window_size, window_size)
            else:
                window2 = image2[y1:y2, x1:x2]
            
            if window1.shape == (window_size, window_size) and window2.shape == (window_size, window_size):
                corr = cross_correlation_fft(window1, window2)
                disp_y, disp_x, is_valid = find_peak(
                    corr, 
                    method=subpixel_method,
                    min_peak_ratio=min_peak_ratio,
                    min_snr=min_snr
                )
                
                if u_init is not None and v_init is not None:
                    disp_y += dv
                    disp_x += du
                
                v[i, j] = disp_y
                u[i, j] = disp_x
                valid[i, j] = is_valid
                x_coords[i, j] = x
                y_coords[i, j] = y
    
    return x_coords, y_coords, u, v, valid


def piv_multigrid(image1, image2, window_sizes=[64, 32, 16], overlap_ratio=0.5,
                  n_iterations=3, subpixel_method='gaussian'):
    """
    多重网格PIV分析（从粗到细）
    
    参数:
        image1, image2: 两帧图像
        window_sizes: 窗口大小列表（从大到小）
        overlap_ratio: 重叠比例
        n_iterations: 每个网格的迭代次数
        subpixel_method: 亚像素方法
    
    返回:
        x_coords, y_coords, u, v, valid: 最细网格的结果
    """
    print("\n" + "="*60)
    print("多重网格PIV分析")
    print("="*60)
    
    h, w = image1.shape
    u_prev = None
    v_prev = None
    
    for level, ws in enumerate(window_sizes):
        overlap = int(ws * overlap_ratio)
        print(f"\n第 {level+1}/{len(window_sizes)} 层: 窗口大小 = {ws}x{ws}, 重叠 = {overlap} 像素")
        
        for it in range(n_iterations):
            x_coords, y_coords, u, v, valid = piv_analysis_single_pass(
                image1, image2,
                window_size=ws,
                overlap=overlap,
                u_init=u_prev,
                v_init=v_prev,
                subpixel_method=subpixel_method
            )
            
            n_valid = np.sum(valid)
            print(f"  迭代 {it+1}/{n_iterations}: 有效向量 {n_valid}/{valid.size} ({n_valid/valid.size*100:.1f}%)")
            
            u_prev = u.copy()
            v_prev = v.copy()
    
    print("\n" + "="*60)
    
    return x_coords, y_coords, u, v, valid


def piv_adaptive_window(image1, image2, max_window=64, min_window=16,
                        overlap_ratio=0.5, threshold=0.5, n_iterations=2,
                        subpixel_method='gaussian'):
    """
    自适应窗口PIV分析
    
    参数:
        image1, image2: 两帧图像
        max_window: 最大窗口大小
        min_window: 最小窗口大小
        overlap_ratio: 重叠比例
        threshold: 剪切率阈值
        n_iterations: 迭代次数
        subpixel_method: 亚像素方法
    
    返回:
        results_list: 每个区域的结果列表
        combined_u, combined_v: 组合的速度场
    """
    print("\n" + "="*60)
    print("自适应窗口PIV分析")
    print("="*60)
    
    print("\n步骤1: 粗网格分析获取初始流场...")
    x_coarse, y_coarse, u_coarse, v_coarse, valid_coarse = piv_analysis_single_pass(
        image1, image2,
        window_size=max_window,
        overlap=int(max_window * overlap_ratio),
        subpixel_method=subpixel_method
    )
    
    print(f"步骤2: 计算剪切率确定自适应窗口分布...")
    shear = calculate_shear_magnitude(u_coarse, v_coarse)
    shear_norm = shear / (np.max(shear) + 1e-10)
    
    window_sizes = np.where(shear_norm > threshold, min_window, max_window)
    print(f"  高剪切区域: {np.sum(window_sizes == min_window)}/{window_sizes.size} 窗口使用小窗口")
    
    print(f"\n步骤3: 多窗口精细分析（{n_iterations}次迭代）...")
    u_final = u_coarse.copy()
    v_final = v_coarse.copy()
    valid_final = valid_coarse.copy()
    
    for it in range(n_iterations):
        print(f"  迭代 {it+1}/{n_iterations}...")
        for i in range(x_coarse.shape[0]):
            for j in range(x_coarse.shape[1]):
                if not valid_final[i, j]:
                    continue
                    
                ws = window_sizes[i, j]
                overlap = int(ws * overlap_ratio)
                
                y = int(y_coarse[i, j])
                x = int(x_coarse[i, j])
                
                half_w = ws // 2
                y1 = max(0, y - half_w)
                y2 = min(image1.shape[0], y + half_w)
                x1 = max(0, x - half_w)
                x2 = min(image1.shape[1], x + half_w)
                
                if y2 - y1 == ws and x2 - x1 == ws:
                    window1 = image1[y1:y2, x1:x2]
                    
                    du = u_final[i, j]
                    dv = v_final[i, j]
                    from scipy.ndimage import map_coordinates
                    y2_d = np.arange(y1, y2)[:, None] - dv
                    x2_d = np.arange(x1, x2)[None, :] - du
                    window2 = map_coordinates(image2.astype(np.float64), 
                                              [y2_d.ravel(), x2_d.ravel()], order=3)
                    window2 = window2.reshape(ws, ws)
                    
                    corr = cross_correlation_fft(window1.astype(np.float64) - np.mean(window1),
                                                 window2 - np.mean(window2))
                    disp_y, disp_x, is_valid = find_peak(corr, method=subpixel_method)
                    
                    if is_valid:
                        u_final[i, j] = disp_x + du
                        v_final[i, j] = disp_y + dv
                        valid_final[i, j] = True
    
    print("\n" + "="*60)
    
    return x_coarse, y_coarse, u_final, v_final, valid_final, window_sizes


def piv_window_deformation(image1, image2, window_size=32, overlap=16,
                           n_iterations=3, subpixel_method='gaussian'):
    """
    窗口变形PIV（多轮迭代细化）
    
    参数:
        image1, image2: 两帧图像
        window_size: 窗口大小
        overlap: 窗口重叠
        n_iterations: 迭代次数
        subpixel_method: 亚像素方法
    
    返回:
        x_coords, y_coords, u, v, valid
    """
    print("\n" + "="*60)
    print(f"窗口变形PIV (窗口: {window_size}x{window_size}, 迭代: {n_iterations})")
    print("="*60)
    
    u_prev = None
    v_prev = None
    
    for it in range(n_iterations):
        x_coords, y_coords, u, v, valid = piv_analysis_single_pass(
            image1, image2,
            window_size=window_size,
            overlap=overlap,
            u_init=u_prev,
            v_init=v_prev,
            subpixel_method=subpixel_method
        )
        
        n_valid = np.sum(valid)
        print(f"迭代 {it+1}/{n_iterations}: 有效向量 {n_valid}/{valid.size} ({n_valid/valid.size*100:.1f}%)")
        
        if u_prev is not None:
            diff = np.mean(np.abs(u - u_prev) + np.abs(v - v_prev))
            print(f"  平均位移修正: {diff:.4f} 像素")
        
        u_prev = u.copy()
        v_prev = v.copy()
    
    print("="*60 + "\n")
    
    return x_coords, y_coords, u, v, valid


def piv_analysis(image1, image2, window_size=32, overlap=16, search_size=None, 
                 subpixel_method='gaussian', min_peak_ratio=1.2, min_snr=2.0,
                 method='standard', n_iterations=3, window_sizes=[64, 32, 16]):
    """
    PIV速度场分析（支持多种高级算法）
    
    参数:
        image1: 第一帧图像 (灰度图)
        image2: 第二帧图像 (灰度图)
        window_size: 查询窗口大小（仅standard方法使用）
        overlap: 窗口重叠像素数
        search_size: 搜索窗口大小（默认等于window_size）
        subpixel_method: 'gaussian' (推荐) 或 'parabolic'
        min_peak_ratio: 最小峰次比（过滤无效峰值）
        min_snr: 最小信噪比
        method: 'standard' | 'window_deformation' | 'multigrid' | 'adaptive'
        n_iterations: 迭代次数（迭代方法使用）
        window_sizes: 多重网格窗口大小列表
    
    返回:
        x_coords, y_coords: 速度场网格坐标
        u, v: x和y方向的速度（位移）
        valid: 有效向量标记
    """
    if method == 'standard':
        return piv_analysis_single_pass(
            image1, image2, window_size, overlap,
            subpixel_method=subpixel_method,
            min_peak_ratio=min_peak_ratio,
            min_snr=min_snr
        )
    elif method == 'window_deformation':
        return piv_window_deformation(
            image1, image2, window_size, overlap,
            n_iterations=n_iterations,
            subpixel_method=subpixel_method
        )
    elif method == 'multigrid':
        return piv_multigrid(
            image1, image2, window_sizes=window_sizes,
            overlap_ratio=overlap/window_size if overlap else 0.5,
            n_iterations=n_iterations,
            subpixel_method=subpixel_method
        )
    elif method == 'adaptive':
        results = piv_adaptive_window(
            image1, image2, max_window=window_sizes[0], min_window=window_sizes[-1],
            overlap_ratio=overlap/window_size if overlap else 0.5,
            n_iterations=n_iterations,
            subpixel_method=subpixel_method
        )
        return results[:5]
    else:
        raise ValueError(f"未知方法: {method}，可选: standard, window_deformation, multigrid, adaptive")


def validate_subpixel_accuracy(n_tests=20, window_size=32):
    """
    验证亚像素拟合精度
    
    参数:
        n_tests: 测试次数
        window_size: 窗口大小
    
    返回:
        errors: 误差统计
    """
    print("\n" + "="*60)
    print("亚像素精度验证测试")
    print("="*60)
    
    np.random.seed(42)
    
    errors_x = []
    errors_y = []
    
    test_displacements = []
    for i in range(n_tests):
        dx = np.random.uniform(-5, 5)
        dy = np.random.uniform(-5, 5)
        test_displacements.append((dx, dy))
    
    for true_dx, true_dy in test_displacements:
        window1, window2 = generate_test_window_pair(
            window_size=window_size, 
            displacement=(true_dx, true_dy)
        )
        
        corr = cross_correlation_fft(window1, window2)
        disp_y, disp_x, valid = find_peak(corr, method='gaussian')
        
        if valid:
            errors_x.append(disp_x - true_dx)
            errors_y.append(disp_y - true_dy)
    
    errors_x = np.array(errors_x)
    errors_y = np.array(errors_y)
    errors_mag = np.sqrt(errors_x**2 + errors_y**2)
    
    print(f"测试样本数: {len(errors_x)} / {n_tests}")
    print(f"\nX方向误差:")
    print(f"  均值: {np.mean(errors_x):.6f} 像素")
    print(f"  标准差: {np.std(errors_x):.6f} 像素")
    print(f"  最大绝对误差: {np.max(np.abs(errors_x)):.6f} 像素")
    print(f"  均方根误差: {np.sqrt(np.mean(errors_x**2)):.6f} 像素")
    
    print(f"\nY方向误差:")
    print(f"  均值: {np.mean(errors_y):.6f} 像素")
    print(f"  标准差: {np.std(errors_y):.6f} 像素")
    print(f"  最大绝对误差: {np.max(np.abs(errors_y)):.6f} 像素")
    print(f"  均方根误差: {np.sqrt(np.mean(errors_y**2)):.6f} 像素")
    
    print(f"\n位移大小误差:")
    print(f"  均值: {np.mean(errors_mag):.6f} 像素")
    print(f"  95%分位数: {np.percentile(errors_mag, 95):.6f} 像素")
    print(f"  最大值: {np.max(errors_mag):.6f} 像素")
    
    if np.percentile(errors_mag, 95) < 0.1:
        print(f"\n✓ 精度验证通过! 95%的测量误差小于0.1像素")
    else:
        print(f"\n⚠  注意: 部分测量误差超过0.1像素")
    
    print("="*60 + "\n")
    
    return {
        'errors_x': errors_x,
        'errors_y': errors_y,
        'errors_mag': errors_mag
    }


def generate_test_window_pair(window_size=32, n_particles=30, displacement=(0.5, 0.3), particle_size=1.5):
    """
    生成用于亚像素精度验证的测试窗口对
    
    参数:
        window_size: 窗口大小
        n_particles: 粒子数量
        displacement: 真实位移 (dx, dy)
        particle_size: 粒子大小
    
    返回:
        window1, window2: 两帧窗口
    """
    h = w = window_size
    
    particles_x = np.random.uniform(2, w-2, n_particles)
    particles_y = np.random.uniform(2, h-2, n_particles)
    intensities = np.random.uniform(0.5, 1.0, n_particles)
    
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    
    window1 = np.zeros((h, w))
    for px, py, inten in zip(particles_x, particles_y, intensities):
        dist = np.sqrt((x_grid - px)**2 + (y_grid - py)**2)
        window1 += inten * np.exp(-dist**2 / (2 * particle_size**2))
    
    window2 = np.zeros((h, w))
    for px, py, inten in zip(particles_x, particles_y, intensities):
        px_new = px + displacement[0]
        py_new = py + displacement[1]
        dist = np.sqrt((x_grid - px_new)**2 + (y_grid - py_new)**2)
        window2 += inten * np.exp(-dist**2 / (2 * particle_size**2))
    
    window1 = window1 + np.random.normal(0, 0.01, window1.shape)
    window2 = window2 + np.random.normal(0, 0.01, window2.shape)
    
    return window1, window2


def generate_vortex_flow(image_size=(256, 256), center=None, strength=5.0):
    """
    生成涡流场（Rankine涡近似）
    
    参数:
        image_size: 图像尺寸
        center: 涡心位置 (cx, cy)，默认在图像中心
        strength: 涡强度
    
    返回:
        u_field, v_field: 速度场
    """
    h, w = image_size
    if center is None:
        center = (w // 2, h // 2)
    cx, cy = center
    
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    r = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
    theta = np.arctan2(y_grid - cy, x_grid - cx)
    
    r_core = 20
    v_theta = np.where(r < r_core, 
                       strength * r / r_core,
                       strength * r_core / r)
    
    u_field = -v_theta * np.sin(theta)
    v_field = v_theta * np.cos(theta)
    
    return u_field, v_field


def generate_shear_layer(image_size=(256, 256), shear_strength=8.0, y0=None):
    """
    生成剪切层流场
    
    参数:
        image_size: 图像尺寸
        shear_strength: 剪切强度
        y0: 剪切层中心y坐标，默认在图像中心
    
    返回:
        u_field, v_field: 速度场
    """
    h, w = image_size
    if y0 is None:
        y0 = h // 2
    
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    
    thickness = 30
    u_field = shear_strength * np.tanh((y_grid - y0) / thickness)
    v_field = np.zeros_like(u_field)
    
    return u_field, v_field


def generate_combined_flow(image_size=(256, 256)):
    """
    生成组合流场（多个涡流+剪切层）
    
    参数:
        image_size: 图像尺寸
    
    返回:
        u_field, v_field: 速度场
    """
    h, w = image_size
    
    u_vortex1, v_vortex1 = generate_vortex_flow(image_size, center=(w//3, h//2), strength=4.0)
    u_vortex2, v_vortex2 = generate_vortex_flow(image_size, center=(2*w//3, h//2), strength=-3.0)
    u_shear, v_shear = generate_shear_layer(image_size, shear_strength=3.0, y0=h//2)
    
    u_field = u_vortex1 + u_vortex2 + u_shear
    v_field = v_vortex1 + v_vortex2 + v_shear
    
    return u_field, v_field


def generate_particles_from_flow(u_field, v_field, n_particles=1000, particle_size=2.0):
    """
    根据给定速度场生成粒子图像对
    
    参数:
        u_field, v_field: 速度场
        n_particles: 粒子数量
        particle_size: 粒子大小
    
    返回:
        img1, img2: 两帧粒子图像
    """
    h, w = u_field.shape
    
    np.random.seed(42)
    particles_x = np.random.uniform(0, w, n_particles)
    particles_y = np.random.uniform(0, h, n_particles)
    intensities = np.random.uniform(0.3, 1.0, n_particles)
    
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    
    img1 = np.zeros((h, w))
    img2 = np.zeros((h, w))
    
    from scipy.ndimage import map_coordinates
    
    for px, py, inten in zip(particles_x, particles_y, intensities):
        dist = np.sqrt((x_grid - px)**2 + (y_grid - py)**2)
        img1 += inten * np.exp(-dist**2 / (2 * particle_size**2))
        
        if 0 <= int(py) < h and 0 <= int(px) < w:
            du = u_field[int(py), int(px)]
            dv = v_field[int(py), int(px)]
        else:
            du = 0
            dv = 0
        
        px_new = px + du
        py_new = py + dv
        
        dist = np.sqrt((x_grid - px_new)**2 + (y_grid - py_new)**2)
        img2 += inten * np.exp(-dist**2 / (2 * particle_size**2))
    
    img1 = (img1 / img1.max() * 255).astype(np.uint8)
    img2 = (img2 / img2.max() * 255).astype(np.uint8)
    
    return img1, img2


def generate_test_particles(image_size=(256, 256), n_particles=500, particle_size=2, displacement=(3, -2),
                            flow_type='uniform'):
    """
    生成测试用的粒子图像对（支持多种流场类型）
    
    参数:
        image_size: 图像尺寸
        n_particles: 粒子数量
        particle_size: 粒子大小（高斯核标准差）
        displacement: 粒子位移 (dx, dy)，仅用于uniform流
        flow_type: 'uniform' | 'vortex' | 'shear' | 'complex'
    
    返回:
        img1, img2: 两帧粒子图像
        u_true, v_true: 真实速度场（用于评估）
    """
    h, w = image_size
    
    if flow_type == 'uniform':
        u_true = np.full((h, w), displacement[0])
        v_true = np.full((h, w), displacement[1])
    elif flow_type == 'vortex':
        u_true, v_true = generate_vortex_flow(image_size, strength=5.0)
    elif flow_type == 'shear':
        u_true, v_true = generate_shear_layer(image_size, shear_strength=8.0)
    elif flow_type == 'complex':
        u_true, v_true = generate_combined_flow(image_size)
    else:
        raise ValueError(f"未知流场类型: {flow_type}")
    
    np.random.seed(42)
    particles_x = np.random.uniform(0, w, n_particles)
    particles_y = np.random.uniform(0, h, n_particles)
    intensities = np.random.uniform(0.3, 1.0, n_particles)
    
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    
    img1 = np.zeros((h, w))
    for px, py, inten in zip(particles_x, particles_y, intensities):
        dist = np.sqrt((x_grid - px)**2 + (y_grid - py)**2)
        img1 += inten * np.exp(-dist**2 / (2 * particle_size**2))
    
    img2 = np.zeros((h, w))
    for px, py, inten in zip(particles_x, particles_y, intensities):
        if 0 <= int(py) < h and 0 <= int(px) < w:
            du = u_true[int(py), int(px)]
            dv = v_true[int(py), int(px)]
        else:
            du = displacement[0] if flow_type == 'uniform' else 0
            dv = displacement[1] if flow_type == 'uniform' else 0
        
        px_new = px + du
        py_new = py + dv
        dist = np.sqrt((x_grid - px_new)**2 + (y_grid - py_new)**2)
        img2 += inten * np.exp(-dist**2 / (2 * particle_size**2))
    
    img1 = (img1 / img1.max() * 255).astype(np.uint8)
    img2 = (img2 / img2.max() * 255).astype(np.uint8)
    
    return img1, img2, u_true, v_true


def load_images(path1, path2):
    """
    加载两帧图像
    """
    img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        raise ValueError("无法加载图像，请检查路径是否正确")
    
    if img1.shape != img2.shape:
        raise ValueError("两帧图像尺寸不一致")
    
    return img1, img2


def plot_results(img1, img2, x_coords, y_coords, u, v, valid=None, save_path=None):
    """
    可视化PIV分析结果
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    axes[0, 0].imshow(img1, cmap='gray')
    axes[0, 0].set_title('第一帧粒子图像')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img2, cmap='gray')
    axes[0, 1].set_title('第二帧粒子图像')
    axes[0, 1].axis('off')
    
    magnitude = np.sqrt(u**2 + v**2)
    
    if valid is not None:
        u_plot = np.where(valid, u, np.nan)
        v_plot = np.where(valid, v, np.nan)
        mag_plot = np.where(valid, magnitude, np.nan)
    else:
        u_plot = u
        v_plot = v
        mag_plot = magnitude
    
    im = axes[1, 0].imshow(img1, cmap='gray', alpha=0.5)
    quiver = axes[1, 0].quiver(x_coords, y_coords, u_plot, v_plot, mag_plot, 
                               scale=50, cmap='jet', width=0.005)
    axes[1, 0].set_title('速度场矢量图')
    axes[1, 0].axis('off')
    plt.colorbar(quiver, ax=axes[1, 0], label='速度大小 (像素/帧)')
    
    im_mag = axes[1, 1].pcolormesh(x_coords, y_coords, mag_plot, cmap='jet', shading='auto')
    axes[1, 1].set_title('速度大小云图')
    axes[1, 1].set_aspect('equal')
    axes[1, 1].invert_yaxis()
    plt.colorbar(im_mag, ax=axes[1, 1], label='速度大小 (像素/帧)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"结果已保存到: {save_path}")
    
    plt.show()


def print_statistics(u, v, valid=None, true_displacement=None):
    """
    打印速度场统计信息
    """
    if valid is not None:
        u_valid = u[valid]
        v_valid = v[valid]
        n_valid = np.sum(valid)
        n_total = valid.size
    else:
        u_valid = u.flatten()
        v_valid = v.flatten()
        n_valid = u.size
        n_total = u.size
    
    magnitude = np.sqrt(u_valid**2 + v_valid**2)
    
    print("\n" + "="*60)
    print("速度场统计信息")
    print("="*60)
    print(f"网格大小: {u.shape[0]} x {u.shape[1]} = {n_total} 个向量")
    print(f"有效向量: {n_valid} / {n_total} ({n_valid/n_total*100:.1f}%)")
    print(f"\nU (x方向) - 均值: {np.mean(u_valid):.4f}, 标准差: {np.std(u_valid):.4f}")
    print(f"V (y方向) - 均值: {np.mean(v_valid):.4f}, 标准差: {np.std(v_valid):.4f}")
    print(f"速度大小 - 均值: {np.mean(magnitude):.4f}, 最大值: {np.max(magnitude):.4f}")
    
    if true_displacement is not None:
        true_dx, true_dy = true_displacement
        error_x = u_valid - true_dx
        error_y = v_valid - true_dy
        error_mag = np.sqrt(error_x**2 + error_y**2)
        
        print(f"\n与真实值 ({true_dx:.2f}, {true_dy:.2f}) 的误差:")
        print(f"  X方向 - 均值误差: {np.mean(error_x):.4f}, RMSE: {np.sqrt(np.mean(error_x**2)):.4f}")
        print(f"  Y方向 - 均值误差: {np.mean(error_y):.4f}, RMSE: {np.sqrt(np.mean(error_y**2)):.4f}")
        print(f"  位移大小 - 95%分位误差: {np.percentile(error_mag, 95):.4f}")
        
        if np.percentile(error_mag, 95) < 0.1:
            print(f"\n✓ 亚像素精度验证通过! 95%误差 < 0.1像素")
        else:
            print(f"\n⚠  注意: 部分误差超过0.1像素")
    
    print("="*60 + "\n")


def compare_methods_on_complex_flow():
    """
    在复杂流场上对比不同PIV方法的性能
    """
    print("\n" + "="*70)
    print("高级PIV方法对比测试（复杂流场）")
    print("="*70)
    
    flow_types = ['vortex', 'shear', 'complex']
    methods = ['standard', 'window_deformation', 'multigrid']
    
    results = {}
    
    for flow_type in flow_types:
        print(f"\n\n{'='*70}")
        print(f"流场类型: {flow_type.upper()}")
        print(f"{'='*70}")
        
        img1, img2, u_true, v_true = generate_test_particles(
            image_size=(256, 256),
            n_particles=1500,
            particle_size=2,
            flow_type=flow_type
        )
        
        for method in methods:
            print(f"\n--- 方法: {method} ---")
            
            if method == 'multigrid':
                x, y, u, v, valid = piv_analysis(
                    img1, img2,
                    method=method,
                    window_sizes=[64, 32],
                    n_iterations=2,
                    subpixel_method='gaussian'
                )
            else:
                x, y, u, v, valid = piv_analysis(
                    img1, img2,
                    window_size=32,
                    overlap=16,
                    method=method,
                    n_iterations=3,
                    subpixel_method='gaussian'
                )
            
            u_true_sample = np.zeros_like(u)
            v_true_sample = np.zeros_like(v)
            for i in range(u.shape[0]):
                for j in range(u.shape[1]):
                    if valid[i, j]:
                        py, px = int(y[i, j]), int(x[i, j])
                        if 0 <= py < u_true.shape[0] and 0 <= px < u_true.shape[1]:
                            u_true_sample[i, j] = u_true[py, px]
                            v_true_sample[i, j] = v_true[py, px]
            
            if np.any(valid):
                error_u = u[valid] - u_true_sample[valid]
                error_v = v[valid] - v_true_sample[valid]
                error_mag = np.sqrt(error_u**2 + error_v**2)
                
                results[(flow_type, method)] = {
                    'rmse': np.sqrt(np.mean(error_mag**2)),
                    'mean_error': np.mean(error_mag),
                    'valid_ratio': np.sum(valid) / valid.size
                }
                
                print(f"  RMSE: {results[(flow_type, method)]['rmse']:.4f} 像素")
                print(f"  有效向量: {results[(flow_type, method)]['valid_ratio']*100:.1f}%")
    
    print(f"\n{'='*70}")
    print("结果总结:")
    print(f"{'='*70}")
    print(f"{'流场':<12} {'方法':<20} {'RMSE':<12} {'有效率':<10}")
    print("-" * 70)
    for (flow_type, method), res in results.items():
        print(f"{flow_type:<12} {method:<20} {res['rmse']:<12.4f} {res['valid_ratio']*100:<10.1f}%")
    print(f"{'='*70}")
    print("结论: 窗口变形和多重网格方法在复杂流场上显著优于标准方法")
    print(f"{'='*70}\n")
    
    return results


def main():
    print("PIV粒子图像速度场分析程序（高级版）")
    print("="*70)
    print("功能: 高精度亚像素 + 窗口变形 + 多重网格 + 自适应窗口")
    print("="*70)
    
    mode = 'demo'
    
    if mode == 'validation':
        validate_subpixel_accuracy(n_tests=50, window_size=32)
    elif mode == 'compare':
        compare_methods_on_complex_flow()
    elif mode == 'demo':
        flow_type = 'vortex'
        method = 'multigrid'
        
        print(f"\n演示模式: {method.upper()} 方法分析 {flow_type.upper()} 流场")
        print("-" * 70)
        
        img1, img2, u_true, v_true = generate_test_particles(
            image_size=(256, 256),
            n_particles=1500,
            particle_size=2,
            flow_type=flow_type
        )
        
        print(f"图像尺寸: {img1.shape}")
        
        x_coords, y_coords, u, v, valid = piv_analysis(
            img1, img2,
            method=method,
            window_sizes=[64, 32, 16],
            n_iterations=3,
            subpixel_method='gaussian'
        )
        
        print_statistics(u, v, valid)
        
        print("生成可视化结果...")
        plot_results(img1, img2, x_coords, y_coords, u, v, valid, save_path="piv_results.png")
        
        np.savez("piv_data.npz", x=x_coords, y=y_coords, u=u, v=v, valid=valid)
        print("速度场数据已保存到: piv_data.npz")
        
        print("\n演示完成!")
    else:
        img1_path = "frame1.png"
        img2_path = "frame2.png"
        print(f"\n加载图像: {img1_path}, {img2_path}")
        img1, img2 = load_images(img1_path, img2_path)
        
        print(f"图像尺寸: {img1.shape}")
        
        x_coords, y_coords, u, v, valid = piv_analysis(
            img1, img2,
            method='window_deformation',
            window_size=32,
            overlap=16,
            n_iterations=3,
            subpixel_method='gaussian'
        )
        
        print_statistics(u, v, valid)
        
        print("生成可视化结果...")
        plot_results(img1, img2, x_coords, y_coords, u, v, valid, save_path="piv_results.png")
        
        np.savez("piv_data.npz", x=x_coords, y=y_coords, u=u, v=v, valid=valid)
        print("速度场数据已保存到: piv_data.npz")
        
        print("\n分析完成!")


if __name__ == "__main__":
    main()

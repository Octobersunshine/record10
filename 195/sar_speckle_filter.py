import numpy as np
from PIL import Image
import argparse
import os

try:
    from polsar_filter import (
        create_covariance_matrix_3x3,
        boxcar_filter_covariance,
        refined_lee_filter_polarimetric,
        polarimetric_whitening_filter,
        enhanced_polarimetric_whitening_filter,
        idan_filter,
        cloude_pottier_decomposition,
        yamaguchi_decomposition,
        calculate_polarization_fidelity,
        covariance_to_intensity,
    )
    POLSAR_AVAILABLE = True
except ImportError:
    POLSAR_AVAILABLE = False
    print("注意: PolSAR滤波模块未加载，仅支持单极化滤波")


def lee_filter(img, window_size=3):
    """
    Lee滤波 - SAR图像相干斑抑制
    
    参数:
        img: 输入图像（灰度图）
        window_size: 滑动窗口大小（奇数）
    
    返回:
        滤波后的图像
    """
    if window_size % 2 == 0:
        window_size += 1
    
    pad = window_size // 2
    img_padded = np.pad(img, pad, mode='reflect')
    img_filtered = np.zeros_like(img, dtype=np.float64)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            window = img_padded[i:i+window_size, j:j+window_size]
            
            mean_win = np.mean(window)
            var_win = np.var(window)
            
            var_noise = var_win / (mean_win ** 2 + 1e-10) if mean_win != 0 else 0
            
            if var_win <= 1e-10:
                img_filtered[i, j] = mean_win
            else:
                weight = var_win / (var_win + var_noise * mean_win ** 2)
                img_filtered[i, j] = mean_win + weight * (img[i, j] - mean_win)
    
    return img_filtered


def enhanced_lee_filter(img, window_size=7, num_looks=1):
    """
    Enhanced Lee滤波 - SAR图像相干斑抑制
    
    参数:
        img: 输入图像（灰度图）
        window_size: 滑动窗口大小（奇数）
        num_looks: SAR图像视数，用于计算噪声方差
    
    返回:
        滤波后的图像
    """
    if window_size % 2 == 0:
        window_size += 1
    
    pad = window_size // 2
    img_padded = np.pad(img, pad, mode='reflect')
    img_filtered = np.zeros_like(img, dtype=np.float64)
    
    cu = np.sqrt(2.0 / num_looks)
    cmax = np.sqrt(2.0) * cu
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            window = img_padded[i:i+window_size, j:j+window_size]
            
            mean_win = np.mean(window)
            std_win = np.std(window)
            
            if mean_win < 1e-10:
                ci = 0
            else:
                ci = std_win / mean_win
            
            if ci <= cu:
                img_filtered[i, j] = mean_win
            elif ci < cmax:
                weight = (ci - cu) / (cmax - cu)
                img_filtered[i, j] = mean_win + weight * (img[i, j] - mean_win)
            else:
                img_filtered[i, j] = img[i, j]
    
    return img_filtered


def load_image(image_path):
    """
    加载图像并转换为灰度图
    """
    img = Image.open(image_path).convert('L')
    return np.array(img, dtype=np.float64)


def save_image(img_array, output_path):
    """
    保存图像
    """
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    img.save(output_path)
    print(f"图像已保存到: {output_path}")


def calculate_enl(img):
    """
    计算等效视数(ENL) - 用于评估滤波效果
    """
    mean_val = np.mean(img)
    std_val = np.std(img)
    return (mean_val ** 2) / (std_val ** 2) if std_val > 0 else 0


def adaptive_window_lee_filter(img, min_window=3, max_window=11, num_looks=1):
    """
    自适应窗口Lee滤波 - 根据局部方差系数动态调整窗口大小
    
    核心思想：在均匀区域使用大窗口获得更好的平滑效果，
             在边缘区域使用小窗口保持边缘细节。
    
    三区域模型：
    - 均匀区域 (Ci <= Cu): 使用最大可用窗口，强平滑
    - 过渡区域 (Cu < Ci < Cmax): 使用中等窗口，适度平滑
    - 边缘区域 (Ci >= Cmax): 直接使用原始像素，保护边缘
    
    参数:
        img: 输入图像（灰度图）
        min_window: 最小窗口大小
        max_window: 最大窗口大小
        num_looks: SAR图像视数
    
    返回:
        滤波后的图像
    """
    if min_window % 2 == 0:
        min_window += 1
    if max_window % 2 == 0:
        max_window += 1
    if min_window >= max_window:
        min_window = max_window - 2 if max_window > 3 else 3
    
    window_sizes = list(range(min_window, max_window + 1, 2))
    
    pad = max_window // 2
    img_padded = np.pad(img, pad, mode='reflect')
    img_filtered = np.zeros_like(img, dtype=np.float64)
    
    cu = np.sqrt(2.0 / num_looks)
    cmax = np.sqrt(2.0) * cu
    
    var_noise = 2.0 / num_looks if num_looks > 0 else 2.0
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            ci_min = float('inf')
            selected_ws = min_window
            is_edge = False
            
            for ws in window_sizes:
                half = ws // 2
                window = img_padded[i+pad-half:i+pad+half+1, 
                                   j+pad-half:j+pad+half+1]
                
                mean_win = np.mean(window)
                std_win = np.std(window)
                
                if mean_win < 1e-10:
                    ci = 0
                else:
                    ci = std_win / mean_win
                
                if ci >= cmax:
                    is_edge = True
                    break
                
                if ci < ci_min:
                    ci_min = ci
                    selected_ws = ws
                
                if ci > cu:
                    break
            
            if is_edge:
                img_filtered[i, j] = img[i, j]
            else:
                half = selected_ws // 2
                window = img_padded[i+pad-half:i+pad+half+1, 
                                   j+pad-half:j+pad+half+1]
                
                mean_win = np.mean(window)
                var_win = np.var(window)
                
                if var_win <= 1e-10:
                    img_filtered[i, j] = mean_win
                else:
                    weight = var_win / (var_win + var_noise * mean_win ** 2)
                    img_filtered[i, j] = mean_win + weight * (img[i, j] - mean_win)
    
    return img_filtered


def sar_nlm_filter(img, window_size=7, search_window=21, h=10.0, num_looks=1):
    """
    基于非局部均值(NLM)的SAR图像相干斑抑制
    
    核心思想：利用整幅图像中相似的斑块进行加权平均，
             能够在有效抑制斑点噪声的同时很好地保持边缘和细节。
    
    针对SAR的改进：
    - 使用概率分布距离（基于瑞利分布）代替欧氏距离
    - 引入预分类以加速计算
    - 结合视数参数调整权重
    
    参数:
        img: 输入图像（灰度图）
        window_size: 相似性比较窗口大小
        search_window: 搜索窗口大小
        h: 滤波强度参数，越大平滑效果越强
        num_looks: SAR图像视数
    
    返回:
        滤波后的图像
    """
    if window_size % 2 == 0:
        window_size += 1
    if search_window % 2 == 0:
        search_window += 1
    
    h = float(h)
    h2 = h * h
    
    half_w = window_size // 2
    half_s = search_window // 2
    
    pad = half_s
    img_padded = np.pad(img, pad, mode='reflect')
    img_filtered = np.zeros_like(img, dtype=np.float64)
    
    noise_var = 2.0 / num_looks if num_looks > 0 else 2.0
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            i_pad = i + pad
            j_pad = j + pad
            
            center_patch = img_padded[i_pad-half_w:i_pad+half_w+1, 
                                     j_pad-half_w:j_pad+half_w+1]
            
            weights = []
            values = []
            
            for di in range(-half_s, half_s + 1):
                for dj in range(-half_s, half_s + 1):
                    ni = i_pad + di
                    nj = j_pad + dj
                    
                    if (ni < half_w or ni >= img_padded.shape[0] - half_w or
                        nj < half_w or nj >= img_padded.shape[1] - half_w):
                        continue
                    
                    patch = img_padded[ni-half_w:ni+half_w+1, 
                                      nj-half_w:nj+half_w+1]
                    
                    mean1 = np.mean(center_patch)
                    mean2 = np.mean(patch)
                    
                    if mean1 < 1e-10 or mean2 < 1e-10:
                        dist = np.sum((center_patch - patch) ** 2)
                    else:
                        norm_patch1 = center_patch / (mean1 + 1e-10)
                        norm_patch2 = patch / (mean2 + 1e-10)
                        
                        dist1 = np.sum((norm_patch1 - norm_patch2) ** 2)
                        dist2 = np.log((mean1 ** 2 + 1e-10) / (mean2 ** 2 + 1e-10)) ** 2
                        dist = dist1 + 0.5 * dist2
                    
                    sigma2 = noise_var * window_size * window_size
                    weight = np.exp(-dist / (h2 * sigma2 + 1e-10))
                    
                    if di == 0 and dj == 0:
                        weight *= 2.0
                    
                    weights.append(weight)
                    values.append(img_padded[ni, nj])
            
            weights = np.array(weights)
            values = np.array(values)
            
            weight_sum = np.sum(weights)
            if weight_sum > 1e-10:
                img_filtered[i, j] = np.sum(weights * values) / weight_sum
            else:
                img_filtered[i, j] = img[i, j]
    
    return img_filtered


def fast_sar_nlm_filter(img, window_size=5, search_window=15, h=15.0, num_looks=1, edge_protect=True):
    """
    快速版非局部均值SAR滤波 - 使用预分类和步长采样加速
    
    改进：增加边缘检测和保护机制
    
    参数:
        img: 输入图像（灰度图）
        window_size: 相似性比较窗口大小
        search_window: 搜索窗口大小
        h: 滤波强度参数
        num_looks: SAR图像视数
        edge_protect: 是否启用边缘保护
    
    返回:
        滤波后的图像
    """
    if window_size % 2 == 0:
        window_size += 1
    if search_window % 2 == 0:
        search_window += 1
    
    h = float(h)
    h2 = h * h
    
    half_w = window_size // 2
    half_s = search_window // 2
    
    step = 2 if search_window > 11 else 1
    
    pad = half_s
    img_padded = np.pad(img, pad, mode='reflect')
    img_filtered = np.zeros_like(img, dtype=np.float64)
    
    noise_var = 2.0 / num_looks if num_looks > 0 else 2.0
    sigma2 = noise_var * window_size * window_size
    
    cu = np.sqrt(2.0 / num_looks)
    cmax = np.sqrt(2.0) * cu
    
    rows, cols = img.shape
    
    for i in range(rows):
        for j in range(cols):
            i_pad = i + pad
            j_pad = j + pad
            
            center_patch = img_padded[i_pad-half_w:i_pad+half_w+1, 
                                     j_pad-half_w:j_pad+half_w+1]
            center_mean = np.mean(center_patch)
            center_std = np.std(center_patch)
            
            if edge_protect and center_mean > 1e-10:
                ci = center_std / center_mean
                if ci >= cmax:
                    img_filtered[i, j] = img[i, j]
                    continue
            
            total_weight = 0.0
            total_value = 0.0
            
            center_weight = np.exp(0) * 2.0
            total_weight += center_weight
            total_value += center_weight * img[i, j]
            
            local_search = half_s
            if edge_protect and center_mean > 1e-10:
                ci = center_std / center_mean
                if ci > cu:
                    local_search = max(3, half_s // 2)
            
            for di in range(-local_search, local_search + 1, step):
                for dj in range(-local_search, local_search + 1, step):
                    if di == 0 and dj == 0:
                        continue
                    
                    ni = i_pad + di
                    nj = j_pad + dj
                    
                    if (ni < half_w or ni >= img_padded.shape[0] - half_w or
                        nj < half_w or nj >= img_padded.shape[1] - half_w):
                        continue
                    
                    patch = img_padded[ni-half_w:ni+half_w+1, 
                                      nj-half_w:nj+half_w+1]
                    patch_mean = np.mean(patch)
                    
                    mean_ratio = abs(center_mean - patch_mean) / (center_mean + patch_mean + 1e-10)
                    if mean_ratio > 0.4:
                        continue
                    
                    if center_mean < 1e-10 or patch_mean < 1e-10:
                        dist = np.sum((center_patch - patch) ** 2)
                    else:
                        norm1 = center_patch / (center_mean + 1e-10)
                        norm2 = patch / (patch_mean + 1e-10)
                        dist = np.sum((norm1 - norm2) ** 2)
                    
                    weight = np.exp(-dist / (h2 * sigma2 + 1e-10))
                    
                    dist_pixel = np.sqrt(di**2 + dj**2)
                    weight *= np.exp(-dist_pixel / (local_search + 1e-10))
                    
                    total_weight += weight
                    total_value += weight * img_padded[ni, nj]
            
            if total_weight > 1e-10:
                img_filtered[i, j] = total_value / total_weight
            else:
                img_filtered[i, j] = img[i, j]
    
    return img_filtered


def main():
    parser = argparse.ArgumentParser(description='SAR图像相干斑抑制 - 支持单极化和极化SAR滤波')
    
    parser.add_argument('--mode', type=str, default='single', 
                       choices=['single', 'polarimetric'],
                       help='滤波模式: single (单极化) 或 polarimetric (极化SAR)')
    parser.add_argument('--input', type=str, help='输入图像路径（单极化模式）')
    parser.add_argument('--output', type=str, default='filtered_sar_image.png', help='输出图像路径')
    
    parser.add_argument('--filter', type=str, default='adaptive_lee', 
                       choices=['lee', 'enhanced_lee', 'adaptive_lee', 'nlm', 'fast_nlm',
                               'boxcar', 'refined_lee', 'pwf', 'epwf', 'idan'], 
                       help='滤波方法')
    parser.add_argument('--window', type=int, default=7, help='窗口大小（推荐3-11）')
    parser.add_argument('--min-window', type=int, default=3, help='自适应最小窗口大小')
    parser.add_argument('--max-window', type=int, default=11, help='自适应最大窗口大小')
    parser.add_argument('--search-window', type=int, default=15, help='NLM搜索窗口大小')
    parser.add_argument('--looks', type=int, default=1, help='SAR图像视数')
    parser.add_argument('--h', type=float, default=15.0, help='NLM滤波强度参数')
    parser.add_argument('--no-edge-protect', action='store_true', help='禁用NLM边缘保护')
    
    parser.add_argument('--hh', type=str, help='HH极化通道图像路径 (PolSAR模式)')
    parser.add_argument('--vv', type=str, help='VV极化通道图像路径 (PolSAR模式)')
    parser.add_argument('--hv', type=str, help='HV极化通道图像路径 (PolSAR模式, 可选)')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        process_single_polarization(args)
    elif args.mode == 'polarimetric':
        if POLSAR_AVAILABLE:
            process_polarimetric(args)
        else:
            print("错误: PolSAR滤波模块不可用，请确保polsar_filter.py存在")
            return


def process_single_polarization(args):
    if not args.input:
        print("错误: 单极化模式需要指定 --input 参数")
        return
    
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return
    
    print(f"加载图像: {args.input}")
    img = load_image(args.input)
    print(f"图像尺寸: {img.shape}")
    
    print(f"原始图像ENL: {calculate_enl(img):.4f}")
    
    if args.filter in ['lee', 'enhanced_lee', 'adaptive_lee', 'nlm', 'fast_nlm']:
        if args.filter == 'lee':
            print(f"应用Lee滤波，窗口大小: {args.window}")
            img_filtered = lee_filter(img, window_size=args.window)
        elif args.filter == 'enhanced_lee':
            print(f"应用Enhanced Lee滤波，窗口大小: {args.window}, 视数: {args.looks}")
            img_filtered = enhanced_lee_filter(img, window_size=args.window, num_looks=args.looks)
        elif args.filter == 'adaptive_lee':
            print(f"应用自适应窗口Lee滤波，窗口范围: {args.min_window}-{args.max_window}, 视数: {args.looks}")
            img_filtered = adaptive_window_lee_filter(img, min_window=args.min_window, 
                                                       max_window=args.max_window, num_looks=args.looks)
        elif args.filter == 'nlm':
            print(f"应用NLM滤波，窗口: {args.window}, 搜索窗口: {args.search_window}, h={args.h}, 视数: {args.looks}")
            img_filtered = sar_nlm_filter(img, window_size=args.window, search_window=args.search_window,
                                          h=args.h, num_looks=args.looks)
        elif args.filter == 'fast_nlm':
            edge_protect = not args.no_edge_protect
            print(f"应用快速NLM滤波，窗口: {args.window}, 搜索窗口: {args.search_window}, h={args.h}, 视数: {args.looks}, 边缘保护: {edge_protect}")
            img_filtered = fast_sar_nlm_filter(img, window_size=args.window, search_window=args.search_window,
                                                h=args.h, num_looks=args.looks, edge_protect=edge_protect)
        
        print(f"滤波后图像ENL: {calculate_enl(img_filtered):.4f}")
        save_image(img_filtered, args.output)
    else:
        print(f"错误: 滤波方法 '{args.filter}' 不适用于单极化模式")
        print("可用的单极化滤波方法: lee, enhanced_lee, adaptive_lee, nlm, fast_nlm")


def process_polarimetric(args):
    if not args.hh or not args.vv:
        print("错误: PolSAR模式需要指定 --hh 和 --vv 参数")
        return
    
    if not os.path.exists(args.hh):
        print(f"错误: HH图像文件不存在: {args.hh}")
        return
    if not os.path.exists(args.vv):
        print(f"错误: VV图像文件不存在: {args.vv}")
        return
    
    print(f"加载HH极化图像: {args.hh}")
    img_hh = load_image(args.hh)
    print(f"加载VV极化图像: {args.vv}")
    img_vv = load_image(args.vv)
    
    if args.hv and os.path.exists(args.hv):
        print(f"加载HV极化图像: {args.hv}")
        img_hv = load_image(args.hv)
        hv_real = img_hv
        hv_imag = np.zeros_like(img_hv)
    else:
        print("未提供HV图像，将使用零值")
        hv_real = np.zeros_like(img_hh)
        hv_imag = np.zeros_like(img_hh)
    
    print(f"图像尺寸: {img_hh.shape}")
    
    C = create_covariance_matrix_3x3(
        img_hh.astype(np.complex128),
        img_vv.astype(np.complex128),
        hv_real,
        hv_imag
    )
    
    hh_orig, vv_orig, hv_orig = covariance_to_intensity(C)
    print(f"原始图像ENL (HH): {calculate_enl(hh_orig):.4f}")
    print(f"原始图像ENL (VV): {calculate_enl(vv_orig):.4f}")
    
    if args.filter == 'boxcar':
        print(f"应用Boxcar滤波，窗口大小: {args.window}")
        C_filtered = boxcar_filter_covariance(C, window_size=args.window)
    elif args.filter == 'refined_lee':
        print(f"应用Refined Lee滤波，窗口大小: {args.window}, 视数: {args.looks}")
        C_filtered = refined_lee_filter_polarimetric(C, window_size=args.window, num_looks=args.looks)
    elif args.filter == 'pwf':
        print(f"应用极化白化滤波(PWF)，窗口大小: {args.window}, 视数: {args.looks}")
        C_filtered = polarimetric_whitening_filter(C, window_size=args.window, num_looks=args.looks)
    elif args.filter == 'epwf':
        print(f"应用增强型PWF(EPWF)，窗口大小: {args.window}, 视数: {args.looks}")
        C_filtered = enhanced_polarimetric_whitening_filter(C, window_size=args.window, num_looks=args.looks)
    elif args.filter == 'idan':
        print(f"应用IDAN滤波，窗口大小: {args.window}, 视数: {args.looks}")
        C_filtered = idan_filter(C, window_size=args.window, num_looks=args.looks)
    else:
        print(f"错误: 滤波方法 '{args.filter}' 不适用于PolSAR模式")
        print("可用的PolSAR滤波方法: boxcar, refined_lee, pwf, epwf, idan")
        return
    
    hh_filtered, vv_filtered, hv_filtered = covariance_to_intensity(C_filtered)
    print(f"滤波后图像ENL (HH): {calculate_enl(hh_filtered):.4f}")
    print(f"滤波后图像ENL (VV): {calculate_enl(vv_filtered):.4f}")
    
    fidelity = calculate_polarization_fidelity(C, C_filtered)
    print(f"平均极化保真度: {np.mean(fidelity):.4f}")
    
    base, ext = os.path.splitext(args.output)
    
    save_image(hh_filtered, f"{base}_hh{ext}")
    print(f"HH滤波结果已保存: {base}_hh{ext}")
    
    save_image(vv_filtered, f"{base}_vv{ext}")
    print(f"VV滤波结果已保存: {base}_vv{ext}")
    
    save_image(hv_filtered, f"{base}_hv{ext}")
    print(f"HV滤波结果已保存: {base}_hv{ext}")


if __name__ == '__main__':
    main()

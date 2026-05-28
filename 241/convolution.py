import numpy as np


def convolve_direct(x, kernel, mode='full'):
    x = list(x)
    kernel = list(kernel)
    n = len(x)
    m = len(kernel)
    
    if mode == 'full':
        result_len = n + m - 1
    elif mode == 'same':
        result_len = max(n, m)
    elif mode == 'valid':
        result_len = max(n, m) - min(n, m) + 1
        if result_len < 1:
            raise ValueError("valid mode requires longer signal than kernel")
    else:
        raise ValueError("mode must be 'full', 'same', or 'valid'")
    
    result = []
    
    if mode == 'full':
        for i in range(result_len):
            s = 0
            for j in range(m):
                x_idx = i - j
                if 0 <= x_idx < n:
                    s += x[x_idx] * kernel[j]
            result.append(s)
    
    elif mode == 'same':
        pad = (m - 1) // 2
        for i in range(result_len):
            s = 0
            for j in range(m):
                x_idx = i + pad - j
                if 0 <= x_idx < n:
                    s += x[x_idx] * kernel[j]
            result.append(s)
    
    elif mode == 'valid':
        for i in range(result_len):
            s = 0
            for j in range(m):
                x_idx = i + m - 1 - j
                s += x[x_idx] * kernel[j]
            result.append(s)
    
    return result


def _next_power_of_two(n):
    return 1 << (n - 1).bit_length()


def convolve_fft(x, kernel, mode='full'):
    x = np.asarray(x, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    n = len(x)
    m = len(kernel)
    
    full_len = n + m - 1
    fft_len = _next_power_of_two(full_len)
    
    X = np.fft.fft(x, fft_len)
    K = np.fft.fft(kernel, fft_len)
    
    result_fft = X * K
    result_full = np.fft.ifft(result_fft).real[:full_len]
    
    if mode == 'full':
        result = result_full
    elif mode == 'same':
        result_len = max(n, m)
        start = (full_len - result_len) // 2
        result = result_full[start:start + result_len]
    elif mode == 'valid':
        result_len = max(n, m) - min(n, m) + 1
        start = min(n, m) - 1
        result = result_full[start:start + result_len]
    else:
        raise ValueError("mode must be 'full', 'same', or 'valid'")
    
    return result.tolist()


AUTO_THRESHOLD = 1000


def convolve1d(x, kernel, mode='full', method='auto'):
    n = len(x)
    m = len(kernel)
    
    if method == 'auto':
        if max(n, m) < AUTO_THRESHOLD or n * m < AUTO_THRESHOLD * 10:
            method = 'direct'
        else:
            method = 'fft'
    
    if method == 'direct':
        return convolve_direct(x, kernel, mode)
    elif method == 'fft':
        return convolve_fft(x, kernel, mode)
    else:
        raise ValueError("method must be 'auto', 'direct', or 'fft'")


def cross_correlation(x, y, mode='full', method='auto'):
    return convolve1d(x, list(y)[::-1], mode=mode, method=method)


def find_signal_alignment(signal, template, method='auto'):
    corr = cross_correlation(signal, template, mode='valid', method=method)
    max_corr = max(corr)
    lag = corr.index(max_corr)
    return lag, max_corr, corr


def _pad2d(image, pad_h, pad_w, padding='zero'):
    h, w = image.shape
    padded = np.zeros((h + 2 * pad_h, w + 2 * pad_w), dtype=image.dtype)
    padded[pad_h:pad_h + h, pad_w:pad_w + w] = image
    
    if padding == 'zero':
        pass
    
    elif padding == 'replicate':
        padded[:pad_h, pad_w:pad_w + w] = image[0:1, :]
        padded[pad_h + h:, pad_w:pad_w + w] = image[-1:, :]
        padded[pad_h:pad_h + h, :pad_w] = image[:, 0:1]
        padded[pad_h:pad_h + h, pad_w + w:] = image[:, -1:]
        padded[:pad_h, :pad_w] = image[0, 0]
        padded[:pad_h, pad_w + w:] = image[0, -1]
        padded[pad_h + h:, :pad_w] = image[-1, 0]
        padded[pad_h + h:, pad_w + w:] = image[-1, -1]
    
    elif padding == 'mirror':
        for i in range(pad_h):
            padded[pad_h - 1 - i, pad_w:pad_w + w] = image[min(i + 1, h - 1), :]
            padded[pad_h + h + i, pad_w:pad_w + w] = image[max(h - 2 - i, 0), :]
        for j in range(pad_w):
            padded[pad_h:pad_h + h, pad_w - 1 - j] = image[:, min(j + 1, w - 1)]
            padded[pad_h:pad_h + h, pad_w + w + j] = image[:, max(w - 2 - j, 0)]
        for i in range(pad_h):
            for j in range(pad_w):
                padded[pad_h - 1 - i, pad_w - 1 - j] = image[min(i + 1, h - 1), min(j + 1, w - 1)]
                padded[pad_h - 1 - i, pad_w + w + j] = image[min(i + 1, h - 1), max(w - 2 - j, 0)]
                padded[pad_h + h + i, pad_w - 1 - j] = image[max(h - 2 - i, 0), min(j + 1, w - 1)]
                padded[pad_h + h + i, pad_w + w + j] = image[max(h - 2 - i, 0), max(w - 2 - j, 0)]
    
    elif padding == 'circular':
        for i in range(pad_h):
            padded[pad_h - 1 - i, pad_w:pad_w + w] = image[h - 1 - i, :]
            padded[pad_h + h + i, pad_w:pad_w + w] = image[i, :]
        for j in range(pad_w):
            padded[pad_h:pad_h + h, pad_w - 1 - j] = image[:, w - 1 - j]
            padded[pad_h:pad_h + h, pad_w + w + j] = image[:, j]
        for i in range(pad_h):
            for j in range(pad_w):
                padded[pad_h - 1 - i, pad_w - 1 - j] = image[h - 1 - i, w - 1 - j]
                padded[pad_h - 1 - i, pad_w + w + j] = image[h - 1 - i, j]
                padded[pad_h + h + i, pad_w - 1 - j] = image[i, w - 1 - j]
                padded[pad_h + h + i, pad_w + w + j] = image[i, j]
    
    else:
        raise ValueError("padding must be 'zero', 'replicate', 'mirror', or 'circular'")
    
    return padded


def convolve2d(image, kernel, mode='same', padding='zero', method='auto'):
    image = np.asarray(image, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape
    
    if method == 'auto':
        if img_h * img_w * k_h * k_w > AUTO_THRESHOLD * AUTO_THRESHOLD:
            method = 'fft'
        else:
            method = 'direct'
    
    if mode == 'valid':
        out_h = img_h - k_h + 1
        out_w = img_w - k_w + 1
        if out_h < 1 or out_w < 1:
            raise ValueError("valid mode requires image larger than kernel")
        
        if method == 'fft':
            return _convolve2d_fft(image, kernel, 'valid')
        
        result = np.zeros((out_h, out_w))
        flipped = kernel[::-1, ::-1]
        for i in range(out_h):
            for j in range(out_w):
                result[i, j] = np.sum(image[i:i + k_h, j:j + k_w] * flipped)
        return result
    
    elif mode == 'same':
        pad_h = k_h // 2
        pad_w = k_w // 2
        padded = _pad2d(image, pad_h, pad_w, padding)
        out_h = img_h
        out_w = img_w
        
        if method == 'fft':
            full_result = _convolve2d_fft(padded, kernel, 'valid')
            return full_result[:out_h, :out_w]
        
        result = np.zeros((out_h, out_w))
        flipped = kernel[::-1, ::-1]
        for i in range(out_h):
            for j in range(out_w):
                result[i, j] = np.sum(padded[i:i + k_h, j:j + k_w] * flipped)
        return result
    
    elif mode == 'full':
        pad_h = k_h - 1
        pad_w = k_w - 1
        padded = _pad2d(image, pad_h, pad_w, padding)
        out_h = img_h + k_h - 1
        out_w = img_w + k_w - 1
        
        if method == 'fft':
            full_result = _convolve2d_fft(padded, kernel, 'valid')
            return full_result[:out_h, :out_w]
        
        result = np.zeros((out_h, out_w))
        flipped = kernel[::-1, ::-1]
        for i in range(out_h):
            for j in range(out_w):
                result[i, j] = np.sum(padded[i:i + k_h, j:j + k_w] * flipped)
        return result
    
    else:
        raise ValueError("mode must be 'full', 'same', or 'valid'")


def _convolve2d_fft(image, kernel, mode='valid'):
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape
    
    fft_h = _next_power_of_two(img_h + k_h - 1)
    fft_w = _next_power_of_two(img_w + k_w - 1)
    
    F_image = np.fft.fft2(image, s=(fft_h, fft_w))
    F_kernel = np.fft.fft2(kernel, s=(fft_h, fft_w))
    
    result_full = np.fft.ifft2(F_image * F_kernel).real
    
    full_h = img_h + k_h - 1
    full_w = img_w + k_w - 1
    result_full = result_full[:full_h, :full_w]
    
    if mode == 'valid':
        out_h = img_h - k_h + 1
        out_w = img_w - k_w + 1
        start_h = k_h - 1
        start_w = k_w - 1
        return result_full[start_h:start_h + out_h, start_w:start_w + out_w]
    elif mode == 'full':
        return result_full
    elif mode == 'same':
        start_h = (k_h - 1) // 2
        start_w = (k_w - 1) // 2
        out_h = img_h
        out_w = img_w
        return result_full[start_h:start_h + out_h, start_w:start_w + out_w]


def normalized_cross_correlation(image, template, padding='zero'):
    image = np.asarray(image, dtype=np.float64)
    template = np.asarray(template, dtype=np.float64)
    
    img_h, img_w = image.shape
    t_h, t_w = template.shape
    
    if img_h < t_h or img_w < t_w:
        raise ValueError("Template must be smaller than image")
    
    template_mean = np.mean(template)
    template_centered = template - template_mean
    template_std = np.sqrt(np.sum(template_centered ** 2))
    
    if template_std < 1e-10:
        return np.zeros((img_h - t_h + 1, img_w - t_w + 1))
    
    out_h = img_h - t_h + 1
    out_w = img_w - t_w + 1
    ncc = np.zeros((out_h, out_w))
    
    for i in range(out_h):
        for j in range(out_w):
            patch = image[i:i + t_h, j:j + t_w]
            patch_mean = np.mean(patch)
            patch_centered = patch - patch_mean
            patch_std = np.sqrt(np.sum(patch_centered ** 2))
            
            if patch_std < 1e-10:
                ncc[i, j] = 0.0
            else:
                ncc[i, j] = np.sum(patch_centered * template_centered) / (template_std * patch_std)
    
    return ncc


def normalized_cross_correlation_fft(image, template, padding='zero'):
    image = np.asarray(image, dtype=np.float64)
    template = np.asarray(template, dtype=np.float64)
    
    img_h, img_w = image.shape
    t_h, t_w = template.shape
    
    if img_h < t_h or img_w < t_w:
        raise ValueError("Template must be smaller than image")
    
    template_mean = np.mean(template)
    template_centered = template - template_mean
    template_norm = np.sqrt(np.sum(template_centered ** 2))
    
    if template_norm < 1e-10:
        return np.zeros((img_h - t_h + 1, img_w - t_w + 1))
    
    out_h = img_h - t_h + 1
    out_w = img_w - t_w + 1
    
    numerator = _cross_correlate2d_fft(image, template_centered)
    
    ones = np.ones((t_h, t_w), dtype=np.float64)
    image_sum = _cross_correlate2d_fft(image, ones)
    image_sq_sum = _cross_correlate2d_fft(image ** 2, ones)
    
    n_pixels = t_h * t_w
    local_var = image_sq_sum - image_sum ** 2 / n_pixels
    local_var = np.maximum(local_var, 0.0)
    denominator = np.sqrt(local_var) * template_norm
    
    ncc = np.zeros((out_h, out_w))
    valid_mask = denominator > 1e-10
    ncc[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
    
    return ncc


def _cross_correlate2d_fft(image, template):
    img_h, img_w = image.shape
    t_h, t_w = template.shape
    
    fft_h = _next_power_of_two(img_h + t_h - 1)
    fft_w = _next_power_of_two(img_w + t_w - 1)
    
    F_image = np.fft.fft2(image, s=(fft_h, fft_w))
    F_template = np.fft.fft2(template, s=(fft_h, fft_w))
    
    result_full = np.fft.ifft2(F_image * np.conj(F_template)).real
    
    out_h = img_h - t_h + 1
    out_w = img_w - t_w + 1
    
    return result_full[:out_h, :out_w]


def template_match(image, template, method='ncc', padding='zero'):
    if method == 'ncc':
        if image.shape[0] * image.shape[1] > AUTO_THRESHOLD * AUTO_THRESHOLD:
            ncc = normalized_cross_correlation_fft(image, template, padding)
        else:
            ncc = normalized_cross_correlation(image, template, padding)
        max_val = np.max(ncc)
        max_loc = np.unravel_index(np.argmax(ncc), ncc.shape)
        return max_loc, max_val, ncc
    elif method == 'cc':
        image = np.asarray(image, dtype=np.float64)
        template = np.asarray(template, dtype=np.float64)
        corr = _cross_correlate2d_fft(image, template)
        max_val = np.max(corr)
        max_loc = np.unravel_index(np.argmax(corr), corr.shape)
        return max_loc, max_val, corr
    else:
        raise ValueError("method must be 'ncc' or 'cc'")


if __name__ == "__main__":
    print("=" * 60)
    print("一维卷积测试")
    print("=" * 60)
    x = [1, 2, 3, 4, 5]
    kernel = [1, 0, -1]
    
    print("直接卷积 (full):", convolve1d(x, kernel, 'full', method='direct'))
    print("FFT卷积 (full):", convolve1d(x, kernel, 'full', method='fft'))
    print("numpy (full):", np.convolve(x, kernel, 'full').tolist())
    print()
    
    print("=" * 60)
    print("二维卷积测试")
    print("=" * 60)
    
    image = np.array([
        [1, 2, 3, 4, 5],
        [5, 6, 7, 8, 9],
        [9, 10, 11, 12, 13],
        [13, 14, 15, 16, 17],
        [17, 18, 19, 20, 21]
    ], dtype=np.float64)
    
    kernel2d = np.array([
        [1, 0],
        [0, -1]
    ], dtype=np.float64)
    
    from scipy import signal as sp_signal
    
    for mode in ['full', 'same', 'valid']:
        result = convolve2d(image, kernel2d, mode=mode, padding='zero')
        ref = sp_signal.convolve2d(image, kernel2d, mode=mode)
        match = np.allclose(result, ref)
        print(f"模式 {mode:5s} | 结果一致: {match}")
        if not match:
            print(f"  我的结果: {result}")
            print(f"  scipy:    {ref}")
    
    print()
    print("=" * 60)
    print("边缘填充测试")
    print("=" * 60)
    
    small_img = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=np.float64)
    
    blur_kernel = np.ones((3, 3), dtype=np.float64) / 9.0
    
    for pad_mode in ['zero', 'replicate', 'mirror']:
        result = convolve2d(small_img, blur_kernel, mode='same', padding=pad_mode)
        print(f"填充 '{pad_mode:10s}': {np.round(result, 2).tolist()}")
    
    print()
    print("=" * 60)
    print("归一化互相关 (NCC) 测试")
    print("=" * 60)
    
    scene = np.zeros((10, 10), dtype=np.float64)
    scene[3:6, 4:7] = np.array([
        [10, 20, 30],
        [40, 50, 60],
        [70, 80, 90]
    ], dtype=np.float64)
    
    template = np.array([
        [10, 20, 30],
        [40, 50, 60],
        [70, 80, 90]
    ], dtype=np.float64)
    
    ncc = normalized_cross_correlation(scene, template)
    max_loc = np.unravel_index(np.argmax(ncc), ncc.shape)
    max_val = np.max(ncc)
    print(f"模板真实位置: (3, 4)")
    print(f"NCC检测位置:   {max_loc}")
    print(f"NCC最大值:     {max_val:.6f}")
    
    ncc_fft = normalized_cross_correlation_fft(scene, template)
    max_loc_fft = np.unravel_index(np.argmax(ncc_fft), ncc_fft.shape)
    max_val_fft = np.max(ncc_fft)
    print(f"FFT-NCC检测位置: {max_loc_fft}")
    print(f"FFT-NCC最大值:   {max_val_fft:.6f}")
    print(f"NCC结果一致: {np.allclose(ncc, ncc_fft, atol=1e-6)}")
    
    print()
    print("=" * 60)
    print("模板匹配测试 (template_match)")
    print("=" * 60)
    
    loc, val, ncc_map = template_match(scene, template, method='ncc')
    print(f"NCC匹配位置: {loc}, 相似度: {val:.6f}")
    
    loc_cc, val_cc, cc_map = template_match(scene, template, method='cc')
    print(f"CC匹配位置:  {loc_cc}, 相似度: {val_cc:.6f}")

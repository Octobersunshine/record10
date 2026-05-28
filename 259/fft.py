import cmath


def _next_power_of_two(n):
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def _is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0


def _fft_core(signal):
    N = len(signal)
    
    if N == 1:
        return signal
    
    even = _fft_core(signal[0::2])
    odd = _fft_core(signal[1::2])
    
    result = [0] * N
    for k in range(N // 2):
        t = cmath.exp(-2j * cmath.pi * k / N) * odd[k]
        result[k] = even[k] + t
        result[k + N // 2] = even[k] - t
    
    return result


def _ifft_core(spectrum):
    N = len(spectrum)
    
    if N == 1:
        return spectrum
    
    even = _ifft_core(spectrum[0::2])
    odd = _ifft_core(spectrum[1::2])
    
    result = [0] * N
    for k in range(N // 2):
        t = cmath.exp(2j * cmath.pi * k / N) * odd[k]
        result[k] = (even[k] + t) / 2
        result[k + N // 2] = (even[k] - t) / 2
    
    return result


def fft(signal):
    original_length = len(signal)
    
    if original_length == 0:
        raise ValueError("输入序列不能为空")
    
    pad_info = {
        'original_length': original_length,
        'padded_length': original_length,
        'pad_count': 0,
        'was_padded': False
    }
    
    if not _is_power_of_two(original_length):
        padded_length = _next_power_of_two(original_length)
        pad_count = padded_length - original_length
        signal = signal + [0 + 0j] * pad_count
        pad_info['padded_length'] = padded_length
        pad_info['pad_count'] = pad_count
        pad_info['was_padded'] = True
    
    result = _fft_core(signal)
    return result, pad_info


def ifft(spectrum, original_length=None):
    padded_length = len(spectrum)
    
    if padded_length == 0:
        raise ValueError("输入序列不能为空")
    
    pad_info = {
        'original_length': original_length if original_length else padded_length,
        'padded_length': padded_length,
        'pad_count': 0,
        'was_padded': False
    }
    
    if not _is_power_of_two(padded_length):
        raise ValueError("频域序列长度必须是2的幂次")
    
    result = _ifft_core(spectrum)
    
    if original_length and original_length < padded_length:
        result = result[:original_length]
        pad_info['pad_count'] = padded_length - original_length
        pad_info['was_padded'] = True
    
    return result, pad_info


def rfft(real_signal):
    original_length = len(real_signal)
    
    if original_length == 0:
        raise ValueError("输入序列不能为空")
    
    pad_info = {
        'original_length': original_length,
        'padded_length': original_length,
        'pad_count': 0,
        'was_padded': False
    }
    
    if not _is_power_of_two(original_length):
        padded_length = _next_power_of_two(original_length)
        pad_count = padded_length - original_length
        real_signal = real_signal + [0.0] * pad_count
        pad_info['padded_length'] = padded_length
        pad_info['pad_count'] = pad_count
        pad_info['was_padded'] = True
    else:
        padded_length = original_length
    
    complex_signal = [x + 0j for x in real_signal]
    full_spectrum = _fft_core(complex_signal)
    half_length = padded_length // 2 + 1
    half_spectrum = full_spectrum[:half_length]
    
    return half_spectrum, pad_info


def irfft(half_spectrum, original_length=None):
    half_N = len(half_spectrum)
    N = 2 * (half_N - 1)
    
    if not _is_power_of_two(N):
        raise ValueError("半谱序列长度对应的完整长度必须是2的幂次")
    
    full_spectrum = [0] * N
    for k in range(half_N):
        full_spectrum[k] = half_spectrum[k]
    
    for k in range(1, half_N - 1):
        full_spectrum[N - k] = half_spectrum[k].conjugate()
    
    result, pad_info = ifft(full_spectrum, original_length=original_length)
    real_result = [x.real for x in result]
    
    return real_result, pad_info


def fft2(image):
    rows = len(image)
    if rows == 0:
        raise ValueError("输入图像不能为空")
    cols = len(image[0])
    
    pad_info = {
        'original_shape': (rows, cols),
        'padded_shape': (rows, cols),
        'pad_rows': 0,
        'pad_cols': 0,
        'was_padded': False
    }
    
    padded_rows = rows if _is_power_of_two(rows) else _next_power_of_two(rows)
    padded_cols = cols if _is_power_of_two(cols) else _next_power_of_two(cols)
    
    if padded_rows != rows or padded_cols != cols:
        pad_info['padded_shape'] = (padded_rows, padded_cols)
        pad_info['pad_rows'] = padded_rows - rows
        pad_info['pad_cols'] = padded_cols - cols
        pad_info['was_padded'] = True
        
        padded_image = []
        for i in range(padded_rows):
            if i < rows:
                row = image[i] + [0 + 0j] * (padded_cols - cols)
            else:
                row = [0 + 0j] * padded_cols
            padded_image.append(row)
    else:
        padded_image = [row[:] for row in image]
    
    row_result = []
    for row in padded_image:
        row_fft, _ = fft(row)
        row_result.append(row_fft)
    
    col_result = [[0 + 0j] * padded_cols for _ in range(padded_rows)]
    for j in range(padded_cols):
        col = [row_result[i][j] for i in range(padded_rows)]
        col_fft, _ = fft(col)
        for i in range(padded_rows):
            col_result[i][j] = col_fft[i]
    
    return col_result, pad_info


def ifft2(spectrum2d, original_shape=None):
    padded_rows = len(spectrum2d)
    if padded_rows == 0:
        raise ValueError("输入频谱不能为空")
    padded_cols = len(spectrum2d[0])
    
    pad_info = {
        'original_shape': original_shape if original_shape else (padded_rows, padded_cols),
        'padded_shape': (padded_rows, padded_cols),
        'pad_rows': 0,
        'pad_cols': 0,
        'was_padded': False
    }
    
    col_result = [[0 + 0j] * padded_cols for _ in range(padded_rows)]
    for j in range(padded_cols):
        col = [spectrum2d[i][j] for i in range(padded_rows)]
        col_ifft, _ = ifft(col)
        for i in range(padded_rows):
            col_result[i][j] = col_ifft[i]
    
    row_result = []
    for row in col_result:
        row_ifft, _ = ifft(row)
        row_result.append(row_ifft)
    
    if original_shape:
        orig_rows, orig_cols = original_shape
        if orig_rows < padded_rows or orig_cols < padded_cols:
            trimmed = []
            for i in range(orig_rows):
                trimmed.append(row_result[i][:orig_cols])
            pad_info['pad_rows'] = padded_rows - orig_rows
            pad_info['pad_cols'] = padded_cols - orig_cols
            pad_info['was_padded'] = True
            return trimmed, pad_info
    
    return row_result, pad_info


if __name__ == "__main__":
    print("=" * 60)
    print("一维复数FFT测试")
    print("=" * 60)
    signal = [1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j, 5 + 0j]
    print("输入信号:", signal)
    spectrum, pad_info = fft(signal)
    print("补零信息:", pad_info)
    reconstructed, _ = ifft(spectrum, original_length=pad_info['original_length'])
    print("重建信号:", reconstructed)
    
    print("\n" + "=" * 60)
    print("实信号FFT优化测试")
    print("=" * 60)
    real_signal = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    print("输入实信号:", real_signal)
    half_spectrum, rpad_info = rfft(real_signal)
    print("补零信息:", rpad_info)
    print("半谱长度:", len(half_spectrum))
    reconstructed_real, _ = irfft(half_spectrum, original_length=rpad_info['original_length'])
    print("重建实信号:", reconstructed_real)
    
    print("\n" + "=" * 60)
    print("二维FFT测试 (3x3矩阵)")
    print("=" * 60)
    image = [
        [1 + 0j, 2 + 0j, 3 + 0j],
        [4 + 0j, 5 + 0j, 6 + 0j],
        [7 + 0j, 8 + 0j, 9 + 0j]
    ]
    print("输入图像:")
    for row in image:
        print(row)
    spectrum2d, pad2d_info = fft2(image)
    print("补零信息:", pad2d_info)
    reconstructed_image, _ = ifft2(spectrum2d, original_shape=pad2d_info['original_shape'])
    print("重建图像:")
    for row in reconstructed_image:
        print([round(x.real, 10) + round(x.imag, 10) * 1j for x in row])

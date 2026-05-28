import math
from typing import List, Optional


def get_window(n: int, window_type: str = 'hann') -> List[float]:
    """
    生成窗函数用于减少边界效应。

    参数:
        n: 窗长
        window_type: 窗类型，可选值：
            - 'hann': 汉宁窗（余弦窗）
            - 'hamming': 汉明窗
            - 'blackman': 布莱克曼窗
            - 'rectangular': 矩形窗（即不加窗）
            - None: 同矩形窗

    返回:
        窗函数序列，长度为n
    """
    if window_type is None or window_type == 'rectangular':
        return [1.0] * n

    if n == 1:
        return [1.0]

    window = []
    for i in range(n):
        if window_type == 'hann':
            w = 0.5 * (1 - math.cos(2 * math.pi * i / (n - 1)))
        elif window_type == 'hamming':
            w = 0.54 - 0.46 * math.cos(2 * math.pi * i / (n - 1))
        elif window_type == 'blackman':
            w = (0.42 - 0.5 * math.cos(2 * math.pi * i / (n - 1)) +
                 0.08 * math.cos(4 * math.pi * i / (n - 1)))
        else:
            raise ValueError(f"不支持的窗类型: {window_type}")
        window.append(w)

    return window


def apply_window(x: List[float], window_type: Optional[str] = 'hann') -> List[float]:
    """
    对输入序列应用窗函数以减少边界效应。

    参数:
        x: 输入实序列
        window_type: 窗类型

    返回:
        加窗后的序列
    """
    if window_type is None or window_type == 'rectangular':
        return x.copy()

    n = len(x)
    window = get_window(n, window_type)
    return [x[i] * window[i] for i in range(n)]


def dct_i(x: List[float], n: Optional[int] = None, norm: Optional[str] = None,
          scale: str = 'standard', window: Optional[str] = None) -> List[float]:
    """
    实现离散余弦变换 DCT-I。

    DCT-I 对应边界延拓方式：两端点为偶对称（关于边界点本身对称），
    延拓后周期为 2(N-1)，可减少边界不连续性。

    参数:
        x: 输入实序列
        n: 变换长度，默认使用输入序列的长度。
           DCT-I要求 n >= 2。
        norm: 归一化类型，可选值：
            - None: 不归一化（默认）
            - 'ortho': 正交归一化
        scale: 缩放约定，可选值：
            - 'standard': 标准定义（默认）
            - 'scipy': 与scipy.fftpack.dct(type=1)兼容，额外乘以因子2
        window: 加窗类型，可选值：None, 'hann', 'hamming', 'blackman', 'rectangular'

    返回:
        DCT-I 系数序列，长度为n

    数学公式（不归一化）：
        X[k] = 0.5*x[0] + Σ(x[i] * cos(π*k*i/(N-1))) ，其中 i=1..N-2, k=0..N-1
               + 0.5*(-1)^k * x[N-1]
    """
    if n is None:
        n = len(x)

    if n < 2:
        raise ValueError("DCT-I 要求变换长度 n >= 2")

    if norm is not None and norm != 'ortho':
        raise ValueError("归一化类型 norm 只能是 None 或 'ortho'")

    if scale not in ('standard', 'scipy'):
        raise ValueError("scale 参数只能是 'standard' 或 'scipy'")

    x_processed = apply_window(x, window) if window is not None else x.copy()

    x_padded = x_processed[:n] + [0.0] * (n - len(x_processed))

    result = [0.0] * n

    for k in range(n):
        result[k] = 0.5 * x_padded[0]
        for i in range(1, n - 1):
            angle = math.pi * k * i / (n - 1)
            result[k] += x_padded[i] * math.cos(angle)
        result[k] += 0.5 * ((-1) ** k) * x_padded[n - 1]

    if norm == 'ortho':
        factor = math.sqrt(2.0 / (n - 1))
        for k in range(n):
            result[k] *= factor
        result[0] *= math.sqrt(0.5)
        result[n - 1] *= math.sqrt(0.5)
    elif scale == 'scipy':
        for k in range(n):
            result[k] *= 2.0

    return result


def idct_i(X: List[float], n: Optional[int] = None, norm: Optional[str] = None,
           scale: str = 'standard') -> List[float]:
    """
    实现 DCT-I 的逆变换（即 IDCT-I）。

    参数:
        X: DCT-I 系数序列
        n: 变换长度，默认使用输入序列的长度
        norm: 归一化类型，需与正向变换保持一致
        scale: 缩放约定，需与正向变换保持一致，可选值：
            - 'standard': 标准定义（默认）
            - 'scipy': 与scipy.fftpack.idct(type=1)兼容

    返回:
        重建的实序列
    """
    if n is None:
        n = len(X)

    if n < 2:
        raise ValueError("DCT-I 要求变换长度 n >= 2")

    if norm is not None and norm != 'ortho':
        raise ValueError("归一化类型 norm 只能是 None 或 'ortho'")

    if scale not in ('standard', 'scipy'):
        raise ValueError("scale 参数只能是 'standard' 或 'scipy'")

    X_padded = X[:n] + [0.0] * (n - len(X))

    X_scaled = X_padded.copy()
    if norm == 'ortho':
        factor = math.sqrt(2.0 / (n - 1))
        for k in range(n):
            X_scaled[k] /= factor
        X_scaled[0] /= math.sqrt(0.5)
        X_scaled[n - 1] /= math.sqrt(0.5)
    elif scale == 'scipy':
        for k in range(n):
            X_scaled[k] /= 2.0

    result = [0.0] * n

    for i in range(n):
        result[i] = 0.5 * X_scaled[0]
        for k in range(1, n - 1):
            angle = math.pi * k * i / (n - 1)
            result[i] += X_scaled[k] * math.cos(angle)
        result[i] += 0.5 * ((-1) ** i) * X_scaled[n - 1]
        result[i] *= 2.0 / (n - 1)

    return result


def dct_ii(x: List[float], n: Optional[int] = None, norm: Optional[str] = None,
           scale: str = 'standard', window: Optional[str] = None) -> List[float]:
    """
    实现离散余弦变换 DCT-II（最常用的DCT类型）。

    DCT-II 对应边界延拓方式：关于边界点半采样对称，
    延拓后周期为 2N，具有更好的边界连续性。

    参数:
        x: 输入实序列
        n: 变换长度，如果n大于输入长度，输入序列自动补零；
           如果n小于输入长度，输入序列自动截断。
           默认使用输入序列的长度。
        norm: 归一化类型，可选值：
            - None: 不归一化（默认）
            - 'ortho': 正交归一化，使得DCT成为正交变换
        scale: 缩放约定，可选值：
            - 'standard': 标准定义（默认），与大多数教材一致
            - 'scipy': 与scipy.fftpack.dct(type=2)兼容，额外乘以因子2
        window: 加窗类型，可选值：None, 'hann', 'hamming', 'blackman', 'rectangular'

    返回:
        DCT-II 系数序列，长度为n

    数学公式（standard, 不归一化）：
        X[k] = Σ(x[i] * cos(π*k*(2i+1)/(2N))) ，其中 i=0..N-1, k=0..N-1
    """
    if n is None:
        n = len(x)

    if n <= 0:
        raise ValueError("变换长度n必须是正整数")

    if norm is not None and norm != 'ortho':
        raise ValueError("归一化类型 norm 只能是 None 或 'ortho'")

    if scale not in ('standard', 'scipy'):
        raise ValueError("scale 参数只能是 'standard' 或 'scipy'")

    x_processed = apply_window(x, window) if window is not None else x.copy()

    x_padded = x_processed[:n] + [0.0] * (n - len(x_processed))

    result = [0.0] * n

    for k in range(n):
        for i in range(n):
            angle = math.pi * k * (2 * i + 1) / (2 * n)
            result[k] += x_padded[i] * math.cos(angle)

    if norm == 'ortho':
        result[0] *= math.sqrt(1.0 / n)
        factor = math.sqrt(2.0 / n)
        for k in range(1, n):
            result[k] *= factor
    elif scale == 'scipy':
        for k in range(n):
            result[k] *= 2.0

    return result


def idct_ii(X: List[float], n: Optional[int] = None, norm: Optional[str] = None,
            scale: str = 'standard') -> List[float]:
    """
    实现 DCT-II 的逆变换（即 IDCT-III，也就是 DCT-III）。

    参数:
        X: DCT-II 系数序列
        n: 变换长度，默认使用输入序列的长度
        norm: 归一化类型，需与正向变换保持一致
        scale: 缩放约定，需与正向变换保持一致，可选值：
            - 'standard': 标准定义（默认）
            - 'scipy': 与scipy.fftpack.idct(type=2)兼容

    返回:
        重建的实序列
    """
    if n is None:
        n = len(X)

    if n <= 0:
        raise ValueError("变换长度n必须是正整数")

    if norm is not None and norm != 'ortho':
        raise ValueError("归一化类型 norm 只能是 None 或 'ortho'")

    if scale not in ('standard', 'scipy'):
        raise ValueError("scale 参数只能是 'standard' 或 'scipy'")

    X_padded = X[:n] + [0.0] * (n - len(X))

    result = [0.0] * n

    if norm == 'ortho':
        for i in range(n):
            s = X_padded[0] * math.sqrt(1.0 / n)
            for k in range(1, n):
                angle = math.pi * k * (2 * i + 1) / (2 * n)
                s += X_padded[k] * math.sqrt(2.0 / n) * math.cos(angle)
            result[i] = s
    else:
        if scale == 'standard':
            for i in range(n):
                s = X_padded[0] / 2.0
                for k in range(1, n):
                    angle = math.pi * k * (2 * i + 1) / (2 * n)
                    s += X_padded[k] * math.cos(angle)
                result[i] = 2.0 * s / n
        else:
            for i in range(n):
                s = X_padded[0] / 2.0
                for k in range(1, n):
                    angle = math.pi * k * (2 * i + 1) / (2 * n)
                    s += X_padded[k] * math.cos(angle)
                result[i] = s / n

    return result


def dct_iii(x: List[float], n: Optional[int] = None, norm: Optional[str] = None,
            scale: str = 'standard', window: Optional[str] = None) -> List[float]:
    """
    实现离散余弦变换 DCT-III（DCT-II的逆变换形式）。

    DCT-III 对应边界延拓方式：关于边界点半采样反对称，
    常作为 DCT-II 的逆变换使用。

    参数:
        x: 输入实序列
        n: 变换长度，如果n大于输入长度，输入序列自动补零；
           如果n小于输入长度，输入序列自动截断。
           默认使用输入序列的长度。
        norm: 归一化类型，可选值：
            - None: 不归一化（默认）
            - 'ortho': 正交归一化
        scale: 缩放约定，可选值：
            - 'standard': 标准定义（默认）
            - 'scipy': 与scipy.fftpack.dct(type=3)兼容
        window: 加窗类型，可选值：None, 'hann', 'hamming', 'blackman', 'rectangular'

    返回:
        DCT-III 系数序列，长度为n

    数学公式（standard, 不归一化）：
        X[k] = 0.5*x[0] + Σ(x[i] * cos(π*(2k+1)*i/(2N))) ，其中 i=1..N-1, k=0..N-1
    """
    if n is None:
        n = len(x)

    if n <= 0:
        raise ValueError("变换长度n必须是正整数")

    if norm is not None and norm != 'ortho':
        raise ValueError("归一化类型 norm 只能是 None 或 'ortho'")

    if scale not in ('standard', 'scipy'):
        raise ValueError("scale 参数只能是 'standard' 或 'scipy'")

    x_processed = apply_window(x, window) if window is not None else x.copy()

    x_padded = x_processed[:n] + [0.0] * (n - len(x_processed))

    result = [0.0] * n

    for k in range(n):
        result[k] = 0.5 * x_padded[0]
        for i in range(1, n):
            angle = math.pi * (2 * k + 1) * i / (2 * n)
            result[k] += x_padded[i] * math.cos(angle)

    if norm == 'ortho':
        factor = math.sqrt(2.0 / n)
        for k in range(n):
            result[k] *= factor
    elif scale == 'scipy':
        for k in range(n):
            result[k] *= 2.0

    return result


def idct_iii(X: List[float], n: Optional[int] = None, norm: Optional[str] = None,
             scale: str = 'standard') -> List[float]:
    """
    实现 DCT-III 的逆变换（即 IDCT-III，也就是 DCT-II）。

    参数:
        X: DCT-III 系数序列
        n: 变换长度，默认使用输入序列的长度
        norm: 归一化类型，需与正向变换保持一致
        scale: 缩放约定，需与正向变换保持一致

    返回:
        重建的实序列
    """
    if n is None:
        n = len(X)

    if n <= 0:
        raise ValueError("变换长度n必须是正整数")

    if norm is not None and norm != 'ortho':
        raise ValueError("归一化类型 norm 只能是 None 或 'ortho'")

    if scale not in ('standard', 'scipy'):
        raise ValueError("scale 参数只能是 'standard' 或 'scipy'")

    X_padded = X[:n] + [0.0] * (n - len(X))

    X_scaled = X_padded.copy()
    if norm == 'ortho':
        factor = math.sqrt(2.0 / n)
        for k in range(n):
            X_scaled[k] /= factor
    elif scale == 'scipy':
        for k in range(n):
            X_scaled[k] /= 2.0

    result = [0.0] * n
    for i in range(n):
        for k in range(n):
            angle = math.pi * (2 * k + 1) * i / (2 * n)
            result[i] += X_scaled[k] * math.cos(angle)
        result[i] *= 2.0 / n

    return result


def dct_ii_2d(matrix: List[List[float]], norm: Optional[str] = None,
              scale: str = 'standard') -> List[List[float]]:
    """
    实现二维离散余弦变换 DCT-II（可分离变换：先对每行，再对每列）。

    适用于图像压缩（如JPEG中的8x8块变换）。

    参数:
        matrix: 输入二维矩阵（图像块），形状为 M x N
        norm: 归一化类型，可选值：None 或 'ortho'
        scale: 缩放约定，可选值：'standard' 或 'scipy'

    返回:
        二维 DCT-II 系数矩阵，形状为 M x N
    """
    if not matrix or not matrix[0]:
        return []

    rows = len(matrix)
    cols = len(matrix[0])

    row_result = []
    for i in range(rows):
        row_result.append(dct_ii(matrix[i], n=cols, norm=norm, scale=scale))

    result = [[0.0] * cols for _ in range(rows)]
    for j in range(cols):
        col = [row_result[i][j] for i in range(rows)]
        col_dct = dct_ii(col, n=rows, norm=norm, scale=scale)
        for i in range(rows):
            result[i][j] = col_dct[i]

    return result


def idct_ii_2d(matrix: List[List[float]], norm: Optional[str] = None,
               scale: str = 'standard') -> List[List[float]]:
    """
    实现二维逆离散余弦变换 IDCT-II。

    参数:
        matrix: 二维 DCT 系数矩阵，形状为 M x N
        norm: 归一化类型，需与正向变换保持一致
        scale: 缩放约定，需与正向变换保持一致

    返回:
        重建的二维矩阵，形状为 M x N
    """
    if not matrix or not matrix[0]:
        return []

    rows = len(matrix)
    cols = len(matrix[0])

    row_result = []
    for i in range(rows):
        row_result.append(idct_ii(matrix[i], n=cols, norm=norm, scale=scale))

    result = [[0.0] * cols for _ in range(rows)]
    for j in range(cols):
        col = [row_result[i][j] for i in range(rows)]
        col_idct = idct_ii(col, n=rows, norm=norm, scale=scale)
        for i in range(rows):
            result[i][j] = col_idct[i]

    return result


def energy_compactness_analysis(coefficients: List[float] | List[List[float]],
                                energy_ratio: float = 0.95) -> dict:
    """
    分析 DCT 系数的能量集中度。

    计算保留指定比例能量所需的系数数量。
    能量定义为系数的平方和。

    参数:
        coefficients: DCT 系数（一维列表或二维矩阵）
        energy_ratio: 目标能量比例 (0, 1]，默认 0.95

    返回:
        包含分析结果的字典：
            - total_energy: 总能量
            - num_coefficients: 总系数数量
            - num_kept: 保留的系数数量
            - kept_ratio: 保留系数比例
            - actual_energy_ratio: 实际达到的能量比例
            - sorted_coefficients: 按绝对值降序排列的系数
    """
    if isinstance(coefficients[0], list):
        flat = []
        for row in coefficients:
            flat.extend(row)
        coeffs = flat
    else:
        coeffs = coefficients

    if energy_ratio <= 0 or energy_ratio > 1:
        raise ValueError("energy_ratio 必须在 (0, 1] 范围内")

    abs_coeffs = [(abs(c), c) for c in coeffs]
    abs_coeffs.sort(reverse=True, key=lambda x: x[0])

    total_energy = sum(c ** 2 for c in coeffs)

    if total_energy == 0:
        return {
            'total_energy': 0.0,
            'num_coefficients': len(coeffs),
            'num_kept': 0,
            'kept_ratio': 0.0,
            'actual_energy_ratio': 0.0,
            'sorted_coefficients': []
        }

    accumulated_energy = 0.0
    num_kept = 0
    sorted_coeffs = []

    for abs_c, c in abs_coeffs:
        accumulated_energy += c ** 2
        num_kept += 1
        sorted_coeffs.append(c)
        if accumulated_energy / total_energy >= energy_ratio:
            break

    actual_ratio = accumulated_energy / total_energy

    return {
        'total_energy': total_energy,
        'num_coefficients': len(coeffs),
        'num_kept': num_kept,
        'kept_ratio': num_kept / len(coeffs),
        'actual_energy_ratio': actual_ratio,
        'sorted_coefficients': sorted_coeffs
    }


def print_matrix(matrix: List[List[float]], title: str = "", precision: int = 4):
    """
    格式化打印二维矩阵。
    """
    if title:
        print(f"\n{title}:")
    for row in matrix:
        formatted = [f"{v:>{precision + 4}.{precision}f}" for v in row]
        print("  " + " ".join(formatted))


if __name__ == '__main__':
    print("=" * 70)
    print("DCT 完整演示：一维/二维变换 + 能量集中度分析")
    print("=" * 70)

    x = [1.0, 2.0, 3.0, 4.0]
    print("\n输入序列:", x)

    print("\n" + "-" * 70)
    print("1. 一维 DCT 变体演示")
    print("-" * 70)

    print("\n【DCT-I】（偶对称边界延拓）")
    X_i = dct_i(x)
    print("  DCT-I (不归一化):", [round(v, 6) for v in X_i])
    x_recon_i = idct_i(X_i)
    print("  IDCT-I 重建:", [round(v, 6) for v in x_recon_i])

    print("\n【DCT-II】（半采样偶对称，最常用）")
    X_ii = dct_ii(x)
    print("  DCT-II (不归一化):", [round(v, 6) for v in X_ii])
    x_recon_ii = idct_ii(X_ii)
    print("  IDCT-II 重建:", [round(v, 6) for v in x_recon_ii])

    print("\n【DCT-III】（半采样奇对称，DCT-II的逆）")
    X_iii = dct_iii(x)
    print("  DCT-III (不归一化):", [round(v, 6) for v in X_iii])
    x_recon_iii = idct_iii(X_iii)
    print("  IDCT-III 重建:", [round(v, 6) for v in x_recon_iii])

    print("\n" + "-" * 70)
    print("2. 加窗效果演示（减少边界效应）")
    print("-" * 70)
    print("  原始序列:", x)
    x_hann = apply_window(x, 'hann')
    print("  汉宁窗后:", [round(v, 6) for v in x_hann])
    x_hamming = apply_window(x, 'hamming')
    print("  汉明窗后:", [round(v, 6) for v in x_hamming])
    X_ii_hann = dct_ii(x, window='hann')
    print("  DCT-II (加汉宁窗):", [round(v, 6) for v in X_ii_hann])

    print("\n" + "-" * 70)
    print("3. 二维 DCT-II 演示（8x8 图像块）")
    print("-" * 70)

    test_block = [
        [52, 55, 61, 66, 70, 61, 64, 73],
        [63, 59, 55, 90, 109, 85, 69, 72],
        [62, 59, 68, 113, 144, 104, 66, 73],
        [63, 58, 71, 122, 154, 106, 70, 69],
        [67, 61, 68, 104, 126, 88, 68, 70],
        [79, 65, 60, 70, 77, 68, 58, 75],
        [85, 71, 64, 59, 55, 61, 65, 83],
        [87, 79, 69, 68, 65, 76, 78, 94]
    ]
    print("\n  原始 8x8 图像块（亮度值）:")
    print_matrix(test_block, precision=0)

    dct_block = dct_ii_2d(test_block, norm='ortho')
    print("\n  二维 DCT-II 系数（正交归一化）:")
    print_matrix(dct_block, precision=2)

    recon_block = idct_ii_2d(dct_block, norm='ortho')
    print("\n  重建图像块:")
    print_matrix(recon_block, precision=1)

    print("\n" + "-" * 70)
    print("4. 能量集中度分析")
    print("-" * 70)

    print("\n  【一维 DCT-II 系数能量分析】")
    analysis_1d = energy_compactness_analysis(X_ii, energy_ratio=0.95)
    print(f"    总系数数量: {analysis_1d['num_coefficients']}")
    print(f"    总能量: {analysis_1d['total_energy']:.4f}")
    print(f"    保留 95% 能量需要 {analysis_1d['num_kept']} 个系数")
    print(f"    压缩率: {analysis_1d['kept_ratio']*100:.1f}%")
    print(f"    实际能量保留: {analysis_1d['actual_energy_ratio']*100:.2f}%")

    print("\n  【二维 DCT 系数能量分析 (8x8块)】")
    analysis_2d = energy_compactness_analysis(dct_block, energy_ratio=0.99)
    print(f"    总系数数量: {analysis_2d['num_coefficients']} (64)")
    print(f"    保留 99% 能量需要 {analysis_2d['num_kept']} 个系数")
    print(f"    压缩率: {analysis_2d['kept_ratio']*100:.1f}% ({analysis_2d['num_kept']}/64)")
    print(f"    实际能量保留: {analysis_2d['actual_energy_ratio']*100:.2f}%")

    print("\n  【不同能量比例对比】")
    for ratio in [0.90, 0.95, 0.99]:
        a = energy_compactness_analysis(dct_block, energy_ratio=ratio)
        print(f"    保留 {ratio*100:3.0f}% 能量: {a['num_kept']:2d} 个系数 ({a['kept_ratio']*100:4.1f}%)")

    print("\n" + "-" * 70)
    print("5. 边界延拓方式对比")
    print("-" * 70)
    print("""
    DCT 变体    |   延拓周期   | 边界处特性          |  典型应用
    -----------|------------|--------------------|-----------------
    DCT-I     |  2(N-1)    | 关于端点偶对称，端点重复 | 对称信号处理
    DCT-II    |   2N       | 半采样点偶对称，平滑过渡 | 图像压缩(JPEG)
    DCT-III   |   2N       | 半采样点奇对称        | DCT-II的逆变换
    DFT       |    N       | 硬截断，边界不连续     | 频谱分析
    """)

    print("\n" + "=" * 70)
    print("DCT 能量集中特性：")
    print("-" * 70)
    print("  DCT 将信号能量集中在低频系数（左上角），这是图像压缩的基础。")
    print("  典型 8x8 图像块：~15% 的系数即可保留 99% 的能量。")
    print("  高频系数通常很小，可以量化为零，实现高压缩比。")
    print("=" * 70)

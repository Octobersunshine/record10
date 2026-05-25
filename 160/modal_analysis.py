import numpy as np
from scipy.linalg import eigh


def modal_analysis(K, M, num_modes=None, normalize='mass'):
    """
    有限元模态分析（求解广义特征值问题）
    
    求解方程: K @ Φ = ω² @ M @ Φ
    其中: K - 刚度矩阵, M - 质量矩阵, ω - 固有角频率, Φ - 振型向量
    
    参数:
        K: 刚度矩阵 (n x n)
        M: 质量矩阵 (n x n)
        num_modes: 要求解的模态数 (默认求解所有模态)
        normalize: 振型归一化方式 ('mass'质量归一化, 'max'最大值归一化, None不归一化)
    
    返回:
        frequencies: 固有频率 (Hz), 按从小到大排序
        natural_frequencies: 固有角频率 (rad/s)
        mode_shapes: 振型矩阵 (每列为一个振型)
    """
    K = np.asarray(K, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    
    if K.shape[0] != K.shape[1] or M.shape[0] != M.shape[1]:
        raise ValueError("刚度矩阵和质量矩阵必须是方阵")
    if K.shape != M.shape:
        raise ValueError("刚度矩阵和质量矩阵维度必须相同")
    
    n = K.shape[0]
    if num_modes is None:
        num_modes = n
    num_modes = min(num_modes, n)
    
    eigenvalues, eigenvectors = eigh(K, M, subset_by_index=[0, num_modes - 1])
    
    eigenvalues = np.maximum(eigenvalues, 0.0)
    natural_frequencies = np.sqrt(eigenvalues)
    frequencies = natural_frequencies / (2 * np.pi)
    
    mode_shapes = eigenvectors.copy()
    
    if normalize == 'mass':
        for i in range(num_modes):
            phi = mode_shapes[:, i]
            mass_norm = phi.T @ M @ phi
            if mass_norm > 1e-12:
                mode_shapes[:, i] = phi / np.sqrt(mass_norm)
    
    elif normalize == 'max':
        for i in range(num_modes):
            phi = mode_shapes[:, i]
            max_val = np.max(np.abs(phi))
            if max_val > 1e-12:
                mode_shapes[:, i] = phi / max_val
    
    sort_idx = np.argsort(frequencies)
    frequencies = frequencies[sort_idx]
    natural_frequencies = natural_frequencies[sort_idx]
    mode_shapes = mode_shapes[:, sort_idx]
    
    return frequencies, natural_frequencies, mode_shapes


def print_modal_results(frequencies, natural_frequencies, mode_shapes, precision=4):
    """打印模态分析结果"""
    num_modes = len(frequencies)
    
    print("=" * 70)
    print("模态分析结果")
    print("=" * 70)
    print(f"{'阶数':<6} {'角频率(rad/s)':<18} {'频率(Hz)':<15} {'周期(s)':<15}")
    print("-" * 70)
    
    for i in range(num_modes):
        freq = frequencies[i]
        omega = natural_frequencies[i]
        period = 1.0 / freq if freq > 1e-10 else np.inf
        print(f"{i+1:<6} {omega:<18.{precision}f} {freq:<15.{precision}f} {period:<15.{precision}f}")
    
    print("-" * 70)
    print("\n振型矩阵 (每列为一个振型):")
    print(np.round(mode_shapes, precision))
    print()


def verify_orthogonality(K, M, mode_shapes, frequencies, tol=1e-6):
    """验证振型正交性"""
    num_modes = mode_shapes.shape[1]
    Phi = mode_shapes
    
    mass_ortho = Phi.T @ M @ Phi
    stiffness_ortho = Phi.T @ K @ Phi
    
    print("=" * 70)
    print("振型正交性验证")
    print("=" * 70)
    
    print("\n质量正交性 (Φ^T M Φ 应为对角阵):")
    print(np.round(mass_ortho, 8))
    
    print("\n刚度正交性 (Φ^T K Φ 应为对角阵):")
    print(np.round(stiffness_ortho, 8))
    
    omega_diag = np.diag((2 * np.pi * frequencies) ** 2)
    error = np.max(np.abs(stiffness_ortho - np.diag(np.diag(stiffness_ortho))))
    print(f"\n非对角元最大误差: {error:.2e}")
    print(f"正交性验证: {'通过' if error < tol else '未通过'}")
    
    return error < tol


def example_spring_mass_system():
    """示例: 3自由度弹簧-质量系统"""
    print("\n" + "=" * 70)
    print("示例: 3自由度弹簧-质量系统")
    print("=" * 70)
    
    m1, m2, m3 = 1.0, 1.0, 1.0
    k1, k2, k3, k4 = 1000.0, 1000.0, 1000.0, 1000.0
    
    K = np.array([
        [k1 + k2, -k2, 0],
        [-k2, k2 + k3, -k3],
        [0, -k3, k3 + k4]
    ])
    
    M = np.diag([m1, m2, m3])
    
    print("\n刚度矩阵 K:")
    print(K)
    print("\n质量矩阵 M:")
    print(M)
    print()
    
    frequencies, natural_frequencies, mode_shapes = modal_analysis(
        K, M, normalize='mass'
    )
    
    print_modal_results(frequencies, natural_frequencies, mode_shapes)
    verify_orthogonality(K, M, mode_shapes, frequencies)
    
    return frequencies, natural_frequencies, mode_shapes


def example_simple_beam():
    """示例: 简支梁的简化有限元模型"""
    print("\n" + "=" * 70)
    print("示例: 简支梁简化模型 (4个集中质量)")
    print("=" * 70)
    
    n = 4
    L = 1.0
    m_total = 10.0
    m = m_total / n
    
    EI = 1e6
    k = EI / (L / (n + 1)) ** 3 / 100
    
    K = np.zeros((n, n))
    for i in range(n):
        K[i, i] = 2 * k
        if i > 0:
            K[i, i-1] = -k
        if i < n - 1:
            K[i, i+1] = -k
    
    M = np.diag([m] * n)
    
    print(f"\n梁长度: {L} m")
    print(f"总质量: {m_total} kg")
    print(f"单元数: {n}")
    print(f"EI: {EI} N·m²")
    print()
    
    frequencies, natural_frequencies, mode_shapes = modal_analysis(
        K, M, normalize='max'
    )
    
    print_modal_results(frequencies, natural_frequencies, mode_shapes)
    
    return frequencies, natural_frequencies, mode_shapes


if __name__ == "__main__":
    print("有限元模态分析工具")
    print("求解广义特征值问题: KΦ = ω²MΦ")
    
    example_spring_mass_system()
    example_simple_beam()
    
    print("\n" + "=" * 70)
    print("使用说明:")
    print("1. 定义刚度矩阵 K 和质量矩阵 M")
    print("2. 调用: frequencies, natural_frequencies, mode_shapes = modal_analysis(K, M)")
    print("3. 参数 num_modes 可指定求解的模态数")
    print("4. 参数 normalize 可选 'mass'质量归一化 或 'max'最大值归一化")
    print("=" * 70)

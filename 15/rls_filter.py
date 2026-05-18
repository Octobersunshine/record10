import numpy as np


def rls_filter(input_signal, desired_signal, order, forgetting_factor=0.99, delta=0.01, regularization=1e-8, reset_interval=None, adaptive_lambda=False, lambda_min=0.95, lambda_max=0.999, alpha=0.1, error_window=10):
    """
    RLS（递归最小二乘）滤波器实现，用于系统辨识（数值稳定版 + 自适应遗忘因子）
    
    参数:
        input_signal: 输入信号数组 (n_samples,)
        desired_signal: 期望信号数组 (n_samples,)
        order: 滤波器阶数
        forgetting_factor: 初始遗忘因子 (0 < lambda <= 1)，默认0.99
        delta: 初始化P矩阵的系数，默认0.01
        regularization: 对角正则化项，保证正定性，默认1e-8
        reset_interval: P矩阵重置间隔，None表示不重置，默认None
        adaptive_lambda: 是否启用自适应遗忘因子，默认False
        lambda_min: 遗忘因子下界，默认0.95
        lambda_max: 遗忘因子上界，默认0.999
        alpha: 遗忘因子调整步长，默认0.1
        error_window: 误差平滑窗口大小，默认10
    
    返回:
        weights: 最终滤波器系数 (order,)
        weights_history: 每一步的滤波器系数 (n_samples, order)
        error_history: 每一步的预测误差 (n_samples,)
        lambda_history: 每一步的遗忘因子值 (n_samples,)
    """
    n_samples = len(input_signal)
    assert len(desired_signal) == n_samples, "输入信号和期望信号长度必须相同"
    assert 0 < forgetting_factor <= 1, "遗忘因子必须在(0, 1]范围内"
    assert lambda_min < lambda_max, "lambda_min必须小于lambda_max"
    
    weights = np.zeros(order)
    weights_history = np.zeros((n_samples, order))
    error_history = np.zeros(n_samples)
    lambda_history = np.zeros(n_samples)
    
    P = np.eye(order) / delta
    input_buffer = np.zeros(order)
    
    current_lambda = forgetting_factor
    error_buffer = np.zeros(error_window)
    
    for n in range(n_samples):
        input_buffer = np.roll(input_buffer, 1)
        input_buffer[0] = input_signal[n]
        
        prediction = np.dot(weights, input_buffer)
        error = desired_signal[n] - prediction
        
        if adaptive_lambda:
            error_buffer = np.roll(error_buffer, 1)
            error_buffer[0] = abs(error)
            avg_error = np.mean(error_buffer) if n >= error_window else np.mean(error_buffer[:n+1])
            
            if avg_error > 0:
                error_ratio = abs(error) / (avg_error + 1e-10)
            else:
                error_ratio = 1.0
            
            if error_ratio > 1.5:
                current_lambda = current_lambda - alpha * (current_lambda - lambda_min)
            elif error_ratio < 0.5:
                current_lambda = current_lambda + alpha * (lambda_max - current_lambda)
            
            current_lambda = np.clip(current_lambda, lambda_min, lambda_max)
        
        denominator = current_lambda + input_buffer @ P @ input_buffer
        if abs(denominator) < 1e-15:
            denominator = 1e-15
        K = (P @ input_buffer) / denominator
        weights = weights + K * error
        
        P = P - np.outer(K, input_buffer) @ P
        
        P = (P + P.T) / 2
        
        P = P / current_lambda
        
        P = P + regularization * np.eye(order)
        
        if reset_interval is not None and (n + 1) % reset_interval == 0:
            min_eig = np.min(np.linalg.eigvalsh(P))
            if min_eig < 1e-6:
                P = np.eye(order) / delta
        
        weights_history[n] = weights
        error_history[n] = error
        lambda_history[n] = current_lambda
    
    return weights, weights_history, error_history, lambda_history


def identify_system(true_system, n_samples=1000, snr_db=30, **kwargs):
    """
    使用RLS滤波器进行系统辨识测试
    
    参数:
        true_system: 真实系统系数
        n_samples: 采样点数
        snr_db: 信噪比(dB)
        **kwargs: 传递给rls_filter的参数
    
    返回:
        identified_weights: 辨识得到的系数
        mse: 均方误差
        lambda_history: 遗忘因子历史（如果启用自适应）
    """
    order = len(true_system)
    input_signal = np.random.randn(n_samples)
    
    true_output = np.convolve(input_signal, true_system, mode='same')
    
    signal_power = np.var(true_output)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power) * np.random.randn(n_samples)
    desired_signal = true_output + noise
    
    identified_weights, _, error_history, lambda_history = rls_filter(input_signal, desired_signal, order, **kwargs)
    
    mse = np.mean((identified_weights - true_system) ** 2)
    
    return identified_weights, mse, lambda_history


def check_matrix_properties(P):
    """
    检查矩阵的对称性和正定性
    """
    is_symmetric = np.allclose(P, P.T, rtol=1e-10)
    eigenvalues = np.linalg.eigvalsh(P)
    min_eigenvalue = np.min(eigenvalues)
    is_positive_definite = min_eigenvalue > 0
    
    return {
        'is_symmetric': is_symmetric,
        'is_positive_definite': is_positive_definite,
        'min_eigenvalue': min_eigenvalue,
        'max_eigenvalue': np.max(eigenvalues),
        'condition_number': np.max(eigenvalues) / np.min(eigenvalues) if min_eigenvalue > 0 else np.inf
    }


def test_numerical_stability(n_iterations=5000):
    """
    测试P矩阵的数值稳定性
    """
    order = 4
    input_signal = np.random.randn(n_iterations)
    true_system = np.array([0.5, 0.3, 0.2, -0.1])
    desired_signal = np.convolve(input_signal, true_system, mode='same')
    
    n_samples = len(input_signal)
    forgetting_factor = 0.99
    delta = 0.01
    regularization = 1e-8
    
    weights = np.zeros(order)
    P = np.eye(order) / delta
    input_buffer = np.zeros(order)
    
    symmetry_checks = []
    eigenvalue_checks = []
    
    for n in range(n_samples):
        input_buffer = np.roll(input_buffer, 1)
        input_buffer[0] = input_signal[n]
        
        denominator = forgetting_factor + input_buffer @ P @ input_buffer
        if abs(denominator) < 1e-15:
            denominator = 1e-15
        K = (P @ input_buffer) / denominator
        
        P = P - np.outer(K, input_buffer) @ P
        P = (P + P.T) / 2
        P = P / forgetting_factor
        P = P + regularization * np.eye(order)
        
        if n % 500 == 0:
            props = check_matrix_properties(P)
            symmetry_checks.append(props['is_symmetric'])
            eigenvalue_checks.append(props['min_eigenvalue'])
            print(f"第{n}步 - 对称:{props['is_symmetric']}, 最小特征值:{props['min_eigenvalue']:.2e}, 条件数:{props['condition_number']:.2e}")
    
    print(f"\n始终保持对称: {all(symmetry_checks)}")
    print(f"始终保持正定 (最小特征值>0): {all(eig > 0 for eig in eigenvalue_checks)}")
    
    return P


def test_time_varying_system():
    """
    测试自适应遗忘因子在时变系统中的性能
    """
    n_samples = 3000
    order = 4
    input_signal = np.random.randn(n_samples)
    
    system1 = np.array([0.5, 0.3, 0.2, -0.1])
    system2 = np.array([0.3, 0.5, -0.2, 0.1])
    
    true_output1 = np.convolve(input_signal[:n_samples//2], system1, mode='same')
    true_output2 = np.convolve(input_signal[n_samples//2:], system2, mode='same')
    true_output = np.concatenate([true_output1, true_output2])
    
    noise = 0.01 * np.random.randn(n_samples)
    desired_signal = true_output + noise
    
    weights_fixed, _, error_fixed, _ = rls_filter(
        input_signal, desired_signal, order, 
        forgetting_factor=0.99,
        adaptive_lambda=False
    )
    
    weights_adaptive, _, error_adaptive, lambda_hist = rls_filter(
        input_signal, desired_signal, order,
        forgetting_factor=0.99,
        adaptive_lambda=True,
        lambda_min=0.95,
        lambda_max=0.995,
        alpha=0.2
    )
    
    mse_fixed_1 = np.mean((weights_fixed - system1)**2)
    mse_adaptive_1 = np.mean((weights_adaptive - system1)**2)
    mse_fixed_2 = np.mean((weights_fixed - system2)**2)
    mse_adaptive_2 = np.mean((weights_adaptive - system2)**2)
    
    print(f"固定lambda (0.99):")
    print(f"  系统1 MSE: {mse_fixed_1:.2e}")
    print(f"  系统2 MSE: {mse_fixed_2:.2e}")
    print(f"  最终预测误差MSE: {np.mean(error_fixed[-500:]**2):.2e}")
    
    print(f"\n自适应lambda:")
    print(f"  系统1 MSE: {mse_adaptive_1:.2e}")
    print(f"  系统2 MSE: {mse_adaptive_2:.2e}")
    print(f"  最终预测误差MSE: {np.mean(error_adaptive[-500:]**2):.2e}")
    print(f"  lambda范围: [{np.min(lambda_hist):.4f}, {np.max(lambda_hist):.4f}]")
    print(f"  最终lambda: {lambda_hist[-1]:.4f}")
    
    return lambda_hist


if __name__ == "__main__":
    print("="*60)
    print("测试1: 系统辨识性能（固定遗忘因子）")
    print("="*60)
    true_system = np.array([0.5, 0.3, 0.2, -0.1])
    
    identified_weights, mse, _ = identify_system(
        true_system, 
        n_samples=2000, 
        snr_db=30,
        forgetting_factor=0.995
    )
    
    print("真实系统系数:", true_system)
    print("辨识得到系数:", identified_weights)
    print(f"系数均方误差: {mse:.2e}")
    
    order = len(true_system)
    input_signal = np.random.randn(1000)
    desired_signal = np.convolve(input_signal, true_system, mode='same') + 0.01 * np.random.randn(1000)
    
    weights, weights_history, error_history, _ = rls_filter(input_signal, desired_signal, order)
    
    print(f"\n最终均方预测误差: {np.mean(error_history[-100:]**2):.2e}")
    
    print("\n" + "="*60)
    print("测试2: P矩阵数值稳定性")
    print("="*60)
    final_P = test_numerical_stability(n_iterations=5000)
    
    print("\n" + "="*60)
    print("最终P矩阵性质:")
    print("="*60)
    props = check_matrix_properties(final_P)
    print(f"对称: {props['is_symmetric']}")
    print(f"正定: {props['is_positive_definite']}")
    print(f"最小特征值: {props['min_eigenvalue']:.2e}")
    print(f"最大特征值: {props['max_eigenvalue']:.2e}")
    print(f"条件数: {props['condition_number']:.2e}")
    
    print("\n" + "="*60)
    print("测试3: 自适应遗忘因子 vs 固定遗忘因子（时变系统）")
    print("="*60)
    lambda_hist = test_time_varying_system()
    
    print("\n" + "="*60)
    print("测试4: 系统辨识性能（自适应遗忘因子）")
    print("="*60)
    identified_weights_adaptive, mse_adaptive, lambda_hist = identify_system(
        true_system, 
        n_samples=2000, 
        snr_db=30,
        adaptive_lambda=True,
        lambda_min=0.95,
        lambda_max=0.999
    )
    
    print("真实系统系数:", true_system)
    print("辨识得到系数:", identified_weights_adaptive)
    print(f"系数均方误差: {mse_adaptive:.2e}")
    print(f"最终遗忘因子: {lambda_hist[-1]:.4f}")

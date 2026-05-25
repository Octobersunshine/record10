import numpy as np
from scipy.optimize import minimize
from modal_analysis_lanczos import LanczosSolver
from ema_polymax import PolyMAX, generate_synthetic_frf
import warnings


class ModelUpdating:
    """
    有限元模型修正（Model Updating）
    
    使用试验模态分析（EMA）识别的模态参数来修正有限元模型
    """
    
    def __init__(self, K_initial, M_initial, sensor_indices=None):
        """
        初始化模型修正
        
        参数:
            K_initial: 初始刚度矩阵
            M_initial: 初始质量矩阵
            sensor_indices: 传感器位置索引（用于振型匹配）
        """
        self.K0 = np.asarray(K_initial, dtype=np.float64)
        self.M0 = np.asarray(M_initial, dtype=np.float64)
        self.n = self.K0.shape[0]
        
        if sensor_indices is None:
            self.sensor_indices = list(range(self.n))
        else:
            self.sensor_indices = sensor_indices
        
        self.N_sensors = len(self.sensor_indices)
        
        self.ema_frequencies = None
        self.ema_damping = None
        self.ema_mode_shapes = None
        
        self.update_params = None
        self.updated_K = None
        self.updated_M = None
    
    def set_ema_data(self, frequencies, damping=None, mode_shapes=None):
        """
        设置EMA识别的试验数据
        
        参数:
            frequencies: EMA识别的固有频率
            damping: EMA识别的阻尼比（可选）
            mode_shapes: EMA识别的振型（N_sensors x N_modes）
        """
        self.ema_frequencies = np.asarray(frequencies)
        self.ema_damping = np.asarray(damping) if damping is not None else None
        
        if mode_shapes is not None:
            self.ema_mode_shapes = np.asarray(mode_shapes)
            if self.ema_mode_shapes.shape[0] != self.N_sensors:
                raise ValueError(f"振型传感器数 {self.ema_mode_shapes.shape[0]} 与设置 {self.N_sensors} 不符")
    
    def compute_mac(self, mode1, mode2):
        """
        计算模态置信准则（Modal Assurance Criterion）
        
        MAC = |mode1^H mode2|^2 / (|mode1|^2 |mode2|^2)
        """
        mode1 = np.asarray(mode1).flatten()
        mode2 = np.asarray(mode2).flatten()
        
        numerator = np.abs(np.vdot(mode1, mode2)) ** 2
        denominator = (np.vdot(mode1, mode1) * np.vdot(mode2, mode2))
        
        return numerator / denominator if denominator > 0 else 0.0
    
    def compute_frequency_error(self, fem_frequencies, weights=None):
        """
        计算频率误差目标函数
        """
        if self.ema_frequencies is None:
            raise ValueError("请先设置EMA数据")
        
        n_modes = min(len(fem_frequencies), len(self.ema_frequencies))
        
        if weights is None:
            weights = np.ones(n_modes) / n_modes
        
        error = 0.0
        for i in range(n_modes):
            rel_error = (fem_frequencies[i] - self.ema_frequencies[i]) / self.ema_frequencies[i]
            error += weights[i] * rel_error ** 2
        
        return error
    
    def compute_mac_error(self, fem_modes, weights=None):
        """
        计算MAC误差目标函数（1 - MAC，越小越好）
        """
        if self.ema_mode_shapes is None:
            return 0.0
        
        n_modes = min(fem_modes.shape[1], self.ema_mode_shapes.shape[1])
        
        if weights is None:
            weights = np.ones(n_modes) / n_modes
        
        fem_modes_sensor = fem_modes[self.sensor_indices, :]
        
        error = 0.0
        for i in range(n_modes):
            mac = self.compute_mac(fem_modes_sensor[:, i], self.ema_mode_shapes[:, i])
            error += weights[i] * (1 - mac)
        
        return error
    
    def objective_function(self, x, param_type='stiffness', 
                          freq_weight=1.0, mac_weight=1.0,
                          num_modes=None):
        """
        目标函数：频率误差 + MAC误差
        
        参数:
            x: 设计变量向量
            param_type: 'stiffness' 或 'mass'
        """
        if num_modes is None:
            num_modes = len(self.ema_frequencies) if self.ema_frequencies is not None else 5
        
        if param_type == 'stiffness':
            K = self._update_stiffness(x)
            M = self.M0
        elif param_type == 'mass':
            K = self.K0
            M = self._update_mass(x)
        else:
            raise ValueError("param_type must be 'stiffness' or 'mass'")
        
        try:
            solver = LanczosSolver(K, M)
            solver.solve(num_eigenvalues=num_modes, tol=1e-9, verbose=False)
            
            fem_freqs = solver.get_frequencies()
            fem_modes = solver.eigenvectors
            
            freq_err = self.compute_frequency_error(fem_freqs)
            mac_err = self.compute_mac_error(fem_modes)
            
            total_error = freq_weight * freq_err + mac_weight * mac_err
            
            return total_error
        
        except Exception as e:
            warnings.warn(f"求解器错误: {e}")
            return 1e10
    
    def _update_stiffness(self, x):
        """
        更新刚度矩阵（按单元/子区域缩放）
        
        x: 缩放因子向量
        """
        K = self.K0.copy()
        
        n_params = len(x)
        elements_per_param = self.n // n_params
        
        for i, scale in enumerate(x):
            start = i * elements_per_param
            end = min(start + elements_per_param, self.n)
            
            for j in range(start, end):
                K[j, j] *= scale
                if j > start:
                    K[j, j-1] *= np.sqrt(scale)
                    K[j-1, j] *= np.sqrt(scale)
        
        return K
    
    def _update_mass(self, x):
        """
        更新质量矩阵（按区域缩放）
        """
        M = self.M0.copy()
        
        n_params = len(x)
        elements_per_param = self.n // n_params
        
        for i, scale in enumerate(x):
            start = i * elements_per_param
            end = min(start + elements_per_param, self.n)
            M[start:end, start:end] *= scale
        
        return M
    
    def update_model(self, initial_guess, param_type='stiffness',
                     bounds=None, method='L-BFGS-B',
                     freq_weight=1.0, mac_weight=1.0,
                     num_modes=None, verbose=True):
        """
        执行模型修正
        
        参数:
            initial_guess: 初始设计变量
            param_type: 'stiffness' 或 'mass'
            bounds: 变量上下界 (min, max) 或 None
            method: 优化方法
            freq_weight: 频率误差权重
            mac_weight: MAC误差权重
            num_modes: 考虑的模态数
            verbose: 是否打印过程
        
        返回:
            result: 优化结果
        """
        if verbose:
            print("开始模型修正...")
            print(f"参数类型: {param_type}")
            print(f"初始参数: {initial_guess}")
        
        if num_modes is None:
            num_modes = len(self.ema_frequencies)
        
        if bounds is None:
            bounds = [(0.5, 2.0) for _ in initial_guess]
        
        def objective(x):
            return self.objective_function(x, param_type, freq_weight, mac_weight, num_modes)
        
        result = minimize(
            objective,
            initial_guess,
            method=method,
            bounds=bounds,
            options={'disp': verbose, 'maxiter': 100}
        )
        
        self.update_params = result.x
        
        if param_type == 'stiffness':
            self.updated_K = self._update_stiffness(result.x)
            self.updated_M = self.M0
        else:
            self.updated_K = self.K0
            self.updated_M = self._update_mass(result.x)
        
        if verbose:
            print(f"\n优化完成!")
            print(f"最终参数: {result.x}")
            print(f"目标函数值: {result.fun:.6e}")
        
        return result
    
    def compare_results(self, num_modes=None):
        """
        对比修正前后与EMA数据的差异
        """
        if num_modes is None:
            num_modes = len(self.ema_frequencies)
        
        print("\n" + "=" * 80)
        print("模型修正前后对比")
        print("=" * 80)
        
        solver_initial = LanczosSolver(self.K0, self.M0)
        solver_initial.solve(num_eigenvalues=num_modes, tol=1e-9, verbose=False)
        initial_freqs = solver_initial.get_frequencies()
        initial_modes = solver_initial.eigenvectors
        
        if self.updated_K is not None:
            solver_updated = LanczosSolver(self.updated_K, self.updated_M)
            solver_updated.solve(num_eigenvalues=num_modes, tol=1e-9, verbose=False)
            updated_freqs = solver_updated.get_frequencies()
            updated_modes = solver_updated.eigenvectors
        else:
            updated_freqs = None
            updated_modes = None
        
        print(f"\n{'模态':<6} {'EMA频率':<12} {'初始FEM':<12} {'修正FEM':<12} {'初始误差':<12} {'修正误差':<12}")
        print("-" * 80)
        
        for i in range(num_modes):
            ema_f = self.ema_frequencies[i]
            init_f = initial_freqs[i]
            upd_f = updated_freqs[i] if updated_freqs is not None else 0
            init_err = abs(init_f - ema_f) / ema_f * 100
            upd_err = abs(upd_f - ema_f) / ema_f * 100 if updated_freqs is not None else 0
            
            print(f"{i+1:<6} {ema_f:<12.4f} {init_f:<12.4f} {upd_f:<12.4f} {init_err:<11.2f}% {upd_err:<11.2f}%")
        
        if self.ema_mode_shapes is not None:
            print("\nMAC对比:")
            print("-" * 50)
            print(f"{'模态':<6} {'初始MAC':<12} {'修正MAC':<12}")
            
            init_modes_sensor = initial_modes[self.sensor_indices, :]
            upd_modes_sensor = updated_modes[self.sensor_indices, :] if updated_modes is not None else None
            
            for i in range(min(num_modes, self.ema_mode_shapes.shape[1])):
                init_mac = self.compute_mac(init_modes_sensor[:, i], self.ema_mode_shapes[:, i])
                upd_mac = self.compute_mac(upd_modes_sensor[:, i], self.ema_mode_shapes[:, i]) if upd_modes_sensor is not None else 0
                print(f"{i+1:<6} {init_mac:<12.4f} {upd_mac:<12.4f}")
        
        print("=" * 80)


def example_model_updating():
    """示例: 完整的模型修正流程"""
    print("=" * 80)
    print("示例: 有限元模型修正完整流程")
    print("=" * 80)
    
    np.random.seed(42)
    
    n = 10
    M_true = np.eye(n) * 1.0
    
    K_true = np.zeros((n, n))
    k_base = 10000.0
    for i in range(n):
        K_true[i, i] = 2.0 * k_base
        if i > 0:
            K_true[i, i-1] = -1.0 * k_base
        if i < n - 1:
            K_true[i, i+1] = -1.0 * k_base
    
    K_initial = K_true.copy()
    K_initial[3:7, 3:7] *= 0.8
    
    print(f"\n真实模型: 刚度均匀 k={k_base}")
    print(f"初始模型: 中间区域刚度降低20%")
    
    frequencies = np.linspace(0, 100, 1025)
    
    solver_true = LanczosSolver(K_true, M_true)
    solver_true.solve(num_eigenvalues=5, tol=1e-9, verbose=False)
    true_freqs = solver_true.get_frequencies()
    true_modes = solver_true.eigenvectors
    
    damping_ratios = np.array([0.01, 0.012, 0.015, 0.018, 0.02])
    
    sensor_indices = [0, 2, 4, 6, 8]
    
    frf = generate_synthetic_frf(
        frequencies, true_freqs, damping_ratios,
        true_modes, sensor_indices=sensor_indices,
        noise_level=0.01
    )
    
    print(f"\n生成模拟EMA数据...")
    print(f"传感器位置: {sensor_indices}")
    print(f"真实固有频率: {true_freqs}")
    
    print("\n使用PolyMAX识别模态参数...")
    polymax = PolyMAX(frequencies, frf)
    stable_poles, _ = polymax.stabilization_diagram(max_order=15, freq_tol=0.02)
    modal_params = polymax.extract_modal_params()
    
    ema_freqs = modal_params['frequencies'][:5]
    ema_modes = modal_params['mode_shapes'][:, :5]
    ema_damping = modal_params['damping'][:5]
    
    print(f"\nEMA识别频率: {ema_freqs}")
    print(f"与真实值的误差: {abs(ema_freqs - true_freqs) / true_freqs * 100}%")
    
    print("\n" + "-" * 60)
    print("开始模型修正...")
    print("-" * 60)
    
    updater = ModelUpdating(K_initial, M_true, sensor_indices=sensor_indices)
    updater.set_ema_data(ema_freqs, damping=ema_damping, mode_shapes=ema_modes)
    
    initial_guess = np.array([1.0, 1.2, 1.0])
    
    result = updater.update_model(
        initial_guess,
        param_type='stiffness',
        bounds=[(0.5, 2.0), (0.5, 2.0), (0.5, 2.0)],
        freq_weight=1.0,
        mac_weight=0.5,
        num_modes=5
    )
    
    updater.compare_results(num_modes=5)
    
    print("\n修正参数说明:")
    print(f"  区域1 (节点0-3): 刚度修正系数 = {result.x[0]:.4f} (真实=1.0)")
    print(f"  区域2 (节点3-7): 刚度修正系数 = {result.x[1]:.4f} (真实=1.25, 因为初始降低了20%)")
    print(f"  区域3 (节点7-9): 刚度修正系数 = {result.x[2]:.4f} (真实=1.0)")
    
    return updater


def example_simple_updating():
    """简化示例：单参数修正"""
    print("\n" + "=" * 80)
    print("简化示例: 单参数刚度修正")
    print("=" * 80)
    
    np.random.seed(123)
    
    n = 5
    M = np.eye(n)
    K = np.zeros((n, n))
    k_true = 5000.0
    
    for i in range(n):
        K[i, i] = 2.0 * k_true
        if i > 0:
            K[i, i-1] = -k_true
        if i < n - 1:
            K[i, i+1] = -k_true
    
    solver = LanczosSolver(K, M)
    solver.solve(num_eigenvalues=3, tol=1e-9, verbose=False)
    true_freqs = solver.get_frequencies()
    true_modes = solver.eigenvectors
    
    K_wrong = K * 0.7
    
    print(f"\n真实刚度: {k_true}")
    print(f"错误刚度: {k_true * 0.7} (降低30%)")
    print(f"真实频率: {true_freqs}")
    
    updater = ModelUpdating(K_wrong, M)
    updater.set_ema_data(true_freqs)
    
    initial_guess = np.array([0.8])
    result = updater.update_model(
        initial_guess,
        param_type='stiffness',
        bounds=[(0.5, 1.5)],
        freq_weight=1.0,
        mac_weight=0.0,
        num_modes=3
    )
    
    print(f"\n识别的刚度修正系数: {result.x[0]:.4f}")
    print(f"期望的修正系数: {1.0/0.7:.4f} (补偿30%的降低)")
    print(f"误差: {abs(result.x[0] - 1.0/0.7) / (1.0/0.7) * 100:.2f}%")
    
    updater.compare_results(num_modes=3)
    
    return updater


if __name__ == "__main__":
    example_model_updating()
    example_simple_updating()
    
    print("\n" + "=" * 80)
    print("模型修正使用指南:")
    print("=" * 80)
    print("1. 准备初始FEM模型: K_initial, M_initial")
    print("2. 获取EMA数据: frequencies, mode_shapes = 从PolyMAX获取")
    print("3. 创建修正器: updater = ModelUpdating(K_initial, M_initial, sensor_indices)")
    print("4. 设置EMA数据: updater.set_ema_data(frequencies, mode_shapes=mode_shapes)")
    print("5. 执行修正: result = updater.update_model(initial_guess, param_type='stiffness')")
    print("6. 对比结果: updater.compare_results()")
    print("=" * 80)

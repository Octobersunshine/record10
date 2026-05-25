import numpy as np
from scipy.linalg import eigh


class ModalAnalysis:
    """有限元模态分析类"""
    
    def __init__(self, K, M):
        """
        初始化模态分析
        
        参数:
            K: 刚度矩阵 (n x n)
            M: 质量矩阵 (n x n)
        """
        self.K = np.asarray(K, dtype=np.float64)
        self.M = np.asarray(M, dtype=np.float64)
        self.n = self.K.shape[0]
        
        self._validate_matrices()
        
        self.frequencies = None
        self.natural_frequencies = None
        self.mode_shapes = None
        self.num_modes = None
    
    def _validate_matrices(self):
        """验证矩阵有效性"""
        if self.K.shape[0] != self.K.shape[1]:
            raise ValueError("刚度矩阵必须是方阵")
        if self.M.shape[0] != self.M.shape[1]:
            raise ValueError("质量矩阵必须是方阵")
        if self.K.shape != self.M.shape:
            raise ValueError("刚度矩阵和质量矩阵维度必须相同")
    
    def solve(self, num_modes=None, normalize='mass'):
        """
        求解广义特征值问题
        
        参数:
            num_modes: 要求解的模态数 (默认求解所有模态)
            normalize: 振型归一化方式 ('mass', 'max', None)
        """
        if num_modes is None:
            num_modes = self.n
        self.num_modes = min(num_modes, self.n)
        
        eigenvalues, eigenvectors = eigh(
            self.K, self.M, 
            subset_by_index=[0, self.num_modes - 1]
        )
        
        eigenvalues = np.maximum(eigenvalues, 0.0)
        self.natural_frequencies = np.sqrt(eigenvalues)
        self.frequencies = self.natural_frequencies / (2 * np.pi)
        
        self.mode_shapes = eigenvectors.copy()
        self._normalize_mode_shapes(normalize)
        
        sort_idx = np.argsort(self.frequencies)
        self.frequencies = self.frequencies[sort_idx]
        self.natural_frequencies = self.natural_frequencies[sort_idx]
        self.mode_shapes = self.mode_shapes[:, sort_idx]
        
        return self.frequencies, self.natural_frequencies, self.mode_shapes
    
    def _normalize_mode_shapes(self, normalize):
        """振型归一化"""
        if normalize == 'mass':
            for i in range(self.num_modes):
                phi = self.mode_shapes[:, i]
                mass_norm = phi.T @ self.M @ phi
                if mass_norm > 1e-12:
                    self.mode_shapes[:, i] = phi / np.sqrt(mass_norm)
        
        elif normalize == 'max':
            for i in range(self.num_modes):
                phi = self.mode_shapes[:, i]
                max_val = np.max(np.abs(phi))
                if max_val > 1e-12:
                    self.mode_shapes[:, i] = phi / max_val
    
    def print_results(self, precision=4):
        """打印模态分析结果"""
        if self.frequencies is None:
            print("请先调用 solve() 方法求解")
            return
        
        print("=" * 70)
        print("模态分析结果")
        print("=" * 70)
        print(f"{'阶数':<6} {'角频率(rad/s)':<18} {'频率(Hz)':<15} {'周期(s)':<15}")
        print("-" * 70)
        
        for i in range(self.num_modes):
            freq = self.frequencies[i]
            omega = self.natural_frequencies[i]
            period = 1.0 / freq if freq > 1e-10 else np.inf
            print(f"{i+1:<6} {omega:<18.{precision}f} {freq:<15.{precision}f} {period:<15.{precision}f}")
        
        print("-" * 70)
        print("\n振型矩阵 (每列为一个振型):")
        print(np.round(self.mode_shapes, precision))
        print()
    
    def verify_orthogonality(self, tol=1e-6):
        """验证振型正交性"""
        if self.mode_shapes is None:
            print("请先调用 solve() 方法求解")
            return False
        
        Phi = self.mode_shapes
        mass_ortho = Phi.T @ self.M @ Phi
        stiffness_ortho = Phi.T @ self.K @ Phi
        
        print("=" * 70)
        print("振型正交性验证")
        print("=" * 70)
        
        print("\n质量正交性 (Φ^T M Φ 应为对角阵):")
        print(np.round(mass_ortho, 8))
        
        print("\n刚度正交性 (Φ^T K Φ 应为对角阵):")
        print(np.round(stiffness_ortho, 8))
        
        error = np.max(np.abs(stiffness_ortho - np.diag(np.diag(stiffness_ortho))))
        print(f"\n非对角元最大误差: {error:.2e}")
        print(f"正交性验证: {'通过' if error < tol else '未通过'}")
        
        return error < tol
    
    def get_participation_factors(self):
        """计算模态参与因子"""
        if self.mode_shapes is None:
            raise ValueError("请先调用 solve() 方法求解")
        
        ones = np.ones(self.n)
        participation_factors = np.zeros(self.num_modes)
        
        for i in range(self.num_modes):
            phi = self.mode_shapes[:, i]
            participation_factors[i] = phi.T @ self.M @ ones
        
        return participation_factors
    
    def get_modal_mass(self):
        """计算模态质量"""
        if self.mode_shapes is None:
            raise ValueError("请先调用 solve() 方法求解")
        
        modal_mass = np.zeros(self.num_modes)
        for i in range(self.num_modes):
            phi = self.mode_shapes[:, i]
            modal_mass[i] = phi.T @ self.M @ phi
        
        return modal_mass
    
    def get_modal_stiffness(self):
        """计算模态刚度"""
        if self.mode_shapes is None:
            raise ValueError("请先调用 solve() 方法求解")
        
        modal_stiffness = np.zeros(self.num_modes)
        for i in range(self.num_modes):
            phi = self.mode_shapes[:, i]
            modal_stiffness[i] = phi.T @ self.K @ phi
        
        return modal_stiffness


def example_2dof_oscillator():
    """示例1: 2自由度振荡器"""
    print("\n" + "=" * 70)
    print("示例1: 2自由度弹簧-质量系统")
    print("=" * 70)
    
    K = np.array([
        [2000.0, -1000.0],
        [-1000.0, 2000.0]
    ])
    
    M = np.diag([1.0, 1.0])
    
    print("\n刚度矩阵 K:")
    print(K)
    print("\n质量矩阵 M:")
    print(M)
    print()
    
    ma = ModalAnalysis(K, M)
    ma.solve(normalize='mass')
    ma.print_results()
    ma.verify_orthogonality()
    
    print("\n模态参与因子:", ma.get_participation_factors())
    print("模态质量:", ma.get_modal_mass())
    print("模态刚度:", ma.get_modal_stiffness())


def example_shear_building():
    """示例2: 5层剪切型建筑"""
    print("\n" + "=" * 70)
    print("示例2: 5层剪切型建筑模型")
    print("=" * 70)
    
    n_floors = 5
    floor_mass = 100.0
    story_stiffness = 10000.0
    
    K = np.zeros((n_floors, n_floors))
    for i in range(n_floors):
        if i == 0:
            K[i, i] = story_stiffness
            K[i, i+1] = -story_stiffness
        elif i == n_floors - 1:
            K[i, i-1] = -story_stiffness
            K[i, i] = story_stiffness
        else:
            K[i, i-1] = -story_stiffness
            K[i, i] = 2 * story_stiffness
            K[i, i+1] = -story_stiffness
    
    M = np.diag([floor_mass] * n_floors)
    
    print(f"\n层数: {n_floors}")
    print(f"每层质量: {floor_mass} kg")
    print(f"层间刚度: {story_stiffness} N/m")
    print()
    
    ma = ModalAnalysis(K, M)
    ma.solve(num_modes=3, normalize='max')
    ma.print_results()
    ma.verify_orthogonality()


def example_lumped_mass_beam(n_nodes=6):
    """示例3: 简支梁集中质量模型"""
    print("\n" + "=" * 70)
    print(f"示例3: 简支梁集中质量模型 ({n_nodes}个节点)")
    print("=" * 70)
    
    L = 2.0
    EI = 1e5
    rhoA = 10.0
    
    dx = L / (n_nodes + 1)
    
    K = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        coeff = EI / dx**3
        if i == 0:
            K[i, i] = 2 * coeff
            K[i, i+1] = -coeff
        elif i == n_nodes - 1:
            K[i, i-1] = -coeff
            K[i, i] = 2 * coeff
        else:
            K[i, i-1] = -coeff
            K[i, i] = 2 * coeff
            K[i, i+1] = -coeff
    
    M = np.diag([rhoA * dx] * n_nodes)
    
    print(f"\n梁长度 L = {L} m")
    print(f"抗弯刚度 EI = {EI} N·m²")
    print(f"线密度 ρA = {rhoA} kg/m")
    print(f"节点数 = {n_nodes}")
    print(f"节点间距 dx = {dx:.4f} m")
    print()
    
    ma = ModalAnalysis(K, M)
    ma.solve(normalize='max')
    ma.print_results()
    
    print("\n理论一阶频率(近似): f1 ≈ π²/(2πL²) * sqrt(EI/(ρA))")
    f1_theory = (np.pi**2 / (2 * np.pi * L**2)) * np.sqrt(EI / rhoA)
    print(f"f1 ≈ {f1_theory:.4f} Hz")
    print(f"计算一阶频率: {ma.frequencies[0]:.4f} Hz")


if __name__ == "__main__":
    print("=" * 70)
    print("有限元模态分析工具")
    print("求解广义特征值问题: KΦ = ω²MΦ")
    print("=" * 70)
    
    example_2dof_oscillator()
    example_shear_building()
    example_lumped_mass_beam(n_nodes=6)
    
    print("\n" + "=" * 70)
    print("快速使用指南:")
    print("-" * 70)
    print("1. 导入类: from modal_analysis_viz import ModalAnalysis")
    print("2. 创建对象: ma = ModalAnalysis(K, M)")
    print("3. 求解: freq, omega, modes = ma.solve(num_modes=5, normalize='mass')")
    print("4. 查看结果: ma.print_results()")
    print("5. 验证正交性: ma.verify_orthogonality()")
    print("=" * 70)

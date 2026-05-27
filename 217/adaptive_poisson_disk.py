import numpy as np
from typing import Tuple, Optional, List
from scipy.spatial import KDTree
import time


class LocalFeatureAnalyzer:
    """
    局部特征分析器
    
    计算点云的局部特征：
    1. 曲率 (Curvature) - 衡量曲面弯曲程度
    2. 特征密度 (Feature density) - 衡量局部几何复杂度
    3. 法向量 (Normal vectors) - 用于曲率计算
    """
    
    def __init__(self, k_neighbors: int = 20):
        self.k_neighbors = k_neighbors
        
    def compute_local_normals(self, points: np.ndarray) -> np.ndarray:
        """
        计算每个点的局部法向量
        
        使用 PCA 分析 K 近邻点的协方差矩阵，
        最小特征值对应的特征向量即为法向量
        """
        n_points = len(points)
        normals = np.zeros((n_points, 3))
        
        tree = KDTree(points)
        
        for i in range(n_points):
            distances, indices = tree.query(points[i], k=min(self.k_neighbors + 1, n_points))
            
            neighbors = points[indices[1:]] if len(indices) > 1 else points[i:i+1]
            
            centered = neighbors - np.mean(neighbors, axis=0)
            cov = centered.T @ centered / len(neighbors)
            
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            normals[i] = eigenvectors[:, 0]
            
            if np.dot(normals[i], points[i]) < 0:
                normals[i] = -normals[i]
        
        return normals
    
    def compute_curvature(self, points: np.ndarray, 
                         normals: Optional[np.ndarray] = None) -> np.ndarray:
        """
        计算每个点的局部曲率
        
        使用法向量变化率估计曲率：
        曲率 ∝ |∇n|，其中 n 是法向量
        """
        if normals is None:
            normals = self.compute_local_normals(points)
        
        n_points = len(points)
        curvatures = np.zeros(n_points)
        
        tree = KDTree(points)
        
        for i in range(n_points):
            distances, indices = tree.query(points[i], k=min(self.k_neighbors + 1, n_points))
            
            neighbor_normals = normals[indices[1:]] if len(indices) > 1 else normals[i:i+1]
            
            normal_diffs = np.linalg.norm(neighbor_normals - normals[i], axis=1)
            avg_normal_diff = np.mean(normal_diffs)
            
            avg_distance = np.mean(distances[1:]) if len(distances) > 1 else 1.0
            
            curvatures[i] = avg_normal_diff / (avg_distance + 1e-10)
        
        curvatures = (curvatures - np.min(curvatures)) / (np.max(curvatures) - np.min(curvatures) + 1e-10)
        
        return curvatures
    
    def compute_feature_density(self, points: np.ndarray) -> np.ndarray:
        """
        计算局部特征密度
        
        基于距离分布的熵来衡量局部几何复杂度：
        - 平坦区域：距离分布集中 → 熵低
        - 复杂区域：距离分布分散 → 熵高
        """
        n_points = len(points)
        feature_density = np.zeros(n_points)
        
        tree = KDTree(points)
        k_for_density = min(30, n_points - 1)
        
        for i in range(n_points):
            distances, indices = tree.query(points[i], k=k_for_density + 1)
            distances = distances[1:]
            
            distances = distances / (np.max(distances) + 1e-10)
            hist, _ = np.histogram(distances, bins=10, range=(0, 1), density=True)
            hist = hist + 1e-10
            entropy = -np.sum(hist * np.log(hist))
            
            feature_density[i] = entropy
        
        feature_density = (feature_density - np.min(feature_density)) / (np.max(feature_density) - np.min(feature_density) + 1e-10)
        
        return feature_density
    
    def compute_local_scale(self, points: np.ndarray) -> np.ndarray:
        """
        计算每个点的局部尺度
        
        局部尺度定义为 K 近邻的平均距离
        """
        n_points = len(points)
        local_scales = np.zeros(n_points)
        
        tree = KDTree(points)
        
        for i in range(n_points):
            distances, indices = tree.query(points[i], k=min(self.k_neighbors + 1, n_points))
            local_scales[i] = np.mean(distances[1:]) if len(distances) > 1 else 1.0
        
        local_scales = local_scales / np.max(local_scales)
        
        return local_scales
    
    def analyze(self, points: np.ndarray) -> dict:
        """
        完整分析点云的局部特征
        
        返回:
            包含法向量、曲率、特征密度、局部尺度的字典
        """
        print("计算局部法向量...")
        normals = self.compute_local_normals(points)
        
        print("计算局部曲率...")
        curvatures = self.compute_curvature(points, normals)
        
        print("计算特征密度...")
        feature_density = self.compute_feature_density(points)
        
        print("计算局部尺度...")
        local_scales = self.compute_local_scale(points)
        
        return {
            'normals': normals,
            'curvatures': curvatures,
            'feature_density': feature_density,
            'local_scales': local_scales
        }


class AdaptiveRadiusEstimator:
    """
    自适应半径估计器
    
    根据局部特征计算每个点的采样半径：
    - 高曲率/高特征密度区域：小半径（密集采样）
    - 低曲率/低特征密度区域：大半径（稀疏采样）
    """
    
    def __init__(self, 
                 base_radius: float,
                 min_radius_factor: float = 0.3,
                 max_radius_factor: float = 3.0,
                 curvature_weight: float = 0.5,
                 density_weight: float = 0.3,
                 scale_weight: float = 0.2):
        """
        参数:
            base_radius: 基础采样半径
            min_radius_factor: 最小半径相对于 base_radius 的比例
            max_radius_factor: 最大半径相对于 base_radius 的比例
            curvature_weight: 曲率的权重
            density_weight: 特征密度的权重
            scale_weight: 局部尺度的权重
        """
        self.base_radius = base_radius
        self.min_radius_factor = min_radius_factor
        self.max_radius_factor = max_radius_factor
        self.curvature_weight = curvature_weight
        self.density_weight = density_weight
        self.scale_weight = scale_weight
        
    def compute_adaptive_radii(self, 
                               points: np.ndarray,
                               features: dict) -> np.ndarray:
        """
        计算每个点的自适应采样半径
        
        参数:
            points: 输入点云
            features: 局部特征字典（来自 LocalFeatureAnalyzer）
            
        返回:
            每个点的自适应采样半径数组
        """
        n_points = len(points)
        radii = np.ones(n_points) * self.base_radius
        
        curvatures = features['curvatures']
        feature_density = features['feature_density']
        local_scales = features['local_scales']
        
        combined_feature = (self.curvature_weight * curvatures + 
                          self.density_weight * feature_density + 
                          self.scale_weight * (1 - local_scales))
        
        combined_feature = (combined_feature - np.min(combined_feature)) / (np.max(combined_feature) - np.min(combined_feature) + 1e-10)
        
        radius_factors = self.max_radius_factor - (self.max_radius_factor - self.min_radius_factor) * combined_feature
        
        radii = self.base_radius * radius_factors
        
        return radii
    
    def compute_radius_for_point(self, 
                                 point_idx: int,
                                 features: dict) -> float:
        """
        计算单个点的自适应采样半径
        """
        curvature = features['curvatures'][point_idx]
        feature_density = features['feature_density'][point_idx]
        local_scale = features['local_scales'][point_idx]
        
        combined_feature = (self.curvature_weight * curvature + 
                          self.density_weight * feature_density + 
                          self.scale_weight * (1 - local_scale))
        
        radius_factor = self.max_radius_factor - (self.max_radius_factor - self.min_radius_factor) * combined_feature
        
        return self.base_radius * radius_factor


class AdaptiveGridIndex:
    """
    自适应网格索引
    
    支持变半径的邻域查询
    """
    
    def __init__(self, base_cell_size: float):
        self.base_cell_size = base_cell_size
        self.grid = {}
        self.point_radii = {}
        
    def _point_to_key(self, point: np.ndarray, radius: float) -> Tuple[int, int, int]:
        cell_size = min(self.base_cell_size, radius / 2.0)
        return tuple(np.floor(point / cell_size).astype(int))
    
    def insert(self, point: np.ndarray, radius: float, point_idx: int):
        key = self._point_to_key(point, radius)
        if key not in self.grid:
            self.grid[key] = []
        self.grid[key].append(point_idx)
        self.point_radii[point_idx] = radius
    
    def has_conflict(self, point: np.ndarray, radius: float, 
                     all_points: np.ndarray) -> bool:
        """
        检查是否存在与其他采样点的冲突
        
        采用双边检查策略：
        - 候选点半径范围内是否已有采样点
        - 已有采样点的半径范围内是否包含候选点
        """
        cell_size = min(self.base_cell_size, radius / 2.0)
        key = tuple(np.floor(point / cell_size).astype(int))
        
        search_range = int(np.ceil(2 * radius / cell_size))
        
        for dx in range(-search_range, search_range + 1):
            for dy in range(-search_range, search_range + 1):
                for dz in range(-search_range, search_range + 1):
                    neighbor_key = (key[0] + dx, key[1] + dy, key[2] + dz)
                    if neighbor_key in self.grid:
                        for neighbor_idx in self.grid[neighbor_key]:
                            neighbor_point = all_points[neighbor_idx]
                            neighbor_radius = self.point_radii[neighbor_idx]
                            dist = np.linalg.norm(point - neighbor_point)
                            min_dist = max(radius, neighbor_radius)
                            if dist < min_dist:
                                return True
        return False


class AdaptivePoissonDiskSampler:
    """
    自适应泊松圆盘采样器
    
    核心特性：
    1. 根据局部曲率调整采样密度
    2. 根据特征密度调整采样密度
    3. 根据局部尺度调整采样密度
    4. 细节区域密集采样，平坦区域稀疏采样
    
    优势：
    - 提高表示效率：用更少的点表示相同的细节
    - 自适应：自动识别并保留重要特征
    - 高效：结合优先级采样和间隙填充
    """
    
    def __init__(self, 
                 base_radius: float,
                 min_radius_factor: float = 0.3,
                 max_radius_factor: float = 3.0,
                 curvature_weight: float = 0.5,
                 density_weight: float = 0.3,
                 scale_weight: float = 0.2,
                 k_neighbors: int = 20):
        """
        参数:
            base_radius: 基础采样半径
            min_radius_factor: 最小半径比例
            max_radius_factor: 最大半径比例
            curvature_weight: 曲率权重
            density_weight: 特征密度权重
            scale_weight: 局部尺度权重
            k_neighbors: K近邻数量
        """
        self.base_radius = base_radius
        self.min_radius_factor = min_radius_factor
        self.max_radius_factor = max_radius_factor
        self.curvature_weight = curvature_weight
        self.density_weight = density_weight
        self.scale_weight = scale_weight
        self.k_neighbors = k_neighbors
        
        self.feature_analyzer = LocalFeatureAnalyzer(k_neighbors=k_neighbors)
        self.radius_estimator = AdaptiveRadiusEstimator(
            base_radius=base_radius,
            min_radius_factor=min_radius_factor,
            max_radius_factor=max_radius_factor,
            curvature_weight=curvature_weight,
            density_weight=density_weight,
            scale_weight=scale_weight
        )
        
    def sample(self, points: np.ndarray, 
               target_count: Optional[int] = None,
               precomputed_features: Optional[dict] = None) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        执行自适应泊松圆盘采样
        
        参数:
            points: 输入点云 (N, 3)
            target_count: 目标采样点数（可选）
            precomputed_features: 预计算的局部特征（可选）
            
        返回:
            (采样点数组, 采样点索引数组, 局部特征字典)
        """
        if len(points) == 0:
            return np.array([]), np.array([]), {}
        
        if target_count is not None and target_count >= len(points):
            features = precomputed_features if precomputed_features else self.feature_analyzer.analyze(points)
            return points.copy(), np.arange(len(points)), features
        
        start_time = time.time()
        
        print("分析局部特征...")
        if precomputed_features is None:
            features = self.feature_analyzer.analyze(points)
        else:
            features = precomputed_features
        
        print("计算自适应采样半径...")
        adaptive_radii = self.radius_estimator.compute_adaptive_radii(points, features)
        
        print("计算采样优先级...")
        priorities = features['curvatures'] * self.curvature_weight + \
                     features['feature_density'] * self.density_weight
        
        sorted_indices = np.argsort(-priorities)
        
        print("执行优先级采样...")
        grid_index = AdaptiveGridIndex(self.base_radius * self.min_radius_factor)
        sample_indices = []
        
        for idx in sorted_indices:
            if target_count is not None and len(sample_indices) >= target_count:
                break
            
            point = points[idx]
            radius = adaptive_radii[idx]
            
            if not grid_index.has_conflict(point, radius, points):
                sample_indices.append(idx)
                grid_index.insert(point, radius, idx)
        
        print(f"初始采样完成: {len(sample_indices)} 个点")
        
        if target_count is not None and len(sample_indices) < target_count:
            print("填充间隙...")
            remaining_indices = [i for i in range(len(points)) if i not in set(sample_indices)]
            np.random.shuffle(remaining_indices)
            
            for idx in remaining_indices:
                if len(sample_indices) >= target_count:
                    break
                
                point = points[idx]
                radius = adaptive_radii[idx]
                
                if not grid_index.has_conflict(point, radius, points):
                    sample_indices.append(idx)
                    grid_index.insert(point, radius, idx)
        
        sample_indices = np.array(sample_indices)
        samples = points[sample_indices]
        sample_radii = adaptive_radii[sample_indices]
        
        elapsed = time.time() - start_time
        print(f"自适应采样完成: {len(samples)} 个点, 耗时: {elapsed:.3f}秒")
        
        return samples, sample_indices, features


class CurvatureAwareSimplifier:
    """
    曲率感知的点云简化器
    
    专门用于点云/网格模型简化的自适应采样器：
    - 高曲率区域（边缘、角落）：保留更多细节
    - 低曲率区域（平面）：大幅简化
    
    应用场景：
    - 三维模型简化
    - 点云压缩
    - 实时渲染
    """
    
    def __init__(self, 
                 target_count: Optional[int] = None,
                 min_radius: Optional[float] = None,
                 sensitivity: float = 1.0):
        """
        参数:
            target_count: 目标点数（与 min_radius 二选一）
            min_radius: 最小采样半径（与 target_count 二选一）
            sensitivity: 曲率敏感度，值越高曲率区域越密集
        """
        self.target_count = target_count
        self.min_radius = min_radius
        self.sensitivity = sensitivity
        
    def simplify(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        简化点云
        
        参数:
            points: 输入点云 (N, 3)
            
        返回:
            (简化后的点云, 采样点的原始索引)
        """
        if len(points) == 0:
            return np.array([]), np.array([])
        
        if self.target_count is not None and self.target_count >= len(points):
            return points.copy(), np.arange(len(points))
        
        if self.min_radius is None and self.target_count is None:
            raise ValueError("必须指定 target_count 或 min_radius")
        
        bbox_diag = np.linalg.norm(np.max(points, axis=0) - np.min(points, axis=0))
        
        if self.min_radius is None:
            density = self.target_count / (bbox_diag ** 3)
            base_radius = 1 / (2 * np.cbrt(density * np.pi * 4 / 3))
        else:
            base_radius = self.min_radius
        
        sampler = AdaptivePoissonDiskSampler(
            base_radius=base_radius,
            min_radius_factor=0.2,
            max_radius_factor=5.0,
            curvature_weight=0.7 * self.sensitivity,
            density_weight=0.2,
            scale_weight=0.1
        )
        
        samples, indices, features = sampler.sample(
            points, target_count=self.target_count
        )
        
        return samples, indices


def compare_uniform_vs_adaptive():
    """
    对比均匀采样和自适应采样的效果
    """
    print("=" * 70)
    print("均匀采样 vs 自适应采样 对比")
    print("=" * 70)
    
    np.random.seed(42)
    
    print("\n生成带特征的测试点云...")
    n_points = 20000
    
    theta = np.random.uniform(0, 2*np.pi, n_points)
    phi = np.arccos(np.random.uniform(-1, 1, n_points))
    r = 10 + 3 * np.sin(5 * theta) * np.cos(3 * phi)
    
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    points = np.column_stack([x, y, z])
    
    print(f"原始点数: {n_points}")
    
    target_count = 2000
    print(f"目标采样数: {target_count} (采样率: {target_count/n_points*100:.1f}%)")
    
    print("\n--- 均匀采样 (MaximalPoissonDiskSampler) ---")
    from advanced_poisson_disk import MaximalPoissonDiskSampler
    
    bbox_diag = np.linalg.norm(np.max(points, axis=0) - np.min(points, axis=0))
    density = target_count / (bbox_diag ** 3)
    uniform_radius = 1 / (2 * np.cbrt(density * np.pi * 4 / 3))
    
    start = time.time()
    uniform_sampler = MaximalPoissonDiskSampler(min_radius=uniform_radius)
    uniform_samples = uniform_sampler.sample(points, target_count=target_count)
    uniform_time = time.time() - start
    
    print(f"  采样点数: {len(uniform_samples)}")
    print(f"  耗时: {uniform_time:.3f}秒")
    
    print("\n--- 自适应采样 (AdaptivePoissonDiskSampler) ---")
    start = time.time()
    adaptive_sampler = AdaptivePoissonDiskSampler(
        base_radius=uniform_radius,
        min_radius_factor=0.3,
        max_radius_factor=3.0,
        curvature_weight=0.6,
        density_weight=0.3,
        scale_weight=0.1
    )
    adaptive_samples, adaptive_indices, features = adaptive_sampler.sample(
        points, target_count=target_count
    )
    adaptive_time = time.time() - start
    
    print(f"  采样点数: {len(adaptive_samples)}")
    print(f"  耗时: {adaptive_time:.3f}秒")
    
    print("\n--- 特征保留分析 ---")
    analyzer = LocalFeatureAnalyzer()
    
    print("  计算曲率分布...")
    uniform_curvatures = analyzer.compute_curvature(uniform_samples)
    adaptive_curvatures = features['curvatures'][adaptive_indices]
    
    high_curvature_threshold = np.percentile(features['curvatures'], 75)
    
    orig_high_curv = np.sum(features['curvatures'] >= high_curvature_threshold)
    uniform_high_curv = np.sum(uniform_curvatures >= high_curvature_threshold)
    adaptive_high_curv = np.sum(adaptive_curvatures >= high_curvature_threshold)
    
    print(f"  原始高曲率点数: {orig_high_curv} ({orig_high_curv/n_points*100:.1f}%)")
    print(f"  均匀采样保留高曲率点: {uniform_high_curv} ({uniform_high_curv/len(uniform_samples)*100:.1f}%)")
    print(f"  自适应采样保留高曲率点: {adaptive_high_curv} ({adaptive_high_curv/len(adaptive_samples)*100:.1f}%)")
    
    if adaptive_high_curv > uniform_high_curv:
        improvement = (adaptive_high_curv - uniform_high_curv) / uniform_high_curv * 100
        print(f"\n✓ 自适应采样在高曲率特征保留上提升了 {improvement:.1f}%")
    
    print("\n" + "=" * 70)
    print("总结:")
    print("=" * 70)
    print("  均匀采样: 整个空间使用相同的采样密度")
    print("  自适应采样: 高曲率区域密集采样，平坦区域稀疏采样")
    print("  → 自适应采样能用相同数量的点更好地保留细节特征")
    print()


if __name__ == "__main__":
    compare_uniform_vs_adaptive()

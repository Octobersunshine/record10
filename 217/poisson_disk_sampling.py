import numpy as np
from typing import Tuple, Optional


class PoissonDiskSampler3D:
    """
    三维泊松圆盘采样器
    
    用于从点云中生成分布均匀且满足最小距离约束的采样点。
    适用于点云简化、模型降采样等场景。
    
    参数:
        min_radius: 采样点之间的最小距离
        max_samples: 最大采样点数（可选）
        num_candidates: 每个采样点生成的候选点数量
    """
    
    def __init__(self, 
                 min_radius: float, 
                 max_samples: Optional[int] = None,
                 num_candidates: int = 30):
        self.min_radius = min_radius
        self.max_samples = max_samples
        self.num_candidates = num_candidates
        self.cell_size = min_radius / 2.0
        
    def _build_grid(self, points: np.ndarray) -> Tuple[dict, np.ndarray, np.ndarray]:
        """
        构建空间网格用于快速邻域查询
        """
        grid = {}
        
        if points.shape[0] == 0:
            return grid, np.zeros(3), np.zeros(3)
        
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        
        grid_dims = np.ceil((max_coords - min_coords) / self.cell_size).astype(int) + 1
        
        for idx, point in enumerate(points):
            grid_coord = self._point_to_grid(point, min_coords)
            grid_key = tuple(grid_coord)
            if grid_key not in grid:
                grid[grid_key] = []
            grid[grid_key].append(idx)
        
        return grid, min_coords, grid_dims
    
    def _point_to_grid(self, point: np.ndarray, min_coords: np.ndarray) -> np.ndarray:
        """
        将点坐标转换为网格坐标
        """
        return np.floor((point - min_coords) / self.cell_size).astype(int)
    
    def _check_neighbors(self, 
                         point: np.ndarray, 
                         sample_points: np.ndarray,
                         grid: dict, 
                         min_coords: np.ndarray) -> bool:
        """
        检查点周围网格中是否存在距离过近的点
        使用 cell_size = min_radius / 2，检查 5x5x5 邻域确保不会漏掉任何可能的邻居
        """
        grid_coord = self._point_to_grid(point, min_coords)
        
        for dx in [-2, -1, 0, 1, 2]:
            for dy in [-2, -1, 0, 1, 2]:
                for dz in [-2, -1, 0, 1, 2]:
                    neighbor_key = (grid_coord[0] + dx, 
                                   grid_coord[1] + dy, 
                                   grid_coord[2] + dz)
                    if neighbor_key in grid:
                        for neighbor_idx in grid[neighbor_key]:
                            neighbor_point = sample_points[neighbor_idx]
                            dist = np.linalg.norm(point - neighbor_point)
                            if dist < self.min_radius:
                                return False
        return True
    
    def sample(self, points: np.ndarray) -> np.ndarray:
        """
        对输入点云进行泊松圆盘采样
        
        参数:
            points: 输入点云，形状为 (N, 3)
            
        返回:
            采样后的点云，形状为 (M, 3)，其中 M <= N
        """
        if len(points) == 0:
            return np.array([])
            
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("输入点云必须是形状为 (N, 3) 的二维数组")
        
        n_points = len(points)
        if self.max_samples is not None and self.max_samples >= n_points:
            return points.copy()
        
        shuffled_indices = np.random.permutation(n_points)
        shuffled_points = points[shuffled_indices]
        
        sample_points_list = []
        sample_points_array = np.empty((0, 3))
        
        grid = {}
        min_coords = np.min(points, axis=0)
        
        for idx, point in enumerate(shuffled_points):
            if self.max_samples is not None and len(sample_points_list) >= self.max_samples:
                break
            
            if self._check_neighbors(point, sample_points_array, grid, min_coords):
                sample_points_list.append(point)
                new_idx = len(sample_points_list) - 1
                sample_points_array = np.vstack([sample_points_array, point]) if sample_points_array.size else point.reshape(1, 3)
                
                grid_coord = self._point_to_grid(point, min_coords)
                grid_key = tuple(grid_coord)
                if grid_key not in grid:
                    grid[grid_key] = []
                grid[grid_key].append(new_idx)
        
        return np.array(sample_points_list)


class BridsonSampler3D:
    """
    基于 Bridson 算法的三维泊松圆盘采样器
    
    直接在空间中生成泊松圆盘分布的采样点，不依赖输入点云。
    适用于需要在指定体积内生成均匀分布点的场景。
    
    参数:
        min_radius: 采样点之间的最小距离
        bounds: 采样空间边界，形状为 (3, 2)，分别为 x, y, z 的 [min, max]
        num_candidates: 每个采样点生成的候选点数量
    """
    
    def __init__(self, 
                 min_radius: float, 
                 bounds: np.ndarray,
                 num_candidates: int = 30):
        self.min_radius = min_radius
        self.bounds = np.asarray(bounds)
        self.num_candidates = num_candidates
        self.cell_size = min_radius / 2.0
        
        self.grid_dims = np.ceil(
            (self.bounds[:, 1] - self.bounds[:, 0]) / self.cell_size
        ).astype(int) + 1
        
        self.grid = np.full(self.grid_dims.tolist(), -1, dtype=int)
        self.samples = []
        
    def _point_to_grid(self, point: np.ndarray) -> np.ndarray:
        """将点坐标转换为网格坐标"""
        return np.floor((point - self.bounds[:, 0]) / self.cell_size).astype(int)
    
    def _is_valid(self, point: np.ndarray) -> bool:
        """检查点是否在边界内且距离其他点足够远"""
        if not np.all(point >= self.bounds[:, 0]) or not np.all(point <= self.bounds[:, 1]):
            return False
        
        grid_coord = self._point_to_grid(point)
        
        min_search = np.maximum(grid_coord - 2, 0)
        max_search = np.minimum(grid_coord + 3, self.grid_dims)
        
        for x in range(min_search[0], max_search[0]):
            for y in range(min_search[1], max_search[1]):
                for z in range(min_search[2], max_search[2]):
                    neighbor_idx = self.grid[x, y, z]
                    if neighbor_idx != -1:
                        dist = np.linalg.norm(point - self.samples[neighbor_idx])
                        if dist < self.min_radius:
                            return False
        return True
    
    def sample(self, max_samples: Optional[int] = None) -> np.ndarray:
        """
        执行泊松圆盘采样
        
        参数:
            max_samples: 最大采样点数（可选）
            
        返回:
            采样点数组，形状为 (N, 3)
        """
        center = np.mean(self.bounds, axis=1)
        self.samples = [center]
        grid_coord = self._point_to_grid(center)
        self.grid[tuple(grid_coord)] = 0
        
        active_list = [0]
        
        while active_list:
            current_idx = np.random.choice(len(active_list))
            sample_idx = active_list[current_idx]
            current_point = self.samples[sample_idx]
            
            found = False
            for _ in range(self.num_candidates):
                angle1 = np.random.uniform(0, 2 * np.pi)
                angle2 = np.arccos(np.random.uniform(-1, 1))
                radius = np.random.uniform(self.min_radius, 2 * self.min_radius)
                
                candidate = current_point + np.array([
                    radius * np.sin(angle2) * np.cos(angle1),
                    radius * np.sin(angle2) * np.sin(angle1),
                    radius * np.cos(angle2)
                ])
                
                if self._is_valid(candidate):
                    new_idx = len(self.samples)
                    self.samples.append(candidate)
                    grid_coord = self._point_to_grid(candidate)
                    self.grid[tuple(grid_coord)] = new_idx
                    active_list.append(new_idx)
                    found = True
                    
                    if max_samples is not None and len(self.samples) >= max_samples:
                        return np.array(self.samples)
                    break
            
            if not found:
                active_list.pop(current_idx)
        
        return np.array(self.samples)


def point_cloud_simplification(points: np.ndarray, 
                               target_count: Optional[int] = None,
                               min_radius: Optional[float] = None) -> np.ndarray:
    """
    点云简化函数 - 根据目标点数或最小距离进行泊松圆盘采样
    
    参数:
        points: 输入点云，形状为 (N, 3)
        target_count: 目标采样点数
        min_radius: 最小距离（与 target_count 二选一）
        
    返回:
        简化后的点云
    """
    if min_radius is None and target_count is None:
        raise ValueError("必须指定 target_count 或 min_radius")
    
    if min_radius is None:
        bbox_diag = np.linalg.norm(np.max(points, axis=0) - np.min(points, axis=0))
        volume = bbox_diag ** 3
        density = target_count / volume
        min_radius = 1 / (2 * np.cbrt(density * np.pi * 4 / 3))
    
    sampler = PoissonDiskSampler3D(min_radius=min_radius, max_samples=target_count)
    return sampler.sample(points)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    print("=" * 60)
    print("泊松圆盘采样演示")
    print("=" * 60)
    
    print("\n1. 点云简化示例")
    print("-" * 60)
    np.random.seed(42)
    n_original = 10000
    original_points = np.random.rand(n_original, 3) * 10
    
    print(f"原始点数: {n_original}")
    
    simplified = point_cloud_simplification(original_points, target_count=500)
    print(f"简化后点数: {len(simplified)}")
    print(f"压缩率: {(1 - len(simplified)/n_original)*100:.1f}%")
    
    distances = []
    for i in range(len(simplified)):
        for j in range(i+1, len(simplified)):
            dist = np.linalg.norm(simplified[i] - simplified[j])
            distances.append(dist)
    
    print(f"最小采样点间距: {np.min(distances):.4f}")
    print(f"平均采样点间距: {np.mean(distances):.4f}")
    
    print("\n2. Bridson 算法空间采样示例")
    print("-" * 60)
    bounds = np.array([[0, 10], [0, 10], [0, 10]])
    bridson_sampler = BridsonSampler3D(min_radius=1.0, bounds=bounds)
    bridson_samples = bridson_sampler.sample()
    print(f"空间采样点数: {len(bridson_samples)}")
    
    print("\n采样完成！")

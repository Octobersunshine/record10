import open3d as o3d
import numpy as np
from typing import Tuple, Optional


def compute_point_density(
    pcd: o3d.geometry.PointCloud,
    radius: float = 0.1
) -> np.ndarray:
    """
    计算点云中点的密度
    
    Args:
        pcd: 输入点云
        radius: 密度计算半径
    
    Returns:
        每个点的密度值数组
    """
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    points = np.asarray(pcd.points)
    densities = np.zeros(len(points))
    
    for i in range(len(points)):
        k, _, _ = pcd_tree.search_radius_vector_3d(points[i], radius)
        densities[i] = k / (4 / 3 * np.pi * radius ** 3)
    
    return densities


def uniform_bidirectional_correspondence_check(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    双向对应点验证：确保点对是相互的最近邻
    
    Args:
        source: 源点云
        target: 目标点云
        threshold: 最大距离阈值
    
    Returns:
        source_indices: 源点云中匹配点的索引
        target_indices: 目标点云中对应点的索引
    """
    source_tree = o3d.geometry.KDTreeFlann(source)
    target_tree = o3d.geometry.KDTreeFlann(target)
    
    source_points = np.asarray(source.points)
    target_points = np.asarray(target.points)
    
    source_to_target = []
    target_to_source = []
    
    for i in range(len(source_points)):
        k, idx, dist = target_tree.search_knn_vector_3d(source_points[i], 1)
        if dist[0] < threshold ** 2:
            source_to_target.append((i, idx[0]))
    
    for i in range(len(target_points)):
        k, idx, dist = source_tree.search_knn_vector_3d(target_points[i], 1)
        if dist[0] < threshold ** 2:
            target_to_source.append((idx[0], i))
    
    bidirectional_matches = set(source_to_target) & set(target_to_source)
    
    if len(bidirectional_matches) == 0:
        return np.array([]), np.array([])
    
    source_indices = np.array([m[0] for m in bidirectional_matches])
    target_indices = np.array([m[1] for m in bidirectional_matches])
    
    return source_indices, target_indices


def density_guided_sampling(
    pcd: o3d.geometry.PointCloud,
    num_samples: int = 5000,
    density_radius: float = 0.1
) -> o3d.geometry.PointCloud:
    """
    密度引导采样：降低高密度区域的采样概率
    
    Args:
        pcd: 输入点云
        num_samples: 采样点数
        density_radius: 密度计算半径
    
    Returns:
        采样后的点云
    """
    if len(pcd.points) <= num_samples:
        return pcd
    
    densities = compute_point_density(pcd, density_radius)
    
    weights = 1.0 / (densities + 1e-8)
    weights = weights / weights.sum()
    
    indices = np.random.choice(
        len(pcd.points),
        size=min(num_samples, len(pcd.points)),
        replace=False,
        p=weights
    )
    
    return pcd.select_by_index(indices)


def voxel_downsample_uniform(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float = 0.05
) -> o3d.geometry.PointCloud:
    """
    体素下采样：统一点云密度
    
    Args:
        pcd: 输入点云
        voxel_size: 体素大小
    
    Returns:
        下采样后的点云
    """
    return pcd.voxel_down_sample(voxel_size=voxel_size)


def icp_registration(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    threshold: float = 0.02,
    max_iteration: int = 2000,
    init_transformation: Optional[np.ndarray] = None,
    voxel_size: Optional[float] = 0.05,
    use_density_sampling: bool = True,
    use_bidirectional_check: bool = True,
    num_samples: int = 8000
) -> Tuple[np.ndarray, o3d.pipelines.registration.RegistrationResult]:
    """
    改进的迭代最近点（ICP）配准算法
    解决点云密度不均导致的配准偏差问题
    
    改进策略：
    1. 体素下采样 - 统一点云密度
    2. 密度引导采样 - 降低高密度区域权重
    3. 双向对应点验证 - 确保匹配质量
    
    Args:
        source: 源点云
        target: 目标点云
        threshold: 对应点对的最大距离阈值
        max_iteration: 最大迭代次数
        init_transformation: 初始变换矩阵 (4x4)，默认为单位矩阵
        voxel_size: 体素下采样大小，None表示不进行下采样
        use_density_sampling: 是否使用密度引导采样
        use_bidirectional_check: 是否使用双向对应点验证
        num_samples: 密度引导采样的点数
    
    Returns:
        transformation_matrix: 4x4变换矩阵
        result: Open3D配准结果对象
    """
    if init_transformation is None:
        init_transformation = np.eye(4)
    
    source_processed = o3d.geometry.PointCloud(source)
    target_processed = o3d.geometry.PointCloud(target)
    
    if voxel_size is not None:
        source_processed = voxel_downsample_uniform(source_processed, voxel_size)
        target_processed = voxel_downsample_uniform(target_processed, voxel_size)
    
    if use_density_sampling:
        source_processed = density_guided_sampling(source_processed, num_samples)
        target_processed = density_guided_sampling(target_processed, num_samples)
    
    source_processed.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    target_processed.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    
    loss = o3d.pipelines.registration.TukeyLoss(k=0.1)
    
    result = o3d.pipelines.registration.registration_icp(
        source_processed,
        target_processed,
        threshold,
        init_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(loss),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6,
            relative_rmse=1e-6,
            max_iteration=max_iteration
        )
    )
    
    return result.transformation, result


def load_point_cloud(file_path: str) -> o3d.geometry.PointCloud:
    """
    加载点云文件
    
    Args:
        file_path: 点云文件路径
    
    Returns:
        加载的点云对象
    """
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd


def visualize_registration(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    transformation: np.ndarray = None
) -> None:
    """
    可视化配准结果
    
    Args:
        source: 源点云
        target: 目标点云
        transformation: 变换矩阵，如提供则将源点云应用此变换
    """
    source_temp = o3d.geometry.PointCloud(source)
    if transformation is not None:
        source_temp.transform(transformation)
    
    source_temp.paint_uniform_color([1, 0.706, 0])
    target.paint_uniform_color([0, 0.651, 0.929])
    
    o3d.visualization.draw_geometries([source_temp, target])


if __name__ == "__main__":
    print("ICP配准模块已加载")
    print("使用方法:")
    print("1. 加载点云: source = load_point_cloud('source.pcd')")
    print("2. 加载点云: target = load_point_cloud('target.pcd')")
    print("3. 执行配准: transform, result = icp_registration(source, target)")
    print("4. 可视化: visualize_registration(source, target, transform)")

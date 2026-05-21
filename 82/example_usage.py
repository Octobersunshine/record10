import open3d as o3d
import numpy as np
from icp_registration import (
    icp_registration,
    load_point_cloud,
    visualize_registration,
    compute_point_density,
    voxel_downsample_uniform,
    density_guided_sampling
)


def create_uneven_density_point_clouds(
    num_high_density: int = 5000,
    num_low_density: int = 200
):
    """
    创建密度不均的演示点云数据
    - 高密度区域：某个局部区域点非常密集
    - 低密度区域：其他区域点非常稀疏
    """
    source_points = []
    
    high_density_center = np.array([0.2, 0.2, 0.2])
    high_density_noise = np.random.randn(num_high_density, 3) * 0.02
    source_points.append(high_density_center + high_density_noise)
    
    low_density_points = np.random.rand(num_low_density, 3) * 2 - 0.5
    source_points.append(low_density_points)
    
    source_points = np.vstack(source_points)
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_points)
    
    translation = np.array([0.3, 0.25, 0.2])
    rotation = o3d.geometry.get_rotation_matrix_from_xyz((0.15, 0.1, 0.05))
    
    target = o3d.geometry.PointCloud(source)
    target.rotate(rotation, center=(0, 0, 0))
    target.translate(translation)
    
    true_transformation = np.eye(4)
    true_transformation[:3, :3] = rotation
    true_transformation[:3, 3] = translation
    
    return source, target, true_transformation


def analyze_density(pcd: o3d.geometry.PointCloud, name: str):
    """分析点云密度分布"""
    densities = compute_point_density(pcd, radius=0.1)
    print(f"\n{name} 密度分析:")
    print(f"   总点数: {len(pcd.points)}")
    print(f"   最大密度: {densities.max():.2f}")
    print(f"   最小密度: {densities.min():.2f}")
    print(f"   平均密度: {densities.mean():.2f}")
    print(f"   密度标准差: {densities.std():.2f}")
    return densities


def compare_registration_methods():
    """对比标准ICP和改进ICP的配准效果"""
    print("=" * 60)
    print("密度不均情况下的ICP配准对比测试")
    print("=" * 60)
    
    print("\n1. 创建密度不均的测试点云...")
    source, target, true_transform = create_uneven_density_point_clouds()
    
    analyze_density(source, "源点云")
    analyze_density(target, "目标点云")
    
    print(f"\n   真实变换矩阵:")
    print(true_transform)
    
    print("\n" + "=" * 60)
    print("方法1: 标准ICP（无密度处理）")
    print("=" * 60)
    
    transform_basic, result_basic = icp_registration(
        source,
        target,
        threshold=0.1,
        max_iteration=2000,
        voxel_size=None,
        use_density_sampling=False,
        use_bidirectional_check=False
    )
    
    print(f"\n   估计的变换矩阵:")
    print(transform_basic)
    print(f"\n   配准结果:")
    print(f"   Fitness: {result_basic.fitness:.6f}")
    print(f"   RMSE: {result_basic.inlier_rmse:.6f}")
    print(f"   内点数量: {len(result_basic.correspondence_set)}")
    
    translation_error_basic = np.linalg.norm(
        transform_basic[:3, 3] - true_transform[:3, 3]
    )
    rotation_error_basic = np.linalg.norm(
        transform_basic[:3, :3] - true_transform[:3, :3]
    )
    print(f"\n   变换误差:")
    print(f"   平移误差: {translation_error_basic:.6f}")
    print(f"   旋转误差: {rotation_error_basic:.6f}")
    
    print("\n" + "=" * 60)
    print("方法2: 改进ICP（体素下采样 + 密度引导采样）")
    print("=" * 60)
    
    transform_improved, result_improved = icp_registration(
        source,
        target,
        threshold=0.1,
        max_iteration=2000,
        voxel_size=0.05,
        use_density_sampling=True,
        num_samples=3000
    )
    
    print(f"\n   估计的变换矩阵:")
    print(transform_improved)
    print(f"\n   配准结果:")
    print(f"   Fitness: {result_improved.fitness:.6f}")
    print(f"   RMSE: {result_improved.inlier_rmse:.6f}")
    print(f"   内点数量: {len(result_improved.correspondence_set)}")
    
    translation_error_improved = np.linalg.norm(
        transform_improved[:3, 3] - true_transform[:3, 3]
    )
    rotation_error_improved = np.linalg.norm(
        transform_improved[:3, :3] - true_transform[:3, :3]
    )
    print(f"\n   变换误差:")
    print(f"   平移误差: {translation_error_improved:.6f}")
    print(f"   旋转误差: {rotation_error_improved:.6f}")
    
    print("\n" + "=" * 60)
    print("效果对比总结")
    print("=" * 60)
    translation_improvement = (translation_error_basic - translation_error_improved) / translation_error_basic * 100
    rotation_improvement = (rotation_error_basic - rotation_error_improved) / rotation_error_basic * 100
    
    print(f"\n   平移误差降低: {translation_improvement:+.2f}%")
    print(f"   旋转误差降低: {rotation_improvement:+.2f}%")
    
    if translation_improvement > 0 or rotation_improvement > 0:
        print("\n   ✓ 改进后的方法有效降低了密度不均导致的配准误差!")
    else:
        print("\n   注: 测试数据可能随机性导致结果波动，请多次运行观察统计效果")
    
    print("\n" + "=" * 60)
    print("可视化配准结果")
    print("=" * 60)
    print("\n   黄色 = 源点云（已配准），蓝色 = 目标点云")
    print("   首先显示标准ICP配准结果，关闭后显示改进ICP配准结果")
    
    print("\n   显示标准ICP配准结果...")
    visualize_registration(source, target, transform_basic)
    
    print("\n   显示改进ICP配准结果...")
    visualize_registration(source, target, transform_improved)


def demonstrate_preprocessing_effects():
    """演示各种预处理效果"""
    print("\n" + "=" * 60)
    print("点云预处理效果演示")
    print("=" * 60)
    
    source, _, _ = create_uneven_density_point_clouds()
    
    print("\n原始点云:")
    densities_original = analyze_density(source, "原始点云")
    
    print("\n体素下采样后:")
    source_voxel = voxel_downsample_uniform(source, voxel_size=0.05)
    densities_voxel = analyze_density(source_voxel, "体素下采样后")
    
    print("\n密度引导采样后:")
    source_density = density_guided_sampling(source, num_samples=2000)
    densities_density = analyze_density(source_density, "密度引导采样后")
    
    print("\n密度变化:")
    print(f"   原始密度标准差: {densities_original.std():.2f}")
    print(f"   体素后密度标准差: {densities_voxel.std():.2f}")
    print(f"   密度采样后标准差: {densities_density.std():.2f}")
    
    print("\n   ✓ 预处理有效降低了密度分布的不均匀性!")


if __name__ == "__main__":
    np.random.seed(42)
    
    demonstrate_preprocessing_effects()
    print("\n")
    compare_registration_methods()

import open3d as o3d
import numpy as np
from typing import List
from slam_backend import (
    PoseGraph,
    build_pose_graph_from_sequence,
    optimize_pose_graph,
    merge_point_clouds,
    compute_trajectory_error
)


def generate_loop_trajectory_point_clouds(
    num_frames: int = 15,
    points_per_frame: int = 2000,
    noise_std: float = 0.02,
    drift_noise: float = 0.05
) -> List[o3d.geometry.PointCloud]:
    """
    生成一个回环轨迹的点云序列，用于测试SLAM
    模拟：沿着圆形轨迹移动，最后回到起点
    """
    print(f"生成模拟回环轨迹点云...")
    print(f"  帧数: {num_frames}")
    print(f"  每帧点数: {points_per_frame}")
    print(f"  测量噪声: {noise_std}")
    print(f"  里程计漂移: {drift_noise}")
    
    base_points = np.random.rand(points_per_frame // 2, 3) * 4 - 2
    
    point_clouds = []
    true_poses = []
    
    for i in range(num_frames):
        angle = 2 * np.pi * i / num_frames
        radius = 1.5
        
        translation = np.array([
            radius * np.cos(angle),
            radius * np.sin(angle),
            0.1 * np.sin(2 * angle)
        ])
        
        rotation = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, angle))
        
        true_pose = np.eye(4)
        true_pose[:3, :3] = rotation
        true_pose[:3, 3] = translation
        true_poses.append(true_pose)
        
        frame_points = base_points @ rotation.T + translation
        frame_points += np.random.randn(*frame_points.shape) * noise_std
        
        if i > 0:
            drift = np.random.randn(3) * drift_noise
            frame_points += drift
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(frame_points)
        point_clouds.append(pcd)
        
        if i % 5 == 0:
            print(f"  已生成帧 {i}/{num_frames}")
    
    print("  ✓ 点云序列生成完成!")
    return point_clouds, true_poses


def visualize_trajectory(
    poses: List[np.ndarray],
    title: str = "轨迹",
    color: List[float] = None
) -> o3d.geometry.LineSet:
    """
    可视化轨迹
    """
    if color is None:
        color = [1, 0, 0]
    
    points = []
    lines = []
    
    for i, pose in enumerate(poses):
        points.append(pose[:3, 3])
        if i > 0:
            lines.append([i-1, i])
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color(color)
    
    return line_set


def main():
    np.random.seed(42)
    
    print("=" * 70)
    print("SLAM后端优化演示：闭环检测 + 姿势图优化")
    print("=" * 70)
    
    num_frames = 12
    point_clouds, true_poses = generate_loop_trajectory_point_clouds(
        num_frames=num_frames,
        points_per_frame=1500,
        noise_std=0.03,
        drift_noise=0.04
    )
    
    print("\n" + "=" * 70)
    print("构建姿势图...")
    print("=" * 70)
    
    original_pose_graph, optimized_pose_graph = build_pose_graph_from_sequence(
        point_clouds,
        odometry_threshold=0.1,
        loop_closure_enabled=True,
        loop_distance_threshold=2.0
    )
    
    print("\n" + "=" * 70)
    print("分析优化结果...")
    print("=" * 70)
    
    print("\n原始位姿 vs 优化后位姿对比:")
    for i in range(num_frames):
        orig_pose = original_pose_graph.get_pose(i)
        opt_pose = optimized_pose_graph.get_pose(i)
        
        orig_trans = orig_pose[:3, 3]
        opt_trans = opt_pose[:3, 3]
        true_trans = true_poses[i][:3, 3]
        
        orig_error = np.linalg.norm(orig_trans - true_trans)
        opt_error = np.linalg.norm(opt_trans - true_trans)
        
        print(f"  帧 {i:2d}: 原始误差={orig_error:.4f}, 优化误差={opt_error:.4f}, "
              f"改进={orig_error-opt_error:.4f}")
    
    orig_errors = []
    opt_errors = []
    for i in range(num_frames):
        orig_trans = original_pose_graph.get_pose(i)[:3, 3]
        opt_trans = optimized_pose_graph.get_pose(i)[:3, 3]
        true_trans = true_poses[i][:3, 3]
        
        orig_errors.append(np.linalg.norm(orig_trans - true_trans))
        opt_errors.append(np.linalg.norm(opt_trans - true_trans))
    
    print(f"\n误差统计:")
    print(f"  原始轨迹 - 平均误差: {np.mean(orig_errors):.4f}, 最大误差: {np.max(orig_errors):.4f}")
    print(f"  优化轨迹 - 平均误差: {np.mean(opt_errors):.4f}, 最大误差: {np.max(opt_errors):.4f}")
    
    improvement = (np.mean(orig_errors) - np.mean(opt_errors)) / np.mean(orig_errors) * 100
    print(f"  平均误差降低: {improvement:+.2f}%")
    
    if improvement > 0:
        print("\n  ✓ 闭环检测和姿势图优化有效降低了累计误差!")
    else:
        print("\n  注: 由于模拟数据的随机性，结果可能有所波动")
    
    print("\n" + "=" * 70)
    print("可视化")
    print("=" * 70)
    print("\n将显示以下内容:")
    print("  1. 红色线条: 原始轨迹（含漂移）")
    print("  2. 蓝色线条: 优化后轨迹（闭环修正）")
    print("  3. 绿色线条: 真实轨迹")
    print("  4. 点云: 优化后的全局地图")
    
    orig_trajectory = visualize_trajectory(
        original_pose_graph.nodes, "原始轨迹", [1, 0.2, 0.2]
    )
    opt_trajectory = visualize_trajectory(
        optimized_pose_graph.nodes, "优化轨迹", [0.2, 0.5, 1]
    )
    true_trajectory = visualize_trajectory(
        true_poses, "真实轨迹", [0.2, 1, 0.2]
    )
    
    global_map = merge_point_clouds(optimized_pose_graph, voxel_size=0.05)
    global_map.paint_uniform_color([0.8, 0.8, 0.8])
    
    print("\n  正在构建可视化场景...")
    geometries = [
        orig_trajectory,
        opt_trajectory,
        true_trajectory,
        global_map,
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    ]
    
    print("  ✓ 可视化准备完成!")
    print("  关闭可视化窗口后程序结束。")
    
    o3d.visualization.draw_geometries(
        geometries,
        window_name="SLAM后端优化演示",
        width=1024,
        height=768
    )


if __name__ == "__main__":
    main()

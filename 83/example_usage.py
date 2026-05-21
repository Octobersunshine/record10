import numpy as np
import open3d as o3d
from poisson_reconstruction import PoissonSurfaceReconstructor, NormalOrientationFixer


def example_normal_consistency_fix():
    print("=" * 60)
    print("示例1: 法向一致性修复演示")
    print("=" * 60)
    
    print("\n1. 生成法向不一致的球面点云...")
    n_points = 5000
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    phi = np.random.uniform(0, np.pi, n_points)
    r = 1.0
    
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    
    points = np.column_stack([x, y, z])
    normals = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    flip_mask = np.random.choice([True, False], n_points, p=[0.3, 0.7])
    normals[flip_mask] = -normals[flip_mask]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    
    is_consistent, outward_ratio = NormalOrientationFixer.check_normal_consistency(pcd)
    print(f"   原始法向 - 一致性: {'通过' if is_consistent else '不通过'}")
    print(f"   向外法向比例: {outward_ratio:.2%}")
    
    print("\n2. 使用传播法修复法向...")
    reconstructor = PoissonSurfaceReconstructor(depth=8)
    pcd_fixed = reconstructor.estimate_normals(pcd, orientation_method='propagation')
    
    is_consistent, outward_ratio = NormalOrientationFixer.check_normal_consistency(pcd_fixed)
    print(f"   修复后法向 - 一致性: {'通过' if is_consistent else '不通过'}")
    print(f"   向外法向比例: {outward_ratio:.2%}")
    
    print("\n3. 泊松重建...")
    mesh, densities = reconstructor.reconstruct(pcd_fixed)
    reconstructor.save_mesh(mesh, "fixed_normal_sphere.ply")
    
    return mesh


def example_screened_poisson():
    print("\n" + "=" * 60)
    print("示例2: 屏幕空间泊松重建 (Screened Poisson)")
    print("=" * 60)
    
    print("\n生成带细节的环面点云...")
    n_points = 15000
    u = np.random.uniform(0, 2 * np.pi, n_points)
    v = np.random.uniform(0, 2 * np.pi, n_points)
    R = 3.0
    r = 1.0
    
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    
    noise = np.random.normal(0, 0.05, (n_points, 3))
    points = np.column_stack([x, y, z]) + noise
    
    nx = -np.cos(u) * np.cos(v)
    ny = -np.sin(u) * np.cos(v)
    nz = -np.sin(v)
    normals = np.column_stack([nx, ny, nz])
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    
    reconstructor = PoissonSurfaceReconstructor(depth=9)
    
    print("\n--- 标准泊松重建 ---")
    mesh_std, _ = reconstructor.reconstruct(pcd, density_threshold=0.05)
    reconstructor.save_mesh(mesh_std, "torus_standard_poisson.ply")
    print(f"标准泊松: {len(mesh_std.triangles)} 个三角形")
    
    print("\n--- 屏幕空间泊松重建 (Screened Poisson) ---")
    print("  特点: 自适应密度阈值 + 特征保留 + 距离筛选")
    mesh_screened, _ = reconstructor.reconstruct_screened(
        pcd,
        screen_weight=3.0,
        confidence_based=True,
        preserve_features=True
    )
    reconstructor.save_mesh(mesh_screened, "torus_screened_poisson.ply")
    print(f"屏幕空间泊松: {len(mesh_screened.triangles)} 个三角形")
    
    print("\n对比结果:")
    print(f"  标准泊松: {len(mesh_std.triangles)} 三角形 (可能有多余的面)")
    print(f"  屏幕空间: {len(mesh_screened.triangles)} 三角形 (更干净)")


def example_screened_parameter_tuning():
    print("\n" + "=" * 60)
    print("示例3: Screened Poisson 参数调优演示")
    print("=" * 60)
    
    bunny_mesh = o3d.data.BunnyMesh()
    mesh_gt = o3d.io.read_triangle_mesh(bunny_mesh.path)
    pcd = mesh_gt.sample_points_poisson_disk(8000)
    
    reconstructor = PoissonSurfaceReconstructor(depth=9)
    pcd = reconstructor.estimate_normals(pcd, orientation_method='tangent_plane')
    
    print("\n测试不同 screen_weight 参数:")
    weights = [2.0, 4.0, 6.0]
    for w in weights:
        mesh, _ = reconstructor.reconstruct_screened(
            pcd,
            screen_weight=w,
            confidence_based=True
        )
        reconstructor.save_mesh(mesh, f"bunny_screened_w{w}.ply")
        print(f"  screen_weight={w}: {len(mesh.triangles)} 三角形")
    
    print("\n测试不同 density_threshold:")
    thresholds = [0.01, 0.05, 0.1]
    for t in thresholds:
        mesh, _ = reconstructor.reconstruct_screened(
            pcd,
            density_threshold=t,
            screen_weight=4.0,
            confidence_based=False
        )
        reconstructor.save_mesh(mesh, f"bunny_screened_t{t:.2f}.ply")
        print(f"  density_threshold={t:.2f}: {len(mesh.triangles)} 三角形")
    
    print("\n测试特征保留功能:")
    mesh_with_features, _ = reconstructor.reconstruct_screened(
        pcd, screen_weight=4.0, preserve_features=True
    )
    mesh_no_features, _ = reconstructor.reconstruct_screened(
        pcd, screen_weight=4.0, preserve_features=False
    )
    reconstructor.save_mesh(mesh_with_features, "bunny_with_features.ply")
    reconstructor.save_mesh(mesh_no_features, "bunny_no_features.ply")
    print(f"  保留特征: {len(mesh_with_features.triangles)} 三角形")
    print(f"  不保留特征: {len(mesh_no_features.triangles)} 三角形")


def example_hole_prevention():
    print("\n" + "=" * 60)
    print("示例4: 防孔洞泊松重建演示")
    print("=" * 60)
    
    print("\n生成有孔洞风险的点云 (球面加噪声)...")
    n_points = 6000
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    phi = np.random.uniform(0, np.pi, n_points)
    r = 1.0
    
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    
    hole_mask = (theta > 1.0) & (theta < 2.0) & (phi > 1.0) & (phi < 2.0)
    points = np.column_stack([x, y, z])[~hole_mask]
    normals = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    points += np.random.normal(0, 0.02, points.shape)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    
    reconstructor = PoissonSurfaceReconstructor(depth=8)
    
    print("\n--- 标准泊松重建 ---")
    mesh_std, _ = reconstructor.reconstruct(pcd, density_threshold=0.05)
    reconstructor.save_mesh(mesh_std, "hole_prone_standard.ply")
    print(f"标准泊松: {len(mesh_std.triangles)} 三角形 (可能有孔洞)")
    
    print("\n--- 防孔洞泊松重建 ---")
    mesh_hole_prevent, _ = reconstructor.reconstruct_with_hole_prevention(
        pcd,
        edge_sensitivity=2.0,
        smooth_iterations=2
    )
    reconstructor.save_mesh(mesh_hole_prevent, "hole_prone_prevented.ply")
    print(f"防孔洞: {len(mesh_hole_prevent.triangles)} 三角形 (孔洞更少)")
    
    print("\n防孔洞重建特点:")
    print("  - 边缘保护: 保留高梯度区域顶点")
    print("  - 平滑处理: 拉普拉斯平滑防止孔洞")
    print("  - 自适应阈值: 根据密度分布调整")


def example_orientation_method_comparison():
    print("\n" + "=" * 60)
    print("示例5: 不同法向定向方法对比")
    print("=" * 60)
    
    bunny_mesh = o3d.data.BunnyMesh()
    mesh_gt = o3d.io.read_triangle_mesh(bunny_mesh.path)
    pcd = mesh_gt.sample_points_poisson_disk(3000)
    
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    
    methods = ['tangent_plane', 'towards_viewpoint', 'outward', 'propagation']
    
    for method in methods:
        print(f"\n方法: {method}")
        reconstructor = PoissonSurfaceReconstructor(depth=8)
        
        pcd_method = o3d.geometry.PointCloud(pcd)
        pcd_method = reconstructor.estimate_normals(pcd_method, orientation_method=method)
        
        is_consistent, outward_ratio = NormalOrientationFixer.check_normal_consistency(pcd_method)
        print(f"   一致性: {'通过' if is_consistent else '不通过'}")
        print(f"   向外法向比例: {outward_ratio:.2%}")
        
        mesh, _ = reconstructor.reconstruct(pcd_method, remove_low_density=True)
        reconstructor.save_mesh(mesh, f"bunny_{method}.ply")


def example_complete_workflow():
    print("\n" + "=" * 60)
    print("示例6: 完整工作流演示 - 从点云到高质量网格")
    print("=" * 60)
    
    bunny_mesh = o3d.data.BunnyMesh()
    mesh_gt = o3d.io.read_triangle_mesh(bunny_mesh.path)
    pcd = mesh_gt.sample_points_poisson_disk(10000)
    
    reconstructor = PoissonSurfaceReconstructor(depth=10)
    
    print("\n步骤1: 法向估计与定向")
    pcd = reconstructor.estimate_normals(
        pcd,
        orientation_method='tangent_plane',
        radius=0.01,
        max_nn=30
    )
    
    is_consistent, outward_ratio = NormalOrientationFixer.check_normal_consistency(pcd)
    print(f"   法向一致性: {'通过' if is_consistent else '不通过'}")
    print(f"   向外法向比例: {outward_ratio:.2%}")
    
    print("\n步骤2: 屏幕空间泊松重建")
    mesh, densities = reconstructor.reconstruct_screened(
        pcd,
        screen_weight=4.0,
        confidence_based=True,
        preserve_features=True
    )
    
    print("\n步骤3: 后处理 (可选)")
    print("   - 移除退化三角形")
    mesh.remove_degenerate_triangles()
    print("   - 移除重复三角形")
    mesh.remove_duplicated_triangles()
    print("   - 移除未引用顶点")
    mesh.remove_unreferenced_vertices()
    
    reconstructor.save_mesh(mesh, "bunny_final_high_quality.ply")
    
    print(f"\n最终结果:")
    print(f"   顶点数: {len(mesh.vertices)}")
    print(f"   三角形数: {len(mesh.triangles)}")
    print(f"   网格文件: bunny_final_high_quality.ply")


def quick_start_guide():
    print("\n" + "=" * 60)
    print("快速开始指南")
    print("=" * 60)
    print("\n命令行使用示例:")
    print("-" * 40)
    print("1. 标准泊松重建:")
    print("   python poisson_reconstruction.py --input your.ply --output mesh.ply --visualize")
    
    print("\n2. 屏幕空间泊松重建 (推荐用于高质量输出):")
    print("   python poisson_reconstruction.py --input your.ply --screened --visualize")
    
    print("\n3. 屏幕空间泊松 (自定义参数):")
    print("   python poisson_reconstruction.py --input your.ply --screened --screen-weight 4.0 --density-threshold 0.05")
    
    print("\n4. 防孔洞重建 (适用于稀疏点云):")
    print("   python poisson_reconstruction.py --input your.ply --hole-prevention --edge-sensitivity 2.0")
    
    print("\n5. 演示法向不一致修复:")
    print("   python poisson_reconstruction.py --demo-inconsistent --visualize --show-normals")
    
    print("\n" + "-" * 40)
    print("关键参数说明:")
    print("  --depth: 八叉树深度 (6-12), 值越大越精细越慢")
    print("  --screen-weight: 屏幕空间筛选严格度 (2-8), 越大越保守")
    print("  --density-threshold: 低密度移除比例")
    print("  --edge-sensitivity: 防孔洞模式的边缘保护强度")
    print("\nPython API 示例:")
    print("-" * 40)
    print("from poisson_reconstruction import PoissonSurfaceReconstructor")
    print("")
    print("reconstructor = PoissonSurfaceReconstructor(depth=9)")
    print("pcd = reconstructor.load_point_cloud('input.ply')")
    print("pcd = reconstructor.estimate_normals(pcd)")
    print("mesh, _ = reconstructor.reconstruct_screened(pcd, screen_weight=4.0)")
    print("reconstructor.save_mesh(mesh, 'output.ply')")


def main():
    print("泊松表面重建 - 完整功能演示集\n")
    
    example_normal_consistency_fix()
    example_screened_poisson()
    example_screened_parameter_tuning()
    example_hole_prevention()
    example_orientation_method_comparison()
    example_complete_workflow()
    quick_start_guide()
    
    print("\n" + "=" * 60)
    print("所有示例完成! 生成的 .ply 文件可在 MeshLab 中查看对比")
    print("=" * 60)


if __name__ == '__main__':
    main()

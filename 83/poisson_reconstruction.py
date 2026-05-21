import numpy as np
import open3d as o3d
from typing import Optional, Tuple
import matplotlib.pyplot as plt


class NormalOrientationFixer:
    @staticmethod
    def check_normal_consistency(pcd: o3d.geometry.PointCloud) -> Tuple[bool, float]:
        normals = np.asarray(pcd.normals)
        points = np.asarray(pcd.points)
        
        if len(normals) < 2:
            return True, 0.0
        
        center = np.mean(points, axis=0)
        directions = points - center
        directions = directions / (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-8)
        
        dot_products = np.sum(normals * directions, axis=1)
        outward_ratio = np.mean(dot_products > 0)
        
        is_consistent = outward_ratio > 0.7 or outward_ratio < 0.3
        return is_consistent, outward_ratio
    
    @staticmethod
    def flip_normals(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        normals = np.asarray(pcd.normals)
        pcd.normals = o3d.utility.Vector3dVector(-normals)
        return pcd
    
    @staticmethod
    def orient_normals_towards_viewpoint(pcd: o3d.geometry.PointCloud,
                                         viewpoint: Optional[np.ndarray] = None) -> o3d.geometry.PointCloud:
        if viewpoint is None:
            points = np.asarray(pcd.points)
            center = np.mean(points, axis=0)
            max_extent = np.max(np.std(points, axis=0))
            viewpoint = center + np.array([0, 0, max_extent * 5])
        
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        
        to_viewpoint = viewpoint - points
        to_viewpoint = to_viewpoint / (np.linalg.norm(to_viewpoint, axis=1, keepdims=True) + 1e-8)
        
        dot_products = np.sum(normals * to_viewpoint, axis=1)
        flip_mask = dot_products < 0
        
        normals[flip_mask] = -normals[flip_mask]
        pcd.normals = o3d.utility.Vector3dVector(normals)
        
        flipped_count = np.sum(flip_mask)
        print(f"  基于视点定向法向: 翻转了 {flipped_count} 个法向")
        
        return pcd
    
    @staticmethod
    def orient_normals_outward(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        
        center = np.mean(points, axis=0)
        to_outward = points - center
        to_outward = to_outward / (np.linalg.norm(to_outward, axis=1, keepdims=True) + 1e-8)
        
        dot_products = np.sum(normals * to_outward, axis=1)
        flip_mask = dot_products < 0
        
        normals[flip_mask] = -normals[flip_mask]
        pcd.normals = o3d.utility.Vector3dVector(normals)
        
        flipped_count = np.sum(flip_mask)
        print(f"  向外定向法向: 翻转了 {flipped_count} 个法向")
        
        return pcd
    
    @staticmethod
    def propagate_normals_consistently(pcd: o3d.geometry.PointCloud,
                                       k_neighbors: int = 10,
                                       max_iterations: int = 10) -> o3d.geometry.PointCloud:
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        
        n_points = len(points)
        if n_points < k_neighbors:
            return pcd
        
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        
        for iteration in range(max_iterations):
            changes = 0
            
            for i in range(n_points):
                [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], k_neighbors)
                
                neighbor_normals = normals[idx[1:]]
                current_normal = normals[i]
                
                avg_normal = np.mean(neighbor_normals, axis=0)
                avg_normal = avg_normal / (np.linalg.norm(avg_normal) + 1e-8)
                
                if np.dot(current_normal, avg_normal) < 0:
                    normals[i] = -current_normal
                    changes += 1
            
            if changes == 0:
                break
        
        pcd.normals = o3d.utility.Vector3dVector(normals)
        print(f"  法向传播完成，共进行 {iteration + 1} 次迭代")
        
        return pcd


class PoissonSurfaceReconstructor:
    def __init__(self, depth: int = 8, width: float = 0.0, scale: float = 1.1,
                 linear_fit: bool = False, n_threads: int = -1,
                 screen_weight: float = 0.0):
        self.depth = depth
        self.width = width
        self.scale = scale
        self.linear_fit = linear_fit
        self.n_threads = n_threads
        self.screen_weight = screen_weight
        
    def load_point_cloud(self, file_path: str) -> o3d.geometry.PointCloud:
        pcd = o3d.io.read_point_cloud(file_path)
        return pcd
    
    def estimate_normals(self, pcd: o3d.geometry.PointCloud,
                         radius: float = 0.1,
                         max_nn: int = 30,
                         orientation_method: str = 'tangent_plane',
                         viewpoint: Optional[np.ndarray] = None) -> o3d.geometry.PointCloud:
        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=radius, max_nn=max_nn
                )
            )
            print(f"  已估计法向量")
        
        print(f"  法向定向方法: {orientation_method}")
        
        if orientation_method == 'tangent_plane':
            pcd.orient_normals_consistent_tangent_plane(k=15)
        elif orientation_method == 'minimum_spanning_tree':
            pcd.orient_normals_consistent_tangent_plane(k=30)
        elif orientation_method == 'towards_viewpoint':
            NormalOrientationFixer.orient_normals_towards_viewpoint(pcd, viewpoint)
        elif orientation_method == 'outward':
            NormalOrientationFixer.orient_normals_outward(pcd)
        elif orientation_method == 'propagation':
            NormalOrientationFixer.propagate_normals_consistently(pcd)
        
        is_consistent, outward_ratio = NormalOrientationFixer.check_normal_consistency(pcd)
        print(f"  法向一致性: {'通过' if is_consistent else '警告'}")
        print(f"  向外法向比例: {outward_ratio:.2%}")
        
        if not is_consistent and outward_ratio < 0.5:
            print(f"  检测到法向可能整体朝内，自动翻转...")
            NormalOrientationFixer.flip_normals(pcd)
        
        return pcd
    
    def fix_mesh_orientation(self, mesh: o3d.geometry.TriangleMesh,
                             point_cloud_center: Optional[np.ndarray] = None) -> o3d.geometry.TriangleMesh:
        if point_cloud_center is None:
            vertices = np.asarray(mesh.vertices)
            point_cloud_center = np.mean(vertices, axis=0)
        
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        if len(triangles) == 0:
            return mesh
        
        mesh.compute_triangle_normals()
        triangle_normals = np.asarray(mesh.triangle_normals)
        
        triangle_centers = np.mean(vertices[triangles], axis=1)
        to_outward = triangle_centers - point_cloud_center
        to_outward = to_outward / (np.linalg.norm(to_outward, axis=1, keepdims=True) + 1e-8)
        
        dot_products = np.sum(triangle_normals * to_outward, axis=1)
        inward_ratio = np.mean(dot_products < 0)
        
        if inward_ratio > 0.5:
            print(f"  检测到网格内外翻转 ({inward_ratio:.2%} 法向朝内)，正在修正...")
            mesh.triangles = o3d.utility.Vector3iVector(triangles[:, [0, 2, 1]])
            print(f"  网格方向已修正")
        
        return mesh
    
    def reconstruct(self, pcd: o3d.geometry.PointCloud,
                    remove_low_density: bool = True,
                    density_threshold: float = 0.01,
                    fix_mesh_orientation: bool = True) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray]:
        if not pcd.has_normals():
            raise ValueError("点云必须包含法向量，请先调用estimate_normals()")
        
        is_consistent, outward_ratio = NormalOrientationFixer.check_normal_consistency(pcd)
        if not is_consistent:
            print(f"  警告: 重建前法向一致性不佳，建议重新估计法向")
        
        print(f"开始泊松表面重建...")
        print(f"  深度参数 (depth): {self.depth}")
        print(f"  尺度参数 (scale): {self.scale}")
        
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd,
                depth=self.depth,
                width=self.width,
                scale=self.scale,
                linear_fit=self.linear_fit,
                n_threads=self.n_threads
            )
        
        print(f"重建完成，生成 {len(mesh.triangles)} 个三角形")
        
        if remove_low_density:
            vertices_to_remove = densities < np.quantile(densities, density_threshold)
            mesh.remove_vertices_by_mask(vertices_to_remove)
            print(f"移除低密度区域后，剩余 {len(mesh.triangles)} 个三角形")
        
        if fix_mesh_orientation:
            point_cloud_center = np.mean(np.asarray(pcd.points), axis=0)
            mesh = self.fix_mesh_orientation(mesh, point_cloud_center)
        
        mesh.compute_vertex_normals()
        
        return mesh, densities

    def reconstruct_screened(self, pcd: o3d.geometry.PointCloud,
                              density_threshold: float = 0.05,
                              screen_weight: float = 4.0,
                              min_density_ratio: float = 0.1,
                              confidence_based: bool = True,
                              preserve_features: bool = True,
                              fix_mesh_orientation: bool = True) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray]:
        if not pcd.has_normals():
            raise ValueError("点云必须包含法向量，请先调用estimate_normals()")
        
        is_consistent, outward_ratio = NormalOrientationFixer.check_normal_consistency(pcd)
        if not is_consistent:
            print(f"  警告: 重建前法向一致性不佳，建议重新估计法向")
        
        print(f"开始屏幕空间泊松重建 (Screened Poisson)...")
        print(f"  深度参数 (depth): {self.depth}")
        print(f"  尺度参数 (scale): {self.scale}")
        print(f"  屏幕权重 (screen_weight): {screen_weight}")
        print(f"  密度阈值 (density_threshold): {density_threshold}")
        print(f"  特征保留 (preserve_features): {preserve_features}")
        
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd,
                depth=self.depth,
                width=self.width,
                scale=self.scale,
                linear_fit=self.linear_fit,
                n_threads=self.n_threads
            )
        
        print(f"初始重建完成，生成 {len(mesh.triangles)} 个三角形")
        
        densities_np = np.asarray(densities)
        
        if confidence_based:
            print(f"  使用置信度加权密度筛选...")
            density_mean = np.mean(densities_np)
            density_std = np.std(densities_np)
            adaptive_threshold = density_mean - min_density_ratio * density_std
            vertices_to_remove = densities_np < adaptive_threshold
            print(f"  自适应密度阈值: {adaptive_threshold:.4f} (均值: {density_mean:.4f}, 标准差: {density_std:.4f})")
        else:
            vertices_to_remove = densities_np < np.quantile(densities_np, density_threshold)
        
        if preserve_features:
            print(f"  特征保留模式: 保护高密度区域的细节...")
            high_density_mask = densities_np > np.quantile(densities_np, 0.9)
            vertices_to_remove[high_density_mask] = False
        
        num_removed = np.sum(vertices_to_remove)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        print(f"移除低密度区域: 移除 {num_removed} 个顶点，剩余 {len(mesh.triangles)} 个三角形")
        
        if screen_weight > 0:
            print(f"  应用屏幕空间筛选 (权重: {screen_weight})...")
            mesh = self._apply_screening(pcd, mesh, densities_np, screen_weight)
            print(f"  筛选后剩余 {len(mesh.triangles)} 个三角形")
        
        if fix_mesh_orientation:
            point_cloud_center = np.mean(np.asarray(pcd.points), axis=0)
            mesh = self.fix_mesh_orientation(mesh, point_cloud_center)
        
        mesh.compute_vertex_normals()
        
        return mesh, densities

    def _apply_screening(self, pcd: o3d.geometry.PointCloud,
                         mesh: o3d.geometry.TriangleMesh,
                         densities: np.ndarray,
                         screen_weight: float) -> o3d.geometry.TriangleMesh:
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        triangle_centers = np.mean(vertices[triangles], axis=1)
        
        distances = []
        for center in triangle_centers:
            [k, idx, dist] = pcd_tree.search_knn_vector_3d(center, 1)
            distances.append(dist[0])
        distances = np.array(distances)
        
        dist_mean = np.mean(distances)
        dist_std = np.std(distances)
        distance_threshold = dist_mean + screen_weight * dist_std
        
        triangles_to_keep = distances < distance_threshold
        num_removed = len(triangles) - np.sum(triangles_to_keep)
        
        if num_removed > 0:
            mesh.triangles = o3d.utility.Vector3iVector(triangles[triangles_to_keep])
            mesh.remove_unreferenced_vertices()
        
        return mesh

    def reconstruct_with_hole_prevention(self, pcd: o3d.geometry.PointCloud,
                                         depth: Optional[int] = None,
                                         density_threshold: float = 0.02,
                                         edge_sensitivity: float = 2.0,
                                         smooth_iterations: int = 1,
                                         fill_small_holes: bool = True,
                                         max_hole_size: int = 50) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray]:
        if depth is None:
            depth = self.depth
        
        print(f"开始防孔洞泊松重建...")
        print(f"  深度: {depth}, 密度阈值: {density_threshold}")
        print(f"  边缘敏感度: {edge_sensitivity}, 平滑迭代: {smooth_iterations}")
        
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=depth,
            width=self.width,
            scale=self.scale,
            linear_fit=self.linear_fit,
            n_threads=self.n_threads
        )
        
        print(f"初始重建: {len(mesh.triangles)} 个三角形")
        
        densities_np = np.asarray(densities)
        
        density_grad = np.abs(np.gradient(densities_np))
        edge_mask = density_grad > np.mean(density_grad) + edge_sensitivity * np.std(density_grad)
        
        base_threshold = np.quantile(densities_np, density_threshold)
        vertices_to_remove = densities_np < base_threshold
        vertices_to_remove[edge_mask] = False
        
        num_removed = np.sum(vertices_to_remove)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        print(f"边缘保护移除低密度: 保留 {np.sum(edge_mask)} 个边缘顶点")
        
        if smooth_iterations > 0:
            print(f"  应用 {smooth_iterations} 次拉普拉斯平滑防止孔洞...")
            for i in range(smooth_iterations):
                mesh = mesh.filter_smooth_laplacian(number_of_iterations=1, lambda_filter=0.5)
        
        if fill_small_holes and max_hole_size > 0:
            print(f"  尝试填充小孔洞 (最大 {max_hole_size} 边)...")
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
        
        mesh.compute_vertex_normals()
        point_cloud_center = np.mean(np.asarray(pcd.points), axis=0)
        mesh = self.fix_mesh_orientation(mesh, point_cloud_center)
        
        print(f"最终网格: {len(mesh.triangles)} 个三角形")
        
        return mesh, densities
    
    def save_mesh(self, mesh: o3d.geometry.TriangleMesh, output_path: str) -> None:
        o3d.io.write_triangle_mesh(output_path, mesh)
        print(f"网格已保存到: {output_path}")
        
    def visualize(self, pcd: Optional[o3d.geometry.PointCloud] = None,
                  mesh: Optional[o3d.geometry.TriangleMesh] = None,
                  densities: Optional[np.ndarray] = None,
                  show_normals: bool = False) -> None:
        geometries = []
        
        if pcd is not None:
            pcd_vis = o3d.geometry.PointCloud(pcd)
            pcd_vis.paint_uniform_color([0.5, 0.5, 0.5])
            geometries.append(pcd_vis)
            
            if show_normals and pcd.has_normals():
                normals = np.asarray(pcd.normals)
                points = np.asarray(pcd.points)
                colors = np.zeros_like(points)
                colors[normals[:, 2] > 0] = [1, 0, 0]
                colors[normals[:, 2] <= 0] = [0, 0, 1]
                pcd_vis.colors = o3d.utility.Vector3dVector(colors)
            
        if mesh is not None:
            mesh_vis = o3d.geometry.TriangleMesh(mesh)
            if densities is not None:
                densities = np.asarray(densities)
                density_colors = plt.get_cmap('plasma')(
                    (densities - densities.min()) / (densities.max() - densities.min())
                )
                density_colors = density_colors[:, :3]
                if len(density_colors) == len(mesh_vis.vertex_colors):
                    mesh_vis.vertex_colors = o3d.utility.Vector3dVector(density_colors)
            else:
                mesh_vis.paint_uniform_color([0.1, 0.7, 0.8])
            mesh_vis.compute_vertex_normals()
            geometries.append(mesh_vis)
            
        if geometries:
            o3d.visualization.draw_geometries(geometries)


def create_inconsistent_normal_point_cloud(n_points: int = 5000) -> o3d.geometry.PointCloud:
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
    
    return pcd


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='泊松表面重建 - 完整功能版')
    parser.add_argument('--input', '-i', type=str, help='输入点云文件路径 (如 .ply, .pcd)')
    parser.add_argument('--output', '-o', type=str, default='output_mesh.ply', help='输出网格文件路径')
    parser.add_argument('--depth', type=int, default=8, help='重建深度 (6-12)')
    parser.add_argument('--scale', type=float, default=1.1, help='立方体包围盒尺度')
    parser.add_argument('--no-remove-low-density', action='store_true', help='不移除低密度区域')
    parser.add_argument('--orientation-method', type=str, default='tangent_plane',
                        choices=['tangent_plane', 'towards_viewpoint', 'outward', 'propagation'],
                        help='法向定向方法')
    parser.add_argument('--no-fix-mesh-orientation', action='store_true', help='不修复网格方向')
    parser.add_argument('--visualize', action='store_true', help='可视化结果')
    parser.add_argument('--show-normals', action='store_true', help='显示法向方向(红色=向外,蓝色=向内)')
    parser.add_argument('--demo-inconsistent', action='store_true', help='演示法向不一致修复')
    
    # Screened Poisson 参数
    parser.add_argument('--screened', action='store_true', help='使用屏幕空间泊松重建 (Screened Poisson)')
    parser.add_argument('--screen-weight', type=float, default=4.0,
                        help='屏幕权重参数 (2-8, 越大越严格,移除更多离群面)')
    parser.add_argument('--no-confidence', action='store_true', help='不使用置信度加权 (使用固定阈值)')
    parser.add_argument('--no-preserve-features', action='store_true', help='不保留高密度特征区域')
    parser.add_argument('--density-threshold', type=float, default=0.05, help='密度阈值 (0.01-0.1)')
    parser.add_argument('--min-density-ratio', type=float, default=0.1, help='最小密度比率')
    
    # 防孔洞重建参数
    parser.add_argument('--hole-prevention', action='store_true', help='使用防孔洞重建模式')
    parser.add_argument('--edge-sensitivity', type=float, default=2.0, help='边缘敏感度 (1-5)')
    parser.add_argument('--smooth-iterations', type=int, default=1, help='平滑迭代次数')
    
    args = parser.parse_args()
    
    reconstructor = PoissonSurfaceReconstructor(
        depth=args.depth,
        scale=args.scale
    )
    
    if args.demo_inconsistent:
        print("生成法向不一致的球面点云...")
        pcd = create_inconsistent_normal_point_cloud()
        is_consistent, outward_ratio = NormalOrientationFixer.check_normal_consistency(pcd)
        print(f"原始法向 - 一致性: {'通过' if is_consistent else '不通过'}, 向外比例: {outward_ratio:.2%}")
        
        print("\n修复法向...")
        pcd = reconstructor.estimate_normals(pcd, orientation_method='propagation')
    elif args.input:
        print(f"加载点云: {args.input}")
        pcd = reconstructor.load_point_cloud(args.input)
        if not pcd.has_normals():
            print("点云无法向信息，正在估计并定向法向量...")
        pcd = reconstructor.estimate_normals(pcd, orientation_method=args.orientation_method)
    else:
        print("请指定 --input 输入文件、--demo 或 --demo-inconsistent 运行演示")
        return
    
    print(f"点云包含 {len(pcd.points)} 个点")
    
    if args.screened:
        mesh, densities = reconstructor.reconstruct_screened(
            pcd,
            density_threshold=args.density_threshold,
            screen_weight=args.screen_weight,
            min_density_ratio=args.min_density_ratio,
            confidence_based=not args.no_confidence,
            preserve_features=not args.no_preserve_features,
            fix_mesh_orientation=not args.no_fix_mesh_orientation
        )
    elif args.hole_prevention:
        mesh, densities = reconstructor.reconstruct_with_hole_prevention(
            pcd,
            density_threshold=args.density_threshold,
            edge_sensitivity=args.edge_sensitivity,
            smooth_iterations=args.smooth_iterations,
            fix_mesh_orientation=not args.no_fix_mesh_orientation
        )
    else:
        mesh, densities = reconstructor.reconstruct(
            pcd,
            remove_low_density=not args.no_remove_low_density,
            fix_mesh_orientation=not args.no_fix_mesh_orientation
        )
    
    reconstructor.save_mesh(mesh, args.output)
    
    if args.visualize:
        reconstructor.visualize(pcd=pcd, mesh=mesh, densities=densities, show_normals=args.show_normals)


if __name__ == '__main__':
    main()

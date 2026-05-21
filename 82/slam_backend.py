import open3d as o3d
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from icp_registration import icp_registration


class PoseGraphEdge:
    """姿势图边：表示两个位姿之间的约束"""
    def __init__(
        self,
        source_id: int,
        target_id: int,
        transformation: np.ndarray,
        information: np.ndarray,
        is_loop_closure: bool = False
    ):
        self.source_id = source_id
        self.target_id = target_id
        self.transformation = transformation
        self.information = information
        self.is_loop_closure = is_loop_closure


class PoseGraph:
    """姿势图数据结构"""
    def __init__(self):
        self.nodes: List[np.ndarray] = []
        self.edges: List[PoseGraphEdge] = []
        self.node_point_clouds: List[o3d.geometry.PointCloud] = []
    
    def add_node(
        self,
        pose: np.ndarray,
        point_cloud: Optional[o3d.geometry.PointCloud] = None
    ) -> int:
        """添加位姿节点"""
        node_id = len(self.nodes)
        self.nodes.append(pose.copy())
        if point_cloud is not None:
            self.node_point_clouds.append(point_cloud)
        return node_id
    
    def add_edge(
        self,
        source_id: int,
        target_id: int,
        transformation: np.ndarray,
        information: Optional[np.ndarray] = None,
        is_loop_closure: bool = False
    ) -> None:
        """添加边约束"""
        if information is None:
            information = np.eye(6)
        
        edge = PoseGraphEdge(
            source_id, target_id, transformation, information, is_loop_closure
        )
        self.edges.append(edge)
    
    def get_pose(self, node_id: int) -> np.ndarray:
        """获取节点位姿"""
        return self.nodes[node_id]
    
    def update_pose(self, node_id: int, new_pose: np.ndarray) -> None:
        """更新节点位姿"""
        self.nodes[node_id] = new_pose.copy()
    
    def to_open3d_pose_graph(self) -> o3d.pipelines.registration.PoseGraph:
        """转换为Open3D的PoseGraph格式"""
        o3d_pose_graph = o3d.pipelines.registration.PoseGraph()
        
        for pose in self.nodes:
            o3d_node = o3d.pipelines.registration.PoseGraphNode(pose)
            o3d_pose_graph.nodes.append(o3d_node)
        
        for edge in self.edges:
            o3d_edge = o3d.pipelines.registration.PoseGraphEdge(
                edge.source_id,
                edge.target_id,
                edge.transformation,
                edge.information,
                uncertain=edge.is_loop_closure
            )
            o3d_pose_graph.edges.append(o3d_edge)
        
        return o3d_pose_graph
    
    def from_open3d_pose_graph(self, o3d_pose_graph: o3d.pipelines.registration.PoseGraph) -> None:
        """从Open3D的PoseGraph更新"""
        for i, node in enumerate(o3d_pose_graph.nodes):
            if i < len(self.nodes):
                self.nodes[i] = node.pose.copy()


def compute_information_matrix(
    rmse: float,
    fitness: float,
    num_inliers: int
) -> np.ndarray:
    """
    计算信息矩阵（协方差矩阵的逆）
    基于配准质量动态调整权重
    """
    base_weight = fitness * np.clip(num_inliers / 100, 0.1, 1.0)
    rotation_weight = base_weight / max(rmse, 0.001)
    translation_weight = base_weight / max(rmse, 0.001)
    
    information = np.eye(6)
    information[0:3, 0:3] *= rotation_weight
    information[3:6, 3:6] *= translation_weight
    
    return information


def extract_fpfh_features(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float = 0.05
) -> o3d.pipelines.registration.Feature:
    """
    提取FPFH特征用于闭环检测
    """
    radius_normal = voxel_size * 2
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    
    radius_feature = voxel_size * 5
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    
    return fpfh


def feature_matching(
    source_feature: o3d.pipelines.registration.Feature,
    target_feature: o3d.pipelines.registration.Feature,
    threshold: float = 0.9
) -> Tuple[np.ndarray, np.ndarray]:
    """
    基于FPFH特征的匹配
    """
    source_data = np.asarray(source_feature.data)
    target_data = np.asarray(target_feature.data)
    
    source_tree = o3d.geometry.KDTreeFlann(target_feature)
    target_tree = o3d.geometry.KDTreeFlann(source_feature)
    
    matches = []
    
    for i in range(source_data.shape[1]):
        k, idx1, dist1 = source_tree.search_knn_vector_xd(source_data[:, i], 1)
        if k > 0:
            k, idx2, dist2 = target_tree.search_knn_vector_xd(target_data[:, idx1[0]], 1)
            if k > 0 and idx2[0] == i:
                if dist1[0] < threshold * threshold:
                    matches.append((i, idx1[0]))
    
    if len(matches) == 0:
        return np.array([]), np.array([])
    
    source_indices = np.array([m[0] for m in matches])
    target_indices = np.array([m[1] for m in matches])
    
    return source_indices, target_indices


def detect_loop_closures(
    pose_graph: PoseGraph,
    current_id: int,
    min_overlap: float = 0.3,
    distance_threshold: float = 1.5,
    feature_voxel_size: float = 0.05
) -> List[Tuple[int, int, np.ndarray, np.ndarray]]:
    """
    闭环检测
    
    检测策略：
    1. 基于位姿距离筛选候选帧
    2. FPFH特征匹配
    3. ICP精配准验证
    """
    loop_closures = []
    
    if current_id < 10:
        return loop_closures
    
    current_pose = pose_graph.get_pose(current_id)
    current_pcd = pose_graph.node_point_clouds[current_id]
    
    current_feature = extract_fpfh_features(current_pcd, feature_voxel_size)
    
    for candidate_id in range(current_id - 10):
        candidate_pose = pose_graph.get_pose(candidate_id)
        
        translation_distance = np.linalg.norm(
            current_pose[:3, 3] - candidate_pose[:3, 3]
        )
        
        if translation_distance > distance_threshold:
            continue
        
        candidate_pcd = pose_graph.node_point_clouds[candidate_id]
        candidate_feature = extract_fpfh_features(candidate_pcd, feature_voxel_size)
        
        source_idx, target_idx = feature_matching(current_feature, candidate_feature)
        
        if len(source_idx) < 30:
            continue
        
        current_pcd_sample = current_pcd.select_by_index(source_idx)
        candidate_pcd_sample = candidate_pcd.select_by_index(target_idx)
        
        init_transform = np.linalg.inv(candidate_pose) @ current_pose
        
        transform, result = icp_registration(
            current_pcd_sample,
            candidate_pcd_sample,
            threshold=feature_voxel_size * 2,
            max_iteration=100,
            init_transformation=init_transform,
            voxel_size=None,
            use_density_sampling=False
        )
        
        if result.fitness > min_overlap and result.inlier_rmse < feature_voxel_size * 3:
            relative_transform = np.linalg.inv(candidate_pose) @ current_pose @ transform
            
            information = compute_information_matrix(
                result.inlier_rmse,
                result.fitness,
                len(result.correspondence_set)
            )
            
            loop_closures.append((
                current_id,
                candidate_id,
                relative_transform,
                information
            ))
            
            print(f"  检测到闭环: 帧 {current_id} <-> 帧 {candidate_id}, "
                  f"匹配度: {result.fitness:.3f}, RMSE: {result.inlier_rmse:.4f}")
    
    return loop_closures


def optimize_pose_graph(
    pose_graph: PoseGraph,
    max_iteration: int = 1000,
    preference_loop_closure: float = 0.1
) -> PoseGraph:
    """
    全局姿势图优化
    
    使用g2o或Open3D内置的优化器进行全局一致性调整
    """
    o3d_pose_graph = pose_graph.to_open3d_pose_graph()
    
    method = o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt()
    criteria = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(
        max_iteration=max_iteration
    )
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=0.05,
        edge_prune_threshold=0.25,
        preference_loop_closure=preference_loop_closure,
        reference_node=0
    )
    
    o3d.pipelines.registration.global_optimization(
        o3d_pose_graph, method, criteria, option
    )
    
    pose_graph.from_open3d_pose_graph(o3d_pose_graph)
    
    return pose_graph


def build_pose_graph_from_sequence(
    point_clouds: List[o3d.geometry.PointCloud],
    odometry_threshold: float = 0.05,
    loop_closure_enabled: bool = True,
    loop_distance_threshold: float = 1.5
) -> Tuple[PoseGraph, PoseGraph]:
    """
    从点云序列构建姿势图
    
    Returns:
        (original_pose_graph, optimized_pose_graph)
    """
    print("=" * 60)
    print("构建姿势图并进行SLAM后端优化")
    print("=" * 60)
    
    pose_graph = PoseGraph()
    
    print(f"\n1. 添加初始节点...")
    pose_graph.add_node(np.eye(4), point_clouds[0])
    print(f"   帧 0: 初始位姿")
    
    print(f"\n2. 添加连续帧间约束（里程计边）...")
    for i in range(1, len(point_clouds)):
        source = point_clouds[i-1]
        target = point_clouds[i]
        
        prev_pose = pose_graph.get_pose(i-1)
        
        transform, result = icp_registration(
            source, target,
            threshold=odometry_threshold,
            max_iteration=200,
            voxel_size=0.05,
            use_density_sampling=True
        )
        
        current_pose = prev_pose @ np.linalg.inv(transform)
        pose_graph.add_node(current_pose, point_clouds[i])
        
        information = compute_information_matrix(
            result.inlier_rmse, result.fitness, len(result.correspondence_set)
        )
        
        pose_graph.add_edge(
            source_id=i-1,
            target_id=i,
            transformation=transform,
            information=information,
            is_loop_closure=False
        )
        
        print(f"   帧 {i-1} -> 帧 {i}: Fitness={result.fitness:.3f}, "
              f"RMSE={result.inlier_rmse:.4f}")
    
    original_pose_graph = PoseGraph()
    original_pose_graph.nodes = [p.copy() for p in pose_graph.nodes]
    original_pose_graph.edges = pose_graph.edges.copy()
    original_pose_graph.node_point_clouds = pose_graph.node_point_clouds.copy()
    
    if loop_closure_enabled:
        print(f"\n3. 检测闭环...")
        total_loop_closures = 0
        
        for i in range(len(point_clouds)):
            loops = detect_loop_closures(
                pose_graph, i,
                distance_threshold=loop_distance_threshold
            )
            
            for (current_id, candidate_id, transform, information) in loops:
                pose_graph.add_edge(
                    source_id=current_id,
                    target_id=candidate_id,
                    transformation=transform,
                    information=information,
                    is_loop_closure=True
                )
                total_loop_closures += 1
        
        print(f"   共检测到 {total_loop_closures} 个闭环")
    
    print(f"\n4. 姿势图统计:")
    print(f"   节点数: {len(pose_graph.nodes)}")
    print(f"   边数: {len(pose_graph.edges)}")
    loop_count = sum(1 for e in pose_graph.edges if e.is_loop_closure)
    print(f"   闭环边数: {loop_count}")
    
    print(f"\n5. 执行全局优化...")
    optimized_pose_graph = optimize_pose_graph(pose_graph)
    
    print("\n   ✓ 姿势图优化完成!")
    
    return original_pose_graph, optimized_pose_graph


def compute_trajectory_error(
    optimized_poses: List[np.ndarray],
    ground_truth_poses: Optional[List[np.ndarray]] = None
) -> Dict[str, float]:
    """
    计算轨迹误差
    """
    errors = []
    for i in range(1, len(optimized_poses)):
        delta_optimized = np.linalg.inv(optimized_poses[i-1]) @ optimized_poses[i]
        translation_error = np.linalg.norm(delta_optimized[:3, 3])
        errors.append(translation_error)
    
    error_array = np.array(errors)
    
    return {
        'mean': np.mean(error_array),
        'std': np.std(error_array),
        'max': np.max(error_array),
        'ate': np.mean(error_array)
    }


def merge_point_clouds(
    pose_graph: PoseGraph,
    voxel_size: float = 0.02
) -> o3d.geometry.PointCloud:
    """
    融合所有点云生成全局地图
    """
    merged = o3d.geometry.PointCloud()
    
    for i, pcd in enumerate(pose_graph.node_point_clouds):
        pose = pose_graph.get_pose(i)
        transformed_pcd = o3d.geometry.PointCloud(pcd)
        transformed_pcd.transform(pose)
        merged += transformed_pcd
    
    merged = merged.voxel_down_sample(voxel_size=voxel_size)
    
    return merged

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from deepsc import DeepSC
from threedsc import ThreeDSC


def generate_sphere(n_points: int = 500, radius: float = 1.0) -> np.ndarray:
    phi = np.random.uniform(0, np.pi, n_points)
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.column_stack([x, y, z])


def generate_ellipsoid(n_points: int = 500, rx: float = 1.0, ry: float = 1.0, rz: float = 1.0) -> np.ndarray:
    phi = np.random.uniform(0, np.pi, n_points)
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    x = rx * np.sin(phi) * np.cos(theta)
    y = ry * np.sin(phi) * np.sin(theta)
    z = rz * np.cos(phi)
    return np.column_stack([x, y, z])


def deform_point_cloud(points: np.ndarray, deformation_strength: float = 0.1) -> np.ndarray:
    noise = np.random.randn(*points.shape) * deformation_strength
    warped = points.copy()
    warped[:, 2] += np.sin(points[:, 0] * 4) * deformation_strength * 2
    warped[:, 1] += np.cos(points[:, 2] * 3) * deformation_strength
    return warped + noise


def rotate_point_cloud(points: np.ndarray, angle_x: float, angle_y: float, angle_z: float) -> np.ndarray:
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(angle_x), -np.sin(angle_x)],
        [0, np.sin(angle_x), np.cos(angle_x)]
    ])
    Ry = np.array([
        [np.cos(angle_y), 0, np.sin(angle_y)],
        [0, 1, 0],
        [-np.sin(angle_y), 0, np.cos(angle_y)]
    ])
    Rz = np.array([
        [np.cos(angle_z), -np.sin(angle_z), 0],
        [np.sin(angle_z), np.cos(angle_z), 0],
        [0, 0, 1]
    ])
    R = Rz @ Ry @ Rx
    return points @ R.T


def visualize_comparison(pc1: np.ndarray, pc2: np.ndarray, matches_deepsc: np.ndarray, 
                         matches_3dsc: np.ndarray, title: str = ""):
    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')

    ax1.scatter(pc1[:, 0], pc1[:, 1], pc1[:, 2], c='b', s=10, alpha=0.6)
    ax1.set_title('Original Point Cloud')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    ax2.scatter(pc2[:, 0], pc2[:, 1], pc2[:, 2], c='r', s=10, alpha=0.6)
    if len(matches_deepsc) > 0:
        for idx1, idx2 in matches_deepsc[:15]:
            color = np.random.rand(3)
            ax1.scatter(pc1[idx1, 0], pc1[idx1, 1], pc1[idx1, 2], c=color, s=80, marker='o', edgecolors='k')
            ax2.scatter(pc2[idx2, 0], pc2[idx2, 1], pc2[idx2, 2], c=color, s=80, marker='o', edgecolors='k')
    ax2.set_title(f'DeepSC Matches ({len(matches_deepsc)})')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    ax3.scatter(pc2[:, 0], pc2[:, 1], pc2[:, 2], c='r', s=10, alpha=0.6)
    if len(matches_3dsc) > 0:
        for idx1, idx2 in matches_3dsc[:15]:
            color = np.random.rand(3)
            ax1.scatter(pc1[idx1, 0], pc1[idx1, 1], pc1[idx1, 2], c=color, s=80, marker='^', edgecolors='k')
            ax3.scatter(pc2[idx2, 0], pc2[idx2, 1], pc2[idx2, 2], c=color, s=80, marker='^', edgecolors='k')
    ax3.set_title(f'3DSC Matches ({len(matches_3dsc)})')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def main():
    print("=" * 70)
    print("深度形状上下文 (DeepSC) vs 传统 3DSC - 形变鲁棒性对比")
    print("=" * 70)

    np.random.seed(42)

    print("\n1. 生成测试数据...")
    sphere_original = generate_sphere(n_points=300, radius=1.0)
    print(f"   原始球体点数: {sphere_original.shape[0]}")

    deformation_levels = [0.02, 0.05, 0.1, 0.15, 0.2]
    deformed_spheres = []
    for level in deformation_levels:
        deformed = deform_point_cloud(sphere_original.copy(), deformation_strength=level)
        deformed = rotate_point_cloud(deformed, 0.2, 0.3, 0.1)
        deformed_spheres.append(deformed)
        print(f"   形变强度 {level}: {deformed.shape[0]} 点")

    cube = generate_ellipsoid(n_points=300, rx=0.8, ry=0.6, rz=1.2)
    print(f"   椭球体(异类)点数: {cube.shape[0]}")

    print("\n2. 初始化描述子...")
    deepsc = DeepSC(
        feature_dim=128,
        num_groups=[128, 64, 32],
        group_radii=[0.2, 0.4, 0.8],
        nsamples=[16, 32, 64]
    )
    print(f"   DeepSC 特征维度: {deepsc.feature_dim}")

    threedsc = ThreeDSC(
        radius=0.5,
        num_bins_r=5,
        num_bins_azim=6,
        num_bins_polar=3,
        use_log=True
    )
    print(f"   3DSC 描述子维度: {threedsc.descriptor_size}")

    print("\n3. 自监督预训练 DeepSC...")
    training_data = [
        generate_sphere(200, 1.0),
        generate_sphere(200, 0.9),
        generate_sphere(200, 1.1),
        deform_point_cloud(generate_sphere(200, 1.0), 0.05),
        deform_point_cloud(generate_sphere(200, 1.0), 0.08),
    ]
    deepsc.self_supervised_train(training_data, epochs=5)

    print("\n4. 提取原始点云特征...")
    deepsc_original = deepsc.extract_features(sphere_original)
    threedsc_original = threedsc.compute_descriptor(sphere_original)
    print(f"   DeepSC 特征形状: {deepsc_original.shape}")
    print(f"   3DSC 描述子形状: {threedsc_original.shape}")

    print("\n5. 形变鲁棒性测试...")
    print("-" * 70)
    print(f"{'形变强度':<12} {'DeepSC匹配数':<15} {'3DSC匹配数':<15} {'DeepSC/3DSC':<12}")
    print("-" * 70)

    deepsc_match_counts = []
    threedsc_match_counts = []

    for i, level in enumerate(deformation_levels):
        deepsc_deformed = deepsc.extract_features(deformed_spheres[i])
        threedsc_deformed = threedsc.compute_descriptor(deformed_spheres[i])

        matches_deepsc, _ = deepsc.match_features(deepsc_original, deepsc_deformed, ratio_threshold=0.9)
        matches_3dsc, _ = threedsc.match_descriptors(threedsc_original, threedsc_deformed)

        deepsc_match_counts.append(len(matches_deepsc))
        threedsc_match_counts.append(len(matches_3dsc))

        ratio = len(matches_deepsc) / max(len(matches_3dsc), 1)
        print(f"{level:<12} {len(matches_deepsc):<15} {len(matches_3dsc):<15} {ratio:<12.2f}x")

    print("-" * 70)

    print("\n6. 异类区分测试...")
    deepsc_cube = deepsc.extract_features(cube)
    threedsc_cube = threedsc.compute_descriptor(cube)

    matches_deepsc_cross, _ = deepsc.match_features(deepsc_original, deepsc_cube, ratio_threshold=0.9)
    matches_3dsc_cross, _ = threedsc.match_descriptors(threedsc_original, threedsc_cube)

    print(f"   DeepSC 同类匹配数: {deepsc_match_counts[1]}")
    print(f"   DeepSC 异类匹配数: {len(matches_deepsc_cross)}")
    print(f"   DeepSC 区分度: {deepsc_match_counts[1] / max(len(matches_deepsc_cross), 1):.2f}x")
    print(f"   3DSC 同类匹配数: {threedsc_match_counts[1]}")
    print(f"   3DSC 异类匹配数: {len(matches_3dsc_cross)}")
    print(f"   3DSC 区分度: {threedsc_match_counts[1] / max(len(matches_3dsc_cross), 1):.2f}x")

    print("\n7. 可视化对比...")
    best_idx = 1
    deepsc_deformed = deepsc.extract_features(deformed_spheres[best_idx])
    threedsc_deformed = threedsc.compute_descriptor(deformed_spheres[best_idx])
    matches_deepsc, _ = deepsc.match_features(deepsc_original, deepsc_deformed, ratio_threshold=0.9)
    matches_3dsc, _ = threedsc.match_descriptors(threedsc_original, threedsc_deformed)

    visualize_comparison(
        sphere_original, deformed_spheres[best_idx],
        matches_deepsc, matches_3dsc,
        title=f"形变强度 {deformation_levels[best_idx]} - DeepSC vs 3DSC 匹配对比"
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(deformation_levels))
    width = 0.35

    ax.bar(x - width/2, deepsc_match_counts, width, label='DeepSC', alpha=0.8)
    ax.bar(x + width/2, threedsc_match_counts, width, label='3DSC', alpha=0.8)

    ax.set_xlabel('Deformation Strength')
    ax.set_ylabel('Number of Matches')
    ax.set_title('DeepSC vs 3DSC: Deformation Robustness Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in deformation_levels])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 70)
    print("DeepSC 优势总结:")
    print("  1. 学习的特征表示对非刚性形变更鲁棒")
    print("  2. 多尺度特征聚合捕获不同层次的几何信息")
    print("  3. 自监督学习适应特定任务的特征空间")
    print("  4. 端到端可训练，可针对具体应用优化")
    print("=" * 70)


if __name__ == "__main__":
    main()

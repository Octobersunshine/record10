import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from threedsc import ThreeDSC


def generate_sphere(n_points: int = 500, radius: float = 1.0) -> np.ndarray:
    phi = np.random.uniform(0, np.pi, n_points)
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.column_stack([x, y, z])


def generate_cube(n_points: int = 500, size: float = 1.0) -> np.ndarray:
    points = []
    for _ in range(n_points):
        face = np.random.randint(0, 6)
        if face == 0:
            x, y, z = np.random.uniform(-size, size), np.random.uniform(-size, size), size
        elif face == 1:
            x, y, z = np.random.uniform(-size, size), np.random.uniform(-size, size), -size
        elif face == 2:
            x, y, z = np.random.uniform(-size, size), size, np.random.uniform(-size, size)
        elif face == 3:
            x, y, z = np.random.uniform(-size, size), -size, np.random.uniform(-size, size)
        elif face == 4:
            x, y, z = size, np.random.uniform(-size, size), np.random.uniform(-size, size)
        else:
            x, y, z = -size, np.random.uniform(-size, size), np.random.uniform(-size, size)
        points.append([x, y, z])
    return np.array(points)


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


def visualize_point_clouds(pc1: np.ndarray, pc2: np.ndarray, matches: np.ndarray = None, title: str = ""):
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    ax1.scatter(pc1[:, 0], pc1[:, 1], pc1[:, 2], c='b', s=10, alpha=0.6)
    ax1.set_title('Point Cloud 1')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    ax2.scatter(pc2[:, 0], pc2[:, 1], pc2[:, 2], c='r', s=10, alpha=0.6)
    ax2.set_title('Point Cloud 2')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    if matches is not None and len(matches) > 0:
        for idx1, idx2 in matches[:20]:
            color = np.random.rand(3)
            ax1.scatter(pc1[idx1, 0], pc1[idx1, 1], pc1[idx1, 2], c=color, s=50, marker='o')
            ax2.scatter(pc2[idx2, 0], pc2[idx2, 1], pc2[idx2, 2], c=color, s=50, marker='o')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def main():
    print("=" * 60)
    print("3D Shape Context (3DSC) 描述子示例")
    print("=" * 60)

    np.random.seed(42)

    print("\n1. 生成测试点云...")
    sphere1 = generate_sphere(n_points=300, radius=1.0)
    sphere2 = generate_sphere(n_points=350, radius=1.0)
    sphere2_rotated = rotate_point_cloud(sphere2, 0.3, 0.5, 0.2) + np.array([0.5, 0.3, 0.1])
    cube = generate_cube(n_points=300, size=0.8)

    print(f"   球体1点数: {sphere1.shape[0]}")
    print(f"   球体2点数: {sphere2_rotated.shape[0]}")
    print(f"   立方体点数: {cube.shape[0]}")

    print("\n2. 初始化3DSC描述子...")
    descriptor = ThreeDSC(
        radius=0.5,
        num_bins_r=5,
        num_bins_azim=6,
        num_bins_polar=3,
        use_log=True
    )
    print(f"   描述子维度: {descriptor.descriptor_size}")

    print("\n3. 计算3DSC描述子...")
    print("   计算球体1描述子...")
    desc_sphere1 = descriptor.compute_descriptor(sphere1)
    print(f"   球体1描述子形状: {desc_sphere1.shape}")

    print("   计算旋转球体描述子...")
    desc_sphere2 = descriptor.compute_descriptor(sphere2_rotated)
    print(f"   旋转球体描述子形状: {desc_sphere2.shape}")

    print("   计算立方体描述子...")
    desc_cube = descriptor.compute_descriptor(cube)
    print(f"   立方体描述子形状: {desc_cube.shape}")

    print("\n4. 进行形状匹配...")
    print("   匹配: 球体1 vs 旋转球体 (同类)...")
    matches_sphere, dists_sphere = descriptor.match_descriptors(desc_sphere1, desc_sphere2)
    print(f"   找到匹配点对数量: {len(matches_sphere)}")
    if len(dists_sphere) > 0:
        print(f"   平均匹配距离: {np.mean(dists_sphere):.4f}")

    print("\n   匹配: 球体1 vs 立方体 (异类)...")
    matches_cross, dists_cross = descriptor.match_descriptors(desc_sphere1, desc_cube)
    print(f"   找到匹配点对数量: {len(matches_cross)}")
    if len(dists_cross) > 0:
        print(f"   平均匹配距离: {np.mean(dists_cross):.4f}")

    print("\n5. 匹配结果分析:")
    sphere_similarity = len(matches_sphere) / min(sphere1.shape[0], sphere2_rotated.shape[0])
    cross_similarity = len(matches_cross) / min(sphere1.shape[0], cube.shape[0])
    print(f"   同类匹配率: {sphere_similarity:.4f}")
    print(f"   异类匹配率: {cross_similarity:.4f}")
    print(f"   区分度: {sphere_similarity / (cross_similarity + 1e-10):.2f}x")

    print("\n6. 旋转不变性验证...")
    print("   测试不同旋转角度下描述子的一致性...")
    correlations = []
    angles = [np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2]
    for angle in angles:
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        sphere_rot = rotate_point_cloud(sphere1, angle, 0, 0)
        desc_rot = descriptor.compute_descriptor(sphere_rot)
        corr = np.corrcoef(np.mean(desc_sphere1, axis=0), np.mean(desc_rot, axis=0))[0, 1]
        correlations.append(corr)
        print(f"   旋转 {np.degrees(angle):.1f}°, 描述子相关性: {corr:.4f}")
    print(f"   平均相关性: {np.mean(correlations):.4f}")
    print(f"   最小相关性: {np.min(correlations):.4f}")

    print("\n7. 可视化结果...")
    visualize_point_clouds(
        sphere1, sphere2_rotated, matches_sphere,
        title="3DSC 匹配: 球体 vs 旋转平移球体 (前20对匹配点)"
    )

    print("\n" + "=" * 60)
    print("示例完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

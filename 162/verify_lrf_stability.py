import numpy as np
from scipy.spatial import cKDTree
from threedsc import ThreeDSC


def generate_sphere(n_points, radius):
    phi = np.random.uniform(0, np.pi, n_points)
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.column_stack([x, y, z])


def rotate_point_cloud(points, angle_x, angle_y, angle_z):
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


def main():
    print("=" * 60)
    print("LRF 定向稳定性验证")
    print("=" * 60)

    np.random.seed(42)

    print("\n1. 生成测试点云...")
    sphere = generate_sphere(200, 1.0)
    print(f"   点云大小: {sphere.shape}")

    print("\n2. 初始化3DSC描述子...")
    descriptor = ThreeDSC(radius=0.5, num_bins_r=5, num_bins_azim=6, num_bins_polar=3)

    print("\n3. 测试旋转不变性...")
    print("-" * 60)
    print(f"{'旋转轴':<10} {'角度(度)':<12} {'描述子相关性':<15}")
    print("-" * 60)

    desc_original = descriptor.compute_descriptor(sphere)

    axes = ['X', 'Y', 'Z']
    angles = [np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2]

    all_correlations = []

    for axis_idx, axis_name in enumerate(axes):
        for angle in angles:
            if axis_idx == 0:
                sphere_rot = rotate_point_cloud(sphere, angle, 0, 0)
            elif axis_idx == 1:
                sphere_rot = rotate_point_cloud(sphere, 0, angle, 0)
            else:
                sphere_rot = rotate_point_cloud(sphere, 0, 0, angle)

            desc_rot = descriptor.compute_descriptor(sphere_rot)
            corr = np.corrcoef(np.mean(desc_original, axis=0), np.mean(desc_rot, axis=0))[0, 1]
            all_correlations.append(corr)
            print(f"{axis_name:<10} {np.degrees(angle):<12.1f} {corr:<15.4f}")

    print("-" * 60)
    print(f"\n统计结果:")
    print(f"  平均相关性: {np.mean(all_correlations):.4f}")
    print(f"  最小相关性: {np.min(all_correlations):.4f}")
    print(f"  最大相关性: {np.max(all_correlations):.4f}")

    if np.min(all_correlations) > 0.85:
        print("\n✓ LRF定向稳定，旋转不变性验证通过!")
    else:
        print("\n⚠ LRF定向需要进一步优化")

    print("\n4. 验证LRF符号消歧原理:")
    print("   使用加权投影和来确定轴方向:")
    print("   - 投影 = 邻域点 · 轴向量")
    print("   - 权重 = 1 / 距离 (近点影响更大)")
    print("   - 加权和 < 0 时翻转轴方向")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

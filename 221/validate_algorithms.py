import numpy as np
import matplotlib.pyplot as plt
from wetland_hydrology import (
    depression_filling_simple, d8_flow_direction,
    d8_flow_accumulation, identify_wetland_connectivity
)


def test_depression_filling():
    """测试填洼算法"""
    print("测试1: 填洼算法验证")
    print("-" * 50)

    dem = np.array([
        [10, 10, 10, 10, 10],
        [10, 5, 8, 5, 10],
        [10, 8, 3, 8, 10],
        [10, 5, 8, 5, 10],
        [10, 10, 10, 10, 10]
    ], dtype=np.float64)

    print("原始DEM:")
    print(dem)
    print()

    filled = depression_filling_simple(dem, epsilon=0.001)
    print("填洼后DEM:")
    print(filled)
    print()

    fill_depth = filled - dem
    print("填洼深度:")
    print(fill_depth)
    print()

    print(f"中心点填洼深度: {fill_depth[2, 2]:.4f}")
    print(f"算法正确性: {'✓ 通过' if filled[2, 2] > dem[2, 2] else '✗ 失败'}")
    print()


def test_d8_flow_direction():
    """测试D8流向算法"""
    print("测试2: D8流向算法验证")
    print("-" * 50)

    dem = np.array([
        [100, 100, 100, 100, 100],
        [100, 20, 15, 12, 100],
        [100, 25, 10, 8, 100],
        [100, 30, 18, 14, 100],
        [100, 100, 100, 100, 100]
    ], dtype=np.float64)

    print("测试DEM:")
    print(dem)
    print()

    direction = d8_flow_direction(dem)
    print("流向编码:")
    print(direction)
    print()

    dir_names = {
        0: "无", 1: "东→", 2: "东南↘", 4: "南↓",
        8: "西南↙", 16: "西←", 32: "西北↖",
        64: "北↑", 128: "东北↗"
    }

    print("流向解读:")
    for i in range(1, 4):
        row_str = []
        for j in range(1, 4):
            row_str.append(f"{dir_names.get(direction[i, j], '?'):>6}")
        print(f"  {i}: {' '.join(row_str)}")
    print()

    print(f"算法正确性: {'✓ 通过' if direction[2, 2] == 1 else '✗ 失败'}")
    print()


def test_flow_accumulation():
    """测试流量累积算法"""
    print("测试3: 流量累积算法验证")
    print("-" * 50)

    slope_dem = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            slope_dem[i, j] = 100 - i * 5 - j * 2

    print("斜坡DEM (西高东低，北高南低):")
    print(slope_dem)
    print()

    direction = d8_flow_direction(slope_dem)
    print("流向矩阵:")
    print(direction)
    print()

    accumulation = d8_flow_accumulation(direction)
    print("流量累积矩阵:")
    print(accumulation)
    print()

    max_accum = np.max(accumulation)
    max_pos = np.unravel_index(np.argmax(accumulation), accumulation.shape)
    print(f"最大流量累积: {max_accum} 位于位置 {max_pos}")
    print(f"算法正确性: {'✓ 通过' if max_accum >= 8 else '✗ 失败'}")
    print()


def test_connectivity_analysis():
    """测试连通性分析"""
    print("测试4: 连通性分析验证")
    print("-" * 50)

    np.random.seed(123)

    dem = np.zeros((30, 30))
    x, y = np.meshgrid(np.arange(30), np.arange(30))

    for cx, cy, r in [(10, 10, 5), (20, 15, 6), (15, 22, 4)]:
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        dem[dist < r] -= 5

    dem += 0.1 * x + 0.05 * y
    dem += np.random.normal(0, 0.2, dem.shape)
    dem = dem - dem.min() + 1

    filled = depression_filling_simple(dem)
    direction = d8_flow_direction(filled)
    accumulation = d8_flow_accumulation(direction)

    labeled, conn_score, num_regions = identify_wetland_connectivity(
        filled, direction, accumulation, dem,
        wetland_threshold=0.1, accumulation_threshold=10
    )

    print(f"识别到的连通区域数量: {num_regions}")
    print(f"最大连通性指数: {np.max(conn_score):.2f}")
    print(f"算法正确性: {'✓ 通过' if num_regions >= 2 else '✗ 失败'}")
    print()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    im1 = axes[0].imshow(dem, cmap='terrain')
    axes[0].set_title('原始DEM')
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(labeled, cmap='tab20')
    axes[1].set_title('连通区域')
    plt.colorbar(im2, ax=axes[1])

    im3 = axes[2].imshow(conn_score, cmap='RdYlBu_r')
    axes[2].set_title('连通性指数')
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    plt.savefig('validation_connectivity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("验证结果图已保存至: validation_connectivity.png")
    print()


def test_synthetic_wetland():
    """测试合成湿地场景"""
    print("测试5: 合成湿地水文连通性分析")
    print("-" * 50)

    size = 50
    np.random.seed(456)

    x, y = np.meshgrid(np.arange(size), np.arange(size))
    dem = np.zeros((size, size))

    cx1, cy1, r1 = 15, 15, 10
    cx2, cy2, r2 = 35, 30, 12

    dist1 = np.sqrt((x - cx1)**2 + (y - cy1)**2)
    dist2 = np.sqrt((x - cx2)**2 + (y - cy2)**2)

    channel_width = 3
    channel_path = np.abs((y - cy1) * (cx2 - cx1) - (x - cx1) * (cy2 - cy1)) / np.sqrt((cx2-cx1)**2 + (cy2-cy1)**2)
    channel_mask = (channel_path < channel_width) & (np.minimum(dist1, dist2) < np.max([r1, r2]) + 5)

    dem[dist1 < r1] -= 8
    dem[dist2 < r2] -= 6
    dem[channel_mask] -= 3

    dem += 0.05 * x + 0.03 * y
    dem += np.random.normal(0, 0.3, dem.shape)
    dem = dem - dem.min() + 2

    print("执行完整分析流程...")
    filled = depression_filling_simple(dem)
    direction = d8_flow_direction(filled)
    accumulation = d8_flow_accumulation(direction)

    labeled, conn_score, num_regions = identify_wetland_connectivity(
        filled, direction, accumulation, dem,
        wetland_threshold=0.3, accumulation_threshold=20
    )

    print(f"湿地数量: {num_regions}")
    print(f"最大连通性指数: {np.max(conn_score):.2f}")

    region_sizes = np.bincount(labeled.ravel())[1:] if num_regions > 0 else []
    if len(region_sizes) > 0:
        print(f"区域大小: {sorted(region_sizes, reverse=True)[:5]}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    im1 = axes[0, 0].imshow(dem, cmap='terrain')
    axes[0, 0].set_title('合成湿地DEM')
    plt.colorbar(im1, ax=axes[0, 0])

    im2 = axes[0, 1].imshow(filled - dem, cmap='Blues')
    axes[0, 1].set_title('填洼深度')
    plt.colorbar(im2, ax=axes[0, 1])

    im3 = axes[0, 2].imshow(direction, cmap='tab10')
    axes[0, 2].set_title('D8流向')
    plt.colorbar(im3, ax=axes[0, 2])

    im4 = axes[1, 0].imshow(np.log1p(accumulation), cmap='YlGnBu')
    axes[1, 0].set_title('流量累积')
    plt.colorbar(im4, ax=axes[1, 0])

    im5 = axes[1, 1].imshow(labeled, cmap='tab20')
    axes[1, 1].set_title('连通区域')
    plt.colorbar(im5, ax=axes[1, 1])

    im6 = axes[1, 2].imshow(conn_score, cmap='RdYlBu_r')
    axes[1, 2].set_title('连通性指数')
    plt.colorbar(im6, ax=axes[1, 2])

    plt.tight_layout()
    plt.savefig('synthetic_wetland_analysis.png', dpi=200, bbox_inches='tight')
    plt.close()

    print("合成湿地分析图已保存至: synthetic_wetland_analysis.png")
    print()

    print(f"算法正确性: {'✓ 通过' if num_regions >= 2 else '✗ 失败'}")
    print()


def main():
    print("=" * 60)
    print("算法正确性验证测试")
    print("=" * 60)
    print()

    test_depression_filling()
    test_d8_flow_direction()
    test_flow_accumulation()
    test_connectivity_analysis()
    test_synthetic_wetland()

    print("=" * 60)
    print("所有测试完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()

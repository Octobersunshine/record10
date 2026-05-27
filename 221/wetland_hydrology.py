import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from collections import deque
from scipy.ndimage import label, generate_binary_structure
import warnings
warnings.filterwarnings('ignore')


def generate_demo_dem(size=100, wetland_center=None):
    """生成模拟湿地DEM数据"""
    x = np.linspace(0, size, size)
    y = np.linspace(0, size, size)
    X, Y = np.meshgrid(x, y)

    dem = np.zeros_like(X)

    dem += 0.5 * np.sin(X / 15) * np.cos(Y / 12)

    for _ in range(8):
        cx, cy = np.random.randint(10, size-10, 2)
        r = np.random.randint(8, 25)
        depression = np.maximum(0, r - np.sqrt((X - cx)**2 + (Y - cy)**2))
        dem -= depression * 0.4

    if wetland_center is None:
        wetland_center = (size // 2, size // 2)

    wetland_r = 30
    wetland_depression = np.maximum(0, wetland_r - np.sqrt(
        (X - wetland_center[0])**2 + (Y - wetland_center[1])**2
    ))
    dem -= wetland_depression * 0.8

    dem += 0.02 * X + 0.015 * Y

    noise = np.random.normal(0, 0.1, dem.shape)
    dem += noise

    dem = dem - dem.min() + 1.0

    return dem, X, Y


def depression_filling(dem, nodata=None):
    """
    优先溢出算法实现DEM填洼
    """
    dem = dem.astype(np.float64).copy()
    if nodata is None:
        nodata = -9999

    rows, cols = dem.shape
    filled = dem.copy()

    in_queue = np.zeros((rows, cols), dtype=bool)
    is_processed = np.zeros((rows, cols), dtype=bool)

    heap = []
    neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),          (0, 1),
                      (1, -1),  (1, 0), (1, 1)]

    for i in range(rows):
        for j in range(cols):
            if i == 0 or i == rows - 1 or j == 0 or j == cols - 1:
                if dem[i, j] != nodata:
                    heapq.heappush(heap, (dem[i, j], i, j))
                    in_queue[i, j] = True

    while heap:
        current_elev, i, j = heapq.heappop(heap)
        is_processed[i, j] = True

        for di, dj in neighbor_offsets:
            ni, nj = i + di, j + dj
            if 0 <= ni < rows and 0 <= nj < cols:
                if not in_queue[ni, nj] and dem[ni, nj] != nodata:
                    spill_elev = max(dem[ni, nj], current_elev)
                    if spill_elev < filled[ni, nj]:
                        filled[ni, nj] = spill_elev
                    if not in_queue[ni, nj]:
                        heapq.heappush(heap, (spill_elev, ni, nj))
                        in_queue[ni, nj] = True

    return filled


def depression_filling_simple(dem, epsilon=0.001):
    """
    改进版填洼算法 - 优先溢出算法
    """
    import heapq
    dem = dem.astype(np.float64).copy()
    rows, cols = dem.shape

    filled = dem.copy()
    in_queue = np.zeros((rows, cols), dtype=bool)
    is_border = np.zeros((rows, cols), dtype=bool)

    heap = []
    neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1),
                       (0, -1),          (0, 1),
                       (1, -1),  (1, 0), (1, 1)]

    for i in range(rows):
        for j in range(cols):
            if i == 0 or i == rows - 1 or j == 0 or j == cols - 1:
                is_border[i, j] = True
                heapq.heappush(heap, (dem[i, j], i, j))
                in_queue[i, j] = True

    while heap:
        current_elev, i, j = heapq.heappop(heap)

        for di, dj in neighbor_offsets:
            ni, nj = i + di, j + dj
            if 0 <= ni < rows and 0 <= nj < cols:
                if not in_queue[ni, nj] and not is_border[ni, nj]:
                    spill_elev = max(dem[ni, nj], current_elev + epsilon)
                    if spill_elev < filled[ni, nj] or not in_queue[ni, nj]:
                        filled[ni, nj] = spill_elev
                        heapq.heappush(heap, (spill_elev, ni, nj))
                        in_queue[ni, nj] = True

    return filled


def d8_flow_direction(filled_dem):
    """
    D8算法计算流向
    """
    rows, cols = filled_dem.shape
    direction = np.zeros((rows, cols), dtype=np.int16)

    d8_offsets = [
        (0, 1),    # 1 - 东
        (1, 1),    # 2 - 东南
        (1, 0),    # 4 - 南
        (1, -1),   # 8 - 西南
        (0, -1),    # 16 - 西
        (-1, -1),   # 32 - 西北
        (-1, 0),    # 64 - 北
        (-1, 1),    # 128 - 东北
    ]
    d8_codes = [1, 2, 4, 8, 16, 32, 64, 128]

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            max_slope = 0
            best_dir = 0
            current_elev = filled_dem[i, j]

            for idx, (di, dj) in enumerate(d8_offsets):
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    neighbor_elev = filled_dem[ni, nj]
                    distance = np.sqrt(di**2 + dj**2)
                    slope = (current_elev - neighbor_elev) / distance

                    if slope > max_slope and slope > 1e-10:
                        max_slope = slope
                        best_dir = d8_codes[idx]

            direction[i, j] = best_dir

    return direction


def d8_flow_accumulation(direction):
    """
    计算流量累积"""
    rows, cols = direction.shape
    accumulation = np.ones((rows, cols), dtype=np.float64)

    d8_offsets = {
        1: (0, 1),
        2: (1, 1),
        4: (1, 0),
        8: (1, -1),
        16: (0, -1),
        32: (-1, -1),
        64: (-1, 0),
        128: (-1, 1),
    }

    in_degree = np.zeros((rows, cols), dtype=np.int32)

    for i in range(rows):
        for j in range(cols):
            dir_code = direction[i, j]
            if dir_code in d8_offsets:
                di, dj = d8_offsets[dir_code]
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    in_degree[ni, nj] += 1

    queue = deque()
    for i in range(rows):
        for j in range(cols):
            if in_degree[i, j] == 0:
                queue.append((i, j))

    while queue:
        i, j = queue.popleft()
        dir_code = direction[i, j]
        if dir_code in d8_offsets:
            di, dj = d8_offsets[dir_code]
            ni, nj = i + di, j + dj
            if 0 <= ni < rows and 0 <= nj < cols:
                accumulation[ni, nj] += accumulation[i, j]
                in_degree[ni, nj] -= 1
                if in_degree[ni, nj] == 0:
                    queue.append((ni, nj))

    return accumulation


def identify_wetland_connectivity(filled_dem, direction, accumulation,
                              dem_original, wetland_threshold=0.3,
                               accumulation_threshold=50):
    """
    识别湿地连通区域"""
    rows, cols = filled_dem.shape

    dep_depth = dem_original - filled_dem
    wetland_mask = dep_depth > wetland_threshold

    high_accum = accumulation > accumulation_threshold
    wetland_mask = wetland_mask | high_accum

    structure = generate_binary_structure(2, 2)
    labeled_regions, num_regions = label(wetland_mask, structure=structure)

    region_sizes = np.bincount(labeled_regions.ravel())
    region_sizes = region_sizes[1:]

    connectivity_map = np.zeros((rows, cols), dtype=np.int32)

    d8_offsets = {
        1: (0, 1),
        2: (1, 1),
        4: (1, 0),
        8: (1, -1),
        16: (0, -1),
        32: (-1, -1),
        64: (-1, 0),
        128: (-1, 1),
    }

    region_connectivity = {}

    for region_id in range(1, num_regions + 1):
        region_mask = labeled_regions == region_id
        downstream_regions = set()

        for i in range(rows):
            for j in range(cols):
                if region_mask[i, j]:
                    dir_code = direction[i, j]
                    if dir_code in d8_offsets:
                        di, dj = d8_offsets[dir_code]
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols:
                            downstream_id = labeled_regions[ni, nj]
                            if downstream_id > 0 and downstream_id != region_id:
                                downstream_regions.add(downstream_id)

        region_connectivity[region_id] = list(downstream_regions)

    connectivity_score = np.zeros((rows, cols), dtype=np.float64)
    for region_id in range(1, num_regions + 1):
        region_mask = labeled_regions == region_id
        size = np.sum(region_mask)
        downstream_count = len(region_connectivity.get(region_id))
        score = size * (1 + downstream_count * 0.5)
        connectivity_score[region_mask] = score

    return labeled_regions, connectivity_score, num_regions


def visualize_results(dem_original, filled_dem, direction,
                    accumulation, labeled_regions, connectivity_score):
    """
    可视化分析结果"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    im1 = axes[0, 0].imshow(dem_original, cmap='terrain')
    axes[0, 0].set_title('原始DEM')
    plt.colorbar(im1, ax=axes[0, 0])

    dem_diff = filled_dem - dem_original
    im2 = axes[0, 1].imshow(dem_diff, cmap='Blues')
    axes[0, 1].set_title('填洼深度')
    plt.colorbar(im2, ax=axes[0, 1])

    dir_cmap = colors.ListedColormap([
        'white', 'red', 'orange', 'yellow',
        'green', 'cyan', 'blue', 'purple', 'pink'
    ])
    bounds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    dir_norm = colors.BoundaryNorm(bounds, dir_cmap.N)
    im3 = axes[0, 2].imshow(direction, cmap=dir_cmap, norm=dir_norm)
    axes[0, 2].set_title('D8流向')
    cbar3 = plt.colorbar(im3, ax=axes[0, 2], ticks=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])
    cbar3.ax.set_yticklabels(['无', '东', '东南', '南', '西南', '西', '西北', '北', '东北'])

    im4 = axes[1, 0].imshow(np.log1p(accumulation), cmap='YlGnBu')
    axes[1, 0].set_title('流量累积 (对数尺度)')
    plt.colorbar(im4, ax=axes[1, 0])

    region_cmap = plt.cm.get_cmap('tab20')
    im5 = axes[1, 1].imshow(labeled_regions, cmap=region_cmap)
    axes[1, 1].set_title('连通区域编号')
    plt.colorbar(im5, ax=axes[1, 1])

    im6 = axes[1, 2].imshow(connectivity_score, cmap='RdYlBu_r')
    axes[1, 2].set_title('连通性指数')
    plt.colorbar(im6, ax=axes[1, 2])

    plt.tight_layout()
    plt.savefig('wetland_connectivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    return fig


def main():
    """
    主函数：执行完整的湿地水文连通性分析流程"""
    print("=" * 60)
    print("湿地水文连通性分析工具")
    print("=" * 60)
    print()

    np.random.seed(42)

    print("1. 生成模拟DEM数据...")
    dem_original, X, Y = generate_demo_dem(size=100)
    print(f"   DEM尺寸: {dem_original.shape}")
    print(f"   高程范围: {dem_original.min():.2f} - {dem_original.max():.2f}")
    print()

    print("2. 执行DEM填洼处理...")
    filled_dem = depression_filling_simple(dem_original)
    fill_depth = filled_dem - dem_original
    print(f"   最大填洼深度: {fill_depth.max():.4f}")
    print(f"   平均填洼深度: {fill_depth.mean():.4f}")
    print()

    print("3. 计算D8流向...")
    direction = d8_flow_direction(filled_dem)
    unique_dirs = np.unique(direction)
    print(f"   流向编码: {unique_dirs}")
    print()

    print("4. 计算流量累积...")
    accumulation = d8_flow_accumulation(direction)
    print(f"   最大流量累积: {accumulation.max():.0f}")
    print(f"   平均流量累积: {accumulation.mean():.2f}")
    print()

    print("5. 识别湿地连通区域...")
    labeled_regions, connectivity_score, num_regions = identify_wetland_connectivity(
        filled_dem, direction, accumulation, dem_original,
        wetland_threshold=0.2, accumulation_threshold=30
    )
    print(f"   识别到的连通区域数量: {num_regions}")
    if num_regions > 0:
        region_sizes = np.bincount(labeled_regions.ravel())[1:]
        print(f"   区域大小范围: {region_sizes.min()} - {region_sizes.max()} 像素")
    else:
        print("   警告: 未识别到连通区域，请调整阈值参数")
    print()

    print("6. 生成可视化结果...")
    fig = visualize_results(
        dem_original, filled_dem, direction,
        accumulation, labeled_regions, connectivity_score
    )
    print()

    print("=" * 60)
    print("分析完成!")
    print("结果已保存至: wetland_connectivity_analysis.png")
    print("=" * 60)

    return {
        'dem_original': dem_original,
        'filled_dem': filled_dem,
        'direction': direction,
        'accumulation': accumulation,
        'labeled_regions': labeled_regions,
        'connectivity_score': connectivity_score,
        'num_regions': num_regions
    }


if __name__ == '__main__':
    import heapq
    results = main()

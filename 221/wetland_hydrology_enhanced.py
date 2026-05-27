import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from collections import deque
from scipy.ndimage import label, generate_binary_structure, sobel, gaussian_filter
import heapq
import warnings
warnings.filterwarnings('ignore')


def generate_demo_dem(size=100, wetland_center=None, seed=42):
    """生成模拟湿地DEM数据，包含平坦区域"""
    np.random.seed(seed)
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

    flat_region = (X > 60) & (X < 85) & (Y > 20) & (Y < 45)
    dem[flat_region] = np.mean(dem[flat_region])

    dem += 0.02 * X + 0.015 * Y

    noise = np.random.normal(0, 0.05, dem.shape)
    dem += noise

    dem = dem - dem.min() + 1.0

    return dem, X, Y


def generate_flat_test_dem(size=50):
    """生成用于测试平坦区域的DEM"""
    x = np.linspace(0, size, size)
    y = np.linspace(0, size, size)
    X, Y = np.meshgrid(x, y)

    dem = np.zeros_like(X)

    dem[:, :20] = 10 - 0.1 * X[:, :20]
    dem[:, 20:30] = 8.0
    dem[:, 30:] = 8 - 0.15 * (X[:, 30:] - 30)

    dem += np.random.normal(0, 0.001, dem.shape)

    return dem, X, Y


def depression_filling_simple(dem, epsilon=0.001):
    """
    改进版填洼算法 - 优先溢出算法
    """
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
    传统D8算法计算流向（用于对比）
    """
    rows, cols = filled_dem.shape
    direction = np.zeros((rows, cols), dtype=np.int16)

    d8_offsets = [
        (0, 1),    # 1 - 东
        (1, 1),    # 2 - 东南
        (1, 0),    # 4 - 南
        (1, -1),   # 8 - 西南
        (0, -1),   # 16 - 西
        (-1, -1),  # 32 - 西北
        (-1, 0),   # 64 - 北
        (-1, 1),   # 128 - 东北
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


def calculate_curvature(dem, cell_size=1.0):
    """
    计算地表曲率（平面曲率和剖面曲率）
    使用二次曲面拟合方法
    """
    rows, cols = dem.shape

    Z = gaussian_filter(dem, sigma=1.0)

    dz_dx = sobel(Z, axis=1) / (8 * cell_size)
    dz_dy = sobel(Z, axis=0) / (8 * cell_size)

    d2z_dx2 = sobel(dz_dx, axis=1) / (8 * cell_size)
    d2z_dy2 = sobel(dz_dy, axis=0) / (8 * cell_size)
    d2z_dxdy = sobel(dz_dx, axis=0) / (8 * cell_size)

    p = dz_dx
    q = dz_dy
    r = d2z_dx2
    s = d2z_dxdy
    t = d2z_dy2

    denom_plan = (p**2 + q**2) * (p**2 + q**2)**0.5
    denom_prof = (p**2 + q**2) * (1 + p**2 + q**2)**1.5

    with np.errstate(divide='ignore', invalid='ignore'):
        plan_curvature = np.where(
            denom_plan > 1e-10,
            -(q**2 * r - 2 * p * q * s + p**2 * t) / denom_plan,
            0
        )
        prof_curvature = np.where(
            denom_prof > 1e-10,
            -(p**2 * r + 2 * p * q * s + q**2 * t) / denom_prof,
            0
        )

    slope = np.sqrt(p**2 + q**2)

    return plan_curvature, prof_curvature, slope


def fd8_flow_direction(filled_dem, exponent=1.0, slope_threshold=1e-6):
    """
    FD8 (Flow Direction 8) 概率多流向算法
    将流量按坡度比例分配到多个下坡方向
    """
    rows, cols = filled_dem.shape

    d8_offsets = [
        (0, 1),    # 1 - 东
        (1, 1),    # 2 - 东南
        (1, 0),    # 4 - 南
        (1, -1),   # 8 - 西南
        (0, -1),   # 16 - 西
        (-1, -1),  # 32 - 西北
        (-1, 0),   # 64 - 北
        (-1, 1),   # 128 - 东北
    ]
    d8_codes = [1, 2, 4, 8, 16, 32, 64, 128]

    flow_weights = np.zeros((rows, cols, 8), dtype=np.float64)
    direction = np.zeros((rows, cols), dtype=np.int16)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            current_elev = filled_dem[i, j]
            slopes = []
            valid_indices = []

            for idx, (di, dj) in enumerate(d8_offsets):
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    neighbor_elev = filled_dem[ni, nj]
                    distance = np.sqrt(di**2 + dj**2)
                    slope = (current_elev - neighbor_elev) / distance

                    if slope > slope_threshold:
                        slopes.append(slope)
                        valid_indices.append(idx)

            if len(slopes) > 0:
                slopes = np.array(slopes)
                weights = slopes ** exponent
                weights = weights / weights.sum()

                max_idx = valid_indices[np.argmax(weights)]
                direction[i, j] = d8_codes[max_idx]

                for k, idx in enumerate(valid_indices):
                    flow_weights[i, j, idx] = weights[k]
            else:
                direction[i, j] = 0

    return direction, flow_weights


def curvature_based_flow_direction(filled_dem, cell_size=1.0, slope_threshold=1e-6):
    """
    基于地表曲率的流向算法
    在平坦区域利用曲率信息确定流向，避免发散
    """
    rows, cols = filled_dem.shape

    plan_curvature, prof_curvature, slope = calculate_curvature(filled_dem, cell_size)

    d8_offsets = [
        (0, 1),    # 1 - 东
        (1, 1),    # 2 - 东南
        (1, 0),    # 4 - 南
        (1, -1),   # 8 - 西南
        (0, -1),   # 16 - 西
        (-1, -1),  # 32 - 西北
        (-1, 0),   # 64 - 北
        (-1, 1),   # 128 - 东北
    ]
    d8_codes = [1, 2, 4, 8, 16, 32, 64, 128]
    angles = [0, 45, 90, 135, 180, 225, 270, 315]

    direction = np.zeros((rows, cols), dtype=np.int16)
    flow_weights = np.zeros((rows, cols, 8), dtype=np.float64)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            current_elev = filled_dem[i, j]
            slopes = []
            valid_indices = []

            for idx, (di, dj) in enumerate(d8_offsets):
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    neighbor_elev = filled_dem[ni, nj]
                    distance = np.sqrt(di**2 + dj**2)
                    slope_val = (current_elev - neighbor_elev) / distance

                    if slope_val > slope_threshold:
                        slopes.append(slope_val)
                        valid_indices.append(idx)

            if len(slopes) > 0:
                slopes = np.array(slopes)
                weights = slopes.copy()

                if len(slopes) > 1 and np.max(slopes) - np.min(slopes) < 0.01:
                    pc = plan_curvature[i, j]
                    aspect = np.arctan2(-sobel(filled_dem, axis=0)[i, j],
                                       sobel(filled_dem, axis=1)[i, j])
                    aspect_deg = np.degrees(aspect) % 360

                    curvature_weights = []
                    for idx in valid_indices:
                        angle_diff = np.abs(angles[idx] - aspect_deg)
                        angle_diff = min(angle_diff, 360 - angle_diff)

                        if pc > 0:
                            cw = np.exp(-angle_diff / 30)
                        elif pc < 0:
                            cw = np.exp(angle_diff / 30)
                        else:
                            cw = 1.0

                        curvature_weights.append(cw)

                    curvature_weights = np.array(curvature_weights)
                    weights = weights * curvature_weights

                weights = weights / weights.sum()

                max_idx = valid_indices[np.argmax(weights)]
                direction[i, j] = d8_codes[max_idx]

                for k, idx in enumerate(valid_indices):
                    flow_weights[i, j, idx] = weights[k]
            else:
                pc = plan_curvature[i, j]
                if abs(pc) > 1e-8:
                    aspect = np.arctan2(-sobel(filled_dem, axis=0)[i, j],
                                       sobel(filled_dem, axis=1)[i, j])
                    aspect_deg = np.degrees(aspect) % 360

                    angle_diffs = [abs(a - aspect_deg) for a in angles]
                    angle_diffs = [min(d, 360 - d) for d in angle_diffs]

                    if pc > 0:
                        best_idx = np.argmin(angle_diffs)
                    else:
                        best_idx = np.argmax(angle_diffs)

                    direction[i, j] = d8_codes[best_idx]
                    flow_weights[i, j, best_idx] = 1.0
                else:
                    direction[i, j] = 0

    return direction, flow_weights, plan_curvature, prof_curvature


def dinfinity_flow_direction(filled_dem, slope_threshold=1e-6):
    """
    D-Infinity (D∞) 无穷方向算法
    使用三角形面元计算连续的流向角度
    """
    rows, cols = filled_dem.shape

    direction_angle = np.zeros((rows, cols), dtype=np.float64)
    direction = np.zeros((rows, cols), dtype=np.int16)
    flow_weights = np.zeros((rows, cols, 8), dtype=np.float64)

    d8_offsets = [
        (0, 1),    # 1 - 东
        (1, 1),    # 2 - 东南
        (1, 0),    # 4 - 南
        (1, -1),   # 8 - 西南
        (0, -1),   # 16 - 西
        (-1, -1),  # 32 - 西北
        (-1, 0),   # 64 - 北
        (-1, 1),   # 128 - 东北
    ]
    d8_codes = [1, 2, 4, 8, 16, 32, 64, 128]
    angles = [0, 45, 90, 135, 180, 225, 270, 315]

    triangle_offsets = [
        [(0, 1), (1, 1), (1, 0)],
        [(1, 0), (1, -1), (0, -1)],
        [(0, -1), (-1, -1), (-1, 0)],
        [(-1, 0), (-1, 1), (0, 1)],
    ]
    triangle_angles = [
        [(0, 45), (45, 90), (0, 90)],
        [(90, 135), (135, 180), (90, 180)],
        [(180, 225), (225, 270), (180, 270)],
        [(270, 315), (315, 360), (270, 360)],
    ]

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            z0 = filled_dem[i, j]
            max_slope = -np.inf
            best_angle = 0
            best_triangle = None
            best_alpha = 0

            for tri_idx, tri in enumerate(triangle_offsets):
                (di1, dj1), (di2, dj2), (di3, dj3) = tri
                z1 = filled_dem[i + di1, j + dj1]
                z2 = filled_dem[i + di2, j + dj2]
                z3 = filled_dem[i + di3, j + dj3]

                angle_ranges = triangle_angles[tri_idx]

                for k, (a1, a2) in enumerate(angle_ranges):
                    if k == 0:
                        za, zb = z1, z2
                    elif k == 1:
                        za, zb = z2, z3
                    else:
                        za, zb = z1, z3

                    dx1, dy1 = tri[k][1], -tri[k][0]
                    dx2, dy2 = tri[(k+1) % 3][1], -tri[(k+1) % 3][0]

                    s1 = (z0 - za) / np.sqrt(dx1**2 + dy1**2)
                    s2 = (z0 - zb) / np.sqrt(dx2**2 + dy2**2)

                    if s1 > slope_threshold and s2 > slope_threshold:
                        alpha = (a2 - a1) * (s1 / (s1 + s2)) if (s1 + s2) > 0 else (a2 - a1) / 2
                        flow_angle = a1 + alpha
                        slope_mag = max(s1, s2)

                        if slope_mag > max_slope:
                            max_slope = slope_mag
                            best_angle = flow_angle
                            best_alpha = alpha
                            best_triangle = tri_idx

            if max_slope > slope_threshold:
                direction_angle[i, j] = best_angle % 360

                angle_deg = best_angle % 360
                for idx, a in enumerate(angles):
                    if abs(angle_deg - a) < 22.5 or abs(angle_deg - a - 360) < 22.5:
                        direction[i, j] = d8_codes[idx]
                        flow_weights[i, j, idx] = 1.0
                        break
            else:
                direction[i, j] = 0
                direction_angle[i, j] = np.nan

    return direction, direction_angle, flow_weights


def multi_flow_accumulation(flow_weights):
    """
    多流向流量累积计算
    """
    rows, cols, _ = flow_weights.shape
    accumulation = np.ones((rows, cols), dtype=np.float64)

    d8_offsets = [
        (0, 1),    # 1 - 东
        (1, 1),    # 2 - 东南
        (1, 0),    # 4 - 南
        (1, -1),   # 8 - 西南
        (0, -1),   # 16 - 西
        (-1, -1),  # 32 - 西北
        (-1, 0),   # 64 - 北
        (-1, 1),   # 128 - 东北
    ]

    in_degree = np.zeros((rows, cols), dtype=np.float64)

    for i in range(rows):
        for j in range(cols):
            for idx, (di, dj) in enumerate(d8_offsets):
                weight = flow_weights[i, j, idx]
                if weight > 0:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        in_degree[ni, nj] += weight

    queue = deque()
    for i in range(rows):
        for j in range(cols):
            if in_degree[i, j] == 0:
                queue.append((i, j))

    while queue:
        i, j = queue.popleft()

        for idx, (di, dj) in enumerate(d8_offsets):
            weight = flow_weights[i, j, idx]
            if weight > 0:
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    accumulation[ni, nj] += accumulation[i, j] * weight
                    in_degree[ni, nj] -= weight
                    if in_degree[ni, nj] < 1e-10:
                        queue.append((ni, nj))

    return accumulation


def d8_flow_accumulation(direction):
    """
    传统D8流量累积
    """
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
    识别湿地连通区域
    """
    rows, cols = filled_dem.shape

    dep_depth = dem_original - filled_dem
    wetland_mask = dep_depth > wetland_threshold

    high_accum = accumulation > accumulation_threshold
    wetland_mask = wetland_mask | high_accum

    structure = generate_binary_structure(2, 2)
    labeled_regions, num_regions = label(wetland_mask, structure=structure)

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
        downstream_count = len(region_connectivity.get(region_id, []))
        score = size * (1 + downstream_count * 0.5)
        connectivity_score[region_mask] = score

    return labeled_regions, connectivity_score, num_regions


def visualize_comparison(dem_original, filled_dem,
                        d8_dir, d8_accum,
                        fd8_dir, fd8_accum,
                        curv_dir, curv_accum, curv_weights,
                        dinf_dir, dinf_accum, dinf_angle,
                        plan_curvature, prof_curvature):
    """
    可视化多种算法的对比结果
    """
    fig, axes = plt.subplots(4, 4, figsize=(20, 18))

    im1 = axes[0, 0].imshow(dem_original, cmap='terrain')
    axes[0, 0].set_title('原始DEM')
    plt.colorbar(im1, ax=axes[0, 0])

    dem_diff = filled_dem - dem_original
    im2 = axes[0, 1].imshow(dem_diff, cmap='Blues')
    axes[0, 1].set_title('填洼深度')
    plt.colorbar(im2, ax=axes[0, 1])

    im3 = axes[0, 2].imshow(plan_curvature, cmap='RdBu_r', vmin=-0.1, vmax=0.1)
    axes[0, 2].set_title('平面曲率')
    plt.colorbar(im3, ax=axes[0, 2])

    im4 = axes[0, 3].imshow(prof_curvature, cmap='RdBu_r', vmin=-0.1, vmax=0.1)
    axes[0, 3].set_title('剖面曲率')
    plt.colorbar(im4, ax=axes[0, 3])

    dir_cmap = colors.ListedColormap([
        'white', 'red', 'orange', 'yellow',
        'green', 'cyan', 'blue', 'purple', 'pink'
    ])
    bounds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    dir_norm = colors.BoundaryNorm(bounds, dir_cmap.N)

    im5 = axes[1, 0].imshow(d8_dir, cmap=dir_cmap, norm=dir_norm)
    axes[1, 0].set_title('传统D8流向')
    cbar5 = plt.colorbar(im5, ax=axes[1, 0], ticks=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])
    cbar5.ax.set_yticklabels(['无', '东', '东南', '南', '西南', '西', '西北', '北', '东北'])

    im6 = axes[1, 1].imshow(fd8_dir, cmap=dir_cmap, norm=dir_norm)
    axes[1, 1].set_title('FD8多流向')
    cbar6 = plt.colorbar(im6, ax=axes[1, 1], ticks=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])
    cbar6.ax.set_yticklabels(['无', '东', '东南', '南', '西南', '西', '西北', '北', '东北'])

    im7 = axes[1, 2].imshow(curv_dir, cmap=dir_cmap, norm=dir_norm)
    axes[1, 2].set_title('曲率修正流向')
    cbar7 = plt.colorbar(im7, ax=axes[1, 2], ticks=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])
    cbar7.ax.set_yticklabels(['无', '东', '东南', '南', '西南', '西', '西北', '北', '东北'])

    im8 = axes[1, 3].imshow(dinf_angle, cmap='hsv', vmin=0, vmax=360)
    axes[1, 3].set_title('D∞流向角度')
    cbar8 = plt.colorbar(im8, ax=axes[1, 3])
    cbar8.set_label('角度 (°)')

    im9 = axes[2, 0].imshow(np.log1p(d8_accum), cmap='YlGnBu')
    axes[2, 0].set_title(f'D8流量累积 (max={d8_accum.max():.0f})')
    plt.colorbar(im9, ax=axes[2, 0])

    im10 = axes[2, 1].imshow(np.log1p(fd8_accum), cmap='YlGnBu')
    axes[2, 1].set_title(f'FD8流量累积 (max={fd8_accum.max():.0f})')
    plt.colorbar(im10, ax=axes[2, 1])

    im11 = axes[2, 2].imshow(np.log1p(curv_accum), cmap='YlGnBu')
    axes[2, 2].set_title(f'曲率修正累积 (max={curv_accum.max():.0f})')
    plt.colorbar(im11, ax=axes[2, 2])

    im12 = axes[2, 3].imshow(np.log1p(dinf_accum), cmap='YlGnBu')
    axes[2, 3].set_title(f'D∞流量累积 (max={dinf_accum.max():.0f})')
    plt.colorbar(im12, ax=axes[2, 3])

    d8_zero = (d8_dir == 0).sum()
    fd8_zero = (fd8_dir == 0).sum()
    curv_zero = (curv_dir == 0).sum()
    dinf_zero = (dinf_dir == 0).sum()

    methods = ['D8', 'FD8', '曲率修正', 'D∞']
    zero_counts = [d8_zero, fd8_zero, curv_zero, dinf_zero]
    bars = axes[3, 0].bar(methods, zero_counts, color=['red', 'orange', 'green', 'blue'])
    axes[3, 0].set_title('未确定流向的像素数')
    axes[3, 0].set_ylabel('像素数量')
    for bar, count in zip(bars, zero_counts):
        axes[3, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{count}', ha='center', va='bottom')

    accum_means = [d8_accum.mean(), fd8_accum.mean(), curv_accum.mean(), dinf_accum.mean()]
    bars2 = axes[3, 1].bar(methods, accum_means, color=['red', 'orange', 'green', 'blue'])
    axes[3, 1].set_title('平均流量累积')
    axes[3, 1].set_ylabel('累积值')
    for bar, val in zip(bars2, accum_means):
        axes[3, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{val:.1f}', ha='center', va='bottom')

    d8_spread = np.std(d8_accum)
    fd8_spread = np.std(fd8_accum)
    curv_spread = np.std(curv_accum)
    dinf_spread = np.std(dinf_accum)
    spreads = [d8_spread, fd8_spread, curv_spread, dinf_spread]
    bars3 = axes[3, 2].bar(methods, spreads, color=['red', 'orange', 'green', 'blue'])
    axes[3, 2].set_title('流量累积标准差（扩散程度）')
    axes[3, 2].set_ylabel('标准差')
    for bar, val in zip(bars3, spreads):
        axes[3, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{val:.1f}', ha='center', va='bottom')

    axes[3, 3].axis('off')
    axes[3, 3].text(0.1, 0.9, '算法对比总结:', fontsize=12, fontweight='bold')
    axes[3, 3].text(0.1, 0.75, '• D8: 简单但在平坦区发散', fontsize=10)
    axes[3, 3].text(0.1, 0.65, '• FD8: 多流向分配，物理更合理', fontsize=10)
    axes[3, 3].text(0.1, 0.55, '• 曲率修正: 利用地形信息定方向', fontsize=10)
    axes[3, 3].text(0.1, 0.45, '• D∞: 连续角度，精度最高', fontsize=10)
    axes[3, 3].text(0.1, 0.3, f'未确定流向减少: {d8_zero - min(zero_counts)} 像素', fontsize=10, color='green')
    axes[3, 3].text(0.1, 0.2, f'改进比例: {(1 - min(zero_counts)/max(d8_zero,1))*100:.1f}%', fontsize=10, color='green')

    plt.tight_layout()
    plt.savefig('flow_direction_comparison.png', dpi=200, bbox_inches='tight')
    plt.show()

    return fig


def visualize_flat_region_comparison(dem, dem_flat, filled,
                                     d8_dir_flat, d8_accum_flat,
                                     fd8_dir_flat, fd8_accum_flat,
                                     curv_dir_flat, curv_accum_flat,
                                     dinf_dir_flat, dinf_accum_flat):
    """
    专门可视化平坦区域的对比
    """
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))

    im1 = axes[0, 0].imshow(dem_flat, cmap='terrain')
    axes[0, 0].set_title('平坦测试DEM')
    plt.colorbar(im1, ax=axes[0, 0])

    dir_cmap = colors.ListedColormap([
        'white', 'red', 'orange', 'yellow',
        'green', 'cyan', 'blue', 'purple', 'pink'
    ])
    bounds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    dir_norm = colors.BoundaryNorm(bounds, dir_cmap.N)

    im2 = axes[0, 1].imshow(d8_dir_flat, cmap=dir_cmap, norm=dir_norm)
    axes[0, 1].set_title('D8流向 (平坦区问题明显)')
    cbar2 = plt.colorbar(im2, ax=axes[0, 1], ticks=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])
    cbar2.ax.set_yticklabels(['无', '东', '东南', '南', '西南', '西', '西北', '北', '东北'])

    im3 = axes[0, 2].imshow(fd8_dir_flat, cmap=dir_cmap, norm=dir_norm)
    axes[0, 2].set_title('FD8流向 (多流向)')
    cbar3 = plt.colorbar(im3, ax=axes[0, 2], ticks=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])
    cbar3.ax.set_yticklabels(['无', '东', '东南', '南', '西南', '西', '西北', '北', '东北'])

    im4 = axes[0, 3].imshow(curv_dir_flat, cmap=dir_cmap, norm=dir_norm)
    axes[0, 3].set_title('曲率修正流向')
    cbar4 = plt.colorbar(im4, ax=axes[0, 3], ticks=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])
    cbar4.ax.set_yticklabels(['无', '东', '东南', '南', '西南', '西', '西北', '北', '东北'])

    im5 = axes[1, 0].imshow(np.log1p(d8_accum_flat), cmap='YlGnBu')
    axes[1, 0].set_title(f'D8累积 (发散)')
    plt.colorbar(im5, ax=axes[1, 0])

    im6 = axes[1, 1].imshow(np.log1p(fd8_accum_flat), cmap='YlGnBu')
    axes[1, 1].set_title(f'FD8累积 (扩散)')
    plt.colorbar(im6, ax=axes[1, 1])

    im7 = axes[1, 2].imshow(np.log1p(curv_accum_flat), cmap='YlGnBu')
    axes[1, 2].set_title(f'曲率修正累积 (汇聚)')
    plt.colorbar(im7, ax=axes[1, 2])

    im8 = axes[1, 3].imshow(np.log1p(dinf_accum_flat), cmap='YlGnBu')
    axes[1, 3].set_title(f'D∞累积 (连续)')
    plt.colorbar(im8, ax=axes[1, 3])

    flat_mask = (dem_flat == dem_flat[25, 25])
    d8_flat_zero = ((d8_dir_flat == 0) & flat_mask).sum()
    fd8_flat_zero = ((fd8_dir_flat == 0) & flat_mask).sum()
    curv_flat_zero = ((curv_dir_flat == 0) & flat_mask).sum()
    dinf_flat_zero = ((dinf_dir_flat == 0) & flat_mask).sum()

    axes[2, 0].axis('off')
    axes[2, 0].text(0.1, 0.9, '平坦区域统计 (共{}像素):'.format(flat_mask.sum()), fontsize=11, fontweight='bold')
    axes[2, 0].text(0.1, 0.75, 'D8未确定流向: {} ({:.1f}%)'.format(d8_flat_zero, d8_flat_zero/flat_mask.sum()*100), fontsize=10, color='red')
    axes[2, 0].text(0.1, 0.65, 'FD8未确定流向: {} ({:.1f}%)'.format(fd8_flat_zero, fd8_flat_zero/flat_mask.sum()*100), fontsize=10, color='orange')
    axes[2, 0].text(0.1, 0.55, '曲率修正未确定: {} ({:.1f}%)'.format(curv_flat_zero, curv_flat_zero/flat_mask.sum()*100), fontsize=10, color='green')
    axes[2, 0].text(0.1, 0.45, 'D∞未确定流向: {} ({:.1f}%)'.format(dinf_flat_zero, dinf_flat_zero/flat_mask.sum()*100), fontsize=10, color='blue')

    axes[2, 1].axis('off')
    axes[2, 1].text(0.1, 0.9, '算法特点:', fontsize=11, fontweight='bold')
    axes[2, 1].text(0.1, 0.75, 'D8: 单一方向，平坦区失效', fontsize=10)
    axes[2, 1].text(0.1, 0.65, 'FD8: 按坡度比例分配流量', fontsize=10)
    axes[2, 1].text(0.1, 0.55, '曲率修正: 利用地形曲率定方向', fontsize=10)
    axes[2, 1].text(0.1, 0.45, 'D∞: 连续角度，最精确', fontsize=10)

    axes[2, 2].axis('off')
    axes[2, 2].text(0.1, 0.9, '适用场景:', fontsize=11, fontweight='bold')
    axes[2, 2].text(0.1, 0.75, 'D8: 地形陡峭、数据精度高', fontsize=10)
    axes[2, 2].text(0.1, 0.65, 'FD8: 漫流、汇流扩散模拟', fontsize=10)
    axes[2, 2].text(0.1, 0.55, '曲率修正: 需保留单一流向时', fontsize=10)
    axes[2, 2].text(0.1, 0.45, 'D∞: 科研、高精度要求', fontsize=10)

    axes[2, 3].axis('off')
    axes[2, 3].text(0.1, 0.9, '推荐方案:', fontsize=11, fontweight='bold')
    axes[2, 3].text(0.1, 0.75, '湿地水文分析:', fontsize=10, fontweight='bold')
    axes[2, 3].text(0.1, 0.65, '  • FD8 + 曲率修正组合', fontsize=10, color='green')
    axes[2, 3].text(0.1, 0.55, '  • 平坦区用曲率定向', fontsize=10, color='green')
    axes[2, 3].text(0.1, 0.45, '  • 陡坡区用D8即可', fontsize=10, color='green')

    plt.tight_layout()
    plt.savefig('flat_region_comparison.png', dpi=200, bbox_inches='tight')
    plt.show()

    return fig


def main():
    """
    主函数：对比多种流向算法在湿地水文分析中的表现
    """
    print("=" * 70)
    print("湿地水文连通性分析 - 流向算法改进版")
    print("=" * 70)
    print()

    np.random.seed(42)

    print("1. 生成模拟DEM数据（包含人工平坦区域）...")
    dem_original, X, Y = generate_demo_dem(size=100, seed=42)
    print(f"   DEM尺寸: {dem_original.shape}")
    print(f"   高程范围: {dem_original.min():.2f} - {dem_original.max():.2f}")
    print()

    print("2. 执行DEM填洼处理...")
    filled_dem = depression_filling_simple(dem_original)
    fill_depth = filled_dem - dem_original
    print(f"   最大填洼深度: {fill_depth.max():.4f}")
    print(f"   平均填洼深度: {fill_depth.mean():.4f}")
    print()

    print("3. 计算地表曲率...")
    plan_curvature, prof_curvature, slope = calculate_curvature(filled_dem)
    print(f"   平面曲率范围: {plan_curvature.min():.4f} - {plan_curvature.max():.4f}")
    print(f"   剖面曲率范围: {prof_curvature.min():.4f} - {prof_curvature.max():.4f}")
    print(f"   坡度小于0.01的平坦区: {(slope < 0.01).sum()} 像素")
    print()

    print("4. 计算传统D8流向（对比基准）...")
    d8_dir = d8_flow_direction(filled_dem)
    d8_accum = d8_flow_accumulation(d8_dir)
    d8_zero = (d8_dir == 0).sum()
    print(f"   D8未确定流向: {d8_zero} 像素 ({d8_zero/d8_dir.size*100:.2f}%)")
    print(f"   D8最大累积: {d8_accum.max():.0f}")
    print()

    print("5. 计算FD8概率多流向...")
    fd8_dir, fd8_weights = fd8_flow_direction(filled_dem, exponent=1.5)
    fd8_accum = multi_flow_accumulation(fd8_weights)
    fd8_zero = (fd8_dir == 0).sum()
    print(f"   FD8未确定流向: {fd8_zero} 像素 ({fd8_zero/fd8_dir.size*100:.2f}%)")
    print(f"   FD8最大累积: {fd8_accum.max():.0f}")
    print()

    print("6. 计算基于曲率修正的流向...")
    curv_dir, curv_weights, _, _ = curvature_based_flow_direction(filled_dem)
    curv_accum = multi_flow_accumulation(curv_weights)
    curv_zero = (curv_dir == 0).sum()
    print(f"   曲率修正未确定流向: {curv_zero} 像素 ({curv_zero/curv_dir.size*100:.2f}%)")
    print(f"   曲率修正最大累积: {curv_accum.max():.0f}")
    print()

    print("7. 计算D-Infinity无穷方向...")
    dinf_dir, dinf_angle, dinf_weights = dinfinity_flow_direction(filled_dem)
    dinf_accum = multi_flow_accumulation(dinf_weights)
    dinf_zero = (dinf_dir == 0).sum()
    print(f"   D∞未确定流向: {dinf_zero} 像素 ({dinf_zero/dinf_dir.size*100:.2f}%)")
    print(f"   D∞最大累积: {dinf_accum.max():.0f}")
    print()

    print("8. 生成算法对比可视化...")
    fig = visualize_comparison(
        dem_original, filled_dem,
        d8_dir, d8_accum,
        fd8_dir, fd8_accum,
        curv_dir, curv_accum, curv_weights,
        dinf_dir, dinf_accum, dinf_angle,
        plan_curvature, prof_curvature
    )
    print("   已保存: flow_direction_comparison.png")
    print()

    print("9. 生成平坦区域专项测试...")
    dem_flat, X_flat, Y_flat = generate_flat_test_dem(size=50)
    filled_flat = depression_filling_simple(dem_flat)
    plan_c_flat, prof_c_flat, slope_flat = calculate_curvature(filled_flat)

    d8_dir_flat = d8_flow_direction(filled_flat)
    d8_accum_flat = d8_flow_accumulation(d8_dir_flat)

    fd8_dir_flat, fd8_w_flat = fd8_flow_direction(filled_flat, exponent=1.5)
    fd8_accum_flat = multi_flow_accumulation(fd8_w_flat)

    curv_dir_flat, curv_w_flat, _, _ = curvature_based_flow_direction(filled_flat)
    curv_accum_flat = multi_flow_accumulation(curv_w_flat)

    dinf_dir_flat, dinf_a_flat, dinf_w_flat = dinfinity_flow_direction(filled_flat)
    dinf_accum_flat = multi_flow_accumulation(dinf_w_flat)

    fig2 = visualize_flat_region_comparison(
        dem_original, dem_flat, filled_flat,
        d8_dir_flat, d8_accum_flat,
        fd8_dir_flat, fd8_accum_flat,
        curv_dir_flat, curv_accum_flat,
        dinf_dir_flat, dinf_accum_flat
    )
    print("   已保存: flat_region_comparison.png")
    print()

    print("=" * 70)
    print("分析完成! 改进效果总结:")
    print("=" * 70)
    print(f"  传统D8未确定流向: {d8_zero} 像素")
    print(f"  FD8改进后: {fd8_zero} 像素 (减少 {d8_zero - fd8_zero}, {(1-fd8_zero/max(d8_zero,1))*100:.1f}%)")
    print(f"  曲率修正后: {curv_zero} 像素 (减少 {d8_zero - curv_zero}, {(1-curv_zero/max(d8_zero,1))*100:.1f}%)")
    print(f"  D∞改进后: {dinf_zero} 像素 (减少 {d8_zero - dinf_zero}, {(1-dinf_zero/max(d8_zero,1))*100:.1f}%)")
    print()
    print("  推荐湿地分析方案: FD8多流向 + 曲率修正组合")
    print("=" * 70)

    return {
        'dem_original': dem_original,
        'filled_dem': filled_dem,
        'd8_dir': d8_dir,
        'd8_accum': d8_accum,
        'fd8_dir': fd8_dir,
        'fd8_accum': fd8_accum,
        'curv_dir': curv_dir,
        'curv_accum': curv_accum,
        'dinf_dir': dinf_dir,
        'dinf_accum': dinf_accum,
        'plan_curvature': plan_curvature,
        'prof_curvature': prof_curvature
    }


if __name__ == '__main__':
    results = main()

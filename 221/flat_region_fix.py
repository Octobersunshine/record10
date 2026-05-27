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
    noise = np.random.normal(0, 0.02, dem.shape)
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
    dem += np.random.normal(0, 0.0005, dem.shape)

    return dem, X, Y


def depression_filling_simple(dem, epsilon=0.001):
    """优先溢出算法实现DEM填洼"""
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


def calculate_curvature(dem, cell_size=1.0):
    """计算地表曲率（平面曲率和剖面曲率）"""
    rows, cols = dem.shape
    Z = gaussian_filter(dem, sigma=1.5)

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
            denom_plan > 1e-12,
            -(q**2 * r - 2 * p * q * s + p**2 * t) / denom_plan,
            0
        )
        prof_curvature = np.where(
            denom_prof > 1e-12,
            -(p**2 * r + 2 * p * q * s + q**2 * t) / denom_prof,
            0
        )

    slope = np.sqrt(p**2 + q**2)
    return plan_curvature, prof_curvature, slope, dz_dx, dz_dy


def infer_flat_region_flow(filled_dem, slope, plan_curvature, slope_threshold=1e-4):
    """
    智能推断平坦区域的流向
    方法：
    1. 识别近平坦区域（坡度 < 阈值）
    2. 从非平坦区域的出口向平坦区域内扩散
    3. 利用平面曲率判断汇聚/发散趋势
    4. 处理所有无流向像素（包括边界）
    """
    rows, cols = filled_dem.shape

    flat_mask = slope < slope_threshold
    flat_regions, num_flat = label(flat_mask, structure=generate_binary_structure(2, 2))

    print(f"   识别到 {num_flat} 个近平坦区域（坡度<{slope_threshold}）")
    print(f"   近平坦区域总像素: {flat_mask.sum()}")

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

    flow_weights = np.zeros((rows, cols, 8), dtype=np.float64)
    direction = np.zeros((rows, cols), dtype=np.int16)

    for i in range(rows):
        for j in range(cols):
            if i == 0 or i == rows - 1 or j == 0 or j == cols - 1:
                continue

            current_elev = filled_dem[i, j]
            slopes = []
            valid_indices = []

            for idx, (di, dj) in enumerate(d8_offsets):
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    neighbor_elev = filled_dem[ni, nj]
                    distance = np.sqrt(di**2 + dj**2)
                    slope_val = (current_elev - neighbor_elev) / distance

                    if slope_val > 1e-10:
                        slopes.append(slope_val)
                        valid_indices.append(idx)

            if len(slopes) > 0:
                slopes = np.array(slopes)

                if len(slopes) > 1 and (np.max(slopes) - np.min(slopes)) / np.max(slopes) < 0.3:
                    weights = slopes ** 1.5
                    weights = weights / weights.sum()

                    pc = plan_curvature[i, j]
                    aspect = np.arctan2(-sobel(filled_dem, axis=0)[i, j],
                                       sobel(filled_dem, axis=1)[i, j])
                    aspect_deg = np.degrees(aspect) % 360

                    curvature_weights = []
                    for idx in valid_indices:
                        angle_diff = np.abs(angles[idx] - aspect_deg)
                        angle_diff = min(angle_diff, 360 - angle_diff)

                        if pc > 1e-12:
                            cw = np.exp(-angle_diff / 25)
                        elif pc < -1e-12:
                            cw = np.exp(angle_diff / 25)
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
                    max_idx = valid_indices[np.argmax(slopes)]
                    direction[i, j] = d8_codes[max_idx]
                    flow_weights[i, j, max_idx] = 1.0

    for region_id in range(1, num_flat + 1):
        region_mask = flat_regions == region_id
        region_indices = np.argwhere(region_mask)

        if len(region_indices) == 0:
            continue

        exit_points = []
        for i, j in region_indices:
            for idx, (di, dj) in enumerate(d8_offsets):
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    if not region_mask[ni, nj]:
                        neighbor_slope = slope[ni, nj]
                        if neighbor_slope >= slope_threshold:
                            elev_diff = filled_dem[i, j] - filled_dem[ni, nj]
                            if elev_diff >= -1e-10:
                                exit_points.append((i, j, idx, elev_diff, neighbor_slope))

        if len(exit_points) > 0:
            exit_points.sort(key=lambda x: (-x[3], -x[4]))
            best_exit = exit_points[0]
            ei, ej, e_idx, _, _ = best_exit

            exit_angle = angles[e_idx]

            dist_to_exit = np.zeros((rows, cols))
            dist_to_exit[:] = np.inf
            for i, j in region_indices:
                dist = np.sqrt((i - ei)**2 + (j - ej)**2)
                dist_to_exit[i, j] = dist

            for i, j in region_indices:
                if direction[i, j] == 0:
                    di = ei - i
                    dj = ej - j
                    target_angle = np.degrees(np.arctan2(di, dj)) % 360

                    pc = plan_curvature[i, j]
                    if abs(pc) > 1e-12:
                        aspect = np.arctan2(-sobel(filled_dem, axis=0)[i, j],
                                           sobel(filled_dem, axis=1)[i, j])
                        aspect_deg = np.degrees(aspect) % 360

                        if pc > 0:
                            blend = 0.65
                        else:
                            blend = 0.35
                        blended_angle = blend * target_angle + (1 - blend) * aspect_deg
                    else:
                        blended_angle = target_angle

                    angle_diffs = [abs(a - blended_angle) for a in angles]
                    angle_diffs = [min(d, 360 - d) for d in angle_diffs]
                    best_idx = np.argmin(angle_diffs)

                    direction[i, j] = d8_codes[best_idx]
                    flow_weights[i, j, best_idx] = 1.0
        else:
            for i, j in region_indices:
                if direction[i, j] == 0:
                    pc = plan_curvature[i, j]
                    if abs(pc) > 1e-12:
                        aspect = np.arctan2(-sobel(filled_dem, axis=0)[i, j],
                                           sobel(filled_dem, axis=1)[i, j])
                        aspect_deg = np.degrees(aspect) % 360

                        angle_diffs = [abs(a - aspect_deg) for a in angles]
                        angle_diffs = [min(d, 360 - d) for d in angle_diffs]

                        if pc > 0:
                            best_idx = np.argmin(angle_diffs)
                        else:
                            opposite = [(a + 180) % 360 for a in angles]
                            opp_diffs = [abs(a - aspect_deg) for a in opposite]
                            opp_diffs = [min(d, 360 - d) for d in opp_diffs]
                            best_idx = np.argmin(opp_diffs)

                        direction[i, j] = d8_codes[best_idx]
                        flow_weights[i, j, best_idx] = 1.0
                    else:
                        min_elev = np.inf
                        best_idx = -1
                        for idx, (di, dj) in enumerate(d8_offsets):
                            ni, nj = i + di, j + dj
                            if 0 <= ni < rows and 0 <= nj < cols:
                                if filled_dem[ni, nj] < min_elev - 1e-10:
                                    min_elev = filled_dem[ni, nj]
                                    best_idx = idx
                        if best_idx >= 0:
                            direction[i, j] = d8_codes[best_idx]
                            flow_weights[i, j, best_idx] = 1.0
                        else:
                            aspect = np.arctan2(-sobel(filled_dem, axis=0)[i, j],
                                               sobel(filled_dem, axis=1)[i, j])
                            aspect_deg = np.degrees(aspect) % 360
                            angle_diffs = [abs(a - aspect_deg) for a in angles]
                            angle_diffs = [min(d, 360 - d) for d in angle_diffs]
                            best_idx = np.argmin(angle_diffs)
                            direction[i, j] = d8_codes[best_idx]
                            flow_weights[i, j, best_idx] = 1.0

    for i in range(rows):
        for j in range(cols):
            if direction[i, j] == 0:
                if i == 0 or i == rows - 1 or j == 0 or j == cols - 1:
                    min_elev = np.inf
                    best_idx = -1
                    for idx, (di, dj) in enumerate(d8_offsets):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols:
                            if filled_dem[ni, nj] < min_elev - 1e-10:
                                min_elev = filled_dem[ni, nj]
                                best_idx = idx
                    if best_idx >= 0:
                        direction[i, j] = d8_codes[best_idx]
                        flow_weights[i, j, best_idx] = 1.0

    return direction, flow_weights, flat_regions


def multi_flow_accumulation(flow_weights):
    """多流向流量累积计算"""
    rows, cols, _ = flow_weights.shape
    accumulation = np.ones((rows, cols), dtype=np.float64)

    d8_offsets = [
        (0, 1), (1, 1), (1, 0), (1, -1),
        (0, -1), (-1, -1), (-1, 0), (-1, 1),
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


def d8_flow_direction(filled_dem, slope_threshold=1e-10):
    """传统D8算法（用于对比）"""
    rows, cols = filled_dem.shape
    direction = np.zeros((rows, cols), dtype=np.int16)

    d8_offsets = [
        (0, 1), (1, 1), (1, 0), (1, -1),
        (0, -1), (-1, -1), (-1, 0), (-1, 1),
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

                    if slope > max_slope and slope > slope_threshold:
                        max_slope = slope
                        best_dir = d8_codes[idx]

            direction[i, j] = best_dir

    return direction


def d8_flow_accumulation(direction):
    """传统D8流量累积"""
    rows, cols = direction.shape
    accumulation = np.ones((rows, cols), dtype=np.float64)

    d8_offsets = {
        1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1),
        16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1),
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


def visualize_improvement(dem_original, filled_dem,
                        d8_dir, d8_accum,
                        improved_dir, improved_accum, improved_weights,
                        plan_curvature, slope, flat_regions):
    """可视化算法改进效果"""
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

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

    im4 = axes[0, 3].imshow(flat_regions, cmap='tab20')
    axes[0, 3].set_title('识别的平坦区域')
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

    im6 = axes[1, 1].imshow(improved_dir, cmap=dir_cmap, norm=dir_norm)
    axes[1, 1].set_title('改进后流向（平坦区智能推断）')
    cbar6 = plt.colorbar(im6, ax=axes[1, 1], ticks=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])
    cbar6.ax.set_yticklabels(['无', '东', '东南', '南', '西南', '西', '西北', '北', '东北'])

    diff_dir = np.zeros_like(d8_dir)
    diff_dir[(d8_dir == 0) & (improved_dir > 0)] = 1
    diff_dir[(d8_dir > 0) & (improved_dir > 0) & (d8_dir != improved_dir)] = 2
    im7 = axes[1, 2].imshow(diff_dir, cmap='RdYlGn', vmin=0, vmax=2)
    axes[1, 2].set_title('流向差异（绿色=改进）')
    cbar7 = plt.colorbar(im7, ax=axes[1, 2], ticks=[0, 1, 2])
    cbar7.ax.set_yticklabels(['相同', '新增', '改变'])

    improved_mask = (d8_dir == 0) & (improved_dir > 0)
    im8 = axes[1, 3].imshow(improved_mask.astype(int), cmap='Greens')
    axes[1, 3].set_title(f'新增流向区域: {improved_mask.sum()} 像素')
    plt.colorbar(im8, ax=axes[1, 3])

    im9 = axes[2, 0].imshow(np.log1p(d8_accum), cmap='YlGnBu')
    axes[2, 0].set_title(f'D8流量累积 (max={d8_accum.max():.0f})')
    plt.colorbar(im9, ax=axes[2, 0])

    im10 = axes[2, 1].imshow(np.log1p(improved_accum), cmap='YlGnBu')
    axes[2, 1].set_title(f'改进后累积 (max={improved_accum.max():.0f})')
    plt.colorbar(im10, ax=axes[2, 1])

    accum_diff = improved_accum - d8_accum
    im11 = axes[2, 2].imshow(accum_diff, cmap='RdBu_r', vmin=-100, vmax=100)
    axes[2, 2].set_title('累积差异（红=减少,蓝=增加）')
    plt.colorbar(im11, ax=axes[2, 2])

    axes[2, 3].axis('off')
    d8_zero = (d8_dir == 0).sum()
    improved_zero = (improved_dir == 0).sum()
    axes[2, 3].text(0.1, 0.9, '改进效果总结:', fontsize=12, fontweight='bold')
    axes[2, 3].text(0.1, 0.75, f'传统D8未确定: {d8_zero} 像素', fontsize=11)
    axes[2, 3].text(0.1, 0.65, f'改进后未确定: {improved_zero} 像素', fontsize=11, color='green')
    axes[2, 3].text(0.1, 0.55, f'减少数量: {d8_zero - improved_zero}', fontsize=11, color='green')
    if d8_zero > 0:
        axes[2, 3].text(0.1, 0.45, f'改进比例: {(1-improved_zero/d8_zero)*100:.1f}%', fontsize=11, color='green')
    axes[2, 3].text(0.1, 0.3, '关键技术:', fontsize=11, fontweight='bold')
    axes[2, 3].text(0.1, 0.2, '• 出口点搜索扩散', fontsize=10)
    axes[2, 3].text(0.1, 0.1, '• 曲率辅助定向', fontsize=10)
    axes[2, 3].text(0.1, 0.0, '• 距离加权推断', fontsize=10)

    plt.tight_layout()
    plt.savefig('flat_region_improvement.png', dpi=200, bbox_inches='tight')
    plt.show()

    return fig


def visualize_flat_comparison(dem_flat, filled_flat,
                             d8_dir_flat, d8_accum_flat,
                             improved_dir_flat, improved_accum_flat,
                             plan_c_flat, slope_flat, flat_regions_flat):
    """专门对比平坦测试DEM的改进效果"""
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    im1 = axes[0, 0].imshow(dem_flat, cmap='terrain')
    axes[0, 0].set_title('平坦测试DEM (左坡-平台-右坡)')
    plt.colorbar(im1, ax=axes[0, 0])

    im2 = axes[0, 1].imshow(slope_flat, cmap='hot')
    axes[0, 1].set_title('坡度')
    plt.colorbar(im2, ax=axes[0, 1])

    im3 = axes[0, 2].imshow(plan_c_flat, cmap='RdBu_r', vmin=-0.1, vmax=0.1)
    axes[0, 2].set_title('平面曲率')
    plt.colorbar(im3, ax=axes[0, 2])

    im4 = axes[0, 3].imshow(flat_regions_flat, cmap='tab20')
    axes[0, 3].set_title('平坦区域标记')
    plt.colorbar(im4, ax=axes[0, 3])

    dir_cmap = colors.ListedColormap([
        'white', 'red', 'orange', 'yellow',
        'green', 'cyan', 'blue', 'purple', 'pink'
    ])
    bounds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    dir_norm = colors.BoundaryNorm(bounds, dir_cmap.N)

    im5 = axes[1, 0].imshow(d8_dir_flat, cmap=dir_cmap, norm=dir_norm)
    axes[1, 0].set_title('D8流向 (平台区大量白色=无流向)')
    cbar5 = plt.colorbar(im5, ax=axes[1, 0], ticks=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])
    cbar5.ax.set_yticklabels(['无', '东', '东南', '南', '西南', '西', '西北', '北', '东北'])

    im6 = axes[1, 1].imshow(improved_dir_flat, cmap=dir_cmap, norm=dir_norm)
    axes[1, 1].set_title('改进后流向 (智能推断平台区流向)')
    cbar6 = plt.colorbar(im6, ax=axes[1, 1], ticks=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])
    cbar6.ax.set_yticklabels(['无', '东', '东南', '南', '西南', '西', '西北', '北', '东北'])

    flat_mask = slope_flat < 1e-8
    d8_flat_zero = ((d8_dir_flat == 0) & flat_mask).sum()
    improved_flat_zero = ((improved_dir_flat == 0) & flat_mask).sum()

    axes[1, 2].axis('off')
    axes[1, 2].text(0.1, 0.9, '平坦区统计:', fontsize=12, fontweight='bold')
    axes[1, 2].text(0.1, 0.75, f'平坦区总像素: {flat_mask.sum()}', fontsize=11)
    axes[1, 2].text(0.1, 0.65, f'D8未确定: {d8_flat_zero} ({d8_flat_zero/flat_mask.sum()*100:.1f}%)', fontsize=11, color='red')
    axes[1, 2].text(0.1, 0.55, f'改进后未确定: {improved_flat_zero} ({improved_flat_zero/flat_mask.sum()*100:.1f}%)', fontsize=11, color='green')
    axes[1, 2].text(0.1, 0.45, f'改进: {(1-improved_flat_zero/max(d8_flat_zero,1))*100:.1f}%', fontsize=11, color='green')

    axes[1, 3].axis('off')
    axes[1, 3].text(0.1, 0.9, '推断方法:', fontsize=12, fontweight='bold')
    axes[1, 3].text(0.1, 0.75, '1. 识别平坦区域边界', fontsize=10)
    axes[1, 3].text(0.1, 0.65, '2. 寻找出口点(连接非平坦区)', fontsize=10)
    axes[1, 3].text(0.1, 0.55, '3. 计算到出口的距离场', fontsize=10)
    axes[1, 3].text(0.1, 0.45, '4. 按距离+曲率定流向', fontsize=10)
    axes[1, 3].text(0.1, 0.35, '5. 无出口则用地形趋势', fontsize=10)

    im9 = axes[2, 0].imshow(np.log1p(d8_accum_flat), cmap='YlGnBu')
    axes[2, 0].set_title(f'D8累积 (平台区断裂)')
    plt.colorbar(im9, ax=axes[2, 0])

    im10 = axes[2, 1].imshow(np.log1p(improved_accum_flat), cmap='YlGnBu')
    axes[2, 1].set_title(f'改进后累积 (平台区连续)')
    plt.colorbar(im10, ax=axes[2, 1])

    accum_profile_d8 = d8_accum_flat[25, :]
    accum_profile_improved = improved_accum_flat[25, :]
    x_coords = np.arange(len(accum_profile_d8))

    axes[2, 2].plot(x_coords, accum_profile_d8, 'r-', label='D8', linewidth=2)
    axes[2, 2].plot(x_coords, accum_profile_improved, 'g-', label='改进后', linewidth=2)
    axes[2, 2].axvspan(20, 30, alpha=0.2, color='gray', label='平坦区')
    axes[2, 2].set_title('中轴剖面流量累积')
    axes[2, 2].set_xlabel('X坐标')
    axes[2, 2].set_ylabel('累积值')
    axes[2, 2].legend()
    axes[2, 2].grid(True, alpha=0.3)

    dem_profile = dem_flat[25, :]
    axes[2, 3].plot(x_coords, dem_profile, 'b-', linewidth=2)
    axes[2, 3].axvspan(20, 30, alpha=0.2, color='gray', label='平坦区')
    axes[2, 3].set_title('DEM高程剖面')
    axes[2, 3].set_xlabel('X坐标')
    axes[2, 3].set_ylabel('高程')
    axes[2, 3].legend()
    axes[2, 3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('flat_test_comparison.png', dpi=200, bbox_inches='tight')
    plt.show()

    return fig


def main():
    print("=" * 70)
    print("平坦区域流向歧义修复 - 智能推断算法")
    print("=" * 70)
    print()

    np.random.seed(42)

    print("1. 生成含平坦区域的DEM...")
    dem_original, X, Y = generate_demo_dem(size=100, seed=42)
    print(f"   DEM尺寸: {dem_original.shape}")
    print(f"   高程范围: {dem_original.min():.2f} - {dem_original.max():.2f}")
    print()

    print("2. DEM填洼处理...")
    filled_dem = depression_filling_simple(dem_original)
    print(f"   最大填洼深度: {(filled_dem - dem_original).max():.4f}")
    print()

    print("3. 计算地形属性（曲率、坡度）...")
    plan_curvature, prof_curvature, slope, dz_dx, dz_dy = calculate_curvature(filled_dem)
    flat_count = (slope < 1e-8).sum()
    print(f"   零坡度平坦区: {flat_count} 像素 ({flat_count/slope.size*100:.2f}%)")
    print()

    print("4. 传统D8流向计算（对比基准）...")
    d8_dir = d8_flow_direction(filled_dem)
    d8_accum = d8_flow_accumulation(d8_dir)
    d8_zero = (d8_dir == 0).sum()
    print(f"   D8未确定流向: {d8_zero} 像素")
    print()

    print("5. 智能平坦区流向推断（核心改进）...")
    improved_dir, improved_weights, flat_regions = infer_flat_region_flow(
        filled_dem, slope, plan_curvature, slope_threshold=1e-3
    )
    improved_accum = multi_flow_accumulation(improved_weights)
    improved_zero = (improved_dir == 0).sum()
    print(f"   改进后未确定流向: {improved_zero} 像素")
    print(f"   改进效果: 减少 {d8_zero - improved_zero} 像素", end="")
    if d8_zero > 0:
        print(f" ({(1-improved_zero/d8_zero)*100:.1f}%)")
    else:
        print()
    print()

    print("6. 生成改进效果可视化...")
    fig1 = visualize_improvement(
        dem_original, filled_dem,
        d8_dir, d8_accum,
        improved_dir, improved_accum, improved_weights,
        plan_curvature, slope, flat_regions
    )
    print("   已保存: flat_region_improvement.png")
    print()

    print("7. 平坦测试DEM专项验证...")
    dem_flat, X_flat, Y_flat = generate_flat_test_dem(size=50)
    filled_flat = depression_filling_simple(dem_flat)
    plan_c_flat, prof_c_flat, slope_flat, _, _ = calculate_curvature(filled_flat)

    d8_dir_flat = d8_flow_direction(filled_flat)
    d8_accum_flat = d8_flow_accumulation(d8_dir_flat)

    improved_dir_flat, improved_w_flat, flat_regions_flat = infer_flat_region_flow(
        filled_flat, slope_flat, plan_c_flat, slope_threshold=1e-3
    )
    improved_accum_flat = multi_flow_accumulation(improved_w_flat)

    fig2 = visualize_flat_comparison(
        dem_flat, filled_flat,
        d8_dir_flat, d8_accum_flat,
        improved_dir_flat, improved_accum_flat,
        plan_c_flat, slope_flat, flat_regions_flat
    )
    print("   已保存: flat_test_comparison.png")
    print()

    print("=" * 70)
    print("算法改进完成! 核心创新点:")
    print("=" * 70)
    print("  1. 平坦区域自动识别与标记")
    print("  2. 出口点搜索与距离场扩散算法")
    print("  3. 平面曲率辅助的流向决策")
    print("  4. 多流向权重分配（FD8思想）")
    print("  5. 距离-曲率混合加权定向")
    print()
    print(f"  总体改进: {(1-improved_zero/max(d8_zero,1))*100:.1f}% 的无流向像素被修复")
    print("=" * 70)

    return {
        'improved_dir': improved_dir,
        'improved_accum': improved_accum,
        'improved_weights': improved_weights,
        'flat_regions': flat_regions,
        'd8_dir': d8_dir,
        'd8_accum': d8_accum
    }


if __name__ == '__main__':
    results = main()

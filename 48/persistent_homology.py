import numpy as np
import gudhi as gd
from typing import List, Tuple, Dict


def normalize_point_cloud(
    point_cloud: np.ndarray,
    method: str = 'standard'
) -> np.ndarray:
    """
    归一化点云数据，消除各向异性
    
    参数:
        point_cloud: 输入点云数据
        method: 归一化方法
            - 'standard': 中心化+标准化 (均值为0，标准差为1)
            - 'minmax': 最小-最大归一化到 [0, 1]
            - 'sphere': 归一化到单位超球 (范数为1)
            - 'center': 仅中心化 (均值为0)
    
    返回:
        归一化后的点云数据
    """
    if point_cloud.ndim != 2:
        raise ValueError("点云数据必须是二维数组")
    
    point_cloud = point_cloud.astype(np.float64)
    
    if method == 'standard':
        mean = np.mean(point_cloud, axis=0)
        std = np.std(point_cloud, axis=0)
        std = np.where(std == 0, 1, std)
        return (point_cloud - mean) / std
    
    elif method == 'minmax':
        min_vals = np.min(point_cloud, axis=0)
        max_vals = np.max(point_cloud, axis=0)
        range_vals = max_vals - min_vals
        range_vals = np.where(range_vals == 0, 1, range_vals)
        return (point_cloud - min_vals) / range_vals
    
    elif method == 'sphere':
        mean = np.mean(point_cloud, axis=0)
        centered = point_cloud - mean
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        max_norm = np.max(norms)
        if max_norm > 0:
            return centered / max_norm
        return centered
    
    elif method == 'center':
        mean = np.mean(point_cloud, axis=0)
        return point_cloud - mean
    
    else:
        raise ValueError(f"未知的归一化方法: {method}")


def compute_persistent_homology(
    point_cloud: np.ndarray,
    max_dimension: int = 2,
    max_edge_length: float = None,
    normalize: bool = True,
    normalize_method: str = 'sphere',
    distance_metric: str = 'euclidean'
) -> Dict:
    """
    计算点云数据的持续同调
    
    参数:
        point_cloud: 输入点云数据，形状为 (n_samples, n_features)
        max_dimension: 持续同调的最大维度
        max_edge_length: Rips复形的最大边长，默认为点云的最大直径
        normalize: 是否归一化点云（推荐True，避免各向异性）
        normalize_method: 归一化方法: 'standard', 'minmax', 'sphere', 'center'
        distance_metric: 距离度量，默认为欧氏距离
    
    返回:
        包含条形码和绘图信息的字典
    """
    if point_cloud.ndim != 2:
        raise ValueError("点云数据必须是二维数组 (n_samples, n_features)")
    
    original_point_cloud = point_cloud.copy()
    
    if normalize:
        point_cloud = normalize_point_cloud(point_cloud, method=normalize_method)
    
    if max_edge_length is None:
        from scipy.spatial.distance import pdist
        distances = pdist(point_cloud, metric=distance_metric)
        if len(distances) > 0:
            max_edge_length = np.max(distances)
        else:
            max_edge_length = 1.0
    
    rips_complex = gd.RipsComplex(
        points=point_cloud.tolist(),
        max_edge_length=max_edge_length
    )
    
    simplex_tree = rips_complex.create_simplex_tree(
        max_dimension=max_dimension
    )
    
    persistence = simplex_tree.persistence()
    
    barcodes = {}
    for dim in range(max_dimension + 1):
        barcodes[dim] = simplex_tree.persistence_intervals_in_dimension(dim)
    
    return {
        'persistence': persistence,
        'barcodes': barcodes,
        'simplex_tree': simplex_tree,
        'num_simplices': simplex_tree.num_simplices(),
        'num_vertices': simplex_tree.num_vertices(),
        'normalized_point_cloud': point_cloud,
        'original_point_cloud': original_point_cloud,
        'was_normalized': normalize,
        'normalize_method': normalize_method if normalize else None,
        'max_edge_length': max_edge_length
    }


def linear_weight(persistence: np.ndarray, max_lifetime: float = None) -> np.ndarray:
    """
    线性权重函数：权重与生命周期成正比
    """
    lifetimes = persistence[:, 1] - persistence[:, 0]
    if max_lifetime is None:
        max_lifetime = np.max(lifetimes) if len(lifetimes) > 0 else 1.0
    if max_lifetime > 0:
        return lifetimes / max_lifetime
    return np.ones_like(lifetimes)


def logarithmic_weight(persistence: np.ndarray, max_lifetime: float = None) -> np.ndarray:
    """
    对数权重函数：对长生命周期给予更高权重
    """
    lifetimes = persistence[:, 1] - persistence[:, 0]
    if max_lifetime is None:
        max_lifetime = np.max(lifetimes) if len(lifetimes) > 0 else 1.0
    if max_lifetime > 0:
        return np.log(1 + lifetimes / max_lifetime)
    return np.ones_like(lifetimes)


def uniform_weight(persistence: np.ndarray, max_lifetime: float = None) -> np.ndarray:
    """
    统一权重函数：所有点权重相同
    """
    return np.ones(len(persistence))


def barcodes_to_persistence_image(
    barcode: np.ndarray,
    resolution: Tuple[int, int] = (50, 50),
    weight_func: str = 'linear',
    sigma: float = 0.1,
    birth_range: Tuple[float, float] = None,
    pers_range: Tuple[float, float] = None,
    normalize: bool = True
) -> np.ndarray:
    """
    将条形码转换为持久图像（Persistence Image）
    
    参数:
        barcode: 条形码数据，形状为 (n_points, 2)，每行为 [birth, death]
        resolution: 图像分辨率 (height, width) = (pers_axis, birth_axis)
        weight_func: 权重函数: 'linear', 'logarithmic', 'uniform'
        sigma: 高斯核带宽
        birth_range: 出生轴范围 [min, max]，自动计算为None
        pers_range: 生命周期轴范围 [min, max]，自动计算为None
        normalize: 是否归一化图像到 [0, 1]
    
    返回:
        持久图像数组，形状为 (height, width)
    """
    if len(barcode) == 0:
        return np.zeros(resolution)
    
    barcode = np.array(barcode, dtype=np.float64)
    
    lifetimes = barcode[:, 1] - barcode[:, 0]
    valid = (lifetimes > 0) & (~np.isinf(lifetimes)) & (~np.isinf(barcode[:, 0]))
    
    if not np.any(valid):
        return np.zeros(resolution)
    
    barcode = barcode[valid]
    lifetimes = lifetimes[valid]
    
    birth = barcode[:, 0]
    pers = lifetimes
    
    if birth_range is None:
        birth_min, birth_max = np.min(birth), np.max(birth)
        if birth_min == birth_max:
            birth_min -= 0.1
            birth_max += 0.1
    else:
        birth_min, birth_max = birth_range
    
    if pers_range is None:
        pers_min, pers_max = np.min(pers), np.max(pers)
        if pers_min == pers_max:
            pers_min -= 0.1
            pers_max += 0.1
    else:
        pers_min, pers_max = pers_range
    
    weight_funcs = {
        'linear': linear_weight,
        'logarithmic': logarithmic_weight,
        'uniform': uniform_weight
    }
    if weight_func not in weight_funcs:
        raise ValueError(f"未知的权重函数: {weight_func}")
    
    max_lifetime = pers_max
    weights = weight_funcs[weight_func](barcode, max_lifetime)
    
    height, width = resolution
    
    pers_bins = np.linspace(pers_min, pers_max, height + 1)
    birth_bins = np.linspace(birth_min, birth_max, width + 1)
    
    pers_centers = (pers_bins[:-1] + pers_bins[1:]) / 2
    birth_centers = (birth_bins[:-1] + birth_bins[1:]) / 2
    
    image = np.zeros((height, width))
    
    for b, p, w in zip(birth, pers, weights):
        b_dist = (b - birth_centers) / (birth_bins[1] - birth_bins[0])
        p_dist = (p - pers_centers) / (pers_bins[1] - pers_bins[0])
        
        b_grid, p_grid = np.meshgrid(b_dist, p_dist)
        dist_sq = b_grid ** 2 + p_grid ** 2
        
        image += w * np.exp(-dist_sq / (2 * sigma ** 2))
    
    if normalize and np.max(image) > 0:
        image = image / np.max(image)
    
    return image


def compute_persistence_images(
    result: Dict,
    resolution: Tuple[int, int] = (50, 50),
    weight_func: str = 'linear',
    sigma: float = 0.1,
    shared_range: bool = True,
    normalize: bool = True
) -> Dict[int, np.ndarray]:
    """
    计算所有维度的持久图像
    
    参数:
        result: compute_persistent_homology的返回结果
        resolution: 图像分辨率
        weight_func: 权重函数
        sigma: 高斯核带宽
        shared_range: 是否所有维度使用相同的坐标轴范围
        normalize: 是否归一化图像
    
    返回:
        各维度持久图像字典 {dim: image_array}
    """
    barcodes = result['barcodes']
    
    if shared_range:
        all_birth = []
        all_pers = []
        for dim in barcodes:
            bc = barcodes[dim]
            if len(bc) > 0:
                lifetimes = bc[:, 1] - bc[:, 0]
                valid = (lifetimes > 0) & (~np.isinf(lifetimes)) & (~np.isinf(bc[:, 0]))
                if np.any(valid):
                    all_birth.extend(bc[valid, 0])
                    all_pers.extend(lifetimes[valid])
        
        if len(all_birth) > 0:
            birth_range = (np.min(all_birth), np.max(all_birth))
            pers_range = (np.min(all_pers), np.max(all_pers))
        else:
            birth_range = (0, 1)
            pers_range = (0, 1)
    else:
        birth_range = None
        pers_range = None
    
    images = {}
    for dim in sorted(barcodes.keys()):
        images[dim] = barcodes_to_persistence_image(
            barcodes[dim],
            resolution=resolution,
            weight_func=weight_func,
            sigma=sigma,
            birth_range=birth_range,
            pers_range=pers_range,
            normalize=normalize
        )
    
    return images


def persistence_images_to_vector(
    images: Dict[int, np.ndarray],
    dims: List[int] = None
) -> np.ndarray:
    """
    将持久图像展平为一维特征向量，供机器学习使用
    
    参数:
        images: 各维度持久图像字典
        dims: 要包含的维度，None表示全部
    
    返回:
        一维特征向量
    """
    if dims is None:
        dims = sorted(images.keys())
    
    vectors = []
    for dim in dims:
        if dim in images:
            vectors.append(images[dim].flatten())
    
    return np.concatenate(vectors) if vectors else np.array([])


def plot_persistence_images(
    images: Dict[int, np.ndarray],
    figsize: Tuple[int, int] = None,
    cmap: str = 'viridis'
):
    """
    绘制持久图像
    """
    import matplotlib.pyplot as plt
    
    n_dims = len(images)
    if figsize is None:
        figsize = (4 * n_dims, 4)
    
    fig, axes = plt.subplots(1, n_dims, figsize=figsize)
    if n_dims == 1:
        axes = [axes]
    
    for ax, dim in zip(axes, sorted(images.keys())):
        img = images[dim]
        im = ax.imshow(img, origin='lower', cmap=cmap, aspect='auto')
        ax.set_title(f'Persistence Image - H{dim}')
        ax.set_xlabel('Birth')
        ax.set_ylabel('Persistence')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    return fig


def plot_barcodes(result: Dict, figsize: Tuple[int, int] = (10, 8)):
    """
    绘制条形码图
    """
    import matplotlib.pyplot as plt
    
    barcodes = result['barcodes']
    max_dim = max(barcodes.keys())
    
    fig, axes = plt.subplots(max_dim + 1, 1, figsize=figsize, sharex=True)
    if max_dim == 0:
        axes = [axes]
    
    colors = ['blue', 'orange', 'green', 'red']
    
    for dim, ax in enumerate(axes):
        if dim in barcodes:
            intervals = barcodes[dim]
            if len(intervals) > 0:
                for i, (birth, death) in enumerate(intervals):
                    ax.plot([birth, death], [i, i], color=colors[dim % len(colors)], linewidth=2)
                
                ax.set_ylabel(f'H{dim}', fontsize=12)
                ax.set_yticks([])
                ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Filtration Value', fontsize=12)
    plt.suptitle('Persistent Homology Barcodes', fontsize=14, y=0.95)
    plt.tight_layout()
    return fig


def plot_persistence_diagram(result: Dict, figsize: Tuple[int, int] = (8, 8)):
    """
    绘制持续图
    """
    import matplotlib.pyplot as plt
    
    persistence = result['persistence']
    
    plt.figure(figsize=figsize)
    gd.plot_persistence_diagram(persistence)
    plt.title('Persistence Diagram', fontsize=14)
    plt.tight_layout()
    return plt.gcf()


def generate_sample_point_cloud(sample_type: str = 'torus', n_points: int = 200) -> np.ndarray:
    """
    生成示例点云数据
    """
    if sample_type == 'torus':
        R, r = 2.0, 1.0
        theta = np.random.uniform(0, 2 * np.pi, n_points)
        phi = np.random.uniform(0, 2 * np.pi, n_points)
        x = (R + r * np.cos(phi)) * np.cos(theta)
        y = (R + r * np.cos(phi)) * np.sin(theta)
        z = r * np.sin(phi)
        return np.column_stack([x, y, z])
    
    elif sample_type == 'sphere':
        theta = np.random.uniform(0, 2 * np.pi, n_points)
        phi = np.random.uniform(0, np.pi, n_points)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        return np.column_stack([x, y, z])
    
    elif sample_type == 'circle':
        theta = np.random.uniform(0, 2 * np.pi, n_points)
        x = np.cos(theta)
        y = np.sin(theta)
        return np.column_stack([x, y])
    
    elif sample_type == 'cloud':
        return np.random.randn(n_points, 3)
    
    else:
        raise ValueError(f"未知的样本类型: {sample_type}")


def print_barcodes_summary(result: Dict):
    """
    打印条形码摘要信息
    """
    print("=" * 60)
    print("持续同调条形码摘要")
    print("=" * 60)
    print(f"单形数量: {result['num_simplices']}")
    print(f"顶点数量: {result['num_vertices']}")
    print()
    
    barcodes = result['barcodes']
    for dim in sorted(barcodes.keys()):
        intervals = barcodes[dim]
        print(f"维度 H{dim}: {len(intervals)} 个条形码")
        if len(intervals) > 0:
            lifetimes = intervals[:, 1] - intervals[:, 0]
            print(f"  最小生命周期: {np.min(lifetimes):.6f}")
            print(f"  最大生命周期: {np.max(lifetimes):.6f}")
            print(f"  平均生命周期: {np.mean(lifetimes):.6f}")
        print()


def main():
    print("持续同调计算示例（含点云归一化和持久图像）")
    print("-" * 70)
    
    print("\n1. 生成测试点云...")
    np.random.seed(42)
    n_points = 100
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    x = 10 * np.cos(theta)
    y = 0.1 * np.sin(theta)
    anisotropic_cloud = np.column_stack([x, y])
    print(f"   点云形状: {anisotropic_cloud.shape}")
    print(f"   X轴范围: [{np.min(x):.2f}, {np.max(x):.2f}]")
    print(f"   Y轴范围: [{np.min(y):.4f}, {np.max(y):.4f}]")
    print(f"   各向异性比 (X/Y): { (np.max(x)-np.min(x)) / (np.max(y)-np.min(y)):.1f}")
    
    print("\n2. 计算持续同调...")
    result = compute_persistent_homology(
        point_cloud=anisotropic_cloud,
        max_dimension=1,
        normalize=True,
        normalize_method='standard'
    )
    print(f"   归一化方法: {result['normalize_method']}")
    print(f"   最大边长: {result['max_edge_length']:.4f}")
    print(f"   H0条形码数: {len(result['barcodes'][0])}")
    print(f"   H1条形码数: {len(result['barcodes'][1])}")
    
    print("\n3. 生成持久图像...")
    resolution = (50, 50)
    images = compute_persistence_images(
        result,
        resolution=resolution,
        weight_func='linear',
        sigma=0.1,
        shared_range=True,
        normalize=True
    )
    
    for dim in sorted(images.keys()):
        print(f"   H{dim}持久图像形状: {images[dim].shape}")
        print(f"   H{dim}持久图像值范围: [{images[dim].min():.4f}, {images[dim].max():.4f}]")
    
    print("\n4. 不同权重函数对比...")
    for weight in ['uniform', 'linear', 'logarithmic']:
        img = barcodes_to_persistence_image(
            result['barcodes'][1],
            resolution=(50, 50),
            weight_func=weight,
            sigma=0.1
        )
        print(f"   {weight}权重: 像素和 = {img.sum():.4f}")
    
    print("\n5. 转换为机器学习特征向量...")
    feature_vector = persistence_images_to_vector(images, dims=[0, 1])
    print(f"   特征向量形状: {feature_vector.shape}")
    print(f"   特征向量统计: 均值={feature_vector.mean():.6f}, 标准差={feature_vector.std():.6f}")
    
    print("\n6. 环面点云测试（H0, H1, H2）...")
    torus_cloud = generate_sample_point_cloud('torus', n_points=80)
    result_torus = compute_persistent_homology(
        point_cloud=torus_cloud,
        max_dimension=2,
        normalize=True,
        normalize_method='sphere'
    )
    
    images_torus = compute_persistence_images(
        result_torus,
        resolution=(40, 40),
        weight_func='linear',
        sigma=0.15
    )
    
    torus_features = persistence_images_to_vector(images_torus)
    print(f"   环面特征向量形状: {torus_features.shape}")
    print(f"   特征向量维度: {len(torus_features)} (适合机器学习分类/回归)")
    
    print(f"\n7. 条形码摘要:")
    print_barcodes_summary(result_torus)
    
    print("\n8. 绘制可视化结果...")
    try:
        fig1 = plot_barcodes(result_torus)
        fig1.savefig('barcodes.png', dpi=150, bbox_inches='tight')
        print("   条形码图已保存到 barcodes.png")
        
        fig2 = plot_persistence_diagram(result_torus)
        fig2.savefig('persistence_diagram.png', dpi=150, bbox_inches='tight')
        print("   持续图已保存到 persistence_diagram.png")
        
        fig3 = plot_persistence_images(images_torus)
        fig3.savefig('persistence_images.png', dpi=150, bbox_inches='tight')
        print("   持久图像已保存到 persistence_images.png")
        
    except Exception as e:
        print(f"   绘图跳过: {e}")
    
    print("\n完成!")
    return result_torus, images_torus, torus_features


if __name__ == "__main__":
    result = main()

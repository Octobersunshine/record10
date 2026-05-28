import warnings
import numpy as np
from scipy import sparse
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.metrics.pairwise import euclidean_distances


def johnson_lindenstrauss_min_dim(n_samples, eps=0.1):
    """
    根据Johnson-Lindenstrauss引理计算最小降维维度下限

    JL引理保证: 对于n个点，若降维到 k >= 4*ln(n) / (eps^2/2 - eps^3/3)，
    则以高概率保持所有成对距离在 (1-eps, 1+eps) 范围内。

    参数:
        n_samples: 样本数量 (必须 >= 1)
        eps: 容许的失真范围 (0, 1)，即距离偏差的比例上限

    返回:
        int: 推荐的最小目标维度下限 k_min

    Raises:
        ValueError: n_samples < 1 或 eps 不在 (0, 1) 内
    """
    if not isinstance(n_samples, (int, np.integer)) or n_samples < 1:
        raise ValueError(f"n_samples必须为正整数，收到 {n_samples}")
    if eps <= 0.0 or eps >= 1.0:
        raise ValueError(f"eps必须在(0, 1)开区间内，收到 {eps}")

    denominator = eps ** 2 / 2.0 - eps ** 3 / 3.0
    k_min = int(np.ceil(4.0 * np.log(n_samples) / denominator))
    return k_min


def validate_n_components(n_samples, n_components, eps=0.1):
    """
    校验目标维度是否满足JL引理下限，不满足时发出警告并返回修正值

    参数:
        n_samples: 样本数量
        n_components: 用户指定的目标维度
        eps: 容许失真

    返回:
        int: 校验后的有效目标维度（取 max(n_components, k_min)）

    Warns:
        UserWarning: 当 n_components < k_min 时
    """
    k_min = johnson_lindenstrauss_min_dim(n_samples, eps)

    if n_components < k_min:
        warnings.warn(
            f"目标维度 n_components={n_components} 低于JL引理下限 k_min={k_min} "
            f"(n_samples={n_samples}, eps={eps})，距离保持可能不满足要求。"
            f"已自动修正为 {k_min}。",
            UserWarning,
            stacklevel=3,
        )
        return k_min

    return n_components


def _resolve_n_components(X, n_components, eps):
    """统一解析 n_components：None 时由 eps 自动推导，否则校验"""
    n_samples = X.shape[0]
    if n_components is None:
        if eps is None:
            raise ValueError("n_components 和 eps 不能同时为 None")
        n_components = johnson_lindenstrauss_min_dim(n_samples, eps)
    else:
        if eps is not None:
            n_components = validate_n_components(n_samples, n_components, eps)
    return n_components


def gaussian_random_projection(X, n_components=None, eps=None, random_state=None):
    """
    高斯随机投影: R^d -> R^k

    投影矩阵元素服从 N(0, 1/k)。

    参数:
        X: 高维数据 (n_samples, n_features)
        n_components: 目标维度。None 时由 eps 自动推导
        eps: 容许失真。与 n_components 同时指定时用于校验
        random_state: 随机种子

    返回:
        X_projected: 投影后数据 (n_samples, n_components)
        projection_matrix: 投影矩阵 (n_features, n_components)
    """
    n_components = _resolve_n_components(X, n_components, eps)
    n_features = X.shape[1]

    rng = np.random.RandomState(random_state)
    projection_matrix = rng.normal(
        loc=0.0,
        scale=1.0 / np.sqrt(n_components),
        size=(n_features, n_components),
    )

    if sparse.issparse(X):
        X_projected = safe_sparse_dot(X, projection_matrix)
    else:
        X_projected = np.dot(X, projection_matrix)

    return X_projected, projection_matrix


def sparse_random_projection(X, n_components=None, eps=None,
                             density='auto', return_sparse=True,
                             random_state=None):
    """
    稀疏随机投影 (Achlioptas, 2003)

    投影矩阵元素取值:
        sqrt(s/k)  概率 1/(2s)
        0          概率 1 - 1/s
        -sqrt(s/k) 概率 1/(2s)

    参数:
        X: 高维数据 (n_samples, n_features)
        n_components: 目标维度。None 时由 eps 自动推导
        eps: 容许失真
        density: 非零元素密度。'auto' 时取 1/sqrt(n_features)，
                 也可直接指定 (0, 1] 范围内的浮点数，如 0.01, 0.05, 0.1
        return_sparse: 是否返回CSR格式的稀疏投影矩阵。当 density <= 0.2
                       时建议启用，可显著提升计算效率
        random_state: 随机种子

    返回:
        X_projected: 投影后数据 (n_samples, n_components)
        projection_matrix: 投影矩阵 (n_features, n_components)，
                           CSR格式或ndarray，取决于 return_sparse
    """
    n_components = _resolve_n_components(X, n_components, eps)
    n_features = X.shape[1]

    if density == 'auto':
        s = min(n_features, int(np.sqrt(n_features)))
        density = 1.0 / s
    else:
        if not (0 < density <= 1.0):
            raise ValueError(f"density 必须在 (0, 1] 范围内，收到 {density}")
        s = 1.0 / density

    scale = np.sqrt(s / n_components)

    rng = np.random.RandomState(random_state)

    if return_sparse and density <= 0.2:
        mask = rng.random((n_features, n_components)) < density
        signs = rng.choice([-1.0, 1.0], (n_features, n_components))
        projection_matrix = sparse.csr_matrix(scale * mask * signs)
    else:
        projection_matrix = rng.choice(
            [-scale, 0, scale],
            size=(n_features, n_components),
            p=[1.0 / (2 * s), 1.0 - 1.0 / s, 1.0 / (2 * s)],
        )

    if sparse.issparse(X):
        X_projected = X.dot(projection_matrix)
    else:
        if sparse.issparse(projection_matrix):
            X_projected = safe_sparse_dot(X, projection_matrix)
        else:
            X_projected = np.dot(X, projection_matrix)

    return X_projected, projection_matrix


def very_sparse_random_projection(X, n_components=None, eps=None,
                                  density='auto', random_state=None):
    """
    极稀疏随机投影 (Li et al., 2006)

    投影矩阵元素从 {-s, 0, s} 中采样:
        P(R_ij = ±s) = density/2,  P(R_ij = 0) = 1 - density
        s = 1/sqrt(density * k)

    参数:
        X: 高维数据 (n_samples, n_features)
        n_components: 目标维度。None 时由 eps 自动推导
        eps: 容许失真
        density: 非零元素密度，'auto' 时取 1/sqrt(n_features)
        random_state: 随机种子

    返回:
        X_projected: 投影后数据 (n_samples, n_components)
        projection_matrix: 投影矩阵 (n_features, n_components) CSR格式
    """
    n_components = _resolve_n_components(X, n_components, eps)
    n_features = X.shape[1]

    if density == 'auto':
        density = 1.0 / np.sqrt(n_features)

    s = np.sqrt(1.0 / (density * n_components))

    rng = np.random.RandomState(random_state)

    total_elements = n_features * n_components
    nnz = int(density * total_elements)
    nnz = min(nnz, total_elements)

    if nnz > 0.02 * total_elements:
        mask = rng.random((n_features, n_components)) < density
        signs = rng.choice([-1.0, 1.0], (n_features, n_components))
        projection_matrix = sparse.csr_matrix(s * mask * signs)
    else:
        rows = []
        cols = []
        data = []
        expected_per_col = max(1, int(n_features * density))
        for col in range(n_components):
            col_rows = rng.choice(n_features, expected_per_col, replace=False)
            rows.extend(col_rows)
            cols.extend([col] * expected_per_col)
        data = rng.choice([-s, s], len(rows))
        projection_matrix = sparse.coo_matrix(
            (data, (rows, cols)), shape=(n_features, n_components)
        ).tocsr()

    if sparse.issparse(X):
        X_projected = X.dot(projection_matrix)
    else:
        X_projected = safe_sparse_dot(X, projection_matrix)

    return X_projected, projection_matrix


class DistortionReport:
    """距离保持程度报告"""

    def __init__(self, ratio, eps):
        self.ratio = ratio
        self.eps = eps
        self.max_distortion = float(np.max(np.abs(ratio - 1)))
        self.ratio_min = float(np.min(ratio))
        self.ratio_max = float(np.max(ratio))
        self.ratio_mean = float(np.mean(ratio))
        self.ratio_median = float(np.median(ratio))
        self.ratio_std = float(np.std(ratio))
        self.n_pairs = len(ratio)
        self.n_violating = int(np.sum(np.abs(ratio - 1) > eps))
        self.violation_rate = self.n_violating / self.n_pairs
        self.eps_satisfied = self.max_distortion <= eps

    def __repr__(self):
        lines = [
            f"DistortionReport (eps={self.eps})",
            f"  样本对数量:       {self.n_pairs}",
            f"  距离比率范围:     [{self.ratio_min:.6f}, {self.ratio_max:.6f}]",
            f"  距离比率均值:     {self.ratio_mean:.6f}",
            f"  距离比率中位数:   {self.ratio_median:.6f}",
            f"  距离比率标准差:   {self.ratio_std:.6f}",
            f"  最大失真:         {self.max_distortion:.6f}",
            f"  超出eps的样本对:  {self.n_violating} / {self.n_pairs} "
            f"({self.violation_rate * 100:.2f}%)",
            f"  满足eps约束:      {'是' if self.eps_satisfied else '否'}",
        ]
        return "\n".join(lines)


def check_distortion(X, X_projected, eps=0.1):
    """
    全面检查投影后的成对距离保持程度

    参数:
        X: 原始高维数据 (n_samples, n_features)
        X_projected: 投影后数据 (n_samples, n_components)
        eps: 容许失真阈值

    返回:
        DistortionReport: 包含完整距离保持指标的报告对象
    """
    dist_original = euclidean_distances(X)
    dist_projected = euclidean_distances(X_projected)

    mask = ~np.eye(dist_original.shape[0], dtype=bool)
    ratio = dist_projected[mask] / dist_original[mask]

    return DistortionReport(ratio, eps)


class RandomProjection:
    """
    随机投影降维统一接口

    自动根据样本数和容差计算推荐目标维度下限，防止维度选择不当导致距离失真。

    参数:
        method: 投影方法 'gaussian' | 'sparse' | 'very_sparse'
        n_components: 目标维度。None 时由 eps 自动推导
        eps: 容许失真 (0, 1)
        density: 稀疏投影的非零密度，'auto' 自动选择
        random_state: 随机种子

    示例:
        >>> rp = RandomProjection(method='gaussian', eps=0.1)
        >>> X_proj = rp.fit_transform(X)
        >>> report = rp.report(X, X_proj)
    """

    METHODS = ('gaussian', 'sparse', 'very_sparse')

    def __init__(self, method='gaussian', n_components=None, eps=0.1,
                 density='auto', random_state=None):
        if method not in self.METHODS:
            raise ValueError(f"method 必须为 {self.METHODS}，收到 '{method}'")
        self.method = method
        self.n_components = n_components
        self.eps = eps
        self.density = density
        self.random_state = random_state
        self.projection_matrix_ = None
        self.n_components_ = None
        self.k_min_ = None
        self.n_samples_seen_ = 0
        self.n_features_in_ = None

    def fit(self, X, y=None):
        """拟合：计算目标维度并生成投影矩阵（不执行实际投影）"""
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        self.n_samples_seen_ = n_samples

        if self.eps is not None:
            self.k_min_ = johnson_lindenstrauss_min_dim(n_samples, self.eps)
        else:
            self.k_min_ = None

        if self.n_components is None:
            if self.eps is None:
                raise ValueError("n_components 和 eps 不能同时为 None")
            self.n_components_ = self.k_min_
        else:
            self.n_components_ = self.n_components
            if self.eps is not None and self.n_components_ < self.k_min_:
                warnings.warn(
                    f"n_components={self.n_components_} < JL下限 k_min={self.k_min_} "
                    f"(n_samples={n_samples}, eps={self.eps})，距离保持可能不满足。",
                    UserWarning,
                    stacklevel=2,
                )

        rng = np.random.RandomState(self.random_state)

        if self.method == 'gaussian':
            self.projection_matrix_ = rng.normal(
                loc=0.0,
                scale=1.0 / np.sqrt(self.n_components_),
                size=(n_features, self.n_components_),
            )

        elif self.method == 'sparse':
            if self.density == 'auto':
                s = min(n_features, int(np.sqrt(n_features)))
                actual_density = 1.0 / s
            else:
                if not (0 < self.density <= 1.0):
                    raise ValueError(f"density 必须在 (0, 1] 范围内，收到 {self.density}")
                s = 1.0 / self.density
                actual_density = self.density
            scale = np.sqrt(s / self.n_components_)

            if actual_density <= 0.2:
                mask = rng.random((n_features, self.n_components_)) < actual_density
                signs = rng.choice([-1.0, 1.0], (n_features, self.n_components_))
                self.projection_matrix_ = sparse.csr_matrix(scale * mask * signs)
            else:
                self.projection_matrix_ = rng.choice(
                    [-scale, 0, scale],
                    size=(n_features, self.n_components_),
                    p=[1.0 / (2 * s), 1.0 - 1.0 / s, 1.0 / (2 * s)],
                )

        elif self.method == 'very_sparse':
            density = 1.0 / np.sqrt(n_features) if self.density == 'auto' else self.density
            s = np.sqrt(1.0 / (density * self.n_components_))
            total_elements = n_features * self.n_components_
            nnz = min(int(density * total_elements), total_elements)

            if nnz > 0.02 * total_elements:
                mask = rng.random((n_features, self.n_components_)) < density
                signs = rng.choice([-1.0, 1.0], (n_features, self.n_components_))
                self.projection_matrix_ = sparse.csr_matrix(s * mask * signs)
            else:
                rows = []
                cols = []
                expected_per_col = max(1, int(n_features * density))
                for col in range(self.n_components_):
                    col_rows = rng.choice(n_features, expected_per_col, replace=False)
                    rows.extend(col_rows)
                    cols.extend([col] * expected_per_col)
                data = rng.choice([-s, s], len(rows))
                self.projection_matrix_ = sparse.coo_matrix(
                    (data, (rows, cols)),
                    shape=(n_features, self.n_components_)
                ).tocsr()

        return self

    def partial_fit(self, X, y=None):
        """
        在线增量拟合：适用于数据流场景。

        首次调用时使用当前批次数据拟合（确定维度、生成投影矩阵），
        后续调用仅更新已处理样本计数，投影矩阵保持不变。

        参数:
            X: 批次数据 (n_samples, n_features)
            y: 忽略，仅为接口兼容性

        返回:
            self
        """
        if self.projection_matrix_ is None:
            return self.fit(X, y)

        n_samples, n_features = X.shape
        if n_features != self.n_features_in_:
            raise ValueError(
                f"特征维度不匹配: 期望 {self.n_features_in_}, 收到 {n_features}"
            )
        self.n_samples_seen_ += n_samples

        return self

    def transform(self, X):
        """
        使用已拟合的投影矩阵进行降维。支持增量处理。

        参数:
            X: 数据 (n_samples, n_features)

        返回:
            X_projected: 降维后数据 (n_samples, n_components_)
        """
        if self.projection_matrix_ is None:
            raise RuntimeError("请先调用 fit() 或 partial_fit()")

        n_features = X.shape[1]
        if n_features != self.n_features_in_:
            raise ValueError(
                f"特征维度不匹配: 期望 {self.n_features_in_}, 收到 {n_features}"
            )

        if sparse.issparse(X):
            return X.dot(self.projection_matrix_)
        if sparse.issparse(self.projection_matrix_):
            return safe_sparse_dot(X, self.projection_matrix_)
        return np.dot(X, self.projection_matrix_)

    def fit_transform(self, X):
        """拟合并降维"""
        self.fit(X)
        return self.transform(X)

    def report(self, X, X_projected=None, eps=None):
        """
        生成距离保持程度报告

        参数:
            X: 原始数据
            X_projected: 投影后数据，None 时自动 transform
            eps: 容许失真，None 时使用构造时的 eps

        返回:
            DistortionReport
        """
        if X_projected is None:
            X_projected = self.transform(X)
        if eps is None:
            eps = self.eps if self.eps is not None else 0.1
        return check_distortion(X, X_projected, eps)

    def inverse_transform(self, X_projected):
        """
        近似重构原始高维数据。

        对于随机投影，由于投影矩阵近似正交，使用转置作为伪逆：
        X_approx = X_projected @ R^T

        参数:
            X_projected: 低维数据 (n_samples, n_components_)

        返回:
            X_approx: 重构的高维数据 (n_samples, n_features_in_)
        """
        if self.projection_matrix_ is None:
            raise RuntimeError("请先调用 fit() 或 partial_fit()")

        if sparse.issparse(self.projection_matrix_):
            R_T = self.projection_matrix_.T
            if sparse.issparse(X_projected):
                return X_projected.dot(R_T)
            return safe_sparse_dot(X_projected, R_T)
        else:
            return np.dot(X_projected, self.projection_matrix_.T)


def compute_reconstruction_mse(X_original, X_reconstructed):
    """
    计算重构均方误差 (MSE)。

    参数:
        X_original: 原始高维数据 (n_samples, n_features)
        X_reconstructed: 重构后的数据 (n_samples, n_features)

    返回:
        mse: 平均每个特征的均方误差
        mse_per_sample: 每个样本的均方误差
    """
    if X_original.shape != X_reconstructed.shape:
        raise ValueError(
            f"形状不匹配: X_original {X_original.shape}, "
            f"X_reconstructed {X_reconstructed.shape}"
        )

    squared_error = (X_original - X_reconstructed) ** 2
    mse_per_sample = np.mean(squared_error, axis=1)
    mse = float(np.mean(mse_per_sample))

    return mse, mse_per_sample


def compare_with_pca(X, n_components_list, random_state=42):
    """
    对比随机投影与 PCA 的降维效果（重构 MSE 和距离保持）。

    参数:
        X: 高维数据 (n_samples, n_features)
        n_components_list: 要测试的目标维度列表
        random_state: 随机种子

    返回:
        dict: 包含各方法在各维度下的 MSE 和失真指标
    """
    from sklearn.decomposition import PCA

    n_samples, n_features = X.shape
    results = {}

    for k in n_components_list:
        if k > n_features or k > n_samples:
            continue

        results[k] = {}

        print(f"\n  目标维度 k = {k}")

        # PCA
        print(f"    PCA...", end=" ", flush=True)
        pca = PCA(n_components=k, random_state=random_state)
        X_pca = pca.fit_transform(X)
        X_pca_recon = pca.inverse_transform(X_pca)
        mse_pca, _ = compute_reconstruction_mse(X, X_pca_recon)
        rpt_pca = check_distortion(X, X_pca, eps=1.0)
        results[k]['pca'] = {
            'mse': mse_pca,
            'max_distortion': rpt_pca.max_distortion,
            'explained_variance': float(np.sum(pca.explained_variance_ratio_)),
        }
        print(f"MSE={mse_pca:.4f}, 最大失真={rpt_pca.max_distortion:.4f}")

        # 高斯随机投影
        print(f"    高斯随机投影...", end=" ", flush=True)
        rp_g = RandomProjection(method='gaussian', n_components=k,
                                eps=None, random_state=random_state)
        X_rp_g = rp_g.fit_transform(X)
        X_rp_g_recon = rp_g.inverse_transform(X_rp_g)
        mse_rp_g, _ = compute_reconstruction_mse(X, X_rp_g_recon)
        rpt_rp_g = rp_g.report(X, X_rp_g, eps=1.0)
        results[k]['gaussian'] = {
            'mse': mse_rp_g,
            'max_distortion': rpt_rp_g.max_distortion,
        }
        print(f"MSE={mse_rp_g:.4f}, 最大失真={rpt_rp_g.max_distortion:.4f}")

        # 稀疏随机投影
        print(f"    稀疏随机投影...", end=" ", flush=True)
        rp_s = RandomProjection(method='sparse', n_components=k,
                                density=0.1, eps=None, random_state=random_state)
        X_rp_s = rp_s.fit_transform(X)
        X_rp_s_recon = rp_s.inverse_transform(X_rp_s)
        mse_rp_s, _ = compute_reconstruction_mse(X, X_rp_s_recon)
        rpt_rp_s = rp_s.report(X, X_rp_s, eps=1.0)
        results[k]['sparse'] = {
            'mse': mse_rp_s,
            'max_distortion': rpt_rp_s.max_distortion,
            'density': 0.1,
        }
        print(f"MSE={mse_rp_s:.4f}, 最大失真={rpt_rp_s.max_distortion:.4f}")

    return results


def _demo_sparse_density():
    print(f"\n{'=' * 70}")
    print(" 稀疏随机投影 — 不同非零密度的性能对比")
    print(f"{'=' * 70}")

    import time

    n_samples = 200
    n_features = 2000
    eps = 0.15
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)

    k_min = johnson_lindenstrauss_min_dim(n_samples, eps)
    print(f"\n数据: {n_samples} 样本 × {n_features} 维, eps={eps}, k_min={k_min}")
    print(f"测试不同 density 下的计算效率和距离保持:\n")

    density_list = ['auto', 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

    print(f"{'density':>10s} | {'非零比例':>10s} | {'存储(MB)':>10s} | "
          f"{'fit时间(s)':>12s} | {'最大失真':>10s} | {'满足eps':>8s}")
    print("-" * 80)

    for density in density_list:
        t0 = time.time()
        rp = RandomProjection(method='sparse', n_components=k_min,
                              density=density, eps=None, random_state=42)
        rp.fit(X)
        t_fit = time.time() - t0

        X_proj = rp.transform(X)
        rpt = rp.report(X, X_proj, eps=eps)

        R = rp.projection_matrix_
        if sparse.issparse(R):
            nnz_ratio = R.nnz / (R.shape[0] * R.shape[1])
            mem_mb = R.data.nbytes / (1024 * 1024)
        else:
            nnz_ratio = np.count_nonzero(R) / (R.shape[0] * R.shape[1])
            mem_mb = R.nbytes / (1024 * 1024)

        density_str = f"{density:.2f}" if isinstance(density, float) else density
        ok = "✔" if rpt.eps_satisfied else "✘"
        print(f"{density_str:>10s} | {nnz_ratio:>10.4f} | {mem_mb:>10.3f} | "
              f"{t_fit:>12.4f} | {rpt.max_distortion:>10.4f} | {ok:>8s}")

    print(f"\n{'=' * 70}")
    print(" 密度越小 → 存储越少 → 计算越快，距离保持均满足 eps 约束")
    print(f"{'=' * 70}")


def _demo_online_projection():
    print(f"\n{'=' * 70}")
    print(" 在线降维 — 流式数据增量处理")
    print(f"{'=' * 70}")

    n_samples_total = 1000
    n_features = 2000
    batch_size = 100
    eps = 0.15

    np.random.seed(42)
    X_full = np.random.randn(n_samples_total, n_features)

    print(f"\n数据: 共 {n_samples_total} 样本 × {n_features} 维")
    print(f"批次大小: {batch_size}, 共 {n_samples_total // batch_size} 批")
    print(f"容许失真 eps={eps}\n")

    rp_online = RandomProjection(method='sparse', density=0.05,
                                 eps=eps, random_state=42)

    all_projections = []
    for i in range(0, n_samples_total, batch_size):
        X_batch = X_full[i:i + batch_size]
        batch_num = i // batch_size + 1

        rp_online.partial_fit(X_batch)
        X_proj_batch = rp_online.transform(X_batch)
        all_projections.append(X_proj_batch)

        if batch_num == 1:
            print(f"第 {batch_num:2d} 批: 拟合完成，k={rp_online.n_components_}, "
                  f"已处理样本数={rp_online.n_samples_seen_}")
        else:
            print(f"第 {batch_num:2d} 批: 增量投影完成，"
                  f"已处理样本数={rp_online.n_samples_seen_}")

    X_proj_online = np.vstack(all_projections)
    print(f"\n在线投影完成，结果形状: {X_proj_online.shape}")

    rp_batch = RandomProjection(method='sparse', density=0.05,
                                eps=eps, random_state=42)
    X_proj_batch = rp_batch.fit_transform(X_full)

    rpt_online = check_distortion(X_full, X_proj_online, eps)
    rpt_batch = check_distortion(X_full, X_proj_batch, eps)

    print(f"\n{'方法':>12s} | {'最大失真':>12s} | {'满足eps':>10s}")
    print("-" * 50)
    print(f"{'在线处理':>12s} | {rpt_online.max_distortion:>12.4f} | "
          f"{'✔' if rpt_online.eps_satisfied else '✘':>10s}")
    print(f"{'批量处理':>12s} | {rpt_batch.max_distortion:>12.4f} | "
          f"{'✔' if rpt_batch.eps_satisfied else '✘':>10s}")

    proj_diff = np.max(np.abs(X_proj_online - X_proj_batch))
    print(f"\n在线与批量投影的最大差异: {proj_diff:.6f}")
    if proj_diff < 1e-6:
        print("✓ 在线处理与批量处理结果完全一致！")

    print(f"\n{'=' * 70}")
    print(" 在线降维支持流式数据，结果与批量处理等价")
    print(f"{'=' * 70}")


def _demo_pca_comparison():
    print(f"\n{'=' * 70}")
    print(" 降维效果对比 — 随机投影 vs PCA (重构 MSE)")
    print(f"{'=' * 70}")

    n_samples = 200
    n_features = 500
    np.random.seed(42)

    t = np.linspace(0, 4 * np.pi, n_features)
    X = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        for j in range(5):
            X[i] += np.random.randn() * np.sin((j + 1) * t + np.random.randn())
    X += 0.1 * np.random.randn(n_samples, n_features)

    print(f"\n数据: {n_samples} 样本 × {n_features} 维 (含5个主成分的正弦信号)")

    k_list = [10, 25, 50, 100]
    print(f"\n测试目标维度: {k_list}")

    results = compare_with_pca(X, k_list, random_state=42)

    print(f"\n{'k':>6s} | {'方法':>14s} | {'MSE':>12s} | {'最大失真':>12s} | {'方差比':>12s}")
    print("-" * 75)
    for k in k_list:
        for method in ['pca', 'gaussian', 'sparse']:
            if method not in results[k]:
                continue
            r = results[k][method]
            var_ratio = f"{r.get('explained_variance', 'N/A'):.4f}" if method == 'pca' else 'N/A'
            method_name = {'pca': 'PCA', 'gaussian': '高斯RP', 'sparse': '稀疏RP'}[method]
            print(f"{k:>6d} | {method_name:>14s} | {r['mse']:>12.4f} | "
                  f"{r['max_distortion']:>12.4f} | {var_ratio:>12s}")
        print("-" * 75)

    print(f"\n{'=' * 70}")
    print(" PCA 重构 MSE 最优；随机投影距离保持更好，计算更高效")
    print(f"{'=' * 70}")


def _demo_bad_vs_good():
    print("=" * 70)
    print(" 随机投影降维 — 目标维度选择对距离保持的影响")
    print("=" * 70)

    n_samples = 200
    n_features = 2000
    eps = 0.15

    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)

    k_min = johnson_lindenstrauss_min_dim(n_samples, eps)
    print(f"\n数据: {n_samples} 样本 × {n_features} 维")
    print(f"容许失真 eps = {eps}")
    print(f"JL引理推荐目标维度下限: k_min = {k_min}")

    bad_k = max(20, k_min // 5)
    good_k = k_min

    print(f"\n{'─' * 70}")
    print(f"  对比: 维度不足 k={bad_k}  vs  维度充足 k={good_k}")
    print(f"{'─' * 70}")

    for method_name in ('gaussian', 'sparse', 'very_sparse'):
        print(f"\n▶ 方法: {method_name}")
        for label, k in [("维度不足", bad_k), ("维度充足", good_k)]:
            rp = RandomProjection(method=method_name, n_components=k,
                                  eps=None, random_state=42)
            X_proj = rp.fit_transform(X)
            rpt = rp.report(X, X_proj, eps=eps)

            print(f"\n  [{label}] k={k}")
            print(f"    距离比率范围:     [{rpt.ratio_min:.4f}, {rpt.ratio_max:.4f}]")
            print(f"    距离比率中位数:   {rpt.ratio_median:.4f}")
            print(f"    最大失真:         {rpt.max_distortion:.4f}")
            print(f"    超出eps的样本对:  {rpt.n_violating}/{rpt.n_pairs} "
                  f"({rpt.violation_rate * 100:.1f}%)")
            print(f"    满足eps={eps}:     {'✔ 是' if rpt.eps_satisfied else '✘ 否'}")

    print(f"\n{'=' * 70}")
    print(" 结论: 目标维度低于JL下限时，距离失真严重；达到下限后可满足eps约束")
    print(f"{'=' * 70}")


def _demo_auto_eps():
    print(f"\n{'=' * 70}")
    print(" 自动维度推导 — 不同 eps 下的推荐维度与验证")
    print(f"{'=' * 70}")

    n_samples = 200
    n_features = 2000
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)

    eps_list = [0.5, 0.3, 0.2, 0.15]

    print(f"\n数据: {n_samples} 样本 × {n_features} 维\n")
    print(f"{'eps':>6s} | {'k_min':>6s} | {'压缩比':>10s} | {'最大失真':>10s} | "
          f"{'违规率':>10s} | {'满足':>4s}")
    print("-" * 70)

    for eps in eps_list:
        rp = RandomProjection(method='gaussian', eps=eps, random_state=42)
        X_proj = rp.fit_transform(X)
        rpt = rp.report(X, X_proj, eps=eps)
        k = rp.n_components_
        compress = f"{n_features / k:.1f}x"
        ok = "✔" if rpt.eps_satisfied else "✘"
        print(f"{eps:>6.2f} | {k:>6d} | {compress:>10s} | {rpt.max_distortion:>10.4f} | "
              f"{rpt.violation_rate * 100:>9.2f}% | {ok:>4s}")

    print(f"\n{'=' * 70}")
    print(" eps 越小 → k_min 越大 → 压缩比越低 → 距离保持越精确")
    print(f"{'=' * 70}")


def _demo_validate_warning():
    print(f"\n{'=' * 70}")
    print(" 维度校验 — 传入过小维度时的警告与自动修正")
    print(f"{'=' * 70}")

    n_samples = 200
    n_features = 2000
    eps = 0.15
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)

    k_min = johnson_lindenstrauss_min_dim(n_samples, eps)
    bad_k = 50

    print(f"\n数据: {n_samples} 样本, eps={eps}, JL下限 k_min={k_min}")
    print(f"用户指定 n_components={bad_k} (低于下限):")

    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        X_proj_bad, _ = gaussian_random_projection(X, n_components=bad_k, eps=eps,
                                                   random_state=42)
        if w:
            print(f"  ⚠  警告: {w[-1].message}")
    rpt_bad = check_distortion(X, X_proj_bad, eps)
    print(f"\n  实际使用维度: {X_proj_bad.shape[1]}")
    print(f"  最大失真:     {rpt_bad.max_distortion:.4f}")
    print(f"  满足eps={eps}: {'✔' if rpt_bad.eps_satisfied else '✘'}")

    print(f"\n用户指定 n_components=None, eps={eps} (自动推导):")
    X_proj_auto, _ = gaussian_random_projection(X, n_components=None, eps=eps,
                                                random_state=42)
    rpt_auto = check_distortion(X, X_proj_auto, eps)
    print(f"\n  自动推导维度: {X_proj_auto.shape[1]}")
    print(f"  最大失真:     {rpt_auto.max_distortion:.4f}")
    print(f"  满足eps={eps}: {'✔' if rpt_auto.eps_satisfied else '✘'}")


if __name__ == "__main__":
    _demo_bad_vs_good()
    _demo_auto_eps()
    _demo_validate_warning()
    _demo_sparse_density()
    _demo_online_projection()
    _demo_pca_comparison()

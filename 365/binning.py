import numpy as np
from sklearn.cluster import KMeans


def equal_width_binning(data, n_bins):
    data = np.asarray(data, dtype=float)
    min_val = data.min()
    max_val = data.max()

    bin_edges = np.linspace(min_val, max_val, n_bins + 1)
    bin_indices = np.clip(np.digitize(data, bin_edges[1:-1], right=False), 0, n_bins - 1)

    return bin_edges, bin_indices


def equal_freq_binning(data, n_bins):
    data = np.asarray(data, dtype=float)

    unique_vals = np.unique(data)
    n_unique = len(unique_vals)

    if n_unique <= n_bins:
        bin_edges = np.append(unique_vals, unique_vals[-1] + 1e-12)
        value_to_bin = np.arange(n_unique)
        actual_n_bins = n_unique
    else:
        bin_edges = np.zeros(n_bins + 1)
        bin_edges[0] = unique_vals[0]
        bin_edges[-1] = unique_vals[-1] + 1e-12

        total_count = len(data)
        target_per_bin = total_count / n_bins

        value_to_bin = np.zeros(n_unique, dtype=int)
        current_bin = 0
        current_count = 0

        for i in range(n_unique):
            val_count = np.sum(data == unique_vals[i])
            if current_count + val_count > target_per_bin * (current_bin + 1) and current_bin < n_bins - 1:
                current_bin += 1
                bin_edges[current_bin] = unique_vals[i]
            value_to_bin[i] = current_bin
            current_count += val_count

        actual_n_bins = current_bin + 1
        bin_edges = bin_edges[:actual_n_bins + 1]

    bin_indices = np.zeros(len(data), dtype=int)
    for i, val in enumerate(unique_vals):
        bin_indices[data == val] = value_to_bin[i]

    return bin_edges, bin_indices


def kmeans_binning(data, n_bins, random_state=42):
    data = np.asarray(data, dtype=float).reshape(-1, 1)

    unique_data = np.unique(data)
    if len(unique_data) <= n_bins:
        return equal_freq_binning(data.flatten(), n_bins)

    kmeans = KMeans(n_clusters=n_bins, random_state=random_state, n_init=10)
    kmeans.fit(data)

    cluster_centers = np.sort(kmeans.cluster_centers_.flatten())
    bin_edges = np.zeros(n_bins + 1)
    bin_edges[0] = data.min()
    for i in range(1, n_bins):
        bin_edges[i] = (cluster_centers[i - 1] + cluster_centers[i]) / 2
    bin_edges[-1] = data.max() + 1e-12

    bin_edges = np.sort(bin_edges)

    bin_indices = np.clip(np.digitize(data.flatten(), bin_edges[1:-1], right=False), 0, n_bins - 1)

    return bin_edges, bin_indices


def custom_binning(data, bin_edges):
    data = np.asarray(data, dtype=float)
    bin_edges = np.asarray(bin_edges, dtype=float)

    if bin_edges[0] > data.min():
        bin_edges = np.insert(bin_edges, 0, data.min())
    if bin_edges[-1] <= data.max():
        bin_edges = np.append(bin_edges, data.max() + 1e-12)

    n_bins = len(bin_edges) - 1
    bin_indices = np.clip(np.digitize(data, bin_edges[1:-1], right=False), 0, n_bins - 1)

    return bin_edges, bin_indices


def encode_bins(bin_indices, n_bins=None, encoding='ordinal'):
    bin_indices = np.asarray(bin_indices, dtype=int)

    if n_bins is None:
        n_bins = bin_indices.max() + 1

    if encoding == 'ordinal':
        return bin_indices.reshape(-1, 1)
    elif encoding == 'onehot':
        onehot = np.zeros((len(bin_indices), n_bins), dtype=int)
        onehot[np.arange(len(bin_indices)), bin_indices] = 1
        return onehot
    else:
        raise ValueError(f"Unknown encoding: {encoding}. Use 'ordinal' or 'onehot'.")


def binning(data, method='equal_width', n_bins=None, bin_edges=None,
            encoding='ordinal', random_state=42):
    data = np.asarray(data, dtype=float)

    if method == 'equal_width':
        if n_bins is None:
            raise ValueError("n_bins must be specified for equal_width binning")
        edges, indices = equal_width_binning(data, n_bins)
    elif method == 'equal_freq':
        if n_bins is None:
            raise ValueError("n_bins must be specified for equal_freq binning")
        edges, indices = equal_freq_binning(data, n_bins)
    elif method == 'kmeans':
        if n_bins is None:
            raise ValueError("n_bins must be specified for kmeans binning")
        edges, indices = kmeans_binning(data, n_bins, random_state=random_state)
    elif method == 'custom':
        if bin_edges is None:
            raise ValueError("bin_edges must be specified for custom binning")
        edges, indices = custom_binning(data, bin_edges)
    else:
        raise ValueError(
            f"Unknown method: {method}. "
            f"Use 'equal_width', 'equal_freq', 'kmeans', or 'custom'."
        )

    encoded = encode_bins(indices, n_bins=len(edges) - 1, encoding=encoding)

    return edges, indices, encoded


if __name__ == "__main__":
    np.random.seed(42)
    data = np.random.randn(20)

    print("原始数据:")
    print(np.round(data, 3))
    print()

    n_bins = 4

    edges_w, indices_w = equal_width_binning(data, n_bins)
    print("=== 等宽分箱 ===")
    print(f"分箱边界: {np.round(edges_w, 3)}")
    print(f"样本箱号: {indices_w}")
    for i in range(n_bins):
        mask = indices_w == i
        print(f"  箱{i} [{edges_w[i]:.3f}, {edges_w[i+1]:.3f}): {np.round(data[mask], 3).tolist()}")
    print()

    edges_f, indices_f = equal_freq_binning(data, n_bins)
    print("=== 等频分箱 ===")
    print(f"分箱边界: {np.round(edges_f, 3)}")
    print(f"样本箱号: {indices_f}")
    n_actual = len(edges_f) - 1
    for i in range(n_actual):
        mask = indices_f == i
        count = mask.sum()
        print(f"  箱{i} [{edges_f[i]:.3f}, {edges_f[i+1]:.3f}): 样本数={count}, 数据={np.round(data[mask], 3).tolist()}")
    print()

    edges_k, indices_k = kmeans_binning(data, n_bins)
    print("=== K-Means 聚类分箱 ===")
    print(f"分箱边界: {np.round(edges_k, 3)}")
    print(f"样本箱号: {indices_k}")
    for i in range(n_bins):
        mask = indices_k == i
        count = mask.sum()
        print(f"  箱{i} [{edges_k[i]:.3f}, {edges_k[i+1]:.3f}): 样本数={count}, 数据={np.round(data[mask], 3).tolist()}")
    print()

    custom_edges = [-1.5, 0, 1.0]
    edges_c, indices_c = custom_binning(data, custom_edges)
    print("=== 自定义分箱 ===")
    print(f"输入边界: {custom_edges}")
    print(f"实际边界: {np.round(edges_c, 3)}")
    print(f"样本箱号: {indices_c}")
    n_actual_c = len(edges_c) - 1
    for i in range(n_actual_c):
        mask = indices_c == i
        count = mask.sum()
        print(f"  箱{i} [{edges_c[i]:.3f}, {edges_c[i+1]:.3f}): 样本数={count}, 数据={np.round(data[mask], 3).tolist()}")
    print()

    print("=" * 60)
    print("测试: 编码方式对比")
    print("=" * 60)
    edges, indices, encoded_ordinal = binning(data, method='equal_freq', n_bins=4, encoding='ordinal')
    _, _, encoded_onehot = binning(data, method='equal_freq', n_bins=4, encoding='onehot')
    print(f"数据: {np.round(data[:8], 3)}")
    print(f"箱号: {indices[:8]}")
    print(f"序号编码:\\n{encoded_ordinal[:8]}")
    print(f"独热编码:\\n{encoded_onehot[:8]}")
    print()

    print("=" * 60)
    print("测试: 含大量重复值的数据")
    print("=" * 60)
    data_with_duplicates = np.array([1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5])
    print(f"原始数据: {data_with_duplicates}")
    print(f"值统计: ", end="")
    for v in np.unique(data_with_duplicates):
        print(f"{v}({np.sum(data_with_duplicates == v)}次) ", end="")
    print("\n")

    edges_dup, indices_dup = equal_freq_binning(data_with_duplicates, 3)
    print(f"分箱边界: {edges_dup}")
    print(f"样本箱号: {indices_dup}")
    n_actual_dup = len(edges_dup) - 1

    print("\n验证相同值是否在同一箱:")
    all_consistent = True
    for v in np.unique(data_with_duplicates):
        bins_for_v = indices_dup[data_with_duplicates == v]
        consistent = len(np.unique(bins_for_v)) == 1
        all_consistent = all_consistent and consistent
        print(f"  值{v}: 箱号={bins_for_v}, 一致性={'✓' if consistent else '✗'}")

    print(f"\n结果: {'全部一致 ✓' if all_consistent else '存在跨箱 ✗'}")
    print("\n各箱详情:")
    for i in range(n_actual_dup):
        mask = indices_dup == i
        vals_in_bin = data_with_duplicates[mask]
        unique_in_bin = np.unique(vals_in_bin)
        print(f"  箱{i} [{edges_dup[i]:.3f}, {edges_dup[i+1]:.3f}): 样本数={mask.sum()}, 包含值={unique_in_bin.tolist()}")

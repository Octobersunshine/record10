import numpy as np


def _compute_cost_matrix(s, t):
    s = np.asarray(s, dtype=float)
    t = np.asarray(t, dtype=float)

    if s.ndim == 1:
        s = s.reshape(-1, 1)
    if t.ndim == 1:
        t = t.reshape(-1, 1)

    n, d = s.shape
    m, d2 = t.shape

    if d != d2:
        raise ValueError(f"Multivariate series must have same dimension: {d} vs {d2}")

    diff = s[:, np.newaxis, :] - t[np.newaxis, :, :]
    cost = np.sum(diff ** 2, axis=2)

    return cost


def dtw_naive(s, t, window=None, return_alignments=False):
    cost_matrix = _compute_cost_matrix(s, t)
    n, m = cost_matrix.shape

    if window is None:
        window = max(n, m)
    window = max(window, abs(n - m))

    INF = float("inf")
    dp = np.full((n + 1, m + 1), INF)
    dp[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_end = min(m, i + window)
        for j in range(j_start, j_end + 1):
            dp[i, j] = cost_matrix[i - 1, j - 1] + min(
                dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1]
            )

    distance = np.sqrt(dp[n, m])

    path = []
    i, j = n, m
    while i > 0 or j > 0:
        path.append((i - 1, j - 1))
        candidates = []
        if i > 0 and j > 0:
            candidates.append((dp[i - 1, j - 1], i - 1, j - 1))
        if i > 0:
            candidates.append((dp[i - 1, j], i - 1, j))
        if j > 0:
            candidates.append((dp[i, j - 1], i, j - 1))
        _, i, j = min(candidates, key=lambda x: x[0])

    path.reverse()

    if return_alignments:
        align_matrix = np.zeros((n, m), dtype=float)
        for (i, j) in path:
            align_matrix[i, j] = 1.0
        return distance, path, align_matrix

    return distance, path


def dtw_windowed(s, t, window, return_alignments=False):
    cost_matrix = _compute_cost_matrix(s, t)
    n, m = cost_matrix.shape

    if n == 0 or m == 0:
        if return_alignments:
            return 0.0, [], np.zeros((0, 0))
        return 0.0, []

    window = max(window, abs(n - m))
    INF = float("inf")

    row_len = 2 * window + 1
    prev_row = np.full(row_len, INF)
    curr_row = np.full(row_len, INF)
    prev_row[window] = 0.0

    path_info = {}

    for i in range(1, n + 1):
        curr_row.fill(INF)
        j_start = max(1, i - window)
        j_end = min(m, i + window)

        for j in range(j_start, j_end + 1):
            offset = j - i + window
            cost = cost_matrix[i - 1, j - 1]

            val_left = curr_row[offset - 1] if offset > 0 else INF
            val_up = prev_row[offset + 1] if offset + 1 < row_len else INF
            val_diag = prev_row[offset]

            min_val = min(val_diag, val_up, val_left)
            curr_row[offset] = cost + min_val

            if min_val == val_diag:
                path_info[(i, j)] = (i - 1, j - 1)
            elif min_val == val_up:
                path_info[(i, j)] = (i - 1, j)
            else:
                path_info[(i, j)] = (i, j - 1)

        prev_row, curr_row = curr_row, prev_row

    final_offset = m - n + window
    distance = np.sqrt(prev_row[final_offset])

    path = []
    i, j = n, m
    while i > 0 or j > 0:
        path.append((i - 1, j - 1))
        if (i, j) in path_info:
            i, j = path_info[(i, j)]
        else:
            if i > 0 and j > 0:
                i -= 1
                j -= 1
            elif i > 0:
                i -= 1
            else:
                j -= 1

    path.reverse()

    if return_alignments:
        align_matrix = np.zeros((n, m), dtype=float)
        for (i, j) in path:
            align_matrix[i, j] = 1.0
        return distance, path, align_matrix

    return distance, path


def _reduce_resolution(s, factor=2):
    s = np.asarray(s, dtype=float)
    if s.ndim == 1:
        s = s.reshape(-1, 1)
    n = len(s)
    reduced_len = n // factor
    if reduced_len == 0:
        return np.mean(s, axis=0).reshape(1, -1)
    s_trimmed = s[:reduced_len * factor]
    return np.mean(s_trimmed.reshape(reduced_len, factor, -1), axis=1)


def fastdtw(s, t, radius=10, return_alignments=False):
    s = np.asarray(s, dtype=float)
    t = np.asarray(t, dtype=float)

    if s.ndim == 1:
        s = s.reshape(-1, 1)
    if t.ndim == 1:
        t = t.reshape(-1, 1)

    n, m = len(s), len(t)

    min_size = max(radius + 2, 50)
    if min(n, m) <= min_size:
        return dtw_windowed(s, t, window=max(n, m), return_alignments=return_alignments)

    shrink_factor = 2
    current_s = s
    current_t = t
    paths = []

    while len(current_s) > min_size and len(current_t) > min_size:
        current_s = _reduce_resolution(current_s, shrink_factor)
        current_t = _reduce_resolution(current_t, shrink_factor)
        dist, path = dtw_windowed(current_s, current_t, window=radius + 2)
        paths.append((len(current_s), len(current_t), path))

    col_min = np.zeros(n + 2, dtype=np.int64)
    col_max = np.zeros(n + 2, dtype=np.int64)

    for i in range(n + 1):
        col_min[i] = max(0, i - radius)
        col_max[i] = min(m, i + radius)

    for (curr_n, curr_m, path) in reversed(paths):
        new_min = np.full(n + 2, m + 1, dtype=np.int64)
        new_max = np.full(n + 2, -1, dtype=np.int64)

        for (i, j) in path:
            i_base = i * 2
            j_base = j * 2
            for di in range(-radius, radius + 1):
                for dj in range(-radius, radius + 1):
                    ni = i_base + di
                    nj = j_base + dj
                    if 0 <= ni <= n and 0 <= nj <= m:
                        if nj < new_min[ni]:
                            new_min[ni] = nj
                        if nj > new_max[ni]:
                            new_max[ni] = nj

        for i in range(n + 1):
            col_min[i] = min(col_min[i], new_min[i])
            col_max[i] = max(col_max[i], new_max[i])

    for i in range(n + 1):
        col_min[i] = max(0, col_min[i])
        col_max[i] = min(m, col_max[i])

    cost_matrix = _compute_cost_matrix(s, t)

    INF = float("inf")
    dp = {}
    path_info = {}
    dp[(0, 0)] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, col_min[i])
        j_end = min(m, col_max[i])
        for j in range(j_start, j_end + 1):
            key = (i, j)
            cost = cost_matrix[i - 1, j - 1]

            candidates = []
            if (i - 1, j - 1) in dp:
                candidates.append((dp[(i - 1, j - 1)], i - 1, j - 1))
            if (i - 1, j) in dp:
                candidates.append((dp[(i - 1, j)], i - 1, j))
            if (i, j - 1) in dp:
                candidates.append((dp[(i, j - 1)], i, j - 1))

            if not candidates:
                continue

            min_val, pi, pj = min(candidates, key=lambda x: x[0])
            dp[key] = cost + min_val
            path_info[key] = (pi, pj)

    final_key = (n, m)
    if final_key not in dp:
        return dtw_windowed(s, t, window=max(n, m) // 4, return_alignments=return_alignments)

    distance = np.sqrt(dp[final_key])

    path = []
    i, j = n, m
    while i > 0 or j > 0:
        path.append((i - 1, j - 1))
        key = (i, j)
        if key in path_info:
            i, j = path_info[key]
        else:
            if i > 0 and j > 0:
                i -= 1
                j -= 1
            elif i > 0:
                i -= 1
            else:
                j -= 1

    path.reverse()

    if return_alignments:
        align_matrix = np.zeros((n, m), dtype=float)
        for (i, j) in path:
            align_matrix[i, j] = 1.0
        return distance, path, align_matrix

    return distance, path


def softdtw(s, t, gamma=1.0, return_gradients=False, return_alignments=False):
    cost_matrix = _compute_cost_matrix(s, t)
    n, m = cost_matrix.shape

    INF = np.inf

    R = np.full((n + 2, m + 2), INF, dtype=np.float64)
    R[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = cost_matrix[i - 1, j - 1]
            r0 = R[i - 1, j]
            r1 = R[i, j - 1]
            r2 = R[i - 1, j - 1]

            if r0 == INF and r1 == INF and r2 == INF:
                R[i, j] = INF
                continue

            min_val = min(r0, r1, r2)
            if gamma < 1e-5:
                R[i, j] = cost + min_val
                continue

            e0 = np.exp((min_val - r0) / gamma) if r0 < INF else 0.0
            e1 = np.exp((min_val - r1) / gamma) if r1 < INF else 0.0
            e2 = np.exp((min_val - r2) / gamma) if r2 < INF else 0.0

            softmin = min_val - gamma * np.log(e0 + e1 + e2)
            R[i, j] = cost + softmin

    distance = np.sqrt(R[n, m]) if R[n, m] < INF else np.inf

    if not (return_gradients or return_alignments):
        return distance

    eps = 1e-12
    E = np.zeros((n + 2, m + 2), dtype=np.float64)
    E[n, m] = 1.0

    for i in range(n, 0, -1):
        for j in range(m, 0, -1):
            if i == n and j == m:
                continue

            r = R[i, j]
            if r == INF:
                continue

            a = R[i + 1, j]
            b = R[i, j + 1]
            c = R[i + 1, j + 1]

            valid = []
            if a < INF:
                valid.append((a, E[i + 1, j]))
            if b < INF:
                valid.append((b, E[i, j + 1]))
            if c < INF:
                valid.append((c, E[i + 1, j + 1]))

            if not valid:
                continue

            min_val = min(v[0] for v in valid)
            e_sum = 0.0
            for val, e_weight in valid:
                e_sum += np.exp((min_val - val) / gamma) * e_weight
            E[i, j] = np.exp((min_val - r) / gamma) * e_sum

    align_matrix = E[1:n + 1, 1:m + 1].copy()
    align_sum = align_matrix.sum()
    if align_sum > eps:
        align_matrix = align_matrix / align_sum

    if return_gradients:
        s_arr = np.asarray(s, dtype=float)
        t_arr = np.asarray(t, dtype=float)

        if s_arr.ndim == 1:
            s_arr = s_arr.reshape(-1, 1)
        if t_arr.ndim == 1:
            t_arr = t_arr.reshape(-1, 1)

        grad_s = np.zeros_like(s_arr)
        grad_t = np.zeros_like(t_arr)

        for i in range(n):
            for j in range(m):
                weight = align_matrix[i, j]
                if weight > eps:
                    diff = 2 * (s_arr[i] - t_arr[j])
                    grad_s[i] += weight * diff
                    grad_t[j] -= weight * diff

        if distance > eps:
            grad_s = grad_s / (2 * distance)
            grad_t = grad_t / (2 * distance)

        if return_alignments:
            return distance, grad_s, grad_t, align_matrix
        return distance, grad_s, grad_t

    return distance, align_matrix


def dtw_multivariate(s, t, method="windowed", **kwargs):
    s = np.asarray(s, dtype=float)
    t = np.asarray(t, dtype=float)

    if s.ndim == 1:
        s = s.reshape(-1, 1)
    if t.ndim == 1:
        t = t.reshape(-1, 1)

    n, d_s = s.shape
    m, d_t = t.shape

    if d_s != d_t:
        raise ValueError(f"Multivariate series must have same dimension: {d_s} vs {d_t}")

    return_alignments = kwargs.get("return_alignments", False)

    if method == "naive":
        result = dtw_naive(s, t, window=kwargs.get("window"), return_alignments=return_alignments)
    elif method == "windowed":
        result = dtw_windowed(s, t, window=kwargs.get("window", 100), return_alignments=return_alignments)
    elif method == "fast":
        result = fastdtw(s, t, radius=kwargs.get("radius", 10), return_alignments=return_alignments)
    elif method == "soft":
        result = softdtw(s, t, gamma=kwargs.get("gamma", 1.0), return_alignments=return_alignments)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'naive', 'windowed', 'fast', or 'soft'.")

    if return_alignments:
        per_dim_distances = []
        for dim in range(d_s):
            dist, _ = dtw_windowed(s[:, dim], t[:, dim], window=kwargs.get("window", 100))
            per_dim_distances.append(dist)
        return result + (np.array(per_dim_distances),)
    return result


if __name__ == "__main__":
    import time

    print("=" * 70)
    print("Test 1: Univariate sequences (basic correctness)")
    print("=" * 70)
    s = [1, 2, 3, 4, 5]
    t = [2, 3, 4, 5, 6]

    dist_naive, path_naive = dtw_naive(s, t, window=10)
    dist_win, path_win = dtw_windowed(s, t, window=10)
    dist_fast, path_fast = fastdtw(s, t, radius=5)
    dist_soft = softdtw(s, t, gamma=0.1)

    print(f"Naive DTW distance: {dist_naive:.4f}")
    print(f"Windowed DTW distance: {dist_win:.4f}")
    print(f"FastDTW distance: {dist_fast:.4f}")
    print(f"SoftDTW distance (gamma=0.1): {dist_soft:.4f}")
    print(f"Paths match (naive vs windowed): {path_naive == path_win}")

    print("\n" + "=" * 70)
    print("Test 2: Multivariate sequences (2D)")
    print("=" * 70)

    np.random.seed(42)
    n = 50
    s_multi = np.column_stack([
        np.sin(np.linspace(0, 4 * np.pi, n)),
        np.cos(np.linspace(0, 4 * np.pi, n))
    ])
    t_multi = np.column_stack([
        np.sin(np.linspace(0.3, 4.3 * np.pi, n)),
        np.cos(np.linspace(0.3, 4.3 * np.pi, n))
    ])

    dist_multi, path_multi, align_multi, per_dim_dist = dtw_multivariate(
        s_multi, t_multi, method="windowed", window=20, return_alignments=True
    )

    print(f"Multivariate DTW distance (2D): {dist_multi:.4f}")
    print(f"Per-dimension distances: {per_dim_dist}")
    print(f"Alignment matrix shape: {align_multi.shape}")
    print(f"Non-zero alignment cells: {np.sum(align_multi > 0)}")

    print("\n" + "=" * 70)
    print("Test 3: Soft-DTW with gradients (differentiable)")
    print("=" * 70)

    s_soft = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    t_soft = np.array([2.0, 3.0, 4.0, 5.0, 6.0])

    dist_soft, grad_s, grad_t, align_soft = softdtw(
        s_soft, t_soft, gamma=0.5, return_gradients=True, return_alignments=True
    )

    print(f"SoftDTW distance: {dist_soft:.4f}")
    print(f"Gradient w.r.t. s: {grad_s.flatten()}")
    print(f"Gradient w.r.t. t: {grad_t.flatten()}")
    print(f"Alignment matrix (soft weights):")
    print(np.round(align_soft, 3))

    print("\n" + "=" * 70)
    print("Test 4: Visualization data with alignment matrix")
    print("=" * 70)

    s_viz = np.sin(np.linspace(0, 6 * np.pi, 30))
    t_viz = np.sin(np.linspace(0.5, 6.5 * np.pi, 25))

    dist_viz, path_viz, align_viz = dtw_naive(
        s_viz, t_viz, window=50, return_alignments=True
    )

    print(f"Distance: {dist_viz:.4f}")
    print(f"Alignment matrix shape: {align_viz.shape}")
    print(f"Path length: {len(path_viz)}")
    print(f"\nAlignment matrix (path points marked):")
    for i in range(min(10, align_viz.shape[0])):
        row_str = "".join("█" if align_viz[i, j] > 0 else "·"
                          for j in range(min(25, align_viz.shape[1])))
        print(f"  {row_str}")

    print("\n" + "=" * 70)
    print("Test 5: Performance comparison on long sequences")
    print("=" * 70)

    n_points = 2000
    s_long = np.sin(np.linspace(0, 20 * np.pi, n_points))
    t_long = np.sin(np.linspace(0.5, 20.5 * np.pi, n_points))

    print(f"Sequence length: {n_points}")

    methods = [
        ("Windowed DTW (w=100)", lambda: dtw_windowed(s_long, t_long, window=100)),
        ("FastDTW (r=20)", lambda: fastdtw(s_long, t_long, radius=20)),
        ("SoftDTW (gamma=1.0)", lambda: softdtw(s_long, t_long, gamma=1.0)),
    ]

    for name, func in methods:
        start = time.time()
        result = func()
        elapsed = time.time() - start
        dist = result[0] if isinstance(result, tuple) else result
        print(f"  {name}: distance={dist:.4f}, time={elapsed:.3f}s")

    print("\n" + "=" * 70)
    print("Summary of Available Functions:")
    print("=" * 70)
    print("  dtw_naive(s, t, window, return_alignments)   - Full matrix DTW")
    print("  dtw_windowed(s, t, window, return_alignments) - Band-limited O(nw)")
    print("  fastdtw(s, t, radius, return_alignments)     - Multi-res approximate O(n)")
    print("  softdtw(s, t, gamma, return_gradients, return_alignments) - Differentiable")
    print("  dtw_multivariate(s, t, method, ...)          - Multi-dimension support")

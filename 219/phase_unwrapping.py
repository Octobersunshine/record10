import numpy as np
from scipy.ndimage import label, uniform_filter, gaussian_filter
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import heapq
from collections import defaultdict


def wrap(phase):
    return np.angle(np.exp(1j * phase))


def create_residues(wrapped_phase):
    rows, cols = wrapped_phase.shape
    residues = np.zeros((rows, cols), dtype=np.int8)

    for i in range(rows - 1):
        for j in range(cols - 1):
            p1 = wrapped_phase[i, j]
            p2 = wrapped_phase[i, j + 1]
            p3 = wrapped_phase[i + 1, j + 1]
            p4 = wrapped_phase[i + 1, j]

            d1 = wrap(p2 - p1)
            d2 = wrap(p3 - p2)
            d3 = wrap(p4 - p3)
            d4 = wrap(p1 - p4)

            sum_d = d1 + d2 + d3 + d4

            if abs(sum_d) > np.pi:
                if sum_d > 0:
                    residues[i, j] = 1
                else:
                    residues[i, j] = -1

    return residues


def branch_cut(unwrapped, wrapped_phase, i, j, visited, direction=0):
    rows, cols = wrapped_phase.shape
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    stack = [(i, j, direction)]

    while stack:
        ci, cj, cd = stack.pop()

        if ci < 0 or ci >= rows - 1 or cj < 0 or cj >= cols - 1:
            continue
        if visited[ci, cj]:
            continue

        visited[ci, cj] = True

        for k in range(4):
            ni, nj = ci + directions[k][0], cj + directions[k][1]
            if ni < 0 or ni >= rows or nj < 0 or nj >= cols:
                continue

            diff = wrap(wrapped_phase[ni, nj] - wrapped_phase[ci, cj])
            unwrapped[ni, nj] = unwrapped[ci, cj] + diff

            if not visited[ni, nj]:
                stack.append((ni, nj, k))

    return unwrapped


def branch_cut_unwrap(wrapped_phase):
    rows, cols = wrapped_phase.shape
    unwrapped = np.zeros_like(wrapped_phase, dtype=np.float64)
    visited = np.zeros((rows, cols), dtype=bool)

    residues = create_residues(wrapped_phase)

    seed_i, seed_j = 0, 0
    for i in range(rows):
        for j in range(cols):
            if residues[i, j] == 0:
                seed_i, seed_j = i, j
                break
        else:
            continue
        break

    unwrapped[seed_i, seed_j] = wrapped_phase[seed_i, seed_j]
    visited[seed_i, seed_j] = True

    branch_cut(unwrapped, wrapped_phase, seed_i, seed_j, visited)

    for i in range(rows):
        for j in range(cols):
            if not visited[i, j]:
                unwrapped[i, j] = wrapped_phase[i, j]
                visited[i, j] = True
                branch_cut(unwrapped, wrapped_phase, i, j, visited)

    return unwrapped


def least_squares_unwrap(wrapped_phase):
    rows, cols = wrapped_phase.shape
    n_pixels = rows * cols

    dx = wrap(np.diff(wrapped_phase, axis=1))
    dy = wrap(np.diff(wrapped_phase, axis=0))

    unwrapped = np.zeros_like(wrapped_phase, dtype=np.float64)
    unwrapped[0, 0] = wrapped_phase[0, 0]

    for i in range(rows):
        for j in range(cols):
            if i == 0 and j == 0:
                continue
            if j > 0:
                unwrapped[i, j] = unwrapped[i, j - 1] + dx[i, j - 1]
            elif i > 0:
                unwrapped[i, j] = unwrapped[i - 1, j] + dy[i - 1, j]

    max_iter = 100
    tol = 1e-3

    for _ in range(max_iter):
        error = 0
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                neighbors = 0
                sum_val = 0

                if j > 0:
                    sum_val += unwrapped[i, j - 1] + dx[i, j - 1]
                    neighbors += 1
                if j < cols - 1:
                    sum_val += unwrapped[i, j + 1] - dx[i, j]
                    neighbors += 1
                if i > 0:
                    sum_val += unwrapped[i - 1, j] + dy[i - 1, j]
                    neighbors += 1
                if i < rows - 1:
                    sum_val += unwrapped[i + 1, j] - dy[i, j]
                    neighbors += 1

                new_val = sum_val / neighbors
                error += abs(new_val - unwrapped[i, j])
                unwrapped[i, j] = new_val

        if error < tol:
            break

    return unwrapped


def quality_guided_unwrap(wrapped_phase):
    rows, cols = wrapped_phase.shape
    unwrapped = np.zeros_like(wrapped_phase, dtype=np.float64)
    unwrapped[:] = np.nan

    quality = np.abs(np.gradient(wrapped_phase)[0]) + np.abs(np.gradient(wrapped_phase)[1])
    quality = -quality

    start_i, start_j = rows // 2, cols // 2
    unwrapped[start_i, start_j] = wrapped_phase[start_i, start_j]

    border = []
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    for di, dj in directions:
        ni, nj = start_i + di, start_j + dj
        if 0 <= ni < rows and 0 <= nj < cols:
            border.append((quality[ni, nj], ni, nj, start_i, start_j))

    import heapq
    heapq.heapify(border)

    while border:
        q, i, j, from_i, from_j = heapq.heappop(border)

        if not np.isnan(unwrapped[i, j]):
            continue

        diff = wrap(wrapped_phase[i, j] - wrapped_phase[from_i, from_j])
        unwrapped[i, j] = unwrapped[from_i, from_j] + diff

        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < rows and 0 <= nj < cols and np.isnan(unwrapped[ni, nj]):
                heapq.heappush(border, (quality[ni, nj], ni, nj, i, j))

    return unwrapped


def estimate_coherence(wrapped_phase, window_size=5):
    rows, cols = wrapped_phase.shape
    pad = window_size // 2

    real = np.cos(wrapped_phase)
    imag = np.sin(wrapped_phase)

    real_sum = uniform_filter(real, size=window_size) * window_size**2
    imag_sum = uniform_filter(imag, size=window_size) * window_size**2
    real2_sum = uniform_filter(real**2, size=window_size) * window_size**2
    imag2_sum = uniform_filter(imag**2, size=window_size) * window_size**2

    numerator = np.sqrt(real_sum**2 + imag_sum**2)
    denominator = np.sqrt(window_size**2 * (real2_sum + imag2_sum))

    coherence = np.ones_like(wrapped_phase)
    valid = denominator > 0
    coherence[valid] = numerator[valid] / denominator[valid]
    coherence = np.clip(coherence, 0, 1)

    return coherence


def phase_derivative_variance(wrapped_phase, window_size=5):
    rows, cols = wrapped_phase.shape

    dx = wrap(np.diff(wrapped_phase, axis=1))
    dy = wrap(np.diff(wrapped_phase, axis=0))

    dx_padded = np.zeros((rows, cols))
    dy_padded = np.zeros((rows, cols))
    dx_padded[:, :-1] = dx
    dy_padded[:-1, :] = dy

    dx_mean = uniform_filter(dx_padded, size=window_size)
    dy_mean = uniform_filter(dy_padded, size=window_size)

    dx_var = uniform_filter((dx_padded - dx_mean)**2, size=window_size)
    dy_var = uniform_filter((dy_padded - dy_mean)**2, size=window_size)

    variance = dx_var + dy_var

    return variance


def compute_cost_gradient(wrapped_phase, coherence=None):
    rows, cols = wrapped_phase.shape

    if coherence is None:
        coherence = estimate_coherence(wrapped_phase)

    dx = np.zeros((rows, cols))
    dy = np.zeros((rows, cols))

    dx[:, :-1] = wrap(np.diff(wrapped_phase, axis=1))
    dy[:-1, :] = wrap(np.diff(wrapped_phase, axis=0))

    weight_x = np.zeros((rows, cols))
    weight_y = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols - 1):
            c = min(coherence[i, j], coherence[i, j + 1])
            weight_x[i, j] = c**2 + 1e-3

    for i in range(rows - 1):
        for j in range(cols):
            c = min(coherence[i, j], coherence[i + 1, j])
            weight_y[i, j] = c**2 + 1e-3

    return dx, dy, weight_x, weight_y


def improved_residue_connection(wrapped_phase, coherence=None, max_distance=20):
    rows, cols = wrapped_phase.shape

    if coherence is None:
        coherence = estimate_coherence(wrapped_phase)

    residues = create_residues(wrapped_phase)

    pos_residues = []
    neg_residues = []

    for i in range(rows - 1):
        for j in range(cols - 1):
            if residues[i, j] == 1:
                pos_residues.append((i, j))
            elif residues[i, j] == -1:
                neg_residues.append((i, j))

    branch_cuts = np.zeros((rows, cols), dtype=bool)

    used_pos = set()
    used_neg = set()

    all_residues = pos_residues + neg_residues

    for idx, (ri, rj) in enumerate(all_residues):
        if (ri, rj) in used_pos or (ri, rj) in used_neg:
            continue

        is_pos = residues[ri, rj] == 1

        candidates = []
        for oi, oj in all_residues[idx + 1:]:
            if (oi, oj) in used_pos or (oi, oj) in used_neg:
                continue
            if residues[oi, oj] == residues[ri, rj]:
                continue

            dist = np.sqrt((ri - oi)**2 + (rj - oj)**2)
            if dist > max_distance:
                continue

            path_coherence = 0
            steps = int(dist) + 1
            for t in range(steps + 1):
                pi = int(ri + (oi - ri) * t / steps)
                pj = int(rj + (oj - rj) * t / steps)
                pi = max(0, min(rows - 1, pi))
                pj = max(0, min(cols - 1, pj))
                path_coherence += coherence[pi, pj]
            path_coherence /= (steps + 1)

            cost = dist / (path_coherence + 0.01)
            candidates.append((cost, oi, oj, dist))

        if not candidates:
            boundary_dist = min(ri, rj, rows - 1 - ri, cols - 1 - rj)
            if boundary_dist < max_distance:
                if ri <= rj and ri <= rows - 1 - ri:
                    for k in range(ri + 1):
                        branch_cuts[k, rj] = True
                elif rj <= cols - 1 - rj:
                    for k in range(rj + 1):
                        branch_cuts[ri, k] = True
                elif rows - 1 - ri <= cols - 1 - rj:
                    for k in range(ri, rows):
                        branch_cuts[k, rj] = True
                else:
                    for k in range(rj, cols):
                        branch_cuts[ri, k] = True
            continue

        candidates.sort()
        best_cost, best_i, best_j, best_dist = candidates[0]

        steps = int(best_dist) + 1
        for t in range(steps + 1):
            pi = int(ri + (best_i - ri) * t / steps)
            pj = int(rj + (best_j - rj) * t / steps)
            pi = max(0, min(rows - 1, pi))
            pj = max(0, min(cols - 1, pj))
            branch_cuts[pi, pj] = True

        if is_pos:
            used_pos.add((ri, rj))
            used_neg.add((best_i, best_j))
        else:
            used_neg.add((ri, rj))
            used_pos.add((best_i, best_j))

    return residues, branch_cuts


def improved_branch_cut_unwrap(wrapped_phase, coherence=None):
    rows, cols = wrapped_phase.shape

    if coherence is None:
        coherence = estimate_coherence(wrapped_phase)

    residues, branch_cuts = improved_residue_connection(wrapped_phase, coherence)

    unwrapped = np.zeros_like(wrapped_phase, dtype=np.float64)
    visited = np.zeros((rows, cols), dtype=bool)

    quality = coherence.copy()
    for i in range(rows):
        for j in range(cols):
            if branch_cuts[i, j]:
                quality[i, j] = -1

    seed_i, seed_j = np.unravel_index(np.argmax(quality), quality.shape)

    unwrapped[seed_i, seed_j] = wrapped_phase[seed_i, seed_j]
    visited[seed_i, seed_j] = True

    border = []
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    for di, dj in directions:
        ni, nj = seed_i + di, seed_j + dj
        if 0 <= ni < rows and 0 <= nj < cols:
            if not branch_cuts[ni, nj]:
                heapq.heappush(border, (-quality[ni, nj], ni, nj, seed_i, seed_j))

    while border:
        q, i, j, from_i, from_j = heapq.heappop(border)

        if visited[i, j]:
            continue

        diff = wrap(wrapped_phase[i, j] - wrapped_phase[from_i, from_j])
        unwrapped[i, j] = unwrapped[from_i, from_j] + diff
        visited[i, j] = True

        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < rows and 0 <= nj < cols:
                if not visited[ni, nj] and not branch_cuts[ni, nj]:
                    heapq.heappush(border, (-quality[ni, nj], ni, nj, i, j))

    for i in range(rows):
        for j in range(cols):
            if not visited[i, j]:
                unwrapped[i, j] = wrapped_phase[i, j]
                visited[i, j] = True

                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        if not visited[ni, nj]:
                            heapq.heappush(border, (-quality[ni, nj], ni, nj, i, j))

                while border:
                    q, ci, cj, fi, fj = heapq.heappop(border)
                    if visited[ci, cj]:
                        continue
                    diff = wrap(wrapped_phase[ci, cj] - wrapped_phase[fi, fj])
                    unwrapped[ci, cj] = unwrapped[fi, fj] + diff
                    visited[ci, cj] = True
                    for di2, dj2 in directions:
                        ni2, nj2 = ci + di2, cj + dj2
                        if 0 <= ni2 < rows and 0 <= nj2 < cols:
                            if not visited[ni2, nj2] and not branch_cuts[ni2, nj2]:
                                heapq.heappush(border, (-quality[ni2, nj2], ni2, nj2, ci, cj))

    return unwrapped, branch_cuts, residues


class GraphCutUnwrapper:
    def __init__(self, wrapped_phase, coherence=None, num_labels=10):
        self.rows, self.cols = wrapped_phase.shape
        self.wrapped = wrapped_phase
        self.num_labels = num_labels

        if coherence is None:
            self.coherence = estimate_coherence(wrapped_phase)
        else:
            self.coherence = coherence

        self.dx, self.dy, self.wx, self.wy = compute_cost_gradient(wrapped_phase, self.coherence)

        self.phase_offsets = np.linspace(-np.pi, np.pi, num_labels, endpoint=False)

    def data_cost(self, i, j, label):
        k = self.phase_offsets[label]
        unwrapped_est = self.wrapped[i, j] + 2 * np.pi * np.round(k / (2 * np.pi))
        return 0.0

    def smooth_cost(self, i, j, label1, label2, axis='x'):
        k1 = self.phase_offsets[label1]
        k2 = self.phase_offsets[label2]

        if axis == 'x':
            diff = (self.wrapped[i, j] + k2) - (self.wrapped[i, j - 1] + k1)
            weight = self.wx[i, j - 1]
            expected_diff = self.dx[i, j - 1]
        else:
            diff = (self.wrapped[i, j] + k2) - (self.wrapped[i - 1, j] + k1)
            weight = self.wy[i - 1, j]
            expected_diff = self.dy[i - 1, j]

        diff_wrapped = wrap(diff - expected_diff)
        cost = weight * diff_wrapped**2

        return cost

    def alpha_expansion(self, max_iter=50):
        rows, cols = self.rows, self.cols
        labels = np.zeros((rows, cols), dtype=np.int32)

        for i in range(rows):
            for j in range(cols):
                labels[i, j] = self.num_labels // 2

        current_energy = self.compute_energy(labels)

        for iteration in range(max_iter):
            improved = False

            for alpha in range(self.num_labels):
                new_labels = self.try_alpha_expansion(labels, alpha)
                new_energy = self.compute_energy(new_labels)

                if new_energy < current_energy - 1e-6:
                    labels = new_labels
                    current_energy = new_energy
                    improved = True

            if not improved:
                break

        return labels

    def try_alpha_expansion(self, current_labels, alpha):
        rows, cols = self.rows, self.cols
        new_labels = current_labels.copy()

        for i in range(rows):
            for j in range(cols):
                old_label = current_labels[i, j]
                if old_label == alpha:
                    continue

                old_cost = self.compute_pixel_cost(i, j, current_labels, old_label)
                new_cost = self.compute_pixel_cost(i, j, current_labels, alpha)

                if new_cost < old_cost:
                    new_labels[i, j] = alpha

        return new_labels

    def compute_pixel_cost(self, i, j, labels, label):
        rows, cols = self.rows, self.cols
        cost = 0.0

        if j > 0:
            cost += self.smooth_cost(i, j, labels[i, j - 1], label, 'x')
        if j < cols - 1:
            cost += self.smooth_cost(i, j + 1, label, labels[i, j + 1], 'x')
        if i > 0:
            cost += self.smooth_cost(i, j, labels[i - 1, j], label, 'y')
        if i < rows - 1:
            cost += self.smooth_cost(i + 1, j, label, labels[i + 1, j], 'y')

        return cost

    def compute_energy(self, labels):
        rows, cols = self.rows, self.cols
        energy = 0.0

        for i in range(rows):
            for j in range(cols):
                if j > 0:
                    energy += self.smooth_cost(i, j, labels[i, j - 1], labels[i, j], 'x')
                if i > 0:
                    energy += self.smooth_cost(i, j, labels[i - 1, j], labels[i, j], 'y')

        return energy

    def unwrap(self):
        labels = self.alpha_expansion()

        unwrapped = np.zeros_like(self.wrapped, dtype=np.float64)
        for i in range(self.rows):
            for j in range(self.cols):
                k = self.phase_offsets[labels[i, j]]
                unwrapped[i, j] = self.wrapped[i, j] + 2 * np.pi * np.round(k / (2 * np.pi))

        unwrapped_ls = least_squares_unwrap(self.wrapped)
        diff = unwrapped_ls - unwrapped
        k_global = np.round(np.median(diff) / (2 * np.pi))
        unwrapped += 2 * np.pi * k_global

        return unwrapped, labels


def graph_cut_unwrap(wrapped_phase, coherence=None, num_labels=10):
    unwrapper = GraphCutUnwrapper(wrapped_phase, coherence, num_labels)
    return unwrapper.unwrap()


def snaphu_style_unwrap(wrapped_phase, coherence=None):
    rows, cols = wrapped_phase.shape

    if coherence is None:
        coherence = estimate_coherence(wrapped_phase)

    unwrapped_ls = least_squares_unwrap(wrapped_phase)

    dx, dy, wx, wy = compute_cost_gradient(wrapped_phase, coherence)

    unwrapped = unwrapped_ls.copy()

    max_iter = 30
    for iteration in range(max_iter):
        max_change = 0
        for i in range(rows):
            for j in range(cols):
                neighbors = 0
                sum_val = 0
                weight_sum = 0

                if j > 0:
                    w = wx[i, j - 1]
                    target = unwrapped[i, j - 1] + dx[i, j - 1]
                    sum_val += w * target
                    weight_sum += w
                    neighbors += 1
                if j < cols - 1:
                    w = wx[i, j]
                    target = unwrapped[i, j + 1] - dx[i, j]
                    sum_val += w * target
                    weight_sum += w
                    neighbors += 1
                if i > 0:
                    w = wy[i - 1, j]
                    target = unwrapped[i - 1, j] + dy[i - 1, j]
                    sum_val += w * target
                    weight_sum += w
                    neighbors += 1
                if i < rows - 1:
                    w = wy[i, j]
                    target = unwrapped[i + 1, j] - dy[i, j]
                    sum_val += w * target
                    weight_sum += w
                    neighbors += 1

                if weight_sum > 0:
                    new_val = sum_val / weight_sum
                    change = abs(new_val - unwrapped[i, j])
                    if change > max_change:
                        max_change = change
                    unwrapped[i, j] = new_val

        if max_change < 1e-3:
            break

    return unwrapped

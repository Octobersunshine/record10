import numpy as np
from typing import Tuple, List


def kmeans(data: np.ndarray, k: int, max_iters: int = 100, tol: float = 1e-4,
           random_state: int = None, init: str = 'kmeans++', n_init: int = 10
           ) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    K-Means clustering using Lloyd's algorithm with K-Means++ initialization.

    Args:
        data: Multidimensional data points with shape (n_samples, n_features)
        k: Number of clusters
        max_iters: Maximum number of iterations
        tol: Tolerance for convergence (minimum centroid shift)
        random_state: Random seed for reproducibility
        init: Initialization method, 'random' or 'kmeans++' (default)
        n_init: Number of times the algorithm will be run with different
            centroid seeds. The final results will be the best output of
            n_init consecutive runs in terms of inertia.

    Returns:
        labels: Cluster labels for each data point with shape (n_samples,)
        centroids: Cluster centers with shape (k, n_features)
        inertia: Sum of squared distances to the closest centroid
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples, n_features = data.shape

    if k > n_samples:
        raise ValueError(f"Number of clusters k={k} cannot exceed number of samples {n_samples}")

    if init not in ('random', 'kmeans++'):
        raise ValueError(f"init must be 'random' or 'kmeans++', got '{init}'")

    best_labels = None
    best_centroids = None
    best_inertia = float('inf')

    base_seed = np.random.randint(0, 2**31 - 1) if random_state is None else random_state

    for run_idx in range(n_init):
        run_seed = base_seed + run_idx
        np.random.seed(run_seed)

        if init == 'kmeans++':
            centroids = _initialize_centroids_kmeanspp(data, k)
        else:
            centroids = _initialize_centroids_random(data, k)

        for _ in range(max_iters):
            labels = _assign_clusters(data, centroids)
            new_centroids = _update_centroids(data, labels, k)

            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids

            if shift < tol:
                break

        labels = _assign_clusters(data, centroids)
        inertia = _calculate_inertia(data, labels, centroids)

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
            best_centroids = centroids

    if random_state is not None:
        np.random.seed(random_state)

    return best_labels, best_centroids, best_inertia


def _initialize_centroids_random(data: np.ndarray, k: int) -> np.ndarray:
    """Initialize centroids by randomly selecting k unique data points."""
    indices = np.random.choice(data.shape[0], size=k, replace=False)
    return data[indices].copy()


def _initialize_centroids_kmeanspp(data: np.ndarray, k: int) -> np.ndarray:
    """
    Initialize centroids using K-Means++ algorithm.

    The first centroid is chosen uniformly at random. Subsequent centroids
    are chosen with probability proportional to the squared distance to the
    nearest already-chosen centroid.
    """
    n_samples = data.shape[0]
    centroids = np.zeros((k, data.shape[1]))

    first_idx = np.random.randint(n_samples)
    centroids[0] = data[first_idx].copy()

    for i in range(1, k):
        distances = np.min(np.linalg.norm(data[:, np.newaxis] - centroids[:i], axis=2), axis=1)
        squared_distances = distances ** 2
        probs = squared_distances / np.sum(squared_distances)
        next_idx = np.random.choice(n_samples, p=probs)
        centroids[i] = data[next_idx].copy()

    return centroids


def _calculate_inertia(data: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """
    Calculate inertia: sum of squared distances of samples to their closest centroid.
    """
    inertia = 0.0
    for i, centroid in enumerate(centroids):
        cluster_points = data[labels == i]
        if len(cluster_points) > 0:
            inertia += np.sum(np.linalg.norm(cluster_points - centroid, axis=1) ** 2)
    return inertia


def _assign_clusters(data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign each data point to the nearest centroid."""
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)


def _update_centroids(data: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Update centroids as the mean of points in each cluster."""
    n_features = data.shape[1]
    new_centroids = np.zeros((k, n_features))

    for i in range(k):
        cluster_points = data[labels == i]
        if len(cluster_points) > 0:
            new_centroids[i] = cluster_points.mean(axis=0)
        else:
            new_centroids[i] = data[np.random.choice(data.shape[0])]

    return new_centroids


def silhouette_score(data: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate the mean Silhouette Coefficient for all samples.

    The Silhouette Coefficient is calculated using the mean intra-cluster
    distance (a) and the mean nearest-cluster distance (b) for each sample.
    The Silhouette Coefficient for a sample is (b - a) / max(a, b).

    Args:
        data: Multidimensional data points with shape (n_samples, n_features)
        labels: Cluster labels for each data point with shape (n_samples,)

    Returns:
        score: Mean Silhouette Coefficient, range [-1, 1]
            1: Perfect clustering
            0: Overlapping clusters
           -1: Incorrect clustering
    """
    n_samples = data.shape[0]
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters <= 1 or n_clusters >= n_samples:
        return 0.0

    distances = np.linalg.norm(data[:, np.newaxis] - data, axis=2)
    silhouette_values = np.zeros(n_samples)

    for i in range(n_samples):
        own_cluster = labels[i]

        same_cluster_mask = labels == own_cluster
        same_cluster_mask[i] = False
        a = np.mean(distances[i, same_cluster_mask]) if np.sum(same_cluster_mask) > 0 else 0

        b_values = []
        for cluster in unique_labels:
            if cluster != own_cluster:
                other_cluster_mask = labels == cluster
                if np.sum(other_cluster_mask) > 0:
                    b_values.append(np.mean(distances[i, other_cluster_mask]))

        if len(b_values) == 0:
            b = 0
        else:
            b = np.min(b_values)

        if a == 0 and b == 0:
            silhouette_values[i] = 0
        else:
            silhouette_values[i] = (b - a) / max(a, b)

    return np.mean(silhouette_values)


def elbow_method(data: np.ndarray, k_range: Tuple[int, int] = (2, 10),
                 **kmeans_kwargs) -> Tuple[List[int], List[float]]:
    """
    Run K-Means for multiple K values and return inertia values for elbow method.

    Args:
        data: Multidimensional data points with shape (n_samples, n_features)
        k_range: Tuple (min_k, max_k) specifying range of K values to test
        **kmeans_kwargs: Additional arguments passed to kmeans function

    Returns:
        k_values: List of K values tested
        inertias: List of inertia values for each K
    """
    min_k, max_k = k_range
    k_values = list(range(min_k, max_k + 1))
    inertias = []

    for k in k_values:
        _, _, inertia = kmeans(data, k=k, **kmeans_kwargs)
        inertias.append(inertia)

    return k_values, inertias


def find_optimal_k_elbow(data: np.ndarray, k_range: Tuple[int, int] = (2, 10),
                         **kmeans_kwargs) -> Tuple[int, List[int], List[float]]:
    """
    Automatically find optimal K using the elbow method.

    The elbow point is found by locating the point with maximum curvature
    (maximum second derivative) in the inertia curve.

    Args:
        data: Multidimensional data points with shape (n_samples, n_features)
        k_range: Tuple (min_k, max_k) specifying range of K values to test
        **kmeans_kwargs: Additional arguments passed to kmeans function

    Returns:
        optimal_k: The estimated optimal number of clusters
        k_values: List of K values tested
        inertias: List of inertia values for each K
    """
    k_values, inertias = elbow_method(data, k_range, **kmeans_kwargs)

    if len(k_values) < 3:
        return k_values[0], k_values, inertias

    inertias_arr = np.array(inertias)
    k_arr = np.array(k_values)

    first_diff = np.diff(inertias_arr)
    second_diff = np.diff(first_diff)

    if len(second_diff) >= 1:
        curvature = second_diff
        elbow_idx = np.argmax(curvature) + 1
    else:
        elbow_idx = np.argmin(first_diff)

    optimal_k = k_values[elbow_idx]

    return optimal_k, k_values, inertias


def find_optimal_k_silhouette(data: np.ndarray, k_range: Tuple[int, int] = (2, 10),
                              **kmeans_kwargs) -> Tuple[int, List[int], List[float]]:
    """
    Find optimal K using Silhouette score.

    The optimal K is the one that maximizes the mean Silhouette Coefficient.

    Args:
        data: Multidimensional data points with shape (n_samples, n_features)
        k_range: Tuple (min_k, max_k) specifying range of K values to test
        **kmeans_kwargs: Additional arguments passed to kmeans function

    Returns:
        optimal_k: The estimated optimal number of clusters
        k_values: List of K values tested
        silhouette_scores: List of Silhouette scores for each K
    """
    min_k, max_k = k_range
    k_values = list(range(min_k, max_k + 1))
    silhouette_scores = []

    for k in k_values:
        labels, _, _ = kmeans(data, k=k, **kmeans_kwargs)
        score = silhouette_score(data, labels)
        silhouette_scores.append(score)

    optimal_idx = np.argmax(silhouette_scores)
    optimal_k = k_values[optimal_idx]

    return optimal_k, k_values, silhouette_scores


if __name__ == "__main__":
    np.random.seed(42)

    n_per_cluster = 60
    data = np.vstack([
        np.random.randn(n_per_cluster, 5) * 0.6 + np.array([2, 0, 0, 0, 0]) * 0.8,
        np.random.randn(n_per_cluster, 5) * 0.6 + np.array([0, 2, 0, 0, 0]) * 0.8,
        np.random.randn(n_per_cluster, 5) * 0.6 + np.array([0, 0, 2, 0, 0]) * 0.8,
        np.random.randn(n_per_cluster, 5) * 0.6 + np.array([0, 0, 0, 2, 0]) * 0.8,
    ])

    true_k = 4

    print("=" * 70)
    print("=== K-Means Clustering with Elbow Method & Silhouette Score ===")
    print("=" * 70)

    print(f"\nData shape: {data.shape}")
    print(f"True number of clusters: {true_k}")

    print("\n" + "=" * 70)
    print("1. Elbow Method for Optimal K Selection")
    print("=" * 70)
    optimal_k_elbow, k_values, inertias = find_optimal_k_elbow(
        data, k_range=(2, 8), n_init=5, random_state=42
    )
    print(f"\nK values tested: {k_values}")
    print(f"Inertia values: {[f'{v:.1f}' for v in inertias]}")
    print(f"Optimal K (elbow method): {optimal_k_elbow}")

    print("\n" + "=" * 70)
    print("2. Silhouette Score for Optimal K Selection")
    print("=" * 70)
    optimal_k_sil, k_values_sil, sil_scores = find_optimal_k_silhouette(
        data, k_range=(2, 8), n_init=5, random_state=42
    )
    print(f"\nK values tested: {k_values_sil}")
    print(f"Silhouette scores: {[f'{s:.3f}' for s in sil_scores]}")
    print(f"Optimal K (silhouette method): {optimal_k_sil}")

    print("\n" + "=" * 70)
    print("3. Clustering with Detected Optimal K")
    print("=" * 70)
    k = optimal_k_sil
    labels, centroids, inertia = kmeans(data, k=k, n_init=10, random_state=42)
    sil_score = silhouette_score(data, labels)

    print(f"\nUsing K = {k}:")
    print(f"  Inertia: {inertia:.2f}")
    print(f"  Silhouette Score: {sil_score:.3f}")
    print(f"  Points per cluster: {np.bincount(labels)}")
    print(f"\n  Cluster centroids (first 2 dimensions):")
    for i, centroid in enumerate(centroids):
        print(f"    Cluster {i}: [{centroid[0]:.3f}, {centroid[1]:.3f}, ...]")

    print("\n" + "=" * 70)
    print("4. Silhouette Score Interpretation")
    print("=" * 70)
    print("""
  Score Range   | Interpretation
  --------------|------------------
  +0.7 to +1.0  | Excellent clustering
  +0.5 to +0.7  | Good clustering
  +0.25 to +0.5 | Fair clustering
  0.0 to +0.25  | Poor clustering / overlapping
  -1.0 to 0.0   | Incorrect clustering
""")
    print(f"  Current score: {sil_score:.3f}")
    if sil_score >= 0.5:
        quality = "Good to Excellent"
    elif sil_score >= 0.25:
        quality = "Fair"
    else:
        quality = "Poor"
    print(f"  Quality: {quality}")

    print("\n" + "=" * 70)
    print("5. API Usage Examples")
    print("=" * 70)
    print("""
  # Basic K-Means with K-Means++ initialization
  labels, centroids, inertia = kmeans(data, k=4)

  # Elbow method - get inertia curve
  k_values, inertias = elbow_method(data, k_range=(2, 10))

  # Auto-detect optimal K using elbow method
  best_k, k_vals, inertias = find_optimal_k_elbow(data, k_range=(2, 10))

  # Auto-detect optimal K using silhouette score
  best_k, k_vals, scores = find_optimal_k_silhouette(data, k_range=(2, 10))

  # Evaluate clustering quality
  score = silhouette_score(data, labels)
""")

import numpy as np
import pickle
import time
from collections import defaultdict
from typing import List, Tuple, Set, Optional, Dict, Any
from itertools import combinations


def bucket_utilization(tables: List[defaultdict], n_data: int) -> dict:
    all_sizes = []
    total_buckets = 0
    for t in tables:
        sizes = [len(v) for v in t.values()]
        all_sizes.extend(sizes)
        total_buckets += len(t)
    arr = np.array(all_sizes, dtype=float)
    sorted_arr = np.sort(arr)
    cum = np.cumsum(sorted_arr)
    n = len(arr)
    gini = 0.0
    if n > 0 and cum[-1] > 0:
        gini = float((2 * np.sum((np.arange(1, n + 1)) * sorted_arr)) / (n * cum[-1]) - 1)
    ideal_per_bucket = n_data / max(total_buckets, 1)
    load_balance = float(np.std(arr) / max(np.mean(arr), 1e-9))
    max_ratio = float(np.max(arr) / max(np.min(arr), 1))
    return {
        "total_buckets": total_buckets,
        "non_empty_ratio": total_buckets / max(n_data, 1),
        "mean_bucket_size": float(np.mean(arr)) if n > 0 else 0.0,
        "std_bucket_size": float(np.std(arr)) if n > 0 else 0.0,
        "max_bucket_size": int(np.max(arr)) if n > 0 else 0,
        "min_bucket_size": int(np.min(arr)) if n > 0 else 0,
        "gini": gini,
        "load_balance_cv": load_balance,
        "max_min_ratio": max_ratio,
    }


class LSH:
    def __init__(
        self,
        dim: int,
        num_tables: int = 10,
        num_hashes: int = 8,
        seed: Optional[int] = None,
    ):
        self.dim = dim
        self.num_tables = num_tables
        self.num_hashes = num_hashes
        self.rng = np.random.RandomState(seed)
        self.hyperplanes = [
            self.rng.randn(num_hashes, dim) for _ in range(num_tables)
        ]
        self.tables: List[defaultdict] = [defaultdict(list) for _ in range(num_tables)]
        self.data_points: Optional[np.ndarray] = None

    def _hash(self, point: np.ndarray, table_idx: int) -> Tuple[int, ...]:
        projections = self.hyperplanes[table_idx] @ point
        bits = (projections > 0).astype(int)
        return tuple(bits.tolist())

    def build(self, data: np.ndarray):
        self.data_points = data.copy()
        n = data.shape[0]
        for t in range(self.num_tables):
            for i in range(n):
                key = self._hash(data[i], t)
                self.tables[t][key].append(i)

    def add_points(self, new_data: np.ndarray):
        if self.data_points is None:
            self.build(new_data)
            return
        start_idx = len(self.data_points)
        self.data_points = np.vstack([self.data_points, new_data])
        for t in range(self.num_tables):
            for i in range(len(new_data)):
                key = self._hash(new_data[i], t)
                self.tables[t][key].append(start_idx + i)

    def query(self, point: np.ndarray, max_candidates: Optional[int] = None) -> Set[int]:
        candidates: Set[int] = set()
        for t in range(self.num_tables):
            key = self._hash(point, t)
            if key in self.tables[t]:
                candidates.update(self.tables[t][key])
        if max_candidates is not None and len(candidates) > max_candidates:
            dists = np.linalg.norm(self.data_points[list(candidates)] - point, axis=1)
            sorted_idx = np.argsort(dists)[:max_candidates]
            candidates = set(np.array(list(candidates))[sorted_idx].tolist())
        return candidates

    def brute_force_knn(
        self, point: np.ndarray, k: int
    ) -> List[int]:
        dists = np.linalg.norm(self.data_points - point, axis=1)
        return np.argsort(dists)[:k].tolist()

    def recall_at_k(
        self,
        queries: np.ndarray,
        k: int,
        max_candidates: Optional[int] = None,
    ) -> float:
        total_recall = 0.0
        for q in queries:
            true_nn = set(self.brute_force_knn(q, k))
            candidates = self.query(q, max_candidates)
            hits = len(true_nn & candidates)
            total_recall += hits / k
        return total_recall / len(queries)

    def evaluate_detailed(
        self,
        queries: np.ndarray,
        k: int,
        max_candidates: Optional[int] = None,
    ) -> dict:
        recalls = []
        candidate_counts = []
        for q in queries:
            true_nn = set(self.brute_force_knn(q, k))
            candidates = self.query(q, max_candidates)
            hits = len(true_nn & candidates)
            recalls.append(hits / k)
            candidate_counts.append(len(candidates))
        result = {
            "mean_recall": float(np.mean(recalls)),
            "std_recall": float(np.std(recalls)),
            "min_recall": float(np.min(recalls)),
            "max_recall": float(np.max(recalls)),
            "mean_candidates": float(np.mean(candidate_counts)),
            "min_candidates": int(np.min(candidate_counts)),
            "max_candidates": int(np.max(candidate_counts)),
        }
        result.update(bucket_utilization(self.tables, len(self.data_points)))
        return result

    def _serialize_state(self) -> Dict[str, Any]:
        tables_as_dicts = []
        for t in self.tables:
            tables_as_dicts.append({k: list(v) for k, v in t.items()})
        state = {
            "dim": self.dim,
            "num_tables": self.num_tables,
            "num_hashes": self.num_hashes,
            "hyperplanes": [h.astype(np.float32) for h in self.hyperplanes],
            "tables": tables_as_dicts,
            "data_points": self.data_points,
        }
        return state

    def _deserialize_state(self, state: Dict[str, Any]):
        self.dim = state["dim"]
        self.num_tables = state["num_tables"]
        self.num_hashes = state["num_hashes"]
        self.hyperplanes = [h.astype(np.float64) for h in state["hyperplanes"]]
        self.tables = []
        for t_dict in state["tables"]:
            dd = defaultdict(list)
            for k, v in t_dict.items():
                dd[k] = v
            self.tables.append(dd)
        self.data_points = state["data_points"]

    def save(self, path: str):
        state = self._serialize_state()
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "LSH":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls.__new__(cls)
        obj._deserialize_state(state)
        return obj

    def benchmark(
        self,
        queries: np.ndarray,
        k: int,
        repeat: int = 3,
    ) -> dict:
        n_q = len(queries)

        lsh_times = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            for q in queries:
                self.query(q)
            lsh_times.append(time.perf_counter() - t0)
        lsh_time = min(lsh_times)

        bf_times = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            for q in queries:
                self.brute_force_knn(q, k)
            bf_times.append(time.perf_counter() - t0)
        bf_time = min(bf_times)

        recall = self.recall_at_k(queries, k)
        speedup = bf_time / max(lsh_time, 1e-9)

        return {
            "lsh_time_s": lsh_time,
            "bf_time_s": bf_time,
            "speedup": speedup,
            "mean_recall": recall,
            "queries_per_sec_lsh": n_q / max(lsh_time, 1e-9),
            "queries_per_sec_bf": n_q / max(bf_time, 1e-9),
        }


class MultiProbeLSH(LSH):
    def __init__(
        self,
        dim: int,
        num_tables: int = 10,
        num_hashes: int = 8,
        num_probes: int = 5,
        seed: Optional[int] = None,
    ):
        super().__init__(dim=dim, num_tables=num_tables, num_hashes=num_hashes, seed=seed)
        self.num_probes = num_probes

    def _generate_perturbations(self, projections: np.ndarray, num_probes: int) -> List[Tuple[int, ...]]:
        distances = np.abs(projections)
        sorted_indices = np.argsort(distances)
        base_bits = (projections > 0).astype(int)
        perturbations = []
        for num_flips in range(1, min(num_probes + 1, len(projections) + 1)):
            flip_combos = list(combinations(sorted_indices[: num_probes * 2], num_flips))
            flip_combos.sort(key=lambda combo: sum(distances[i] for i in combo))
            perturbations.extend(flip_combos)
        perturbations.sort(key=lambda combo: sum(distances[i] for i in combo))
        perturbations = perturbations[:num_probes]
        result = []
        for flip_set in perturbations:
            perturbed = base_bits.copy()
            for idx in flip_set:
                perturbed[idx] = 1 - perturbed[idx]
            result.append(tuple(perturbed.tolist()))
        return result

    def query(self, point: np.ndarray, max_candidates: Optional[int] = None) -> Set[int]:
        candidates: Set[int] = set()
        for t in range(self.num_tables):
            projections = self.hyperplanes[t] @ point
            base_key = tuple((projections > 0).astype(int).tolist())
            if base_key in self.tables[t]:
                candidates.update(self.tables[t][base_key])
            perturbed_keys = self._generate_perturbations(projections, self.num_probes)
            for key in perturbed_keys:
                if key in self.tables[t]:
                    candidates.update(self.tables[t][key])
        if max_candidates is not None and len(candidates) > max_candidates:
            dists = np.linalg.norm(self.data_points[list(candidates)] - point, axis=1)
            sorted_idx = np.argsort(dists)[:max_candidates]
            candidates = set(np.array(list(candidates))[sorted_idx].tolist())
        return candidates

    def _serialize_state(self) -> Dict[str, Any]:
        state = super()._serialize_state()
        state["num_probes"] = self.num_probes
        return state

    def _deserialize_state(self, state: Dict[str, Any]):
        super()._deserialize_state(state)
        self.num_probes = state["num_probes"]


class PCAHashLSH(LSH):
    def __init__(
        self,
        dim: int,
        num_tables: int = 10,
        num_hashes: int = 8,
        seed: Optional[int] = None,
    ):
        self.dim = dim
        self.num_tables = num_tables
        self.num_hashes = num_hashes
        self.rng = np.random.RandomState(seed)
        self.hyperplanes = []
        self.tables: List[defaultdict] = []
        self.data_points: Optional[np.ndarray] = None
        self.means: Optional[np.ndarray] = None

    def build(self, data: np.ndarray):
        self.data_points = data.copy()
        self.means = np.mean(data, axis=0)
        centered = data - self.means
        cov = centered.T @ centered / len(data)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        top_indices = np.argsort(eigenvalues)[::-1]
        top_eigvecs = eigenvectors[:, top_indices[:max(self.num_hashes, 64)]]
        self.hyperplanes = []
        self.tables = []
        for t in range(self.num_tables):
            indices = self.rng.choice(top_eigvecs.shape[1], size=self.num_hashes, replace=False)
            selected = top_eigvecs[:, indices].T
            selected = selected / np.linalg.norm(selected, axis=1, keepdims=True)
            self.hyperplanes.append(selected)
            self.tables.append(defaultdict(list))
        n = data.shape[0]
        for t in range(self.num_tables):
            for i in range(n):
                key = self._hash(centered[i], t)
                self.tables[t][key].append(i)

    def add_points(self, new_data: np.ndarray):
        if self.data_points is None:
            self.build(new_data)
            return
        start_idx = len(self.data_points)
        self.data_points = np.vstack([self.data_points, new_data])
        centered_new = new_data - self.means
        for t in range(self.num_tables):
            for i in range(len(new_data)):
                key = self._hash(centered_new[i], t)
                self.tables[t][key].append(start_idx + i)

    def query(self, point: np.ndarray, max_candidates: Optional[int] = None) -> Set[int]:
        centered = point - self.means
        candidates: Set[int] = set()
        for t in range(self.num_tables):
            key = self._hash(centered, t)
            if key in self.tables[t]:
                candidates.update(self.tables[t][key])
        if max_candidates is not None and len(candidates) > max_candidates:
            dists = np.linalg.norm(self.data_points[list(candidates)] - point, axis=1)
            sorted_idx = np.argsort(dists)[:max_candidates]
            candidates = set(np.array(list(candidates))[sorted_idx].tolist())
        return candidates

    def _serialize_state(self) -> Dict[str, Any]:
        state = super()._serialize_state()
        state["means"] = self.means
        return state

    def _deserialize_state(self, state: Dict[str, Any]):
        super()._deserialize_state(state)
        self.means = state["means"]


class PCAHashMultiProbeLSH(MultiProbeLSH):
    def __init__(
        self,
        dim: int,
        num_tables: int = 10,
        num_hashes: int = 8,
        num_probes: int = 5,
        seed: Optional[int] = None,
    ):
        self.dim = dim
        self.num_tables = num_tables
        self.num_hashes = num_hashes
        self.num_probes = num_probes
        self.rng = np.random.RandomState(seed)
        self.hyperplanes = []
        self.tables: List[defaultdict] = []
        self.data_points: Optional[np.ndarray] = None
        self.means: Optional[np.ndarray] = None

    def build(self, data: np.ndarray):
        self.data_points = data.copy()
        self.means = np.mean(data, axis=0)
        centered = data - self.means
        cov = centered.T @ centered / len(data)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        top_indices = np.argsort(eigenvalues)[::-1]
        top_eigvecs = eigenvectors[:, top_indices[:max(self.num_hashes, 64)]]
        self.hyperplanes = []
        self.tables = []
        for t in range(self.num_tables):
            indices = self.rng.choice(top_eigvecs.shape[1], size=self.num_hashes, replace=False)
            selected = top_eigvecs[:, indices].T
            selected = selected / np.linalg.norm(selected, axis=1, keepdims=True)
            self.hyperplanes.append(selected)
            self.tables.append(defaultdict(list))
        n = data.shape[0]
        for t in range(self.num_tables):
            for i in range(n):
                key = self._hash(centered[i], t)
                self.tables[t][key].append(i)

    def add_points(self, new_data: np.ndarray):
        if self.data_points is None:
            self.build(new_data)
            return
        start_idx = len(self.data_points)
        self.data_points = np.vstack([self.data_points, new_data])
        centered_new = new_data - self.means
        for t in range(self.num_tables):
            for i in range(len(new_data)):
                key = self._hash(centered_new[i], t)
                self.tables[t][key].append(start_idx + i)

    def query(self, point: np.ndarray, max_candidates: Optional[int] = None) -> Set[int]:
        centered = point - self.means
        candidates: Set[int] = set()
        for t in range(self.num_tables):
            projections = self.hyperplanes[t] @ centered
            base_key = tuple((projections > 0).astype(int).tolist())
            if base_key in self.tables[t]:
                candidates.update(self.tables[t][base_key])
            perturbed_keys = self._generate_perturbations(projections, self.num_probes)
            for key in perturbed_keys:
                if key in self.tables[t]:
                    candidates.update(self.tables[t][key])
        if max_candidates is not None and len(candidates) > max_candidates:
            dists = np.linalg.norm(self.data_points[list(candidates)] - point, axis=1)
            sorted_idx = np.argsort(dists)[:max_candidates]
            candidates = set(np.array(list(candidates))[sorted_idx].tolist())
        return candidates

    def _serialize_state(self) -> Dict[str, Any]:
        state = super()._serialize_state()
        state["means"] = self.means
        return state

    def _deserialize_state(self, state: Dict[str, Any]):
        super()._deserialize_state(state)
        self.means = state["means"]


def generate_imbalanced_data(n: int, dim: int, n_clusters: int = 5, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_clusters, dim) * 10
    sizes = rng.dirichlet(np.array([0.5] * n_clusters)) * n
    sizes = sizes.astype(int)
    sizes[-1] = n - sizes[:-1].sum()
    data_parts = []
    for i in range(n_clusters):
        cluster_data = rng.randn(sizes[i], dim) + centers[i]
        data_parts.append(cluster_data)
    data = np.vstack(data_parts).astype(np.float32)
    perm = rng.permutation(n)
    return data[perm]


if __name__ == "__main__":
    np.random.seed(42)
    import os

    dim = 128
    n_data = 30000
    n_queries = 50
    k = 10

    data = np.random.randn(n_data, dim).astype(np.float32)
    queries = np.random.randn(n_queries, dim).astype(np.float32)

    print("=" * 90)
    print("  LSH Feature Demo: Save/Load, Incremental Insert, Speed Benchmark")
    print(f"  {n_data} points, {dim}D, {n_queries} queries, k={k}")
    print("=" * 90)

    # ── 1. Speed benchmark ──
    print("\n" + "=" * 90)
    print("  [1] Speed Benchmark: LSH vs Brute-Force")
    print("=" * 90)

    models = [
        ("Baseline LSH (T=10,H=8)", LSH(dim=dim, num_tables=10, num_hashes=8, seed=42)),
        ("Multi-Probe LSH (T=10,H=8,P=5)", MultiProbeLSH(dim=dim, num_tables=10, num_hashes=8, num_probes=5, seed=42)),
        ("PCA-Hash LSH (T=10,H=8)", PCAHashLSH(dim=dim, num_tables=10, num_hashes=8, seed=42)),
        ("PCA+MultiProbe (T=10,H=8,P=5)", PCAHashMultiProbeLSH(dim=dim, num_tables=10, num_hashes=8, num_probes=5, seed=42)),
    ]

    print(f"\n  {'Method':<35s} {'Recall':>7s} {'LSH(s)':>8s} {'BF(s)':>8s} {'Speedup':>8s} {'QPS_LSH':>10s} {'QPS_BF':>10s}")
    print("  " + "-" * 90)

    for name, model in models:
        model.build(data)
        bench = model.benchmark(queries, k=k, repeat=3)
        print(
            f"  {name:<35s} "
            f"{bench['mean_recall']:>7.4f} "
            f"{bench['lsh_time_s']:>8.4f} "
            f"{bench['bf_time_s']:>8.4f} "
            f"{bench['speedup']:>7.1f}x "
            f"{bench['queries_per_sec_lsh']:>10.1f} "
            f"{bench['queries_per_sec_bf']:>10.1f}"
        )

    # ── 2. Save / Load ──
    print("\n" + "=" * 90)
    print("  [2] Save / Load Serialization")
    print("=" * 90)

    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_lsh_index.pkl")

    original = MultiProbeLSH(dim=dim, num_tables=10, num_hashes=8, num_probes=5, seed=42)
    original.build(data)
    recall_before = original.recall_at_k(queries, k=k)
    original.save(save_path)
    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)

    loaded = MultiProbeLSH.load(save_path)
    recall_after = loaded.recall_at_k(queries, k=k)

    print(f"  Saved index to: {save_path}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Recall before save: {recall_before:.4f}")
    print(f"  Recall after load:  {recall_after:.4f}")
    print(f"  Recall match: {'YES' if abs(recall_before - recall_after) < 1e-6 else 'NO'}")

    bench_loaded = loaded.benchmark(queries, k=k, repeat=2)
    print(f"  Loaded index speedup: {bench_loaded['speedup']:.1f}x")

    if os.path.exists(save_path):
        os.remove(save_path)

    # ── 3. Incremental Insert ──
    print("\n" + "=" * 90)
    print("  [3] Incremental Insert (Dynamic Addition)")
    print("=" * 90)

    base_data = data[:20000]
    batch1 = data[20000:25000]
    batch2 = data[25000:30000]

    lsh_inc = LSH(dim=dim, num_tables=10, num_hashes=8, seed=42)
    lsh_inc.build(base_data)
    print(f"  Initial index: {len(lsh_inc.data_points)} points")

    lsh_inc.add_points(batch1)
    print(f"  After +5000:   {len(lsh_inc.data_points)} points")

    lsh_inc.add_points(batch2)
    print(f"  After +5000:   {len(lsh_inc.data_points)} points")

    lsh_full = LSH(dim=dim, num_tables=10, num_hashes=8, seed=42)
    lsh_full.build(data)

    recall_inc = lsh_inc.recall_at_k(queries, k=k)
    recall_full = lsh_full.recall_at_k(queries, k=k)
    print(f"\n  Recall (incremental build): {recall_inc:.4f}")
    print(f"  Recall (one-shot build):    {recall_full:.4f}")
    print(f"  Recall match: {'YES' if abs(recall_inc - recall_full) < 1e-6 else 'NO'}")

    bench_inc = lsh_inc.benchmark(queries, k=k, repeat=3)
    print(f"  Speedup after incremental insert: {bench_inc['speedup']:.1f}x")

    # ── 4. PCA variant incremental + save/load ──
    print("\n" + "=" * 90)
    print("  [4] PCA-Hash LSH: Incremental + Save/Load")
    print("=" * 90)

    pca = PCAHashLSH(dim=dim, num_tables=10, num_hashes=8, seed=42)
    pca.build(base_data)
    pca.add_points(batch1)
    pca.add_points(batch2)

    pca_full = PCAHashLSH(dim=dim, num_tables=10, num_hashes=8, seed=42)
    pca_full.build(data)

    recall_pca_inc = pca.recall_at_k(queries, k=k)
    recall_pca_full = pca_full.recall_at_k(queries, k=k)
    print(f"  PCA Recall (incremental): {recall_pca_inc:.4f}")
    print(f"  PCA Recall (one-shot):    {recall_pca_full:.4f}")

    pca.save(save_path)
    pca_loaded = PCAHashLSH.load(save_path)
    recall_pca_loaded = pca_loaded.recall_at_k(queries, k=k)
    print(f"  PCA Recall after load:    {recall_pca_loaded:.4f}")
    print(f"  Save/Load match: {'YES' if abs(recall_pca_inc - recall_pca_loaded) < 1e-6 else 'NO'}")

    bench_pca = pca.benchmark(queries, k=k, repeat=3)
    print(f"  PCA-Hash speedup: {bench_pca['speedup']:.1f}x")

    if os.path.exists(save_path):
        os.remove(save_path)

    # ── 5. Scalability: speedup vs data size ──
    print("\n" + "=" * 90)
    print("  [5] Scalability: Speedup vs Data Size")
    print("=" * 90)

    data_sizes = [10000, 30000, 50000]
    print(f"\n  {'N':>8s} {'LSH(s)':>8s} {'BF(s)':>8s} {'Speedup':>8s} {'Recall':>7s}")
    print("  " + "-" * 45)

    for n in data_sizes:
        scale_data = np.random.randn(n, dim).astype(np.float32)
        scale_queries = np.random.randn(50, dim).astype(np.float32)
        lsh_s = LSH(dim=dim, num_tables=10, num_hashes=8, seed=42)
        lsh_s.build(scale_data)
        b = lsh_s.benchmark(scale_queries, k=k, repeat=2)
        print(f"  {n:>8d} {b['lsh_time_s']:>8.4f} {b['bf_time_s']:>8.4f} {b['speedup']:>7.1f}x {b['mean_recall']:>7.4f}")

    print("\n" + "=" * 90)
    print("  All features verified successfully.")
    print("=" * 90)

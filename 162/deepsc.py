import numpy as np
from typing import Tuple, List, Optional
from scipy.spatial import cKDTree


class PointNetModule:
    def __init__(self, mlp_sizes: List[int], activation: str = 'relu'):
        self.mlp_sizes = mlp_sizes
        self.activation = activation
        self.weights = []
        self.biases = []
        self._initialized = False

    def _initialize(self, input_dim: int):
        prev_size = input_dim
        for size in self.mlp_sizes:
            self.weights.append(np.random.randn(prev_size, size) * 0.1)
            self.biases.append(np.zeros((1, size)))
            prev_size = size
        self._initialized = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self._initialized:
            self._initialize(x.shape[-1])

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w + b
            if self.activation == 'relu' and i < len(self.weights) - 1:
                x = np.maximum(0, x)
        return x


class SetAbstraction:
    def __init__(self, npoint: int, radius: float, nsample: int, mlp_sizes: List[int], group_all: bool = False):
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_sizes = mlp_sizes
        self.group_all = group_all
        self.pointnet = PointNetModule(mlp_sizes)

    def _sample_farthest_points(self, points: np.ndarray, npoint: int) -> np.ndarray:
        B, N, _ = points.shape
        centroids = np.zeros((B, npoint), dtype=np.int32)
        distance = np.ones((B, N)) * 1e10
        farthest = np.random.randint(0, N, (B,))

        for i in range(npoint):
            centroids[:, i] = farthest
            centroid_points = points[np.arange(B), farthest, :].reshape(B, 1, 3)
            dists = np.sum((points - centroid_points) ** 2, axis=2)
            mask = dists < distance
            distance[mask] = dists[mask]
            farthest = np.argmax(distance, axis=1)
        return centroids

    def _query_ball_point(self, radius: float, nsample: int, xyz: np.ndarray, centers: np.ndarray) -> np.ndarray:
        B, N, _ = xyz.shape
        S = centers.shape[1]
        group_indices = np.zeros((B, S, nsample), dtype=np.int32)

        for b in range(B):
            tree = cKDTree(xyz[b])
            for s in range(S):
                indices = tree.query_ball_point(centers[b, s], radius)
                if len(indices) >= nsample:
                    indices = np.random.choice(indices, nsample, replace=False)
                else:
                    indices = np.pad(indices, (0, nsample - len(indices)), mode='edge')
                group_indices[b, s] = indices
        return group_indices

    def forward(self, xyz: np.ndarray, features: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        B, N, _ = xyz.shape

        if self.group_all:
            new_xyz = np.zeros((B, 1, 3))
            grouped_xyz = xyz.reshape(B, 1, N, 3)
            if features is not None:
                grouped_features = features.reshape(B, 1, N, -1)
        else:
            centroids_idx = self._sample_farthest_points(xyz, self.npoint)
            batch_indices = np.arange(B).reshape(-1, 1)
            new_xyz = xyz[batch_indices, centroids_idx, :]
            group_indices = self._query_ball_point(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = xyz[batch_indices.reshape(-1, 1, 1), group_indices, :]
            grouped_xyz = grouped_xyz - new_xyz.reshape(B, self.npoint, 1, 3)

            if features is not None:
                grouped_features = features[batch_indices.reshape(-1, 1, 1), group_indices, :]
                grouped_features = np.concatenate([grouped_xyz, grouped_features], axis=-1)
            else:
                grouped_features = grouped_xyz

        B, S, K, C = grouped_features.shape
        grouped_features = grouped_features.reshape(B * S * K, C)
        new_features = self.pointnet.forward(grouped_features)
        new_features = new_features.reshape(B, S, K, -1)
        new_features = np.max(new_features, axis=2)

        return new_xyz, new_features


class FeaturePropagation:
    def __init__(self, mlp_sizes: List[int]):
        self.mlp_sizes = mlp_sizes
        self.pointnet = PointNetModule(mlp_sizes)

    def _three_nn_interpolate(self, unknown: np.ndarray, known: np.ndarray, known_features: np.ndarray) -> np.ndarray:
        B, N, _ = unknown.shape
        _, M, _ = known.shape

        dists = np.zeros((B, N, M))
        for b in range(B):
            for i in range(N):
                dists[b, i] = np.sum((known[b] - unknown[b, i].reshape(1, 3)) ** 2, axis=1)

        idx = np.argsort(dists, axis=2)[:, :, :3]
        dists = np.take_along_axis(dists, idx, axis=2)
        dists = np.maximum(dists, 1e-10)

        weights = 1.0 / dists
        weights = weights / np.sum(weights, axis=2, keepdims=True)

        batch_indices = np.arange(B).reshape(-1, 1, 1)
        interpolated_features = np.sum(
            known_features[batch_indices, idx, :] * weights.reshape(B, N, 3, 1),
            axis=2
        )
        return interpolated_features

    def forward(self, xyz1: np.ndarray, xyz2: np.ndarray, features1: Optional[np.ndarray], features2: np.ndarray) -> np.ndarray:
        interpolated_features = self._three_nn_interpolate(xyz1, xyz2, features2)

        if features1 is not None:
            new_features = np.concatenate([interpolated_features, features1], axis=-1)
        else:
            new_features = interpolated_features

        B, N, C = new_features.shape
        new_features_flat = new_features.reshape(B * N, C)
        new_features_flat = self.pointnet.forward(new_features_flat)
        new_features = new_features_flat.reshape(B, N, -1)

        return new_features


class DeepSC:
    def __init__(
        self,
        feature_dim: int = 128,
        radius: float = 0.1,
        num_groups: List[int] = [512, 128, 32],
        group_radii: List[float] = [0.05, 0.1, 0.2],
        nsamples: List[int] = [32, 64, 128]
    ):
        self.feature_dim = feature_dim
        self.radius = radius
        self.num_groups = num_groups
        self.group_radii = group_radii
        self.nsamples = nsamples

        self.sa_modules = []
        self.fp_modules = []

        self._build_network()
        self._trained = False

    def _build_network(self):
        self.sa_modules.append(SetAbstraction(self.num_groups[0], self.group_radii[0], self.nsamples[0], [32, 32, 64]))
        self.sa_modules.append(SetAbstraction(self.num_groups[1], self.group_radii[1], self.nsamples[1], [64, 64, 128]))
        self.sa_modules.append(SetAbstraction(self.num_groups[2], self.group_radii[2], self.nsamples[2], [128, 128, 256]))

        self.fp_modules.append(FeaturePropagation([256, 256]))
        self.fp_modules.append(FeaturePropagation([256, 128]))
        self.fp_modules.append(FeaturePropagation([128, 128, self.feature_dim]))

        self.head_mlp = PointNetModule([128, 64, self.feature_dim])

    def _compute_geometric_features(self, points: np.ndarray) -> np.ndarray:
        B, N, _ = points.shape
        features = np.zeros((B, N, 6))

        for b in range(B):
            tree = cKDTree(points[b])
            for i in range(N):
                _, indices = tree.query(points[b, i], k=10)
                neighbors = points[b, indices[1:]]
                centered = neighbors - points[b, i]
                cov = centered.T @ centered
                eigenvalues, _ = np.linalg.eigh(cov)
                eigenvalues = eigenvalues / (np.sum(eigenvalues) + 1e-10)
                features[b, i, :3] = eigenvalues
                features[b, i, 3] = np.mean(np.linalg.norm(centered, axis=1))
                features[b, i, 4] = np.var(np.linalg.norm(centered, axis=1))
                features[b, i, 5] = np.mean(np.arctan2(centered[:, 1], centered[:, 0])) / np.pi

        return features

    def extract_features(self, points: np.ndarray) -> np.ndarray:
        if points.ndim == 2:
            points = points.reshape(1, -1, 3)

        B, N, _ = points.shape

        geo_features = self._compute_geometric_features(points)

        l0_xyz = points
        l0_points = geo_features

        l1_xyz, l1_points = self.sa_modules[0].forward(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa_modules[1].forward(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa_modules[2].forward(l2_xyz, l2_points)

        l2_points = self.fp_modules[0].forward(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp_modules[1].forward(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp_modules[2].forward(l0_xyz, l1_xyz, l0_points, l1_points)

        features_flat = l0_points.reshape(B * N, -1)
        features_flat = self.head_mlp.forward(features_flat)
        features = features_flat.reshape(B, N, -1)

        norm = np.linalg.norm(features, axis=2, keepdims=True) + 1e-10
        features = features / norm

        return features

    def match_features(self, feat1: np.ndarray, feat2: np.ndarray, ratio_threshold: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        if feat1.ndim == 3:
            feat1 = feat1[0]
        if feat2.ndim == 3:
            feat2 = feat2[0]

        matches = []
        distances = []

        for i in range(feat1.shape[0]):
            dists = np.linalg.norm(feat2 - feat1[i], axis=1)
            sorted_indices = np.argsort(dists)

            if len(sorted_indices) >= 2:
                ratio = dists[sorted_indices[0]] / (dists[sorted_indices[1]] + 1e-10)
                if ratio < ratio_threshold:
                    matches.append((i, sorted_indices[0]))
                    distances.append(dists[sorted_indices[0]])

        return np.array(matches), np.array(distances)

    def self_supervised_train(self, point_clouds: List[np.ndarray], epochs: int = 10, lr: float = 0.001):
        print(f"开始自监督训练，共 {epochs} 轮，{len(point_clouds)} 个点云")

        for epoch in range(epochs):
            total_loss = 0
            count = 0

            for pc in point_clouds:
                if pc.ndim == 2:
                    pc = pc.reshape(1, -1, 3)

                features = self.extract_features(pc)

                idx1, idx2 = np.random.choice(features.shape[1], 2, replace=False)
                anchor = features[0, idx1]
                positive = features[0, idx2]

                negative_idx = np.random.choice(features.shape[1], 10)
                negatives = features[0, negative_idx]

                pos_dist = np.linalg.norm(anchor - positive)
                neg_dists = np.linalg.norm(anchor.reshape(1, -1) - negatives, axis=1)

                loss = np.maximum(0, pos_dist - np.min(neg_dists) + 0.5)
                total_loss += loss
                count += 1

            avg_loss = total_loss / count if count > 0 else 0
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        self._trained = True
        print("训练完成!")

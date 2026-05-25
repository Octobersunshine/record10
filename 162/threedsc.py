import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple, List, Optional


class ThreeDSC:
    def __init__(
        self,
        radius: float = 0.1,
        num_bins_r: int = 5,
        num_bins_azim: int = 6,
        num_bins_polar: int = 3,
        min_radius: float = 0.01,
        use_log: bool = True
    ):
        self.radius = radius
        self.num_bins_r = num_bins_r
        self.num_bins_azim = num_bins_azim
        self.num_bins_polar = num_bins_polar
        self.min_radius = min_radius
        self.use_log = use_log
        self.descriptor_size = num_bins_r * num_bins_azim * num_bins_polar

    def _estimate_normals(self, points: np.ndarray, k: int = 10) -> np.ndarray:
        n_points = points.shape[0]
        normals = np.zeros((n_points, 3))
        tree = cKDTree(points)

        for i in range(n_points):
            _, indices = tree.query(points[i], k=k + 1)
            indices = indices[1:]
            neighbors = points[indices]
            centered = neighbors - points[i]
            cov = centered.T @ centered
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            normals[i] = eigenvectors[:, 0]

            if normals[i].dot(-points[i]) > 0:
                normals[i] = -normals[i]

        return normals

    def _compute_lrf(self, point: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
        centered = neighbors - point
        cov = centered.T @ centered
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        x_axis = eigenvectors[:, 2]
        y_axis = eigenvectors[:, 1]
        z_axis = eigenvectors[:, 0]

        x_axis = self._disambiguate_axis_sign(x_axis, centered)
        y_axis = self._disambiguate_axis_sign(y_axis, centered)
        z_axis = self._disambiguate_axis_sign(z_axis, centered)

        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-10)
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-10)

        lrf = np.column_stack([x_axis, y_axis, z_axis])

        if np.linalg.det(lrf) < 0:
            lrf[:, 0] = -lrf[:, 0]

        return lrf

    def _disambiguate_axis_sign(self, axis: np.ndarray, centered_neighbors: np.ndarray) -> np.ndarray:
        projections = centered_neighbors @ axis
        weights = 1.0 / (np.linalg.norm(centered_neighbors, axis=1) + 1e-10)
        weighted_sum = np.sum(projections * weights)
        if weighted_sum < 0:
            axis = -axis
        return axis

    def _transform_to_local(self, points: np.ndarray, center: np.ndarray, lrf: np.ndarray) -> np.ndarray:
        centered = points - center
        local_points = centered @ lrf
        return local_points

    def _compute_histogram(self, local_points: np.ndarray) -> np.ndarray:
        n_points = local_points.shape[0]
        histogram = np.zeros((self.num_bins_r, self.num_bins_azim, self.num_bins_polar))

        if n_points == 0:
            return histogram.flatten()

        x = local_points[:, 0]
        y = local_points[:, 1]
        z = local_points[:, 2]

        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        valid = r > self.min_radius

        if not np.any(valid):
            return histogram.flatten()

        x_valid = x[valid]
        y_valid = y[valid]
        z_valid = z[valid]
        r_valid = r[valid]

        if self.use_log:
            r_normalized = np.log(r_valid / self.min_radius) / np.log(self.radius / self.min_radius)
        else:
            r_normalized = (r_valid - self.min_radius) / (self.radius - self.min_radius)

        r_normalized = np.clip(r_normalized, 0, 1 - 1e-10)

        theta = np.arctan2(y_valid, x_valid)
        theta = (theta + np.pi) / (2 * np.pi)

        cos_phi = z_valid / r_valid
        phi = np.arccos(np.clip(cos_phi, -1, 1))
        phi_normalized = phi / np.pi

        r_bin = (r_normalized * self.num_bins_r).astype(int)
        theta_bin = (theta * self.num_bins_azim).astype(int)
        phi_bin = (phi_normalized * self.num_bins_polar).astype(int)

        r_bin = np.clip(r_bin, 0, self.num_bins_r - 1)
        theta_bin = np.clip(theta_bin, 0, self.num_bins_azim - 1)
        phi_bin = np.clip(phi_bin, 0, self.num_bins_polar - 1)

        for i in range(len(r_bin)):
            histogram[r_bin[i], theta_bin[i], phi_bin[i]] += 1

        if np.sum(histogram) > 0:
            histogram = histogram / np.sum(histogram)

        return histogram.flatten()

    def compute_descriptor(self, points: np.ndarray, normals: Optional[np.ndarray] = None) -> np.ndarray:
        n_points = points.shape[0]
        descriptors = np.zeros((n_points, self.descriptor_size))
        tree = cKDTree(points)

        if normals is None:
            normals = self._estimate_normals(points)

        for i in range(n_points):
            point = points[i]
            indices = tree.query_ball_point(point, self.radius)
            indices = [idx for idx in indices if idx != i]

            if len(indices) < 3:
                continue

            neighbors = points[indices]
            lrf = self._compute_lrf(point, neighbors)
            local_points = self._transform_to_local(neighbors, point, lrf)
            descriptor = self._compute_histogram(local_points)
            descriptors[i] = descriptor

        return descriptors

    def match_descriptors(self, desc1: np.ndarray, desc2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        matches = []
        distances = []

        for i in range(desc1.shape[0]):
            if np.sum(desc1[i]) == 0:
                continue
            dists = np.linalg.norm(desc2 - desc1[i], axis=1)
            sorted_indices = np.argsort(dists)
            if len(sorted_indices) >= 2:
                ratio = dists[sorted_indices[0]] / (dists[sorted_indices[1]] + 1e-10)
                if ratio < 0.8:
                    matches.append((i, sorted_indices[0]))
                    distances.append(dists[sorted_indices[0]])

        return np.array(matches), np.array(distances)

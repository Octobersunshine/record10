import numpy as np
import unittest
from scipy.spatial import cKDTree
from threedsc import ThreeDSC


class TestThreeDSC(unittest.TestCase):
    def setUp(self):
        self.descriptor = ThreeDSC(
            radius=0.5,
            num_bins_r=3,
            num_bins_azim=4,
            num_bins_polar=2,
            use_log=True
        )

    def test_initialization(self):
        self.assertEqual(self.descriptor.radius, 0.5)
        self.assertEqual(self.descriptor.num_bins_r, 3)
        self.assertEqual(self.descriptor.num_bins_azim, 4)
        self.assertEqual(self.descriptor.num_bins_polar, 2)
        self.assertEqual(self.descriptor.descriptor_size, 3 * 4 * 2)

    def test_normal_estimation(self):
        np.random.seed(42)
        points = np.random.rand(50, 3)
        normals = self.descriptor._estimate_normals(points, k=10)
        self.assertEqual(normals.shape, (50, 3))
        norms = np.linalg.norm(normals, axis=1)
        self.assertTrue(np.allclose(norms, 1.0, atol=0.1))

    def test_lrf_computation(self):
        np.random.seed(42)
        center = np.array([0.0, 0.0, 0.0])
        neighbors = np.random.rand(20, 3) - 0.5
        lrf = self.descriptor._compute_lrf(center, neighbors)
        self.assertEqual(lrf.shape, (3, 3))
        det = np.linalg.det(lrf)
        self.assertGreater(det, 0)
        self.assertTrue(np.allclose(lrf.T @ lrf, np.eye(3), atol=0.1))

    def test_histogram_computation(self):
        np.random.seed(42)
        local_points = np.random.rand(100, 3) - 0.5
        histogram = self.descriptor._compute_histogram(local_points)
        self.assertEqual(histogram.shape, (self.descriptor.descriptor_size,))
        self.assertGreaterEqual(np.sum(histogram), 0)
        self.assertLessEqual(np.sum(histogram), 1 + 1e-10)

    def test_empty_histogram(self):
        local_points = np.array([]).reshape(0, 3)
        histogram = self.descriptor._compute_histogram(local_points)
        self.assertEqual(histogram.shape, (self.descriptor.descriptor_size,))
        self.assertTrue(np.all(histogram == 0))

    def test_descriptor_computation(self):
        np.random.seed(42)
        points = np.random.rand(30, 3)
        descriptors = self.descriptor.compute_descriptor(points)
        self.assertEqual(descriptors.shape, (30, self.descriptor.descriptor_size))
        self.assertTrue(np.all(descriptors >= 0))

    def test_descriptor_rotation_invariance(self):
        np.random.seed(42)
        sphere = self._generate_sphere(200, 1.0)
        angles = [np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2]

        desc_original = self.descriptor.compute_descriptor(sphere)

        for angle in angles:
            for axis_idx in range(3):
                if axis_idx == 0:
                    R = np.array([
                        [1, 0, 0],
                        [0, np.cos(angle), -np.sin(angle)],
                        [0, np.sin(angle), np.cos(angle)]
                    ])
                elif axis_idx == 1:
                    R = np.array([
                        [np.cos(angle), 0, np.sin(angle)],
                        [0, 1, 0],
                        [-np.sin(angle), 0, np.cos(angle)]
                    ])
                else:
                    R = np.array([
                        [np.cos(angle), -np.sin(angle), 0],
                        [np.sin(angle), np.cos(angle), 0],
                        [0, 0, 1]
                    ])

                points_rotated = sphere @ R.T
                desc_rotated = self.descriptor.compute_descriptor(points_rotated)

                mean_original = np.mean(desc_original, axis=0)
                mean_rotated = np.mean(desc_rotated, axis=0)
                correlation = np.corrcoef(mean_original, mean_rotated)[0, 1]
                self.assertGreater(correlation, 0.85,
                    f"Rotation invariance failed for angle {angle:.2f} around axis {axis_idx}. Correlation: {correlation:.3f}")

    def test_lrf_stability_under_rotation(self):
        np.random.seed(42)
        sphere = self._generate_sphere(100, 1.0)
        angle = np.pi / 3
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        sphere_rotated = sphere @ R.T

        tree = cKDTree(sphere)
        tree_rotated = cKDTree(sphere_rotated)

        point_idx = 50
        point = sphere[point_idx]
        point_rotated = sphere_rotated[point_idx]

        _, indices = tree.query(point, k=30)
        neighbors = sphere[indices]
        lrf = self.descriptor._compute_lrf(point, neighbors)

        _, indices_rotated = tree_rotated.query(point_rotated, k=30)
        neighbors_rotated = sphere_rotated[indices_rotated]
        lrf_rotated = self.descriptor._compute_lrf(point_rotated, neighbors_rotated)

        expected_lrf_rotated = R.T @ lrf
        diff = np.abs(lrf_rotated) - np.abs(expected_lrf_rotated)
        self.assertLess(np.mean(np.abs(diff)), 0.2)

    def test_matching(self):
        np.random.seed(42)
        desc1 = np.random.rand(50, 90)
        desc2 = np.random.rand(60, 90)
        desc2[:10] = desc1[:10] + np.random.randn(10, 90) * 0.01

        matches, distances = self.descriptor.match_descriptors(desc1, desc2)
        self.assertIsInstance(matches, np.ndarray)
        self.assertIsInstance(distances, np.ndarray)
        if len(matches) > 0:
            self.assertEqual(matches.shape[1], 2)

    def test_different_shapes_discrimination(self):
        np.random.seed(42)

        sphere = self._generate_sphere(100, 1.0)
        cube = self._generate_cube(100, 0.8)

        desc_sphere = self.descriptor.compute_descriptor(sphere)
        desc_cube = self.descriptor.compute_descriptor(cube)

        mean_sphere = np.mean(desc_sphere, axis=0)
        mean_cube = np.mean(desc_cube, axis=0)

        distance = np.linalg.norm(mean_sphere - mean_cube)
        self.assertGreater(distance, 0.05)

    def _generate_sphere(self, n_points, radius):
        phi = np.random.uniform(0, np.pi, n_points)
        theta = np.random.uniform(0, 2 * np.pi, n_points)
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        return np.column_stack([x, y, z])

    def _generate_cube(self, n_points, size):
        points = []
        for _ in range(n_points):
            face = np.random.randint(0, 6)
            if face == 0:
                x, y, z = np.random.uniform(-size, size), np.random.uniform(-size, size), size
            elif face == 1:
                x, y, z = np.random.uniform(-size, size), np.random.uniform(-size, size), -size
            elif face == 2:
                x, y, z = np.random.uniform(-size, size), size, np.random.uniform(-size, size)
            elif face == 3:
                x, y, z = np.random.uniform(-size, size), -size, np.random.uniform(-size, size)
            elif face == 4:
                x, y, z = size, np.random.uniform(-size, size), np.random.uniform(-size, size)
            else:
                x, y, z = -size, np.random.uniform(-size, size), np.random.uniform(-size, size)
            points.append([x, y, z])
        return np.array(points)


if __name__ == '__main__':
    unittest.main(verbosity=2)

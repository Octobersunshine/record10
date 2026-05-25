import numpy as np
import unittest
from deepsc import DeepSC, PointNetModule, SetAbstraction, FeaturePropagation


class TestPointNetModule(unittest.TestCase):
    def test_initialization(self):
        module = PointNetModule([32, 64, 128])
        self.assertEqual(len(module.weights), 0)
        self.assertEqual(len(module.biases), 0)
        self.assertFalse(module._initialized)

    def test_forward(self):
        module = PointNetModule([32, 64])
        x = np.random.randn(100, 16)
        output = module.forward(x)
        self.assertEqual(output.shape, (100, 64))
        self.assertTrue(module._initialized)


class TestSetAbstraction(unittest.TestCase):
    def test_farthest_point_sampling(self):
        sa = SetAbstraction(16, 0.5, 32, [64, 128])
        points = np.random.randn(2, 100, 3)
        centroids = sa._sample_farthest_points(points, 16)
        self.assertEqual(centroids.shape, (2, 16))

    def test_ball_query(self):
        sa = SetAbstraction(16, 0.5, 32, [64, 128])
        points = np.random.randn(2, 100, 3)
        centroids = points[:, :16, :]
        indices = sa._query_ball_point(10.0, 32, points, centroids)
        self.assertEqual(indices.shape, (2, 16, 32))

    def test_forward(self):
        sa = SetAbstraction(16, 0.5, 8, [32, 64])
        xyz = np.random.randn(2, 64, 3)
        features = np.random.randn(2, 64, 6)
        new_xyz, new_features = sa.forward(xyz, features)
        self.assertEqual(new_xyz.shape, (2, 16, 3))
        self.assertEqual(new_features.shape, (2, 16, 64))


class TestFeaturePropagation(unittest.TestCase):
    def test_three_nn_interpolate(self):
        fp = FeaturePropagation([128, 64])
        unknown = np.random.randn(2, 64, 3)
        known = np.random.randn(2, 16, 3)
        known_features = np.random.randn(2, 16, 128)
        interpolated = fp._three_nn_interpolate(unknown, known, known_features)
        self.assertEqual(interpolated.shape, (2, 64, 128))

    def test_forward(self):
        fp = FeaturePropagation([128, 64])
        xyz1 = np.random.randn(2, 64, 3)
        xyz2 = np.random.randn(2, 16, 3)
        features1 = np.random.randn(2, 64, 64)
        features2 = np.random.randn(2, 16, 128)
        output = fp.forward(xyz1, xyz2, features1, features2)
        self.assertEqual(output.shape, (2, 64, 64))


class TestDeepSC(unittest.TestCase):
    def setUp(self):
        self.deepsc = DeepSC(
            feature_dim=64,
            num_groups=[32, 16, 8],
            group_radii=[0.2, 0.4, 0.8],
            nsamples=[8, 16, 32]
        )

    def test_initialization(self):
        self.assertEqual(self.deepsc.feature_dim, 64)
        self.assertEqual(len(self.deepsc.sa_modules), 3)
        self.assertEqual(len(self.deepsc.fp_modules), 3)

    def test_geometric_features(self):
        points = np.random.randn(2, 50, 3)
        features = self.deepsc._compute_geometric_features(points)
        self.assertEqual(features.shape, (2, 50, 6))

    def test_extract_features(self):
        points = np.random.randn(100, 3)
        features = self.deepsc.extract_features(points)
        self.assertEqual(features.shape, (1, 100, 64))

    def test_extract_features_batch(self):
        points = np.random.randn(2, 50, 3)
        features = self.deepsc.extract_features(points)
        self.assertEqual(features.shape, (2, 50, 64))

    def test_feature_normalization(self):
        points = np.random.randn(1, 50, 3)
        features = self.deepsc.extract_features(points)
        norms = np.linalg.norm(features[0], axis=1)
        self.assertTrue(np.allclose(norms, 1.0, atol=0.1))

    def test_match_features(self):
        feat1 = np.random.randn(1, 30, 64)
        feat2 = np.random.randn(1, 40, 64)
        feat2[0, :10] = feat1[0, :10] + np.random.randn(10, 64) * 0.01

        matches, distances = self.deepsc.match_features(feat1, feat2)
        self.assertIsInstance(matches, np.ndarray)
        self.assertIsInstance(distances, np.ndarray)
        if len(matches) > 0:
            self.assertEqual(matches.shape[1], 2)

    def test_rotation_invariance(self):
        np.random.seed(42)
        points = np.random.randn(100, 3)

        angle = np.pi / 4
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        points_rotated = points @ R.T

        feat1 = self.deepsc.extract_features(points)
        feat2 = self.deepsc.extract_features(points_rotated)

        mean_feat1 = np.mean(feat1[0], axis=0)
        mean_feat2 = np.mean(feat2[0], axis=0)
        correlation = np.corrcoef(mean_feat1, mean_feat2)[0, 1]
        self.assertGreater(correlation, 0.5)

    def test_deformation_robustness(self):
        np.random.seed(42)
        points = np.random.randn(100, 3)
        deformed = points + np.random.randn(*points.shape) * 0.05

        feat1 = self.deepsc.extract_features(points)
        feat2 = self.deepsc.extract_features(deformed)

        matches, _ = self.deepsc.match_features(feat1, feat2, ratio_threshold=0.95)
        self.assertGreater(len(matches), 5)

    def test_self_supervised_training(self):
        training_data = [
            np.random.randn(50, 3),
            np.random.randn(50, 3),
            np.random.randn(50, 3),
        ]
        self.deepsc.self_supervised_train(training_data, epochs=2)
        self.assertTrue(self.deepsc._trained)


if __name__ == '__main__':
    unittest.main(verbosity=2)

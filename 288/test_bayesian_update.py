import unittest
import numpy as np
from bayesian_update import (
    BetaBinomial, NormalNormal, MCMCSampler, compute_hpd
)


class TestBetaBinomial(unittest.TestCase):
    
    def test_uniform_prior(self):
        model = BetaBinomial(1, 1)
        alpha_post, beta_post = model.update(5, 10)
        self.assertEqual(alpha_post, 6.0)
        self.assertEqual(beta_post, 6.0)
    
    def test_posterior_mean(self):
        model = BetaBinomial(2, 2)
        mean = model.posterior_mean(3, 5)
        self.assertAlmostEqual(mean, 0.5555555, places=4)
    
    def test_posterior_predictive_sum(self):
        model = BetaBinomial(1, 1)
        total_prob = sum(
            model.posterior_predictive(3, 5, k, 2)
            for k in range(3)
        )
        self.assertAlmostEqual(total_prob, 1.0, places=5)

    def test_smooth_correction_alpha_zero(self):
        model = BetaBinomial(0, 2, prior_correction="smooth")
        self.assertTrue(model.corrected)
        self.assertAlmostEqual(model.alpha_prior, 1e-6)
        self.assertEqual(model.beta_prior, 2.0)

    def test_smooth_correction_beta_zero(self):
        model = BetaBinomial(3, 0, prior_correction="smooth")
        self.assertTrue(model.corrected)
        self.assertEqual(model.alpha_prior, 3.0)
        self.assertAlmostEqual(model.beta_prior, 1e-6)

    def test_smooth_correction_both_zero(self):
        model = BetaBinomial(0, 0, prior_correction="smooth")
        self.assertTrue(model.corrected)
        self.assertAlmostEqual(model.alpha_prior, 1e-6)
        self.assertAlmostEqual(model.beta_prior, 1e-6)

    def test_smooth_correction_negative(self):
        model = BetaBinomial(-1, -2, prior_correction="smooth")
        self.assertTrue(model.corrected)
        self.assertAlmostEqual(model.alpha_prior, 1e-6)
        self.assertAlmostEqual(model.beta_prior, 1e-6)

    def test_smooth_no_correction_when_valid(self):
        model = BetaBinomial(1, 1, prior_correction="smooth")
        self.assertFalse(model.corrected)
        self.assertEqual(model.alpha_prior, 1.0)
        self.assertEqual(model.beta_prior, 1.0)

    def test_jeffreys_correction_alpha_zero(self):
        model = BetaBinomial(0, 2, prior_correction="jeffreys")
        self.assertTrue(model.corrected)
        self.assertEqual(model.alpha_prior, 0.5)
        self.assertEqual(model.beta_prior, 0.5)

    def test_jeffreys_correction_both_zero(self):
        model = BetaBinomial(0, 0, prior_correction="jeffreys")
        self.assertTrue(model.corrected)
        self.assertEqual(model.alpha_prior, 0.5)
        self.assertEqual(model.beta_prior, 0.5)

    def test_jeffreys_correction_negative(self):
        model = BetaBinomial(-1, 0.5, prior_correction="jeffreys")
        self.assertTrue(model.corrected)
        self.assertEqual(model.alpha_prior, 0.5)
        self.assertEqual(model.beta_prior, 0.5)

    def test_jeffreys_no_correction_when_valid(self):
        model = BetaBinomial(2, 3, prior_correction="jeffreys")
        self.assertFalse(model.corrected)
        self.assertEqual(model.alpha_prior, 2.0)
        self.assertEqual(model.beta_prior, 3.0)

    def test_invalid_correction_strategy(self):
        with self.assertRaises(ValueError):
            BetaBinomial(1, 1, prior_correction="invalid")

    def test_degenerate_smooth_posterior_predictive(self):
        model = BetaBinomial(0, 0, prior_correction="smooth")
        alpha_post, beta_post = model.update(5, 10)
        self.assertGreater(alpha_post, 0)
        self.assertGreater(beta_post, 0)
        pred = model.posterior_predictive(5, 10, 2, 3)
        self.assertGreater(pred, 0)
        self.assertLess(pred, 1)

    def test_degenerate_jeffreys_posterior_predictive(self):
        model = BetaBinomial(0, 0, prior_correction="jeffreys")
        pred = model.posterior_predictive(5, 10, 2, 3)
        self.assertGreater(pred, 0)
        self.assertLess(pred, 1)


class TestNormalNormal(unittest.TestCase):
    
    def test_no_data(self):
        model = NormalNormal(0, 1, 1)
        mu_post, tau_post = model.update([])
        self.assertEqual(mu_post, 0.0)
        self.assertEqual(tau_post, 1.0)
    
    def test_posterior_precision(self):
        model = NormalNormal(0, 1, 1)
        _, tau_post = model.update([1, 2, 3])
        self.assertEqual(tau_post, 4.0)


class TestComputeHPD(unittest.TestCase):
    
    def test_hpd_1d_known_distribution(self):
        np.random.seed(42)
        samples = np.random.normal(0, 1, size=100000)
        hpd = compute_hpd(samples, cred_mass=0.95)
        self.assertEqual(hpd.shape, (1, 2))
        self.assertAlmostEqual(hpd[0, 0], -1.96, places=1)
        self.assertAlmostEqual(hpd[0, 1], 1.96, places=1)
    
    def test_hpd_2d(self):
        np.random.seed(42)
        samples = np.random.multivariate_normal(
            [0, 1], [[1, 0], [0, 2]], size=10000
        )
        hpd = compute_hpd(samples, cred_mass=0.68)
        self.assertEqual(hpd.shape, (2, 2))
        self.assertAlmostEqual(hpd[0, 0], -1.0, places=1)
        self.assertAlmostEqual(hpd[0, 1], 1.0, places=1)
        self.assertAlmostEqual(hpd[1, 0], 1 - np.sqrt(2), places=1)
        self.assertAlmostEqual(hpd[1, 1], 1 + np.sqrt(2), places=1)
    
    def test_hpd_credible_mass_bounds(self):
        samples = np.array([1, 2, 3, 4, 5])
        with self.assertRaises(ValueError):
            compute_hpd(samples, cred_mass=0)
        with self.assertRaises(ValueError):
            compute_hpd(samples, cred_mass=1.0)
    
    def test_hpd_list_input(self):
        samples = [1.0, 2.0, 3.0, 4.0, 5.0]
        hpd = compute_hpd(samples, cred_mass=0.6)
        self.assertEqual(hpd.shape, (1, 2))


class TestMCMCSampler(unittest.TestCase):
    
    def test_log_posterior_valid(self):
        def log_prior(theta):
            if -10 < theta[0] < 10:
                return 0.0
            return -np.inf
        
        def log_likelihood(theta, data):
            return -0.5 * np.sum((data - theta[0])**2)
        
        sampler = MCMCSampler(
            log_prior=log_prior,
            log_likelihood=log_likelihood,
            ndim=1,
            data=np.array([1.0, 2.0, 3.0])
        )
        
        lp = sampler.log_posterior(np.array([2.0]))
        self.assertTrue(np.isfinite(lp))
    
    def test_log_posterior_outside_prior(self):
        def log_prior(theta):
            if -10 < theta[0] < 10:
                return 0.0
            return -np.inf
        
        def log_likelihood(theta, data):
            return -0.5 * np.sum((data - theta[0])**2)
        
        sampler = MCMCSampler(
            log_prior=log_prior,
            log_likelihood=log_likelihood,
            ndim=1,
            data=np.array([1.0, 2.0, 3.0])
        )
        
        lp = sampler.log_posterior(np.array([100.0]))
        self.assertEqual(lp, -np.inf)
    
    def test_sample_insufficient_walkers(self):
        def log_prior(theta):
            return 0.0
        
        def log_likelihood(theta, data):
            return 0.0
        
        sampler = MCMCSampler(
            log_prior=log_prior,
            log_likelihood=log_likelihood,
            ndim=2
        )
        
        with self.assertRaises(ValueError):
            sampler.sample(n_walkers=4, n_steps=100)
    
    def test_sample_returns_expected_keys(self):
        np.random.seed(42)
        
        def log_prior(theta):
            if -10 < theta[0] < 10:
                return 0.0
            return -np.inf
        
        def log_likelihood(theta, data):
            return -0.5 * np.sum((data - theta[0])**2)
        
        sampler = MCMCSampler(
            log_prior=log_prior,
            log_likelihood=log_likelihood,
            ndim=1,
            data=np.array([1.0, 2.0, 3.0])
        )
        
        result = sampler.sample(
            n_walkers=10,
            n_steps=500,
            n_burn=100,
            initial_guess=np.array([0.0])
        )
        
        expected_keys = [
            "samples", "log_prob", "acceptance_rate",
            "n_walkers", "n_steps", "n_burn", "ndim"
        ]
        for key in expected_keys:
            self.assertIn(key, result)
        
        self.assertEqual(result["samples"].shape[1], 1)
        self.assertGreater(len(result["samples"]), 0)
        self.assertGreater(result["acceptance_rate"], 0)
        self.assertLess(result["acceptance_rate"], 1)
    
    def test_sample_bernoulli_conjugate(self):
        np.random.seed(42)
        
        successes = 7
        trials = 10
        
        def log_prior(theta):
            p = theta[0]
            if 0 < p < 1:
                return 0.0
            return -np.inf
        
        def log_likelihood(theta, data):
            p = theta[0]
            successes, trials = data
            if 0 < p < 1:
                return successes * np.log(p) + (trials - successes) * np.log(1 - p)
            return -np.inf
        
        sampler = MCMCSampler(
            log_prior=log_prior,
            log_likelihood=log_likelihood,
            ndim=1,
            data=(successes, trials)
        )
        
        result = sampler.sample(
            n_walkers=50,
            n_steps=2000,
            n_burn=500,
            initial_guess=np.array([0.5])
        )
        
        stats = sampler.compute_statistics(
            result["samples"],
            cred_mass=0.95,
            param_names=["p"]
        )
        
        analytical_mean = (successes + 1) / (trials + 2)
        self.assertAlmostEqual(stats["p"]["mean"], analytical_mean, places=1)
        self.assertGreater(stats["p"]["hpd_high"], stats["p"]["hpd_low"])
    
    def test_compute_statistics_1d(self):
        np.random.seed(42)
        samples = np.random.normal(5, 1, size=1000)
        
        def log_prior(theta):
            return 0.0
        
        def log_likelihood(theta, data):
            return 0.0
        
        sampler = MCMCSampler(
            log_prior=log_prior,
            log_likelihood=log_likelihood,
            ndim=1
        )
        
        stats = sampler.compute_statistics(
            samples, cred_mass=0.68, param_names=["mu"]
        )
        
        self.assertIn("mu", stats)
        self.assertAlmostEqual(stats["mu"]["mean"], 5.0, places=1)
        self.assertAlmostEqual(stats["mu"]["median"], 5.0, places=1)
        self.assertAlmostEqual(stats["mu"]["std"], 1.0, places=1)
        self.assertLess(stats["mu"]["hpd_low"], 5.0)
        self.assertGreater(stats["mu"]["hpd_high"], 5.0)
    
    def test_compute_statistics_2d(self):
        np.random.seed(42)
        samples = np.random.multivariate_normal(
            [1.0, 2.0], [[1.0, 0.0], [0.0, 0.5]], size=1000
        )
        
        def log_prior(theta):
            return 0.0
        
        def log_likelihood(theta, data):
            return 0.0
        
        sampler = MCMCSampler(
            log_prior=log_prior,
            log_likelihood=log_likelihood,
            ndim=2
        )
        
        stats = sampler.compute_statistics(
            samples, cred_mass=0.95, param_names=["a", "b"]
        )
        
        self.assertIn("a", stats)
        self.assertIn("b", stats)
        self.assertAlmostEqual(stats["a"]["mean"], 1.0, places=1)
        self.assertAlmostEqual(stats["b"]["mean"], 2.0, places=1)
    
    def test_compute_statistics_mismatched_names(self):
        def log_prior(theta):
            return 0.0
        
        def log_likelihood(theta, data):
            return 0.0
        
        sampler = MCMCSampler(
            log_prior=log_prior,
            log_likelihood=log_likelihood,
            ndim=2
        )
        
        with self.assertRaises(ValueError):
            sampler.compute_statistics(
                np.random.randn(100, 2), param_names=["only_one"]
            )


if __name__ == "__main__":
    unittest.main()

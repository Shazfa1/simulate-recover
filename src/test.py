import unittest
import numpy as np
from ez_diffusion import forward_equations, inverse_equations, simulate_observed

class TestEZDiffusion(unittest.TestCase):

    def test_forward_equations(self):
        v, a, T = 0.5, 1.0, 0.3
        Rpred, Mpred, Vpred = forward_equations(v, a, T)
        self.assertIsNotNone(Rpred)
        self.assertIsNotNone(Mpred)
        self.assertIsNotNone(Vpred)
        # Add more specific assertions based on expected values

    def test_inverse_equations(self):
        R, M, V = 0.7, 1.2, 0.4
        vest, aest, Test = inverse_equations(R, M, V)
        self.assertIsNotNone(vest)
        self.assertIsNotNone(aest)
        self.assertIsNotNone(Test)
        # Add more specific assertions based on expected values

    def test_simulate_observed(self):
        Rpred, Mpred, Vpred = 0.7, 1.2, 0.4
        N = 100
        Robs, Mobs, Vobs = simulate_observed(Rpred, Mpred, Vpred, N)
        self.assertIsNotNone(Robs)
        self.assertIsNotNone(Mobs)
        self.assertIsNotNone(Vobs)
        # Add more specific assertions based on expected values

    def test_no_noise_case(self):
        # Test the case when there's no noise (slide 17)
        v_true, a_true, T_true = 0.5, 1.0, 0.3
        Rpred, Mpred, Vpred = forward_equations(v_true, a_true, T_true)
        
        # In the no-noise case, observed should equal predicted
        Robs, Mobs, Vobs = Rpred, Mpred, Vpred
        
        v_est, a_est, T_est = inverse_equations(Robs, Mobs, Vobs)
        
        # Calculate bias
        bias = np.array([v_true, a_true, T_true]) - np.array([v_est, a_est, T_est])
        
        # Assert that bias is close to zero (use np.allclose for floating-point comparison)
        self.assertTrue(np.allclose(bias, np.zeros(3), atol=1e-6))

    def test_parameter_recovery(self):
        # Test if parameters can be recovered accurately
        v_true, a_true, T_true = 0.5, 1.0, 0.3
        N = 10000  # Large sample size for more accurate recovery
        
        Rpred, Mpred, Vpred = forward_equations(v_true, a_true, T_true)
        Robs, Mobs, Vobs = simulate_observed(Rpred, Mpred, Vpred, N)
        v_est, a_est, T_est = inverse_equations(Robs, Mobs, Vobs)
        
        # Assert that estimated parameters are close to true parameters
        self.assertAlmostEqual(v_true, v_est, places=2)
        self.assertAlmostEqual(a_true, a_est, places=2)
        self.assertAlmostEqual(T_true, T_est, places=2)

    def test_sample_size_effect(self):
        # Test if increasing sample size decreases squared error
        v_true, a_true, T_true = 0.5, 1.0, 0.3
        N_small, N_large = 100, 10000
        
        Rpred, Mpred, Vpred = forward_equations(v_true, a_true, T_true)
        
        # Small sample size
        Robs_small, Mobs_small, Vobs_small = simulate_observed(Rpred, Mpred, Vpred, N_small)
        v_est_small, a_est_small, T_est_small = inverse_equations(Robs_small, Mobs_small, Vobs_small)
        bias_small = np.array([v_true, a_true, T_true]) - np.array([v_est_small, a_est_small, T_est_small])
        squared_error_small = np.sum(bias_small ** 2)
        
        # Large sample size
        Robs_large, Mobs_large, Vobs_large = simulate_observed(Rpred, Mpred, Vpred, N_large)
        v_est_large, a_est_large, T_est_large = inverse_equations(Robs_large, Mobs_large, Vobs_large)
        bias_large = np.array([v_true, a_true, T_true]) - np.array([v_est_large, a_est_large, T_est_large])
        squared_error_large = np.sum(bias_large ** 2)
        
        # Assert that squared error decreases with larger sample size
        self.assertLess(squared_error_large, squared_error_small)

if __name__ == '__main__':
    unittest.main()

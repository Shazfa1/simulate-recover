import unittest
import numpy as np
from simulate_recover import forward_equations, inverse_equations, simulate_and_recover

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

    def test_no_noise_case(self):
        v_true, a_true, T_true = 0.5, 1.0, 0.3
        Rpred, Mpred, Vpred = forward_equations(v_true, a_true, T_true)
        v_est, a_est, T_est = inverse_equations(Rpred, Mpred, Vpred)
        bias = np.array([v_true, a_true, T_true]) - np.array([v_est, a_est, T_est])
        # Check if bias is close to zero within a tolerance
        tolerance = 1e-6  # Adjust this value as needed
        self.assertTrue(np.allclose(bias, np.zeros(3), atol=1e-5))

    def test_simulate_and_recover(self):
        sample_sizes = [10, 40, 4000]
        iterations = 1000
        
        for N in sample_sizes:
            biases, squared_errors = simulate_and_recover(N, iterations)
            
            # Check shape of results
            self.assertEqual(biases.shape, (iterations, 3))
            self.assertEqual(squared_errors.shape, (iterations,))
            
            # Check average bias is close to 0
            avg_bias = np.mean(biases, axis=0)
            print(f"This is average bias {avg_bias}")
            self.assertTrue(np.all(np.abs(avg_bias) < 0.1))
            
            # Store average squared error for comparison
            if N == 10:
                prev_avg_squared_error = np.mean(squared_errors)
            else:
                curr_avg_squared_error = np.mean(squared_errors)
                # Check that squared error decreases with increasing N
                self.assertLess(curr_avg_squared_error, prev_avg_squared_error)
                prev_avg_squared_error = curr_avg_squared_error

    def test_parameter_ranges(self):
        N = 1000
        iterations = 100
        biases, _ = simulate_and_recover(N, iterations)
        
        # Calculate mean biases across iterations
        mean_biases = np.mean(biases, axis=0)
        
        # Add mean of range to mean biases
        v_est, a_est, T_est = mean_biases + np.array([1.25, 1.25, 0.3])
        
        self.assertTrue(np.all((v_est >= 0.5) & (v_est <= 2)))
        self.assertTrue(np.all((a_est >= 0.5) & (a_est <= 2)))
        self.assertTrue(np.all((T_est >= 0.1) & (T_est <= 0.5)))

if __name__ == '__main__':
    unittest.main()

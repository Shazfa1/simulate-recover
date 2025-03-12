import numpy as np
import argparse
from scipy.stats import norm, binom, gamma

def forward_equations(v, a, T):
    """
    Implement forward equations (1, 2, 3) to generate predicted summary statistics
    """
    Rpred = (1 + np.exp(-2*v*a)) ** -1
    Mpred = (a / (2*v)) * (1 - np.exp(-2*v*a)) / (1 + np.exp(-2*v*a)) + T
    Vpred = (a**2 / (4*v**2)) * (1 - np.exp(-4*v*a)) / (1 + np.exp(-2*v*a)) - ((a / (2*v)) * (1 - np.exp(-2*v*a)) / (1 + np.exp(-2*v*a)))**2
    return Rpred, Mpred, Vpred

def sampling_distribution(Rpred, Mpred, Vpred, N):
    """
    Implement sampling distribution equations (7, 8, 9) to simulate observed summary statistics
    """
    Robs = binom.rvs(N, Rpred) / N
    Mobs = norm.rvs(Mpred, np.sqrt(Vpred/N))
    Vobs = gamma.rvs((N-1)/2, scale=2*Vpred/(N-1))
    return Robs, Mobs, Vobs

def inverse_equations(Robs, Mobs, Vobs):
    """
    Implement inverse equations (4, 5, 6) to compute estimated parameters
    """
    s = Mobs / Vobs
    vest = np.sign(Robs - 0.5) * (((7 * Robs**2 - 7*Robs + 4) / (Vobs * (Robs**2 - Robs + 0.5))) ** 0.25)
    aest = s * vest * Mobs
    Test = Mobs - (aest / (2 * vest)) * ((1 - np.exp(-2 * vest * aest)) / (1 + np.exp(-2 * vest * aest)))
    return vest, aest, Test

def simulate_and_recover(N, iterations):
    biases = []
    squared_errors = []

    for _ in range(iterations):
        # Select true parameters
        v = np.random.uniform(0.5, 2)
        a = np.random.uniform(0.5, 2)
        T = np.random.uniform(0.1, 0.5)

        # Generate predicted summary statistics
        Rpred, Mpred, Vpred = forward_equations(v, a, T)

        # Simulate observed summary statistics
        Robs, Mobs, Vobs = sampling_distribution(Rpred, Mpred, Vpred, N)

        # Compute estimated parameters
        vest, aest, Test = inverse_equations(Robs, Mobs, Vobs)

        # Compute bias and squared error
        bias = np.array([v, a, T]) - np.array([vest, aest, Test])
        squared_error = np.sum(bias**2)

        biases.append(bias)
        squared_errors.append(squared_error)

    return np.array(biases), np.array(squared_errors)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run EZ diffusion simulate-and-recover')
    parser.add_argument('--n', type=int, required=True, help='Sample size')
    parser.add_argument('--iterations', type=int, required=True, help='Number of iterations')
    args = parser.parse_args()

    biases, squared_errors = simulate_and_recover(args.n, args.iterations)

    print(f"Average bias: {np.mean(biases, axis=0)}")
    print(f"Average squared error: {np.mean(squared_errors)}")

    # Save results to file
    np.savez(f'results_N{args.n}.npz', biases=biases, squared_errors=squared_errors)

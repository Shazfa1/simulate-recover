import numpy as np
from scipy.stats import norm, binom, gamma
import matplotlib.pyplot as plt

def test_no_noise():
    v, a, T = 1.0, 1.5, 0.3  # Example true parameters
    Rpred, Mpred, Vpred = forward_equations(v, a, T)
    vest, aest, Test = inverse_equations(Rpred, Mpred, Vpred)
    b = np.array([v, a, T]) - np.array([vest, aest, Test])
    assert np.allclose(b, 0, atol=1e-6), f"Bias should be 0 when there's no noise, but got {b}"
    print("No-noise test passed successfully.")

def forward_equations(v, a, T):
    y = np.exp(-1*a*v)
    Rpred = 1 / (1 + y)
    Mpred = (a / (2 * v)) * ((1 - y) / (1 + y)) + T
    Vpred = (a / (2* v**3)) * ((1 - 2*a*v*y - y**2) / (y + 1)**2)
    return Rpred, Mpred, Vpred

def sampling_distribution(Rpred, Mpred, Vpred, N):
    Robs = binom.rvs(N, Rpred) / N
    Mobs = norm.rvs(Mpred, np.sqrt(Vpred / N))
    Vobs = gamma.rvs((N - 1) / 2, scale=2 * Vpred / (N - 1))
    return Robs, Mobs, Vobs

def sign(x):
    if x < 0:
        return -1
    elif x > 0:
        return 1
    else:
        return 0

def inverse_equations(Robs, Mobs, Vobs):
    epsilon = 1e-10  # Small value to avoid division by zero
    Robs = np.clip(Robs, epsilon, 1 - epsilon)  # Clip Robs to avoid 0 or 1
    L = np.log(Robs / (1-Robs))
    vest = sign(Robs - 0.5) * ((L*(((Robs**2)*(L)) - (Robs*L) + Robs - 0.5)) / Vobs)**0.25
    aest = L / vest
    Test = Mobs - (aest / (2 * vest)) * ((1 - np.exp(-vest * aest)) / (1 + np.exp(-vest * aest)))
    return vest, aest, Test

def simulate_and_recover(N, iterations):
    biases = []
    squared_errors = []
    for _ in range(iterations):
        v = np.random.uniform(0.5, 2)
        a = np.random.uniform(0.5, 2)
        T = np.random.uniform(0.1, 0.5)
        
        Rpred, Mpred, Vpred = forward_equations(v, a, T)
        Robs, Mobs, Vobs = sampling_distribution(Rpred, Mpred, Vpred, N)
        vest, aest, Test = inverse_equations(Robs, Mobs, Vobs)
        
        bias = np.array([v, a, T]) - np.array([vest, aest, Test])
        biases.append(bias)
        squared_errors.append(np.sum(bias**2))
    
    return np.array(biases), np.array(squared_errors)

# Call this function before running the main simulation
test_no_noise()
results = []
# Run simulations
sample_sizes = [10, 40, 4000]
all_biases = []
all_squared_errors = []

for N in sample_sizes:
    biases, squared_errors = simulate_and_recover(N, 1000)
    for bias, squared_error in zip(biases, squared_errors):
        results.append({
            'N': N,
            'bias': bias,
            'squared_error': squared_error
        })
    print(f"N = {N}:")
    print(f"Average bias: {np.mean(biases, axis=0)}")
    print(f"Average squared error: {np.mean(squared_errors)}")
    print()

np.save('simulation_results.npy', results)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for i, N in enumerate(sample_sizes):
    ax1.boxplot(all_biases[i], positions=[i*3, i*3+1, i*3+2], widths=0.6)
    ax2.boxplot(all_squared_errors[i], positions=[i], widths=0.6)

ax1.set_title('Parameter Biases')
ax1.set_xticklabels(['v', 'a', 'T'] * 3)
ax1.set_ylabel('Bias')
ax1.axhline(y=0, color='r', linestyle='--')

ax2.set_title('Squared Errors')
ax2.set_xticklabels(sample_sizes)
ax2.set_ylabel('Squared Error')
ax2.set_xlabel('Sample Size (N)')

plt.tight_layout()
plt.show()

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
    Mpred = T + ((a / (2 * v)) * ((1 - y) / (1 + y)))
    Vpred = (a / (2* v**3)) * ((1 - 2*a*v*y - y**2) / (y + 1)**2)

    print(f"Forward equations results:")
    print(f"Rpred: {Rpred}")
    print(f"Mpred: {Mpred}")
    print(f"Vpred: {Vpred}")
    
    return Rpred, Mpred, Vpred

def sampling_distribution(Rpred, Mpred, Vpred, N, epsilon=1e-6):
    while True:
        Robs = binom.rvs(N, Rpred) / N
        if epsilon < Robs < 1 - epsilon:
            break
    
    Mobs = norm.rvs(Mpred, np.sqrt(Vpred / N))
    Vobs = gamma.rvs((N - 1) / 2, scale=2 * Vpred / (N - 1))
    
    # Resample if necessary
    while Mobs <= 0 or Vobs <= 0:
        if Mobs <= 0:
            Mobs = norm.rvs(Mpred, np.sqrt(Vpred / N))
        if Vobs <= 0:
            Vobs = gamma.rvs((N - 1) / 2, scale=2 * Vpred / (N - 1))
    print(f"Sampling distribution results:")
    print(f"Robs: {Robs}")
    print(f"Mobs: {Mobs}")
    print(f"Vobs: {Vobs}")
    return Robs, Mobs, Vobs

def sign(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)

def inverse_equations(Robs, Mobs, Vobs):
    epsilon = 1e-10  # Small value to avoid division by zero

    # Check for invalid input values
    if not (0 <= Robs <= 1):
        raise ValueError(f"Robs must be between 0 and 1, got {Robs}")
    if Mobs <= 0:
        raise ValueError(f"Mobs must be positive, got {Mobs}")
    if Vobs <= 0:
        raise ValueError(f"Vobs must be positive, got {Vobs}")

    # Clip Robs to avoid 0 or 1
    Robs = np.clip(Robs, epsilon, 1 - epsilon)  # Clip Robs to avoid 0 or 1

    L = np.log(Robs / (1-Robs))
    print(f"Putting following into the inverse")
    
    print(f"Robs: {Robs}")
    print(f"Mobs: {Mobs}")
    print(f"Vobs: {Vobs}")
    # Check for potential division by zero or negative values under root
    #if Vobs == 0 or (Robs**2 * L - Robs * L + Robs - 0.5) <= 0:

        #raise ValueError("Invalid combination of Robs and Vobs")

    vest = sign(Robs - 0.5) * ((L*(((Robs**2)*(L)) - (Robs*L) + Robs - 0.5)) / Vobs)**0.25
    print(f"vest: {vest}")
    # Check for division by zero
    if abs(vest) < epsilon:
        raise ValueError("vest is too close to zero, causing division issues")

    aest = L / (vest + epsilon)
    print(f"aest: {aest}")

    # Check for potential issues in Test calculation
    denominator = 1 + np.exp(-vest * aest)
    if denominator < epsilon:
        raise ValueError("Denominator in Test calculation is too close to zero")

    print(f"part one:")
    print(f"{((aest / (2 * vest)))}")
    print(f"{((1 - np.exp(-vest * aest)) / denominator)}")
    
    Test = Mobs - ((aest / (2 * vest)) * ((1 - np.exp(-vest * aest)) / denominator))

    # Check for negative Test
    if Test < 0:
        raise ValueError(f"Calculated Test is negative: {Test}")

    return vest, aest, Test
def simulate_and_recover(N, iterations):
    biases = []
    squared_errors = []
    for _ in range(iterations):
        v = np.random.uniform(0.5, 2)
        a = np.random.uniform(0.5, 2)
        T = np.random.uniform(0.1, 0.5)

        print(f"N:{N}")
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
    all_biases.append(biases)
    all_squared_errors.append(squared_errors)
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

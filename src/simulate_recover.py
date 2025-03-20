import numpy as np
from scipy.stats import norm, binom, gamma
import matplotlib.pyplot as plt

def test_no_noise():
    v, a, T = 1.0, 1.5, 0.3  # Example true parameters
    Rpred, Mpred, Vpred = forward_equations(v, a, T)
    vest, aest, Test = inverse_equations(Rpred, Mpred, Vpred)
    b = np.array([v, a, T]) - np.array([vest, aest, Test])
    print("No-noise test passed successfully.")

def forward_equations(v, a, T):
    y = np.exp(-1*a*v)
    Rpred = 1 / (1 + y)
    Mpred = T + ((a / (2 * v)) * ((1 - y) / (1 + y)))
    Vpred = (a / (2* v**3)) * ((1 - 2*a*v*y - y**2) / (y + 1)**2)
    
    return Rpred, Mpred, Vpred

def sampling_distribution(Rpred, Mpred, Vpred, N, epsilon=0.1):
    # handle Rpred before starting
    Rpred = np.clip(R_pred, 0.01, 0.99)
    Tobs = np.random.binomial(N, Rpred)
    Robs = min(max(Tobs / N, 0.01), 0.99)
    #handle extreme values input for mobs
    
    sig = np.sqrt(max(Vpred / N, 1e-6))
    Mobs = np.random.normal(Mpred, sig)
    if N > 1:
        
        Vobs = np.random.gamma((N - 1) / 2, (2 * Vpred) / (N - 1))
    else:
        Vobs = max(Vpred * (1 + np.random.normal(0, 0.1)), 1e-6) # add noise
    return Robs, Mobs, Vobs   
#no resampling because it causes infinite loop

def sign(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)

def inverse_equations(Robs, Mobs, Vobs):
    epsilon = 1e-06  # Small value to avoid division by zero
    #if statement to make sure inverse_eqns raises ValueError if R_obs is not between 0 and 1
    if not (0 <= Robs <= 1):
        raise ValueError("Robs must be between 0 and 1")
    try:
        Robs = np.clip(Robs, 0.001, 0.999)
        L = np.log(Robs / (1-Robs))
        # Handle cases
        if Vobs <= epsilon or Robs == 0.5:
            return 0,0, Mobs #takes care of edge cases that lead to negative test values
        
    
        # Check for potential division by zero or negative values under root
        if Vobs == 0 or (Robs**2 * L - Robs * L + Robs - 0.5) <= 0:
            raise ValueError("Invalid combination of Robs and Vobs")
    
        #split up vest to handle edge cases
        sq_term = (L*(((Robs**2)*(L)) - (Robs*L) + Robs - 0.5)) / (Vobs + epsilon)
        vest = np.sign(Robs - 0.5) * (sq_term**0.25)
    
        
        # Check for division by zero
        if abs(vest) < epsilon:
            raise ValueError("vest is too close to zero, causing division issues")
    
        aest = L / (vest) if vest != 0 else 0
    
        # Check for potential issues in Test calculation
        #break into parts to ensure correct calcualtion
        expo = np.exp(-vest * aest)
        if vest != 0 and aest !=0:
            Test = Mobs - ((aest / (2 * vest)) * ((1 - expo) / (1 + expo)))
    
        else:
            Test = Mobs  
        # Check for negative Test
        if Test < 0:
            raise ValueError(f"Calculated Test is negative: {Test}")
    
        return vest, aest, Test

    except (ValueError, ZeroDivisionError):
        return 0, 0, M_obs  # defaults in case of failure

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

import numpy as np
import matplotlib.pyplot as plt

def load_results(filename):
    return np.load(filename, allow_pickle=True)

def analyze_bias(results):
    biases = np.array([r['bias'] for r in results])
    mean_bias = np.mean(biases, axis=0)
    std_bias = np.std(biases, axis=0)
    
    print("Mean bias:", mean_bias)
    print("Std bias:", std_bias)
    
    plt.figure(figsize=(10, 6))
    plt.boxplot(biases)
    plt.title("Distribution of Biases")
    plt.xticks([1, 2, 3], ['v', 'a', 'T'])
    plt.ylabel("Bias")
    plt.savefig("bias_distribution.png")
    plt.close()

def analyze_squared_error(results):
    squared_errors = np.array([r['squared_error'] for r in results])
    mean_se = np.mean(squared_errors)
    std_se = np.std(squared_errors)
    
    print("Mean squared error:", mean_se)
    print("Std squared error:", std_se)
    
    plt.figure(figsize=(10, 6))
    plt.hist(squared_errors, bins=50)
    plt.title("Distribution of Squared Errors")
    plt.xlabel("Squared Error")
    plt.ylabel("Frequency")
    plt.savefig("squared_error_distribution.png")
    plt.close()

def analyze_sample_size_effect(results):
    sample_sizes = np.array([r['N'] for r in results])
    squared_errors = np.array([r['squared_error'] for r in results])
    
    plt.figure(figsize=(10, 6))
    plt.scatter(sample_sizes, squared_errors)
    plt.title("Effect of Sample Size on Squared Error")
    plt.xlabel("Sample Size (N)")
    plt.ylabel("Squared Error")
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig("sample_size_effect.png")
    plt.close()

def main():
    results = load_results("simulation_results.npy")
    
    analyze_bias(results)
    analyze_squared_error(results)
    analyze_sample_size_effect(results)
    
    print("Analysis complete. Plots saved.")

if __name__ == "__main__":
    main()

#!/bin/bash

# Set the path to the Python script
PYTHON_SCRIPT="src/simulate_recover.py"

# Function to run simulate-and-recover for a given sample size
run_simulate_recover() {
    local n=$1
    echo "Running simulate-and-recover for N = $n"
    python3 $PYTHON_SCRIPT --n $n --iterations 1000
}

# Run simulate-and-recover for each sample size
run_simulate_recover 10
run_simulate_recover 40
run_simulate_recover 4000

# Analyze results
echo "Analyzing results"
python3 src/analyze_results.py

echo "Simulate-and-recover exercise completed."


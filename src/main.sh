#!/bin/bash

# Run the Python script for each sample size
python3 src/simulate_recover.py --n 10 --iterations 1000
python3 src/simulate_recover.py --n 40 --iterations 1000
python3 src/simulate_recover.py --n 4000 --iterations 1000

# Analyze results
python3 src/analyze_results.py

echo "Simulation and analysis complete."

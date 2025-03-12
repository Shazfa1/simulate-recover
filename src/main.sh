#!/bin/bash

# Run the Python script for each sample size
python src/simulate_recover.py --n 10 --iterations 1000
python src/simulate_recover.py --n 40 --iterations 1000
python src/simulate_recover.py --n 4000 --iterations 1000

# Analyze results
python src/analyze_results.py

echo "Simulation and analysis complete."

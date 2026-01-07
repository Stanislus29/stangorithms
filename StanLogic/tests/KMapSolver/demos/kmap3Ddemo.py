"""
KMapSolver3D Demo: 12-Variable K-Map Minimization
==================================================

Demonstrates the use of KMapSolver3D for minimizing a 12-variable
Boolean function with random pattern generation.

Author: Somtochukwu Stanislus Emeka-Onwuneme
Date: November 2025
"""

from stanlogic import BoolMinGeo
import random, re

# Example: 8-variable K-map with random pattern
print("EXAMPLE: 12-VARIABLE K-MAP (Random Pattern)")
print("="*60)
num_vars = 12

# Generate random output values (256 total for 8 variables)# Set random seed for reproducibility
random.seed(42)
output_values_16 = [random.choice([0, 1, 'd']) for _ in range(2**num_vars)]

# Create and minimize
kmap_solver_16 = BoolMinGeo(num_vars, output_values_16)

# Minimize the 12-variable K-map
terms, expression = kmap_solver_16.minimize(form='sop')
num_terms, num_literals = BoolMinGeo.count_literals(expression, form='sop')

print(f"Number of terms: {num_terms}")
print(f"Number of literals in final expression: {num_literals}")
print("Final minimized expression:")
print(expression)
"""
Simple test for 2D/3D cluster integration.
"""

import sys
sys.path.insert(0, 'c:/Users/DELL/Documents/Research Projects/Mathematical_models/StanLogic/src')

from stanlogic import BoolMinGeo

# Simple 5-variable case
print("Creating 5-variable solver...")
output_values = [0] * 32

# Set some minterms
output_values[0] = 1
output_values[1] = 1
output_values[8] = 1
output_values[9] = 1

print(f"Output values set: {[i for i, v in enumerate(output_values) if v == 1]}")

solver = BoolMinGeo(5, output_values)

print("\nCalling minimize_3d...")
try:
    terms, expression = solver.minimize_3d(form='sop')
    print(f"\nResult: {len(terms)} terms")
    print(f"F = {expression}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

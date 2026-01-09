"""
Quick test to verify the benchmark variable ordering fix.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')))

from stanlogic.BoolMinGeo import BoolMinGeo

# Simple 3-variable test case
num_vars = 3
# Truth table: index 0=000, 1=001, 2=010, 3=011, 4=100, 5=101, 6=110, 7=111
# Let's set minterms 1, 3, 5, 7 to 1 (all where x1=1)
output_values = [0, 1, 0, 1, 0, 1, 0, 1]

print("="*70)
print("VARIABLE ORDERING VERIFICATION TEST")
print("="*70)
print(f"\nTesting {num_vars}-variable function")
print(f"Output values: {output_values}")
print("\nExpected: This should simplify to just x1")
print("Because minterms where x1=1 (bit 0 set) are: 1,3,5,7")

# Create solver
solver = BoolMinGeo(num_vars, output_values)

# Minimize
import io
import sys as sys_module
old_stdout = sys_module.stdout
sys_module.stdout = io.StringIO()

terms, expression = solver.minimize_3d(form='sop')

sys_module.stdout = old_stdout

print(f"\nResult: {expression}")
print(f"Terms: {terms}")

# Verify
print("\nVerifying all minterms...")
mismatches = []
for i in range(2**num_vars):
    binary = format(i, f'0{num_vars}b')
    # Evaluate expression manually
    x1, x2, x3 = int(binary[0]), int(binary[1]), int(binary[2])
    
    # For expression "x1", result should equal x1
    expected = output_values[i]
    
    # Simple evaluation: if expression is "x1", check if x1 matches expected
    if expression.strip() == "x1":
        actual = x1
    else:
        # More complex - would need full evaluation
        actual = -1
    
    if actual != expected:
        mismatches.append((i, binary, expected, actual))

if mismatches:
    print(f"✗ FAILED: {len(mismatches)} mismatches")
    for m, b, e, a in mismatches:
        print(f"  Minterm {m} ({b}): expected {e}, got {a}")
else:
    print(f"✓ PASSED: All minterms match")

print("\n" + "="*70)

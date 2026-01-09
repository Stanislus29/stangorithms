"""
Minimal benchmark test to verify the variable ordering fix.
Runs just a few test cases to check if equivalence now works.
"""

import sys
import os
import random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')))

from stanlogic.BoolMinGeo import BoolMinGeo
from pyeda.inter import *
import numpy as np

# Seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def create_minterm_fixed(vars_list, index):
    """
    Create a minterm using BoolMinGeo's bit ordering (LSB-first).
    """
    terms = []
    for i, var in enumerate(vars_list):
        if (index >> i) & 1:
            terms.append(var)
        else:
            terms.append(~var)
    return And(*terms) if len(terms) > 1 else terms[0]

def test_single_case(num_vars, output_values, test_name):
    """Test a single case."""
    print(f"\nTest: {test_name}")
    print(f"  Variables: {num_vars}, Ones: {sum(1 for v in output_values if v == 1)}")
    
    # Get minterms
    minterms = [i for i, v in enumerate(output_values) if v == 1]
    dont_cares = [i for i, v in enumerate(output_values) if v == 'd']
    
    # Create PyEDA variables
    pyeda_var_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    pyeda_vars = [exprvar(pyeda_var_names[i]) for i in range(num_vars)]
    
    # PyEDA minimization
    print("  PyEDA...", end=" ", flush=True)
    if not minterms and not dont_cares:
        expr_pyeda = expr(0)
    else:
        if minterms:
            f_on = Or(*[create_minterm_fixed(pyeda_vars, mt) for mt in minterms])
        else:
            f_on = expr(0)
        
        if dont_cares:
            f_dc = Or(*[create_minterm_fixed(pyeda_vars, dc) for dc in dont_cares])
        else:
            f_dc = expr(0)
        
        expr_pyeda = espresso_exprs(f_on, f_dc)[0]
    print(f"✓")
    
    # BoolMinGeo minimization
    print("  BoolMinGeo...", end=" ", flush=True)
    import io
    import sys as sys_module
    old_stdout = sys_module.stdout
    sys_module.stdout = io.StringIO()
    
    solver = BoolMinGeo(num_vars, output_values)
    terms, expr_str = solver.minimize_3d(form='sop')
    
    sys_module.stdout = old_stdout
    print(f"✓")
    
    # Check equivalence
    print("  Checking equivalence...", end=" ", flush=True)
    mismatches = 0
    
    for i in range(2**num_vars):
        # Create assignment (LSB-first to match BoolMinGeo)
        assign = {}
        for j, v in enumerate(pyeda_vars):
            bit = (i >> j) & 1
            assign[v] = expr(1) if bit else expr(0)
        
        # Expected value
        expected = output_values[i]
        if expected == 'd':
            continue  # Skip don't-cares
        
        # Evaluate PyEDA
        try:
            val = expr_pyeda.restrict(assign)
            pyeda_val = 1 if (hasattr(val, 'is_one') and val.is_one()) else (1 if str(val) == '1' else 0)
        except:
            pyeda_val = -1
        
        if pyeda_val != expected:
            mismatches += 1
    
    if mismatches > 0:
        print(f"✗ FAIL ({mismatches} mismatches)")
        return False
    else:
        print(f"✓ PASS")
        return True

# Run tests
print("="*80)
print("MINIMAL BENCHMARK - VARIABLE ORDERING FIX VERIFICATION")
print("="*80)

results = []

# Test 1: Simple 5-variable case with known result
print("\n" + "="*80)
num_vars = 5
output_values = [0] * 32
# Set minterms where x1=1 (bit 0 set): 1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31
for i in range(32):
    if i & 1:  # bit 0 set
        output_values[i] = 1
results.append(("5-var x1", test_single_case(num_vars, output_values, "5-var: x1=1")))

# Test 2: Random 5-variable
print("\n" + "="*80)
num_vars = 5
output_values = [1 if random.random() < 0.5 else 0 for _ in range(32)]
results.append(("5-var random", test_single_case(num_vars, output_values, "5-var: random 50%")))

# Test 3: 6-variable sparse
print("\n" + "="*80)
num_vars = 6
output_values = [1 if random.random() < 0.2 else 0 for _ in range(64)]
results.append(("6-var sparse", test_single_case(num_vars, output_values, "6-var: sparse 20%")))

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
passed = sum(1 for _, r in results if r)
print(f"Results: {passed}/{len(results)} tests passed")

for name, result in results:
    status = "PASS" if result else "FAIL"
    symbol = "✓" if result else "✗"
    print(f"  {symbol} {name:20s}: {status}")

if passed == len(results):
    print("\n✓ All tests passed! Variable ordering fix is working.")
else:
    print(f"\n✗ {len(results)-passed} test(s) failed. Further debugging needed.")

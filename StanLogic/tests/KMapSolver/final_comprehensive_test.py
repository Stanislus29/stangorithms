"""
Final comprehensive test of BoolMin2D optimizations
Validates that all optimizations ensure complete coverage and logical correctness
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from stanlogic.BoolMin2D import BoolMin2D
from pyeda.inter import *
import random
import re

def validate_kmap_result(kmap, form='sop'):
    """
    Validate that BoolMin2D result correctly implements the K-map specification.
    Checks that the expression produces the correct output for all non-don't-care inputs.
    """
    num_vars = 3 if len(kmap) * len(kmap[0]) == 8 else (2 if len(kmap) * len(kmap[0]) == 4 else 4)
    
    # Get BoolMin2D result
    solver = BoolMin2D(kmap)
    terms, expr_str = solver.minimize(form=form)
    
    # Parse to PyEDA for evaluation
    pyeda_vars = [exprvar(f'x{i+1}') for i in range(num_vars)]
    var_map = {f'x{i+1}': pyeda_vars[i] for i in range(num_vars)}
    
    def parse_sop(expr_str):
        if not expr_str or expr_str == "0":
            return expr(0)
        if expr_str == "1":
            return expr(1)
        
        or_terms = []
        for product in expr_str.split('+'):
            product = product.strip()
            if not product:
                continue
            literals = re.findall(r"[A-Za-z]+\d*'?", product)
            and_terms = []
            for lit in literals:
                if lit.endswith("'"):
                    var_name = lit[:-1]
                    if var_name in var_map:
                        and_terms.append(~var_map[var_name])
                else:
                    if lit in var_map:
                        and_terms.append(var_map[lit])
            if len(and_terms) == 1:
                or_terms.append(and_terms[0])
            elif and_terms:
                or_terms.append(And(*and_terms))
        return Or(*or_terms) if or_terms else expr(0)
    
    expr_pyeda = parse_sop(expr_str)
    
    # Validate against K-map
    target_val = 0 if form == 'pos' else 1
    
    for i in range(2**num_vars):
        assign = {}
        for j, v in enumerate(pyeda_vars):
            bit = (i >> (num_vars - 1 - j)) & 1
            assign[v] = expr(1) if bit else expr(0)
        
        # Get expected value from K-map
        if num_vars == 3:
            col = (i >> 1) & 3
            row = i & 1
            col_idx = [0, 1, 3, 2][col]  # Binary to Gray code
            expected = kmap[row][col_idx]
        elif num_vars == 2:
            col = i & 1
            row = (i >> 1) & 1
            expected = kmap[row][col]
        else:  # 4 vars
            col = (i >> 2) & 3
            row = i & 3
            col_idx = [0, 1, 3, 2][col]
            row_idx = [0, 1, 3, 2][row]
            expected = kmap[row_idx][col_idx]
        
        # Skip don't-cares
        if expected == 'd':
            continue
        
        # Evaluate expression
        val = expr_pyeda.restrict(assign)
        try:
            result = 1 if val.is_one() else 0
        except:
            result = 1 if str(val) == '1' else 0
        
        # Must match expected value
        if result != expected:
            return False, f"Mismatch at input {i}: expected {expected}, got {result}"
    
    return True, "OK"

# Run comprehensive tests
print("COMPREHENSIVE BOOLMIN2D VALIDATION")
print("="*70)
print()

test_cases = [
    # Edge cases
    ("All zeros 2-var", [[0, 0], [0, 0]], 'sop'),
    ("All ones 2-var", [[1, 1], [1, 1]], 'sop'),
    ("All don't-cares 2-var", [['d', 'd'], ['d', 'd']], 'sop'),
    ("Single 1", [[1, 0], [0, 0]], 'sop'),
    ("Single 0", [[0, 1], [1, 1]], 'pos'),
    
    # 3-variable cases with don't-cares
    ("Sparse 3-var", [[1, 0, 0, 'd'], [0, 0, 1, 0]], 'sop'),
    ("Dense 3-var", [[1, 1, 1, 0], [1, 'd', 1, 1]], 'sop'),
    ("Balanced 3-var", [[1, 'd', 1, 0], [0, 1, 'd', 1]], 'sop'),
    
    # Cases that previously failed
    ("Test 5 regression", [[0, 0, 0, 'd'], [0, 'd', 1, 0]], 'sop'),
    ("Test 8 regression", [[1, 'd', 1, 0], [1, 1, 'd', 1]], 'sop'),
    ("Test 19 regression", [[0, 0, 0, 0], [1, 'd', 1, 1]], 'sop'),
    
    # 4-variable case
    ("Simple 4-var", [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], 'sop'),
]

pass_count = 0
fail_count = 0

for name, kmap, form in test_cases:
    valid, msg = validate_kmap_result(kmap, form)
    
    if valid:
        print(f"[PASS] {name}")
        pass_count += 1
    else:
        print(f"[FAIL] {name}: {msg}")
        print(f"       K-map: {kmap}")
        fail_count += 1

print()
print("="*70)
print(f"RESULTS: {pass_count} passed, {fail_count} failed")
if fail_count == 0:
    print("SUCCESS: All tests passed! BoolMin2D ensures complete coverage.")
else:
    print(f"FAILURE: {fail_count} test(s) failed.")
print("="*70)

# Run random tests
print()
print("Running 100 random tests...")
random.seed(42)

random_pass = 0
random_fail = 0

for i in range(100):
    # Generate random K-map
    kmap = [[random.choice([0, 1, 'd', 0]) for _ in range(4)] for _ in range(2)]
    
    valid, msg = validate_kmap_result(kmap, 'sop')
    
    if valid:
        random_pass += 1
    else:
        random_fail += 1
        if random_fail <= 5:  # Print first 5 failures
            print(f"  Random test {i+1} FAILED: {msg}")
            print(f"  K-map: {kmap}")

print(f"\nRandom tests: {random_pass}/100 passed ({100*random_pass/100:.0f}%)")
if random_pass == 100:
    print("SUCCESS: All random tests passed!")
print("="*70)

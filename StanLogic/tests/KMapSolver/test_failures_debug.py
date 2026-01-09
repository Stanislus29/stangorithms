"""
Debug script to analyze specific benchmark failures
Reproduces failing test cases to understand the issue
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from stanlogic.BoolMin2D import BoolMin2D
from pyeda.inter import *
from sympy import symbols, sympify, simplify, Equivalent as SymEquivalent, And as SymAnd, Or as SymOr, Not as SymNot
import random

def check_equivalence_detailed(kmap, pyeda_result, kmap_result, num_vars):
    """Check if two expressions are logically equivalent"""
    print(f"\nChecking equivalence:")
    print(f"  PyEDA: {pyeda_result}")
    print(f"  KMap:  {kmap_result}")
    
    # Create variable symbols
    if num_vars == 2:
        x1, x2 = map(exprvar, ['x1', 'x2'])
        vars_list = [x1, x2]
    elif num_vars == 3:
        x1, x2, x3 = map(exprvar, ['x1', 'x2', 'x3'])
        vars_list = [x1, x2, x3]
    else:  # 4 vars
        x1, x2, x3, x4 = map(exprvar, ['x1', 'x2', 'x3', 'x4'])
        vars_list = [x1, x2, x3, x4]
    
    # Build truth tables for both expressions
    print("\n  Truth table comparison:")
    print("  " + " ".join([f"x{i+1}" for i in range(num_vars)]) + " | PyEDA | KMap | Match")
    print("  " + "-" * (num_vars*3 + 20))
    
    all_match = True
    for i in range(2**num_vars):
        # Generate input values
        values = []
        for j in range(num_vars):
            values.append((i >> (num_vars-1-j)) & 1)
        
        # Evaluate PyEDA expression
        point = {f'x{j+1}': values[j] for j in range(num_vars)}
        try:
            pyeda_val = pyeda_result.restrict(point)
            pyeda_val = 1 if pyeda_val.is_one() else 0
        except:
            pyeda_val = "?"
        
        # Evaluate KMap expression by checking original K-map
        # Convert input values to row/col based on convention
        if num_vars == 2:
            row = values[1]
            col = values[0]
        elif num_vars == 3:
            row = values[2]
            col = values[0] * 2 + values[1]
        else:  # 4 vars
            row = values[2] * 2 + values[3]
            col = values[0] * 2 + values[1]
        
        kmap_val = kmap[row][col]
        if kmap_val == 'd':
            kmap_expected = "d"
        else:
            kmap_expected = kmap_val
        
        match = "✓" if (pyeda_val == kmap_expected or kmap_expected == "d") else "✗"
        if match == "✗":
            all_match = False
        
        values_str = " ".join([str(v) for v in values])
        print(f"  {values_str}  | {pyeda_val:^5} | {kmap_expected:^4} | {match}")
    
    return all_match

def test_case(test_num, kmap_3var, form='sop'):
    """Test a specific case"""
    print(f"\n{'='*70}")
    print(f"TEST {test_num}")
    print(f"{'='*70}")
    print(f"K-map: {kmap_3var}")
    print(f"Form: {form}")
    
    # Run BoolMin2D
    solver = BoolMin2D(kmap_3var)
    kmap_terms, kmap_expr = solver.minimize(form=form)
    
    print(f"\nBoolMin2D Result:")
    print(f"  Terms: {kmap_terms}")
    print(f"  Expression: {kmap_expr}")
    print(f"  Literal count: {len(kmap_expr.replace(' ', '').replace('+', '').replace("'", ''))}")
    
    # Run PyEDA
    x1, x2, x3 = map(exprvar, ['x1', 'x2', 'x3'])
    
    # Build minterms from k-map
    minterms = []
    for r in range(len(kmap_3var)):
        for c in range(len(kmap_3var[0])):
            if kmap_3var[r][c] == 1:
                # Convert position to minterm
                # Convention: cols = x1x2, rows = x3
                if len(kmap_3var[0]) == 4:  # 2x4
                    col_bits = ['00', '01', '11', '10'][c]
                    row_bit = str(r)
                    minterm_idx = int(col_bits + row_bit, 2)
                    minterms.append(minterm_idx)
    
    if not minterms:
        pyeda_expr = expr(0)
        pyeda_terms_count = 0
    else:
        # Build expression from minterms
        terms = []
        for m in minterms:
            bits = format(m, '03b')  # 3-bit binary
            term = expr(1)
            for i, bit in enumerate(bits):
                var = [x1, x2, x3][i]
                if bit == '1':
                    term = term & var
                else:
                    term = term & ~var
            terms.append(term)
        
        pyeda_expr = expr(0)
        for term in terms:
            pyeda_expr = pyeda_expr | term
        
        # Minimize with PyEDA
        pyeda_expr = pyeda_expr.simplify()
        pyeda_str = str(pyeda_expr)
        pyeda_terms_count = len(pyeda_str.replace(' ', '').replace('|', '').replace('~', ''))
    
    print(f"\nPyEDA Result:")
    print(f"  Expression: {pyeda_expr}")
    print(f"  Literal count: {pyeda_terms_count}")
    
    # Check equivalence
    equiv = check_equivalence_detailed(kmap_3var, pyeda_expr, kmap_expr, 3)
    print(f"\n{'✓' if equiv else '✗'} Equivalence: {equiv}")
    
    return equiv

# Test failing cases from benchmark
print("Testing failing cases from benchmark...")

# Test 8: 3-sop, sparse, False (PyEDA: 6 literals, KMap: 5 literals)
random.seed(8)
kmap = []
for _ in range(2):
    row = []
    for _ in range(4):
        val = random.choice([0, 1, 'd', 0, 0])  # sparse
        row.append(val)
    kmap.append(row)
test_case(8, kmap, 'sop')

# Test 13: 3-sop, sparse, False (PyEDA: 3 literals, KMap: 2 literals)
random.seed(13)
kmap = []
for _ in range(2):
    row = []
    for _ in range(4):
        val = random.choice([0, 1, 'd', 0, 0])  # sparse
        row.append(val)
    kmap.append(row)
test_case(13, kmap, 'sop')

# Test 45: 3-sop, balanced, False (PyEDA: 6 literals, KMap: 0 literals!)
random.seed(45)
kmap = []
for _ in range(2):
    row = []
    for _ in range(4):
        val = random.choice([0, 1, 'd'])  # balanced
        row.append(val)
    kmap.append(row)
test_case(45, kmap, 'sop')

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

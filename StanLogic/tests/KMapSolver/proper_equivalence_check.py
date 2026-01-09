"""
Proper equivalence check considering don't-cares
Two expressions are equivalent w.r.t. a K-map if they agree on all non-don't-care cells
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from stanlogic.BoolMin2D import BoolMin2D
from pyeda.inter import *
import random
import re

def proper_equivalence_check(kmap, expr_pyeda, kmap_expr_str, num_vars=3):
    """
    Check if two expressions are equivalent with respect to the K-map specification.
    They must agree on all non-don't-care inputs.
    """
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
    
    expr_kmap = parse_sop(kmap_expr_str)
    
    # Check all inputs
    all_match = True
    for i in range(2**num_vars):
        assign = {}
        for j, v in enumerate(pyeda_vars):
            bit = (i >> (num_vars - 1 - j)) & 1
            assign[v] = expr(1) if bit else expr(0)
        
        # Get expected value from K-map
        col = (i >> 1) & 3  # x1 x2
        row = i & 1         # x3
        col_idx = [0, 1, 3, 2][col]  # Binary to Gray code
        expected = kmap[row][col_idx]
        
        # If this is a don't-care, skip the check
        if expected == 'd':
            continue
        
        # Evaluate both expressions
        val1 = expr_pyeda.restrict(assign)
        try:
            pyeda_val = 1 if val1.is_one() else 0
        except:
            pyeda_val = 1 if str(val1) == '1' else 0
        
        val2 = expr_kmap.restrict(assign)
        try:
            kmap_val = 1 if val2.is_one() else 0
        except:
            kmap_val = 1 if str(val2) == '1' else 0
        
        # Both must match the expected value
        if pyeda_val != expected or kmap_val != expected:
            all_match = False
            return False
    
    return True

# Test the failures with proper checking
tests = [
    (8, [[1, 'd', 1, 0], [1, 1, 'd', 1]]),
    (19, [[0, 0, 0, 0], [1, 'd', 1, 1]]),
    (22, [['d', 1, 1, 'd'], [1, 1, 0, 1]]),
    (27, [[0, 0, 'd', 1], [1, 0, 1, 1]]),
    (32, [[1, 1, 1, 'd'], [0, 0, 1, 0]]),
    (41, [[1, 1, 1, 'd'], [0, 0, 0, 1]]),
]

print("Testing with PROPER equivalence check (considering don't-cares):")
print("="*70)

for test_num, kmap in tests:
    random.seed(test_num)
    
    # BoolMin2D
    solver = BoolMin2D(kmap)
    kmap_terms, kmap_expr = solver.minimize(form='sop')
    
    # PyEDA
    x1, x2, x3 = map(exprvar, ['x1', 'x2', 'x3'])
    pyeda_vars = [x1, x2, x3]
    
    minterms = []
    for r in range(2):
        for c in range(4):
            if kmap[r][c] == 1:
                col_bits = ['00', '01', '11', '10'][c]
                row_bit = str(r)
                minterm_idx = int(col_bits + row_bit, 2)
                minterms.append(minterm_idx)
    
    if not minterms:
        expr_pyeda = expr(0)
    else:
        terms = []
        for m in minterms:
            bits = format(m, '03b')
            term = expr(1)
            for i, bit in enumerate(bits):
                var = pyeda_vars[i]
                if bit == '1':
                    term = term & var
                else:
                    term = term & ~var
            terms.append(term)
        
        expr_pyeda = expr(0)
        for term in terms:
            expr_pyeda = expr_pyeda | term
        expr_pyeda = expr_pyeda.simplify()
    
    # Proper equivalence check
    equiv = proper_equivalence_check(kmap, expr_pyeda, kmap_expr)
    
    status = "PASS" if equiv else "FAIL"
    print(f"Test {test_num}: {status}")
    if not equiv:
        print(f"  K-map:     {kmap}")
        print(f"  PyEDA:     {expr_pyeda}")
        print(f"  BoolMin2D: {kmap_expr}")

print("="*70)
print("\nConclusion: If all pass, then BoolMin2D is producing valid")
print("minimizations that just handle don't-cares differently from PyEDA.")

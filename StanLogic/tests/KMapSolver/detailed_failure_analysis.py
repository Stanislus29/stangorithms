"""
Detailed analysis of remaining failures
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from stanlogic.BoolMin2D import BoolMin2D
from pyeda.inter import *
import random

def detailed_check(test_num, kmap):
    """Detailed truth table check for a specific test"""
    print(f"\n{'='*70}")
    print(f"TEST {test_num}")
    print(f"{'='*70}")
    print(f"K-map: {kmap}")
    
    # BoolMin2D
    solver = BoolMin2D(kmap)
    kmap_terms, kmap_expr = solver.minimize(form='sop')
    print(f"\nBoolMin2D: {kmap_expr}")
    
    # PyEDA
    x1, x2, x3 = map(exprvar, ['x1', 'x2', 'x3'])
    pyeda_vars = [x1, x2, x3]
    
    # Build minterms
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
    
    print(f"PyEDA:     {expr_pyeda}")
    
    # Truth table comparison
    print(f"\nTruth Table:")
    print("  x1 x2 x3 | Expected | PyEDA | KMap | Match")
    print("  " + "-"*45)
    
    # Parse BoolMin2D expression
    import re
    var_map = {'x1': x1, 'x2': x2, 'x3': x3}
    
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
    
    expr_kmap = parse_sop(kmap_expr)
    
    all_match = True
    mismatches = []
    for i in range(8):
        assign = {}
        for j, v in enumerate(pyeda_vars):
            bit = (i >> (2 - j)) & 1
            assign[v] = expr(1) if bit else expr(0)
        
        # Expected value from K-map
        col = (i >> 1) & 3  # x1 x2 (2 bits)
        row = i & 1         # x3 (1 bit)
        # Map to Gray code positions
        col_idx = [0, 1, 3, 2][col]  # Binary to Gray code
        expected = kmap[row][col_idx]
        expected_val = '?' if expected == 'd' else str(expected)
        
        # PyEDA value
        val1 = expr_pyeda.restrict(assign)
        try:
            pyeda_val = 1 if val1.is_one() else 0
        except:
            pyeda_val = 1 if str(val1) == '1' else 0
        
        # KMap value
        val2 = expr_kmap.restrict(assign)
        try:
            kmap_val = 1 if val2.is_one() else 0
        except:
            kmap_val = 1 if str(val2) == '1' else 0
        
        # Check match
        match = (pyeda_val == kmap_val)
        if not match:
            all_match = False
            mismatches.append((i, pyeda_val, kmap_val))
        
        match_sym = "OK" if match else "MISMATCH"
        print(f"  {(i>>2)&1}  {(i>>1)&1}  {i&1}  | {expected_val:^8} | {pyeda_val:^5} | {kmap_val:^4} | {match_sym}")
    
    print(f"\n{'='*70}")
    if all_match:
        print("VERDICT: EQUIVALENT (BoolMin2D is correct!)")
    else:
        print(f"VERDICT: NOT EQUIVALENT ({len(mismatches)} mismatches)")
        print(f"Mismatches: {mismatches}")
    print(f"{'='*70}")
    
    return all_match

# Test the failures
random.seed(8)
kmap8 = [[1, 'd', 1, 0], [1, 1, 'd', 1]]
detailed_check(8, kmap8)

random.seed(19)
kmap19 = [[0, 0, 0, 0], [1, 'd', 1, 1]]
detailed_check(19, kmap19)

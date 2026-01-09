"""
Quick benchmark test to check equivalence rates
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from stanlogic.BoolMin2D import BoolMin2D
from pyeda.inter import *
import random

# Configure
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def random_kmap(rows=2, cols=4, p_one=0.45, p_dc=0.1):
    """Generate random K-map"""
    p_zero = 1 - p_one - p_dc
    choices = [1, 0, 'd']
    weights = [p_one, p_zero, p_dc]
    return [[random.choices(choices, weights=weights, k=1)[0] 
             for _ in range(cols)] for _ in range(rows)]

def check_truth_table_equivalence(kmap, expr_pyeda, expr_kmap_str, num_vars):
    """Check equivalence via truth table"""
    pyeda_vars = [exprvar(f'x{i+1}') for i in range(num_vars)]
    
    # Parse BoolMin2D expression to PyEDA
    import re
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
    
    expr_kmap = parse_sop(expr_kmap_str)
    
    # Compare truth tables
    for i in range(2**num_vars):
        assign = {}
        for j, v in enumerate(pyeda_vars):
            bit = (i >> (num_vars - 1 - j)) & 1
            assign[v] = expr(1) if bit else expr(0)
        
        val1 = expr_pyeda.restrict(assign)
        val2 = expr_kmap.restrict(assign)
        
        try:
            b1 = 1 if val1.is_one() else 0
        except:
            b1 = 1 if str(val1) == '1' else 0
        
        try:
            b2 = 1 if val2.is_one() else 0
        except:
            b2 = 1 if str(val2) == '1' else 0
        
        if b1 != b2:
            return False
    
    return True

# Run quick tests
print("Running quick benchmark...")
print("="*60)

results = {"pass": 0, "fail": 0, "error": 0}
num_tests = 50

for test_idx in range(1, num_tests + 1):
    random.seed(test_idx)
    kmap = random_kmap(rows=2, cols=4, p_one=0.45, p_dc=0.1)
    
    try:
        # BoolMin2D
        solver = BoolMin2D(kmap)
        kmap_terms, kmap_expr = solver.minimize(form='sop')
        
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
        
        # PyEDA minimization
        if not minterms:
            expr_pyeda = expr(0)
        else:
            # Build expression
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
        
        # Check equivalence via truth table
        equiv = check_truth_table_equivalence(kmap, expr_pyeda, kmap_expr, 3)
        
        if equiv:
            results["pass"] += 1
            status = "PASS"
        else:
            results["fail"] += 1
            status = "FAIL"
            print(f"Test {test_idx}: {status}")
            print(f"  K-map: {kmap}")
            print(f"  PyEDA: {expr_pyeda}")
            print(f"  KMap:  {kmap_expr}")
        
    except Exception as e:
        results["error"] += 1
        print(f"Test {test_idx}: ERROR - {str(e)[:50]}")

print("\n" + "="*60)
print("RESULTS:")
print(f"  Pass:  {results['pass']}/{num_tests} ({100*results['pass']/num_tests:.1f}%)")
print(f"  Fail:  {results['fail']}/{num_tests} ({100*results['fail']/num_tests:.1f}%)")
print(f"  Error: {results['error']}/{num_tests} ({100*results['error']/num_tests:.1f}%)")
print("="*60)

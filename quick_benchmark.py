"""Quick benchmark to test optimization improvements."""
import sys
import os
import random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'StanLogic', 'src')))

from stanlogic.BoolMinGeo import BoolMinGeo
from pyeda.inter import *
import time
import re

random.seed(42)

def count_literals(expr_str):
    if not expr_str or expr_str.strip() == "":
        return 0
    s = expr_str.replace(" ", "")
    terms = [t for t in s.split('+') if t]
    return sum(len(re.findall(r"[A-Za-z]+\d*'?", t)) for t in terms)

def count_pyeda_literals(expr):
    s = str(expr).strip()
    if s in ("1", "0", "True", "False"):
        return 0
    
    def split_top_level(text, sep):
        parts = []
        buf = []
        depth = 0
        for ch in text:
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            if ch == sep and depth == 0:
                if buf:
                    parts.append(''.join(buf).strip())
                buf = []
            else:
                buf.append(ch)
        if buf:
            parts.append(''.join(buf).strip())
        return parts
    
    top_terms = split_top_level(s, '|')
    total_literals = 0
    for term in top_terms:
        term = term.strip()
        while term.startswith('(') and term.endswith(')'):
            term = term[1:-1].strip()
        if not term:
            continue
        lits = split_top_level(term, '&')
        literal_count = sum(1 for lit in lits if re.fullmatch(r"~?[A-Za-z_]\w*", lit.strip()))
        total_literals += literal_count
    return total_literals

def create_minterm_fixed(vars_list, index):
    terms = []
    for i, var in enumerate(vars_list):
        if (index >> i) & 1:
            terms.append(var)
        else:
            terms.append(~var)
    return And(*terms) if len(terms) > 1 else terms[0]

# Test 5 random functions
test_cases = [
    (5, [1 if random.random() < 0.3 else 0 for _ in range(32)]),
    (6, [1 if random.random() < 0.5 else 0 for _ in range(64)]),
    (7, [1 if random.random() < 0.4 else 0 for _ in range(128)]),
    (8, [1 if random.random() < 0.5 else 0 for _ in range(256)]),
    (8, [1 if random.random() < 0.6 else 0 for _ in range(256)]),
]

print("="*80)
print("QUICK BENCHMARK - 5 TEST CASES")
print("="*80)

total_pyeda_lits = 0
total_kmap_lits = 0

for i, (num_vars, output_values) in enumerate(test_cases, 1):
    print(f"\nTest {i}: {num_vars} variables, {sum(output_values)} ones")
    print("-"*80)
    
    # PyEDA
    minterms = [i for i, v in enumerate(output_values) if v == 1]
    pyeda_vars = [exprvar(chr(97+i)) for i in range(num_vars)]
    
    if minterms:
        f_on = Or(*[create_minterm_fixed(pyeda_vars, mt) for mt in minterms])
        expr_pyeda = espresso_exprs(f_on)[0]
        pyeda_lits = count_pyeda_literals(expr_pyeda)
    else:
        pyeda_lits = 0
    
    print(f"  PyEDA: {pyeda_lits} literals")
    
    # BoolMinGeo
    import io
    import sys as sys_module
    old_stdout = sys_module.stdout
    sys_module.stdout = io.StringIO()
    
    solver = BoolMinGeo(num_vars, output_values)
    terms, expr_str = solver.minimize_3d(form='sop')
    
    sys_module.stdout = old_stdout
    
    kmap_lits = count_literals(expr_str)
    print(f"  BoolMinGeo: {kmap_lits} literals")
    print(f"  Ratio: {kmap_lits / max(pyeda_lits, 1):.3f}")
    
    total_pyeda_lits += pyeda_lits
    total_kmap_lits += kmap_lits

print("\n" + "="*80)
print("OVERALL RESULTS")
print("="*80)
print(f"Total PyEDA literals: {total_pyeda_lits}")
print(f"Total BoolMinGeo literals: {total_kmap_lits}")
print(f"Overall ratio: {total_kmap_lits / max(total_pyeda_lits, 1):.3f}")

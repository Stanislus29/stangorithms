"""
Quick test to verify the equivalence checking fix works
"""

from pyeda.inter import *
from stanlogic import BoolMin2D
from sympy import symbols
import re

# Create a simple 2x2 K-map
kmap = [
    [0, 1],
    [1, 1]
]

print("K-map:")
for row in kmap:
    print(f"  {row}")

# Get BoolMin2D result
solver = BoolMin2D(kmap)
terms, expr_str = solver.minimize(form='sop')
print(f"\nBoolMin2D result: '{expr_str}'")

# Create PyEDA variables with x1, x2 names (matching BoolMin2D)
pyeda_vars = [exprvar('x1'), exprvar('x2')]
x1, x2 = pyeda_vars

# Create PyEDA expression from K-map truth table
# Minterms: 01 (x1=0,x2=1), 10 (x1=1,x2=0), 11 (x1=1,x2=1)
pyeda_expr_manual = Or(And(~x1, x2), And(x1, ~x2), And(x1, x2))
print(f"PyEDA expression (from minterms): {pyeda_expr_manual}")

# Minimize with espresso
pyeda_minimized = espresso_exprs(pyeda_expr_manual)[0]
print(f"PyEDA minimized: {pyeda_minimized}")

# Parse BoolMin2D output to PyEDA using the parse function
def parse_kmap_to_pyeda(expr_str, pyeda_vars, form="sop"):
    """Parse BoolMin2D expression to PyEDA format"""
    if not expr_str or expr_str.strip() == "":
        return expr(0) if form == "sop" else expr(1)

    var_map = {str(v): v for v in pyeda_vars}

    if form.lower() == "sop":
        or_terms = []
        for product in expr_str.split('+'):
            product = product.strip()
            if not product:
                continue
            literals = re.findall(r"[A-Za-z_]\w*'?", product)
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

    raise ValueError(f"Unsupported form '{form}'.")

parsed_expr = parse_kmap_to_pyeda(expr_str, pyeda_vars, form='sop')
print(f"Parsed BoolMin2D to PyEDA: {parsed_expr}")

# Check equivalence
equiv = pyeda_minimized.equivalent(parsed_expr)
print(f"\nEquivalence check: {equiv}")

if equiv:
    print("✓ SUCCESS: Equivalence checking works correctly!")
else:
    print("✗ FAILURE: Equivalence check failed")
    print(f"  PyEDA: {pyeda_minimized}")
    print(f"  Parsed: {parsed_expr}")

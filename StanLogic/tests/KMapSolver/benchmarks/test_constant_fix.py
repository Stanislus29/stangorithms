"""
Test constant expression parsing fix
"""

from pyeda.inter import *
from stanlogic import BoolMin2D
import re

def parse_kmap_to_pyeda(expr_str, pyeda_vars, terms=None, form="sop"):
    """Parse BoolMin2D expression to PyEDA format"""
    # Handle empty expression
    if not expr_str or expr_str.strip() == "":
        if terms is not None:
            if terms == ['']:
                return expr(1)  # Constant 1
            elif not terms:
                return expr(0)  # Constant 0
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

pyeda_vars = [exprvar('x1'), exprvar('x2')]

print("Test 1: Constant 1 (all ones)")
kmap1 = [[1, 1], [1, 1]]
solver1 = BoolMin2D(kmap1)
terms1, expr1 = solver1.minimize(form='sop')
print(f"  BoolMin2D: expr='{expr1}', terms={terms1}")
parsed1 = parse_kmap_to_pyeda(expr1, pyeda_vars, terms=terms1, form='sop')
print(f"  Parsed: {parsed1}")
print(f"  Expected: 1")
print(f"  Match: {str(parsed1) == '1'}")

print("\nTest 2: Constant 0 (all zeros)")
kmap2 = [[0, 0], [0, 0]]
solver2 = BoolMin2D(kmap2)
terms2, expr2 = solver2.minimize(form='sop')
print(f"  BoolMin2D: expr='{expr2}', terms={terms2}")
parsed2 = parse_kmap_to_pyeda(expr2, pyeda_vars, terms=terms2, form='sop')
print(f"  Parsed: {parsed2}")
print(f"  Expected: 0")
print(f"  Match: {str(parsed2) == '0'}")

print("\nTest 3: XOR pattern")
kmap3 = [[1, 0], [0, 1]]
solver3 = BoolMin2D(kmap3)
terms3, expr3 = solver3.minimize(form='sop')
print(f"  BoolMin2D: expr='{expr3}', terms={terms3}")
parsed3 = parse_kmap_to_pyeda(expr3, pyeda_vars, terms=terms3, form='sop')
print(f"  Parsed: {parsed3}")

# Build expected PyEDA expression
x1, x2 = pyeda_vars
expected3 = Or(And(x1, x2), And(~x1, ~x2))
print(f"  Expected: {expected3}")
print(f"  Equivalent: {parsed3.equivalent(expected3)}")

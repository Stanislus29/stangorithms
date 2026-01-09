"""
Test the actual parse_kmap_to_pyeda from benchmark file
"""

import sys
sys.path.insert(0, '.')

from pyeda.inter import *
from stanlogic import BoolMin2D

# Import the actual function
from benchmark_test2D_pyeda import parse_kmap_to_pyeda

pyeda_vars = [exprvar('x1'), exprvar('x2')]

print("Test 1: Constant 1 (all ones)")
kmap1 = [[1, 1], [1, 1]]
solver1 = BoolMin2D(kmap1)
terms1, expr1 = solver1.minimize(form='sop')
print(f"  BoolMin2D: expr='{expr1}', terms={terms1}")
parsed1 = parse_kmap_to_pyeda(expr1, pyeda_vars, terms=terms1, form='sop')
print(f"  Parsed: {parsed1}")
print(f"  Expected: 1")
print(f"  Match: {str(parsed1) == '1'}\n")

print("Test 2: Constant 0 (all zeros)")
kmap2 = [[0, 0], [0, 0]]
solver2 = BoolMin2D(kmap2)
terms2, expr2 = solver2.minimize(form='sop')
print(f"  BoolMin2D: expr='{expr2}', terms={terms2}")
parsed2 = parse_kmap_to_pyeda(expr2, pyeda_vars, terms=terms2, form='sop')
print(f"  Parsed: {parsed2}")
print(f"  Expected: 0")
print(f"  Match: {str(parsed2) == '0'}\n")

print("Test 3: XOR pattern")
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
print(f"  Equivalent: {parsed3.equivalent(expected3)}\n")

print("Test 4: Simple OR")
kmap4 = [[0, 1], [1, 1]]
solver4 = BoolMin2D(kmap4)
terms4, expr4 = solver4.minimize(form='sop')
print(f"  BoolMin2D: expr='{expr4}', terms={terms4}")
parsed4 = parse_kmap_to_pyeda(expr4, pyeda_vars, terms=terms4, form='sop')
print(f"  Parsed: {parsed4}")
expected4 = Or(x1, x2)
print(f"  Expected: {expected4}")
print(f"  Equivalent: {parsed4.equivalent(expected4)}")

"""
Test a single case to debug equivalence checking
"""
import sys
sys.path.insert(0, 'c:\\Users\\DELL\\Documents\\Research Projects\\Mathematical_models\\StanLogic\\src')

from stanlogic import BoolMin2D
from pyeda.inter import *
from sympy import symbols, sympify, simplify
from sympy.logic.boolalg import And as SymAnd, Or as SymOr, Not as SymNot, true, false
import random

# Create variables
x1, x2 = symbols('x1 x2', boolean=True)
a, b = map(exprvar, 'ab')

# Simple test case: 2 variables, minterms [1, 2]
# x1=0, x2=1: minterm 1
# x1=1, x2=0: minterm 2

# Build PyEDA expression
pyeda_vars = [a, b]
def pyeda_minterm(idx, nvars, pvars):
    """Build a PyEDA minterm"""
    return And(*[pvars[i] if (idx >> (nvars - 1 - i)) & 1 else ~pvars[i] for i in range(nvars)])

minterms = [1, 2]
f_on = Or(*[pyeda_minterm(m, 2, pyeda_vars) for m in minterms])
print(f"PyEDA ON-set: {f_on}")

# Minimize with PyEDA
try:
    pyeda_result = espresso_exprs(f_on.to_dnf())[0]
    print(f"PyEDA result: {pyeda_result}")
except Exception as e:
    print(f"PyEDA error: {e}")
    pyeda_result = f_on

# Parse PyEDA to SymPy
pyeda_to_sympy_map = {'a': x1, 'b': x2}
local_map = {"And": SymAnd, "Or": SymOr, "Not": SymNot}
local_map.update(pyeda_to_sympy_map)

pyeda_str = str(pyeda_result)
print(f"PyEDA string: {pyeda_str}")

sympy_from_pyeda = sympify(pyeda_str, locals=local_map)
print(f"SymPy from PyEDA: {sympy_from_pyeda}")
print(f"  Free symbols: {sympy_from_pyeda.free_symbols}")

# Build KMap expression
kmap_grid = [[0, 0], [0, 0]]  # 2x2 for 2-variable
# Set minterms
for m in minterms:
    row = m >> 1
    col = m & 1
    kmap_grid[row][col] = 1

print(f"\nKMap grid:")
for row in kmap_grid:
    print(row)

kmap = BoolMin2D(kmap_grid)
kmap_result = kmap.get_function()
print(f"KMap result: {kmap_result}")

# Parse KMap to SymPy
# KMap uses 'x1', 'x2' etc.
kmap_str = kmap_result.replace('·', '*').replace('\'', '~')
print(f"KMap string (cleaned): {kmap_str}")

# Parse it
var_names = [x1, x2]
kmap_namespace = {"And": SymAnd, "Or": SymOr, "Not": SymNot}
for v in var_names:
    kmap_namespace[str(v)] = v

# Convert operators
kmap_str_python = kmap_str.replace('*', ' & ').replace('+', ' | ').replace('~', ' ~ ')
print(f"KMap string (Python ops): {kmap_str_python}")

sympy_from_kmap = sympify(kmap_str_python, locals=kmap_namespace)
print(f"SymPy from KMap: {sympy_from_kmap}")
print(f"  Free symbols: {sympy_from_kmap.free_symbols}")

# Check equivalence
print(f"\n=== EQUIVALENCE CHECK ===")
print(f"PyEDA SymPy: {sympy_from_pyeda}")
print(f"KMap SymPy: {sympy_from_kmap}")
print(f"XOR: {sympy_from_pyeda ^ sympy_from_kmap}")
print(f"Simplified XOR: {simplify(sympy_from_pyeda ^ sympy_from_kmap)}")
print(f"Are they equivalent? {simplify(sympy_from_pyeda ^ sympy_from_kmap) == False}")

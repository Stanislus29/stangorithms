"""
Simple test to understand PyEDA expressions and parsing
"""
from pyeda.inter import *
from sympy import symbols, sympify
from sympy.logic.boolalg import And as SymAnd, Or as SymOr, Not as SymNot

# Create PyEDA variables
a, b, c = map(exprvar, 'abc')

# Create SymPy variables
x1, x2, x3 = symbols('x1 x2 x3', boolean=True)

# Test function
f = a & b | c
print(f"PyEDA expression: {f}")
print(f"Type: {type(f)}")
print(f"String: {str(f)}")
print(f"Repr: {repr(f)}")

# Try sympify with mapping
mapping = {'a': x1, 'b': x2, 'c': x3}
local_map = {"And": SymAnd, "Or": SymOr, "Not": SymNot}
local_map.update(mapping)

print(f"\nMapping: {mapping}")
print(f"Local map keys: {local_map.keys()}")

try:
    result = sympify(str(f), locals=local_map)
    print(f"Sympify result: {result}")
    print(f"Free symbols: {result.free_symbols}")
except Exception as e:
    print(f"Error: {e}")

# Let me try a different approach - replace the string first
f_str = str(f)
print(f"\nOriginal string: {f_str}")
for pvar, svar in mapping.items():
    f_str = f_str.replace(pvar, str(svar))
print(f"Modified string: {f_str}")

# Now sympify
try:
    result2 = sympify(f_str, locals={"And": SymAnd, "Or": SymOr, "Not": SymNot, 'x1': x1, 'x2': x2, 'x3': x3})
    print(f"Sympify result 2: {result2}")
    print(f"Free symbols: {result2.free_symbols}")
except Exception as e:
    print(f"Error: {e}")

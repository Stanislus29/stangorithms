from pyeda.inter import *
from sympy import symbols, simplify, Equivalent

a, b, c = map(exprvar, 'abc')

f1 = (a & b) | (~a & c)
f2 = (a & b) | (~a & c)

def check_equivalence(expr1, expr2):
    """Check if two SymPy expressions are logically equivalent."""
    try:
        return bool(simplify(Equivalent(expr1, expr2)))
    except Exception as e:
        return False

print(check_equivalence(f1,f2))   # True or False

"""
Quick test to verify PyEDA integration fix
"""
from pyeda.inter import *

# Test the fix: Create PyEDA variables and build minterms
pyeda_vars = [exprvar(name) for name in ['a', 'b']]

def pyeda_minterm(m, pvars):
    return And(*[
        pvar if (m >> i) & 1 else ~pvar
        for i, pvar in enumerate(reversed(pvars))
    ])

# Test case: 2-variable, minterms [1, 3], no don't cares
minterms = [1, 3]
dont_cares = []

print("Testing PyEDA integration...")
print(f"Variables: {pyeda_vars}")
print(f"Minterms: {minterms}")

# Build ON-set
if minterms:
    f_on = Or(*[pyeda_minterm(m, pyeda_vars) for m in minterms])
else:
    f_on = expr(0)

if dont_cares:
    f_dc = Or(*[pyeda_minterm(m, pyeda_vars) for m in dont_cares])
else:
    f_dc = expr(0)

print(f"\nON-set: {f_on}")
print(f"DC-set: {f_dc}")

# Try espresso
try:
    result = espresso_exprs(f_on, f_dc)[0]
    print(f"\n✓ SUCCESS!")
    print(f"Result: {result}")
    print(f"Type: {type(result)}")
except Exception as e:
    print(f"\n✗ FAILED: {e}")

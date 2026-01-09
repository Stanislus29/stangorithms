"""
Test to understand BoolMin2D output format for empty/constant cases
"""

from stanlogic import BoolMin2D

print("Testing BoolMin2D output formats\n")

# Test 1: All ones (should be constant 1)
kmap1 = [[1, 1], [1, 1]]
solver1 = BoolMin2D(kmap1)
terms1, expr1 = solver1.minimize(form='sop')
print(f"All ones (constant 1):")
print(f"  Expression: '{expr1}'")
print(f"  Terms: {terms1}")
print(f"  Length: {len(expr1)}")
print(f"  Repr: {repr(expr1)}")

# Test 2: All zeros (should be constant 0)
kmap2 = [[0, 0], [0, 0]]
solver2 = BoolMin2D(kmap2)
terms2, expr2 = solver2.minimize(form='sop')
print(f"\nAll zeros (constant 0):")
print(f"  Expression: '{expr2}'")
print(f"  Terms: {terms2}")
print(f"  Length: {len(expr2)}")
print(f"  Repr: {repr(expr2)}")

# Test 3: XOR pattern
kmap3 = [[1, 0], [0, 1]]
solver3 = BoolMin2D(kmap3)
terms3, expr3 = solver3.minimize(form='sop')
print(f"\nXOR pattern:")
print(f"  Expression: '{expr3}'")
print(f"  Terms: {terms3}")
print(f"  Split by +: {expr3.split('+')}")

# Test 4: Simple OR
kmap4 = [[0, 1], [1, 1]]
solver4 = BoolMin2D(kmap4)
terms4, expr4 = solver4.minimize(form='sop')
print(f"\nSimple OR:")
print(f"  Expression: '{expr4}'")
print(f"  Terms: {terms4}")
print(f"  Split by +: {expr4.split('+')}")

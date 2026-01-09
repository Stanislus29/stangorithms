"""
Debug POS form output from BoolMin2D
"""

from stanlogic import BoolMin2D

print("Testing BoolMin2D POS form outputs\n")

# Test 1: XOR
kmap1 = [[1, 0], [0, 1]]
solver1 = BoolMin2D(kmap1)
terms1, expr1 = solver1.minimize(form='pos')
print(f"XOR pattern:")
print(f"  Expression: '{expr1}'")
print(f"  Terms: {terms1}")
print(f"  Contains '*': {'*' in expr1}")

# Test 2: All zeros (constant 0 in POS)
kmap2 = [[0, 0], [0, 0]]
solver2 = BoolMin2D(kmap2)
terms2, expr2 = solver2.minimize(form='pos')
print(f"\nAll zeros (constant 0):")
print(f"  Expression: '{expr2}'")
print(f"  Terms: {terms2}")
print(f"  Repr: {repr(expr2)}")

# Test 3: All ones (constant 1 in POS)
kmap3 = [[1, 1], [1, 1]]
solver3 = BoolMin2D(kmap3)
terms3, expr3 = solver3.minimize(form='pos')
print(f"\nAll ones (constant 1):")
print(f"  Expression: '{expr3}'")
print(f"  Terms: {terms3}")
print(f"  Repr: {repr(expr3)}")

# Test 4: Single zero
kmap4 = [[0, 1], [1, 1]]
solver4 = BoolMin2D(kmap4)
terms4, expr4 = solver4.minimize(form='pos')
print(f"\nSingle zero:")
print(f"  Expression: '{expr4}'")
print(f"  Terms: {terms4}")

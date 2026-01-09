from sympy import symbols, simplify
from sympy.logic.boolalg import And, Or, Not, Equivalent

x1, x2 = symbols('x1 x2', boolean=True)

# Test equivalent expressions
expr1 = (x1 & x2) | (~x1 & ~x2)
expr2 = ~(x1 ^ x2)

print(f"expr1: {expr1}")
print(f"expr2: {expr2}")

# Using Equivalent function
equiv_expr = Equivalent(expr1, expr2)
print(f"\nEquivalent(expr1, expr2): {equiv_expr}")
print(f"Simplify: {simplify(equiv_expr)}")

# Another way - check if XOR is always false
xor_expr = expr1 ^ expr2
print(f"\nXOR: {xor_expr}")

# Truth table check
print("\nTruth table:")
for v1 in [True, False]:
    for v2 in [True, False]:
        val1 = expr1.subs({x1: v1, x2: v2})
        val2 = expr2.subs({x1: v1, x2: v2})
        print(f"x1={v1}, x2={v2}: expr1={val1}, expr2={val2}, match={val1==val2}")

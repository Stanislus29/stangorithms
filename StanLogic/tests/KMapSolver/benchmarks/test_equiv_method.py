from sympy import symbols, sympify
from sympy.logic.boolalg import And, Or, Not

x1, x2 = symbols('x1 x2', boolean=True)

# Test equivalent expressions
expr1 = (x1 & x2) | (~x1 & ~x2)
expr2 = ~(x1 ^ x2)

print(f"expr1: {expr1}")
print(f"expr2: {expr2}")
print(f"XOR: {expr1 ^ expr2}")
print(f"Are they equal by ==: {expr1 == expr2}")
print(f"Are they equivalent by .equivalent(): {expr1.equivalent(expr2)}")

# Test with exact same expression
expr3 = (x1 & x2) | (~x1 & ~x2)
print(f"\nexpr1: {expr1}")
print(f"expr3: {expr3}")
print(f"Are they equal by ==: {expr1 == expr3}")
print(f"Are they equivalent by .equivalent(): {expr1.equivalent(expr3)}")


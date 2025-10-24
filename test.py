from sympy import symbols, simplify_logic

x1, x2, x3, x4 = symbols('x1 x2 x3 x4')
E1 = (~x3) | (~x1 & ~x4) | (~x2 & ~x4)
E2 = (~x2 & ~x4) | (~x2 & ~x3) | (~x1)
equiv = simplify_logic(E1 ^ E2, form='dnf')
print(equiv)

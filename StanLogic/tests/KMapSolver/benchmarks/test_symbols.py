from sympy import symbols

var_pool = symbols('x1 x2 x3 x4')
print(f"Type: {type(var_pool)}")
print(f"Content: {var_pool}")
print(f"var_pool[:2]: {var_pool[:2]}")
print(f"var_pool[0]: {var_pool[0]}, type: {type(var_pool[0])}")

# Test mapping
mapping = {'a': var_pool[0], 'b': var_pool[1]}
print(f"\nMapping: {mapping}")
print(f"Keys: {mapping.keys()}")
print(f"Values: {mapping.values()}")

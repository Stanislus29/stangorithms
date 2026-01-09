"""
Quick test to see PyEDA DNF output format
"""
from pyeda.inter import *

# Create variables
a, b, c = map(exprvar, 'abc')

print("=" * 60)
print("Testing PyEDA Espresso Output Formats")
print("=" * 60)

# Test 1: Simple SOP minimization
print("\n1. Simple SOP (minterms [1, 2, 5, 7]):")
print("-" * 40)
f_on = Or(And(~a, ~b, c), And(~a, b, ~c), And(a, ~b, c), And(a, b, c))
f_dc = expr(0)  # No don't cares

result = espresso_exprs(f_on, f_dc)[0]
print(f"ON-set:  {f_on}")
print(f"Result:  {result}")
print(f"Type:    {type(result)}")
print(f"String:  {str(result)}")

# Test 2: With don't cares
print("\n2. With Don't Cares (minterms [0, 1], dc [2, 3]):")
print("-" * 40)
f_on2 = Or(And(~a, ~b, ~c), And(~a, ~b, c))
f_dc2 = Or(And(~a, b, ~c), And(~a, b, c))

result2 = espresso_exprs(f_on2, f_dc2)[0]
print(f"ON-set:  {f_on2}")
print(f"DC-set:  {f_dc2}")
print(f"Result:  {result2}")
print(f"Type:    {type(result2)}")

# Test 3: All ones
print("\n3. All Ones (should return constant 1):")
print("-" * 40)
f_on3 = Or(And(~a, ~b), And(~a, b), And(a, ~b), And(a, b))
f_dc3 = expr(0)

result3 = espresso_exprs(f_on3, f_dc3)[0]
print(f"Result:  {result3}")
print(f"Type:    {type(result3)}")
print(f"Is constant: {isinstance(result3, (int, bool))}")

# Test 4: Complement for POS (OFF-set minimization)
print("\n4. POS via OFF-set complement:")
print("-" * 40)
# Minimize OFF-set [0, 2, 3]
f_off = Or(And(~a, ~b, ~c), And(~a, b, ~c), And(~a, b, c))
f_dc4 = expr(0)

result_off = espresso_exprs(f_off, f_dc4)[0]
result_pos = ~result_off
print(f"OFF-set:     {f_off}")
print(f"OFF result:  {result_off}")
print(f"POS (~OFF):  {result_pos}")

# Test 5: What inputs does the result have?
print("\n5. Checking result inputs/variables:")
print("-" * 40)
print(f"Variables in result: {list(result.inputs)}")
print(f"Can iterate inputs:  {[str(v) for v in result.inputs]}")

print("\n" + "=" * 60)
print("Test Complete!")
print("=" * 60)

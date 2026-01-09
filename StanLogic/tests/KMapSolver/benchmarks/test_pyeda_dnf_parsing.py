"""
Test script to verify PyEDA DNF output format and parsing from BoolMin2D
"""

from pyeda.inter import *
from stanlogic import BoolMin2D
import re

print("=" * 70)
print("TEST 1: PyEDA DNF Output Format")
print("=" * 70)

# Create PyEDA variables
a, b, c, d = map(exprvar, 'abcd')

# Test various expressions
print("\n1. Simple expression: a*b + a*~c")
expr1 = Or(And(a, b), And(a, ~c))
print(f"   Expression: {expr1}")
print(f"   Type: {type(expr1)}")
print(f"   String representation: '{str(expr1)}'")

print("\n2. Using espresso_exprs on a simple function")
# Create a simple function: F = Σ(1,2,5,6) with 2 variables (a,b)
# Minterms: 01, 10, 11 (using 2 variables)
f = Or(And(~a, b), And(a, ~b), And(a, b))
print(f"   Original: {f}")
minimized = espresso_exprs(f)[0]
print(f"   Minimized: {minimized}")
print(f"   String: '{str(minimized)}'")

print("\n3. Three variable example")
# F = Σ(1,3,5,7) = b (all minterms where b=1)
f3 = Or(And(~a, ~b, c), And(~a, b, c), And(a, ~b, c), And(a, b, c))
print(f"   Original: {f3}")
minimized3 = espresso_exprs(f3)[0]
print(f"   Minimized: {minimized3}")
print(f"   String: '{str(minimized3)}'")

print("\n" + "=" * 70)
print("TEST 2: BoolMin2D Output Format")
print("=" * 70)

# Create a simple 2x2 K-map
kmap_2x2 = [
    [0, 1],
    [1, 1]
]

print("\n1. K-map (2 variables):")
for row in kmap_2x2:
    print(f"   {row}")

solver = BoolMin2D(kmap_2x2)
terms, expr_str = solver.minimize(form='sop')
print(f"\n   BoolMin2D output: '{expr_str}'")
print(f"   Terms: {terms}")

# Create a 2x4 K-map (3 variables)
kmap_2x4 = [
    [0, 1, 1, 0],
    [0, 1, 1, 0]
]

print("\n2. K-map (3 variables):")
for row in kmap_2x4:
    print(f"   {row}")

solver2 = BoolMin2D(kmap_2x4)
terms2, expr_str2 = solver2.minimize(form='sop')
print(f"\n   BoolMin2D output: '{expr_str2}'")
print(f"   Terms: {terms2}")

print("\n" + "=" * 70)
print("TEST 3: Parsing BoolMin2D to PyEDA")
print("=" * 70)

def parse_kmap_to_pyeda_test(expr_str, pyeda_vars, form="sop"):
    """Test version of parse function with debug output"""
    print(f"\n   Input expression: '{expr_str}'")
    print(f"   Variables: {[str(v) for v in pyeda_vars]}")
    print(f"   Form: {form}")
    
    if not expr_str or expr_str.strip() == "":
        result = expr(0) if form == "sop" else expr(1)
        print(f"   Result (empty): {result}")
        return result

    # Build variable name mapping
    var_map = {str(v): v for v in pyeda_vars}
    print(f"   Variable map: {var_map}")

    if form.lower() == "sop":
        # SOP: product terms separated by '+'
        products = expr_str.split('+')
        print(f"   Products after split: {products}")
        
        or_terms = []
        for product in products:
            product = product.strip()
            if not product:
                continue
            print(f"   Processing product: '{product}'")
            
            # Extract literals (variables with optional prime)
            literals = re.findall(r"[A-Za-z_]\w*'?", product)
            print(f"      Literals found: {literals}")
            
            and_terms = []
            for lit in literals:
                if lit.endswith("'"):
                    var_name = lit[:-1]
                    print(f"      Complemented: {lit} -> ~{var_name}")
                    if var_name in var_map:
                        and_terms.append(~var_map[var_name])
                else:
                    print(f"      Normal: {lit}")
                    if lit in var_map:
                        and_terms.append(var_map[lit])
            
            print(f"      AND terms: {and_terms}")
            if len(and_terms) == 1:
                or_terms.append(and_terms[0])
            elif and_terms:
                or_terms.append(And(*and_terms))
        
        print(f"   OR terms: {or_terms}")
        result = Or(*or_terms) if or_terms else expr(0)
        print(f"   Final result: {result}")
        return result
    
    raise ValueError(f"Only SOP supported in test")

# Test parsing with 2-variable case
print("\n1. Parsing 2-variable BoolMin2D output")
pyeda_vars_2 = [exprvar(name) for name in ['a', 'b']]
# Note: BoolMin2D uses x1, x2 notation, we need to map to a, b
# We need to create a mapping: x1 -> a, x2 -> b

# First, let's see what variables BoolMin2D actually uses
print(f"\n   BoolMin2D expression: '{expr_str}'")

# Create variables matching BoolMin2D output
if 'x1' in expr_str or 'x2' in expr_str:
    print("   BoolMin2D uses x1, x2 notation")
    pyeda_vars_test = [exprvar(name) for name in ['x1', 'x2']]
else:
    print("   BoolMin2D uses different notation")
    pyeda_vars_test = pyeda_vars_2

parsed = parse_kmap_to_pyeda_test(expr_str, pyeda_vars_test, form='sop')

print("\n2. Testing equivalence")
# Create the expected PyEDA expression from the K-map
# K-map minterms: (0,1)=1, (1,0)=1, (1,1)=1 -> should be: a + b (or x1 + x2)
x1, x2 = pyeda_vars_test[0], pyeda_vars_test[1]
expected = Or(x1, x2)
print(f"   Expected (from truth table): {expected}")
print(f"   Parsed: {parsed}")
print(f"   Are they equivalent? {expected.equivalent(parsed)}")

print("\n" + "=" * 70)
print("TEST 4: Variable Naming Issue")
print("=" * 70)

print("\nBoolMin2D likely uses x1, x2, x3, x4 notation")
print("PyEDA benchmark uses a, b, c, d notation")
print("WE NEED TO MAP: x1->a, x2->b, x3->c, x4->d")
print("\nThis means we need to either:")
print("1. Replace x1 with a, x2 with b, etc. in the BoolMin2D string")
print("2. Create PyEDA variables named x1, x2, x3, x4 instead of a, b, c, d")

print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)
print("\nThe parse_kmap_to_pyeda function should:")
print("1. Accept the BoolMin2D expression string (uses x1, x2, x3, x4)")
print("2. Accept PyEDA variables (currently a, b, c, d)")
print("3. Do variable name substitution: replace 'x1' with 'a', etc.")
print("4. Then parse the modified string")
print("\nOR create PyEDA variables as x1, x2, x3, x4 from the start.")

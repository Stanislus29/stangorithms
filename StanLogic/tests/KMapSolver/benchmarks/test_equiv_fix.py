"""
Test equivalence check with variable mapping
"""
from pyeda.inter import *
from sympy import symbols, simplify, SOPform, POSform
from sympy.logic.boolalg import to_cnf, to_dnf

# Create PyEDA variables
a, b, c, d = map(exprvar, 'abcd')

# Create SymPy variables
x1, x2, x3, x4 = symbols('x1 x2 x3 x4')

# Test 1: Simple function - a AND b
print("Test 1: Simple function a AND b")
pyeda_expr = a & b
print(f"PyEDA expression: {pyeda_expr}")
print(f"PyEDA variables: {[str(v) for v in pyeda_expr.support]}")

# Parse PyEDA to string
pyeda_str = str(pyeda_expr)
print(f"PyEDA string: {pyeda_str}")

# Now we need to map a->x1, b->x2 when parsing
# Simple string replacement
mapped_str = pyeda_str.replace('a', 'x1').replace('b', 'x2').replace('c', 'x3').replace('d', 'x4')
print(f"Mapped string: {mapped_str}")

# Parse to SymPy
sympy_expr = eval(mapped_str, {'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'And': lambda *args: x1.func(*[a & b for a, b in zip(args, args)]), 'Or': lambda *args: x1.func(*[a | b for a, b in zip(args, args)])})
print(f"SymPy expression: {sympy_expr}")

# Better approach - direct construction
sympy_expr2 = x1 & x2
print(f"Direct SymPy: {sympy_expr2}")

print("\n" + "="*50 + "\n")

# Test 2: Let's see what parse_pyeda_expression should do
def parse_pyeda_expression(pyeda_expr, var_mapping=None):
    """
    Parse PyEDA expression to SymPy expression.
    var_mapping: dict mapping PyEDA var names to SymPy symbols
    """
    from sympy.logic.boolalg import And, Or, Not
    
    expr_str = str(pyeda_expr)
    
    # If constant
    if expr_str == '0':
        return False
    elif expr_str == '1':
        return True
    
    # Apply variable mapping if provided
    if var_mapping:
        for pyeda_var, sympy_var in var_mapping.items():
            # Replace whole words only
            import re
            expr_str = re.sub(r'\b' + pyeda_var + r'\b', f'_VAR_{pyeda_var}_', expr_str)
        
        # Now replace with actual SymPy variable names
        for pyeda_var, sympy_var in var_mapping.items():
            expr_str = expr_str.replace(f'_VAR_{pyeda_var}_', str(sympy_var))
    
    # Replace PyEDA operators with Python boolean ops
    expr_str = expr_str.replace('~', ' not ')
    expr_str = expr_str.replace('&', ' and ')
    expr_str = expr_str.replace('|', ' or ')
    
    # Build namespace for eval
    namespace = {'and': And, 'or': Or, 'not': Not, 'And': And, 'Or': Or, 'Not': Not}
    if var_mapping:
        namespace.update(var_mapping.values())
        # Also add by name
        for v in var_mapping.values():
            namespace[str(v)] = v
    
    try:
        result = eval(expr_str, namespace)
        return result
    except Exception as e:
        print(f"Error parsing: {expr_str}")
        print(f"Namespace keys: {namespace.keys()}")
        raise e

# Test the function
print("Test 2: Using parse_pyeda_expression")
mapping = {'a': x1, 'b': x2, 'c': x3, 'd': x4}
pyeda_func = a & b | c
print(f"PyEDA: {pyeda_func}")
sympy_func = parse_pyeda_expression(pyeda_func, mapping)
print(f"SymPy (parsed): {sympy_func}")
print(f"SymPy variables: {sympy_func.free_symbols}")

# Compare with direct construction
sympy_direct = (x1 & x2) | x3
print(f"SymPy (direct): {sympy_direct}")
print(f"Are they equivalent? {simplify(sympy_func ^ sympy_direct) == False}")

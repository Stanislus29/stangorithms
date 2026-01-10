"""
Simple test to verify 9-variable function minimization using 4D method.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'StanLogic', 'src'))

from stanlogic.BoolMinGeo import BoolMinGeo
from sympy import symbols, And as SymAnd, Or as SymOr, Not as SymNot, true, false
import re

def parse_boolmingeo_expression(expr_str, num_vars):
    """
    Parse BoolMinGeo expression string to SymPy for evaluation.
    
    Args:
        expr_str: Expression like "x1x2' + x3x4"
        num_vars: Number of variables
        
    Returns:
        SymPy Boolean expression
    """
    if not expr_str or expr_str.strip() in ["0", ""]:
        return false
    if expr_str.strip() == "1":
        return true
    
    var_symbols = symbols(f'x1:{num_vars+1}')
    if not isinstance(var_symbols, tuple):
        var_symbols = (var_symbols,)
    
    # Parse SOP format: x1x2' + x3x4
    or_terms = []
    for product in expr_str.split('+'):
        product = product.strip()
        if not product:
            continue
        
        # Extract literals (variables with optional prime)
        literals = re.findall(r"x\d+'?", product)
        and_terms = []
        for lit in literals:
            if lit.endswith("'"):
                var_idx = int(lit[1:-1]) - 1  # x1' -> index 0
                and_terms.append(SymNot(var_symbols[var_idx]))
            else:
                var_idx = int(lit[1:]) - 1  # x1 -> index 0
                and_terms.append(var_symbols[var_idx])
        
        if and_terms:
            or_terms.append(SymAnd(*and_terms) if len(and_terms) > 1 else and_terms[0])
    
    return SymOr(*or_terms) if len(or_terms) > 1 else (or_terms[0] if or_terms else false)

def verify_expression(expr_str, output_values, num_vars):
    """
    Verify that the expression matches the truth table.
    
    Args:
        expr_str: Boolean expression string
        output_values: List of expected output values
        num_vars: Number of variables
        
    Returns:
        Tuple of (passed, mismatches)
    """
    var_symbols = symbols(f'x1:{num_vars+1}')
    if not isinstance(var_symbols, tuple):
        var_symbols = (var_symbols,)
    
    expr_sympy = parse_boolmingeo_expression(expr_str, num_vars)
    
    mismatches = 0
    for i in range(2**num_vars):
        expected = output_values[i]
        
        # Create assignment (MSB-first for BoolMinGeo)
        assign = {}
        for j in range(num_vars):
            bit = (i >> (num_vars - 1 - j)) & 1
            assign[var_symbols[j]] = True if bit else False
        
        # Evaluate expression
        try:
            result = expr_sympy.subs(assign)
            actual = 1 if (result == true or result == True or result == 1) else 0
        except:
            actual = -1
        
        if actual != expected:
            mismatches += 1
    
    return mismatches == 0, mismatches

def test_9var_simple():
    """Test a simple 9-variable function: x1'x2' (should work with first 2 variables)"""
    print("="*70)
    print("TEST 1: 9-variable function x1'x2'")
    print("="*70)
    
    num_vars = 9
    output_values = [0] * 512
    
    # x1'x2' means first two bits are '00'
    for i in range(512):
        bits = format(i, '09b')
        if bits[0:2] == '00':
            output_values[i] = 1
    
    ones = sum(output_values)
    print(f"Variables: {num_vars}")
    print(f"Minterms: {ones}/{512}")
    
    # Minimize
    solver = BoolMinGeo(num_vars, output_values)
    terms, expr = solver.minimize_4d(form='sop')
    
    print(f"\nResult: {expr}")
    print(f"Terms: {len(terms)}")
    print(f"Expected: x1'x2'")
    
    # Verify
    passed, mismatches = verify_expression(expr, output_values, num_vars)
    
    if passed:
        print(f"\n✓ PASS - Expression matches truth table!")
    else:
        print(f"\n✗ FAIL - {mismatches} mismatches found!")
    
    return passed

def test_9var_or():
    """Test 9-variable function: x1 + x2"""
    print("\n" + "="*70)
    print("TEST 2: 9-variable function x1 + x2")
    print("="*70)
    
    num_vars = 9
    output_values = [0] * 512
    
    # x1 + x2 means bit[0]=='1' OR bit[1]=='1'
    for i in range(512):
        bits = format(i, '09b')
        if bits[0] == '1' or bits[1] == '1':
            output_values[i] = 1
    
    ones = sum(output_values)
    print(f"Variables: {num_vars}")
    print(f"Minterms: {ones}/{512}")
    
    # Minimize
    solver = BoolMinGeo(num_vars, output_values)
    terms, expr = solver.minimize_4d(form='sop')
    
    print(f"\nResult: {expr}")
    print(f"Terms: {len(terms)}")
    print(f"Expected: x1 + x2")
    
    # Verify
    passed, mismatches = verify_expression(expr, output_values, num_vars)
    
    if passed:
        print(f"\n✓ PASS - Expression matches truth table!")
    else:
        print(f"\n✗ FAIL - {mismatches} mismatches found!")
    
    return passed

def test_9var_complex():
    """Test 9-variable function: x1'x2' + x3'x4'"""
    print("\n" + "="*70)
    print("TEST 3: 9-variable function x1'x2' + x3'x4'")
    print("="*70)
    
    num_vars = 9
    output_values = [0] * 512
    
    for i in range(512):
        bits = format(i, '09b')
        # x1'x2' (bits 0,1 are 00) OR x3'x4' (bits 2,3 are 00)
        if bits[0:2] == '00' or bits[2:4] == '00':
            output_values[i] = 1
    
    ones = sum(output_values)
    print(f"Variables: {num_vars}")
    print(f"Minterms: {ones}/{512}")
    
    # Minimize
    solver = BoolMinGeo(num_vars, output_values)
    terms, expr = solver.minimize_4d(form='sop')
    
    print(f"\nResult: {expr}")
    print(f"Terms: {len(terms)}")
    print(f"Expected: x1'x2' + x3'x4'")
    
    # Verify
    passed, mismatches = verify_expression(expr, output_values, num_vars)
    
    if passed:
        print(f"\n✓ PASS - Expression matches truth table!")
    else:
        print(f"\n✗ FAIL - {mismatches} mismatches found!")
    
    return passed

if __name__ == "__main__":
    results = []
    
    results.append(("Test 1: x1'x2'", test_9var_simple()))
    results.append(("Test 2: x1 + x2", test_9var_or()))
    results.append(("Test 3: x1'x2' + x3'x4'", test_9var_complex()))
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed!")
    else:
        print(f"\n⚠ {total - passed} test(s) failed")

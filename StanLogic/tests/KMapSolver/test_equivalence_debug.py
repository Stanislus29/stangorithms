"""
Comprehensive equivalence test suite for BoolMinGeo.
Tests various edge cases to identify logical errors.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')))

from stanlogic.BoolMinGeo import BoolMinGeo
import random

def test_random_case():
    """Test a single 5-variable case to check equivalence."""
    
    print("="*70)
    print("TESTING SINGLE 5-VARIABLE CASE FOR EQUIVALENCE")
    print("="*70)
    
    # Create a simple 5-variable function
    # Set seed for reproducibility
    random.seed(42)
    
    # Generate random output values (50% ones, rest zeros)
    num_vars = 5
    num_minterms = 2 ** num_vars
    
    output_values = []
    for i in range(num_minterms):
        if random.random() < 0.5:
            output_values.append(1)
        else:
            output_values.append(0)
    
    print(f"\nGenerated {num_vars}-variable truth table")
    print(f"Total minterms: {num_minterms}")
    print(f"Ones: {sum(output_values)}")
    print(f"Zeros: {num_minterms - sum(output_values)}")
    
    # Show which minterms are 1
    ones_indices = [i for i, v in enumerate(output_values) if v == 1]
    print(f"\nMinterms with value 1: {ones_indices}")
    
    # Create solver
    print("\nCreating BoolMinGeo solver...")
    solver = BoolMinGeo(num_vars, output_values)
    
    # Minimize
    print("\nMinimizing SOP form...")
    try:
        terms, expression = solver.minimize_3d(form='sop')
        print(f"\nResult:")
        print(f"  Terms: {len(terms)}")
        print(f"  Expression: F = {expression}")
        print(f"\nTerms:")
        for i, term in enumerate(terms, 1):
            print(f"  {i}. {term}")
    except Exception as e:
        print(f"ERROR during minimization: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Verify equivalence by evaluating both
    print("\n" + "="*70)
    print("VERIFYING EQUIVALENCE")
    print("="*70)
    
    # Evaluate the minimized expression for all inputs
    print("\nEvaluating minimized expression on all minterms...")
    
    mismatches = []
    for minterm_idx in range(num_minterms):
        # Convert minterm index to binary values
        binary = format(minterm_idx, f'0{num_vars}b')
        var_values = [int(b) for b in binary]
        
        # Expected output
        expected = output_values[minterm_idx]
        
        # Evaluate expression
        actual = evaluate_expression(expression, var_values, form='sop')
        
        if actual != expected:
            mismatches.append({
                'minterm': minterm_idx,
                'binary': binary,
                'expected': expected,
                'actual': actual
            })
    
    if mismatches:
        print(f"\n*** EQUIVALENCE FAILED ***")
        print(f"Found {len(mismatches)} mismatches:")
        for m in mismatches[:10]:  # Show first 10
            print(f"  Minterm {m['minterm']:2d} ({m['binary']}): expected {m['expected']}, got {m['actual']}")
        if len(mismatches) > 10:
            print(f"  ... and {len(mismatches)-10} more")
        return False
    else:
        print(f"\n*** EQUIVALENCE VERIFIED ***")
        print(f"All {num_minterms} minterms match!")
        return True


def evaluate_expression(expr_str, var_values, form='sop'):
    """
    Evaluate a Boolean expression for given variable values.
    
    Args:
        expr_str: Expression string (e.g., "x1x2'x4 + x3x5")
        var_values: List of values for [x1, x2, x3, x4, x5, ...]
        form: 'sop' or 'pos'
    
    Returns:
        int: 0 or 1
    """
    if not expr_str or expr_str.strip() == '0':
        return 0
    if expr_str.strip() == '1':
        return 1
    
    if form == 'sop':
        # Split by '+' to get product terms
        terms = expr_str.split('+')
        result = 0
        
        for term in terms:
            term = term.strip()
            if not term:
                continue
            
            # Evaluate product term (all ANDed literals must be true)
            term_value = 1
            i = 0
            while i < len(term):
                if term[i] == 'x':
                    # Found a variable
                    j = i + 1
                    while j < len(term) and term[j].isdigit():
                        j += 1
                    
                    var_num = int(term[i+1:j])
                    var_idx = var_num - 1
                    
                    # Check for complement
                    if j < len(term) and term[j] == "'":
                        # Complemented variable
                        term_value &= (1 - var_values[var_idx])
                        i = j + 1
                    else:
                        # Non-complemented variable
                        term_value &= var_values[var_idx]
                        i = j
                else:
                    i += 1
            
            result |= term_value
        
        return result
    
    elif form == 'pos':
        # Split by '*' to get sum terms
        clauses = expr_str.split('*')
        result = 1
        
        for clause in clauses:
            clause = clause.strip().strip('()')
            if not clause:
                continue
            
            # Evaluate sum term (any ORed literal can be true)
            clause_value = 0
            
            literals = clause.split('+')
            for lit in literals:
                lit = lit.strip()
                if not lit:
                    continue
                
                # Parse variable
                if lit.startswith('x'):
                    var_str = lit[1:]
                    if var_str.endswith("'"):
                        var_num = int(var_str[:-1])
                        var_idx = var_num - 1
                        clause_value |= (1 - var_values[var_idx])
                    else:
                        var_num = int(var_str)
                        var_idx = var_num - 1
                        clause_value |= var_values[var_idx]
            
            result &= clause_value
        
        return result
    
    return 0


if __name__ == "__main__":
    print("="*80)
    print("COMPREHENSIVE EQUIVALENCE TEST SUITE FOR BOOLMINGEO")
    print("="*80)
    
    test_results = []
    
    # Test 1: Random case
    print("\n" + "="*80)
    print("TEST 1: Random 50% density case")
    print("="*80)
    test_results.append(("Random 50%", test_random_case()))
    
    # Test 2: All zeros
    print("\n" + "="*80)
    print("TEST 2: All zeros (constant 0)")
    print("="*80)
    test_results.append(("All zeros", test_all_zeros()))
    
    # Test 3: All ones
    print("\n" + "="*80)
    print("TEST 3: All ones (constant 1)")
    print("="*80)
    test_results.append(("All ones", test_all_ones()))
    
    # Test 4: Sparse
    print("\n" + "="*80)
    print("TEST 4: Sparse (10% ones)")
    print("="*80)
    test_results.append(("Sparse 10%", test_sparse()))
    
    # Test 5: Dense
    print("\n" + "="*80)
    print("TEST 5: Dense (90% ones)")
    print("="*80)
    test_results.append(("Dense 90%", test_dense()))
    
    # Test 6: Checkerboard
    print("\n" + "="*80)
    print("TEST 6: Checkerboard pattern")
    print("="*80)
    test_results.append(("Checkerboard", test_checkerboard()))
    
    # Test 7: Single one
    print("\n" + "="*80)
    print("TEST 7: Single one (minterm 0)")
    print("="*80)
    test_results.append(("Single one", test_single_one()))
    
    # Test 8: All but one
    print("\n" + "="*80)
    print("TEST 8: All ones except one")
    print("="*80)
    test_results.append(("All but one", test_all_but_one()))
    
    # Test 9: 6-variable case
    print("\n" + "="*80)
    print("TEST 9: 6-variable random case")
    print("="*80)
    test_results.append(("6-var random", test_6_variable()))
    
    # Test 10: 7-variable case
    print("\n" + "="*80)
    print("TEST 10: 7-variable random case")
    print("="*80)
    test_results.append(("7-var random", test_7_variable()))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    passed = sum(1 for _, result in test_results if result)
    failed = len(test_results) - passed
    
    for test_name, result in test_results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"  {symbol} {test_name:20s}: {status}")
    
    print("\n" + "="*80)
    print(f"Results: {passed}/{len(test_results)} tests passed")
    if failed > 0:
        print(f"⚠️  {failed} test(s) failed - LOGICAL EQUIVALENCE ISSUES DETECTED")
    else:
        print("✓ All tests passed - No logical equivalence issues detected")
    print("="*80)

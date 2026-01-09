"""
Comprehensive equivalence test suite for BoolMinGeo.
Tests various edge cases to identify logical errors.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')))

from stanlogic.BoolMinGeo import BoolMinGeo
import random


def run_equivalence_test(num_vars, output_values, description=""):
    """
    Run equivalence test for given truth table.
    
    Args:
        num_vars: Number of variables
        output_values: List of output values for each minterm
        description: Test description
        
    Returns:
        bool: True if equivalent, False otherwise
    """
    num_minterms = 2 ** num_vars
    ones_count = sum(output_values)
    
    print(f"Testing {num_vars}-variable: {description}")
    print(f"Ones: {ones_count}/{num_minterms} ({100*ones_count/num_minterms:.1f}%)")
    
    try:
        # Create solver
        solver = BoolMinGeo(num_vars, output_values)
        
        # Minimize - suppress verbose output
        import io
        import sys as sys_module
        old_stdout = sys_module.stdout
        sys_module.stdout = io.StringIO()
        
        terms, expression = solver.minimize_3d(form='sop')
        
        # Restore stdout
        sys_module.stdout = old_stdout
        
        print(f"Result: {len(terms)} terms")
        if len(expression) <= 80:
            print(f"Expression: F = {expression}")
        else:
            print(f"Expression: F = {expression[:77]}...")
        
    except Exception as e:
        print(f"✗ ERROR during minimization: {e}")
        return False
    
    # Verify equivalence
    print("Verifying...", end=" ")
    
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
        print(f"✗ FAIL ({len(mismatches)} mismatches)")
        for m in mismatches[:3]:
            print(f"    Minterm {m['minterm']:3d} ({m['binary']}): expected {m['expected']}, got {m['actual']}")
        if len(mismatches) > 3:
            print(f"    ... and {len(mismatches)-3} more")
        return False
    else:
        print(f"✓ PASS")
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


# Test Functions
def test_random_50():
    """Random 5-var with 50% ones."""
    random.seed(42)
    num_vars = 5
    output_values = [1 if random.random() < 0.5 else 0 for _ in range(2**num_vars)]
    return run_equivalence_test(num_vars, output_values, "Random 50% density")


def test_all_zeros():
    """All zeros (constant 0)."""
    num_vars = 5
    output_values = [0] * (2 ** num_vars)
    return run_equivalence_test(num_vars, output_values, "All zeros (constant 0)")


def test_all_ones():
    """All ones (constant 1)."""
    num_vars = 5
    output_values = [1] * (2 ** num_vars)
    return run_equivalence_test(num_vars, output_values, "All ones (constant 1)")


def test_sparse_10():
    """Sparse with 10% ones."""
    random.seed(123)
    num_vars = 5
    output_values = [1 if random.random() < 0.1 else 0 for _ in range(2**num_vars)]
    return run_equivalence_test(num_vars, output_values, "Sparse 10% density")


def test_dense_90():
    """Dense with 90% ones."""
    random.seed(456)
    num_vars = 5
    output_values = [1 if random.random() < 0.9 else 0 for _ in range(2**num_vars)]
    return run_equivalence_test(num_vars, output_values, "Dense 90% density")


def test_checkerboard():
    """Checkerboard pattern."""
    num_vars = 5
    output_values = [i % 2 for i in range(2**num_vars)]
    return run_equivalence_test(num_vars, output_values, "Checkerboard pattern")


def test_single_one():
    """Single minterm = 1."""
    num_vars = 5
    output_values = [0] * (2 ** num_vars)
    output_values[0] = 1
    return run_equivalence_test(num_vars, output_values, "Single one (minterm 0)")


def test_all_but_one():
    """All ones except one."""
    num_vars = 5
    output_values = [1] * (2 ** num_vars)
    output_values[0] = 0
    return run_equivalence_test(num_vars, output_values, "All except one")


def test_first_half():
    """First half ones, second half zeros."""
    num_vars = 5
    mid = 2 ** (num_vars - 1)
    output_values = [1] * mid + [0] * mid
    return run_equivalence_test(num_vars, output_values, "First half ones")


def test_6var_random():
    """6-variable random."""
    random.seed(789)
    num_vars = 6
    output_values = [1 if random.random() < 0.5 else 0 for _ in range(2**num_vars)]
    return run_equivalence_test(num_vars, output_values, "6-var random 50%")


def test_7var_random():
    """7-variable random."""
    random.seed(101112)
    num_vars = 7
    output_values = [1 if random.random() < 0.5 else 0 for _ in range(2**num_vars)]
    return run_equivalence_test(num_vars, output_values, "7-var random 50%")


def test_8var_sparse():
    """8-variable sparse."""
    random.seed(131415)
    num_vars = 8
    output_values = [1 if random.random() < 0.2 else 0 for _ in range(2**num_vars)]
    return run_equivalence_test(num_vars, output_values, "8-var sparse 20%")


def test_powers_of_2():
    """Only power-of-2 minterms."""
    num_vars = 5
    output_values = [0] * (2 ** num_vars)
    for i in range(num_vars + 1):
        if 2**i < len(output_values):
            output_values[2**i] = 1
    return run_equivalence_test(num_vars, output_values, "Power-of-2 minterms")


def test_corners_5var():
    """Corner minterms (all 0s and all 1s)."""
    num_vars = 5
    output_values = [0] * (2 ** num_vars)
    output_values[0] = 1  # 00000
    output_values[-1] = 1  # 11111
    return run_equivalence_test(num_vars, output_values, "Corner minterms")


def test_alternating_vars():
    """Alternating variable pattern."""
    num_vars = 5
    output_values = []
    for i in range(2**num_vars):
        binary = format(i, f'0{num_vars}b')
        # Count ones in binary representation
        ones = sum(int(b) for b in binary)
        output_values.append(1 if ones % 2 == 0 else 0)
    return run_equivalence_test(num_vars, output_values, "Even parity function")


if __name__ == "__main__":
    print("="*80)
    print("COMPREHENSIVE EQUIVALENCE TEST SUITE FOR BOOLMINGEO")
    print("="*80)
    print("Testing various edge cases to identify logical errors\n")
    
    tests = [
        ("Random 50%", test_random_50),
        ("All zeros", test_all_zeros),
        ("All ones", test_all_ones),
        ("Sparse 10%", test_sparse_10),
        ("Dense 90%", test_dense_90),
        ("Checkerboard", test_checkerboard),
        ("Single one", test_single_one),
        ("All but one", test_all_but_one),
        ("First half", test_first_half),
        ("Powers of 2", test_powers_of_2),
        ("Corners", test_corners_5var),
        ("Even parity", test_alternating_vars),
        ("6-var random", test_6var_random),
        ("7-var random", test_7var_random),
        ("8-var sparse", test_8var_sparse),
    ]
    
    results = []
    
    for i, (name, test_func) in enumerate(tests, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}/{len(tests)}: {name}")
        print(f"{'='*80}")
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ Exception: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    failed = len(results) - passed
    
    for name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"  {symbol} {name:20s}: {status}")
    
    print("\n" + "="*80)
    print(f"Results: {passed}/{len(results)} tests passed ({100*passed/len(results):.1f}%)")
    
    if failed > 0:
        print(f"⚠️  {failed} test(s) failed - LOGICAL EQUIVALENCE ISSUES DETECTED")
        print("="*80)
        sys.exit(1)
    else:
        print("✓ All tests passed - No logical equivalence issues detected")
        print("="*80)
        sys.exit(0)

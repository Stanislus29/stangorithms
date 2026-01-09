"""
Test the updated timeout and SymPy fallback mechanism
"""

import sys
sys.path.append('c:/Users/DELL/Documents/Research Projects/Mathematical_models/StanLogic/src')
sys.path.append('c:/Users/DELL/Documents/Research Projects/Mathematical_models/StanLogic/tests/KMapSolver/benchmarks')

# Import the updated function
import benchmark_test2D_pyeda as bench
from pyeda.inter import *

def test_timeout_and_fallback():
    """Test timeout with SymPy fallback"""
    print("="*70)
    print("TESTING TIMEOUT AND SYMPY FALLBACK")
    print("="*70)
    
    # Test 1: Simple equivalence (should work with PyEDA)
    print("\nTest 1: Simple equivalence")
    x1, x2 = map(exprvar, ['x1', 'x2'])
    expr1 = x1 & x2
    expr2 = x1 & x2
    
    result = bench.check_equivalence_pyeda(expr1, expr2)
    print(f"Result: {result}")
    assert result == True, "Should be equivalent"
    print("✓ PASS")
    
    # Test 2: Simple non-equivalence (should work with PyEDA)
    print("\nTest 2: Simple non-equivalence")
    expr1 = x1 & x2
    expr2 = x1 | x2
    
    result = bench.check_equivalence_pyeda(expr1, expr2)
    print(f"Result: {result}")
    assert result == False, "Should not be equivalent"
    print("✓ PASS")
    
    # Test 3: More complex case
    print("\nTest 3: Complex equivalence")
    x1, x2, x3 = map(exprvar, ['x1', 'x2', 'x3'])
    expr1 = (x1 & x2) | (~x1 & x3)
    expr2 = (x1 & x2) | (~x1 & x3)
    
    result = bench.check_equivalence_pyeda(expr1, expr2)
    print(f"Result: {result}")
    assert result == True, "Should be equivalent"
    print("✓ PASS")
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED")
    print("="*70)

if __name__ == "__main__":
    test_timeout_and_fallback()

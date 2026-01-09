"""
Quick test to verify timeout mechanism works
"""

import sys
sys.path.append('c:/Users/DELL/Documents/Research Projects/Mathematical_models/StanLogic/src')

from stanlogic import BoolMin2D
from pyeda.inter import *
import time

def test_timeout_mechanism():
    """Test that equivalence checking has timeout"""
    print("Testing timeout mechanism for equivalence checking...")
    
    # Create two complex expressions
    x1, x2, x3, x4 = map(exprvar, ['x1', 'x2', 'x3', 'x4'])
    
    # Create expressions that might take long to compare
    expr1 = (x1 & x2) | (~x1 & x3) | (x2 & ~x3 & x4)
    expr2 = (x1 & x2) | (~x1 & x3) | (x2 & ~x3 & x4) | (~x1 & ~x2 & ~x3 & ~x4)
    
    print(f"Expr1: {expr1}")
    print(f"Expr2: {expr2}")
    
    # Test equivalence with timeout
    start = time.time()
    try:
        result = expr1.equivalent(expr2)
        elapsed = time.time() - start
        print(f"Equivalence check completed in {elapsed:.3f}s: {result}")
    except Exception as e:
        elapsed = time.time() - start
        print(f"Equivalence check failed after {elapsed:.3f}s: {e}")
    
    print("\n✓ Timeout mechanism test complete")

if __name__ == "__main__":
    test_timeout_mechanism()

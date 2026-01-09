"""
Quick test to isolate the crash at test 22 in benchmark_test2D_pyeda.py
"""

import sys
sys.path.append('c:/Users/DELL/Documents/Research Projects/Mathematical_models/StanLogic/src')

from stanlogic import BoolMin2D
from sympy import symbols
import random

def generate_random_kmap(num_vars, p_one=0.5, p_dc=0.1):
    """Generate a random K-map"""
    if num_vars == 2:
        rows, cols = 2, 2
    elif num_vars == 3:
        rows, cols = 2, 4
    elif num_vars == 4:
        rows, cols = 4, 4
    else:
        raise ValueError("Only 2-4 variables supported")
    
    kmap = []
    for r in range(rows):
        row = []
        for c in range(cols):
            rand = random.random()
            if rand < p_dc:
                row.append('d')
            elif rand < p_dc + p_one:
                row.append(1)
            else:
                row.append(0)
        kmap.append(row)
    
    return kmap

def test_single_case(test_num, num_vars=3, form='sop'):
    """Test a single case"""
    print(f"\nTest {test_num}: {num_vars}-var {form.upper()}", end=" ... ")
    
    try:
        # Generate K-map
        random.seed(test_num)
        kmap = generate_random_kmap(num_vars, p_one=0.6, p_dc=0.15)
        
        # Print K-map for debugging
        print(f"\n  K-map:")
        for row in kmap:
            print(f"    {row}")
        
        # Minimize
        solver = BoolMin2D(kmap)
        terms, expr = solver.minimize(form=form)
        
        print(f"\n  Result: {expr}")
        print(f"  Terms: {len(terms)}")
        print("  ✓ SUCCESS")
        return True
        
    except Exception as e:
        print(f"\n  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run tests around test 22"""
    print("="*70)
    print("ISOLATING TEST 22 CRASH")
    print("="*70)
    
    # Test cases around 22
    test_cases = [
        (20, 3, 'sop'),
        (21, 3, 'sop'),
        (22, 3, 'sop'),  # The problematic one
        (23, 3, 'sop'),
        (24, 3, 'sop'),
    ]
    
    passed = 0
    failed = 0
    
    for test_num, num_vars, form in test_cases:
        if test_single_case(test_num, num_vars, form):
            passed += 1
        else:
            failed += 1
            print(f"\n⚠️  Test {test_num} failed - stopping here")
            break
    
    print("\n" + "="*70)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*70)

if __name__ == "__main__":
    main()

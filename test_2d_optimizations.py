"""
Test script to verify BoolMin2D optimizations
Tests the enhanced weighted greedy set cover and redundancy removal
"""

import sys
sys.path.insert(0, r'c:\Users\DELL\Documents\Research Projects\Mathematical_models\StanLogic\src')

from stanlogic.BoolMin2D import BoolMin2D
import random

def test_basic_minimization():
    """Test basic minimization with known examples"""
    print("=" * 80)
    print("TEST 1: Basic 3-variable K-map")
    print("=" * 80)
    
    # Example: F(A,B,C) - 3 variables require 2x4 K-map
    # Gray code column ordering: 00, 01, 11, 10
    kmap = [
        [1, 1, 0, 1],  # Row 0: A=0, BC=00,01,11,10
        [1, 0, 1, 0]   # Row 1: A=1, BC=00,01,11,10
    ]
    
    solver = BoolMin2D(kmap)
    terms, expr = solver.minimize(form='sop')
    
    print(f"K-map: {kmap}")
    print(f"Minimized SOP: {expr}")
    print(f"Number of terms: {len(terms)}")
    print(f"Terms: {terms}")
    print()
    
    return len(terms), expr

def test_with_dont_cares():
    """Test minimization with don't-cares"""
    print("=" * 80)
    print("TEST 2: K-map with don't-cares")
    print("=" * 80)
    
    # 3-variable K-map with don't-cares (2x4 format)
    kmap = [
        [1, 'd', 1, 0],  # Row 0
        [1, 0, 1, 'd']   # Row 1
    ]
    
    solver = BoolMin2D(kmap)
    terms, expr = solver.minimize(form='sop')
    
    print(f"K-map: {kmap}")
    print(f"Minimized SOP: {expr}")
    print(f"Number of terms: {len(terms)}")
    print(f"Terms: {terms}")
    print()
    
    return len(terms), expr

def test_4_variable():
    """Test 4-variable minimization"""
    print("=" * 80)
    print("TEST 3: 4-variable K-map")
    print("=" * 80)
    
    # 4x4 K-map with some pattern
    kmap = [
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 1]
    ]
    
    solver = BoolMin2D(kmap)
    terms, expr = solver.minimize(form='sop')
    
    print(f"Minimized SOP: {expr}")
    print(f"Number of terms: {len(terms)}")
    print(f"Terms: {terms}")
    print()
    
    return len(terms), expr

def test_random_functions():
    """Test with random truth tables"""
    print("=" * 80)
    print("TEST 4: Random 3-variable functions")
    print("=" * 80)
    
    results = []
    for i in range(5):
        # Generate random 3-variable truth table (2x4 format)
        kmap = []
        for r in range(2):
            row = []
            for c in range(4):
                row.append(random.choice([0, 1]))
            kmap.append(row)
        
        solver = BoolMin2D(kmap)
        terms, expr = solver.minimize(form='sop')
        
        # Count total literals
        literal_count = sum(len([c for c in term if c in "x'"]) for term in terms)
        
        results.append({
            'test': i+1,
            'terms': len(terms),
            'literals': literal_count,
            'expr': expr
        })
        
        print(f"Test {i+1}:")
        print(f"  K-map: {kmap}")
        print(f"  Expression: {expr}")
        print(f"  Terms: {len(terms)}, Literals: {literal_count}")
    
    print()
    avg_terms = sum(r['terms'] for r in results) / len(results)
    avg_literals = sum(r['literals'] for r in results) / len(results)
    print(f"Average terms: {avg_terms:.2f}")
    print(f"Average literals: {avg_literals:.2f}")
    print()
    
    return results

def test_optimization_effectiveness():
    """Test that optimizations actually improve results"""
    print("=" * 80)
    print("TEST 5: Optimization Effectiveness Check")
    print("=" * 80)
    
    # Create a K-map that benefits from weighted scoring
    # This example should favor general (low literal) terms
    kmap = [
        [1, 1, 1, 1],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [1, 1, 1, 1]
    ]
    
    solver = BoolMin2D(kmap)
    terms, expr = solver.minimize(form='sop')
    
    print(f"K-map with corner pattern:")
    for row in kmap:
        print(f"  {row}")
    print(f"\nMinimized SOP: {expr}")
    print(f"Number of terms: {len(terms)}")
    print(f"Terms: {terms}")
    
    # Count literals
    literal_count = 0
    for term in terms:
        # Count variables (x followed by digit)
        literal_count += len([c for c in term if c in "x'"])
    
    print(f"Total literals: {literal_count}")
    print()
    
    return len(terms), literal_count

def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("BOOLMIN2D OPTIMIZATION TESTING")
    print("Testing: Weighted Greedy Set Cover, Enhanced Redundancy Removal")
    print("=" * 80 + "\n")
    
    test_basic_minimization()
    test_with_dont_cares()
    test_4_variable()
    test_random_functions()
    test_optimization_effectiveness()
    
    print("=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    main()

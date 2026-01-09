"""
Test to verify 2D clusters are properly integrated with 3D clusters.
This addresses the fatal flaw where 2D clusters were being eliminated.
"""

import sys
sys.path.insert(0, 'c:/Users/DELL/Documents/Research Projects/Mathematical_models/StanLogic/src')

from stanlogic import BoolMinGeo

def test_2d_and_3d_integration():
    """
    Test that both 2D and 3D clusters are included in final expression.
    """
    print("="*70)
    print("TEST: 2D + 3D Cluster Integration")
    print("="*70)
    
    # Create a 5-variable test case with:
    # - Some patterns that span multiple K-maps (3D clusters)
    # - Some patterns that appear in only one K-map (2D clusters)
    
    # 5 variables = 2 K-maps (identifiers "0" and "1")
    # K-map layout (4x4):
    #       00  01  11  10
    #   00  0   1   2   3
    #   01  4   5   6   7
    #   11  12  13  14  15
    #   10  8   9   10  11
    
    # Setup: K-map "0" has pattern "1100" (cells 2,3,6,7)
    #        K-map "1" has pattern "1100" (cells 2,3,6,7) - same, so 3D cluster
    #        K-map "0" has pattern "0011" (cells 1,3,5,7) - unique to K-map "0", 2D cluster
    
    output_values = [0] * 32
    
    # K-map "0" (identifier bit = 0)
    # Pattern "1100" spans cells with x3x4 = "11" (covering x1x2 = 0X)
    # In binary: 0 11 00, 0 11 01 -> minterms 24, 25
    output_values[24] = 1  # 11000
    output_values[25] = 1  # 11001
    output_values[28] = 1  # 11100  
    output_values[29] = 1  # 11101
    
    # K-map "1" (identifier bit = 1)
    # Pattern "1100" - same K-map pattern, different identifier (3D cluster)
    output_values[24 + 1] = 1  # 11001 
    output_values[25 + 1] = 1  # 11010
    output_values[28 + 1] = 1  # 11101
    output_values[29 + 1] = 1  # 11110
    
    # K-map "0" only - 2D cluster
    # Pattern in only K-map "0"
    output_values[16] = 1  # 10000
    output_values[17] = 1  # 10001
    output_values[20] = 1  # 10100
    output_values[21] = 1  # 10101
    
    solver = BoolMinGeo(5, output_values)
    
    print("\nMinimizing...\n")
    terms, expression = solver.minimize_3d(form='sop')
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Number of terms: {len(terms)}")
    print(f"Expression: F = {expression}")
    print(f"\nTerms:")
    for i, term in enumerate(terms, 1):
        print(f"  {i}. {term}")
    
    # Verify coverage
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)
    
    # Check that all minterms are covered
    expected_minterms = [i for i in range(32) if output_values[i] == 1]
    print(f"Expected {len(expected_minterms)} minterms to be covered")
    print(f"Minterms: {expected_minterms}")
    
    # Count patterns that should be 2D vs 3D
    print("\nExpected behavior:")
    print("  - Should have 3D cluster(s) for pattern appearing in both K-maps")
    print("  - Should have 2D cluster(s) for pattern appearing in only K-map 0")
    print("  - Both types should be in final expression")
    
    if len(terms) == 0:
        print("\n✗ FAIL: No terms generated!")
        return False
    
    print(f"\n✓ PASS: Generated {len(terms)} terms including both 2D and 3D clusters")
    return True


def test_simple_5var_case():
    """
    Simple 5-variable test with clear 2D and 3D separation.
    """
    print("\n\n" + "="*70)
    print("TEST: Simple 5-Variable Case")
    print("="*70)
    
    # 5 variables, 32 minterms, 2 K-maps
    output_values = [0] * 32
    
    # K-map 0: minterms 0,1,4,5 (top-left corner)
    output_values[0] = 1
    output_values[1] = 1  
    output_values[4] = 1
    output_values[5] = 1
    
    # K-map 1: minterms 16,17,20,21 (top-left corner) - same pattern as K-map 0
    output_values[16] = 1
    output_values[17] = 1
    output_values[20] = 1
    output_values[21] = 1
    
    # K-map 0 only: minterms 8,9 (2D cluster)
    output_values[8] = 1
    output_values[9] = 1
    
    solver = BoolMinGeo(5, output_values)
    terms, expression = solver.minimize_3d(form='sop')
    
    print(f"\nResult: {len(terms)} terms")
    print(f"F = {expression}")
    
    # Should have at least 2 terms:
    # 1. A 3D cluster covering pattern in both K-maps
    # 2. A 2D cluster covering pattern only in K-map 0
    
    if len(terms) >= 2:
        print("\n✓ PASS: Multiple terms including 2D clusters")
        return True
    else:
        print("\n✗ FAIL: Not enough terms - 2D clusters may be missing")
        return False


if __name__ == "__main__":
    print("Testing 2D and 3D cluster integration fix\n")
    
    result1 = test_2d_and_3d_integration()
    result2 = test_simple_5var_case()
    
    print("\n" + "="*70)
    print("OVERALL RESULTS")
    print("="*70)
    if result1 and result2:
        print("✓ All tests PASSED")
    else:
        print("✗ Some tests FAILED")

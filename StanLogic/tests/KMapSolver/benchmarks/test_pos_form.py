"""
Test POS form parsing
"""

from pyeda.inter import *
from stanlogic import BoolMin2D
from sympy import symbols
from benchmark_test2D_pyeda import benchmark_case

print("="*70)
print("POS FORM PARSING TEST")
print("="*70)

var_names = symbols('x1 x2', boolean=True)

# Test cases with POS form
test_cases = [
    ([[0, 1], [1, 1]], "2-var: zero in position (0,0)"),
    ([[1, 0], [1, 1]], "2-var: zero in position (0,1)"),
    ([[1, 1], [0, 1]], "2-var: zero in position (1,0)"),
    ([[1, 1], [1, 0]], "2-var: zero in position (1,1)"),
    ([[1, 0], [0, 1]], "2-var: XOR (two zeros)"),
    ([[0, 0], [0, 0]], "2-var: all zeros (constant 0)"),
    ([[1, 1], [1, 1]], "2-var: all ones (constant 1)"),
]

results = []
for idx, (kmap, name) in enumerate(test_cases, 1):
    print(f"\nTest {idx}: {name}")
    print(f"  K-map: {kmap}")
    
    # Get BoolMin2D POS result
    solver = BoolMin2D(kmap)
    terms, expr_str = solver.minimize(form='pos')
    print(f"  BoolMin2D POS: '{expr_str}'")
    print(f"  Terms: {terms}")
    
    try:
        result = benchmark_case(kmap, var_names, 'pos', idx, 'test')
        equiv = result['equiv']
        
        if equiv is True:
            print(f"  Result: PASS")
            results.append(True)
        elif equiv is False:
            print(f"  Result: FAIL")
            print(f"    PyEDA: {result.get('pyeda_literals')} literals")
            print(f"    KMap: {result.get('kmap_literals')} literals")
            results.append(False)
        else:
            print(f"  Result: N/A (error)")
            results.append(None)
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        results.append(False)

print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")
passed = sum(1 for r in results if r is True)
total = len(results)
print(f"Passed: {passed}/{total}")

if passed == total:
    print("SUCCESS: All POS form tests passed!")
else:
    print(f"FAILED: {total - passed} tests failed")

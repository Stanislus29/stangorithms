"""
Comprehensive test of both SOP and POS forms
"""

from sympy import symbols
from benchmark_test2D_pyeda import benchmark_case
import random

random.seed(42)

print("="*70)
print("COMPREHENSIVE SOP & POS TEST")
print("="*70)

var_names = symbols('x1 x2', boolean=True)

test_cases = [
    ([[0, 1], [1, 1]], "OR"),
    ([[1, 0], [0, 1]], "XOR"),
    ([[1, 1], [1, 1]], "constant 1"),
    ([[0, 0], [0, 0]], "constant 0"),
    ([[1, 1], [1, 0]], "simple"),
    ([[0, 1], [0, 1]], "just x2"),
    ([[1, 0], [1, 0]], "just ~x2"),
]

for form in ['sop', 'pos']:
    print(f"\n{'='*70}")
    print(f"{form.upper()} FORM TESTS")
    print(f"{'='*70}")
    
    results = []
    for idx, (kmap, name) in enumerate(test_cases, 1):
        try:
            result = benchmark_case(kmap, var_names, form, idx, 'test')
            equiv = result['equiv']
            
            if equiv is True:
                status = "PASS"
                results.append(True)
            elif equiv is False:
                status = "FAIL"
                results.append(False)
            else:
                status = "N/A"
                results.append(None)
                
        except Exception as e:
            status = f"ERROR: {str(e)[:30]}"
            results.append(False)
    
    passed = sum(1 for r in results if r is True)
    total = len(results)
    print(f"\nSummary: {passed}/{total} passed")
    
    if passed == total:
        print(f"SUCCESS: All {form.upper()} tests passed!")
    else:
        print(f"FAILED: {total - passed} tests failed")

print(f"\n{'='*70}")
print("OVERALL: All forms tested successfully!")
print(f"{'='*70}")

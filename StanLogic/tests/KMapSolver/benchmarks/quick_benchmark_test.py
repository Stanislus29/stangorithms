"""
Quick benchmark test with a few cases
"""

import sys
import random
random.seed(42)

# Import benchmark
from benchmark_test2D_pyeda import benchmark_case, random_kmap
from sympy import symbols

print("Running quick benchmark test...\n")

# Test with 2 variables
var_names = symbols('x1 x2', boolean=True)

test_cases = [
    ([[0, 1], [1, 1]], "2-var OR"),
    ([[1, 0], [0, 1]], "2-var XOR"),
    ([[1, 1], [1, 1]], "2-var constant 1"),
    ([[0, 0], [0, 0]], "2-var constant 0"),
    ([[1, 1], [1, 0]], "2-var simple"),
]

results = []
for idx, (kmap, name) in enumerate(test_cases, 1):
    print(f"Test {idx}: {name}")
    try:
        result = benchmark_case(kmap, var_names, 'sop', idx, 'test')
        equiv = result['equiv']
        if equiv is True:
            print(f"  Result: PASS (equivalent)\n")
            results.append(True)
        elif equiv is False:
            print(f"  Result: FAIL (not equivalent)")
            print(f"    PyEDA literals: {result['pyeda_literals']}")
            print(f"    KMap literals: {result['kmap_literals']}\n")
            results.append(False)
        else:
            print(f"  Result: N/A (algorithm error)\n")
            results.append(None)
    except Exception as e:
        print(f"  ERROR: {e}\n")
        results.append(False)

print(f"\n{'='*60}")
print(f"Summary: {sum(1 for r in results if r is True)}/{len(results)} passed")
if any(r is False for r in results):
    print("FAILURES DETECTED")
else:
    print("ALL TESTS PASSED!")

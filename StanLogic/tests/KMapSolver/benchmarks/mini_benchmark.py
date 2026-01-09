"""
Mini benchmark run to verify all fixes work in full pipeline
"""

import os
import sys
import random
import datetime

# Reduce test count for quick validation
TESTS_PER_CONFIG = 10

# Override constants
import benchmark_test2D_pyeda as bench
bench.TESTS_PER_CONFIG = TESTS_PER_CONFIG

# Set seed
random.seed(42)

print("="*70)
print("MINI BENCHMARK - Validation Run")
print("="*70)
print(f"Tests per configuration: {TESTS_PER_CONFIG}")
print(f"Date: {datetime.datetime.now()}")
print("="*70)

# Generate small test suite for 2 variables only
from sympy import symbols
from benchmark_test2D_pyeda import generate_test_suite, benchmark_case

var_names = symbols('x1 x2', boolean=True)
num_vars = 2

print(f"\nGenerating test suite for {num_vars} variables...")
test_suite = generate_test_suite(num_vars, tests_per_distribution=TESTS_PER_CONFIG // 5)
print(f"Generated {len(test_suite)} test cases")

# Run benchmarks for SOP form only
form_type = 'sop'
print(f"\n{'='*70}")
print(f"Running {num_vars}-variable {form_type.upper()} tests")
print(f"{'='*70}")

results = []
equiv_pass = 0
equiv_fail = 0
equiv_na = 0

for idx, (kmap, distribution) in enumerate(test_suite[:TESTS_PER_CONFIG], 1):
    try:
        result = benchmark_case(kmap, var_names, form_type, idx, distribution)
        results.append(result)
        
        if result['equiv'] is True:
            equiv_pass += 1
        elif result['equiv'] is False:
            equiv_fail += 1
        else:
            equiv_na += 1
            
    except Exception as e:
        print(f"    ERROR in test {idx}: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*70}")
print(f"RESULTS SUMMARY")
print(f"{'='*70}")
print(f"Total tests: {len(results)}")
print(f"Equivalence PASS: {equiv_pass}")
print(f"Equivalence FAIL: {equiv_fail}")
print(f"Equivalence N/A: {equiv_na}")
print(f"Pass rate: {100*equiv_pass/len(results):.1f}%")

if equiv_fail > 0:
    print(f"\nWARNING: {equiv_fail} equivalence failures detected!")
    print("Failed cases:")
    for r in results:
        if r['equiv'] is False:
            print(f"  Test {r['index']}: {r['distribution']}")
else:
    print(f"\nSUCCESS: All equivalence checks passed!")

print(f"{'='*70}")

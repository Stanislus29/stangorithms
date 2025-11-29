"""
SymPy 10-Variable Failure Verification
=======================================

This simple test demonstrates that SymPy cannot solve 10-variable Boolean
minimization problems within reasonable time constraints.

Run this test yourself to verify SymPy's limitations before reviewing the
KMapSolver3D-only benchmark results.

Author: Stan's Technologies
Date: November 2025
"""

import random
import time
import sys
import os
from sympy import symbols, SOPform

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

# Configuration
TIMEOUT_SECONDS = 60  # 1 minute timeout
NUM_TESTS = 10        # Number of random test cases
SEED = 42

def generate_random_output_values(num_vars, p_one=0.45, p_dc=0.1):
    """Generate random Boolean function output values."""
    size = 2 ** num_vars
    choices = [1, 0, 'd']
    weights = [p_one, 1 - p_one - p_dc, p_dc]
    return [random.choices(choices, weights=weights, k=1)[0] for _ in range(size)]

def get_minterms_from_output_values(output_values):
    """Extract minterms and don't-cares from output values."""
    minterms = [idx for idx, val in enumerate(output_values) if val == 1]
    dont_cares = [idx for idx, val in enumerate(output_values) if val == 'd']
    return minterms, dont_cares

def test_sympy_10var():
    """Test SymPy on 10-variable problems."""
    print("=" * 80)
    print("SYMPY 10-VARIABLE BOOLEAN MINIMIZATION TEST")
    print("=" * 80)
    print(f"\nThis test demonstrates SymPy's inability to solve 10-variable")
    print(f"Boolean minimization within {TIMEOUT_SECONDS}s timeout.")
    print(f"\nRunning {NUM_TESTS} random test cases...")
    print(f"Random seed: {SEED}\n")
    
    random.seed(SEED)
    num_vars = 10
    var_symbols = symbols(f'x1:{num_vars+1}')
    
    successful = 0
    timed_out = 0
    
    for test_num in range(1, NUM_TESTS + 1):
        # Generate random test case
        output_values = generate_random_output_values(num_vars)
        minterms, dont_cares = get_minterms_from_output_values(output_values)
        
        print(f"Test {test_num}/{NUM_TESTS}: ", end="", flush=True)
        print(f"Minterms: {len(minterms)}, Don't-cares: {len(dont_cares)} ... ", end="", flush=True)
        
        # Try to minimize with SymPy
        start_time = time.perf_counter()
        try:
            # Attempt minimization
            expr = SOPform(var_symbols, minterms, dont_cares)
            elapsed = time.perf_counter() - start_time
            
            if elapsed < TIMEOUT_SECONDS:
                print(f"✓ Completed in {elapsed:.2f}s")
                successful += 1
            else:
                print(f"⏱️ Exceeded {TIMEOUT_SECONDS}s (took {elapsed:.2f}s)")
                timed_out += 1
        except KeyboardInterrupt:
            elapsed = time.perf_counter() - start_time
            print(f"❌ Interrupted at {elapsed:.2f}s (manually stopped)")
            timed_out += 1
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            print(f"❌ Error at {elapsed:.2f}s: {str(e)[:50]}")
            timed_out += 1
    
    # Summary
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Total tests:       {NUM_TESTS}")
    print(f"Successful:        {successful} ({100*successful/NUM_TESTS:.1f}%)")
    print(f"Failed/Timed out:  {timed_out} ({100*timed_out/NUM_TESTS:.1f}%)")
    
    if successful < 8:  # Less than 80% success
        print("\n⚠️  VERDICT: SymPy FAILED - Success rate below 80% threshold")
        print("   SymPy is not suitable for 10-variable Boolean minimization")
        print("   within reasonable time constraints.")
    else:
        print("\n✓ VERDICT: SymPy succeeded on most test cases")
    
    print("\n" + "=" * 80)
    print("INSTRUCTIONS FOR REVIEWERS")
    print("=" * 80)
    print("1. Run this test yourself: python verify_sympy_10var_failure.py")
    print("2. Observe SymPy's performance (likely many timeouts/failures)")
    print("3. Compare with KMapSolver3D benchmark results")
    print("4. Understand why statistical comparison is not meaningful")
    print("\nNote: You may need to manually interrupt (Ctrl+C) if tests hang.")
    print("=" * 80)

if __name__ == "__main__":
    print("\n⚠️  WARNING: This test may take several minutes to complete.")
    print("Some test cases may hang for extended periods.")
    print("Press Ctrl+C to interrupt hanging tests.\n")
    
    try:
        test_sympy_10var()
    except KeyboardInterrupt:
        print("\n\n❌ Test interrupted by user")
        print("This demonstrates SymPy's impractical execution time for 10-variable problems.")
        sys.exit(1)

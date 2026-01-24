"""
Test 5D Clustering for n > 10 Variables

This script tests the new minimize_5d function which handles
Boolean functions with more than 10 variables using 5D clustering.

Note: While 4D can handle up to 16 variables, the worst-case 
optimality bound is 10 variables, so 5D clustering activates 
for n > 10, partitioning into 10-variable hyperchunks.
"""

import sys
import os
import random
from datetime import datetime
import io

# Set UTF-8 encoding for console output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')))
from stanlogic.BoolMinGeo import BoolMinGeo

# Configuration
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def test_5d_clustering(num_vars=12, density=0.5):
    """
    Test 5D clustering with a function of specified density.
    
    Args:
        num_vars: Number of variables (should be > 10 for 5D)
        density: Fraction of minterms that are 1 (0.0 to 1.0)
    """
    print(f"\n{'='*80}")
    print(f"5D CLUSTERING TEST")
    print(f"{'='*80}")
    print(f"Variables: {num_vars}")
    print(f"Density: {density*100:.1f}%")
    print(f"Total minterms: {2**num_vars:,}")
    print(f"{'='*80}\n")
    
    # Generate random function
    total_minterms = 2 ** num_vars
    num_ones = int(total_minterms * density)
    
    output_values = [0] * total_minterms
    ones_indices = random.sample(range(total_minterms), num_ones)
    
    for idx in ones_indices:
        output_values[idx] = 1
    
    print(f"Generated function with {num_ones:,} ones out of {total_minterms:,} minterms")
    print(f"\nStarting 5D minimization at {datetime.now().strftime('%H:%M:%S')}...")
    
    print(f"Generated function with {num_ones:,} ones out of {total_minterms:,} minterms")
    print(f"\nStarting 5D minimization at {datetime.now().strftime('%H:%M:%S')}...")
    
    # Create solver and minimize
    solver = BoolMinGeo(num_vars, output_values)
    
    # Store 5D EPIs by temporarily storing them
    original_minimize_with_5d = solver._minimize_with_5d_clustering
    epis_5d_captured = []
    
    def capture_5d_clustering(hyperchunk_results):
        result = original_minimize_with_5d(hyperchunk_results)
        epis_5d_captured.extend(result)
        return result
    
    solver._minimize_with_5d_clustering = capture_5d_clustering
    
    start_time = datetime.now()
    terms, expression = solver.minimize_5d(form='sop')
    end_time = datetime.now()
    
    elapsed = (end_time - start_time).total_seconds()
    
    print(f"\n{'='*80}")
    print(f"5D CLUSTER ANALYSIS")
    print(f"{'='*80}")
    print(f"Found {len(epis_5d_captured)} total EPIs from 5D clustering\n")
    
    if len(epis_5d_captured) > 0:
        # Analyze 5D clusters
        hyperchunk_bits = num_vars - 10
        
        # Categorize clusters by dimensionality
        true_5d_clusters = []  # Patterns with don't-cares in hyperchunk bits
        for epi in epis_5d_captured:
            pattern = epi['full_pattern']
            hyperchunk_pattern = pattern[:hyperchunk_bits]
            
            # Check if pattern spans multiple hyperchunks (has '-' in hyperchunk bits)
            if '-' in hyperchunk_pattern:
                true_5d_clusters.append(epi)
        
        print(f"True 5D clusters (spanning hyperchunks): {len(true_5d_clusters)}")
        print(f"4D and lower clusters: {len(epis_5d_captured) - len(true_5d_clusters)}")
        
        if true_5d_clusters:
            print(f"\n{'='*80}")
            print(f"5D CLUSTER DETAILS")
            print(f"{'='*80}\n")
            
            densities = []
            coverages = []
            hyperspans = []
            
            for i, cluster in enumerate(true_5d_clusters[:20], 1):  # Show first 20
                pattern = cluster['full_pattern']
                hyperspan = cluster['hyperspan']
                
                # Calculate information density (ratio of don't-cares)
                dont_cares = pattern.count('-')
                info_density = dont_cares / num_vars
                
                # Calculate coverage (number of minterms)
                coverage = 2 ** dont_cares
                
                densities.append(info_density)
                coverages.append(coverage)
                hyperspans.append(hyperspan)
                
                print(f"Cluster {i}:")
                print(f"  Pattern: {cluster['hyperchunk_pattern']} + {cluster['inner_pattern']}")
                print(f"  Full pattern: {pattern}")
                print(f"  Hyperspan: {hyperspan} hyperchunks")
                print(f"  Information density: {info_density:.3f} ({dont_cares}/{num_vars} don't-cares)")
                print(f"  Coverage: {coverage:,} minterms")
                print()
            
            if len(true_5d_clusters) > 20:
                print(f"... and {len(true_5d_clusters) - 20} more 5D clusters\n")
            
            # Summary statistics
            print(f"{'='*80}")
            print(f"5D CLUSTER STATISTICS")
            print(f"{'='*80}")
            print(f"Total 5D clusters: {len(true_5d_clusters)}")
            print(f"Average information density: {sum(densities)/len(densities):.3f}")
            print(f"Min density: {min(densities):.3f}, Max density: {max(densities):.3f}")
            print(f"Average coverage: {sum(coverages)/len(coverages):,.0f} minterms")
            print(f"Total coverage (sum): {sum(coverages):,} minterms")
            print(f"Average hyperspan: {sum(hyperspans)/len(hyperspans):.1f} hyperchunks")
            print(f"Max hyperspan: {max(hyperspans)} hyperchunks")
        else:
            print("\nNo true 5D clusters found (no patterns spanning hyperchunks)")
    else:
        print("No 5D clusters found")
    
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Execution time: {elapsed:.2f} seconds")
    print(f"Number of terms: {len(terms)}")
    
    print(f"\n{'='*80}")
    print(f"TEST COMPLETE")
    print(f"{'='*80}\n")
    
    return terms, expression


def main():
    """Main test function."""
    print(f"\n{'='*80}")
    print(f"5D K-MAP CLUSTERING TEST SUITE")
    print(f"Starting: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    # Test 1: 12 variables (2 hyperchunk bits) - 50% density
    print("\n" + "="*80)
    print("TEST 1: 12 Variables, 50% Density")
    print("="*80)
    print("This should show 5D clustering with 2-bit hyperchunk dimension")
    test_5d_clustering(num_vars=12, density=0.5)
    
    # Test 2: 14 variables (4 hyperchunk bits) - 70% density (dense function)
    print("\n" + "="*80)
    print("TEST 2: 14 Variables, 70% Density (Dense Function)")
    print("="*80)
    print("According to theory, we should find 5D clusters in dense functions")
    test_5d_clustering(num_vars=14, density=0.7)
    
    print(f"\n{'='*80}")
    print(f"ALL TESTS COMPLETE")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

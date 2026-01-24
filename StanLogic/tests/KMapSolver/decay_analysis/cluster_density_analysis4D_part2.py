"""
4D Cluster Density Analysis - Part 2 (19-20 Variables)
"""

import sys
import os
import random
import csv
from datetime import datetime
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')))
from stanlogic.BoolMinGeo import BoolMinGeo

# Configuration
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Output directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, "..", "outputs", "decay", "analysis")
RESULTS_CSV = os.path.join(OUTPUTS_DIR, "cluster_density_analysis_4D_part2.csv")

os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Test configuration - PART 2
VARIABLE_RANGE = range(19, 21)  # 19-20 variables
DENSITY_LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9]
TESTS_PER_CONFIG = 2


def generate_random_function(num_vars, density):
    """Generate random Boolean function with specified density."""
    total_minterms = 2 ** num_vars
    num_ones = round(total_minterms * density)
    
    output_values = [0] * total_minterms
    ones_indices = random.sample(range(total_minterms), num_ones)
    
    for idx in ones_indices:
        output_values[idx] = 1
    
    return output_values


def extract_4d_clusters(solver, form='sop'):
    """Extract 4D clusters from the minimization process."""
    if solver.num_vars <= 8:
        # Fall back to 3D for small variable counts
        id_set = sorted(solver.kmaps.keys())
        β = {}
        for idx in id_set:
            result = solver._solve_single_kmap(idx, form)
            β[idx] = result['terms_bits']
        epis = solver._minimize_with_3d_clustering(β, id_set)
        patterns = list(epis)  # epis is a set of strings, not list of dicts
    else:
        # Use 4D clustering for n > 8
        chunks = solver._partition_into_chunks()
        
        chunk_results = {}
        for chunk_id, chunk_kmap in chunks.items():
            chunk_minimizer = solver._create_chunk_minimizer(chunk_kmap)
            terms, _ = chunk_minimizer.minimize_3d(form)
            patterns_3d = solver._extract_patterns_from_terms(terms, chunk_minimizer)
            chunk_results[chunk_id] = patterns_3d
        
        epis = solver._minimize_with_4d_clustering(chunk_results)
        patterns = [c['full_pattern'] for c in epis]
    
    if not patterns:
        return {
            'patterns': [],
            'coverage': {},
            'total_clusters': 0,
            'cluster_sizes': [],
            'avg_density': 0,
            'max_density': 0,
            'min_density': 0,
            'std_density': 0
        }
    
    coverage = solver.get_cluster_coordinates(patterns)
    cluster_sizes = [len(minterms) for minterms in coverage.values()]
    
    return {
        'patterns': patterns,
        'coverage': coverage,
        'total_clusters': len(patterns),
        'cluster_sizes': cluster_sizes,
        'avg_density': np.mean(cluster_sizes) if cluster_sizes else 0,
        'max_density': max(cluster_sizes) if cluster_sizes else 0,
        'min_density': min(cluster_sizes) if cluster_sizes else 0,
        'std_density': np.std(cluster_sizes) if cluster_sizes else 0
    }


def run_analysis():
    """Run comprehensive cluster density analysis - PART 2."""
    print(f"\n{'='*80}")
    print(f"4D CLUSTER DENSITY ANALYSIS - PART 2 (Variables 19-20)")
    print(f"{'='*80}")
    print(f"Variables: {min(VARIABLE_RANGE)} to {max(VARIABLE_RANGE)}")
    print(f"Density levels: {DENSITY_LEVELS}")
    print(f"Tests per config: {TESTS_PER_CONFIG}")
    print(f"Total tests: {len(VARIABLE_RANGE) * len(DENSITY_LEVELS) * TESTS_PER_CONFIG}")
    print(f"{'='*80}\n")
    
    results = []
    test_count = 0
    total_tests = len(VARIABLE_RANGE) * len(DENSITY_LEVELS) * TESTS_PER_CONFIG
    
    with open(RESULTS_CSV, 'w', newline='') as csvfile:
        fieldnames = [
            'test_id', 'num_vars', 'density', 'total_minterms', 'ones_count',
            'num_4d_clusters', 'avg_cluster_size', 'max_cluster_size', 'min_cluster_size',
            'std_cluster_size', 'total_coverage', 'coverage_by_4d_percent'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for num_vars in VARIABLE_RANGE:
            print(f"\n{'='*80}")
            print(f"Testing {num_vars}-variable functions")
            print(f"{'='*80}")
            
            for density in DENSITY_LEVELS:
                print(f"\n  Density: {density*100:.0f}% ", end='')
                
                for test_num in range(TESTS_PER_CONFIG):
                    test_count += 1
                    print(f".", end='', flush=True)
                    
                    output_values = generate_random_function(num_vars, density)
                    ones_count = sum(output_values)
                    total_minterms = 2 ** num_vars
                    
                    solver = BoolMinGeo(num_vars, output_values)
                    
                    # Extract 4D clusters (suppress output)
                    import io
                    import contextlib
                    
                    f = io.StringIO()
                    with contextlib.redirect_stdout(f):
                        cluster_info = extract_4d_clusters(solver, form='sop')
                    
                    # Calculate coverage (accounting for overlaps)
                    # Get unique minterms covered by all clusters
                    covered_minterms = set()
                    for minterms_list in cluster_info['coverage'].values():
                        covered_minterms.update(minterms_list)
                    
                    total_coverage = len(covered_minterms)
                    coverage_percent = (total_coverage / ones_count * 100) if ones_count > 0 else 0
                    
                    result = {
                        'test_id': test_count,
                        'num_vars': num_vars,
                        'density': density,
                        'total_minterms': total_minterms,
                        'ones_count': ones_count,
                        'num_4d_clusters': cluster_info['total_clusters'],
                        'avg_cluster_size': cluster_info['avg_density'],
                        'max_cluster_size': cluster_info['max_density'],
                        'min_cluster_size': cluster_info['min_density'],
                        'std_cluster_size': cluster_info['std_density'],
                        'total_coverage': total_coverage,
                        'coverage_by_4d_percent': coverage_percent
                    }
                    
                    results.append(result)
                    writer.writerow(result)
                
                print(f" ✓ ({test_count}/{total_tests})")
    
    print(f"\n{'='*80}")
    print(f"PART 2 COMPLETE! Results saved to:")
    print(f"  {RESULTS_CSV}")
    print(f"{'='*80}\n")
    
    return results


def main():
    """Main execution function."""
    print(f"\n{'='*80}")
    print(f"4D CLUSTER DENSITY ANALYSIS - PART 2")
    print(f"Starting: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    results = run_analysis()
    
    print(f"\n{'='*80}")
    print(f"PART 2 COMPLETE")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Completed {len(results)} tests")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

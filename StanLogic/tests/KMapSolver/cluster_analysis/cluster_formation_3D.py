from stanlogic.BoolMinGeo import BoolMinGeo
import random
import csv
import os 
from datetime import datetime

# Set random seed for reproducibility
random.seed(42)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Output directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, "..", "outputs", "cluster_formation")
RESULTS_CSV = os.path.join(OUTPUTS_DIR, f"cluster_analysis3d_results_{timestamp}.csv")

# Create output directory if it doesn't exist
os.makedirs(OUTPUTS_DIR, exist_ok=True)

print(f"{'='*70}")
print("CLUSTER FORMATION 3D ANALYSIS")
print(f"{'='*70}")
print(f"Script directory: {SCRIPT_DIR}")
print(f"Output directory: {OUTPUTS_DIR}")
print(f"CSV will be written to: {RESULTS_CSV}")
print(f"Output directory exists: {os.path.exists(OUTPUTS_DIR)}")
print(f"{'='*70}\n")

def generate_random_function(num_vars, density):
    """Generate random Boolean function with specified density."""
    total_minterms = 2 ** num_vars
    num_ones = round(total_minterms * density)
    
    output_values = [0] * total_minterms
    ones_indices = random.sample(range(total_minterms), num_ones)
    
    for idx in ones_indices:
        output_values[idx] = 1
    
    return output_values

def extract_3d_clusters(solver, form='sop'):
    """Extract 3D clusters from the solver."""
    # Build dictionary β mapping identifiers to patterns
    β = {}
    id_set = []
    
    # For 6 variables, we have 2^(6-4) = 4 maps: "00", "01", "10", "11"
    num_extra_vars = solver.num_vars - 4
    num_maps = 2 ** num_extra_vars
    
    # Collect all target minterms
    target_val = 0 if form.lower() == 'pos' else 1
    all_target_minterms = set()
    for i, val in enumerate(solver.output_values):
        if val == target_val:
            bits = format(i, f'0{solver.num_vars}b')
            all_target_minterms.add(bits)
    
    # Solve each K-map and collect patterns
    for i in range(num_maps):
        # Convert index to binary string (e.g., 0 -> "00", 1 -> "01", etc.)
        extra_combo = format(i, f'0{num_extra_vars}b')
        id_set.append(extra_combo)
        
        # Solve this K-map to get patterns
        result = solver._solve_single_kmap(extra_combo, form='sop')
        
        # Store the patterns (terms_bits) for this identifier
        β[extra_combo] = result['terms_bits']
    
    # Apply 3D clustering
    epis = solver._minimize_with_3d_clustering(β, id_set)
    
    return {
        'num_3d_clusters': len(epis),
        '3d_clusters': epis,
        'patterns_by_map': β,
        'map_identifiers': id_set,
        'target_minterms': all_target_minterms
    }

def get_covered_minterms(solver, clusters):
    """
    Get all minterms covered by a set of clusters.
    
    Args:
        solver: BoolMinGeo solver instance
        clusters: Set of full pattern strings
        
    Returns:
        set: Set of minterms covered
    """
    covered = set()
    
    for pattern in clusters:
        # Expand pattern to all concrete minterms it represents
        covered.update(solver._expand_pattern(pattern))
    
    return covered

def get_information_density(solver, clusters):
    """
    Calculate information density metrics from cluster sizes.
    """
    information = {} #empty dict to contain number of minterms in a cluster

    for pattern in clusters:
        minterms = solver._expand_pattern(pattern)
        information[pattern] = len(minterms)

    return information

def build_cluster_minterm_mapping(solver, clusters):
    """
    Build a dictionary mapping each cluster to the set of minterms it covers.
    
    Args:
        solver: BoolMinGeo solver instance
        clusters: Set of cluster patterns
        
    Returns:
        dict: Mapping of cluster pattern -> set of minterms
    """
    cluster_to_minterms = {}
    
    for pattern in clusters:
        minterms = solver._expand_pattern(pattern)
        cluster_to_minterms[pattern] = set(minterms)
    
    return cluster_to_minterms

def compute_unique_coverage(cluster_to_minterms):
    """
    For each cluster, find minterms it uniquely covers (not covered by any other cluster).
    
    Args:
        cluster_to_minterms: Dictionary mapping cluster -> set of minterms
        
    Returns:
        dict: Mapping of cluster -> set of unique minterms
    """
    unique_coverage = {}
    cluster_list = list(cluster_to_minterms.keys())
    
    for i, cluster in enumerate(cluster_list):
        cluster_minterms = cluster_to_minterms[cluster]
        
        # Get all minterms covered by other clusters
        other_minterms = set()
        for j, other_cluster in enumerate(cluster_list):
            if i != j:
                other_minterms.update(cluster_to_minterms[other_cluster])
        
        # Unique minterms = minterms covered by this cluster but not by others
        unique_minterms = cluster_minterms - other_minterms
        unique_coverage[cluster] = unique_minterms
    
    return unique_coverage

def compute_uniqueness_metrics(cluster_to_minterms, avg_info_density):
    """
    Compute uniqueness ratio for each cluster.
    
    Uniqueness Ratio = Unique minterms / avg_information_density
    
    A ratio close to 1.0 means the cluster is mostly unique.
    A ratio close to 0.0 means the cluster is highly redundant.
    
    Args:
        cluster_to_minterms: Dictionary mapping cluster -> set of minterms
        avg_info_density: Average information density across all clusters
        
    Returns:
        dict: Statistics about uniqueness
    """
    unique_coverage = compute_unique_coverage(cluster_to_minterms)
    
    uniqueness_ratios = []
    for cluster, unique_minterms in unique_coverage.items():
        num_unique = len(unique_minterms)
        
        # Uniqueness ratio
        if avg_info_density > 0:
            ratio = num_unique / avg_info_density
        else:
            ratio = 0.0
        
        uniqueness_ratios.append(ratio)
    
    # Compute statistics
    if uniqueness_ratios:
        avg_uniqueness_ratio = sum(uniqueness_ratios) / len(uniqueness_ratios)
        min_ratio = min(uniqueness_ratios)
        max_ratio = max(uniqueness_ratios)
    else:
        avg_uniqueness_ratio = 0.0
        min_ratio = 0.0
        max_ratio = 0.0
    
    return {
        'avg_uniqueness_ratio': avg_uniqueness_ratio,
        'min_uniqueness_ratio': min_ratio,
        'max_uniqueness_ratio': max_ratio,
        'uniqueness_ratios': uniqueness_ratios,
        'unique_coverage': unique_coverage
    }        

def check_coverage(solver, epis, all_target_minterms):
    """
    Check coverage of target minterms by 3D clusters.
    
    Args:
        solver: BoolMinGeo solver instance
        epis: Set of 3D clusters
        all_target_minterms: Set of all target minterms
        
    Returns:
        dict: Coverage information including uniqueness metrics
    """
    # Check coverage after 3D clustering
    covered_by_3d = get_covered_minterms(solver, epis)
    uncovered_after_3d = all_target_minterms - covered_by_3d

    information_density = get_information_density(solver, epis)
    
    # Calculate average information density
    avg_info_density = (sum(information_density.values()) / len(information_density) 
                       if information_density else 0)
    
    # Calculate min and max information density
    min_info_density = min(information_density.values()) if information_density else 0
    max_info_density = max(information_density.values()) if information_density else 0
    
    # Build cluster-to-minterm mapping
    cluster_to_minterms = build_cluster_minterm_mapping(solver, epis)
    
    # Compute uniqueness metrics
    uniqueness_metrics = compute_uniqueness_metrics(cluster_to_minterms, avg_info_density)
    
    return {
        'covered_minterms': covered_by_3d,
        'information_density': information_density,
        'uncovered_minterms': uncovered_after_3d,
        'coverage_ratio': len(covered_by_3d) / len(all_target_minterms) if all_target_minterms else 1.0,
        'avg_info_density': avg_info_density,
        'min_info_density': min_info_density,
        'max_info_density': max_info_density,
        'avg_uniqueness_ratio': uniqueness_metrics['avg_uniqueness_ratio'],
        'min_uniqueness_ratio': uniqueness_metrics['min_uniqueness_ratio'],
        'max_uniqueness_ratio': uniqueness_metrics['max_uniqueness_ratio']
    }

def main():
    num_vars_range = range(5, 17)  # Variables from 5 to 16
    density_levels = [0.3, 0.5, 0.7, 0.9]
    tests_per_config = 10
    
    all_results = []
    test_count = 0

    for num_vars in num_vars_range:
        print(f"\n{'='*70}")
        print(f"║ Tests for {num_vars}-variables")
        print(f"{'='*70}")
        
        for density in density_levels:
            print(f"\n  Density {density}", end='', flush=True)
            
            for test_num in range(tests_per_config):
                test_count += 1
                
                # Generate random Boolean function
                output_values = generate_random_function(num_vars, density)

                # Initialize BoolMinGeo solver
                solver = BoolMinGeo(num_vars)
                solver.set_output_values(output_values)

                # Extract 3D clusters
                cluster_info = extract_3d_clusters(solver)

                # Check coverage
                coverage_info = check_coverage(solver, cluster_info['3d_clusters'], cluster_info['target_minterms'])

                # Store results
                result = {
                    'test_id': test_count,
                    'num_vars': num_vars,
                    'density': density,
                    'num_3d_clusters': cluster_info['num_3d_clusters'],
                    'avg_information_density': coverage_info['avg_info_density'],
                    'min_information_density': coverage_info['min_info_density'],
                    'max_information_density': coverage_info['max_info_density'],
                    'coverage_ratio': coverage_info['coverage_ratio'],
                    'avg_uniqueness_ratio': coverage_info['avg_uniqueness_ratio'],
                    'min_uniqueness_ratio': coverage_info['min_uniqueness_ratio'],
                    'max_uniqueness_ratio': coverage_info['max_uniqueness_ratio'],
                    'num_target_minterms': len(cluster_info['target_minterms']),
                    'num_covered': len(coverage_info['covered_minterms']),
                    'num_uncovered': len(coverage_info['uncovered_minterms'])
                }
                all_results.append(result)
                
                # Show progress with accumulating dots
                dots = '.' * (test_num + 1)
                print(f"\r  Density {density}{dots}({test_num+1}/{tests_per_config})", end='', flush=True)
            
            # Completed this density
            print(f" ✓ Complete")
                
    # Print final summary
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY: Completed {test_count} tests")
    print(f"{'='*60}")
    
    # Aggregate results by num_vars and density and write to CSV
    print(f"\n{'='*70}")
    print("AGGREGATING RESULTS AND WRITING TO CSV...")
    print(f"{'='*70}")
    
    aggregated_data = []
    
    for num_vars in num_vars_range:
        for density in density_levels:
            # Filter results for this configuration
            config_results = [r for r in all_results 
                            if r['num_vars'] == num_vars and r['density'] == density]
            
            if config_results:
                # Calculate averages and extremes
                avg_clusters = sum(r['num_3d_clusters'] for r in config_results) / len(config_results)
                min_clusters = min(r['num_3d_clusters'] for r in config_results)
                max_clusters = max(r['num_3d_clusters'] for r in config_results)
                
                avg_info_density = sum(r['avg_information_density'] for r in config_results) / len(config_results)
                min_info_density = min(r['min_information_density'] for r in config_results)
                max_info_density = max(r['max_information_density'] for r in config_results)
                
                avg_coverage = sum(r['coverage_ratio'] for r in config_results) / len(config_results)
                avg_uniqueness = sum(r['avg_uniqueness_ratio'] for r in config_results) / len(config_results)
                avg_min_uniqueness = sum(r['min_uniqueness_ratio'] for r in config_results) / len(config_results)
                avg_max_uniqueness = sum(r['max_uniqueness_ratio'] for r in config_results) / len(config_results)
                
                aggregated_data.append({
                    'num_vars': num_vars,
                    'density': density,
                    'num_tests': len(config_results),
                    'avg_3d_clusters': avg_clusters,
                    'min_3d_clusters': min_clusters,
                    'max_3d_clusters': max_clusters,
                    'avg_information_density': avg_info_density,
                    'min_information_density': min_info_density,
                    'max_information_density': max_info_density,
                    'avg_coverage_ratio': avg_coverage,
                    'avg_uniqueness_ratio': avg_uniqueness,
                    'avg_min_uniqueness_ratio': avg_min_uniqueness,
                    'avg_max_uniqueness_ratio': avg_max_uniqueness
                })
                print(f"  Aggregated: {num_vars} vars, density {density} ({len(config_results)} tests)")
    
    # Write to CSV
    print(f"\nWriting to CSV: {RESULTS_CSV}")
    try:
        with open(RESULTS_CSV, 'w', newline='') as csvfile:
            fieldnames = ['num_vars', 'density', 'num_tests', 
                         'avg_3d_clusters', 'min_3d_clusters', 'max_3d_clusters',
                         'avg_information_density', 'min_information_density', 'max_information_density',
                         'avg_coverage_ratio',
                         'avg_uniqueness_ratio', 'avg_min_uniqueness_ratio', 'avg_max_uniqueness_ratio']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in aggregated_data:
                writer.writerow(row)
        
        print(f"✓ CSV file successfully created: {RESULTS_CSV}")
    except Exception as e:
        print(f"✗ ERROR writing CSV: {e}")
        print(f"  Attempted path: {RESULTS_CSV}")
        print(f"  Directory exists: {os.path.exists(OUTPUTS_DIR)}")
    
    print(f"\n{'='*70}")
    print(f"Results written to: {RESULTS_CSV}")
    print(f"Total configurations: {len(aggregated_data)}")
    print(f"{'='*70}")
    
    # Also print summary to console
    print(f"\n{'='*60}")
    print("AGGREGATED RESULTS SUMMARY")
    print(f"{'='*60}\n")
    print(f"{'Vars':<6} {'Density':<8} {'Clusters':<16} {'Info Dens':<16} {'Coverage':<10} {'Uniqueness':<12}")
    print(f"{'':6} {'':8} {'(avg/min/max)':<16} {'(avg/min/max)':<16}")
    print("-" * 80)
    
    for data in aggregated_data:
        print(f"{data['num_vars']:<6} {data['density']:<8.1f} "
              f"{data['avg_3d_clusters']:>5.1f}/{data['min_3d_clusters']:>3.0f}/{data['max_3d_clusters']:>3.0f}  "
              f"{data['avg_information_density']:>5.1f}/{data['min_information_density']:>3.0f}/{data['max_information_density']:>3.0f}  "
              f"{data['avg_coverage_ratio']:<10.2%} "
              f"{data['avg_uniqueness_ratio']:<12.4f}")
    
    return all_results



if __name__ == "__main__":
    main()
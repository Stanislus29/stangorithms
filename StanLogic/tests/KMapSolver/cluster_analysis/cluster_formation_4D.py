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
RESULTS_CSV = os.path.join(OUTPUTS_DIR, f"cluster_analysis4d_results_{timestamp}.csv")

# Create output directory if it doesn't exist
os.makedirs(OUTPUTS_DIR, exist_ok=True)

print(f"{'='*70}")
print("CLUSTER FORMATION 4D ANALYSIS")
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

def extract_4d_clusters(solver, form='sop'):
    """Extract 4D clusters from the solver."""
    # Build dictionary β mapping identifiers to patterns
    
    # Collect all target minterms
    target_val = 0 if form.lower() == 'pos' else 1
    all_target_minterms = set()
    for i, val in enumerate(solver.output_values):
        if val == target_val:
            bits = format(i, f'0{solver.num_vars}b')
            all_target_minterms.add(bits)

    chunk_results = {}
    
    # For 6 variables, we have 2^(6-4) = 4 maps: "00", "01", "10", "11"
    c = solver.num_vars - 8  # chunk bits
    num_chunks = 2 ** c
    
    chunks = {}

    # Generate all chunk identifiers
    for chunk_idx in range(num_chunks):
        chunk_id = format(chunk_idx, f'0{c}b')
        
        # Extract K-maps belonging to this chunk
        chunk_kmaps = {}
        
        # For this chunk, we need the 16 K-maps (4-bit identifiers for 8-var)
        for inner_id_idx in range(16):
            inner_id = format(inner_id_idx, '04b')  # 4 bits for inner identifiers
            
            # Full identifier in original structure
            full_id = chunk_id + inner_id
            
            if full_id in solver.kmaps:
                chunk_kmaps[inner_id] = solver.kmaps[full_id]
        
        chunks[chunk_id] = chunk_kmaps

    for chunk_id, chunk_kmaps in chunks.items():
        print(f"\nChunk {chunk_id}:")
        print(f"  Building 8-variable subproblem...")
        
        # Build output values for this chunk (256 values for 8 variables)
        chunk_output_values = [0] * 256
        gray_code = ['00', '01', '11', '10']
        
        for idx_str, kmap in chunk_kmaps.items():
            idx_val = int(idx_str, 2)  # 4-bit identifier value
            
            for row in range(4):
                for col in range(4):
                    cell = kmap[row][col]
                    if cell and cell.get('value') == 1:
                        # Calculate position in chunk's output values
                        # First 4 bits from identifier, last 4 from K-map position
                        row_bits = gray_code[row]
                        col_bits = gray_code[col]
                        kmap_bits = col_bits + row_bits  # 4 bits
                        full_bits = idx_str + kmap_bits  # 8 bits total
                        position = int(full_bits, 2)
                        chunk_output_values[position] = 1
        
        # Create minimizer with proper output values
        chunk_minimizer = BoolMinGeo(8, chunk_output_values)
        
        # Get 3D clusters for this chunk
        print(f"  Extracting 3D clusters...")
        β = {}
        id_set = []
        
        # For 8 variables, we have 2^(8-4) = 16 maps
        for inner_id_idx in range(16):
            inner_id = format(inner_id_idx, '04b')
            id_set.append(inner_id)
            
            # Solve this K-map to get patterns
            if inner_id in chunk_minimizer.kmaps:
                result = chunk_minimizer._solve_single_kmap(inner_id, form=form)
                β[inner_id] = result['terms_bits']
            else:
                β[inner_id] = []
        
        # Apply 3D clustering to get patterns for this chunk
        patterns = chunk_minimizer._minimize_with_3d_clustering(β, id_set)
        
        chunk_results[chunk_id] = patterns
        
        print(f"  Found {len(patterns)} 3D cluster patterns")
        for pattern in list(patterns)[:5]:  # Show first 5
            print(f"    {pattern}")
        if len(patterns) > 5:
            print(f"    ... and {len(patterns)-5} more")
    
    # Apply 4D clustering
    epis = solver._minimize_with_4d_clustering(chunk_results)
    
    return {
        'num_4d_clusters': len(epis),
        '4d_clusters': epis,
        'patterns_by_chunk': chunk_results,
        'chunk_identifiers': list(chunks.keys()),
        'target_minterms': all_target_minterms
    }

def get_covered_minterms(solver, clusters):
    """
    Get all minterms covered by a set of clusters.
    
    Args:
        solver: BoolMinGeo solver instance
        clusters: List of cluster dictionaries or set of pattern strings
        
    Returns:
        set: Set of minterms covered
    """
    covered = set()
    
    for cluster in clusters:
        # Handle cluster dictionaries from 4D clustering
        if isinstance(cluster, dict):
            pattern = cluster.get('full_pattern', cluster)
        else:
            pattern = cluster
            
        # Expand pattern to all concrete minterms it represents
        covered.update(solver._expand_pattern(pattern))
    
    return covered

def get_information_density(solver, clusters):
    """
    Calculate information density metrics from cluster sizes.
    """
    information = {} #empty dict to contain number of minterms in a cluster

    for cluster in clusters:
        # Handle cluster dictionaries from 4D clustering
        if isinstance(cluster, dict):
            pattern = cluster.get('full_pattern', cluster)
        else:
            pattern = cluster
            
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
    
    for cluster in clusters:
        # Handle cluster dictionaries from 4D clustering
        if isinstance(cluster, dict):
            pattern = cluster.get('full_pattern', cluster)
        else:
            pattern = cluster
            
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
    Check coverage of target minterms by 4D clusters.
    
    Args:
        solver: BoolMinGeo solver instance
        epis: Set of 4D clusters
        all_target_minterms: Set of all target minterms
        
    Returns:
        dict: Coverage information including uniqueness metrics
    """
    # Check coverage after 4D clustering
    covered_by_4d = get_covered_minterms(solver, epis)
    uncovered_after_4d = all_target_minterms - covered_by_4d

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
        'covered_minterms': covered_by_4d,
        'information_density': information_density,
        'uncovered_minterms': uncovered_after_4d,
        'coverage_ratio': len(covered_by_4d) / len(all_target_minterms) if all_target_minterms else 1.0,
        'avg_info_density': avg_info_density,
        'min_info_density': min_info_density,
        'max_info_density': max_info_density,
        'avg_uniqueness_ratio': uniqueness_metrics['avg_uniqueness_ratio'],
        'min_uniqueness_ratio': uniqueness_metrics['min_uniqueness_ratio'],
        'max_uniqueness_ratio': uniqueness_metrics['max_uniqueness_ratio']
    }

def main():
    num_vars_range = range(10,16)  # Variables from 5 to 16
    density_levels = [0.3,0.5,0.7,0.9]
    tests_per_config = 2
    
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

                # Extract 4D clusters
                cluster_info = extract_4d_clusters(solver)

                # Check coverage
                coverage_info = check_coverage(solver, cluster_info['4d_clusters'], cluster_info['target_minterms'])

                # Store results
                result = {
                    'test_id': test_count,
                    'num_vars': num_vars,
                    'density': density,
                    'num_4d_clusters': cluster_info['num_4d_clusters'],
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
                avg_clusters = sum(r['num_4d_clusters'] for r in config_results) / len(config_results)
                min_clusters = min(r['num_4d_clusters'] for r in config_results)
                max_clusters = max(r['num_4d_clusters'] for r in config_results)
                
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
                    'avg_4d_clusters': avg_clusters,
                    'min_4d_clusters': min_clusters,
                    'max_4d_clusters': max_clusters,
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
                         'avg_4d_clusters', 'min_4d_clusters', 'max_4d_clusters',
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
              f"{data['avg_4d_clusters']:>5.1f}/{data['min_4d_clusters']:>3.0f}/{data['max_4d_clusters']:>3.0f}  "
              f"{data['avg_information_density']:>5.1f}/{data['min_information_density']:>3.0f}/{data['max_information_density']:>3.0f}  "
              f"{data['avg_coverage_ratio']:<10.2%} "
              f"{data['avg_uniqueness_ratio']:<12.4f}")
    
    return all_results



if __name__ == "__main__":
    main()
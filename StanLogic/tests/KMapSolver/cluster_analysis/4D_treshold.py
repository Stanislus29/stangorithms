import csv 
import os
import statistics

# Use script directory as base
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path3d = os.path.join(script_dir, "..", "outputs", "cluster_formation", "cluster_analysis3d_results_20260206_164938.csv")
csv_path4d = os.path.join(script_dir, "..", "outputs", "cluster_formation", "cluster_analysis4d_results_20260207_011244.csv")
output_csv = os.path.join(script_dir, "..", "outputs", "cluster_formation", "4d_threshold_analysis.csv")

def read_csv_with_info_density(csv_path):
    """Read CSV and organize data by density including info density."""
    data_by_density = {}
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            density = float(row['density']) if 'density' in row else float(row['Density'])
            num_vars = int(row['num_vars']) if 'num_vars' in row else int(row['Vars'])
            
            # Get info density
            if 'avg_information_density' in row:
                info_density = float(row['avg_information_density'])
            elif 'Avg_Info_Density' in row:
                info_density = float(row['Avg_Info_Density'])
            else:
                info_density = None
            
            if density not in data_by_density:
                data_by_density[density] = []
            
            data_by_density[density].append((num_vars, info_density))
    
    # Sort by num_vars for each density
    for density in data_by_density:
        data_by_density[density].sort(key=lambda x: x[0])
    
    return data_by_density

def find_saturation_point(data_points, tolerance=2.0):
    """
    Find the saturation point where info density stabilizes within ±tolerance.
    Returns the saturation value if found, otherwise None.
    """
    if len(data_points) < 3:
        return None
    
    # Check consecutive values
    for i in range(len(data_points) - 2):
        n1, val1 = data_points[i]
        n2, val2 = data_points[i + 1]
        n3, val3 = data_points[i + 2]
        
        if val1 is None or val2 is None or val3 is None:
            continue
        
        # Check if values are within tolerance of each other
        if abs(val2 - val1) <= tolerance and abs(val3 - val2) <= tolerance:
            # Found saturation point - return the average of stable values
            stable_values = [val1, val2, val3]
            
            # Look ahead to include more stable values
            for j in range(i + 3, len(data_points)):
                _, val_next = data_points[j]
                if val_next is not None and abs(val_next - val3) <= tolerance:
                    stable_values.append(val_next)
                else:
                    break
            
            return statistics.mean(stable_values), n1
    
    return None, None

def calculate_average_geometric_ratio(results_list):
    """Calculate the overall average geometric ratio across all densities."""
    all_ratios = []
    for result in results_list:
        all_ratios.append(result['avg_geometric_ratio'])
    
    if all_ratios:
        return statistics.mean(all_ratios)
    return None

def save_collapse_analysis_to_csv(collapse_data, output_path):
    """Save 4D coverage collapse analysis results to CSV."""
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['n', 'k', 'raw_coverage', 'capped_coverage', 'status', 
                         'C_18', 'geometric_ratio', 'I_sat', 'density']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(collapse_data)
        
        print(f"\nCoverage collapse analysis saved to: {output_path}")
        return True
    except Exception as e:
        print(f"\nError saving collapse analysis to CSV: {e}")
        return False

def analyze_information_saturation(csv_path, label):
    """Analyze information density saturation for each density."""
    print(f"\n{'='*70}")
    print(f"{label} INFORMATION SATURATION ANALYSIS")
    print(f"{'='*70}\n")
    
    data_by_density = read_csv_with_info_density(csv_path)
    
    saturation_results = []
    
    for density in sorted(data_by_density.keys()):
        data_points = data_by_density[density]
        
        print(f"Density: {density}")
        print(f"  Info density progression:")
        for n, info_d in data_points:
            if info_d is not None:
                print(f"    n={n}: {info_d:.2f}")
        
        sat_value, sat_n = find_saturation_point(data_points)
        
        if sat_value is not None:
            print(f"  ✓ Saturation found at n={sat_n}: I_sat ≈ {sat_value:.2f}")
            saturation_results.append({
                'density': density,
                'saturation_value': sat_value,
                'saturation_n': sat_n
            })
        else:
            print(f"  ✗ No clear saturation point found")
        print()
    
    return saturation_results

def read_csv_data(csv_path, cluster_column='avg_3d_clusters'):
    """Read CSV and organize data by density."""
    data_by_density = {}
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            density = float(row['density']) if 'density' in row else float(row['Density'])
            num_vars = int(row['num_vars']) if 'num_vars' in row else int(row['Vars'])
            
            # Handle different column names
            if cluster_column in row:
                clusters = float(row[cluster_column])
            elif 'Avg_Clusters' in row:
                clusters = float(row['Avg_Clusters'])
            else:
                clusters = float(row['avg_4d_clusters'])
            
            if density not in data_by_density:
                data_by_density[density] = []
            
            data_by_density[density].append((num_vars, clusters))
    
    # Sort by num_vars for each density
    for density in data_by_density:
        data_by_density[density].sort(key=lambda x: x[0])
    
    return data_by_density

def calculate_geometric_ratios(data_points):
    """Calculate geometric ratios between consecutive n values."""
    ratios = []
    
    for i in range(len(data_points) - 1):
        n1, clusters1 = data_points[i]
        n2, clusters2 = data_points[i + 1]
        
        if clusters1 > 0:
            ratio = clusters2 / clusters1
            ratios.append(ratio)
    
    return ratios

def calculate_statistics(ratios):
    """Calculate average ratio and deviations."""
    if not ratios:
        return None, None, None
    
    avg_ratio = statistics.mean(ratios)
    
    # Calculate deviations from average
    deviations = [abs(r - avg_ratio) for r in ratios]
    avg_deviation = statistics.mean(deviations)
    
    # Calculate standard deviation
    std_dev = statistics.stdev(ratios) if len(ratios) > 1 else 0
    
    return avg_ratio, avg_deviation, std_dev

def analyze_clustering_growth(csv_path, label, cluster_column='avg_3d_clusters'):
    """Analyze cluster growth for a dataset."""
    print(f"\n{'='*70}")
    print(f"{label} CLUSTER GROWTH ANALYSIS")
    print(f"{'='*70}\n")
    
    data_by_density = read_csv_data(csv_path, cluster_column)
    
    results = []
    
    for density in sorted(data_by_density.keys()):
        data_points = data_by_density[density]
        
        print(f"Density: {density}")
        print(f"  Data points: {len(data_points)}")
        
        # Show the data
        for n, clusters in data_points:
            print(f"    n={n}: {clusters:.2f} clusters")
        
        # Calculate ratios
        ratios = calculate_geometric_ratios(data_points)
        
        if ratios:
            print(f"  Growth ratios between consecutive n:")
            for i, ratio in enumerate(ratios):
                n1, _ = data_points[i]
                n2, _ = data_points[i + 1]
                print(f"    n={n1}→{n2}: {ratio:.4f}x")
            
            avg_ratio, avg_deviation, std_dev = calculate_statistics(ratios)
            
            print(f"\n  Average geometric ratio: {avg_ratio:.4f}")
            print(f"  Average deviation: {avg_deviation:.4f}")
            print(f"  Standard deviation: {std_dev:.4f}")
            print()
            
            results.append({
                'density': density,
                'num_points': len(data_points),
                'num_ratios': len(ratios),
                'avg_geometric_ratio': avg_ratio,
                'avg_deviation': avg_deviation,
                'std_deviation': std_dev,
                'min_ratio': min(ratios),
                'max_ratio': max(ratios)
            })
        else:
            print(f"  Not enough data points to calculate ratios\n")
    
    return results

# Analyze 3D clustering
print("\n" + "="*70)
print("ANALYZING 3D CLUSTER FORMATION DATA")
print("="*70)
results_3d = analyze_clustering_growth(csv_path3d, "3D", 'avg_3d_clusters')

# Analyze 4D clustering
print("\n" + "="*70)
print("ANALYZING 4D CLUSTER FORMATION DATA")
print("="*70)
results_4d = analyze_clustering_growth(csv_path4d, "4D", 'Avg_Clusters')

# Write results to CSV
print(f"\n{'='*70}")
print("WRITING RESULTS TO CSV")
print(f"{'='*70}\n")

with open(output_csv, 'w', newline='') as f:
    fieldnames = ['type', 'density', 'num_points', 'num_ratios', 'avg_geometric_ratio', 
                  'avg_deviation', 'std_deviation', 'min_ratio', 'max_ratio']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    
    writer.writeheader()
    
    for result in results_3d:
        result['type'] = '3D'
        writer.writerow(result)
    
    for result in results_4d:
        result['type'] = '4D'
        writer.writerow(result)

print(f"✓ Results written to: {output_csv}")

# Calculate overall average geometric ratios
print(f"\n{'='*70}")
print("OVERALL AVERAGE GEOMETRIC RATIOS")
print(f"{'='*70}\n")

avg_ratio_3d = calculate_average_geometric_ratio(results_3d)
avg_ratio_4d = calculate_average_geometric_ratio(results_4d)

print(f"3D Average Geometric Ratio (across all densities): {avg_ratio_3d:.4f}")
print(f"4D Average Geometric Ratio (across all densities): {avg_ratio_4d:.4f}")

# Analyze information saturation for 3D
saturation_3d = analyze_information_saturation(csv_path3d, "3D")

# Analyze information saturation for 4D
saturation_4d = analyze_information_saturation(csv_path4d, "4D")

# Calculate mean I_sat across all densities
print(f"\n{'='*70}")
print("INFORMATION SATURATION SUMMARY")
print(f"{'='*70}\n")

all_sat_values = []
print("3D Saturation Values:")
for result in saturation_3d:
    print(f"  Density {result['density']}: I_sat = {result['saturation_value']:.2f} (at n={result['saturation_n']})")
    all_sat_values.append(result['saturation_value'])

print("\n4D Saturation Values:")
for result in saturation_4d:
    print(f"  Density {result['density']}: I_sat = {result['saturation_value']:.2f} (at n={result['saturation_n']})")
    all_sat_values.append(result['saturation_value'])

if all_sat_values:
    mean_i_sat = statistics.mean(all_sat_values)
    print(f"\n{'='*70}")
    print(f"Mean I_sat (across all densities, 3D and 4D): {mean_i_sat:.2f}")
    print(f"{'='*70}")

# Model coverage collapse for 4D - analyze each density separately
print(f"\n{'='*70}")
print("4D COVERAGE COLLAPSE ANALYSIS (ALL DENSITIES)")
print(f"{'='*70}\n")

# Read all 4D data
with open(csv_path4d, 'r') as f:
    reader = csv.DictReader(f)
    data_4d = list(reader)

# Group by density
data_by_density_4d = {}
for row in data_4d:
    density = float(row['density'])
    if density not in data_by_density_4d:
        data_by_density_4d[density] = []
    data_by_density_4d[density].append(row)

# Analyze each density separately
for density in sorted(data_by_density_4d.keys()):
    print(f"\n{'='*70}")
    print(f"ANALYZING DENSITY {density}")
    print(f"{'='*70}\n")
    
    density_data = data_by_density_4d[density]
    
    # Find n=18 data for this density
    n18_data = [row for row in density_data if int(row.get('num_vars', 0)) == 18]
    
    # Get I_sat for this density
    density_i_sat = None
    for result in saturation_4d:
        if result['density'] == density:
            density_i_sat = result['saturation_value']
            break
    
    # Get geometric ratio for this density
    density_ratio = None
    for result in results_4d:
        if result['density'] == density:
            density_ratio = result['avg_geometric_ratio']
            break
    
    if n18_data and density_i_sat and density_ratio:
        C_18 = float(n18_data[0].get('avg_4d_clusters', 0))
        actual_coverage_18 = float(n18_data[0].get('avg_coverage_ratio', 0)) * 100  # Convert to percentage
            
        print(f"Parameters:")
        print(f"  C_18 (clusters at n=18): {C_18:.2f}")
        print(f"  r (geometric ratio): {density_ratio:.4f}")
        print(f"  I_sat: {density_i_sat:.2f}")
        print(f"  Density: {density}")
        print(f"  Actual coverage at n=18: {actual_coverage_18:.2f}%")
        print(f"\nModeling coverage for n ≥ 19:")
        print(f"  Formula: coverage = (C_18 * r^k * I_sat) / (density * 2^n) * 100")
        print()
        
        # Model coverage for increasing n
        n_start = 19
        max_n = 50  # Set a reasonable limit
        
        coverages = []
        collapse_data = []  # Store data for CSV export
        stagnant_count = 0
        n_collapse = None
        
        print(f"{'n':<6} {'k':<6} {'Raw Model':<15} {'Capped (%)':<15} {'Status':<25}")
        print("-" * 80)
        
        for n in range(n_start, max_n + 1):
            k = n - 18  # k starts at 1 for n=19
            
            # Calculate predicted coverage (uncapped)
            numerator = C_18 * (density_ratio ** k) * density_i_sat
            denominator = density * (2 ** n)
            coverage_raw = (numerator / denominator) * 100
                
            # Cap at 100% (physical limit)
            coverage_pct = min(coverage_raw, 100.0)
            
            coverages.append(coverage_pct)
            
            # Determine status
            status = ""
            if coverage_raw >= 100.0:
                status = "⚠ REDUNDANT (>100%)"
                stagnant_count += 1
            elif len(coverages) >= 2:
                change = coverage_pct - coverages[-2]
                
                if change <= 0.001:  # Essentially no growth
                    stagnant_count += 1
                    status = f"↓ Stagnant ({stagnant_count}/3)"
                else:
                    stagnant_count = 0
                    status = f"↑ Growing (+{change:.4f}%)"
            else:
                status = "Initial"
            
            print(f"{n:<6} {k:<6} {coverage_raw:<15.4f} {coverage_pct:<15.4f} {status:<25}")
            
            # Store data for CSV export
            collapse_data.append({
                'n': n,
                'k': k,
                'raw_coverage': round(coverage_raw, 4),
                'capped_coverage': round(coverage_pct, 4),
                'status': status,
                'C_18': C_18,
                'geometric_ratio': round(density_ratio, 4),
                'I_sat': round(density_i_sat, 2),
                'density': density
            })
                
            # Check termination conditions
            if coverage_raw >= 100.0 and n_collapse is None:
                n_collapse = n
                print(f"\n{'='*80}")
                print(f"REDUNDANCY THRESHOLD REACHED!")
                print(f"{'='*80}")
                print(f"n_collapse = {n_collapse}")
                print(f"At n={n_collapse}, model predicts {coverage_raw:.2f}% coverage")
                print(f"Coverage exceeds 100% - system produces redundant clusters")
                print(f"Continuing simulation to show redundancy growth...")
                print()
                if n < max_n - 5:
                    max_n = n + 5  # Show 5 more iterations after redundancy
            elif stagnant_count >= 3 and n_collapse is None:
                n_collapse = n - 2  # The point where stagnation started
                print(f"\n{'='*80}")
                print(f"EFFICIENCY COLLAPSE DETECTED!")
                print(f"{'='*80}")
                print(f"n_collapse = {n_collapse}")
                print(f"Coverage stagnated at ~{coverage_pct:.4f}% (before reaching 100%)")
                print(f"System becomes inefficient - clusters grow but coverage doesn't")
                print(f"Diminishing returns make further scaling impractical")
                break
        
        if n_collapse is None:
            print(f"\nNote: No collapse detected up to n={max_n}")
            print(f"Coverage continues to change with increasing n")
        
        # Interpret results
        print(f"\n{'='*80}")
        print("INTERPRETATION")
        print(f"{'='*80}")
        if n_collapse and coverages[-1] >= 100:
            print(f"✓ System reaches full coverage at n={n_collapse}")
            print(f"✓ Beyond n={n_collapse}, all clusters are redundant")
            print(f"✓ Optimal operating point: n < {n_collapse}")
        elif n_collapse:
            print(f"⚠ System stagnates at {coverages[n_collapse - n_start]:.2f}% coverage")
            print(f"⚠ Cannot reach 100% due to diminishing returns")
            print(f"⚠ Optimal operating point: n ≈ {n_collapse - 1}")
        else:
            print(f"→ Model suggests continued growth potential")
            print(f"→ May need larger n to reach saturation")
        
        # Save collapse analysis to CSV for this density
        collapse_csv_path = os.path.join(script_dir, "..", "outputs", "cluster_formation", 
                                        f"4d_collapse_density_{density}_analysis.csv")
        save_collapse_analysis_to_csv(collapse_data, collapse_csv_path)
    else:
        print(f"⚠ Skipping density {density}: Missing data (n=18, I_sat, or geometric ratio)")

# Summary table
print(f"\n{'='*70}")
print("SUMMARY TABLE")
print(f"{'='*70}\n")
print(f"{'Type':<6} {'Density':<10} {'Avg Ratio':<12} {'Avg Dev':<12} {'Std Dev':<12}")
print("-" * 70)

for result in results_3d:
    print(f"{'3D':<6} {result['density']:<10.1f} {result['avg_geometric_ratio']:<12.4f} "
          f"{result['avg_deviation']:<12.4f} {result['std_deviation']:<12.4f}")

for result in results_4d:
    print(f"{'4D':<6} {result['density']:<10.1f} {result['avg_geometric_ratio']:<12.4f} "
          f"{result['avg_deviation']:<12.4f} {result['std_deviation']:<12.4f}")
    



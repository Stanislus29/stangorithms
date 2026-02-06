import csv
import os
from datetime import datetime

# Use script directory as base
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path1 = os.path.join(script_dir, "..", "outputs", "cluster_formation", 
                         "cluster_analysis4d_results_20260205_035542.csv")
csv_path2 = os.path.join(script_dir, "..", "outputs", "cluster_formation", 
                         "cluster_analysis4d_results_20260205_035127.csv")
output_dir = os.path.join(script_dir, "..", "outputs", "cluster_formation")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_csv = os.path.join(output_dir, f"merged_4d_results_{timestamp}.csv")

def read_cluster_analysis_file(csv_path):
    """Read cluster_analysis4d_results CSV with full column structure."""
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'num_vars': int(row['num_vars']),
                'density': float(row['density']),
                'num_tests': int(row['num_tests']),
                'avg_4d_clusters': float(row['avg_4d_clusters']),
                'min_4d_clusters': float(row['min_4d_clusters']),
                'max_4d_clusters': float(row['max_4d_clusters']),
                'avg_information_density': float(row['avg_information_density']),
                'min_information_density': float(row['min_information_density']),
                'max_information_density': float(row['max_information_density']),
                'avg_coverage_ratio': float(row['avg_coverage_ratio']),
                'avg_uniqueness_ratio': float(row['avg_uniqueness_ratio']),
                'avg_min_uniqueness_ratio': float(row['avg_min_uniqueness_ratio']),
                'avg_max_uniqueness_ratio': float(row['avg_max_uniqueness_ratio'])
            })
    return data

def merge_data(data_lists):
    """Merge all rows from multiple files."""
    # Concatenate all data from all files
    merged = []
    
    for i, data in enumerate(data_lists):
        merged.extend(data)
        print(f"  File {i+1}: Added {len(data)} rows")
    
    # Sort by num_vars, then density
    merged.sort(key=lambda x: (x['num_vars'], x['density']))
    
    return merged

# Read both files
print("Reading CSV files...")
print(f"  File 1: {csv_path1}")
data1 = read_cluster_analysis_file(csv_path1)
print(f"    Found {len(data1)} rows")

print(f"  File 2: {csv_path2}")
data2 = read_cluster_analysis_file(csv_path2)
print(f"    Found {len(data2)} rows")

# Merge data
print("\nMerging data...")
merged_data = merge_data([data1, data2])
print(f"  Total merged rows: {len(merged_data)}")

# Write merged data to new CSV
print(f"\nWriting merged data to: {output_csv}")
with open(output_csv, 'w', newline='') as f:
    fieldnames = ['num_vars', 'density', 'num_tests', 'avg_4d_clusters',
                 'min_4d_clusters', 'max_4d_clusters', 'avg_information_density',
                 'min_information_density', 'max_information_density', 'avg_coverage_ratio',
                 'avg_uniqueness_ratio', 'avg_min_uniqueness_ratio', 'avg_max_uniqueness_ratio']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(merged_data)

print(f"✓ Merged CSV saved successfully!")

# Display summary
print(f"\n{'='*70}")
print("MERGED DATA SUMMARY")
print(f"{'='*70}\n")
print(f"{'Vars':<6} {'Density':<8} {'Tests':<8} {'Avg Clusters':<14} {'Avg Info Density':<18} {'Avg Coverage':<14} {'Avg Uniqueness':<15}")
print("-" * 90)

for row in merged_data:
    print(f"{row['num_vars']:<6} {row['density']:<8.1f} {row['num_tests']:<8} "
          f"{row['avg_4d_clusters']:<14.2f} {row['avg_information_density']:<18.2f} "
          f"{row['avg_coverage_ratio']:<14.2%} {row['avg_uniqueness_ratio']:<15.2%}")

print(f"\n{'='*70}")
print(f"Output file: {output_csv}")
print(f"{'='*70}")

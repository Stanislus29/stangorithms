"""
Script to extract and calculate the average and mode number of clusters and average cluster size 
formed at each variable count, grouped by their respective densities.

Reads from cluster_density_analysis.csv and outputs results to cluster_averages.txt
"""

import pandas as pd
from pathlib import Path
from scipy import stats as scipy_stats

# Set up file paths
csv_file = Path(__file__).parent.parent / "outputs" / "decay" / "analysis" / "cluster_density_analysis.csv"
output_file = Path(__file__).parent.parent / "outputs" / "decay" / "analysis" / "cluster_averages.txt"

# Read the CSV file
df = pd.read_csv(csv_file)

# Group by num_vars and density, then calculate average and mode number of clusters
def calculate_mode(x):
    mode_result = scipy_stats.mode(x, keepdims=True)
    return mode_result.mode[0] if len(mode_result.mode) > 0 else x.iloc[0]

grouped = df.groupby(['num_vars', 'density']).agg({
    'num_3d_clusters': ['mean', calculate_mode, 'count'],
    'avg_cluster_size': ['mean', calculate_mode]
})
grouped.columns = ['avg_clusters', 'mode_clusters', 'num_tests', 'avg_cluster_size', 'mode_cluster_size']
grouped = grouped.reset_index()

# Open output file and write results
with open(output_file, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("AVERAGE AND MODE NUMBER OF CLUSTERS AND CLUSTER SIZE BY VARIABLE COUNT AND DENSITY\n")
    f.write("=" * 80 + "\n\n")
    
    # Group by variable count for organized output
    for num_vars in sorted(grouped['num_vars'].unique()):
        f.write(f"\n{num_vars}-Variable Functions:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Density':<10} {'Avg Clusters':<15} {'Mode Clusters':<15} {'Avg Size':<15} {'Mode Size':<15} {'Tests':<10}\n")
        f.write("-" * 80 + "\n")
        
        var_data = grouped[grouped['num_vars'] == num_vars]
        for _, row in var_data.iterrows():
            f.write(f"{row['density']:<10.1f} {row['avg_clusters']:<15.4f} {int(row['mode_clusters']):<15} "
                   f"{row['avg_cluster_size']:<15.4f} {row['mode_cluster_size']:<15.4f} {int(row['num_tests']):<10}\n")
    
    # Summary statistics
    f.write("\n\n" + "=" * 80 + "\n")
    f.write("SUMMARY STATISTICS\n")
    f.write("=" * 80 + "\n\n")
    
    # Overall statistics by variable count
    overall_by_vars = df.groupby('num_vars').agg({
        'num_3d_clusters': ['mean', calculate_mode, 'std', 'min', 'max', 'count'],
        'avg_cluster_size': ['mean', calculate_mode, 'std', 'min', 'max']
    })
    overall_by_vars.columns = ['avg_clusters', 'mode_clusters', 'std_clusters', 'min_clusters', 'max_clusters', 'total_tests',
                                'avg_cluster_size', 'mode_cluster_size', 'std_cluster_size', 'min_cluster_size', 'max_cluster_size']
    overall_by_vars = overall_by_vars.reset_index()
    
    f.write("Overall Statistics by Variable Count:\n")
    f.write("-" * 80 + "\n")
    for _, row in overall_by_vars.iterrows():
        f.write(f"{row['num_vars']}-Variable:\n")
        f.write(f"  Clusters:\n")
        f.write(f"    Average:      {row['avg_clusters']:.4f}\n")
        f.write(f"    Mode:         {int(row['mode_clusters'])}\n")
        f.write(f"    Std dev:      {row['std_clusters']:.4f}\n")
        f.write(f"    Min:          {int(row['min_clusters'])}\n")
        f.write(f"    Max:          {int(row['max_clusters'])}\n")
        f.write(f"  Cluster Size:\n")
        f.write(f"    Average:      {row['avg_cluster_size']:.4f}\n")
        f.write(f"    Mode:         {row['mode_cluster_size']:.4f}\n")
        f.write(f"    Std dev:      {row['std_cluster_size']:.4f}\n")
        f.write(f"    Min:          {row['min_cluster_size']:.4f}\n")
        f.write(f"    Max:          {row['max_cluster_size']:.4f}\n")
        f.write(f"  Total tests:    {int(row['total_tests'])}\n\n")
    
    # Overall statistics by density
    f.write("\nOverall Statistics by Density:\n")
    f.write("-" * 80 + "\n")
    overall_by_density = df.groupby('density').agg({
        'num_3d_clusters': ['mean', calculate_mode, 'std', 'min', 'max', 'count'],
        'avg_cluster_size': ['mean', calculate_mode, 'std', 'min', 'max']
    })
    overall_by_density.columns = ['avg_clusters', 'mode_clusters', 'std_clusters', 'min_clusters', 'max_clusters', 'total_tests',
                                   'avg_cluster_size', 'mode_cluster_size', 'std_cluster_size', 'min_cluster_size', 'max_cluster_size']
    overall_by_density = overall_by_density.reset_index()
    
    for _, row in overall_by_density.iterrows():
        f.write(f"Density {row['density']:.1f}:\n")
        f.write(f"  Clusters:\n")
        f.write(f"    Average:      {row['avg_clusters']:.4f}\n")
        f.write(f"    Mode:         {int(row['mode_clusters'])}\n")
        f.write(f"    Std dev:      {row['std_clusters']:.4f}\n")
        f.write(f"    Min:          {int(row['min_clusters'])}\n")
        f.write(f"    Max:          {int(row['max_clusters'])}\n")
        f.write(f"  Cluster Size:\n")
        f.write(f"    Average:      {row['avg_cluster_size']:.4f}\n")
        f.write(f"    Mode:         {row['mode_cluster_size']:.4f}\n")
        f.write(f"    Std dev:      {row['std_cluster_size']:.4f}\n")
        f.write(f"    Min:          {row['min_cluster_size']:.4f}\n")
        f.write(f"    Max:          {row['max_cluster_size']:.4f}\n")
        f.write(f"  Total tests:    {int(row['total_tests'])}\n\n")

print(f"Analysis complete! Results written to: {output_file}")
print(f"Total records analyzed: {len(df)}")
print(f"Variable counts found: {sorted(df['num_vars'].unique())}")
print(f"Density values found: {sorted(df['density'].unique())}")

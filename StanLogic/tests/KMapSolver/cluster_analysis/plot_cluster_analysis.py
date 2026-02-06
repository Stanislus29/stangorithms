import csv
import os
import matplotlib.pyplot as plt
import numpy as np

# Use script directory as base
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "..", "outputs", "cluster_formation", 
                        "cluster_analysis3d_results_20260205_035043.csv")
output_dir = os.path.join(script_dir, "..", "outputs", "cluster_formation")

def read_csv_data(csv_path):
    """Read CSV and organize data by density."""
    data_by_density = {}
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            density = float(row['density'])
            num_vars = int(row['num_vars'])
            avg_clusters = float(row['avg_3d_clusters'])
            min_clusters = float(row['min_3d_clusters'])
            max_clusters = float(row['max_3d_clusters'])
            avg_info_density = float(row['avg_information_density'])
            min_info_density = float(row['min_information_density'])
            max_info_density = float(row['max_information_density'])
            avg_coverage = float(row['avg_coverage_ratio']) * 100  # Convert to percentage
            avg_uniqueness = float(row['avg_uniqueness_ratio'])
            min_uniqueness = float(row['avg_min_uniqueness_ratio'])
            max_uniqueness = float(row['avg_max_uniqueness_ratio'])
            
            if density not in data_by_density:
                data_by_density[density] = {
                    'num_vars': [],
                    'clusters': [],
                    'min_clusters': [],
                    'max_clusters': [],
                    'info_density': [],
                    'min_info_density': [],
                    'max_info_density': [],
                    'coverage': [],
                    'uniqueness': [],
                    'min_uniqueness': [],
                    'max_uniqueness': []
                }
            
            data_by_density[density]['num_vars'].append(num_vars)
            data_by_density[density]['clusters'].append(avg_clusters)
            data_by_density[density]['min_clusters'].append(min_clusters)
            data_by_density[density]['max_clusters'].append(max_clusters)
            data_by_density[density]['info_density'].append(avg_info_density)
            data_by_density[density]['min_info_density'].append(min_info_density)
            data_by_density[density]['max_info_density'].append(max_info_density)
            data_by_density[density]['coverage'].append(avg_coverage)
            data_by_density[density]['uniqueness'].append(avg_uniqueness)
            data_by_density[density]['min_uniqueness'].append(min_uniqueness)
            data_by_density[density]['max_uniqueness'].append(max_uniqueness)
    
    return data_by_density

# Read data
print("Reading data from CSV...")
data = read_csv_data(csv_path)

# Create figure with 4 subplots (2x2)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('3D Cluster Formation Analysis', fontsize=16, fontweight='bold')

# Define colors for different densities
colors = {0.3: '#1f77b4', 0.5: '#ff7f0e', 0.7: '#2ca02c', 0.9: '#d62728'}
markers = {0.3: 'o', 0.5: 's', 0.7: '^', 0.9: 'D'}

# Plot 1: Cluster Growth vs n
ax1 = axes[0, 0]
for density in sorted(data.keys()):
    ax1.plot(data[density]['num_vars'], data[density]['clusters'], 
             marker=markers[density], color=colors[density], 
             label=f'Density {density}', linewidth=2, markersize=6)

ax1.set_xlabel('Number of Variables (n)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Average Number of 3D Clusters', fontsize=12, fontweight='bold')
ax1.set_title('Cluster Growth', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(loc='upper left', framealpha=0.9)
ax1.set_yscale('log')  # Log scale to better show exponential growth

# Plot 2: Information Density vs n (with min/max range)
ax2 = axes[0, 1]
for density in sorted(data.keys()):
    # Plot average line
    ax2.plot(data[density]['num_vars'], data[density]['info_density'], 
             marker=markers[density], color=colors[density], 
             label=f'Density {density}', linewidth=2, markersize=6)
    # Add shaded region for min/max range
    ax2.fill_between(data[density]['num_vars'], 
                     data[density]['min_info_density'],
                     data[density]['max_info_density'],
                     color=colors[density], alpha=0.15)

ax2.set_xlabel('Number of Variables (n)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Information Density\n(Minterms per Cluster)', fontsize=12, fontweight='bold')
ax2.set_title('Information Density (avg ± range)', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(loc='best', framealpha=0.9)

# Plot 3: Coverage Effectiveness vs n
ax3 = axes[1, 0]
for density in sorted(data.keys()):
    ax3.plot(data[density]['num_vars'], data[density]['coverage'], 
             marker=markers[density], color=colors[density], 
             label=f'Density {density}', linewidth=2, markersize=6)

ax3.set_xlabel('Number of Variables (n)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Coverage (%)', fontsize=12, fontweight='bold')
ax3.set_title('Coverage Effectiveness', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.legend(loc='best', framealpha=0.9)
ax3.set_ylim(bottom=0)

# Plot 4: Uniqueness Ratio vs n (with min/max range)
ax4 = axes[1, 1]
for density in sorted(data.keys()):
    # Plot average line
    ax4.plot(data[density]['num_vars'], data[density]['uniqueness'], 
             marker=markers[density], color=colors[density], 
             label=f'Density {density}', linewidth=2, markersize=6)
    # Add shaded region for min/max range
    ax4.fill_between(data[density]['num_vars'], 
                     data[density]['min_uniqueness'],
                     data[density]['max_uniqueness'],
                     color=colors[density], alpha=0.15)

ax4.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Perfect Uniqueness')
ax4.set_xlabel('Number of Variables (n)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Uniqueness Ratio\n(Unique Minterms / Avg Info Density)', fontsize=12, fontweight='bold')
ax4.set_title('Cluster Uniqueness (avg ± range)', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3, linestyle='--')
ax4.legend(loc='best', framealpha=0.9)
ax4.set_ylim(bottom=0)

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save the plot
output_path = os.path.join(output_dir, "cluster_analysis_3d_plots2.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {output_path}")

# Show the plot
plt.show()

print("\nAnalysis complete!")

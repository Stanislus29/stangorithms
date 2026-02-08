import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import glob

# Use script directory as base
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "..", "outputs", "cluster_formation")
merged_csv_path = os.path.join(output_dir, "cluster_analysis4d_results_20260205_035127.csv")

def find_collapse_files():
    """Find all 4D collapse density analysis CSV files."""
    pattern = os.path.join(output_dir, "4d_collapse_density_*_analysis.csv")
    files = glob.glob(pattern)
    return sorted(files)

def read_collapse_data(csv_path):
    """Read 4D coverage collapse analysis data."""
    n_values = []
    k_values = []
    raw_coverage = []
    capped_coverage = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_values.append(int(row['n']))
            k_values.append(int(row['k']))
            raw_coverage.append(float(row['raw_coverage']))
            capped_coverage.append(float(row['capped_coverage']))
    
    # Get parameters from first row
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        first_row = next(reader)
        params = {
            'C_18': float(first_row['C_18']),
            'geometric_ratio': float(first_row['geometric_ratio']),
            'I_sat': float(first_row['I_sat']),
            'density': float(first_row['density'])
        }
    
    return n_values, k_values, raw_coverage, capped_coverage, params

def read_merged_data_for_density(density):
    """Read information density, cluster data, and uniqueness ratio for specific density from merged file."""
    n_values = []
    info_density = []
    min_info_density = []
    max_info_density = []
    clusters = []
    uniqueness_ratio = []
    min_uniqueness = []
    max_uniqueness = []
    
    with open(merged_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if float(row['density']) == density:
                n_values.append(int(row['num_vars']))
                info_density.append(float(row['avg_information_density']))
                min_info_density.append(float(row['min_information_density']))
                max_info_density.append(float(row['max_information_density']))
                clusters.append(float(row['avg_4d_clusters']))
                uniqueness_ratio.append(float(row['avg_uniqueness_ratio']))
                min_uniqueness.append(float(row['avg_min_uniqueness_ratio']))
                max_uniqueness.append(float(row['avg_max_uniqueness_ratio']))
    
    return n_values, info_density, min_info_density, max_info_density, clusters, uniqueness_ratio, min_uniqueness, max_uniqueness

# Find all collapse analysis files
print("Finding collapse analysis files...")
collapse_files = find_collapse_files()
print(f"Found {len(collapse_files)} files")

if not collapse_files:
    print("Error: No collapse analysis files found!")
    exit(1)

# Read all data
all_density_data = {}
for file in collapse_files:
    density = float(file.split('density_')[1].split('_analysis')[0])
    n_values, k_values, raw_coverage, capped_coverage, params = read_collapse_data(file)
    n_merged, info_density, min_info_density, max_info_density, clusters, uniqueness_ratio, min_uniqueness, max_uniqueness = read_merged_data_for_density(density)
    
    all_density_data[density] = {
        'n_values': n_values,
        'k_values': k_values,
        'raw_coverage': raw_coverage,
        'capped_coverage': capped_coverage,
        'params': params,
        'n_merged': n_merged,
        'info_density': info_density,
        'min_info_density': min_info_density,
        'max_info_density': max_info_density,
        'clusters': clusters,
        'uniqueness_ratio': uniqueness_ratio,
        'min_uniqueness': min_uniqueness,
        'max_uniqueness': max_uniqueness
    }
    print(f"  Loaded data for density {density}")

# Create PDF
pdf_path = os.path.join(output_dir, f"4d_collapse_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
print(f"\nCreating PDF report: {pdf_path}")

with PdfPages(pdf_path) as pdf:
    
    # Page 1: Title and Model Parameters
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    title_text = f"""
{'='*70}
4D COVERAGE COLLAPSE ANALYSIS REPORT
{'='*70}

Date: {datetime.now().strftime('%B %d, %Y')}
Experiment: 4D Boolean Minimization Coverage Modeling

{'='*70}
MODEL DESCRIPTION
{'='*70}

This report analyzes the coverage collapse behavior of 4D cluster
formation in Boolean function minimization across different minterm
densities (0.3, 0.5, 0.7, 0.9).

The model predicts when adding more variables (n) stops improving
coverage and begins producing redundant clusters.

{'='*70}
MATHEMATICAL MODEL
{'='*70}

Coverage(n) = (C₁₈ × r^k × I_sat) / (density × 2^n) × 100

where:
  • C₁₈   : Number of 4D clusters at n=18
  • r     : Geometric growth ratio (cluster increase rate)
  • k     : Counter (k = n - 18, starts at 1 for n=19)
  • I_sat : Saturation information density (minterms/cluster)
  • n     : Number of variables

{'='*70}
PARAMETERS BY DENSITY
{'='*70}
"""
    
    # Add parameters for each density
    for density in sorted(all_density_data.keys()):
        params = all_density_data[density]['params']
        title_text += f"\nDensity {density}:\n"
        title_text += f"  C₁₈ = {params['C_18']:.0f} clusters\n"
        title_text += f"  r   = {params['geometric_ratio']:.4f}\n"
        title_text += f"  I_sat = {params['I_sat']:.2f} minterms/cluster\n"
    
    title_text += f"\n{'='*70}\n"
    title_text += "Analysis follows on subsequent pages, one per density.\n"
    title_text += f"{'='*70}"
    
    ax.text(0.5, 0.5, title_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3, pad=1))
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    print("  Created title page")
    
    # Create a page for each density
    for density in sorted(all_density_data.keys()):
        data = all_density_data[density]
        n_values = data['n_values']
        raw_coverage = data['raw_coverage']
        capped_coverage = data['capped_coverage']
        n_merged = data['n_merged']
        info_density = data['info_density']
        min_info_density = data['min_info_density']
        max_info_density = data['max_info_density']
        clusters = data['clusters']
        uniqueness_ratio = data['uniqueness_ratio']
        min_uniqueness = data['min_uniqueness']
        max_uniqueness = data['max_uniqueness']
        params = data['params']
        
        # Find collapse point
        collapse_idx = None
        for i, raw in enumerate(raw_coverage):
            if raw >= 100.0:
                collapse_idx = i
                break
        
        # Determine the range for Plot 1 (up to 10 values after collapse)
        if collapse_idx is not None:
            plot1_end_idx = min(collapse_idx + 11, len(n_values))  # +11 to include collapse point + 10 after
        else:
            plot1_end_idx = len(n_values)  # Use all data if no collapse
        
        # Create sliced data for Plot 1
        n_values_plot1 = n_values[:plot1_end_idx]
        raw_coverage_plot1 = raw_coverage[:plot1_end_idx]
        redundancy_plot1 = [max(0, raw - 100) for raw in raw_coverage_plot1]
        effective_coverage_plot1 = [min(raw, 100) for raw in raw_coverage_plot1]
        
        # Create figure with 4 plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'4D Cluster Analysis - Density {density}', fontsize=16, fontweight='bold')
        
        # Plot 1: Effective vs Redundant Coverage
        ax1 = axes[0, 0]
        
        ax1.fill_between(n_values_plot1, 0, effective_coverage_plot1, color='#2ca02c', alpha=0.6, label='Effective Coverage')
        ax1.fill_between(n_values_plot1, effective_coverage_plot1, [eff + red for eff, red in zip(effective_coverage_plot1, redundancy_plot1)], 
                         color='#d62728', alpha=0.6, label='Redundant Clusters')
        ax1.plot(n_values_plot1, raw_coverage_plot1, 'ko-', linewidth=2, markersize=5, label='Total Model Output')
        ax1.axhline(y=100, color='black', linestyle='--', linewidth=1.5, label='100% Limit')
        
        if collapse_idx is not None:
            ax1.axvline(x=n_values[collapse_idx], color='orange', linestyle=':', 
                        linewidth=2, label=f'Collapse n={n_values[collapse_idx]}')
        
        ax1.set_xlabel('Number of Variables (n)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Coverage (%)', fontsize=11, fontweight='bold')
        ax1.set_title('Effective vs Redundant Coverage', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(loc='upper left', framealpha=0.9, fontsize=9)
        
        # Plot 2: Information Density Saturation (with min/max range)
        ax2 = axes[0, 1]
        ax2.plot(n_merged, info_density, 'o-', color='#1f77b4', linewidth=2, markersize=7, label='Info Density')
        ax2.fill_between(n_merged, min_info_density, max_info_density, color='#1f77b4', alpha=0.15, label='Min-Max Range')
        ax2.axhline(y=params['I_sat'], color='red', linestyle='--', linewidth=2, label=f'I_sat = {params["I_sat"]:.2f}')
        
        ax2.set_xlabel('Number of Variables (n)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Information Density\n(Minterms per Cluster)', fontsize=11, fontweight='bold')
        ax2.set_title('Information Density (avg ± range)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend(loc='best', framealpha=0.9, fontsize=9)
        
        # Plot 3: Cluster Growth
        ax3 = axes[1, 0]
        ax3.plot(n_merged, clusters, 's-', color='#ff7f0e', linewidth=2, markersize=7, label='4D Clusters')
        ax3.set_xlabel('Number of Variables (n)', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Number of 4D Clusters', fontsize=11, fontweight='bold')
        ax3.set_title('Cluster Growth', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.legend(loc='upper left', framealpha=0.9, fontsize=9)
        ax3.set_yscale('log')  # Log scale for exponential growth
        
        # Plot 4: Uniqueness Ratio (with min/max range and projection)
        ax4 = axes[1, 1]
        ax4.plot(n_merged, uniqueness_ratio, '^-', color='#9467bd', linewidth=2, markersize=7, label='Uniqueness Ratio')
        ax4.fill_between(n_merged, min_uniqueness, max_uniqueness, color='#9467bd', alpha=0.15, label='Min-Max Range')
        
        # Calculate and plot projected uniqueness at collapse point
        if collapse_idx is not None:
            collapse_n = n_values[collapse_idx]
            
            # Calculate arithmetic progression
            if len(uniqueness_ratio) > 1:
                differences = [uniqueness_ratio[i] - uniqueness_ratio[i-1] for i in range(1, len(uniqueness_ratio))]
                avg_diff = sum(differences) / len(differences)
                
                # Project to collapse point
                if collapse_n > n_merged[-1]:
                    last_n = n_merged[-1]
                    last_uniqueness = uniqueness_ratio[-1]
                    steps = collapse_n - last_n
                    projected_uniqueness = last_uniqueness + steps * avg_diff
                    
                    # Plot projection line
                    projection_n = [last_n, collapse_n]
                    projection_u = [last_uniqueness, projected_uniqueness]
                    ax4.plot(projection_n, projection_u, '--', color='#9467bd', linewidth=2, alpha=0.6, label='Projected')
                    ax4.plot(collapse_n, projected_uniqueness, 'X', color='red', markersize=12, 
                            label=f'Projected at collapse: {projected_uniqueness:.3f}', zorder=5)
        
        ax4.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Perfect Uniqueness')
        ax4.set_xlabel('Number of Variables (n)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Uniqueness Ratio\n(Unique Minterms / Avg Info Density)', fontsize=11, fontweight='bold')
        ax4.set_title('Cluster Uniqueness (avg ± range)', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.legend(loc='best', framealpha=0.9, fontsize=8)
        ax4.set_ylim(bottom=0)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        print(f"  Created page for density {density}")

print(f"\n✓ PDF report created: {pdf_path}")

# Analyze uniqueness ratio arithmetic progression within each density
print("\n" + "="*70)
print("UNIQUENESS RATIO ARITHMETIC PROGRESSION ANALYSIS")
print("="*70)

for density in sorted(all_density_data.keys()):
    data = all_density_data[density]
    n_values = data['n_values']
    raw_coverage = data['raw_coverage']
    n_merged = data['n_merged']
    uniqueness_ratio = data['uniqueness_ratio']
    
    # Find collapse point
    collapse_n = None
    for i, raw in enumerate(raw_coverage):
        if raw >= 100.0:
            collapse_n = n_values[i]
            break
    
    print(f"\nDensity {density}:")
    print("-" * 70)
    
    if collapse_n is not None:
        print(f"Collapse point: n = {collapse_n}")
    else:
        print(f"No collapse detected in range (max n = {n_values[-1]})")
    
    # Calculate differences in uniqueness ratio between consecutive n values
    differences = []
    print(f"\nUniqueness Ratio progression:")
    for i in range(len(n_merged)):
        if i == 0:
            print(f"  n={n_merged[i]:2d}: Uniqueness = {uniqueness_ratio[i]:.6f}")
        else:
            diff = uniqueness_ratio[i] - uniqueness_ratio[i-1]
            differences.append(diff)
            print(f"  n={n_merged[i]:2d}: Uniqueness = {uniqueness_ratio[i]:.6f}, d = {diff:.6f}")
    
    if len(differences) >= 2:
        avg_diff = sum(differences) / len(differences)
        variance = sum((d - avg_diff)**2 for d in differences) / len(differences)
        std_dev = variance ** 0.5
        
        print(f"\nArithmetic progression statistics:")
        print(f"  Average common difference (d): {avg_diff:.6f}")
        print(f"  Standard deviation: {std_dev:.6f}")
        
        if std_dev < abs(avg_diff) * 0.1:  # Within 10% variation
            print(f"  → Approximately arithmetic progression ✓")
        else:
            print(f"  → Variable progression (not strictly arithmetic)")
        
        # Project uniqueness ratio at collapse point
        if collapse_n is not None and collapse_n > n_merged[-1]:
            # Use arithmetic progression formula: a_n = a_k + (n - k) * d
            last_n = n_merged[-1]
            last_uniqueness = uniqueness_ratio[-1]
            steps = collapse_n - last_n
            projected_uniqueness = last_uniqueness + steps * avg_diff
            
            print(f"\nProjection to collapse point (n={collapse_n}):")
            print(f"  From n={last_n}, Uniqueness = {last_uniqueness:.6f}")
            print(f"  Steps: {steps}, Common difference: {avg_diff:.6f}")
            print(f"  Projected Uniqueness at n={collapse_n}: {projected_uniqueness:.6f}")
        elif collapse_n is not None:
            # Collapse point is within available data
            if collapse_n in n_merged:
                idx = n_merged.index(collapse_n)
                actual_uniqueness = uniqueness_ratio[idx]
                print(f"\nActual Uniqueness at collapse (n={collapse_n}): {actual_uniqueness:.6f}")
            else:
                # Interpolate
                below_idx = None
                above_idx = None
                for i, n in enumerate(n_merged):
                    if n < collapse_n:
                        below_idx = i
                    elif n > collapse_n:
                        above_idx = i
                        break
                
                if below_idx is not None and above_idx is not None:
                    n1, n2 = n_merged[below_idx], n_merged[above_idx]
                    u1, u2 = uniqueness_ratio[below_idx], uniqueness_ratio[above_idx]
                    interpolated_uniqueness = u1 + (u2 - u1) * (collapse_n - n1) / (n2 - n1)
                    print(f"\nInterpolated Uniqueness at collapse (n={collapse_n}): {interpolated_uniqueness:.6f}")

print("\n" + "="*70)
print("Analysis complete!")

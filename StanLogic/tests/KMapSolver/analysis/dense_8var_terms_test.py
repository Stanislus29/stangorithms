"""
Dense 8-Variable 3D EPI Analysis with QM Merging
==================================================

Analyzes the number of 3D EPIs and their QM-merged result for dense
8-variable K-maps (70%+ density). This test examines how many patterns
combine to form the final expression assuming 2D clusters aren't needed.

This test generates dense 8-variable K-maps and examines:
- Number of 3D EPIs before merging
- Number of terms after QM merging of 3D EPIs
- Reduction ratio (EPIs → merged terms)
- Statistics: mean, median, standard deviation

Author: Somtochukwu Stanislus Emeka-Onwuneme
Date: January 2026
"""

import random
import csv
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

# Add parent directory to path to import stanlogic
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from stanlogic.BoolMinGeo import BoolMinGeo

# ============================================================================
# CONFIGURATION
# ============================================================================

# Random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Number of test cases
NUM_TEST_CASES = 500

# Density range for "dense" functions (90% fixed)
DENSITY_MIN = 0.9
DENSITY_MAX = 0.9

# Output directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, "..", "outputs", "obtain_terms_frequency")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

RESULTS_CSV = os.path.join(OUTPUTS_DIR, "dense_8var_3d_epi_results.csv")
STATS_TXT = os.path.join(OUTPUTS_DIR, "dense_8var_3d_epi_stats.txt")
HISTOGRAM_PNG = os.path.join(OUTPUTS_DIR, "dense_8var_3d_epi_histogram.png")

# ============================================================================
# TEST DATA GENERATION
# ============================================================================

def random_dense_output_values_8var():
    """
    Generate a dense random 8-variable output values list.
    Dense means high probability of 1s (70-90%).
    
    For 8 variables, we need 2^8 = 256 output values.
    
    Returns:
        List of 256 values (0 or 1)
    """
    size = 2**8  # 256 cells
    density = random.uniform(DENSITY_MIN, DENSITY_MAX)
    ones_count = round(size * density)
    
    # Create list with specified number of 1s
    output_values = [1] * ones_count + [0] * (size - ones_count)
    random.shuffle(output_values)
    return output_values


# ============================================================================
# K-MAP SOLVER ANALYSIS
# ============================================================================

def analyze_dense_kmaps_3d_epis():
    """
    Generate dense 8-variable K-maps and analyze 3D EPIs with QM merging.
    
    Returns:
        List of dictionaries with test results
    """
    results = []
    
    print(f"Generating and analyzing {NUM_TEST_CASES} dense 8-variable K-maps...")
    print("=" * 70)
    print(f"Density range: {DENSITY_MIN*100:.0f}% - {DENSITY_MAX*100:.0f}%")
    print("Note: Each test processes 256 cells (2^8 variables)")
    print("=" * 70)
    
    for test_num in range(1, NUM_TEST_CASES + 1):
        # Generate dense output values
        output_values = random_dense_output_values_8var()
        
        # Count ones
        num_ones = sum(1 for v in output_values if v == 1)
        density = num_ones / 256.0
        
        # Create solver
        solver = BoolMinGeo(8, output_values)
        
        # Suppress verbose output during 3D clustering
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        # Get 3D EPIs directly (this is what minimize_3d returns)
        # Extract EPIs from 3D clustering
        id_set = sorted(solver.kmaps.keys())
        β = {}
        for idx in id_set:
            result = solver._solve_single_kmap(idx, 'sop')
            β[idx] = result['terms_bits']
        
        # Get 3D EPIs (returns set of pattern strings)
        epis_3d = solver._minimize_with_3d_clustering(β, id_set)
        
        # Restore stdout
        sys.stdout = old_stdout
        
        # EPIs are already strings, convert set to list
        epi_patterns = list(epis_3d)
        num_3d_epis = len(epi_patterns)
        
        # Get all target minterms for QM optimization
        all_target_minterms = set()
        for i, val in enumerate(output_values):
            if val == 1:
                bits = format(i, f'08b')
                all_target_minterms.add(bits)
        
        # Perform QM optimization on the 3D EPI patterns (patterns with don't-cares)
        sys.stdout = io.StringIO()
        merged_patterns = solver._optimize_with_quine_mccluskey(set(epi_patterns), all_target_minterms)
        sys.stdout = old_stdout
        
        num_merged_terms = len(merged_patterns)
        reduction_ratio = num_merged_terms / num_3d_epis if num_3d_epis > 0 else 0
        
        result = {
            "test_num": test_num,
            "num_ones": num_ones,
            "density": density,
            "num_3d_epis": num_3d_epis,
            "num_merged_terms": num_merged_terms,
            "reduction_ratio": reduction_ratio,
            "epis_reduced": num_3d_epis - num_merged_terms
        }
        
        results.append(result)
        
        # Print progress
        if test_num % 50 == 0 or test_num == 1:
            print(f"Test {test_num:3d}: {num_ones:3d} ones ({density*100:.1f}%) -> {num_3d_epis:3d} EPIs -> {num_merged_terms:3d} merged ({reduction_ratio:.2%} of original)")
    
    print("=" * 70)
    return results

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def compute_statistics(results):
    """
    Compute statistical measures for 3D EPI counts and merged terms.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Dictionary with statistical measures
    """
    epi_counts = [r["num_3d_epis"] for r in results]
    merged_counts = [r["num_merged_terms"] for r in results]
    reduction_ratios = [r["reduction_ratio"] for r in results]
    densities = [r["density"] for r in results]
    reduced_counts = [r["epis_reduced"] for r in results]
    
    # Calculate frequency distribution
    epi_freq = defaultdict(int)
    for ec in epi_counts:
        epi_freq[ec] += 1
    
    merged_freq = defaultdict(int)
    for mc in merged_counts:
        merged_freq[mc] += 1
    
    # Calculate mode (most frequent value)
    mode_3d_epis = max(epi_freq.items(), key=lambda x: x[1])[0] if epi_freq else 0
    mode_merged = max(merged_freq.items(), key=lambda x: x[1])[0] if merged_freq else 0
    
    stats = {
        # 3D EPI stats
        "mean_3d_epis": np.mean(epi_counts),
        "median_3d_epis": np.median(epi_counts),
        "mode_3d_epis": mode_3d_epis,
        "std_3d_epis": np.std(epi_counts, ddof=1),
        "min_3d_epis": np.min(epi_counts),
        "max_3d_epis": np.max(epi_counts),
        
        # Merged term stats
        "mean_merged": np.mean(merged_counts),
        "median_merged": np.median(merged_counts),
        "mode_merged": mode_merged,
        "std_merged": np.std(merged_counts, ddof=1),
        "min_merged": np.min(merged_counts),
        "max_merged": np.max(merged_counts),
        
        # Reduction stats
        "mean_reduction_ratio": np.mean(reduction_ratios),
        "median_reduction_ratio": np.median(reduction_ratios),
        "mean_epis_reduced": np.mean(reduced_counts),
        
        # Density stats
        "mean_density": np.mean(densities),
        "median_density": np.median(densities),
        
        # Distributions
        "epi_distribution": dict(sorted(epi_freq.items())),
        "merged_distribution": dict(sorted(merged_freq.items())),
    }
    
    return stats

def print_statistics(stats):
    """Print statistical summary."""
    print("\nSTATISTICAL SUMMARY")
    print("=" * 70)
    print("3D EPIs (before QM merging):")
    print(f"  Mean:     {stats['mean_3d_epis']:.2f}")
    print(f"  Median:   {stats['median_3d_epis']:.1f}")
    print(f"  Mode:     {stats['mode_3d_epis']:.0f}")
    print(f"  Std Dev:  {stats['std_3d_epis']:.2f}")
    print(f"  Range:    [{stats['min_3d_epis']}, {stats['max_3d_epis']}]")
    
    print("\nMerged Terms (after QM merging):")
    print(f"  Mean:     {stats['mean_merged']:.2f}")
    print(f"  Median:   {stats['median_merged']:.1f}")
    print(f"  Mode:     {stats['mode_merged']:.0f}")
    print(f"  Std Dev:  {stats['std_merged']:.2f}")
    print(f"  Range:    [{stats['min_merged']}, {stats['max_merged']}]")
    
    print("\nReduction Analysis:")
    print(f"  Mean reduction ratio:  {stats['mean_reduction_ratio']:.2%}")
    print(f"  Median reduction ratio: {stats['median_reduction_ratio']:.2%}")
    print(f"  Mean EPIs reduced:     {stats['mean_epis_reduced']:.2f}")
    
    print("\nFunction Density:")
    print(f"  Mean density:   {stats['mean_density']:.2%}")
    print(f"  Median density: {stats['median_density']:.2%}")
    print("=" * 70)

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_histogram(results, stats):
    """Create histogram comparing 3D EPIs with merged terms."""
    epi_counts = [r["num_3d_epis"] for r in results]
    merged_counts = [r["num_merged_terms"] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histogram 1: 3D EPIs
    ax1.hist(epi_counts, bins=range(min(epi_counts), max(epi_counts) + 2),
             edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(stats['mean_3d_epis'], color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {stats["mean_3d_epis"]:.2f}')
    ax1.axvline(stats['median_3d_epis'], color='green', linestyle='--', 
                linewidth=2, label=f'Median: {stats["median_3d_epis"]:.1f}')
    ax1.set_xlabel('Number of 3D EPIs', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('3D EPIs (Before QM Merge)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # Histogram 2: Merged Terms
    ax2.hist(merged_counts, bins=range(min(merged_counts), max(merged_counts) + 2),
             edgecolor='black', alpha=0.7, color='coral')
    ax2.axvline(stats['mean_merged'], color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {stats["mean_merged"]:.2f}')
    ax2.axvline(stats['median_merged'], color='green', linestyle='--', 
                linewidth=2, label=f'Median: {stats["median_merged"]:.1f}')
    ax2.set_xlabel('Number of Merged Terms', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('After QM Merge', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    fig.suptitle(f'Dense 8-Variable Functions (N={len(results)}, {stats["mean_density"]:.1%} density)', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(HISTOGRAM_PNG, dpi=150)
    print(f"\nHistogram saved to: {HISTOGRAM_PNG}")
    plt.close()

# ============================================================================
# RESULTS EXPORT
# ============================================================================

def save_results_csv(results):
    """Save detailed results to CSV."""
    with open(RESULTS_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "test_num", "num_ones", "density", "num_3d_epis", "num_merged_terms",
            "reduction_ratio", "epis_reduced"
        ])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Results saved to: {RESULTS_CSV}")

def save_statistics_txt(stats, results):
    """Save statistical summary to text file."""
    with open(STATS_TXT, 'w') as f:
        f.write("Dense 8-Variable 3D EPI Analysis with QM Merging\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Configuration:\n")
        f.write(f"  Random Seed: {RANDOM_SEED}\n")
        f.write(f"  Number of Test Cases: {NUM_TEST_CASES}\n")
        f.write(f"  Function Type: Dense 8-variable ({DENSITY_MIN:.0%}-{DENSITY_MAX:.0%} density)\n")
        f.write(f"  K-Map Size: 256 cells (2^8)\n\n")
        
        f.write("3D EPIs (Before QM Merging):\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Mean:       {stats['mean_3d_epis']:.2f} EPIs\n")
        f.write(f"  Median:     {stats['median_3d_epis']:.1f} EPIs\n")
        f.write(f"  Mode:       {stats['mode_3d_epis']:.0f} EPIs\n")
        f.write(f"  Std Dev:    {stats['std_3d_epis']:.2f}\n")
        f.write(f"  Range:      [{stats['min_3d_epis']}, {stats['max_3d_epis']}]\n")
        f.write(f"  Distribution: {stats['epi_distribution']}\n\n")
        
        f.write("Merged Terms (After QM Merging):\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Mean:       {stats['mean_merged']:.2f} terms\n")
        f.write(f"  Median:     {stats['median_merged']:.1f} terms\n")
        f.write(f"  Mode:       {stats['mode_merged']:.0f} terms\n")
        f.write(f"  Std Dev:    {stats['std_merged']:.2f}\n")
        f.write(f"  Range:      [{stats['min_merged']}, {stats['max_merged']}]\n")
        f.write(f"  Distribution: {stats['merged_distribution']}\n\n")
        
        f.write("Reduction Analysis:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Mean reduction ratio:   {stats['mean_reduction_ratio']:.2%}\n")
        f.write(f"  Median reduction ratio: {stats['median_reduction_ratio']:.2%}\n")
        f.write(f"  Mean EPIs reduced:      {stats['mean_epis_reduced']:.2f}\n\n")
        
        f.write("Function Density:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Mean density:   {stats['mean_density']:.2%}\n")
        f.write(f"  Median density: {stats['median_density']:.2%}\n")
        f.write("=" * 70 + "\n")
        
        # Distribution of term counts
        f.write("\\nDistribution of 3D EPI Counts:\\n")
        f.write("-" * 70 + "\\n")
        for count in sorted(stats['epi_distribution'].keys()):
            f.write(f"  {count:3d} EPIs: {stats['epi_distribution'][count]:2d} cases\\n")
        
        f.write("\\nDistribution of Merged Term Counts:\\n")
        f.write("-" * 70 + "\\n")
        for count in sorted(stats['merged_distribution'].keys()):
            f.write(f"  {count:3d} terms: {stats['merged_distribution'][count]:2d} cases\n")
        
        f.write("\n" + "=" * 70 + "\n\n")
        
        # Add analysis notes
        f.write("Analysis Notes:\n")
        f.write("-" * 70 + "\n")
        f.write(f"1. 8-variable K-maps have 2^8 = 256 cells\n")
        f.write(f"2. Dense functions have {DENSITY_MIN:.0%}-{DENSITY_MAX:.0%} of cells as 1s\n")
        f.write(f"3. 3D EPIs are extracted using depth-based clustering without 2D optimization\n")
        f.write(f"4. QM merging combines 3D EPIs to produce final minimal terms\n")
        f.write(f"5. Reduction ratio shows compression achieved by QM merging\n")
    
    print(f"Statistics saved to: {STATS_TXT}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("DENSE 8-VARIABLE 3D EPI ANALYSIS WITH QM MERGING")
    print("=" * 70)
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Test Cases: {NUM_TEST_CASES}")
    print(f"K-Map Size: 256 cells (2^8)")
    print(f"Density Range: {DENSITY_MIN:.0%}-{DENSITY_MAX:.0%}")
    print("")
    
    # Generate and analyze K-maps
    results = analyze_dense_kmaps_3d_epis()
    
    # Compute statistics
    stats = compute_statistics(results)
    print_statistics(stats)
    
    # Create visualizations
    create_histogram(results, stats)
    
    # Save results
    save_results_csv(results)
    save_statistics_txt(stats, results)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nOutput files:")
    print(f"  - CSV: {RESULTS_CSV}")
    print(f"  - Stats: {STATS_TXT}")
    print(f"  - Histogram: {HISTOGRAM_PNG}")
    print("")

if __name__ == "__main__":
    main()

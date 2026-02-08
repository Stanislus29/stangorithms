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
from matplotlib.backends.backend_pdf import PdfPages

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

# Density configuration - test multiple densities
DENSITIES = [0.3, 0.5, 0.7, 0.9]  # Test at 30%, 50%, 70%, and 90% densities

# Output directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, "..", "outputs", "obtain_terms_frequency")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Unified output files
STATS_TXT = os.path.join(OUTPUTS_DIR, "dense_8var_3d_epi_unified_results.txt")
HISTOGRAM_PDF = os.path.join(OUTPUTS_DIR, "dense_8var_3d_epi_histograms.pdf")

# ============================================================================
# TEST DATA GENERATION
# ============================================================================

def random_dense_output_values_8var(density):
    """
    Generate a dense random 8-variable output values list.
    
    For 8 variables, we need 2^8 = 256 output values.
    
    Args:
        density: Probability of a cell being 1
    
    Returns:
        List of 256 values (0 or 1)
    """
    size = 2**8  # 256 cells
    ones_count = round(size * density)
    
    # Create list with specified number of 1s
    output_values = [1] * ones_count + [0] * (size - ones_count)
    random.shuffle(output_values)
    return output_values


# ============================================================================
# K-MAP SOLVER ANALYSIS
# ============================================================================

def analyze_dense_kmaps_3d_epis(density):
    """
    Generate dense 8-variable K-maps and analyze 3D EPIs with QM merging.
    
    Args:
        density: Probability of a cell being 1
                
    Returns:
        List of dictionaries with test results
    """
    results = []
    
    print(f"Generating and analyzing {NUM_TEST_CASES} 8-variable K-maps at {density*100:.0f}% density...")
    print("=" * 70)
    print("Note: Each test processes 256 cells (2^8 variables)")
    print("=" * 70)
    
    for test_num in range(1, NUM_TEST_CASES + 1):
        # Generate dense output values
        output_values = random_dense_output_values_8var(density)
        
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
        
        result = {
            "test_num": test_num,
            "num_ones": num_ones,
            "density": density,
            "num_3d_epis": num_3d_epis
        }
        
        results.append(result)
        
        # Print progress
        if test_num % 50 == 0 or test_num == 1:
            print(f"Test {test_num:3d}: {num_ones:3d} ones ({density*100:.1f}%) -> {num_3d_epis:3d} EPIs")
    
    print("=" * 70)
    return results

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def compute_statistics(results):
    """
    Compute statistical measures for 3D EPI counts.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Dictionary with statistical measures
    """
    epi_counts = [r["num_3d_epis"] for r in results]
    densities = [r["density"] for r in results]
    
    # Calculate frequency distribution
    epi_freq = defaultdict(int)
    for ec in epi_counts:
        epi_freq[ec] += 1
    
    # Calculate mode (most frequent value)
    mode_3d_epis = max(epi_freq.items(), key=lambda x: x[1])[0] if epi_freq else 0
    
    stats = {
        # 3D EPI stats
        "mean_3d_epis": np.mean(epi_counts),
        "median_3d_epis": np.median(epi_counts),
        "mode_3d_epis": mode_3d_epis,
        "std_3d_epis": np.std(epi_counts, ddof=1),
        "min_3d_epis": np.min(epi_counts),
        "max_3d_epis": np.max(epi_counts),
        
        # Density stats
        "mean_density": np.mean(densities),
        "median_density": np.median(densities),
        
        # Distributions
        "epi_distribution": dict(sorted(epi_freq.items())),
    }
    
    return stats

def print_statistics(stats):
    """Print statistical summary."""
    print("\nSTATISTICAL SUMMARY")
    print("=" * 70)
    print("3D EPIs:")
    print(f"  Mean:     {stats['mean_3d_epis']:.2f}")
    print(f"  Median:   {stats['median_3d_epis']:.1f}")
    print(f"  Mode:     {stats['mode_3d_epis']:.0f}")
    print(f"  Std Dev:  {stats['std_3d_epis']:.2f}")
    print(f"  Range:    [{stats['min_3d_epis']}, {stats['max_3d_epis']}]")
    
    print("\nFunction Density:")
    print(f"  Mean density:   {stats['mean_density']:.2%}")
    print(f"  Median density: {stats['median_density']:.2%}")
    print("=" * 70)

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_histograms_pdf(all_results, all_stats):
    """Create PDF with histograms for each density."""
    with PdfPages(HISTOGRAM_PDF) as pdf:
        for density in DENSITIES:
            results = all_results[density]
            stats = all_stats[density]
            
            epi_counts = [r["num_3d_epis"] for r in results]
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            # Histogram: 3D EPIs
            ax.hist(epi_counts, bins=range(min(epi_counts), max(epi_counts) + 2),
                    edgecolor='black', alpha=0.7, color='steelblue')
            ax.axvline(stats['mean_3d_epis'], color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {stats["mean_3d_epis"]:.2f}')
            ax.axvline(stats['median_3d_epis'], color='green', linestyle='--', 
                       linewidth=2, label=f'Median: {stats["median_3d_epis"]:.1f}')
            ax.set_xlabel('Number of 3D EPIs', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title(f'8-Variable Functions ({density*100:.0f}% density, N={len(results)})', 
                        fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            pdf.savefig(fig, dpi=150)
            plt.close()
    
    print(f"\nHistograms saved to: {HISTOGRAM_PDF}")

# ============================================================================
# RESULTS EXPORT
# ============================================================================

def save_unified_statistics_txt(all_stats, all_results):
    """Save unified statistical summary to text file for all densities."""
    with open(STATS_TXT, 'w') as f:
        f.write("="*80 + "\n")
        f.write("  DENSE 8-VARIABLE 3D EPI ANALYSIS - UNIFIED RESULTS\n")
        f.write("="*80 + "\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  Random Seed: {RANDOM_SEED}\n")
        f.write(f"  Test Cases per Density: {NUM_TEST_CASES}\n")
        f.write(f"  Tested Densities: {', '.join([f'{d*100:.0f}%' for d in DENSITIES])}\n")
        f.write(f"  K-Map Size: 256 cells (2^8)\n")
        f.write(f"  Total Tests: {NUM_TEST_CASES * len(DENSITIES)}\n\n")
        f.write("="*80 + "\n\n")
        
        # Results for each density
        for density in DENSITIES:
            stats = all_stats[density]
            results = all_results[density]
            
            f.write(f"\n{'='*80}\n")
            f.write(f"  DENSITY: {density*100:.0f}% ({density:.1f} probability of 1s)\n")
            f.write(f"{'='*80}\n\n")
            
            f.write("3D EPIs:\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Mean:       {stats['mean_3d_epis']:8.2f} EPIs\n")
            f.write(f"  Median:     {stats['median_3d_epis']:8.1f} EPIs\n")
            f.write(f"  Mode:       {stats['mode_3d_epis']:8.0f} EPIs\n")
            f.write(f"  Std Dev:    {stats['std_3d_epis']:8.2f}\n")
            f.write(f"  Range:      [{stats['min_3d_epis']}, {stats['max_3d_epis']}]\n\n")
            
            f.write("Function Density:\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Mean density:   {stats['mean_density']:8.2%}\n")
            f.write(f"  Median density: {stats['median_density']:8.2%}\n\n")
        
        # Summary comparison table
        f.write("\n" + "="*80 + "\n")
        f.write("  DENSITY COMPARISON SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"{'Density':>10} | {'Mean EPIs':>11} | {'Median EPIs':>13} | ")
        f.write(f"{'Std Dev':>10} | {'Mean Density':>13}\n")
        f.write("-" * 80 + "\n")
        
        for density in DENSITIES:
            stats = all_stats[density]
            f.write(f"{density*100:>9.0f}% | {stats['mean_3d_epis']:>11.2f} | ")
            f.write(f"{stats['median_3d_epis']:>13.2f} | ")
            f.write(f"{stats['std_3d_epis']:>9.2f} | ")
            f.write(f"{stats['mean_density']:>13.2%}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"Unified statistics saved to: {STATS_TXT}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("DENSE 8-VARIABLE 3D EPI ANALYSIS - MULTIPLE DENSITIES")
    print("=" * 80)
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Test Cases per Density: {NUM_TEST_CASES}")
    print(f"K-Map Size: 256 cells (2^8)")
    print(f"Densities to Test: {', '.join([f'{d*100:.0f}%' for d in DENSITIES])}")
    print("")
    
    # Store results for all densities
    all_results = {}
    all_stats = {}
    
    # Analyze for each density
    for density in DENSITIES:
        print("\n" + "=" * 80)
        print(f"ANALYZING 8-VARIABLE K-MAPS AT {density*100:.0f}% DENSITY")
        print("=" * 80)
        
        results = analyze_dense_kmaps_3d_epis(density)
        stats = compute_statistics(results)
        
        all_results[density] = results
        all_stats[density] = stats
        
        print_statistics(stats)
    
    # Create unified visualizations
    print("\n" + "=" * 80)
    print("Creating unified outputs...")
    print("=" * 80)
    create_histograms_pdf(all_results, all_stats)
    
    # Save unified results
    save_unified_statistics_txt(all_stats, all_results)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nOutput files:")
    print(f"  - Unified Stats: {STATS_TXT}")
    print(f"  - Histograms PDF: {HISTOGRAM_PDF}")
    print("")

if __name__ == "__main__":
    main()

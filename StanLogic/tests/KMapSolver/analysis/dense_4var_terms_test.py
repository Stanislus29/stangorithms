"""
Dense 4-Variable K-Map Terms Analysis
======================================

Analyzes the number of terms produced by the K-Map solver for dense
4-variable K-maps. Dense maps have 90% probability of being 1 and 10% of being 0.

This test examines:
- Number of terms in minimized SOP expression for dense functions
- Statistics: mean, median, standard deviation, mode
- Distribution visualization
- Analysis of 4-variable results with high density

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
from stanlogic import BoolMin2D
from tabulate import tabulate
from matplotlib.backends.backend_pdf import PdfPages

# ============================================================================
# CONFIGURATION
# ============================================================================

# Random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Number of test cases - testing only 4 variables with varying densities
NUM_TEST_CASES_4VAR = 200  # 4-var has 16 cells

# Density configuration - test multiple densities
DENSITIES = [0.3, 0.5, 0.7, 0.9]  # Test at 30%, 50%, 70%, and 90% densities

# Output directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, "..", "outputs", "obtain_terms_frequency")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Unified output files
STATS_TXT = os.path.join(OUTPUTS_DIR, "dense_4var_terms_unified_results.txt")
HISTOGRAM_PDF = os.path.join(OUTPUTS_DIR, "dense_4var_terms_histograms.pdf")

# ============================================================================
# TEST DATA GENERATION
# ============================================================================

def random_dense_kmap_4var(density=0.90):
    """
    Generate a dense random K-map for 4 variables.
    Dense means high probability of 1s (default 90%), low probability of 0s (10%).
    
    Structure: 4x4 K-map (4 rows, 4 columns)
    
    Args:
        density: Probability of a cell being 1 (default 0.90)
        
    Returns:
        K-map as 2D list [rows][cols]
    """
    # 4x4 K-map with weighted random selection
    kmap = []
    for _ in range(4):
        row = []
        for _ in range(4):
            # Generate 1 with probability = density
            value = 1 if random.random() < density else 0
            row.append(value)
        kmap.append(row)
    return kmap

def kmap_to_minterms(kmap):
    """
    Convert 4-variable K-map to minterms list.
    
    Uses Vranseic convention: bits = col_labels[c] + row_labels[r]
    This matches kmapsolver.py _cell_to_term() method.
    
    Args:
        kmap: 2D list representing the 4x4 K-map [rows][cols]
        
    Returns:
        List of minterm indices
    """
    minterms = []
    
    # 4x4 K-map (both rows and columns use Gray code)
    row_labels = ["00", "01", "11", "10"]  # Gray code
    col_labels = ["00", "01", "11", "10"]  # Gray code
    for r in range(4):
        for c in range(4):
            if kmap[r][c] == 1:
                bits = col_labels[c] + row_labels[r]  # Vranseic: col + row
                idx = int(bits, 2)
                minterms.append(idx)
    
    return sorted(minterms)

def count_terms_in_expression(expr_str):
    """
    Count the number of terms in a SOP expression.
    A term is a product (AND) of literals, separated by + (OR).
    
    Args:
        expr_str: Boolean expression string (e.g., "AB + CD + E")
        
    Returns:
        Number of terms
    """
    if not expr_str or expr_str.strip() == "" or expr_str.strip() == "0":
        return 0
    
    if expr_str.strip() == "1":
        return 1
    
    # Split by + to get terms
    terms = expr_str.split('+')
    return len(terms)

# ============================================================================
# K-MAP SOLVER ANALYSIS
# ============================================================================

def analyze_dense_kmaps_4var(num_tests, density=0.90):
    """
    Generate dense 4-variable K-maps and analyze term counts.
    
    Args:
        num_tests: Number of test cases to run
        density: Probability of 1s (default 0.90)
        
    Returns:
        List of dictionaries with test results
    """
    results = []
    
    print(f"\nGenerating and analyzing {num_tests} dense 4-variable K-maps ({density*100:.0f}% density)...")
    print("=" * 70)
    
    for test_num in range(1, num_tests + 1):
        # Generate dense K-map
        kmap = random_dense_kmap_4var(density)
        
        # Convert to minterms
        minterms = kmap_to_minterms(kmap)
        
        # Solve using KMapSolver
        solver = BoolMin2D(kmap)
        terms_list, sop_expr = solver.minimize()  # Returns (terms_list, expression_string)
        
        # Count terms
        num_terms = count_terms_in_expression(sop_expr)
        
        # Count ones in K-map
        num_ones = sum(sum(row) for row in kmap)
        
        result = {
            "test_num": test_num,
            "num_vars": 4,
            "num_minterms": len(minterms),
            "num_ones": num_ones,
            "num_terms": num_terms,
            "sop_expression": sop_expr,
            "minterms": str(minterms)
        }
        
        results.append(result)
        
        # Print progress every 20 tests
        if test_num % 20 == 0 or test_num == num_tests:
            print(f"Test {test_num:3d}: {num_ones:2d} ones -> {num_terms:2d} terms | {sop_expr[:40]}")
    
    print("=" * 70)
    return results

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def compute_statistics(results, density=0.90):
    """
    Compute statistical measures for term counts.
    
    Args:
        results: List of result dictionaries
        density: Target density of 1s
        
    Returns:
        Dictionary with statistical measures
    """
    term_counts = [r["num_terms"] for r in results]
    ones_counts = [r["num_ones"] for r in results]
    
    # Calculate frequency distribution for mode
    term_freq = defaultdict(int)
    for tc in term_counts:
        term_freq[tc] += 1
    
    # Find mode (most frequent term count)
    max_freq = max(term_freq.values())
    mode_terms = [term for term, freq in term_freq.items() if freq == max_freq]
    
    num_vars = 4
    expected_ones = int((2**num_vars) * density)  # Expected for dense functions
    
    stats = {
        "num_vars": num_vars,
        "density": density,
        "mean_terms": np.mean(term_counts),
        "median_terms": np.median(term_counts),
        "std_terms": np.std(term_counts, ddof=1),
        "min_terms": np.min(term_counts),
        "max_terms": np.max(term_counts),
        "mode_terms": mode_terms,  # List of mode(s)
        "mode_frequency": max_freq,  # How many times mode occurred
        "term_distribution": dict(sorted(term_freq.items())),  # Full distribution
        "mean_ones": np.mean(ones_counts),
        "median_ones": np.median(ones_counts),
        "expected_ones": expected_ones,
    }
    
    return stats

def print_statistics(stats):
    """Print statistical summary."""
    print(f"\n{stats['num_vars']}-VARIABLE DENSE ({stats['density']*100:.0f}% DENSITY) STATISTICAL SUMMARY")
    print("=" * 70)
    print(f"Mean number of terms:     {stats['mean_terms']:.2f}")
    print(f"Median number of terms:   {stats['median_terms']:.1f}")
    print(f"Std deviation of terms:   {stats['std_terms']:.2f}")
    print(f"Min number of terms:      {stats['min_terms']}")
    print(f"Max number of terms:      {stats['max_terms']}")
    
    # Display mode (most frequent term count)
    if len(stats['mode_terms']) == 1:
        print(f"Mode (most frequent):     {stats['mode_terms'][0]} terms (occurred {stats['mode_frequency']} times)")
    else:
        mode_str = ", ".join(str(m) for m in sorted(stats['mode_terms']))
        print(f"Mode (most frequent):     {mode_str} terms (each occurred {stats['mode_frequency']} times)")
    
    print(f"\nMean number of ones:      {stats['mean_ones']:.2f}")
    print(f"Median number of ones:    {stats['median_ones']:.1f}")
    print(f"Expected ones (dense):    {stats['expected_ones']}")
    
    # Display frequency distribution
    print("\nTerm Count Distribution:")
    for terms, count in stats['term_distribution'].items():
        bar = "█" * min(count, 50)  # Cap bar length at 50
        print(f"  {terms} terms: {count:2d} cases {bar}")
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
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            # Histogram for current density
            term_counts = [r["num_terms"] for r in results]
            ax.hist(term_counts, bins=range(min(term_counts), max(term_counts) + 2),
                     edgecolor='black', alpha=0.7, color='darkgreen')
            ax.axvline(stats['mean_terms'], color='red', linestyle='--', 
                        linewidth=2, label=f'Mean: {stats["mean_terms"]:.2f}')
            ax.axvline(stats['median_terms'], color='blue', linestyle='--', 
                        linewidth=2, label=f'Median: {stats["median_terms"]:.1f}')
            ax.set_xlabel('Number of Terms', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title(f'4-Variable K-Maps ({density*100:.0f}% Density, 16 cells)', 
                         fontsize=13, fontweight='bold')
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
        f.write("  DENSE 4-VARIABLE K-MAP TERMS ANALYSIS - UNIFIED RESULTS\n")
        f.write("="*80 + "\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  Random Seed: {RANDOM_SEED}\n")
        f.write(f"  Test Cases per Density: {NUM_TEST_CASES_4VAR}\n")
        f.write(f"  Tested Densities: {', '.join([f'{d*100:.0f}%' for d in DENSITIES])}\n")
        f.write(f"  Total Tests: {NUM_TEST_CASES_4VAR * len(DENSITIES)}\n\n")
        f.write("="*80 + "\n\n")
        
        # Results for each density
        for density in DENSITIES:
            stats = all_stats[density]
            results = all_results[density]
            
            f.write(f"\n{'='*80}\n")
            f.write(f"  DENSITY: {density*100:.0f}% ({density:.1f} probability of 1s)\n")
            f.write(f"{'='*80}\n\n")
            
            f.write("Statistical Summary:\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Mean number of terms:     {stats['mean_terms']:8.2f}\n")
            f.write(f"  Median number of terms:   {stats['median_terms']:8.1f}\n")
            f.write(f"  Std deviation of terms:   {stats['std_terms']:8.2f}\n")
            f.write(f"  Min number of terms:      {stats['min_terms']:8d}\n")
            f.write(f"  Max number of terms:      {stats['max_terms']:8d}\n")
            
            if len(stats['mode_terms']) == 1:
                f.write(f"  Mode (most frequent):     {stats['mode_terms'][0]:8d} terms ")
                f.write(f"(occurred {stats['mode_frequency']} times)\n")
            else:
                mode_str = ", ".join(str(m) for m in sorted(stats['mode_terms']))
                f.write(f"  Mode (most frequent):     {mode_str} terms ")
                f.write(f"(each occurred {stats['mode_frequency']} times)\n")
            
            f.write(f"\n  Mean number of ones:      {stats['mean_ones']:8.2f}\n")
            f.write(f"  Median number of ones:    {stats['median_ones']:8.1f}\n")
            f.write(f"  Expected ones:            {stats['expected_ones']:8d}\n")
            f.write("\n")
            
            # Distribution table
            f.write("Term Count Distribution:\n")
            f.write("-" * 80 + "\n")
            for terms in sorted(stats['term_distribution'].keys()):
                count = stats['term_distribution'][terms]
                bar = "█" * min(count, 40)  # Visual bar
                f.write(f"  {terms:3d} terms: {count:4d} cases  {bar}\n")
            
            f.write("\n")
        
        # Summary comparison table
        f.write("\n" + "="*80 + "\n")
        f.write("  DENSITY COMPARISON SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"{'Density':>10} | {'Mean Terms':>12} | {'Median':>9} | {'Std Dev':>9} | ")
        f.write(f"{'Min':>5} | {'Max':>5}\n")
        f.write("-" * 80 + "\n")
        
        for density in DENSITIES:
            stats = all_stats[density]
            f.write(f"{density*100:>9.0f}% | {stats['mean_terms']:>12.2f} | ")
            f.write(f"{stats['median_terms']:>9.1f} | {stats['std_terms']:>9.2f} | ")
            f.write(f"{stats['min_terms']:>5d} | {stats['max_terms']:>5d}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"Unified statistics saved to: {STATS_TXT}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("DENSE 4-VARIABLE K-MAP TERMS ANALYSIS - MULTIPLE DENSITIES")
    print("=" * 80)
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Test Cases per Density: {NUM_TEST_CASES_4VAR}")
    print(f"Densities to Test: {', '.join([f'{d*100:.0f}%' for d in DENSITIES])}")
    print("")
    
    # Store results for all densities
    all_results = {}
    all_stats = {}
    
    # Analyze for each density
    for density in DENSITIES:
        print("\n" + "=" * 80)
        print(f"ANALYZING 4-VARIABLE K-MAPS AT {density*100:.0f}% DENSITY")
        print("=" * 80)
        
        results = analyze_dense_kmaps_4var(NUM_TEST_CASES_4VAR, density)
        stats = compute_statistics(results, density)
        
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
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nOutput files:")
    print(f"  - Unified Stats: {STATS_TXT}")
    print(f"  - Histograms PDF: {HISTOGRAM_PDF}")
    print("")

if __name__ == "__main__":
    main()

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

# ============================================================================
# CONFIGURATION
# ============================================================================

# Random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Number of test cases - testing only 4 variables with high density
NUM_TEST_CASES_4VAR = 200  # 4-var has 16 cells

# Density configuration
DENSITY = 0.90  # 90% probability of 1s, 10% probability of 0s

# Output directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, "..", "outputs", "obtain_terms_frequency")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

RESULTS_CSV_4VAR = os.path.join(OUTPUTS_DIR, "dense_90pct_4var_terms_results.csv")
STATS_TXT = os.path.join(OUTPUTS_DIR, "dense_90pct_4var_terms_stats.txt")
HISTOGRAM_PNG = os.path.join(OUTPUTS_DIR, "dense_90pct_4var_terms_histogram.png")

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

def create_histogram(results_4var, stats_4var):
    """Create histogram for dense 4-variable results."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # 4-variable histogram
    term_counts_4 = [r["num_terms"] for r in results_4var]
    ax.hist(term_counts_4, bins=range(min(term_counts_4), max(term_counts_4) + 2),
             edgecolor='black', alpha=0.7, color='darkgreen')
    ax.axvline(stats_4var['mean_terms'], color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {stats_4var["mean_terms"]:.2f}')
    ax.axvline(stats_4var['median_terms'], color='blue', linestyle='--', 
                linewidth=2, label=f'Median: {stats_4var["median_terms"]:.1f}')
    ax.set_xlabel('Number of Terms', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Dense 4-Variable K-Maps ({stats_4var["density"]*100:.0f}% Density, 16 cells)', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(HISTOGRAM_PNG, dpi=150)
    print(f"\nHistogram saved to: {HISTOGRAM_PNG}")
    plt.close()

# ============================================================================
# RESULTS EXPORT
# ============================================================================

def save_results_csv(results, filepath):
    """Save detailed results to CSV."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "test_num", "num_vars", "num_minterms", "num_ones", "num_terms", 
            "sop_expression", "minterms"
        ])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Results saved to: {filepath}")

def save_statistics_txt(stats_4var, results_4var):
    """Save statistical summary to text file."""
    with open(STATS_TXT, 'w') as f:
        f.write("Dense 4-Variable K-Map Terms Analysis\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Configuration:\n")
        f.write(f"  Random Seed: {RANDOM_SEED}\n")
        f.write(f"  4-Variable Test Cases: {NUM_TEST_CASES_4VAR}\n")
        f.write(f"  K-Map Type: Dense (p={DENSITY:.2f} for 1s)\n")
        f.write(f"  Density: {DENSITY*100:.0f}%\n\n")
        
        # 4-variable statistics
        f.write("4-VARIABLE DENSE STATISTICAL SUMMARY:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Mean number of terms:     {stats_4var['mean_terms']:.2f}\n")
        f.write(f"Median number of terms:   {stats_4var['median_terms']:.1f}\n")
        f.write(f"Std deviation of terms:   {stats_4var['std_terms']:.2f}\n")
        f.write(f"Min number of terms:      {stats_4var['min_terms']}\n")
        f.write(f"Max number of terms:      {stats_4var['max_terms']}\n")
        
        if len(stats_4var['mode_terms']) == 1:
            f.write(f"Mode (most frequent):     {stats_4var['mode_terms'][0]} terms (occurred {stats_4var['mode_frequency']} times)\n")
        else:
            mode_str = ", ".join(str(m) for m in sorted(stats_4var['mode_terms']))
            f.write(f"Mode (most frequent):     {mode_str} terms (each occurred {stats_4var['mode_frequency']} times)\n")
        
        f.write(f"\nMean number of ones:      {stats_4var['mean_ones']:.2f}\n")
        f.write(f"Median number of ones:    {stats_4var['median_ones']:.1f}\n")
        f.write(f"Expected ones (dense):    {stats_4var['expected_ones']}\n")
        f.write("=" * 70 + "\n\n")
        
        # Distribution table
        f.write("4-VARIABLE Term Count Distribution:\n")
        f.write("-" * 70 + "\n")
        for terms in sorted(stats_4var['term_distribution'].keys()):
            f.write(f"  {terms} terms: {stats_4var['term_distribution'][terms]} cases\n")
    
    print(f"Statistics saved to: {STATS_TXT}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("DENSE 4-VARIABLE K-MAP TERMS ANALYSIS")
    print("=" * 70)
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"4-Variable Test Cases: {NUM_TEST_CASES_4VAR}")
    print(f"Density: {DENSITY*100:.0f}% (probability of 1s)")
    print("")
    
    # Analyze 4-variable dense K-maps
    print("\n" + "=" * 70)
    print(f"ANALYZING DENSE 4-VARIABLE K-MAPS ({DENSITY*100:.0f}% DENSITY)")
    print("=" * 70)
    results_4var = analyze_dense_kmaps_4var(NUM_TEST_CASES_4VAR, DENSITY)
    stats_4var = compute_statistics(results_4var, DENSITY)
    print_statistics(stats_4var)
    
    # Create visualizations
    create_histogram(results_4var, stats_4var)
    
    # Save results
    save_results_csv(results_4var, RESULTS_CSV_4VAR)
    save_statistics_txt(stats_4var, results_4var)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nOutput files:")
    print(f"  - 4-var CSV: {RESULTS_CSV_4VAR}")
    print(f"  - Stats: {STATS_TXT}")
    print(f"  - Histogram: {HISTOGRAM_PNG}")
    print("")

if __name__ == "__main__":
    main()

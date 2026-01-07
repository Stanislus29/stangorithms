"""
Balanced 2, 3 & 4-Variable K-Map Terms Analysis
================================================

Analyzes the number of terms produced by the K-Map solver for balanced
2-variable, 3-variable, and 4-variable K-maps. Balanced maps have equal probability (50%) of being 1 or 0.

This test examines:
- Number of terms in minimized SOP expression
- Statistics: mean, median, standard deviation, mode
- Distribution visualization
- Comparison between 2-var, 3-var, and 4-var results

Author: Somtochukwu Stanislus Emeka-Onwuneme
Date: December 2025
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

# Number of test cases per variable count
NUM_TEST_CASES_2VAR = 50   # 2-var has 4 cells
NUM_TEST_CASES_3VAR = 100  # 3-var has 8 cells
NUM_TEST_CASES_4VAR = 200  # 4-var has 16 cells

# Output directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, "..", "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

RESULTS_CSV_2VAR = os.path.join(OUTPUTS_DIR, "balanced_2var_terms_results.csv")
RESULTS_CSV_3VAR = os.path.join(OUTPUTS_DIR, "balanced_3var_terms_results.csv")
RESULTS_CSV_4VAR = os.path.join(OUTPUTS_DIR, "balanced_4var_terms_results.csv")
STATS_TXT = os.path.join(OUTPUTS_DIR, "balanced_2_3_4var_terms_stats.txt")
HISTOGRAM_PNG = os.path.join(OUTPUTS_DIR, "balanced_2_3_4var_terms_histogram.png")

# ============================================================================
# TEST DATA GENERATION
# ============================================================================

def random_balanced_kmap(num_vars):
    """
    Generate a balanced random K-map for 2, 3, or 4 variables.
    Balanced means p(1) = 0.5, p(0) = 0.5, no don't cares.
    
    Structure follows kmapsolver.py:
    - 2-var: 2x2 (rows x cols)
    - 3-var: 2x4 (2 rows, 4 columns)
    - 4-var: 4x4 (4 rows, 4 columns)
    
    Args:
        num_vars: 2, 3, or 4
        
    Returns:
        K-map as 2D list [rows][cols]
    """
    if num_vars == 2:
        # 2x2 K-map
        return [[random.choice([0, 1]) for _ in range(2)] for _ in range(2)]
    elif num_vars == 3:
        # 2x4 K-map (2 rows, 4 columns)
        return [[random.choice([0, 1]) for _ in range(4)] for _ in range(2)]
    elif num_vars == 4:
        # 4x4 K-map (4 rows, 4 columns)
        return [[random.choice([0, 1]) for _ in range(4)] for _ in range(4)]
    else:
        raise ValueError("Only 2, 3, or 4 variables supported")

def kmap_to_minterms(kmap, num_vars):
    """
    Convert K-map to minterms list.
    
    Uses Vranseic convention: bits = col_labels[c] + row_labels[r]
    This matches kmapsolver.py _cell_to_term() method.
    
    Args:
        kmap: 2D list representing the K-map [rows][cols]
        num_vars: Number of variables (2, 3, or 4)
        
    Returns:
        List of minterm indices
    """
    minterms = []
    
    if num_vars == 2:
        # 2x2 K-map
        row_labels = ["0", "1"]
        col_labels = ["0", "1"]
        for r in range(2):
            for c in range(2):
                if kmap[r][c] == 1:
                    bits = col_labels[c] + row_labels[r]  # Vranseic: col + row
                    idx = int(bits, 2)
                    minterms.append(idx)
    elif num_vars == 3:
        # 2x4 K-map (2 rows, 4 columns with Gray code)
        row_labels = ["0", "1"]
        col_labels = ["00", "01", "11", "10"]  # Gray code
        for r in range(2):
            for c in range(4):
                if kmap[r][c] == 1:
                    bits = col_labels[c] + row_labels[r]  # Vranseic: col + row
                    idx = int(bits, 2)
                    minterms.append(idx)
    elif num_vars == 4:
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

def analyze_balanced_kmaps(num_vars, num_tests):
    """
    Generate balanced K-maps and analyze term counts.
    
    Args:
        num_vars: Number of variables (2, 3, or 4)
        num_tests: Number of test cases to run
        
    Returns:
        List of dictionaries with test results
    """
    results = []
    
    print(f"\nGenerating and analyzing {num_tests} balanced {num_vars}-variable K-maps...")
    print("=" * 70)
    
    for test_num in range(1, num_tests + 1):
        # Generate balanced K-map
        kmap = random_balanced_kmap(num_vars)
        
        # Convert to minterms
        minterms = kmap_to_minterms(kmap, num_vars)
        
        # Solve using KMapSolver
        solver = BoolMin2D(kmap)
        terms_list, sop_expr = solver.minimize()  # Returns (terms_list, expression_string)
        
        # Count terms
        num_terms = count_terms_in_expression(sop_expr)
        
        # Count ones in K-map
        num_ones = sum(sum(row) for row in kmap)
        
        result = {
            "test_num": test_num,
            "num_vars": num_vars,
            "num_minterms": len(minterms),
            "num_ones": num_ones,
            "num_terms": num_terms,
            "sop_expression": sop_expr,
            "minterms": str(minterms)
        }
        
        results.append(result)
        
        # Print progress (every 10 tests for 2-var, every 20 for 3-var)
        if test_num % (10 if num_vars == 2 else 20) == 0 or test_num == num_tests:
            print(f"Test {test_num:3d}: {num_ones:2d} ones -> {num_terms:2d} terms | {sop_expr[:40]}")
    
    print("=" * 70)
    return results

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def compute_statistics(results, num_vars):
    """
    Compute statistical measures for term counts.
    
    Args:
        results: List of result dictionaries
        num_vars: Number of variables (2, 3, or 4)
        
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
    
    expected_ones = (2**num_vars) // 2  # Expected 50% for balanced
    
    stats = {
        "num_vars": num_vars,
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
    print(f"\n{stats['num_vars']}-VARIABLE STATISTICAL SUMMARY")
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
    print(f"Expected ones (balanced): {stats['expected_ones']}")
    
    # Display frequency distribution
    print("\nTerm Count Distribution:")
    for terms, count in stats['term_distribution'].items():
        bar = "█" * min(count, 50)  # Cap bar length at 50
        print(f"  {terms} terms: {count:2d} cases {bar}")
    print("=" * 70)

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_comparison_histogram(results_2var, results_3var, results_4var, stats_2var, stats_3var, stats_4var):
    """Create side-by-side histograms comparing 2-var, 3-var, and 4-var results."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # 2-variable histogram
    term_counts_2 = [r["num_terms"] for r in results_2var]
    ax1.hist(term_counts_2, bins=range(min(term_counts_2), max(term_counts_2) + 2),
             edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(stats_2var['mean_terms'], color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {stats_2var["mean_terms"]:.2f}')
    ax1.axvline(stats_2var['median_terms'], color='green', linestyle='--', 
                linewidth=2, label=f'Median: {stats_2var["median_terms"]:.1f}')
    ax1.set_xlabel('Number of Terms', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('2-Variable K-Maps (4 cells)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    
    # 3-variable histogram
    term_counts_3 = [r["num_terms"] for r in results_3var]
    ax2.hist(term_counts_3, bins=range(min(term_counts_3), max(term_counts_3) + 2),
             edgecolor='black', alpha=0.7, color='coral')
    ax2.axvline(stats_3var['mean_terms'], color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {stats_3var["mean_terms"]:.2f}')
    ax2.axvline(stats_3var['median_terms'], color='green', linestyle='--', 
                linewidth=2, label=f'Median: {stats_3var["median_terms"]:.1f}')
    ax2.set_xlabel('Number of Terms', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('3-Variable K-Maps (8 cells)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    
    # 4-variable histogram
    term_counts_4 = [r["num_terms"] for r in results_4var]
    ax3.hist(term_counts_4, bins=range(min(term_counts_4), max(term_counts_4) + 2),
             edgecolor='black', alpha=0.7, color='mediumseagreen')
    ax3.axvline(stats_4var['mean_terms'], color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {stats_4var["mean_terms"]:.2f}')
    ax3.axvline(stats_4var['median_terms'], color='green', linestyle='--', 
                linewidth=2, label=f'Median: {stats_4var["median_terms"]:.1f}')
    ax3.set_xlabel('Number of Terms', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('4-Variable K-Maps (16 cells)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Balanced K-Map Term Count Distribution Comparison', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(HISTOGRAM_PNG, dpi=150)
    print(f"\nComparison histogram saved to: {HISTOGRAM_PNG}")
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

def save_statistics_txt(stats_2var, stats_3var, stats_4var, results_2var, results_3var, results_4var):
    """Save statistical summary to text file."""
    with open(STATS_TXT, 'w') as f:
        f.write("Balanced 2, 3 & 4-Variable K-Map Terms Analysis\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Configuration:\n")
        f.write(f"  Random Seed: {RANDOM_SEED}\n")
        f.write(f"  2-Variable Test Cases: {NUM_TEST_CASES_2VAR}\n")
        f.write(f"  3-Variable Test Cases: {NUM_TEST_CASES_3VAR}\n")
        f.write(f"  4-Variable Test Cases: {NUM_TEST_CASES_4VAR}\n")
        f.write(f"  K-Map Type: Balanced (p=0.5 for 1s)\n\n")
        
        # 2-variable statistics
        f.write("2-VARIABLE STATISTICAL SUMMARY:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Mean number of terms:     {stats_2var['mean_terms']:.2f}\n")
        f.write(f"Median number of terms:   {stats_2var['median_terms']:.1f}\n")
        f.write(f"Std deviation of terms:   {stats_2var['std_terms']:.2f}\n")
        f.write(f"Min number of terms:      {stats_2var['min_terms']}\n")
        f.write(f"Max number of terms:      {stats_2var['max_terms']}\n")
        
        if len(stats_2var['mode_terms']) == 1:
            f.write(f"Mode (most frequent):     {stats_2var['mode_terms'][0]} terms (occurred {stats_2var['mode_frequency']} times)\n")
        else:
            mode_str = ", ".join(str(m) for m in sorted(stats_2var['mode_terms']))
            f.write(f"Mode (most frequent):     {mode_str} terms (each occurred {stats_2var['mode_frequency']} times)\n")
        
        f.write(f"\nMean number of ones:      {stats_2var['mean_ones']:.2f}\n")
        f.write(f"Median number of ones:    {stats_2var['median_ones']:.1f}\n")
        f.write(f"Expected ones (balanced): {stats_2var['expected_ones']}\n")
        f.write("=" * 70 + "\n\n")
        
        # 3-variable statistics
        f.write("3-VARIABLE STATISTICAL SUMMARY:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Mean number of terms:     {stats_3var['mean_terms']:.2f}\n")
        f.write(f"Median number of terms:   {stats_3var['median_terms']:.1f}\n")
        f.write(f"Std deviation of terms:   {stats_3var['std_terms']:.2f}\n")
        f.write(f"Min number of terms:      {stats_3var['min_terms']}\n")
        f.write(f"Max number of terms:      {stats_3var['max_terms']}\n")
        
        if len(stats_3var['mode_terms']) == 1:
            f.write(f"Mode (most frequent):     {stats_3var['mode_terms'][0]} terms (occurred {stats_3var['mode_frequency']} times)\n")
        else:
            mode_str = ", ".join(str(m) for m in sorted(stats_3var['mode_terms']))
            f.write(f"Mode (most frequent):     {mode_str} terms (each occurred {stats_3var['mode_frequency']} times)\n")
        
        f.write(f"\nMean number of ones:      {stats_3var['mean_ones']:.2f}\n")
        f.write(f"Median number of ones:    {stats_3var['median_ones']:.1f}\n")
        f.write(f"Expected ones (balanced): {stats_3var['expected_ones']}\n")
        f.write("=" * 70 + "\n\n")
        
        # 4-variable statistics
        f.write("4-VARIABLE STATISTICAL SUMMARY:\n")
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
        f.write(f"Expected ones (balanced): {stats_4var['expected_ones']}\n")
        f.write("=" * 70 + "\n\n")
        
        # Distribution tables
        f.write("2-VARIABLE Term Count Distribution:\n")
        f.write("-" * 70 + "\n")
        for terms in sorted(stats_2var['term_distribution'].keys()):
            f.write(f"  {terms} terms: {stats_2var['term_distribution'][terms]} cases\n")
        
        f.write("\n3-VARIABLE Term Count Distribution:\n")
        f.write("-" * 70 + "\n")
        for terms in sorted(stats_3var['term_distribution'].keys()):
            f.write(f"  {terms} terms: {stats_3var['term_distribution'][terms]} cases\n")
        
        f.write("\n4-VARIABLE Term Count Distribution:\n")
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
    print("BALANCED 2, 3 & 4-VARIABLE K-MAP TERMS ANALYSIS")
    print("=" * 70)
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"2-Variable Test Cases: {NUM_TEST_CASES_2VAR}")
    print(f"3-Variable Test Cases: {NUM_TEST_CASES_3VAR}")
    print(f"4-Variable Test Cases: {NUM_TEST_CASES_4VAR}")
    print("")
    
    # Analyze 2-variable K-maps
    print("\n" + "=" * 70)
    print("ANALYZING 2-VARIABLE K-MAPS")
    print("=" * 70)
    results_2var = analyze_balanced_kmaps(2, NUM_TEST_CASES_2VAR)
    stats_2var = compute_statistics(results_2var, 2)
    print_statistics(stats_2var)
    
    # Analyze 3-variable K-maps
    print("\n" + "=" * 70)
    print("ANALYZING 3-VARIABLE K-MAPS")
    print("=" * 70)
    results_3var = analyze_balanced_kmaps(3, NUM_TEST_CASES_3VAR)
    stats_3var = compute_statistics(results_3var, 3)
    print_statistics(stats_3var)
    
    # Analyze 4-variable K-maps
    print("\n" + "=" * 70)
    print("ANALYZING 4-VARIABLE K-MAPS")
    print("=" * 70)
    results_4var = analyze_balanced_kmaps(4, NUM_TEST_CASES_4VAR)
    stats_4var = compute_statistics(results_4var, 4)
    print_statistics(stats_4var)
    
    # Create visualizations
    create_comparison_histogram(results_2var, results_3var, results_4var, 
                                stats_2var, stats_3var, stats_4var)
    
    # Save results
    save_results_csv(results_2var, RESULTS_CSV_2VAR)
    save_results_csv(results_3var, RESULTS_CSV_3VAR)
    save_results_csv(results_4var, RESULTS_CSV_4VAR)
    save_statistics_txt(stats_2var, stats_3var, stats_4var, 
                       results_2var, results_3var, results_4var)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nOutput files:")
    print(f"  - 2-var CSV: {RESULTS_CSV_2VAR}")
    print(f"  - 3-var CSV: {RESULTS_CSV_3VAR}")
    print(f"  - 4-var CSV: {RESULTS_CSV_4VAR}")
    print(f"  - Stats: {STATS_TXT}")
    print(f"  - Histogram: {HISTOGRAM_PNG}")
    print("")

if __name__ == "__main__":
    main()

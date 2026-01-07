"""
Balanced 8-Variable K-Map Terms Analysis
==========================================

Analyzes the number of terms produced by the K-Map solver for balanced
8-variable K-maps. Balanced maps have equal probability (50%) of being 1 or 0.

This test generates balanced 8-variable K-maps and examines:
- Number of terms in minimized SOP expression
- Statistics: mean, median, standard deviation, mode
- Distribution visualization

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
NUM_TEST_CASES = 500  # 8-variable has 256 cells, so fewer test cases

# Output directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, "..", "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

RESULTS_CSV = os.path.join(OUTPUTS_DIR, "balanced_8var_terms_results.csv")
STATS_TXT = os.path.join(OUTPUTS_DIR, "balanced_8var_terms_stats.txt")
HISTOGRAM_PNG = os.path.join(OUTPUTS_DIR, "balanced_8var_terms_histogram.png")

# ============================================================================
# TEST DATA GENERATION
# ============================================================================

def random_balanced_output_values_8var():
    """
    Generate a balanced random 8-variable output values list.
    Balanced means p(1) = 0.5, p(0) = 0.5, no don't cares.
    
    For 8 variables, we need 2^8 = 256 output values.
    
    Returns:
        List of 256 values (0 or 1)
    """
    size = 2**8  # 256 cells
    return [random.choice([0, 1]) for _ in range(size)]

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

def analyze_balanced_kmaps():
    """
    Generate balanced 8-variable K-maps and analyze term counts.
    
    Returns:
        List of dictionaries with test results
    """
    results = []
    
    print(f"Generating and analyzing {NUM_TEST_CASES} balanced 8-variable K-maps...")
    print("=" * 70)
    print("Note: Each test processes 256 cells (2^8 variables)")
    print("=" * 70)
    
    for test_num in range(1, NUM_TEST_CASES + 1):
        # Generate balanced output values
        output_values = random_balanced_output_values_8var()
        
        # Count ones
        num_ones = sum(1 for v in output_values if v == 1)
        
        # Create solver and minimize
        solver = BoolMinGeo(8, output_values)
        
        # Suppress verbose output
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        terms_list, sop_expr = solver.minimize(form='sop')
        
        # Restore stdout
        sys.stdout = old_stdout
        
        # Count terms
        num_terms = count_terms_in_expression(sop_expr)
        
        result = {
            "test_num": test_num,
            "num_ones": num_ones,
            "num_terms": num_terms,
            "sop_expression": sop_expr[:100] + "..." if len(sop_expr) > 100 else sop_expr,
            "full_expression": sop_expr
        }
        
        results.append(result)
        
        # Print progress
        print(f"Test {test_num:2d}: {num_ones:3d} ones -> {num_terms:3d} terms | {sop_expr[:60]}")
    
    print("=" * 70)
    return results

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def compute_statistics(results):
    """
    Compute statistical measures for term counts.
    
    Args:
        results: List of result dictionaries
        
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
    
    stats = {
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
        "expected_ones": 128,  # For 8-var balanced, expected is 256 * 0.5 = 128
    }
    
    return stats

def print_statistics(stats):
    """Print statistical summary."""
    print("\nSTATISTICAL SUMMARY")
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
    print(f"Expected ones (balanced): {stats['expected_ones']} (out of 256)")
    
    # Display frequency distribution
    print("\nTerm Count Distribution:")
    for terms, count in stats['term_distribution'].items():
        bar = "█" * count
        print(f"  {terms:3d} terms: {count:2d} cases {bar}")
    print("=" * 70)

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_histogram(results, stats):
    """Create histogram of term counts."""
    term_counts = [r["num_terms"] for r in results]
    
    plt.figure(figsize=(12, 6))
    
    # Create histogram
    n, bins, patches = plt.hist(term_counts, bins=range(min(term_counts), 
                                                        max(term_counts) + 2),
                                 edgecolor='black', alpha=0.7, color='steelblue')
    
    # Add mean line
    plt.axvline(stats['mean_terms'], color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {stats["mean_terms"]:.2f}')
    
    # Add median line
    plt.axvline(stats['median_terms'], color='green', linestyle='--', 
                linewidth=2, label=f'Median: {stats["median_terms"]:.1f}')
    
    # Add mode line(s)
    for mode in stats['mode_terms']:
        plt.axvline(mode, color='purple', linestyle=':', 
                    linewidth=2, alpha=0.7)
    
    plt.xlabel('Number of Terms', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Terms in Balanced 8-Variable K-Maps (256 cells)', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    
    # Add text box with statistics
    textstr = f'N = {len(results)}\nStd Dev: {stats["std_terms"]:.2f}\n'
    textstr += f'Range: [{stats["min_terms"]}, {stats["max_terms"]}]\n'
    if len(stats['mode_terms']) == 1:
        textstr += f'Mode: {stats["mode_terms"][0]} (×{stats["mode_frequency"]})'
    else:
        textstr += f'Modes: {", ".join(str(m) for m in sorted(stats["mode_terms"]))}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.98, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right', bbox=props)
    
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
            "test_num", "num_ones", "num_terms", 
            "sop_expression", "full_expression"
        ])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Results saved to: {RESULTS_CSV}")

def save_statistics_txt(stats, results):
    """Save statistical summary to text file."""
    with open(STATS_TXT, 'w') as f:
        f.write("Balanced 8-Variable K-Map Terms Analysis\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Configuration:\n")
        f.write(f"  Random Seed: {RANDOM_SEED}\n")
        f.write(f"  Number of Test Cases: {NUM_TEST_CASES}\n")
        f.write(f"  K-Map Type: Balanced 8-variable (p=0.5 for 1s)\n")
        f.write(f"  K-Map Size: 256 cells (2^8)\n\n")
        
        f.write("Statistical Summary:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Mean number of terms:     {stats['mean_terms']:.2f}\n")
        f.write(f"Median number of terms:   {stats['median_terms']:.1f}\n")
        f.write(f"Std deviation of terms:   {stats['std_terms']:.2f}\n")
        f.write(f"Min number of terms:      {stats['min_terms']}\n")
        f.write(f"Max number of terms:      {stats['max_terms']}\n")
        
        # Write mode information
        if len(stats['mode_terms']) == 1:
            f.write(f"Mode (most frequent):     {stats['mode_terms'][0]} terms (occurred {stats['mode_frequency']} times)\n")
        else:
            mode_str = ", ".join(str(m) for m in sorted(stats['mode_terms']))
            f.write(f"Mode (most frequent):     {mode_str} terms (each occurred {stats['mode_frequency']} times)\n")
        
        f.write(f"\nMean number of ones:      {stats['mean_ones']:.2f}\n")
        f.write(f"Median number of ones:    {stats['median_ones']:.1f}\n")
        f.write(f"Expected ones (balanced): {stats['expected_ones']} (out of 256)\n")
        f.write("=" * 70 + "\n\n")
        
        # Distribution of term counts
        f.write("Distribution of Term Counts:\n")
        f.write("-" * 70 + "\n")
        for terms in sorted(stats['term_distribution'].keys()):
            f.write(f"  {terms:3d} terms: {stats['term_distribution'][terms]:2d} cases\n")
        
        f.write("\n" + "=" * 70 + "\n\n")
        
        # Add analysis notes
        f.write("Analysis Notes:\n")
        f.write("-" * 70 + "\n")
        f.write(f"1. 8-variable K-maps have 2^8 = 256 cells\n")
        f.write(f"2. Balanced distribution means ~50% of cells are 1s (~128 ones)\n")
        f.write(f"3. The mode represents the most common term count observed\n")
        f.write(f"4. Higher variable counts typically produce more complex expressions\n")
    
    print(f"Statistics saved to: {STATS_TXT}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("BALANCED 8-VARIABLE K-MAP TERMS ANALYSIS")
    print("=" * 70)
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Test Cases: {NUM_TEST_CASES}")
    print(f"K-Map Size: 256 cells (2^8)")
    print("")
    
    # Generate and analyze K-maps
    results = analyze_balanced_kmaps()
    
    # Compute statistics
    stats = compute_statistics(results)
    print_statistics(stats)
    
    # Create visualizations
    create_histogram(results, stats)
    
    # Save results
    save_results_csv(results)
    save_statistics_txt(stats, results)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nOutput files:")
    print(f"  - CSV: {RESULTS_CSV}")
    print(f"  - Stats: {STATS_TXT}")
    print(f"  - Histogram: {HISTOGRAM_PNG}")
    print("")

if __name__ == "__main__":
    main()

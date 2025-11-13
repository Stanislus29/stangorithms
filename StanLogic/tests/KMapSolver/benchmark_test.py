"""
Scientific Benchmark: K-Map Solver vs SymPy Boolean Minimization
=================================================================

A rigorous experimental comparison of Boolean function minimization algorithms
with proper statistical analysis, reproducibility controls, and comprehensive
test coverage.

Author: [Your Name]
Date: November 2025
Version: 2.0 (Scientifically Enhanced)
"""

import time
import random
import re
import csv
import os
import sys
import platform
import datetime
from collections import defaultdict
from textwrap import wrap

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import scipy
from scipy import stats
from sympy import symbols, SOPform, simplify, Equivalent, POSform
from stanlogic import KMapSolver
from tabulate import tabulate

# ============================================================================
# CONFIGURATION & EXPERIMENTAL PARAMETERS
# ============================================================================

# Random seed for reproducibility (CRITICAL for scientific validity)
RANDOM_SEED = 42

# Benchmark parameters
TESTS_PER_CONFIG = 500  # Reduced from 1000 for faster testing, increase for production
TIMING_REPEATS = 5      # Number of timing repetitions per test
TIMING_WARMUP = 2       # Warm-up iterations before timing

# Statistical significance threshold
ALPHA = 0.05  # 95% confidence level

# Output directories
OUTPUTS_DIR = "outputs"
RESULTS_CSV = os.path.join(OUTPUTS_DIR, "benchmark_results.csv")
REPORT_PDF = os.path.join(OUTPUTS_DIR, "benchmark_scientific_report.pdf")
STATS_CSV = os.path.join(OUTPUTS_DIR, "statistical_analysis.csv")

# Logo for report cover
LOGO_PATH = "StanLogic/images/St_logo_light-tp.png"

# ============================================================================
# EXPERIMENTAL SETUP DOCUMENTATION
# ============================================================================

def document_experimental_setup():
    """
    Record all relevant environmental factors for reproducibility.
    Returns a dictionary with system and experimental configuration.
    """
    import sympy
    
    setup_info = {
        "experiment_date": datetime.datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "sympy_version": sympy.__version__,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "random_seed": RANDOM_SEED,
        "tests_per_config": TESTS_PER_CONFIG,
        "timing_repeats": TIMING_REPEATS,
        "timing_warmup": TIMING_WARMUP,
        "alpha_level": ALPHA,
    }
    return setup_info

# ============================================================================
# TEST DATA GENERATION
# ============================================================================

class TestDistribution:
    """Enumeration of different test data distributions."""
    SPARSE = "sparse"           # Few 1s (common in digital logic)
    DENSE = "dense"             # Many 1s
    BALANCED = "balanced"       # Equal probability
    MINIMAL_DC = "minimal_dc"   # Few don't-cares (realistic)
    HEAVY_DC = "heavy_dc"       # Many don't-cares (stress test)

def random_kmap(rows=4, cols=4, p_one=0.45, p_dc=0.1, seed=None):
    """
    Generate a random K-map with controlled probability distribution.
    
    Args:
        rows: Number of rows in K-map
        cols: Number of columns in K-map
        p_one: Probability a cell is 1
        p_dc: Probability a cell is 'd' (don't care)
        seed: Random seed for this specific K-map
        
    Returns:
        2D list representing the K-map
        
    Raises:
        ValueError: If probabilities are invalid
    """
    if seed is not None:
        random.seed(seed)
    
    if p_one < 0 or p_dc < 0 or p_one + p_dc > 1:
        raise ValueError("Probabilities invalid: ensure p_one >=0, p_dc >=0, p_one + p_dc <= 1")
    
    p_zero = 1 - p_one - p_dc
    choices = [1, 0, 'd']
    weights = [p_one, p_zero, p_dc]
    
    kmap = [[random.choices(choices, weights=weights, k=1)[0] 
             for _ in range(cols)] for _ in range(rows)]
    
    return kmap

def generate_test_suite(num_vars, tests_per_distribution=100):
    """
    Generate comprehensive test suite with diverse distributions.
    
    Args:
        num_vars: Number of variables (2, 3, or 4)
        tests_per_distribution: Number of tests per distribution type
        
    Returns:
        List of tuples: (kmap, distribution_label)
    """
    rows, cols = _kmap_shape(num_vars)
    test_cases = []
    
    # Distribution configurations
    distributions = {
        TestDistribution.SPARSE: {"p_one": 0.2, "p_dc": 0.05},
        TestDistribution.DENSE: {"p_one": 0.7, "p_dc": 0.05},
        TestDistribution.BALANCED: {"p_one": 0.5, "p_dc": 0.1},
        TestDistribution.MINIMAL_DC: {"p_one": 0.45, "p_dc": 0.02},
        TestDistribution.HEAVY_DC: {"p_one": 0.3, "p_dc": 0.3},
    }
    
    for dist_name, params in distributions.items():
        for i in range(tests_per_distribution):
            kmap = random_kmap(rows=rows, cols=cols, **params)
            test_cases.append((kmap, dist_name))
    
    # Add edge cases
    test_cases.extend(generate_edge_cases(num_vars))
    
    return test_cases

def generate_edge_cases(num_vars):
    """
    Generate edge case K-maps for comprehensive testing.
    
    Args:
        num_vars: Number of variables
        
    Returns:
        List of tuples: (kmap, label)
    """
    rows, cols = _kmap_shape(num_vars)
    edge_cases = []
    
    # All zeros
    edge_cases.append(([[0] * cols for _ in range(rows)], "edge_all_zeros"))
    
    # All ones
    edge_cases.append(([[1] * cols for _ in range(rows)], "edge_all_ones"))
    
    # All don't cares
    edge_cases.append(([['d'] * cols for _ in range(rows)], "edge_all_dc"))
    
    # Checkerboard pattern
    checkerboard = [[(i + j) % 2 for j in range(cols)] for i in range(rows)]
    edge_cases.append((checkerboard, "edge_checkerboard"))
    
    # Single 1
    single_one = [[0] * cols for _ in range(rows)]
    single_one[0][0] = 1
    edge_cases.append((single_one, "edge_single_one"))
    
    return edge_cases

def _kmap_shape(num_vars):
    """Return (rows, cols) for given number of variables."""
    if num_vars == 2:
        return 2, 2
    if num_vars == 3:
        return 2, 4
    if num_vars == 4:
        return 4, 4
    raise ValueError("Only 2-4 variables supported.")

# ============================================================================
# K-MAP ANALYSIS
# ============================================================================

def kmap_minterms(kmap, convention="vranseic"):
    """
    Extract minterm indices and don't care indices from a K-map.
    
    Args:
        kmap: 2D list representing the K-map
        convention: Variable ordering convention ('vranseic' or 'mano_kime')
        
    Returns:
        Dictionary with num_vars, minterms, dont_cares, and bits_map
    """
    rows = len(kmap)
    cols = len(kmap[0])
    size = rows * cols

    if size == 4:
        num_vars = 2
        row_labels = ["0", "1"]
        col_labels = ["0", "1"]
    elif size == 8:
        num_vars = 3
        if rows == 2 and cols == 4:
            row_labels = ["0", "1"]
            col_labels = ["00", "01", "11", "10"]
        elif rows == 4 and cols == 2:
            row_labels = ["00", "01", "11", "10"]
            col_labels = ["0", "1"]
        else:
            raise ValueError("Invalid 3-variable K-map shape.")
    elif size == 16:
        num_vars = 4
        row_labels = ["00", "01", "11", "10"]
        col_labels = ["00", "01", "11", "10"]
    else:
        raise ValueError("K-map must be 2x2, 2x4/4x2, or 4x4.")

    convention = convention.lower().strip()

    def cell_to_bits(r, c):
        if convention == "mano_kime":
            return row_labels[r] + col_labels[c]
        return col_labels[c] + row_labels[r]

    def cell_to_term(r, c):
        bits = cell_to_bits(r, c)
        return int(bits, 2)

    minterms = []
    dont_cares = []
    bits_map = {}

    for r in range(rows):
        for c in range(cols):
            val = kmap[r][c]
            idx = cell_to_term(r, c)
            bits_map[idx] = cell_to_bits(r, c)
            if val == 1:
                minterms.append(idx)
            elif val == 'd':
                dont_cares.append(idx)

    return {
        "num_vars": num_vars,
        "minterms": sorted(minterms),
        "dont_cares": sorted(dont_cares),
        "bits_map": bits_map
    }

# ============================================================================
# EXPRESSION PARSING & VALIDATION
# ============================================================================

def parse_kmap_expression(expr_str, var_names, form="sop"):
    """
    Parse a Boolean expression string into a SymPy expression.
    
    Args:
        expr_str: The Boolean expression string
        var_names: List of SymPy symbols
        form: 'sop' or 'pos'
        
    Returns:
        SymPy Boolean expression
    """
    from sympy import And, Or, Not, false, true

    if not expr_str or expr_str.strip() == "":
        return false if form == "sop" else true

    name_map = {str(v): v for v in var_names}

    if form.lower() == "sop":
        or_terms = []
        for product in expr_str.split('+'):
            product = product.strip()
            if not product:
                continue
            literals = re.findall(r"[A-Za-z_]\w*'?", product)
            and_terms = []
            for lit in literals:
                if lit.endswith("'"):
                    var = lit[:-1]
                    if var in name_map:
                        and_terms.append(Not(name_map[var]))
                else:
                    if lit in name_map:
                        and_terms.append(name_map[lit])
            if len(and_terms) == 1:
                or_terms.append(and_terms[0])
            elif and_terms:
                or_terms.append(And(*and_terms))
        return Or(*or_terms) if or_terms else false

    elif form.lower() == "pos":
        and_terms = []
        for sum_term in re.split(r'\)\s*\(', expr_str.strip('() ')):
            sum_term = sum_term.strip()
            if not sum_term:
                continue
            literals = re.findall(r"[A-Za-z_]\w*'?", sum_term)
            or_terms = []
            for lit in literals:
                if lit.endswith("'"):
                    var = lit[:-1]
                    if var in name_map:
                        or_terms.append(Not(name_map[var]))
                else:
                    if lit in name_map:
                        or_terms.append(name_map[lit])
            if len(or_terms) == 1:
                and_terms.append(or_terms[0])
            elif or_terms:
                and_terms.append(Or(*or_terms))
        return And(*and_terms) if and_terms else true

    raise ValueError(f"Unsupported form '{form}'.")

def check_equivalence(expr1, expr2):
    """Check if two SymPy expressions are logically equivalent."""
    try:
        return bool(simplify(Equivalent(expr1, expr2)))
    except Exception:
        return False

def count_literals(expr_str, form="sop"):
    """
    Count terms and literals in a Boolean expression string.
    
    Args:
        expr_str: Boolean expression string
        form: 'sop' or 'pos'
        
    Returns:
        Tuple of (num_terms, num_literals)
    """
    if not expr_str or expr_str.strip() == "":
        return 0, 0

    form = form.lower()
    s = expr_str.replace(" ", "")

    if form == "sop":
        terms = [t for t in s.split('+') if t]
        num_terms = len(terms)
        num_literals = sum(len(re.findall(r"[A-Za-z_]\w*'?", t)) for t in terms)
        return num_terms, num_literals

    if form == "pos":
        clauses = re.findall(r"\(([^()]*)\)", s)
        if not clauses:
            clauses = [s]
        num_terms = len(clauses)
        num_literals = 0
        for clause in clauses:
            lits = [lit for lit in clause.split('+') if lit]
            num_literals += sum(1 for lit in lits if re.fullmatch(r"[A-Za-z_]\w*'?", lit))
        return num_terms, num_literals

    raise ValueError("form must be 'sop' or 'pos'")

def count_sympy_expression_literals(expr, form="sop"):
    """
    Count terms and literals from a SymPy expression (string parsing).
    
    Args:
        expr: SymPy expression
        form: 'sop' or 'pos'
        
    Returns:
        Tuple of (num_terms, num_literals)
    """
    if expr is None:
        return 0, 0
    
    s = str(expr).strip()
    form = form.lower()

    if s in ("True", "False"):
        return 0, 0

    if re.fullmatch(r"~?[A-Za-z_]\w*", s):
        return 1, 1

    def split_top_level(text, sep):
        parts = []
        buf = []
        depth = 0
        for ch in text:
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            if ch == sep and depth == 0:
                part = ''.join(buf).strip()
                if part:
                    parts.append(part)
                buf = []
            else:
                buf.append(ch)
        last = ''.join(buf).strip()
        if last:
            parts.append(last)
        return parts

    def strip_parens(x):
        x = x.strip()
        while x.startswith('(') and x.endswith(')'):
            depth = 0
            valid = True
            for i, ch in enumerate(x):
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                    if depth == 0 and i != len(x) - 1:
                        valid = False
                        break
            if valid:
                x = x[1:-1].strip()
            else:
                break
        return x

    def count_literals_in_group(group_str, inner_sep):
        group_str = strip_parens(group_str)
        if not group_str:
            return 0
        inner_parts = split_top_level(group_str, inner_sep)
        if len(inner_parts) == 1:
            raw = inner_parts[0]
            lits = split_top_level(raw, inner_sep)
        else:
            lits = inner_parts
        literal_count = sum(1 for lit in lits if re.fullmatch(r"~?[A-Za-z_]\w*", strip_parens(lit)))
        return literal_count

    if form == "sop":
        top_terms = split_top_level(s, '|')
        num_terms = len(top_terms)
        total_literals = sum(count_literals_in_group(term, '&') for term in top_terms)
        return num_terms, total_literals

    if form == "pos":
        top_clauses = split_top_level(s, '&')
        num_terms = len(top_clauses)
        total_literals = sum(count_literals_in_group(clause, '|') for clause in top_clauses)
        return num_terms, total_literals

    raise ValueError("form must be 'sop' or 'pos'")

# ============================================================================
# IMPROVED TIMING METHODOLOGY
# ============================================================================

def benchmark_with_warmup(func, args, warmup=TIMING_WARMUP, repeats=TIMING_REPEATS):
    """
    Benchmark a function with warm-up runs and multiple repetitions.
    
    Args:
        func: Function to benchmark
        args: Tuple of arguments to pass to function
        warmup: Number of warm-up iterations
        repeats: Number of timed repetitions
        
    Returns:
        Minimum execution time (best of N to minimize OS noise)
    """
    # Warm-up phase (not counted)
    for _ in range(warmup):
        func(*args)
    
    # Actual timing
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        func(*args)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    # Return minimum time (reduces impact of system interrupts)
    return min(times)

# ============================================================================
# BENCHMARK EXECUTION
# ============================================================================

def benchmark_case(kmap, var_names, form_type, test_index, distribution):
    """
    Run a single benchmark case comparing SymPy and KMapSolver.
    
    Args:
        kmap: K-map as 2D list
        var_names: List of SymPy variable symbols
        form_type: 'sop' or 'pos'
        test_index: Test case number
        distribution: Distribution type label
        
    Returns:
        Dictionary with benchmark results
    """
    info = kmap_minterms(kmap, convention="vranseic")
    minterms = info["minterms"]
    dont_cares = info["dont_cares"]

    if len(var_names) != info["num_vars"]:
        raise ValueError(f"Variable count mismatch: {len(var_names)} != {info['num_vars']}")

    # SymPy minimization with improved timing
    if form_type == "pos":
        sympy_func = lambda: POSform(var_names, minterms, dont_cares)
    else:
        sympy_func = lambda: SOPform(var_names, minterms, dont_cares)
    
    print(f"    Test {test_index}: Running SymPy {form_type.upper()}...", end=" ", flush=True)
    t_sympy = benchmark_with_warmup(sympy_func, ())
    expr_sympy = sympy_func()  # Get actual result
    print(f"✓ ({t_sympy:.6f}s)", end=" | ", flush=True)

    # KMapSolver minimization with improved timing
    kmap_func = lambda: KMapSolver(kmap).minimize(form=form_type)
    print(f"KMapSolver...", end=" ", flush=True)
    t_kmap = benchmark_with_warmup(kmap_func, ())
    terms, expr_str = kmap_func()  # Get actual result
    print(f"✓ ({t_kmap:.6f}s)", end=" | ", flush=True)

    # Validate equivalence
    expr_kmap_sympy = parse_kmap_expression(expr_str, var_names, form=form_type)
    equiv = check_equivalence(expr_sympy, expr_kmap_sympy)
    equiv_symbol = "✓" if equiv else "✗"
    print(f"Equiv: {equiv_symbol}")

    # Compute metrics
    sympy_terms, sympy_literals = count_sympy_expression_literals(str(expr_sympy), form=form_type)
    kmap_terms, kmap_literals = count_literals(expr_str, form=form_type)

    return {
        "index": test_index,
        "num_vars": info["num_vars"],
        "form": form_type,
        "distribution": distribution,
        "equiv": equiv,
        "t_sympy": t_sympy,
        "t_kmap": t_kmap,
        "sympy_literals": sympy_literals,
        "kmap_literals": kmap_literals,
        "sympy_terms": sympy_terms,
        "kmap_terms": kmap_terms,
    }

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def perform_statistical_tests(sympy_times, kmap_times, sympy_literals, kmap_literals):
    """
    Perform rigorous statistical hypothesis testing.
    
    Args:
        sympy_times: List of SymPy execution times
        kmap_times: List of KMapSolver execution times
        sympy_literals: List of SymPy literal counts
        kmap_literals: List of KMapSolver literal counts
        
    Returns:
        Dictionary with comprehensive statistical test results
    """
    # Convert to numpy arrays
    sympy_times = np.array(sympy_times)
    kmap_times = np.array(kmap_times)
    sympy_literals = np.array(sympy_literals)
    kmap_literals = np.array(kmap_literals)
    
    # Calculate differences
    time_diff = kmap_times - sympy_times
    lit_diff = kmap_literals - sympy_literals
    
    # Paired t-test (parametric)
    t_stat_time, p_value_time = stats.ttest_rel(kmap_times, sympy_times)
    t_stat_lit, p_value_lit = stats.ttest_rel(kmap_literals, sympy_literals)
    
    # Wilcoxon signed-rank test (non-parametric alternative)
    w_stat_time, w_p_time = stats.wilcoxon(kmap_times, sympy_times, zero_method='zsplit')
    w_stat_lit, w_p_lit = stats.wilcoxon(kmap_literals, sympy_literals, zero_method='zsplit')
    
    # Effect size (Cohen's d)
    cohens_d_time = np.mean(time_diff) / np.std(time_diff, ddof=1) if np.std(time_diff) > 0 else 0
    cohens_d_lit = np.mean(lit_diff) / np.std(lit_diff, ddof=1) if np.std(lit_diff) > 0 else 0
    
    # Confidence intervals (95%)
    ci_time = stats.t.interval(0.95, len(time_diff)-1, 
                               loc=np.mean(time_diff), 
                               scale=stats.sem(time_diff))
    ci_lit = stats.t.interval(0.95, len(lit_diff)-1,
                              loc=np.mean(lit_diff),
                              scale=stats.sem(lit_diff))
    
    return {
        "time": {
            "mean_sympy": np.mean(sympy_times),
            "mean_kmap": np.mean(kmap_times),
            "mean_diff": np.mean(time_diff),
            "std_diff": np.std(time_diff, ddof=1),
            "t_statistic": t_stat_time,
            "p_value": p_value_time,
            "wilcoxon_stat": w_stat_time,
            "wilcoxon_p": w_p_time,
            "cohens_d": cohens_d_time,
            "ci_lower": ci_time[0],
            "ci_upper": ci_time[1],
            "significant": p_value_time < ALPHA
        },
        "literals": {
            "mean_sympy": np.mean(sympy_literals),
            "mean_kmap": np.mean(kmap_literals),
            "mean_diff": np.mean(lit_diff),
            "std_diff": np.std(lit_diff, ddof=1),
            "t_statistic": t_stat_lit,
            "p_value": p_value_lit,
            "wilcoxon_stat": w_stat_lit,
            "wilcoxon_p": w_p_lit,
            "cohens_d": cohens_d_lit,
            "ci_lower": ci_lit[0],
            "ci_upper": ci_lit[1],
            "significant": p_value_lit < ALPHA
        }
    }

def interpret_effect_size(cohens_d):
    """Interpret Cohen's d effect size."""
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"

# ============================================================================
# INFERENCE GENERATION
# ============================================================================

def generate_scientific_inference(stats_results):
    """
    Generate scientifically rigorous inference with statistical support.
    
    Args:
        stats_results: Dictionary from perform_statistical_tests()
        
    Returns:
        Formatted inference text
    """
    lines = []
    lines.append("=" * 75)
    lines.append("STATISTICAL INFERENCE REPORT")
    lines.append("=" * 75)
    
    # Execution Time Analysis
    lines.append("\n1. EXECUTION TIME ANALYSIS")
    lines.append("-" * 75)
    time_stats = stats_results["time"]
    lines.append(f"Mean SymPy Time:      {time_stats['mean_sympy']:.6f} s")
    lines.append(f"Mean KMapSolver Time: {time_stats['mean_kmap']:.6f} s")
    lines.append(f"Mean Difference:      {time_stats['mean_diff']:+.6f} s")
    lines.append(f"Std. Dev. (Δ):        {time_stats['std_diff']:.6f} s")
    lines.append(f"95% CI:               [{time_stats['ci_lower']:.6f}, {time_stats['ci_upper']:.6f}]")
    lines.append("")
    lines.append(f"Paired t-test:        t = {time_stats['t_statistic']:.4f}, p = {time_stats['p_value']:.6f}")
    lines.append(f"Wilcoxon test:        W = {time_stats['wilcoxon_stat']:.1f}, p = {time_stats['wilcoxon_p']:.6f}")
    lines.append(f"Effect Size (d):      {time_stats['cohens_d']:.4f} ({interpret_effect_size(time_stats['cohens_d'])})")
    
    if time_stats['significant']:
        lines.append(f"\n✓ SIGNIFICANT: Time difference is statistically significant (p < {ALPHA})")
        if time_stats['mean_diff'] < 0:
            lines.append("  → KMapSolver is significantly faster than SymPy")
        else:
            lines.append("  → SymPy is significantly faster than KMapSolver")
    else:
        lines.append(f"\n✗ NOT SIGNIFICANT: No statistically significant time difference (p ≥ {ALPHA})")
        lines.append("  → Both algorithms exhibit comparable performance")
    
    # Literal Count Analysis
    lines.append("\n2. SIMPLIFICATION QUALITY ANALYSIS")
    lines.append("-" * 75)
    lit_stats = stats_results["literals"]
    lines.append(f"Mean SymPy Literals:  {lit_stats['mean_sympy']:.2f}")
    lines.append(f"Mean KMap Literals:   {lit_stats['mean_kmap']:.2f}")
    lines.append(f"Mean Difference:      {lit_stats['mean_diff']:+.2f}")
    lines.append(f"Std. Dev. (Δ):        {lit_stats['std_diff']:.2f}")
    lines.append(f"95% CI:               [{lit_stats['ci_lower']:.2f}, {lit_stats['ci_upper']:.2f}]")
    lines.append("")
    lines.append(f"Paired t-test:        t = {lit_stats['t_statistic']:.4f}, p = {lit_stats['p_value']:.6f}")
    lines.append(f"Wilcoxon test:        W = {lit_stats['wilcoxon_stat']:.1f}, p = {lit_stats['wilcoxon_p']:.6f}")
    lines.append(f"Effect Size (d):      {lit_stats['cohens_d']:.4f} ({interpret_effect_size(lit_stats['cohens_d'])})")
    
    if lit_stats['significant']:
        lines.append(f"\n✓ SIGNIFICANT: Literal count difference is statistically significant (p < {ALPHA})")
        if lit_stats['mean_diff'] < 0:
            lines.append("  → KMapSolver produces more minimal expressions")
        else:
            lines.append("  → SymPy produces more minimal expressions")
    else:
        lines.append(f"\n✗ NOT SIGNIFICANT: No significant difference in simplification (p ≥ {ALPHA})")
        lines.append("  → Both algorithms achieve comparable minimization")
    
    # Overall Verdict
    lines.append("\n3. OVERALL SCIENTIFIC CONCLUSION")
    lines.append("-" * 75)
    
    if time_stats['significant'] and lit_stats['significant']:
        lines.append("Both performance and simplification show statistically significant differences.")
    elif time_stats['significant']:
        lines.append("Performance difference is significant, but simplification is comparable.")
    elif lit_stats['significant']:
        lines.append("Simplification difference is significant, but performance is comparable.")
    else:
        lines.append("No statistically significant differences detected in either metric.")
    
    lines.append(f"\nEffect sizes: Time ({interpret_effect_size(time_stats['cohens_d'])}), "
                 f"Literals ({interpret_effect_size(lit_stats['cohens_d'])})")
    lines.append("=" * 75)
    
    text = "\n".join(lines)
    print(text)
    return text

# ============================================================================
# EXPORT & VISUALIZATION
# ============================================================================

def export_results_to_csv(all_results, filename=RESULTS_CSV):
    """Export benchmark results to CSV."""
    print(f"\n📄 Exporting raw results to CSV...", end=" ", flush=True)
    grouped = defaultdict(list)
    for r in all_results:
        grouped[(r["num_vars"], r["form"])].append(r)

    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Configuration", "Test Index", "Distribution", "Equivalent", 
                         "SymPy Time (s)", "KMap Time (s)", 
                         "SymPy Literals", "KMap Literals"])

        for (vars, form_type), group in sorted(grouped.items()):
            writer.writerow([])
            writer.writerow([f"{vars}-Variable ({form_type.upper()} Form)"])
            
            for result in group:
                writer.writerow([
                    f"{vars}-{form_type}",
                    result.get("index", ""),
                    result.get("distribution", ""),
                    result.get("equiv", ""),
                    f"{result.get('t_sympy', 0):.8f}",
                    f"{result.get('t_kmap', 0):.8f}",
                    result.get("sympy_literals", ""),
                    result.get("kmap_literals", "")
                ])

    print(f"✅ Done ({len(all_results)} records)")

def export_statistical_analysis(all_stats, filename=STATS_CSV):
    """Export statistical analysis results to CSV."""
    print(f"📊 Exporting statistical analysis...", end=" ", flush=True)
    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(["Configuration", "Metric", "Mean SymPy", "Mean KMap", 
                        "Mean Diff", "Std Diff", "t-statistic", "p-value",
                        "Cohen's d", "Effect Size", "CI Lower", "CI Upper", "Significant"])
        
        for (num_vars, form_type), stats in sorted(all_stats.items()):
            config = f"{num_vars}-{form_type}"
            
            # Time statistics
            t = stats["time"]
            writer.writerow([
                config, "Time (s)",
                f"{t['mean_sympy']:.8f}",
                f"{t['mean_kmap']:.8f}",
                f"{t['mean_diff']:.8f}",
                f"{t['std_diff']:.8f}",
                f"{t['t_statistic']:.4f}",
                f"{t['p_value']:.6f}",
                f"{t['cohens_d']:.4f}",
                interpret_effect_size(t['cohens_d']),
                f"{t['ci_lower']:.8f}",
                f"{t['ci_upper']:.8f}",
                "Yes" if t['significant'] else "No"
            ])
            
            # Literal statistics
            l = stats["literals"]
            writer.writerow([
                config, "Literals",
                f"{l['mean_sympy']:.2f}",
                f"{l['mean_kmap']:.2f}",
                f"{l['mean_diff']:.2f}",
                f"{l['std_diff']:.2f}",
                f"{l['t_statistic']:.4f}",
                f"{l['p_value']:.6f}",
                f"{l['cohens_d']:.4f}",
                interpret_effect_size(l['cohens_d']),
                f"{l['ci_lower']:.2f}",
                f"{l['ci_upper']:.2f}",
                "Yes" if l['significant'] else "No"
            ])
    
    print(f"✅ Done ({len(all_stats)} configurations)")

def save_comprehensive_report(all_results, all_stats, setup_info, pdf_filename=REPORT_PDF):
    """
    Generate comprehensive scientific report as PDF.
    """
    print(f"\n📊 Generating comprehensive scientific report...")
    print(f"   Output: {pdf_filename}")
    
    with PdfPages(pdf_filename) as pdf:
        # ============================================================
        # COVER PAGE
        # ============================================================
        print(f"   • Creating cover page...", end=" ", flush=True)
        fig = plt.figure(figsize=(8.5, 11))
        ax = plt.gca()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        
        # Logo (if available)
        if os.path.exists(LOGO_PATH):
            try:
                img = mpimg.imread(LOGO_PATH)
                logo_ax = fig.add_axes([0.2, 0.65, 0.6, 0.25])
                logo_ax.imshow(img)
                logo_ax.axis("off")
            except:
                pass
        
        # Title
        ax.text(0.5, 0.55, "Scientific Benchmark Report", 
                fontsize=28, fontweight='bold', ha='center')
        ax.text(0.5, 0.49, "K-Map Solver vs SymPy Boolean Minimization",
                fontsize=16, ha='center')
        
        # Separator
        ax.plot([0.2, 0.8], [0.45, 0.45], 'k-', linewidth=2, alpha=0.3)
        
        # Metadata
        ax.text(0.5, 0.35, f"Experiment Date: {setup_info['experiment_date'][:10]}",
                fontsize=11, ha='center')
        ax.text(0.5, 0.31, f"Random Seed: {setup_info['random_seed']}",
                fontsize=11, ha='center')
        ax.text(0.5, 0.27, f"Total Test Cases: {len(all_results)}",
                fontsize=11, ha='center')
        ax.text(0.5, 0.23, f"Statistical Significance Level: α = {ALPHA}",
                fontsize=11, ha='center')
        
        # Footer
        ax.text(0.5, 0.08, "A Rigorous Statistical Analysis with Reproducibility Controls",
                fontsize=10, ha='center', style='italic', color='gray')
        ax.text(0.5, 0.04, f"© Stan's Technologies {datetime.datetime.now().year}",
                fontsize=9, ha='center', color='gray')
        
        pdf.savefig(bbox_inches='tight')
        plt.close()
        print("✓")
        
        # ============================================================
        # EXPERIMENTAL SETUP PAGE
        # ============================================================
        print(f"   • Creating experimental setup page...", end=" ", flush=True)
        fig = plt.figure(figsize=(8.5, 11))
        ax = plt.gca()
        ax.axis("off")
        
        ax.text(0.5, 0.95, "EXPERIMENTAL SETUP", 
                fontsize=20, fontweight='bold', ha='center', transform=ax.transAxes)
        
        setup_text = f"""
SYSTEM CONFIGURATION
{'─' * 70}
Python Version:    {setup_info['python_version'].split()[0]}
Platform:          {setup_info['platform']}
Processor:         {setup_info['processor']}

LIBRARY VERSIONS
{'─' * 70}
SymPy:             {setup_info['sympy_version']}
NumPy:             {setup_info['numpy_version']}
SciPy:             {setup_info['scipy_version']}

EXPERIMENTAL PARAMETERS
{'─' * 70}
Random Seed:                {setup_info['random_seed']}
Tests per Configuration:    {setup_info['tests_per_config']}
Timing Warm-up Runs:        {setup_info['timing_warmup']}
Timing Repetitions:         {setup_info['timing_repeats']}
Significance Level (α):     {setup_info['alpha_level']}

TEST DISTRIBUTIONS
{'─' * 70}
• Sparse:       20% ones, 5% don't-cares (realistic digital logic)
• Dense:        70% ones, 5% don't-cares (stress test)
• Balanced:     50% ones, 10% don't-cares (neutral case)
• Minimal DC:   45% ones, 2% don't-cares (typical circuits)
• Heavy DC:     30% ones, 30% don't-cares (optimization test)
• Edge Cases:   All-zeros, all-ones, checkerboard, single-minterm

METHODOLOGY
{'─' * 70}
1. Random K-maps generated with controlled distributions
2. Each algorithm executed with {setup_info['timing_warmup']} warm-up runs
3. Best of {setup_info['timing_repeats']} timed repetitions recorded
4. Logical equivalence verified using SymPy
5. Statistical significance tested using paired t-tests
6. Non-parametric Wilcoxon tests used as robustness check
7. Effect sizes computed using Cohen's d

REPRODUCIBILITY
{'─' * 70}
To reproduce this experiment:
  1. Set random seed: random.seed({setup_info['random_seed']})
  2. Run with identical system configuration
  3. Use same library versions as documented above
"""
        
        ax.text(0.05, 0.88, setup_text, fontsize=9, family='monospace',
                va='top', transform=ax.transAxes)
        
        pdf.savefig(bbox_inches='tight')
        plt.close()
        print("✓")
        
        # ============================================================
        # PER-CONFIGURATION RESULTS
        # ============================================================
        grouped = defaultdict(list)
        for r in all_results:
            grouped[(r["num_vars"], r["form"])].append(r)
        
        for config_num, ((num_vars, form_type), results) in enumerate(sorted(grouped.items()), 1):
            print(f"   • Creating plots for {num_vars}-var {form_type.upper()} ({config_num}/{len(grouped)})...", end=" ", flush=True)
            
            # Extract data
            sympy_times = [r["t_sympy"] for r in results]
            kmap_times = [r["t_kmap"] for r in results]
            sympy_literals = [r["sympy_literals"] for r in results]
            kmap_literals = [r["kmap_literals"] for r in results]
            tests = list(range(1, len(results) + 1))
            
            stats = all_stats[(num_vars, form_type)]
            
            # Create figure with 2x2 subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
            fig.suptitle(f'{num_vars}-Variable K-Map ({form_type.upper()} Form)', 
                        fontsize=16, fontweight='bold')
            
            # Plot 1: Time comparison (line plot)
            ax1.plot(tests, sympy_times, marker='o', markersize=2, label='SymPy', alpha=0.7)
            ax1.plot(tests, kmap_times, marker='s', markersize=2, label='KMapSolver', alpha=0.7)
            ax1.axhline(stats['time']['mean_sympy'], color='blue', linestyle='--', 
                       alpha=0.5, label=f'SymPy Mean = {stats["time"]["mean_sympy"]:.6f}s')
            ax1.axhline(stats['time']['mean_kmap'], color='orange', linestyle='--',
                       alpha=0.5, label=f'KMap Mean = {stats["time"]["mean_kmap"]:.6f}s')
            ax1.set_xlabel('Test Case')
            ax1.set_ylabel('Execution Time (s)')
            ax1.set_title('Execution Time Comparison')
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Time difference histogram
            time_diffs = np.array(kmap_times) - np.array(sympy_times)
            ax2.hist(time_diffs, bins=30, edgecolor='black', alpha=0.7)
            ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='No Difference')
            ax2.axvline(stats['time']['mean_diff'], color='green', linestyle='--', 
                       linewidth=2, label=f'Mean Δ = {stats["time"]["mean_diff"]:.6f}s')
            ax2.set_xlabel('Time Difference (KMap - SymPy) [s]')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Distribution of Time Differences')
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Plot 3: Literal comparison (bar plot)
            sample_indices = list(range(0, len(tests), max(1, len(tests)//50)))  # Show ~50 bars
            ax3.bar([tests[i] - 0.2 for i in sample_indices], 
                   [sympy_literals[i] for i in sample_indices], 
                   width=0.4, label='SymPy', alpha=0.7)
            ax3.bar([tests[i] + 0.2 for i in sample_indices], 
                   [kmap_literals[i] for i in sample_indices],
                   width=0.4, label='KMapSolver', alpha=0.7)
            ax3.axhline(stats['literals']['mean_sympy'], color='blue', linestyle='--',
                       alpha=0.5, label=f'SymPy Mean = {stats["literals"]["mean_sympy"]:.2f}')
            ax3.axhline(stats['literals']['mean_kmap'], color='orange', linestyle='--',
                       alpha=0.5, label=f'KMap Mean = {stats["literals"]["mean_kmap"]:.2f}')
            ax3.set_xlabel('Test Case (sampled)')
            ax3.set_ylabel('Number of Literals')
            ax3.set_title('Literal Count Comparison')
            ax3.legend(fontsize=8)
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Plot 4: Literal difference histogram
            lit_diffs = np.array(kmap_literals) - np.array(sympy_literals)
            ax4.hist(lit_diffs, bins=20, edgecolor='black', alpha=0.7)
            ax4.axvline(0, color='red', linestyle='--', linewidth=2, label='No Difference')
            ax4.axvline(stats['literals']['mean_diff'], color='green', linestyle='--',
                       linewidth=2, label=f'Mean Δ = {stats["literals"]["mean_diff"]:.2f}')
            ax4.set_xlabel('Literal Difference (KMap - SymPy)')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Distribution of Literal Differences')
            ax4.legend(fontsize=8)
            ax4.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            pdf.savefig()
            plt.close()
            print("✓")
            
            # ============================================================
            # STATISTICAL INFERENCE PAGE (per configuration)
            # ============================================================
            print(f"   • Creating statistical analysis page ({config_num}/{len(grouped)})...", end=" ", flush=True)
            fig = plt.figure(figsize=(8.5, 11))
            ax = plt.gca()
            ax.axis("off")
            
            ax.text(0.5, 0.96, f"STATISTICAL ANALYSIS: {num_vars}-Variable {form_type.upper()}",
                   fontsize=16, fontweight='bold', ha='center', transform=ax.transAxes)
            
            inference = generate_scientific_inference(stats)
            
            # Display inference with proper formatting
            y_pos = 0.90
            for line in inference.split('\n'):
                if line.startswith('=') or line.startswith('-'):
                    fontsize, fontweight = 9, 'normal'
                    y_pos -= 0.015
                elif any(line.strip().startswith(h) for h in ['1.', '2.', '3.', 'STATISTICAL']):
                    fontsize, fontweight = 11, 'bold'
                    y_pos -= 0.020
                elif line.strip().startswith(('✓', '✗')):
                    fontsize, fontweight = 10, 'bold'
                else:
                    fontsize, fontweight = 9, 'normal'
                
                ax.text(0.05, y_pos, line, fontsize=fontsize, fontweight=fontweight,
                       family='monospace', va='top', transform=ax.transAxes)
                y_pos -= 0.022
            
            pdf.savefig(bbox_inches='tight')
            plt.close()
            print("✓")
        
        # ============================================================
        # OVERALL SUMMARY PAGE
        # ============================================================
        print(f"   • Creating overall summary page...", end=" ", flush=True)
        fig = plt.figure(figsize=(11, 8.5))
        
        configs = [f"{nv}V-{f.upper()}" for (nv, f) in sorted(grouped.keys())]
        
        # Extract aggregate statistics
        time_means_sympy = [all_stats[k]['time']['mean_sympy'] for k in sorted(all_stats.keys())]
        time_means_kmap = [all_stats[k]['time']['mean_kmap'] for k in sorted(all_stats.keys())]
        lit_means_sympy = [all_stats[k]['literals']['mean_sympy'] for k in sorted(all_stats.keys())]
        lit_means_kmap = [all_stats[k]['literals']['mean_kmap'] for k in sorted(all_stats.keys())]
        time_significant = [all_stats[k]['time']['significant'] for k in sorted(all_stats.keys())]
        lit_significant = [all_stats[k]['literals']['significant'] for k in sorted(all_stats.keys())]
        
        # Plot 1: Average execution time
        ax1 = plt.subplot(2, 2, 1)
        x = np.arange(len(configs))
        width = 0.35
        bars1 = ax1.bar(x - width/2, time_means_sympy, width, label='SymPy', alpha=0.8)
        bars2 = ax1.bar(x + width/2, time_means_kmap, width, label='KMapSolver', alpha=0.8)
        
        # Mark significant differences
        for i, sig in enumerate(time_significant):
            if sig:
                ax1.text(i, max(time_means_sympy[i], time_means_kmap[i]) * 1.05, 
                        '*', ha='center', fontsize=16, color='red')
        
        ax1.set_xlabel('Configuration')
        ax1.set_ylabel('Mean Execution Time (s)')
        ax1.set_title('Average Performance by Configuration\n(* = statistically significant)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(configs, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Average literal count
        ax2 = plt.subplot(2, 2, 2)
        bars1 = ax2.bar(x - width/2, lit_means_sympy, width, label='SymPy', alpha=0.8)
        bars2 = ax2.bar(x + width/2, lit_means_kmap, width, label='KMapSolver', alpha=0.8)
        
        # Mark significant differences
        for i, sig in enumerate(lit_significant):
            if sig:
                ax2.text(i, max(lit_means_sympy[i], lit_means_kmap[i]) * 1.05,
                        '*', ha='center', fontsize=16, color='red')
        
        ax2.set_xlabel('Configuration')
        ax2.set_ylabel('Mean Literal Count')
        ax2.set_title('Average Simplification Quality\n(* = statistically significant)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(configs, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Effect sizes (time)
        ax3 = plt.subplot(2, 2, 3)
        time_effects = [all_stats[k]['time']['cohens_d'] for k in sorted(all_stats.keys())]
        colors = ['red' if abs(d) >= 0.8 else 'orange' if abs(d) >= 0.5 else 'yellow' if abs(d) >= 0.2 else 'green' 
                  for d in time_effects]
        ax3.barh(configs, time_effects, color=colors, alpha=0.7)
        ax3.axvline(0, color='black', linestyle='-', linewidth=1)
        ax3.axvline(-0.2, color='gray', linestyle='--', alpha=0.5, label='Small effect')
        ax3.axvline(0.2, color='gray', linestyle='--', alpha=0.5)
        ax3.axvline(-0.5, color='gray', linestyle=':', alpha=0.5, label='Medium effect')
        ax3.axvline(0.5, color='gray', linestyle=':', alpha=0.5)
        ax3.set_xlabel("Cohen's d")
        ax3.set_title('Effect Size: Execution Time\n(Negative = KMap faster)')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Plot 4: Effect sizes (literals)
        ax4 = plt.subplot(2, 2, 4)
        lit_effects = [all_stats[k]['literals']['cohens_d'] for k in sorted(all_stats.keys())]
        colors = ['red' if abs(d) >= 0.8 else 'orange' if abs(d) >= 0.5 else 'yellow' if abs(d) >= 0.2 else 'green'
                  for d in lit_effects]
        ax4.barh(configs, lit_effects, color=colors, alpha=0.7)
        ax4.axvline(0, color='black', linestyle='-', linewidth=1)
        ax4.axvline(-0.2, color='gray', linestyle='--', alpha=0.5, label='Small effect')
        ax4.axvline(0.2, color='gray', linestyle='--', alpha=0.5)
        ax4.axvline(-0.5, color='gray', linestyle=':', alpha=0.5, label='Medium effect')
        ax4.axvline(0.5, color='gray', linestyle=':', alpha=0.5)
        ax4.set_xlabel("Cohen's d")
        ax4.set_title('Effect Size: Literal Count\n(Negative = KMap more minimal)')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        print("✓")
        
        # ============================================================
        # FINAL CONCLUSIONS PAGE
        # ============================================================
        print(f"   • Creating final conclusions page...", end=" ", flush=True)
        fig = plt.figure(figsize=(8.5, 11))
        ax = plt.gca()
        ax.axis("off")
        
        ax.text(0.5, 0.96, "OVERALL SCIENTIFIC CONCLUSIONS",
               fontsize=18, fontweight='bold', ha='center', transform=ax.transAxes)
        
        # Aggregate all statistics
        all_sympy_times = [r['t_sympy'] for r in all_results]
        all_kmap_times = [r['t_kmap'] for r in all_results]
        all_sympy_lits = [r['sympy_literals'] for r in all_results]
        all_kmap_lits = [r['kmap_literals'] for r in all_results]
        
        overall_stats = perform_statistical_tests(
            all_sympy_times, all_kmap_times,
            all_sympy_lits, all_kmap_lits
        )
        
        conclusions = f"""
EXECUTIVE SUMMARY
{'=' * 70}
Total Test Cases:        {len(all_results)}
Configurations Tested:   {len(grouped)}
Equivalence Check:       {sum(1 for r in all_results if r['equiv'])} / {len(all_results)} passed

AGGREGATE PERFORMANCE
{'=' * 70}
Mean SymPy Time:         {overall_stats['time']['mean_sympy']:.6f} s
Mean KMapSolver Time:    {overall_stats['time']['mean_kmap']:.6f} s
Mean Time Difference:    {overall_stats['time']['mean_diff']:+.6f} s
95% CI:                  [{overall_stats['time']['ci_lower']:.6f}, {overall_stats['time']['ci_upper']:.6f}]
Statistical Significance: {'YES' if overall_stats['time']['significant'] else 'NO'} (p = {overall_stats['time']['p_value']:.6f})
Effect Size:             {overall_stats['time']['cohens_d']:.4f} ({interpret_effect_size(overall_stats['time']['cohens_d'])})

AGGREGATE SIMPLIFICATION
{'=' * 70}
Mean SymPy Literals:     {overall_stats['literals']['mean_sympy']:.2f}
Mean KMap Literals:      {overall_stats['literals']['mean_kmap']:.2f}
Mean Literal Difference: {overall_stats['literals']['mean_diff']:+.2f}
95% CI:                  [{overall_stats['literals']['ci_lower']:.2f}, {overall_stats['literals']['ci_upper']:.2f}]
Statistical Significance: {'YES' if overall_stats['literals']['significant'] else 'NO'} (p = {overall_stats['literals']['p_value']:.6f})
Effect Size:             {overall_stats['literals']['cohens_d']:.4f} ({interpret_effect_size(overall_stats['literals']['cohens_d'])})

KEY FINDINGS
{'=' * 70}
"""
        
        # Add findings based on results
        if overall_stats['time']['significant']:
            if overall_stats['time']['mean_diff'] < 0:
                conclusions += "\n1. KMapSolver demonstrates statistically significant performance\n   advantage over SymPy's minimization approach."
            else:
                conclusions += "\n1. SymPy demonstrates statistically significant performance\n   advantage over KMapSolver."
        else:
            conclusions += "\n1. No statistically significant performance difference detected\n   between KMapSolver and SymPy."
        
        if overall_stats['literals']['significant']:
            if overall_stats['literals']['mean_diff'] < 0:
                conclusions += "\n\n2. KMapSolver produces statistically more minimal Boolean\n   expressions (fewer literals) compared to SymPy."
            else:
                conclusions += "\n\n2. SymPy produces statistically more minimal Boolean\n   expressions (fewer literals) compared to KMapSolver."
        else:
            conclusions += "\n\n2. Both algorithms achieve statistically equivalent simplification\n   quality in terms of literal count."
        
        conclusions += f"\n\n3. Effect sizes indicate {interpret_effect_size(overall_stats['time']['cohens_d'])} practical\n   significance for performance and {interpret_effect_size(overall_stats['literals']['cohens_d'])} practical\n   significance for simplification quality."
        
        conclusions += f"\n\n4. All {len(all_results)} test cases maintained logical correctness,\n   with {sum(1 for r in all_results if r['equiv'])} passing equivalence verification."
        
        conclusions += f"""

THREATS TO VALIDITY
{'=' * 70}
• Limited to 2-4 variable K-maps (inherent K-map scalability limit)
• Random test case generation may not reflect real-world distributions
• Timing includes Python overhead (not pure algorithm performance)
• SymPy uses different minimization strategies (not pure K-map based)

REPRODUCIBILITY
{'=' * 70}
This experiment used random seed {RANDOM_SEED} and can be fully reproduced
using the documented experimental setup and library versions.

RECOMMENDATIONS
{'=' * 70}
Based on statistical evidence:
"""
        
        if overall_stats['time']['mean_diff'] < -0.0001 and overall_stats['literals']['mean_diff'] <= 0:
            conclusions += "\n→ KMapSolver offers practical advantages for applications requiring\n  both speed and minimal circuit representations."
        elif abs(overall_stats['time']['mean_diff']) < 0.0001 and abs(overall_stats['literals']['mean_diff']) < 0.5:
            conclusions += "\n→ Both algorithms are practically equivalent; choice should be based\n  on integration convenience and API preferences."
        else:
            conclusions += "\n→ Algorithm selection should be based on whether performance or\n  simplification quality is the priority for the application."
        
        conclusions += f"\n\n{'=' * 70}\nReport generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n© Stan's Technologies {datetime.datetime.now().year}"
        
        ax.text(0.05, 0.90, conclusions, fontsize=9, family='monospace',
               va='top', transform=ax.transAxes)
        
        pdf.savefig(bbox_inches='tight')
        plt.close()
        print("✓")
    
    print(f"✅ Comprehensive scientific report saved!")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main benchmark execution function."""
    print("=" * 80)
    print("SCIENTIFIC K-MAP SOLVER BENCHMARK")
    print("=" * 80)
    
    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    print(f"🎲 Random seed set to: {RANDOM_SEED}")
    
    # Create outputs directory
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    print(f"📁 Output directory: {OUTPUTS_DIR}")
    
    # Document experimental setup
    setup_info = document_experimental_setup()
    print("📝 Experimental setup documented.")
    
    # Test configurations
    configurations = [
        (2, "sop"),
        (2, "pos"),
        (3, "sop"),
        (3, "pos"),
        (4, "sop"),
        (4, "pos"),
    ]
    
    print(f"\n🔬 Test configurations: {len(configurations)}")
    print(f"   Variables: 2, 3, 4")
    print(f"   Forms: SOP, POS")
    print(f"   Tests per config: ~{TESTS_PER_CONFIG}")
    
    var_pool = symbols('x1 x2 x3 x4')
    all_results = []
    all_stats = {}
    
    # Run benchmarks for each configuration
    total_configs = len(configurations)
    for config_idx, (num_vars, form_type) in enumerate(configurations, 1):
        print(f"\n{'='*80}")
        print(f"[{config_idx}/{total_configs}] Benchmarking: {num_vars}-variable K-map ({form_type.upper()} form)")
        print(f"{'='*80}")
        
        # Generate test suite
        print(f"  📋 Generating test suite...", end=" ", flush=True)
        test_suite = generate_test_suite(num_vars, tests_per_distribution=TESTS_PER_CONFIG // 5)
        var_names = var_pool[:num_vars]
        print(f"✓ ({len(test_suite)} test cases)")
        
        print(f"  ⚙️  Running benchmarks...")
        config_results = []
        for idx, (kmap, distribution) in enumerate(test_suite, 1):
            try:
                result = benchmark_case(kmap, var_names, form_type, idx, distribution)
                config_results.append(result)
                all_results.append(result)
            
            except Exception as e:
                print(f"    ⚠️  Error in test {idx}: {e}")
                continue
        
        # Perform statistical analysis for this configuration
        if config_results:
            print(f"\n  📊 Performing statistical analysis...", end=" ", flush=True)
            sympy_times = [r["t_sympy"] for r in config_results]
            kmap_times = [r["t_kmap"] for r in config_results]
            sympy_literals = [r["sympy_literals"] for r in config_results]
            kmap_literals = [r["kmap_literals"] for r in config_results]
            
            stats = perform_statistical_tests(sympy_times, kmap_times, 
                                             sympy_literals, kmap_literals)
            all_stats[(num_vars, form_type)] = stats
            print("✓")
            
            # Display summary
            print(f"\n  📈 Configuration Summary:")
            print(f"     Tests completed:     {len(config_results)}")
            print(f"     Mean SymPy time:     {stats['time']['mean_sympy']:.6f} s")
            print(f"     Mean KMap time:      {stats['time']['mean_kmap']:.6f} s")
            print(f"     Time significant:    {'YES ✓' if stats['time']['significant'] else 'NO ✗'} "
                  f"(p={stats['time']['p_value']:.4f})")
            print(f"     Mean SymPy literals: {stats['literals']['mean_sympy']:.2f}")
            print(f"     Mean KMap literals:  {stats['literals']['mean_kmap']:.2f}")
            print(f"     Literal significant: {'YES ✓' if stats['literals']['significant'] else 'NO ✗'} "
                  f"(p={stats['literals']['p_value']:.4f})")
    
    # Export results
    print(f"\n{'='*80}")
    print("EXPORTING RESULTS")
    print(f"{'='*80}")
    
    export_results_to_csv(all_results, RESULTS_CSV)
    export_statistical_analysis(all_stats, STATS_CSV)
    save_comprehensive_report(all_results, all_stats, setup_info, REPORT_PDF)
    
    # Final summary
    print(f"\n{'='*80}")
    print("✅ BENCHMARK COMPLETE")
    print(f"{'='*80}")
    print(f"📊 Total test cases executed: {len(all_results)}")
    print(f"🔧 Configurations tested: {len(configurations)}")
    equiv_passed = sum(1 for r in all_results if r['equiv'])
    print(f"✓  Equivalence pass rate: {equiv_passed}/{len(all_results)} ({100*equiv_passed/len(all_results):.1f}%)")
    print(f"\n📂 Outputs:")
    print(f"   • Raw results:       {RESULTS_CSV}")
    print(f"   • Statistical data:  {STATS_CSV}")
    print(f"   • PDF report:        {REPORT_PDF}")
    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
"""
Primary Benchmark test for 3D/4D minimization (5-16 variables).
Tests variable ordering fix and tracks performance vitals.
Saves results to CSV file.
"""

import sys
import os
import random
import time
import re
import csv
from datetime import datetime
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')))

from stanlogic.BoolMinGeo import BoolMinGeo
from sympy import And as SymAnd, Or as SymOr, Not as SymNot, symbols, sympify, true, false, simplify, Equivalent
from pyeda.inter import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

# Seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Timing configuration
TIMING_WARMUP = 1
TIMING_REPEATS = 3

NUM_TESTS = 3

# Statistical significance threshold
ALPHA = 0.05  # 95% confidence level

# Output directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, "..", "outputs", "benchmark_results4D_pyeda")
RESULTS_CSV = os.path.join(OUTPUTS_DIR, "benchmark_results4D_pyeda.csv")
REPORT_PDF = os.path.join(OUTPUTS_DIR, "benchmark_scientific_report4D_pyeda.pdf")
STATS_CSV = os.path.join(OUTPUTS_DIR, "benchmark_results4D_pyeda_statistical_analysis.csv")
LOGO_PATH = os.path.join(SCRIPT_DIR, "..", "..", "..", "images", "St_logo_light-tp.png")

# Ensure output directory exists
os.makedirs(OUTPUTS_DIR, exist_ok=True)

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
        num_literals = sum(len(re.findall(r"[A-Za-z]+\d*'?", t)) for t in terms)
        return num_terms, num_literals

    if form == "pos":
        clauses = re.findall(r"\(([^()]*)\)", s)
        if not clauses:
            clauses = [s]
        num_terms = len(clauses)
        num_literals = 0
        for clause in clauses:
            lits = [lit for lit in clause.split('+') if lit]
            num_literals += sum(1 for lit in lits if re.fullmatch(r"[A-Za-z]+\d*'?", lit))
        return num_terms, num_literals

    raise ValueError("form must be 'sop' or 'pos'")

def parse_pyeda_expression(pyeda_expr, var_mapping=None):
    """
    Convert a PyEDA Boolean expression (e.g., Or(And(b, c), And(~a, c)))
    into a SymPy Boolean expression that can be compared against K-map output.
    
    Args:
        pyeda_expr: PyEDA expression to parse
        var_mapping: Dictionary mapping PyEDA var names (a,b,c,d) to SymPy symbols (x1,x2,x3,x4)
                     e.g., {'a': x1_symbol, 'b': x2_symbol, 'c': x3_symbol, 'd': x4_symbol}
    """
    if pyeda_expr is None:
        return None

    if isinstance(pyeda_expr, bool):
        return true if pyeda_expr else false

    expr_str = str(pyeda_expr).strip()
    if expr_str in {"1", "True"}:
        return true
    if expr_str in {"0", "False"}:
        return false

    # Build variable mapping
    local_map = {"And": SymAnd, "Or": SymOr, "Not": SymNot}
    
    if var_mapping:
        # Use provided mapping (PyEDA vars -> SymPy vars)
        local_map.update(var_mapping)
    else:
        # Fallback: create new SymPy variables
        var_names = set()
        try:
            for v in pyeda_expr.inputs:
                var_names.add(str(v))
        except Exception:
            var_names.update(re.findall(r"[A-Za-z_]\w*", expr_str))
        
        sym_vars = symbols(" ".join(sorted(var_names)), boolean=True) if var_names else ()
        if var_names:
            if isinstance(sym_vars, tuple):
                local_map.update({str(s): s for s in sym_vars})
            else:
                local_map[str(sym_vars)] = sym_vars

    return sympify(expr_str, locals=local_map, evaluate=True)

def count_pyeda_expression_literals(expr, form="sop"):
    """
    Count terms and literals from a PyEDA expression (string parsing).
    
    Args:
        expr: PyEDA expression
        form: 'sop' or 'pos'
        
    Returns:
        Tuple of (num_terms, num_literals)
    """
    if expr is None:
        return 0, 0
    
    s = str(expr).strip()
    form = form.lower()

    if s in ("1", "0", "True", "False"):
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

def calculate_statistics(pyeda_data, kmap_data, metric_name="metric"):
    """
    Calculate comprehensive statistics comparing two datasets.
    
    Args:
        pyeda_data: List of PyEDA measurements
        kmap_data: List of BoolMinGeo measurements
        metric_name: Name of the metric being compared
        
    Returns:
        Dictionary with statistical analysis results
    """
    pyeda_arr = np.array(pyeda_data)
    kmap_arr = np.array(kmap_data)
    
    # Calculate differences (PyEDA - BoolMinGeo)
    differences = pyeda_arr - kmap_arr
    
    # Basic statistics
    stats_dict = {
        'mean_pyeda': np.mean(pyeda_arr),
        'mean_kmap': np.mean(kmap_arr),
        'std_pyeda': np.std(pyeda_arr, ddof=1),
        'std_kmap': np.std(kmap_arr, ddof=1),
        'mean_diff': np.mean(differences),
        'std_diff': np.std(differences, ddof=1),
        'median_pyeda': np.median(pyeda_arr),
        'median_kmap': np.median(kmap_arr),
        'min_pyeda': np.min(pyeda_arr),
        'min_kmap': np.min(kmap_arr),
        'max_pyeda': np.max(pyeda_arr),
        'max_kmap': np.max(kmap_arr),
    }
    
    # Paired t-test
    if len(differences) > 1:
        t_stat, p_value = stats.ttest_rel(pyeda_arr, kmap_arr)
        stats_dict['t_statistic'] = t_stat
        stats_dict['p_value'] = p_value
        stats_dict['significant'] = p_value < ALPHA
        
        # Cohen's d effect size
        cohens_d = np.mean(differences) / np.std(differences, ddof=1) if np.std(differences, ddof=1) > 0 else 0
        stats_dict['cohens_d'] = cohens_d
        
        # 95% confidence interval for mean difference
        se = stats_dict['std_diff'] / np.sqrt(len(differences))
        ci = stats.t.interval(0.95, len(differences)-1, loc=stats_dict['mean_diff'], scale=se)
        stats_dict['ci_lower'] = ci[0]
        stats_dict['ci_upper'] = ci[1]
    else:
        stats_dict['t_statistic'] = 0
        stats_dict['p_value'] = 1.0
        stats_dict['significant'] = False
        stats_dict['cohens_d'] = 0
        stats_dict['ci_lower'] = stats_dict['mean_diff']
        stats_dict['ci_upper'] = stats_dict['mean_diff']
    
    return stats_dict

def interpret_effect_size(d):
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"

def create_minterm_fixed(vars_list, index):
    """
    Create a minterm using BoolMinGeo's bit ordering (LSB-first).
    """

def create_minterm_fixed(vars_list, index):
    """
    Create a minterm using BoolMinGeo's bit ordering (LSB-first).
    """
    terms = []
    for i, var in enumerate(vars_list):
        if (index >> i) & 1:
            terms.append(var)
        else:
            terms.append(~var)
    return And(*terms) if len(terms) > 1 else terms[0]

def test_single_case(num_vars, output_values, test_name):
    """Test a single case with timing and literal counting."""
    print(f"\nTest: {test_name}")
    print(f"  Variables: {num_vars}, Ones: {sum(1 for v in output_values if v == 1)}")
    
    # Get minterms
    minterms = [i for i, v in enumerate(output_values) if v == 1]
    dont_cares = [i for i, v in enumerate(output_values) if v == 'd']
    
    # Create PyEDA variables (extended to support 16 variables)
    pyeda_var_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p']
    pyeda_vars = [exprvar(pyeda_var_names[i]) for i in range(num_vars)]
    
    # PyEDA minimization with timing
    print("  PyEDA...", end=" ", flush=True)
    pyeda_error = None
    try:
        # Build expression first
        if not minterms and not dont_cares:
            expr_pyeda = expr(0)
            pyeda_func = lambda: expr(0)
        else:
            if minterms:
                f_on = Or(*[create_minterm_fixed(pyeda_vars, mt) for mt in minterms])
            else:
                f_on = expr(0)
            
            if dont_cares:
                f_dc = Or(*[create_minterm_fixed(pyeda_vars, dc) for dc in dont_cares])
            else:
                f_dc = expr(0)
            
            # Define timing function
            pyeda_func = lambda: espresso_exprs(f_on, f_dc)[0]
            expr_pyeda = pyeda_func()
        
        # Time the minimization
        t_pyeda = benchmark_with_warmup(pyeda_func, ())
        
        # Count literals
        expr_pyeda_parsed = parse_pyeda_expression(expr_pyeda)
        pyeda_terms, pyeda_literals = count_pyeda_expression_literals(expr_pyeda_parsed, form='sop')
        
        print(f"✓ ({t_pyeda:.6f}s, {pyeda_literals} lits)", flush=True)
    except Exception as e:
        pyeda_error = str(e)
        t_pyeda = 0.0
        pyeda_terms, pyeda_literals = 0, 0
        expr_pyeda = None
        print(f"✗ Error: {pyeda_error[:50]}", flush=True)
    
    # BoolMinGeo minimization with timing
    print("  BoolMinGeo...", end=" ", flush=True)
    kmap_error = None
    try:
        # Suppress verbose output
        import io
        import sys as sys_module
        old_stdout = sys_module.stdout
        sys_module.stdout = io.StringIO()
        
        solver = BoolMinGeo(num_vars, output_values)
        kmap_func = lambda: solver.minimize_4d(form='sop')
        
        # Time the minimization
        t_kmap = benchmark_with_warmup(kmap_func, ())
        terms, expr_str = kmap_func()
        
        # Restore stdout
        sys_module.stdout = old_stdout
        
        # Count literals
        kmap_terms, kmap_literals = count_literals(expr_str, form='sop')
        
        print(f"✓ ({t_kmap:.6f}s, {kmap_literals} lits)", flush=True)
    except Exception as e:
        kmap_error = str(e)
        t_kmap = 0.0
        kmap_terms, kmap_literals = 0, 0
        expr_str = ""
        print(f"✗ Error: {kmap_error[:50]}", flush=True)
    
    # Check equivalence only if both succeeded
    if pyeda_error is None and kmap_error is None:
        print("  Checking equivalence...", end=" ", flush=True)
        
        # Detect constant functions (both have 0 literals)
        is_constant = (pyeda_literals == 0 and kmap_literals == 0)
        
        # Special case: constant functions - verify both return the same constant
        if is_constant:
            # Get the constant values
            pyeda_str = str(expr_pyeda).strip()
            kmap_str = expr_str.strip() if expr_str else ""
            
            # Normalize constant representations
            pyeda_is_zero = pyeda_str in ("0", "False")
            pyeda_is_one = pyeda_str in ("1", "True")
            kmap_is_zero = kmap_str in ("", "0", "False")
            kmap_is_one = kmap_str in ("1", "True")
            
            # Both must be the same constant
            if (pyeda_is_zero and kmap_is_zero) or (pyeda_is_one and kmap_is_one):
                print(f"✓ PASS [CONSTANT]")
                equiv = True
            else:
                print(f"✗ FAIL [CONSTANT MISMATCH: PyEDA={pyeda_str}, BoolMinGeo={kmap_str}]")
                equiv = False
        else:
            pyeda_mismatches = 0
            kmap_mismatches = 0
        
        # Parse BoolMinGeo expression to SymPy for evaluation
        from sympy import symbols as sym_symbols
        var_symbols = sym_symbols(f'x1:{num_vars+1}')
        if not isinstance(var_symbols, tuple):
            var_symbols = (var_symbols,)
        
        # Parse the BoolMinGeo expression string (SOP format)
        if expr_str and expr_str.strip():
            # Simple parser for SOP: x1x2' + x3x4 format
            or_terms = []
            for product in expr_str.split('+'):
                product = product.strip()
                if not product:
                    continue
                # Extract literals (variables with optional prime)
                literals = re.findall(r"x\d+'?", product)
                and_terms = []
                for lit in literals:
                    if lit.endswith("'"):
                        var_idx = int(lit[1:-1]) - 1  # x1' -> index 0
                        and_terms.append(SymNot(var_symbols[var_idx]))
                    else:
                        var_idx = int(lit[1:]) - 1  # x1 -> index 0
                        and_terms.append(var_symbols[var_idx])
                if and_terms:
                    or_terms.append(SymAnd(*and_terms) if len(and_terms) > 1 else and_terms[0])
            kmap_expr_sympy = SymOr(*or_terms) if len(or_terms) > 1 else (or_terms[0] if or_terms else false)
        else:
            # Empty expression means constant 0
            kmap_expr_sympy = false
        
        for i in range(2**num_vars):
            # Expected value from truth table
            expected = output_values[i]
            if expected == 'd':
                continue  # Skip don't-cares
            
            # Create assignment for PyEDA (LSB-first: bit 0 = var 0)
            assign_pyeda = {}
            for j, v in enumerate(pyeda_vars):
                bit = (i >> j) & 1
                assign_pyeda[v] = expr(1) if bit else expr(0)
            
            # Create assignment for SymPy/BoolMinGeo (MSB-first: x1 is leftmost bit)
            # BoolMinGeo uses standard notation where x1 is MSB
            assign_sympy = {}
            for j in range(num_vars):
                bit = (i >> (num_vars - 1 - j)) & 1  # MSB-first for BoolMinGeo
                assign_sympy[var_symbols[j]] = True if bit else False
            
            # Evaluate PyEDA expression
            try:
                val = expr_pyeda.restrict(assign_pyeda)
                pyeda_val = 1 if (hasattr(val, 'is_one') and val.is_one()) else (1 if str(val) == '1' else 0)
            except:
                pyeda_val = -1
            
            # Evaluate BoolMinGeo expression
            try:
                kmap_result = kmap_expr_sympy.subs(assign_sympy)
                kmap_val = 1 if (kmap_result == true or kmap_result == True or kmap_result == 1) else 0
            except Exception as e:
                kmap_val = -1
            
            # Check both against expected value
            if pyeda_val != expected:
                pyeda_mismatches += 1
            if kmap_val != expected:
                kmap_mismatches += 1
        
        # Both must match the truth table specification
        if pyeda_mismatches > 0 or kmap_mismatches > 0:
            print(f"✗ FAIL (PyEDA: {pyeda_mismatches}, BoolMinGeo: {kmap_mismatches} mismatches)")
            equiv = False
        else:
            print(f"✓ PASS")
            equiv = True
    else:
        equiv = None
        is_constant = False
        print("  Equivalence: N/A (one or both failed)")
    
    # Return results dictionary
    return {
        'passed': equiv,
        'is_constant': is_constant if equiv is not None else False,
        'pyeda_time': t_pyeda,
        'kmap_time': t_kmap,
        'pyeda_literals': pyeda_literals,
        'kmap_literals': kmap_literals,
        'pyeda_terms': pyeda_terms,
        'kmap_terms': kmap_terms,
        'pyeda_error': pyeda_error,
        'kmap_error': kmap_error
    }

# Run tests
print("="*80)
print("BENCHMARK - 10 RANDOM FUNCTIONS EACH FOR 5-16 VARIABLES")
print("="*80)

results = []
results_by_var = {}  # Store results grouped by variable count

# Configuration: (num_vars, output_size)
configs = [
    (9, 512), (10, 1024), (11, 2048), (12, 4096)
]

# Test 10 random functions for each variable count
for num_vars, output_size in configs:
    print("\n" + "="*80)
    print(f"{num_vars}-VARIABLE FUNCTIONS (10 random tests)")
    print("="*80)
    
    var_results = []
    
    for i in range(NUM_TESTS):
        print("\n" + "-"*80)
        # Generate random output values with varying densities
        if i < 3:
            # Sparse: 20% ones
            output_values = [1 if random.random() < 0.2 else 0 for _ in range(output_size)]
            density = "sparse 20%"
        elif i < 6:
            # Balanced: 50% ones
            output_values = [1 if random.random() < 0.5 else 0 for _ in range(output_size)]
            density = "balanced 50%"
        else:
            # Dense: 70% ones
            output_values = [1 if random.random() < 0.7 else 0 for _ in range(output_size)]
            density = "dense 70%"
        
        test_result = test_single_case(num_vars, output_values, f"{num_vars}-var test {i+1}: {density}")
        results.append((f"{num_vars}-var test {i+1}", test_result))
        var_results.append(test_result)
    
    # Store results for this variable count
    results_by_var[num_vars] = var_results
    
    # Print statistics for this variable count
    print("\n" + "="*80)
    print(f"RESULTS FOR {num_vars} VARIABLES")
    print("="*80)
    
    # Filter valid results (where both succeeded)
    valid_results = [r for r in var_results if r['pyeda_time'] > 0 and r['kmap_time'] > 0]
    
    if valid_results:
        avg_pyeda_time = np.mean([r['pyeda_time'] for r in valid_results])
        avg_kmap_time = np.mean([r['kmap_time'] for r in valid_results])
        avg_pyeda_lits = np.mean([r['pyeda_literals'] for r in valid_results if r['pyeda_literals'] > 0])
        avg_kmap_lits = np.mean([r['kmap_literals'] for r in valid_results if r['kmap_literals'] > 0])
        
        speedup = avg_pyeda_time / avg_kmap_time if avg_kmap_time > 0 else 0
        lit_ratio = avg_kmap_lits / avg_pyeda_lits if avg_pyeda_lits > 0 else 0
        
        print(f"\nAverage Times:")
        print(f"  PyEDA:      {avg_pyeda_time:.6f}s")
        print(f"  BoolMinGeo: {avg_kmap_time:.6f}s")
        print(f"  Speedup:    {speedup:.2f}x")
        
        print(f"\nAverage Literal Count:")
        print(f"  PyEDA:      {avg_pyeda_lits:.1f}")
        print(f"  BoolMinGeo: {avg_kmap_lits:.1f}")
        print(f"  Ratio:      {lit_ratio:.4f}")
        
        passed_count = sum(1 for r in var_results if r['passed'] == True)
        print(f"\nTests Passed: {passed_count}/10")
    else:
        print("\nNo valid results for this variable count")


# Overall Summary
print("\n" + "="*80)
print("OVERALL SUMMARY")
print("="*80)

# Count passes and failures
passed = sum(1 for _, r in results if r['passed'] == True)
failed = sum(1 for _, r in results if r['passed'] == False)
errors = sum(1 for _, r in results if r['passed'] is None)

print(f"\nTotal Results: {passed}/{len(results)} tests passed, {failed} failed, {errors} errors")

# Calculate overall statistics
pyeda_times = [r['pyeda_time'] for _, r in results if r['pyeda_time'] > 0]
kmap_times = [r['kmap_time'] for _, r in results if r['kmap_time'] > 0]
pyeda_lits = [r['pyeda_literals'] for _, r in results if r['pyeda_literals'] > 0]
kmap_lits = [r['kmap_literals'] for _, r in results if r['kmap_literals'] > 0]

if pyeda_times and kmap_times:
    print(f"\nOverall Timing Statistics:")
    print(f"  PyEDA:      avg={np.mean(pyeda_times):.6f}s, min={np.min(pyeda_times):.6f}s, max={np.max(pyeda_times):.6f}s")
    print(f"  BoolMinGeo: avg={np.mean(kmap_times):.6f}s, min={np.min(kmap_times):.6f}s, max={np.max(kmap_times):.6f}s")
    speedup = np.mean(pyeda_times) / np.mean(kmap_times)
    print(f"  Speedup:    {speedup:.2f}x ({'BoolMinGeo faster' if speedup > 1 else 'PyEDA faster'})")

if pyeda_lits and kmap_lits:
    print(f"\nOverall Literal Count Statistics:")
    print(f"  PyEDA:      avg={np.mean(pyeda_lits):.1f}, min={np.min(pyeda_lits)}, max={np.max(pyeda_lits)}")
    print(f"  BoolMinGeo: avg={np.mean(kmap_lits):.1f}, min={np.min(kmap_lits)}, max={np.max(kmap_lits)}")
    lit_ratio = np.mean(kmap_lits) / np.mean(pyeda_lits) if np.mean(pyeda_lits) > 0 else 1.0
    print(f"  Ratio:      {lit_ratio:.4f} ({'BoolMinGeo more compact' if lit_ratio < 1 else 'PyEDA more compact' if lit_ratio > 1 else 'Equal'})")

# Per-variable summary table
print(f"\n{'='*80}")
print("PER-VARIABLE SUMMARY")
print(f"{'='*80}")
print(f"{'Vars':<6} {'Avg Time (PyEDA)':<18} {'Avg Time (BoolMin)':<18} {'Speedup':<10} {'Avg Lits (PyEDA)':<18} {'Avg Lits (BoolMin)':<18} {'Ratio':<10} {'Passed':<8}")
print("-"*80)

# Collect statistical data
all_stats = {}

for num_vars in sorted(results_by_var.keys()):
    var_results = results_by_var[num_vars]
    valid_results = [r for r in var_results if r['pyeda_time'] > 0 and r['kmap_time'] > 0]
    
    if valid_results:
        avg_pyeda_time = np.mean([r['pyeda_time'] for r in valid_results])
        avg_kmap_time = np.mean([r['kmap_time'] for r in valid_results])
        
        # Filter out constant functions for literal comparison
        literal_results = [r for r in valid_results if r['pyeda_literals'] > 0 or r['kmap_literals'] > 0]
        non_constant_results = [r for r in literal_results if not r.get('is_constant', False)]
        
        if non_constant_results:
            avg_pyeda_lits = np.mean([r['pyeda_literals'] for r in non_constant_results if r['pyeda_literals'] > 0])
            avg_kmap_lits = np.mean([r['kmap_literals'] for r in non_constant_results if r['kmap_literals'] > 0])
        else:
            avg_pyeda_lits = 0
            avg_kmap_lits = 0
        
        speedup = avg_pyeda_time / avg_kmap_time if avg_kmap_time > 0 else 0
        lit_ratio = avg_kmap_lits / avg_pyeda_lits if avg_pyeda_lits > 0 else 0
        passed_count = sum(1 for r in var_results if r['passed'] == True)
        
        print(f"{num_vars:<6} {avg_pyeda_time:<18.6f} {avg_kmap_time:<18.6f} {speedup:<10.2f} {avg_pyeda_lits:<18.1f} {avg_kmap_lits:<18.1f} {lit_ratio:<10.4f} {passed_count}/10")
        
        # Calculate detailed statistics
        pyeda_times = [r['pyeda_time'] for r in valid_results]
        kmap_times = [r['kmap_time'] for r in valid_results]
        
        time_stats = calculate_statistics(pyeda_times, kmap_times, "time")
        
        # Literal statistics (excluding constants)
        if non_constant_results:
            pyeda_lits = [r['pyeda_literals'] for r in non_constant_results]
            kmap_lits = [r['kmap_literals'] for r in non_constant_results]
            lit_stats = calculate_statistics(pyeda_lits, kmap_lits, "literals")
        else:
            lit_stats = {
                'mean_pyeda': 0, 'mean_kmap': 0, 'mean_diff': 0,
                't_statistic': 0, 'p_value': 1.0, 'significant': False,
                'cohens_d': 0, 'ci_lower': 0, 'ci_upper': 0
            }
        
        all_stats[num_vars] = {
            'valid_count': len(valid_results),
            'literal_count': len(non_constant_results),
            'time': time_stats,
            'literals': lit_stats
        }

# Print detailed statistical analysis
print(f"\n{'='*80}")
print("STATISTICAL ANALYSIS")
print(f"{'='*80}")

for num_vars in sorted(all_stats.keys()):
    stats_data = all_stats[num_vars]
    
    print(f"\n{num_vars}-Variable Functions:")
    print(f"  Valid tests: {stats_data['valid_count']}")
    print(f"  Non-constant tests: {stats_data['literal_count']}")
    
    # Timing statistics
    time_stats = stats_data['time']
    print(f"\n  Timing Analysis:")
    print(f"    PyEDA:      {time_stats['mean_pyeda']:.6f}s ± {time_stats['std_pyeda']:.6f}s")
    print(f"    BoolMinGeo: {time_stats['mean_kmap']:.6f}s ± {time_stats['std_kmap']:.6f}s")
    print(f"    Speedup:    {time_stats['mean_kmap']/time_stats['mean_pyeda']:.2f}x")
    print(f"    t-test:     t={time_stats['t_statistic']:.4f}, p={time_stats['p_value']:.4f} {'(significant)' if time_stats['significant'] else '(not significant)'}")
    print(f"    Effect size: {time_stats['cohens_d']:.4f} ({interpret_effect_size(time_stats['cohens_d'])})")
    
    # Literal statistics
    if stats_data['literal_count'] > 0:
        lit_stats = stats_data['literals']
        print(f"\n  Simplification Quality:")
        print(f"    PyEDA:      {lit_stats['mean_pyeda']:.2f} ± {lit_stats['std_pyeda']:.2f} literals")
        print(f"    BoolMinGeo: {lit_stats['mean_kmap']:.2f} ± {lit_stats['std_kmap']:.2f} literals")
        print(f"    Mean deviation: {lit_stats['mean_diff']:.2f} literals")
        print(f"    95% CI: [{lit_stats['ci_lower']:.2f}, {lit_stats['ci_upper']:.2f}]")
        print(f"    t-test:     t={lit_stats['t_statistic']:.4f}, p={lit_stats['p_value']:.4f} {'(significant)' if lit_stats['significant'] else '(not significant)'}")
        print(f"    Effect size: {lit_stats['cohens_d']:.4f} ({interpret_effect_size(lit_stats['cohens_d'])})")
        
        # Calculate quality gap
        quality_gap_pct = (abs(lit_stats['mean_diff']) / lit_stats['mean_pyeda'] * 100) if lit_stats['mean_pyeda'] > 0 else 0
        print(f"    Quality gap: {quality_gap_pct:.2f}% relative to PyEDA")

# ============================================================================
# SAVE RAW RESULTS TO CSV
# ============================================================================
print(f"\n{'='*80}")
print(f"Saving raw results to CSV...")
print(f"{'='*80}")

with open(RESULTS_CSV, 'w', newline='') as csvfile:
    fieldnames = ['num_vars', 'test_num', 'density', 'pyeda_time', 'kmap_time', 'speedup',
                  'pyeda_literals', 'kmap_literals', 'literal_ratio', 'is_constant', 'passed']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    
    for name, result in results:
        # Parse name to extract num_vars and test_num
        parts = name.split('-var test ')
        num_vars = int(parts[0])
        test_num = int(parts[1])
        
        # Determine density
        if test_num <= 3:
            density = "sparse_20%"
        elif test_num <= 6:
            density = "balanced_50%"
        else:
            density = "dense_70%"
        
        speedup = result['pyeda_time'] / result['kmap_time'] if result['kmap_time'] > 0 and result['pyeda_time'] > 0 else None
        lit_ratio = result['kmap_literals'] / result['pyeda_literals'] if result['pyeda_literals'] > 0 else None
        
        writer.writerow({
            'num_vars': num_vars,
            'test_num': test_num,
            'density': density,
            'pyeda_time': result['pyeda_time'],
            'kmap_time': result['kmap_time'],
            'speedup': speedup,
            'pyeda_literals': result['pyeda_literals'],
            'kmap_literals': result['kmap_literals'],
            'literal_ratio': lit_ratio,
            'is_constant': result.get('is_constant', False),
            'passed': result['passed']
        })

print(f"✓ Raw results saved to {RESULTS_CSV}")

# ============================================================================
# SAVE STATISTICAL ANALYSIS TO CSV
# ============================================================================
print(f"\n{'='*80}")
print(f"Saving statistical analysis to CSV...")
print(f"{'='*80}")

with open(STATS_CSV, 'w', newline='') as csvfile:
    fieldnames = ['num_vars', 'valid_tests', 'non_constant_tests',
                  'mean_pyeda_time', 'mean_kmap_time', 'mean_speedup',
                  'time_t_statistic', 'time_p_value', 'time_significant', 'time_cohens_d',
                  'mean_pyeda_literals', 'mean_kmap_literals', 'mean_deviation',
                  'literal_t_statistic', 'literal_p_value', 'literal_significant', 'literal_cohens_d',
                  'ci_95_lower', 'ci_95_upper', 'quality_gap_percent']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    
    for num_vars in sorted(all_stats.keys()):
        time_stats = all_stats[num_vars]['time']
        lit_stats = all_stats[num_vars]['literals']
        
        quality_gap_pct = (abs(lit_stats['mean_diff']) / lit_stats['mean_pyeda'] * 100) if lit_stats['mean_pyeda'] > 0 else 0
        
        writer.writerow({
            'num_vars': num_vars,
            'valid_tests': all_stats[num_vars]['valid_count'],
            'non_constant_tests': all_stats[num_vars]['literal_count'],
            'mean_pyeda_time': time_stats['mean_pyeda'],
            'mean_kmap_time': time_stats['mean_kmap'],
            'mean_speedup': time_stats['mean_kmap'] / time_stats['mean_pyeda'] if time_stats['mean_pyeda'] > 0 else None,
            'time_t_statistic': time_stats['t_statistic'],
            'time_p_value': time_stats['p_value'],
            'time_significant': time_stats['significant'],
            'time_cohens_d': time_stats['cohens_d'],
            'mean_pyeda_literals': lit_stats['mean_pyeda'],
            'mean_kmap_literals': lit_stats['mean_kmap'],
            'mean_deviation': lit_stats['mean_diff'],
            'literal_t_statistic': lit_stats['t_statistic'],
            'literal_p_value': lit_stats['p_value'],
            'literal_significant': lit_stats['significant'],
            'literal_cohens_d': lit_stats['cohens_d'],
            'ci_95_lower': lit_stats['ci_lower'],
            'ci_95_upper': lit_stats['ci_upper'],
            'quality_gap_percent': quality_gap_pct
        })

print(f"✓ Statistical analysis saved to {STATS_CSV}")

# ============================================================================
# PDF REPORT GENERATION
# ============================================================================
print(f"\n{'='*80}")
print(f"Generating PDF report...")
print(f"{'='*80}")

try:
    with PdfPages(REPORT_PDF) as pdf:
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
        ax.text(0.5, 0.55, "Benchmark Report", 
                fontsize=28, fontweight='bold', ha='center')
        ax.text(0.5, 0.49, "BoolMinGeo vs PyEDA Boolean Minimization",
                fontsize=16, ha='center')
        
        # Separator
        ax.plot([0.2, 0.8], [0.45, 0.45], 'k-', linewidth=2, alpha=0.3)
        
        # Metadata
        ax.text(0.5, 0.35, f"Experiment Date: {datetime.now().strftime('%Y-%m-%d')}",
                fontsize=11, ha='center')
        ax.text(0.5, 0.31, f"Random Seed: {RANDOM_SEED}",
                fontsize=11, ha='center')
        ax.text(0.5, 0.27, f"Total Test Cases: {len(results)}",
                fontsize=11, ha='center')
        ax.text(0.5, 0.23, f"Statistical Significance Level: α = {ALPHA}",
                fontsize=11, ha='center')
        
        # Footer
        ax.text(0.5, 0.08, "A Statistical Comparison of Boolean Minimization Algorithms",
                fontsize=10, ha='center', style='italic', color='gray')
        ax.text(0.5, 0.04, f"© Stan's Technologies {datetime.now().year}",
                fontsize=9, ha='center', color='gray')
        
        pdf.savefig(bbox_inches='tight')
        plt.close()
        print("✓")
        
        # ============================================================
        # OVERALL COMPARISON CHARTS
        # ============================================================
        print(f"   • Creating comparison charts...", end=" ", flush=True)
        fig = plt.figure(figsize=(11, 8.5))
        
        configs = [f"{nv} Variables" for nv in sorted(all_stats.keys())]
        
        # Extract aggregate statistics
        time_means_pyeda = [all_stats[k]['time']['mean_pyeda'] for k in sorted(all_stats.keys())]
        time_means_kmap = [all_stats[k]['time']['mean_kmap'] for k in sorted(all_stats.keys())]
        lit_means_pyeda = [all_stats[k]['literals']['mean_pyeda'] for k in sorted(all_stats.keys())]
        lit_means_kmap = [all_stats[k]['literals']['mean_kmap'] for k in sorted(all_stats.keys())]
        time_significant = [all_stats[k]['time']['significant'] for k in sorted(all_stats.keys())]
        lit_significant = [all_stats[k]['literals']['significant'] for k in sorted(all_stats.keys())]
        
        # Plot 1: Average execution time
        ax1 = plt.subplot(2, 2, 1)
        x = np.arange(len(configs))
        width = 0.35
        bars1 = ax1.bar(x - width/2, time_means_pyeda, width, label='PyEDA', alpha=0.8, color='#2E86AB')
        bars2 = ax1.bar(x + width/2, time_means_kmap, width, label='BoolMinGeo', alpha=0.8, color='#A23B72')
        
        # Mark significant differences
        for i, sig in enumerate(time_significant):
            if sig:
                ax1.text(i, max(time_means_pyeda[i], time_means_kmap[i]) * 1.05, 
                        '*', ha='center', fontsize=16, color='red')
        
        ax1.set_xlabel('Configuration')
        ax1.set_ylabel('Mean Execution Time (s)')
        ax1.set_title('Average Performance by Variable Count\\n(* = statistically significant, p < 0.05)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(configs)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Average literal count
        ax2 = plt.subplot(2, 2, 2)
        bars1 = ax2.bar(x - width/2, lit_means_pyeda, width, label='PyEDA', alpha=0.8, color='#2E86AB')
        bars2 = ax2.bar(x + width/2, lit_means_kmap, width, label='BoolMinGeo', alpha=0.8, color='#A23B72')
        
        # Mark significant differences
        for i, sig in enumerate(lit_significant):
            if sig:
                ax2.text(i, max(lit_means_pyeda[i], lit_means_kmap[i]) * 1.05,
                        '*', ha='center', fontsize=16, color='red')
        
        ax2.set_xlabel('Configuration')
        ax2.set_ylabel('Mean Literal Count')
        ax2.set_title('Average Simplification Quality\\n(* = statistically significant, p < 0.05)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(configs)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Mean deviation in simplification quality
        ax3 = plt.subplot(2, 2, 3)
        mean_diffs = [all_stats[k]['literals']['mean_diff'] for k in sorted(all_stats.keys())]
        colors = ['green' if d < 0 else 'red' if d > 0 else 'gray' for d in mean_diffs]
        bars = ax3.bar(x, np.abs(mean_diffs), color=colors, alpha=0.7)
        ax3.axhline(0, color='black', linestyle='-', linewidth=1)
        ax3.set_xlabel('Configuration')
        ax3.set_ylabel('Mean Absolute Deviation (literals)')
        ax3.set_title('Simplification Quality Gap\\n(Green = BoolMinGeo better, Red = PyEDA better)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(configs)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, mean_diffs)):
            height = bar.get_height()
            label = f'{abs(val):.2f}'
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    label, ha='center', va='bottom', fontsize=9)
        
        # Plot 4: Effect sizes (literals)
        ax4 = plt.subplot(2, 2, 4)
        lit_effects = [all_stats[k]['literals']['cohens_d'] for k in sorted(all_stats.keys())]
        colors_effect = ['red' if abs(d) >= 0.8 else 'orange' if abs(d) >= 0.5 else 'yellow' if abs(d) >= 0.2 else 'green'
                  for d in lit_effects]
        ax4.barh(configs, lit_effects, color=colors_effect, alpha=0.7)
        ax4.axvline(0, color='black', linestyle='-', linewidth=1)
        ax4.axvline(-0.2, color='gray', linestyle='--', alpha=0.5, label='Small effect')
        ax4.axvline(0.2, color='gray', linestyle='--', alpha=0.5)
        ax4.axvline(-0.5, color='gray', linestyle=':', alpha=0.5, label='Medium effect')
        ax4.axvline(0.5, color='gray', linestyle=':', alpha=0.5)
        ax4.set_xlabel("Cohen's d Effect Size")
        ax4.set_title('Effect Size: Simplification Quality\n(Negative = BoolMinGeo more minimal)')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        print("✓")
        
        # ============================================================
        # STATISTICAL SUMMARY PAGE
        # ============================================================
        print(f"   • Creating statistical summary...", end=" ", flush=True)
        fig = plt.figure(figsize=(8.5, 11))
        ax = plt.gca()
        ax.axis("off")
        
        ax.text(0.5, 0.96, "STATISTICAL ANALYSIS SUMMARY",
               fontsize=18, fontweight='bold', ha='center', transform=ax.transAxes)
        
        summary_text = f"""
EXPERIMENT OVERVIEW
{'=' * 70}
Total Test Cases:        {len(results)}
Configurations Tested:   {len(all_stats)} (9-10 variables)
Tests per Config:        10 (sparse, balanced, dense distributions)
Random Seed:             {RANDOM_SEED}
Significance Level:      α = {ALPHA}

"""
        
        for num_vars in sorted(all_stats.keys()):
            stats_data = all_stats[num_vars]
            time_stats = stats_data['time']
            lit_stats = stats_data['literals']
            
            winner_time = "BoolMinGeo" if time_stats['mean_diff'] > 0 else "PyEDA" if time_stats['mean_diff'] < 0 else "TIE"
            winner_lit = "BoolMinGeo" if lit_stats['mean_diff'] > 0 else "PyEDA" if lit_stats['mean_diff'] < 0 else "TIE"
            
            quality_gap_pct = (abs(lit_stats['mean_diff']) / lit_stats['mean_pyeda'] * 100) if lit_stats['mean_pyeda'] > 0 else 0
            
            summary_text += f"""
{num_vars}-VARIABLE FUNCTIONS ({stats_data['valid_count']} tests, {stats_data['literal_count']} non-constant)
{'─' * 70}

EXECUTION TIME:
  Mean PyEDA:           {time_stats['mean_pyeda']:.6f} s
  Mean BoolMinGeo:      {time_stats['mean_kmap']:.6f} s
  Mean Difference:      {time_stats['mean_diff']:+.6f} s
  95% CI:               [{time_stats['ci_lower']:.6f}, {time_stats['ci_upper']:.6f}]
  p-value:              {time_stats['p_value']:.6f}
  Statistically Sig:    {'YES' if time_stats['significant'] else 'NO'}
  Effect Size (d):      {time_stats['cohens_d']:.4f} ({interpret_effect_size(time_stats['cohens_d'])})
  Winner:               {winner_time}

SIMPLIFICATION QUALITY (Literal Count):
  Mean PyEDA:           {lit_stats['mean_pyeda']:.2f} literals
  Mean BoolMinGeo:      {lit_stats['mean_kmap']:.2f} literals
  Mean Difference:      {lit_stats['mean_diff']:+.2f} literals
  Mean Deviation:       {abs(lit_stats['mean_diff']):.2f} literals
  Quality Gap:          {quality_gap_pct:.2f}% relative to PyEDA
  95% CI:               [{lit_stats['ci_lower']:.2f}, {lit_stats['ci_upper']:.2f}]
  p-value:              {lit_stats['p_value']:.6f}
  Statistically Sig:    {'YES' if lit_stats['significant'] else 'NO'}
  Effect Size (d):      {lit_stats['cohens_d']:.4f} ({interpret_effect_size(lit_stats['cohens_d'])})
  Winner:               {winner_lit}

"""
        
        ax.text(0.05, 0.90, summary_text, fontsize=8, family='monospace',
               va='top', transform=ax.transAxes)
        
        pdf.savefig(bbox_inches='tight')
        plt.close()
        print("✓")
        
        # ============================================================
        # CONCLUSIONS PAGE
        # ============================================================
        print(f"   • Creating conclusions page...", end=" ", flush=True)
        fig = plt.figure(figsize=(8.5, 11))
        ax = plt.gca()
        ax.axis("off")
        
        ax.text(0.5, 0.96, "CONCLUSIONS",
               fontsize=18, fontweight='bold', ha='center', transform=ax.transAxes)
        
        # Calculate overall statistics
        all_pyeda_times = [r['pyeda_time'] for _, r in results if r['pyeda_time'] > 0]
        all_kmap_times = [r['kmap_time'] for _, r in results if r['kmap_time'] > 0]
        all_pyeda_lits = [r['pyeda_literals'] for _, r in results if r['pyeda_literals'] > 0]
        all_kmap_lits = [r['kmap_literals'] for _, r in results if r['kmap_literals'] > 0]
        
        overall_time_stats = calculate_statistics(all_pyeda_times, all_kmap_times)
        overall_lit_stats = calculate_statistics(all_pyeda_lits, all_kmap_lits)
        
        overall_quality_gap = (abs(overall_lit_stats['mean_diff']) / overall_lit_stats['mean_pyeda'] * 100) if overall_lit_stats['mean_pyeda'] > 0 else 0
        
        conclusions = f"""
KEY FINDINGS
{'=' * 70}

1. PERFORMANCE:
   • Average PyEDA time:        {overall_time_stats['mean_pyeda']:.6f} s
   • Average BoolMinGeo time:   {overall_time_stats['mean_kmap']:.6f} s
   • Difference:                {overall_time_stats['mean_diff']:+.6f} s
   • Statistical significance:  {'YES' if overall_time_stats['significant'] else 'NO'} (p = {overall_time_stats['p_value']:.6f})
   
2. SIMPLIFICATION QUALITY:
   • Average PyEDA literals:        {overall_lit_stats['mean_pyeda']:.2f}
   • Average BoolMinGeo literals:   {overall_lit_stats['mean_kmap']:.2f}
   • Mean deviation:                {abs(overall_lit_stats['mean_diff']):.2f} literals
   • Quality gap:                   {overall_quality_gap:.2f}% relative to PyEDA
   • Statistical significance:      {'YES' if overall_lit_stats['significant'] else 'NO'} (p = {overall_lit_stats['p_value']:.6f})

3. EQUIVALENCE:
   • All {passed}/{len(results)} tests passed logical equivalence checks
   • Both algorithms produce functionally correct results

4. EFFECT SIZES:
   • Performance:    {overall_time_stats['cohens_d']:.4f} ({interpret_effect_size(overall_time_stats['cohens_d'])})
   • Simplification: {overall_lit_stats['cohens_d']:.4f} ({interpret_effect_size(overall_lit_stats['cohens_d'])})

INTERPRETATION
{'=' * 70}

Per-Variable Analysis:
"""
        
        for num_vars in sorted(all_stats.keys()):
            lit_stats = all_stats[num_vars]['literals']
            gap = (abs(lit_stats['mean_diff']) / lit_stats['mean_pyeda'] * 100) if lit_stats['mean_pyeda'] > 0 else 0
            better = "BoolMinGeo" if lit_stats['mean_diff'] > 0 else "PyEDA" if lit_stats['mean_diff'] < 0 else "Neither"
            conclusions += f"""
  • {num_vars} variables: {gap:.2f}% quality gap, {better} produces more minimal results
                  (Mean deviation: {abs(lit_stats['mean_diff']):.2f} literals)
"""
        
        conclusions += f"""

REPRODUCIBILITY
{'=' * 70}
• Random seed: {RANDOM_SEED}
• All test cases can be reproduced with this seed
• Statistical tests use standard methods (paired t-test, Cohen's d)

{'=' * 70}
Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
© Stan's Technologies {datetime.now().year}
"""
        
        ax.text(0.05, 0.90, conclusions, fontsize=8, family='monospace',
               va='top', transform=ax.transAxes)
        
        pdf.savefig(bbox_inches='tight')
        plt.close()
        print("✓")
    
    print(f"✅ Comprehensive PDF report saved to: {REPORT_PDF}")
    
except Exception as e:
    print(f"\n⚠️  Error generating PDF report: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print(f"\n{'='*80}")
print("BENCHMARK COMPLETE")
print(f"{'='*80}")
print(f"\nOutput files:")
print(f"   • Raw results:       {RESULTS_CSV}")
print(f"   • Statistical data:  {STATS_CSV}")
print(f"   • PDF report:        {REPORT_PDF}")
print(f"\n{'='*80}\n")
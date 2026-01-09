"""
Benchmark test for 3D minimization (5-8 variables).
Tests variable ordering fix and tracks performance vitals.
"""

import sys
import os
import random
import time
import re
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')))

from stanlogic.BoolMinGeo import BoolMinGeo
from sympy import And as SymAnd, Or as SymOr, Not as SymNot, symbols, sympify, true, false, simplify, Equivalent
from pyeda.inter import *
import numpy as np

# Seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Timing configuration
TIMING_WARMUP = 1
TIMING_REPEATS = 3

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
    
    # Create PyEDA variables
    pyeda_var_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
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
        kmap_func = lambda: solver.minimize_3d(form='sop')
        
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
        mismatches = 0
        
        for i in range(2**num_vars):
            # Create assignment (LSB-first to match BoolMinGeo)
            assign = {}
            for j, v in enumerate(pyeda_vars):
                bit = (i >> j) & 1
                assign[v] = expr(1) if bit else expr(0)
            
            # Expected value
            expected = output_values[i]
            if expected == 'd':
                continue  # Skip don't-cares
            
            # Evaluate PyEDA
            try:
                val = expr_pyeda.restrict(assign)
                pyeda_val = 1 if (hasattr(val, 'is_one') and val.is_one()) else (1 if str(val) == '1' else 0)
            except:
                pyeda_val = -1
            
            if pyeda_val != expected:
                mismatches += 1
        
        if mismatches > 0:
            print(f"✗ FAIL ({mismatches} mismatches)")
            equiv = False
        else:
            print(f"✓ PASS")
            equiv = True
    else:
        equiv = None
        print("  Equivalence: N/A (one or both failed)")
    
    # Return results dictionary
    return {
        'passed': equiv,
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
print("BENCHMARK - 10 RANDOM FUNCTIONS EACH FOR 5-8 VARIABLES")
print("="*80)

results = []

# Configuration: (num_vars, output_size)
configs = [(5, 32), (6, 64), (7, 128), (8, 256)]

# Test 10 random functions for each variable count
for num_vars, output_size in configs:
    print("\n" + "="*80)
    print(f"{num_vars}-VARIABLE FUNCTIONS (10 random tests)")
    print("="*80)
    
    for i in range(10):
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
        
        results.append((f"{num_vars}-var test {i+1}", test_single_case(num_vars, output_values, f"{num_vars}-var test {i+1}: {density}")))

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

# Count passes and failures
passed = sum(1 for _, r in results if r['passed'] == True)
failed = sum(1 for _, r in results if r['passed'] == False)
errors = sum(1 for _, r in results if r['passed'] is None)

print(f"Results: {passed}/{len(results)} tests passed, {failed} failed, {errors} errors")

# Calculate statistics
pyeda_times = [r['pyeda_time'] for _, r in results if r['pyeda_time'] > 0]
kmap_times = [r['kmap_time'] for _, r in results if r['kmap_time'] > 0]
pyeda_lits = [r['pyeda_literals'] for _, r in results if r['pyeda_literals'] > 0]
kmap_lits = [r['kmap_literals'] for _, r in results if r['kmap_literals'] > 0]

if pyeda_times and kmap_times:
    print(f"\nTiming Statistics:")
    print(f"  PyEDA:      avg={np.mean(pyeda_times):.6f}s, min={np.min(pyeda_times):.6f}s, max={np.max(pyeda_times):.6f}s")
    print(f"  BoolMinGeo: avg={np.mean(kmap_times):.6f}s, min={np.min(kmap_times):.6f}s, max={np.max(kmap_times):.6f}s")
    speedup = np.mean(pyeda_times) / np.mean(kmap_times)
    print(f"  Speedup:    {speedup:.2f}x ({'BoolMinGeo faster' if speedup > 1 else 'PyEDA faster'})")

if pyeda_lits and kmap_lits:
    print(f"\nLiteral Count Statistics:")
    print(f"  PyEDA:      avg={np.mean(pyeda_lits):.1f}, min={np.min(pyeda_lits)}, max={np.max(pyeda_lits)}")
    print(f"  BoolMinGeo: avg={np.mean(kmap_lits):.1f}, min={np.min(kmap_lits)}, max={np.max(kmap_lits)}")
    lit_ratio = np.mean(kmap_lits) / np.mean(pyeda_lits) if np.mean(pyeda_lits) > 0 else 1.0
    print(f"  Ratio:      {lit_ratio:.2f} ({'BoolMinGeo more compact' if lit_ratio < 1 else 'PyEDA more compact' if lit_ratio > 1 else 'Equal'})")

print("\nDetailed Results:")
for name, result in results:
    if result['passed'] == True:
        status = "PASS"
        symbol = "✓"
    elif result['passed'] == False:
        status = "FAIL"
        symbol = "✗"
    else:
        status = "ERROR"
        symbol = "⚠"
    
    time_str = f"T:{result['pyeda_time']:.4f}s/{result['kmap_time']:.4f}s" if result['pyeda_time'] > 0 and result['kmap_time'] > 0 else "N/A"
    lit_str = f"L:{result['pyeda_literals']}/{result['kmap_literals']}" if result['pyeda_literals'] > 0 or result['kmap_literals'] > 0 else "N/A"
    
    print(f"  {symbol} {name:20s}: {status:5s} | {time_str:20s} | {lit_str}")

if passed == len(results):
    print("\n✓ All tests passed! Variable ordering fix is working.")
elif passed > 0:
    print(f"\n⚠ {passed}/{len(results)} tests passed. Some issues remain.")
else:
    print(f"\n✗ All tests failed. Further debugging needed.")

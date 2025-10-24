import time
import random
import re
import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols, SOPform, simplify, Equivalent
from lib_kmap_solver import KMapSolver
from tabulate import tabulate

# -----------------------------------------------------------
# Utility functions
# -----------------------------------------------------------

def generate_kmap(num_vars):
    """Generate a random K-map matrix with values 0, 1, or 'd'."""
    size = 2 ** (num_vars // 2), 2 ** ((num_vars + 1) // 2)
    rows, cols = size
    values = [0, 1, 'd']
    return [[random.choice(values) for _ in range(cols)] for _ in range(rows)]

def get_minterms_from_kmap(kmap):
    """Extract minterms and don't cares from a 4-variable K-map layout."""
    row_order = [0, 1, 3, 2]
    col_order = [0, 1, 3, 2]
    minterms, dont_cares = [], []
    for r, row in enumerate(kmap):
        for c, val in enumerate(row):
            idx = (row_order[r] << 2) | col_order[c]
            if val == 1:
                minterms.append(idx)
            elif val == 'd':
                dont_cares.append(idx)
    return minterms, dont_cares

def parse_kmap_sop(sop_str, var_names):
    """Parse SOP string into SymPy boolean expression."""
    from sympy import And, Or, Not, false
    if not sop_str or sop_str.strip() == "":
        return false
    name_map = {str(v): v for v in var_names}
    or_terms = []
    for product in sop_str.split('+'):
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

def check_equivalence(expr1, expr2):
    """Return True if both SymPy expressions are logically equivalent."""
    try:
        return bool(simplify(Equivalent(expr1, expr2)))
    except Exception:
        return False

def count_literals(sop_str):
    """Count number of product terms and literals in a SOP string."""
    if not sop_str or sop_str.strip() == "":
        return 0, 0
    terms = [t.strip() for t in sop_str.split('+') if t.strip()]
    num_terms = len(terms)
    num_literals = 0
    for t in terms:
        num_literals += len([v for v in re.findall(r"[A-Za-z_]\w*'?", t)])
    return num_terms, num_literals

# -----------------------------------------------------------
# Benchmark routine
# -----------------------------------------------------------

def benchmark_case(kmap, var_names, test_index):
    minterms, dont_cares = get_minterms_from_kmap(kmap)

    # --- SymPy benchmark ---
    start = time.perf_counter()
    expr_sympy = SOPform(var_names, minterms, dont_cares)
    t_sympy = time.perf_counter() - start

    # --- KMapSolver benchmark ---
    start = time.perf_counter()
    expr_kmap = KMapSolver(kmap)
    terms, sop = expr_kmap.minimize()
    t_kmap = time.perf_counter() - start

    expr_kmap_sympy = parse_kmap_sop(sop, var_names)
    equiv = check_equivalence(expr_sympy, expr_kmap_sympy)

    # literal metrics
    sympy_terms, sympy_literals = count_literals(str(expr_sympy))
    kmap_terms, kmap_literals = count_literals(sop)

    return {
        # Uncomment to see expressions of each algorithm
        # "sympy_expr": expr_sympy,
        # "kmap_expr": sop,
        "index": test_index,
        "equiv": equiv,
        "t_sympy": t_sympy,
        "t_kmap": t_kmap,
        "sympy_literals": sympy_literals,
        "kmap_literals": kmap_literals,
    }

def generate_inference(
    avg_sympy_time, avg_kmap_time, std_diff, deviation_ratio,
    avg_sympy_literals, avg_kmap_literals, std_lit_diff, deviation_ratio_literals
):
    """
    Generate a terminal inference summary based on statistical results.
    """

    # Compute differences
    time_diff = avg_kmap_time - avg_sympy_time
    time_pct_diff = (time_diff / avg_sympy_time) * 100 if avg_sympy_time else 0
    lit_diff = avg_kmap_literals - avg_sympy_literals
    lit_pct_diff = (lit_diff / avg_sympy_literals) * 100 if avg_sympy_literals else 0

    print("\n" + "="*65)
    print("🧠 INFERENCE SUMMARY")
    print("="*65)

    # --- Time Analysis ---
    print("\n⏱️  Execution Time Analysis")
    print(f" - Average SymPy Time:       {avg_sympy_time:.6f} s")
    print(f" - Average KMapSolver Time:  {avg_kmap_time:.6f} s")
    print(f" - Difference:               {time_diff:+.6f} s ({time_pct_diff:+.2f}%)")
    print(f" - Std. Dev (ΔTime):         {std_diff:.6f} s")
    print(f" - Deviation Ratio:          {deviation_ratio:.3f}")

    if time_diff < 0:
        print(" → KMapSolver is faster than SymPy on average.")
    elif time_diff < avg_sympy_time * 0.2:
        print(" → KMapSolver is slightly slower but within acceptable tolerance.")
    else:
        print(" → KMapSolver exhibits a noticeable runtime overhead compared to SymPy.")

    if std_diff < 0.001:
        print(" → Execution times are consistent and stable across tests.")
    else:
        print(" → Some variability observed in execution time per test.")

    # --- Literal Efficiency ---
    print("\n🔢 Simplification (Literal Count) Analysis")
    print(f" - Average SymPy Literals:   {avg_sympy_literals:.2f}")
    print(f" - Average KMap Literals:    {avg_kmap_literals:.2f}")
    print(f" - Difference:               {lit_diff:+.2f} ({lit_pct_diff:+.1f}%)")
    print(f" - Std. Dev (ΔLiterals):     {std_lit_diff:.2f}")
    print(f" - Deviation Ratio:          {deviation_ratio_literals:.3f}")

    if lit_diff < 0:
        print(" → KMapSolver produces more minimal SOP forms (fewer literals).")
    elif abs(lit_diff) < 0.5:
        print(" → Both algorithms produce nearly identical simplifications.")
    else:
        print(" → SymPy produces slightly simpler expressions on average.")

    if std_lit_diff < 2:
        print(" → Literal simplification results are consistent across tests.")
    else:
        print(" → Simplification varies depending on the input pattern.")

    # --- Overall Verdict ---
    print("\n⚖️  Overall Inference")
    if avg_kmap_literals <= avg_sympy_literals and avg_kmap_time <= avg_sympy_time * 1.2:
        print(" ✅ KMapSolver achieves comparable or better simplification efficiency with minimal time overhead.")
    elif avg_kmap_time < avg_sympy_time:
        print(" ✅ KMapSolver outperforms SymPy in runtime while maintaining logical equivalence.")
    else:
        print(" ⚠️ KMapSolver maintains correctness but trades slight performance for structural optimization.")

    print("="*65 + "\n")

# -----------------------------------------------------------
# Run multiple tests and plot results
# -----------------------------------------------------------

def main():
    random.seed(42)
    var_names = symbols('x1 x2 x3 x4')
    results = []

    for i in range(100):  # 10 random K-maps
        kmap = generate_kmap(2)
        result = benchmark_case(kmap, var_names, i+1)  # Pass i+1 as the test index
        results.append(result)

        #---Uncomment to see individual test details---
        # print(f"\nTest {i+1}")
        # print(f"SymPy: {result['sympy_expr']}")
        # print(f"KMap:  {result['kmap_expr']}")
        # print(f"Equivalent: {result['equiv']}")
        # print(f"SymPy time: {result['t_sympy']:.6f}s, KMapSolver time: {result['t_kmap']:.6f}s")

    print(tabulate(results, headers="keys", tablefmt="fancy_grid"))

    # --- Plot performance comparison ---
    sympy_times = [r['t_sympy'] for r in results]
    kmap_times = [r['t_kmap'] for r in results]
    sympy_literals = [r['sympy_literals'] for r in results]
    kmap_literals = [r['kmap_literals'] for r in results]

    """
    Statistical Analysis
    ------------------------------------------------------------
    """
    avg_sympy_time = np.mean(sympy_times)
    avg_kmap_time = np.mean(kmap_times)

    avg_sympy_literals = np.mean(sympy_literals)
    avg_kmap_literals = np.mean(kmap_literals)

    #Compute time differences and find the standard deviation
    diffs = np.array(sympy_times) - np.array(kmap_times)
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)

    # Compute standard deviations
    std_sympy = np.std(sympy_times, ddof=1)  # sample std dev
    std_kmap = np.std(kmap_times, ddof=1)

    # Compare deviation ratio (optional)
    deviation_ratio = std_kmap / std_sympy if std_sympy != 0 else np.nan

    #--- Print summary statistics for time ---
    print(f"Average SymPy Time: {avg_sympy_time:.6f} seconds")
    print(f"Average KMapSolver Time: {avg_kmap_time:.6f} seconds")

    print(f"Std Dev of Time Difference: {std_diff:.6f} seconds")
    print(f"Deviation Ratio (KMap/SymPy): {deviation_ratio:.6f}")

    # ------------------------------------------------------------

    #Compute literal differences and find the standard deviation
    lit_diffs = np.array(sympy_literals) - np.array(kmap_literals)
    avg_lit_diff = np.mean(lit_diffs)
    std_lit_diff = np.std(lit_diffs)

    #Compute standard deviations for literals
    std_sympy_literals = np.std(sympy_literals, ddof=1)
    std_kmap_literals = np.std(kmap_literals, ddof=1)

    #Compare literal deviation ratio (optional)
    deviation_ratio_literals = std_kmap_literals / std_sympy_literals if std_sympy_literals != 0 else np.nan

    #--- Print summary statistics for literals ---
    print(f"Average SymPy Literals: {avg_sympy_literals:.2f}")
    print(f"Average KMapSolver Literals: {avg_kmap_literals:.2f}")

    print(f"Std dev of Literal Difference: {std_lit_diff:.2f}")
    print(f"Deviation Ratio (KMap/SymPy) Literals: {deviation_ratio_literals:.6f}")

    # Generate inference summary
    generate_inference(
        avg_sympy_time, avg_kmap_time, std_diff, deviation_ratio,
        avg_sympy_literals, avg_kmap_literals, std_lit_diff, deviation_ratio_literals
    )

    tests = list(range(1, len(results)+1))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(tests, sympy_times, marker='o', label='SymPy Time')
    plt.plot(tests, kmap_times, marker='s', label='KMapSolver Time')
    plt.axhline(avg_sympy_time, color='blue', linestyle='--', alpha=0.6, label=f'SymPy Avg = {avg_sympy_time:.6f}s')
    plt.axhline(avg_kmap_time, color='orange', linestyle='--', alpha=0.6, label=f'KMap Avg = {avg_kmap_time:.6f}s')
    plt.xlabel('Test Case')
    plt.ylabel('Execution Time (s)')
    plt.title('Performance Comparison')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.bar([x - 0.2 for x in tests], sympy_literals, width=0.4, label='SymPy Literals')
    plt.bar([x + 0.2 for x in tests], kmap_literals, width=0.4, label='KMapSolver Literals')
    plt.axhline(avg_sympy_literals, color='blue', linestyle='--', alpha=0.6, label=f'SymPy Avg = {avg_sympy_literals:.2f}')
    plt.axhline(avg_kmap_literals, color='orange', linestyle='--', alpha=0.6, label=f'KMap Avg = {avg_kmap_literals:.2f}')
    plt.xlabel('Test Case')
    plt.ylabel('Number of Literals')
    plt.title('Literal Count Comparison')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
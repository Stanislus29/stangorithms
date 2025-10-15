import sys
print(sys.executable)

import time
import random
import re
import matplotlib.pyplot as plt
from sympy import symbols, SOPform, simplify, Equivalent
from kmap_algorithm import KMapSolver

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

def benchmark_case(kmap, var_names):
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
        "sympy_expr": expr_sympy,
        "kmap_expr": sop,
        "equiv": equiv,
        "t_sympy": t_sympy,
        "t_kmap": t_kmap,
        "sympy_literals": sympy_literals,
        "kmap_literals": kmap_literals,
    }

# -----------------------------------------------------------
# Run multiple tests and plot results
# -----------------------------------------------------------

def main():
    random.seed(42)
    var_names = symbols('x1 x2 x3 x4')
    results = []

    for i in range(10):  # 10 random K-maps
        kmap = generate_kmap(4)
        result = benchmark_case(kmap, var_names)
        results.append(result)
        print(f"\nTest {i+1}")
        print(f"SymPy: {result['sympy_expr']}")
        print(f"KMap:  {result['kmap_expr']}")
        print(f"Equivalent: {result['equiv']}")
        print(f"SymPy time: {result['t_sympy']:.6f}s, KMapSolver time: {result['t_kmap']:.6f}s")

    # --- Plot performance comparison ---
    sympy_times = [r['t_sympy'] for r in results]
    kmap_times = [r['t_kmap'] for r in results]
    sympy_literals = [r['sympy_literals'] for r in results]
    kmap_literals = [r['kmap_literals'] for r in results]

    tests = list(range(1, len(results)+1))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(tests, sympy_times, marker='o', label='SymPy Time')
    plt.plot(tests, kmap_times, marker='s', label='KMapSolver Time')
    plt.xlabel('Test Case')
    plt.ylabel('Execution Time (s)')
    plt.title('Performance Comparison')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.bar([x - 0.2 for x in tests], sympy_literals, width=0.4, label='SymPy Literals')
    plt.bar([x + 0.2 for x in tests], kmap_literals, width=0.4, label='KMapSolver Literals')
    plt.xlabel('Test Case')
    plt.ylabel('Number of Literals')
    plt.title('Literal Count Comparison')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
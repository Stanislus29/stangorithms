# ...existing code...
import random
import time
import re
import matplotlib.pyplot as plt
from sympy import symbols, And, Or, Not, simplify_logic, Equivalent, false
from kmap_algorithm import KMapSolver


# ---------- K-map Utilities ----------

def get_minterms_from_kmap(kmap):
    """Extract minterm and don't-care indices using KMapSolver's mapping."""
    solver = KMapSolver(kmap)
    minterms, dont_cares = [], []
    for r, row in enumerate(kmap):
        for c, val in enumerate(row):
            idx = solver._cell_to_minterm(r, c)
            if val == 1:
                minterms.append(idx)
            elif val == 'd':
                dont_cares.append(idx)
    return minterms, dont_cares


def canonical_SOP(var_names, minterms):
    """Build canonical SOP (unsimplified) from minterm indices."""
    # var_names is expected as a list of strings
    vars_tuple = symbols(" ".join(var_names))
    # ensure we always have a sequence
    if not isinstance(vars_tuple, tuple):
        vars = [vars_tuple]
    else:
        vars = list(vars_tuple)

    n = len(vars)
    terms = []

    for m in minterms:
        bits = [(m >> (n - 1 - i)) & 1 for i in range(n)]
        term_lits = [var if bit else Not(var) for var, bit in zip(vars, bits)]
        if len(term_lits) == 1:
            terms.append(term_lits[0])
        else:
            terms.append(And(*term_lits))

    return Or(*terms) if terms else false


# ---------- Random K-map Generator ----------

def random_kmap(num_vars: int, dc_prob: float = 0.1) -> list:
    """Generate a random 2D K-map with 0, 1, and 'd' values (for up to 4 variables)."""
    if num_vars not in (2, 3, 4):
        raise ValueError("Only 2-, 3-, and 4-variable K-maps are supported.")

    rows = 2 ** (num_vars // 2)
    cols = 2 ** (num_vars - num_vars // 2)

    kmap = []
    for _ in range(rows):
        row = [random.choices([0, 1, 'd'], weights=[0.6, 0.3, dc_prob])[0] for _ in range(cols)]
        kmap.append(row)
    return kmap


# ---------- Helper: parse SOP string from KMapSolver ----------

def parse_kmap_sop(sop_str, var_names):
    """
    Try to parse a SOP string returned by KMapSolver into a SymPy boolean expression.
    Handles simple formats like: "x1 x2' + x3' x4" (products separated by '+', literals
    may have a trailing apostrophe for negation).
    """
    if not sop_str:
        return false

    sym_vars = symbols(" ".join(var_names))
    if not isinstance(sym_vars, tuple):
        vars_list = [sym_vars]
    else:
        vars_list = list(sym_vars)
    name_map = {v.name: v for v in vars_list}

    or_terms = []
    try:
        product_terms = [t.strip() for t in sop_str.split('+') if t.strip()]
        for term in product_terms:
            literals = re.findall(r"[A-Za-z_]\w*'?", term)
            lits = []
            for lit in literals:
                if lit.endswith("'"):
                    varname = lit[:-1]
                    if varname in name_map:
                        lits.append(Not(name_map[varname]))
                    else:
                        raise KeyError(f"Unknown variable '{varname}' in term '{term}'")
                else:
                    if lit in name_map:
                        lits.append(name_map[lit])
                    else:
                        raise KeyError(f"Unknown variable '{lit}' in term '{term}'")
            if not lits:
                continue
            if len(lits) == 1:
                or_terms.append(lits[0])
            else:
                or_terms.append(And(*lits))
        return Or(*or_terms) if or_terms else false
    except Exception:
        # fallback: try a simple textual replacement and let simplify_logic handle it
        try:
            fallback = sop_str.replace("'", " not ")
            return simplify_logic(fallback, form="dnf")
        except Exception:
            return false


# ---------- Benchmark Runner ----------

def benchmark_case(kmap, var_names):
    # Extract minterms & don't-cares
    minterms, dont_cares = get_minterms_from_kmap(kmap)

    # --- Canonical SOP (SymPy) ---
    start = time.perf_counter()
    expr_sympy_canonical = canonical_SOP(var_names, minterms)
    expr_sympy_simplified = simplify_logic(expr_sympy_canonical, form='dnf')
    t_sympy = time.perf_counter() - start

    # --- KMapSolver ---
    start = time.perf_counter()
    solver = KMapSolver(kmap)
    try:
        terms, sop_str = solver.minimize()
    except Exception:
        # If minimize fails, record and continue
        terms, sop_str = None, ""
    t_kmap = time.perf_counter() - start

    # --- Convert KMap output to SymPy expr ---
    expr_kmap_sympy = parse_kmap_sop(sop_str, var_names)

    # --- Check equivalence ---
    try:
        equivalent = Equivalent(expr_sympy_simplified, expr_kmap_sympy).simplify()
    except Exception:
        # fallback boolean check via simplify of XOR (if either parse failed)
        try:
            xor = simplify_logic(expr_sympy_simplified ^ expr_kmap_sympy, form="dnf")
            equivalent = (xor == false)
        except Exception:
            equivalent = False

    return {
        "sympy_expr": expr_sympy_simplified,
        "kmap_expr": sop_str,
        "equiv": equivalent,
        "t_sympy": t_sympy,
        "t_kmap": t_kmap
    }

def count_literals(expr):
    """Count the total number of literals in a SymPy SOP expression."""
    if expr == false:
        return 0
    expr = simplify_logic(expr, form='dnf')
    # flatten Or terms
    if expr.func == Or:
        terms = expr.args
    else:
        terms = [expr]
    count = 0
    for t in terms:
        if t.func == And:
            count += len(t.args)
        else:
            count += 1
    return count

def count_literals_kmap(sop_str, var_names):
    """Count the exact number of literals in a KMapSolver SOP string."""
    if not sop_str:
        return 0
    sym_vars = symbols(" ".join(var_names))
    if not isinstance(sym_vars, tuple):
        vars_list = [sym_vars]
    else:
        vars_list = list(sym_vars)
    name_map = {v.name: v for v in vars_list}

    or_terms = [t.strip() for t in sop_str.split('+') if t.strip()]
    count = 0
    for term in or_terms:
        literals = re.findall(r"[A-Za-z_]\w*'?", term)
        count += len(literals)
    return count

def run_benchmark_with_plots(num_tests_per_var=3):
    """Run benchmark and produce matplotlib plots for time and literal count (accurate)."""
    random.seed(42)
    all_results = []

    for num_vars in [2, 3, 4]:
        var_names = [f"x{i+1}" for i in range(num_vars)]
        times_sympy, times_kmap = [], []
        literals_sympy, literals_kmap = [], []

        print(f"\n=== Benchmarking {num_tests_per_var} random {num_vars}-variable K-maps ===\n")
        for i in range(1, num_tests_per_var + 1):
            kmap = random_kmap(num_vars)
            result = benchmark_case(kmap, var_names)
            all_results.append((num_vars, result))

            # Count literals accurately
            lit_sympy = count_literals(result['sympy_expr'])
            lit_kmap = count_literals_kmap(result['kmap_expr'], var_names)
            literals_sympy.append(lit_sympy)
            literals_kmap.append(lit_kmap)
            times_sympy.append(result['t_sympy'])
            times_kmap.append(result['t_kmap'])

            print(f"Test {i}")
            print(f"SymPy: {result['sympy_expr']}")
            print(f"KMap:  {result['kmap_expr']}")
            print(f"Equivalent: {result['equiv']}")
            print(f"SymPy time: {result['t_sympy']:.6f}s, KMapSolver time: {result['t_kmap']:.6f}s")
            print(f"Literals -> SymPy: {lit_sympy}, KMapSolver: {lit_kmap}\n")

        # Plot time comparison
        plt.figure(figsize=(10, 4))
        plt.plot(range(1, num_tests_per_var+1), times_sympy, 'o-', label='SymPy')
        plt.plot(range(1, num_tests_per_var+1), times_kmap, 's-', label='KMapSolver')
        plt.title(f"Runtime Comparison ({num_vars}-variable K-maps)")
        plt.xlabel("Test Case")
        plt.ylabel("Time (s)")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot literal count comparison (accurate)
        plt.figure(figsize=(10, 4))
        plt.bar([x-0.15 for x in range(1, num_tests_per_var+1)], literals_sympy, width=0.3, label='SymPy')
        plt.bar([x+0.15 for x in range(1, num_tests_per_var+1)], literals_kmap, width=0.3, label='KMapSolver')
        plt.title(f"Literal Count Comparison ({num_vars}-variable K-maps)")
        plt.xlabel("Test Case")
        plt.ylabel("Number of Literals")
        plt.xticks(range(1, num_tests_per_var+1))
        plt.legend()
        plt.grid(axis='y')
        plt.show()

    # --- Summary ---
    print("\n=== Summary ===")
    print(f"{'Vars':<6}{'SymPy(s)':<12}{'KMap(s)':<12}{'Equivalent'}")
    for i, (num_vars, res) in enumerate(all_results, start=1):
        print(f"{num_vars:<6}{res['t_sympy']:<12.6f}{res['t_kmap']:<12.6f}{res['equiv']}")

# ---------- Run ----------
if __name__ == "__main__":
    run_benchmark_with_plots(num_tests_per_var=10)
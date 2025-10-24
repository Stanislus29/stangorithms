from kmap_algorithm import KMapSolver
from tabulate import tabulate 
from sympy import symbols, And, Or, Not, simplify_logic, Equivalent, false
import matplotlib.pyplot as plt 
import numpy as np
import time
import re

#------------ Examples extracted from Brown and Vranesic ----------
"""
Reference:
Test cases adapted from: S. Brown and Z. Vranesic, *Digital Logic with Verilog Design*, McGraw-Hill.
"""
# -Chapter 2, 2.14a, p. 95.
test1 = [
    [0, 1, 'd', 0],
    [0, 1, 'd', 0],
    [0, 0, 'd', 0],
    [1, 1, 'd', 1]
]
test1_equivalent = "x3x4' + x2x3'"

# -Chapter 2, 2.11, p. 85. 
test2 = [
    [0, 0, 0, 0],
    [0, 0, 1, 1],
    [1, 0, 0, 1],
    [1, 0, 0, 1]
]
test2_equivalent = "x2'x3 + x1x3'x4"

test3 = [
    [0, 0, 0, 0],
    [0, 0, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1]
]
test3_equivalent = "x3 + x1x4"

test4 = [
    [1, 0, 0, 1],
    [0, 0, 0, 0],
    [1, 1, 1, 0],
    [1, 1, 0, 1]
]
test4_equivalent = "x2'x4' + x1'x3 + x2x3x4"

test5 = [
    [1, 1, 1, 0],
    [1, 1, 1, 0],
    [0, 0, 1, 1],
    [0, 0, 1, 1]
]
test5_equivalent = "x1'x3' + x1x3 + x1x2"

# -Chapter 2, Example 2.27, p. 106.
test6 = [
    [1, 1, 'd', 0],
    [1, 0, 1, 'd'],
    [1, 1, 1, 1],
    [0, 0, 'd', 0]
]
test6_equivalent = "x3x4' + x1'x3'x4' + x2x4 +x1x4"

# -Chapter 2, Example 2.26a, p. 105.
test7 = [
    [0, 1, 1, 1],
    [0, 'd', 0, 'd'],
    ['d', 'd', 1, 1],
    [0, 1, 0, 'd']
]
test7_equivalent = "x1'x2 + x1x2' + x3x4 + x1x3'x4'"

#------------ Examples extracted from Mano and Kine ----------
"""
Reference:
Test cases adapted from: M. Mano and C. Kine, *Logic and Computer Design Fundamentals*, Pearson.
"""

# -Chapter 2, Example 8, p. 65.
test8 = [
    ['d', 1, 1, 'd'],
    [0, 'd', 1, 0],
    [0, 0, 1, 0],
    [1, 0, 1, 0]
]
test8_equivalent = "x3x4 + x1'x4"

# -Chapter 2, Example 9, p. 66.
test9 = [
    [1, 1, 1, 1],
    [1, 1, 1, 0],
    [0, 0, 1, 0],
    [1, 0, 1, 1]
]
test9_equivalent = "x2'x4' + x1'x3' + x3x4"

# -Chapter 2, Example 9, p. 66.
test10 = [
    ['d', 1, 1, 'd'],
    [0, 'd', 1, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 0]
]
test10_equivalent = "x3x4 + x1'x2'"

kmaps = [test1, test2, test3, test4, test5, 
         test6, test7, test8, test9, test10]

equivalents = [test1_equivalent, test2_equivalent, test3_equivalent, test4_equivalent, 
               test5_equivalent, test6_equivalent, test7_equivalent, test8_equivalent,
               test9_equivalent, test10_equivalent]

results = []

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


var_names = ['x1', 'x2', 'x3', 'x4']
literals_kmap, literals_equiv = [], []

for idx, kmap in enumerate(kmaps):
    solver = KMapSolver(kmap)
    start_time = time.perf_counter()
    terms, sop = solver.minimize()
    duration = time.perf_counter() - start_time

    # Parse both the solver's output and expected equivalent
    expr_kmap = parse_kmap_sop(sop, var_names)
    expr_equiv = parse_kmap_sop(equivalents[idx], var_names)

    # expr_kmap = simplify_logic(expr_kmap, form="dnf")
    # expr_equiv = simplify_logic(expr_equiv, form="dnf")

    # Literal counts
    lit_equiv = count_literals_kmap(equivalents[idx], var_names)
    lit_kmap = count_literals_kmap(sop, var_names)
    literals_equiv.append(lit_equiv)
    literals_kmap.append(lit_kmap)

    # Use SymPy equivalence checking
    try:
        equivalent = Equivalent(expr_kmap, expr_equiv).simplify()
    except Exception:
        # Fallback: use XOR-based equivalence check
        try:
            diff = simplify_logic(expr_kmap ^ expr_equiv, form="dnf")
            equivalent = (diff == false)
        except Exception:
            equivalent = False

    print(f"Test {idx + 1} SOP: {sop}")
    
    results.append({
    'Test': idx + 1,
    'Time Taken (s)': duration,
    'Equivalent': equivalent,
    'Literals (KMap)': lit_kmap,
    'Literals (Expected)': lit_equiv
    })

# --- Prepare data ---
tests = [r['Test'] for r in results]
times = [r['Time Taken (s)'] for r in results]
lits_kmap = [r['Literals (KMap)'] for r in results]
lits_equiv = [r['Literals (Expected)'] for r in results]

avg_time = np.mean(times)
avg_kmap_lits = np.mean(lits_kmap)
avg_equiv_lits = np.mean(lits_equiv)

print(f"Average Time Taken: {avg_time:.6f} seconds")
print(tabulate(results, headers="keys", tablefmt="fancy_grid"))

# --- Create side-by-side plots ---
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# --- Plot 1: Execution Time ---
axs[0].plot(tests, times, marker='o', color='tab:blue', label='KMapSolver Time')
axs[0].axhline(avg_time, color='red', linestyle='--', alpha=0.7, label=f'Avg Time = {avg_time:.6f}s')
axs[0].set_title('Execution Time per Test')
axs[0].set_xlabel('Test Number')
axs[0].set_ylabel('Time (seconds)')
axs[0].grid(True, linestyle='--', alpha=0.6)
axs[0].legend()

# --- Plot 2: Literal Comparison ---
axs[1].plot(tests, lits_equiv, marker='o', color='tab:orange', label='Expected Literals')
axs[1].plot(tests, lits_kmap, marker='s', color='tab:green', label='KMapSolver Literals')
axs[1].axhline(avg_equiv_lits, color='orange', linestyle='--', alpha=0.6, label=f'Expected Avg = {avg_equiv_lits:.2f}')
axs[1].axhline(avg_kmap_lits, color='green', linestyle='--', alpha=0.6, label=f'KMap Avg = {avg_kmap_lits:.2f}')
axs[1].set_title('Literal Count Comparison')
axs[1].set_xlabel('Test Number')
axs[1].set_ylabel('Number of Literals')
axs[1].grid(True, linestyle='--', alpha=0.6)
axs[1].legend()

plt.tight_layout()
plt.show()
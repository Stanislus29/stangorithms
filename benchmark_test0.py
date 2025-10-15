from sympy import symbols, SOPform, POSform, simplify, Equivalent
from kmap_algorithm import KMapSolver

import time

# Example: 3-variable K-map with don't cares
kmap4 = [
    [0, 1, 'd', 0],
    [0, 1, 'd', 0],
    [0, 0, 'd', 0],
    [1, 1, 'd', 1]
]

# SymPy equivalent
# Define the minterms and don't cares
minterms = [4, 5, 2, 6, 10]
dont_cares = [12,13,14,15]

start = time.time()
expr_sympy = SOPform(['x1', 'x2', 'x3', 'x4'], minterms, dont_cares)
t_sympy = time.time() - start
print("SymPy Simplified SOP:", expr_sympy)

#--------------------------------------
start = time.time()
expr_kmap = KMapSolver(kmap4)
# expr_kmap.print_kmap()
terms, sop = expr_kmap.minimize()
t_kmap = time.time() - start
print("\n4-var result:", terms, sop)

print(f"SymPy took {t_sympy:.6f} seconds and KMapSolver took {t_kmap:.6f} seconds")
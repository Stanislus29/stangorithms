#Demo program using the library "lib_kmap_solver.py"

from lib_kmap_solver import KMapSolver

# Example: 3-variable K-map with don't cares
kmap4 = [
    [0, 1, 'd', 0],
    [0, 1, 'd', 0],
    [0, 0, 'd', 0],
    [1, 1, 'd', 1]
]

kmap4a = [
    [0, 0, 0, 0],
    [0, 0, 1, 1],
    [1, 0, 0, 1],
    [1, 0, 0, 1]
]

solver3 = KMapSolver(kmap4)
solver3.print_kmap()
terms, sop = solver3.minimize()
print("\n4-var result:", terms, sop)

solver4 = KMapSolver(kmap4a)
solver4.print_kmap()
terms, sop = solver4.minimize()
print("\n4-var result:", terms, sop)

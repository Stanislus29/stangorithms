#Demo program using the library "lib_kmap_solver.py"

# Author: Somtochukwu Stanislus Emeka-Onwuneme 
# Publication Date: 8th September, 2025
# Copyright © 2025 Somtochukwu Stanislus Emeka-Onwuneme
#---------------------------------------------------------

from stanlogic import KMapSolver

# Example: 3-variable K-map with don't cares
kmap4 = [
    [0, 1, 'd', 0],
    [0, 1, 'd', 0],
    [0, 0, 'd', 0],
    [1, 1, 'd', 1]
]

# kmap4a = [
#     [0, 0, 0, 0],
#     [0, 0, 1, 1],
#     [1, 0, 0, 1],
#     [1, 0, 0, 1]
# ]

pos_test = KMapSolver(kmap4)
pos_test.print_kmap()
terms, pos = pos_test.minimize(form ="pos")
print("\n4-var result:", terms, pos)

sop_test = KMapSolver(kmap4)
sop_test.print_kmap()
terms, sop = sop_test.minimize(form ="sop")
print("\n4-var result:", terms, sop)

# solver4 = KMapSolver(kmap4a)
# solver4.print_kmap()
# terms, sop = solver4.minimize()
# print("\n4-var result:", terms, sop)

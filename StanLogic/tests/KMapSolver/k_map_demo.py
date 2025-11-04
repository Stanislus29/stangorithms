#Demo program using the library "lib_kmap_solver.py"

# Author: Somtochukwu Stanislus Emeka-Onwuneme 
# Publication Date: 8th September, 2025
# Copyright © 2025 Somtochukwu Stanislus Emeka-Onwuneme
#---------------------------------------------------------

from stanlogic import KMapSolver

# Example: 4-variable K-map with don't cares
kmap4 = [
    [0, 1, 'd', 0],
    [0, 1, 'd', 0],
    [0, 0, 'd', 0],
    [1, 1, 'd', 1]
]

pos_test = KMapSolver(kmap4)
pos_test.print_kmap()
terms, pos = pos_test.minimize(form ="pos")
print("\n4-var result:", terms, pos)

sop_test = KMapSolver(kmap4)
sop_test.print_kmap()
terms, sop = sop_test.minimize(form ="sop")
print("\n4-var result:", terms, sop)

# ---------------Using the "mano_kime" convention-----------------
kmap = [
    [1, 1, 0, 0],
    [1, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 0, 1, 1]
]

solver = KMapSolver(kmap, convention="mano_kime")
solver.print_kmap()
terms, sop2 = solver.minimize(form="sop")
terms_pos2, pos2 = solver.minimize(form="pos")
print("Minimal SOP:", sop2)
print("Minimal POS:", pos2)
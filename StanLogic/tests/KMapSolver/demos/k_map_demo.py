#Demo program using the library "lib_kmap_solver.py"

# Author: Somtochukwu Stanislus Emeka-Onwuneme 
# Publication Date: 8th September, 2025
# Copyright © 2025 Somtochukwu Stanislus Emeka-Onwuneme
#---------------------------------------------------------

from stanlogic import BoolMin2D

# # Example: 4-variable K-map with don't cares
# kmap4 = [
#     [0, 1, 'd', 0],
#     [0, 1, 'd', 0],
#     [0, 0, 'd', 0],
#     [1, 1, 'd', 1]
# ]

# pos_test = BoolMin2D(kmap4)
# pos_test.print_kmap()
# terms, pos = pos_test.minimize(form ="pos")
# print("\n4-var result:", terms, pos)

# sop_test = BoolMin2D(kmap4)
# sop_test.print_kmap()
# terms, sop = sop_test.minimize(form ="sop")
# print("\n4-var result:", terms, sop)

# ---------------Using the "mano_kime" convention-----------------
kmap = [
    ['d', 1, 0, 1],
    [0, 0, 'd', 'd'],
    ['d', 1, 0, 1],
    [1, 0, 0, 0]
]

solver = BoolMin2D(kmap)
solver.generate_html_report(form="sop", filename="kmap_report.html")
print("HTML report generated: kmap_report.html")
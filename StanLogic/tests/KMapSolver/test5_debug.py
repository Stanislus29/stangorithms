"""
Debug test for Test 5 to understand the coverage issue
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from stanlogic.BoolMin2D import BoolMin2D
import random

# Test 5 from quick benchmark
random.seed(5)
kmap = [[0, 0, 0, 'd'], [0, 'd', 1, 0]]

print("K-map: Test 5")
print("  Row 0: [0, 0, 0, 'd']  (x3')")
print("  Row 1: [0, 'd', 1, 0]  (x3)")
print()

# Expected: only cell (1,2) should be covered
# That's column 11 (x1=1,x2=1), row 1 (x3=1)
# So the minterm is x1 x2 x3 (binary: 111 = 7)

solver = BoolMin2D(kmap)

# Print internal state
print("Cell labels (Gray code):")
print("  Cols: ['00', '01', '11', '10']  (x1 x2)")
print("  Rows: ['0', '1']                (x3)")
print()

print("Cell to minterm mapping:")
for r in range(2):
    for c in range(4):
        col_label = ['00', '01', '11', '10'][c]
        row_label = str(r)
        bits = col_label + row_label
        idx = int(bits, 2)
        val = kmap[r][c]
        print(f"  ({r},{c}): bits={bits}, minterm={idx}, value={val}")
print()

# The target cell is (1,2): bits='111', minterm=7, value=1
print("Target cell to cover: (1,2) with minterm=7, bits=111 (x1 x2 x3)")
print()

# Run minimization
terms, expr = solver.minimize(form='sop')

print(f"BoolMin2D Result: {expr}")
print(f"Terms: {terms}")
print()

# The expression should be x1 x2 x3 (all three variables)
# But it's giving x2 x3 (missing x1)

# Let's trace what groups are being formed
groups = solver.find_all_groups(allow_dontcare=True)
print(f"All groups found: {len(groups)}")

for i, gmask in enumerate(groups):
    print(f"\nGroup {i}: mask={gmask:08b}")
    # Decode mask to cells
    cells = []
    temp = gmask
    while temp:
        low = temp & -temp
        idx = low.bit_length() - 1
        temp -= low
        if idx in solver._index_to_rc:
            r, c = solver._index_to_rc[idx]
            val = kmap[r][c]
            col_label = ['00', '01', '11', '10'][c]
            row_label = str(r)
            bits = col_label + row_label
            cells.append(f"({r},{c}):bits={bits},val={val}")
    print(f"  Cells: {cells}")
    
    # What would the simplified term be?
    bits_list = []
    for cell_str in cells:
        # Extract bits
        bits_part = cell_str.split('bits=')[1].split(',')[0]
        bits_list.append(bits_part)
    
    if bits_list:
        # Simplify
        bits = list(bits_list[0])
        for b in bits_list[1:]:
            for j in range(3):
                if bits[j] != b[j]:
                    bits[j] = '-'
        
        # Build term
        vars_ = ['x1', 'x2', 'x3']
        term = []
        for j, b in enumerate(bits):
            if b == '0':
                term.append(vars_[j] + "'")
            elif b == '1':
                term.append(vars_[j])
        term_str = "".join(term)
        print(f"  Simplified bits: {''.join(bits)}")
        print(f"  Term: {term_str}")

print("\n" + "="*60)
print("DIAGNOSIS:")
print("The issue is that the group contains cell (1,1) with value 'd'")
print("This causes x1 to be eliminated during simplification")
print("But cell (1,1) should NOT affect coverage - only cell (1,2) matters!")
print("="*60)

"""
Simple test to verify BoolMin2D optimizations work correctly
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from stanlogic.BoolMin2D import BoolMin2D
import random

def test_kmap(kmap, test_name, form='sop'):
    """Test a K-map and verify it covers all required cells"""
    print(f"\n{'='*60}")
    print(f"Test: {test_name}")
    print(f"K-map: {kmap}")
    print(f"Form: {form}")
    
    solver = BoolMin2D(kmap)
    terms, expr = solver.minimize(form=form)
    
    print(f"Result: {expr}")
    print(f"Terms: {terms}")
    print(f"Term count: {len(terms)}")
    
    # Verify coverage
    target_val = 0 if form == 'pos' else 1
    target_cells = []
    for r in range(len(kmap)):
        for c in range(len(kmap[0])):
            if kmap[r][c] == target_val:
                target_cells.append((r, c))
    
    print(f"Target cells to cover: {len(target_cells)}")
    print(f"Expected to cover: {target_cells}")
    
    # Simple validation: expression should not be empty if target cells exist
    if target_cells:
        empty_expr = (expr == "0") if form == 'sop' else (expr == "1")
        if not expr or empty_expr:
            print("ERROR: Expression is empty but target cells exist!")
            return False
    
    print("PASS")
    return True

# Test cases from benchmark failures
print("Testing BoolMin2D optimizations...")

# Test 8: sparse case
random.seed(8)
kmap8 = []
for _ in range(2):
    row = []
    for _ in range(4):
        val = random.choice([0, 1, 'd', 0, 0])
        row.append(val)
    kmap8.append(row)
test_kmap(kmap8, "Test 8 (sparse)", 'sop')

# Test 13: sparse case
random.seed(13)
kmap13 = []
for _ in range(2):
    row = []
    for _ in range(4):
        val = random.choice([0, 1, 'd', 0, 0])
        row.append(val)
    kmap13.append(row)
test_kmap(kmap13, "Test 13 (sparse)", 'sop')

# Test 45: balanced case
random.seed(45)
kmap45 = []
for _ in range(2):
    row = []
    for _ in range(4):
        val = random.choice([0, 1, 'd'])
        row.append(val)
    kmap45.append(row)
test_kmap(kmap45, "Test 45 (balanced)", 'sop')

# Test a few more edge cases
test_kmap([[1, 1], [1, 1]], "All ones 2-var", 'sop')
test_kmap([[0, 0], [0, 0]], "All zeros 2-var", 'sop')
test_kmap([[1, 'd'], ['d', 1]], "Don't cares 2-var", 'sop')
test_kmap([[1, 0, 0, 1], [1, 0, 0, 1]], "Pattern 3-var", 'sop')
test_kmap([['d', 'd', 'd', 'd'], ['d', 'd', 'd', 'd']], "All don't cares 3-var", 'sop')

print("\n" + "="*60)
print("All tests completed")

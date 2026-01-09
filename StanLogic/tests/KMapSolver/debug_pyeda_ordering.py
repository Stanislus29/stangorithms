"""Debug script to verify PyEDA variable ordering."""

from pyeda.inter import *

# Create 3 variables
x1, x2, x3 = map(exprvar, ['x1', 'x2', 'x3'])

# Test case: x1 = 1 (all odd minterms in 3-var truth table)
# Truth table indices: 0,1,2,3,4,5,6,7
# Binary:              000,001,010,011,100,101,110,111
# If LSB-first (x1=bit0): x1=1 means bit 0 set → indices 1,3,5,7
# If MSB-first (x1=bit2): x1=1 means bit 2 set → indices 4,5,6,7

print("Testing PyEDA variable ordering")
print("="*60)

# Create minterm for index 1 (binary 001)
# LSB-first: bit0=1, bit1=0, bit2=0 → x1=1, x2=0, x3=0 → x1 & ~x2 & ~x3
# MSB-first: bit0=0, bit1=0, bit2=1 → x1=0, x2=0, x3=1 → ~x1 & ~x2 & x3

def create_minterm_lsb(index, vars_list):
    """LSB-first: bit 0 = first variable."""
    terms = []
    for i, var in enumerate(vars_list):
        bit = (index >> i) & 1
        terms.append(var if bit else ~var)
    return And(*terms)

def create_minterm_msb(index, vars_list):
    """MSB-first: bit 0 = last variable."""
    n = len(vars_list)
    terms = []
    for i, var in enumerate(vars_list):
        bit = (index >> (n-1-i)) & 1
        terms.append(var if bit else ~var)
    return And(*terms)

vars_list = [x1, x2, x3]

print("\nTesting minterm for index=1 (binary 001):")
print(f"  LSB-first approach: {create_minterm_lsb(1, vars_list)}")
print(f"  MSB-first approach: {create_minterm_msb(1, vars_list)}")

# Now test which one PyEDA uses internally
# Create expression x1 (just the variable) and test all 8 inputs
print("\nTesting PyEDA expression 'x1' against all 8 inputs:")
expr_x1 = x1

for i in range(8):
    # Try LSB-first assignment
    lsb_assign = {}
    for j, var in enumerate(vars_list):
        bit = (i >> j) & 1
        lsb_assign[var] = expr(1) if bit else expr(0)
    
    # Try MSB-first assignment  
    msb_assign = {}
    for j, var in enumerate(vars_list):
        bit = (i >> (2-j)) & 1
        msb_assign[var] = expr(1) if bit else expr(0)
    
    lsb_result = expr_x1.restrict(lsb_assign)
    msb_result = expr_x1.restrict(msb_assign)
    
    lsb_val = 1 if (hasattr(lsb_result, 'is_one') and lsb_result.is_one()) else 0
    msb_val = 1 if (hasattr(msb_result, 'is_one') and msb_result.is_one()) else 0
    
    print(f"  Index {i} (binary {i:03b}): LSB-first={lsb_val}, MSB-first={msb_val}")

print("\nExpected for 'x1':")
print("  If x1 corresponds to bit 0 (LSB): should be 1 for indices 1,3,5,7")
print("  If x1 corresponds to bit 2 (MSB): should be 1 for indices 4,5,6,7")

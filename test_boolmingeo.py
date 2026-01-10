"""
Test BoolMinGeo with the fixed coverage and Quine-McCluskey optimization
"""
from StanLogic.src.stanlogic.BoolMinGeo import BoolMinGeo

def test_6var_simple():
    """Test x1'x2' function"""
    print("="*70)
    print("TEST 1: 6-variable function x1'x2'")
    print("="*70)
    
    num_vars = 6
    output_values = [0] * 64
    
    # x1'x2' means bits[0:2] == '00'
    for i in range(64):
        bits = format(i, '06b')
        if bits[0:2] == '00':
            output_values[i] = 1
    
    print(f"Minterms: {sum(output_values)}")
    
    solver = BoolMinGeo(num_vars, output_values)
    terms, expr = solver.minimize_3d(form='sop')
    
    print(f"\nResult: {expr}")
    print(f"Expected: x1'x2'")
    print()
    
    return expr

def test_6var_or():
    """Test x1 + x2 function"""
    print("="*70)
    print("TEST 2: 6-variable function x1 + x2")
    print("="*70)
    
    num_vars = 6
    output_values = [0] * 64
    
    # x1 + x2 means bit[0]=='1' OR bit[1]=='1'
    for i in range(64):
        bits = format(i, '06b')
        if bits[0] == '1' or bits[1] == '1':
            output_values[i] = 1
    
    print(f"Minterms: {sum(output_values)}")
    
    solver = BoolMinGeo(num_vars, output_values)
    terms, expr = solver.minimize_3d(form='sop')
    
    print(f"\nResult: {expr}")
    print(f"Expected: x1 + x2")
    print()
    
    return expr

def test_6var_complex():
    """Test x1'x2' + x3'x4' function"""
    print("="*70)
    print("TEST 3: 6-variable function x1'x2' + x3'x4'")
    print("="*70)
    
    num_vars = 6
    output_values = [0] * 64
    
    for i in range(64):
        bits = format(i, '06b')
        # x1'x2' (bits 0,1 are 00)
        if bits[0:2] == '00':
            output_values[i] = 1
        # x3'x4' (bits 2,3 are 00)
        elif bits[2:4] == '00':
            output_values[i] = 1
    
    print(f"Minterms: {sum(output_values)}")
    
    solver = BoolMinGeo(num_vars, output_values)
    terms, expr = solver.minimize_3d(form='sop')
    
    print(f"\nResult: {expr}")
    print(f"Expected: x1'x2' + x3'x4'")
    print()
    
    return expr

if __name__ == "__main__":
    test_6var_simple()
    print("\n" + "="*70 + "\n")
    test_6var_or()
    print("\n" + "="*70 + "\n")
    test_6var_complex()
    print("\n" + "="*70)
    print("ALL TESTS COMPLETED!")
    print("="*70)

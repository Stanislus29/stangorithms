"""
Comprehensive test for the fixed kmapsolver3D
"""
from StanLogic.src.stanlogic.kmapsolver3D import KMapSolver3D

def test_6var_complex():
    """Test a more complex 6-variable function"""
    print("="*60)
    print("TEST: 6-variable complex function")
    print("="*60)
    
    num_vars = 6
    # Function: x1'x2' + x3'x4'
    output_values = [0] * 64
    
    for i in range(64):
        bits = format(i, '06b')
        # x1'x2' (bits 0,1 are 00)
        if bits[0:2] == '00':
            output_values[i] = 1
        # x3'x4' (bits 2,3 are 00)
        elif bits[2:4] == '00':
            output_values[i] = 1
    
    print(f"\nNumber of minterms: {sum(output_values)}")
    
    kmap_solver = KMapSolver3D(num_vars, output_values)
    terms, expression = kmap_solver.minimize_3d(form='sop')
    
    print(f"\nResult: {expression}")
    print(f"Expected: x1'x2' + x3'x4'")
    print()
    
    return expression

def test_7var():
    """Test a 7-variable function: x1'x2' + x3'x4'x5'"""
    print("="*60)
    print("TEST: 7-variable function x1'x2' + x3'x4'x5'")
    print("="*60)
    
    num_vars = 7
    output_values = [0] * 128
    
    for i in range(128):
        bits = format(i, '07b')
        # x1'x2' (bits 0,1 are 00)
        if bits[0:2] == '00':
            output_values[i] = 1
        # x3'x4'x5' (bits 2,3,4 are 000)
        elif bits[2:5] == '000':
            output_values[i] = 1
    
    print(f"\nNumber of minterms: {sum(output_values)}")
    
    kmap_solver = KMapSolver3D(num_vars, output_values)
    terms, expression = kmap_solver.minimize_3d(form='sop')
    
    print(f"\nResult: {expression}")
    print(f"Expected: x1'x2' + x3'x4'x5'")
    print()
    
    return expression

def test_8var():
    """Test an 8-variable function"""
    print("="*60)
    print("TEST: 8-variable function x1'x2'x3' + x4x5")
    print("="*60)
    
    num_vars = 8
    output_values = [0] * 256
    
    for i in range(256):
        bits = format(i, '08b')
        # x1'x2'x3' (bits 0,1,2 are 000)
        if bits[0:3] == '000':
            output_values[i] = 1
        # x4x5 (bits 3,4 are 11)
        elif bits[3:5] == '11':
            output_values[i] = 1
    
    print(f"\nNumber of minterms: {sum(output_values)}")
    
    kmap_solver = KMapSolver3D(num_vars, output_values)
    terms, expression = kmap_solver.minimize_3d(form='sop')
    
    print(f"\nResult: {expression}")
    print(f"Expected: x1'x2'x3' + x4x5")
    print()
    
    return expression

if __name__ == "__main__":
    test_6var_complex()
    test_7var()
    test_8var()
    print("\n" + "="*60)
    print("ALL COMPREHENSIVE TESTS COMPLETED!")
    print("="*60)

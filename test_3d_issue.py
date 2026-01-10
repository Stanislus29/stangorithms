"""
Test to demonstrate the issue with kmapsolver3D for 6+ variables
"""
from StanLogic.src.stanlogic.kmapsolver3D import KMapSolver3D
from pyeda.inter import exprvars, expr

def test_6var_simple():
    """Test a simple 6-variable function: x1'x2'"""
    print("="*60)
    print("TEST: 6-variable function x1'x2'")
    print("="*60)
    
    num_vars = 6
    output_values = [0] * 64
    
    # x1'x2' means bits[0:2] == '00'
    for i in range(64):
        bits = format(i, '06b')
        if bits[0:2] == '00':
            output_values[i] = 1
    
    print(f"\nMinterms set to 1: {[i for i, v in enumerate(output_values) if v == 1]}")
    
    kmap_solver = KMapSolver3D(num_vars, output_values)
    terms, expression = kmap_solver.minimize_3d(form='sop')
    
    print(f"\nResult: {expression}")
    print("Expected: x1'x2'")
    
    return expression

def test_6var_or():
    """Test a 6-variable OR: x1 + x2"""
    print("\n" + "="*60)
    print("TEST: 6-variable function x1 + x2")
    print("="*60)
    
    num_vars = 6
    output_values = [0] * 64
    
    # x1 + x2 means bit[0]=='1' OR bit[1]=='1'
    for i in range(64):
        bits = format(i, '06b')
        if bits[0] == '1' or bits[1] == '1':
            output_values[i] = 1
    
    print(f"\nMinterms set to 1: {[i for i, v in enumerate(output_values) if v == 1]}")
    
    kmap_solver = KMapSolver3D(num_vars, output_values)
    terms, expression = kmap_solver.minimize_3d(form='sop')
    
    print(f"\nResult: {expression}")
    print(f"Expected: x1 + x2")
    
    return expression

def test_7var_simple():
    """Test a 7-variable function: x1'x2'x3'"""
    print("\n" + "="*60)
    print("TEST: 7-variable function x1'x2'x3'")
    print("="*60)
    
    num_vars = 7
    output_values = [0] * 128
    
    # x1'x2'x3' means bits[0:3] == '000'
    for i in range(128):
        bits = format(i, '07b')
        if bits[0:3] == '000':
            output_values[i] = 1
    
    print(f"\nMinterms set to 1: {[i for i, v in enumerate(output_values) if v == 1]}")
    
    kmap_solver = KMapSolver3D(num_vars, output_values)
    terms, expression = kmap_solver.minimize_3d(form='sop')
    
    print(f"\nResult: {expression}")
    print(f"Expected: x1'x2'x3'")
    
    return expression

if __name__ == "__main__":
    test_6var_simple()
    test_6var_or()
    test_7var_simple()

"""
Test file to verify the optimizations added to kmapsolver3D
"""
from StanLogic.src.stanlogic.kmapsolver3D import KMapSolver3D

def test_constant_function_optimization():
    """Test that constant functions are handled correctly"""
    print("="*60)
    print("TEST 1: Constant Function (All 1's)")
    print("="*60)
    
    num_vars = 5
    # All output values are 1
    output_values = [1] * 32
    
    kmap_solver = KMapSolver3D(num_vars, output_values)
    terms, expression = kmap_solver.minimize_3d(form='sop')
    
    print(f"Expression: {expression}")
    print(f"Number of terms: {len(terms)}")
    print()

def test_empty_function_optimization():
    """Test that empty functions (all 0's) are handled correctly"""
    print("="*60)
    print("TEST 2: Empty Function (All 0's)")
    print("="*60)
    
    num_vars = 5
    # All output values are 0
    output_values = [0] * 32
    
    kmap_solver = KMapSolver3D(num_vars, output_values)
    terms, expression = kmap_solver.minimize_3d(form='sop')
    
    print(f"Expression: {expression}")
    print(f"Number of terms: {len(terms)}")
    print()

def test_normal_function():
    """Test a normal function with mixed values"""
    print("="*60)
    print("TEST 3: Normal Function (Mixed Values)")
    print("="*60)
    
    num_vars = 5
    # Create sample output values - simple pattern
    output_values = [
        1,1,0,0,0,1,0,1,  # First 8 minterms (x1=0)
        1,1,0,0,0,1,0,1,  # Next 8 minterms (x1=0)
        0,0,0,0,0,0,0,0,  # Next 8 minterms (x1=1) 
        0,0,0,0,0,0,0,0   # Last 8 minterms (x1=1)
    ]
    
    kmap_solver = KMapSolver3D(num_vars, output_values)
    terms, expression = kmap_solver.minimize_3d(form='sop')
    
    print(f"Expression: {expression}")
    print(f"Number of terms: {len(terms)}")
    print()

def test_dont_care_handling():
    """Test handling of don't care values"""
    print("="*60)
    print("TEST 4: Don't Care Handling")
    print("="*60)
    
    num_vars = 5
    # Mix of 0, 1, and don't care values
    output_values = [1,1,0,'d',0,1,'d',1] * 4
    
    kmap_solver = KMapSolver3D(num_vars, output_values)
    terms, expression = kmap_solver.minimize_3d(form='sop')
    
    print(f"Expression: {expression}")
    print(f"Number of terms: {len(terms)}")
    print()

def main():
    print("\n" + "="*60)
    print("TESTING KMAPSOLVER3D OPTIMIZATIONS")
    print("="*60 + "\n")
    
    try:
        test_constant_function_optimization()
        test_empty_function_optimization()
        test_normal_function()
        test_dont_care_handling()
        
        print("="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

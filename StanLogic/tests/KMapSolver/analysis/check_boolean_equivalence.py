"""
Script to check equivalence between two Boolean functions using SymPy

Author: Somtochukwu Stanislus Emeka-Onwuneme
Date: November 2025
"""

import sys
import io
# Set UTF-8 encoding for console output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from sympy import symbols
from sympy.logic.boolalg import And, Or, Not, simplify_logic, SOPform, to_cnf
from sympy.logic.inference import satisfiable
sys.path.append('c:/Users/DELL/Documents/Research Projects/Mathematical_models/StanLogic/src')
from stanlogic import BoolMinGeo


def check_equivalence(func1, func2, variables):
    """
    Check if two Boolean functions are equivalent.
    
    Args:
        func1: First Boolean function (sympy expression)
        func2: Second Boolean function (sympy expression)
        variables: List of Boolean variables used
    
    Returns:
        bool: True if equivalent, False otherwise
    """
    # Method 1: Check if (f1 XOR f2) is unsatisfiable
    # If f1 and f2 are equivalent, then f1 XOR f2 should always be False
    xor_expr = (func1 & ~func2) | (~func1 & func2)
    
    # If the XOR is unsatisfiable, the functions are equivalent
    result = satisfiable(xor_expr)
    
    return result is False


def simplify_and_compare(func1, func2):
    """
    Simplify both functions and compare them.
    
    Args:
        func1: First Boolean function
        func2: Second Boolean function
    
    Returns:
        tuple: (simplified_f1, simplified_f2, are_equal)
    """
    simplified_f1 = simplify_logic(func1)
    simplified_f2 = simplify_logic(func2)
    
    # Check if simplified forms are equal
    are_equal = simplified_f1.equals(simplified_f2)
    
    return simplified_f1, simplified_f2, are_equal


def extract_truth_table(func, variables):
    """
    Extract truth table from a Boolean function.
    
    Args:
        func: Boolean function (sympy expression)
        variables: List of Boolean variables (ordered as x1, x2, ..., xn)
    
    Returns:
        list: List of output values (0 or 1) for all 2^n combinations
    """
    n = len(variables)
    output_values = []
    
    # Generate all possible combinations
    # Standard truth table ordering: x1 is MSB, xn is LSB
    for i in range(2**n):
        # Create substitution dictionary for this combination
        subs_dict = {}
        for j, var in enumerate(variables):
            # Extract bit for variable j (x1 is MSB, so it's bit n-1-j)
            bit_value = (i >> (n - 1 - j)) & 1
            subs_dict[var] = bool(bit_value)
        
        # Evaluate function at this point
        result = func.subs(subs_dict)
        output_values.append(1 if result else 0)
    
    return output_values


def parse_kmapsolver_result(result_str, variables):
    """
    Parse kmapsolver3D result string into a SymPy Boolean expression.
    
    Args:
        result_str: Result string from kmapsolver3D (e.g., "x3x5' + x2x3'x5")
        variables: List of variable symbols in order
    
    Returns:
        SymPy Boolean expression
    """
    if not result_str or result_str == '0':
        return False
    if result_str == '1':
        return True
    
    # Split by '+' for OR terms
    terms = result_str.split('+')
    or_terms = []
    
    for term in terms:
        term = term.strip()
        and_factors = []
        
        # Parse variables in format like "x3x5'" or "x2x3'x5"
        i = 0
        while i < len(term):
            if term[i] == 'x':
                # Found a variable
                i += 1
                # Read the variable number
                var_num = ''
                while i < len(term) and term[i].isdigit():
                    var_num += term[i]
                    i += 1
                
                if var_num:
                    var_index = int(var_num) - 1  # Convert to 0-based index
                    
                    # Check for negation (prime symbol)
                    if i < len(term) and term[i] == "'":
                        and_factors.append(Not(variables[var_index]))
                        i += 1
                    else:
                        and_factors.append(variables[var_index])
            else:
                i += 1
        
        if and_factors:
            if len(and_factors) == 1:
                or_terms.append(and_factors[0])
            else:
                or_terms.append(And(*and_factors))
    
    if not or_terms:
        return False
    if len(or_terms) == 1:
        return or_terms[0]
    return Or(*or_terms)


def main():
    # Define Boolean variables
    A, B, C, D, E = symbols('A B C D E')
    
    # Define the two functions
    # f1 = ~AC + A~CD = (~A)C + A(~C)D
    f1 = Or(And(Not(A), C), And(A, Not(C), D))
    
    # f2 = ~AC + A~CD + A~B~CE = AC + A(~C)D + A(~B)(~C)E
    f2 = Or(
        And(Not(A), C),
        And(A, Not(C), D),
        And(A, Not(B), Not(C), E)
    )
    
    print("Boolean Function Equivalence Check using SymPy and KMapSolver3D")
    print("=" * 70)
    print()
    
    # Display original functions
    print("Function 1: f1 = ~AC + A~CD")
    print(f"  SymPy form: {f1}")
    print()
    
    print("Function 2: f2 = ~AC + A~CD + A~B~CE")
    print(f"  SymPy form: {f2}")
    print()
    
    # Extract truth table for f2
    print("Extracting Truth Table for Function 2:")
    print("-" * 70)
    variables = [A, B, C, D, E]
    print(f"Variable order: {[str(v) for v in variables]} (MSB to LSB: A is MSB, E is LSB)")
    output_values_f2 = extract_truth_table(f2, variables)
    minterms_f2 = [i for i, val in enumerate(output_values_f2) if val == 1]
    print(f"Output values: {output_values_f2[:16]}... (first 16 of {len(output_values_f2)})")
    print(f"Minterms (indices where f2 = 1): {minterms_f2}")
    print(f"Total number of minterms: {len(minterms_f2)}")
    
    # Print some specific truth table entries for verification
    print("\nSample truth table entries:")
    for i in [4, 6, 9, 11, 12]:
        bits = [(i >> (4 - j)) & 1 for j in range(5)]
        print(f"  Index {i:2d}: A={bits[0]} B={bits[1]} C={bits[2]} D={bits[3]} E={bits[4]} => f2={output_values_f2[i]}")
    print()
    
    # Solve using kmapsolver3D
    print("Solving with KMapSolver3D:")
    print("-" * 70)
    num_vars = len(variables)
    var_names = ['A', 'B', 'C', 'D', 'E']
    
    try:
        # Create K-map solver
        kmap_solver = BoolMinGeo(num_vars, output_values_f2)
        
        # Print K-maps for visualization
        print("K-Map Structure:")
        print("-" * 70)
        kmap_solver.print_kmaps()
        print()
        
        # Suppress verbose output from minimization
        import contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            terms, kmap_result = kmap_solver.minimize_3d(form='sop')
        
        print(f"KMapSolver3D result: {kmap_result}")
        print(f"Number of terms: {len(terms)}")
        print()
        
        # Parse kmapsolver result into SymPy expression
        kmap_expr = parse_kmapsolver_result(kmap_result, variables)
        print(f"Parsed as SymPy expression: {kmap_expr}")
        print()
        
        # Check equivalence between f2 and kmapsolver result
        print("Checking Equivalence between f2 and KMapSolver3D result:")
        print("-" * 70)
        
        equivalent = check_equivalence(f2, kmap_expr, variables)
        
        if equivalent:
            print("✓ KMapSolver3D solution is EQUIVALENT to f2")
        else:
            print("✗ KMapSolver3D solution is NOT EQUIVALENT to f2")
            
            # Find a counterexample if not equivalent
            print("\nFinding counterexample...")
            xor_expr = (f2 & ~kmap_expr) | (~f2 & kmap_expr)
            counterexample = satisfiable(xor_expr)
            
            if counterexample:
                print("Counterexample found:")
                for var, value in counterexample.items():
                    print(f"  {var} = {value}")
                
                # Evaluate both functions at the counterexample
                subs_dict = {k: v for k, v in counterexample.items()}
                val_f2 = f2.subs(subs_dict)
                val_kmap = kmap_expr.subs(subs_dict)
                print(f"\nAt this assignment:")
                print(f"  f2 = {val_f2}")
                print(f"  KMapSolver3D result = {val_kmap}")
        
        # Also simplify both for comparison
        print()
        print("Simplified Forms:")
        print("-" * 70)
        simplified_f2 = simplify_logic(f2)
        simplified_kmap = simplify_logic(kmap_expr)
        print(f"f2 simplified: {simplified_f2}")
        print(f"KMapSolver3D simplified: {simplified_kmap}")
        
    except Exception as e:
        print(f"Error running kmapsolver3D: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    
    # Original comparison between f1 and f2
    print()
    print("Original Comparison: f1 vs f2")
    print("=" * 70)
    
    # Simplify both functions
    simplified_f1, simplified_f2, direct_equal = simplify_and_compare(f1, f2)
    
    print("Simplified Forms:")
    print("-" * 70)
    print(f"f1 simplified: {simplified_f1}")
    print(f"f2 simplified: {simplified_f2}")
    print()
    
    # Check equivalence using satisfiability
    equivalent = check_equivalence(f1, f2, variables)
    
    print("Equivalence Check Results:")
    print("-" * 70)
    print(f"Direct comparison of simplified forms: {direct_equal}")
    print(f"Satisfiability-based equivalence: {equivalent}")
    print()
    
    if equivalent:
        print("✓ The two Boolean functions are EQUIVALENT")
    else:
        print("✗ The two Boolean functions are NOT EQUIVALENT")
        print()
        
        # Find a counterexample if not equivalent
        print("Finding counterexample...")
        xor_expr = (f1 & ~f2) | (~f1 & f2)
        counterexample = satisfiable(xor_expr)
        
        if counterexample:
            print("Counterexample found:")
            for var, value in counterexample.items():
                print(f"  {var} = {value}")
            
            # Evaluate both functions at the counterexample
            subs_dict = {k: v for k, v in counterexample.items()}
            val1 = f1.subs(subs_dict)
            val2 = f2.subs(subs_dict)
            print(f"\nAt this assignment:")
            print(f"  f1 = {val1}")
            print(f"  f2 = {val2}")
    
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()

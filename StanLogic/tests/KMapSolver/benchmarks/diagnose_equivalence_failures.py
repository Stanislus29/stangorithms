"""
Diagnostic test to identify equivalence check failures
"""

from pyeda.inter import *
from stanlogic import BoolMin2D
from sympy import symbols, And as SymAnd, Or as SymOr, Not as SymNot, simplify, Equivalent
import re

def parse_kmap_to_pyeda(expr_str, pyeda_vars, form="sop"):
    """Parse BoolMin2D expression to PyEDA format"""
    if not expr_str or expr_str.strip() == "":
        return expr(0) if form == "sop" else expr(1)

    var_map = {str(v): v for v in pyeda_vars}

    if form.lower() == "sop":
        or_terms = []
        for product in expr_str.split('+'):
            product = product.strip()
            if not product:
                continue
            literals = re.findall(r"[A-Za-z_]\w*'?", product)
            and_terms = []
            for lit in literals:
                if lit.endswith("'"):
                    var_name = lit[:-1]
                    if var_name in var_map:
                        and_terms.append(~var_map[var_name])
                else:
                    if lit in var_map:
                        and_terms.append(var_map[lit])
            if len(and_terms) == 1:
                or_terms.append(and_terms[0])
            elif and_terms:
                or_terms.append(And(*and_terms))
        return Or(*or_terms) if or_terms else expr(0)
    
    elif form.lower() == "pos":
        and_terms = []
        for sum_term in re.split(r'\)\s*\(', expr_str.strip('() ')):
            sum_term = sum_term.strip()
            if not sum_term:
                continue
            literals = re.findall(r"[A-Za-z_]\w*'?", sum_term)
            or_terms = []
            for lit in literals:
                if lit.endswith("'"):
                    var_name = lit[:-1]
                    if var_name in var_map:
                        or_terms.append(~var_map[var_name])
                else:
                    if lit in var_map:
                        or_terms.append(var_map[lit])
            if len(or_terms) == 1:
                and_terms.append(or_terms[0])
            elif or_terms:
                and_terms.append(Or(*or_terms))
        return And(*and_terms) if and_terms else expr(1)

def test_kmap(kmap, test_name, form='sop'):
    """Test a specific K-map configuration"""
    print(f"\n{'='*70}")
    print(f"TEST: {test_name}")
    print(f"{'='*70}")
    print(f"K-map:")
    for row in kmap:
        print(f"  {row}")
    print(f"Form: {form.upper()}")
    
    # Determine number of variables
    size = len(kmap) * len(kmap[0])
    if size == 4:
        num_vars = 2
    elif size == 8:
        num_vars = 3
    elif size == 16:
        num_vars = 4
    else:
        print(f"ERROR: Invalid K-map size {size}")
        return False
    
    # Create variables
    pyeda_vars = [exprvar(f'x{i+1}') for i in range(num_vars)]
    
    # Get BoolMin2D result
    try:
        solver = BoolMin2D(kmap)
        terms, expr_str = solver.minimize(form=form)
        print(f"BoolMin2D result: '{expr_str}'")
        print(f"Terms: {terms}")
    except Exception as e:
        print(f"❌ BoolMin2D error: {e}")
        return False
    
    # Build PyEDA expression from truth table
    try:
        if form == 'sop':
            # Build from minterms
            minterms = []
            for r in range(len(kmap)):
                for c in range(len(kmap[0])):
                    if kmap[r][c] == 1:
                        # Calculate minterm index
                        if num_vars == 2:
                            idx = r * 2 + c
                        elif num_vars == 3:
                            if len(kmap) == 2:  # 2x4
                                idx = c * 2 + r
                            else:  # 4x2
                                idx = r * 2 + c
                        elif num_vars == 4:
                            idx = c * 4 + r
                        
                        # Build minterm expression
                        minterm = And(*[
                            pyeda_vars[i] if (idx >> (num_vars - 1 - i)) & 1 else ~pyeda_vars[i]
                            for i in range(num_vars)
                        ])
                        minterms.append(minterm)
            
            if minterms:
                pyeda_expr_manual = Or(*minterms)
            else:
                pyeda_expr_manual = expr(0)
        else:  # POS
            # Build from maxterms (zeros)
            maxterms = []
            for r in range(len(kmap)):
                for c in range(len(kmap[0])):
                    if kmap[r][c] == 0:
                        # Calculate maxterm index
                        if num_vars == 2:
                            idx = r * 2 + c
                        elif num_vars == 3:
                            if len(kmap) == 2:  # 2x4
                                idx = c * 2 + r
                            else:  # 4x2
                                idx = r * 2 + c
                        elif num_vars == 4:
                            idx = c * 4 + r
                        
                        # Build maxterm expression (OR of literals)
                        maxterm = Or(*[
                            ~pyeda_vars[i] if (idx >> (num_vars - 1 - i)) & 1 else pyeda_vars[i]
                            for i in range(num_vars)
                        ])
                        maxterms.append(maxterm)
            
            if maxterms:
                pyeda_expr_manual = And(*maxterms)
            else:
                pyeda_expr_manual = expr(1)
        
        print(f"PyEDA from truth table: {pyeda_expr_manual}")
        
        # Minimize
        if form == 'sop':
            # For SOP, handle don't cares
            dc_terms = []
            for r in range(len(kmap)):
                for c in range(len(kmap[0])):
                    if kmap[r][c] == 'd':
                        if num_vars == 2:
                            idx = r * 2 + c
                        elif num_vars == 3:
                            if len(kmap) == 2:
                                idx = c * 2 + r
                            else:
                                idx = r * 2 + c
                        elif num_vars == 4:
                            idx = c * 4 + r
                        
                        dc_term = And(*[
                            pyeda_vars[i] if (idx >> (num_vars - 1 - i)) & 1 else ~pyeda_vars[i]
                            for i in range(num_vars)
                        ])
                        dc_terms.append(dc_term)
            
            if dc_terms:
                f_dc = Or(*dc_terms)
            else:
                f_dc = expr(0)
            
            if minterms:
                pyeda_minimized = espresso_exprs(pyeda_expr_manual, f_dc)[0]
            else:
                pyeda_minimized = expr(0)
        else:  # POS
            # For POS, minimize the OFF-set and complement
            if maxterms:
                # Convert to DNF of zeros
                dnf_zeros = Or(*[And(*[
                    pyeda_vars[i] if (idx >> (num_vars - 1 - i)) & 1 else ~pyeda_vars[i]
                    for i in range(num_vars)
                ]) for r in range(len(kmap)) for c in range(len(kmap[0])) 
                   if kmap[r][c] == 0 
                   and (idx := (c * 4 + r if num_vars == 4 else 
                               (c * 2 + r if (num_vars == 3 and len(kmap) == 2) else r * 2 + c))) >= 0])
                
                pyeda_minimized = (~espresso_exprs(dnf_zeros)[0]).to_cnf()
            else:
                pyeda_minimized = expr(1)
        
        print(f"PyEDA minimized: {pyeda_minimized}")
    except Exception as e:
        print(f"❌ PyEDA error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Parse BoolMin2D output
    try:
        parsed_expr = parse_kmap_to_pyeda(expr_str, pyeda_vars, form=form)
        print(f"Parsed BoolMin2D: {parsed_expr}")
    except Exception as e:
        print(f"❌ Parsing error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check equivalence
    try:
        equiv = pyeda_minimized.equivalent(parsed_expr)
        status = "PASS" if equiv else "FAIL"
        print(f"\nEquivalence: {equiv} [{status}]")
        
        if not equiv:
            print(f"\nMISMATCH DETAILS:")
            print(f"  PyEDA minimized:  {pyeda_minimized}")
            print(f"  Type: {type(pyeda_minimized)}")
            print(f"  Parsed BoolMin2D: {parsed_expr}")
            print(f"  Type: {type(parsed_expr)}")
            
            # Try truth table comparison
            print(f"\n  Testing truth table equivalence...")
            all_match = True
            for i in range(2**num_vars):
                inputs = {pyeda_vars[j]: (i >> (num_vars - 1 - j)) & 1 for j in range(num_vars)}
                val1 = pyeda_minimized.restrict(inputs)
                val2 = parsed_expr.restrict(inputs)
                if val1 != val2:
                    print(f"    Input {bin(i)[2:].zfill(num_vars)}: PyEDA={val1}, Parsed={val2}")
                    all_match = False
            
            if all_match:
                print(f"    All truth table entries match! Issue may be with .equivalent() method")
        
        return equiv
    except Exception as e:
        print(f"ERROR: Equivalence check error: {e}")
        import traceback
        traceback.print_exc()
        return False

# Run tests
print("DIAGNOSTIC TEST SUITE")
print("=" * 70)

test_cases = [
    # 2-variable tests
    ([[0, 1], [1, 1]], "2-var: Simple OR", 'sop'),
    ([[1, 1], [1, 0]], "2-var: Simple AND", 'sop'),
    ([[1, 0], [0, 1]], "2-var: XOR", 'sop'),
    ([[1, 1], [1, 1]], "2-var: All ones", 'sop'),
    ([[0, 0], [0, 0]], "2-var: All zeros", 'sop'),
    
    # 3-variable tests
    ([[0, 1, 1, 0], [0, 1, 1, 0]], "3-var: Middle columns", 'sop'),
    ([[1, 0, 0, 1], [1, 0, 0, 1]], "3-var: Outer columns", 'sop'),
    
    # With don't cares
    ([[0, 1], [1, 'd']], "2-var: With DC", 'sop'),
    ([['d', 'd'], ['d', 'd']], "2-var: All DC", 'sop'),
    
    # POS tests
    ([[0, 1], [1, 1]], "2-var: POS form", 'pos'),
    ([[1, 1], [1, 0]], "2-var: POS form 2", 'pos'),
]

results = []
for kmap, name, form in test_cases:
    result = test_kmap(kmap, name, form)
    results.append((name, result))

print(f"\n\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
passed = sum(1 for _, r in results if r)
total = len(results)
print(f"Passed: {passed}/{total}")
print(f"\nFailed tests:")
for name, result in results:
    if not result:
        print(f"  X {name}")

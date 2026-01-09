"""
Debug full parse_kmap_to_pyeda function
"""

from pyeda.inter import *
import re

def parse_kmap_to_pyeda_debug(expr_str, pyeda_vars, terms=None, form="sop"):
    """Parse with debug output"""
    print(f"\n=== PARSING ===")
    print(f"Input: '{expr_str}'")
    print(f"Terms: {terms}")
    print(f"Form: {form}")
    
    if not expr_str or expr_str.strip() == "":
        if terms is not None:
            if terms == ['']:
                print("-> Constant 1")
                return expr(1)
            elif not terms:
                print("-> Constant 0")
                return expr(0)
        return expr(0) if form == "sop" else expr(1)

    var_map = {str(v): v for v in pyeda_vars}
    print(f"Variable map: {var_map}")

    if form.lower() == "sop":
        or_terms = []
        products = expr_str.split('+')
        print(f"Products: {products}")
        
        for product in products:
            product = product.strip()
            if not product:
                print(f"  Skipping empty product")
                continue
            
            print(f"  Processing product: '{product}'")
            literals = re.findall(r"[A-Za-z]+\d*'?", product)
            print(f"    Literals found: {literals}")
            
            and_terms = []
            for lit in literals:
                if lit.endswith("'"):
                    var_name = lit[:-1]
                    print(f"      {lit} -> ~{var_name}")
                    if var_name in var_map:
                        and_terms.append(~var_map[var_name])
                else:
                    print(f"      {lit} -> {lit}")
                    if lit in var_map:
                        and_terms.append(var_map[lit])
            
            print(f"    AND terms: {and_terms}")
            if len(and_terms) == 1:
                or_terms.append(and_terms[0])
                print(f"    Added single term")
            elif and_terms:
                or_terms.append(And(*and_terms))
                print(f"    Added AND({and_terms})")
        
        print(f"OR terms collected: {or_terms}")
        result = Or(*or_terms) if or_terms else expr(0)
        print(f"Final result: {result}")
        return result

# Test
pyeda_vars = [exprvar('x1'), exprvar('x2')]
expr_str = "x1x2 + x1'x2'"
terms = ['x1x2', "x1'x2'"]

result = parse_kmap_to_pyeda_debug(expr_str, pyeda_vars, terms=terms, form='sop')
print(f"\n=== RESULT ===")
print(f"Parsed expression: {result}")
print(f"Type: {type(result)}")

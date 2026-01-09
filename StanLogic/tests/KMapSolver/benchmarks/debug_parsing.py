"""
Debug multi-term parsing
"""

import re

expr_str = "x1x2 + x1'x2'"
print(f"Expression: '{expr_str}'")
print(f"\nSplit by '+': {expr_str.split('+')}")

# Old regex
print("\n=== OLD REGEX: r\"[A-Za-z_]\\w*'?\" ===")
for product in expr_str.split('+'):
    product = product.strip()
    print(f"\nProcessing: '{product}'")
    literals = re.findall(r"[A-Za-z_]\w*'?", product)
    print(f"  Literals: {literals}")

# New regex
print("\n=== NEW REGEX: r\"[A-Za-z]+\\d*'?\" ===")
for product in expr_str.split('+'):
    product = product.strip()
    print(f"\nProcessing: '{product}'")
    literals = re.findall(r"[A-Za-z]+\d*'?", product)
    print(f"  Literals: {literals}")
    
    for lit in literals:
        if lit.endswith("'"):
            var_name = lit[:-1]
            print(f"    {lit} -> complemented {var_name}")
        else:
            print(f"    {lit} -> normal")

import random 
import time 
import re
from stanlogic import KMapSolver
from sympy import symbols, SOPform, simplify, Equivalent, POSform
from tabulate import tabulate
import random

# ...existing code...
def random_kmap(rows=4, cols=4, formatted=False, p_one=0.55, p_dc=0.07, seed=None):
    """
    Generate a random K-map with controlled probability of 1, 0, and 'd'.
    p_one: probability a cell is 1
    p_dc: probability a cell is 'd' (kept low to avoid excessive don't cares)
    Remaining probability becomes 0.
    """
    if seed is not None:
        random.seed(seed)
    if p_one < 0 or p_dc < 0 or p_one + p_dc > 1:
        raise ValueError("Probabilities invalid: ensure p_one >=0, p_dc >=0, p_one + p_dc <= 1")
    p_zero = 1 - p_one - p_dc
    choices = [1, 0, 'd']
    weights = [p_one, p_zero, p_dc]
    kmap = [[random.choices(choices, weights=weights, k=1)[0] for _ in range(cols)] for _ in range(rows)]
    if formatted:
        return "kmap = [\n" + "\n".join("    " + repr(row) + "," for row in kmap) + "\n]"
    return kmap
# ...existing code...

def kmap_minterms(kmap, convention="vranseic"):
    """
    Extract minterm indices (cells with 1) and don't care indices (cells with 'd')
    from a K-map using Gray code labeling and variable ordering convention.
    convention: 'vranseic' (cols=x1x2, rows=x3x4) or 'mano_kime' (rows=x1x2, cols=x3x4)
    Returns dict with minterms, dont_cares, bits_map.
    """
    rows = len(kmap)
    cols = len(kmap[0])
    size = rows * cols

    if size == 4:          # 2 vars (2x2)
        num_vars = 2
        row_labels = ["0", "1"]
        col_labels = ["0", "1"]
    elif size == 8:        # 3 vars (2x4 or 4x2)
        num_vars = 3
        if rows == 2 and cols == 4:
            row_labels = ["0", "1"]
            col_labels = ["00", "01", "11", "10"]  # Gray code
        elif rows == 4 and cols == 2:
            row_labels = ["00", "01", "11", "10"]
            col_labels = ["0", "1"]
        else:
            raise ValueError("Invalid 3-variable K-map shape.")
    elif size == 16:       # 4 vars (4x4)
        num_vars = 4
        row_labels = ["00", "01", "11", "10"]
        col_labels = ["00", "01", "11", "10"]
    else:
        raise ValueError("K-map must be 2x2, 2x4/4x2, or 4x4.")

    convention = convention.lower().strip()

    def cell_to_bits(r, c):
        if convention == "mano_kime":
            return row_labels[r] + col_labels[c]
        return col_labels[c] + row_labels[r]

    def cell_to_term(r, c):
        bits = cell_to_bits(r, c)
        return int(bits, 2)

    minterms = []
    dont_cares = []
    bits_map = {}

    for r in range(rows):
        for c in range(cols):
            val = kmap[r][c]
            idx = cell_to_term(r, c)
            bits_map[idx] = cell_to_bits(r, c)
            if val == 1:
                minterms.append(idx)
            elif val == 'd':
                dont_cares.append(idx)

    return {
        "num_vars": num_vars,
        "minterms": sorted(minterms),
        "dont_cares": sorted(dont_cares),
        "bits_map": bits_map
    }

def parse_kmap_expression(expr_str, var_names, form="sop"):
    """
    Parse a K-map expression (SOP or POS) string into a SymPy boolean expression.

    Args:
        expr_str (str): The Boolean expression string (SOP or POS form).
        var_names (list): List of SymPy symbols (variable names).
        form (str): 'sop' or 'pos' (Sum of Products or Product of Sums).

    Returns:
        SymPy Boolean expression equivalent to the input string.
    """
    from sympy import And, Or, Not, false, true

    if not expr_str or expr_str.strip() == "":
        return false if form == "sop" else true

    name_map = {str(v): v for v in var_names}

    # --- SOP (Sum of Products): A'B + BC' + ...
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
                    var = lit[:-1]
                    if var in name_map:
                        and_terms.append(Not(name_map[var]))
                else:
                    if lit in name_map:
                        and_terms.append(name_map[lit])
            if len(and_terms) == 1:
                or_terms.append(and_terms[0])
            elif and_terms:
                or_terms.append(And(*and_terms))
        return Or(*or_terms) if or_terms else false

    # --- POS (Product of Sums): (A + B')(A' + C) + ...
    elif form.lower() == "pos":
        and_terms = []
        # Split between parentheses (handle (A + B')(A' + C))
        for sum_term in re.split(r'\)\s*\(\s*', expr_str.strip('() ')):
            sum_term = sum_term.strip()
            if not sum_term:
                continue
            literals = re.findall(r"[A-Za-z_]\w*'?", sum_term)
            or_terms = []
            for lit in literals:
                if lit.endswith("'"):
                    var = lit[:-1]
                    if var in name_map:
                        or_terms.append(Not(name_map[var]))
                else:
                    if lit in name_map:
                        or_terms.append(name_map[lit])
            if len(or_terms) == 1:
                and_terms.append(or_terms[0])
            elif or_terms:
                and_terms.append(Or(*or_terms))
        return And(*and_terms) if and_terms else true

    else:
        raise ValueError(f"Unsupported form '{form}'. Must be 'sop' or 'pos'.")

def check_equivalence(expr1, expr2):
    """Return True if both SymPy expressions are logically equivalent."""
    try:
        return bool(simplify(Equivalent(expr1, expr2)))
    except Exception:
        return False

def count_literals(expr_str, form="sop"):
    """
    Count number of terms and total literals in a Boolean expression.
    form='sop': expression like A'B + BC' + D
        - terms = product terms (separated by +)
        - literals = variable appearances with negation counted separately
    form='pos': expression like (A + B')(A' + C + D')
        - terms = sum clauses (parenthesized groups)
        - literals = variable appearances across all sum clauses
    Returns (num_terms, num_literals).
    """
    if not expr_str or expr_str.strip() == "":
        return 0, 0

    form = form.lower()
    s = expr_str.replace(" ", "")

    if form == "sop":
        terms = [t for t in s.split('+') if t]
        num_terms = len(terms)
        num_literals = 0
        for t in terms:
            lits = re.findall(r"[A-Za-z_]\w*'?", t)
            num_literals += len(lits)
        return num_terms, num_literals

    if form == "pos":
        # Extract clauses inside parentheses; if none, treat whole string as one clause
        clauses = re.findall(r"\(([^()]*)\)", s)
        if not clauses:
            clauses = [s]
        num_terms = len(clauses)
        num_literals = 0
        for clause in clauses:
            # Split by '+' inside clause
            lits = [lit for lit in clause.split('+') if lit]
            # Each lit may be like A or A'
            for lit in lits:
                if re.fullmatch(r"[A-Za-z_]\w*'?", lit):
                    num_literals += 1
        return num_terms, num_literals

    raise ValueError("form must be 'sop' or 'pos'")


def benchmark_case(kmap, var_names, form_type, test_index):
    info = kmap_minterms(kmap, convention="vranseic")
    minterms = info["minterms"]
    dont_cares = info["dont_cares"]

    if len(var_names) != info["num_vars"]:
        raise ValueError(f"Variable symbol count ({len(var_names)}) does not match K-map ({info['num_vars']}).")

    # SymPy minimization
    start = time.perf_counter()
    if form_type == "pos":
        expr_sympy = POSform(var_names, minterms, dont_cares)
    else:
        expr_sympy = SOPform(var_names, minterms, dont_cares)
    t_sympy = time.perf_counter() - start

    # KMapSolver minimization
    start = time.perf_counter()
    solver = KMapSolver(kmap)
    terms, expr_str = solver.minimize(form=form_type)
    t_kmap = time.perf_counter() - start

    # Parse solver expression to SymPy
    expr_kmap_sympy = parse_kmap_expression(expr_str, var_names, form=form_type)
    equiv = check_equivalence(expr_sympy, expr_kmap_sympy)

    # Literal metrics (pass form so POS counted correctly)
    sympy_terms, sympy_literals = count_literals(str(expr_sympy), form=form_type)
    kmap_terms, kmap_literals = count_literals(expr_str, form=form_type)

    return {
        "index": test_index,
        "num_vars": info["num_vars"],
        "form": form_type,
        "equiv": equiv,
        "t_sympy": t_sympy,
        "t_kmap": t_kmap,
        "sympy_literals": sympy_literals,
        "kmap_literals": kmap_literals,
        "sympy_terms": sympy_terms,
        "kmap_terms": kmap_terms,
    }

def _kmap_shape(num_vars):
    if num_vars == 2:
        return 2, 2
    if num_vars == 3:
        return 2, 4   # matches kmap_minterms 3-variable case rows=2, cols=4
    if num_vars == 4:
        return 4, 4
    raise ValueError("Only 2–4 variables supported.")

def main():
    config = [
        (2, "sop"),
        (2, "pos"),
        (3, "sop"),
        (3, "pos"),
        (4, "sop"),
        (4, "pos"),
    ]

    random.seed(42)
    var_pool = symbols('x1 x2 x3 x4')
    all_results = []

    for num_vars, form_type in config:
        print(f"\n{'='*70}")
        print(f" Benchmark: {num_vars}-variable K-map ({form_type.upper()})")
        print(f"{'='*70}")
        rows, cols = _kmap_shape(num_vars)
        results = []
        for i in range(100):  # 100 random K-maps per configuration
            kmap = random_kmap(rows=rows, cols=cols)
            var_names = var_pool[:num_vars]
            result = benchmark_case(kmap, var_names, form_type, i + 1)
            results.append(result)
            all_results.append(result)

        print(tabulate(results, headers="keys", tablefmt="fancy_grid"))        
        
    #     sop_expr = SOPform(vars_syms, minterms, dont_cares)

    #     expr_kmap_sympy = parse_kmap_expression(sop, vars_syms)
    #     equiv = check_equivalence(expr_kmap_sympy, sop_expr)

    #     if not equiv:
    #         print("Discrepancy found!")
    #         print("K-map:")
    #         solver.print_kmap()
    #         print("K-map SOP:", sop)
    #         print("SymPy SOP:", sop_expr)
    #         print("Minterms:", info["minterms"])
    #         print("Don't cares:", info["dont_cares"])
    #         break

if __name__ == "__main__":
    main()


# # Example usage
# kmap = random_kmap(rows=4, cols=4)  # list, suitable for KMapSolver
# solver = KMapSolver(kmap)
# terms, sop = solver.minimize()
# solver.print_kmap()
# print("K-map SOP:", sop)


# info = kmap_minterms(kmap, convention="vranseic")
# minterms = info["minterms"]
# dont_cares = info["dont_cares"]

# vars_syms = symbols('x1 x2 x3 x4')
# sop_expr = SOPform(vars_syms, minterms, dont_cares)
# print("SymPy SOP:", sop_expr)

# print("Minterms:", info["minterms"])
# print("Don't cares:", info["dont_cares"])

# expr_kmap_sympy = parse_kmap_expression(sop, vars_syms)
# equiv = check_equivalence(expr_kmap_sympy, sop_expr)

# print("Equivalence:", equiv)
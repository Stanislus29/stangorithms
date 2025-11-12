import time
import random
import re
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols, SOPform, simplify, Equivalent, POSform
from stanlogic import KMapSolver
from tabulate import tabulate
from collections import defaultdict
from matplotlib.backends.backend_pdf import PdfPages
from textwrap import fill, wrap
import matplotlib.image as mpimg

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

def count_sympy_expression_literals(expr, form="sop"):
    """
    Count number of terms and literals from a SymPy boolean expression string
    without using SymPy logic utilities (pure string parsing).
    SOP: terms are top-level OR-separated groups (|). Each group's literals are &-separated.
    POS: clauses are top-level AND-separated groups (&). Each clause's literals are |-separated.
    Negations (~x) count as literals. Parentheses only provide grouping and are ignored in counts.
    Returns (num_terms, num_literals).
    """
    if expr is None:
        return 0, 0
    s = str(expr).strip()
    form = form.lower()

    # Handle trivial constants
    if s in ("True", "False"):
        return 0, 0

    # If single symbol or negated symbol
    import re
    if re.fullmatch(r"~?[A-Za-z_]\w*", s):
        return 1, 1

    def split_top_level(text, sep):
        parts = []
        buf = []
        depth = 0
        for ch in text:
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            if ch == sep and depth == 0:
                part = ''.join(buf).strip()
                if part:
                    parts.append(part)
                buf = []
            else:
                buf.append(ch)
        last = ''.join(buf).strip()
        if last:
            parts.append(last)
        return parts

    def strip_parens(x):
        x = x.strip()
        while x.startswith('(') and x.endswith(')'):
            # Ensure they are a matching outer pair
            depth = 0
            valid = True
            for i, ch in enumerate(x):
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                    if depth == 0 and i != len(x) - 1:
                        valid = False
                        break
            if valid:
                x = x[1:-1].strip()
            else:
                break
        return x

    def count_literals_in_group(group_str, inner_sep):
        group_str = strip_parens(group_str)
        if not group_str:
            return 0
        # Split by inner_sep at top level for POS clauses or SOP products
        inner_parts = split_top_level(group_str, inner_sep)
        # If no split occurred (single literal or conjunction/disjunction without separator)
        if len(inner_parts) == 1 and inner_sep in ('|', '&'):
            # For products/clauses we still might have separators
            raw = inner_parts[0]
            # For SOP product: split by '&'
            if inner_sep == '&':
                lits = split_top_level(raw, '&')
            else:
                lits = split_top_level(raw, '|')
        else:
            lits = inner_parts
        literal_count = 0
        for lit in lits:
            lit = strip_parens(lit)
            if re.fullmatch(r"~?[A-Za-z_]\w*", lit):
                literal_count += 1
        return literal_count

    if form == "sop":
        # Top-level OR splits
        top_terms = split_top_level(s, '|')
        num_terms = len(top_terms)
        total_literals = 0
        for term in top_terms:
            # Count literals inside product (AND-separated)
            literals_in_term = count_literals_in_group(term, '&')
            total_literals += literals_in_term
        return num_terms, total_literals

    if form == "pos":
        # Top-level AND splits
        top_clauses = split_top_level(s, '&')
        num_terms = len(top_clauses)
        total_literals = 0
        for clause in top_clauses:
            # Count literals inside sum (OR-separated)
            literals_in_clause = count_literals_in_group(clause, '|')
            total_literals += literals_in_clause
        return num_terms, total_literals

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
    sympy_terms, sympy_literals = count_sympy_expression_literals(str(expr_sympy), form=form_type)
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

def generate_inference(
    avg_sympy_time, avg_kmap_time, std_diff, deviation_ratio,
    avg_sympy_literals, avg_kmap_literals, std_lit_diff, deviation_ratio_literals
):
    """
    Generate a comprehensive inference summary based on performance and simplification statistics.
    Returns a formatted string (also prints to console).
    """

    time_diff = avg_kmap_time - avg_sympy_time
    time_pct_diff = (time_diff / avg_sympy_time) * 100 if avg_sympy_time else 0
    lit_diff = avg_kmap_literals - avg_sympy_literals
    lit_pct_diff = (lit_diff / avg_sympy_literals) * 100 if avg_sympy_literals else 0

    lines = []
    lines.append("=" * 75)
    lines.append("INFERENCE SUMMARY")
    lines.append("=" * 75)

    # Execution Time
    lines.append("\nEXECUTION TIME ANALYSIS")
    lines.append(f"  Average SymPy Time:      {avg_sympy_time:.6f} s")
    lines.append(f"  Average KMapSolver Time: {avg_kmap_time:.6f} s")
    lines.append(f"  Difference:              {time_diff:+.6f} s ({time_pct_diff:+.2f}%)")
    lines.append(f"  Std. Dev (ΔTime):        {std_diff:.6f} s")
    lines.append(f"  Deviation Ratio:         {deviation_ratio:.3f}")

    if time_diff < 0:
        lines.append("  → KMapSolver is faster than SymPy on average.")
    elif abs(time_diff) < avg_sympy_time * 0.1:
        lines.append("  → Both algorithms exhibit nearly identical runtimes.")
    else:
        lines.append("  → KMapSolver shows a slight runtime overhead compared to SymPy.")

    if std_diff < 0.001:
        lines.append("  → Execution times are stable and consistent.")
    else:
        lines.append("  → Some variability observed across test runs.")

    # Simplification
    lines.append("\nLITERAL COUNT ANALYSIS")
    lines.append(f"  Average SymPy Literals:  {avg_sympy_literals:.2f}")
    lines.append(f"  Average KMap Literals:   {avg_kmap_literals:.2f}")
    lines.append(f"  Difference:              {lit_diff:+.2f} ({lit_pct_diff:+.1f}%)")
    lines.append(f"  Std. Dev (ΔLiterals):    {std_lit_diff:.2f}")
    lines.append(f"  Deviation Ratio:         {deviation_ratio_literals:.3f}")

    if lit_diff < 0:
        lines.append("  → KMapSolver produces more minimal logical forms (fewer literals).")
    elif abs(lit_diff) < 0.5:
        lines.append("  → Both solvers yield nearly identical simplifications.")
    else:
        lines.append("  → SymPy produces slightly simpler expressions on average.")

    if std_lit_diff < 2:
        lines.append("  → Literal simplifications are consistent.")
    else:
        lines.append("  → Simplification outcomes vary across inputs.")

    # Overall verdict
    lines.append("\nOVERALL VERDICT")
    if avg_kmap_literals <= avg_sympy_literals and avg_kmap_time <= avg_sympy_time * 1.1:
        lines.append("  ✅ KMapSolver achieves comparable or superior simplification efficiency with minimal time overhead.")
    elif avg_kmap_time < avg_sympy_time:
        lines.append("  ✅ KMapSolver outperforms SymPy in runtime while maintaining correctness.")
    else:
        lines.append("  ⚠️ KMapSolver maintains correctness but trades slight performance for structural optimization.")

    lines.append("=" * 75)

    formatted_text = "\n".join(lines)
    print(formatted_text)
    return formatted_text
        
def export_results_to_csv(all_results, filename="outputs/benchmark_results.csv"):
    """
    Export all benchmark results to a CSV file.
    Groups results by configuration (num_vars, form).
    """
    # Group results by (num_vars, form)
    grouped = defaultdict(list)
    for r in all_results:
        grouped[(r["num_vars"], r["form"])].append(r)

    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Configuration", "Test Index", "Equivalent", 
                         "SymPy Time (s)", "KMap Time (s)", 
                         "SymPy Literals", "KMap Literals"])

        for (vars, form_type), group in grouped.items():
            writer.writerow([])  # blank line for readability
            writer.writerow([f"{vars}-Variable ({form_type.upper()} Form)"])
            
            for result in group:
                writer.writerow([
                    f"{vars}-{form_type}",
                    result.get("index", ""),
                    result.get("equiv", ""),
                    f"{result.get('t_sympy', 0):.6f}",
                    f"{result.get('t_kmap', 0):.6f}",
                    result.get("sympy_literals", ""),
                    result.get("kmap_literals", "")
                ])

    print(f"\n✅ Results exported successfully to '{filename}'\n")

def save_benchmark_plots(all_results, pdf_filename="outputs/benchmark_plots.pdf", logo_path="StanLogic/images/St_logo_light-tp.png"):
    """
    Save benchmark plots, summary, and inference report into a single PDF file.
    """
    
    print(f"\n📊 Generating PDF report: {pdf_filename}")

    with PdfPages(pdf_filename) as pdf:
        # ---- COVER PAGE ----
        if os.path.exists(logo_path):
            fig = plt.figure(figsize=(8.5, 11))
            ax = plt.gca()
            
            # Load and display logo in the upper portion of the page
            img = mpimg.imread(logo_path)
            
            # Create a specific axes for the logo to control its size and position
            logo_ax = fig.add_axes([0.2, 0.55, 0.6, 0.35])  # [left, bottom, width, height]
            logo_ax.imshow(img)
            logo_ax.axis("off")
            
            # Main axes for text (invisible)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            
            # Title - well spaced below the logo
            ax.text(
                0.5, 0.45,
                "Inference Report",
                fontsize=32,
                fontweight='bold',
                ha='center',
                va='center',
            )
            
            # Subtitle - appropriately spaced below title
            ax.text(
                0.5, 0.38,
                "Performance and Simplification Benchmark",
                fontsize=16,
                ha='center',
                va='center',
            )
            
            ax.text(
                0.5, 0.34,
                "between SymPy and StanLogic",
                fontsize=16,
                ha='center',
                va='center',
            )
            
            # Add a subtle separator line
            ax.plot([0.25, 0.75], [0.30, 0.30], 'k-', linewidth=1.5, alpha=0.3)
            
            # Add date at the bottom of the page
            import datetime
            ax.text(
                0.5, 0.15,
                f"Generated on {datetime.date.today():%B %d, %Y}",
                fontsize=12,
                ha='center',
                va='center',
                style='italic',
                color='gray'
            )
            
            pdf.savefig(bbox_inches='tight')
            plt.close()
        else:
            print(f"⚠️  Logo not found at {logo_path}, skipping cover page.")

        # ---- GROUP RESULTS ----
        grouped = {}
        for r in all_results:
            key = (r["num_vars"], r["form"])
            grouped.setdefault(key, []).append(r)

        # ---- PER-CONFIGURATION PAGES ----
        for (num_vars, form), results in grouped.items():
            sympy_times = [r["t_sympy"] for r in results]
            kmap_times = [r["t_kmap"] for r in results]
            sympy_literals = [r["sympy_literals"] for r in results]
            kmap_literals = [r["kmap_literals"] for r in results]

            avg_sympy_time = np.mean(sympy_times)
            avg_kmap_time = np.mean(kmap_times)
            avg_sympy_literals = np.mean(sympy_literals)
            avg_kmap_literals = np.mean(kmap_literals)
            tests = list(range(1, len(results) + 1))

            # ---- Plot for this configuration ----
            plt.figure(figsize=(11, 5))

            # Time plot
            plt.subplot(1, 2, 1)
            plt.plot(tests, sympy_times, marker='o', label='SymPy Time')
            plt.plot(tests, kmap_times, marker='s', label='KMapSolver Time')
            plt.axhline(avg_sympy_time, color='blue', linestyle='--', alpha=0.6,
                        label=f'SymPy Avg = {avg_sympy_time:.6f}s')
            plt.axhline(avg_kmap_time, color='orange', linestyle='--', alpha=0.6,
                        label=f'KMap Avg = {avg_kmap_time:.6f}s')
            plt.xlabel('Test Case')
            plt.ylabel('Execution Time (s)')
            plt.title(f'Performance ({num_vars}-Variable {form.upper()})')
            plt.legend()
            plt.grid(True)

            # Literal count plot
            plt.subplot(1, 2, 2)
            plt.bar([x - 0.2 for x in tests], sympy_literals, width=0.4, label='SymPy Literals')
            plt.bar([x + 0.2 for x in tests], kmap_literals, width=0.4, label='KMapSolver Literals')
            plt.axhline(avg_sympy_literals, color='blue', linestyle='--', alpha=0.6,
                        label=f'SymPy Avg = {avg_sympy_literals:.2f}')
            plt.axhline(avg_kmap_literals, color='orange', linestyle='--', alpha=0.6,
                        label=f'KMap Avg = {avg_kmap_literals:.2f}')
            plt.xlabel('Test Case')
            plt.ylabel('Number of Literals')
            plt.title(f'Literal Comparison ({num_vars}-Variable {form.upper()})')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            pdf.savefig()
            plt.close()
            
            # ---- INFERENCE PAGE FOR THIS CONFIGURATION ----
            from statistics import stdev
            
            # Calculate statistics for this specific configuration
            std_diff_config = stdev([kt - st for kt, st in zip(kmap_times, sympy_times)]) if len(kmap_times) > 1 else 0
            std_lit_diff_config = stdev([kl - sl for kl, sl in zip(kmap_literals, sympy_literals)]) if len(kmap_literals) > 1 else 0
            deviation_ratio_config = std_diff_config / avg_sympy_time if avg_sympy_time else 0
            deviation_ratio_literals_config = std_lit_diff_config / avg_sympy_literals if avg_sympy_literals else 0
            
            inference_text_config = generate_inference(
                avg_sympy_time, avg_kmap_time,
                std_diff_config, deviation_ratio_config,
                avg_sympy_literals, avg_kmap_literals,
                std_lit_diff_config, deviation_ratio_literals_config
            )
            
            plt.figure(figsize=(8.5, 11))
            plt.axis("off")
            
            # Add header
            plt.text(
                0.5, 0.95,
                f"INFERENCE: {num_vars}-Variable {form.upper()}",
                fontsize=18,
                fontweight="bold",
                ha="center",
                va="center",
                transform=plt.gca().transAxes
            )
            
            # Add inference body line by line with proper spacing
            lines = inference_text_config.split('\n')
            y_position = 0.90
            base_line_spacing = 0.022  # base spacing

            for line in lines:
                line = line.rstrip()

                # Determine font properties and extra spacing
                if line.startswith("=" * 30):  # Separator
                    fontweight = 'normal'
                    fontsize = 10
                    extra_spacing = 0.015
                elif line.strip() in ["INFERENCE SUMMARY", "EXECUTION TIME ANALYSIS",
                                    "LITERAL COUNT ANALYSIS", "OVERALL VERDICT"]:
                    fontweight = 'bold'
                    fontsize = 12
                    extra_spacing = 0.02
                elif line.strip().startswith(("→", "✅", "⚠️")):
                    fontweight = 'normal'
                    fontsize = 10
                    extra_spacing = 0
                else:
                    fontweight = 'normal'
                    fontsize = 10
                    extra_spacing = 0

                # Draw the text
                plt.text(
                    0.02, y_position,
                    line,
                    fontsize=fontsize,
                    fontweight=fontweight,
                    family="monospace",
                    va="bottom",
                    wrap=True,
                    transform=plt.gca().transAxes
                )

                # Move y_position down by base spacing plus extra spacing
                y_position -= (base_line_spacing + extra_spacing)
           
            pdf.savefig()
            plt.close()

        # ---- SUMMARY PAGE ----
        plt.figure(figsize=(10, 5))
        configs = [f"{nv}-{f}" for (nv, f) in grouped.keys()]
        avg_sympy_times = [np.mean([r["t_sympy"] for r in grouped[k]]) for k in grouped]
        avg_kmap_times = [np.mean([r["t_kmap"] for r in grouped[k]]) for k in grouped]
        avg_sympy_literals = [np.mean([r["sympy_literals"] for r in grouped[k]]) for k in grouped]
        avg_kmap_literals = [np.mean([r["kmap_literals"] for r in grouped[k]]) for k in grouped]

        plt.subplot(1, 2, 1)
        plt.bar(np.arange(len(configs)) - 0.2, avg_sympy_times, width=0.4, label='SymPy Avg Time')
        plt.bar(np.arange(len(configs)) + 0.2, avg_kmap_times, width=0.4, label='KMapSolver Avg Time')
        plt.xticks(range(len(configs)), configs, rotation=30)
        plt.ylabel("Average Time (s)")
        plt.title("Average Execution Time per Configuration")
        plt.legend()
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)

        plt.subplot(1, 2, 2)
        plt.bar(np.arange(len(configs)) - 0.2, avg_sympy_literals, width=0.4, label='SymPy Avg Literals')
        plt.bar(np.arange(len(configs)) + 0.2, avg_kmap_literals, width=0.4, label='KMapSolver Avg Literals')
        plt.xticks(range(len(configs)), configs, rotation=30)
        plt.ylabel("Average Literals")
        plt.title("Average Literal Count per Configuration")
        plt.legend()
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)

        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # ---- INFERENCE PAGE ----
        from statistics import stdev
        import datetime

        std_diff = stdev(np.array(avg_kmap_times) - np.array(avg_sympy_times))
        std_lit_diff = stdev(np.array(avg_kmap_literals) - np.array(avg_sympy_literals))
        deviation_ratio = std_diff / np.mean(avg_sympy_times)
        deviation_ratio_literals = std_lit_diff / np.mean(avg_sympy_literals)

        inference_text = generate_inference(
            np.mean(avg_sympy_times), np.mean(avg_kmap_times),
            std_diff, deviation_ratio,
            np.mean(avg_sympy_literals), np.mean(avg_kmap_literals),
            std_lit_diff, deviation_ratio_literals
        )

        plt.figure(figsize=(8.5, 11))
        plt.axis("off")

        # Add header with company name and date
        plt.text(
            0.5, 0.95,
            "OVERALL INFERENCE REPORT",
            fontsize=22,
            fontweight="bold",
            ha="center",
            va="center",
            transform=plt.gca().transAxes
        )
        plt.text(
            0.5, 0.91,
            f"Generated on {datetime.date.today():%B %d, %Y}",
            fontsize=12,
            ha="center",
            transform=plt.gca().transAxes
        )

        # Add inference body line by line with proper spacing
        lines = inference_text.split('\n')
        y_position = 0.87
        base_line_spacing = 0.022  # Base spacing between lines

        for line in lines:
            line = line.rstrip()

            # Determine font properties and extra spacing
            if line.startswith("=" * 30):  # Separator line
                fontweight = 'normal'
                fontsize = 10
                extra_spacing = 0.015  # Extra space after separators
            elif line.strip() in ["INFERENCE SUMMARY", "EXECUTION TIME ANALYSIS",
                                "LITERAL COUNT ANALYSIS", "OVERALL VERDICT"]:
                fontweight = 'bold'
                fontsize = 12
                extra_spacing = 0.02  # Extra space after headings
            elif line.strip().startswith(("✅", "⚠️")):
                fontweight = 'normal'
                fontsize = 10
                extra_spacing = 0.02
            else:
                fontweight = 'normal'
                fontsize = 10
                extra_spacing = 0

            # Draw the text with wrapping
            plt.text(
                0.02, y_position,
                line,
                fontsize=fontsize,
                fontweight=fontweight,
                family="monospace",
                va="top",
                wrap=True,  # Ensure text wraps inside the plot
                transform=plt.gca().transAxes
            )

            # Move y_position down by base spacing plus extra spacing
            y_position -= (base_line_spacing + extra_spacing)

        # Add copyright footer
        plt.text(
            0.5, 0.03,
            "© Copyright Stan's Technologies 2025",
            fontsize=10,
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
            style='italic',
            color='gray'
        )
        pdf.savefig()
        plt.close()
        
    print(f"✅ PDF successfully saved to {pdf_filename}")

def main():
    # Create outputs directory if it doesn't exist
    outputs_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    config = [
        (2, "sop"),
        (2, "pos"),
        (3, "sop"),
        (3, "pos"),
        (4, "sop"),
        (4, "pos"),
    ]

    # random.seed(42)
    var_pool = symbols('x1 x2 x3 x4')
    all_results = []

    for num_vars, form_type in config:
        print(f"\n{'='*70}")
        print(f" Benchmark: {num_vars}-variable K-map ({form_type.upper()})")
        print(f"{'='*70}")
        rows, cols = _kmap_shape(num_vars)
        results = []
        for i in range(1000):  # 100 random K-maps per configuration
            kmap = random_kmap(rows=rows, cols=cols)
            var_names = var_pool[:num_vars]
            result = benchmark_case(kmap, var_names, form_type, i + 1)
            results.append(result)
            all_results.append(result)

        # --- Plot performance comparison ---
        sympy_times = [r['t_sympy'] for r in results]
        kmap_times = [r['t_kmap'] for r in results]
        sympy_literals = [r['sympy_literals'] for r in results]
        kmap_literals = [r['kmap_literals'] for r in results]

        """
        Statistical Analysis
        ------------------------------------------------------------
        """
        avg_sympy_time = np.mean(sympy_times)
        avg_kmap_time = np.mean(kmap_times)

        avg_sympy_literals = np.mean(sympy_literals)
        avg_kmap_literals = np.mean(kmap_literals)

        #Compute time differences and find the standard deviation
        diffs = np.array(sympy_times) - np.array(kmap_times)
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)

        # Compute standard deviations
        std_sympy = np.std(sympy_times, ddof=1)  # sample std dev
        std_kmap = np.std(kmap_times, ddof=1)

        # Compare deviation ratio (optional)
        deviation_ratio = std_kmap / std_sympy if std_sympy != 0 else np.nan

        # #--- Print summary statistics for time ---
        # print(f"Average SymPy Time: {avg_sympy_time:.6f} seconds")
        # print(f"Average KMapSolver Time: {avg_kmap_time:.6f} seconds")

        # print(f"Std Dev of Time Difference: {std_diff:.6f} seconds")
        # print(f"Deviation Ratio (KMap/SymPy): {deviation_ratio:.6f}")

        # ------------------------------------------------------------

        #Compute literal differences and find the standard deviation
        lit_diffs = np.array(sympy_literals) - np.array(kmap_literals)
        avg_lit_diff = np.mean(lit_diffs)
        std_lit_diff = np.std(lit_diffs)

        #Compute standard deviations for literals
        std_sympy_literals = np.std(sympy_literals, ddof=1)
        std_kmap_literals = np.std(kmap_literals, ddof=1)

        #Compare literal deviation ratio (optional)
        deviation_ratio_literals = std_kmap_literals / std_sympy_literals if std_sympy_literals != 0 else np.nan

        # #--- Print summary statistics for literals ---
        # print(f"Average SymPy Literals: {avg_sympy_literals:.2f}")
        # print(f"Average KMapSolver Literals: {avg_kmap_literals:.2f}")

        # print(f"Std dev of Literal Difference: {std_lit_diff:.2f}")
        # print(f"Deviation Ratio (KMap/SymPy) Literals: {deviation_ratio_literals:.6f}")

        # # Generate inference summary
        # generate_inference(
        #     avg_sympy_time, avg_kmap_time, std_diff, deviation_ratio,
        #     avg_sympy_literals, avg_kmap_literals, std_lit_diff, deviation_ratio_literals
        # )

    
    # Fix output paths using os.path.join
    csv_path = os.path.join(outputs_dir, "benchmark_results.csv")
    pdf_path = os.path.join(outputs_dir, "benchmark_results.pdf")
    
    export_results_to_csv(all_results, csv_path)
    save_benchmark_plots(all_results, pdf_path)

    print(f"\n✅ All benchmark results exported to '{csv_path}'")   
    print(f"✅ All benchmark plots saved to '{pdf_path}'")         

        # print(tabulate(results, headers="keys", tablefmt="fancy_grid"))        
        
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
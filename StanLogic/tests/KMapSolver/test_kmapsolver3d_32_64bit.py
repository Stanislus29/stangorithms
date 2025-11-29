"""
KMapSolver3D Extreme Scale Test: 32-bit and 64-bit Boolean Minimization
=========================================================================

Performance evaluation of KMapSolver3D for extreme-scale Boolean function 
minimization (32 and 64 variables). These correspond to 32-bit and 64-bit 
computer architectures.

32-bit: 2^32 = 4,294,967,296 truth table entries
64-bit: 2^64 = 18,446,744,073,709,551,616 truth table entries

Author: Stan's Technologies
Date: November 2025
Version: 1.0 (Extreme Scale Performance)
"""

import csv
import datetime
import os
import random
import re
import sys
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# Add parent directory to path to import stanlogic
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from stanlogic.kmapsolver3D import KMapSolver3D

# ============================================================================
# CONFIGURATION
# ============================================================================

RANDOM_SEED = 42
TESTS_PER_DISTRIBUTION = 3  # Tests per distribution type
TIMING_REPEATS = 1
TIMING_WARMUP = 0

# Output files
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, "outputs")
RESULTS_CSV = os.path.join(OUTPUTS_DIR, "kmapsolver3d_32_64bit_performance.csv")
REPORT_PDF = os.path.join(OUTPUTS_DIR, "kmapsolver3d_32_64bit_performance_report.pdf")
LOGO_PATH = os.path.join(SCRIPT_DIR, "..", "..", "images", "St_logo_light-tp.png")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def count_literals(expr_str, form="sop"):
    """Count terms and literals in a Boolean expression string."""
    if not expr_str or expr_str.strip() == "":
        return 0, 0
    
    form = form.lower()
    s = expr_str.replace(" ", "")
    
    if form == "sop":
        terms = [t for t in s.split('+') if t]
        num_terms = len(terms)
        num_literals = sum(len(re.findall(r"[A-Za-z_]\w*'?", t)) for t in terms)
        return num_terms, num_literals
    
    if form == "pos":
        clauses = re.findall(r"\(([^()]*)\)", s)
        if not clauses:
            clauses = [s]
        num_terms = len(clauses)
        num_literals = 0
        for clause in clauses:
            lits = [lit for lit in clause.split('+') if lit]
            num_literals += sum(1 for lit in lits if re.fullmatch(r"[A-Za-z_]\w*'?", lit))
        return num_terms, num_literals
    
    raise ValueError("form must be 'sop' or 'pos'")


def is_truly_constant_function(output_values):
    """Determine if a Boolean function is genuinely constant based on its truth table."""
    defined_values = [v for v in output_values if v != 'd']
    
    if not defined_values:
        return True, "all_dc"
    
    if all(v == 0 for v in defined_values):
        return True, "all_zeros"
    elif all(v == 1 for v in defined_values):
        return True, "all_ones"
    
    return False, None


def benchmark_with_warmup(func, args=(), warmup=TIMING_WARMUP, repeats=TIMING_REPEATS):
    """Benchmark a function with warm-up runs and multiple repetitions."""
    for _ in range(warmup):
        func(*args)
    
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        func(*args)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    return min(times)


def format_large_number(n):
    """Format large numbers with comma separators."""
    return f"{n:,}"


def format_time(seconds):
    """Format time in human-readable format."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.2f}s"


# ============================================================================
# TEST DATA GENERATION
# ============================================================================

class TestDistribution:
    SPARSE = "sparse"
    DENSE = "dense"
    BALANCED = "balanced"
    MINIMAL_DC = "minimal_dc"
    HEAVY_DC = "heavy_dc"


def random_output_values(size, p_one=0.45, p_dc=0.1, seed=None):
    """Generate random output values with controlled probability distribution."""
    if seed is not None:
        random.seed(seed)
    
    if p_one < 0 or p_dc < 0 or p_one + p_dc > 1:
        raise ValueError("Probabilities invalid")
    
    p_zero = 1 - p_one - p_dc
    choices = [1, 0, 'd']
    weights = [p_one, p_zero, p_dc]
    
    return [random.choices(choices, weights=weights, k=1)[0] for _ in range(size)]


def generate_test_suite(num_vars, tests_per_distribution=TESTS_PER_DISTRIBUTION):
    """Generate comprehensive test suite with diverse distributions."""
    size = 2**num_vars
    test_cases = []
    
    print(f"      Truth table size: {format_large_number(size)} entries")
    
    distributions = {
        TestDistribution.SPARSE: {"p_one": 0.2, "p_dc": 0.05},
        TestDistribution.DENSE: {"p_one": 0.7, "p_dc": 0.05},
        TestDistribution.BALANCED: {"p_one": 0.5, "p_dc": 0.1},
        TestDistribution.MINIMAL_DC: {"p_one": 0.45, "p_dc": 0.02},
        TestDistribution.HEAVY_DC: {"p_one": 0.3, "p_dc": 0.3},
    }
    
    for dist_name, params in distributions.items():
        for i in range(tests_per_distribution):
            output_values = random_output_values(size=size, **params)
            test_cases.append((output_values, dist_name))
    
    # Add edge cases
    test_cases.append(([0] * size, "edge_all_zeros"))
    test_cases.append(([1] * size, "edge_all_ones"))
    test_cases.append((['d'] * size, "edge_all_dc"))
    test_cases.append(([(i % 2) for i in range(size)], "edge_checkerboard"))
    single_one = [0] * size
    single_one[0] = 1
    test_cases.append((single_one, "edge_single_one"))
    
    return test_cases


def generate_test_cases():
    """Generate all test cases for 32-bit and 64-bit K-maps."""
    random.seed(RANDOM_SEED)
    test_cases = []
    
    for num_vars in [32, 64]:
        bit_size = num_vars
        print(f"\n  Generating {num_vars}-variable ({bit_size}-bit) test suite...")
        suite = generate_test_suite(num_vars, tests_per_distribution=TESTS_PER_DISTRIBUTION)
        for output_values, distribution in suite:
            test_cases.append((num_vars, output_values, f"{bit_size}-bit: {distribution}"))
        print(f"      ✓ {len(suite)} test cases generated")
    
    return test_cases


# ============================================================================
# TEST EXECUTION
# ============================================================================

def benchmark_case(test_num, num_vars, output_values, description, form='sop'):
    """Run a single performance test case for KMapSolver3D."""
    bit_size = num_vars
    print(f"    Test {test_num}: {bit_size}-bit ({description})...", end=" ", flush=True)
    
    # Suppress verbose output from minimize_3d
    import io
    import sys as sys_module
    old_stdout = sys_module.stdout
    sys_module.stdout = io.StringIO()
    
    solver = KMapSolver3D(num_vars, output_values)
    kmap_func = lambda: solver.minimize_3d(form=form)
    
    start_time = time.perf_counter()
    t_kmap = benchmark_with_warmup(kmap_func, ())
    terms, expr_str = kmap_func()
    elapsed_time = time.perf_counter() - start_time
    
    sys_module.stdout = old_stdout
    
    # Compute metrics
    kmap_terms, kmap_literals = count_literals(expr_str, form=form)
    
    # Detect constants
    is_constant, constant_type = is_truly_constant_function(output_values)
    result_category = constant_type if is_constant else "expression"
    
    if is_constant:
        print(f"✓ {format_time(elapsed_time)} [CONSTANT: {constant_type}]")
    else:
        print(f"✓ {format_time(elapsed_time)} | Literals: {kmap_literals}")
    
    return {
        "index": test_num,
        "num_vars": num_vars,
        "bit_size": bit_size,
        "description": description,
        "form": form,
        "result_category": result_category,
        "is_constant": is_constant,
        "t_kmap": t_kmap,
        "elapsed_time": elapsed_time,
        "kmap_literals": kmap_literals,
        "kmap_terms": kmap_terms,
    }


# ============================================================================
# RESULTS EXPORT
# ============================================================================

def export_results_to_csv(all_results, csv_path):
    """Export results to CSV file."""
    print(f"\n   • Exporting to CSV...", end=" ", flush=True)
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            "Index", "Variables", "Bit_Size", "Description", "Form",
            "Category", "Is_Constant", "Time_KMap(s)", "Elapsed_Time(s)",
            "KMap_Literals", "KMap_Terms"
        ])
        
        # Data rows
        for r in all_results:
            writer.writerow([
                r["index"],
                r["num_vars"],
                r["bit_size"],
                r["description"],
                r["form"],
                r["result_category"],
                r["is_constant"],
                f"{r['t_kmap']:.6f}",
                f"{r['elapsed_time']:.6f}",
                r["kmap_literals"],
                r["kmap_terms"]
            ])
    
    print("✓")
    print(f"      {csv_path}")


def save_performance_report(all_results, pdf_path):
    """Generate comprehensive PDF performance report."""
    print(f"\n   • Generating PDF report...")
    
    with PdfPages(pdf_path) as pdf:
        # Title Page
        print(f"      • Creating title page...", end=" ", flush=True)
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Try to add logo
        if os.path.exists(LOGO_PATH):
            try:
                img = mpimg.imread(LOGO_PATH)
                logo_ax = fig.add_axes([0.35, 0.70, 0.30, 0.15])
                logo_ax.imshow(img)
                logo_ax.axis("off")
            except:
                pass
        
        ax.text(0.5, 0.55, "KMapSolver3D Extreme Scale Report", 
                fontsize=26, fontweight='bold', ha='center')
        ax.text(0.5, 0.49, "32-bit & 64-bit Boolean Minimization",
                fontsize=16, ha='center')
        
        ax.plot([0.2, 0.8], [0.45, 0.45], 'k-', linewidth=2, alpha=0.3)
        
        ax.text(0.5, 0.38, f"32-bit: 2³² = {format_large_number(2**32)} entries",
                fontsize=11, ha='center', family='monospace')
        ax.text(0.5, 0.34, f"64-bit: 2⁶⁴ = {format_large_number(2**64)} entries",
                fontsize=11, ha='center', family='monospace')
        
        ax.text(0.5, 0.25, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d')}",
                fontsize=11, ha='center')
        ax.text(0.5, 0.21, f"Random Seed: {RANDOM_SEED}",
                fontsize=11, ha='center')
        ax.text(0.5, 0.17, f"Total Test Cases: {len(all_results)}",
                fontsize=11, ha='center')
        
        ax.text(0.5, 0.08, "Testing extreme-scale Boolean minimization capabilities",
                fontsize=9, ha='center', style='italic', color='gray')
        ax.text(0.5, 0.04, f"© Stan's Technologies {datetime.datetime.now().year}",
                fontsize=9, ha='center', color='gray')
        
        pdf.savefig(bbox_inches='tight')
        plt.close()
        print("✓")
        
        # Performance Summary
        print(f"      • Creating performance summary...", end=" ", flush=True)
        
        grouped = defaultdict(list)
        for r in all_results:
            grouped[r['num_vars']].append(r)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('32-bit & 64-bit Performance Analysis', fontsize=16, fontweight='bold')
        
        bit_sizes = sorted(grouped.keys())
        colors = ['#1f77b4', '#ff7f0e']
        
        for idx, bit_size in enumerate(bit_sizes):
            results = grouped[bit_size]
            non_const = [r for r in results if not r.get('is_constant', False)]
            
            times = [r['elapsed_time'] for r in results]
            literals = [r['kmap_literals'] for r in non_const] if non_const else [0]
            
            label = f"{bit_size}-bit"
            
            # Plot 1: Execution Time Distribution
            ax1.hist(times, bins=10, alpha=0.7, label=label, color=colors[idx])
            
            # Plot 2: Mean Time
            ax2.bar(idx, np.mean(times), color=colors[idx], alpha=0.7)
            ax2.text(idx, np.mean(times), f"{format_time(np.mean(times))}", 
                    ha='center', va='bottom', fontsize=9)
            
            # Plot 3: Literal Count Distribution
            if non_const:
                ax3.hist(literals, bins=10, alpha=0.7, label=label, color=colors[idx])
            
            # Plot 4: Mean Literals
            if non_const:
                ax4.bar(idx, np.mean(literals), color=colors[idx], alpha=0.7)
                ax4.text(idx, np.mean(literals), f"{np.mean(literals):.1f}", 
                        ha='center', va='bottom', fontsize=9)
        
        ax1.set_xlabel('Execution Time (s)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Execution Time Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_ylabel('Mean Execution Time (s)')
        ax2.set_title('Average Performance')
        ax2.set_xticks(range(len(bit_sizes)))
        ax2.set_xticklabels([f"{b}-bit" for b in bit_sizes])
        ax2.grid(True, alpha=0.3, axis='y')
        
        ax3.set_xlabel('Literal Count')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Solution Complexity Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4.set_ylabel('Mean Literal Count')
        ax4.set_title('Average Solution Size')
        ax4.set_xticks(range(len(bit_sizes)))
        ax4.set_xticklabels([f"{b}-bit" for b in bit_sizes])
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()
        print("✓")
        
        # Detailed Statistics Page
        print(f"      • Creating statistics page...", end=" ", flush=True)
        
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        y_pos = 0.95
        ax.text(0.5, y_pos, 'Detailed Performance Statistics', 
                fontsize=16, fontweight='bold', ha='center')
        y_pos -= 0.08
        
        for bit_size in bit_sizes:
            results = grouped[bit_size]
            non_const = [r for r in results if not r.get('is_constant', False)]
            const_count = len(results) - len(non_const)
            
            times = [r['elapsed_time'] for r in results]
            literals = [r['kmap_literals'] for r in non_const] if non_const else [0]
            
            ax.text(0.5, y_pos, f'{bit_size}-bit Performance ({2**bit_size:,} entries)', 
                    fontsize=14, fontweight='bold', ha='center')
            y_pos -= 0.05
            
            stats_text = f"""
Test Cases: {len(results)}
  • Constant Functions: {const_count} ({100*const_count/len(results):.1f}%)
  • Non-constant: {len(non_const)} ({100*len(non_const)/len(results):.1f}%)

Execution Time:
  • Mean: {format_time(np.mean(times))}
  • Median: {format_time(np.median(times))}
  • Min: {format_time(np.min(times))}
  • Max: {format_time(np.max(times))}
  • Std Dev: {np.std(times):.3f}s
"""
            
            if non_const:
                stats_text += f"""
Solution Complexity (Non-constant):
  • Mean Literals: {np.mean(literals):.2f}
  • Median Literals: {np.median(literals):.2f}
  • Min Literals: {int(np.min(literals))}
  • Max Literals: {int(np.max(literals))}
"""
            
            ax.text(0.1, y_pos, stats_text, fontsize=10, va='top', family='monospace')
            y_pos -= 0.35
            
            if y_pos < 0.15:
                break
        
        ax.text(0.5, 0.05, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                fontsize=8, ha='center', color='gray')
        
        pdf.savefig(bbox_inches='tight')
        plt.close()
        print("✓")
        
        # Set PDF metadata
        d = pdf.infodict()
        d['Title'] = 'KMapSolver3D 32-bit & 64-bit Performance Report'
        d['Author'] = "Stan's Technologies"
        d['Subject'] = 'Extreme Scale Boolean Minimization Performance'
        d['Keywords'] = '32-bit, 64-bit, Boolean minimization, K-map, performance'
        d['CreationDate'] = datetime.datetime.now()
    
    print(f"      {pdf_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main test execution function."""
    print("=" * 80)
    print("KMapSolver3D EXTREME SCALE TEST (32-bit & 64-bit)")
    print("=" * 80)
    print("\n⚠️  WARNING: These tests involve extremely large truth tables:")
    print(f"    32-bit: {format_large_number(2**32)} entries (4.3 billion)")
    print(f"    64-bit: {format_large_number(2**64)} entries (18.4 quintillion)")
    print("\n    Execution may take considerable time and memory.")
    
    # Set random seed
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    print(f"\n🎲 Random seed: {RANDOM_SEED}")
    
    # Create outputs directory
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    print(f"📁 Output directory: {OUTPUTS_DIR}")
    
    # Generate test cases
    print(f"\n📋 Generating test suite...")
    test_cases = generate_test_cases()
    print(f"\n   Total: {len(test_cases)} test cases")
    
    # Run tests
    print(f"\n⚙️  Running extreme scale performance tests...")
    print(f"\n{'='*80}")
    all_results = []
    
    grouped_cases = defaultdict(list)
    for num_vars, output_values, description in test_cases:
        grouped_cases[num_vars].append((output_values, description))
    
    for var_num, cases in sorted(grouped_cases.items()):
        bit_size = var_num
        print(f"\n  {bit_size}-bit K-maps ({len(cases)} cases)")
        print(f"  " + "-" * 60)
        
        for idx, (output_values, description) in enumerate(cases, 1):
            try:
                result = benchmark_case(
                    len(all_results) + 1,
                    var_num,
                    output_values,
                    description,
                    form='sop'
                )
                all_results.append(result)
            except Exception as e:
                print(f"    ⚠️  Error in test {idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Export results
    print(f"\n{'='*80}")
    print("EXPORTING RESULTS")
    print(f"{'='*80}")
    
    export_results_to_csv(all_results, RESULTS_CSV)
    save_performance_report(all_results, REPORT_PDF)
    
    # Final summary
    print(f"\n{'='*80}")
    print("✅ EXTREME SCALE TEST COMPLETE")
    print(f"{'='*80}")
    print(f"📊 Total tests: {len(all_results)}")
    constant_count = sum(1 for r in all_results if r.get('is_constant', False))
    print(f"   Constant functions: {constant_count} ({100*constant_count/len(all_results):.1f}%)")
    print(f"   Non-constant: {len(all_results) - constant_count}")
    
    # Time statistics
    total_time = sum(r['elapsed_time'] for r in all_results)
    print(f"\n⏱️  Total execution time: {format_time(total_time)}")
    print(f"   Average per test: {format_time(total_time/len(all_results))}")
    
    print(f"\n📂 Outputs:")
    print(f"   • CSV results:  {RESULTS_CSV}")
    print(f"   • PDF report:   {REPORT_PDF}")
    print(f"\n{'='*80}")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

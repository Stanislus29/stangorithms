"""
KMapSolver3D Performance Characterization: 9-16 Variable Boolean Minimization
==============================================================================

PERFORMANCE-ONLY STUDY (No SymPy comparison - see verify_sympy_failure.py)

This benchmark characterizes KMapSolver3D's:
- Execution time scaling (9-16 variables)
- Memory consumption patterns
- Solution quality (literal counts)
- Distribution sensitivity
- Practical performance limits

Author: Stan's Technologies  
Date: November 2025
Version: 5.0 (Performance Characterization)
"""

import csv
import datetime
import os
import platform
import random
import re
import sys
import time
import tracemalloc
import psutil
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
from stanlogic.kmapsolver3D import KMapSolver3D

# ============================================================================
# CONFIGURATION
# ============================================================================

RANDOM_SEED = 42
TESTS_PER_DISTRIBUTION = 3  # Per distribution type
VAR_RANGE = range(9, 17)    # 9-16 variables

# Output paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, "outputs")
RESULTS_CSV = os.path.join(OUTPUTS_DIR, "kmapsolver3d_9to16var_performance.csv")
REPORT_PDF = os.path.join(OUTPUTS_DIR, "kmapsolver3d_9to16var_performance_report.pdf")
LOGO_PATH = os.path.join(SCRIPT_DIR, "..", "..", "images", "St_logo_light-tp.png")

# ============================================================================
# PERFORMANCE MEASUREMENT
# ============================================================================

def get_memory_usage_mb():
    """Get current process memory in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

def benchmark_with_resources(func):
    """Measure execution time and memory usage."""
    tracemalloc.start()
    mem_before = get_memory_usage_mb()
    
    start = time.perf_counter()
    result = func()
    elapsed = time.perf_counter() - start
    
    mem_after = get_memory_usage_mb()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return result, elapsed, mem_after - mem_before, peak / 1024 / 1024

# ============================================================================
# TEST GENERATION
# ============================================================================

class Distribution:
    SPARSE = {"p_one": 0.2, "p_dc": 0.05, "label": "Sparse (20% 1s)"}
    DENSE = {"p_one": 0.7, "p_dc": 0.05, "label": "Dense (70% 1s)"}
    BALANCED = {"p_one": 0.5, "p_dc": 0.1, "label": "Balanced (50% 1s)"}
    MINIMAL_DC = {"p_one": 0.45, "p_dc": 0.02, "label": "Minimal DC (2%)"}
    HEAVY_DC = {"p_one": 0.3, "p_dc": 0.3, "label": "Heavy DC (30%)"}

def generate_output_values(size, p_one, p_dc):
    """Generate random Boolean function output."""
    p_zero = 1 - p_one - p_dc
    choices = [1, 0, 'd']
    weights = [p_one, p_zero, p_dc]
    return [random.choices(choices, weights=weights)[0] for _ in range(size)]

def is_constant_function(output_values):
    """Check if function is trivially constant."""
    defined = [v for v in output_values if v != 'd']
    if not defined:
        return True, "all_dc"
    if all(v == 0 for v in defined):
        return True, "all_zeros"
    if all(v == 1 for v in defined):
        return True, "all_ones"
    return False, None

def count_literals(expr_str):
    """Count literals in SOP expression."""
    if not expr_str or not expr_str.strip():
        return 0, 0
    terms = [t for t in expr_str.split('+') if t.strip()]
    literals = sum(len(re.findall(r"[A-Za-z_]\w*'?", t)) for t in terms)
    return len(terms), literals

# ============================================================================
# BENCHMARK EXECUTION
# ============================================================================

def run_single_test(test_num, num_vars, output_values, dist_name):
    """Execute single performance test."""
    print(f"    Test {test_num}: ", end="", flush=True)
    
    # Check if constant
    is_const, const_type = is_constant_function(output_values)
    
    # Suppress verbose output
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    solver = KMapSolver3D(num_vars, output_values)
    (terms, expr), time_s, mem_mb, peak_mem_mb = benchmark_with_resources(
        lambda: solver.minimize_3d(form='sop')
    )
    
    sys.stdout = old_stdout
    
    num_terms, num_literals = count_literals(expr)
    
    print(f"✓ {time_s:.3f}s | {mem_mb:.1f}MB | {num_literals} lits", end="")
    if is_const:
        print(f" [CONST:{const_type}]")
    else:
        print()
    
    return {
        "test_num": test_num,
        "num_vars": num_vars,
        "distribution": dist_name,
        "is_constant": is_const,
        "constant_type": const_type,
        "time_s": time_s,
        "memory_mb": mem_mb,
        "peak_memory_mb": peak_mem_mb,
        "num_terms": num_terms,
        "num_literals": num_literals,
        "truth_table_size": 2**num_vars,
        "time_per_entry_ms": (time_s * 1000) / (2**num_vars),
        "memory_per_entry_kb": (mem_mb * 1024) / (2**num_vars),
    }

# ============================================================================
# COMPREHENSIVE VISUALIZATIONS
# ============================================================================

def create_performance_report(all_results, pdf):
    """Generate comprehensive performance visualization suite."""
    
    # Data aggregation
    configs = sorted(set(r['num_vars'] for r in all_results))
    distributions = ['Sparse', 'Dense', 'Balanced', 'Minimal DC', 'Heavy DC']
    
    # ===== PLOT 1: Scalability Dashboard (4 subplots) =====
    print("   • Creating scalability dashboard...", end=" ", flush=True)
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Subplot 1: Absolute Time (log scale)
    ax1 = fig.add_subplot(gs[0, 0])
    times_by_var = [np.mean([r['time_s'] for r in all_results if r['num_vars'] == v]) 
                    for v in configs]
    ax1.semilogy(configs, times_by_var, 'bo-', linewidth=2, markersize=8)
    ax1.fill_between(configs, 
                     [np.min([r['time_s'] for r in all_results if r['num_vars'] == v]) for v in configs],
                     [np.max([r['time_s'] for r in all_results if r['num_vars'] == v]) for v in configs],
                     alpha=0.2)
    ax1.set_xlabel('Number of Variables', fontsize=11)
    ax1.set_ylabel('Execution Time (s) - Log Scale', fontsize=11)
    ax1.set_title('A) Absolute Execution Time', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0.1, color='green', linestyle='--', alpha=0.5, linewidth=1, label='Real-time (100ms)')
    ax1.axhline(1.0, color='orange', linestyle='--', alpha=0.5, linewidth=1, label='Interactive (1s)')
    ax1.legend(fontsize=9)
    
    # Subplot 2: Time per Truth Table Entry
    ax2 = fig.add_subplot(gs[0, 1])
    time_per_entry = [np.mean([r['time_per_entry_ms'] for r in all_results if r['num_vars'] == v]) 
                      for v in configs]
    ax2.plot(configs, time_per_entry, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Variables', fontsize=11)
    ax2.set_ylabel('Time per Entry (ms)', fontsize=11)
    ax2.set_title('B) Efficiency (Time per Truth Table Entry)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Memory Consumption
    ax3 = fig.add_subplot(gs[1, 0])
    mem_by_var = [np.mean([r['memory_mb'] for r in all_results if r['num_vars'] == v]) 
                  for v in configs]
    peak_by_var = [np.mean([r['peak_memory_mb'] for r in all_results if r['num_vars'] == v]) 
                   for v in configs]
    ax3.plot(configs, mem_by_var, 'go-', label='Average', linewidth=2, markersize=8)
    ax3.plot(configs, peak_by_var, 'g^--', label='Peak', linewidth=2, markersize=8, alpha=0.7)
    ax3.set_xlabel('Number of Variables', fontsize=11)
    ax3.set_ylabel('Memory Usage (MB)', fontsize=11)
    ax3.set_title('C) Memory Consumption', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Solution Complexity (Literal Count)
    ax4 = fig.add_subplot(gs[1, 1])
    non_const = [r for r in all_results if not r['is_constant']]
    lits_by_var = [np.mean([r['num_literals'] for r in non_const if r['num_vars'] == v]) 
                   for v in configs]
    ax4.plot(configs, lits_by_var, 'mo-', linewidth=2, markersize=8)
    ax4.set_xlabel('Number of Variables', fontsize=11)
    ax4.set_ylabel('Average Literal Count', fontsize=11)
    ax4.set_title('D) Solution Complexity (Non-constant Functions)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Subplot 5-6: Exponential Model Fit
    ax5 = fig.add_subplot(gs[2, :])
    
    # Fit exponential model
    def exp_model(n, a, b):
        return a * (b ** n)
    
    params, _ = curve_fit(exp_model, configs, times_by_var, p0=[0.001, 1.5])
    
    # Plot measured vs predicted
    ax5.semilogy(configs, times_by_var, 'bo', markersize=10, label='Measured', zorder=3)
    extended_vars = np.linspace(min(configs), max(configs)+4, 100)
    predicted = exp_model(extended_vars, *params)
    ax5.semilogy(extended_vars, predicted, 'r-', linewidth=2, label=f'Model: T ≈ {params[0]:.2e} × {params[1]:.3f}^n')
    
    # Add projections
    for proj_var in [17, 18, 19, 20]:
        proj_time = exp_model(proj_var, *params)
        ax5.plot(proj_var, proj_time, 'rs', markersize=8, zorder=2)
        ax5.annotate(f'{proj_var}-var\n{proj_time:.2f}s', 
                    xy=(proj_var, proj_time), xytext=(5, 10),
                    textcoords='offset points', fontsize=9, color='red')
    
    ax5.set_xlabel('Number of Variables', fontsize=11)
    ax5.set_ylabel('Execution Time (s) - Log Scale', fontsize=11)
    ax5.set_title('E) Exponential Growth Model & Extrapolation', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    ax5.axhline(1.0, color='orange', linestyle='--', alpha=0.4, linewidth=1)
    ax5.axhline(10.0, color='red', linestyle='--', alpha=0.4, linewidth=1)
    
    plt.suptitle('KMapSolver3D Performance Characterization (9-16 Variables)', 
                 fontsize=16, fontweight='bold', y=0.995)
    pdf.savefig(bbox_inches='tight')
    plt.close()
    print("✓")
    
    # ===== PLOT 2: Distribution Sensitivity =====
    print("   • Creating distribution analysis...", end=" ", flush=True)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    dist_names = ['sparse', 'dense', 'balanced', 'minimal_dc', 'heavy_dc']
    dist_labels = ['Sparse\n20% 1s', 'Dense\n70% 1s', 'Balanced\n50% 1s', 
                   'Minimal DC\n2% dc', 'Heavy DC\n30% dc']
    
    # Time by distribution (aggregated)
    times_by_dist = []
    for dist in dist_names:
        results = [r for r in all_results if dist in r['distribution'].lower() and not r['is_constant']]
        times_by_dist.append(np.mean([r['time_s'] for r in results]) if results else 0)
    
    ax1.bar(range(len(dist_labels)), times_by_dist, color='steelblue', alpha=0.8)
    ax1.set_xticks(range(len(dist_labels)))
    ax1.set_xticklabels(dist_labels, fontsize=10)
    ax1.set_ylabel('Average Time (s)', fontsize=11)
    ax1.set_title('A) Execution Time by Distribution', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Literals by distribution
    lits_by_dist = []
    for dist in dist_names:
        results = [r for r in all_results if dist in r['distribution'].lower() and not r['is_constant']]
        lits_by_dist.append(np.mean([r['num_literals'] for r in results]) if results else 0)
    
    ax2.bar(range(len(dist_labels)), lits_by_dist, color='coral', alpha=0.8)
    ax2.set_xticks(range(len(dist_labels)))
    ax2.set_xticklabels(dist_labels, fontsize=10)
    ax2.set_ylabel('Average Literals', fontsize=11)
    ax2.set_title('B) Solution Complexity by Distribution', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Heatmap: Time by (Variables × Distribution)
    time_matrix = np.zeros((len(configs), len(dist_names)))
    for i, nv in enumerate(configs):
        for j, dist in enumerate(dist_names):
            results = [r for r in all_results 
                      if r['num_vars'] == nv and dist in r['distribution'].lower() and not r['is_constant']]
            time_matrix[i, j] = np.mean([r['time_s'] for r in results]) if results else 0
    
    im = ax3.imshow(time_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax3.set_xticks(range(len(dist_labels)))
    ax3.set_xticklabels([d.split('\n')[0] for d in dist_labels], rotation=45, ha='right', fontsize=9)
    ax3.set_yticks(range(len(configs)))
    ax3.set_yticklabels([f'{v}-var' for v in configs], fontsize=9)
    ax3.set_title('C) Time Heatmap (s)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax3, label='Time (s)')
    
    # Heatmap: Literals
    lit_matrix = np.zeros((len(configs), len(dist_names)))
    for i, nv in enumerate(configs):
        for j, dist in enumerate(dist_names):
            results = [r for r in all_results 
                      if r['num_vars'] == nv and dist in r['distribution'].lower() and not r['is_constant']]
            lit_matrix[i, j] = np.mean([r['num_literals'] for r in results]) if results else 0
    
    im2 = ax4.imshow(lit_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
    ax4.set_xticks(range(len(dist_labels)))
    ax4.set_xticklabels([d.split('\n')[0] for d in dist_labels], rotation=45, ha='right', fontsize=9)
    ax4.set_yticks(range(len(configs)))
    ax4.set_yticklabels([f'{v}-var' for v in configs], fontsize=9)
    ax4.set_title('D) Literal Count Heatmap', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=ax4, label='Literals')
    
    plt.suptitle('Distribution Sensitivity Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    pdf.savefig(bbox_inches='tight')
    plt.close()
    print("✓")
    
    # ===== PLOT 3: Practical Performance Limits =====
    print("   • Creating performance limits analysis...", end=" ", flush=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Determine practical limits
    thresholds = {
        "Real-time\n(<100ms)": 0.1,
        "Interactive\n(<1s)": 1.0,
        "Acceptable\n(<10s)": 10.0,
        "Batch\n(<60s)": 60.0
    }
    
    limit_vars = []
    for label, threshold in thresholds.items():
        for n in range(min(configs), 25):
            if exp_model(n, *params) > threshold:
                limit_vars.append(n - 1)
                break
        else:
            limit_vars.append(24)  # Beyond 24 vars
    
    ax1.barh(list(thresholds.keys()), limit_vars, color='green', alpha=0.7)
    ax1.set_xlabel('Maximum Variables', fontsize=11)
    ax1.set_title('A) Practical Performance Limits', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Project to larger sizes
    proj_vars = list(range(9, 21))
    proj_times = [exp_model(v, *params) for v in proj_vars]
    
    ax2.semilogy(proj_vars, proj_times, 'bo-', linewidth=2, markersize=8)
    ax2.fill_between(proj_vars[:len(configs)], 
                     times_by_var, 
                     alpha=0.3, label='Measured range')
    ax2.axhline(0.1, color='green', linestyle='--', alpha=0.6, label='Real-time')
    ax2.axhline(1.0, color='orange', linestyle='--', alpha=0.6, label='Interactive')
    ax2.axhline(10.0, color='red', linestyle='--', alpha=0.6, label='Batch limit')
    ax2.axvline(max(configs), color='gray', linestyle=':', alpha=0.6, label='Extrapolation')
    ax2.set_xlabel('Number of Variables', fontsize=11)
    ax2.set_ylabel('Execution Time (s) - Log Scale', fontsize=11)
    ax2.set_title('B) Performance Projection to 20 Variables', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Practical Application Limits', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(bbox_inches='tight')
    plt.close()
    print("✓")
    
    return params

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*80)
    print("KMapSolver3D Performance Characterization (9-16 Variables)")
    print("="*80)
    print("\nNOTE: This is a performance-only study.")
    print("      SymPy comparison omitted - see verify_sympy_failure.py\n")
    
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    all_results = []
    test_counter = 0
    
    distributions = [
        ("sparse", Distribution.SPARSE),
        ("dense", Distribution.DENSE),
        ("balanced", Distribution.BALANCED),
        ("minimal_dc", Distribution.MINIMAL_DC),
        ("heavy_dc", Distribution.HEAVY_DC)
    ]
    
    # Run tests
    for num_vars in VAR_RANGE:
        print(f"\n{'='*80}")
        print(f"Testing {num_vars}-variable K-maps (2^{num_vars} = {2**num_vars} entries)")
        print(f"{'='*80}")
        
        for dist_name, dist_config in distributions:
            print(f"  Distribution: {dist_config['label']}")
            
            for i in range(TESTS_PER_DISTRIBUTION):
                test_counter += 1
                output_values = generate_output_values(
                    2**num_vars, 
                    dist_config['p_one'], 
                    dist_config['p_dc']
                )
                
                result = run_single_test(test_counter, num_vars, output_values, dist_name)
                all_results.append(result)
        
        # Add edge cases
        print("  Edge cases:")
        for case_name, values in [
            ("all_zeros", [0] * (2**num_vars)),
            ("all_ones", [1] * (2**num_vars)),
            ("all_dc", ['d'] * (2**num_vars))
        ]:
            test_counter += 1
            result = run_single_test(test_counter, num_vars, values, f"edge_{case_name}")
            all_results.append(result)
    
    # Export CSV
    print(f"\n{'='*80}")
    print("Exporting Results")
    print(f"{'='*80}")
    
    with open(RESULTS_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"✓ CSV exported: {RESULTS_CSV}")
    
    # Generate PDF report
    print(f"\n📊 Generating comprehensive PDF report...")
    with PdfPages(REPORT_PDF) as pdf:
        # Cover page
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.gca()
        ax.axis('off')
        ax.text(0.5, 0.6, 'KMapSolver3D', fontsize=36, ha='center', fontweight='bold')
        ax.text(0.5, 0.52, 'Performance Characterization', fontsize=24, ha='center')
        ax.text(0.5, 0.45, '9-16 Variable Boolean Functions', fontsize=18, ha='center')
        ax.text(0.5, 0.3, f'Total Tests: {len(all_results)}', fontsize=14, ha='center')
        ax.text(0.5, 0.25, f'Date: {datetime.datetime.now().strftime("%Y-%m-%d")}', 
                fontsize=12, ha='center')
        ax.text(0.5, 0.1, '© Stan\'s Technologies 2025', fontsize=10, ha='center', style='italic')
        pdf.savefig()
        plt.close()
        
        # Generate plots
        model_params = create_performance_report(all_results, pdf)
    
    print(f"✓ PDF report: {REPORT_PDF}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    print(f"Total tests: {len(all_results)}")
    print(f"Variable range: {min(VAR_RANGE)}-{max(VAR_RANGE)}")
    print(f"\nExponential Model: T ≈ {model_params[0]:.2e} × {model_params[1]:.3f}^n")
    print(f"\nProjected Performance:")
    for n in [17, 18, 19, 20]:
        def exp_model(n, a, b):
            return a * (b ** n)
        t = exp_model(n, *model_params)
        print(f"  {n} variables: {t:.3f}s ({t/60:.2f} min)")
    
    print(f"\n{'='*80}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
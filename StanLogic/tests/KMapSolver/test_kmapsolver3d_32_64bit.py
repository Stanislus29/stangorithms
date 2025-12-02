"""
KMapSolver3D Extreme Scale Test: 32-bit Boolean Minimization
==============================================================

Performance evaluation of KMapSolver3D for 32-bit Boolean function 
minimization using chunked processing with step-by-step truth table generation.

32-bit: 2^32 = 4,294,967,296 truth table entries

Algorithm (matching KMapSolver3D structure):
- Process truth table in chunks (2^16 = 65,536 entries per chunk)
- For each chunk (defined by first 16 variables fixed):
  - First 16 variables (bits 0-15) = chunk_index (constant for this chunk)
  - Last 16 variables (bits 16-31) = vary through all 65,536 combinations
  - Build 16-variable K-map with these combinations
  - Minimize using KMapSolver3D (which uses first 12 as extra_vars, last 4 as map)
  - Extract ESSENTIAL prime implicant bitmasks using _solve_single_kmap()
  - Write essential prime implicants to CSV
- After all chunks processed, perform bitwise union on bitmasks
- Extract final minimal terms

Bit structure (32 variables total):
  Bits 0-15:  chunk_index (first 16 variables)
  Bits 16-27: extra_combo (next 12 variables, selects 4×4 map)
  Bits 28-31: map position (last 4 variables, position in 4×4 map)

This approach ensures efficient stack management by completing and 
discarding each chunk before processing the next.

Author: Stan's Technologies
Date: December 2025
Version: 2.0 (Chunked Processing with 16-Variable Maps and CSV Storage)
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

from stanlogic import KMapSolver3D

# ============================================================================
# CONFIGURATION
# ============================================================================

RANDOM_SEED = 42
TIMING_REPEATS = 1
TIMING_WARMUP = 0

# Output files
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, "outputs")
RESULTS_CSV = os.path.join(OUTPUTS_DIR, "kmapsolver3d_32bit_performance.csv")
INTERMEDIATE_CSV = os.path.join(OUTPUTS_DIR, "kmapsolver3d_32bit_terms.csv")
REPORT_PDF = os.path.join(OUTPUTS_DIR, "kmapsolver3d_32bit_performance_report.pdf")
LOGO_PATH = os.path.join(SCRIPT_DIR, "..", "..", "images", "St_logo_light-tp.png")

# Configuration for 32-bit chunked processing
NUM_VARS = 32
NUM_KMAP_VARS = 16  # Variables used for K-map gray code labeling (16-variable maps)
NUM_EXTRA_VARS = NUM_VARS - NUM_KMAP_VARS  # 16 extra variables per chunk
CHUNK_SIZE = 2**NUM_KMAP_VARS  # 65,536 truth table entries per K-map
TOTAL_CHUNKS = 2**NUM_EXTRA_VARS  # 65,536 chunks to process

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
# CHUNKED PROCESSING FUNCTIONS
# ============================================================================

def generate_chunk_output(chunk_index, seed=None):
    """
    Generate outputs for a single chunk of the 32-bit truth table.
    
    A chunk represents all 2^16 = 65,536 combinations where:
    - The first 16 variables (bits 0-15) are fixed to chunk_index
    - The last 16 variables (bits 16-31) vary through all combinations
    
    This matches KMapSolver3D's structure where first variables are extra_vars.
    
    Returns:
        List of 65,536 output values (0, 1, or 'd') for this chunk
    """
    if seed is not None:
        random.seed(seed + chunk_index)
    
    # Dense distribution: ~70% ones, ~5% don't-cares, ~25% zeros
    outputs = []
    for _ in range(CHUNK_SIZE):
        r = random.random()
        if r < 0.70:
            outputs.append(1)
        elif r < 0.75:
            outputs.append('d')
        else:
            outputs.append(0)
    
    return outputs


def process_single_chunk(chunk_index, csv_writer, seed=None, verbose=False):
    """
    Process a single chunk: generate outputs, minimize with 16-variable K-map, write bitmasks to CSV.
    
    Args:
        chunk_index: Index of the current chunk (0 to 2^16 - 1 = 65,535)
        csv_writer: CSV writer for intermediate bitmask storage
        seed: Random seed for reproducibility
        verbose: If True, print detailed feedback about sub-maps being solved
    
    Returns:
        Dictionary with chunk statistics
    """
    # Print chunk being processed
    if verbose:
        print(f"\n{'-'*70}")
        print(f"CHUNK {chunk_index:,} / {TOTAL_CHUNKS:,} (bits 0-15 = {format(chunk_index, '016b')})")
        print(f"{'-'*70}")
    
    # Generate outputs for this chunk
    chunk_outputs = generate_chunk_output(chunk_index, seed=seed)
    
    # Check if chunk is constant (all 0, all 1, or all don't-care)
    defined_vals = [v for v in chunk_outputs if v != 'd']
    if not defined_vals:
        # All don't-care - skip minimization
        if verbose:
            print("  [x] All don't-care - skipping minimization")
        del chunk_outputs
        return {"chunk": chunk_index, "type": "all_dc", "terms": 0, "time": 0}
    
    if all(v == 0 for v in defined_vals):
        # All zeros - no terms needed
        if verbose:
            print("  [x] All zeros - no terms")
        del chunk_outputs
        return {"chunk": chunk_index, "type": "all_zeros", "terms": 0, "time": 0}
    
    if all(v == 1 for v in defined_vals):
        # All ones - single bitmask covering entire chunk
        # For 16 variables, all bits set = 0xFFFF
        bitmask = (chunk_index << NUM_KMAP_VARS) | 0xFFFF
        csv_writer.writerow([chunk_index, hex(bitmask), "all_ones"])
        if verbose:
            print("  [+] All ones - constant function")
        del chunk_outputs
        return {"chunk": chunk_index, "type": "all_ones", "terms": 1, "time": 0}
    
    # Minimize using KMapSolver3D
    if verbose:
        print(f"  → Processing 16-variable K-map ({CHUNK_SIZE:,} entries)...")
        print(f"     Solving {solver.num_maps if 'solver' in locals() else 2**(NUM_KMAP_VARS-4)} sub-maps (4×4 each)...")
    
    start_time = time.perf_counter()
    
    # Suppress verbose output
    import io
    import sys as sys_module
    old_stdout = sys_module.stdout
    sys_module.stdout = io.StringIO()
    
    solver = KMapSolver3D(NUM_KMAP_VARS, chunk_outputs)
    terms, expr_str = solver.minimize_3d(form='sop')
    
    sys_module.stdout = old_stdout
    
    elapsed = time.perf_counter() - start_time
    
    # Access internal bitmask data from KMapSolver3D
    # The solver stores all_map_results which contain bitmasks
    # We need to extract these directly
    term_count = 0
    
    # Get ESSENTIAL prime implicant bitmasks from all solved K-maps within this chunk
    # KMapSolver3D._solve_single_kmap() returns only essential prime implicants
    # Each essential prime implicant is a bitmask covering multiple minterms in the 16-var space
    # Write directly to CSV without storing in RAM
    
    total_submaps = len(solver.kmaps)
    solved_submaps = 0
    
    for extra_combo in sorted(solver.kmaps.keys()):
        solved_submaps += 1
        
        if verbose:
            extra_combo_int = int(extra_combo, 2) if extra_combo else 0
            print(f"     Sub-map {solved_submaps}/{total_submaps}: bits 16-27 = {extra_combo} (decimal {extra_combo_int})", end="")
        
        # _solve_single_kmap returns ESSENTIAL prime implicants (not all primes)
        result = solver._solve_single_kmap(extra_combo, form='sop')
        extra_combo_int = int(extra_combo, 2) if extra_combo else 0
        
        # Each bitmask from the 4x4 map represents an ESSENTIAL prime implicant
        # Write directly to CSV without storing in RAM
        for idx, (bitmask_4x4, term_bits) in enumerate(zip(result['bitmasks'], result['terms_bits'])):
            # term_bits is like "01-1" showing which of the 4 variables are fixed
            # extra_combo shows which of the 12 extra variables are fixed
            # Together they form a 16-variable ESSENTIAL prime implicant pattern
            
            # Write to CSV with structured representation
            # Format: chunk_index | extra_combo | bitmask_4x4 | pattern | type
            # Type = "essential_prime" indicates this is an essential prime implicant
            csv_writer.writerow([
                chunk_index, 
                extra_combo,
                hex(bitmask_4x4),
                term_bits,
                "essential_prime"
            ])
            term_count += 1
        
        if verbose:
            print(f" → {len(result['bitmasks'])} essential prime implicants covering {sum(bin(bm).count('1') for bm in result['bitmasks'])} minterms")
    
    if verbose:
        print(f"  [+] Chunk complete: {term_count} essential prime implicants in {elapsed:.3f}s")
    
    # Explicitly clean up chunk data to free memory
    del chunk_outputs
    del solver
    del terms
    del expr_str
    
    return {
        "chunk": chunk_index,
        "type": "minimized",
        "terms": term_count,
        "time": elapsed,
        "literals": 0  # Not tracking literals to save memory
    }


def perform_bitwise_union(csv_path):
    """
    Read all bitmasks from CSV and perform bitwise union to extract final terms.
    
    Args:
        csv_path: Path to intermediate bitmask CSV file
    
    Returns:
        List of final minimized terms
    """
    print(f"\n   • Reading bitmasks from CSV...")
    bitmasks = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        
        for row in reader:
            chunk_idx, bitmask_hex, term_type = row
            if term_type != "all_dc" and term_type != "all_zeros":
                # Convert hex string to integer
                bitmasks.append(int(bitmask_hex, 16))
    
    print(f"      Found {len(bitmasks)} bitmasks")
    
    # Perform bitwise union to merge overlapping terms
    print(f"   • Performing bitwise union...")
    
    if not bitmasks:
        return {"total_bitmasks": 0, "unique_bitmasks": 0, "final_terms": 0}
    
    # Remove duplicates
    unique_bitmasks = sorted(set(bitmasks))
    print(f"      Unique bitmasks: {len(unique_bitmasks)}")
    
    # Simple union: combine bitmasks that differ by only one bit
    final_terms = []
    processed = set()
    
    for mask in unique_bitmasks:
        if mask in processed:
            continue
        
        # Try to merge with other masks
        merged = mask
        processed.add(mask)
        
        for other_mask in unique_bitmasks:
            if other_mask in processed:
                continue
            # Check if they differ by exactly one bit
            diff = merged ^ other_mask
            if bin(diff).count('1') == 1:
                merged |= other_mask
                processed.add(other_mask)
        
        final_terms.append(merged)
    
    print(f"      Final term count after union: {len(final_terms)}")
    
    return {
        "total_bitmasks": len(bitmasks),
        "unique_bitmasks": len(unique_bitmasks),
        "final_terms": len(final_terms)
    }


# ============================================================================
# TEST EXECUTION
# ============================================================================

def run_chunked_32bit_test(seed=None, progress_interval=10000):
    """
    Run the complete 32-bit test using chunked processing.
    
    Process all 2^16 = 65,536 chunks sequentially:
    - Generate outputs for each chunk (65,536 entries per chunk)
    - Minimize using KMapSolver3D with 16 variables
    - Write intermediate terms to CSV
    - Consolidate terms across all chunks
    
    Args:
        seed: Random seed for reproducibility
        progress_interval: Print progress every N chunks (default 10,000)
    
    Returns:
        Dictionary with test results
    """
    print(f"\n{'='*80}")
    print("CHUNKED 32-BIT PROCESSING")
    print(f"{'='*80}")
    print(f"   Total chunks to process: {format_large_number(TOTAL_CHUNKS)}")
    print(f"   Entries per chunk: {CHUNK_SIZE}")
    print(f"   Total truth table entries: {format_large_number(2**NUM_VARS)}")
    print(f"   Progress updates every: {format_large_number(progress_interval)} chunks")
    
    # Initialize intermediate CSV
    print(f"\n   • Initializing intermediate bitmask CSV...")
    with open(INTERMEDIATE_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["chunk_index", "bitmask", "type"])
    
    print(f"      {INTERMEDIATE_CSV}")
    
    # Statistics tracking
    stats = {
        "total_chunks": TOTAL_CHUNKS,
        "processed_chunks": 0,
        "all_dc_chunks": 0,
        "all_zeros_chunks": 0,
        "all_ones_chunks": 0,
        "minimized_chunks": 0,
        "total_terms": 0,
        "total_time": 0,
        "chunk_times": []
    }
    
    start_time = time.perf_counter()
    
    print(f"\n   • Processing chunks...")
    print(f"      Verbose mode: Showing detailed feedback for first 3 chunks\n")
    
    # Process chunks in batches to manage CSV writing
    with open(INTERMEDIATE_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        for chunk_idx in range(TOTAL_CHUNKS):
            # Enable verbose output for first 3 chunks only
            verbose = (chunk_idx < 3)
            
            # Process single chunk
            chunk_result = process_single_chunk(chunk_idx, writer, seed=seed, verbose=True)
            
            # Force garbage collection after each chunk to free memory immediately
            import gc
            gc.collect()
            
            # Update statistics
            stats["processed_chunks"] += 1
            stats["total_terms"] += chunk_result.get("terms", 0)
            
            chunk_type = chunk_result["type"]
            if chunk_type == "all_dc":
                stats["all_dc_chunks"] += 1
            elif chunk_type == "all_zeros":
                stats["all_zeros_chunks"] += 1
            elif chunk_type == "all_ones":
                stats["all_ones_chunks"] += 1
            elif chunk_type == "minimized":
                stats["minimized_chunks"] += 1
                stats["total_time"] += chunk_result.get("time", 0)
                stats["chunk_times"].append(chunk_result.get("time", 0))
            
            # Progress update (standard mode)
            if (chunk_idx + 1) % progress_interval == 0:
                elapsed = time.perf_counter() - start_time
                progress_pct = 100 * (chunk_idx + 1) / TOTAL_CHUNKS
                chunks_per_sec = (chunk_idx + 1) / elapsed
                eta = (TOTAL_CHUNKS - chunk_idx - 1) / chunks_per_sec if chunks_per_sec > 0 else 0
                
                print(f"\n      ═══ PROGRESS UPDATE ═══")
                print(f"      Chunks: {format_large_number(chunk_idx + 1)} / {format_large_number(TOTAL_CHUNKS)} ({progress_pct:.2f}%)")
                print(f"      Speed: {chunks_per_sec:.2f} chunks/s | Avg time/chunk: {elapsed/(chunk_idx+1):.3f}s")
                print(f"      Terms so far: {format_large_number(stats['total_terms'])}")
                print(f"      Minimized: {stats['minimized_chunks']:,} | Constant: {stats['all_ones_chunks']+stats['all_zeros_chunks']:,} | DC: {stats['all_dc_chunks']:,}")
                print(f"      ETA: {format_time(eta)}\n")
    
    total_elapsed = time.perf_counter() - start_time
    stats["total_elapsed"] = total_elapsed
    
    print(f"\n   [+] All chunks processed in {format_time(total_elapsed)}")
    print(f"      Average: {stats['processed_chunks']/total_elapsed:.1f} chunks/second")
    
    # Perform bitwise union
    print(f"\n   • Performing final bitwise union...")
    union_result = perform_bitwise_union(INTERMEDIATE_CSV)
    stats["total_bitmasks"] = union_result["total_bitmasks"]
    stats["unique_bitmasks"] = union_result["unique_bitmasks"]
    stats["final_terms"] = union_result["final_terms"]
    
    return stats


# ============================================================================
# RESULTS EXPORT
# ============================================================================

def export_results_to_csv(stats, csv_path):
    """Export test statistics to CSV file."""
    print(f"\n   • Exporting statistics to CSV...", end=" ", flush=True)
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            "Metric", "Value"
        ])
        
        # Data rows
        writer.writerow(["Variables", NUM_VARS])
        writer.writerow(["Bit_Size", "32-bit"])
        writer.writerow(["Total_Chunks", stats["total_chunks"]])
        writer.writerow(["Processed_Chunks", stats["processed_chunks"]])
        writer.writerow(["All_DC_Chunks", stats["all_dc_chunks"]])
        writer.writerow(["All_Zeros_Chunks", stats["all_zeros_chunks"]])
        writer.writerow(["All_Ones_Chunks", stats["all_ones_chunks"]])
        writer.writerow(["Minimized_Chunks", stats["minimized_chunks"]])
        writer.writerow(["Total_Intermediate_Terms", stats["total_terms"]])
        writer.writerow(["Total_Bitmasks", stats.get("total_bitmasks", 0)])
        writer.writerow(["Unique_Bitmasks", stats.get("unique_bitmasks", 0)])
        writer.writerow(["Final_Terms_After_Union", stats["final_terms"]])
        writer.writerow(["Total_Elapsed_Time_s", f"{stats['total_elapsed']:.6f}"])
        writer.writerow(["Total_Minimization_Time_s", f"{stats['total_time']:.6f}"])
        
        if stats["chunk_times"]:
            writer.writerow(["Avg_Chunk_Time_s", f"{np.mean(stats['chunk_times']):.6f}"])
            writer.writerow(["Min_Chunk_Time_s", f"{np.min(stats['chunk_times']):.6f}"])
            writer.writerow(["Max_Chunk_Time_s", f"{np.max(stats['chunk_times']):.6f}"])
    
    print("[+] Done")
    print(f"      {csv_path}")


def save_performance_report(stats, pdf_path):
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
        
        ax.text(0.5, 0.55, "KMapSolver3D 32-bit Test Report", 
                fontsize=26, fontweight='bold', ha='center')
        ax.text(0.5, 0.49, "Chunked Processing with CSV Storage",
                fontsize=16, ha='center')
        
        ax.plot([0.2, 0.8], [0.45, 0.45], 'k-', linewidth=2, alpha=0.3)
        
        ax.text(0.5, 0.38, f"32-bit: 2³² = {format_large_number(2**NUM_VARS)} entries",
                fontsize=11, ha='center', family='monospace')
        ax.text(0.5, 0.34, f"Chunks: 2²⁸ = {format_large_number(TOTAL_CHUNKS)} chunks",
                fontsize=11, ha='center', family='monospace')
        ax.text(0.5, 0.30, f"Chunk size: 2⁴ = {CHUNK_SIZE} entries per chunk",
                fontsize=11, ha='center', family='monospace')
        
        ax.text(0.5, 0.22, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d')}",
                fontsize=11, ha='center')
        ax.text(0.5, 0.18, f"Random Seed: {RANDOM_SEED}",
                fontsize=11, ha='center')
        ax.text(0.5, 0.14, f"Processing Time: {format_time(stats['total_elapsed'])}",
                fontsize=11, ha='center')
        
        ax.text(0.5, 0.08, "Step-by-step truth table generation with efficient stack management",
                fontsize=9, ha='center', style='italic', color='gray')
        ax.text(0.5, 0.04, f"© Stan's Technologies {datetime.datetime.now().year}",
                fontsize=9, ha='center', color='gray')
        
        pdf.savefig(bbox_inches='tight')
        plt.close()
        print("[+] Done")
        
        # Chunk Processing Statistics
        print(f"      • Creating chunk processing statistics...", end=" ", flush=True)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('32-bit Chunked Processing Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Chunk Type Distribution
        chunk_types = ['All DC', 'All Zeros', 'All Ones', 'Minimized']
        chunk_counts = [
            stats['all_dc_chunks'],
            stats['all_zeros_chunks'],
            stats['all_ones_chunks'],
            stats['minimized_chunks']
        ]
        colors = ['#d62728', '#8c564b', '#2ca02c', '#1f77b4']
        
        wedges, texts, autotexts = ax1.pie(chunk_counts, labels=chunk_types, colors=colors,
                                             autopct='%1.1f%%', startangle=90)
        ax1.set_title('Chunk Type Distribution')
        
        # Plot 2: Processing Time Breakdown
        time_data = [stats['total_time'], stats['total_elapsed'] - stats['total_time']]
        time_labels = ['Minimization', 'Overhead']
        ax2.pie(time_data, labels=time_labels, colors=['#1f77b4', '#ff7f0e'],
                autopct='%1.1f%%', startangle=90)
        ax2.set_title('Time Distribution')
        
        # Plot 3: Chunk Processing Time (if available)
        if stats['chunk_times']:
            ax3.hist(stats['chunk_times'], bins=50, color='#1f77b4', alpha=0.7, edgecolor='black')
            ax3.set_xlabel('Chunk Processing Time (s)')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Minimization Time Distribution')
            ax3.grid(True, alpha=0.3)
            ax3.axvline(np.mean(stats['chunk_times']), color='red', linestyle='--',
                       label=f"Mean: {np.mean(stats['chunk_times']):.6f}s")
            ax3.legend()
        
        # Plot 4: Term Count Comparison
        term_labels = ['Intermediate\nTerms', 'Final Terms\n(After Union)']
        term_counts = [stats['total_terms'], stats['final_terms']]
        bars = ax4.bar(term_labels, term_counts, color=['#ff7f0e', '#2ca02c'], alpha=0.7)
        ax4.set_ylabel('Term Count')
        ax4.set_title('Term Count: Intermediate vs Final')
        ax4.grid(True, alpha=0.3, axis='y')
        
        for bar, count in zip(bars, term_counts):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{format_large_number(count)}',
                    ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()
        print("[+] Done")
        
        # Detailed Statistics Page
        print(f"      • Creating detailed statistics page...", end=" ", flush=True)
        
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        y_pos = 0.95
        ax.text(0.5, y_pos, '32-bit Performance Statistics', 
                fontsize=16, fontweight='bold', ha='center')
        y_pos -= 0.08
        
        stats_text = f"""
CHUNKED PROCESSING SUMMARY
{'='*70}

Configuration:
  • Variables: {NUM_VARS} (32-bit architecture)
  • K-map Variables: {NUM_KMAP_VARS}
  • Extra Variables per Chunk: {NUM_EXTRA_VARS}
  • Chunk Size: {CHUNK_SIZE} entries (2^{NUM_KMAP_VARS})
  • Total Chunks: {format_large_number(TOTAL_CHUNKS)} (2^{NUM_EXTRA_VARS})
  • Truth Table Size: {format_large_number(2**NUM_VARS)} entries (2^{NUM_VARS})

Chunk Processing Results:
  • Processed Chunks: {format_large_number(stats['processed_chunks'])}
  • All Don't-Care: {format_large_number(stats['all_dc_chunks'])} ({100*stats['all_dc_chunks']/stats['total_chunks']:.2f}%)
  • All Zeros: {format_large_number(stats['all_zeros_chunks'])} ({100*stats['all_zeros_chunks']/stats['total_chunks']:.2f}%)
  • All Ones: {format_large_number(stats['all_ones_chunks'])} ({100*stats['all_ones_chunks']/stats['total_chunks']:.2f}%)
  • Minimized: {format_large_number(stats['minimized_chunks'])} ({100*stats['minimized_chunks']/stats['total_chunks']:.2f}%)

Time Statistics:
  • Total Elapsed Time: {format_time(stats['total_elapsed'])}
  • Minimization Time: {format_time(stats['total_time'])}
  • Overhead Time: {format_time(stats['total_elapsed'] - stats['total_time'])}
  • Average Chunks/Second: {stats['processed_chunks']/stats['total_elapsed']:.2f}
"""

        if stats['chunk_times']:
            stats_text += f"""
  • Avg Chunk Time: {np.mean(stats['chunk_times']):.6f}s
  • Min Chunk Time: {np.min(stats['chunk_times']):.6f}s
  • Max Chunk Time: {np.max(stats['chunk_times']):.6f}s
  • Std Dev: {np.std(stats['chunk_times']):.6f}s
"""

        stats_text += f"""

Term Statistics:
  • Total Intermediate Terms: {format_large_number(stats['total_terms'])}
  • Final Terms (After Union): {format_large_number(stats['final_terms'])}
  • Reduction: {100*(1-stats['final_terms']/stats['total_terms']) if stats['total_terms'] > 0 else 0:.2f}%

Algorithm Details:
  • Step-by-step truth table generation
  • Chunk-wise K-map minimization
  • CSV-based intermediate storage
  • Bitwise union for final term extraction
  • Efficient stack management (completed chunks discarded)

Random Seed: {RANDOM_SEED}
Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        ax.text(0.05, y_pos, stats_text, fontsize=9, va='top', family='monospace')
        
        pdf.savefig(bbox_inches='tight')
        plt.close()
        print("[+] Done")
        
        # Set PDF metadata
        d = pdf.infodict()
        d['Title'] = 'KMapSolver3D 32-bit Chunked Processing Report'
        d['Author'] = "Stan's Technologies"
        d['Subject'] = '32-bit Boolean Minimization with Chunked Processing'
        d['Keywords'] = '32-bit, Boolean minimization, K-map, chunked processing, CSV storage'
        d['CreationDate'] = datetime.datetime.now()
    
    print(f"      [+] PDF report saved")
    print(f"      {pdf_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main test execution function."""
    print("=" * 80)
    print("KMapSolver3D 32-BIT CHUNKED PROCESSING TEST")
    print("=" * 80)
    print("\n[*] Test Configuration:")
    print(f"    Architecture: 32-bit")
    print(f"    Truth Table Size: {format_large_number(2**NUM_VARS)} entries")
    print(f"    Processing Strategy: Chunked (16-variable K-maps per chunk)")
    print(f"    Chunk Count: {format_large_number(TOTAL_CHUNKS)}")
    print(f"    Chunk Size: {format_large_number(CHUNK_SIZE)} entries per chunk")
    print(f"    K-map Variables: {NUM_KMAP_VARS}")
    print(f"    Extra Variables (chunk selectors): {NUM_EXTRA_VARS}")
    
    print("\n[!] NOTE: This test processes the truth table in chunks to manage memory efficiently.")
    print("    Each chunk is minimized, results written to CSV, then discarded.")
    print("    Final terms extracted via bitwise union of all chunk results.")
    print("    Execution may take considerable time due to the large number of chunks.")
    
    # Set random seed
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    print(f"\n[#] Random seed: {RANDOM_SEED}")
    
    # Create outputs directory
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    print(f"[>] Output directory: {OUTPUTS_DIR}")
    
    # Run chunked processing test
    print(f"\n[~] Starting chunked 32-bit test...")
    
    try:
        stats = run_chunked_32bit_test(seed=RANDOM_SEED, progress_interval=10000)
    except Exception as e:
        print(f"\n[ERROR] ERROR during chunked processing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Export results
    print(f"\n{'='*80}")
    print("EXPORTING RESULTS")
    print(f"{'='*80}")
    
    export_results_to_csv(stats, RESULTS_CSV)
    save_performance_report(stats, REPORT_PDF)
    
    # Final summary
    print(f"\n{'='*80}")
    print("[SUCCESS] 32-BIT CHUNKED PROCESSING TEST COMPLETE")
    print(f"{'='*80}")
    print(f"[*] Processing Statistics:")
    print(f"   Total chunks: {format_large_number(stats['total_chunks'])}")
    print(f"   Processed: {format_large_number(stats['processed_chunks'])} (100%)")
    print(f"   Minimized chunks: {format_large_number(stats['minimized_chunks'])} ({100*stats['minimized_chunks']/stats['total_chunks']:.2f}%)")
    print(f"   Constant chunks: {format_large_number(stats['all_dc_chunks'] + stats['all_zeros_chunks'] + stats['all_ones_chunks'])} ({100*(stats['all_dc_chunks'] + stats['all_zeros_chunks'] + stats['all_ones_chunks'])/stats['total_chunks']:.2f}%)")
    
    print(f"\n[*] Time Statistics:")
    print(f"   Total elapsed: {format_time(stats['total_elapsed'])}")
    print(f"   Minimization time: {format_time(stats['total_time'])}")
    print(f"   Overhead: {format_time(stats['total_elapsed'] - stats['total_time'])}")
    print(f"   Average: {stats['processed_chunks']/stats['total_elapsed']:.2f} chunks/second")
    
    print(f"\n[*] Term Statistics:")
    print(f"   Intermediate terms: {format_large_number(stats['total_terms'])}")
    print(f"   Final terms: {format_large_number(stats['final_terms'])}")
    if stats['total_terms'] > 0:
        print(f"   Reduction: {100*(1-stats['final_terms']/stats['total_terms']):.2f}%")
    
    print(f"\n[*] Output Files:")
    print(f"   • Statistics CSV:    {RESULTS_CSV}")
    print(f"   • Intermediate CSV:  {INTERMEDIATE_CSV}")
    print(f"   • PDF Report:        {REPORT_PDF}")
    print(f"\n{'='*80}")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

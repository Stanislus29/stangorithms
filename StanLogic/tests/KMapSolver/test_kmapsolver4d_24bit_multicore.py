"""
24-Bit Boolean Minimization with Multicore Processing
======================================================

Tests KMapSolver4D with 24-bit Boolean functions using parallel processing
across 4 CPU cores.

24-bit configuration:
- Total variables: 24
- K-map variables: 16 (per chunk)
- Chunk selector variables: 8
- Total chunks: 256
- Chunks per core: 64

Estimated time (at 76s/chunk):
- Sequential: 256 × 76s = 5.4 hours
- 4 cores: 64 × 76s = 1.3 hours

Author: Stan's Technologies
Date: December 2025
"""

import os
import sys
import csv
import time
import random
import multiprocessing as mp
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from stanlogic.kmapsolver4D import KMapSolver4D, merge_csv_files

# ============================================================================
# CONFIGURATION
# ============================================================================

NUM_VARS = 24
NUM_CORES = 4
RANDOM_SEED = 42

# Output files
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# CSV files for each core
CSV_PATHS = [
    os.path.join(OUTPUTS_DIR, f"kmapsolver4d_24bit_core{i}.csv")
    for i in range(NUM_CORES)
]
MERGED_CSV = os.path.join(OUTPUTS_DIR, "kmapsolver4d_24bit_merged.csv")
RESULTS_CSV = os.path.join(OUTPUTS_DIR, "kmapsolver4d_24bit_results.csv")


# ============================================================================
# WORKER FUNCTION
# ============================================================================

def worker_process(core_id, chunk_start, chunk_end, csv_path, seed):
    """
    Worker function for parallel chunk processing.
    Runs in a separate terminal window.
    
    Args:
        core_id: Core identifier (0-3)
        chunk_start: Starting chunk index
        chunk_end: Ending chunk index (exclusive)
        csv_path: CSV output path for this core
        seed: Random seed
    """
    try:
        # Set terminal title
        if sys.platform == 'win32':
            os.system(f'title Core {core_id} - Processing chunks {chunk_start}-{chunk_end-1}')
        
        print(f"\n{'='*70}")
        print(f"CORE {core_id} - PROCESSING STARTED")
        print(f"{'='*70}")
        print(f"Chunk range: {chunk_start:,} to {chunk_end-1:,} ({chunk_end - chunk_start:,} chunks)")
        print(f"Output CSV: {os.path.basename(csv_path)}")
        print(f"{'='*70}\n")
        
        # Create solver for this chunk range
        solver = KMapSolver4D(
            num_vars=NUM_VARS,
            csv_path=csv_path,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            seed=seed,
            verbose=True  # All cores are verbose in their own windows
        )
        
        # Process chunks with progress updates every 10 chunks
        stats = solver.minimize(progress_interval=10)
        
        # Write statistics to a temp file for the main process to read
        stats_file = csv_path.replace('.csv', '_stats.txt')
        with open(stats_file, 'w') as f:
            f.write(f"processed_chunks:{stats['processed_chunks']}\n")
            f.write(f"total_terms:{stats['total_terms']}\n")
            f.write(f"minimized_chunks:{stats['minimized_chunks']}\n")
            f.write(f"total_time:{stats['total_time']}\n")
            f.write(f"total_elapsed:{stats['total_elapsed']}\n")
        
        print(f"\n{'='*70}")
        print(f"CORE {core_id} - COMPLETED")
        print(f"{'='*70}")
        print(f"Processed {stats['processed_chunks']:,} chunks in {stats['total_elapsed']:.1f}s")
        print(f"Generated {stats['total_terms']:,} terms")
        print(f"{'='*70}\n")
        print("Press any key to close this window...")
        input()
        
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"CORE {core_id} - ERROR")
        print(f"{'='*70}")
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        # Write error to stats file
        stats_file = csv_path.replace('.csv', '_stats.txt')
        with open(stats_file, 'w') as f:
            f.write(f"error:{e}\n")
        
        print("\nPress any key to close this window...")
        input()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 80)
    print("24-BIT BOOLEAN MINIMIZATION WITH MULTICORE PROCESSING")
    print("=" * 80)
    
    # Calculate chunk distribution
    num_kmap_vars = 16
    num_chunk_vars = NUM_VARS - num_kmap_vars
    total_chunks = 2**num_chunk_vars
    chunks_per_core = total_chunks // NUM_CORES
    
    print(f"\n[*] Configuration:")
    print(f"    Variables: {NUM_VARS}")
    print(f"    Total chunks: {total_chunks:,}")
    print(f"    CPU cores: {NUM_CORES}")
    print(f"    Chunks per core: {chunks_per_core:,}")
    print(f"    Random seed: {RANDOM_SEED}")
    
    # Calculate chunk ranges for each core
    chunk_ranges = []
    for i in range(NUM_CORES):
        start = i * chunks_per_core
        end = (i + 1) * chunks_per_core if i < NUM_CORES - 1 else total_chunks
        chunk_ranges.append((start, end))
    
    print(f"\n[*] Chunk Distribution:")
    for i, (start, end) in enumerate(chunk_ranges):
        print(f"    Core {i}: chunks {start:,} to {end-1:,} ({end-start:,} chunks)")
    
    # Estimate time
    avg_time_per_chunk = 76  # seconds (based on observation)
    estimated_time = chunks_per_core * avg_time_per_chunk
    print(f"\n[*] Estimated Time:")
    print(f"    Per chunk: ~{avg_time_per_chunk}s")
    print(f"    Per core: ~{estimated_time/3600:.1f} hours")
    print(f"    Total wall-clock: ~{estimated_time/3600:.1f} hours (parallel)")
    
    input(f"\nPress ENTER to start processing with {NUM_CORES} cores in separate windows...")
    
    # Start parallel processing
    print(f"\n{'='*80}")
    print("STARTING PARALLEL PROCESSING IN SEPARATE TERMINALS")
    print(f"{'='*80}")
    print(f"   Opening {NUM_CORES} terminal windows...")
    
    start_time = time.perf_counter()
    
    # Launch each core in a separate terminal window
    import subprocess
    processes = []
    
    # Get the directory containing the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), 'src')
    
    for i in range(NUM_CORES):
        start, end = chunk_ranges[i]
        
        # Create a Python command that will run in the new terminal
        python_code = f"""
import sys
import os
sys.path.insert(0, r'{src_dir}')
os.chdir(r'{script_dir}')
from stanlogic.kmapsolver4D import KMapSolver4D
import random
import time

# Worker function inline
def run_worker():
    core_id = {i}
    chunk_start = {start}
    chunk_end = {end}
    csv_path = r'{CSV_PATHS[i]}'
    seed = {RANDOM_SEED}
    num_vars = {NUM_VARS}
    
    try:
        os.system('title Core {{}} - Processing chunks {{}}-{{}}'.format(core_id, chunk_start, chunk_end-1))
        
        print('\\n' + '='*70)
        print('CORE {{}} - PROCESSING STARTED'.format(core_id))
        print('='*70)
        print('Chunk range: {{:,}} to {{:,}} ({{:,}} chunks)'.format(chunk_start, chunk_end-1, chunk_end - chunk_start))
        print('Output CSV: {{}}'.format(os.path.basename(csv_path)))
        print('='*70 + '\\n')
        
        solver = KMapSolver4D(
            num_vars=num_vars,
            csv_path=csv_path,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            seed=seed,
            verbose=True
        )
        
        stats = solver.minimize(progress_interval=10)
        
        stats_file = csv_path.replace('.csv', '_stats.txt')
        with open(stats_file, 'w') as f:
            f.write('processed_chunks:{{}}\\n'.format(stats['processed_chunks']))
            f.write('total_terms:{{}}\\n'.format(stats['total_terms']))
            f.write('minimized_chunks:{{}}\\n'.format(stats['minimized_chunks']))
            f.write('total_time:{{}}\\n'.format(stats['total_time']))
            f.write('total_elapsed:{{}}\\n'.format(stats['total_elapsed']))
        
        print('\\n' + '='*70)
        print('CORE {{}} - COMPLETED'.format(core_id))
        print('='*70)
        print('Processed {{:,}} chunks in {{:.1f}}s'.format(stats['processed_chunks'], stats['total_elapsed']))
        print('Generated {{:,}} terms'.format(stats['total_terms']))
        print('='*70 + '\\n')
        print('Press any key to close this window...')
        input()
        
    except Exception as e:
        print('\\n' + '='*70)
        print('CORE {{}} - ERROR'.format(core_id))
        print('='*70)
        print('ERROR: {{}}'.format(e))
        import traceback
        traceback.print_exc()
        
        stats_file = csv_path.replace('.csv', '_stats.txt')
        with open(stats_file, 'w') as f:
            f.write('error:{{}}\\n'.format(e))
        
        print('\\nPress any key to close this window...')
        input()

run_worker()
"""
        
        # Launch in new PowerShell window (Windows)
        if sys.platform == 'win32':
            # Write the Python code to a temp file
            temp_script = os.path.join(OUTPUTS_DIR, f'worker_core{i}.py')
            with open(temp_script, 'w') as f:
                f.write(python_code)
            
            # Launch PowerShell with the script
            cmd = f'Start-Process powershell -ArgumentList "-NoExit", "-Command", "python {temp_script}"'
            p = subprocess.Popen(['powershell', '-Command', cmd], shell=True)
        else:
            # For Linux/Mac, use xterm or gnome-terminal
            p = subprocess.Popen(['xterm', '-hold', '-e', f'python3 -c "{python_code}"'])
        
        processes.append(p)
        print(f"   [+] Core {i}: Terminal window opened (chunks {start}-{end-1})")
        time.sleep(0.5)  # Small delay between window launches
    
    print(f"\n   All {NUM_CORES} terminal windows launched!")
    print(f"   Waiting for all cores to complete...")
    print(f"   Monitor progress in each terminal window.")
    
    # Wait for all processes by checking for stats files
    completed = [False] * NUM_CORES
    while not all(completed):
        for i in range(NUM_CORES):
            if not completed[i]:
                stats_file = CSV_PATHS[i].replace('.csv', '_stats.txt')
                if os.path.exists(stats_file):
                    completed[i] = True
                    print(f"   [+] Core {i} completed!")
        time.sleep(2)  # Check every 2 seconds
    
    total_elapsed = time.perf_counter() - start_time
    
    # Collect statistics from files
    print(f"\n{'='*80}")
    print("PARALLEL PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"   Total wall-clock time: {total_elapsed/3600:.2f} hours ({total_elapsed/60:.1f} minutes)")
    
    # Aggregate statistics from stats files
    total_terms = 0
    total_minimized = 0
    total_processing_time = 0
    all_errors = []
    
    for i in range(NUM_CORES):
        stats_file = CSV_PATHS[i].replace('.csv', '_stats.txt')
        if os.path.exists(stats_file):
            stats = {}
            with open(stats_file, 'r') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.strip().split(':', 1)
                        if key == 'error':
                            stats['error'] = value
                        else:
                            stats[key] = float(value) if '.' in value else int(value)
            
            if "error" in stats:
                all_errors.append(f"Core {i}: {stats['error']}")
            else:
                total_terms += stats.get("total_terms", 0)
                total_minimized += stats.get("minimized_chunks", 0)
                total_processing_time += stats.get("total_time", 0)
    
    if all_errors:
        print(f"\n[!] ERRORS OCCURRED:")
        for err in all_errors:
            print(f"    {err}")
    
    print(f"\n[*] Aggregate Statistics:")
    print(f"    Total terms generated: {total_terms:,}")
    print(f"    Minimized chunks: {total_minimized:,} / {total_chunks:,}")
    print(f"    Total CPU time: {total_processing_time/3600:.2f} hours")
    print(f"    Parallel efficiency: {(total_processing_time/total_elapsed):.1f}x")
    
    print(f"\n[*] You can now close the worker terminal windows (they're waiting for a keypress)")
    
    # Merge CSV files
    print(f"\n{'='*80}")
    print("MERGING RESULTS")
    print(f"{'='*80}")
    
    existing_csvs = [path for path in CSV_PATHS if os.path.exists(path)]
    if existing_csvs:
        merge_csv_files(existing_csvs, MERGED_CSV)
        print(f"   [+] Merged CSV saved: {MERGED_CSV}")
    else:
        print(f"   [!] No CSV files found to merge")
    
    # Save summary statistics
    print(f"\n   • Saving results summary...")
    with open(RESULTS_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["num_vars", NUM_VARS])
        writer.writerow(["num_cores", NUM_CORES])
        writer.writerow(["total_chunks", total_chunks])
        writer.writerow(["chunks_per_core", chunks_per_core])
        writer.writerow(["total_terms", total_terms])
        writer.writerow(["minimized_chunks", total_minimized])
        writer.writerow(["wall_clock_time_hours", f"{total_elapsed/3600:.2f}"])
        writer.writerow(["total_cpu_time_hours", f"{total_processing_time/3600:.2f}"])
        writer.writerow(["parallel_efficiency", f"{total_processing_time/total_elapsed:.2f}"])
        writer.writerow(["random_seed", RANDOM_SEED])
    
    print(f"   [+] Results saved: {RESULTS_CSV}")
    
    # Final summary
    print(f"\n{'='*80}")
    print("[SUCCESS] 24-BIT MULTICORE PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"\n[*] Output Files:")
    print(f"    Core CSVs: {len(existing_csvs)} files in {OUTPUTS_DIR}")
    print(f"    Merged CSV: {MERGED_CSV}")
    print(f"    Results: {RESULTS_CSV}")
    
    print(f"\n[*] Next Steps:")
    print(f"    1. Review {RESULTS_CSV} for statistics")
    print(f"    2. Use {MERGED_CSV} for bitwise union operation")
    print(f"    3. Extract final minimized terms")
    
    return True


if __name__ == "__main__":
    # Set start method for multiprocessing (required on Windows)
    mp.set_start_method('spawn', force=True)
    
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n[!] Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

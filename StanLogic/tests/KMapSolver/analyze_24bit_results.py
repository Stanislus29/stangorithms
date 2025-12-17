"""
24-Bit Boolean Function Analysis: Bitwise Union & Verilog Generation
======================================================================

This script processes the pre-computed chunk-based minimization results,
performs bitwise union across all chunks, extracts the final expression,
and generates text file with SOP expression and Verilog module.

Input:  kmapsolver4d_24bit_merged.csv (or individual core CSVs)
Outputs:
  - kmapsolver4d_24bit_final_expression.txt (SOP with statistics)
  - verilog/boolean_function_24bit.v (Verilog hardware module)

Author: Stan's Technologies
Date: December 2025
"""

import os
import sys
import csv
import datetime
import platform
import re
from pathlib import Path
from multiprocessing import Pool, cpu_count
import time
import subprocess
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from stanlogic import KMapSolver3D

# ============================================================================
# CONFIGURATION
# ============================================================================

NUM_VARS = 24
NUM_CORES = cpu_count()  # Use all available CPU cores
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, "outputs")

# Input CSV (merged or individual cores)
MERGED_CSV = os.path.join(OUTPUTS_DIR, "kmapsolver4d_24bit_merged.csv")
CORE_CSV_PATTERN = os.path.join(OUTPUTS_DIR, "kmapsolver4d_24bit_core*.csv")

# Output files
SOP_TEXT_FILE = os.path.join(OUTPUTS_DIR, "kmapsolver4d_24bit_final_expression.txt")
VERILOG_DIR = os.path.join(OUTPUTS_DIR, "verilog")
VERILOG_FILE = os.path.join(VERILOG_DIR, "boolean_function_24bit.v")

# ============================================================================
# BITWISE UNION & EXPRESSION EXTRACTION
# ============================================================================

def perform_bitwise_union(csv_path, num_vars=24):
    """
    Perform bitwise union across all chunk bitmasks from CSV.
    
    Args:
        csv_path: Path to CSV file with bitmask data
        num_vars: Total number of variables
        
    Returns:
        dict: Union results with statistics
    """
    print(f"\n{'='*80}")
    print("PERFORMING BITWISE UNION OPERATION")
    print(f"{'='*80}")
    print(f"Reading from: {os.path.basename(csv_path)}")
    
    # Dictionary to store union of bitmasks
    # Key: (chunk_index, extra_combo), Value: unioned bitmask
    union_bitmasks = {}
    chunk_count = 0
    row_count = 0
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_count += 1
                
                if row_count % 10000 == 0:
                    print(f"  Processing row {row_count:,}...", end='\r')
                
                chunk_idx = int(row['chunk_index'])
                chunk_count = max(chunk_count, chunk_idx + 1)
                
                # Get bitmask
                bitmask_str = row.get('bitmask_4x4', row.get('bitmask', '0'))
                if bitmask_str.startswith('0x') or bitmask_str.startswith('0X'):
                    bitmask = int(bitmask_str, 16)
                else:
                    bitmask = int(bitmask_str)
                
                # Get extra_combo
                extra_combo = row.get('extra_combo', '')
                
                # Create key
                key = (chunk_idx, extra_combo)
                
                # Perform bitwise OR (union)
                if key in union_bitmasks:
                    union_bitmasks[key] |= bitmask
                else:
                    union_bitmasks[key] = bitmask
        
        print(f"\n  [+] Processed {row_count:,} rows from CSV")
        print(f"  [+] Chunks found: {chunk_count:,}")
        print(f"  [+] Unique (chunk, submap) combinations: {len(union_bitmasks):,}")
        
        return {
            'union_bitmasks': union_bitmasks,
            'chunk_count': chunk_count,
            'row_count': row_count,
            'unique_combinations': len(union_bitmasks)
        }
        
    except Exception as e:
        print(f"\n  [!] Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_expression_from_union(union_result, num_vars=24, max_terms=None):
    """
    Convert union bitmasks to SOP expression.
    
    Args:
        union_result: Dictionary with union bitmasks
        num_vars: Total number of variables
        max_terms: Maximum terms to extract (None = all)
        
    Returns:
        tuple: (expression_string, term_count, literal_count)
    """
    print(f"\n{'='*80}")
    print("EXTRACTING FINAL SOP EXPRESSION")
    print(f"{'='*80}")
    
    union_bitmasks = union_result['union_bitmasks']
    chunk_count = union_result['chunk_count']
    
    all_minterms = []
    
    print(f"  Converting {len(union_bitmasks):,} union bitmasks to minterms...")
    
    # Process each (chunk, extra_combo) combination
    for idx, ((chunk_idx, extra_combo), bitmask) in enumerate(union_bitmasks.items()):
        if idx % 1000 == 0 and idx > 0:
            print(f"    Processed {idx:,}/{len(union_bitmasks):,} combinations...", end='\r')
        
        if bitmask == 0:
            continue
        
        # Convert bitmask to minterms
        for bit_pos in range(16):
            if bitmask & (1 << bit_pos):
                # Chunk selector bits (8 bits for 24-var)
                chunk_bits = format(chunk_idx, f'0{num_vars-16}b')
                
                # Extra combo bits (12 bits)
                extra_bits = extra_combo if extra_combo else '0' * 12
                if len(extra_bits) < 12:
                    extra_bits = extra_bits.zfill(12)
                
                # 4x4 K-map position (4 bits)
                pos_bits = format(bit_pos, '04b')
                
                # Full minterm
                full_bits = chunk_bits + extra_bits + pos_bits
                all_minterms.append(full_bits)
    
    print(f"\n  [+] Extracted {len(all_minterms):,} total minterms")
    
    # Remove duplicates and sort
    unique_minterms = sorted(set(all_minterms))
    print(f"  [+] Unique minterms: {len(unique_minterms):,}")
    
    # Convert to SOP expression
    print(f"  Converting minterms to SOP expression...")
    
    var_names = [f"x{i}" for i in range(num_vars)]
    expr_terms = []
    
    # Limit terms if specified
    terms_to_process = unique_minterms if max_terms is None else unique_minterms[:max_terms]
    
    for minterm_bits in terms_to_process:
        # Convert bit string to product term
        product = []
        for i, bit in enumerate(minterm_bits):
            if bit == '1':
                product.append(var_names[i])
            elif bit == '0':
                product.append(f"{var_names[i]}'")
        
        if product:
            expr_terms.append('*'.join(product))
    
    # Calculate statistics
    total_literals = sum(len(term.replace("'", "").replace("*", "")) for term in expr_terms)
    
    print(f"  [+] Generated {len(expr_terms):,} product terms")
    print(f"  [+] Total literals: {total_literals:,}")
    
    if max_terms and len(unique_minterms) > max_terms:
        print(f"  [!] Note: Limited to first {max_terms:,} terms (total: {len(unique_minterms):,})")
    
    # Join terms
    final_expr = ' + '.join(expr_terms)
    
    return final_expr, len(expr_terms), total_literals

# ============================================================================
# VERILOG CODE GENERATION
# ============================================================================

def count_literals_in_term(term):
    """Count literals in a product term."""
    return len(re.findall(r"[A-Za-z_]\w*'?", term))

def analyze_expression(expr_str):
    """Analyze expression and return statistics."""
    if not expr_str or not expr_str.strip():
        return {
            'num_terms': 0,
            'num_literals': 0,
            'avg_literals_per_term': 0,
            'max_term_size': 0,
            'min_term_size': 0
        }
    
    terms = [t.strip() for t in expr_str.split('+') if t.strip()]
    literal_counts = [count_literals_in_term(t) for t in terms]
    total_literals = sum(literal_counts)
    
    return {
        'num_terms': len(terms),
        'num_literals': total_literals,
        'avg_literals_per_term': total_literals / len(terms) if terms else 0,
        'max_term_size': max(literal_counts) if literal_counts else 0,
        'min_term_size': min(literal_counts) if literal_counts else 0
    }

def save_sop_expression_to_file(expression, expr_stats, union_result, output_path):
    """Save SOP expression to text file with statistics."""
    print(f"\n{'='*80}")
    print("SAVING SOP EXPRESSION TO TEXT FILE")
    print(f"{'='*80}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("24-BIT BOOLEAN FUNCTION - FINAL MINIMIZED EXPRESSION\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Variables: {NUM_VARS}\n")
        f.write(f"Form: Sum of Products (SOP)\n")
        f.write("\n")
        
        f.write("-"*80 + "\n")
        f.write("STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total unique terms:      {expr_stats['num_terms']:,}\n")
        f.write(f"Total literals:          {expr_stats['num_literals']:,}\n")
        f.write(f"Average literals/term:   {expr_stats['avg_literals_per_term']:.2f}\n")
        f.write(f"Maximum term size:       {expr_stats['max_term_size']}\n")
        f.write(f"Minimum term size:       {expr_stats['min_term_size']}\n")
        f.write(f"Chunks processed:        {union_result['chunk_count']:,}\n")
        f.write(f"Union combinations:      {union_result['unique_combinations']:,}\n")
        f.write("\n")
        
        f.write("-"*80 + "\n")
        f.write("FINAL EXPRESSION (ALL TERMS)\n")
        f.write("-"*80 + "\n\n")
        f.write(expression)
        f.write("\n\n")
        
        f.write("-"*80 + "\n")
        f.write("NOTES\n")
        f.write("-"*80 + "\n")
        f.write("• This expression represents the bitwise union of all chunks\n")
        f.write("• Each term is a product of literals (variables or their complements)\n")
        f.write("• Format: x0*x1'*x2 means (x0 AND NOT x1 AND x2)\n")
        f.write("• Terms are joined with '+' representing OR operation\n")
        f.write(f"• Total expression size: {len(expression):,} characters\n")
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  [+] SOP expression saved: {output_path}")
    print(f"  [+] File size: {file_size_mb:.2f} MB")
    print(f"  [+] Terms: {expr_stats['num_terms']:,}")
    print(f"  [+] Literals: {expr_stats['num_literals']:,}")

def convert_term_to_verilog(args):
    """
    Convert a single Boolean term to Verilog syntax.
    Worker function for parallel processing.
    
    Args:
        args: tuple of (term_string, var_names_dict)
    
    Returns:
        str: Verilog representation of the term
    """
    term, var_names = args
    term = term.strip()
    
    if not term:
        return ""
    
    # Split into literals
    literals = re.findall(r"[A-Za-z_]\w*'?", term)
    verilog_literals = []
    
    for lit in literals:
        if lit.endswith("'"):
            verilog_literals.append(f"~{lit[:-1]}")
        else:
            verilog_literals.append(lit)
    
    return f"({' & '.join(verilog_literals)})"

def convert_boolean_to_verilog(expr, var_names):
    """Convert Boolean SOP expression to Verilog syntax."""
    if not expr or not expr.strip():
        return "1'b0"
    
    terms = expr.split('+')
    verilog_terms = []
    
    for term in terms:
        term = term.strip()
        if not term:
            continue
        
        # Split into literals
        literals = re.findall(r"[A-Za-z_]\w*'?", term)
        verilog_literals = []
        
        for lit in literals:
            if lit.endswith("'"):
                verilog_literals.append(f"~{lit[:-1]}")
            else:
                verilog_literals.append(lit)
        
        verilog_terms.append(f"({' & '.join(verilog_literals)})")
    
    return ' | '.join(verilog_terms)

def worker_convert_terms(worker_id, terms_subset, var_names, output_file, total_terms):
    """
    Worker function to convert terms in a separate terminal window.
    
    Args:
        worker_id: Core/worker identifier
        terms_subset: List of terms to convert
        var_names: List of variable names
        output_file: Path to write results
        total_terms: Total number of terms across all workers
    """
    try:
        if sys.platform == 'win32':
            os.system(f'title Core {worker_id} - Converting {len(terms_subset):,} terms')
        
        print(f"\n{'='*70}")
        print(f"CORE {worker_id} - VERILOG CONVERSION STARTED")
        print(f"{'='*70}")
        print(f"Terms to convert: {len(terms_subset):,} / {total_terms:,}")
        print(f"Output: {os.path.basename(output_file)}")
        print(f"{'='*70}\n")
        
        verilog_terms = []
        start_time = time.perf_counter()
        
        for i, term in enumerate(terms_subset):
            term = term.strip()
            if not term:
                continue
            
            # Split into literals
            literals = re.findall(r"[A-Za-z_]\w*'?", term)
            verilog_literals = []
            
            for lit in literals:
                if lit.endswith("'"):
                    verilog_literals.append(f"~{lit[:-1]}")
                else:
                    verilog_literals.append(lit)
            
            verilog_terms.append(f"({' & '.join(verilog_literals)})")
            
            # Progress updates
            if (i + 1) % 10000 == 0:
                elapsed = time.perf_counter() - start_time
                progress = (i + 1) / len(terms_subset) * 100
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(terms_subset) - i - 1) / rate if rate > 0 else 0
                print(f"  Progress: {i+1:,}/{len(terms_subset):,} ({progress:.1f}%) | "
                      f"Rate: {rate:,.0f} terms/s | ETA: {eta:.1f}s")
        
        # Write results to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(verilog_terms))
        
        elapsed = time.perf_counter() - start_time
        
        print(f"\n{'='*70}")
        print(f"CORE {worker_id} - COMPLETED")
        print(f"{'='*70}")
        print(f"Converted: {len(verilog_terms):,} terms in {elapsed:.2f}s")
        print(f"Rate: {len(verilog_terms)/elapsed:,.0f} terms/s")
        print(f"{'='*70}\n")
        
        # Write completion marker
        marker_file = output_file.replace('.txt', '_complete.txt')
        with open(marker_file, 'w') as f:
            f.write(f"complete:{elapsed:.2f}")
        
        print("Press any key to close this window...")
        input()
        
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"CORE {worker_id} - ERROR")
        print(f"{'='*70}")
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        error_file = output_file.replace('.txt', '_error.txt')
        with open(error_file, 'w') as f:
            f.write(f"error:{e}")
        
        print("\nPress any key to close this window...")
        input()

def convert_boolean_to_verilog_parallel_terminals(expr, var_names, num_cores=None):
    """
    Convert Boolean SOP expression to Verilog syntax using parallel processing
    with separate terminal windows for each core.
    
    Args:
        expr: Boolean expression string
        var_names: List of variable names
        num_cores: Number of cores to use (None = all available)
    
    Returns:
        str: Verilog expression
    """
    if not expr or not expr.strip():
        return "1'b0"
    
    if num_cores is None:
        num_cores = cpu_count()
    
    terms = [t.strip() for t in expr.split('+') if t.strip()]
    
    if len(terms) < 10000:
        # For small expressions, parallel overhead not worth it
        print(f"      Expression too small for parallel processing - using single core...")
        return convert_boolean_to_verilog(expr, var_names)
    
    print(f"      Using {num_cores} CPU cores with separate terminal windows...")
    
    # Split terms across cores
    terms_per_core = len(terms) // num_cores
    term_chunks = []
    for i in range(num_cores):
        start_idx = i * terms_per_core
        end_idx = start_idx + terms_per_core if i < num_cores - 1 else len(terms)
        term_chunks.append(terms[start_idx:end_idx])
    
    print(f"      Split {len(terms):,} terms into {num_cores} chunks")
    for i, chunk in enumerate(term_chunks):
        print(f"        Core {i}: {len(chunk):,} terms")
    
    # Create temporary directory for outputs
    temp_dir = tempfile.mkdtemp(prefix='verilog_conversion_')
    output_files = [os.path.join(temp_dir, f'core{i}_output.txt') for i in range(num_cores)]
    
    print(f"      Temporary output directory: {temp_dir}")
    print(f"\n      Launching {num_cores} terminal windows...")
    
    # Launch workers in separate terminals
    processes = []
    for i in range(num_cores):
        # Create Python script for worker
        worker_script = f"""
import sys
import os
import time
import re

sys.path.insert(0, r'{os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))}')

# Worker function inline
worker_id = {i}
output_file = r'{output_files[i]}'
var_names = {var_names}
total_terms = {len(terms)}

terms_subset = {repr(term_chunks[i])}

try:
    if sys.platform == 'win32':
        os.system(f'title Core {{worker_id}} - Converting {{len(terms_subset):,}} terms')
    
    print(f"\\n{{'='*70}}")
    print(f"CORE {{worker_id}} - VERILOG CONVERSION STARTED")
    print(f"{{'='*70}}")
    print(f"Terms to convert: {{len(terms_subset):,}} / {{total_terms:,}}")
    print(f"Output: {{os.path.basename(output_file)}}")
    print(f"{{'='*70}}\\n")
    
    verilog_terms = []
    start_time = time.perf_counter()
    
    for idx, term in enumerate(terms_subset):
        term = term.strip()
        if not term:
            continue
        
        literals = re.findall(r"[A-Za-z_]\\w*'?", term)
        verilog_literals = []
        
        for lit in literals:
            if lit.endswith("'"):
                verilog_literals.append(f"~{{lit[:-1]}}")
            else:
                verilog_literals.append(lit)
        
        verilog_terms.append(f"({{' & '.join(verilog_literals)}})")
        
        if (idx + 1) % 10000 == 0:
            elapsed = time.perf_counter() - start_time
            progress = (idx + 1) / len(terms_subset) * 100
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (len(terms_subset) - idx - 1) / rate if rate > 0 else 0
            print(f"  Progress: {{idx+1:,}}/{{len(terms_subset):,}} ({{progress:.1f}}%) | "
                  f"Rate: {{rate:,.0f}} terms/s | ETA: {{eta:.1f}}s")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\\n'.join(verilog_terms))
    
    elapsed = time.perf_counter() - start_time
    
    print(f"\\n{{'='*70}}")
    print(f"CORE {{worker_id}} - COMPLETED")
    print(f"{{'='*70}}")
    print(f"Converted: {{len(verilog_terms):,}} terms in {{elapsed:.2f}}s")
    print(f"Rate: {{len(verilog_terms)/elapsed:,.0f}} terms/s")
    print(f"{{'='*70}}\\n")
    
    marker_file = output_file.replace('.txt', '_complete.txt')
    with open(marker_file, 'w') as f:
        f.write(f"complete:{{elapsed:.2f}}")
    
    print("Press any key to close this window...")
    input()
    
except Exception as e:
    print(f"\\n{{'='*70}}")
    print(f"CORE {{worker_id}} - ERROR")
    print(f"{{'='*70}}")
    print(f"ERROR: {{e}}")
    import traceback
    traceback.print_exc()
    
    error_file = output_file.replace('.txt', '_error.txt')
    with open(error_file, 'w') as f:
        f.write(f"error:{{e}}")
    
    print("\\nPress any key to close this window...")
    input()
"""
        
        # Write worker script to temp file
        worker_script_path = os.path.join(temp_dir, f'worker_core{i}.py')
        with open(worker_script_path, 'w', encoding='utf-8') as f:
            f.write(worker_script)
        
        # Launch in PowerShell window
        if sys.platform == 'win32':
            cmd = f'Start-Process powershell -ArgumentList "-NoExit", "-Command", "python {worker_script_path}"'
            p = subprocess.Popen(['powershell', '-Command', cmd], shell=True)
            processes.append(p)
            print(f"        [+] Core {i}: Terminal window opened")
            time.sleep(0.3)
    
    print(f"\n      All {num_cores} terminal windows launched!")
    print(f"      Waiting for workers to complete...")
    
    # Wait for all workers to complete
    completed = [False] * num_cores
    start_wait = time.perf_counter()
    
    while not all(completed):
        for i in range(num_cores):
            if not completed[i]:
                marker_file = output_files[i].replace('.txt', '_complete.txt')
                error_file = output_files[i].replace('.txt', '_error.txt')
                
                if os.path.exists(marker_file):
                    completed[i] = True
                    print(f"      [+] Core {i} completed!")
                elif os.path.exists(error_file):
                    completed[i] = True
                    print(f"      [!] Core {i} failed!")
        
        time.sleep(1)
    
    total_wait = time.perf_counter() - start_wait
    print(f"\n      All workers finished in {total_wait:.2f}s")
    
    # Collect results from files
    print(f"      Collecting results from {num_cores} cores...")
    all_verilog_terms = []
    
    for i, output_file in enumerate(output_files):
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                core_terms = f.read().strip().split('\n')
                all_verilog_terms.extend(core_terms)
            print(f"        Core {i}: Collected {len(core_terms):,} terms")
    
    print(f"      Total terms collected: {len(all_verilog_terms):,}")
    
    # Clean up temp directory
    import shutil
    try:
        shutil.rmtree(temp_dir)
        print(f"      Cleaned up temporary directory")
    except:
        print(f"      Note: Temp directory not cleaned: {temp_dir}")
    
    return ' | '.join(all_verilog_terms)

def format_verilog_terms_parallel(verilog_terms, num_cores=None):
    """
    Format Verilog terms into multi-line output using parallel processing.
    Each core handles a chunk of terms and formats them with proper indentation.
    
    Args:
        verilog_terms: List of Verilog term strings
        num_cores: Number of cores to use (None = all available)
    
    Returns:
        str: Formatted multi-line Verilog assignment
    """
    if num_cores is None:
        num_cores = cpu_count()
    
    if len(verilog_terms) < 10000:
        # For small expressions, parallel overhead not worth it
        result = "    assign result = \n        "
        for i, term in enumerate(verilog_terms):
            result += term
            if i < len(verilog_terms) - 1:
                result += " |\n        "
        result += ";\n\n"
        return result
    
    print(f"      Using {num_cores} cores for parallel formatting...")
    
    # Split terms across cores
    terms_per_core = len(verilog_terms) // num_cores
    term_chunks = []
    for i in range(num_cores):
        start_idx = i * terms_per_core
        end_idx = start_idx + terms_per_core if i < num_cores - 1 else len(verilog_terms)
        term_chunks.append(verilog_terms[start_idx:end_idx])
    
    # Format each chunk in parallel
    with Pool(num_cores) as pool:
        formatted_chunks = pool.map(format_term_chunk, 
                                    [(chunk, i, len(verilog_terms)) for i, chunk in enumerate(term_chunks)])
    
    # Merge formatted chunks
    result = "    assign result = \n        "
    result += " |\n        ".join(formatted_chunks)
    result += ";\n\n"
    
    return result

def format_term_chunk(args):
    """
    Format a chunk of Verilog terms (worker function).
    
    Args:
        args: tuple of (terms_list, chunk_id, total_terms)
    
    Returns:
        str: Formatted terms joined with ' | '
    """
    terms, chunk_id, total_terms = args
    return ' |\n        '.join(terms)

def generate_verilog_module(expr_str, num_vars, term_count, literal_count, module_name="boolean_function_24bit"):
    """Generate Verilog HDL module from Boolean expression."""
    import time
    
    print(f"\n  Generating Verilog module...")
    print(f"    Terms to process: {term_count:,}")
    print(f"    Literals to process: {literal_count:,}")
    
    # Estimate time based on observed performance
    # Approximately 0.5 microseconds per term for conversion + formatting
    estimated_seconds = (term_count * 0.5e-6) + (literal_count * 0.1e-6)
    if estimated_seconds > 60:
        print(f"    Estimated time: {estimated_seconds/60:.1f} minutes")
    else:
        print(f"    Estimated time: {estimated_seconds:.1f} seconds")
    
    start_time = time.perf_counter()
    
    var_names = [f"x{i}" for i in range(num_vars)]
    
    print(f"    [1/4] Building module header...")
    verilog = f"""// Source SOP Expression: {os.path.basename(SOP_TEXT_FILE)}
// Verilog Module: {module_name}
// Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
// Variables: {num_vars}
// Terms: {term_count:,}
// Literals: {literal_count:,}
// Description: 24-bit Boolean function minimized with KMapSolver4D

module {module_name} (
    input [{num_vars-1}:0] inputs,
    output result
);

// Variable mapping
"""
    
    print(f"    [2/4] Adding wire declarations (multicore)...")
    # Parallelize wire declarations for large designs
    if num_vars > 16:
        wire_chunks = []
        chunk_size = (num_vars + NUM_CORES - 1) // NUM_CORES
        for i in range(0, num_vars, chunk_size):
            chunk = []
            for j in range(i, min(i + chunk_size, num_vars)):
                chunk.append(f"    wire {var_names[j]} = inputs[{j}];\n")
            wire_chunks.append(''.join(chunk))
        verilog += ''.join(wire_chunks)
        print(f"      Generated {num_vars} wire declarations")
    else:
        for i, var in enumerate(var_names):
            verilog += f"    wire {var} = inputs[{i}];\n"
    
    print(f"\n    [3/4] Converting Boolean expression to Verilog...")
    verilog += "\n    // Minimized Boolean expression (SOP form)\n"
    verilog += f"    // WARNING: Large expression with {term_count:,} terms\n"
    verilog += f"    // Synthesis may require significant time and resources\n\n"
    
    if expr_str:
        print(f"      Converting {term_count:,} terms to Verilog syntax...")
        conversion_start = time.perf_counter()
        
        # Use parallel conversion with separate terminals for large expressions
        if term_count > 10000:
            print(f"      Large expression detected - using multicore processing with terminal windows...")
            verilog_expr = convert_boolean_to_verilog_parallel_terminals(expr_str, var_names, num_cores=NUM_CORES)
        else:
            verilog_expr = convert_boolean_to_verilog(expr_str, var_names)
        
        conversion_time = time.perf_counter() - conversion_start
        print(f"      Conversion complete in {conversion_time:.2f}s")
        
        # Split long expressions across multiple lines with parallel formatting
        print(f"      Formatting multi-line output (multicore)...")
        if len(verilog_expr) > 100:
            format_start = time.perf_counter()
            terms = verilog_expr.split(' | ')
            
            # Use parallel formatting for large expressions
            verilog += format_verilog_terms_parallel(terms, num_cores=NUM_CORES)
            
            format_time = time.perf_counter() - format_start
            print(f"      Formatted {len(terms):,} terms in {format_time:.2f}s (multicore)")
        else:
            verilog += f"    assign result = {verilog_expr};\n\n"
    else:
        verilog += "    assign result = 1'b0;  // Empty expression\n\n"
    
    verilog += "endmodule\n\n"
    
    # Add testbench
    print(f"    [4/4] Adding testbench module...")
    verilog += f"""// Testbench for {module_name}
module {module_name}_tb;

    reg [{num_vars-1}:0] inputs;
    wire result;

    {module_name} uut (
        .inputs(inputs),
        .result(result)
    );

    initial begin
        $dumpfile("{module_name}.vcd");
        $dumpvars(0, {module_name}_tb);
        
        // Test cases
        inputs = {num_vars}'b0;
        #10;
        $display("Input: %b, Output: %b", inputs, result);
        
        inputs = {num_vars}'h1;
        #10;
        $display("Input: %b, Output: %b", inputs, result);
        
        inputs = {num_vars}'hFFFFFF;
        #10;
        $display("Input: %b, Output: %b", inputs, result);
        
        $finish;
    end

endmodule
"""
    
    total_time = time.perf_counter() - start_time
    verilog_size_mb = len(verilog) / (1024 * 1024)
    
    print(f"\n  [+] Verilog generation complete!")
    print(f"      Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"      Module size: {verilog_size_mb:.2f} MB")
    print(f"      Lines of code: {verilog.count(chr(10)):,}")
    
    return verilog

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def estimate_processing_time(row_count):
    """
    Estimate total processing time based on data size.
    Based on benchmarks: ~1M rows takes ~10-15 seconds for union,
    ~1M terms takes ~5-10 seconds for expression extraction,
    ~1M terms takes ~30-60 seconds for Verilog generation.
    """
    # Rough estimates
    union_time = row_count * 10e-6  # 10 microseconds per row
    extraction_time = row_count * 5e-6  # 5 microseconds per minterm
    verilog_time = row_count * 50e-6  # 50 microseconds per term for Verilog
    
    total_seconds = union_time + extraction_time + verilog_time
    return total_seconds

def estimate_processing_time(row_count):
    """
    Estimate total processing time based on data size.
    Based on benchmarks: ~1M rows takes ~10-15 seconds for union,
    ~1M terms takes ~5-10 seconds for expression extraction,
    ~1M terms takes ~30-60 seconds for Verilog generation.
    """
    # Rough estimates
    union_time = row_count * 10e-6  # 10 microseconds per row
    extraction_time = row_count * 5e-6  # 5 microseconds per minterm
    verilog_time = row_count * 50e-6  # 50 microseconds per term for Verilog
    
    total_seconds = union_time + extraction_time + verilog_time
    return total_seconds

def main():
    """Main execution function."""
    print("="*80)
    print("24-BIT BOOLEAN FUNCTION ANALYSIS")
    print("Bitwise Union & Verilog Generation (MULTICORE)")
    print("="*80)
    print(f"Using {NUM_CORES} CPU cores for parallel processing")
    
    # Check if merged CSV exists
    if not os.path.exists(MERGED_CSV):
        print(f"\n[!] Error: Merged CSV not found: {MERGED_CSV}")
        print(f"[!] Please run the 24-bit multicore test first to generate the CSV file.")
        return False
    
    # Quick estimate based on file size
    file_size_mb = os.path.getsize(MERGED_CSV) / (1024 * 1024)
    estimated_rows = file_size_mb * 20000  # Rough estimate: ~20k rows per MB
    estimated_time = estimate_processing_time(estimated_rows)
    
    print(f"\n[*] Pre-flight Analysis:")
    print(f"    CSV file size: {file_size_mb:.1f} MB")
    print(f"    Estimated rows: ~{estimated_rows:,.0f}")
    print(f"    Available CPU cores: {NUM_CORES}")
    print(f"    Estimated processing time: {estimated_time/60:.1f} - {estimated_time/60*1.5:.1f} minutes")
    print(f"\n    Note: For ~11.7M terms with ~727M literals ({NUM_CORES} cores):")
    print(f"          - Union operation: ~10-15 seconds")
    print(f"          - Expression extraction: ~30-60 seconds")
    print(f"          - Verilog generation (parallel): ~2-4 minutes (vs 10-15 min single-core)")
    print(f"          - File writing: ~1-2 seconds")
    print(f"          Total estimated: ~3-6 minutes (with {NUM_CORES}x parallel speedup)")
    print()
    
    # Step 1: Perform bitwise union
    union_result = perform_bitwise_union(MERGED_CSV, NUM_VARS)
    if union_result is None:
        print("\n[!] Failed to perform bitwise union")
        return False
    
    # Step 2: Extract expression from union
    expression, term_count, literal_count = extract_expression_from_union(
        union_result, 
        NUM_VARS,
        max_terms=None  # Extract all terms
    )
    
    # Step 3: Analyze expression
    expr_stats = analyze_expression(expression)
    
    # Step 4: Save SOP expression to text file FIRST (before Verilog generation)
    save_sop_expression_to_file(expression, expr_stats, union_result, SOP_TEXT_FILE)
    
    # Step 5: Generate Verilog code (references the text file created above)
    print(f"\n{'='*80}")
    print("GENERATING VERILOG MODULE")
    print(f"{'='*80}")
    verilog_code = generate_verilog_module(
        expression, 
        NUM_VARS, 
        expr_stats['num_terms'],
        expr_stats['num_literals']
    )
    
    # Step 6: Save Verilog code to .v file
    print(f"\n{'='*80}")
    print("SAVING VERILOG MODULE")
    print(f"{'='*80}")
    
    # Create verilog directory if it doesn't exist
    os.makedirs(VERILOG_DIR, exist_ok=True)
    
    with open(VERILOG_FILE, 'w', encoding='utf-8') as f:
        f.write(verilog_code)
    
    verilog_size_mb = os.path.getsize(VERILOG_FILE) / (1024 * 1024)
    print(f"  [+] Verilog module saved: {VERILOG_FILE}")
    print(f"  [+] File size: {verilog_size_mb:.2f} MB")
    print(f"  [+] Module name: boolean_function_24bit")
    
    # Final summary
    print(f"\n{'='*80}")
    print("[SUCCESS] ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\n[*] Results:")
    print(f"    Terms extracted: {expr_stats['num_terms']:,}")
    print(f"    Total literals: {expr_stats['num_literals']:,}")
    print(f"    Average literals/term: {expr_stats['avg_literals_per_term']:.2f}")
    print(f"\n[*] Output Files:")
    print(f"    SOP Expression: {SOP_TEXT_FILE}")
    print(f"    Verilog Module: {VERILOG_FILE}")
    print(f"    Verilog Directory: {VERILOG_DIR}")
    print(f"\n[*] Next Steps:")
    print(f"    1. Review SOP expression in {os.path.basename(SOP_TEXT_FILE)}")
    print(f"    2. Use {os.path.basename(VERILOG_FILE)} for FPGA/ASIC synthesis")
    print(f"    3. Simulate with testbench included in Verilog file")
    
    return True


if __name__ == "__main__":
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

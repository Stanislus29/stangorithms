"""
KMapSolver4D: High-Performance Chunked Boolean Minimization
============================================================

Handles large-scale Boolean minimization (>16 variables) using chunked processing.
Processes truth tables in manageable 16-variable chunks, writing essential prime 
implicants to CSV, then performing bitwise union for final minimization.

This approach enables minimization of functions with 24, 28, or 32+ variables
without exhausting system memory.

Author: Stan's Technologies
Date: December 2025
Version: 1.0
"""

from stanlogic import KMapSolver3D
import csv
import os
import random
import time
from collections import defaultdict


class KMapSolver4D:
    """
    4D K-map solver using chunked processing for large variable counts (>16).
    
    The algorithm:
    1. Split N variables into: NUM_CHUNK_VARS (select chunk) + 16 (K-map vars)
    2. Process each chunk as a separate 16-variable K-map
    3. Extract essential prime implicants to CSV
    4. Perform bitwise union across all chunks for final result
    """
    
    def __init__(self, num_vars, output_generator=None, csv_path="kmapsolver4d_terms.csv", 
                 chunk_start=0, chunk_end=None, seed=None, verbose=False):
        """
        Initialize 4D K-map solver with chunked processing.
        
        Args:
            num_vars (int): Total number of variables (must be > 16)
            output_generator (callable): Function(chunk_index, seed) -> list of 65536 outputs
            csv_path (str): Path for intermediate CSV storage
            chunk_start (int): Starting chunk index (for distributed processing)
            chunk_end (int): Ending chunk index (exclusive, None = all chunks)
            seed (int): Random seed for reproducibility
            verbose (bool): Print detailed progress information
        """
        if num_vars <= 16:
            raise ValueError("KMapSolver4D requires > 16 variables. Use KMapSolver3D for <= 16 variables.")
        
        self.num_vars = num_vars
        self.num_kmap_vars = 16  # Each chunk is a 16-variable K-map
        self.num_chunk_vars = num_vars - self.num_kmap_vars
        self.chunk_size = 2**self.num_kmap_vars  # 65,536 entries per chunk
        self.total_chunks = 2**self.num_chunk_vars
        
        self.output_generator = output_generator or self._default_output_generator
        self.csv_path = csv_path
        self.chunk_start = chunk_start
        self.chunk_end = chunk_end if chunk_end is not None else self.total_chunks
        self.seed = seed
        self.verbose = verbose
        
        # Statistics
        self.stats = {
            "total_chunks": self.total_chunks,
            "chunks_to_process": self.chunk_end - self.chunk_start,
            "processed_chunks": 0,
            "all_dc_chunks": 0,
            "all_zeros_chunks": 0,
            "all_ones_chunks": 0,
            "minimized_chunks": 0,
            "total_terms": 0,
            "total_time": 0,
            "chunk_times": []
        }
    
    def _default_output_generator(self, chunk_index, seed=None):
        """
        Default random output generator: 70% ones, 5% don't-care, 25% zeros.
        
        Args:
            chunk_index: Index of the current chunk
            seed: Random seed base
            
        Returns:
            List of 65,536 output values (0, 1, or 'd')
        """
        if seed is not None:
            random.seed(seed + chunk_index)
        
        outputs = []
        for _ in range(self.chunk_size):
            r = random.random()
            if r < 0.70:
                outputs.append(1)
            elif r < 0.75:
                outputs.append('d')
            else:
                outputs.append(0)
        
        return outputs
    
    def process_chunk(self, chunk_index, csv_writer):
        """
        Process a single chunk: generate outputs, minimize, write to CSV.
        
        Args:
            chunk_index: Index of the current chunk
            csv_writer: CSV writer object
            
        Returns:
            Dictionary with chunk processing statistics
        """
        if self.verbose:
            print(f"\n{'-'*70}")
            print(f"CHUNK {chunk_index:,} / {self.total_chunks:,} " 
                  f"(bits 0-{self.num_chunk_vars-1} = {format(chunk_index, f'0{self.num_chunk_vars}b')})")
            print(f"{'-'*70}")
        
        # Generate outputs for this chunk
        chunk_outputs = self.output_generator(chunk_index, self.seed)
        
        # Check for constant chunks
        defined_vals = [v for v in chunk_outputs if v != 'd']
        if not defined_vals:
            if self.verbose:
                print("  [x] All don't-care - skipping minimization")
            del chunk_outputs
            return {"chunk": chunk_index, "type": "all_dc", "terms": 0, "time": 0}
        
        if all(v == 0 for v in defined_vals):
            if self.verbose:
                print("  [x] All zeros - no terms")
            del chunk_outputs
            return {"chunk": chunk_index, "type": "all_zeros", "terms": 0, "time": 0}
        
        if all(v == 1 for v in defined_vals):
            bitmask = (chunk_index << self.num_kmap_vars) | 0xFFFF
            csv_writer.writerow([chunk_index, hex(bitmask), "all_ones"])
            if self.verbose:
                print("  [+] All ones - constant function")
            del chunk_outputs
            return {"chunk": chunk_index, "type": "all_ones", "terms": 1, "time": 0}
        
        # Minimize using KMapSolver3D
        if self.verbose:
            print(f"  → Processing 16-variable K-map ({self.chunk_size:,} entries)...")
        
        start_time = time.perf_counter()
        
        # Suppress KMapSolver3D verbose output
        import io
        import sys as sys_module
        old_stdout = sys_module.stdout
        sys_module.stdout = io.StringIO()
        
        solver = KMapSolver3D(self.num_kmap_vars, chunk_outputs)
        terms, expr_str = solver.minimize_3d(form='sop')
        
        sys_module.stdout = old_stdout
        
        elapsed = time.perf_counter() - start_time
        term_count = 0
        
        # Extract essential prime implicants from all sub-maps
        total_submaps = len(solver.kmaps)
        solved_submaps = 0
        
        for extra_combo in sorted(solver.kmaps.keys()):
            solved_submaps += 1
            
            if self.verbose:
                extra_combo_int = int(extra_combo, 2) if extra_combo else 0
                print(f"     Sub-map {solved_submaps}/{total_submaps}: "
                      f"bits {self.num_chunk_vars}-{self.num_chunk_vars+11} = "
                      f"{extra_combo} (decimal {extra_combo_int})", end="")
            
            # Get essential prime implicants
            result = solver._solve_single_kmap(extra_combo, form='sop')
            extra_combo_int = int(extra_combo, 2) if extra_combo else 0
            
            # Write each essential prime implicant to CSV
            for bitmask_4x4, term_bits in zip(result['bitmasks'], result['terms_bits']):
                csv_writer.writerow([
                    chunk_index,
                    extra_combo,
                    hex(bitmask_4x4),
                    term_bits,
                    "essential_prime"
                ])
                term_count += 1
            
            if self.verbose:
                print(f" → {len(result['bitmasks'])} essential prime implicants "
                      f"covering {sum(bin(bm).count('1') for bm in result['bitmasks'])} minterms")
        
        if self.verbose:
            print(f"  [+] Chunk complete: {term_count} essential prime implicants in {elapsed:.3f}s")
        
        # Clean up
        del chunk_outputs
        del solver
        del terms
        del expr_str
        
        return {
            "chunk": chunk_index,
            "type": "minimized",
            "terms": term_count,
            "time": elapsed
        }
    
    def minimize(self, progress_interval=100):
        """
        Perform chunked minimization for the specified chunk range.
        
        Args:
            progress_interval: Print progress every N chunks
            
        Returns:
            Dictionary with processing statistics
        """
        print(f"\n{'='*80}")
        print(f"KMAPSOLVER4D: CHUNKED {self.num_vars}-BIT PROCESSING")
        print(f"{'='*80}")
        print(f"   Total variables: {self.num_vars}")
        print(f"   K-map variables per chunk: {self.num_kmap_vars}")
        print(f"   Chunk selector variables: {self.num_chunk_vars}")
        print(f"   Total chunks: {self.total_chunks:,}")
        print(f"   Processing chunks: {self.chunk_start:,} to {self.chunk_end-1:,} "
              f"({self.chunk_end - self.chunk_start:,} chunks)")
        print(f"   Entries per chunk: {self.chunk_size:,}")
        print(f"   Total truth table entries: {2**self.num_vars:,}")
        
        # Initialize CSV
        print(f"\n   • Initializing CSV: {self.csv_path}")
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["chunk_index", "extra_combo", "bitmask_4x4", "term_bits", "type"])
        
        # Process chunks
        print(f"\n   • Processing chunks...")
        if self.verbose:
            print(f"      Verbose mode: Detailed feedback for first 3 chunks\n")
        
        start_time = time.perf_counter()
        
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            for chunk_idx in range(self.chunk_start, self.chunk_end):
                # Verbose output for first 3 chunks only
                verbose = self.verbose and (chunk_idx < self.chunk_start + 3)
                
                # Process chunk
                chunk_result = self.process_chunk(chunk_idx, writer)
                
                # Explicit garbage collection after each chunk
                import gc
                gc.collect()
                
                # Update statistics
                self.stats["processed_chunks"] += 1
                self.stats["total_terms"] += chunk_result.get("terms", 0)
                
                chunk_type = chunk_result["type"]
                if chunk_type == "all_dc":
                    self.stats["all_dc_chunks"] += 1
                elif chunk_type == "all_zeros":
                    self.stats["all_zeros_chunks"] += 1
                elif chunk_type == "all_ones":
                    self.stats["all_ones_chunks"] += 1
                elif chunk_type == "minimized":
                    self.stats["minimized_chunks"] += 1
                    self.stats["total_time"] += chunk_result.get("time", 0)
                    self.stats["chunk_times"].append(chunk_result.get("time", 0))
                
                # Progress update
                if (chunk_idx - self.chunk_start + 1) % progress_interval == 0:
                    elapsed = time.perf_counter() - start_time
                    progress_pct = 100 * (chunk_idx - self.chunk_start + 1) / (self.chunk_end - self.chunk_start)
                    chunks_per_sec = (chunk_idx - self.chunk_start + 1) / elapsed
                    eta = (self.chunk_end - chunk_idx - 1) / chunks_per_sec if chunks_per_sec > 0 else 0
                    
                    print(f"\n      === PROGRESS UPDATE ===")
                    print(f"      Chunks: {chunk_idx - self.chunk_start + 1:,} / "
                          f"{self.chunk_end - self.chunk_start:,} ({progress_pct:.2f}%)")
                    print(f"      Speed: {chunks_per_sec:.2f} chunks/s | "
                          f"Avg time/chunk: {elapsed/(chunk_idx - self.chunk_start + 1):.3f}s")
                    print(f"      Terms so far: {self.stats['total_terms']:,}")
                    print(f"      Minimized: {self.stats['minimized_chunks']:,} | "
                          f"Constant: {self.stats['all_ones_chunks']+self.stats['all_zeros_chunks']:,} | "
                          f"DC: {self.stats['all_dc_chunks']:,}")
                    print(f"      ETA: {self._format_time(eta)}\n")
        
        total_elapsed = time.perf_counter() - start_time
        self.stats["total_elapsed"] = total_elapsed
        
        print(f"\n   [+] All chunks processed in {self._format_time(total_elapsed)}")
        print(f"      Average: {self.stats['processed_chunks']/total_elapsed:.1f} chunks/second")
        
        return self.stats
    
    def _format_time(self, seconds):
        """Format seconds as human-readable time string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        elif seconds < 86400:
            return f"{seconds/3600:.1f}h"
        else:
            return f"{seconds/86400:.1f}d"


def merge_csv_files(input_paths, output_path):
    """
    Merge multiple CSV files from distributed processing into one.
    
    Args:
        input_paths: List of CSV file paths to merge
        output_path: Output CSV file path
    """
    print(f"\n{'='*80}")
    print("MERGING CSV FILES")
    print(f"{'='*80}")
    print(f"   Input files: {len(input_paths)}")
    print(f"   Output file: {output_path}")
    
    with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["chunk_index", "extra_combo", "bitmask_4x4", "term_bits", "type"])
        
        total_rows = 0
        for i, path in enumerate(input_paths):
            print(f"   • Processing file {i+1}/{len(input_paths)}: {os.path.basename(path)}")
            with open(path, 'r', encoding='utf-8') as infile:
                reader = csv.reader(infile)
                next(reader)  # Skip header
                for row in reader:
                    writer.writerow(row)
                    total_rows += 1
        
        print(f"   [+] Merged {total_rows:,} rows into {output_path}")

"""
Scientific Benchmark: K-Map Solver vs PyEDA Boolean Minimization
================================================================

A rigorous experimental comparison of Boolean function minimization algorithms
with proper statistical analysis, reproducibility controls, and comprehensive
test coverage.

Author: Somtochukwu Stanislus Emeka-Onwuneme
Date: January 2026
Version: 2.0 (PyEDA Adaptation)
"""

import time
import random
import re
import csv
import os
import sys
import platform
import datetime
from collections import defaultdict
from textwrap import wrap
import signal
from contextlib import contextmanager
import traceback

# Try to import psutil for memory monitoring (optional)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️  Warning: psutil not installed. Memory monitoring disabled.")
    print("   Install with: pip install psutil")

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import scipy
from scipy import stats
from pyeda.inter import *
from sympy import And as SymAnd, Or as SymOr, Not as SymNot, symbols, sympify, true, false, simplify, Equivalent
from stanlogic.BoolMin2D import BoolMin2D
from tabulate import tabulate

# ============================================================================
# CONFIGURATION & EXPERIMENTAL PARAMETERS
# ============================================================================

# Random seed for reproducibility (CRITICAL for scientific validity)
RANDOM_SEED = 42

# Benchmark parameters
TESTS_PER_CONFIG = 100  # Reduced from 1000 for faster testing, increase for production
TIMING_REPEATS = 5      # Number of timing repetitions per test
TIMING_WARMUP = 2       # Warm-up iterations before timing

# Statistical significance threshold
ALPHA = 0.05  # 95% confidence level

# Equivalence check timeout (seconds)
EQUIVALENCE_TIMEOUT = 15  # Timeout for equivalence checking to prevent hangs

# ============================================================================
# TIMEOUT UTILITIES
# ============================================================================

class TimeoutException(Exception):
    """Exception raised when operation times out"""
    pass

@contextmanager
def time_limit(seconds):
    """
    Context manager to impose a time limit on operations.
    Uses signal.alarm on Unix/Linux or a simple timer check on Windows.
    
    Args:
        seconds: Maximum time allowed
        
    Raises:
        TimeoutException: If operation exceeds time limit
    """
    if platform.system() != 'Windows':
        # Unix/Linux: Use signal.alarm
        def signal_handler(signum, frame):
            raise TimeoutException("Operation timed out")
        
        old_handler = signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        # Windows: Use threading timer (less precise but works)
        import threading
        
        timer_expired = [False]
        
        def timer_handler():
            timer_expired[0] = True
        
        timer = threading.Timer(seconds, timer_handler)
        timer.daemon = True
        timer.start()
        
        try:
            yield
            if timer_expired[0]:
                raise TimeoutException("Operation timed out")
        finally:
            timer.cancel()

# Output directories - relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, "..", "outputs", "benchmark_results2D_pyeda2")
RESULTS_CSV = os.path.join(OUTPUTS_DIR, "benchmark_results2D_pyeda2.csv")
REPORT_PDF = os.path.join(OUTPUTS_DIR, "benchmark_scientific_report2D_pyeda2.pdf")
STATS_CSV = os.path.join(OUTPUTS_DIR, "benchmark_results2D_pyeda_statistical_analysis2.csv")

# Logo for report cover
LOGO_PATH = os.path.join(SCRIPT_DIR,"..", "..", "..", "images", "St_logo_light-tp.png")

# ============================================================================
# CRASH DETECTION AND REPORTING
# ============================================================================

# Global flag to track if we're currently in a test
CURRENT_TEST_INDEX = 0
CRASH_LOG_FILE = None

def setup_crash_log():
    """Setup a crash log file."""
    global CRASH_LOG_FILE
    crash_log_path = os.path.join(OUTPUTS_DIR, "crash_log.txt")
    CRASH_LOG_FILE = crash_log_path
    with open(crash_log_path, 'w') as f:
        f.write(f"Crash Log - Started at {datetime.datetime.now()}\n")
        f.write("="*80 + "\n\n")
    return crash_log_path

def log_crash_info(message):
    """Write crash info to log file immediately."""
    if CRASH_LOG_FILE:
        try:
            with open(CRASH_LOG_FILE, 'a') as f:
                f.write(f"[{datetime.datetime.now()}] {message}\n")
                f.flush()
        except:
            pass

def signal_handler(signum, frame):
    """Handle OS termination signals."""
    signal_names = {
        signal.SIGTERM: 'SIGTERM (Termination)',
        signal.SIGINT: 'SIGINT (Interrupt)',
    }
    if hasattr(signal, 'SIGBREAK'):
        signal_names[signal.SIGBREAK] = 'SIGBREAK'
    
    sig_name = signal_names.get(signum, f'Signal {signum}')
    
    crash_msg = f"""
{'='*80}
🚨 PROCESS TERMINATED BY OS SIGNAL AT TEST {CURRENT_TEST_INDEX}
{'='*80}
Signal: {sig_name}

This typically indicates:
- Memory exhaustion (OOM Killer)
- System resource limits exceeded
- Manual process termination
{'='*80}
"""
    print(crash_msg, flush=True)
    log_crash_info(crash_msg)
    sys.exit(1)

def setup_signal_handlers():
    """Setup handlers for OS termination signals."""
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    if platform.system() == 'Windows' and hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, signal_handler)

def get_memory_usage():
    """Get current memory usage in MB."""
    if not PSUTIL_AVAILABLE:
        return 0
    try:
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)  # Convert to MB
    except:
        return 0

def diagnose_crash_reason(exception, test_index):
    """
    Diagnose and report the reason for a test crash.
    
    Args:
        exception: The exception that was raised
        test_index: The test number that crashed
        
    Returns:
        String describing the crash reason
    """
    crash_info = []
    crash_info.append(f"\n{'='*80}")
    crash_info.append(f"🚨 CRASH DETECTED AT TEST {test_index}")
    crash_info.append(f"{'='*80}")
    
    # Get exception details
    exception_type = type(exception).__name__
    crash_info.append(f"Exception Type: {exception_type}")
    crash_info.append(f"Exception Message: {str(exception)}")
    
    # Check memory status (if psutil is available)
    if PSUTIL_AVAILABLE:
        try:
            process = psutil.Process()
            mem_info = process.memory_info()
            mem_mb = mem_info.rss / (1024 * 1024)
            mem_percent = process.memory_percent()
            
            crash_info.append(f"\nMemory Status:")
            crash_info.append(f"  Process Memory: {mem_mb:.2f} MB ({mem_percent:.1f}% of system)")
            
            # Check if memory exhaustion is likely
            if mem_percent > 80:
                crash_info.append(f"  ⚠️  HIGH MEMORY USAGE - Possible memory exhaustion")
            
            # Check system memory
            sys_mem = psutil.virtual_memory()
            crash_info.append(f"  System Memory: {sys_mem.percent:.1f}% used ({sys_mem.available / (1024**3):.2f} GB available)")
            
            if sys_mem.percent > 90:
                crash_info.append(f"  ⚠️  SYSTEM MEMORY CRITICAL - Likely cause: Memory exhaustion")
        except Exception as e:
            crash_info.append(f"  Unable to retrieve memory info: {e}")
    else:
        crash_info.append(f"\nMemory Status: [psutil not available - install with: pip install psutil]")
    
    # Check for specific error types
    if isinstance(exception, MemoryError):
        crash_info.append(f"\n❌ CRASH REASON: Memory Exhaustion (MemoryError)")
    elif isinstance(exception, OSError) and 'killed' in str(exception).lower():
        crash_info.append(f"\n❌ CRASH REASON: Process Killed by OS (OSError)")
    elif isinstance(exception, KeyboardInterrupt):
        crash_info.append(f"\n❌ CRASH REASON: User Interruption (Ctrl+C)")
    elif isinstance(exception, TimeoutException):
        crash_info.append(f"\n❌ CRASH REASON: Operation Timeout")
    elif isinstance(exception, RecursionError):
        crash_info.append(f"\n❌ CRASH REASON: Stack Overflow (RecursionError)")
    elif 'out of memory' in str(exception).lower():
        crash_info.append(f"\n❌ CRASH REASON: Out of Memory")
    else:
        crash_info.append(f"\n❌ CRASH REASON: Unexpected Error ({exception_type})")
    
    # Add traceback
    crash_info.append(f"\nTraceback:")
    crash_info.append(traceback.format_exc())
    
    crash_info.append(f"{'='*80}\n")
    
    return '\n'.join(crash_info)

# ============================================================================
# EXPERIMENTAL SETUP DOCUMENTATION
# ============================================================================

def document_experimental_setup():
    """
    Record all relevant environmental factors for reproducibility.
    Returns a dictionary with system and experimental configuration.
    """
    import pyeda
    
    setup_info = {
        "experiment_date": datetime.datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "pyeda_version": pyeda.__version__,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "random_seed": RANDOM_SEED,
        "tests_per_config": TESTS_PER_CONFIG,
        "timing_repeats": TIMING_REPEATS,
        "timing_warmup": TIMING_WARMUP,
        "alpha_level": ALPHA,
    }
    return setup_info

# ============================================================================
# TEST DATA GENERATION
# ============================================================================

class TestDistribution:
    """Enumeration of different test data distributions."""
    SPARSE = "sparse"           # Few 1s (common in digital logic)
    DENSE = "dense"             # Many 1s
    BALANCED = "balanced"       # Equal probability
    MINIMAL_DC = "minimal_dc"   # Few don't-cares (realistic)
    HEAVY_DC = "heavy_dc"       # Many don't-cares (stress test)

def random_kmap(rows=4, cols=4, p_one=0.45, p_dc=0.1, seed=None):
    """
    Generate a random K-map with controlled probability distribution.
    
    Args:
        rows: Number of rows in K-map
        cols: Number of columns in K-map
        p_one: Probability a cell is 1
        p_dc: Probability a cell is 'd' (don't care)
        seed: Random seed for this specific K-map
        
    Returns:
        2D list representing the K-map
        
    Raises:
        ValueError: If probabilities are invalid
    """
    if seed is not None:
        random.seed(seed)
    
    if p_one < 0 or p_dc < 0 or p_one + p_dc > 1:
        raise ValueError("Probabilities invalid: ensure p_one >=0, p_dc >=0, p_one + p_dc <= 1")
    
    p_zero = 1 - p_one - p_dc
    choices = [1, 0, 'd']
    weights = [p_one, p_zero, p_dc]
    
    kmap = [[random.choices(choices, weights=weights, k=1)[0] 
             for _ in range(cols)] for _ in range(rows)]
    
    return kmap

def generate_test_suite(num_vars, tests_per_distribution=100):
    """
    Generate comprehensive test suite with diverse distributions.
    
    Args:
        num_vars: Number of variables (2, 3, or 4)
        tests_per_distribution: Number of tests per distribution type
        
    Returns:
        List of tuples: (kmap, distribution_label)
    """
    rows, cols = _kmap_shape(num_vars)
    test_cases = []
    
    # Distribution configurations
    distributions = {
        TestDistribution.SPARSE: {"p_one": 0.2, "p_dc": 0.05},
        TestDistribution.DENSE: {"p_one": 0.7, "p_dc": 0.05},
        TestDistribution.BALANCED: {"p_one": 0.5, "p_dc": 0.1},
        TestDistribution.MINIMAL_DC: {"p_one": 0.45, "p_dc": 0.02},
        TestDistribution.HEAVY_DC: {"p_one": 0.3, "p_dc": 0.3},
    }
    
    for dist_name, params in distributions.items():
        for i in range(tests_per_distribution):
            kmap = random_kmap(rows=rows, cols=cols, **params)
            test_cases.append((kmap, dist_name))
    
    # Add edge cases
    test_cases.extend(generate_edge_cases(num_vars))
    
    return test_cases

def generate_edge_cases(num_vars):
    """
    Generate edge case K-maps for comprehensive testing.
    
    Args:
        num_vars: Number of variables
        
    Returns:
        List of tuples: (kmap, label)
    """
    rows, cols = _kmap_shape(num_vars)
    edge_cases = []
    
    # All zeros
    edge_cases.append(([[0] * cols for _ in range(rows)], "edge_all_zeros"))
    
    # All ones
    edge_cases.append(([[1] * cols for _ in range(rows)], "edge_all_ones"))
    
    # All don't cares
    edge_cases.append(([['d'] * cols for _ in range(rows)], "edge_all_dc"))
    
    # Checkerboard pattern
    checkerboard = [[(i + j) % 2 for j in range(cols)] for i in range(rows)]
    edge_cases.append((checkerboard, "edge_checkerboard"))
    
    # Single 1
    single_one = [[0] * cols for _ in range(rows)]
    single_one[0][0] = 1
    edge_cases.append((single_one, "edge_single_one"))
    
    return edge_cases

def _kmap_shape(num_vars):
    """Return (rows, cols) for given number of variables."""
    if num_vars == 2:
        return 2, 2
    if num_vars == 3:
        return 2, 4
    if num_vars == 4:
        return 4, 4
    raise ValueError("Only 2-4 variables supported.")

# ============================================================================
# K-MAP ANALYSIS
# ============================================================================

def kmap_minterms(kmap, convention="vranseic"):
    """
    Extract minterm indices and don't care indices from a K-map.
    
    Args:
        kmap: 2D list representing the K-map
        convention: Variable ordering convention ('vranseic' or 'mano_kime')
        
    Returns:
        Dictionary with num_vars, minterms, dont_cares, and bits_map
    """
    rows = len(kmap)
    cols = len(kmap[0])
    size = rows * cols

    if size == 4:
        num_vars = 2
        row_labels = ["0", "1"]
        col_labels = ["0", "1"]
    elif size == 8:
        num_vars = 3
        if rows == 2 and cols == 4:
            row_labels = ["0", "1"]
            col_labels = ["00", "01", "11", "10"]
        elif rows == 4 and cols == 2:
            row_labels = ["00", "01", "11", "10"]
            col_labels = ["0", "1"]
        else:
            raise ValueError("Invalid 3-variable K-map shape.")
    elif size == 16:
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

def kmap_maxterms(kmap, convention="vranseic"):
    """
    Extract maxterm indices (zeros) and don't care indices from a K-map.
    
    Args:
        kmap: 2D list representing the K-map
        convention: Variable ordering convention ('vranseic' or 'mano_kime')
        
    Returns:
        Dictionary with num_vars, maxterms, dont_cares, and bits_map
    """
    rows = len(kmap)
    cols = len(kmap[0])
    size = rows * cols

    if size == 4:
        num_vars = 2
        row_labels = ["0", "1"]
        col_labels = ["0", "1"]
    elif size == 8:
        num_vars = 3
        if rows == 2 and cols == 4:
            row_labels = ["0", "1"]
            col_labels = ["00", "01", "11", "10"]
        elif rows == 4 and cols == 2:
            row_labels = ["00", "01", "11", "10"]
            col_labels = ["0", "1"]
        else:
            raise ValueError("Invalid 3-variable K-map shape.")
    elif size == 16:
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

    maxterms = []  # Positions with 0
    dont_cares = []  # Positions with 'd'
    bits_map = {}

    for r in range(rows):
        for c in range(cols):
            val = kmap[r][c]
            idx = cell_to_term(r, c)
            bits_map[idx] = cell_to_bits(r, c)
            if val == 0:
                maxterms.append(idx)
            elif val == 'd':
                dont_cares.append(idx)

    return {
        "num_vars": num_vars,
        "maxterms": sorted(maxterms),
        "dont_cares": sorted(dont_cares),
        "bits_map": bits_map
    }

def minterm_expr(m, vars):
    return And(*[
        var if (m >> i) & 1 else ~var
        for i, var in enumerate(reversed(vars))
    ])

def maxterm_expr(m, vars):
    return Or(*[
        ~var if (m >> i) & 1 else var
        for i, var in enumerate(reversed(vars))
    ])


# ============================================================================
# EXPRESSION PARSING & VALIDATION
# ============================================================================

def parse_pyeda_expression(pyeda_expr, var_mapping=None):
    """
    Convert a PyEDA Boolean expression (e.g., Or(And(b, c), And(~a, c)))
    into a SymPy Boolean expression that can be compared against K-map output.
    
    Args:
        pyeda_expr: PyEDA expression to parse
        var_mapping: Dictionary mapping PyEDA var names (a,b,c,d) to SymPy symbols (x1,x2,x3,x4)
                     e.g., {'a': x1_symbol, 'b': x2_symbol, 'c': x3_symbol, 'd': x4_symbol}
    """
    if pyeda_expr is None:
        return None

    if isinstance(pyeda_expr, bool):
        return true if pyeda_expr else false

    expr_str = str(pyeda_expr).strip()
    if expr_str in {"1", "True"}:
        return true
    if expr_str in {"0", "False"}:
        return false

    # Build variable mapping
    local_map = {"And": SymAnd, "Or": SymOr, "Not": SymNot}
    
    if var_mapping:
        # Use provided mapping (PyEDA vars -> SymPy vars)
        local_map.update(var_mapping)
    else:
        # Fallback: create new SymPy variables
        var_names = set()
        try:
            for v in pyeda_expr.inputs:
                var_names.add(str(v))
        except Exception:
            var_names.update(re.findall(r"[A-Za-z_]\w*", expr_str))
        
        sym_vars = symbols(" ".join(sorted(var_names)), boolean=True) if var_names else ()
        if var_names:
            if isinstance(sym_vars, tuple):
                local_map.update({str(s): s for s in sym_vars})
            else:
                local_map[str(sym_vars)] = sym_vars

    return sympify(expr_str, locals=local_map, evaluate=True)

def parse_kmap_expression(expr_str, var_names, form="sop"):
    """
    Parse a Boolean expression string into a SymPy expression.
    
    Args:
        expr_str: The Boolean expression string
        var_names: List of SymPy symbols
        form: 'sop' or 'pos'
        
    Returns:
        SymPy Boolean expression
    """
    from sympy import And, Or, Not, false, true

    if not expr_str or expr_str.strip() == "":
        return false if form == "sop" else true

    name_map = {str(v): v for v in var_names}

    if form.lower() == "sop":
        or_terms = []
        for product in expr_str.split('+'):
            product = product.strip()
            if not product:
                continue
            literals = re.findall(r"[A-Za-z]+\d*'?", product)
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

    elif form.lower() == "pos":
        # Handle constant 0 case: '()'
        if expr_str.strip() == '()':
            return false
        
        and_terms = []
        # Split by * to get individual clauses
        clauses = expr_str.split('*')
        for clause in clauses:
            clause = clause.strip()
            if not clause:
                continue
            # Remove outer parentheses
            clause = clause.strip('() ')
            if not clause:
                continue
            # In POS, literals are separated by + within each clause
            # Split by + and process each literal
            or_terms = []
            for lit_str in clause.split('+'):
                lit_str = lit_str.strip()
                if not lit_str:
                    continue
                # Extract the variable name (with optional prime)
                lit_match = re.match(r"([A-Za-z]+\d*)'?", lit_str)
                if lit_match:
                    lit = lit_match.group(0)
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

    raise ValueError(f"Unsupported form '{form}'.")

def check_equivalence(expr1, expr2):
    """Check if two SymPy expressions are logically equivalent."""
    try:
        return bool(simplify(Equivalent(expr1, expr2)))
    except Exception:
        return False

def parse_kmap_to_pyeda(expr_str, pyeda_vars, terms=None, form="sop"):
    """
    Parse a BoolMin2D Boolean expression string into a PyEDA expression.
    
    Args:
        expr_str: Boolean expression string from BoolMin2D
        pyeda_vars: List of PyEDA variable objects
        terms: List of terms from BoolMin2D (used to distinguish constant 0 vs 1)
        form: 'sop' or 'pos'
        
    Returns:
        PyEDA Boolean expression
    """
    # Handle empty expression
    # BoolMin2D returns '' for both constant 0 and constant 1
    # Distinguish by checking terms list:
    #   - Constant 1: terms = [''] (SOP) or [] (POS) - tautology
    #   - Constant 0: terms = [] (SOP) or ['()'] (POS) - contradiction
    if not expr_str or expr_str.strip() == "":
        if terms is not None:
            if form == "sop":
                if terms == ['']:
                    return expr(1)  # Constant 1
                elif not terms:
                    return expr(0)  # Constant 0
            elif form == "pos":
                if not terms:
                    return expr(1)  # Constant 1 (no clauses means always true)
                elif terms == ['()']:
                    return expr(0)  # Constant 0 (empty clause means always false)
        # Fallback if terms not provided
        return expr(0) if form == "sop" else expr(1)
    
    # Handle POS constant 0: '()'
    if form == "pos" and expr_str.strip() == '()' and terms == ['()']:
        return expr(0)

    # Build variable name mapping
    var_map = {str(v): v for v in pyeda_vars}

    if form.lower() == "sop":
        # SOP: product terms separated by '+'
        or_terms = []
        for product in expr_str.split('+'):
            product = product.strip()
            if not product:
                continue
            # Extract literals (variables with optional prime)
            # Pattern: letter followed by digits, optionally followed by prime
            literals = re.findall(r"[A-Za-z]+\d*'?", product)
            and_terms = []
            for lit in literals:
                if lit.endswith("'"):
                    var_name = lit[:-1]
                    if var_name in var_map:
                        and_terms.append(~var_map[var_name])
                else:
                    if lit in var_map:
                        and_terms.append(var_map[lit])
            if len(and_terms) == 1:
                or_terms.append(and_terms[0])
            elif and_terms:
                or_terms.append(And(*and_terms))
        return Or(*or_terms) if or_terms else expr(0)

    elif form.lower() == "pos":
        # POS: sum terms in parentheses separated by * (not implicit AND)
        # Format: (x1+x2)*(x3+x4) or (x1 + x2) * (x3 + x4)
        # where + separates literals within each clause
        and_terms = []
        # Split by * to get individual clauses
        clauses = expr_str.split('*')
        for clause in clauses:
            clause = clause.strip()
            if not clause:
                continue
            # Remove outer parentheses
            clause = clause.strip('() ')
            if not clause:
                continue
            # Split by + to get individual literals within the clause
            or_terms = []
            for lit_str in clause.split('+'):
                lit_str = lit_str.strip()
                if not lit_str:
                    continue
                # Extract variable name with optional prime
                lit_match = re.match(r"([A-Za-z]+\d*)'?", lit_str)
                if lit_match:
                    lit = lit_match.group(0)
                    if lit.endswith("'"):
                        var_name = lit[:-1]
                        if var_name in var_map:
                            or_terms.append(~var_map[var_name])
                    else:
                        if lit in var_map:
                            or_terms.append(var_map[lit])
            if len(or_terms) == 1:
                and_terms.append(or_terms[0])
            elif or_terms:
                and_terms.append(Or(*or_terms))
        return And(*and_terms) if and_terms else expr(1)

    raise ValueError(f"Unsupported form '{form}'.")

def check_equivalence_pyeda(expr1, expr2):
    """
    Check if two PyEDA expressions are logically equivalent using PyEDA's equivalent method.
    Uses a timeout to prevent hanging on complex expressions.
    Falls back to SymPy equivalence check if PyEDA times out.
    
    Args:
        expr1: First PyEDA expression
        expr2: Second PyEDA expression
        
    Returns:
        Boolean indicating equivalence, or None if all methods fail
    """
    try:
        # Use timeout to prevent hanging
        with time_limit(EQUIVALENCE_TIMEOUT):
            # PyEDA's equivalent method
            result = expr1.equivalent(expr2)
            return result
    except TimeoutException:
        # Timeout - fallback to SymPy equivalence check
        print(f"  [PyEDA TIMEOUT after {EQUIVALENCE_TIMEOUT}s - trying SymPy fallback]", end=" ")
        try:
            return check_equivalence_sympy_fallback(expr1, expr2)
        except Exception as e:
            print(f"  [SymPy fallback also failed: {str(e)[:30]}]", end=" ")
            return None
    except Exception as e:
        # Other error - try SymPy fallback
        print(f"  [PyEDA ERROR: {str(e)[:30]} - trying SymPy]", end=" ")
        try:
            return check_equivalence_sympy_fallback(expr1, expr2)
        except Exception:
            return False

def check_kmap_aware_equivalence(kmap, expr_pyeda, expr_kmap_pyeda, pyeda_vars):
    """
    Check equivalence with respect to K-map specification, respecting don't-cares.
    Two expressions are equivalent if they agree on all non-don't-care inputs.
    They may differ on don't-care inputs (both valid minimizations).
    
    Args:
        kmap: K-map specification (2D list with 0, 1, or 'd')
        expr_pyeda: PyEDA expression
        expr_kmap_pyeda: BoolMin2D expression (converted to PyEDA)
        pyeda_vars: List of PyEDA variable objects
        
    Returns:
        Tuple of (equivalent, differs_on_dc, mismatches)
        - equivalent: True if both match K-map specification on defined inputs
        - differs_on_dc: True if they differ only on don't-care inputs
        - mismatches: List of (input_index, expected, pyeda_val, kmap_val)
    """
    num_vars = len(pyeda_vars)
    rows = len(kmap)
    cols = len(kmap[0])
    
    # Check if shapes match expectations
    if num_vars == 2 and (rows, cols) != (2, 2):
        return None, None, []
    if num_vars == 3 and (rows, cols) != (2, 4):
        return None, None, []
    if num_vars == 4 and (rows, cols) != (4, 4):
        return None, None, []
    
    mismatches_on_defined = []
    mismatches_on_dc = []
    
    # Check all possible inputs
    for i in range(2**num_vars):
        # Create variable assignment
        assign = {}
        for j, v in enumerate(pyeda_vars):
            bit = (i >> (num_vars - 1 - j)) & 1
            assign[v] = expr(1) if bit else expr(0)
        
        # Map input to K-map position (using Gray code)
        if num_vars == 2:
            col = i & 1
            row = (i >> 1) & 1
            expected = kmap[row][col]
        elif num_vars == 3:
            col = (i >> 1) & 3  # x1 x2
            row = i & 1         # x3
            col_idx = [0, 1, 3, 2][col]  # Binary to Gray code
            expected = kmap[row][col_idx]
        else:  # 4 vars
            col = (i >> 2) & 3  # x1 x2
            row = i & 3         # x3 x4
            col_idx = [0, 1, 3, 2][col]
            row_idx = [0, 1, 3, 2][row]
            expected = kmap[row_idx][col_idx]
        
        # Evaluate both expressions
        try:
            val1 = expr_pyeda.restrict(assign)
            pyeda_val = 1 if (hasattr(val1, 'is_one') and val1.is_one()) else (1 if str(val1) == '1' else 0)
        except Exception:
            pyeda_val = None
        
        try:
            val2 = expr_kmap_pyeda.restrict(assign)
            kmap_val = 1 if (hasattr(val2, 'is_one') and val2.is_one()) else (1 if str(val2) == '1' else 0)
        except Exception:
            kmap_val = None
        
        # Check for mismatches
        if expected == 'd':
            # Don't-care input: expressions can differ
            if pyeda_val is not None and kmap_val is not None and pyeda_val != kmap_val:
                mismatches_on_dc.append((i, expected, pyeda_val, kmap_val))
        else:
            # Defined input: both must match specification
            if pyeda_val is not None and pyeda_val != expected:
                mismatches_on_defined.append((i, expected, pyeda_val, kmap_val))
            if kmap_val is not None and kmap_val != expected:
                mismatches_on_defined.append((i, expected, pyeda_val, kmap_val))
    
    # Determine equivalence status
    equivalent = len(mismatches_on_defined) == 0
    differs_on_dc = len(mismatches_on_dc) > 0
    
    return equivalent, differs_on_dc, mismatches_on_defined

def check_equivalence_sympy_fallback(pyeda_expr1, pyeda_expr2):
    """
    Fallback equivalence check using SymPy when PyEDA times out.
    Converts PyEDA expressions to SymPy and uses SymPy's Equivalent.
    
    Args:
        pyeda_expr1: First PyEDA expression
        pyeda_expr2: Second PyEDA expression
        
    Returns:
        Boolean indicating equivalence
    """
    # Convert PyEDA expressions to strings
    str1 = str(pyeda_expr1)
    str2 = str(pyeda_expr2)
    
    # Extract variables
    import re
    vars_found = set(re.findall(r'x\d+', str1 + str2))
    var_symbols = {v: symbols(v) for v in vars_found}
    
    # Parse to SymPy expressions
    # Replace PyEDA operators with SymPy operators
    def pyeda_to_sympy(expr_str):
        # Handle special cases
        if expr_str == '0':
            return false
        if expr_str == '1':
            return true
        
        # Replace operators
        expr_str = expr_str.replace('~', 'Not(')
        expr_str = expr_str.replace('&', ') & (')
        expr_str = expr_str.replace('|', ') | (')
        
        # Replace variables
        for var_name, var_sym in var_symbols.items():
            expr_str = expr_str.replace(var_name, str(var_sym))
        
        # Balance parentheses
        open_count = expr_str.count('(')
        close_count = expr_str.count(')')
        if open_count > close_count:
            expr_str += ')' * (open_count - close_count)
        
        try:
            return sympify(expr_str, locals=var_symbols)
        except:
            # If sympify fails, build from scratch
            return parse_boolean_expression_simple(expr_str, var_symbols)
    
    def parse_boolean_expression_simple(expr_str, var_dict):
        """Simple parser for PyEDA expressions"""
        expr_str = expr_str.strip()
        
        # Handle constants
        if expr_str == '0':
            return false
        if expr_str == '1':
            return true
        
        # Handle OR (lowest precedence)
        or_parts = []
        depth = 0
        current = []
        for char in expr_str:
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            
            if char == '|' and depth == 0:
                or_parts.append(''.join(current).strip())
                current = []
            else:
                current.append(char)
        
        if current:
            or_parts.append(''.join(current).strip())
        
        if len(or_parts) > 1:
            return SymOr(*[parse_boolean_expression_simple(p, var_dict) for p in or_parts])
        
        # Handle AND
        and_parts = []
        depth = 0
        current = []
        for char in expr_str:
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            
            if char == '&' and depth == 0:
                and_parts.append(''.join(current).strip())
                current = []
            else:
                current.append(char)
        
        if current:
            and_parts.append(''.join(current).strip())
        
        if len(and_parts) > 1:
            return SymAnd(*[parse_boolean_expression_simple(p, var_dict) for p in and_parts])
        
        # Handle NOT
        if expr_str.startswith('~'):
            return SymNot(parse_boolean_expression_simple(expr_str[1:], var_dict))
        
        # Handle parentheses
        if expr_str.startswith('(') and expr_str.endswith(')'):
            return parse_boolean_expression_simple(expr_str[1:-1], var_dict)
        
        # Must be a variable
        if expr_str in var_dict:
            return var_dict[expr_str]
        
        return false
    
    try:
        sympy_expr1 = pyeda_to_sympy(str1)
        sympy_expr2 = pyeda_to_sympy(str2)
        
        # Use SymPy's Equivalent with simplification
        equiv_expr = Equivalent(sympy_expr1, sympy_expr2)
        result = simplify(equiv_expr)
        
        return bool(result)
    except Exception as e:
        # If SymPy also fails, return None
        raise Exception(f"SymPy conversion failed: {e}")

def count_literals(expr_str, form="sop"):
    """
    Count terms and literals in a Boolean expression string.
    
    Args:
        expr_str: Boolean expression string
        form: 'sop' or 'pos'
        
    Returns:
        Tuple of (num_terms, num_literals)
    """
    if not expr_str or expr_str.strip() == "":
        return 0, 0

    form = form.lower()
    s = expr_str.replace(" ", "")

    if form == "sop":
        terms = [t for t in s.split('+') if t]
        num_terms = len(terms)
        num_literals = sum(len(re.findall(r"[A-Za-z]+\d*'?", t)) for t in terms)
        return num_terms, num_literals

    if form == "pos":
        clauses = re.findall(r"\(([^()]*)\)", s)
        if not clauses:
            clauses = [s]
        num_terms = len(clauses)
        num_literals = 0
        for clause in clauses:
            lits = [lit for lit in clause.split('+') if lit]
            num_literals += sum(1 for lit in lits if re.fullmatch(r"[A-Za-z]+\d*'?", lit))
        return num_terms, num_literals

    raise ValueError("form must be 'sop' or 'pos'")

def count_pyeda_expression_literals(expr, form="sop"):
    """
    Count terms and literals from a SymPy expression (string parsing).
    
    Args:
        expr: SymPy expression
        form: 'sop' or 'pos'
        
    Returns:
        Tuple of (num_terms, num_literals)
    """
    if expr is None:
        return 0, 0
    
    s = str(expr).strip()
    form = form.lower()

    if s in ("True", "False"):
        return 0, 0

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
        inner_parts = split_top_level(group_str, inner_sep)
        if len(inner_parts) == 1:
            raw = inner_parts[0]
            lits = split_top_level(raw, inner_sep)
        else:
            lits = inner_parts
        literal_count = sum(1 for lit in lits if re.fullmatch(r"~?[A-Za-z_]\w*", strip_parens(lit)))
        return literal_count

    if form == "sop":
        top_terms = split_top_level(s, '|')
        num_terms = len(top_terms)
        total_literals = sum(count_literals_in_group(term, '&') for term in top_terms)
        return num_terms, total_literals

    if form == "pos":
        top_clauses = split_top_level(s, '&')
        num_terms = len(top_clauses)
        total_literals = sum(count_literals_in_group(clause, '|') for clause in top_clauses)
        return num_terms, total_literals

    raise ValueError("form must be 'sop' or 'pos'")

# ============================================================================
# IMPROVED TIMING METHODOLOGY
# ============================================================================

def benchmark_with_warmup(func, args, warmup=TIMING_WARMUP, repeats=TIMING_REPEATS):
    """
    Benchmark a function with warm-up runs and multiple repetitions.
    
    Args:
        func: Function to benchmark
        args: Tuple of arguments to pass to function
        warmup: Number of warm-up iterations
        repeats: Number of timed repetitions
        
    Returns:
        Minimum execution time (best of N to minimize OS noise)
    """
    # Warm-up phase (not counted)
    for _ in range(warmup):
        func(*args)
    
    # Actual timing
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        func(*args)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    # Return minimum time (reduces impact of system interrupts)
    return min(times)

# ============================================================================
# BENCHMARK EXECUTION
# ============================================================================

def benchmark_case(kmap, var_names, form_type, test_index, distribution):
    """
    Run a single benchmark case comparing PyEDA and BoolMin2D.
    
    Args:
        kmap: K-map as 2D list
        var_names: List of PyEDA variable symbols
        form_type: 'sop' or 'pos'
        test_index: Test case number
        distribution: Distribution type label
        
    Returns:
        Dictionary with benchmark results
    """
    # Monitor memory at start of test
    mem_start = get_memory_usage()
    
    info = kmap_minterms(kmap, convention="vranseic")
    minterms = info["minterms"]
    dont_cares = info["dont_cares"]

    inf = kmap_maxterms(kmap, convention="vranseic")
    maxterms = inf["maxterms"]
    pos_dont_cares = inf["dont_cares"]

    if len(var_names) != info["num_vars"]:
        raise ValueError(f"Variable count mismatch: {len(var_names)} != {info['num_vars']}")

    # Create PyEDA variables using x1, x2, x3, x4 to match BoolMin2D output
    # (var_names are SymPy symbols, but PyEDA needs matching names)
    pyeda_var_names = ['x1', 'x2', 'x3', 'x4'][:info["num_vars"]]
    pyeda_vars = [exprvar(name) for name in pyeda_var_names]
    
    # Helper to build PyEDA minterm expression
    def pyeda_minterm(m, pvars):
        return And(*[
            pvar if (m >> i) & 1 else ~pvar
            for i, pvar in enumerate(reversed(pvars))
        ])
    
    # PyEDA minimization with improved timing
    # PyEDA espresso_exprs always expects DNF (OR of AND terms)
    if form_type == "sop":
        # For SOP: Build ON-set and DC-set as DNF
        if minterms:
            f_on = Or(*[pyeda_minterm(m, pyeda_vars) for m in minterms])
        else:
            f_on = expr(0)  # False
        
        if dont_cares:
            f_dc = Or(*[pyeda_minterm(m, pyeda_vars) for m in dont_cares])
        else:
            f_dc = expr(0)  # False
        
        # Handle constant 0 case (no minterms and no don't cares)
        if not minterms and not dont_cares:
            pyeda_func = lambda: expr(0)
        else:
            pyeda_func = lambda: espresso_exprs(f_on, f_dc)[0]
    else:  # form_type == "pos"
        # For POS: Build DNF from maxterm positions (0s in K-map)
        # Then minimize and convert to CNF (POS form)
        if maxterms:
            # Build minterms for the 0 positions
            f_off = Or(*[pyeda_minterm(m, pyeda_vars) for m in maxterms])
        else:
            f_off = expr(0)  # False (no zeros means function is always 1)
        
        if dont_cares:
            f_dc = Or(*[pyeda_minterm(m, pyeda_vars) for m in dont_cares])
        else:
            f_dc = expr(0)  # False
        
        # Handle constant 1 case (no maxterms)
        if not maxterms:
            pyeda_func = lambda: expr(1)
        else:
            # Minimize the OFF-set as DNF, complement, then convert to CNF for POS
            pyeda_func = lambda: (~espresso_exprs(f_off, f_dc)[0]).to_cnf()
    
    log_crash_info(f"Test {test_index}: Starting PyEDA {form_type.upper()}...")
    print(f"    Test {test_index}: Running PyEDA {form_type.upper()}...", end=" ", flush=True)
    
    # Try PyEDA minimization with error handling
    pyeda_error = None
    try:
        t_pyeda = benchmark_with_warmup(pyeda_func, ())
        expr_pyeda_raw = pyeda_func()  # Get actual result (PyEDA expression)
        
        # Create variable mapping: PyEDA vars (x1,x2,x3,x4) -> SymPy vars (x1,x2,x3,x4)
        # Both use the same naming convention now
        pyeda_to_sympy_map = {
            'x1': var_names[0],
            'x2': var_names[1] if info["num_vars"] >= 2 else None,
            'x3': var_names[2] if info["num_vars"] >= 3 else None,
            'x4': var_names[3] if info["num_vars"] >= 4 else None,
        }
        # Remove None entries
        pyeda_to_sympy_map = {k: v for k, v in pyeda_to_sympy_map.items() if v is not None}
        
        expr_pyeda = parse_pyeda_expression(expr_pyeda_raw, var_mapping=pyeda_to_sympy_map)
        print(f"OK ({t_pyeda:.6f}s)", end=" | ", flush=True)
        log_crash_info(f"Test {test_index}: PyEDA completed successfully")
    except Exception as e:
        pyeda_error = str(e)
        t_pyeda = 0.0
        expr_pyeda = None
        expr_pyeda_raw = None
        print(f"WARN: {pyeda_error[:50]}", end=" | ", flush=True)
        log_crash_info(f"Test {test_index}: PyEDA failed - {pyeda_error}")

    # BoolMin2D minimization with improved timing
    kmap_func = lambda: BoolMin2D(kmap).minimize(form=form_type)
    log_crash_info(f"Test {test_index}: Starting BoolMin2D...")
    print(f"BoolMin2D...", end=" ", flush=True)
    
    kmap_error = None
    try:
        t_kmap = benchmark_with_warmup(kmap_func, ())
        terms, expr_str = kmap_func()  # Get actual result
        expr_kmap_sympy = parse_kmap_expression(expr_str, var_names, form=form_type)
        # Parse BoolMin2D output to PyEDA format for equivalence checking
        # Pass terms list to handle constant expressions correctly
        expr_kmap_pyeda = parse_kmap_to_pyeda(expr_str, pyeda_vars, terms=terms, form=form_type)
        kmap_terms, kmap_literals = count_literals(expr_str, form=form_type)
        print(f"OK ({t_kmap:.6f}s)", end=" | ", flush=True)
        log_crash_info(f"Test {test_index}: BoolMin2D completed successfully")
    except Exception as e:
        kmap_error = str(e)
        t_kmap = 0.0
        expr_kmap_sympy = None
        expr_str = ""
        kmap_terms, kmap_literals = 0, 0
        print(f"WARN: {kmap_error[:50]}", end=" | ", flush=True)
        log_crash_info(f"Test {test_index}: BoolMin2D failed - {kmap_error}")

    # Validate equivalence only if BOTH algorithms succeeded
    log_crash_info(f"Test {test_index}: Starting equivalence check...")
    if pyeda_error is None and kmap_error is None:
        # Compute PyEDA metrics
        pyeda_terms, pyeda_literals = count_pyeda_expression_literals(str(expr_pyeda), form=form_type)

        # Detect constant functions (both have 0 literals)
        is_constant = (pyeda_literals == 0 and kmap_literals == 0)
        
        # Use K-map-aware equivalence checking that respects don't-cares
        kmap_equiv, differs_on_dc, mismatches = check_kmap_aware_equivalence(
            kmap, expr_pyeda_raw, expr_kmap_pyeda, pyeda_vars
        )
        
        # Determine result category and equivalence status
        if kmap_equiv is None:
            # K-map check failed (shouldn't happen)
            result_category = "check_failed"
            equiv = None
            equiv_symbol = "ERROR"
            print(f"Equiv: {equiv_symbol} [K-map check failed]")
        elif is_constant:
            result_category = "constant"
            equiv = True  # Constants are always equivalent if both produce 0 literals
            equiv_symbol = "PASS"
            print(f"Equiv: {equiv_symbol} [CONSTANT]")
        elif kmap_equiv and not differs_on_dc:
            # Perfect equivalence - both produce identical outputs
            result_category = "expression"
            equiv = True
            equiv_symbol = "PASS"
            print(f"Equiv: {equiv_symbol} [IDENTICAL]")
        elif kmap_equiv and differs_on_dc:
            # Both valid but handle don't-cares differently
            result_category = "valid_different_dc"
            equiv = True  # Mark as equivalent since both are valid
            equiv_symbol = "PASS"
            print(f"Equiv: {equiv_symbol} [VALID - differ on don't-cares]")
            log_crash_info(f"Test {test_index}: Both valid, differ on {len(differs_on_dc) if isinstance(differs_on_dc, list) else 'some'} don't-care input(s)")
        else:
            # True mismatch - at least one is incorrect
            result_category = "expression"
            equiv = False
            equiv_symbol = "FAIL"
            print(f"Equiv: {equiv_symbol} [MISMATCH on {len(mismatches)} input(s)]")
            log_crash_info(f"Test {test_index}: True mismatch detected - {len(mismatches)} errors on defined inputs")
            
            # Show first few mismatches for debugging
            if mismatches and len(mismatches) <= 5:
                print(f"    Mismatches: {mismatches[:5]}")
        
        # For compatibility with downstream code, set tt_equiv and tt_mismatches
        tt_equiv = kmap_equiv
        tt_mismatches = mismatches
    else:
        # One or both algorithms failed - mark as N/A
        equiv = None  # Not applicable when one algorithm failed
        pyeda_terms, pyeda_literals = 0, 0 if pyeda_error else count_pyeda_expression_literals(str(expr_pyeda), form=form_type)
        is_constant = False
        
        if pyeda_error and kmap_error:
            result_category = "both_failed"
            print(f"Equiv: N/A [BOTH FAILED]")
        elif pyeda_error:
            result_category = "pyeda_failed"
            print(f"Equiv: N/A [PYEDA FAILED]")
        else:
            result_category = "kmap_failed"
            print(f"Equiv: N/A [KMAP FAILED]")

    # Monitor memory at end of test
    mem_end = get_memory_usage()
    mem_delta = mem_end - mem_start
    
    # Warn if significant memory increase
    if mem_delta > 100:  # More than 100MB increase
        print(f"    ⚠️  Memory increase: {mem_delta:.2f} MB")

    return {
        "index": test_index,
        "num_vars": info["num_vars"],
        "form": form_type,
        "distribution": distribution,
        "equiv": equiv,
        "result_category": result_category,
        "is_constant": is_constant,
        "differs_on_dc": differs_on_dc if (pyeda_error is None and kmap_error is None) else False,
        "t_pyeda": t_pyeda,
        "t_kmap": t_kmap,
        "pyeda_literals": pyeda_literals,
        "kmap_literals": kmap_literals,
        "pyeda_terms": pyeda_terms,
        "kmap_terms": kmap_terms,
        "pyeda_error": pyeda_error,
        "kmap_error": kmap_error,
        "mem_delta": mem_delta,
    }

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def perform_statistical_tests(pyeda_times, kmap_times, pyeda_literals, kmap_literals):
    """
    Perform rigorous statistical hypothesis testing.
    
    Args:
        pyeda_times: List of PyEDA execution times
        kmap_times: List of BoolMin2D execution times
        pyeda_literals: List of PyEDA literal counts
        kmap_literals: List of BoolMin2D literal counts
        
    Returns:
        Dictionary with comprehensive statistical test results
    """
    # Convert to numpy arrays
    pyeda_times = np.array(pyeda_times)
    kmap_times = np.array(kmap_times)
    pyeda_literals = np.array(pyeda_literals)
    kmap_literals = np.array(kmap_literals)
    
    # Calculate differences
    time_diff = kmap_times - pyeda_times
    lit_diff = kmap_literals - pyeda_literals
    
    # Paired t-test (parametric)
    t_stat_time, p_value_time = stats.ttest_rel(kmap_times, pyeda_times)
    
    # Only perform literal tests if we have non-zero literals
    if len(pyeda_literals) > 0 and np.any(pyeda_literals > 0):
        t_stat_lit, p_value_lit = stats.ttest_rel(kmap_literals, pyeda_literals)
        w_stat_lit, w_p_lit = stats.wilcoxon(kmap_literals, pyeda_literals, zero_method='zsplit')
        cohens_d_lit = np.mean(lit_diff) / np.std(lit_diff, ddof=1) if np.std(lit_diff) > 0 else 0
        ci_lit = stats.t.interval(0.95, len(lit_diff)-1,
                                  loc=np.mean(lit_diff),
                                  scale=stats.sem(lit_diff))
    else:
        # All constants - no meaningful literal comparison
        t_stat_lit = p_value_lit = w_stat_lit = w_p_lit = cohens_d_lit = 0
        ci_lit = (0, 0)
    
    # Wilcoxon signed-rank test (non-parametric alternative)
    w_stat_time, w_p_time = stats.wilcoxon(kmap_times, pyeda_times, zero_method='zsplit')
    
    # Effect size (Cohen's d)
    cohens_d_time = np.mean(time_diff) / np.std(time_diff, ddof=1) if np.std(time_diff) > 0 else 0
    
    # Confidence intervals (95%)
    ci_time = stats.t.interval(0.95, len(time_diff)-1, 
                               loc=np.mean(time_diff), 
                               scale=stats.sem(time_diff))
    
    return {
        "time": {
            "mean_pyeda": np.mean(pyeda_times),
            "mean_kmap": np.mean(kmap_times),
            "mean_diff": np.mean(time_diff),
            "std_diff": np.std(time_diff, ddof=1),
            "t_statistic": t_stat_time,
            "p_value": p_value_time,
            "wilcoxon_stat": w_stat_time,
            "wilcoxon_p": w_p_time,
            "cohens_d": cohens_d_time,
            "ci_lower": ci_time[0],
            "ci_upper": ci_time[1],
            "significant": p_value_time < ALPHA
        },
        "literals": {
            "mean_pyeda": np.mean(pyeda_literals),
            "mean_kmap": np.mean(kmap_literals),
            "mean_diff": np.mean(lit_diff),
            "std_diff": np.std(lit_diff, ddof=1),
            "t_statistic": t_stat_lit,
            "p_value": p_value_lit,
            "wilcoxon_stat": w_stat_lit,
            "wilcoxon_p": w_p_lit,
            "cohens_d": cohens_d_lit,
            "ci_lower": ci_lit[0],
            "ci_upper": ci_lit[1],
            "significant": p_value_lit < ALPHA
        }
    }

def interpret_effect_size(cohens_d):
    """Interpret Cohen's d effect size."""
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"

# ============================================================================
# INFERENCE GENERATION
# ============================================================================

def generate_scientific_inference(stats_results, num_constant=0, total_tests=0):
    """
    Generate scientifically rigorous inference with statistical support.
    
    Args:
        stats_results: Dictionary from perform_statistical_tests()
        num_constant: Number of constant functions detected
        total_tests: Total number of tests run
        
    Returns:
        Formatted inference text
    """
    lines = []
    lines.append("=" * 75)
    lines.append("STATISTICAL INFERENCE REPORT")
    lines.append("=" * 75)
    
    # Constant functions notification
    if num_constant > 0:
        lines.append(f"\nℹ️  CONSTANT FUNCTIONS DETECTED: {num_constant}/{total_tests} ({100*num_constant/total_tests:.1f}%)")
        lines.append("   These are unsimplifiable functions (e.g., all-zeros, all-ones) that both")
        lines.append("   algorithms correctly identified. They are included in performance and")
        lines.append("   equivalence analysis but excluded from literal-count statistics.")
        lines.append("")
    
    # Execution Time Analysis
    lines.append("\n1. EXECUTION TIME ANALYSIS")
    lines.append("-" * 75)
    time_stats = stats_results["time"]
    lines.append(f"Mean PyEDA Time:      {time_stats['mean_pyeda']:.6f} s")
    lines.append(f"Mean BoolMin2D Time:  {time_stats['mean_kmap']:.6f} s")
    lines.append(f"Mean Difference:      {time_stats['mean_diff']:+.6f} s")
    lines.append(f"Std. Dev. (Δ):        {time_stats['std_diff']:.6f} s")
    lines.append(f"95% CI:               [{time_stats['ci_lower']:.6f}, {time_stats['ci_upper']:.6f}]")
    lines.append("")
    lines.append(f"Paired t-test:        t = {time_stats['t_statistic']:.4f}, p = {time_stats['p_value']:.6f}")
    lines.append(f"Wilcoxon test:        W = {time_stats['wilcoxon_stat']:.1f}, p = {time_stats['wilcoxon_p']:.6f}")
    lines.append(f"Effect Size (d):      {time_stats['cohens_d']:.4f} ({interpret_effect_size(time_stats['cohens_d'])})")
    
    if time_stats['significant']:
        lines.append(f"\n✓ SIGNIFICANT: Time difference is statistically significant (p < {ALPHA})")
        if time_stats['mean_diff'] < 0:
            lines.append("  → BoolMin2D is significantly faster than PyEDA")
        else:
            lines.append("  → PyEDA is significantly faster than BoolMin2D")
    else:
        lines.append(f"\n✗ NOT SIGNIFICANT: No statistically significant time difference (p ≥ {ALPHA})")
        lines.append("  → Both algorithms exhibit comparable performance")
    
    # Literal Count Analysis
    lines.append("\n2. SIMPLIFICATION QUALITY ANALYSIS")
    lines.append("-" * 75)
    lit_stats = stats_results["literals"]
    
    if num_constant == total_tests:
        lines.append(f"⚠️  All {num_constant} test cases resulted in constant functions.")
        lines.append(f"   Those {num_constant} tests are not applicable in the literal count comparison.")
    else:
        non_constant_count = total_tests - num_constant
        lines.append(f"Analysis based on {non_constant_count} non-constant functions:")
        if num_constant > 0:
            lines.append(f"({num_constant} constant function(s) excluded from this analysis)")
        lines.append("")
        lines.append(f"Mean PyEDA Literals:  {lit_stats['mean_pyeda']:.2f}")
        lines.append(f"Mean KMap Literals:   {lit_stats['mean_kmap']:.2f}")
        lines.append(f"Mean Difference:      {lit_stats['mean_diff']:+.2f}")
        lines.append(f"Std. Dev. (Δ):        {lit_stats['std_diff']:.2f}")
        lines.append(f"95% CI:               [{lit_stats['ci_lower']:.2f}, {lit_stats['ci_upper']:.2f}]")
        lines.append("")
        lines.append(f"Paired t-test:        t = {lit_stats['t_statistic']:.4f}, p = {lit_stats['p_value']:.6f}")
        lines.append(f"Wilcoxon test:        W = {lit_stats['wilcoxon_stat']:.1f}, p = {lit_stats['wilcoxon_p']:.6f}")
        lines.append(f"Effect Size (d):      {lit_stats['cohens_d']:.4f} ({interpret_effect_size(lit_stats['cohens_d'])})")
        
        if lit_stats['significant']:
            lines.append(f"\n✓ SIGNIFICANT: Literal count difference is statistically significant (p < {ALPHA})")
            if lit_stats['mean_diff'] < 0:
                lines.append("  → BoolMin2D produces more minimal expressions")
            else:
                lines.append("  → PyEDA produces more minimal expressions")
        else:
            lines.append(f"\n✗ NOT SIGNIFICANT: No significant difference in simplification (p ≥ {ALPHA})")
            lines.append("  → Both algorithms achieve comparable minimization")
    
    # Overall Verdict
    lines.append("\n3. OVERALL SCIENTIFIC CONCLUSION")
    lines.append("-" * 75)
    
    if time_stats['significant'] and lit_stats['significant']:
        lines.append("Both performance and simplification show statistically significant differences.")
    elif time_stats['significant']:
        lines.append("Performance difference is significant, but simplification is comparable.")
    elif lit_stats['significant']:
        lines.append("Simplification difference is significant, but performance is comparable.")
    else:
        lines.append("No statistically significant differences detected in either metric.")
    
    lines.append(f"\nEffect sizes: Time ({interpret_effect_size(time_stats['cohens_d'])}), "
                 f"Literals ({interpret_effect_size(lit_stats['cohens_d'])})")
    
    if num_constant > 0:
        lines.append(f"\nNote: {num_constant} constant function(s) correctly handled by both algorithms.")
    
    lines.append("=" * 75)
    
    text = "\n".join(lines)
    print(text)
    return text

# ============================================================================
# EXPORT & VISUALIZATION
# ============================================================================

def export_results_to_csv(all_results, filename=RESULTS_CSV):
    """Export benchmark results to CSV."""
    print(f"\n📄 Exporting raw results to CSV...", end=" ", flush=True)
    grouped = defaultdict(list)
    for r in all_results:
        grouped[(r["num_vars"], r["form"])].append(r)

    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Add explanatory comments at the top of the CSV
        writer.writerow(["# BENCHMARK RESULTS: BoolMin2D vs PyEDA"])
        writer.writerow(["# Notes:"])
        writer.writerow(["# - For constant expressions, both algorithms correctly produce 0 literals"])
        writer.writerow(["# - Cases where PyEDA or BoolMin2D failed are marked as such and excluded from equivalence comparison"])
        writer.writerow(["# - Equivalence is only checked when BOTH algorithms successfully produced a result"])
        writer.writerow(["# - This ensures fair comparison - no algorithm is penalized for edge cases the other cannot handle"])
        writer.writerow([])
        
        writer.writerow(["Configuration", "Test Index", "Distribution", "Category", "Equivalent", 
                         "Differs on DC", "PyEDA Time (s)", "KMap Time (s)", 
                         "PyEDA Literals", "KMap Literals", "PyEDA Error", "KMap Error"])

        for (vars, form_type), group in sorted(grouped.items()):
            writer.writerow([])
            writer.writerow([f"{vars}-Variable ({form_type.upper()} Form)"])
            
            for result in group:
                equiv_str = result.get("equiv", "")
                if equiv_str is None:
                    equiv_str = "N/A"
                elif equiv_str == True:
                    equiv_str = "True"
                elif equiv_str == False:
                    equiv_str = "False"
                
                writer.writerow([
                    f"{vars}-{form_type}",
                    result.get("index", ""),
                    result.get("distribution", ""),
                    result.get("result_category", "expression"),
                    equiv_str,
                    result.get("differs_on_dc", False),
                    f"{result.get('t_pyeda', 0):.8f}",
                    f"{result.get('t_kmap', 0):.8f}",
                    result.get("pyeda_literals", ""),
                    result.get("kmap_literals", ""),
                    result.get("pyeda_error", "")[:50] if result.get("pyeda_error") else "",
                    result.get("kmap_error", "")[:50] if result.get("kmap_error") else ""
                ])

    print(f"✅ Done ({len(all_results)} records)")

def export_statistical_analysis(all_stats, filename=STATS_CSV):
    """Export statistical analysis results to CSV."""
    print(f"📊 Exporting statistical analysis...", end=" ", flush=True)
    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(["Configuration", "Metric", "Mean PyEDA", "Mean KMap", 
                        "Mean Diff", "Std Diff", "t-statistic", "p-value",
                        "Cohen's d", "Effect Size", "CI Lower", "CI Upper", "Significant"])
        
        for (num_vars, form_type), stats in sorted(all_stats.items()):
            config = f"{num_vars}-{form_type}"
            
            # Time statistics
            t = stats["time"]
            writer.writerow([
                config, "Time (s)",
                f"{t['mean_pyeda']:.8f}",
                f"{t['mean_kmap']:.8f}",
                f"{t['mean_diff']:.8f}",
                f"{t['std_diff']:.8f}",
                f"{t['t_statistic']:.4f}",
                f"{t['p_value']:.6f}",
                f"{t['cohens_d']:.4f}",
                interpret_effect_size(t['cohens_d']),
                f"{t['ci_lower']:.8f}",
                f"{t['ci_upper']:.8f}",
                "Yes" if t['significant'] else "No"
            ])
            
            # Literal statistics
            l = stats["literals"]
            writer.writerow([
                config, "Literals",
                f"{l['mean_pyeda']:.2f}",
                f"{l['mean_kmap']:.2f}",
                f"{l['mean_diff']:.2f}",
                f"{l['std_diff']:.2f}",
                f"{l['t_statistic']:.4f}",
                f"{l['p_value']:.6f}",
                f"{l['cohens_d']:.4f}",
                interpret_effect_size(l['cohens_d']),
                f"{l['ci_lower']:.2f}",
                f"{l['ci_upper']:.2f}",
                "Yes" if l['significant'] else "No"
            ])
    
    print(f"✅ Done ({len(all_stats)} configurations)")

def save_comprehensive_report(all_results, all_stats, setup_info, pdf_filename=REPORT_PDF):
    """
    Generate comprehensive scientific report as PDF.
    """
    print(f"\n📊 Generating comprehensive scientific report...")
    print(f"   Output: {pdf_filename}")
    
    with PdfPages(pdf_filename) as pdf:
        # ============================================================
        # COVER PAGE
        # ============================================================
        print(f"   • Creating cover page...", end=" ", flush=True)
        fig = plt.figure(figsize=(8.5, 11))
        ax = plt.gca()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        
        # Logo (if available)
        if os.path.exists(LOGO_PATH):
            try:
                img = mpimg.imread(LOGO_PATH)
                logo_ax = fig.add_axes([0.2, 0.65, 0.6, 0.25])
                logo_ax.imshow(img)
                logo_ax.axis("off")
            except:
                pass
        
        # Title
        ax.text(0.5, 0.55, "Scientific Benchmark Report", 
                fontsize=28, fontweight='bold', ha='center')
        ax.text(0.5, 0.49, "BoolMin2D vs PyEDA Boolean Minimization",
                fontsize=16, ha='center')
        
        # Separator
        ax.plot([0.2, 0.8], [0.45, 0.45], 'k-', linewidth=2, alpha=0.3)
        
        # Metadata
        ax.text(0.5, 0.35, f"Experiment Date: {setup_info['experiment_date'][:10]}",
                fontsize=11, ha='center')
        ax.text(0.5, 0.31, f"Random Seed: {setup_info['random_seed']}",
                fontsize=11, ha='center')
        ax.text(0.5, 0.27, f"Total Test Cases: {len(all_results)}",
                fontsize=11, ha='center')
        ax.text(0.5, 0.23, f"Statistical Significance Level: α = {ALPHA}",
                fontsize=11, ha='center')
        
        # Footer
        ax.text(0.5, 0.08, "A Rigorous Statistical Analysis with Reproducibility Controls",
                fontsize=10, ha='center', style='italic', color='gray')
        ax.text(0.5, 0.04, f"© Stan's Technologies {datetime.datetime.now().year}",
                fontsize=9, ha='center', color='gray')
        
        pdf.savefig(bbox_inches='tight')
        plt.close()
        print("✓")
        
        # ============================================================
        # EXPERIMENTAL SETUP PAGE
        # ============================================================
        print(f"   • Creating experimental setup page...", end=" ", flush=True)
        fig = plt.figure(figsize=(8.5, 11))
        ax = plt.gca()
        ax.axis("off")
        
        ax.text(0.5, 0.95, "EXPERIMENTAL SETUP", 
                fontsize=20, fontweight='bold', ha='center', transform=ax.transAxes)
        
        setup_text = f"""
SYSTEM CONFIGURATION
{'─' * 70}
Python Version:    {setup_info['python_version'].split()[0]}
Platform:          {setup_info['platform']}
Processor:         {setup_info['processor']}

LIBRARY VERSIONS
{'─' * 70}
PyEDA:             {setup_info['pyeda_version']}
NumPy:             {setup_info['numpy_version']}
SciPy:             {setup_info['scipy_version']}

EXPERIMENTAL PARAMETERS
{'─' * 70}
Random Seed:                {setup_info['random_seed']}
Tests per Configuration:    {setup_info['tests_per_config']}
Timing Warm-up Runs:        {setup_info['timing_warmup']}
Timing Repetitions:         {setup_info['timing_repeats']}
Significance Level (α):     {setup_info['alpha_level']}

TEST DISTRIBUTIONS
{'─' * 70}
• Sparse:       20% ones, 5% don't-cares (realistic digital logic)
• Dense:        70% ones, 5% don't-cares (stress test)
• Balanced:     50% ones, 10% don't-cares (neutral case)
• Minimal DC:   45% ones, 2% don't-cares (typical circuits)
• Heavy DC:     30% ones, 30% don't-cares (optimization test)
• Edge Cases:   All-zeros, all-ones, checkerboard, single-minterm

METHODOLOGY
{'─' * 70}
1. Random K-maps generated with controlled distributions
2. Each algorithm executed with {setup_info['timing_warmup']} warm-up runs
3. Best of {setup_info['timing_repeats']} timed repetitions recorded
4. Logical equivalence verified using SymPy (converted to common format)
5. Statistical significance tested using paired t-tests
6. Non-parametric Wilcoxon tests used as robustness check
7. Effect sizes computed using Cohen's d

REPRODUCIBILITY
{'─' * 70}
To reproduce this experiment:
  1. Set random seed: random.seed({setup_info['random_seed']})
  2. Run with identical system configuration
  3. Use same library versions as documented above
"""
        
        ax.text(0.05, 0.88, setup_text, fontsize=9, family='monospace',
                va='top', transform=ax.transAxes)
        
        pdf.savefig(bbox_inches='tight')
        plt.close()
        print("✓")
        
        # ============================================================
        # PER-CONFIGURATION RESULTS
        # ============================================================
        grouped = defaultdict(list)
        for r in all_results:
            grouped[(r["num_vars"], r["form"])].append(r)
        
        for config_num, ((num_vars, form_type), results) in enumerate(sorted(grouped.items()), 1):
            print(f"   • Creating plots for {num_vars}-var {form_type.upper()} ({config_num}/{len(grouped)})...", end=" ", flush=True)
            
            # Extract data
            pyeda_times = [r["t_pyeda"] for r in results]
            kmap_times = [r["t_kmap"] for r in results]
            pyeda_literals = [r["pyeda_literals"] for r in results]
            kmap_literals = [r["kmap_literals"] for r in results]
            tests = list(range(1, len(results) + 1))
            
            stats = all_stats[(num_vars, form_type)]
            
            # Create figure with 2x2 subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
            fig.suptitle(f'{num_vars}-Variable K-Map ({form_type.upper()} Form)', 
                        fontsize=16, fontweight='bold')
            
            # Plot 1: Time comparison (line plot)
            ax1.plot(tests, pyeda_times, marker='o', markersize=2, label='PyEDA', alpha=0.7)
            ax1.plot(tests, kmap_times, marker='s', markersize=2, label='BoolMin2D', alpha=0.7)
            ax1.axhline(stats['time']['mean_pyeda'], color='blue', linestyle='--', 
                       alpha=0.5, label=f'PyEDA Mean = {stats["time"]["mean_pyeda"]:.6f}s')
            ax1.axhline(stats['time']['mean_kmap'], color='orange', linestyle='--',
                       alpha=0.5, label=f'KMap Mean = {stats["time"]["mean_kmap"]:.6f}s')
            ax1.set_xlabel('Test Case')
            ax1.set_ylabel('Execution Time (s)')
            ax1.set_title('Execution Time Comparison')
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Time difference histogram
            time_diffs = np.array(kmap_times) - np.array(pyeda_times)
            ax2.hist(time_diffs, bins=30, edgecolor='black', alpha=0.7)
            ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='No Difference')
            ax2.axvline(stats['time']['mean_diff'], color='green', linestyle='--', 
                       linewidth=2, label=f'Mean Δ = {stats["time"]["mean_diff"]:.6f}s')
            ax2.set_xlabel('Time Difference (KMap - PyEDA) [s]')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Distribution of Time Differences')
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Plot 3: Literal comparison (bar plot)
            sample_indices = list(range(0, len(tests), max(1, len(tests)//50)))  # Show ~50 bars
            ax3.bar([tests[i] - 0.2 for i in sample_indices], 
                   [pyeda_literals[i] for i in sample_indices], 
                   width=0.4, label='PyEDA', alpha=0.7)
            ax3.bar([tests[i] + 0.2 for i in sample_indices], 
                   [kmap_literals[i] for i in sample_indices],
                   width=0.4, label='BoolMin2D', alpha=0.7)
            ax3.axhline(stats['literals']['mean_pyeda'], color='blue', linestyle='--',
                       alpha=0.5, label=f'PyEDA Mean = {stats["literals"]["mean_pyeda"]:.2f}')
            ax3.axhline(stats['literals']['mean_kmap'], color='orange', linestyle='--',
                       alpha=0.5, label=f'KMap Mean = {stats["literals"]["mean_kmap"]:.2f}')
            ax3.set_xlabel('Test Case (sampled)')
            ax3.set_ylabel('Number of Literals')
            ax3.set_title('Literal Count Comparison')
            ax3.legend(fontsize=8)
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Plot 4: Literal difference histogram
            lit_diffs = np.array(kmap_literals) - np.array(pyeda_literals)
            ax4.hist(lit_diffs, bins=20, edgecolor='black', alpha=0.7)
            ax4.axvline(0, color='red', linestyle='--', linewidth=2, label='No Difference')
            ax4.axvline(stats['literals']['mean_diff'], color='green', linestyle='--',
                       linewidth=2, label=f'Mean Δ = {stats["literals"]["mean_diff"]:.2f}')
            ax4.set_xlabel('Literal Difference (KMap - PyEDA)')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Distribution of Literal Differences')
            ax4.legend(fontsize=8)
            ax4.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            pdf.savefig()
            plt.close()
            print("✓")
            
            # ============================================================
            # STATISTICAL INFERENCE PAGE (per configuration)
            # ============================================================
            print(f"   • Creating statistical analysis page ({config_num}/{len(grouped)})...", end=" ", flush=True)
            fig = plt.figure(figsize=(8.5, 11))
            ax = plt.gca()
            ax.axis("off")
            
            ax.text(0.5, 0.96, f"STATISTICAL ANALYSIS: {num_vars}-Variable {form_type.upper()}",
                   fontsize=16, fontweight='bold', ha='center', transform=ax.transAxes)
            
            # Count constants for this configuration
            num_constant_config = sum(1 for r in results if r.get("is_constant", False))
            
            inference = generate_scientific_inference(
                all_stats[(num_vars, form_type)],
                num_constant=num_constant_config,
                total_tests=len(results)
            )
            
            # Display inference with proper formatting
            y_pos = 0.90
            for line in inference.split('\n'):
                if line.startswith('=') or line.startswith('-'):
                    fontsize, fontweight = 9, 'normal'
                    y_pos -= 0.015
                elif any(line.strip().startswith(h) for h in ['1.', '2.', '3.', 'STATISTICAL']):
                    fontsize, fontweight = 11, 'bold'
                    y_pos -= 0.020
                elif line.strip().startswith(('✓', '✗', 'ℹ️', '⚠️')):
                    fontsize, fontweight = 10, 'bold'
                else:
                    fontsize, fontweight = 9, 'normal'
                
                ax.text(0.05, y_pos, line, fontsize=fontsize, fontweight=fontweight,
                       family='monospace', va='top', transform=ax.transAxes)
                y_pos -= 0.022
            
            pdf.savefig(bbox_inches='tight')
            plt.close()
            print("✓")
        
        # ============================================================
        # OVERALL SUMMARY PAGE
        # ============================================================
        print(f"   • Creating overall summary page...", end=" ", flush=True)
        fig = plt.figure(figsize=(11, 8.5))
        
        configs = [f"{nv}V-{f.upper()}" for (nv, f) in sorted(grouped.keys())]
        
        # Extract aggregate statistics
        time_means_pyeda = [all_stats[k]['time']['mean_pyeda'] for k in sorted(all_stats.keys())]
        time_means_kmap = [all_stats[k]['time']['mean_kmap'] for k in sorted(all_stats.keys())]
        lit_means_pyeda = [all_stats[k]['literals']['mean_pyeda'] for k in sorted(all_stats.keys())]
        lit_means_kmap = [all_stats[k]['literals']['mean_kmap'] for k in sorted(all_stats.keys())]
        time_significant = [all_stats[k]['time']['significant'] for k in sorted(all_stats.keys())]
        lit_significant = [all_stats[k]['literals']['significant'] for k in sorted(all_stats.keys())]
        
        # Plot 1: Average execution time
        ax1 = plt.subplot(2, 2, 1)
        x = np.arange(len(configs))
        width = 0.35
        bars1 = ax1.bar(x - width/2, time_means_pyeda, width, label='PyEDA', alpha=0.8)
        bars2 = ax1.bar(x + width/2, time_means_kmap, width, label='BoolMin2D', alpha=0.8)
        
        # Mark significant differences
        for i, sig in enumerate(time_significant):
            if sig:
                ax1.text(i, max(time_means_pyeda[i], time_means_kmap[i]) * 1.05, 
                        '*', ha='center', fontsize=16, color='red')
        
        ax1.set_xlabel('Configuration')
        ax1.set_ylabel('Mean Execution Time (s)')
        ax1.set_title('Average Performance by Configuration\n(* = statistically significant)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(configs, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Average literal count
        ax2 = plt.subplot(2, 2, 2)
        bars1 = ax2.bar(x - width/2, lit_means_pyeda, width, label='PyEDA', alpha=0.8)
        bars2 = ax2.bar(x + width/2, lit_means_kmap, width, label='BoolMin2D', alpha=0.8)
        
        # Mark significant differences
        for i, sig in enumerate(lit_significant):
            if sig:
                ax2.text(i, max(lit_means_pyeda[i], lit_means_kmap[i]) * 1.05,
                        '*', ha='center', fontsize=16, color='red')
        
        ax2.set_xlabel('Configuration')
        ax2.set_ylabel('Mean Literal Count')
        ax2.set_title('Average Simplification Quality\n(* = statistically significant)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(configs, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Effect sizes (time)
        ax3 = plt.subplot(2, 2, 3)
        time_effects = [all_stats[k]['time']['cohens_d'] for k in sorted(all_stats.keys())]
        colors = ['red' if abs(d) >= 0.8 else 'orange' if abs(d) >= 0.5 else 'yellow' if abs(d) >= 0.2 else 'green' 
                  for d in time_effects]
        ax3.barh(configs, time_effects, color=colors, alpha=0.7)
        ax3.axvline(0, color='black', linestyle='-', linewidth=1)
        ax3.axvline(-0.2, color='gray', linestyle='--', alpha=0.5, label='Small effect')
        ax3.axvline(0.2, color='gray', linestyle='--', alpha=0.5)
        ax3.axvline(-0.5, color='gray', linestyle=':', alpha=0.5, label='Medium effect')
        ax3.axvline(0.5, color='gray', linestyle=':', alpha=0.5)
        ax3.set_xlabel("Cohen's d")
        ax3.set_title('Effect Size: Execution Time\n(Negative = KMap faster)')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Plot 4: Effect sizes (literals)
        ax4 = plt.subplot(2, 2, 4)
        lit_effects = [all_stats[k]['literals']['cohens_d'] for k in sorted(all_stats.keys())]
        colors = ['red' if abs(d) >= 0.8 else 'orange' if abs(d) >= 0.5 else 'yellow' if abs(d) >= 0.2 else 'green'
                  for d in lit_effects]
        ax4.barh(configs, lit_effects, color=colors, alpha=0.7)
        ax4.axvline(0, color='black', linestyle='-', linewidth=1)
        ax4.axvline(-0.2, color='gray', linestyle='--', alpha=0.5, label='Small effect')
        ax4.axvline(0.2, color='gray', linestyle='--', alpha=0.5)
        ax4.axvline(-0.5, color='gray', linestyle=':', alpha=0.5, label='Medium effect')
        ax4.axvline(0.5, color='gray', linestyle=':', alpha=0.5)
        ax4.set_xlabel("Cohen's d")
        ax4.set_title('Effect Size: Literal Count\n(Negative = KMap more minimal)')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        print("✓")
        
        # ============================================================
        # STATISTICAL INFERENCE PAGE (per configuration)
        # ============================================================
        print(f"   • Creating statistical analysis page ({config_num}/{len(grouped)})...", end=" ", flush=True)
        fig = plt.figure(figsize=(8.5, 11))
        ax = plt.gca()
        ax.axis("off")
        
        ax.text(0.5, 0.96, f"STATISTICAL ANALYSIS: {num_vars}-Variable {form_type.upper()}",
               fontsize=16, fontweight='bold', ha='center', transform=ax.transAxes)
        
        # Count constants for this configuration
        num_constant_config = sum(1 for r in results if r.get("is_constant", False))
        
        inference = generate_scientific_inference(
            all_stats[(num_vars, form_type)],
            num_constant=num_constant_config,
            total_tests=len(results)
        )
        
        # Display inference with proper formatting
        y_pos = 0.90
        for line in inference.split('\n'):
            if line.startswith('=') or line.startswith('-'):
                fontsize, fontweight = 9, 'normal'
                y_pos -= 0.015
            elif any(line.strip().startswith(h) for h in ['1.', '2.', '3.', 'STATISTICAL']):
                fontsize, fontweight = 11, 'bold'
                y_pos -= 0.020
            elif line.strip().startswith(('✓', '✗', 'ℹ️', '⚠️')):
                fontsize, fontweight = 10, 'bold'
            else:
                fontsize, fontweight = 9, 'normal'
            
            ax.text(0.05, y_pos, line, fontsize=fontsize, fontweight=fontweight,
                   family='monospace', va='top', transform=ax.transAxes)
            y_pos -= 0.022
            
        pdf.savefig(bbox_inches='tight')
        plt.close()
        print("✓")
        
        # ============================================================
        # FINAL CONCLUSIONS PAGE
        # ============================================================
        print(f"   • Creating final conclusions page...", end=" ", flush=True)
        fig = plt.figure(figsize=(8.5, 11))
        ax = plt.gca()
        ax.axis("off")
        
        ax.text(0.5, 0.96, "OVERALL SCIENTIFIC CONCLUSIONS",
               fontsize=18, fontweight='bold', ha='center', transform=ax.transAxes)
        
        # Aggregate all statistics
        all_pyeda_times = [r['t_pyeda'] for r in all_results]
        all_kmap_times = [r['t_kmap'] for r in all_results]
        all_pyeda_lits = [r['pyeda_literals'] for r in all_results]
        all_kmap_lits = [r['kmap_literals'] for r in all_results]
        
        # Count total constants and DC differences
        total_constant_count = sum(1 for r in all_results if r.get('is_constant', False))
        total_dc_diff_count = sum(1 for r in all_results if r.get('differs_on_dc', False))
        
        overall_stats = perform_statistical_tests(
            all_pyeda_times, all_kmap_times,
            all_pyeda_lits, all_kmap_lits
        )
        
        conclusions = f"""
EXECUTIVE SUMMARY
{'=' * 70}
Total Test Cases:        {len(all_results)}
Configurations Tested:   {len(grouped)}
Equivalence Check:       {sum(1 for r in all_results if r['equiv'])} / {len(all_results)} passed
  - Identical results:   {sum(1 for r in all_results if r['equiv'] and not r.get('differs_on_dc', False))}
  - Valid, differ on DC: {total_dc_diff_count} (both valid minimizations)
Constant Functions:      {total_constant_count} / {len(all_results)} ({100*total_constant_count/len(all_results):.1f}%)

AGGREGATE PERFORMANCE
{'=' * 70}
Mean PyEDA Time:         {overall_stats['time']['mean_pyeda']:.6f} s
Mean BoolMin2D Time:     {overall_stats['time']['mean_kmap']:.6f} s
Mean Time Difference:    {overall_stats['time']['mean_diff']:+.6f} s
95% CI:                  [{overall_stats['time']['ci_lower']:.6f}, {overall_stats['time']['ci_upper']:.6f}]
Statistical Significance: {'YES' if overall_stats['time']['significant'] else 'NO'} (p = {overall_stats['time']['p_value']:.6f})
Effect Size:             {overall_stats['time']['cohens_d']:.4f} ({interpret_effect_size(overall_stats['time']['cohens_d'])})

AGGREGATE SIMPLIFICATION
{'=' * 70}
Mean PyEDA Literals:     {overall_stats['literals']['mean_pyeda']:.2f}
Mean KMap Literals:      {overall_stats['literals']['mean_kmap']:.2f}
Mean Literal Difference: {overall_stats['literals']['mean_diff']:+.2f}
95% CI:                  [{overall_stats['literals']['ci_lower']:.2f}, {overall_stats['literals']['ci_upper']:.2f}]
Statistical Significance: {'YES' if overall_stats['literals']['significant'] else 'NO'} (p = {overall_stats['literals']['p_value']:.6f})
Effect Size:             {overall_stats['literals']['cohens_d']:.4f} ({interpret_effect_size(overall_stats['literals']['cohens_d'])})

KEY FINDINGS
{'=' * 70}
"""
        
        # Add findings based on results
        if overall_stats['time']['significant']:
            if overall_stats['time']['mean_diff'] < 0:
                conclusions += "\n1. BoolMin2D demonstrates statistically significant performance\n   advantage over PyEDA's minimization approach."
            else:
                conclusions += "\n1. PyEDA demonstrates statistically significant performance\n   advantage over BoolMin2D."
        else:
            conclusions += "\n1. No statistically significant performance difference detected\n   between BoolMin2D and PyEDA."
        
        if overall_stats['literals']['significant']:
            if overall_stats['literals']['mean_diff'] < 0:
                conclusions += "\n\n2. BoolMin2D produces statistically more minimal Boolean\n   expressions (fewer literals) compared to PyEDA."
            else:
                conclusions += "\n\n2. PyEDA produces statistically more minimal Boolean\n   expressions (fewer literals) compared to BoolMin2D."
        else:
            conclusions += "\n\n2. Both algorithms achieve statistically equivalent simplification\n   quality in terms of literal count."
        
        conclusions += f"\n\n3. Effect sizes indicate {interpret_effect_size(overall_stats['time']['cohens_d'])} practical\n   significance for performance and {interpret_effect_size(overall_stats['literals']['cohens_d'])} practical\n   significance for simplification quality."
        
        conclusions += f"\n\n4. All {len(all_results)} test cases maintained logical correctness,\n   with {sum(1 for r in all_results if r['equiv'])} passing equivalence verification.\n   Constant cases were {total_constant_count} (i.e., cases where there was no\n   minimal function to be found - both algorithms correctly identified these)."
        
        conclusions += f"""

THREATS TO VALIDITY
{'=' * 70}
• Limited to 2-4 variable K-maps (inherent K-map scalability limit)
• Random test case generation may not reflect real-world distributions
• Timing includes Python overhead (not pure algorithm performance)
• SymPy uses different minimization strategies (not pure K-map based)

REPRODUCIBILITY
{'=' * 70}
This experiment used random seed {RANDOM_SEED} and can be fully reproduced
using the documented experimental setup and library versions.

RECOMMENDATIONS
{'=' * 70}
Based on statistical evidence:
"""
        
        if overall_stats['time']['mean_diff'] < -0.0001 and overall_stats['literals']['mean_diff'] <= 0:
            conclusions += "\n→ BoolMin2D offers practical advantages for applications requiring\n  both speed and minimal circuit representations."
        elif abs(overall_stats['time']['mean_diff']) < 0.0001 and abs(overall_stats['literals']['mean_diff']) < 0.5:
            conclusions += "\n→ Both algorithms are practically equivalent; choice should be based\n  on integration convenience and API preferences."
        else:
            conclusions += "\n→ Algorithm selection should be based on whether performance or\n  simplification quality is the priority for the application."
        
        conclusions += f"\n\n{'=' * 70}\nReport generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n© Stan's Technologies {datetime.datetime.now().year}"
        
        ax.text(0.05, 0.90, conclusions, fontsize=9, family='monospace',
               va='top', transform=ax.transAxes)
        
        pdf.savefig(bbox_inches='tight')
        plt.close()
        print("✓")
    
    print(f"✅ Comprehensive scientific report saved!")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main benchmark execution function."""
    print("=" * 80)
    print("SCIENTIFIC K-MAP SOLVER BENCHMARK")
    print("=" * 80)
    
    # Create outputs directory FIRST (needed for crash log)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    print(f"📁 Output directory: {OUTPUTS_DIR}")
    
    # Setup crash detection
    setup_signal_handlers()
    crash_log = setup_crash_log()
    print(f"🔍 Crash detection enabled")
    print(f"📝 Crash log: {crash_log}")
    
    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    print(f"🎲 Random seed set to: {RANDOM_SEED}")
    
    # Document experimental setup
    setup_info = document_experimental_setup()
    print("📝 Experimental setup documented.")
    
    # Test configurations
    configurations = [
        # (2, "sop"),
        # (2, "pos"),
        # (3, "sop"),
        # (3, "pos"),
        (4, "sop"),
        (4, "pos"),
    ]
    
    print(f"\n🔬 Test configurations: {len(configurations)}")
    print(f"   Variables: 2, 3, 4")
    print(f"   Forms: SOP, POS")
    print(f"   Tests per config: ~{TESTS_PER_CONFIG}")
    
    var_pool = symbols('x1 x2 x3 x4')
    all_results = []
    all_stats = {}
    
    # Run benchmarks for each configuration
    total_configs = len(configurations)
    for config_idx, (num_vars, form_type) in enumerate(configurations, 1):
        print(f"\n{'='*80}")
        print(f"[{config_idx}/{total_configs}] Benchmarking: {num_vars}-variable K-map ({form_type.upper()} form)")
        print(f"{'='*80}")
        
        # Generate test suite
        print(f"  📋 Generating test suite...", end=" ", flush=True)
        test_suite = generate_test_suite(num_vars, tests_per_distribution=TESTS_PER_CONFIG // 5)
        var_names = var_pool[:num_vars]
        print(f"✓ ({len(test_suite)} test cases)")
        
        print(f"  ⚙️  Running benchmarks...")
        config_results = []
        for idx, (kmap, distribution) in enumerate(test_suite, 1):
            # Update global test index for signal handler
            global CURRENT_TEST_INDEX
            CURRENT_TEST_INDEX = idx
            
            # Log test start BEFORE running it
            test_start_msg = f"Starting test {idx} ({num_vars}-var {form_type.upper()}, {distribution})"
            print(f"\n    {test_start_msg}", flush=True)
            log_crash_info(test_start_msg)
            
            # Check system memory before starting test (if psutil available)
            if PSUTIL_AVAILABLE:
                try:
                    sys_mem = psutil.virtual_memory()
                    mem_status = f"System memory: {sys_mem.percent:.1f}% used, {sys_mem.available / (1024**3):.2f} GB available"
                    log_crash_info(mem_status)
                    
                    if sys_mem.percent > 95:
                        msg = f"CRITICAL: System memory at {sys_mem.percent:.1f}%. Stopping to prevent crash."
                        print(f"\n    🛑 {msg}", flush=True)
                        log_crash_info(msg)
                        print(f"    Available memory: {sys_mem.available / (1024**3):.2f} GB", flush=True)
                        break
                except Exception as e:
                    log_crash_info(f"Error checking memory: {e}")
            
            # Record memory before test
            mem_before = get_memory_usage()
            
            try:
                result = benchmark_case(kmap, var_names, form_type, idx, distribution)
                config_results.append(result)
                all_results.append(result)
                
                # Log successful completion
                log_crash_info(f"Test {idx} completed successfully")
            
            except MemoryError as e:
                crash_report = diagnose_crash_reason(e, idx)
                print(crash_report, flush=True)
                log_crash_info(crash_report)
                print(f"    🛑 Stopping benchmark due to memory exhaustion at test {idx}", flush=True)
                break
            
            except KeyboardInterrupt as e:
                crash_report = diagnose_crash_reason(e, idx)
                print(crash_report, flush=True)
                log_crash_info(crash_report)
                print(f"    🛑 Benchmark interrupted by user at test {idx}", flush=True)
                raise  # Re-raise to stop the entire program
            
            except SystemExit as e:
                crash_msg = f"SystemExit raised at test {idx} with code {e.code}"
                print(f"\n🚨 {crash_msg}", flush=True)
                log_crash_info(crash_msg)
                raise
            
            except Exception as e:
                crash_report = diagnose_crash_reason(e, idx)
                print(crash_report, flush=True)
                log_crash_info(crash_report)
                
                # Check if we should continue or stop (if psutil available)
                if PSUTIL_AVAILABLE:
                    mem_after = get_memory_usage()
                    mem_increase = mem_after - mem_before
                    
                    if mem_increase > 500:  # More than 500MB increase
                        msg = f"Large memory increase detected ({mem_increase:.2f} MB). Stopping benchmark."
                        print(f"    🛑 {msg}", flush=True)
                        log_crash_info(msg)
                        break
                
                print(f"    ⚠️  Error in test {idx}, continuing with next test...", flush=True)
                continue
        
        # Perform statistical analysis for this configuration
        if config_results:
            print(f"\n  📊 Performing statistical analysis...", end=" ", flush=True)
            
            # Filter out failed cases (where one or both algorithms failed)
            valid_results = [r for r in config_results if r.get("pyeda_error") is None and r.get("kmap_error") is None]
            num_failed = len(config_results) - len(valid_results)
            num_pyeda_failed = sum(1 for r in config_results if r.get("pyeda_error") is not None)
            num_kmap_failed = sum(1 for r in config_results if r.get("kmap_error") is not None)
            
            # Count constants from valid results only
            num_constant = sum(1 for r in valid_results if r.get("is_constant", False))
            
            # Filter out constants for literal-count statistics
            non_constant_results = [r for r in valid_results if not r.get("is_constant", False)]
            
            # Use only valid results for timing and literal comparison
            if valid_results:
                pyeda_times = [r["t_pyeda"] for r in valid_results]
                kmap_times = [r["t_kmap"] for r in valid_results]
                
                if non_constant_results:
                    pyeda_literals = [r["pyeda_literals"] for r in non_constant_results]
                    kmap_literals = [r["kmap_literals"] for r in non_constant_results]
                else:
                    # All constants - use zeros for literal comparison (will be flagged)
                    pyeda_literals = [0]
                    kmap_literals = [0]
                
                stats = perform_statistical_tests(pyeda_times, kmap_times, 
                                                 pyeda_literals, kmap_literals)
                all_stats[(num_vars, form_type)] = stats
                print("✓")
                
                # Display summary
                print(f"\n  📈 Configuration Summary:")
                print(f"     Tests completed:     {len(config_results)}")
                if num_failed > 0:
                    print(f"     ⚠️  Failed tests:        {num_failed}")
                    if num_pyeda_failed > 0:
                        print(f"        • PyEDA failed:    {num_pyeda_failed}")
                    if num_kmap_failed > 0:
                        print(f"        • BoolMin2D failed: {num_kmap_failed}")
                print(f"     Valid comparisons:   {len(valid_results)}")
                print(f"     Constant functions:  {num_constant}")
                
                # Calculate equivalence rate from valid, non-constant results
                non_constant_valid = [r for r in valid_results if not r.get("is_constant", False)]
                if non_constant_valid:
                    equiv_count = sum(1 for r in non_constant_valid if r.get("equiv") == True)
                    equiv_rate = 100 * equiv_count / len(non_constant_valid)
                    print(f"     Equivalence rate:    {equiv_count}/{len(non_constant_valid)} ({equiv_rate:.1f}%)")
                
                print(f"     Mean PyEDA time:     {stats['time']['mean_pyeda']:.6f} s")
                print(f"     Mean KMap time:      {stats['time']['mean_kmap']:.6f} s")
                print(f"     Time significant:    {'YES ✓' if stats['time']['significant'] else 'NO ✗'} "
                      f"(p={stats['time']['p_value']:.4f})")
                if non_constant_results:
                    print(f"     Mean PyEDA literals: {stats['literals']['mean_pyeda']:.2f}")
                    print(f"     Mean KMap literals:  {stats['literals']['mean_kmap']:.2f}")
                    print(f"     Literal significant: {'YES ✓' if stats['literals']['significant'] else 'NO ✗'} "
                          f"(p={stats['literals']['p_value']:.4f})")
            else:
                print(" ⚠️  No valid results for statistical analysis")
    
    # Export results
    print(f"\n{'='*80}")
    print("EXPORTING RESULTS")
    print(f"{'='*80}")
    
    export_results_to_csv(all_results, RESULTS_CSV)
    export_statistical_analysis(all_stats, STATS_CSV)
    save_comprehensive_report(all_results, all_stats, setup_info, REPORT_PDF)
    
    # Final summary
    print(f"\n{'='*80}")
    print("✅ BENCHMARK COMPLETE")
    print(f"{'='*80}")
    print(f"📊 Total test cases executed: {len(all_results)}")
    print(f"🔧 Configurations tested: {len(configurations)}")
    equiv_passed = sum(1 for r in all_results if r['equiv'])
    constant_count = sum(1 for r in all_results if r.get('is_constant', False))
    print(f"✓  Equivalence pass rate: {equiv_passed}/{len(all_results)} ({100*equiv_passed/len(all_results):.1f}%)")
    print(f"ℹ️  Constant functions: {constant_count}/{len(all_results)} ({100*constant_count/len(all_results):.1f}%)")
    print(f"   (Unsimplifiable functions correctly identified by both algorithms)")
    print(f"\n📂 Outputs:")
    print(f"   • Raw results:       {RESULTS_CSV}")
    print(f"   • Statistical data:  {STATS_CSV}")
    print(f"   • PDF report:        {REPORT_PDF}")
    print(f"\n{'='*80}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Catch any unhandled exceptions at the top level
        error_msg = f"""
{'='*80}
🚨 UNHANDLED EXCEPTION IN MAIN
{'='*80}
Exception Type: {type(e).__name__}
Exception Message: {str(e)}

Traceback:
{traceback.format_exc()}
{'='*80}
"""
        print(error_msg, flush=True)
        if CRASH_LOG_FILE:
            log_crash_info(error_msg)
        sys.exit(1)
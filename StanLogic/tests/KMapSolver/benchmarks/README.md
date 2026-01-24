# Benchmarks Directory

## Overview

This directory contains comprehensive benchmark suites for evaluating StanLogic's Boolean minimization algorithms against established computational algebra systems (SymPy, PyEDA). The benchmarks employ rigorous statistical methodologies, reproducibility controls, and multi-dimensional performance characterization.

## Benchmark Scripts

### benchmark_test2D_sympy.py
**2-4 Variable Comparative Benchmark: BoolMin2D vs SymPy**

Rigorous experimental comparison of BoolMin2D against SymPy's SOPform minimization for 2-4 variable Boolean functions.

**Features:**
- Comprehensive statistical analysis with significance testing
- Multiple distribution pattern testing (sparse, dense, balanced)
- Reproducible results via fixed random seeds
- Performance metrics: execution time, memory consumption, solution quality
- PDF report generation with visualizations
- CSV output for raw data and statistical analysis

**Test Distributions:**
- Sparse (20% ones): Minimal TRUE outputs
- Dense (70% ones): Predominantly TRUE outputs  
- Balanced (50% ones): Equal TRUE/FALSE distribution
- Minimal don't-care (2%): Few optimization opportunities
- Heavy don't-care (30%): Extensive optimization potential

**Metrics Collected:**
- Execution time (microseconds)
- Memory consumption (MB)
- Literal count in minimized expression
- Term count in minimized expression
- Solution quality comparison

**Outputs:**
- `../outputs/benchmark_results2D_sympy/benchmark_results2D.csv` - Raw performance data
- `../outputs/benchmark_results2D_sympy/statistical_analysis2D.csv` - Statistical summaries
- `../outputs/benchmark_results2D_sympy/benchmark_scientific_report2D.pdf` - Comprehensive report

### benchmark_test2D_pyeda.py
**2-4 Variable Comparative Benchmark: BoolMin2D vs PyEDA**

Comparative performance analysis of BoolMin2D against PyEDA's minimization algorithms.

**Features:**
- PyEDA library integration
- Alternative minimization method comparison
- Distribution-specific performance analysis
- Statistical validation and reporting

**Outputs:**
- `../outputs/benchmark_results2D_pyeda/` - Raw and analyzed results

### benchmark_test3D_kmapsolver.py
**5-8 Variable Performance Characterization**

Internal performance characterization of BoolMinGeo's hierarchical minimization for 5-8 variable functions without external library comparisons.

**Focus:**
- Algorithm-specific performance profiling
- Hierarchical decomposition efficiency
- Geometric minimization effectiveness
- Scaling behavior analysis

**Outputs:**
- Performance metrics for BoolMinGeo standalone operation
- Efficiency characterization data

### benchmark_test3D_sympy.py
**5-8 Variable Comparative Benchmark: BoolMinGeo vs SymPy**

Comparative analysis of BoolMinGeo's hierarchical minimization against SymPy for medium-complexity Boolean functions.

**Significance:**
- Demonstrates BoolMinGeo advantages at moderate scales
- Characterizes SymPy limitations beyond 4 variables
- Establishes performance boundaries for both methods

**Outputs:**
- `../outputs/benchmark_results3D_sympy/` - Comparative benchmark results

### benchmark_test3D_pyeda.py
**5-8 Variable Comparative Benchmark: BoolMinGeo vs PyEDA**

Performance comparison of BoolMinGeo against PyEDA for 5-8 variable Boolean functions.

**Features:**
- PyEDA comparative analysis at medium scales
- Distribution sensitivity evaluation
- Multi-algorithm performance profiling

**Outputs:**
- `../outputs/benchmark_results3D_pyeda/` - Comparative results

### benchmark_test3D_pyeda2.py
**5-8 Variable Alternative PyEDA Benchmark**

Alternative comparative benchmark implementation for BoolMinGeo vs PyEDA with different test configurations or methodologies.

**Purpose:**
- Alternative test approach validation
- Additional PyEDA comparison data points
- Methodological verification

**Outputs:**
- `../outputs/benchmark_results3D_pyeda2/` - Alternative comparative results

### benchmark_test4D_pyeda.py
**9-10 Variable Comparative Benchmark: BoolMinGeo vs PyEDA**

Performance characterization at the boundary of practical geometric minimization methods.

**Research Focus:**
- High-complexity function handling
- PyEDA comparison at extended scales
- Performance boundary identification
- Method transition threshold analysis

**Note:** SymPy comparison excluded due to computational infeasibility (see `../analysis/verify_sympy_10var_failure.py`)

**Outputs:**
- `../outputs/benchmark_results4D_pyeda/` - Performance comparison data

### benchmark_test_hierarchical_9to16.py
**9-16 Variable Performance Characterization: Hierarchical Methods**

Comprehensive performance study of BoolMinGeo's hierarchical minimization across extended variable ranges. This is a performance-only study without external library comparisons.

**Key Features:**
- Wide variable range coverage (9-16 variables)
- Exponential growth modeling
- Scalability projections
- Practical performance limit identification
- Distribution sensitivity across extended scales

**Test Parameters:**
- Variable counts: 9, 10, 11, 12, 13, 14, 15, 16
- Multiple test iterations per configuration
- Memory profiling and time measurements
- Solution quality metrics

**Statistical Analysis:**
- Descriptive statistics (mean, median, std dev)
- Performance scaling characterization
- Exponential model fitting
- Practical boundary recommendations

**Outputs:**
- `../outputs/benchmark_hierarchical_9to16/BoolMinGeo_hierarchical.csv` - Raw performance data
- `../outputs/benchmark_hierarchical_9to16/BoolMinGeo_hierarchical.pdf` - Performance report
- `../outputs/benchmark_hierarchical_9to16/kmapsolver_hierarchical_statistical_analysis.csv` - Statistical analysis

## Benchmark Execution

### Basic Execution

Run individual benchmarks:

```bash
cd StanLogic/tests/KMapSolver/benchmarks

# 2-4 variable benchmarks
python benchmark_test2D_sympy.py
python benchmark_test2D_pyeda.py

# 5-8 variable benchmarks
python benchmark_test3D_kmapsolver.py
python benchmark_test3D_sympy.py
python benchmark_test3D_pyeda.py
python benchmark_test3D_pyeda2.py

# 9-10 variable benchmark
python benchmark_test4D_pyeda.py

# 9-16 variable hierarchical benchmark
python benchmark_test_hierarchical_9to16.py
```

### Execution Time Estimates

- 2-4 variable benchmarks: 5-15 minutes
- 5-8 variable benchmarks: 15-45 minutes
- 9-10 variable benchmark: 30-90 minutes
- 9-16 variable benchmark: 1-4 hours

**Note:** Times vary significantly based on hardware, test configuration, and number of iterations.

## Configuration Parameters

### Reproducibility Controls

All benchmarks use fixed random seeds (typically `RANDOM_SEED = 42`) to ensure reproducible results across runs and platforms.

### Test Iteration Counts

Benchmark scripts define iteration counts via constants:
- `TESTS_PER_CONFIG` or `TESTS_PER_DISTRIBUTION`: Number of test cases per configuration
- `TIMING_REPEATS`: Number of timing repetitions for precision
- `TIMING_WARMUP`: Warm-up iterations before measurement

### Statistical Parameters

- `ALPHA = 0.05`: Statistical significance threshold (95% confidence)
- Effect size calculations (Cohen's d)
- Distribution normality testing

## Output Structure

### CSV Files

**Raw Data Format:**
- Test case identifiers
- Configuration parameters (num_vars, distribution)
- Performance metrics (time, memory)
- Solution quality metrics (literals, terms)

**Statistical Analysis Format:**
- Configuration groupings
- Descriptive statistics (mean, std dev, quartiles)
- Comparative metrics (t-statistics, p-values, effect sizes)

### PDF Reports

**Standard Report Sections:**
1. **Cover Page**: Study identification and metadata
2. **Experimental Setup**: Configuration, methodology, reproducibility
3. **Performance Analysis**: Distribution-specific results
4. **Statistical Validation**: Significance testing, effect sizes
5. **Scalability Analysis**: Performance scaling across variables
6. **Conclusions**: Key findings and practical recommendations

## Dependencies

### Required Libraries

**Core:**
- `stanlogic` - StanLogic Boolean minimization library
- `sympy` - Symbolic mathematics (for SymPy benchmarks)
- `pyeda` - Python Electronic Design Automation (for PyEDA benchmarks)

**Scientific Computing:**
- `numpy` - Numerical operations
- `scipy` - Statistical analysis
- `matplotlib` - Visualization and plotting

**System:**
- `psutil` - System resource monitoring
- `tracemalloc` - Memory profiling

**Utilities:**
- `tabulate` - Table formatting
- Standard library: `csv`, `time`, `random`, `platform`, `datetime`

Install dependencies:
```bash
pip install stanlogic sympy pyeda numpy scipy matplotlib psutil tabulate
```

## Benchmark Methodology

### Experimental Protocol

1. **Initialization**: Set random seeds, create output directories
2. **Test Case Generation**: Generate truth tables with specified distributions
3. **Warm-up Phase**: Execute warm-up iterations (discarded)
4. **Measurement Phase**: Timed execution with memory profiling
5. **Statistical Analysis**: Aggregate results, compute statistics
6. **Visualization**: Generate plots and charts
7. **Report Generation**: Create PDF reports with findings

### Performance Measurement

**Timing:**
- Multiple repetitions for precision
- Warm-up iterations to stabilize cache effects
- Microsecond precision via `time.perf_counter()`

**Memory:**
- `tracemalloc` for detailed memory profiling
- Peak memory consumption tracking
- Memory-per-operation metrics

**Solution Quality:**
- Literal count in minimized expressions
- Term count in minimized expressions
- Correctness verification (some benchmarks)

## Research Applications

Benchmark results support:
- Algorithm validation and verification
- Competitive performance analysis
- Publication-quality performance data
- Algorithm optimization prioritization
- User guidance on method selection
- Theoretical complexity validation

## Troubleshooting

### Common Issues

**SymPy Hangs/Crashes:**
- Expected behavior for 10+ variables
- See `../analysis/verify_sympy_10var_failure.py` for justification
- Use PyEDA or performance-only benchmarks for large functions

**Memory Errors:**
- Reduce `TESTS_PER_CONFIG` parameter
- Close other applications
- Consider smaller variable ranges

**Import Errors:**
- Ensure StanLogic is installed: `pip install -e .` from project root
- Verify all dependencies installed
- Check Python path configuration

## Benchmark Output Files

Benchmark results are automatically saved to `../outputs/` in subdirectories organized by benchmark type:
- `benchmark_results2D_sympy/`
- `benchmark_results2D_pyeda/`
- `benchmark_results3D_sympy/`
- `benchmark_results3D_pyeda/`
- `benchmark_results4D_pyeda/`
- `benchmark_hierarchical_9to16/`

See `../outputs/README.md` for detailed output file descriptions.

## Version Control Considerations

**Include in Repository:**
- Small CSV files (< 1 MB)
- PDF reports (compressed)
- Statistical summaries

**Exclude from Repository:**
- Large raw data files (> 1 MB)
- Intermediate computation files
- Temporary benchmark outputs

See `.gitignore` for specific exclusions.

## Performance Optimization

To reduce benchmark execution time:
1. Decrease `TESTS_PER_CONFIG` parameter
2. Reduce `TIMING_REPEATS` count
3. Limit variable range tested
4. Skip specific distribution patterns
5. Run subsets of benchmarks

**Warning:** Reducing test iterations may impact statistical validity and result reproducibility.

## Historical Context

The `benchmark_output.txt` file contains legacy benchmark results for reference. Current benchmarks supersede this data with more rigorous statistical methodology.

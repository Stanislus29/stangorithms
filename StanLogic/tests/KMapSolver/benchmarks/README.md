# Benchmarks Directory

## Overview

This directory contains performance benchmarking tests for the StanLogic Boolean minimization algorithms. The benchmarks provide empirical analysis of execution time, memory consumption, and solution quality across different variable ranges and distribution patterns.

## Benchmark Files

### benchmark_test2D.py
**Scope:** 2-4 variable Boolean functions  
**Algorithm:** BoolMin2D (classical 2D Karnaugh map approach)  
**Comparison:** Performance benchmarked against SymPy's SOPform minimization  
**Metrics:**
- Execution time per test case
- Memory consumption
- Literal count in minimized expressions
- Statistical analysis with paired t-tests and effect sizes

**Output Location:** `../outputs/benchmark_results2D/`

**Key Features:**
- Tests five distribution types: sparse (20% ones), dense (70% ones), balanced (50% ones), minimal don't-care (2%), and heavy don't-care (30%)
- Includes edge case testing (all-zeros, all-ones, all-don't-cares)
- Generates comprehensive PDF scientific reports with statistical validation
- Exports raw data CSV and statistical analysis CSV

### benchmark_test3D.py
**Scope:** 5-8 variable Boolean functions  
**Algorithm:** BoolMinGeo (3D geometric minimization using minimize_3d method)  
**Comparison:** Performance benchmarked against SymPy's SOPform minimization  
**Metrics:**
- Execution time scaling
- Memory usage patterns
- Solution quality (literal and term counts)
- Statistical significance testing

**Output Location:** `../outputs/benchmark_results3D/`

**Key Features:**
- Evaluates 3D K-map visualization approach for medium-complexity functions
- Comprehensive distribution sensitivity analysis
- Scalability projections and exponential growth modeling
- Statistical analysis with confidence intervals and Cohen's d effect sizes

### benchmark_test4D.py
**Scope:** 9-10 variable Boolean functions  
**Algorithm:** BoolMinGeo (4D minimization using minimize_4d method)  
**Comparison:** Performance-only study (no SymPy comparison due to computational infeasibility)  
**Metrics:**
- Execution time characterization
- Memory consumption analysis
- Solution complexity metrics
- Performance efficiency (time per truth table entry)

**Output Location:** `../outputs/benchmark_results4D/`

**Key Features:**
- Tests 4D extension of geometric minimization
- Descriptive statistical analysis (no comparative benchmarking)
- Evaluates performance across five distribution patterns
- Exports statistical summaries for research paper data extraction

### benchmark_test_hierarchical_9to16.py
**Scope:** 9-16 variable Boolean functions  
**Algorithm:** BoolMinGeo hierarchical minimization  
**Comparison:** Performance-only study (no SymPy comparison)  
**Metrics:**
- Time complexity modeling (exponential growth patterns)
- Space complexity analysis
- Solution quality assessment
- Scalability projections

**Output Location:** `../outputs/benchmark_hierarchical_9to16/`

**Key Features:**
- Comprehensive variable range testing (9 through 16 variables)
- Exponential model fitting for time and memory scaling
- Practical application limit identification
- Distribution sensitivity characterization across wide variable range

## Benchmark Methodology

All benchmarks follow a consistent experimental protocol:

1. **Test Generation:** Random Boolean functions generated according to specified probability distributions
2. **Execution Measurement:** High-precision timing using `time.perf_counter()`
3. **Memory Tracking:** Memory profiling with `tracemalloc` and `psutil`
4. **Solution Analysis:** Literal and term counting in minimized expressions
5. **Statistical Analysis:** Paired statistical tests, effect size calculations, and confidence intervals (where applicable)
6. **Reproducibility:** Fixed random seeds and documented system configurations

## Output Structure

Each benchmark produces three primary outputs:

1. **Raw Data CSV:** Complete test results with all measured metrics
2. **Statistical Analysis CSV:** Aggregated statistics for research and publication
3. **Scientific Report PDF:** Comprehensive visualization and analysis document

Statistical analysis files include:
- Descriptive statistics (mean, standard deviation, quartiles)
- Comparative statistics (t-tests, p-values, effect sizes) where applicable
- Coefficient of variation for variability assessment

## Distribution Types

All benchmarks test the following distribution patterns:

- **Sparse (20% ones):** Functions with predominantly zero outputs
- **Dense (70% ones):** Functions with predominantly one outputs  
- **Balanced (50% ones):** Equal distribution of zeros and ones
- **Minimal DC (2%):** Few don't-care conditions
- **Heavy DC (30%):** Significant don't-care conditions for optimization

Edge cases (all-zeros, all-ones, all-don't-cares) are included to validate correctness.

## Usage

To run a specific benchmark:

```bash
cd StanLogic/tests/KMapSolver/benchmarks
python benchmark_test2D.py  # or benchmark_test3D.py, etc.
```

Ensure the Python environment has all required dependencies installed:
- numpy
- scipy
- matplotlib
- psutil

Results will be automatically saved to the corresponding output directory.

## Research Applications

These benchmarks support:
- Algorithm performance characterization studies
- Comparative analysis against established tools (SymPy)
- Scalability assessment for educational and research applications
- Publication-ready statistical data extraction
- Identification of practical computational limits

## Citation

When using benchmark data in research publications, please cite the StanLogic package and reference the specific benchmark methodology employed.

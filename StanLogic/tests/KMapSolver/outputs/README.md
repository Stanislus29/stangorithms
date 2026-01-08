# Outputs Directory

## Overview

This directory serves as the centralized repository for all test results, benchmark data, performance analyses, and generated reports from the StanLogic test suite. Output files are organized by test category to maintain clarity and facilitate data retrieval.

## Directory Structure

### benchmark_results2D/
**2-4 Variable Benchmark Outputs**

Contains performance benchmarking results for BoolMin2D algorithm testing.

**Typical Contents:**
- `benchmark_results2D.csv` - Raw performance data for all 2-4 variable test cases
- `statistical_analysis2D.csv` - Aggregated statistics with comparative metrics vs SymPy
- `benchmark_scientific_report2D.pdf` - Comprehensive visualization and analysis report
- `scalability_analysis.png` - Performance scaling visualizations (if generated separately)

**Data Characteristics:**
- Comparative benchmarks against SymPy's SOPform
- Statistical significance testing (t-tests, effect sizes)
- Distribution-specific performance characterization
- Time, memory, and solution quality metrics

### benchmark_results3D/
**5-8 Variable Benchmark Outputs**

Contains performance benchmarking results for BoolMinGeo's 3D minimization method.

**Typical Contents:**
- `benchmark_results3D.csv` - Raw performance data for 3D minimization tests
- `benchmark_results3D_statistical_analysis.csv` - Statistical analysis with SymPy comparison
- `benchmark_results3D.pdf` - Scientific report with comprehensive visualizations
- `3D_decay.csv` - Decay analysis data (9-12 variables)
- `3D_decay.pdf` - Decay study report documenting performance degradation
- `scalability_analysis.png` - Scaling behavior visualizations

**Data Characteristics:**
- 3D geometric minimization performance metrics
- Comparative analysis against SymPy (5-8 variables)
- Decay characterization beyond 8-variable threshold
- Exponential growth modeling and projections

### benchmark_results4D/
**9-10 Variable Benchmark Outputs**

Contains performance characterization results for BoolMinGeo's 4D minimization extension.

**Typical Contents:**
- `benchmark_results4D.csv` - Raw performance data for 4D minimization
- `benchmark_results4D_statistical_analysis.csv` - Descriptive statistical analysis
- `benchmark_results4D_performance_report.pdf` - Performance characterization report
- `4D_decay.csv` - Decay analysis data (11-13 variables)
- `4D_decay.pdf` - Decay study report for 4D method limitations

**Data Characteristics:**
- Performance-only study (no SymPy comparison due to infeasibility)
- Descriptive statistics for algorithm characterization
- 4D method decay analysis beyond 10-variable threshold
- Efficiency metrics and practical boundary identification

### benchmark_hierarchical_9to16/
**9-16 Variable Hierarchical Benchmark Outputs**

Contains results from comprehensive testing of BoolMinGeo's hierarchical minimization across extended variable ranges.

**Typical Contents:**
- `BoolMinGeo_hierarchical.csv` - Raw performance data (9-16 variables)
- `BoolMinGeo_hierarchical.pdf` - Comprehensive performance report
- `kmapsolver_hierarchical_statistical_analysis.csv` - Statistical characterization
- Exponential model parameters and scalability projections

**Data Characteristics:**
- Wide variable range coverage (9 through 16 variables)
- Performance scaling analysis
- Practical application limit identification
- Distribution sensitivity across extended scales

### obtain_terms_frequency/
**Term Distribution Analysis Outputs**

Contains results from balanced distribution term count studies.

**Typical Contents:**
- `balanced_Nvar_terms_results.csv` - Term count data for N-variable functions
- `balanced_Nvar_terms_stats.txt` - Statistical summary (mean, std dev, quartiles)
- `balanced_Nvar_terms_histogram.png` - Visual distribution representations
- Comparative analyses across different variable counts

**Research Value:**
- Solution complexity characterization
- Distribution pattern analysis
- Theoretical validation data
- Educational visualization materials

## File Naming Conventions

**Standard Patterns:**
- `benchmark_results[N]D.csv` - Raw benchmark data for N-dimensional methods
- `[test_name]_statistical_analysis.csv` - Aggregated statistical metrics
- `[test_name].pdf` - Comprehensive PDF reports with visualizations
- `[N]D_decay.csv/pdf` - Decay analysis results for N-dimensional methods

**Timestamp Suffixes:**
Some scripts may append timestamps to prevent overwriting:
- `results_20260107_143022.csv` - ISO format date-time stamps
- Facilitates tracking multiple test runs
- Enables historical performance comparison

## Data Retention Guidelines

**Version Control:**
- Small CSV files (< 1 MB): Included in repository
- Large CSV files (> 1 MB): Excluded via `.gitignore`
- PDF reports: Typically included (compressed format)
- PNG visualizations: Included for documentation

**Local Storage:**
- All outputs retained locally for analysis
- Large files archived separately if needed
- Backup important benchmark results before rerunning tests

## Data Format Specifications

### CSV Files

**Raw Benchmark Data:**
Columns typically include:
- `test_num` - Sequential test identifier
- `num_vars` - Number of Boolean variables
- `distribution` - Test distribution type (sparse, dense, balanced, etc.)
- `time_s` - Execution time in seconds
- `memory_mb` - Memory consumption in megabytes
- `num_literals` - Literal count in minimized expression
- `num_terms` - Term count in minimized expression
- Additional metrics as appropriate

**Statistical Analysis:**
Columns typically include:
- `Configuration` - Test configuration identifier
- `Metric` - Measured quantity (Time, Memory, Literals)
- `Mean`, `Std Dev`, `Min`, `Max` - Descriptive statistics
- `Quartiles` (Q1, Median, Q3) - Distribution characteristics
- Comparative metrics (if applicable): t-statistic, p-value, Cohen's d

### PDF Reports

**Standard Sections:**
1. Cover page with study identification
2. Experimental setup documentation
3. Distribution performance analysis
4. Scalability analysis and projections
5. Statistical conclusions and practical recommendations

## Usage Recommendations

**For Researchers:**
- Consult statistical analysis CSV files for publication data
- Reference PDF reports for methodology and visualization
- Use raw data files for custom analysis and replication

**For Developers:**
- Review benchmark results before algorithm modifications
- Compare outputs after changes to detect regressions
- Use data to inform optimization priorities

**For Educators:**
- Utilize PDF reports for teaching materials
- Extract visualizations for presentations
- Reference statistical analyses for curriculum development

## Output Regeneration

To regenerate outputs:
1. Navigate to appropriate test directory
2. Execute relevant test script
3. Wait for completion (duration varies by test)
4. Check outputs directory for new/updated files

**Warning:** Rerunning tests may overwrite existing results. Back up important data before regeneration.

## Disk Space Considerations

**Typical Storage Requirements:**
- 2D benchmarks: < 10 MB
- 3D benchmarks: 10-50 MB
- 4D benchmarks: 50-200 MB
- Hierarchical tests: 100-500 MB
- 24-bit tests: 1-10 GB (large CSV files)

Monitor available disk space when running extensive test suites.

## Data Interpretation Resources

For detailed interpretation guidance:
- Consult individual test directory README files
- Reference benchmark methodology documentation
- Review statistical analysis headers in CSV files
- Examine PDF report conclusions sections

Each output type is designed to be self-documenting with clear labeling and metric definitions.

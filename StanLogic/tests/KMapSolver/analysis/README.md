# Analysis Directory

## Overview

This directory contains specialized analysis tools for validating Boolean minimization algorithms, characterizing solution distributions, and verifying correctness against established computational algebra systems.

## Analysis Scripts

### check_boolean_equivalence.py
**Boolean Expression Verification**

Validates logical equivalence between original Boolean functions and their minimized expressions. Ensures that minimization preserves functional correctness while reducing complexity.

**Methodology:**
- Truth table generation for both original and minimized expressions
- Bitwise comparison across all input combinations
- Don't-care condition handling validation
- Exhaustive verification for all test cases

**Applications:**
- Correctness validation for minimization algorithms
- Regression testing after algorithm modifications
- Quality assurance for production minimization
- Verification of edge case handling

### verify_sympy_10var_failure.py
**Computational Feasibility Analysis**

Documents and verifies the computational infeasibility of SymPy's SOPform minimization for 10+ variable Boolean functions. Provides empirical justification for excluding SymPy comparisons in large-variable benchmarks.

**Findings:**
- Execution time characteristics beyond practical limits
- Memory consumption exceeding typical system resources
- Comparative analysis demonstrating BoolMinGeo superiority at scale
- Threshold identification for SymPy applicability

**Research Value:**
- Justifies benchmark methodology decisions
- Establishes performance context for StanLogic algorithms
- Provides reproducible evidence of competitive advantage
- Informs practical application boundaries

### balanced_4var_terms_test.py
**4-Variable Term Distribution Analysis**

Characterizes the distribution of product terms in minimized SOP expressions for 4-variable Boolean functions with balanced output distributions (50% ones, 50% zeros).

**Analysis Metrics:**
- Term count frequency distributions
- Statistical descriptors (mean, median, standard deviation)
- Histogram generation for visual analysis
- Comparison against theoretical expectations

**Outputs:**
- CSV files with detailed term counts per test case
- Statistical summary text files
- Histogram visualizations (PNG format)

### balanced_8var_terms_test.py
**8-Variable Term Distribution Analysis**

Extends distribution analysis to 8-variable Boolean functions. Evaluates how solution complexity scales with increased problem dimensionality under balanced output conditions.

**Focus Areas:**
- Term count scaling behavior (4 to 8 variables)
- Distribution shape evolution with dimensionality
- Statistical moment characterization
- Outlier identification and analysis

**Research Applications:**
- Understanding complexity growth patterns
- Validating algorithm behavior at medium scale
- Informing heuristic optimization strategies
- Educational visualization of K-map limitations

### balanced_10var_terms_test.py
**10-Variable Term Distribution Analysis**

Characterizes term distributions at the boundary of practical K-map visualization. Provides insight into algorithm behavior when transitioning to hierarchical minimization methods.

**Significance:**
- Represents threshold between 3D and hierarchical approaches
- Demonstrates computational challenges at large scales
- Validates correctness of hierarchical decomposition
- Establishes baseline for higher-dimensional analysis

**Data Generated:**
- Comprehensive term count datasets
- Statistical analysis across multiple test iterations
- Visual representations of distribution patterns
- Comparative metrics against lower-dimensional cases

## Common Analysis Workflow

1. **Test Case Generation:** Random or systematic Boolean function creation
2. **Minimization Execution:** StanLogic algorithm application
3. **Metric Collection:** Solution characteristic measurement
4. **Statistical Analysis:** Distribution characterization and hypothesis testing
5. **Visualization:** Histogram and plot generation
6. **Result Interpretation:** Findings documentation and analysis

## Output Locations

Analysis results are typically saved to:
- `../outputs/obtain_terms_frequency/` for term distribution studies
- Individual analysis script directories (when applicable)
- Timestamped result files to prevent overwriting

## Usage

Execute individual analysis scripts:

```bash
cd StanLogic/tests/KMapSolver/analysis
python check_boolean_equivalence.py
python balanced_4var_terms_test.py
# etc.
```

Each script is self-contained and documents its specific configuration and output format.

## Research Applications

Analysis tools support:
- Algorithm validation and verification
- Performance characterization studies
- Solution quality assessment
- Comparative analysis against established tools
- Educational material development
- Publication-ready data generation

## Dependencies

Standard scientific Python stack:
- numpy (numerical operations)
- matplotlib (visualization)
- scipy (statistical analysis, if used)
- sympy (for comparative validation)

Ensure all dependencies are installed before executing analysis scripts.

# Decay Analysis Directory

## Overview

This directory contains specialized performance characterization studies that demonstrate algorithmic decay patterns beyond critical variable thresholds. These studies document performance degradation when geometric K-map advantages are eliminated by problem dimensionality.

## Decay Studies

### 3D_decay.py
**Three-Dimensional Minimization Decay (9-12 Variables)**

Characterizes performance degradation in BoolMinGeo's 3D minimization approach beyond the 8-variable threshold where three-dimensional geometric visualization advantages diminish.

**Research Focus:**
- Performance decay patterns from 9 to 12 variables
- Execution time degradation analysis
- Memory consumption increase characterization
- Solution quality metrics across decay region

**Theoretical Context:**

Three-dimensional K-map visualization provides geometric advantages for functions up to 8 variables (2^8 = 256 entries fit within practical 3D cube representations). Beyond 8 variables, the geometric intuition breaks down, requiring hierarchical decomposition that loses the direct spatial reasoning benefits.

**Key Metrics:**
- Time per truth table entry (efficiency degradation)
- Memory scaling beyond optimal range
- Solution complexity evolution
- Distribution sensitivity at extended scales

**Outputs:**
- Performance decay CSV data
- Statistical analysis of degradation patterns
- PDF report with decay visualization
- Comparative analysis across distributions

### 4D_decay.py
**Four-Dimensional Minimization Decay (11-13 Variables)**

Documents performance characteristics of BoolMinGeo's 4D minimization extension beyond the 10-variable threshold. Demonstrates the limits of four-dimensional geometric reasoning for Boolean minimization.

**Research Focus:**
- Decay patterns in 4D geometric approach (11-13 variables)
- Execution time scaling beyond 4D optimization threshold
- Memory consumption in extended 4D space
- Effectiveness of 4D decomposition strategies

**Theoretical Context:**

Four-dimensional minimization extends the geometric approach by decomposing higher-dimensional problems into 4D subspaces. This provides advantages up to approximately 10 variables, after which the overhead of hierarchical management and 4D-to-3D reduction dominates, causing performance decay.

**Significance:**
- Establishes upper bounds for 4D geometric methods
- Informs transition points to pure hierarchical approaches
- Validates theoretical predictions of algorithmic limitations
- Provides empirical evidence for method selection criteria

**Analysis Components:**
- Distribution-specific decay characterization
- Exponential growth modeling beyond threshold
- Practical performance boundary identification
- Recommendation generation for method selection

## Common Decay Study Methodology

**Experimental Protocol:**

1. **Baseline Establishment:** Characterize optimal performance within valid range
2. **Decay Region Testing:** Systematic testing beyond optimal threshold
3. **Performance Measurement:** High-precision timing and memory profiling
4. **Statistical Characterization:** Distribution analysis and model fitting
5. **Visualization Generation:** Decay curve plotting and trend analysis
6. **Threshold Identification:** Practical limit determination

**Distribution Testing:**

All decay studies evaluate five distribution patterns:
- Sparse (20% ones): Minimal TRUE outputs
- Dense (70% ones): Predominantly TRUE outputs
- Balanced (50% ones): Equal TRUE/FALSE distribution
- Minimal don't-care (2%): Few optimization opportunities
- Heavy don't-care (30%): Extensive optimization potential

**Output Structure:**

Each study produces:
- Raw performance data CSV (all test cases)
- Decay analysis PDF report with visualizations
- Statistical summaries and model parameters
- Practical recommendation documentation

## Theoretical Significance

Decay studies provide empirical validation of theoretical complexity analysis. They demonstrate:

1. **Geometric Method Limitations:** Physical visualization constraints impose practical boundaries
2. **Hierarchical Necessity:** Pure geometric approaches become inefficient beyond certain scales
3. **Transition Thresholds:** Optimal method switching points for algorithm selection
4. **Scalability Boundaries:** Practical limits for real-world applications

## Usage

Execute decay analysis:

```bash
cd StanLogic/tests/KMapSolver/decay_analysis
python 3D_decay.py  # For 3D method decay study
python 4D_decay.py  # For 4D method decay study
```

Studies may require extended execution time (minutes to hours) depending on hardware and configured test parameters.

## Output Locations

Results are saved to:
- `../outputs/benchmark_results3D/` for 3D decay studies
- `../outputs/benchmark_results4D/` for 4D decay studies

Filenames include "decay" designation for clear identification.

## Research Applications

Decay studies inform:
- Algorithm selection criteria for specific problem scales
- Performance expectation setting for users
- Theoretical model validation
- Hybrid method development strategies
- Educational material on algorithmic limitations
- Publication data on practical boundaries

## Interpretation Guidelines

**Performance Decay Indicators:**
- Non-linear execution time increases
- Memory consumption exceeding proportional growth
- Efficiency metrics (time per entry) degradation
- Solution quality consistency maintenance despite slower execution

**Practical Recommendations:**
- Use 3D methods up to 8 variables for optimal performance
- Transition to 4D methods for 9-10 variables
- Apply pure hierarchical methods beyond 10 variables
- Consider computational budget when selecting methods at boundaries

These studies provide quantitative justification for the multi-method architecture employed in StanLogic's Boolean minimization toolkit.

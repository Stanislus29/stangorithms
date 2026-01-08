# 24-Bit Test Directory

## Overview

This directory contains the infrastructure for testing BoolMinHcal's hierarchical minimization method on 24-variable Boolean functions. The test suite employs multicore parallel processing to manage the computational complexity of minimizing functions with 16,777,216 truth table entries.

## Test Files

### test_kmapsolver4d_24bit_multicore.py
**Primary Test Script**

Orchestrates parallel execution of 24-variable Boolean function minimization across multiple processor cores. The script partitions the computational workload and distributes it to dedicated worker processes.

**Key Features:**
- Multicore task distribution for computational efficiency
- Dynamic worker process generation
- Result aggregation from parallel executions
- Memory-efficient processing of large truth tables

### Worker Scripts (worker_core0.py through worker_core3.py)
**Parallel Execution Units**

Individual worker processes responsible for executing subset computations. Each worker handles a portion of the 24-variable minimization workload independently.

**Characteristics:**
- Core-specific processing assignments
- Independent execution contexts
- Isolated result generation
- Minimal inter-process communication overhead

### analyze_24bit_results.py
**Result Analysis Utility**

Post-processing script for aggregating and analyzing results from parallel worker executions. Generates statistical summaries and validates correctness of minimized expressions.

**Functions:**
- Result consolidation from multiple workers
- Statistical metric computation
- Output validation and verification
- Report generation for 24-variable test cases

### check_verilog_duplicates.py
**Verification Utility**

Validates the uniqueness and correctness of Verilog HDL representations generated during 24-variable testing. Ensures no duplicate test cases or erroneous Boolean function definitions.

**Purpose:**
- Verilog file integrity checking
- Duplicate detection in test case generation
- Hardware description language validation
- Test suite quality assurance

## Directory Structure

### outputs/
Contains execution results, performance metrics, and generated reports from 24-variable tests. Due to file size constraints, large CSV files and Verilog outputs may be excluded from version control.

**Typical Contents:**
- Core-specific CSV result files
- Merged result aggregations
- Statistical analysis summaries
- Performance metrics
- PDF reports (when generated)

### verilog/
Houses Verilog HDL representations of 24-variable Boolean functions used in testing. These files serve as alternative test case representations and validation references.

**Contents:**
- Boolean function Verilog modules
- Test case HDL definitions
- Archived compressed Verilog collections

## Computational Requirements

**Processing Demands:**
- Minimum 4 CPU cores recommended for parallel execution
- High memory availability (8GB+ RAM recommended)
- Extended execution time (hours to days depending on hardware)
- Substantial disk space for result storage

**Scale:**
- 24 variables = 16,777,216 truth table entries
- Exponential complexity in minimization process
- Large intermediate data structures
- Significant memory footprint per worker

## Usage

Execute the multicore test:

```bash
cd StanLogic/tests/KMapSolver/24_bits_test
python test_kmapsolver4d_24bit_multicore.py
```

The test will automatically spawn worker processes and distribute the computational workload. Monitor system resources during execution.

## Research Context

This test suite demonstrates:
- Scalability limits of hierarchical Boolean minimization
- Practical computational boundaries for geometric K-map methods
- Effectiveness of parallel processing strategies
- Memory management challenges at extreme scales

Results from 24-variable testing inform understanding of algorithm behavior beyond typical educational and commercial use cases, establishing theoretical and practical performance boundaries.

## Output Interpretation

Generated outputs provide:
- Execution time per worker core
- Memory consumption profiles
- Solution quality metrics (literal counts, term counts)
- Computational feasibility assessment
- Hardware-specific performance characterization

For detailed results interpretation, consult the README_24bit_multicore.md file in this directory.

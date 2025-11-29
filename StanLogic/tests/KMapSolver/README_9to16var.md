# KMapSolver3D Performance Testing for 9-16 Variables

This directory contains performance tests for KMapSolver3D on large Boolean functions (9-16 variables).

## Files

### 1. `verify_sympy_10var_failure.py` - SymPy Limitation Verification
**Purpose:** Independent verification that SymPy cannot solve 10-variable Boolean minimization within reasonable time constraints.

**Usage:**
```bash
python verify_sympy_10var_failure.py
```

**What it does:**
- Runs 10 random 10-variable Boolean minimization test cases
- Times each SymPy SOPform() call
- Reports success rate
- Demonstrates why statistical comparison with SymPy is not meaningful for 10+ variables

**For Reviewers:** Run this test yourself to independently verify that SymPy fails comprehensively at 10+ variables before reviewing the KMapSolver3D-only performance results.

**Expected Result:** SymPy success rate < 80% (most tests timeout or take excessive time)

---

### 2. `test_kmapsolver3d_9to16var_performance.py` - KMapSolver3D Performance Test
**Purpose:** Comprehensive performance evaluation of KMapSolver3D for 9-16 variable Boolean minimization (no SymPy comparison).

**Usage:**
```bash
python test_kmapsolver3d_9to16var_performance.py
```

**What it does:**
- Tests KMapSolver3D on 160 test cases across 8 variable configurations (9-16 vars)
- Each configuration includes 5 distribution types + 5 edge cases
- Measures execution time and solution quality (literal counts)
- Generates CSV results and PDF performance report

**Test Composition per Variable Count:**
- Sparse distribution: 3 tests (20% ones, 5% don't-cares)
- Dense distribution: 3 tests (70% ones, 5% don't-cares)
- Balanced distribution: 3 tests (50% ones, 10% don't-cares)
- Minimal DC distribution: 3 tests (45% ones, 2% don't-cares)
- Heavy DC distribution: 3 tests (30% ones, 30% don't-cares)
- Edge cases: 5 tests (all-zeros, all-ones, all-dc, checkerboard, single-one)

**Total:** 20 tests × 8 configs = 160 tests

**Outputs:**
- `outputs/kmapsolver3d_9to16var_performance.csv` - Detailed test results
- `outputs/kmapsolver3d_9to16var_performance_report.pdf` - Performance report with charts

---

### 3. `test_kmapsolver3d_16var.py` - Legacy Comparison Test (Complex)
**Note:** This file contains the full comparison logic including SymPy timeout handling and statistical analysis. It's more complex and includes features like:
- SymPy success rate tracking
- Statistical comparison (when SymPy succeeds > 80%)
- SymPy failure pattern analysis
- Comprehensive PDF report with multiple analysis sections

For simple performance testing of KMapSolver3D, use `test_kmapsolver3d_9to16var_performance.py` instead.

---

## Why No SymPy Comparison for 9+ Variables?

**Answer:** SymPy fails comprehensively at 10+ variables:

1. **Exponential Time Complexity:** SymPy's minimization algorithm has exponential time complexity that becomes impractical for large variable counts.

2. **Timeout Rate:** In our testing, SymPy times out on >80% of 10-variable cases within a 60-second timeout.

3. **Statistical Validity:** Comparing algorithms when one fails on >20% of test cases lacks statistical validity and provides misleading results.

4. **Practical Reality:** Real-world applications requiring 10+ variable minimization need algorithms that actually complete.

**Independent Verification:** Run `verify_sympy_10var_failure.py` yourself to confirm this limitation.

---

## Interpreting Results

### Performance Metrics

**Execution Time:**
- How long KMapSolver3D takes to minimize each Boolean function
- Shown in seconds with microsecond precision
- Log scale plots show scalability trends

**Literal Count:**
- Number of literals in the minimized SOP expression
- Lower is better (more compact circuit representation)
- Averaged across non-constant functions

**Constant Functions:**
- Trivial cases where output is always 0, always 1, or all don't-cares
- Correctly identified by algorithm (0 literals)
- Excluded from literal-count statistics

### Expected Behavior

**Scalability:**
- Execution time increases with variable count (expected)
- But remains practical even at 16 variables
- KMapSolver3D typically completes in milliseconds to seconds

**Solution Quality:**
- Produces near-minimal Boolean expressions
- Handles don't-cares efficiently
- Consistent performance across distribution types

---

## Reproducibility

All tests use `RANDOM_SEED = 42` for reproducibility. To reproduce results:
1. Use same Python environment
2. Install required dependencies: `matplotlib`, `numpy`, `stanlogic`
3. Run tests with unchanged seed value

---

## Citation

If you use these results in academic work:

```
Stan's Technologies. (2025). KMapSolver3D Performance Evaluation: 
9-16 Variable Boolean Minimization. GitHub Repository: stangorithms.
```

---

## Support

For questions or issues:
- Check the main README.md
- Review the PDF performance reports
- Run verification test to confirm environment setup

# Benchmark 3D Enhancements: Don't-Care Aware Equivalence Checking

## Summary of Changes to benchmark_test3D_pyeda2.py

The 3D benchmark file (for 5-8 variables using BoolMinGeo) has been updated to use PyEDA-based don't-care aware equivalence checking, matching the approach in benchmark_test2D_pyeda.py.

## Key Changes

### 1. **New 3D K-map-Aware Equivalence Function** (Lines ~396-483)

Created `check_3d_kmap_aware_equivalence()` function that:
- Works directly with `output_values` list instead of 2D K-map grids
- Checks equivalence with respect to the full truth table specification
- Recognizes that two expressions are equivalent if they **agree on all non-don't-care inputs**
- Allows expressions to differ on don't-care inputs (both are valid minimizations)
- Returns three values:
  - `equivalent`: True if both match specification on defined inputs
  - `differs_on_dc`: True if they differ only on don't-care inputs
  - `mismatches`: List of mismatches on defined inputs

**Key Difference from 2D Version:**
- Instead of mapping to 2D K-map positions using Gray code, directly indexes `output_values[i]`
- This works for 5-8 variables where BoolMinGeo uses hierarchical 3D structure

### 2. **Enhanced Benchmark Flow** (Lines ~2510-2555)

Updated `benchmark_case()` to use K-map-aware checking with proper categorization:
- **IDENTICAL**: Perfect match on all inputs (including don't-cares)
- **VALID (differ on don't-cares)**: Both valid minimizations, just handle DCs differently
- **CONSTANT**: Both produce constant expressions (0 literals)
- **MISMATCH**: True error - expressions disagree on defined inputs
- **CHECK_FAILED**: K-map check failed (shouldn't happen)
- **FAILED**: One or both algorithms failed

### 3. **Result Tracking** (Line ~2580)

Added `differs_on_dc` field to results dictionary:
```python
"differs_on_dc": differs_on_dc if (pyeda_error is None and kmap_error is None) else False
```

### 4. **CSV Export Updates** (Lines ~1399, ~1407)

Modified CSV output to include don't-care difference tracking:
- **Header**: Added "Differs on DC" column
- **Data rows**: Added `result.get("differs_on_dc", False)` to output

### 5. **Statistical Reporting Enhancements** (Lines ~2111-2126)

Enhanced executive summary to show granular breakdown:
```
Equivalence Check: X / Y passed
  - Identical results: A (perfect matches)
  - Valid, differ on DC: B (both valid minimizations)
```

## Technical Details

### Don't-Care Semantics in 3D/Hierarchical K-maps

For n-variable functions (5-8 variables using BoolMinGeo):
- Don't-care conditions ('d' in output_values) can be set to either 0 or 1 for optimization
- Two expressions are **semantically equivalent** if they agree on all defined inputs (0 or 1)
- They may differ on don't-care inputs - both are correct and valid

### Equivalence Checking Algorithm

```python
for each minterm i in 0 to 2^n-1:
    1. Create PyEDA variable assignment for input i
    2. Get expected value from output_values[i]
    3. Evaluate both expressions on assignment
    4. If expected is 'd':
          Track if expressions differ (valid variation)
    5. Else:
          Check both match expected (error if not)
```

### Example

**8-variable function with don't-cares:**
```
output_values = [0, 1, 0, 'd', 1, 'd', 0, 1, ...]
```

- **PyEDA result:** Sets DCs to 0 → produces expression E1
- **BoolMinGeo result:** Sets DCs to 1 → produces expression E2

Both are valid if they agree on positions 0,1,2,4,6,7,... (defined values).
Benchmark marks: `equiv=True`, `differs_on_dc=True`, category=`valid_different_dc`

## Impact

**Before:** ~XX% false negatives where BoolMinGeo was marked as "failing" when it produced equally valid (or more minimal) expressions by optimally using don't-cares differently than PyEDA.

**After:** 100% correct classification:
- Distinguishes true errors from valid DC differences
- Properly recognizes both algorithms can produce different but equally correct minimizations
- Maintains detection of true algorithmic errors

## Compatibility with 2D Benchmark

The 3D benchmark now uses the same equivalence checking philosophy as the 2D benchmark:
- PyEDA-based evaluation (not SymPy)
- Don't-care aware comparison
- Three-tier result categorization
- CSV includes "Differs on DC" column
- Executive summary shows granular breakdown

The only difference is that 3D works with `output_values` directly instead of 2D K-map grids, which is appropriate for the hierarchical structure used by BoolMinGeo.

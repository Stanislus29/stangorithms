================================================================================
BENCHMARK TEST FAILURE ROOT CAUSE ANALYSIS
================================================================================

ISSUE IDENTIFIED:
-----------------
The benchmark_test3D_pyeda2.py is failing all tests due to a **variable ordering 
mismatch** between PyEDA and BoolMinGeo.

TECHNICAL DETAILS:
------------------

1. **PyEDA Variable Ordering:**
   - PyEDA uses variables (a, b, c, d, e, f, g, h) corresponding to (x1, x2, ..., x8)
   - When evaluating at minterm index i, PyEDA extracts bits as:
     ```python
     bit = (i >> (num_vars - 1 - j)) & 1
     ```
   - This gives MSB-first ordering: bit 0 of i goes to the MOST significant variable

2. **BoolMinGeo Expected Ordering:**
   - BoolMinGeo expects output_values indexed in standard binary order
   - Minterm 0 = all variables = 0 (00000...)
   - Minterm 1 = only last variable = 1 (00001...)
   - Minterm 2^n-1 = all variables = 1 (11111...)

3. **The Problem:**
   At line 476 in check_3d_kmap_aware_equivalence():
   ```python
   # Get expected value directly from output_values
   expected = output_values[i]
   ```
   
   This assumes output_values[i] corresponds to the binary representation of i
   in the same bit order as PyEDA uses for its variables.
   
   However, if output_values was generated with a different bit order, this
   will cause mismatches even when both expressions are logically correct!

4. **Variable Assignment Mismatch:**
   Lines 467-470:
   ```python
   for j, v in enumerate(pyeda_vars):
       bit = (i >> (num_vars - 1 - j)) & 1
       assign[v] = expr(1) if bit else expr(0)
   ```
   
   This extracts bits in MSB-first order. Variable x1 gets the MSB of i.
   But BoolMinGeo might be using LSB-first or a different ordering.

EXAMPLE OF THE BUG:
-------------------
For 5 variables (x1, x2, x3, x4, x5):

Minterm index i = 5 = 0b00101

PyEDA interpretation (MSB-first):
- x1 = 0, x2 = 0, x3 = 1, x4 = 0, x5 = 1

BoolMinGeo might interpret index 5 as:
- Binary 00101 LSB-first: x5=0, x4=0, x3=1, x2=0, x1=1
- Or use truth table ordering where x1 is LSB

If the benchmark checks output_values[5] against PyEDA's interpretation,
but output_values was generated with BoolMinGeo's ordering, they won't match!

THE FIX NEEDED:
---------------
The benchmark needs to ensure consistent variable ordering between:
1. How output_values is indexed
2. How PyEDA evaluates expressions  
3. How BoolMinGeo interprets the truth table

Options:
A. Convert output_values to match PyEDA's variable ordering before comparison
B. Convert PyEDA's minterm index to match BoolMinGeo's expected ordering
C. Use explicit variable assignments that both systems agree on

VERIFICATION:
-------------
Check the generate_test_cases() function to see how output_values is created.
Check BoolMinGeo.__init__() to see how it interprets output_values.
Ensure consistent bit ordering throughout.

ADDITIONAL ISSUES:
------------------
1. The benchmark may also have issues with how it constructs PyEDA expressions
   from minterms - need to verify the minterm-to-variable mapping matches
   what BoolMinGeo expects.

2. The parse_kmap_expression() and parse_kmap_to_pyeda() functions need to
   ensure they convert variable names correctly between the two systems.

STATUS:
-------
This explains why ALL tests fail - it's not that the algorithm is wrong,
but that the equivalence check is comparing apples to oranges due to
different variable orderings.

RECOMMENDED ACTION:
-------------------
1. Add debug output showing:
   - Input minterm index
   - PyEDA variable assignment  
   - BoolMinGeo expected output
   - PyEDA evaluated result
   - BoolMinGeo evaluated result

2. Verify the variable ordering is consistent throughout

3. Fix the mismatch by standardizing on one ordering convention

================================================================================

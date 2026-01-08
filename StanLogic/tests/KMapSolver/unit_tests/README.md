# Unit Tests Directory

## Overview

This directory contains fundamental unit tests for BoolMin2D's 2-4 variable Boolean minimization functionality. These tests validate core algorithm correctness through systematic truth table verification and edge case analysis.

## Test Files

### tbk_test2var.py
**2-Variable Boolean Function Tests**

Comprehensive test suite for 2-variable Boolean minimization. Validates all 16 possible 2-variable Boolean functions (2^4 truth table combinations).

**Test Coverage:**
- Exhaustive enumeration of all 2-variable functions
- Trivial cases (constant 0, constant 1, identity, negation)
- Standard logic gates (AND, OR, XOR, NAND, NOR, XNOR)
- Complex functions requiring multiple terms
- Don't-care condition handling

**Validation Methodology:**
- Truth table generation and verification
- Expression parsing and evaluation
- Bitwise comparison with expected outputs
- Literal and term count validation

**Significance:**
- Establishes baseline correctness for simplest cases
- Validates fundamental algorithm mechanisms
- Serves as regression test foundation
- Confirms proper handling of all logical operations

### tbk_test3var.py
**3-Variable Boolean Function Tests**

Test suite for 3-variable Boolean minimization covering representative cases from the 256 possible 3-variable functions (2^8 combinations).

**Test Selection:**
- Systematically chosen representative functions
- Common patterns (majority, parity, multiplexer)
- Edge cases (sparse, dense, balanced distributions)
- Functions requiring optimal term selection
- Don't-care optimization scenarios

**Test Categories:**
- **Trivial Functions:** Constants and single-variable expressions
- **Standard Gates:** Three-input AND, OR, XOR, and derivatives
- **Canonical Forms:** Minterms and maxterms
- **Optimization Cases:** Multiple valid minimal forms
- **Don't-Care Effects:** Functions with 'd' entries

**Educational Value:**
- Demonstrates K-map grouping for 3-variable cases
- Illustrates transition from exhaustive to representative testing
- Shows algorithm behavior on real-world digital logic patterns

### tbk_test4var.py
**4-Variable Boolean Function Tests**

Test suite for 4-variable Boolean minimization, the upper limit of classical 2D Karnaugh map visualization. Tests selected from 65,536 possible functions (2^16 combinations).

**Test Strategy:**
- Stratified sampling across function complexity spectrum
- Critical patterns from digital circuit design
- Known difficult cases for minimization algorithms
- Functions demonstrating don't-care optimization advantages
- Performance boundary cases

**Complexity Classes Tested:**
- **Low Complexity:** Functions requiring 1-2 terms (sparse/dense)
- **Medium Complexity:** Standard circuits (decoders, encoders, adders)
- **High Complexity:** Functions near maximal term requirements
- **Optimization Challenges:** Multiple local minima scenarios

**Validation Focus:**
- Correctness verification across complexity range
- Optimal term selection validation
- Don't-care condition proper utilization
- Expression minimality confirmation

**Representative Circuits:**
- 4-input multiplexers
- Half adder and full adder logic
- Priority encoders
- Comparator circuits
- Parity generators

## Test Execution

Run individual test suites:

```bash
cd StanLogic/tests/KMapSolver/unit_tests
python tbk_test2var.py
python tbk_test3var.py
python tbk_test4var.py
```

Tests execute automatically and report pass/fail status for each test case. Failures include detailed diagnostic information.

## Test Output Format

Typical test output structure:
```
Testing 2-variable function [0, 0, 0, 1]...
Expected: AB
Result: AB
Literals: 2
PASS

Testing 3-variable function [1, 1, 0, 1, 1, 0, 0, 1]...
Expected: A'B' + A'C + AB'C
Result: A'B' + A'C + AB'C
Literals: 9
PASS
```

## Regression Testing

These unit tests serve as:
- **Algorithm Validation:** Confirms correctness after modifications
- **Refactoring Safety Net:** Detects unintended behavioral changes
- **Performance Baseline:** Establishes timing benchmarks
- **Documentation:** Provides verified examples of expected behavior

## Test Case Selection Methodology

**2-Variable Tests:**
- Exhaustive coverage (all 16 functions)
- Complete validation of minimal case behavior

**3-Variable Tests:**
- Representative sampling covering major patterns
- Edge cases and corner conditions
- Don't-care optimization verification

**4-Variable Tests:**
- Stratified sampling by complexity
- Industry-relevant circuit patterns
- Known algorithmic challenge cases
- Performance boundary exploration

## Extension Guidelines

To add new test cases:

1. Define input truth table as list or tuple
2. Specify expected minimized expression (optional for validation)
3. Include don't-care conditions as 'd' entries if applicable
4. Document the test case purpose and expected behavior
5. Add to appropriate test file based on variable count

Example test case addition:
```python
# Test XOR function (3 variables)
output_values = [0, 1, 1, 0, 1, 0, 0, 1]
result = minimize_function(3, output_values)
verify_correctness(result, output_values)
```

## Dependencies

Minimal requirements:
- StanLogic package installed
- Python standard library
- No external testing frameworks required (self-contained assertions)

## Relationship to Benchmarks

Unit tests differ from benchmarks:
- **Unit Tests:** Correctness validation, fast execution, deterministic
- **Benchmarks:** Performance characterization, statistical analysis, comparative evaluation

Unit tests ensure the algorithms work correctly; benchmarks measure how well they work under various conditions.

## Continuous Integration

These tests are suitable for:
- Pre-commit hooks
- Automated CI/CD pipelines
- Nightly regression testing
- Release validation

Fast execution time (seconds) enables frequent validation cycles.

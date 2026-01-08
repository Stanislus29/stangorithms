# Demos Directory

## Overview

This directory contains interactive demonstration scripts that illustrate the usage and capabilities of StanLogic's Boolean minimization algorithms. The demos provide practical examples for educational purposes and serve as reference implementations for common use cases.

## Demo Scripts

### k_map_demo.py
**Basic Karnaugh Map Demonstration**

Introductory demonstration of BoolMin2D functionality for 2-4 variable Boolean functions. Provides clear, commented examples of standard K-map minimization workflows.

**Features:**
- Step-by-step minimization examples
- Multiple variable count demonstrations (2, 3, and 4 variables)
- Don't-care condition handling illustrations
- Both SOP and POS form generation examples
- Console output with readable formatted results

**Target Audience:**
- Students learning Boolean algebra and K-map minimization
- Educators seeking example code for teaching materials
- Developers integrating StanLogic into applications
- Researchers evaluating algorithm behavior

**Example Use Cases:**
- Truth table to minimized expression conversion
- Interactive K-map problem solving
- Algorithm behavior visualization
- Quick verification of hand-calculated solutions

**Typical Output:**
```
Original function: [0, 1, 1, 0, 1, 1, 0, 1]
Minimized SOP: A'B + AB' + B
Literal count: 5
Term count: 3
```

### kmap3Ddemo.py
**Three-Dimensional K-Map Demonstration**

Advanced demonstration showcasing BoolMinGeo's 3D minimization capabilities for 5-8 variable Boolean functions. Illustrates extended K-map methods and hierarchical decomposition strategies.

**Features:**
- 3D minimization workflow examples
- Medium-complexity function handling (5-8 variables)
- Geometric visualization concept illustrations
- Performance characteristic demonstrations
- Multiple distribution pattern examples

**Educational Value:**
- Demonstrates transition from 2D to 3D K-map thinking
- Illustrates spatial reasoning in higher dimensions
- Shows practical application of cube-based grouping
- Provides context for when 3D methods are optimal

**Example Applications:**
- Multiplexer truth table minimization
- Digital circuit design optimization
- Computer architecture Boolean function simplification
- Control logic synthesis for moderate complexity systems

**Complexity Range:**
- 5 variables: 32 truth table entries
- 6 variables: 64 truth table entries
- 7 variables: 128 truth table entries
- 8 variables: 256 truth table entries

## Demo Execution

Run demonstrations directly:

```bash
cd StanLogic/tests/KMapSolver/demos
python k_map_demo.py       # Basic 2-4 variable examples
python kmap3Ddemo.py        # Advanced 3D minimization examples
```

Demos are self-contained and require no command-line arguments. Output is printed to console with clear formatting and explanatory text.

## Customization

Demos can be easily modified to test custom Boolean functions:

1. **Modify Truth Tables:** Edit the output value arrays in the script
2. **Adjust Variable Counts:** Change the number of variables parameter
3. **Test Different Distributions:** Create custom TRUE/FALSE patterns
4. **Explore Don't-Care Effects:** Add 'd' entries to observe optimization behavior

Example modification:
```python
# Original demo
output_values = [0, 1, 1, 0, 1, 1, 0, 1]

# Custom test case
output_values = [1, 1, 0, 'd', 1, 0, 'd', 1]  # With don't-cares
```

## Educational Context

Demos support learning objectives:
- Understanding K-map grouping principles
- Visualizing Boolean simplification processes
- Recognizing optimal term selection strategies
- Appreciating don't-care condition advantages
- Transitioning from manual to algorithmic methods

## Integration Examples

Demos serve as templates for integrating StanLogic into:
- Digital logic simulators
- Electronic design automation tools
- Computer architecture education platforms
- Boolean algebra tutoring systems
- Automated circuit optimization workflows

## Output Interpretation

Demo outputs typically include:
- **Original Function:** Input truth table or Boolean expression
- **Minimized Expression:** Simplified SOP or POS form
- **Metrics:** Literal count, term count, optimization percentage
- **Timing:** Execution time for performance reference
- **Verification:** Correctness validation (in some demos)

## Extended Usage

Developers can extend demos by:
- Adding visualization output (matplotlib plots)
- Implementing interactive input prompts
- Creating batch test case processing
- Generating comparison tables (StanLogic vs other tools)
- Building educational GUIs around demo logic

## Dependencies

Minimal dependencies required:
- StanLogic package (installed)
- Python standard library
- Optional: matplotlib for visualization extensions

No heavy computational requirements; demos execute in milliseconds for standard test cases.

## Documentation Reference

For comprehensive algorithm documentation, consult:
- `../../docs/kmapsolver.md` for detailed methodology
- Source code comments within demo scripts
- Main README for package overview

Demos complement formal documentation by providing practical, executable examples of documented concepts.

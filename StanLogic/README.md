<div align="center">
  <b font-size: 64px;>STANLOGIC</b>
  <p font-size: 14px;">by</p>

  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="images\St_logo_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="images\St_logo_light.png">
    <img alt="Project logo" src="images\St_logo_light.png" width="300">
  </picture>

  <hr style="margin-top: 10px; width: 60%;">
</div>

## Overview

StanLogic is a Python package dedicated to Boolean expression simplification and logical reasoning. It provides computational tools that bridge mathematical formalism and algorithmic implementation for digital logic, electronics, and computer science education.

The project aims to serve as a pedagogical and research tool, offering students, educators, and developers a clear computational framework for analyzing and minimizing Boolean logic.

## Research Background

StanLogic is part of an ongoing research initiative by ```Stan's Technologies```, focused on computational models in logic, electronics, and algorithm design. The motivation behind this work is to create open-source computational models that can aid teaching, research, and local problem-solving in Africa and beyond.

This project demonstrates how algorithmic engineering can make mathematical and electronic concepts tangible through simulation and software. The goal is to advance computational literacy by showing how complex logic principles can be represented in executable form.

## Modules within the Package

1. KMapSolver – A module for deriving minimal SOP (Sum of Products) and POS (Product of Sums) expressions for 2, 3, and 4-variable Karnaugh Maps.

    - Supports don’t-care conditions.

    - Uses efficient bitmask operations for implicant filtering and coverage analysis.

    - Outputs minimal Boolean expressions with human-readable and symbolic formats.

Future modules will include additional logic simplifiers, expression parsers, and symbolic verification tools.

### Documentation and Test Files

- Documentation can be found here: [kmapsolver.md](docs/kmapsolver.md)
- Test files can be found here: [kmapsolver tests](tests/)

The documentation provides extensive guides on the thought processes behind the costruction of the algorithm, key optimizations, and future research directions. 

The test files show how the methods within the algorithm may be used, as well as benchmark test cases against ```SymPy```, and the corresponding output 

## Repository Structure 

```css
StanLogic
│
├── docs
│   └── kmapsolver.md /* Documentation for kmap solver */
│
├── images
│   ├── St_logo_dark.png
│   ├── St_logo_light-tp.png
│   └── St_logo_light.png
│
├── prototypes /* Base prototypes to understand development thought process */
│   ├── kmap_solver_prototype.py 
│   └── kmapsolver.py
│
├── src
│   └── stanlogic 
│       ├── init.py
│       └── kmapsolver.py /* Source code for kmap solver */
├── tests
│   └── KMapSolver /* Test files for kmap solver */
│       ├── outputs /* Outputs for test results save here */
│           ├── benchmark_results.csv 
│           └── benchmark_results.pdf
│       ├── benchmark_test.py
│       ├── k_map_demo.py
│       ├── tbk_test2var.py
│       ├── tbk_test3var.py
│       └── tbk_test4var.py
└── README.md
```

## How to Use

Clone the repository and install locally:

``` bash
git clone https://github.com/<your-username>/StanLogic.git
cd StanLogic/src
pip install -e .
```
## Citation

If you use StanLogic in your research, teaching, or software, please cite it as follows:

**Plain Text:**

    Somtochukwu Emeka-Onwuneme. StanLogic: A Python Package for Boolean Simplification and Logic Computation. Stan's Technologies, 2025. GitHub Repository.

**BibTeX:**

```BibTeX
@software{stanlogic2025,
author = {Somtochukwu Emeka-Onwuneme},
title = {StanLogic: A Python Package for Boolean Simplification and Logic Computation},
year = {2025},
institution = {Stan's Technologies},
url = {https://github.com/Stanislus29/stangorithms/tree/main/StanLogic}
}
```

## Licence 
This project is licensed under the MIT Non-Commercial (MIT-NC) License.

You are free to use, modify, and distribute this code for educational and research purposes only. Commercial use requires explicit permission from Somtochukwu Emeka-Onwuneme (Stan's Technologies).

For inquiries regarding collaboration, research use, or licensing, contact:

Email: stanstechnologies@gmail.com

Institution: Stan's Technologies, Ghana

## Acknowledgments
This project was built as part of a broader initiative to develop computational educational tools in logic and mathematics. The author acknowledges the open-source community for inspiring reproducible research and educational innovation.


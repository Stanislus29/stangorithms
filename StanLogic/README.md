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

1. ```BoolMin2D``` – An algorithmic implementation for solving 2D K-maps (2 to 4) variables. 

2. ```BoolMinGeo``` - A geometric clustering algorithm for simplifying Boolean systems greater than 4 variables. Also contains a purely hireachical method for systems greater than 10 variables where geometric clustering fails. 

3. ```BoolMinHcal``` - An algorithm for simplifying Boolean systems greater than 16 variables, where geometric methods fail.

### Documentation and Test Files

- A research paper formulating the proofs and explaining the mathematical concepts behind the algorithms is currently in development, to be published on ArXiv soon.
- Documentation can be found here for the 2D minimization method: [kmapsolver.md](docs/kmapsolver.md)
- Test files can be found here: [kmapsolver tests](tests/)

The documentation provides extensive guides on the thought processes behind the costruction of the algorithm, key optimizations, and future research directions. 

The test files show how the methods within the algorithm may be used, as well as benchmark test cases against ```SymPy```, and the corresponding output 

## Repository Structure 

```css
StanLogic
│
├── docs
│   ├── kmapsolver.md /* Documentation for 2D Boolean minimization */
│   └── ones_complement.md /* Documentation for ones complement */
│
├── images
│   ├── St_logo_dark.png
│   ├── St_logo_light-tp.png
│   └── St_logo_light.png
│
├── prototypes /* Base prototypes to understand development thought process */
│   ├── kmap_solver_prototype.py 
│   ├── kmapsolver.py
│   └── ones_complement_prototype.py
│
├── src
│   └── stanlogic /* Source code for all modules */
│       ├── __init__.py
│       ├── BoolMin2D.py /* 2-4 variable minimization */
│       ├── BoolMinGeo.py /* 5-10 variable 3D/4D minimization */
│       ├── BoolMinHcal.py /* 16+ variable hierarchical minimization */
│       └── ones_complement.py /* Ones complement operations */
│
├── tests
│   └── KMapSolver
│       ├── 24_bits_test /* 24-variable multicore benchmarks */
│       ├── analysis /* Boolean equivalence & analysis tools */
│       ├── benchmarks /* Performance benchmarks (2D, 3D, 4D, hierarchical) */
│       ├── decay_analysis /* Decay studies beyond thresholds */
│       ├── demos /* Interactive demos and examples */
│       ├── outputs /* Test results organized by benchmark type */
│       │   ├── benchmark_results2D
│       │   ├── benchmark_results3D
│       │   └── benchmark_results4D
│       └── unit_tests /* Unit tests (2-4 variables) */
│
└── README.md
```

## How to Use

Clone the repository and install locally:

``` bash
git clone https://github.com/Stanislus29/stangorithms.git
cd StanLogic/src
pip install -e .
```
## Citation

If you use StanLogic in your research, teaching, or software, please cite it as follows:

**Plain Text:**

    Somtochukwu Stanislus Emeka-Onwuneme. StanLogic: A Python Package for Boolean Simplification and Logic Computation. Stan's Technologies, 2025. GitHub Repository.

**BibTeX:**

```BibTeX
@software{stanlogic2025,
author = {Somtochukwu Stanislus Emeka-Onwuneme},
title = {StanLogic: A Python Package for Boolean Simplification and Logic Computation},
year = {2025},
institution = {Stan's Technologies},
url = {https://github.com/Stanislus29/stangorithms/tree/main/StanLogic}
}
```

## Licence 
This project is dual-licensed under AGPL - 3.0, and a commercial license

You are free to use, modify, and distribute this code for educational and research purposes only. Commercial use requires explicit permission from Somtochukwu Emeka-Onwuneme (Stan's Technologies).

For inquiries regarding collaboration, research use, or licensing, contact:

Email: stanstechnologies@gmail.com

Institution: Stan's Technologies, Ghana

## Acknowledgments
This project was built as part of a broader initiative to develop computational educational tools in logic and mathematics. The author acknowledges the open-source community for inspiring reproducible research and educational innovation.


# PyCommend

```
 ____        ____                                          _
|  _ \ _   _/ ___|___  _ __ ___  _ __ ___   ___ _ __   __| |
| |_) | | | | |   / _ \| '_ ` _ \| '_ ` _ \ / _ \ '_ \ / _` |
|  __/| |_| | |__| (_) | | | | | | | | | | |  __/ | | | (_| |
|_|    \__, |\____\___/|_| |_| |_|_| |_| |_|\___|_| |_|\____|
       |___/
```

This project implements multi-objective optimization algorithms for recommending Python packages. The system analyzes dependencies from real GitHub projects and suggests complementary libraries based on usage patterns and semantic similarity.

## Setup

After cloning the repository, run the data rebuild script:

```bash
cd data
python rebuild_data.py
```

This combines the split similarity matrix files into the complete data file required by the algorithms.

## How it works

PyCommend uses data from approximately 10,000 PyPI packages and 24,000 GitHub projects. When you provide a main library, the system finds packages that are commonly used together with it. The search considers both the frequency with which packages appear together and the similarity between their descriptions.

The project compares two different algorithms. MOEA/D divides the problem into smaller subproblems and solves each one separately. MOVNS explores different package combinations and gradually improves the solutions. Both try to balance three objectives at the same time: maximize package relevance, maintain semantic coherence, and avoid suggesting too many packages at once.

## Basic structure

The `data` folder contains pre-calculated matrices with information about co-occurrence and similarity between packages. The `optimizer` folder has the implementations of MOEA/D and MOVNS algorithms. The `evaluation` folder provides metrics to compare the quality of the solutions found.

To use the system, you need to have the data files in the correct folder and then you can run either algorithm by passing a package name as context. The result will be a list of recommended packages that make sense to use together with that context.

## Authors

This work was developed by Augusto Magalhães Pinto de Mendonça from Instituto de Computação at Universidade Federal Fluminense, Filipe Pessoa Sousa from Instituto de Matemática e Estatística at Universidade do Estado do Rio de Janeiro, and Igor Machado Coelho from Instituto de Computação at Universidade Federal Fluminense. The project is part of research on applying multi-objective metaheuristics to software engineering.

This work was presented at the International Conference on Variable Neighborhood Search (ICVNS 2025) held at https://2025.icvns.com/

## Acknowledgments

This system is a tribute to Professor Pierre Hansen In Memoriam. We express our gratitude to Daniel Aloise (Full Professor at Polytechnique Montréal, Canada), Eduardo G. Pardo (Full Professor at Universidad Rey Juan Carlos, Spain), José Andrés Moreno Pérez (Full Professor at Universidad de La Laguna, Spain), and Angelo Sifaleras (Full Professor at University of Macedonia, Greece) for their contributions to the field of Variable Neighborhood Search.

## Contributing

This project is open for collaborations and improvements. Feel free to explore the code, suggest enhancements, or extend the algorithms to other programming language ecosystems.

## License

MIT License

Copyright (c) 2024 Augusto Magalhães Pinto de Mendonça

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

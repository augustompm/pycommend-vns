# PyCommend

```
 ____        ____                                          _
|  _ \ _   _/ ___|___  _ __ ___  _ __ ___   ___ _ __   __| |
| |_) | | | | |   / _ \| '_ ` _ \| '_ ` _ \ / _ \ '_ \ / _` |
|  __/| |_| | |__| (_) | | | | | | | | | | |  __/ | | | (_| |
|_|    \__, |\____\___/|_| |_| |_|_| |_| |_|\___|_| |_|\____|
       |___/
```

Multi-objective optimization for Python package recommendation using real dependency data from GitHub.

## Setup

After cloning, rebuild the similarity matrix:

```bash
cd data
python rebuild_data.py
```

This reconstructs `package_similarity_matrix_10k.pkl` from split files.

## How it works

The system analyzes 10,000 PyPI packages and 24,000 GitHub repositories to recommend complementary libraries. Given a target package, it optimizes three objectives:

- **Linked Usage (LU)**: Co-occurrence frequency in real projects
- **Semantic Similarity (SS)**: SBERT embeddings from package descriptions
- **Recommended Set Size (RSS)**: Number of suggestions (minimize)

Three algorithms solve this multi-objective problem:

- **MO-GVNS**: Multi-Objective General Variable Neighborhood Search following Duarte et al. (2015)
- **NSGA-II**: Non-dominated Sorting Genetic Algorithm II following Deb et al. (2002)
- **MOEA/D**: Multi-Objective Evolutionary Algorithm based on Decomposition following Zhang & Li (2007)

## Data structure

- `data/` - Pre-computed co-occurrence and similarity matrices
- `optimizer/` - MO-GVNS, NSGA-II, and MOEA/D implementations
- `evaluation/` - Quality metrics (hypervolume, spacing, epsilon-indicator)

Run by providing a package name as context. Output is a Pareto front of recommendation sets.

## References

- Duarte et al. (2015) Multi-objective variable neighborhood search. J Glob Optim 63:515-536
- Hansen et al. (2017) Variable neighborhood search: basics and variants. EURO J Comput Optim 5:423-454
- Deb et al. (2002) A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE TEVC 6(2):182-197
- Zhang & Li (2007) MOEA/D: A multiobjective evolutionary algorithm based on decomposition. IEEE TEVC 11(6):712-731

## Authors

Augusto Magalhaes Pinto de Mendonca (IC/UFF), Filipe Pessoa Sousa (IME/UERJ), Igor Machado Coelho (IC/UFF).

Presented at ICVNS 2025: https://2025.icvns.com/

## Acknowledgments

Tribute to Professor Pierre Hansen (In Memoriam). We thank Daniel Aloise (Polytechnique Montreal, Canada), Eduardo G. Pardo (Universidad Rey Juan Carlos, Spain), Jose Andres Moreno Perez (Universidad de La Laguna, Spain), and Angelo Sifaleras (University of Macedonia, Greece) for their contributions to Variable Neighborhood Search.

## License

MIT License - Copyright (c) 2024 Augusto Magalhaes Pinto de Mendonca

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

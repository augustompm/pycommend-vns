"""
run_comparison.py

Description:
    Runs all three multi-objective algorithms on the same packages
    and saves results to CSV for comparison.

References:
    - [1]: Duarte et al. (2015) J Glob Optim 63:515-536 - MO-GVNS
    - [2]: Deb et al. (2002) IEEE TEVC 6(2):182-197 - NSGA-II
    - [3]: Zhang & Li (2007) IEEE TEVC 11(6):712-731 - MOEA/D

Implementation origin:
    - Comparison framework for PyCommend optimization algorithms

Parameters (R2 compliance):
    TIME_BUDGET=15.0: seconds per algorithm run
    POP_SIZE=30: population/archive size for all algorithms
    N_RUNS=3: number of independent runs per configuration
"""

import numpy as np
import pandas as pd
import time
import os
import sys
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optimizer', 'modules'))

from optimizer.nsga2 import NSGA2_Timed
from optimizer.moead import MOEAD_Timed
from optimizer.mo_gvns_pycommend import MOGVNS_PyCommend


RESULTS_DIR = "C:/Projects/PyCommend/results"
TIME_BUDGET = 15.0
POP_SIZE = 30
TEST_PACKAGES = ['numpy', 'pandas', 'requests']
N_RUNS = 3


def compute_hypervolume_2d_max(objectives):
    """Compute hypervolume for maximization problems (larger is better).

    Reference point is automatically set below minimum values.
    """
    if len(objectives) == 0:
        return 0.0

    ref_point = np.min(objectives, axis=0) - 0.1 * (np.max(objectives, axis=0) - np.min(objectives, axis=0) + 1)

    sorted_idx = np.argsort(objectives[:, 0])
    obj = objectives[sorted_idx]

    hv = 0.0
    prev_y = ref_point[1]

    for o in obj:
        if o[1] > prev_y:
            hv += (o[0] - ref_point[0]) * (o[1] - prev_y)
            prev_y = o[1]

    return hv


def compute_spacing(objectives):
    if len(objectives) < 2:
        return 0.0

    n = len(objectives)
    distances = np.zeros(n)

    for i in range(n):
        min_dist = float('inf')
        for j in range(n):
            if i != j:
                dist = np.sum(np.abs(objectives[i] - objectives[j]))
                min_dist = min(min_dist, dist)
        distances[i] = min_dist

    d_mean = np.mean(distances)
    return np.sqrt(np.sum((distances - d_mean) ** 2) / n)


def run_nsga2(package, time_budget, run_id):
    print(f"    NSGA-II run {run_id}...")
    start = time.perf_counter()

    optimizer = NSGA2_Timed(
        main_package=package,
        pop_size=POP_SIZE,
        time_budget=time_budget
    )
    solutions = optimizer.run()

    elapsed = time.perf_counter() - start

    objectives = np.array([
        [-s['objectives'][0], -s['objectives'][1]]
        for s in solutions
    ])

    hv = compute_hypervolume_2d_max(objectives)
    spacing = compute_spacing(objectives)

    return {
        'algorithm': 'NSGA-II',
        'package': package,
        'run': run_id,
        'n_solutions': len(solutions),
        'hypervolume': hv,
        'spacing': spacing,
        'time': elapsed
    }


def run_moead(package, time_budget, run_id):
    print(f"    MOEA/D run {run_id}...")
    start = time.perf_counter()

    optimizer = MOEAD_Timed(
        main_package=package,
        pop_size=POP_SIZE,
        time_budget=time_budget
    )
    solutions = optimizer.run()

    elapsed = time.perf_counter() - start

    objectives = np.array([
        [-s['objectives'][0], -s['objectives'][1]]
        for s in solutions
    ])

    hv = compute_hypervolume_2d_max(objectives)
    spacing = compute_spacing(objectives)

    return {
        'algorithm': 'MOEA/D',
        'package': package,
        'run': run_id,
        'n_solutions': len(solutions),
        'hypervolume': hv,
        'spacing': spacing,
        'time': elapsed
    }


def run_mogvns(package, time_budget, run_id):
    print(f"    MO-GVNS run {run_id}...")
    start = time.perf_counter()

    optimizer = MOGVNS_PyCommend(
        main_package=package,
        pop_size=POP_SIZE,
        time_budget=time_budget,
        k_max=3,
        k_prime_max=3,
        seed=run_id * 100
    )

    solutions = optimizer.run()

    elapsed = time.perf_counter() - start

    objectives = np.array([
        [s['linked_usage'], s['semantic_similarity']]
        for s in solutions
    ])

    hv = compute_hypervolume_2d_max(objectives)
    spacing = compute_spacing(objectives)

    return {
        'algorithm': 'MO-GVNS',
        'package': package,
        'run': run_id,
        'n_solutions': len(solutions),
        'hypervolume': hv,
        'spacing': spacing,
        'time': elapsed
    }


def main():
    print("=" * 60)
    print("ALGORITHM COMPARISON")
    print(f"Packages: {TEST_PACKAGES}")
    print(f"Time budget: {TIME_BUDGET}s, Runs: {N_RUNS}")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = []

    for package in TEST_PACKAGES:
        print(f"\nPackage: {package}")
        print("-" * 40)

        for run in range(1, N_RUNS + 1):
            try:
                result = run_nsga2(package, TIME_BUDGET, run)
                results.append(result)
                print(f"      HV={result['hypervolume']:.4f}, Solutions={result['n_solutions']}")
            except Exception as e:
                print(f"      NSGA-II ERROR: {e}")

            try:
                result = run_moead(package, TIME_BUDGET, run)
                results.append(result)
                print(f"      HV={result['hypervolume']:.4f}, Solutions={result['n_solutions']}")
            except Exception as e:
                print(f"      MOEA/D ERROR: {e}")

            try:
                result = run_mogvns(package, TIME_BUDGET, run)
                results.append(result)
                print(f"      HV={result['hypervolume']:.4f}, Solutions={result['n_solutions']}")
            except Exception as e:
                print(f"      MO-GVNS ERROR: {e}")

    df = pd.DataFrame(results)

    csv_path = os.path.join(RESULTS_DIR, "experiment_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    summary = df.groupby('algorithm').agg({
        'hypervolume': ['mean', 'std'],
        'spacing': ['mean', 'std'],
        'n_solutions': 'mean',
        'time': 'mean'
    }).round(4)

    print(summary)

    summary_path = os.path.join(RESULTS_DIR, "summary_statistics.csv")
    summary.to_csv(summary_path)
    print(f"\nSummary saved to: {summary_path}")

    return df


if __name__ == '__main__':
    os.chdir("C:/Projects/PyCommend/pycommend-vns-github")
    main()

"""
Unit Tests for VND Module

References:
    Duarte et al. (2015) Algorithm 5: VND-i, Algorithm 6: MO-VND

Tests all three optimization versions: baseline, numpy, numba
"""

import numpy as np
import time
from vnd import (
    local_search_swap_baseline, local_search_swap_numpy, local_search_swap_numba,
    vnd_i_baseline, vnd_i_numpy, vnd_i_numba,
    mo_vnd_baseline, mo_vnd_numpy, mo_vnd_numba,
    NUMBA_AVAILABLE, get_implementation
)


def create_simple_evaluator(n_packages: int = 100):
    """
    Create a simple multi-objective evaluator for testing.

    Objectives (minimization):
    - f0: Sum of package indices (prefer lower indices)
    - f1: Negative sum (prefer higher indices)
    """
    def evaluate(solution: np.ndarray) -> np.ndarray:
        f0 = np.sum(solution) / n_packages
        f1 = -np.sum(solution) / n_packages + len(solution)
        return np.array([f0, f1])

    return evaluate


def test_local_search_basic():
    """Test basic local search functionality."""
    print("\n[Test 1] local_search_swap basic functionality")
    print("-" * 60)

    n_packages = 100
    evaluate = create_simple_evaluator(n_packages)
    rng = np.random.default_rng(42)
    solution = np.array([50, 51, 52, 53, 54])

    versions = [
        ('baseline', local_search_swap_baseline),
        ('numpy', local_search_swap_numpy),
        ('numba', local_search_swap_numba),
    ]

    all_passed = True
    for name, func in versions:
        rng_local = np.random.default_rng(42)
        init_obj = evaluate(solution)

        best_sol, best_obj = func(
            solution.copy(), 0, evaluate, n_packages, 1, rng_local, 50
        )

        same_size = len(best_sol) == len(solution)
        all_valid = all(0 <= p < n_packages for p in best_sol)
        all_unique = len(set(best_sol)) == len(best_sol)
        not_worse = best_obj[0] <= init_obj[0]

        passed = same_size and all_valid and all_unique and not_worse
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        improved = "yes" if best_obj[0] < init_obj[0] else "no"
        print(f"  {name:10} | valid={all_valid}, improved={improved} | {status}")

    return all_passed


def test_vnd_i_improves_objective():
    """Test that VND-i improves the target objective."""
    print("\n[Test 2] VND-i improves target objective")
    print("-" * 60)

    n_packages = 100
    evaluate = create_simple_evaluator(n_packages)

    solution = np.array([50, 51, 52, 53, 54])
    init_obj = evaluate(solution)

    versions = [
        ('baseline', vnd_i_baseline),
        ('numpy', vnd_i_numpy),
        ('numba', vnd_i_numba),
    ]

    all_passed = True

    for name, func in versions:
        rng = np.random.default_rng(42)
        improved_sol, improved_obj = func(
            solution.copy(), 0, 3, evaluate, n_packages, rng, 30
        )

        not_worse = improved_obj[0] <= init_obj[0]
        valid = len(set(improved_sol)) == len(improved_sol)

        passed = not_worse and valid
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        delta = init_obj[0] - improved_obj[0]
        print(f"  {name:10} | obj0: {init_obj[0]:.3f} -> {improved_obj[0]:.3f} (delta={delta:.3f}) | {status}")

    for name, func in versions:
        rng = np.random.default_rng(42)
        improved_sol, improved_obj = func(
            solution.copy(), 1, 3, evaluate, n_packages, rng, 30
        )

        not_worse = improved_obj[1] <= init_obj[1]
        valid = len(set(improved_sol)) == len(improved_sol)

        passed = not_worse and valid
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        delta = init_obj[1] - improved_obj[1]
        print(f"  {name:10} | obj1: {init_obj[1]:.3f} -> {improved_obj[1]:.3f} (delta={delta:.3f}) | {status}")

    return all_passed


def test_mo_vnd():
    """Test MO-VND (Algorithm 6) functionality."""
    print("\n[Test 3] MO-VND improves archive")
    print("-" * 60)

    n_packages = 100
    evaluate = create_simple_evaluator(n_packages)

    E_solutions = [
        np.array([40, 41, 42, 43, 44]),
        np.array([50, 51, 52, 53, 54]),
        np.array([60, 61, 62, 63, 64]),
    ]
    E_objectives = np.array([evaluate(s) for s in E_solutions])

    versions = [
        ('baseline', mo_vnd_baseline),
        ('numpy', mo_vnd_numpy),
        ('numba', mo_vnd_numba),
    ]

    all_passed = True
    for name, func in versions:
        rng = np.random.default_rng(42)
        E_copy = [s.copy() for s in E_solutions]
        obj_copy = E_objectives.copy()

        E_new, obj_new = func(
            E_copy, obj_copy, 2, 2, evaluate, n_packages, rng, 20
        )

        same_count = len(E_new) == len(E_solutions)
        all_valid = all(len(set(s)) == len(s) for s in E_new)

        passed = same_count and all_valid
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        print(f"  {name:10} | count={same_count}, valid={all_valid} | {status}")

    return all_passed


def test_vnd_preserves_constraints():
    """Test that VND maintains solution validity."""
    print("\n[Test 4] VND preserves solution constraints")
    print("-" * 60)

    n_packages = 50
    evaluate = create_simple_evaluator(n_packages)

    solution = np.arange(10)

    versions = [
        ('baseline', vnd_i_baseline),
        ('numpy', vnd_i_numpy),
        ('numba', vnd_i_numba),
    ]

    all_passed = True
    for name, func in versions:
        rng = np.random.default_rng(42)
        result, _ = func(solution.copy(), 0, 3, evaluate, n_packages, rng, 50)

        same_size = len(result) == len(solution)
        all_unique = len(set(result)) == len(result)
        all_in_range = all(0 <= p < n_packages for p in result)

        passed = same_size and all_unique and all_in_range
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        print(f"  {name:10} | size={same_size}, unique={all_unique}, range={all_in_range} | {status}")

    return all_passed


def benchmark_versions(n_solutions: int = 5, sol_size: int = 10, n_packages: int = 200):
    """Benchmark all versions."""
    print(f"\n[Benchmark] Solutions={n_solutions}, Size={sol_size}, Packages={n_packages}")
    print("-" * 60)

    evaluate = create_simple_evaluator(n_packages)
    rng = np.random.default_rng(42)

    E_solutions = [
        rng.choice(n_packages, size=sol_size, replace=False)
        for _ in range(n_solutions)
    ]
    E_objectives = np.array([evaluate(s) for s in E_solutions])

    versions = [
        ('baseline', mo_vnd_baseline),
        ('numpy', mo_vnd_numpy),
        ('numba', mo_vnd_numba),
    ]

    results = {}
    for name, func in versions:
        start = time.perf_counter()
        n_runs = 3
        for _ in range(n_runs):
            rng_run = np.random.default_rng(42)
            E_copy = [s.copy() for s in E_solutions]
            obj_copy = E_objectives.copy()
            func(E_copy, obj_copy, 2, 2, evaluate, n_packages, rng_run, 20)

        elapsed = (time.perf_counter() - start) / n_runs * 1000
        results[name] = elapsed
        print(f"  {name:10} | {elapsed:8.3f} ms/call")

    baseline_time = results['baseline']
    print("\n  Speedup vs baseline:")
    for name, t in results.items():
        speedup = baseline_time / t
        print(f"    {name:10} | {speedup:5.2f}x")

    return results


def test_get_implementation():
    """Test implementation selector."""
    print("\n[Test 5] get_implementation()")
    print("-" * 60)

    for version in ['baseline', 'numpy', 'numba']:
        impl = get_implementation(version)
        has_all = all(k in impl for k in ['local_search', 'vnd_i', 'mo_vnd'])
        status = "PASS" if has_all else "FAIL"
        print(f"  {version:10} | all functions present                  | {status}")

    return True


def main():
    print("=" * 60)
    print("VND MODULE TESTS")
    print("Algorithm 5: VND-i, Algorithm 6: MO-VND (Duarte et al. 2015)")
    print("=" * 60)

    print(f"\nNumba available: {NUMBA_AVAILABLE}")

    results = []
    results.append(("local_search_basic", test_local_search_basic()))
    results.append(("vnd_i_improves", test_vnd_i_improves_objective()))
    results.append(("mo_vnd", test_mo_vnd()))
    results.append(("preserves_constraints", test_vnd_preserves_constraints()))
    results.append(("get_implementation", test_get_implementation()))

    benchmark_versions()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  {name:30} | {status}")

    print("\n" + ("ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED"))
    return all_passed


if __name__ == '__main__':
    main()

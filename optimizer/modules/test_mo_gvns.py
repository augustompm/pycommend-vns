"""
Unit Tests for MO-GVNS Module

References:
    Duarte et al. (2015) Algorithm 7: MO-GVNS

Tests the complete MO-GVNS algorithm implementation.
"""

import numpy as np
import time
from mo_gvns import (
    mo_gvns, mo_gvns_baseline, mo_gvns_numpy, mo_gvns_numba,
    create_initial_archive, MOGVNSResult,
    IMPLEMENTATIONS, get_implementation
)


def create_test_evaluator(n_packages: int = 100):
    """
    Create a bi-objective evaluator for testing.

    Objectives (minimization):
    - f0: Prefer lower package indices (sum / n_packages)
    - f1: Prefer higher package indices (inverse sum)
    """
    def evaluate(solution: np.ndarray) -> np.ndarray:
        f0 = np.sum(solution) / n_packages
        f1 = -np.sum(solution) / n_packages + len(solution)
        return np.array([f0, f1])

    return evaluate


def test_initial_archive():
    """Test initial archive creation."""
    print("\n[Test 1] create_initial_archive")
    print("-" * 60)

    n_packages = 100
    evaluate = create_test_evaluator(n_packages)
    rng = np.random.default_rng(42)

    solutions, objectives = create_initial_archive(
        n_solutions=5,
        solution_size=10,
        n_packages=n_packages,
        evaluate_func=evaluate,
        rng=rng
    )

    correct_count = len(solutions) == 5
    correct_obj_shape = objectives.shape == (5, 2)
    all_unique = all(len(set(s)) == len(s) for s in solutions)
    all_valid = all(all(0 <= p < n_packages for p in s) for s in solutions)

    passed = correct_count and correct_obj_shape and all_unique and all_valid
    status = "PASS" if passed else "FAIL"

    print(f"  count={correct_count}, shape={correct_obj_shape}, unique={all_unique}, valid={all_valid} | {status}")

    return passed


def test_mo_gvns_basic():
    """Test basic MO-GVNS execution."""
    print("\n[Test 2] MO-GVNS basic execution")
    print("-" * 60)

    n_packages = 30
    evaluate = create_test_evaluator(n_packages)

    result = mo_gvns(
        evaluate_func=evaluate,
        n_packages=n_packages,
        solution_size=5,
        n_objectives=2,
        k_max=2,
        k_prime_max=1,
        t_max=1,
        initial_archive_size=3,
        seed=42,
        verbose=False
    )

    has_solutions = len(result.solutions) > 0
    has_objectives = len(result.objectives) > 0
    has_history = len(result.history) > 0
    has_time = result.elapsed_time > 0

    passed = has_solutions and has_objectives and has_history and has_time
    status = "PASS" if passed else "FAIL"

    print(f"  solutions={result.n_solutions}, objectives={result.n_objectives}, iterations={len(result.history)} | {status}")

    return passed


def test_mo_gvns_time_limit():
    """Test MO-GVNS with time limit."""
    print("\n[Test 3] MO-GVNS with time limit")
    print("-" * 60)

    n_packages = 30
    evaluate = create_test_evaluator(n_packages)

    time_limit = 0.5

    result = mo_gvns(
        evaluate_func=evaluate,
        n_packages=n_packages,
        solution_size=5,
        n_objectives=2,
        k_max=2,
        k_prime_max=1,
        t_max=100,
        initial_archive_size=3,
        time_limit=time_limit,
        seed=42,
        verbose=False
    )

    within_time = result.elapsed_time <= time_limit + 1.0
    has_results = result.n_solutions > 0

    passed = within_time and has_results
    status = "PASS" if passed else "FAIL"

    print(f"  time={result.elapsed_time:.2f}s (limit={time_limit}s), solutions={result.n_solutions} | {status}")

    return passed


def test_mo_gvns_improves():
    """Test that MO-GVNS improves the initial archive."""
    print("\n[Test 4] MO-GVNS improves archive")
    print("-" * 60)

    n_packages = 50
    evaluate = create_test_evaluator(n_packages)
    rng = np.random.default_rng(42)

    init_solutions, init_objectives = create_initial_archive(
        n_solutions=3,
        solution_size=5,
        n_packages=n_packages,
        evaluate_func=evaluate,
        rng=rng
    )

    result = mo_gvns(
        evaluate_func=evaluate,
        n_packages=n_packages,
        solution_size=5,
        n_objectives=2,
        k_max=2,
        k_prime_max=1,
        t_max=2,
        initial_archive_size=3,
        seed=42,
        verbose=False
    )

    init_min_obj0 = np.min(init_objectives[:, 0])
    init_min_obj1 = np.min(init_objectives[:, 1])
    final_min_obj0 = np.min(result.objectives[:, 0])
    final_min_obj1 = np.min(result.objectives[:, 1])

    improved_obj0 = final_min_obj0 <= init_min_obj0
    improved_obj1 = final_min_obj1 <= init_min_obj1

    passed = improved_obj0 or improved_obj1
    status = "PASS" if passed else "FAIL"

    print(f"  obj0: {init_min_obj0:.3f} -> {final_min_obj0:.3f} | obj1: {init_min_obj1:.3f} -> {final_min_obj1:.3f} | {status}")

    return passed


def test_all_versions():
    """Test all implementation versions produce valid results."""
    print("\n[Test 5] All implementation versions")
    print("-" * 60)

    n_packages = 30
    evaluate = create_test_evaluator(n_packages)

    versions = [
        ('baseline', mo_gvns_baseline),
        ('numpy', mo_gvns_numpy),
        ('numba', mo_gvns_numba),
    ]

    all_passed = True
    for name, func in versions:
        result = func(
            evaluate_func=evaluate,
            n_packages=n_packages,
            solution_size=5,
            n_objectives=2,
            k_max=2,
            k_prime_max=1,
            t_max=1,
            initial_archive_size=3,
            seed=42,
            verbose=False
        )

        valid = result.n_solutions > 0 and result.elapsed_time > 0
        status = "PASS" if valid else "FAIL"
        if not valid:
            all_passed = False

        print(f"  {name:10} | solutions={result.n_solutions}, time={result.elapsed_time:.3f}s | {status}")

    return all_passed


def test_result_container():
    """Test MOGVNSResult container."""
    print("\n[Test 6] MOGVNSResult container")
    print("-" * 60)

    solutions = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    objectives = np.array([[0.5, 0.5], [0.3, 0.7]])
    history = [{'iteration': 0, 'archive_size': 2}]

    result = MOGVNSResult(solutions, objectives, history, 1.5)

    correct_n_solutions = result.n_solutions == 2
    correct_n_objectives = result.n_objectives == 2
    correct_time = result.elapsed_time == 1.5

    passed = correct_n_solutions and correct_n_objectives and correct_time
    status = "PASS" if passed else "FAIL"

    print(f"  n_solutions={result.n_solutions}, n_objectives={result.n_objectives}, time={result.elapsed_time} | {status}")

    return passed


def benchmark_versions(n_packages: int = 30, t_max: int = 1):
    """Benchmark all versions."""
    print(f"\n[Benchmark] Packages={n_packages}, t_max={t_max}")
    print("-" * 60)

    evaluate = create_test_evaluator(n_packages)

    versions = [
        ('baseline', mo_gvns_baseline),
        ('numpy', mo_gvns_numpy),
        ('numba', mo_gvns_numba),
    ]

    results = {}
    for name, func in versions:
        start = time.perf_counter()
        result = func(
            evaluate_func=evaluate,
            n_packages=n_packages,
            solution_size=5,
            n_objectives=2,
            k_max=2,
            k_prime_max=1,
            t_max=t_max,
            initial_archive_size=3,
            seed=42,
            verbose=False
        )
        elapsed = time.perf_counter() - start

        results[name] = elapsed
        print(f"  {name:10} | {elapsed*1000:8.1f} ms | solutions={result.n_solutions}")

    baseline_time = results['baseline']
    print("\n  Speedup vs baseline:")
    for name, t in results.items():
        speedup = baseline_time / t
        print(f"    {name:10} | {speedup:5.2f}x")

    return results


def test_get_implementation():
    """Test implementation selector."""
    print("\n[Test 7] get_implementation()")
    print("-" * 60)

    all_passed = True
    for version in ['baseline', 'numpy', 'numba']:
        impl = get_implementation(version)
        valid = callable(impl)
        status = "PASS" if valid else "FAIL"
        if not valid:
            all_passed = False
        print(f"  {version:10} | callable={valid} | {status}")

    return all_passed


def main():
    print("=" * 60)
    print("MO-GVNS MODULE TESTS")
    print("Algorithm 7: MO-GVNS (Duarte et al. 2015)")
    print("=" * 60)

    results = []
    results.append(("initial_archive", test_initial_archive()))
    results.append(("mo_gvns_basic", test_mo_gvns_basic()))
    results.append(("time_limit", test_mo_gvns_time_limit()))
    results.append(("improves", test_mo_gvns_improves()))
    results.append(("all_versions", test_all_versions()))
    results.append(("result_container", test_result_container()))
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

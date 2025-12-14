"""
Unit Tests for Dominance Module

References:
    Duarte et al. (2015) Algorithm 1: MO-Improvement

Tests all three optimization versions: baseline, numpy, numba
"""

import numpy as np
import time
from dominance import (
    dominates_baseline, dominates_numpy, dominates_numba,
    is_dominated_baseline, is_dominated_numpy, is_dominated_numba,
    mo_improvement_baseline, mo_improvement_numpy, mo_improvement_numba,
    NUMBA_AVAILABLE, get_implementation
)


def test_dominates():
    """Test dominates function for all versions."""
    print("\n[Test 1] dominates(a, b)")
    print("-" * 50)

    test_cases = [
        (np.array([1.0, 1.0]), np.array([2.0, 2.0]), True, "a strictly better in all"),
        (np.array([1.0, 2.0]), np.array([2.0, 2.0]), True, "a equal in one, better in other"),
        (np.array([2.0, 2.0]), np.array([1.0, 1.0]), False, "b dominates a"),
        (np.array([1.0, 3.0]), np.array([2.0, 2.0]), False, "incomparable"),
        (np.array([2.0, 2.0]), np.array([2.0, 2.0]), False, "equal solutions"),
    ]

    versions = [
        ('baseline', dominates_baseline),
        ('numpy', dominates_numpy),
        ('numba', dominates_numba),
    ]

    all_passed = True
    for a, b, expected, description in test_cases:
        for name, func in versions:
            result = func(a, b)
            status = "PASS" if result == expected else "FAIL"
            if result != expected:
                all_passed = False
            print(f"  {name:10} | {description:35} | {status}")

    return all_passed


def test_is_dominated():
    """Test is_dominated function for all versions."""
    print("\n[Test 2] is_dominated(x, E)")
    print("-" * 50)

    E = np.array([
        [1.0, 3.0],
        [2.0, 2.0],
        [3.0, 1.0],
    ])

    test_cases = [
        (np.array([4.0, 4.0]), True, "dominated by all"),
        (np.array([0.5, 4.0]), False, "on Pareto front"),
        (np.array([2.5, 2.5]), True, "dominated by (2,2)"),
        (np.array([0.0, 0.0]), False, "dominates all"),
        (np.array([1.5, 2.5]), False, "incomparable with all"),
    ]

    versions = [
        ('baseline', is_dominated_baseline),
        ('numpy', is_dominated_numpy),
        ('numba', is_dominated_numba),
    ]

    all_passed = True
    for x, expected, description in test_cases:
        for name, func in versions:
            result = func(x, E)
            status = "PASS" if result == expected else "FAIL"
            if result != expected:
                all_passed = False
            print(f"  {name:10} | {description:35} | {status}")

    return all_passed


def test_mo_improvement():
    """Test MO-Improvement (Algorithm 1) for all versions."""
    print("\n[Test 3] MO-Improvement(E, E')")
    print("-" * 50)

    E = np.array([
        [2.0, 2.0],
        [1.0, 3.0],
        [3.0, 1.0],
    ])

    test_cases = [
        (np.array([[1.5, 1.5]]), True, "new non-dominated solution"),
        (np.array([[4.0, 4.0]]), False, "dominated solution"),
        (np.array([[2.0, 2.0]]), False, "solution already in E"),
        (np.array([[0.5, 0.5]]), True, "solution dominating E"),
        (np.array([[4.0, 4.0], [0.5, 0.5]]), True, "mixed: one dominates"),
        (np.array([]).reshape(0, 2), False, "empty E_prime"),
    ]

    versions = [
        ('baseline', mo_improvement_baseline),
        ('numpy', mo_improvement_numpy),
        ('numba', mo_improvement_numba),
    ]

    all_passed = True
    for E_prime, expected, description in test_cases:
        for name, func in versions:
            result = func(E, E_prime)
            status = "PASS" if result == expected else "FAIL"
            if result != expected:
                all_passed = False
            print(f"  {name:10} | {description:35} | {status}")

    return all_passed


def benchmark_versions(n_archive: int = 100, n_candidates: int = 50, n_objectives: int = 3):
    """Benchmark all versions with synthetic data."""
    print(f"\n[Benchmark] Archive={n_archive}, Candidates={n_candidates}, Objectives={n_objectives}")
    print("-" * 50)

    np.random.seed(42)
    E = np.random.rand(n_archive, n_objectives)
    E_prime = np.random.rand(n_candidates, n_objectives)

    versions = [
        ('baseline', mo_improvement_baseline),
        ('numpy', mo_improvement_numpy),
        ('numba', mo_improvement_numba),
    ]

    results = {}
    for name, func in versions:
        if name == 'numba':
            func(E[:5], E_prime[:5])

        start = time.perf_counter()
        n_runs = 10
        for _ in range(n_runs):
            func(E, E_prime)
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
    print("\n[Test 4] get_implementation()")
    print("-" * 50)

    for version in ['baseline', 'numpy', 'numba']:
        impl = get_implementation(version)
        has_all = all(k in impl for k in ['dominates', 'is_dominated', 'mo_improvement'])
        status = "PASS" if has_all else "FAIL"
        print(f"  {version:10} | all functions present | {status}")

    return True


def main():
    print("=" * 60)
    print("DOMINANCE MODULE TESTS")
    print("Algorithm 1: MO-Improvement (Duarte et al. 2015)")
    print("=" * 60)

    print(f"\nNumba available: {NUMBA_AVAILABLE}")

    results = []
    results.append(("dominates", test_dominates()))
    results.append(("is_dominated", test_is_dominated()))
    results.append(("mo_improvement", test_mo_improvement()))
    results.append(("get_implementation", test_get_implementation()))

    benchmark_versions(n_archive=100, n_candidates=50, n_objectives=3)

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

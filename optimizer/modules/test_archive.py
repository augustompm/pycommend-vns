"""
Unit Tests for Archive Module

References:
    Duarte et al. (2015) Algorithm 2: MO-NeighborhoodChange

Tests all three optimization versions: baseline, numpy, numba
"""

import numpy as np
import time
from archive import (
    update_archive_baseline, update_archive_numpy, update_archive_numba,
    mo_neighborhood_change_baseline, mo_neighborhood_change_numpy,
    mo_neighborhood_change_numba,
    NUMBA_AVAILABLE, get_implementation
)


def test_update_archive():
    """Test update_archive function for all versions."""
    print("\n[Test 1] update_archive(E, E')")
    print("-" * 60)

    E = np.array([
        [2.0, 2.0],
        [1.0, 3.0],
        [3.0, 1.0],
    ])

    test_cases = [
        (
            np.array([[1.5, 1.5]]),
            3,
            "add non-dominated (dominates [2,2])"
        ),
        (
            np.array([[0.5, 0.5]]),
            1,
            "add dominant solution (dominates all)"
        ),
        (
            np.array([[4.0, 4.0]]),
            3,
            "reject dominated solution"
        ),
        (
            np.array([[0.5, 4.0], [4.0, 0.5]]),
            5,
            "add two extreme points"
        ),
        (
            np.array([]).reshape(0, 2),
            3,
            "empty E_prime"
        ),
    ]

    versions = [
        ('baseline', update_archive_baseline),
        ('numpy', update_archive_numpy),
        ('numba', update_archive_numba),
    ]

    all_passed = True
    for E_prime, expected_size, description in test_cases:
        for name, func in versions:
            result = func(E.copy(), E_prime.copy())
            status = "PASS" if len(result) == expected_size else f"FAIL (got {len(result)})"
            if len(result) != expected_size:
                all_passed = False
            print(f"  {name:10} | {description:40} | {status}")

    return all_passed


def test_update_archive_removes_dominated():
    """Test that update_archive properly removes dominated solutions."""
    print("\n[Test 2] update_archive removes dominated")
    print("-" * 60)

    E = np.array([
        [2.0, 2.0],
        [3.0, 3.0],
    ])
    E_prime = np.array([[1.0, 1.0]])

    versions = [
        ('baseline', update_archive_baseline),
        ('numpy', update_archive_numpy),
        ('numba', update_archive_numba),
    ]

    all_passed = True
    for name, func in versions:
        result = func(E.copy(), E_prime.copy())
        expected = 1
        status = "PASS" if len(result) == expected else f"FAIL (got {len(result)})"
        if len(result) != expected:
            all_passed = False
        print(f"  {name:10} | removes all dominated ({len(E)} -> {expected}) | {status}")

    return all_passed


def test_mo_neighborhood_change():
    """Test MO-NeighborhoodChange (Algorithm 2) for all versions."""
    print("\n[Test 3] MO-NeighborhoodChange(E, E', k)")
    print("-" * 60)

    E = np.array([
        [2.0, 2.0],
        [1.0, 3.0],
        [3.0, 1.0],
    ])

    test_cases = [
        (np.array([[1.5, 1.5]]), 3, 1, "k resets to 1 on improvement"),
        (np.array([[4.0, 4.0]]), 3, 4, "k increments on no improvement"),
        (np.array([[2.0, 2.0]]), 5, 6, "k increments for duplicate"),
    ]

    versions = [
        ('baseline', mo_neighborhood_change_baseline),
        ('numpy', mo_neighborhood_change_numpy),
        ('numba', mo_neighborhood_change_numba),
    ]

    all_passed = True
    for E_prime, k_in, k_expected, description in test_cases:
        for name, func in versions:
            E_out, k_out = func(E.copy(), E_prime.copy(), k_in)
            status = "PASS" if k_out == k_expected else f"FAIL (k={k_out})"
            if k_out != k_expected:
                all_passed = False
            print(f"  {name:10} | {description:40} | {status}")

    return all_passed


def test_empty_archives():
    """Test edge cases with empty archives."""
    print("\n[Test 4] Empty archive edge cases")
    print("-" * 60)

    empty_E = np.array([]).reshape(0, 2)
    single_point = np.array([[1.0, 1.0]])

    versions = [
        ('baseline', update_archive_baseline),
        ('numpy', update_archive_numpy),
        ('numba', update_archive_numba),
    ]

    all_passed = True

    for name, func in versions:
        result = func(empty_E.copy(), single_point.copy())
        status = "PASS" if len(result) == 1 else f"FAIL (got {len(result)})"
        if len(result) != 1:
            all_passed = False
        print(f"  {name:10} | empty E, one E_prime                   | {status}")

    for name, func in versions:
        result = func(single_point.copy(), empty_E.copy())
        status = "PASS" if len(result) == 1 else f"FAIL (got {len(result)})"
        if len(result) != 1:
            all_passed = False
        print(f"  {name:10} | one E, empty E_prime                   | {status}")

    return all_passed


def benchmark_versions(n_archive: int = 100, n_candidates: int = 50, n_objectives: int = 3):
    """Benchmark all versions with synthetic data."""
    print(f"\n[Benchmark] Archive={n_archive}, Candidates={n_candidates}, Objectives={n_objectives}")
    print("-" * 60)

    np.random.seed(42)
    E = np.random.rand(n_archive, n_objectives)
    E_prime = np.random.rand(n_candidates, n_objectives)

    versions = [
        ('baseline', update_archive_baseline),
        ('numpy', update_archive_numpy),
        ('numba', update_archive_numba),
    ]

    results = {}
    for name, func in versions:
        if name == 'numba':
            func(E[:5].copy(), E_prime[:5].copy())

        start = time.perf_counter()
        n_runs = 5
        for _ in range(n_runs):
            func(E.copy(), E_prime.copy())
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
        has_all = all(k in impl for k in ['update_archive', 'mo_neighborhood_change'])
        status = "PASS" if has_all else "FAIL"
        print(f"  {version:10} | all functions present                  | {status}")

    return True


def main():
    print("=" * 60)
    print("ARCHIVE MODULE TESTS")
    print("Algorithm 2: MO-NeighborhoodChange (Duarte et al. 2015)")
    print("=" * 60)

    print(f"\nNumba available: {NUMBA_AVAILABLE}")

    results = []
    results.append(("update_archive", test_update_archive()))
    results.append(("removes_dominated", test_update_archive_removes_dominated()))
    results.append(("mo_neighborhood_change", test_mo_neighborhood_change()))
    results.append(("empty_archives", test_empty_archives()))
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

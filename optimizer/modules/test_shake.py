"""
Unit Tests for Shake Module

References:
    Duarte et al. (2015) Algorithm 3: MO-Shake

Tests all three optimization versions: baseline, numpy, numba
"""

import numpy as np
import time
from shake import (
    shake_solution_baseline, shake_solution_numpy, shake_solution_numba,
    mo_shake_baseline, mo_shake_numpy, mo_shake_numba,
    NUMBA_AVAILABLE, get_implementation
)


def test_shake_solution_basic():
    """Test basic shake_solution functionality."""
    print("\n[Test 1] shake_solution basic functionality")
    print("-" * 60)

    rng = np.random.default_rng(42)
    solution = np.array([0, 1, 2, 3, 4])
    n_packages = 100
    k = 2

    versions = [
        ('baseline', shake_solution_baseline),
        ('numpy', shake_solution_numpy),
        ('numba', shake_solution_numba),
    ]

    all_passed = True
    for name, func in versions:
        rng_local = np.random.default_rng(42)
        result = func(solution.copy(), k, n_packages, rng_local)

        same_length = len(result) == len(solution)
        all_valid = all(0 <= p < n_packages for p in result)
        all_unique = len(set(result)) == len(result)
        some_different = not np.array_equal(result, solution)

        passed = same_length and all_valid and all_unique
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        print(f"  {name:10} | length={same_length}, valid={all_valid}, unique={all_unique}, changed={some_different} | {status}")

    return all_passed


def test_shake_preserves_size():
    """Test that shake preserves solution size."""
    print("\n[Test 2] shake preserves solution size")
    print("-" * 60)

    versions = [
        ('baseline', shake_solution_baseline),
        ('numpy', shake_solution_numpy),
        ('numba', shake_solution_numba),
    ]

    test_cases = [
        (5, 100, 1),
        (10, 50, 3),
        (20, 100, 5),
    ]

    all_passed = True
    for sol_size, n_packages, k in test_cases:
        for name, func in versions:
            rng = np.random.default_rng(42)
            solution = np.arange(sol_size)
            result = func(solution.copy(), k, n_packages, rng)

            passed = len(result) == sol_size
            status = "PASS" if passed else f"FAIL (got {len(result)})"
            if not passed:
                all_passed = False

            print(f"  {name:10} | size={sol_size}, k={k} | {status}")

    return all_passed


def test_shake_different_k():
    """Test shake with different neighborhood intensities."""
    print("\n[Test 3] shake with different k values")
    print("-" * 60)

    solution = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    n_packages = 100

    versions = [
        ('baseline', shake_solution_baseline),
        ('numpy', shake_solution_numpy),
        ('numba', shake_solution_numba),
    ]

    all_passed = True
    for k in [1, 3, 5, 10]:
        for name, func in versions:
            rng = np.random.default_rng(42)
            result = func(solution.copy(), k, n_packages, rng)

            n_diff = np.sum(~np.isin(result, solution))
            k_actual = min(k, len(solution))

            passed = n_diff <= k_actual
            status = "PASS" if passed else f"FAIL (diff={n_diff})"
            if not passed:
                all_passed = False

            print(f"  {name:10} | k={k:2}, changes={n_diff} (max={k_actual}) | {status}")

    return all_passed


def test_mo_shake():
    """Test MO-Shake (Algorithm 3) for all versions."""
    print("\n[Test 4] MO-Shake(E, k)")
    print("-" * 60)

    E_solutions = [
        np.array([0, 1, 2, 3, 4]),
        np.array([5, 6, 7, 8, 9]),
        np.array([10, 11, 12, 13, 14]),
    ]
    n_packages = 100
    k = 2

    versions = [
        ('baseline', mo_shake_baseline),
        ('numpy', mo_shake_numpy),
        ('numba', mo_shake_numba),
    ]

    all_passed = True
    for name, func in versions:
        rng = np.random.default_rng(42)
        E_copy = [s.copy() for s in E_solutions]
        E_prime = func(E_copy, k, n_packages, rng)

        same_count = len(E_prime) == len(E_solutions)
        all_same_size = all(len(ep) == len(es) for ep, es in zip(E_prime, E_solutions))

        passed = same_count and all_same_size
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        print(f"  {name:10} | count={same_count}, sizes={all_same_size} | {status}")

    return all_passed


def test_edge_cases():
    """Test edge cases."""
    print("\n[Test 5] Edge cases")
    print("-" * 60)

    versions = [
        ('baseline', shake_solution_baseline),
        ('numpy', shake_solution_numpy),
        ('numba', shake_solution_numba),
    ]

    all_passed = True

    for name, func in versions:
        rng = np.random.default_rng(42)
        solution = np.array([0, 1, 2, 3, 4])
        result = func(solution.copy(), 0, 100, rng)

        passed = np.array_equal(result, solution)
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        print(f"  {name:10} | k=0 (no change expected)               | {status}")

    for name, func in versions:
        rng = np.random.default_rng(42)
        solution = np.arange(100)
        result = func(solution.copy(), 5, 100, rng)

        passed = np.array_equal(np.sort(result), np.sort(solution))
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        print(f"  {name:10} | all packages used (no swap possible)   | {status}")

    return all_passed


def benchmark_versions(sol_size: int = 20, n_packages: int = 1000, n_solutions: int = 50):
    """Benchmark all versions."""
    print(f"\n[Benchmark] Solutions={n_solutions}, Size={sol_size}, Packages={n_packages}")
    print("-" * 60)

    rng = np.random.default_rng(42)
    E_solutions = [
        rng.choice(n_packages, size=sol_size, replace=False)
        for _ in range(n_solutions)
    ]
    k = 3

    versions = [
        ('baseline', mo_shake_baseline),
        ('numpy', mo_shake_numpy),
        ('numba', mo_shake_numba),
    ]

    results = {}
    for name, func in versions:
        if name == 'numba' and NUMBA_AVAILABLE:
            rng_warmup = np.random.default_rng(0)
            func([E_solutions[0].copy()], k, n_packages, rng_warmup)

        start = time.perf_counter()
        n_runs = 10
        for _ in range(n_runs):
            rng_run = np.random.default_rng(42)
            E_copy = [s.copy() for s in E_solutions]
            func(E_copy, k, n_packages, rng_run)
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
    print("\n[Test 6] get_implementation()")
    print("-" * 60)

    for version in ['baseline', 'numpy', 'numba']:
        impl = get_implementation(version)
        has_all = all(k in impl for k in ['shake_solution', 'mo_shake'])
        status = "PASS" if has_all else "FAIL"
        print(f"  {version:10} | all functions present                  | {status}")

    return True


def main():
    print("=" * 60)
    print("SHAKE MODULE TESTS")
    print("Algorithm 3: MO-Shake (Duarte et al. 2015)")
    print("=" * 60)

    print(f"\nNumba available: {NUMBA_AVAILABLE}")

    results = []
    results.append(("shake_basic", test_shake_solution_basic()))
    results.append(("preserves_size", test_shake_preserves_size()))
    results.append(("different_k", test_shake_different_k()))
    results.append(("mo_shake", test_mo_shake()))
    results.append(("edge_cases", test_edge_cases()))
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

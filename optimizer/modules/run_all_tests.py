"""
Comprehensive Test Runner for MO-GVNS Modules

References:
    Duarte, A., Marti, R., Alvarez, A., & Angel-Bello, F. (2015).
    Multi-objective variable neighborhood search: an application to
    combinatorial optimization problems. J Glob Optim, 63, 515-536.

Runs all unit tests for the canonical MO-GVNS implementation:
    - Algorithm 1: MO-Improvement (dominance.py)
    - Algorithm 2: MO-NeighborhoodChange (archive.py)
    - Algorithm 3: MO-Shake (shake.py)
    - Algorithm 5: VND-i (vnd.py)
    - Algorithm 6: MO-VND (vnd.py)
    - Algorithm 7: MO-GVNS (mo_gvns.py)
"""

import sys
import time


def run_test_module(module_name: str) -> bool:
    """Run tests for a specific module."""
    print(f"\n{'='*70}")
    print(f"Running {module_name} tests...")
    print('='*70)

    try:
        if module_name == 'dominance':
            from test_dominance import main
        elif module_name == 'archive':
            from test_archive import main
        elif module_name == 'shake':
            from test_shake import main
        elif module_name == 'vnd':
            from test_vnd import main
        elif module_name == 'mo_gvns':
            from test_mo_gvns import main
        else:
            print(f"Unknown module: {module_name}")
            return False

        return main()

    except Exception as e:
        print(f"Error running {module_name} tests: {e}")
        return False


def main():
    """Run all tests."""
    print("="*70)
    print("MO-GVNS COMPREHENSIVE TEST SUITE")
    print("Canonical Implementation (Duarte et al. 2015)")
    print("="*70)

    start_time = time.perf_counter()

    modules = [
        ('dominance', 'Algorithm 1: MO-Improvement'),
        ('archive', 'Algorithm 2: MO-NeighborhoodChange'),
        ('shake', 'Algorithm 3: MO-Shake'),
        ('vnd', 'Algorithms 5-6: VND-i and MO-VND'),
        ('mo_gvns', 'Algorithm 7: MO-GVNS'),
    ]

    results = []
    for module, description in modules:
        passed = run_test_module(module)
        results.append((module, description, passed))

    elapsed = time.perf_counter() - start_time

    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    all_passed = True
    for module, description, passed in results:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  {module:15} | {description:40} | {status}")

    print(f"\nTotal time: {elapsed:.2f}s")
    print("\n" + ("ALL MODULES PASSED" if all_passed else "SOME MODULES FAILED"))

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())

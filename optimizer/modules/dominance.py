"""
Dominance and MO-Improvement Module

References:
    Duarte, A., Marti, R., Alvarez, A., & Angel-Bello, F. (2015).
    Multi-objective variable neighborhood search: an application to
    combinatorial optimization problems. J Glob Optim, 63, 515-536.
    Algorithm 1: MO-Improvement (page 519)

This module implements Pareto dominance checking and the MO-Improvement
procedure from Duarte et al. (2015). Three optimization versions are
provided for performance comparison.
"""

import numpy as np
from typing import Set, Tuple

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


def dominates_baseline(a: np.ndarray, b: np.ndarray) -> bool:
    """
    Check if solution a dominates solution b using pure Python.

    Parameters:
        a: Objective values of solution a (to be minimized)
        b: Objective values of solution b (to be minimized)

    Returns:
        True if a dominates b (a is better or equal in all objectives
        and strictly better in at least one)
    """
    dominated = True
    strictly_better = False

    for i in range(len(a)):
        if a[i] > b[i]:
            dominated = False
            break
        if a[i] < b[i]:
            strictly_better = True

    return dominated and strictly_better


def dominates_numpy(a: np.ndarray, b: np.ndarray) -> bool:
    """
    Check if solution a dominates solution b using NumPy vectorization.

    Parameters:
        a: Objective values of solution a (to be minimized)
        b: Objective values of solution b (to be minimized)

    Returns:
        True if a dominates b
    """
    return np.all(a <= b) and np.any(a < b)


if NUMBA_AVAILABLE:
    @jit(nopython=True, cache=True)
    def dominates_numba(a: np.ndarray, b: np.ndarray) -> bool:
        """
        Check if solution a dominates solution b using Numba JIT.

        Parameters:
            a: Objective values of solution a (to be minimized)
            b: Objective values of solution b (to be minimized)

        Returns:
            True if a dominates b
        """
        dominated = True
        strictly_better = False

        for i in range(len(a)):
            if a[i] > b[i]:
                dominated = False
                break
            if a[i] < b[i]:
                strictly_better = True

        return dominated and strictly_better
else:
    dominates_numba = dominates_baseline


def is_dominated_baseline(x: np.ndarray, E: np.ndarray) -> bool:
    """
    Check if x is dominated by any solution in E using pure Python.

    Parameters:
        x: Objective values of solution x
        E: Array of shape (n_solutions, n_objectives) representing the archive

    Returns:
        True if x is dominated by at least one solution in E
    """
    for i in range(len(E)):
        if dominates_baseline(E[i], x):
            return True
    return False


def is_dominated_numpy(x: np.ndarray, E: np.ndarray) -> bool:
    """
    Check if x is dominated by any solution in E using NumPy vectorization.

    Parameters:
        x: Objective values of solution x
        E: Array of shape (n_solutions, n_objectives) representing the archive

    Returns:
        True if x is dominated by at least one solution in E
    """
    if len(E) == 0:
        return False

    leq = np.all(E <= x, axis=1)
    lt = np.any(E < x, axis=1)

    return np.any(leq & lt)


if NUMBA_AVAILABLE:
    @jit(nopython=True, cache=True)
    def is_dominated_numba(x: np.ndarray, E: np.ndarray) -> bool:
        """
        Check if x is dominated by any solution in E using Numba JIT.

        Parameters:
            x: Objective values of solution x
            E: Array of shape (n_solutions, n_objectives) representing the archive

        Returns:
            True if x is dominated by at least one solution in E
        """
        for i in range(len(E)):
            if dominates_numba(E[i], x):
                return True
        return False
else:
    is_dominated_numba = is_dominated_baseline


def mo_improvement_baseline(E: np.ndarray, E_prime: np.ndarray) -> bool:
    """
    Algorithm 1: MO-Improvement from Duarte et al. (2015) using pure Python.

    Tests whether E_prime contains at least one solution that is not in E
    and is not dominated by any solution in E.

    Parameters:
        E: Current archive (n_solutions, n_objectives)
        E_prime: New candidate solutions (n_candidates, n_objectives)

    Returns:
        True if improvement found (E_prime contains non-dominated solutions not in E)
    """
    E_set = set(map(tuple, E))

    for i in range(len(E_prime)):
        x = E_prime[i]
        x_tuple = tuple(x)

        if x_tuple not in E_set and not is_dominated_baseline(x, E):
            return True

    return False


def mo_improvement_numpy(E: np.ndarray, E_prime: np.ndarray) -> bool:
    """
    Algorithm 1: MO-Improvement from Duarte et al. (2015) using NumPy.

    Tests whether E_prime contains at least one solution that is not in E
    and is not dominated by any solution in E.

    Parameters:
        E: Current archive (n_solutions, n_objectives)
        E_prime: New candidate solutions (n_candidates, n_objectives)

    Returns:
        True if improvement found
    """
    if len(E) == 0:
        return len(E_prime) > 0

    if len(E_prime) == 0:
        return False

    E_set = set(map(tuple, E))

    for i in range(len(E_prime)):
        x = E_prime[i]
        x_tuple = tuple(x)

        if x_tuple not in E_set and not is_dominated_numpy(x, E):
            return True

    return False


if NUMBA_AVAILABLE:
    @jit(nopython=True, cache=True)
    def _solutions_equal(a: np.ndarray, b: np.ndarray) -> bool:
        """Check if two solution vectors are equal."""
        for i in range(len(a)):
            if a[i] != b[i]:
                return False
        return True

    @jit(nopython=True, cache=True)
    def _in_archive(x: np.ndarray, E: np.ndarray) -> bool:
        """Check if x is already in archive E."""
        for i in range(len(E)):
            if _solutions_equal(x, E[i]):
                return True
        return False

    @jit(nopython=True, cache=True)
    def mo_improvement_numba(E: np.ndarray, E_prime: np.ndarray) -> bool:
        """
        Algorithm 1: MO-Improvement from Duarte et al. (2015) using Numba JIT.

        Parameters:
            E: Current archive (n_solutions, n_objectives)
            E_prime: New candidate solutions (n_candidates, n_objectives)

        Returns:
            True if improvement found
        """
        if len(E) == 0:
            return len(E_prime) > 0

        if len(E_prime) == 0:
            return False

        for i in range(len(E_prime)):
            x = E_prime[i]

            if not _in_archive(x, E) and not is_dominated_numba(x, E):
                return True

        return False
else:
    mo_improvement_numba = mo_improvement_baseline


IMPLEMENTATIONS = {
    'baseline': {
        'dominates': dominates_baseline,
        'is_dominated': is_dominated_baseline,
        'mo_improvement': mo_improvement_baseline,
    },
    'numpy': {
        'dominates': dominates_numpy,
        'is_dominated': is_dominated_numpy,
        'mo_improvement': mo_improvement_numpy,
    },
    'numba': {
        'dominates': dominates_numba,
        'is_dominated': is_dominated_numba,
        'mo_improvement': mo_improvement_numba,
    },
}


def get_implementation(version: str = 'numpy') -> dict:
    """
    Get the implementation functions for the specified version.

    Parameters:
        version: One of 'baseline', 'numpy', 'numba'

    Returns:
        Dictionary with dominates, is_dominated, and mo_improvement functions
    """
    if version not in IMPLEMENTATIONS:
        raise ValueError(f"Unknown version: {version}. Use: {list(IMPLEMENTATIONS.keys())}")

    if version == 'numba' and not NUMBA_AVAILABLE:
        print("Warning: Numba not available, falling back to baseline")
        return IMPLEMENTATIONS['baseline']

    return IMPLEMENTATIONS[version]

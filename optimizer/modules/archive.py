"""
Archive Management Module (MO-NeighborhoodChange)

References:
    Duarte, A., Marti, R., Alvarez, A., & Angel-Bello, F. (2015).
    Multi-objective variable neighborhood search: an application to
    combinatorial optimization problems. J Glob Optim, 63, 515-536.
    Algorithm 2: MO-NeighborhoodChange (page 519)

Archive maintenance:
    Cost is O(n*m) per update where n=archive size, m=new candidates.
    Continuous pruning removes dominated solutions immediately (Duarte p.519).
    This avoids post-hoc pruning and keeps archive bounded during search.

This module implements archive update and MO-NeighborhoodChange from Duarte (2015).
"""

import numpy as np
from typing import Tuple

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

from dominance import (
    dominates_baseline, dominates_numpy,
    is_dominated_baseline, is_dominated_numpy,
    mo_improvement_baseline, mo_improvement_numpy, mo_improvement_numba
)

if NUMBA_AVAILABLE:
    from dominance import dominates_numba, is_dominated_numba


def update_archive_baseline(E: np.ndarray, E_prime: np.ndarray) -> np.ndarray:
    """
    Update archive by merging E and E_prime, keeping only non-dominated solutions.

    Parameters:
        E: Current archive (n_solutions, n_objectives)
        E_prime: New candidate solutions (n_candidates, n_objectives)

    Returns:
        Updated archive containing only non-dominated solutions from E ∪ E_prime
    """
    if len(E) == 0:
        return E_prime.copy() if len(E_prime) > 0 else E.copy()

    if len(E_prime) == 0:
        return E.copy()

    combined = np.vstack([E, E_prime])
    n = len(combined)
    is_dominated_mask = np.zeros(n, dtype=bool)

    for i in range(n):
        if is_dominated_mask[i]:
            continue
        for j in range(n):
            if i != j and not is_dominated_mask[j]:
                if dominates_baseline(combined[j], combined[i]):
                    is_dominated_mask[i] = True
                    break

    non_dominated = combined[~is_dominated_mask]

    unique_solutions = []
    seen = set()
    for sol in non_dominated:
        sol_tuple = tuple(sol)
        if sol_tuple not in seen:
            seen.add(sol_tuple)
            unique_solutions.append(sol)

    if len(unique_solutions) == 0:
        return E.copy()

    return np.array(unique_solutions)


def update_archive_numpy(E: np.ndarray, E_prime: np.ndarray) -> np.ndarray:
    """
    Update archive using NumPy vectorization.

    Parameters:
        E: Current archive (n_solutions, n_objectives)
        E_prime: New candidate solutions (n_candidates, n_objectives)

    Returns:
        Updated archive containing only non-dominated solutions
    """
    if len(E) == 0:
        return E_prime.copy() if len(E_prime) > 0 else E.copy()

    if len(E_prime) == 0:
        return E.copy()

    combined = np.vstack([E, E_prime])
    n = len(combined)

    is_dominated_mask = np.zeros(n, dtype=bool)

    for i in range(n):
        if is_dominated_mask[i]:
            continue

        others = combined[~is_dominated_mask]
        others_mask = np.arange(n)[~is_dominated_mask]

        leq = np.all(others <= combined[i], axis=1)
        lt = np.any(others < combined[i], axis=1)
        dominators = leq & lt

        dominators[others_mask == i] = False

        if np.any(dominators):
            is_dominated_mask[i] = True

    non_dominated = combined[~is_dominated_mask]

    if len(non_dominated) == 0:
        return E.copy()

    unique_indices = np.unique(non_dominated, axis=0, return_index=True)[1]
    unique_solutions = non_dominated[np.sort(unique_indices)]

    return unique_solutions


if NUMBA_AVAILABLE:
    @jit(nopython=True, cache=True)
    def update_archive_numba(E: np.ndarray, E_prime: np.ndarray) -> np.ndarray:
        """
        Update archive using Numba JIT.

        Parameters:
            E: Current archive (n_solutions, n_objectives)
            E_prime: New candidate solutions (n_candidates, n_objectives)

        Returns:
            Updated archive containing only non-dominated solutions
        """
        if len(E) == 0:
            if len(E_prime) > 0:
                return E_prime.copy()
            return E.copy()

        if len(E_prime) == 0:
            return E.copy()

        combined = np.vstack((E, E_prime))
        n = len(combined)
        is_dominated_mask = np.zeros(n, dtype=np.bool_)

        for i in range(n):
            if is_dominated_mask[i]:
                continue
            for j in range(n):
                if i != j and not is_dominated_mask[j]:
                    if dominates_numba(combined[j], combined[i]):
                        is_dominated_mask[i] = True
                        break

        count = 0
        for i in range(n):
            if not is_dominated_mask[i]:
                count += 1

        if count == 0:
            return E.copy()

        result = np.empty((count, combined.shape[1]), dtype=combined.dtype)
        idx = 0
        for i in range(n):
            if not is_dominated_mask[i]:
                result[idx] = combined[i]
                idx += 1

        return result
else:
    update_archive_numba = update_archive_baseline


def mo_neighborhood_change_baseline(
    E: np.ndarray,
    E_prime: np.ndarray,
    k: int
) -> Tuple[np.ndarray, int]:
    """
    Algorithm 2: MO-NeighborhoodChange from Duarte et al. (2015) using baseline.

    If improvement is found (E_prime contains non-dominated solutions not in E),
    update the archive and reset k to 1. Otherwise, increment k.

    Parameters:
        E: Current archive (n_solutions, n_objectives)
        E_prime: New candidate solutions (n_candidates, n_objectives)
        k: Current neighborhood index

    Returns:
        Tuple of (updated_archive, new_k)
    """
    if mo_improvement_baseline(E, E_prime):
        E_new = update_archive_baseline(E, E_prime)
        return E_new, 1
    else:
        return E, k + 1


def mo_neighborhood_change_numpy(
    E: np.ndarray,
    E_prime: np.ndarray,
    k: int
) -> Tuple[np.ndarray, int]:
    """
    Algorithm 2: MO-NeighborhoodChange using NumPy vectorization.

    Parameters:
        E: Current archive (n_solutions, n_objectives)
        E_prime: New candidate solutions (n_candidates, n_objectives)
        k: Current neighborhood index

    Returns:
        Tuple of (updated_archive, new_k)
    """
    if mo_improvement_numpy(E, E_prime):
        E_new = update_archive_numpy(E, E_prime)
        return E_new, 1
    else:
        return E, k + 1


def mo_neighborhood_change_numba(
    E: np.ndarray,
    E_prime: np.ndarray,
    k: int
) -> Tuple[np.ndarray, int]:
    """
    Algorithm 2: MO-NeighborhoodChange using Numba JIT.

    Parameters:
        E: Current archive (n_solutions, n_objectives)
        E_prime: New candidate solutions (n_candidates, n_objectives)
        k: Current neighborhood index

    Returns:
        Tuple of (updated_archive, new_k)
    """
    if mo_improvement_numba(E, E_prime):
        E_new = update_archive_numba(E, E_prime)
        return E_new, 1
    else:
        return E, k + 1


IMPLEMENTATIONS = {
    'baseline': {
        'update_archive': update_archive_baseline,
        'mo_neighborhood_change': mo_neighborhood_change_baseline,
    },
    'numpy': {
        'update_archive': update_archive_numpy,
        'mo_neighborhood_change': mo_neighborhood_change_numpy,
    },
    'numba': {
        'update_archive': update_archive_numba,
        'mo_neighborhood_change': mo_neighborhood_change_numba,
    },
}


def get_implementation(version: str = 'numpy') -> dict:
    """
    Get the implementation functions for the specified version.

    Parameters:
        version: One of 'baseline', 'numpy', 'numba'

    Returns:
        Dictionary with update_archive and mo_neighborhood_change functions
    """
    if version not in IMPLEMENTATIONS:
        raise ValueError(f"Unknown version: {version}. Use: {list(IMPLEMENTATIONS.keys())}")

    if version == 'numba' and not NUMBA_AVAILABLE:
        print("Warning: Numba not available, falling back to baseline")
        return IMPLEMENTATIONS['baseline']

    return IMPLEMENTATIONS[version]

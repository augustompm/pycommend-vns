"""
Shake Module (MO-Shake)

References:
    Duarte, A., Marti, R., Alvarez, A., & Angel-Bello, F. (2015).
    Multi-objective variable neighborhood search: an application to
    combinatorial optimization problems. J Glob Optim, 63, 515-536.
    Algorithm 3: MO-Shake (page 519)

    Hansen, P., Mladenovic, N., Todosijevic, R., & Hanafi, S. (2017).
    Variable neighborhood search: basics and variants. EURO J Comput Optim,
    5, 423-454.

This module implements the shaking procedure from Duarte et al. (2015).
The shake function perturbs solutions to escape local optima using
neighborhood structures of increasing intensity.
"""

import numpy as np
from typing import List, Tuple, Optional

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


def shake_solution_baseline(
    solution: np.ndarray,
    k: int,
    n_packages: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Perturb a single solution by swapping k packages.

    Parameters:
        solution: Array of package indices in the current recommendation
        k: Neighborhood index (number of packages to swap)
        n_packages: Total number of available packages
        rng: NumPy random generator

    Returns:
        Perturbed solution with k packages swapped
    """
    solution_new = solution.copy()
    solution_set = set(solution)

    available = [i for i in range(n_packages) if i not in solution_set]

    if len(available) == 0:
        return solution_new

    k_actual = min(k, len(solution), len(available))

    if k_actual == 0:
        return solution_new

    positions_to_swap = rng.choice(len(solution), size=k_actual, replace=False)
    new_packages = rng.choice(available, size=k_actual, replace=False)

    for i, pos in enumerate(positions_to_swap):
        solution_new[pos] = new_packages[i]

    return solution_new


def shake_solution_numpy(
    solution: np.ndarray,
    k: int,
    n_packages: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Perturb a single solution using NumPy operations.

    Parameters:
        solution: Array of package indices in the current recommendation
        k: Neighborhood index (number of packages to swap)
        n_packages: Total number of available packages
        rng: NumPy random generator

    Returns:
        Perturbed solution with k packages swapped
    """
    solution_new = solution.copy()

    all_packages = np.arange(n_packages)
    mask = np.isin(all_packages, solution, invert=True)
    available = all_packages[mask]

    if len(available) == 0:
        return solution_new

    k_actual = min(k, len(solution), len(available))

    if k_actual == 0:
        return solution_new

    positions_to_swap = rng.choice(len(solution), size=k_actual, replace=False)
    new_packages = rng.choice(available, size=k_actual, replace=False)

    solution_new[positions_to_swap] = new_packages

    return solution_new


if NUMBA_AVAILABLE:
    @jit(nopython=True, cache=True)
    def _shake_solution_numba_core(
        solution: np.ndarray,
        k: int,
        n_packages: int,
        positions_to_swap: np.ndarray,
        new_package_indices: np.ndarray
    ) -> np.ndarray:
        """Core shake operation for Numba."""
        solution_new = solution.copy()

        solution_set = set()
        for s in solution:
            solution_set.add(s)

        available = np.empty(n_packages, dtype=np.int64)
        avail_count = 0
        for i in range(n_packages):
            if i not in solution_set:
                available[avail_count] = i
                avail_count += 1

        if avail_count == 0:
            return solution_new

        k_actual = min(k, len(solution), avail_count)

        if k_actual == 0:
            return solution_new

        for i in range(k_actual):
            pos = positions_to_swap[i % len(positions_to_swap)]
            new_idx = new_package_indices[i % len(new_package_indices)]
            if new_idx < avail_count:
                solution_new[pos] = available[new_idx]

        return solution_new

    def shake_solution_numba(
        solution: np.ndarray,
        k: int,
        n_packages: int,
        rng: np.random.Generator
    ) -> np.ndarray:
        """
        Perturb a single solution using Numba JIT.

        Parameters:
            solution: Array of package indices
            k: Neighborhood index
            n_packages: Total number of available packages
            rng: NumPy random generator

        Returns:
            Perturbed solution
        """
        k_actual = min(k, len(solution), n_packages - len(solution))

        if k_actual == 0:
            return solution.copy()

        positions = rng.choice(len(solution), size=k_actual, replace=False)
        indices = rng.integers(0, n_packages, size=k_actual)

        return _shake_solution_numba_core(
            solution, k, n_packages, positions, indices
        )
else:
    shake_solution_numba = shake_solution_baseline


def mo_shake_baseline(
    E_solutions: List[np.ndarray],
    k: int,
    n_packages: int,
    rng: Optional[np.random.Generator] = None
) -> List[np.ndarray]:
    """
    Algorithm 3: MO-Shake from Duarte et al. (2015) using baseline.

    Perturbs all solutions in the archive E using neighborhood k.

    Parameters:
        E_solutions: List of solution arrays (each is array of package indices)
        k: Neighborhood index (intensity of perturbation)
        n_packages: Total number of available packages
        rng: NumPy random generator (optional)

    Returns:
        E_prime: List of perturbed solutions
    """
    if rng is None:
        rng = np.random.default_rng()

    E_prime = []
    for solution in E_solutions:
        solution_new = shake_solution_baseline(solution, k, n_packages, rng)
        E_prime.append(solution_new)

    return E_prime


def mo_shake_numpy(
    E_solutions: List[np.ndarray],
    k: int,
    n_packages: int,
    rng: Optional[np.random.Generator] = None
) -> List[np.ndarray]:
    """
    Algorithm 3: MO-Shake using NumPy operations.

    Parameters:
        E_solutions: List of solution arrays
        k: Neighborhood index
        n_packages: Total number of available packages
        rng: NumPy random generator

    Returns:
        E_prime: List of perturbed solutions
    """
    if rng is None:
        rng = np.random.default_rng()

    E_prime = []
    for solution in E_solutions:
        solution_new = shake_solution_numpy(solution, k, n_packages, rng)
        E_prime.append(solution_new)

    return E_prime


def mo_shake_numba(
    E_solutions: List[np.ndarray],
    k: int,
    n_packages: int,
    rng: Optional[np.random.Generator] = None
) -> List[np.ndarray]:
    """
    Algorithm 3: MO-Shake using Numba JIT.

    Parameters:
        E_solutions: List of solution arrays
        k: Neighborhood index
        n_packages: Total number of available packages
        rng: NumPy random generator

    Returns:
        E_prime: List of perturbed solutions
    """
    if rng is None:
        rng = np.random.default_rng()

    E_prime = []
    for solution in E_solutions:
        solution_new = shake_solution_numba(solution, k, n_packages, rng)
        E_prime.append(solution_new)

    return E_prime


IMPLEMENTATIONS = {
    'baseline': {
        'shake_solution': shake_solution_baseline,
        'mo_shake': mo_shake_baseline,
    },
    'numpy': {
        'shake_solution': shake_solution_numpy,
        'mo_shake': mo_shake_numpy,
    },
    'numba': {
        'shake_solution': shake_solution_numba,
        'mo_shake': mo_shake_numba,
    },
}


def get_implementation(version: str = 'numpy') -> dict:
    """
    Get the implementation functions for the specified version.

    Parameters:
        version: One of 'baseline', 'numpy', 'numba'

    Returns:
        Dictionary with shake_solution and mo_shake functions
    """
    if version not in IMPLEMENTATIONS:
        raise ValueError(f"Unknown version: {version}. Use: {list(IMPLEMENTATIONS.keys())}")

    if version == 'numba' and not NUMBA_AVAILABLE:
        print("Warning: Numba not available, falling back to baseline")
        return IMPLEMENTATIONS['baseline']

    return IMPLEMENTATIONS[version]

"""
Variable Neighborhood Descent Module (VND-i and MO-VND)

References:
    Duarte, A., Marti, R., Alvarez, A., & Angel-Bello, F. (2015).
    Multi-objective variable neighborhood search: an application to
    combinatorial optimization problems. J Glob Optim, 63, 515-536.
    Algorithm 5: VND-i (page 520)
    Algorithm 6: MO-VND (page 520)

    Hansen, P., Mladenovic, N., Todosijevic, R., & Hanafi, S. (2017).
    Variable neighborhood search: basics and variants. EURO J Comput Optim,
    5, 423-454.

Default parameters:
    max_neighbors=100: limits neighborhood exploration per iteration
    max_neighbors_per_k=50: sampling size for large neighborhoods (efficiency)
    Uses first-improvement strategy with random sampling as in Hansen (2017)

This module implements VND-i (single objective) and MO-VND (all objectives).
"""

import numpy as np
from typing import Callable, List, Tuple, Optional

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


def local_search_swap_baseline(
    solution: np.ndarray,
    objective_idx: int,
    evaluate_func: Callable,
    n_packages: int,
    k_prime: int,
    rng: np.random.Generator,
    max_neighbors: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the best neighbor in N_k'(x) for objective i using baseline.

    Neighborhood N_k' swaps k' packages with candidates from the package pool.
    Uses first-improvement strategy with sampling for efficiency.

    Parameters:
        solution: Current solution (array of package indices)
        objective_idx: Index of objective to optimize (minimization)
        evaluate_func: Function(solution) -> array of objective values
        n_packages: Total number of available packages
        k_prime: Neighborhood index (number of swaps)
        rng: NumPy random generator
        max_neighbors: Maximum neighbors to evaluate per iteration

    Returns:
        Tuple of (best_solution, best_objectives)
    """
    current_obj = evaluate_func(solution)
    best_solution = solution.copy()
    best_obj = current_obj.copy()

    solution_set = set(solution)
    available = [i for i in range(n_packages) if i not in solution_set]

    if len(available) == 0:
        return best_solution, best_obj

    k_actual = min(k_prime, len(solution), len(available))

    if k_actual == 0:
        return best_solution, best_obj

    neighbors_checked = 0

    positions = list(range(len(solution)))
    rng.shuffle(positions)

    for pos_combo_start in range(0, len(positions), k_actual):
        if neighbors_checked >= max_neighbors:
            break

        swap_positions = positions[pos_combo_start:pos_combo_start + k_actual]
        if len(swap_positions) < k_actual:
            continue

        for _ in range(min(10, len(available))):
            if neighbors_checked >= max_neighbors:
                break

            new_packages = rng.choice(available, size=len(swap_positions), replace=False)

            neighbor = solution.copy()
            for i, pos in enumerate(swap_positions):
                neighbor[pos] = new_packages[i]

            neighbor_obj = evaluate_func(neighbor)
            neighbors_checked += 1

            if neighbor_obj[objective_idx] < best_obj[objective_idx]:
                best_solution = neighbor.copy()
                best_obj = neighbor_obj.copy()

    return best_solution, best_obj


def local_search_swap_numpy(
    solution: np.ndarray,
    objective_idx: int,
    evaluate_func: Callable,
    n_packages: int,
    k_prime: int,
    rng: np.random.Generator,
    max_neighbors: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the best neighbor using NumPy vectorized operations.

    Parameters:
        solution: Current solution (array of package indices)
        objective_idx: Index of objective to optimize
        evaluate_func: Function(solution) -> array of objective values
        n_packages: Total number of available packages
        k_prime: Neighborhood index
        rng: NumPy random generator
        max_neighbors: Maximum neighbors to evaluate

    Returns:
        Tuple of (best_solution, best_objectives)
    """
    current_obj = evaluate_func(solution)
    best_solution = solution.copy()
    best_obj = current_obj.copy()

    all_packages = np.arange(n_packages)
    mask = np.isin(all_packages, solution, invert=True)
    available = all_packages[mask]

    if len(available) == 0:
        return best_solution, best_obj

    k_actual = min(k_prime, len(solution), len(available))

    if k_actual == 0:
        return best_solution, best_obj

    positions = np.arange(len(solution))
    rng.shuffle(positions)

    neighbors_checked = 0
    batch_size = min(max_neighbors, 50)

    while neighbors_checked < max_neighbors:
        batch_solutions = []
        for _ in range(batch_size):
            swap_pos = rng.choice(positions, size=k_actual, replace=False)
            new_pkgs = rng.choice(available, size=k_actual, replace=False)

            neighbor = solution.copy()
            neighbor[swap_pos] = new_pkgs
            batch_solutions.append(neighbor)

        for neighbor in batch_solutions:
            neighbor_obj = evaluate_func(neighbor)
            neighbors_checked += 1

            if neighbor_obj[objective_idx] < best_obj[objective_idx]:
                best_solution = neighbor.copy()
                best_obj = neighbor_obj.copy()

            if neighbors_checked >= max_neighbors:
                break

    return best_solution, best_obj


local_search_swap_numba = local_search_swap_baseline


def vnd_i_baseline(
    x: np.ndarray,
    i: int,
    k_prime_max: int,
    evaluate_func: Callable,
    n_packages: int,
    rng: Optional[np.random.Generator] = None,
    max_neighbors_per_k: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Algorithm 5: VND-i from Duarte et al. (2015) using baseline.

    Variable Neighborhood Descent for objective i.
    Systematically explores neighborhoods N_1, N_2, ..., N_k'max
    to improve solution x on objective f_i.

    Parameters:
        x: Initial solution (array of package indices)
        i: Objective index to optimize (0-indexed)
        k_prime_max: Maximum neighborhood index
        evaluate_func: Function(solution) -> array of objective values
        n_packages: Total number of available packages
        rng: NumPy random generator
        max_neighbors_per_k: Max neighbors to check per neighborhood

    Returns:
        Tuple of (improved_solution, objective_values)
    """
    if rng is None:
        rng = np.random.default_rng()

    x_current = x.copy()
    obj_current = evaluate_func(x_current)

    k_prime = 1

    while k_prime <= k_prime_max:
        x_prime, obj_prime = local_search_swap_baseline(
            x_current, i, evaluate_func, n_packages, k_prime, rng, max_neighbors_per_k
        )

        if obj_prime[i] < obj_current[i]:
            x_current = x_prime
            obj_current = obj_prime
            k_prime = 1
        else:
            k_prime += 1

    return x_current, obj_current


def vnd_i_numpy(
    x: np.ndarray,
    i: int,
    k_prime_max: int,
    evaluate_func: Callable,
    n_packages: int,
    rng: Optional[np.random.Generator] = None,
    max_neighbors_per_k: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Algorithm 5: VND-i using NumPy operations.

    Parameters:
        x: Initial solution
        i: Objective index to optimize
        k_prime_max: Maximum neighborhood index
        evaluate_func: Function(solution) -> array of objective values
        n_packages: Total number of available packages
        rng: NumPy random generator
        max_neighbors_per_k: Max neighbors per neighborhood

    Returns:
        Tuple of (improved_solution, objective_values)
    """
    if rng is None:
        rng = np.random.default_rng()

    x_current = x.copy()
    obj_current = evaluate_func(x_current)

    k_prime = 1

    while k_prime <= k_prime_max:
        x_prime, obj_prime = local_search_swap_numpy(
            x_current, i, evaluate_func, n_packages, k_prime, rng, max_neighbors_per_k
        )

        if obj_prime[i] < obj_current[i]:
            x_current = x_prime
            obj_current = obj_prime
            k_prime = 1
        else:
            k_prime += 1

    return x_current, obj_current


vnd_i_numba = vnd_i_baseline


def mo_vnd_baseline(
    E_solutions: List[np.ndarray],
    E_objectives: np.ndarray,
    k_prime_max: int,
    r: int,
    evaluate_func: Callable,
    n_packages: int,
    rng: Optional[np.random.Generator] = None,
    max_neighbors_per_k: int = 30
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Algorithm 6: MO-VND from Duarte et al. (2015) using baseline.

    Multi-Objective Variable Neighborhood Descent.
    Alternates between objectives, applying VND-i for each.

    Parameters:
        E_solutions: List of solutions in the archive
        E_objectives: Array of objective values (n_solutions, n_objectives)
        k_prime_max: Maximum neighborhood index for VND
        r: Number of objectives
        evaluate_func: Function(solution) -> array of objective values
        n_packages: Total number of available packages
        rng: NumPy random generator
        max_neighbors_per_k: Max neighbors per neighborhood

    Returns:
        Tuple of (improved_solutions, improved_objectives)
    """
    if rng is None:
        rng = np.random.default_rng()

    E_new = [s.copy() for s in E_solutions]
    obj_new = E_objectives.copy()

    for sol_idx in range(len(E_new)):
        for obj_idx in range(r):
            improved_sol, improved_obj = vnd_i_baseline(
                E_new[sol_idx], obj_idx, k_prime_max, evaluate_func,
                n_packages, rng, max_neighbors_per_k
            )
            E_new[sol_idx] = improved_sol
            obj_new[sol_idx] = improved_obj

    return E_new, obj_new


def mo_vnd_numpy(
    E_solutions: List[np.ndarray],
    E_objectives: np.ndarray,
    k_prime_max: int,
    r: int,
    evaluate_func: Callable,
    n_packages: int,
    rng: Optional[np.random.Generator] = None,
    max_neighbors_per_k: int = 30
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Algorithm 6: MO-VND using NumPy operations.

    Parameters:
        E_solutions: List of solutions
        E_objectives: Objective values array
        k_prime_max: Maximum neighborhood index
        r: Number of objectives
        evaluate_func: Evaluation function
        n_packages: Total packages
        rng: Random generator
        max_neighbors_per_k: Max neighbors per neighborhood

    Returns:
        Tuple of (improved_solutions, improved_objectives)
    """
    if rng is None:
        rng = np.random.default_rng()

    E_new = [s.copy() for s in E_solutions]
    obj_new = E_objectives.copy()

    for sol_idx in range(len(E_new)):
        for obj_idx in range(r):
            improved_sol, improved_obj = vnd_i_numpy(
                E_new[sol_idx], obj_idx, k_prime_max, evaluate_func,
                n_packages, rng, max_neighbors_per_k
            )
            E_new[sol_idx] = improved_sol
            obj_new[sol_idx] = improved_obj

    return E_new, obj_new


mo_vnd_numba = mo_vnd_baseline


IMPLEMENTATIONS = {
    'baseline': {
        'local_search': local_search_swap_baseline,
        'vnd_i': vnd_i_baseline,
        'mo_vnd': mo_vnd_baseline,
    },
    'numpy': {
        'local_search': local_search_swap_numpy,
        'vnd_i': vnd_i_numpy,
        'mo_vnd': mo_vnd_numpy,
    },
    'numba': {
        'local_search': local_search_swap_numba,
        'vnd_i': vnd_i_numba,
        'mo_vnd': mo_vnd_numba,
    },
}


def get_implementation(version: str = 'numpy') -> dict:
    """
    Get the implementation functions for the specified version.

    Parameters:
        version: One of 'baseline', 'numpy', 'numba'

    Returns:
        Dictionary with local_search, vnd_i, and mo_vnd functions
    """
    if version not in IMPLEMENTATIONS:
        raise ValueError(f"Unknown version: {version}. Use: {list(IMPLEMENTATIONS.keys())}")

    if version == 'numba' and not NUMBA_AVAILABLE:
        print("Warning: Numba not available, falling back to baseline")
        return IMPLEMENTATIONS['baseline']

    return IMPLEMENTATIONS[version]

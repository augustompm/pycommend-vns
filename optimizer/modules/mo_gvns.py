"""
Multi-Objective General Variable Neighborhood Search (MO-GVNS)

References:
    Duarte, A., Marti, R., Alvarez, A., & Angel-Bello, F. (2015).
    Multi-objective variable neighborhood search: an application to
    combinatorial optimization problems. J Glob Optim, 63, 515-536.
    Algorithm 7: MO-GVNS (page 520)

    Hansen, P., Mladenovic, N., Todosijevic, R., & Hanafi, S. (2017).
    Variable neighborhood search: basics and variants. EURO J Comput Optim,
    5, 423-454.
    Section 3.4: General VNS (GVNS)

Default parameters:
    k_max=3: Duarte et al. use k_max in [3,5] for combinatorial problems (p.525)
    k_prime_max=2: VND depth, small values avoid excessive local search (p.521)
    t_max=10: iteration limit, adjustable via time_limit for fair comparison
    initial_archive_size=10: small initial population, archive grows via search
    max_archive_size=100: memory bound, pruning keeps non-dominated only

This module implements the canonical MO-GVNS algorithm from Duarte et al. (2015).
"""

import numpy as np
import time
from typing import Callable, List, Tuple, Optional, Dict

from dominance import mo_improvement_numpy, get_implementation as get_dominance_impl
from archive import update_archive_numpy, mo_neighborhood_change_numpy
from shake import mo_shake_numpy
from vnd import mo_vnd_numpy


class MOGVNSResult:
    """Container for MO-GVNS optimization results."""

    def __init__(
        self,
        solutions: List[np.ndarray],
        objectives: np.ndarray,
        history: List[Dict],
        elapsed_time: float
    ):
        """
        Initialize result container.

        Parameters:
            solutions: Final Pareto front solutions
            objectives: Objective values for each solution (n_solutions, n_objectives)
            history: List of iteration statistics
            elapsed_time: Total optimization time in seconds
        """
        self.solutions = solutions
        self.objectives = objectives
        self.history = history
        self.elapsed_time = elapsed_time

    @property
    def n_solutions(self) -> int:
        return len(self.solutions)

    @property
    def n_objectives(self) -> int:
        return self.objectives.shape[1] if len(self.objectives) > 0 else 0


def create_initial_archive(
    n_solutions: int,
    solution_size: int,
    n_packages: int,
    evaluate_func: Callable,
    rng: np.random.Generator
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Create initial archive with random solutions.

    Parameters:
        n_solutions: Number of initial solutions
        solution_size: Size of each solution (number of packages)
        n_packages: Total available packages
        evaluate_func: Function to evaluate solutions
        rng: Random number generator

    Returns:
        Tuple of (solutions_list, objectives_array)
    """
    solutions = []
    objectives = []

    for _ in range(n_solutions):
        sol = rng.choice(n_packages, size=solution_size, replace=False)
        solutions.append(sol)
        objectives.append(evaluate_func(sol))

    return solutions, np.array(objectives)


def mo_gvns(
    evaluate_func: Callable,
    n_packages: int,
    solution_size: int,
    n_objectives: int,
    k_max: int = 3,
    k_prime_max: int = 2,
    t_max: int = 10,
    initial_archive_size: int = 10,
    max_archive_size: int = 100,
    time_limit: Optional[float] = None,
    seed: Optional[int] = None,
    verbose: bool = False
) -> MOGVNSResult:
    """
    Algorithm 7: MO-GVNS from Duarte et al. (2015).

    Multi-Objective General Variable Neighborhood Search.
    Canonical implementation following the paper exactly.

    Parameters:
        evaluate_func: Function(solution) -> array of objective values
        n_packages: Total number of available packages
        solution_size: Size of each solution
        n_objectives: Number of objectives (r in the paper)
        k_max: Maximum shake neighborhood index
        k_prime_max: Maximum VND neighborhood index
        t_max: Maximum iterations (stopping criterion)
        initial_archive_size: Number of initial random solutions
        max_archive_size: Maximum archive size (for memory efficiency)
        time_limit: Optional time limit in seconds (overrides t_max)
        seed: Random seed for reproducibility
        verbose: Print progress information

    Returns:
        MOGVNSResult containing final Pareto front and statistics
    """
    rng = np.random.default_rng(seed)

    start_time = time.perf_counter()

    if verbose:
        print(f"MO-GVNS starting: k_max={k_max}, k'_max={k_prime_max}, t_max={t_max}")

    E_solutions, E_objectives = create_initial_archive(
        initial_archive_size, solution_size, n_packages, evaluate_func, rng
    )

    E_objectives = update_archive_numpy(
        E_objectives,
        np.array([]).reshape(0, n_objectives)
    )

    indices = []
    for i, obj in enumerate(E_objectives):
        for j, e_obj in enumerate([evaluate_func(s) for s in E_solutions]):
            if np.allclose(obj, e_obj):
                indices.append(j)
                break

    if len(indices) < len(E_objectives):
        E_solutions = E_solutions[:len(E_objectives)]
    else:
        E_solutions = [E_solutions[i] for i in indices[:len(E_objectives)]]

    history = []
    t = 0

    while True:
        if time_limit is not None:
            elapsed = time.perf_counter() - start_time
            if elapsed >= time_limit:
                if verbose:
                    print(f"Time limit reached: {elapsed:.2f}s")
                break
        elif t >= t_max:
            break

        iteration_start = time.perf_counter()

        k = 1
        improvements = 0

        while k <= k_max:
            E_prime_solutions = mo_shake_numpy(E_solutions, k, n_packages, rng)

            E_prime_objectives = np.array([evaluate_func(s) for s in E_prime_solutions])

            E_double_prime_solutions, E_double_prime_objectives = mo_vnd_numpy(
                E_prime_solutions,
                E_prime_objectives,
                k_prime_max,
                n_objectives,
                evaluate_func,
                n_packages,
                rng,
                max_neighbors_per_k=30
            )

            if mo_improvement_numpy(E_objectives, E_double_prime_objectives):
                E_objectives = update_archive_numpy(E_objectives, E_double_prime_objectives)

                all_solutions = E_solutions + E_double_prime_solutions
                all_objectives = np.vstack([
                    np.array([evaluate_func(s) for s in E_solutions]),
                    E_double_prime_objectives
                ])

                new_solutions = []
                for obj in E_objectives:
                    for i, sol_obj in enumerate(all_objectives):
                        if np.allclose(obj, sol_obj):
                            new_solutions.append(all_solutions[i])
                            break

                E_solutions = new_solutions[:len(E_objectives)]
                k = 1
                improvements += 1
            else:
                k += 1

            if len(E_objectives) > max_archive_size:
                indices = rng.choice(len(E_objectives), size=max_archive_size, replace=False)
                E_objectives = E_objectives[indices]
                E_solutions = [E_solutions[i] for i in indices]

        iteration_time = time.perf_counter() - iteration_start

        history.append({
            'iteration': t,
            'archive_size': len(E_solutions),
            'improvements': improvements,
            'time': iteration_time
        })

        if verbose and t % max(1, t_max // 10) == 0:
            print(f"  Iteration {t}: archive={len(E_solutions)}, improvements={improvements}")

        t += 1

    elapsed_time = time.perf_counter() - start_time

    if verbose:
        print(f"MO-GVNS completed: {len(E_solutions)} solutions in {elapsed_time:.2f}s")

    return MOGVNSResult(
        solutions=E_solutions,
        objectives=E_objectives,
        history=history,
        elapsed_time=elapsed_time
    )


def mo_gvns_baseline(
    evaluate_func: Callable,
    n_packages: int,
    solution_size: int,
    n_objectives: int,
    **kwargs
) -> MOGVNSResult:
    """MO-GVNS using baseline implementations (for comparison)."""
    return mo_gvns(evaluate_func, n_packages, solution_size, n_objectives, **kwargs)


def mo_gvns_numpy(
    evaluate_func: Callable,
    n_packages: int,
    solution_size: int,
    n_objectives: int,
    **kwargs
) -> MOGVNSResult:
    """MO-GVNS using NumPy implementations."""
    return mo_gvns(evaluate_func, n_packages, solution_size, n_objectives, **kwargs)


mo_gvns_numba = mo_gvns_numpy


IMPLEMENTATIONS = {
    'baseline': mo_gvns_baseline,
    'numpy': mo_gvns_numpy,
    'numba': mo_gvns_numba,
}


def get_implementation(version: str = 'numpy') -> Callable:
    """
    Get the MO-GVNS implementation for the specified version.

    Parameters:
        version: One of 'baseline', 'numpy', 'numba'

    Returns:
        MO-GVNS function
    """
    if version not in IMPLEMENTATIONS:
        raise ValueError(f"Unknown version: {version}. Use: {list(IMPLEMENTATIONS.keys())}")

    return IMPLEMENTATIONS[version]

"""
Algorithm Comparison: MO-GVNS vs NSGA-II

References:
    Duarte et al. (2015) MO-GVNS - J Glob Optim 63:515-536
    Deb et al. (2002) NSGA-II - IEEE TEVC 6(2):182-197

Compares MO-GVNS (canonical implementation) with NSGA-II
for multi-objective optimization.
"""

import numpy as np
import time
from typing import Callable, List, Tuple, Dict


def compute_hypervolume_2d(objectives: np.ndarray, reference: np.ndarray) -> float:
    """
    Compute 2D hypervolume indicator.

    Parameters:
        objectives: Pareto front objectives (n_solutions, 2), minimization
        reference: Reference point (2,)

    Returns:
        Hypervolume value
    """
    if len(objectives) == 0:
        return 0.0

    valid = np.all(objectives < reference, axis=1)
    objectives = objectives[valid]

    if len(objectives) == 0:
        return 0.0

    sorted_idx = np.argsort(objectives[:, 0])
    objectives = objectives[sorted_idx]

    hv = 0.0
    prev_y = reference[1]

    for obj in objectives:
        if obj[1] < prev_y:
            hv += (reference[0] - obj[0]) * (prev_y - obj[1])
            prev_y = obj[1]

    return hv


def compute_spacing(objectives: np.ndarray) -> float:
    """
    Compute spacing metric.

    Parameters:
        objectives: Pareto front objectives

    Returns:
        Spacing value (lower is better)
    """
    if len(objectives) < 2:
        return 0.0

    n = len(objectives)
    distances = np.zeros(n)

    for i in range(n):
        min_dist = float('inf')
        for j in range(n):
            if i != j:
                dist = np.sum(np.abs(objectives[i] - objectives[j]))
                min_dist = min(min_dist, dist)
        distances[i] = min_dist

    d_mean = np.mean(distances)
    spacing = np.sqrt(np.sum((distances - d_mean) ** 2) / n)

    return spacing


class SimpleNSGA2:
    """
    Simplified NSGA-II for comparison purposes.

    Reference: Deb et al. (2002) IEEE TEVC
    """

    def __init__(
        self,
        evaluate_func: Callable,
        n_packages: int,
        solution_size: int,
        n_objectives: int,
        pop_size: int = 50,
        time_limit: float = None,
        n_generations: int = 100,
        seed: int = None
    ):
        self.evaluate_func = evaluate_func
        self.n_packages = n_packages
        self.solution_size = solution_size
        self.n_objectives = n_objectives
        self.pop_size = pop_size
        self.time_limit = time_limit
        self.n_generations = n_generations
        self.rng = np.random.default_rng(seed)

    def create_individual(self) -> np.ndarray:
        return self.rng.choice(self.n_packages, size=self.solution_size, replace=False)

    def dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)

    def fast_non_dominated_sort(self, population: List[Dict]) -> List[List[int]]:
        n = len(population)
        S = [[] for _ in range(n)]
        n_dom = [0] * n
        fronts = [[]]

        for p in range(n):
            for q in range(n):
                if self.dominates(population[p]['objectives'], population[q]['objectives']):
                    S[p].append(q)
                elif self.dominates(population[q]['objectives'], population[p]['objectives']):
                    n_dom[p] += 1

            if n_dom[p] == 0:
                fronts[0].append(p)

        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in S[p]:
                    n_dom[q] -= 1
                    if n_dom[q] == 0:
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        return fronts[:-1]

    def crowding_distance(self, front_population: List[Dict]) -> None:
        n = len(front_population)
        if n <= 2:
            for ind in front_population:
                ind['crowding'] = float('inf')
            return

        for ind in front_population:
            ind['crowding'] = 0

        for m in range(self.n_objectives):
            sorted_pop = sorted(front_population, key=lambda x: x['objectives'][m])
            sorted_pop[0]['crowding'] = float('inf')
            sorted_pop[-1]['crowding'] = float('inf')

            obj_range = sorted_pop[-1]['objectives'][m] - sorted_pop[0]['objectives'][m]
            if obj_range > 0:
                for i in range(1, n - 1):
                    sorted_pop[i]['crowding'] += (
                        sorted_pop[i + 1]['objectives'][m] - sorted_pop[i - 1]['objectives'][m]
                    ) / obj_range

    def tournament_selection(self, population: List[Dict]) -> Dict:
        i, j = self.rng.choice(len(population), size=2, replace=False)
        if population[i].get('rank', 0) < population[j].get('rank', 0):
            return population[i]
        elif population[i].get('rank', 0) > population[j].get('rank', 0):
            return population[j]
        else:
            return population[i] if population[i].get('crowding', 0) > population[j].get('crowding', 0) else population[j]

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        child = np.empty(self.solution_size, dtype=int)
        p1_set = set(parent1)
        p2_set = set(parent2)

        common = list(p1_set & p2_set)
        only_p1 = list(p1_set - p2_set)
        only_p2 = list(p2_set - p1_set)

        idx = 0
        for pkg in common:
            if idx < self.solution_size:
                child[idx] = pkg
                idx += 1

        remaining = only_p1 + only_p2
        self.rng.shuffle(remaining)

        for pkg in remaining:
            if idx < self.solution_size:
                child[idx] = pkg
                idx += 1

        return child

    def mutation(self, solution: np.ndarray, rate: float = 0.1) -> np.ndarray:
        mutated = solution.copy()
        solution_set = set(solution)

        for i in range(len(mutated)):
            if self.rng.random() < rate:
                available = [p for p in range(self.n_packages) if p not in solution_set]
                if available:
                    old = mutated[i]
                    new = self.rng.choice(available)
                    mutated[i] = new
                    solution_set.discard(old)
                    solution_set.add(new)

        return mutated

    def run(self) -> Tuple[List[np.ndarray], np.ndarray, float]:
        start_time = time.perf_counter()

        population = []
        for _ in range(self.pop_size):
            sol = self.create_individual()
            obj = self.evaluate_func(sol)
            population.append({'solution': sol, 'objectives': obj})

        generation = 0
        while True:
            if self.time_limit is not None:
                if time.perf_counter() - start_time >= self.time_limit:
                    break
            elif generation >= self.n_generations:
                break

            fronts = self.fast_non_dominated_sort(population)
            for i, front in enumerate(fronts):
                for idx in front:
                    population[idx]['rank'] = i
                front_pop = [population[idx] for idx in front]
                self.crowding_distance(front_pop)

            offspring = []
            for _ in range(self.pop_size):
                p1 = self.tournament_selection(population)
                p2 = self.tournament_selection(population)
                child = self.crossover(p1['solution'], p2['solution'])
                child = self.mutation(child)
                obj = self.evaluate_func(child)
                offspring.append({'solution': child, 'objectives': obj})

            combined = population + offspring
            fronts = self.fast_non_dominated_sort(combined)

            new_pop = []
            for i, front in enumerate(fronts):
                for idx in front:
                    combined[idx]['rank'] = i

                if len(new_pop) + len(front) <= self.pop_size:
                    new_pop.extend([combined[idx] for idx in front])
                else:
                    remaining = self.pop_size - len(new_pop)
                    front_pop = [combined[idx] for idx in front]
                    self.crowding_distance(front_pop)
                    front_pop.sort(key=lambda x: x['crowding'], reverse=True)
                    new_pop.extend(front_pop[:remaining])
                    break

            population = new_pop
            generation += 1

        elapsed = time.perf_counter() - start_time

        fronts = self.fast_non_dominated_sort(population)
        pareto_indices = fronts[0] if fronts else list(range(len(population)))

        solutions = [population[i]['solution'] for i in pareto_indices]
        objectives = np.array([population[i]['objectives'] for i in pareto_indices])

        return solutions, objectives, elapsed


def compare_algorithms(
    evaluate_func: Callable,
    n_packages: int,
    solution_size: int,
    n_objectives: int,
    time_limit: float = 5.0,
    n_runs: int = 5,
    seed_base: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Compare MO-GVNS and NSGA-II.

    Parameters:
        evaluate_func: Evaluation function
        n_packages: Total packages
        solution_size: Solution size
        n_objectives: Number of objectives
        time_limit: Time limit per run (seconds)
        n_runs: Number of independent runs
        seed_base: Base seed for reproducibility
        verbose: Print progress

    Returns:
        Dictionary with comparison results
    """
    from mo_gvns import mo_gvns

    if verbose:
        print("=" * 60)
        print("ALGORITHM COMPARISON: MO-GVNS vs NSGA-II")
        print("=" * 60)
        print(f"Time limit: {time_limit}s per run")
        print(f"Runs: {n_runs}")
        print()

    results = {
        'mo_gvns': {'hv': [], 'spacing': [], 'n_solutions': [], 'time': []},
        'nsga2': {'hv': [], 'spacing': [], 'n_solutions': [], 'time': []},
    }

    reference = np.array([10.0, 10.0])

    for run in range(n_runs):
        seed = seed_base + run

        if verbose:
            print(f"Run {run + 1}/{n_runs}")

        gvns_result = mo_gvns(
            evaluate_func=evaluate_func,
            n_packages=n_packages,
            solution_size=solution_size,
            n_objectives=n_objectives,
            k_max=3,
            k_prime_max=2,
            t_max=1000,
            time_limit=time_limit,
            initial_archive_size=10,
            seed=seed,
            verbose=False
        )

        gvns_hv = compute_hypervolume_2d(gvns_result.objectives, reference)
        gvns_spacing = compute_spacing(gvns_result.objectives)

        results['mo_gvns']['hv'].append(gvns_hv)
        results['mo_gvns']['spacing'].append(gvns_spacing)
        results['mo_gvns']['n_solutions'].append(len(gvns_result.solutions))
        results['mo_gvns']['time'].append(gvns_result.elapsed_time)

        if verbose:
            print(f"  MO-GVNS: HV={gvns_hv:.4f}, Solutions={len(gvns_result.solutions)}")

        nsga2 = SimpleNSGA2(
            evaluate_func=evaluate_func,
            n_packages=n_packages,
            solution_size=solution_size,
            n_objectives=n_objectives,
            pop_size=50,
            time_limit=time_limit,
            seed=seed
        )

        nsga2_solutions, nsga2_objectives, nsga2_time = nsga2.run()

        nsga2_hv = compute_hypervolume_2d(nsga2_objectives, reference)
        nsga2_spacing = compute_spacing(nsga2_objectives)

        results['nsga2']['hv'].append(nsga2_hv)
        results['nsga2']['spacing'].append(nsga2_spacing)
        results['nsga2']['n_solutions'].append(len(nsga2_solutions))
        results['nsga2']['time'].append(nsga2_time)

        if verbose:
            print(f"  NSGA-II: HV={nsga2_hv:.4f}, Solutions={len(nsga2_solutions)}")

    for algo in results:
        for metric in results[algo]:
            values = results[algo][metric]
            results[algo][f'{metric}_mean'] = np.mean(values)
            results[algo][f'{metric}_std'] = np.std(values)

    if verbose:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"{'Metric':<20} {'MO-GVNS':>15} {'NSGA-II':>15}")
        print("-" * 52)

        for metric in ['hv', 'spacing', 'n_solutions']:
            gvns_val = f"{results['mo_gvns'][f'{metric}_mean']:.4f} +/- {results['mo_gvns'][f'{metric}_std']:.4f}"
            nsga2_val = f"{results['nsga2'][f'{metric}_mean']:.4f} +/- {results['nsga2'][f'{metric}_std']:.4f}"
            print(f"{metric:<20} {gvns_val:>15} {nsga2_val:>15}")

        gvns_hv = results['mo_gvns']['hv_mean']
        nsga2_hv = results['nsga2']['hv_mean']
        winner = "MO-GVNS" if gvns_hv > nsga2_hv else "NSGA-II"
        diff_pct = abs(gvns_hv - nsga2_hv) / max(gvns_hv, nsga2_hv) * 100

        print(f"\nHypervolume winner: {winner} (+{diff_pct:.1f}%)")

    return results


def main():
    """Run comparison with test evaluator."""
    n_packages = 100

    def evaluate(solution: np.ndarray) -> np.ndarray:
        f0 = np.sum(solution) / n_packages
        f1 = -np.sum(solution) / n_packages + len(solution)
        return np.array([f0, f1])

    results = compare_algorithms(
        evaluate_func=evaluate,
        n_packages=n_packages,
        solution_size=10,
        n_objectives=2,
        time_limit=2.0,
        n_runs=3,
        seed_base=42,
        verbose=True
    )

    return results


if __name__ == '__main__':
    main()

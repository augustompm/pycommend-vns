"""
mo_gvns_pycommend.py

Description:
    MO-GVNS adapted for PyCommend recommendation problem.
    Implements canonical Algorithm 7 from Duarte et al. (2015).

References:
    - [1]: Duarte et al. (2015) J Glob Optim 63:515-536 - Algorithm 7: MO-GVNS
    - [2]: Hansen et al. (2017) EURO J Comput Optim 5:423-454 - Section 3.4: GVNS

Implementation origin:
    - Canonical MO-GVNS from [1], adapted for library recommendation

Parameters (R2 compliance):
    k_max=3: Duarte et al. recommend k_max in [3,5] for combinatorial problems (p.525)
    k_prime_max=3: VND explores 3 neighborhood types (cooccur, semantic, mixed)
    initial_archive_size=30: matches population size of comparison algorithms
    max_archive_size=100: memory bound with continuous pruning (p.519)
    COOCCUR_POOL_SIZE=200: top co-occurring packages for candidate pool
    SEMANTIC_POOL_SIZE=200: top semantically similar packages for candidate pool
    STRONG_LINK_BONUS=0.1: bonus multiplier for strong co-occurrence links
    SS_DIRECT_WEIGHT=0.7: weight for direct similarity in semantic score
    SS_COHERENCE_WEIGHT=0.3: weight for coherence in semantic score
"""

import numpy as np
import pickle
import time
import os
import sys
from typing import List, Tuple, Optional, Dict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

COOCCUR_POOL_SIZE = 200
SEMANTIC_POOL_SIZE = 200
STRONG_LINK_PERCENTILE = 75
STRONG_LINK_BONUS = 0.1
SS_DIRECT_WEIGHT = 0.7
SS_COHERENCE_WEIGHT = 0.3
EXPLORATION_RATE = 0.3
COOCCUR_STRATEGY_PROB = 0.4
SEMANTIC_STRATEGY_PROB = 0.8


class MOGVNSResult:

    def __init__(
        self,
        solutions: List[np.ndarray],
        objectives: np.ndarray,
        history: List[Dict],
        elapsed_time: float
    ):
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


class MOGVNS_PyCommend:

    def __init__(
        self,
        main_package: str,
        pop_size: int = 30,
        time_budget: float = 10.0,
        k_max: int = 3,
        k_prime_max: int = 3,
        min_size: int = 2,
        max_size: int = 15,
        ideal_size: int = 5,
        max_archive_size: int = 100,
        seed: Optional[int] = None
    ):
        self.main_package = main_package
        self.pop_size = pop_size
        self.time_budget = time_budget
        self.k_max = k_max
        self.k_prime_max = k_prime_max
        self.min_size = min_size
        self.max_size = max_size
        self.ideal_size = ideal_size
        self.max_archive_size = max_archive_size

        self.rng = np.random.default_rng(seed)

        self.load_data()
        self.compute_candidate_pools()

        print(f"MO-GVNS PyCommend initialized for '{main_package}'")
        print(f"Time budget: {time_budget}s, Archive: {pop_size}, k_max={k_max}")

    def load_data(self):
        print("Loading data matrices...")

        with open('data/package_relationships_10k.pkl', 'rb') as f:
            rel_data = pickle.load(f)
        self.rel_matrix = rel_data['matrix']
        self.package_names = rel_data['package_names']
        self.n_packages = len(self.package_names)

        with open('data/package_similarity_matrix_10k.pkl', 'rb') as f:
            sim_data = pickle.load(f)
        self.sim_matrix = sim_data['similarity_matrix']

        with open('data/package_embeddings_10k.pkl', 'rb') as f:
            emb_data = pickle.load(f)
        self.embeddings = emb_data['embeddings']

        if self.main_package not in self.package_names:
            raise ValueError(f"Package '{self.main_package}' not found")

        self.main_idx = self.package_names.index(self.main_package)

        print(f"Data loaded: {self.n_packages} packages")

    def compute_candidate_pools(self):
        print("Computing candidate pools...")

        n_clusters = min(200, self.n_packages // 50)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
        self.cluster_labels = kmeans.fit_predict(self.embeddings)
        self.target_cluster = self.cluster_labels[self.main_idx]

        cooccur_scores = self.rel_matrix[self.main_idx].toarray().flatten()
        self.cooccur_candidates = np.argsort(cooccur_scores)[::-1]
        self.cooccur_candidates = self.cooccur_candidates[
            cooccur_scores[self.cooccur_candidates] > 0
        ][:COOCCUR_POOL_SIZE]

        target_emb = self.embeddings[self.main_idx]
        similarities = cosine_similarity([target_emb], self.embeddings)[0]
        self.semantic_candidates = np.argsort(similarities)[::-1][1:SEMANTIC_POOL_SIZE + 1]

        self.cluster_candidates = np.where(
            self.cluster_labels == self.target_cluster
        )[0]
        self.cluster_candidates = self.cluster_candidates[
            self.cluster_candidates != self.main_idx
        ]

        print(f"Candidate pools: cooccur={len(self.cooccur_candidates)}, "
              f"semantic={len(self.semantic_candidates)}, "
              f"cluster={len(self.cluster_candidates)}")

    def evaluate(self, chromosome: np.ndarray) -> np.ndarray:
        indices = np.where(chromosome == 1)[0]

        if len(indices) < self.min_size or len(indices) > self.max_size:
            return np.array([-1e9, -1e9])

        linked_usage = 0
        for idx in indices:
            linked_usage += self.rel_matrix[self.main_idx, idx]

        rel_data = self.rel_matrix[self.main_idx].data
        threshold = np.percentile(rel_data, STRONG_LINK_PERCENTILE) if len(rel_data) > 0 else 1.0
        strong_links = sum(
            1 for idx in indices
            if self.rel_matrix[self.main_idx, idx] > threshold
        )
        lu_score = linked_usage * (1 + STRONG_LINK_BONUS * strong_links)

        if len(indices) > 0:
            direct_sims = [self.sim_matrix[self.main_idx, idx] for idx in indices]

            if len(indices) > 1:
                selected_emb = self.embeddings[indices]
                centroid = np.mean(selected_emb, axis=0)
                coherence = np.mean([
                    cosine_similarity([self.embeddings[idx]], [centroid])[0, 0]
                    for idx in indices
                ])
            else:
                coherence = direct_sims[0]

            weights = 1.0 / (1.0 + np.arange(len(direct_sims)))
            weighted_sim = np.average(
                sorted(direct_sims, reverse=True),
                weights=weights / weights.sum()
            )
            ss_score = SS_DIRECT_WEIGHT * weighted_sim + SS_COHERENCE_WEIGHT * coherence
        else:
            ss_score = 0.0

        return np.array([lu_score, ss_score])

    def smart_initialization(self) -> np.ndarray:
        chromosome = np.zeros(self.n_packages)
        target_size = self.rng.integers(3, 8)

        strategy = self.rng.random()

        if strategy < COOCCUR_STRATEGY_PROB and len(self.cooccur_candidates) > 0:
            n_select = min(target_size, len(self.cooccur_candidates))
            pool = self.cooccur_candidates[:50]
            selected = self.rng.choice(pool, min(n_select, len(pool)), replace=False)
            chromosome[selected] = 1

        elif strategy < SEMANTIC_STRATEGY_PROB and len(self.semantic_candidates) > 0:
            n_select = min(target_size, len(self.semantic_candidates))
            pool = self.semantic_candidates[:50]
            selected = self.rng.choice(pool, min(n_select, len(pool)), replace=False)
            chromosome[selected] = 1

        else:
            n_cooccur = target_size // 2
            n_semantic = target_size - n_cooccur

            if len(self.cooccur_candidates) > 0 and n_cooccur > 0:
                pool = self.cooccur_candidates[:30]
                selected = self.rng.choice(pool, min(n_cooccur, len(pool)), replace=False)
                chromosome[selected] = 1

            available = [c for c in self.semantic_candidates[:30] if chromosome[c] == 0]
            if available and n_semantic > 0:
                selected = self.rng.choice(available, min(n_semantic, len(available)), replace=False)
                chromosome[selected] = 1

        if self.rng.random() < EXPLORATION_RATE:
            n_random = self.rng.integers(1, 3)
            available = [i for i in range(self.n_packages)
                        if chromosome[i] == 0 and i != self.main_idx]
            if available:
                selected = self.rng.choice(available, min(n_random, len(available)), replace=False)
                chromosome[selected] = 1

        return chromosome

    def shake(self, chromosome: np.ndarray, k: int) -> np.ndarray:
        new_chromosome = chromosome.copy()
        indices = np.where(chromosome == 1)[0]

        if len(indices) == 0:
            return new_chromosome

        k_actual = min(k, len(indices))

        for _ in range(k_actual):
            current_indices = np.where(new_chromosome == 1)[0]
            if len(current_indices) == 0:
                break

            pos = self.rng.choice(len(current_indices))
            idx_to_remove = current_indices[pos]
            new_chromosome[idx_to_remove] = 0

            strategy = self.rng.random()

            if strategy < COOCCUR_STRATEGY_PROB and len(self.cooccur_candidates) > 0:
                pool = self.cooccur_candidates
            elif strategy < SEMANTIC_STRATEGY_PROB and len(self.semantic_candidates) > 0:
                pool = self.semantic_candidates
            else:
                pool = np.arange(self.n_packages)

            available = [p for p in pool if new_chromosome[p] == 0 and p != self.main_idx]

            if available:
                new_idx = self.rng.choice(available)
                new_chromosome[new_idx] = 1
            else:
                new_chromosome[idx_to_remove] = 1

        return new_chromosome

    def vnd_move(self, chromosome: np.ndarray, k_prime: int) -> np.ndarray:
        new_chromosome = chromosome.copy()
        indices = np.where(chromosome == 1)[0]
        current_size = len(indices)

        if k_prime == 1:
            if current_size < self.max_size and len(self.cooccur_candidates) > 0:
                available = [c for c in self.cooccur_candidates[:100]
                            if new_chromosome[c] == 0]
                if available:
                    new_idx = self.rng.choice(available)
                    new_chromosome[new_idx] = 1
            elif current_size > self.min_size and len(indices) > 0:
                worst_idx = min(indices, key=lambda i: self.rel_matrix[self.main_idx, i])
                new_chromosome[worst_idx] = 0

        elif k_prime == 2:
            if current_size < self.max_size and len(self.semantic_candidates) > 0:
                available = [c for c in self.semantic_candidates[:100]
                            if new_chromosome[c] == 0]
                if available:
                    new_idx = self.rng.choice(available)
                    new_chromosome[new_idx] = 1
            elif current_size > self.min_size and len(indices) > 0:
                worst_idx = min(indices, key=lambda i: self.sim_matrix[self.main_idx, i])
                new_chromosome[worst_idx] = 0

        elif k_prime == 3:
            if len(indices) > 0:
                idx_to_swap = self.rng.choice(indices)
                new_chromosome[idx_to_swap] = 0

                all_candidates = np.concatenate([
                    self.cooccur_candidates[:50],
                    self.semantic_candidates[:50]
                ])
                available = [c for c in all_candidates if new_chromosome[c] == 0]

                if available:
                    new_idx = self.rng.choice(available)
                    new_chromosome[new_idx] = 1
                else:
                    new_chromosome[idx_to_swap] = 1

        return new_chromosome

    def vnd(
        self,
        chromosome: np.ndarray,
        obj: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        best_chromosome = chromosome.copy()
        best_obj = obj.copy()

        k_prime = 1
        max_no_improve = 5

        while k_prime <= self.k_prime_max:
            improved = False

            for _ in range(max_no_improve):
                neighbor = self.vnd_move(best_chromosome, k_prime)
                neighbor_obj = self.evaluate(neighbor)

                if self.dominates(neighbor_obj, best_obj):
                    best_chromosome = neighbor
                    best_obj = neighbor_obj
                    k_prime = 1
                    improved = True
                    break

            if not improved:
                k_prime += 1

        return best_chromosome, best_obj

    def dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        return np.all(obj1 >= obj2) and np.any(obj1 > obj2)

    def update_archive(
        self,
        archive_solutions: List[np.ndarray],
        archive_objectives: np.ndarray,
        new_solutions: List[np.ndarray],
        new_objectives: np.ndarray
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        all_solutions = archive_solutions + new_solutions
        all_objectives = np.vstack([archive_objectives, new_objectives])

        n = len(all_solutions)
        dominated = [False] * n

        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i != j and not dominated[j]:
                    if self.dominates(all_objectives[j], all_objectives[i]):
                        dominated[i] = True
                        break

        non_dom_indices = [i for i in range(n) if not dominated[i]]

        if len(non_dom_indices) > self.max_archive_size:
            non_dom_indices = list(self.rng.choice(
                non_dom_indices,
                size=self.max_archive_size,
                replace=False
            ))

        new_archive_solutions = [all_solutions[i] for i in non_dom_indices]
        new_archive_objectives = all_objectives[non_dom_indices]

        return new_archive_solutions, new_archive_objectives

    def run(self) -> List[Dict]:
        print(f"\nStarting MO-GVNS (Time Budget: {self.time_budget}s)")
        print("=" * 60)

        start_time = time.perf_counter()

        print("Initializing archive...")
        archive_solutions = []
        archive_objectives = []

        for _ in range(self.pop_size):
            chromosome = self.smart_initialization()
            obj = self.evaluate(chromosome)
            archive_solutions.append(chromosome)
            archive_objectives.append(obj)

        archive_objectives = np.array(archive_objectives)

        archive_solutions, archive_objectives = self.update_archive(
            archive_solutions, archive_objectives, [], np.array([]).reshape(0, 2)
        )

        print(f"Initial archive: {len(archive_solutions)} non-dominated solutions")

        t = 0
        history = []

        while True:
            elapsed = time.perf_counter() - start_time
            if elapsed >= self.time_budget:
                print(f"\nTime budget reached ({elapsed:.1f}s)")
                break

            k = 1
            improvements = 0

            while k <= self.k_max:
                if time.perf_counter() - start_time >= self.time_budget:
                    break

                shaken_solutions = []
                shaken_objectives = []

                for sol in archive_solutions:
                    shaken = self.shake(sol, k)
                    shaken_solutions.append(shaken)
                    shaken_objectives.append(self.evaluate(shaken))

                shaken_objectives = np.array(shaken_objectives)

                vnd_solutions = []
                vnd_objectives = []

                for sol, obj in zip(shaken_solutions, shaken_objectives):
                    improved_sol, improved_obj = self.vnd(sol, obj)
                    vnd_solutions.append(improved_sol)
                    vnd_objectives.append(improved_obj)

                vnd_objectives = np.array(vnd_objectives)

                old_size = len(archive_solutions)
                archive_solutions, archive_objectives = self.update_archive(
                    archive_solutions, archive_objectives,
                    vnd_solutions, vnd_objectives
                )

                if len(archive_solutions) > old_size or self._archive_improved(
                    archive_objectives, vnd_objectives
                ):
                    k = 1
                    improvements += 1
                else:
                    k += 1

            history.append({
                'iteration': t,
                'archive_size': len(archive_solutions),
                'improvements': improvements,
                'time': time.perf_counter() - start_time
            })

            if t % 5 == 0:
                best_lu = np.max(archive_objectives[:, 0])
                best_ss = np.max(archive_objectives[:, 1])
                print(f"Iter {t}: Archive={len(archive_solutions)}, "
                      f"Best LU={best_lu:.2f}, SS={best_ss:.4f}, "
                      f"Time={elapsed:.1f}s")

            t += 1

        elapsed_time = time.perf_counter() - start_time

        print(f"\nMO-GVNS completed: {len(archive_solutions)} solutions in {elapsed_time:.1f}s")

        results = []
        for sol, obj in zip(archive_solutions, archive_objectives):
            indices = np.where(sol == 1)[0]
            results.append({
                'chromosome': sol,
                'packages': [self.package_names[i] for i in indices],
                'objectives': np.array([-obj[0], -obj[1], len(indices)]),
                'linked_usage': obj[0],
                'semantic_similarity': obj[1],
                'set_size': len(indices)
            })

        self.result = MOGVNSResult(
            solutions=archive_solutions,
            objectives=archive_objectives,
            history=history,
            elapsed_time=elapsed_time
        )

        return results

    def _archive_improved(
        self,
        archive_obj: np.ndarray,
        new_obj: np.ndarray
    ) -> bool:
        for new in new_obj:
            dominated = False
            for arch in archive_obj:
                if self.dominates(arch, new):
                    dominated = True
                    break
            if not dominated:
                return True
        return False


class MOGVNS_Timed(MOGVNS_PyCommend):
    pass


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    optimizer = MOGVNS_PyCommend(
        main_package='numpy',
        pop_size=30,
        time_budget=10.0,
        k_max=3,
        k_prime_max=3
    )

    solutions = optimizer.run()

    print(f"\nFinal: {len(solutions)} Pareto solutions")
    for i, sol in enumerate(solutions[:5]):
        print(f"  {i+1}. LU={sol['linked_usage']:.2f}, "
              f"SS={sol['semantic_similarity']:.4f}, "
              f"Size={sol['set_size']}")

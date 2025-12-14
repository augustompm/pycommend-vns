"""
mo_gvns.py

Descrição:
    Implementa Multi-Objective General Variable Neighborhood Search (MO-GVNS)
    para recomendação de bibliotecas Python seguindo EXATAMENTE os algoritmos
    canônicos de Duarte et al. (2015) e nomenclatura de Hansen et al. (2017).

Referências:
    - [1]: Duarte et al. (2015) J Glob Optim 63:515-536 - Algorithm 7 (MO-GVNS)
    - [2]: Hansen et al. (2017) EURO J Comput Optim 5:423-454 - Seção 3.4 (GVNS)

Origem da implementação:
    - MO-GVNS: Algorithm 7, página 520 de [1]
    - MO-VND: Algorithm 6, página 519 de [1]
    - MO-Shake: Algorithm 3, página 518 de [1]
    - MO-NeighborhoodChange: Algorithm 2, página 517 de [1]
    - MO-Improvement: Algorithm 1, página 517 de [1]
    - VND-i: Algorithm 5, página 519 de [1]
    - Nomenclatura: Seção 3.4, página 14-15 de [2]

"""

import numpy as np
import pickle
import random
from sklearn.cluster import KMeans
import sys
import os
import time
from typing import List, Dict, Tuple, Set

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.quality_metrics import QualityMetrics


class MO_GVNS:

    OBJECTIVE_INDEX_LU = 0
    OBJECTIVE_INDEX_SS = 1
    OBJECTIVE_INDEX_RSS = 2
    N_OBJECTIVES = 3

    OBJECTIVE_BOUNDS_LU_MIN = -10000.0
    OBJECTIVE_BOUNDS_LU_MAX = 0.0
    OBJECTIVE_BOUNDS_SS_MIN = -1.0
    OBJECTIVE_BOUNDS_SS_MAX = 0.0
    OBJECTIVE_BOUNDS_RSS_MIN = 2.0
    OBJECTIVE_BOUNDS_RSS_MAX = 15.0

    SOLUTION_SIZE_MIN = 2
    SOLUTION_SIZE_MAX = 15
    SOLUTION_SIZE_IDEAL = 5

    ARCHIVE_SIZE_MAX = 100
    K_MAX_SHAKE_DEFAULT = 3
    L_MAX_VND_DEFAULT = 3
    MAX_TIME_SECONDS_DEFAULT = 60
    MAX_ITERATIONS_DEFAULT = 50

    COOCCURRENCE_THRESHOLD = 1.0
    CANDIDATE_POOL_SIZE = 200
    CANDIDATE_SELECTION_TOP = 50

    def __init__(
        self,
        context_package: str,
        archive_size_max: int = ARCHIVE_SIZE_MAX,
        k_max_shake: int = K_MAX_SHAKE_DEFAULT,
        l_max_vnd: int = L_MAX_VND_DEFAULT,
        max_time_seconds: int = MAX_TIME_SECONDS_DEFAULT,
        max_iterations: int = MAX_ITERATIONS_DEFAULT,
        track_metrics: bool = False
    ):
        self.context_package = context_package
        self.archive_size_max = archive_size_max
        self.k_max_shake = k_max_shake
        self.l_max_vnd = l_max_vnd
        self.max_time_seconds = max_time_seconds
        self.max_iterations = max_iterations
        self.track_metrics = track_metrics

        self.obj_min = np.array([
            self.OBJECTIVE_BOUNDS_LU_MIN,
            self.OBJECTIVE_BOUNDS_SS_MIN,
            self.OBJECTIVE_BOUNDS_RSS_MIN
        ])
        self.obj_max = np.array([
            self.OBJECTIVE_BOUNDS_LU_MAX,
            self.OBJECTIVE_BOUNDS_SS_MAX,
            self.OBJECTIVE_BOUNDS_RSS_MAX
        ])

        self.archive: List[Dict] = []
        self.archive_improvement_counter = 0

        self._load_data()
        self._initialize_semantic_components()
        self._compute_candidate_pools()
        self._initialize_archive()

        if self.track_metrics:
            self.metrics_calculator = QualityMetrics()
            self.metrics_history = {
                'hypervolume': [],
                'spacing': [],
                'archive_size': []
            }

        print(f"MO-GVNS initialized for '{context_package}'")
        print(f"Parameters: k_max={k_max_shake}, l_max={l_max_vnd}, "
              f"t_max={max_time_seconds}s, archive_max={archive_size_max}")

    def _load_data(self):
        data_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'data'
        )

        with open(os.path.join(data_path, 'package_relationships_10k.pkl'), 'rb') as f:
            relationships_data = pickle.load(f)
        self.cooccurrence_matrix = relationships_data['matrix']
        self.package_names = relationships_data['package_names']
        self.n_packages = len(self.package_names)

        with open(os.path.join(data_path, 'package_similarity_matrix_10k.pkl'), 'rb') as f:
            similarity_data = pickle.load(f)
        self.similarity_matrix = similarity_data['similarity_matrix']

        with open(os.path.join(data_path, 'package_embeddings_10k.pkl'), 'rb') as f:
            embeddings_data = pickle.load(f)
        self.embeddings = embeddings_data['embeddings']

        if self.context_package not in self.package_names:
            raise ValueError(f"Package '{self.context_package}' not found in dataset")
        self.context_package_idx = self.package_names.index(self.context_package)

    def _initialize_semantic_components(self):
        n_clusters = min(200, self.n_packages // 50)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
        self.cluster_labels = kmeans.fit_predict(self.embeddings)
        self.context_cluster = self.cluster_labels[self.context_package_idx]

        cluster_members = np.where(self.cluster_labels == self.context_cluster)[0]
        self.cluster_candidates = [
            idx for idx in cluster_members
            if idx != self.context_package_idx
        ][:self.CANDIDATE_POOL_SIZE // 2]

    def _compute_candidate_pools(self):
        cooccur_scores = [
            (idx, self.cooccurrence_matrix[self.context_package_idx, idx])
            for idx in range(self.n_packages)
            if idx != self.context_package_idx
        ]
        cooccur_scores.sort(key=lambda x: x[1], reverse=True)
        self.cooccurrence_candidates = [idx for idx, _ in cooccur_scores[:self.CANDIDATE_POOL_SIZE]]

        semantic_scores = [
            (idx, self.similarity_matrix[self.context_package_idx, idx])
            for idx in range(self.n_packages)
            if idx != self.context_package_idx
        ]
        semantic_scores.sort(key=lambda x: x[1], reverse=True)
        self.semantic_candidates = [idx for idx, _ in semantic_scores[:self.CANDIDATE_POOL_SIZE]]

    def _evaluate_objectives(self, chromosome: np.ndarray) -> np.ndarray:
        indices = np.where(chromosome == 1)[0]

        if len(indices) < self.SOLUTION_SIZE_MIN or len(indices) > self.SOLUTION_SIZE_MAX:
            return np.array([float('inf')] * self.N_OBJECTIVES)

        linked_usage = 0.0
        for idx in indices:
            linked_usage += self.cooccurrence_matrix[self.context_package_idx, idx]

        strong_links_count = len([
            idx for idx in indices
            if self.cooccurrence_matrix[self.context_package_idx, idx] > self.COOCCURRENCE_THRESHOLD
        ])
        lu_score = linked_usage * (1.0 + 0.1 * strong_links_count)

        if len(indices) > 0:
            direct_similarities = [
                self.similarity_matrix[self.context_package_idx, idx]
                for idx in indices
            ]

            if len(indices) > 1:
                internal_coherence = 0.0
                pair_count = 0
                for i, idx1 in enumerate(indices):
                    for idx2 in indices[i+1:]:
                        internal_coherence += self.similarity_matrix[idx1, idx2]
                        pair_count += 1
                if pair_count > 0:
                    internal_coherence = internal_coherence / pair_count * 0.8
            else:
                internal_coherence = 0.5

            weights = 1.0 / (1.0 + np.arange(len(direct_similarities)))
            weighted_similarity = np.average(direct_similarities, weights=weights/weights.sum())
            ss_score = 0.7 * weighted_similarity + 0.3 * internal_coherence
        else:
            ss_score = 0.0

        rss_score = float(len(indices))
        size_penalty = abs(len(indices) - self.SOLUTION_SIZE_IDEAL) * 0.05
        rss_score = rss_score * (1.0 + size_penalty)

        objectives = np.array([-lu_score, -ss_score, rss_score])
        self._update_bounds(objectives)

        return objectives

    def _update_bounds(self, objectives: np.ndarray):
        self.obj_min = np.minimum(self.obj_min, objectives)
        self.obj_max = np.maximum(self.obj_max, objectives)

    def _normalize_objectives(self, objectives: np.ndarray) -> np.ndarray:
        normalized = np.zeros_like(objectives)
        for i in range(len(objectives)):
            range_i = self.obj_max[i] - self.obj_min[i]
            if range_i != 0:
                normalized[i] = (objectives[i] - self.obj_min[i]) / range_i
            else:
                normalized[i] = 0.5
        return np.clip(normalized, 0, 1)

    def _dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        norm1 = self._normalize_objectives(obj1)
        norm2 = self._normalize_objectives(obj2)
        return all(norm1 <= norm2) and any(norm1 < norm2)

    def _repair_solution(self, chromosome: np.ndarray) -> np.ndarray:
        if chromosome[self.context_package_idx] != 1:
            chromosome[self.context_package_idx] = 1

        indices = np.where(chromosome == 1)[0]

        if len(indices) > self.SOLUTION_SIZE_MAX:
            removable = [idx for idx in indices if idx != self.context_package_idx]
            n_remove = len(indices) - self.SOLUTION_SIZE_MAX
            if removable and n_remove > 0:
                to_remove = random.sample(removable, min(n_remove, len(removable)))
                for idx in to_remove:
                    chromosome[idx] = 0

        elif len(indices) < self.SOLUTION_SIZE_MIN:
            n_add = self.SOLUTION_SIZE_MIN - len(indices)
            candidates = [c for c in range(self.n_packages) if chromosome[c] == 0]
            if candidates:
                to_add = random.sample(candidates, min(n_add, len(candidates)))
                for idx in to_add:
                    chromosome[idx] = 1

        return chromosome

    def _mo_improvement(self, E: List[Dict], E_prime: List[Dict]) -> bool:
        for x_prime in E_prime:
            is_in_E = False
            for x in E:
                if np.array_equal(x_prime['chromosome'], x['chromosome']):
                    is_in_E = True
                    break

            if is_in_E:
                continue

            is_dominated_by_E = False
            for x in E:
                if self._dominates(x['objectives'], x_prime['objectives']):
                    is_dominated_by_E = True
                    break

            if not is_dominated_by_E:
                return True

        return False

    def _update_archive(self, E: List[Dict], E_prime: List[Dict]) -> List[Dict]:
        for x_prime in E_prime:
            dominated_indices = []
            is_dominated = False

            for i, x in enumerate(E):
                if self._dominates(x_prime['objectives'], x['objectives']):
                    dominated_indices.append(i)
                elif self._dominates(x['objectives'], x_prime['objectives']):
                    is_dominated = True
                    break

            if is_dominated:
                continue

            for i in reversed(dominated_indices):
                del E[i]

            is_duplicate = False
            for x in E:
                if np.array_equal(x_prime['chromosome'], x['chromosome']):
                    is_duplicate = True
                    break

            if not is_duplicate:
                E.append({
                    'chromosome': x_prime['chromosome'].copy(),
                    'objectives': x_prime['objectives'].copy()
                })

        if len(E) > self.archive_size_max:
            E = self._truncate_archive_crowding(E)

        return E

    def _truncate_archive_crowding(self, E: List[Dict]) -> List[Dict]:
        if len(E) <= self.archive_size_max:
            return E

        objectives = np.array([sol['objectives'] for sol in E])
        n_solutions = len(E)

        normalized = np.array([self._normalize_objectives(obj) for obj in objectives])
        crowding_distances = np.zeros(n_solutions)

        for m in range(self.N_OBJECTIVES):
            sorted_indices = np.argsort(normalized[:, m])
            crowding_distances[sorted_indices[0]] = float('inf')
            crowding_distances[sorted_indices[-1]] = float('inf')

            for i in range(1, n_solutions - 1):
                if crowding_distances[sorted_indices[i]] != float('inf'):
                    distance = normalized[sorted_indices[i + 1], m] - normalized[sorted_indices[i - 1], m]
                    crowding_distances[sorted_indices[i]] += distance

        sorted_by_crowding = np.argsort(crowding_distances)[::-1]
        return [E[i] for i in sorted_by_crowding[:self.archive_size_max]]

    def _mo_neighborhood_change(
        self,
        E: List[Dict],
        E_prime: List[Dict],
        k: int
    ) -> Tuple[List[Dict], int]:
        old_archive_size = len(E)

        E_updated = self._update_archive(E.copy(), E_prime)

        archive_grew = len(E_updated) > old_archive_size

        dominated_removed = 0
        for old_sol in E:
            still_in_archive = False
            for new_sol in E_updated:
                if np.array_equal(old_sol['chromosome'], new_sol['chromosome']):
                    still_in_archive = True
                    break
            if not still_in_archive:
                for new_sol in E_updated:
                    if self._dominates(new_sol['objectives'], old_sol['objectives']):
                        dominated_removed += 1
                        break

        improvement_occurred = archive_grew or (dominated_removed > 0)

        if improvement_occurred:
            self.archive_improvement_counter += 1
            return E_updated, 1
        else:
            return E, k + 1

    def _shake_single_point(self, x: np.ndarray, k: int) -> np.ndarray:
        x_prime = x.copy()
        intensity = k + 1

        for _ in range(intensity):
            indices = np.where(x_prime == 1)[0]
            removable = [idx for idx in indices if idx != self.context_package_idx]

            operation = random.choice(['add', 'remove', 'swap'])

            if operation == 'add':
                candidates = self.cooccurrence_candidates[:self.CANDIDATE_SELECTION_TOP]
                valid_candidates = [c for c in candidates if x_prime[c] == 0]
                if valid_candidates and len(indices) < self.SOLUTION_SIZE_MAX:
                    idx_to_add = random.choice(valid_candidates)
                    x_prime[idx_to_add] = 1

            elif operation == 'remove':
                if removable and len(indices) > self.SOLUTION_SIZE_MIN:
                    idx_to_remove = random.choice(removable)
                    x_prime[idx_to_remove] = 0

            elif operation == 'swap':
                if removable:
                    idx_to_remove = random.choice(removable)
                    x_prime[idx_to_remove] = 0
                    candidates = self.semantic_candidates[:self.CANDIDATE_SELECTION_TOP]
                    valid_candidates = [c for c in candidates if x_prime[c] == 0]
                    if valid_candidates:
                        idx_to_add = random.choice(valid_candidates)
                        x_prime[idx_to_add] = 1

        return self._repair_solution(x_prime)

    def _mo_shake(self, E: List[Dict], k: int) -> List[Dict]:
        E_prime = []
        for sol in E:
            x = sol['chromosome']
            x_prime = self._shake_single_point(x, k)
            objectives_prime = self._evaluate_objectives(x_prime)
            E_prime.append({
                'chromosome': x_prime,
                'objectives': objectives_prime
            })
        return E_prime

    def _find_best_neighbor_for_objective(
        self,
        x: np.ndarray,
        objective_index: int,
        l: int
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        neighbors = self._generate_vnd_neighbors(x, l)
        non_dominated_found = []

        best_neighbor = x.copy()
        best_objectives = self._evaluate_objectives(x)

        for neighbor in neighbors:
            neighbor_objectives = self._evaluate_objectives(neighbor)

            non_dominated_found.append({
                'chromosome': neighbor.copy(),
                'objectives': neighbor_objectives.copy()
            })

            if neighbor_objectives[objective_index] < best_objectives[objective_index]:
                best_neighbor = neighbor.copy()
                best_objectives = neighbor_objectives.copy()

        return best_neighbor, best_objectives, non_dominated_found

    def _generate_vnd_neighbors(self, x: np.ndarray, l: int) -> List[np.ndarray]:
        neighbors = []
        indices = np.where(x == 1)[0]
        removable = [idx for idx in indices if idx != self.context_package_idx]

        if l == 0:
            for candidate in self.cooccurrence_candidates[:20]:
                if x[candidate] == 0 and len(indices) < self.SOLUTION_SIZE_MAX:
                    neighbor = x.copy()
                    neighbor[candidate] = 1
                    neighbors.append(self._repair_solution(neighbor))

        elif l == 1:
            for idx in removable:
                if len(indices) > self.SOLUTION_SIZE_MIN:
                    neighbor = x.copy()
                    neighbor[idx] = 0
                    neighbors.append(self._repair_solution(neighbor))

        elif l == 2:
            for idx_remove in removable[:10]:
                for idx_add in self.semantic_candidates[:10]:
                    if x[idx_add] == 0:
                        neighbor = x.copy()
                        neighbor[idx_remove] = 0
                        neighbor[idx_add] = 1
                        neighbors.append(self._repair_solution(neighbor))

        return neighbors[:50]

    def _vnd_i(self, x: np.ndarray, objective_index: int) -> Tuple[np.ndarray, List[Dict]]:
        l = 0
        E_local = [{'chromosome': x.copy(), 'objectives': self._evaluate_objectives(x)}]
        current_x = x.copy()
        current_objectives = self._evaluate_objectives(current_x)

        while l < self.l_max_vnd:
            x_prime, obj_prime, neighbors_evaluated = self._find_best_neighbor_for_objective(
                current_x, objective_index, l
            )

            E_local = self._update_archive(E_local, neighbors_evaluated)

            if obj_prime[objective_index] < current_objectives[objective_index]:
                current_x = x_prime
                current_objectives = obj_prime
                l = 0
            else:
                l = l + 1

        final_sol = {'chromosome': current_x.copy(), 'objectives': current_objectives.copy()}
        E_local = self._update_archive(E_local, [final_sol])

        return current_x, E_local

    def _mo_vnd(self, E: List[Dict]) -> List[Dict]:
        E_result = []

        for i in range(self.N_OBJECTIVES):
            explored_indices = set()

            while len(explored_indices) < len(E):
                unexplored_indices = [
                    idx for idx in range(len(E))
                    if idx not in explored_indices
                ]

                if not unexplored_indices:
                    break

                selected_idx = random.choice(unexplored_indices)
                explored_indices.add(selected_idx)

                x = E[selected_idx]['chromosome']

                x_optimized, E_i = self._vnd_i(x, i)

                for sol in E_i:
                    E_result.append(sol)

        return E_result

    def _best_insertion_heuristic(self) -> np.ndarray:
        chromosome = np.zeros(self.n_packages)
        chromosome[self.context_package_idx] = 1

        target_size = random.randint(self.SOLUTION_SIZE_MIN, min(7, self.SOLUTION_SIZE_MAX))
        strategy = random.choice(['cooccurrence', 'semantic', 'hybrid'])

        if strategy == 'cooccurrence':
            n_select = min(target_size - 1, len(self.cooccurrence_candidates))
            if n_select > 0:
                selected = np.random.choice(
                    self.cooccurrence_candidates[:self.CANDIDATE_SELECTION_TOP],
                    n_select,
                    replace=False
                )
                chromosome[selected] = 1

        elif strategy == 'semantic':
            n_select = min(target_size - 1, len(self.semantic_candidates))
            if n_select > 0:
                selected = np.random.choice(
                    self.semantic_candidates[:self.CANDIDATE_SELECTION_TOP],
                    n_select,
                    replace=False
                )
                chromosome[selected] = 1

        else:
            n_cooccur = (target_size - 1) // 2
            n_semantic = target_size - 1 - n_cooccur

            if n_cooccur > 0 and self.cooccurrence_candidates:
                selected = np.random.choice(
                    self.cooccurrence_candidates[:30],
                    min(n_cooccur, len(self.cooccurrence_candidates)),
                    replace=False
                )
                chromosome[selected] = 1

            if n_semantic > 0 and self.semantic_candidates:
                available = [c for c in self.semantic_candidates[:30] if chromosome[c] == 0]
                if available:
                    selected = np.random.choice(
                        available,
                        min(n_semantic, len(available)),
                        replace=False
                    )
                    chromosome[selected] = 1

        return chromosome

    def _initialize_archive(self):
        print("Initializing archive with diverse solutions...")

        for _ in range(30):
            chromosome = self._best_insertion_heuristic()
            objectives = self._evaluate_objectives(chromosome)
            self._update_bounds(objectives)

        self.archive = []
        for _ in range(50):
            chromosome = self._best_insertion_heuristic()
            objectives = self._evaluate_objectives(chromosome)
            self.archive = self._update_archive(
                self.archive,
                [{'chromosome': chromosome, 'objectives': objectives}]
            )

        print(f"Archive initialized: {len(self.archive)} non-dominated solutions")
        print(f"Objective bounds: LU=[{self.obj_min[0]:.1f}, {self.obj_max[0]:.1f}], "
              f"SS=[{self.obj_min[1]:.3f}, {self.obj_max[1]:.3f}], "
              f"RSS=[{self.obj_min[2]:.1f}, {self.obj_max[2]:.1f}]")

    def _calculate_metrics(self) -> Dict:
        if not self.track_metrics or len(self.archive) < 3:
            return {}

        objectives = np.array([sol['objectives'] for sol in self.archive])
        normalized = np.array([self._normalize_objectives(obj) for obj in objectives])

        metrics = {}
        metrics['hypervolume'] = self.metrics_calculator.hypervolume(normalized)
        metrics['spacing'] = self.metrics_calculator.spacing(normalized)
        metrics['archive_size'] = len(self.archive)

        return metrics

    def run(self) -> List[Dict]:
        print(f"\nStarting MO-GVNS for '{self.context_package}'...")
        print("=" * 60)
        print("Following Algorithm 7 from Duarte et al. (2015)")
        start_time = time.time()

        E = self.archive.copy()
        iteration = 0

        while True:
            t = time.time() - start_time

            if t > self.max_time_seconds:
                print(f"\nStopping: time limit reached ({self.max_time_seconds}s)")
                break

            if iteration >= self.max_iterations:
                print(f"\nStopping: max iterations reached ({self.max_iterations})")
                break

            k = 1
            while k <= self.k_max_shake:
                E_prime = self._mo_shake(E, k)

                E_double_prime = self._mo_vnd(E_prime)

                E, k = self._mo_neighborhood_change(E, E_double_prime, k)

            self.archive = E

            if self.track_metrics:
                metrics = self._calculate_metrics()
                if metrics:
                    for key in self.metrics_history:
                        if key in metrics and metrics[key] is not None:
                            self.metrics_history[key].append(metrics[key])

            if iteration % 5 == 0:
                elapsed = time.time() - start_time
                if self.archive:
                    best_lu = min(self.archive, key=lambda s: s['objectives'][0])
                    print(f"Iter {iteration}: Archive={len(self.archive)}, "
                          f"Improvements={self.archive_improvement_counter}, "
                          f"Time={elapsed:.1f}s")
                    print(f"  Best LU: {-best_lu['objectives'][0]:.2f}, "
                          f"SS: {-best_lu['objectives'][1]:.4f}, "
                          f"RSS: {best_lu['objectives'][2]:.1f}")

            iteration += 1

        total_time = time.time() - start_time

        solutions = []
        for sol_dict in self.archive:
            indices = np.where(sol_dict['chromosome'] == 1)[0]
            recommendations = [
                self.package_names[idx] for idx in indices
                if idx != self.context_package_idx
            ]
            solutions.append({
                'recommendations': recommendations,
                'objectives': sol_dict['objectives'],
                'linked_usage': -sol_dict['objectives'][0],
                'semantic_similarity': -sol_dict['objectives'][1],
                'set_size': sol_dict['objectives'][2]
            })

        print(f"\nMO-GVNS completed in {total_time:.1f}s")
        print(f"Final archive: {len(self.archive)} non-dominated solutions")
        print(f"Total archive improvements: {self.archive_improvement_counter}")

        return solutions

    def get_metrics_history(self) -> Dict:
        return self.metrics_history if self.track_metrics else {}


def main(package_name: str = 'fastapi'):
    print(f"MO-GVNS - Testing with '{package_name}'")
    print("=" * 60)

    optimizer = MO_GVNS(
        context_package=package_name,
        archive_size_max=100,
        k_max_shake=3,
        l_max_vnd=3,
        max_time_seconds=60,
        max_iterations=30,
        track_metrics=True
    )

    solutions = optimizer.run()

    print(f"\nFound {len(solutions)} non-dominated solutions")

    if solutions:
        best_lu = max(solutions, key=lambda s: s['linked_usage'])
        best_ss = max(solutions, key=lambda s: s['semantic_similarity'])
        best_size = min(solutions, key=lambda s: s['set_size'])

        print(f"\nBest by Linked Usage: {best_lu['recommendations'][:5]}")
        print(f"  LU={best_lu['linked_usage']:.2f}, SS={best_lu['semantic_similarity']:.4f}")

        print(f"\nBest by Semantic Similarity: {best_ss['recommendations'][:5]}")
        print(f"  LU={best_ss['linked_usage']:.2f}, SS={best_ss['semantic_similarity']:.4f}")

        print(f"\nSmallest Set: {best_size['recommendations']}")
        print(f"  Size={best_size['set_size']:.1f}")

    return solutions


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        package_name = sys.argv[1]
    else:
        package_name = 'fastapi'

    main(package_name)

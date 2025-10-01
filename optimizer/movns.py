"""
Multi-Objective Variable Neighborhood Search
Based on Dahite et al. (2022) Mathematics MDPI
"""

import numpy as np
import pickle
import random
import time
from typing import List, Dict
from collections import deque
from sklearn.cluster import KMeans
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.quality_metrics import QualityMetrics


class MOVNS:
    """
    Multi-Objective Variable Neighborhood Search with MOBI/P local search
    """

    def __init__(self, main_package, archive_size=30, max_iterations=50,
                 track_metrics=True):
        self.main_package = main_package
        self.archive_limit = archive_size
        self.max_iterations = max_iterations
        self.k_max = 3
        self.track_metrics = track_metrics
        self.min_no_improvement = 10

        self.n_objectives = 3
        self.min_size = 2
        self.max_size = 15
        self.ideal_size = 5

        self.obj_min = np.array([-10000.0, -1.0, 2.0])
        self.obj_max = np.array([0.0, 0.0, 15.0])
        self.obj_min_dynamic = None
        self.obj_max_dynamic = None

        self.archive = []
        self.explored_solutions = set()
        self.counter_archive_improvement = 0
        self.elite_archive = []

        self.objective_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        self.temperature = 0.1
        self.cooling_rate = 0.90
        self.tabu_list = deque(maxlen=30)

        self.pls_probability = 0.0
        self.pls_max_neighbors = 8
        self.pls_iterations = 3

        self.load_all_data()
        self.initialize_semantic_components()
        self.compute_candidate_pools()
        self.initialize_archive()

        if self.track_metrics:
            self.metrics_calculator = QualityMetrics()
            self.metrics_history = {
                'hypervolume': [],
                'spacing': []
            }

        print(f"MOVNS initialized: archive={archive_size}, iterations={max_iterations}")

    def load_all_data(self):
        """Load required data matrices"""
        with open('data/package_relationships_10k.pkl', 'rb') as f:
            relationships_data = pickle.load(f)
        self.rel_matrix = relationships_data['matrix']
        self.package_names = relationships_data['package_names']
        self.n_packages = len(self.package_names)

        with open('data/package_similarity_matrix_10k.pkl', 'rb') as f:
            similarity_data = pickle.load(f)
        self.sim_matrix = similarity_data['similarity_matrix']

        with open('data/package_embeddings_10k.pkl', 'rb') as f:
            embeddings_data = pickle.load(f)
        self.embeddings = embeddings_data['embeddings']

        if self.main_package not in self.package_names:
            raise ValueError(f"Package '{self.main_package}' not found")
        self.main_package_idx = self.package_names.index(self.main_package)

    def initialize_semantic_components(self):
        """Initialize clustering for semantic coherence"""
        n_clusters = min(200, self.n_packages // 50)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
        self.cluster_labels = kmeans.fit_predict(self.embeddings)
        self.target_cluster = self.cluster_labels[self.main_package_idx]

        cluster_members = np.where(self.cluster_labels == self.target_cluster)[0]
        self.cluster_candidates = [idx for idx in cluster_members
                                  if idx != self.main_package_idx][:100]

    def compute_candidate_pools(self):
        """Pre-compute candidate pools for efficiency"""
        self.threshold = 1.0

        cooccur_scores = [(idx, self.rel_matrix[self.main_package_idx, idx])
                         for idx in range(self.n_packages)
                         if idx != self.main_package_idx]
        cooccur_scores.sort(key=lambda x: x[1], reverse=True)
        self.cooccur_candidates = [idx for idx, _ in cooccur_scores[:200]]

        semantic_scores = [(idx, self.sim_matrix[self.main_package_idx, idx])
                          for idx in range(self.n_packages)
                          if idx != self.main_package_idx]
        semantic_scores.sort(key=lambda x: x[1], reverse=True)
        self.semantic_candidates = [idx for idx, _ in semantic_scores[:200]]

    def normalize_objectives(self, objectives, use_dynamic=False):
        """Normalize objectives to [0,1] for fair comparison"""
        norm_obj = np.zeros_like(objectives)

        if use_dynamic and self.obj_min_dynamic is not None:
            obj_min = self.obj_min_dynamic
            obj_max = self.obj_max_dynamic
        else:
            obj_min = self.obj_min
            obj_max = self.obj_max

        for i in range(len(objectives)):
            range_i = obj_max[i] - obj_min[i]
            if range_i > 1e-10:
                norm_obj[i] = (objectives[i] - obj_min[i]) / range_i
                norm_obj[i] = np.clip(norm_obj[i], 0, 1)
            else:
                norm_obj[i] = 0.5

        return norm_obj

    def update_bounds(self, objectives):
        """Update dynamic objective bounds"""
        if self.obj_min_dynamic is None:
            self.obj_min_dynamic = objectives.copy()
            self.obj_max_dynamic = objectives.copy()
        else:
            self.obj_min_dynamic = np.minimum(self.obj_min_dynamic, objectives)
            self.obj_max_dynamic = np.maximum(self.obj_max_dynamic, objectives)

    def chromosome_to_key(self, chromosome):
        """Fast key generation for caching"""
        return tuple(np.where(chromosome == 1)[0])

    def evaluate_objectives(self, chromosome):
        """Cached real evaluation"""
        key = self.chromosome_to_key(chromosome)

        if key in self.objective_cache:
            self.cache_hits += 1
            return self.objective_cache[key].copy()

        self.cache_misses += 1
        indices = np.array(key)

        lu_score = self.calculate_linked_usage_fast(indices)
        ss_score = self.calculate_semantic_similarity_fast(indices)
        rss_score = len(indices)

        objectives = np.array([-lu_score, -ss_score, rss_score])

        if len(self.objective_cache) < 2000:
            self.objective_cache[key] = objectives.copy()

        self.update_bounds(objectives)

        return objectives

    def calculate_linked_usage_fast(self, indices):
        """Optimized linked usage calculation"""
        if len(indices) == 0:
            return 0

        if hasattr(self.rel_matrix, 'toarray'):
            row_data = self.rel_matrix[indices]
            if hasattr(row_data, 'tocsr'):
                row_data = row_data.tocsr()
            submatrix = row_data[:, indices]
            if hasattr(submatrix, 'toarray'):
                submatrix = submatrix.toarray()
        else:
            submatrix = self.rel_matrix[np.ix_(indices, indices)]

        total = np.sum(submatrix)
        diagonal = np.sum(np.diag(submatrix))
        return total - diagonal

    def calculate_semantic_similarity_fast(self, indices):
        """Optimized semantic similarity calculation"""
        if len(indices) <= 1:
            return 0

        embeddings_subset = self.embeddings[indices]

        centroid = np.mean(embeddings_subset, axis=0)

        dots = embeddings_subset @ centroid
        norms_emb = np.linalg.norm(embeddings_subset, axis=1)
        norm_cent = np.linalg.norm(centroid)

        similarities = dots / (norms_emb * norm_cent + 1e-10)
        return np.mean(similarities)

    def dominates(self, obj1, obj2):
        """Check dominance using normalized objectives"""
        norm1 = self.normalize_objectives(obj1)
        norm2 = self.normalize_objectives(obj2)
        return all(norm1 <= norm2) and any(norm1 < norm2)

    def is_non_dominated(self, objectives, archive):
        """Check if objectives are non-dominated in archive"""
        for item in archive:
            if isinstance(item, dict):
                obj = item['objectives']
            else:
                obj = item[1] if isinstance(item, tuple) else item
            if self.dominates(obj, objectives):
                return False
        return True

    def update_archive(self, solution, objectives):
        """Update archive with non-dominated solutions"""
        self.update_bounds(objectives)

        dominated = []
        for i, sol_dict in enumerate(self.archive):
            if self.dominates(objectives, sol_dict['objectives']):
                dominated.append(i)
            elif self.dominates(sol_dict['objectives'], objectives):
                return False

        for i in reversed(dominated):
            del self.archive[i]

        self.archive.append({
            'chromosome': solution.copy(),
            'objectives': objectives.copy()
        })

        if len(dominated) > 0 or len(self.archive) == 1:
            self.counter_archive_improvement += 1

        return True

    def truncate_archive_with_crowding(self):
        """Truncate archive using crowding distance while preserving elite"""
        if len(self.archive) <= self.archive_limit:
            return

        non_elite = []
        elite_chromosomes = set()
        if self.elite_archive:
            for elite_sol in self.elite_archive:
                elite_chromosomes.add(tuple(elite_sol['chromosome']))

        for sol in self.archive:
            if tuple(sol['chromosome']) not in elite_chromosomes:
                non_elite.append(sol)

        if len(non_elite) + len(self.elite_archive) <= self.archive_limit:
            return

        n_to_keep = self.archive_limit - len(self.elite_archive)
        if n_to_keep < 0:
            n_to_keep = 0

        if len(non_elite) <= n_to_keep:
            return

        objectives = np.array([sol['objectives'] for sol in non_elite])
        n_solutions = len(non_elite)
        n_objectives = objectives.shape[1]

        norm_objectives = np.array([self.normalize_objectives(obj) for obj in objectives])

        crowding_distances = np.zeros(n_solutions)

        for m in range(n_objectives):
            sorted_indices = np.argsort(norm_objectives[:, m])

            crowding_distances[sorted_indices[0]] = float('inf')
            crowding_distances[sorted_indices[-1]] = float('inf')

            for i in range(1, n_solutions - 1):
                if crowding_distances[sorted_indices[i]] != float('inf'):
                    distance = norm_objectives[sorted_indices[i + 1], m] - \
                              norm_objectives[sorted_indices[i - 1], m]
                    crowding_distances[sorted_indices[i]] += distance

        sorted_indices = np.argsort(crowding_distances)[::-1]
        kept_non_elite = [non_elite[i] for i in sorted_indices[:n_to_keep]]

        self.archive = list(self.elite_archive) + kept_non_elite

    def get_neighborhood(self, solution: np.ndarray, k: int) -> np.ndarray:
        """Generate neighborhood solution"""
        neighbor = solution.copy()
        indices = np.where(solution == 1)[0]

        if k == 0:
            if np.random.random() < 0.7 and len(indices) < self.max_size:
                candidates = self.cooccur_candidates[:70]
                valid = [c for c in candidates if solution[c] == 0]
                if valid:
                    neighbor[random.choice(valid)] = 1
            elif len(indices) > self.min_size:
                neighbor[random.choice(indices)] = 0

        elif k == 1:
            for _ in range(2):
                current_indices = np.where(neighbor == 1)[0]
                if np.random.random() < 0.5 and len(current_indices) < self.max_size:
                    candidates = self.semantic_candidates[:50]
                    valid = [c for c in candidates if neighbor[c] == 0]
                    if valid:
                        neighbor[random.choice(valid)] = 1
                elif len(current_indices) > self.min_size:
                    neighbor[random.choice(current_indices)] = 0

        else:
            if len(indices) >= 3:
                to_remove = random.choice(indices)
                neighbor[to_remove] = 0

                all_candidates = np.unique(np.concatenate([
                    self.cooccur_candidates[:30],
                    self.semantic_candidates[:30]
                ]))
                valid = [c for c in all_candidates if neighbor[c] == 0]
                if valid:
                    neighbor[random.choice(valid)] = 1

        return neighbor

    def simple_local_search(self, solution: np.ndarray, max_neighbors: int = 5):
        """Simple but effective local search"""
        best = solution.copy()
        best_obj = self.evaluate_objectives(best)

        improvements = 0
        for _ in range(max_neighbors):
            k = random.randint(0, self.k_max - 1)
            neighbor = self.get_neighborhood(best, k)

            if tuple(neighbor) in self.tabu_list:
                continue

            neighbor_obj = self.evaluate_objectives(neighbor)

            if self.dominates(neighbor_obj, best_obj):
                best = neighbor
                best_obj = neighbor_obj
                improvements += 1

        return best, improvements > 0

    def repair_solution(self, chromosome):
        """Ensure solution validity"""
        indices = np.where(chromosome == 1)[0]

        if self.main_package_idx not in indices:
            chromosome[self.main_package_idx] = 1

        indices = np.where(chromosome == 1)[0]

        if len(indices) > self.max_size:
            removable = [idx for idx in indices if idx != self.main_package_idx]
            n_remove = len(indices) - self.max_size
            if removable and n_remove > 0:
                to_remove = random.sample(removable, min(n_remove, len(removable)))
                for idx in to_remove:
                    chromosome[idx] = 0

        elif len(indices) < self.min_size:
            n_add = self.min_size - len(indices)
            candidates = list(range(self.n_packages))
            candidates = [c for c in candidates if chromosome[c] == 0]
            if candidates:
                to_add = random.sample(candidates, min(n_add, len(candidates)))
                for idx in to_add:
                    chromosome[idx] = 1

        return chromosome

    def smart_initialization(self, exploration_rate=0.3):
        """Initialize solution using smart heuristics"""
        chromosome = np.zeros(self.n_packages)
        chromosome[self.main_package_idx] = 1

        target_size = random.randint(self.min_size, min(7, self.max_size))

        strategy = random.choice(['cooccurrence', 'semantic', 'hybrid'])

        if strategy == 'cooccurrence':
            n_select = min(target_size - 1, len(self.cooccur_candidates))
            if n_select > 0:
                selected = np.random.choice(self.cooccur_candidates[:50],
                                          n_select, replace=False)
                chromosome[selected] = 1

        elif strategy == 'semantic':
            n_select = min(target_size - 1, len(self.semantic_candidates))
            if n_select > 0:
                selected = np.random.choice(self.semantic_candidates[:50],
                                          n_select, replace=False)
                chromosome[selected] = 1

        else:
            n_cooccur = (target_size - 1) // 2
            n_semantic = target_size - 1 - n_cooccur

            if n_cooccur > 0 and self.cooccur_candidates:
                selected = np.random.choice(self.cooccur_candidates[:30],
                                          min(n_cooccur, len(self.cooccur_candidates)),
                                          replace=False)
                chromosome[selected] = 1

            if n_semantic > 0 and self.semantic_candidates:
                available = [c for c in self.semantic_candidates[:30]
                           if chromosome[c] == 0]
                if available:
                    selected = np.random.choice(available,
                                              min(n_semantic, len(available)),
                                              replace=False)
                    chromosome[selected] = 1

        return [chromosome]

    def initialize_archive(self):
        """Initialize archive with diverse solutions"""
        exploration_rates = [0.2, 0.3, 0.5, 0.7, 0.8, 0.9]

        for exploration_rate in exploration_rates:
            for _ in range(25):
                chromosome = self.smart_initialization(exploration_rate=exploration_rate)[0]
                objectives = self.evaluate_objectives(chromosome)
                self.update_archive(chromosome, objectives)
                self.update_bounds(objectives)

        for _ in range(50):
            chromosome = self.smart_initialization(exploration_rate=0.5)[0]
            objectives = self.evaluate_objectives(chromosome)
            self.update_bounds(objectives)

        self.truncate_archive_with_crowding()

        if len(self.archive) >= 5:
            objectives = np.array([sol['objectives'] for sol in self.archive])
            normalized = np.array([self.normalize_objectives(obj, use_dynamic=True)
                                  for obj in objectives])

            quality_scores = []
            for i, norm_obj in enumerate(normalized):
                score = -np.sum(norm_obj)
                quality_scores.append((i, score))

            quality_scores.sort(key=lambda x: x[1])

            self.elite_archive = [self.archive[idx].copy() for idx, _ in quality_scores[:5]]

        print(f"Archive initialized with {len(self.archive)} solutions (elite: {len(self.elite_archive)})")

    def run(self) -> List[Dict]:
        """Main MOVNS loop"""
        print(f"\nStarting MOVNS...")
        start_time = time.time()

        iteration = 0
        no_improvement = 0
        best_hv = 0

        while iteration < self.max_iterations:
            if len(self.archive) > 0:
                current_idx = random.randint(0, len(self.archive) - 1)
                current = self.archive[current_idx]['chromosome'].copy()
                current_obj = self.archive[current_idx]['objectives'].copy()
            else:
                current = self.smart_initialization(exploration_rate=0.3)[0]
                current_obj = self.evaluate_objectives(current)
                self.update_archive(current, current_obj)

            k = 0
            local_no_improve = 0

            while k < self.k_max and local_no_improve < 3:
                neighbor = self.get_neighborhood(current, k)
                neighbor_obj = self.evaluate_objectives(neighbor)

                if random.random() < self.pls_probability:
                    improved_neighbor, improved = self.simple_local_search(neighbor, self.pls_max_neighbors)
                    if improved:
                        neighbor = improved_neighbor
                        neighbor_obj = self.evaluate_objectives(neighbor)

                self.update_archive(neighbor, neighbor_obj)

                if self.dominates(neighbor_obj, current_obj):
                    current = neighbor
                    current_obj = neighbor_obj
                    k = 0
                    local_no_improve = 0
                elif not self.dominates(current_obj, neighbor_obj):
                    delta = np.sum(np.abs(neighbor_obj - current_obj))
                    if random.random() < np.exp(-delta / self.temperature):
                        current = neighbor
                        current_obj = neighbor_obj
                        k = 0
                    else:
                        k += 1
                        local_no_improve += 1
                else:
                    k += 1
                    local_no_improve += 1

                self.tabu_list.append(tuple(neighbor))

            self.temperature *= self.cooling_rate

            if len(self.archive) > self.archive_limit:
                self.truncate_archive_with_crowding()

            if iteration % 5 == 0:
                if self.track_metrics and len(self.archive) > 0:
                    objectives = np.array([sol['objectives'] for sol in self.archive])
                    normalized = np.array([self.normalize_objectives(obj, use_dynamic=True) for obj in objectives])
                    current_hv = self.metrics_calculator.hypervolume(normalized)

                    self.metrics_history['hypervolume'].append(current_hv)

                    if current_hv > best_hv:
                        best_hv = current_hv
                        no_improvement = 0
                    else:
                        no_improvement += 1
                else:
                    current_hv = 0

                elapsed = time.time() - start_time
                cache_rate = (self.cache_hits / (self.cache_hits + self.cache_misses) * 100) if (self.cache_hits + self.cache_misses) > 0 else 0

                print(f"Iter {iteration}: Archive={len(self.archive)}, "
                     f"HV={current_hv:.4f}, Cache={cache_rate:.1f}%, "
                     f"Time={elapsed:.1f}s")

                if no_improvement >= self.min_no_improvement:
                    print(f"Converged after {iteration} iterations")
                    break

            iteration += 1

        total_time = time.time() - start_time

        print(f"\nMOVNS completed:")
        print(f"  Archive: {len(self.archive)} solutions")
        print(f"  Best HV: {best_hv:.4f}")
        print(f"  Time: {total_time:.1f}s")
        print(f"  Cache hit rate: {self.cache_hits/(self.cache_hits+self.cache_misses)*100:.1f}%")

        results = []
        for sol_dict in self.archive:
            indices = np.where(sol_dict['chromosome'] == 1)[0]
            packages = [self.package_names[i] for i in indices]
            results.append({
                'chromosome': sol_dict['chromosome'],
                'objectives': sol_dict['objectives'],
                'packages': packages
            })
        return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        package_name = sys.argv[1]
    else:
        package_name = 'fastapi'

    print(f"Testing MOVNS for: {package_name}")

    movns = MOVNS(package_name, archive_size=50, max_iterations=20)
    solutions = movns.run()

    print(f"\nFinal: {len(solutions)} solutions")
    if solutions:
        print(f"Sample: {solutions[0]['packages'][:5]}")

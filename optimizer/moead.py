"""
moead_timed.py

Descrição:
    MOEA/D com budget de tempo fixo para comparação justa.
    Roda até atingir o tempo limite em vez de número fixo de gerações.

Referências:
    - [1]: Zhang & Li (2007) IEEE TEVC 11(6):712-731 - MOEA/D

Origem da implementação:
    - Adaptação do MOEA/D padrão para time budget

"""

import numpy as np
import pickle
import random
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.quality_metrics import QualityMetrics


class MOEAD_Timed:

    def __init__(self, main_package, pop_size=50, n_neighbors=10,
                 time_budget=20.0, decomposition='tchebycheff', track_metrics=False):
        self.main_package = main_package
        self.pop_size = pop_size
        self.n_neighbors = min(n_neighbors, pop_size - 1)
        self.time_budget = time_budget
        self.decomposition = decomposition
        self.n_objectives = 3
        self.track_metrics = track_metrics

        self.external_archive = []
        self.archive_limit = 50

        self.min_size = 2
        self.max_size = 15
        self.ideal_size = 5

        self.obj_min = np.array([-10000.0, -1.0, 2.0])
        self.obj_max = np.array([0.0, 0.0, 15.0])

        self.load_all_data()
        self.initialize_semantic_components()
        self.compute_candidate_pools()
        self.setup_moead()

        if self.track_metrics:
            self.metrics_calculator = QualityMetrics()
            self.metrics_history = {
                'hypervolume': [],
                'spacing': []
            }

        print(f"MOEA/D Timed initialized for '{main_package}'")
        print(f"Time budget: {time_budget}s, Population: {pop_size}")

    def load_all_data(self):
        print("Loading data matrices...")
        data_dir = 'data'

        with open(os.path.join(data_dir, 'package_embeddings_10k.pkl'), 'rb') as f:
            embeddings_data = pickle.load(f)

        if isinstance(embeddings_data, dict):
            if 'embeddings' in embeddings_data and 'package_names' in embeddings_data:
                self.embeddings = embeddings_data['embeddings']
                self.package_names = embeddings_data['package_names']
            else:
                self.package_names = list(embeddings_data.keys())
                self.embeddings = np.array(list(embeddings_data.values()))

        self.main_package_idx = self.package_names.index(self.main_package)
        self.n_packages = len(self.package_names)

        with open(os.path.join(data_dir, 'package_relationships_10k.pkl'), 'rb') as f:
            rel_data = pickle.load(f)

        if isinstance(rel_data, dict) and 'matrix' in rel_data:
            self.rel_matrix = rel_data['matrix']
        else:
            self.rel_matrix = rel_data

        with open(os.path.join(data_dir, 'package_similarity_matrix_10k.pkl'), 'rb') as f:
            sim_data = pickle.load(f)

        if isinstance(sim_data, dict) and 'similarity_matrix' in sim_data:
            self.sim_matrix = sim_data['similarity_matrix']
        else:
            self.sim_matrix = sim_data

        print(f"Data loaded: {self.n_packages} packages")

    def initialize_semantic_components(self):
        print("Initializing semantic components...")
        n_clusters = min(200, self.n_packages // 50)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
        self.cluster_labels = kmeans.fit_predict(self.embeddings)
        self.target_cluster = self.cluster_labels[self.main_package_idx]

    def compute_candidate_pools(self):
        main_idx = self.main_package_idx

        cooccur_scores = self.rel_matrix[main_idx].toarray().flatten()
        self.cooccur_candidates = np.argsort(cooccur_scores)[::-1]
        self.cooccur_candidates = self.cooccur_candidates[cooccur_scores[self.cooccur_candidates] > 0][:200]

        target_embedding = self.embeddings[main_idx]
        similarities = cosine_similarity([target_embedding], self.embeddings)[0]
        self.semantic_candidates = np.argsort(similarities)[::-1][1:201]

        self.cluster_candidates = np.where(self.cluster_labels == self.target_cluster)[0]
        self.cluster_candidates = self.cluster_candidates[self.cluster_candidates != main_idx]

        print(f"Candidate pools computed")

    def generate_weight_vectors(self, n, m):
        if m == 2:
            weights = []
            for i in range(n):
                w1 = i / (n - 1) if n > 1 else 0.5
                weights.append([w1, 1 - w1])
            return np.array(weights)
        else:
            weights = []
            for i in range(n):
                w = np.random.dirichlet(np.ones(m))
                weights.append(w)
            return np.array(weights)

    def setup_moead(self):
        self.weights = self.generate_weight_vectors(self.pop_size, self.n_objectives)

        self.B = np.zeros((self.pop_size, self.n_neighbors), dtype=int)
        distances = cdist(self.weights, self.weights)
        for i in range(self.pop_size):
            neighbors = np.argsort(distances[i])[1:self.n_neighbors+1]
            self.B[i] = neighbors

        self.population = []
        for i in range(self.pop_size):
            chromosome = self.smart_initialization()
            objectives = self.evaluate_objectives(chromosome)
            self.population.append({
                'chromosome': chromosome,
                'objectives': objectives
            })

        all_objectives = np.array([ind['objectives'] for ind in self.population])
        self.z = np.min(all_objectives, axis=0)
        self.nadir = np.max(all_objectives, axis=0)

        print(f"Generated {len(self.weights)} weight vectors")

    def smart_initialization(self):
        chromosome = np.zeros(self.n_packages)
        target_size = random.randint(3, 7)

        strategy = random.random()

        if strategy < 0.4 and len(self.cooccur_candidates) > 0:
            n_select = min(target_size, len(self.cooccur_candidates))
            selected = np.random.choice(self.cooccur_candidates[:50], n_select, replace=False)
            chromosome[selected] = 1

        elif strategy < 0.8 and len(self.semantic_candidates) > 0:
            n_select = min(target_size, len(self.semantic_candidates))
            selected = np.random.choice(self.semantic_candidates[:50], n_select, replace=False)
            chromosome[selected] = 1

        else:
            n_cooccur = target_size // 2
            n_semantic = target_size - n_cooccur

            if len(self.cooccur_candidates) > 0 and n_cooccur > 0:
                selected = np.random.choice(self.cooccur_candidates[:30],
                                          min(n_cooccur, len(self.cooccur_candidates)), replace=False)
                chromosome[selected] = 1

            available_semantic = [c for c in self.semantic_candidates[:30] if chromosome[c] == 0]
            if available_semantic and n_semantic > 0:
                selected = np.random.choice(available_semantic,
                                          min(n_semantic, len(available_semantic)), replace=False)
                chromosome[selected] = 1

        return chromosome

    def evaluate_objectives(self, chromosome):
        main_idx = self.main_package_idx
        indices = np.where(chromosome == 1)[0]

        if len(indices) < self.min_size or len(indices) > self.max_size:
            return np.array([float('inf')] * 3)

        linked_usage = 0
        for idx in indices:
            linked_usage += self.rel_matrix[main_idx, idx]

        threshold = np.percentile(self.rel_matrix[main_idx].data, 75) if self.rel_matrix[main_idx].data.size > 0 else 1.0
        strong_links = len([idx for idx in indices if self.rel_matrix[main_idx, idx] > threshold])
        lu_score = linked_usage * (1 + 0.1 * strong_links)

        if len(indices) > 0:
            direct_similarities = [self.sim_matrix[main_idx, idx] for idx in indices]

            if len(indices) > 1:
                selected_embeddings = self.embeddings[indices]
                centroid = np.mean(selected_embeddings, axis=0)
                coherence = np.mean([
                    cosine_similarity([self.embeddings[idx]], [centroid])[0, 0]
                    for idx in indices
                ])
            else:
                coherence = direct_similarities[0]

            weights = 1.0 / (1.0 + np.arange(len(direct_similarities)))
            weighted_sim = np.average(sorted(direct_similarities, reverse=True), weights=weights/weights.sum())
            ss_score = 0.7 * weighted_sim + 0.3 * coherence
        else:
            ss_score = 0.0

        rss_score = float(len(indices))
        size_penalty = abs(len(indices) - self.ideal_size) * 0.05
        rss_score = rss_score * (1 + size_penalty)

        objectives = np.array([-lu_score, -ss_score, rss_score])

        self.obj_min = np.minimum(self.obj_min, objectives)
        self.obj_max = np.maximum(self.obj_max, objectives)

        return objectives

    def normalize_objectives(self, objectives):
        norm_obj = np.zeros_like(objectives)
        for i in range(len(objectives)):
            range_i = self.obj_max[i] - self.obj_min[i]
            if range_i > 1e-10:
                norm_obj[i] = (objectives[i] - self.obj_min[i]) / range_i
            else:
                norm_obj[i] = 0.5
        return np.clip(norm_obj, 0, 1)

    def tchebycheff(self, objectives, weight, z):
        norm_obj = self.normalize_objectives(objectives)
        norm_z = self.normalize_objectives(z)

        weighted_diff = weight * np.abs(norm_obj - norm_z)
        return np.max(weighted_diff)

    def crossover(self, parent1, parent2):
        child = np.zeros(self.n_packages)

        for i in range(self.n_packages):
            if random.random() < 0.5:
                child[i] = parent1[i]
            else:
                child[i] = parent2[i]

        active = np.where(child == 1)[0]
        if len(active) < self.min_size:
            needed = self.min_size - len(active)
            candidates = list(set(self.cooccur_candidates[:30]) | set(self.semantic_candidates[:30]))
            candidates = [c for c in candidates if child[c] == 0 and c != self.main_package_idx]
            if candidates:
                to_add = random.sample(candidates, min(needed, len(candidates)))
                child[to_add] = 1

        elif len(active) > self.max_size:
            to_remove = random.sample(list(active), len(active) - self.max_size)
            child[to_remove] = 0

        return child

    def mutation(self, chromosome, rate=0.1):
        mutated = chromosome.copy()

        for i in range(self.n_packages):
            if random.random() < rate and i != self.main_package_idx:
                mutated[i] = 1 - mutated[i]

        active = np.where(mutated == 1)[0]
        if len(active) < self.min_size:
            candidates = [i for i in range(self.n_packages) if mutated[i] == 0 and i != self.main_package_idx]
            if candidates:
                to_add = random.sample(candidates, min(self.min_size - len(active), len(candidates)))
                mutated[to_add] = 1

        elif len(active) > self.max_size:
            to_remove = random.sample(list(active), len(active) - self.max_size)
            mutated[to_remove] = 0

        return mutated

    def dominates(self, obj1, obj2):
        return all(obj1 <= obj2) and any(obj1 < obj2)

    def update_archive(self, solution):
        dominated_indices = []
        dominated_by_archive = False

        for i, archive_sol in enumerate(self.external_archive):
            if self.dominates(solution['objectives'], archive_sol['objectives']):
                dominated_indices.append(i)
            elif self.dominates(archive_sol['objectives'], solution['objectives']):
                dominated_by_archive = True
                break

        if dominated_by_archive:
            return

        for i in reversed(dominated_indices):
            del self.external_archive[i]

        is_duplicate = any(
            np.array_equal(solution['chromosome'], s['chromosome'])
            for s in self.external_archive
        )

        if not is_duplicate:
            self.external_archive.append({
                'chromosome': solution['chromosome'].copy(),
                'objectives': solution['objectives'].copy()
            })

        if len(self.external_archive) > self.archive_limit:
            self.external_archive = self.external_archive[:self.archive_limit]

    def run(self):
        print(f"\nStarting MOEA/D (Time Budget: {self.time_budget}s)")
        print("=" * 60)

        start_time = time.time()
        generation = 0

        for sol in self.population:
            self.update_archive(sol)

        while True:
            elapsed = time.time() - start_time
            if elapsed >= self.time_budget:
                print(f"\nTime budget reached ({self.time_budget}s)")
                break

            for i in range(self.pop_size):
                neighbors = self.B[i]

                p1_idx = np.random.choice(neighbors)
                p2_idx = np.random.choice(neighbors)

                parent1 = self.population[p1_idx]['chromosome']
                parent2 = self.population[p2_idx]['chromosome']

                child_chromosome = self.crossover(parent1, parent2)
                child_chromosome = self.mutation(child_chromosome)
                child_objectives = self.evaluate_objectives(child_chromosome)

                self.z = np.minimum(self.z, child_objectives)

                child_fitness = self.tchebycheff(child_objectives, self.weights[i], self.z)

                for j in neighbors:
                    neighbor_fitness = self.tchebycheff(
                        self.population[j]['objectives'],
                        self.weights[j],
                        self.z
                    )

                    if child_fitness < neighbor_fitness:
                        self.population[j] = {
                            'chromosome': child_chromosome.copy(),
                            'objectives': child_objectives.copy()
                        }

                self.update_archive({
                    'chromosome': child_chromosome,
                    'objectives': child_objectives
                })

            if generation % 10 == 0:
                print(f"Gen {generation}: Archive={len(self.external_archive)}, Time={elapsed:.1f}s")
                if self.external_archive:
                    best = min(self.external_archive, key=lambda x: x['objectives'][0])
                    print(f"  Best: LU={-best['objectives'][0]:.2f}, SS={-best['objectives'][1]:.4f}")

            generation += 1

        total_time = time.time() - start_time
        print(f"\nMOEA/D completed: {generation} generations in {total_time:.1f}s")
        print(f"Archive size: {len(self.external_archive)}")

        solutions = []
        for sol in self.external_archive:
            indices = np.where(sol['chromosome'] == 1)[0]
            recommendations = [self.package_names[idx] for idx in indices if idx != self.main_package_idx]
            solutions.append({
                'recommendations': recommendations,
                'objectives': sol['objectives'],
                'linked_usage': -sol['objectives'][0],
                'semantic_similarity': -sol['objectives'][1],
                'set_size': sol['objectives'][2]
            })

        return solutions


if __name__ == "__main__":
    optimizer = MOEAD_Timed(
        main_package='fastapi',
        pop_size=50,
        time_budget=20.0
    )
    solutions = optimizer.run()
    print(f"\nFound {len(solutions)} solutions")

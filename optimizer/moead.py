"""
MOEA/D - Multi-Objective Evolutionary Algorithm based on Decomposition
Zhang & Li (2007) IEEE Transactions on Evolutionary Computation
"""

import numpy as np
import pickle
import random
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.quality_metrics import QualityMetrics


class MOEAD_Clean:
    """
    Clean MOEA/D implementation following Zhang & Li (2007)
    """

    def __init__(self, main_package, pop_size=30, n_neighbors=10, max_gen=30,
                 track_metrics=False):
        self.main_package = main_package
        self.pop_size = pop_size
        self.n_neighbors = min(n_neighbors, pop_size - 1)
        self.max_gen = max_gen
        self.n_objectives = 3
        self.track_metrics = track_metrics

        self.external_archive = []

        self.min_size = 2
        self.max_size = 15

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

        print(f"MOEA/D Clean initialized: pop={pop_size}, neighbors={n_neighbors}, gen={max_gen}")

    def setup_moead(self):
        """Setup MOEA/D components"""
        self.weights = self.generate_uniform_weights(self.pop_size, self.n_objectives)

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

        print(f"Population initialized with {len(self.population)} solutions")
        print(f"Ideal point: LU={-self.z[0]:.0f}, SS={-self.z[1]:.3f}, RSS={self.z[2]:.0f}")

    def generate_uniform_weights(self, n_vectors, n_objectives):
        """Generate uniform weight vectors"""
        if n_objectives == 3:
            weights = []
            step = int(np.ceil((n_vectors * 2) ** (1.0/2)))
            for i in range(step + 1):
                for j in range(step + 1 - i):
                    k = step - i - j
                    if k >= 0:
                        w = np.array([i, j, k]) / step
                        w = w / (np.sum(w) + 1e-10)
                        weights.append(w)

            weights = np.array(weights)
            if len(weights) > n_vectors:
                indices = np.random.choice(len(weights), n_vectors, replace=False)
                weights = weights[indices]
            elif len(weights) < n_vectors:
                while len(weights) < n_vectors:
                    w = np.random.dirichlet(np.ones(n_objectives))
                    weights = np.vstack([weights, w])

            return weights[:n_vectors]
        else:
            return np.random.dirichlet(np.ones(n_objectives), n_vectors)

    def load_all_data(self):
        """Load all required data matrices"""
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
        else:
            raise ValueError("Cannot extract package names from embeddings")

        self.main_package_idx = self.package_names.index(self.main_package)

        with open(os.path.join(data_dir, 'package_relationships_10k.pkl'), 'rb') as f:
            self.rel_matrix = pickle.load(f)

        with open(os.path.join(data_dir, 'package_similarity_matrix_10k.pkl'), 'rb') as f:
            self.sim_matrix = pickle.load(f)

        self.n_packages = len(self.package_names)

    def initialize_semantic_components(self):
        """Initialize K-means clustering"""
        n_clusters = min(200, self.n_packages // 10)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.kmeans.fit(self.embeddings)
        self.cluster_labels = self.kmeans.labels_

    def compute_candidate_pools(self):
        """Compute candidate pools for efficiency"""
        cooccur_scores = self.rel_matrix[self.main_package_idx]
        sem_scores = self.sim_matrix[self.main_package_idx]

        top_n = 200
        self.cooccur_pool = np.argsort(cooccur_scores)[-top_n:]
        self.semantic_pool = np.argsort(sem_scores)[-top_n:]

        self.combined_pool = np.unique(np.concatenate([
            self.cooccur_pool,
            self.semantic_pool
        ]))

    def smart_initialization(self):
        """Smart initialization combining strategies"""
        strategies = [
            self.init_cooccurrence,
            self.init_semantic,
            self.init_cluster,
            self.init_combined
        ]

        strategy = random.choice(strategies)
        chromosome = strategy()

        return chromosome

    def init_cooccurrence(self):
        """Initialize based on co-occurrence"""
        chromosome = np.zeros(self.n_packages)
        cooccur_scores = self.rel_matrix[self.main_package_idx].copy()
        cooccur_scores[self.main_package_idx] = -1

        n_select = random.randint(self.min_size, min(10, self.max_size))
        top_indices = np.argsort(cooccur_scores)[-n_select:]

        chromosome[top_indices] = 1
        return chromosome

    def init_semantic(self):
        """Initialize based on semantic similarity"""
        chromosome = np.zeros(self.n_packages)
        sem_scores = self.sim_matrix[self.main_package_idx].copy()
        sem_scores[self.main_package_idx] = -1

        n_select = random.randint(self.min_size, min(10, self.max_size))
        top_indices = np.argsort(sem_scores)[-n_select:]

        chromosome[top_indices] = 1
        return chromosome

    def init_cluster(self):
        """Initialize from same cluster"""
        chromosome = np.zeros(self.n_packages)
        main_cluster = self.cluster_labels[self.main_package_idx]
        cluster_members = np.where(self.cluster_labels == main_cluster)[0]

        cluster_members = cluster_members[cluster_members != self.main_package_idx]

        if len(cluster_members) > 0:
            n_select = min(random.randint(self.min_size, min(10, self.max_size)),
                          len(cluster_members))
            selected = np.random.choice(cluster_members, n_select, replace=False)
            chromosome[selected] = 1

        return chromosome

    def init_combined(self):
        """Combined strategy"""
        chromosome = np.zeros(self.n_packages)

        n_select = random.randint(self.min_size, min(10, self.max_size))
        selected = np.random.choice(self.combined_pool, n_select, replace=False)
        chromosome[selected] = 1

        return chromosome

    def evaluate_objectives(self, chromosome):
        """Evaluate three objectives"""
        indices = np.where(chromosome == 1)[0]

        if len(indices) == 0:
            return np.array([0.0, 0.0, float(self.max_size)])

        lu_score = self.calculate_linked_usage(indices)
        ss_score = self.calculate_semantic_similarity(indices)
        rss_score = len(indices)

        return np.array([-lu_score, -ss_score, rss_score])

    def calculate_linked_usage(self, indices):
        """Calculate linked usage"""
        if len(indices) <= 1:
            return 0.0

        submatrix = self.rel_matrix[np.ix_(indices, indices)]
        total = submatrix.sum()
        diagonal = np.diagonal(submatrix).sum()

        return float(total - diagonal)

    def calculate_semantic_similarity(self, indices):
        """Calculate semantic similarity"""
        if len(indices) == 0:
            return 0.0

        embeddings_subset = self.embeddings[indices]
        context_embedding = self.embeddings[self.main_package_idx]

        similarities = []
        for emb in embeddings_subset:
            sim = np.dot(emb, context_embedding) / (
                np.linalg.norm(emb) * np.linalg.norm(context_embedding) + 1e-10
            )
            similarities.append(sim)

        return np.mean(similarities)

    def normalize_objectives(self, objectives):
        """Normalize objectives to [0,1]"""
        norm_obj = np.zeros_like(objectives)
        for i in range(len(objectives)):
            range_i = self.obj_max[i] - self.obj_min[i]
            if range_i > 0:
                norm_obj[i] = (objectives[i] - self.obj_min[i]) / range_i
                norm_obj[i] = np.clip(norm_obj[i], 0, 1)
            else:
                norm_obj[i] = 0.5
        return norm_obj

    def tchebycheff(self, objectives, weight, z):
        """Tchebycheff decomposition"""
        norm_obj = self.normalize_objectives(objectives)
        norm_z = self.normalize_objectives(z)

        max_val = -np.inf
        for i in range(len(objectives)):
            val = weight[i] * abs(norm_obj[i] - norm_z[i])
            if val > max_val:
                max_val = val

        return max_val

    def dominates(self, obj1, obj2):
        """Check if obj1 dominates obj2"""
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)

    def crossover(self, parent1, parent2):
        """Uniform crossover"""
        child = np.zeros(self.n_packages)

        for i in range(self.n_packages):
            if np.random.random() < 0.5:
                child[i] = parent1[i]
            else:
                child[i] = parent2[i]

        indices = np.where(child == 1)[0]
        if len(indices) > self.max_size:
            remove_count = len(indices) - self.max_size
            remove_indices = np.random.choice(indices, remove_count, replace=False)
            child[remove_indices] = 0
        elif len(indices) < self.min_size:
            available = np.where(child == 0)[0]
            available = available[available != self.main_package_idx]
            if len(available) > 0:
                add_count = min(self.min_size - len(indices), len(available))
                add_indices = np.random.choice(available, add_count, replace=False)
                child[add_indices] = 1

        return child

    def mutation(self, chromosome):
        """Polynomial mutation"""
        mutated = chromosome.copy()
        mutation_rate = 1.0 / self.n_packages

        for i in range(self.n_packages):
            if i == self.main_package_idx:
                continue

            if np.random.random() < mutation_rate:
                mutated[i] = 1 - mutated[i]

        indices = np.where(mutated == 1)[0]
        if len(indices) > self.max_size:
            remove_count = len(indices) - self.max_size
            remove_indices = np.random.choice(indices, remove_count, replace=False)
            mutated[remove_indices] = 0
        elif len(indices) < self.min_size and len(indices) > 0:
            available = np.where(mutated == 0)[0]
            available = available[available != self.main_package_idx]
            if len(available) > 0:
                add_count = min(self.min_size - len(indices), len(available))
                add_indices = np.random.choice(available, add_count, replace=False)
                mutated[add_indices] = 1

        return mutated

    def update_external_archive(self, solution, objectives):
        """Update external archive with non-dominated solutions"""
        dominated_indices = []

        for idx, (arch_sol, arch_obj) in enumerate(self.external_archive):
            if self.dominates(objectives, arch_obj):
                dominated_indices.append(idx)
            elif self.dominates(arch_obj, objectives):
                return

        for idx in reversed(dominated_indices):
            del self.external_archive[idx]

        self.external_archive.append((solution, objectives))

    def run(self):
        """Main MOEA/D loop"""
        print(f"\nStarting MOEA/D Clean...")
        print("="*60)

        for generation in range(self.max_gen):
            for i in range(self.pop_size):
                k = np.random.randint(0, 2)

                if k == 0:
                    indices = self.B[i]
                else:
                    indices = np.random.choice(self.pop_size, self.n_neighbors, replace=False)

                parent1_idx = indices[np.random.randint(0, len(indices))]
                parent2_idx = indices[np.random.randint(0, len(indices))]

                parent1 = self.population[parent1_idx]['chromosome']
                parent2 = self.population[parent2_idx]['chromosome']

                offspring = self.crossover(parent1, parent2)
                offspring = self.mutation(offspring)

                offspring_obj = self.evaluate_objectives(offspring)

                for j in range(self.n_objectives):
                    if offspring_obj[j] < self.z[j]:
                        self.z[j] = offspring_obj[j]

                update_count = 0
                max_updates = int(self.pop_size * 0.3)

                for j in indices:
                    if update_count >= max_updates:
                        break

                    if self.tchebycheff(offspring_obj, self.weights[j], self.z) < \
                       self.tchebycheff(self.population[j]['objectives'], self.weights[j], self.z):
                        self.population[j] = {
                            'chromosome': offspring,
                            'objectives': offspring_obj
                        }
                        update_count += 1

                self.update_external_archive(offspring, offspring_obj)

            if generation % 10 == 0:
                best_lu = max([-ind['objectives'][0] for ind in self.population])
                best_ss = max([-ind['objectives'][1] for ind in self.population])
                best_rss = min([ind['objectives'][2] for ind in self.population])

                print(f"Generation {generation}: Best LU={best_lu:.0f}, SS={best_ss:.4f}, RSS={best_rss:.0f}")

                if self.track_metrics and self.external_archive:
                    archive_objs = np.array([obj for _, obj in self.external_archive])
                    archive_objs_normalized = np.array([self.normalize_objectives(obj) for obj in archive_objs])
                    hv = self.metrics_calculator.hypervolume(archive_objs_normalized, ref_point=[1, 1, 1])
                    self.metrics_history['hypervolume'].append(hv)
                    print(f"  Metrics: HV={hv:.4f}, Archive={len(self.external_archive)}")

        print("\n" + "="*60)
        print("MOEA/D Clean completed")
        print("-"*60)
        if self.track_metrics and self.metrics_history['hypervolume']:
            final_hv = self.metrics_history['hypervolume'][-1]
            print(f"Final Hypervolume: {final_hv:.4f}")
            print(f"Archive Size: {len(self.external_archive)}")
        print("="*60)

        if self.external_archive:
            pareto_front = []
            for chromosome, objectives in self.external_archive:
                pareto_front.append({
                    'chromosome': chromosome,
                    'objectives': objectives
                })
            return pareto_front

        pareto_front = []
        for i, ind_i in enumerate(self.population):
            dominated = False
            for j, ind_j in enumerate(self.population):
                if i != j and self.dominates(ind_j['objectives'], ind_i['objectives']):
                    dominated = True
                    break

            if not dominated:
                pareto_front.append({
                    'chromosome': ind_i['chromosome'],
                    'objectives': ind_i['objectives']
                })

        return pareto_front

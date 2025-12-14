"""
nsga2_timed.py

Descrição:
    NSGA-II com budget de tempo fixo para comparação justa.
    Roda até atingir o tempo limite em vez de número fixo de gerações.

Referências:
    - [1]: Deb et al. (2002) IEEE TEVC 6(2):182-197 - NSGA-II

Origem da implementação:
    - Adaptação do NSGA-II padrão para time budget

"""

import numpy as np
import pickle
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.quality_metrics import QualityMetrics


class NSGA2_Timed:

    def __init__(self, main_package, pop_size=50, time_budget=20.0, track_metrics=False):
        self.main_package = main_package
        self.pop_size = pop_size
        self.time_budget = time_budget
        self.min_size = 2
        self.max_size = 15
        self.ideal_size = 5
        self.track_metrics = track_metrics

        self.load_all_data()
        self.initialize_semantic_components()
        self.compute_candidate_pools()

        if self.track_metrics:
            self.metrics_calculator = QualityMetrics()
            self.metrics_history = {
                'hypervolume': [],
                'spacing': []
            }

        print(f"NSGA-II Timed initialized for '{main_package}'")
        print(f"Time budget: {time_budget}s, Population: {pop_size}")

    def load_all_data(self):
        print("Loading data matrices...")

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
            raise ValueError(f"Package '{self.main_package}' not found in dataset")
        self.main_package_idx = self.package_names.index(self.main_package)

        print(f"Data loaded: {self.n_packages} packages")

    def initialize_semantic_components(self):
        print("Initializing semantic components...")

        n_clusters = min(200, self.n_packages // 50)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
        self.cluster_labels = kmeans.fit_predict(self.embeddings)
        self.target_cluster = self.cluster_labels[self.main_package_idx]

        cluster_members = np.sum(self.cluster_labels == self.target_cluster)
        print(f"Target package in cluster {self.target_cluster} with {cluster_members} members")

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

        print(f"Candidate pools: cooccur={len(self.cooccur_candidates)}, "
              f"semantic={len(self.semantic_candidates)}, cluster={len(self.cluster_candidates)}")

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

        return np.array([-lu_score, -ss_score, rss_score])

    def smart_initialization(self, exploration_rate=0.3):
        chromosome = np.zeros(self.n_packages)
        target_size = random.randint(3, 7)

        strategy = random.random()

        if strategy < 0.4:
            if len(self.cooccur_candidates) > 0:
                n_select = min(target_size, len(self.cooccur_candidates))
                selected = np.random.choice(self.cooccur_candidates[:50], n_select, replace=False)
                chromosome[selected] = 1

        elif strategy < 0.8:
            if len(self.semantic_candidates) > 0:
                n_select = min(target_size, len(self.semantic_candidates))
                selected = np.random.choice(self.semantic_candidates[:50], n_select, replace=False)
                chromosome[selected] = 1

        else:
            n_cooccur = target_size // 2
            n_semantic = target_size - n_cooccur

            if len(self.cooccur_candidates) > 0 and n_cooccur > 0:
                selected = np.random.choice(self.cooccur_candidates[:30], min(n_cooccur, len(self.cooccur_candidates)), replace=False)
                chromosome[selected] = 1

            available_semantic = [c for c in self.semantic_candidates[:30] if chromosome[c] == 0]
            if available_semantic and n_semantic > 0:
                selected = np.random.choice(available_semantic, min(n_semantic, len(available_semantic)), replace=False)
                chromosome[selected] = 1

        if random.random() < exploration_rate:
            n_random = random.randint(1, 2)
            available = [i for i in range(self.n_packages) if chromosome[i] == 0 and i != self.main_package_idx]
            if available:
                random_selection = random.sample(available, min(n_random, len(available)))
                chromosome[random_selection] = 1

        return chromosome

    def dominates(self, obj1, obj2):
        return all(obj1 <= obj2) and any(obj1 < obj2)

    def fast_non_dominated_sort(self, population):
        n = len(population)
        S = [[] for _ in range(n)]
        n_dom = [0] * n
        rank = [0] * n
        fronts = [[]]

        for p in range(n):
            for q in range(n):
                if self.dominates(population[p]['objectives'], population[q]['objectives']):
                    S[p].append(q)
                elif self.dominates(population[q]['objectives'], population[p]['objectives']):
                    n_dom[p] += 1

            if n_dom[p] == 0:
                rank[p] = 0
                fronts[0].append(p)

        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in S[p]:
                    n_dom[q] -= 1
                    if n_dom[q] == 0:
                        rank[q] = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        return fronts[:-1]

    def crowding_distance_assignment(self, front_population):
        n = len(front_population)
        if n <= 2:
            for ind in front_population:
                ind['crowding_distance'] = float('inf')
            return

        for ind in front_population:
            ind['crowding_distance'] = 0

        for m in range(3):
            sorted_pop = sorted(front_population, key=lambda x: x['objectives'][m])

            sorted_pop[0]['crowding_distance'] = float('inf')
            sorted_pop[-1]['crowding_distance'] = float('inf')

            obj_min = sorted_pop[0]['objectives'][m]
            obj_max = sorted_pop[-1]['objectives'][m]
            obj_range = obj_max - obj_min

            if obj_range > 0:
                for i in range(1, n - 1):
                    sorted_pop[i]['crowding_distance'] += (
                        sorted_pop[i + 1]['objectives'][m] - sorted_pop[i - 1]['objectives'][m]
                    ) / obj_range

    def tournament_selection(self, population):
        i, j = random.sample(range(len(population)), 2)

        if population[i].get('rank', 0) < population[j].get('rank', 0):
            return population[i]
        elif population[i].get('rank', 0) > population[j].get('rank', 0):
            return population[j]
        else:
            if population[i].get('crowding_distance', 0) > population[j].get('crowding_distance', 0):
                return population[i]
            else:
                return population[j]

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

    def survivor_selection(self, population, offspring):
        combined = population + offspring

        fronts = self.fast_non_dominated_sort(combined)

        for i, front in enumerate(fronts):
            for idx in front:
                combined[idx]['rank'] = i

        new_population = []
        for front in fronts:
            if len(new_population) + len(front) <= self.pop_size:
                new_population.extend([combined[i] for i in front])
            else:
                remaining = self.pop_size - len(new_population)
                if remaining > 0:
                    front_individuals = [combined[i] for i in front]
                    self.crowding_distance_assignment(front_individuals)
                    front_individuals.sort(key=lambda x: x['crowding_distance'], reverse=True)
                    new_population.extend(front_individuals[:remaining])
                break

        return new_population

    def run(self):
        print(f"\nStarting NSGA-II (Time Budget: {self.time_budget}s)")
        print("=" * 60)

        start_time = time.time()

        print("Initializing population...")
        population = []
        for i in range(self.pop_size):
            chromosome = self.smart_initialization()
            objectives = self.evaluate_objectives(chromosome)
            population.append({
                'chromosome': chromosome,
                'objectives': objectives
            })

        print(f"Population initialized with {len(population)} individuals")

        generation = 0
        while True:
            elapsed = time.time() - start_time
            if elapsed >= self.time_budget:
                print(f"\nTime budget reached ({self.time_budget}s)")
                break

            offspring = []
            for _ in range(self.pop_size):
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)

                child_chromosome = self.crossover(parent1['chromosome'], parent2['chromosome'])
                child_chromosome = self.mutation(child_chromosome)
                child_objectives = self.evaluate_objectives(child_chromosome)

                offspring.append({
                    'chromosome': child_chromosome,
                    'objectives': child_objectives
                })

            population = self.survivor_selection(population, offspring)

            if generation % 10 == 0:
                fronts = self.fast_non_dominated_sort(population)
                pareto_size = len(fronts[0]) if fronts else 0
                best = min(population, key=lambda x: x['objectives'][0])
                print(f"Gen {generation}: Pareto size={pareto_size}, Time={elapsed:.1f}s")
                print(f"  Best: LU={-best['objectives'][0]:.2f}, SS={-best['objectives'][1]:.4f}, RSS={best['objectives'][2]:.1f}")

            generation += 1

        fronts = self.fast_non_dominated_sort(population)
        pareto_front = [population[i] for i in fronts[0]] if fronts else population

        total_time = time.time() - start_time
        print(f"\nNSGA-II completed: {generation} generations in {total_time:.1f}s")
        print(f"Pareto front size: {len(pareto_front)}")

        solutions = []
        for sol in pareto_front:
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
    optimizer = NSGA2_Timed(
        main_package='fastapi',
        pop_size=50,
        time_budget=20.0
    )
    solutions = optimizer.run()
    print(f"\nFound {len(solutions)} Pareto-optimal solutions")

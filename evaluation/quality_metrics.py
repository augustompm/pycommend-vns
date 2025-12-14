"""
quality_metrics.py

Description:
    Quality metrics for multi-objective optimization evaluation.
    Implements Hypervolume, IGD+, Spacing, Spread, and related indicators.

References:
    - [1]: Zitzler & Thiele (1999) IEEE TEVC - Hypervolume indicator
    - [2]: Ishibuchi et al. (2015) EMO - IGD+ indicator
    - [3]: Schott (1995) - Spacing metric

Implementation origin:
    - Standard implementations from multi-objective optimization literature
"""

import numpy as np
from scipy.spatial.distance import cdist
from itertools import combinations_with_replacement
import warnings


class QualityMetrics:
    """
    Complete suite of quality metrics for multi-objective optimization
    """

    def __init__(self, reference_set=None, ideal_point=None, nadir_point=None):
        """
        Initialize metrics calculator

        Args:
            reference_set: Reference Pareto front for IGD calculations
            ideal_point: Ideal point for normalization
            nadir_point: Nadir point for normalization
        """
        self.reference_set = reference_set
        self.ideal_point = ideal_point
        self.nadir_point = nadir_point

    def normalize_objectives(self, objectives):
        """
        Normalize objectives to [0, 1] range
        """
        if self.ideal_point is None:
            self.ideal_point = np.min(objectives, axis=0)
        if self.nadir_point is None:
            self.nadir_point = np.max(objectives, axis=0)

        range_obj = self.nadir_point - self.ideal_point
        range_obj[range_obj == 0] = 1.0

        return (objectives - self.ideal_point) / range_obj

    def filter_dominated(self, objectives):
        """
        Filter dominated solutions, keep only non-dominated
        """
        n = len(objectives)
        is_dominated = np.zeros(n, dtype=bool)

        for i in range(n):
            if is_dominated[i]:
                continue
            for j in range(n):
                if i == j or is_dominated[j]:
                    continue
                if np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                    is_dominated[i] = True
                    break

        return objectives[~is_dominated]

    def hypervolume(self, objectives, ref_point=None):
        """
        Calculate hypervolume indicator

        Args:
            objectives: Set of objective vectors
            ref_point: Reference point (default: 1.1 * max for each objective)

        Returns:
            Hypervolume value
        """
        if len(objectives) == 0:
            return 0.0

        pareto_front = self.filter_dominated(objectives)

        if len(pareto_front) == 0:
            return 0.0

        if ref_point is None:
            ref_point = np.max(pareto_front, axis=0) * 1.1
        else:
            ref_point = np.array(ref_point)

        n_obj = pareto_front.shape[1]

        if n_obj == 2:
            return self._hv_2d(pareto_front, ref_point)
        elif n_obj == 3:
            return self._hv_3d_exact(pareto_front, ref_point)
        else:
            return self._hv_monte_carlo(pareto_front, ref_point)

    def _hv_2d(self, points, ref_point):
        """
        Calculate 2D hypervolume exactly
        """
        points = points[points[:, 0].argsort()]

        volume = 0.0
        prev_x = 0.0

        for point in points:
            if point[0] > ref_point[0] or point[1] > ref_point[1]:
                continue

            width = point[0] - prev_x
            height = ref_point[1] - point[1]

            if width > 0 and height > 0:
                volume += width * height

            prev_x = point[0]

        if prev_x < ref_point[0]:
            volume += (ref_point[0] - prev_x) * ref_point[1]

        return volume

    def _hv_3d(self, points, ref_point):
        """
        Calculate 3D hypervolume (simplified)
        """
        volume = 0.0

        for point in points:
            if np.any(point > ref_point):
                continue

            box_volume = np.prod(ref_point - point)
            volume += box_volume

        return volume / len(points)

    def _hv_3d_exact(self, points, ref_point):
        """
        Calculate exact 3D hypervolume using a proper algorithm
        """
        if len(points) == 0:
            return 0.0

        points = points[points[:, 0].argsort()]

        total_volume = 0.0

        for i, point in enumerate(points):
            if np.any(point > ref_point):
                continue

            vol = 1.0
            for j in range(3):
                vol *= abs(ref_point[j] - point[j])

            for j in range(i):
                if np.all(points[j] <= point):
                    overlap = 1.0
                    for k in range(3):
                        overlap *= max(0, min(abs(ref_point[k] - point[k]),
                                              abs(ref_point[k] - points[j][k])))
                    vol -= overlap

            if vol > 0:
                total_volume += vol

        return total_volume

    def _hv_monte_carlo(self, points, ref_point, n_samples=10000):
        """
        Monte Carlo approximation for high-dimensional hypervolume
        """
        samples = np.random.uniform(0, ref_point, (n_samples, len(ref_point)))

        dominated_count = 0

        for sample in samples:
            for point in points:
                if np.all(point <= sample):
                    dominated_count += 1
                    break

        ref_volume = np.prod(ref_point)
        return (dominated_count / n_samples) * ref_volume

    def igd_plus(self, objectives, reference_set=None):
        """
        Calculate IGD+ (Inverted Generational Distance Plus)

        Args:
            objectives: Set of objective vectors
            reference_set: Reference Pareto front

        Returns:
            IGD+ value (lower is better)
        """
        if reference_set is None:
            if self.reference_set is None:
                raise ValueError("Reference set required for IGD+")
            reference_set = self.reference_set

        # Normalize both sets
        norm_obj = self.normalize_objectives(objectives)
        norm_ref = self.normalize_objectives(reference_set)

        total_distance = 0.0

        for ref_point in norm_ref:
            min_distance = float('inf')

            for obj_point in norm_obj:
                diff = np.maximum(ref_point - obj_point, 0)
                distance = np.linalg.norm(diff)

                if distance < min_distance:
                    min_distance = distance

            total_distance += min_distance

        return total_distance / len(norm_ref)

    def igd(self, objectives, reference_set=None):
        """
        Calculate standard IGD (for comparison)

        Args:
            objectives: Set of objective vectors
            reference_set: Reference Pareto front

        Returns:
            IGD value (lower is better)
        """
        if reference_set is None:
            if self.reference_set is None:
                raise ValueError("Reference set required for IGD")
            reference_set = self.reference_set

        norm_obj = self.normalize_objectives(objectives)
        norm_ref = self.normalize_objectives(reference_set)

        distances = cdist(norm_ref, norm_obj)
        min_distances = np.min(distances, axis=1)

        return np.mean(min_distances)

    def spacing(self, objectives):
        """
        Calculate spacing metric (uniformity of distribution)

        Args:
            objectives: Set of objective vectors

        Returns:
            Spacing value (lower is better)
        """
        if len(objectives) < 2:
            return 0.0

        norm_obj = self.normalize_objectives(objectives)

        n = len(norm_obj)
        distances = []

        for i in range(n):
            min_dist = float('inf')
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(norm_obj[i] - norm_obj[j])
                    if dist < min_dist:
                        min_dist = dist
            distances.append(min_dist)

        mean_dist = np.mean(distances)
        if mean_dist == 0:
            return 0.0

        spacing = np.sqrt(np.sum((distances - mean_dist) ** 2) / (n - 1))
        return spacing

    def get_non_dominated_set(self, objectives):
        """Get the non-dominated set from a set of objectives"""
        non_dominated = []
        for i, obj_i in enumerate(objectives):
            is_dominated = False
            for j, obj_j in enumerate(objectives):
                if i != j and self._dominates(obj_j, obj_i):
                    is_dominated = True
                    break
            if not is_dominated:
                non_dominated.append(obj_i)
        return np.array(non_dominated) if non_dominated else np.array([])

    def _dominates(self, a, b):
        """Check if solution a dominates solution b"""
        return all(a <= b) and any(a < b)

    def spread(self, objectives):
        """
        Calculate spread metric (distribution and extent)

        Args:
            objectives: Set of objective vectors

        Returns:
            Spread value (lower is better)
        """
        if len(objectives) < 3:
            return 1.0

        norm_obj = self.normalize_objectives(objectives)

        n = len(norm_obj)
        n_obj = norm_obj.shape[1]

        extreme_points = []
        for i in range(n_obj):
            min_idx = np.argmin(norm_obj[:, i])
            max_idx = np.argmax(norm_obj[:, i])
            extreme_points.extend([min_idx, max_idx])

        extreme_points = list(set(extreme_points))

        distances = []
        for i in range(n):
            min_dist = float('inf')
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(norm_obj[i] - norm_obj[j])
                    if dist < min_dist:
                        min_dist = dist
            if min_dist < float('inf'):
                distances.append(min_dist)

        if len(distances) == 0:
            return 1.0

        mean_dist = np.mean(distances)

        df = dl = 0
        if len(extreme_points) >= 2:
            df = np.min([np.linalg.norm(norm_obj[i] - norm_obj[extreme_points[0]])
                        for i in range(n) if i != extreme_points[0]])
            dl = np.min([np.linalg.norm(norm_obj[i] - norm_obj[extreme_points[-1]])
                        for i in range(n) if i != extreme_points[-1]])

        numerator = df + dl + np.sum(np.abs(distances - mean_dist))
        denominator = df + dl + (len(distances)) * mean_dist

        if denominator == 0:
            return 1.0

        return numerator / denominator

    def r2_indicator(self, objectives, weight_vectors=None, ideal_point=None):
        """
        Calculate R2 indicator (utility-based metric)
        Weakly Pareto compliant, correlated with HV

        Args:
            objectives: Set of objective vectors
            weight_vectors: Weight vectors for scalarization
            ideal_point: Ideal point for normalization

        Returns:
            R2 value (lower is better)
        """
        if len(objectives) == 0:
            return float('inf')

        norm_obj = self.normalize_objectives(objectives)

        if weight_vectors is None:
            n_weights = min(100, len(objectives) * 5)
            weight_vectors = []
            for _ in range(n_weights):
                w = np.random.random(norm_obj.shape[1])
                w = w / np.sum(w)
                weight_vectors.append(w)
            weight_vectors = np.array(weight_vectors)

        if ideal_point is None:
            ideal_point = np.zeros(norm_obj.shape[1])

        utilities = []
        for weight in weight_vectors:
            min_utility = float('inf')
            for obj in norm_obj:
                utility = np.max(weight * (obj - ideal_point))
                if utility < min_utility:
                    min_utility = utility
            utilities.append(min_utility)

        return np.mean(utilities)

    def epsilon_indicator(self, objectives, reference_set=None):
        """
        Calculate epsilon indicator (additive version)
        Weakly Pareto compliant metric

        Args:
            objectives: Set of objective vectors from algorithm
            reference_set: Reference Pareto front (or use objectives if None)

        Returns:
            Epsilon value (lower is better)
        """
        if len(objectives) == 0:
            return float('inf')

        if reference_set is None:
            if self.reference_set is None:
                return 0.0
            reference_set = self.reference_set

        norm_obj = self.normalize_objectives(objectives)
        norm_ref = self.normalize_objectives(reference_set)

        max_epsilon = -float('inf')

        for ref_point in norm_ref:
            min_epsilon = float('inf')

            for obj_point in norm_obj:
                epsilon = np.max(obj_point - ref_point)
                if epsilon < min_epsilon:
                    min_epsilon = epsilon

            if min_epsilon > max_epsilon:
                max_epsilon = min_epsilon

        return max_epsilon

    def diversity(self, objectives):
        """
        Calculate diversity metric

        Args:
            objectives: Set of objective vectors

        Returns:
            Diversity value (higher is better)
        """
        if len(objectives) < 2:
            return 0.0

        norm_obj = self.normalize_objectives(objectives)

        distances = cdist(norm_obj, norm_obj)
        np.fill_diagonal(distances, 0)

        k = min(5, len(norm_obj) - 1)
        diversity_scores = []

        for i in range(len(norm_obj)):
            row_distances = distances[i]
            row_distances[i] = float('inf')
            k_nearest = np.sort(row_distances)[:k]
            diversity_scores.append(np.mean(k_nearest))

        return np.mean(diversity_scores)

    def maximum_spread(self, objectives):
        """
        Calculate maximum spread in each objective

        Args:
            objectives: Set of objective vectors

        Returns:
            Maximum spread value (higher is better)
        """
        norm_obj = self.normalize_objectives(objectives)

        spreads = []
        for i in range(norm_obj.shape[1]):
            obj_range = np.max(norm_obj[:, i]) - np.min(norm_obj[:, i])
            spreads.append(obj_range ** 2)

        return np.sqrt(np.sum(spreads))

    def evaluate_all(self, objectives, reference_set=None):
        """
        Calculate all metrics at once

        Args:
            objectives: Set of objective vectors
            reference_set: Reference Pareto front (optional)

        Returns:
            Dictionary with all metric values
        """
        results = {
            'n_solutions': len(objectives),
            'n_nondominated': len(self.filter_dominated(objectives))
        }

        results['hypervolume'] = self.hypervolume(objectives)
        results['spacing'] = self.spacing(objectives)
        results['spread'] = self.spread(objectives)
        results['diversity'] = self.diversity(objectives)
        results['maximum_spread'] = self.maximum_spread(objectives)

        if reference_set is not None or self.reference_set is not None:
            results['igd'] = self.igd(objectives, reference_set)
            results['igd_plus'] = self.igd_plus(objectives, reference_set)

        return results

    def generate_reference_set(self, n_points, n_objectives):
        """
        Generate uniform reference points for IGD calculation

        Args:
            n_points: Number of reference points
            n_objectives: Number of objectives

        Returns:
            Array of reference points
        """
        if n_objectives == 2:
            weights = np.linspace(0, 1, n_points)
            ref_points = np.column_stack([weights, 1 - weights])

        elif n_objectives == 3:
            ref_points = []
            h = int(np.sqrt(n_points))

            for i in range(h):
                for j in range(h):
                    if i + j < h:
                        w1 = i / h
                        w2 = j / h
                        w3 = 1 - w1 - w2
                        ref_points.append([w1, w2, w3])

            ref_points = np.array(ref_points)

            while len(ref_points) < n_points:
                w = np.random.dirichlet(np.ones(n_objectives))
                ref_points = np.vstack([ref_points, w])

            ref_points = ref_points[:n_points]

        else:
            ref_points = np.random.dirichlet(np.ones(n_objectives), n_points)

        return ref_points


def compare_algorithms(objectives1, objectives2, name1="Algorithm 1", name2="Algorithm 2"):
    """
    Compare two algorithms using quality metrics

    Args:
        objectives1: Objectives from algorithm 1
        objectives2: Objectives from algorithm 2
        name1: Name of algorithm 1
        name2: Name of algorithm 2

    Returns:
        Comparison results dictionary
    """
    metrics = QualityMetrics()

    results1 = metrics.evaluate_all(objectives1)
    results2 = metrics.evaluate_all(objectives2)

    comparison = {
        'algorithm_1': name1,
        'algorithm_2': name2,
        'metrics': {}
    }

    for metric in results1.keys():
        value1 = results1[metric]
        value2 = results2[metric]

        if metric in ['hypervolume', 'diversity', 'maximum_spread', 'n_solutions', 'n_nondominated']:
            winner = name1 if value1 > value2 else name2 if value2 > value1 else 'tie'
        else:
            winner = name1 if value1 < value2 else name2 if value2 < value1 else 'tie'

        comparison['metrics'][metric] = {
            name1: value1,
            name2: value2,
            'winner': winner,
            'difference': abs(value1 - value2)
        }

    return comparison
import numpy as np
from numba import jit


class Specific_Sample_Adaptive_Neighborhood:
    def __init__(self, x_data, initial_k):
        self.X = np.array(x_data)
        self.X = np.ascontiguousarray(self.X, dtype=np.float32)
        self.initial_k = initial_k
        self.samples_dist_matrix = []
        self.n_samples = self.X.shape[0]

    def sample_neighborhood(self):
        if self.initial_k < 5 or self.initial_k >= self.n_samples:
            raise ValueError(
                f"initial_k must satisfy 5 <= initial_k < {self.n_samples}, and the current value is {self.initial_k}"
            )

        @jit(nopython=True, fastmath=True)
        def calc_dist(X, i):
            return np.sqrt(np.sum(np.square(X - X[i, :]), axis=1))

        self.samples_dist_matrix = []
        k_neighbor_dict = {}
        for i in range(self.n_samples):
            dists = calc_dist(self.X, i)
            self.samples_dist_matrix.append(dists)
            sorted_indices = np.argsort(dists)
            k_neighbor_dict[i] = sorted_indices[1:1 + self.initial_k]

        return k_neighbor_dict

    def calculate_mean_global_neighbors_distances(self, k_neighbors_dict):
        samples_neighborhood_sum_distance_dict = {}
        for i in range(self.n_samples):
            samples_neighborhood_sum_distance_dict[i] = 0.0
            for j in k_neighbors_dict[i]:
                samples_neighborhood_sum_distance_dict[i] += self.samples_dist_matrix[i][j]

        samples_neighbor_mean = [
            value / self.initial_k
            for value in samples_neighborhood_sum_distance_dict.values()
        ]
        global_mean_distance = np.mean(samples_neighbor_mean)
        return global_mean_distance, samples_neighbor_mean

    def adaptive_neighborhood(self, global_mean_distance, samples_neighbor_mean):
        updated_samples_k_number = {}
        updated_samples_neighbors = {}
        eps = 1e-8

        for i in range(self.n_samples):
            sample_mean = samples_neighbor_mean[i]
            if sample_mean < eps:
                relative_density = 1.0
            else:
                relative_density = global_mean_distance / sample_mean

            if np.isnan(relative_density) or np.isinf(relative_density):
                relative_density = 1.0

            updated_k = relative_density * self.initial_k
            if updated_k > self.initial_k * 2:
                updated_k = self.initial_k * 2
            elif updated_k < 5:
                updated_k = 5
            updated_k = int(updated_k)

            updated_samples_k_number[i] = updated_k
            sorted_distances = np.argsort(self.samples_dist_matrix[i])
            updated_samples_neighbors[i] = sorted_distances[1:1 + updated_k]

        return (
            updated_samples_k_number,
            updated_samples_neighbors,
            np.array(self.samples_dist_matrix, dtype=np.float32),
        )


class LabelDisambiguation:
    def __init__(self, updated_neighbors_dict, dist_matrix, candidate_labels):
        self.updated_neighbors_dict = updated_neighbors_dict
        self.dist_matrix = np.array(dist_matrix, dtype=np.float32)
        self.candidate_labels = np.array(candidate_labels, dtype=np.float32)
        self.n_samples = self.dist_matrix.shape[0]
        self.n_classes = self.candidate_labels.shape[0]

        assert self.dist_matrix.shape[1] == self.n_samples, "The distance matrix has to be an M×M square matrix"
        assert self.candidate_labels.shape[1] == self.n_samples, "The candidate label matrix has to be a Q×M matrix"
        assert len(self.updated_neighbors_dict) == self.n_samples, "The number of samples in the neighborhood dictionary must match the total number of samples."

        self.valid_label_mask = self.candidate_labels.T

    def _calc_sample_weight(self, sample_idx):
        neighbor_indices = self.updated_neighbors_dict[sample_idx]
        if len(neighbor_indices) == 0:
            raise ValueError(f"The neighborhood for sample {sample_idx} is empty, can't compute the weight")

        neighbor_dists = self.dist_matrix[sample_idx, neighbor_indices]
        dist_squared = np.square(neighbor_dists)
        sigma_sq = np.mean(dist_squared) + 1e-8
        numerator = np.exp(-dist_squared / sigma_sq)
        weights = numerator / (np.sum(numerator) + 1e-8)
        return weights, neighbor_indices

    def _calc_label_prob(self, sample_m_idx):
        m_neighbors = self.updated_neighbors_dict[sample_m_idx]
        n_m_neighbors = len(m_neighbors)
        label_prob = np.zeros(self.n_classes, dtype=np.float32)

        if n_m_neighbors == 0:
            m_valid_labels = self.candidate_labels[:, sample_m_idx] == 1
            if np.sum(m_valid_labels) > 0:
                label_prob[m_valid_labels] = 1.0 / np.sum(m_valid_labels)
            return label_prob

        label_counts = np.sum(self.candidate_labels[:, m_neighbors], axis=1)
        label_prob = label_counts / (n_m_neighbors + 1e-8)
        label_prob[self.candidate_labels[:, sample_m_idx] == 0] = 0.0
        return label_prob

    def run_disambiguation(self):
        disambiguated_dist = np.zeros(
            (self.n_samples, self.n_classes), dtype=np.float32
        )

        for i in range(self.n_samples):
            weights, neighbor_indices = self._calc_sample_weight(i)
            for idx, m in enumerate(neighbor_indices):
                label_prob = self._calc_label_prob(m)
                disambiguated_dist[i] += weights[idx] * label_prob

            disambiguated_dist[i][self.valid_label_mask[i] == False] = 0.0
            valid_sum = np.sum(
                disambiguated_dist[i][self.valid_label_mask[i] == True]
            )
            if valid_sum > 1e-8:
                disambiguated_dist[i][self.valid_label_mask[i] == True] /= valid_sum
            else:
                disambiguated_dist[i][self.valid_label_mask[i] == True] = (
                    1.0 / np.sum(self.valid_label_mask[i])
                )

        return disambiguated_dist

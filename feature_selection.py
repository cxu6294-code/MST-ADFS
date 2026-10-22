import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.special import rel_entr
from sklearn.neighbors import NearestNeighbors


class feature_selection:
    def __init__(self, X, LD_matrix, redundancy_weight=0.0):
        self.X = X.astype(np.float32)
        self.LD = LD_matrix.astype(np.float32)
        self.rw = redundancy_weight
        self.N, self.L = self.LD.shape
        self.M = self.X.shape[1]

        epsilon = 1e-8
        self.label_global_mean = np.mean(self.LD, axis=0) + epsilon
        label_total = np.sum(self.LD, axis=0)
        self.label_importance = label_total / (np.sum(label_total) + epsilon)

        if self.rw > 0:
            self.feature_corr_matrix = np.abs(np.corrcoef(self.X, rowvar=False))
            self.feature_corr_matrix = np.nan_to_num(self.feature_corr_matrix, nan=0.0)
        else:
            self.feature_corr_matrix = None

        self._label_sim_cache = None

    def calc_label_similarity_jsd(self):
        if self._label_sim_cache is not None:
            return self._label_sim_cache

        epsilon = 1e-10
        P = (self.LD + epsilon).T
        P = P / np.sum(P, axis=1, keepdims=True)
        sim_matrix = np.eye(self.L)
        dist_matrix = np.zeros((self.L, self.L))

        for i in range(self.L):
            p = P[i]
            m = 0.5 * (P + p)
            js_div = (
                0.5 * np.sum(rel_entr(P, m), axis=1)
                + 0.5 * np.sum(rel_entr(p, m), axis=1)
            )
            js_dist = np.sqrt(np.maximum(js_div, 0))
            dist_matrix[i, :] = js_dist
            sim_matrix[i, :] = 1.0 - js_dist

        self._label_sim_cache = (sim_matrix, dist_matrix)
        return sim_matrix, dist_matrix

    def get_high_similar_label_pairs(self, c=0.5):
        sim_matrix, dist_matrix = self.calc_label_similarity_jsd()
        mst = minimum_spanning_tree(dist_matrix).toarray()
        u, v = np.where(mst > 0)
        mst_pairs = {tuple(sorted((i, j))) for i, j in zip(u, v)}
        u_th, v_th = np.where(np.triu(sim_matrix, k=1) >= c)
        thresh_pairs = {tuple(sorted((i, j))) for i, j in zip(u_th, v_th)}
        return sorted(list(mst_pairs.union(thresh_pairs))), sim_matrix

    def _get_knn_indices(self, feat_idx, k):
        if not feat_idx:
            return None
        nbrs = NearestNeighbors(
            n_neighbors=k, algorithm="auto", n_jobs=1
        ).fit(self.X[:, feat_idx])
        return nbrs.kneighbors(return_distance=False)

    def _calc_gamma_with_indices(self, indices, label_pairs, sim_matrix, beta, rho):
        gamma1 = np.zeros(self.L)
        for l in range(self.L):
            xi = self.LD[:, l].reshape(-1, 1)
            neighbor_ld = self.LD[indices, l]
            is_pos = np.all(
                1 - (np.abs(xi - neighbor_ld) / self.label_global_mean[l]) >= beta,
                axis=1,
            )
            gamma1[l] = (np.sum(is_pos) / self.N) * self.label_importance[l]

        gamma2_dict = {}
        for lk, lm in label_pairs:
            xi_k = self.LD[:, lk].reshape(-1, 1)
            xi_m = self.LD[:, lm].reshape(-1, 1)
            diff_k = np.abs(xi_k - self.LD[indices, lk]) / self.label_global_mean[lk]
            diff_m = np.abs(xi_m - self.LD[indices, lm]) / self.label_global_mean[lm]
            is_pos = np.all(1 - 0.5 * (diff_k + diff_m) >= rho, axis=1)
            w_geo = np.sqrt(self.label_importance[lk] * self.label_importance[lm])
            gamma2_dict[(lk, lm)] = (
                sim_matrix[lk, lm] * w_geo * (np.sum(is_pos) / self.N)
            )

        return gamma1, gamma2_dict

    def cal_strong_correlation_gain(self, delta, avg_base):
        if avg_base < 1e-8:
            return delta
        return delta * (delta / avg_base) if delta > avg_base else delta

    def forward_search_feature_selection(self, beta, rho, c, k):
        remaining_feat = list(range(self.M))
        final_order = []
        label_pairs, sim_matrix = self.get_high_similar_label_pairs(c)
        lambda_val = (
            self.L / (self.L + len(label_pairs))
            if (self.L + len(label_pairs)) > 0 else 1
        )

        while remaining_feat:
            base_indices = self._get_knn_indices(remaining_feat, k)
            base_g1, base_g2_dict = self._calc_gamma_with_indices(
                base_indices, label_pairs, sim_matrix, beta, rho
            )
            best_score, best_feature = -np.inf, None
            len_A = max(len(remaining_feat) - 1, 1)

            for feat in remaining_feat:
                temp_feat_set = [f for f in remaining_feat if f != feat]
                if not temp_feat_set:
                    sig_g1, sig_g2 = 0.0, 0.0
                else:
                    after_indices = self._get_knn_indices(temp_feat_set, k)
                    after_g1, after_g2_dict = self._calc_gamma_with_indices(
                        after_indices, label_pairs, sim_matrix, beta, rho
                    )
                    sig_g1 = np.mean([
                        self.cal_strong_correlation_gain(
                            base_g1[i] - after_g1[i], after_g1[i] / len_A
                        )
                        for i in range(self.L)
                    ])
                    sig_g2 = np.mean([
                        self.cal_strong_correlation_gain(
                            base_g2_dict[p] - after_g2_dict[p],
                            after_g2_dict[p] / len_A,
                        )
                        for p in label_pairs
                    ]) if label_pairs else 0.0

                relevance = lambda_val * sig_g1 + (1 - lambda_val) * sig_g2
                redundancy = (
                    np.mean([
                        self.feature_corr_matrix[feat, selected]
                        for selected in final_order
                    ])
                    if self.rw > 0 and final_order else 0.0
                )
                mrmr_score = relevance - (self.rw * redundancy)

                if mrmr_score > best_score:
                    best_score = mrmr_score
                    best_feature = feat

            remaining_feat.remove(best_feature)
            final_order.append(best_feature)

        return final_order

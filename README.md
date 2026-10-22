# MST-ADFS
# Label Disambiguation and Feature Selection

This project implements label disambiguation using sample-specific adaptive neighborhoods, followed by feature ranking based on the disambiguated label distributions. The code contains the core computations and returns their results directly.

## Files

| File | Purpose |
| --- | --- |
| `disambiguation.py` | Computes sample-specific adaptive neighborhoods and produces disambiguated label distributions subject to candidate-label constraints. |
| `feature_selection.py` | Ranks feature indices using label distributions, label similarities, and feature redundancy. |

## Dependencies

- NumPy
- Numba
- SciPy
- scikit-learn

## Label Disambiguation

`disambiguation.py` contains two classes.

### `Specific_Sample_Adaptive_Neighborhood`

This class computes pairwise Euclidean distances and estimates relative density as the ratio of the global mean neighbor distance to each sample's local mean neighbor distance. It then adjusts the number of neighbors for each sample accordingly.

Constructor: `Specific_Sample_Adaptive_Neighborhood(x_data, initial_k)`.

| Method | Returns |
| --- | --- |
| `sample_neighborhood()` | A dictionary of initial neighbor indices for each sample. The distance matrix is also stored in the object. |
| `calculate_mean_global_neighbors_distances(k_neighbors_dict)` | The global mean neighbor distance and the mean neighbor distance for each sample. |
| `adaptive_neighborhood(global_mean_distance, samples_neighbor_mean)` | A dictionary of adaptive target neighbor counts, a dictionary of actual neighbor indices, and the sample distance matrix. |

The methods are evaluated in the order shown above. Adaptive target neighbor counts are bounded between `5` and `2 * initial_k`; the actual number of available neighbors cannot exceed the number of samples minus one.

### `LabelDisambiguation`

Constructor: `LabelDisambiguation(updated_neighbors_dict, dist_matrix, candidate_labels)`.

The algorithm computes neighbor weights using a Gaussian kernel with a local distance scale. It calculates candidate-label frequencies within each neighbor's own neighborhood and aggregates these frequencies using the neighbor weights to obtain the current sample's label distribution. Candidate-label constraints are preserved: non-candidate labels receive zero weight, and weights are normalized over candidate labels. If the total weight over candidate labels is too small, a uniform distribution over those labels is used.

`run_disambiguation()` returns the disambiguated label distribution matrix, which represents relative confidence in candidate labels rather than binary assignments or final predicted labels.

## Feature Selection

`feature_selection.py` provides the `feature_selection` class.

Constructor: `feature_selection(X, LD_matrix, redundancy_weight=0.0)`.

The core computations are:

1. Compute Jensen–Shannon distances between label distributions across samples and define label similarity as `1 - distance`.
2. Combine label pairs from a minimum spanning tree with pairs that meet the similarity threshold.
3. Evaluate neighborhood consistency for individual labels and label pairs, incorporating label importance into feature evaluation.
4. At each round, compare evaluation scores before and after removing each feature from the remaining feature set. Combine this contribution with a correlation-based redundancy penalty relative to previously selected features to choose the next feature.

| Method | Returns |
| --- | --- |
| `calc_label_similarity_jsd()` | The label similarity matrix and label distance matrix. |
| `get_high_similar_label_pairs(c=0.5)` | A list of retained label pairs and the label similarity matrix. |
| `forward_search_feature_selection(beta, rho, c, k)` | A list of all feature indices in their successive selection order. |

Although the ranking method is named `forward_search_feature_selection`, feature contributions are evaluated through removal from the current remaining feature set. Returned indices are zero-based, and features appearing earlier in the list are selected earlier. The algorithm does not automatically determine how many features to retain.

## Parameters

| Parameter | Description |
| --- | --- |
| `initial_k` | Initial neighbor count for constructing sample-specific adaptive neighborhoods. Must satisfy `5 <= initial_k < number of samples`. |
| `beta` | Neighborhood consistency threshold for individual labels during feature evaluation. |
| `rho` | Neighborhood consistency threshold for label pairs during feature evaluation. |
| `c` | Similarity threshold for retaining highly similar label pairs. Pairs from the minimum spanning tree are also retained. |
| `k` | Neighbor count used during feature evaluation. Must satisfy `1 <= k < number of samples`. It does not specify the number of selected features. |


## Matrix Conventions and Module Relationship

Let `N` denote the number of samples, `D` the number of features, and `Q` the number of labels.

| Object | Shape or Meaning |
| --- | --- |
| `x_data`, `X` | `N × D` feature matrix. |
| `candidate_labels` | `Q × N` candidate-label matrix. A value of `1` indicates a candidate label, and `0` indicates a non-candidate label. Each sample must have at least one candidate label. |
| `dist_matrix` | `N × N` sample distance matrix. |
| Return value of `run_disambiguation()` | `N × Q` disambiguated label distribution matrix. |
| `LD_matrix` | `N × Q` label distribution matrix, which can be taken directly from the label disambiguation module's return value. |

The modules connect in the following order: adaptive neighborhood computation → label disambiguation → feature ranking. Sample ordering must remain consistent across all matrices and neighborhood dictionaries.

The project does not include data preprocessing, file loading or saving, batch experiments, or a standalone execution entry point.

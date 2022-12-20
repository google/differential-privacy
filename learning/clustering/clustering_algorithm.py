# Copyright 2021 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Implement private clustering."""

import dataclasses
import typing

from absl import logging
import numpy as np
import sklearn.cluster

from clustering import clustering_params
from clustering import coreset_params
from clustering import default_clustering_params
from clustering import lsh_tree
from clustering import privacy_calculator
from clustering import private_outputs


class ClusteringMetrics():
  """Class for computing various clustering quality metrics.

  Note: This class is relevant only for data with specified ground truth labels.

  1. Dominant Label Accuracy: For each cluster, as indicated by cluster
  labels, the accuracy is computed for the labeling of the points that assigns
  the most frequently occurring ground truth label to each cluster.

  2. True Non-matches: Fraction of pairs of points with the same ground truth
  label, which get assigned to different clusters.
  The number of pairs of points with the same true label present in different
  clusters is computed as follows: For a single ground truth label with a
  histogram of cluster labels as (n_1, ... , n_k), the number of pairs of points
  in different clusters is given as
  ((n_1 + ... + n_k)^2 - (n_1^2 + ... + n_k^2))/2.

  3. False Matches: Fraction of pairs of points with different ground truth
  labels, which get assigned to the same cluster.
  The number of pairs of points with different true labels in the same cluster
  can also be computed similarly as above.

  Attributes:
    cross_label_histogram: 2D histogram of (cluster label, true label) pairs.
    num_points: total number of points
    dominant_label_correct_count: number of labels correctly predicted
    dominant_label_accuracy: ratio of dominant_label_correct_count and
      num_points
    true_pairs: number of pairs of points with the same true label.
    true_nonmatch_count: number of pairs of points with same true label, but
      assigned to different clusters.
    true_nonmatch_frac: ratio of true_nonmatch_count and true_pairs
    false_pairs: number of pairs of points with different true labels.
    false_match_count: number of pairs of points with different true labels, but
      assigned to the same cluster.
    false_match_frac: ratio of false_match_count and false_pairs
  """
  cross_label_histogram: np.ndarray
  num_points: int
  dominant_label_correct_count: int
  dominant_label_accuracy: float
  true_pairs: int
  true_nonmatch_count: int
  true_nonmatch_frac: float
  false_pairs: int
  false_match_count: int
  false_match_frac: float

  def __init__(self, cross_label_histogram: np.ndarray):
    self.cross_label_histogram = cross_label_histogram
    self.num_points = np.sum(cross_label_histogram)
    hist_square_sum = np.sum(cross_label_histogram**2)
    num_pairs = self.num_points * (self.num_points - 1) / 2

    # Dominant Label Accuracy
    self.dominant_label_correct_count = np.sum(
        np.max(cross_label_histogram, axis=1))
    self.dominant_label_accuracy = (
        self.dominant_label_correct_count / self.num_points)

    # True Non-matches
    true_label_count = np.sum(cross_label_histogram, axis=0)
    self.true_pairs = np.sum(true_label_count * (true_label_count - 1) / 2)
    self.true_nonmatch_count = (
        (np.sum(true_label_count**2) - hist_square_sum) / 2)
    self.true_nonmatch_frac = self.true_nonmatch_count / self.true_pairs

    # False Matches
    cluster_label_count = np.sum(cross_label_histogram, axis=1)
    self.false_pairs = num_pairs - self.true_pairs
    self.false_match_count = (
        (np.sum(cluster_label_count**2) - hist_square_sum) / 2)
    self.false_match_frac = self.false_match_count / self.false_pairs


@dataclasses.dataclass(frozen=True)
class ClusteringResult():
  """Result of labelling the data using the centers.

  Attributes:
    data: Data that is being labelled.
    centers: Cluster centers.
    labels: Indices of the closest center for each datapoint.
    loss: The k-means objective with respect to the centers, i.e., sum of
      squared distances of the data to their closest center.
  """
  data: clustering_params.Data
  centers: clustering_params.Points
  labels: typing.Optional[np.ndarray] = None
  loss: typing.Optional[float] = None

  def __post_init__(self):

    def closest_center(datapoint: np.ndarray):
      """Returns closest center to data point and the squared distance from it.

      Args:
        datapoint: 1D np.ndarray containing a single datapoint
      """
      squared_distances = np.sum((self.centers - datapoint)**2, axis=1)
      min_index = np.argmin(squared_distances)
      return (min_index, squared_distances[min_index])

    if self.labels is None and self.loss is None:
      result = [closest_center(datapoint) for datapoint in self.data.datapoints]
      object.__setattr__(self, "labels",
                         np.array([res[0] for res in result], dtype=int))
      object.__setattr__(self, "loss", sum([res[1] for res in result]))
    if self.labels is None or self.loss is None:
      raise ValueError("Only one of labels or loss was initialized; "
                       "either both should be initialized or none.")

    if self.data.num_points != len(self.labels):
      raise ValueError(f"number of labels ({self.labels.shape[0]}) is not "
                       f"equal to number of points ({self.data.num_points})")
    num_clusters, centers_dim = self.centers.shape
    if centers_dim != self.data.dim:
      raise ValueError(f"Dimension of cluster centers ({centers_dim}) is not "
                       f"equal to dimension of data points ({self.data.dim})")
    if not all([label in list(range(num_clusters)) for label in self.labels]):
      raise ValueError("Labels in incorrect format. Each entry of label must "
                       "be an integer between 0 and number of clusters - 1")

  def cross_label_histogram(self) -> np.ndarray:
    """Returns 2D histogram of (cluster label, true label) pairs.

    Example:
    For cluster labels (self.labels) = [0, 0, 1, 1, 2, 2], and
    true labels (self.data.labels)   = [0, 0, 0, 1, 1, 1]
    the 2D histogram is given as [[2, 0],
                                  [1, 1],
                                  [0, 2]]
    This is computed using np.histogram2d with bins
    [-0.5, 0.5, 1.5, 2.5] for cluster labels and
    [-0.5, 0.5, 1.5] for true labels.

    Raises:
      ValueError: if data does not have any specified true labels.
    """
    if self.data.labels is None:
      raise ValueError("Cross label histogram is undefined since data does not "
                       "have any specified labels")
    bin_start = -0.5
    cluster_label_bins = np.arange(bin_start, np.max(self.labels) + 1, 1)
    true_label_bins = np.arange(bin_start, np.max(self.data.labels) + 1, 1)
    hist, _, _ = np.histogram2d(
        self.labels,
        self.data.labels,
        bins=(cluster_label_bins, true_label_bins))
    return hist.astype(int)

  def get_clustering_metrics(self) -> ClusteringMetrics:
    """Returns various clustering quality metrics, when data labels are given.

    Raises:
      ValueError: if data does not have any specified true labels.
    """
    return ClusteringMetrics(self.cross_label_histogram())


def private_lsh_clustering(
    k: int,
    data: clustering_params.Data,
    privacy_param: clustering_params.DifferentialPrivacyParam,
    privacy_budget_split: typing.Optional[
        clustering_params.PrivacyBudgetSplit] = None,
    tree_param: typing.Optional[clustering_params.TreeParam] = None,
    multipliers: typing.Optional[
        clustering_params.PrivacyCalculatorMultiplier] = None,
    short_description: str = "CoresetParam") -> ClusteringResult:
  """Clusters data into k clusters.

  Args:
    k: Number of clusters to divide the data into.
    data: Data to find centers for. Centering the data around the origin
      beforehand may provide performance improvements.
    privacy_param: Differential privacy parameters.
    privacy_budget_split: Deprecated.
    tree_param: Optional tree parameters for generating the LSH net tree for
      fine-tuning.
    multipliers: Optional multipliers for fine-tuning. These are used to
      determine noise parameters for the clustering algorithm.
      See the clustering_params.PrivacyCalculatorMultiplier documentation for
      details.
    short_description: Optional description to identify this parameter
      configuration.

  Returns:
    ClusteringResult with differentially private centers. The rest of
    ClusteringResult is nonprivate, and only provided for convenience.
  """
  # Warn about deprecated arguments.
  if privacy_budget_split is not None:
    logging.warn(
        "Ignoring privacy_budget_split (%s), privacy_budget_split is deprecated"
        " and has been replaced with multipliers.", privacy_budget_split
    )

  # Note that max_depth is used for the private count calculation so it cannot
  # depend on the count.
  # Chosen experimentally over multiple datasets.
  if tree_param is None:
    max_depth = 20
  else:
    max_depth = tree_param.max_depth

  # Use default multiplier if not provided.
  multipliers = (clustering_params.PrivacyCalculatorMultiplier()
                 if multipliers is None else multipliers)

  pcalc = privacy_calculator.PrivacyCalculator(
      privacy_param, data.radius, max_depth, multipliers)

  logging.debug("Privacy calculator: %s", pcalc)
  pcalc.validate_accounting(privacy_param, max_depth)

  private_count = None
  if tree_param is None:
    # Saves the private count to re-use for the root node of the tree.
    tree_param, private_count = default_clustering_params.default_tree_param(
        k, data, pcalc, max_depth)
  coreset_param = coreset_params.CoresetParam(pcalc, tree_param,
                                              short_description, data.radius)
  logging.debug("coreset_param: %s", coreset_param)

  # To guarantee privacy, enforce the radius provided.
  clipped_data = clustering_params.Data(data.clip_by_radius(), data.radius,
                                        data.labels)

  coreset: private_outputs.PrivateWeightedData = get_private_coreset(
      clipped_data, coreset_param, private_count)

  k = min(k, len(coreset.datapoints))
  logging.debug(
      "Starting k-means++ computation on private coreset with k=%d. This may "
      "be less than the original if generated coreset data ended up with "
      "less than k unique points.", k)
  kmeans = sklearn.cluster.KMeans(
      n_clusters=k, init="k-means++").fit(
          coreset.datapoints, sample_weight=coreset.weights)
  # Calculate the result relative to the original data.
  # Note: the calculations besides the centers are nonprivate.
  return ClusteringResult(data, kmeans.cluster_centers_)


def get_private_coreset(
    data: clustering_params.Data,
    coreset_param: coreset_params.CoresetParam,
    private_count: typing.Optional[int],
) -> private_outputs.PrivateWeightedData:
  """Returns private coreset, when clustered it approximates data clustering.

  Args:
    data: Data to approximate with the coreset.
    coreset_param: Parameters for generating the coreset.
    private_count: Optional private count. If None, the private count will be
      computed.
  """
  logging.debug("Starting process to get private coreset.")
  root = lsh_tree.root_node(data, coreset_param, private_count)

  # Root node must have private count >= 1.
  root.private_count = max(1, root.private_count)
  leaves = lsh_tree.LshTree(root).leaves
  coreset_points = []
  coreset_point_weights = []
  for leaf in leaves:
    coreset_points.append(leaf.get_private_average())
    coreset_point_weights.append(leaf.private_count)

  # To improve accuracy, we can clip the coreset points to the provided radius.
  coreset_points = data.clip_by_radius(np.array(coreset_points))

  logging.debug("Finished generating private coreset.")
  logging.debug("The coreset consists of %s points.", len(coreset_points))
  logging.debug("The coreset weights are: %s.", coreset_point_weights)
  return private_outputs.PrivateWeightedData(
      np.array(coreset_points), np.array(coreset_point_weights))

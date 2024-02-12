# Copyright 2024 Google LLC.
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
"""Base and child classes for different versions of sparse vector technique.

Base class and child classes implementing different versions of the `sparse
vector technique`, a randomized algorithm that receives as input a set of
queries and a dataset and returns a certain output for each query. The
nomenclature is derived from https://arxiv.org/pdf/1603.01699.pdf.
"""

from collections.abc import Callable
import dataclasses

import numpy as np
from typing_extensions import override

from dp_auditorium.configs import mechanism_config


@dataclasses.dataclass
class Query:
  """Class specifying functions to be computed in a dataset.

  Attributes:
    query_calculator: function that receives a one-dimensional array of records
      and outputs a stream of scalar queries computed on the data.
    sensitivity: l_1 sensitivity of the query.
  """

  query_calculator: Callable[[np.ndarray], np.ndarray]
  sensitivity: float


def _get_sum_query(
    min_value: float, max_value: float, num_queries: int
) -> Query:
  """Returns `Query` that computes the sum of clipped data for each query.

  Creates instance of query that computes the sum of clipped records in a `data`
  array.

  Args:
    min_value: Minimum value to which `data` records will be clipped.
    max_value: Maximum value to which `data` record will be clipped.
    num_queries: Number of queries to `data`.
  """

  def clipped_sum_queries(data: np.ndarray) -> np.ndarray:
    clipped_data = np.clip(data, min_value, max_value)
    return np.sum(clipped_data) * np.ones(num_queries)

  return Query(
      sensitivity=max_value - min_value, query_calculator=clipped_sum_queries
  )


def _get_reveal_records_query(
    min_value: float, max_value: float, num_queries: int
) -> Query:
  """Returns `Query` instance that reveals records in data.

  Creates instance of query that computes the sum of clipped records in a `data`
  returns `num_queries-1` times the last clipped record and one time the second
  to last record. When this query is computed on two different one-dimensional
  arrays that differ only in the last position (`neighboring datasets`, e.g.
  [0,1] and [0,1,1]), the output of queries will be very different and some
  child classes `AbstractSVTMechanism` will break the privacy guarantee with
  this query. For details see proof of theorem 3 in
  https://arxiv.org/pdf/1603.01699.pdf

  Args:
    min_value: Minimum value to which `data` records will be clipped.
    max_value: Maximum value to which `data` record will be clipped.
    num_queries: Number of queries to `data`.
  """

  def reveal_records_queries(data: np.ndarray) -> np.ndarray:
    if len(data) < 2:
      raise ValueError(
          "`reveal_records_queries` is only defined for datasets with at least"
          f" two points. Input was {data}"
      )
    clipped_data = np.clip(data, min_value, max_value)

    # When two `data` inputs differ only in the last record, the queries
    # computed below will differ in at least `num_queries-1` positions.
    queries_values = [clipped_data[-1]] * (num_queries - 1)
    # In certain cases this will increase the likelihood of breaking the
    # privacy, e.g. when the two last values on neighboring datasets are
    # flipped: d_0=[0, 1] and d_1=[0, 1, 0].
    queries_values.append(clipped_data[-2])
    return np.array(queries_values)

  return Query(
      sensitivity=max_value - min_value, query_calculator=reveal_records_queries
  )


class AbstractSVTMechanism:
  """Base class for sparse vector technique (SVT) mechanisms.

  Let X={x_1, ..., x_m} be a dataset with m float records. SVT is a mechanism
  for releasing information about a sequence of numerical queries
  `(q_1, q_2, ..., q_n)` computed on X. Given a `threshold`, the mechanism
  outputs randomized information `(o_1,...,o_n)` about `q_i+n_i>threshold+s_i`
  where `n_i` and `s_i` represent noise for i=1,...,n. Each configuration of
  `n_i`(specified by `get_query_noise_scale`) and `s_i`(specified by
  `get_threshold_noise_scale`), and `max_answered_queries` produces a different
  variant of SVT (see https://arxiv.org/pdf/1603.01699.pdf for more details).
  The implementation below assumes datasets are 1 dimensional arrays that will
  be passed to a `Query.query_calculator`.
  """

  def __init__(
      self,
      config: mechanism_config.SVTMechanismConfig,
      rng: np.random.BitGenerator,
  ):
    """Initializes instance of sparse vector technique mechanism.

    Args:
      config: Configuration for SVT mechanism.
      rng: Random number generator.
    """
    self._epsilon = config.epsilon
    self._max_answered_queries = config.max_answered_queries
    self._threshold = config.threshold
    if config.query_type == mechanism_config.QueryType.QUERY_TYPE_SUM:
      self._queries = _get_sum_query(
          config.min_value, config.max_value, config.num_queries
      )
    elif (
        config.query_type
        == mechanism_config.QueryType.QUERY_TYPE_REVEAL_RECORDS
    ):
      self._queries = _get_reveal_records_query(
          config.min_value, config.max_value, config.num_queries
      )
    else:
      raise ValueError(f"Invalid query type: {config.query_type.name}")
    self._num_queries = config.num_queries
    self._rng = rng

  def get_noisy_queries(
      self, queries_values: np.ndarray, num_samples: int
  ) -> np.ndarray:
    """Return `num_samples` arrays of noisy `query_values."""
    raise NotImplementedError("Must implement get_noisy_queries().")

  def get_noisy_thresholds(self, num_samples: int) -> np.ndarray:
    """Return `num_samples` arrays of noisy thresholds."""
    raise NotImplementedError("Must implement get_noisy_thresholds().")

  def get_query_output(
      self, noisy_queries: np.ndarray, noisy_threshold: np.ndarray
  ) -> tuple[np.ndarray, np.ndarray]:
    """Compute output and compare to threshold for each query.

    Each SVT mechanism determines an output for a query depending if the query
    is above the threshold (e.g. binary or the query itself). This method should
    implement a function that computes the output for each query in case the
    mechanism has not aborted yet and an array indicating if the query is above
    or not the noisy threshold.

    Args:
      noisy_queries: array with noisy queries. Can include different trials of
        the mechanism.
      noisy_threshold: array with noisy thresholds for each trial.

    Returns:
      Tuple with query outputs in case the mechanism has not aborted and boolean
      array indicating if the query is above the threshold.
    """
    raise NotImplementedError("Must implement get_query_output().")

  def __call__(self, data: np.ndarray, num_samples: int) -> np.ndarray:
    """Runs the sparse vector technique.

    Runs `num_samples` iterations of sparse vector technique on `data` with
    parameters and subroutines specified by concrete class instantiations.

    Args:
      data: np array with float values specifying a dataset to compute queries
        on. Array must be one-dimensional.
      num_samples: number of samples to draw from the mechanism.

    Returns:
      array with num_samples of the mechanism computed on `data`.
    """
    if np.array(data).ndim > 1:
      raise ValueError(
          "Sparse vector technique implemented for one-dimensional"
          "data arrays where each record is a scalar."
      )
    query_values = self._queries.query_calculator(data)
    if query_values.ndim > 1:
      raise ValueError("The output of query calculator should be a 1D array.")

    noisy_threshold = self.get_noisy_thresholds(num_samples)

    # Compute all queries and whether they exceed the threshold.
    noisy_queries = self.get_noisy_queries(query_values, num_samples)

    queries_outputs, queries_pass_threshold = self.get_query_output(
        noisy_queries, noisy_threshold
    )

    counter = np.zeros((num_samples))

    # Sequentially for each sample but vectorized over samples, check if we
    # already reached the number of answered queries by keeping a `counter` for
    # each sample.
    for i in range(self._num_queries):
      aborted = counter >= self._max_answered_queries
      # If all queries already reached the maximum we fill the rest of the array
      # with `SVT_NO_RESPONSE` and break. Otherwise only fill the ones that
      # reach the maximum, increase counter of samples that produced an output
      # and continue.
      if sum(aborted) == num_samples:
        queries_outputs[:, i:] = mechanism_config.DataValues.SVT_NO_RESPONSE
        break

      queries_outputs[aborted, i] = mechanism_config.DataValues.SVT_NO_RESPONSE
      counter += queries_pass_threshold[:, i]
    return queries_outputs


class SVT1(AbstractSVTMechanism):
  """Private sparse vector technique 1(https://arxiv.org/pdf/1603.01699.pdf).

  SVT1 returns a binary answer for the first `max_answered_queries` informing
  whether the query is above or below the threshold. Each query is independently
  noised with noise scale is `4 * max_answered_queries * sensitivity / epsilon`.
  Threshold noise is computed once for each call to the mechanism and reused
  across queries. Threshold noise scale is `2 * sensitivity / epsilon`.
  """

  @override
  def get_noisy_queries(
      self, queries_values: np.ndarray, num_samples: int
  ) -> np.ndarray:
    query_noise_scale = (
        4
        * self._max_answered_queries
        * self._queries.sensitivity
        / self._epsilon
    )
    return queries_values + self._rng.laplace(
        0, query_noise_scale, size=(num_samples, self._num_queries)
    )

  @override
  def get_noisy_thresholds(self, num_samples: int) -> np.ndarray:
    threshold_noise_scale = 2 * self._queries.sensitivity / self._epsilon
    return self._threshold + self._rng.laplace(
        0, threshold_noise_scale, size=(num_samples, 1)
    )

  @override
  def get_query_output(
      self, noisy_queries: np.ndarray, noisy_threshold: np.ndarray
  ) -> tuple[np.ndarray, np.ndarray]:
    queries_pass_threshold = noisy_queries >= noisy_threshold
    return (
        np.where(
            queries_pass_threshold,
            mechanism_config.DataValues.SVT_ABOVE_THRESHOLD,
            mechanism_config.DataValues.SVT_BELOW_THRESHOLD,
        ),
        queries_pass_threshold,
    )


class SVT2(AbstractSVTMechanism):
  """Private sparse vector technique 2(https://arxiv.org/pdf/1603.01699.pdf).

  SVT2 returns a binary answer for the first `max_answered_queries` informing
  whether the query is above or below the threshold. Each query is independently
  noised with noise scale `4 * max_answered_queries * sensitivity / epsilon`.
  Threshold noise is computed once for each call to the mechanism and reused
  across queries. Threshold noise scale is
  `2 * sensitivity * max_answered_queries / epsilon`.
  """

  @override
  def get_noisy_queries(
      self, queries_values: np.ndarray, num_samples: int
  ) -> np.ndarray:
    query_noise_scale = (
        4
        * self._max_answered_queries
        * self._queries.sensitivity
        / self._epsilon
    )
    return queries_values + self._rng.laplace(
        0, query_noise_scale, size=(num_samples, self._num_queries)
    )

  @override
  def get_noisy_thresholds(self, num_samples: int) -> np.ndarray:
    threshold_noise_scale = (
        2
        * self._queries.sensitivity
        * self._max_answered_queries
        / self._epsilon
    )
    return self._threshold + self._rng.laplace(
        0,
        threshold_noise_scale,
        size=(num_samples, 1),
    )

  @override
  def get_query_output(
      self, noisy_queries: np.ndarray, noisy_threshold: np.ndarray
  ) -> tuple[np.ndarray, np.ndarray]:
    queries_pass_threshold = noisy_queries >= noisy_threshold
    return (
        np.where(
            queries_pass_threshold,
            mechanism_config.DataValues.SVT_ABOVE_THRESHOLD,
            mechanism_config.DataValues.SVT_BELOW_THRESHOLD,
        ),
        queries_pass_threshold,
    )


class SVT3(AbstractSVTMechanism):
  """Non private sparse vector technique.

  Implements non private instance of the sparse vector technique defined as
  algorithm 3 in (https://arxiv.org/pdf/1603.01699.pdf). SVT3 returns either a
  noised query if it is above the noisy threshold or a `below_threshold`
  constant for the first `max_answered_queries` informing. Each query is
  independently noised.  Noise scale is
  `2 * max_answered_queries * sensitivity / epsilon`. Threshold noise is
  computed once for each call to the mechanism and reused across queries.
  Threshold noise scale is `2 * sensitivity / epsilon`.
  """

  @override
  def get_noisy_queries(
      self, queries_values: np.ndarray, num_samples: int
  ) -> np.ndarray:
    query_noise_scale = (
        2
        * self._max_answered_queries
        * self._queries.sensitivity
        / self._epsilon
    )
    return queries_values + self._rng.laplace(
        0, query_noise_scale, size=(num_samples, self._num_queries)
    )

  @override
  def get_noisy_thresholds(self, num_samples: int) -> np.ndarray:
    threshold_noise_scale = 2 * self._queries.sensitivity / self._epsilon
    return self._threshold + self._rng.laplace(
        0, threshold_noise_scale, size=(num_samples, 1)
    )

  @override
  def get_query_output(
      self, noisy_queries: np.ndarray, noisy_threshold: np.ndarray
  ) -> tuple[np.ndarray, np.ndarray]:
    queries_pass_threshold = noisy_queries >= noisy_threshold
    return (
        np.where(
            queries_pass_threshold,
            noisy_queries,
            mechanism_config.DataValues.SVT_BELOW_THRESHOLD,
        ),
        queries_pass_threshold,
    )


class SVT4(AbstractSVTMechanism):
  """Private sparse vector technique 4 (https://arxiv.org/pdf/1603.01699.pdf).

  SVT4 returns a binary answer for the first `max_answered_queries` informing
  whether the query is above or below the threshold. Each query is independently
  noised with noise scale `4 * sensitivity / (3 * epsilon)`. Threshold noise is
  computed once for each call to the mechanism and reused across queries.
  Threshold noise scale is `4 * sensitivity / epsilon`. SVT4 satisfies a
  stronger guarantee than the one claimed. If the intended guarantee is
  `epsilon_0`-DP, setting `epsilon=epsilon_0` at initialization will guarantee
  `(1 + 6 * max_answered_queries) * epsilon / 4`- differential privacy.
  """

  @override
  def get_noisy_queries(
      self, queries_values: np.ndarray, num_samples: int
  ) -> np.ndarray:
    query_noise_scale = 4 * self._queries.sensitivity / (3 * self._epsilon)
    return queries_values + self._rng.laplace(
        0, query_noise_scale, size=(num_samples, self._num_queries)
    )

  @override
  def get_noisy_thresholds(self, num_samples: int) -> np.ndarray:
    threshold_noise_scale = 4 * self._queries.sensitivity / self._epsilon
    return self._threshold + self._rng.laplace(
        0,
        threshold_noise_scale,
        size=(num_samples, 1),
    )

  @override
  def get_query_output(
      self, noisy_queries: np.ndarray, noisy_threshold: np.ndarray
  ) -> tuple[np.ndarray, np.ndarray]:
    queries_pass_threshold = noisy_queries >= noisy_threshold
    return (
        np.where(
            queries_pass_threshold,
            mechanism_config.DataValues.SVT_ABOVE_THRESHOLD,
            mechanism_config.DataValues.SVT_BELOW_THRESHOLD,
        ),
        queries_pass_threshold,
    )


class SVT5(AbstractSVTMechanism):
  """Non private sparse vector technique.

  Implements non private instance of the sparse vector technique defined as
  algorithm 5 in (https://arxiv.org/pdf/1603.01699.pdf). SVT5 returns a binary
  answer informing whether the query is above or below the threshold. Queries
  are not noised. Threshold noise is computed once for each call to the
  mechanism and reused across queries. Threshold noise scale is
  `2 * sensitivity / epsilon`.
  """

  @override
  def get_noisy_queries(
      self, queries_values: np.ndarray, num_samples: int
  ) -> np.ndarray:
    return np.tile(queries_values, (num_samples, 1))

  @override
  def get_noisy_thresholds(self, num_samples: int) -> np.ndarray:
    threshold_noise_scale = 2 * self._queries.sensitivity / self._epsilon
    return self._threshold + self._rng.laplace(
        0,
        threshold_noise_scale,
        size=(num_samples, 1),
    )

  @override
  def get_query_output(
      self, noisy_queries: np.ndarray, noisy_threshold: np.ndarray
  ) -> tuple[np.ndarray, np.ndarray]:
    queries_pass_threshold = noisy_queries >= noisy_threshold
    return (
        np.where(
            queries_pass_threshold,
            mechanism_config.DataValues.SVT_ABOVE_THRESHOLD,
            mechanism_config.DataValues.SVT_BELOW_THRESHOLD,
        ),
        queries_pass_threshold,
    )


class SVT6(AbstractSVTMechanism):
  """Non private sparse vector technique.

  Implements non private instance of the sparse vector technique defined as
  algorithm 6 in (https://arxiv.org/pdf/1603.01699.pdf). SVT6 returns a binary
  answer informing whether the query is above or below the threshold. Queries
  are noised with noise scale `2 * sensitivity / epsilon`. Threshold noise is
  computed once for each call to the mechanism and reused across queries.
  Threshold noise scale is `2 * sensitivity / epsilon`.
  """

  @override
  def get_noisy_queries(
      self, queries_values: np.ndarray, num_samples: int
  ) -> np.ndarray:
    query_noise_scale = 2 * self._queries.sensitivity / self._epsilon
    return queries_values + self._rng.laplace(
        0, query_noise_scale, size=(num_samples, self._num_queries)
    )

  @override
  def get_noisy_thresholds(self, num_samples: int) -> np.ndarray:
    threshold_noise_scale = 2 * self._queries.sensitivity / self._epsilon
    return self._threshold + self._rng.laplace(
        0,
        threshold_noise_scale,
        size=(num_samples, 1),
    )

  @override
  def get_query_output(
      self, noisy_queries: np.ndarray, noisy_threshold: np.ndarray
  ) -> tuple[np.ndarray, np.ndarray]:
    queries_pass_threshold = noisy_queries >= noisy_threshold
    return (
        np.where(
            queries_pass_threshold,
            mechanism_config.DataValues.SVT_ABOVE_THRESHOLD,
            mechanism_config.DataValues.SVT_BELOW_THRESHOLD,
        ),
        queries_pass_threshold,
    )

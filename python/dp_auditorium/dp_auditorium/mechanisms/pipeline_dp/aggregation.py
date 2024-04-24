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
"""Pipeline-DP library mechanisms."""


import itertools

import numpy as np
import pipeline_dp

from dp_auditorium import interfaces
from dp_auditorium.configs import privacy_property


class AggregationMechanism(interfaces.Mechanism):
  """Pipeline DP mechanism wrapper for privacy auditing."""

  def __init__(
      self,
      config: pipeline_dp.AggregateParams,
      tested_privacy_property: privacy_property.ApproximateDp,
      public_partitions: list[int] | None,
  ):
    self._epsilon = tested_privacy_property.epsilon
    self._delta = tested_privacy_property.delta
    self._config = config
    self._public_partitions = public_partitions

  def _compute_aggregations(self, data: list[float]) -> list[float]:
    """Returns one sample of a DP aggregation using the pipeline_dp library.

    Args:
      data: One dimensional array with scalar corresponding to different
        records.
    """
    budget_accountant = pipeline_dp.NaiveBudgetAccountant(
        self._epsilon, self._delta
    )
    dp_engine = pipeline_dp.DPEngine(
        budget_accountant, pipeline_dp.LocalBackend()
    )

    data_extractors = pipeline_dp.DataExtractors(
        partition_extractor=lambda x: x[0],
        privacy_id_extractor=lambda x: x[1],
        value_extractor=lambda x: x[2],
    )

    result = dp_engine.aggregate(
        data,
        self._config,
        data_extractors,
        public_partitions=self._public_partitions,
    )
    budget_accountant.compute_budgets()

    # result is an iterator where each item is a tuple
    # `(`partition_id`, MetricsTuple)`. We drop partition_id and concatenate all
    # metrics' values.
    values = [row[1] for row in result]

    # The output of this wrapper is designed for `interfaces.PropertyTester`
    # which receives arrays of samples where each sample is a one-dimensional
    # array. The specific metric defining each entry does not affect the privacy
    # test result, so for each sample we flatten all metrics across distinct
    # partitions.
    return [x for x in itertools.chain(*values)]

  def __call__(self, data: np.ndarray, num_samples: int) -> np.ndarray:
    """Returns an array of samples of a DP aggregation using pipeline_dp."""
    result = []
    data = list(data)  # PipelineDP works now for list only.
    for _ in range(num_samples):
      result.append(self._compute_aggregations(data))
    return np.array(result)

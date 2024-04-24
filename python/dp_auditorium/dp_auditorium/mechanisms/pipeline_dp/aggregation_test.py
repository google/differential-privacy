#
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the mean mechanism with Pipeline DP."""
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pipeline_dp

from dp_auditorium.configs import privacy_property
from dp_auditorium.mechanisms.pipeline_dp import aggregation


class AggregationMechanismTest(parameterized.TestCase):

  @parameterized.product(
      metrics=[
          [pipeline_dp.Metrics.MEAN],
          [pipeline_dp.Metrics.PERCENTILE(0.1)],
          [pipeline_dp.Metrics.MEAN, pipeline_dp.Metrics.SUM],
      ],
      num_samples=[1, 2, 3],
      public_partitions=[[1], [1, 2, 3]],
      delta=[0.0, 0.5],
  )
  def test_pipeline_dp_mechanism(
      self, metrics, num_samples, public_partitions, delta
  ):
    """Tests that the output of the mechanism has the expected shape.

    Correctness of the implementation and returned values will be verified using
    DP-Auditorium testers. Here we only verify the mechanism wrapper works as
    expected.

    Args:
      metrics: aggregations to be tested.
      num_samples: Number of samples to draw from the mechanism.
      public_partitions: List with ids of public partitions.
      delta: Privacy parameter.
    """
    # Stub data to test the mechanism. The first column represents a partition
    # id, the second column represents the user id, and the third the
    # corresponding value.
    data = np.array([
        [1, 1, 1.0],
        [1, 1, 1.5],
        [2, 1, 3.1],
        [1, 2, 1.0],
        [2, 2, 1.0],
        [1, 3, 1.7],
        [3, 3, 2.0],
    ])
    epsilon = 10000
    tested_privacy_property = privacy_property.ApproximateDp(
        epsilon=epsilon, delta=delta
    )
    config = pipeline_dp.AggregateParams(
        metrics=metrics,
        min_value=0.01,
        max_value=1.0,
        max_partitions_contributed=2,
        max_contributions_per_partition=1,
        contribution_bounds_already_enforced=False,
    )
    aggregation_mechanism = aggregation.AggregationMechanism(
        config=config,
        tested_privacy_property=tested_privacy_property,
        public_partitions=public_partitions,
    )

    result = aggregation_mechanism(data, num_samples=num_samples)
    self.assertEqual(
        result.shape, (num_samples, len(public_partitions)* len(metrics))
    )


if __name__ == "__main__":
  absltest.main()

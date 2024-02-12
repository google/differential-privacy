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
"""Implements a constant dataset generator.

This is a simple wrapper to integrate with the DP tester when using a constant
dataset pair.
"""

from typing import Optional

from dp_auditorium import interfaces


class ConstantDatasetGenerator(interfaces.DatasetGenerator):
  """Implements a constant dataset generator.

  Always returns the dataset pair that is passed during initialization.
  """

  def __init__(self, dataset_pair: interfaces.NeighboringDatasetsType):
    self._dataset_pair = dataset_pair

  def __call__(
      self,
      last_trial_result: Optional[float],
  ) -> interfaces.NeighboringDatasetsType:
    return self._dataset_pair

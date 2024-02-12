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
"""Test for constant dataset generator."""

from absl.testing import absltest
import numpy as np
from dp_auditorium.generators import constant_dataset_generator


class ConstantDatasetGeneratorTest(absltest.TestCase):

  def test_returns_constant_data(self):
    data1, data2 = np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])
    data_generator = constant_dataset_generator.ConstantDatasetGenerator(
        (data1, data2))
    np.testing.assert_array_equal(
        data_generator(None), (data1, data2)
    )
    np.testing.assert_array_equal(
        data_generator(1.0), (data1, data2)
    )


if __name__ == "__main__":
  absltest.main()

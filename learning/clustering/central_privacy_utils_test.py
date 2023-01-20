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
"""Tests for central_privacy_utils."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from scipy import stats

from clustering import central_privacy_utils
from clustering.central_privacy_utils import AveragePrivacyParam
from clustering.central_privacy_utils import CountPrivacyParam


class CentralPrivacyUtilsTest(parameterized.TestCase):

  def test_average_privacy_param_basic(self):
    average_privacy_param = AveragePrivacyParam(2.0, 4.3)
    self.assertEqual(average_privacy_param.gaussian_standard_deviation, 2.0)
    self.assertEqual(average_privacy_param.sensitivity, 4.3)

  def test_average_privacy_param_error(self):
    with self.assertRaises(
        ValueError,
        msg='Gaussian standard deviation was -0.1, but it must be nonnegative.'
    ):
      AveragePrivacyParam(gaussian_standard_deviation=-0.1, sensitivity=2.0)

    with self.assertRaises(
        ValueError,
        msg='Sensitivity was 0, but it must be positive.'
    ):
      AveragePrivacyParam(gaussian_standard_deviation=5.0, sensitivity=0.0)

    with self.assertRaises(
        ValueError,
        msg='Sensitivity was -0.2, but it must be positive.'
    ):
      AveragePrivacyParam(gaussian_standard_deviation=5.0, sensitivity=-0.2)

  def test_average_privacy_param_infinite(self):
    average_privacy_param = AveragePrivacyParam(0, 4.3)
    self.assertEqual(average_privacy_param.gaussian_standard_deviation, 0)
    self.assertEqual(average_privacy_param.sensitivity, 4.3)

  @parameterized.named_parameters(
      ('basic', [[1, 2, 1], [0.4, 0.2, 0.8], [3, 0, 3]], [3.1, 5.55, 0.2]),
      ('empty', [], [2, 5, -1]))
  @mock.patch.object(
      np.random, 'normal', return_value=np.array([8, 20, -4]), autospec=True)
  def test_get_private_average(self, nonprivate_points, expected_center,
                               mock_normal_fn):
    private_count = 4
    average_privacy_param = AveragePrivacyParam(
        gaussian_standard_deviation=1.9, sensitivity=4.3)

    result = central_privacy_utils.get_private_average(
        nonprivate_points, private_count, average_privacy_param, dim=3)
    self.assertSequenceAlmostEqual(result, expected_center)
    mock_normal_fn.assert_called_once()
    self.assertEqual(mock_normal_fn.call_args[1]['size'], 3)
    self.assertEqual(mock_normal_fn.call_args[1]['scale'], 1.9)

  def test_get_private_average_error(self):
    nonprivate_points = [[1, 2, 1], [0.4, 0.2, 0.8], [3, 0, 3]]
    average_privacy_param = AveragePrivacyParam(
        gaussian_standard_deviation=1.9, sensitivity=4.3)

    with self.assertRaises(ValueError):
      central_privacy_utils.get_private_average(
          nonprivate_points, 0, average_privacy_param, dim=3)
    with self.assertRaises(ValueError):
      central_privacy_utils.get_private_average(
          nonprivate_points, -2, average_privacy_param, dim=3)

  def test_get_private_average_zero_std_dev(self):
    nonprivate_points = [[1, 2, 1], [0.2, 0.1, 0.8], [3, 0, 3]]
    private_count = 3
    expected_center = [1.4, 0.7, 1.6]
    average_privacy_param = AveragePrivacyParam(
        gaussian_standard_deviation=0, sensitivity=4.3)
    self.assertSequenceAlmostEqual(
        central_privacy_utils.get_private_average(
            nonprivate_points, private_count, average_privacy_param, dim=3),
        expected_center)

  def test_private_count_param_basic(self):
    count_privacy_param = CountPrivacyParam(2.0)
    self.assertEqual(count_privacy_param.laplace_param, 2.0)

  def test_private_count_param_error(self):
    with self.assertRaises(
        ValueError, msg='Laplace param was 0, but it must be positive.'):
      CountPrivacyParam(laplace_param=0)

    with self.assertRaises(
        ValueError, msg='Laplace param was -0.2, but it must be positive.'):
      CountPrivacyParam(laplace_param=-0.2)

  def test_private_count_param_infinite(self):
    count_privacy_param = CountPrivacyParam(np.inf)
    self.assertEqual(count_privacy_param.laplace_param, np.inf)

  @parameterized.named_parameters(('basic', 10, 70), ('not_clip', -80, -20))
  @mock.patch.object(stats.dlaplace, 'rvs', autospec=True)
  def test_get_private_count(self, dlaplace_noise, expected_private_count,
                             mock_dlaplace_fn):
    mock_dlaplace_fn.return_value = dlaplace_noise

    nonprivate_count = 60
    count_privacy_param = CountPrivacyParam(laplace_param=2.0)

    result = central_privacy_utils.get_private_count(nonprivate_count,
                                                     count_privacy_param)
    self.assertEqual(result, expected_private_count)
    mock_dlaplace_fn.assert_called_once_with(2)

  def test_get_private_count_inf_laplace_param(self):
    nonprivate_count = 60
    count_privacy_param = CountPrivacyParam(laplace_param=np.inf)
    self.assertEqual(
        central_privacy_utils.get_private_count(nonprivate_count,
                                                count_privacy_param),
        nonprivate_count)


if __name__ == '__main__':
  absltest.main()

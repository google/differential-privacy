# Copyright 2022 Google LLC.
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

"""Tests for test_util."""

import unittest

from absl.testing import parameterized

from dp_accounting.pld import test_util


class TestUtilTest(parameterized.TestCase):

  @parameterized.parameters(
      # Dictionary contained
      ({
          1: 0.1,
          2: 0.3
      }, {
          1: 0.1,
          2: 0.3,
          4: 0.2
      }, True),
      ({
          1: 1e-10,
          2: 0.3
      }, {
          2: 0.3,
          4: 0.2
      }, True),
      ({
          1.0: 0.1 + 1e-10,
          2.0: 0.3
      }, {
          1.0 + 1e-10: 0.1,
          2.0: 0.3 + 1e-10,
          4.0: 0.2
      }, True),
      # Dictionary not contained
      ({
          1: 0.1,
          2: 0.3
      }, {
          2: 0.3,
          4: 0.2
      }, False))
  def test_assert_dictionary_contained(self, dict1, dict2, expected_result):
    if expected_result:
      test_util.assert_dictionary_contained(self, dict1, dict2)
    else:
      with self.assertRaises(AssertionError):
        test_util.assert_dictionary_contained(self, dict1, dict2)

  @parameterized.parameters(
      # Key missing
      ({
          1: 0.1,
          2: 0.3
      }, {
          1: 0.1,
          3: 0.3,
      }, True),
      # Value not matching
      ({
          1: 0.1,
          2: 0.3
      }, {
          1: 0.1,
          2: 0.2
      }, False))
  def test_assert_dictionary_contained_error_messages(
      self, dict1, dict2, key_missing):
    with self.assertRaises(AssertionError) as cm:
      test_util.assert_dictionary_contained(self, dict1, dict2)
    if key_missing:
      self.assertStartsWith(str(cm.exception), 'False is not true : Key')
    else:
      self.assertStartsWith(str(cm.exception), 'False is not true : Value')

  @parameterized.parameters(
      # Dictionary almost equal
      ({
          1: 0.1,
          2: 0.3,
      }, {
          1: 0.1,
          2: 0.3
      }, True),
      ({
          1: 1e-10,
          2: 0.3,
          4: 0.2,
      }, {
          2: 0.3,
          4: 0.2
      }, True),
      ({
          1.0: 0.1 + 1e-10,
          2.0: 0.3,
          4.0 + 1e-10: 0.2
      }, {
          1.0 + 1e-10: 0.1,
          2.0: 0.3 + 1e-10,
          4.0: 0.2 - 1e-10
      }, True),
      # Dictionary not almost equal
      ({
          1: 0.1,
          2: 0.3,
      }, {
          2: 0.3,
          4: 0.2
      }, False))
  def test_dictionary_almost_equal(self, dict1, dict2, expected_result):
    if expected_result:
      test_util.assert_dictionary_almost_equal(self, dict1, dict2)
    else:
      with self.assertRaises(AssertionError):
        test_util.assert_dictionary_almost_equal(self, dict1, dict2)

  @parameterized.parameters(
      (2, 1, True), (2, 2+1e-10, True), (2+1e-10, 2, True),
      (1, 2, False))
  def test_assert_almost_greater_equal(self, a, b, expected_result):
    if expected_result:
      test_util.assert_almost_greater_equal(self, a, b)
    else:
      with self.assertRaises(AssertionError):
        test_util.assert_almost_greater_equal(self, a, b)


if __name__ == '__main__':
  unittest.main()

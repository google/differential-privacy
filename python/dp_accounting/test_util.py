# Copyright 2020 Google LLC.
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

"""Helper functions for testing.
"""

import typing
import unittest


def dictionary_almost_equal(
    testcase: 'unittest.TestCase',
    dictionary1: typing.Mapping[typing.Any, float],
    dictionary2: typing.Mapping[typing.Any, float]):
  """Check two dictionaries have almost equal values."""
  for i in dictionary1.keys():
    testcase.assertAlmostEqual(dictionary1[i], dictionary2.get(i, 0))
  for i in dictionary2.keys():
    testcase.assertAlmostEqual(dictionary1.get(i, 0), dictionary2[i])

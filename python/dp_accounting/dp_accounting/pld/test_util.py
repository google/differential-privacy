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

from collections.abc import Mapping
import unittest

import numpy as np

_NumericMapping = (
    Mapping[int, float] | Mapping[float, float] | Mapping[int | float, float]
)


def assert_dictionary_contained(
    testcase: unittest.TestCase,
    dict1: _NumericMapping,
    dict2: _NumericMapping,
) -> None:
  """Check whether first dictionary is contained in the second.

  Keys of type float are checked for almost equality. Values are always checked
  for almost equality.

  Keys corresponding to values close to 0 are ignored in this test.

  Args:
    testcase: unittestTestCase object to assert containment of dictionary.
    dict1: first dictionary
    dict2: second dictionary
  """
  for i, val1 in dict1.items():
    if not np.isclose(val1, 0):
      key_found = False
      value_found = False
      for j, val2 in dict2.items():
        if np.isclose(i, j):
          key_found = True
          if np.isclose(val1, val2):
            value_found = True
            break
      testcase.assertTrue(
          key_found, msg=f'Key {i} in {dict1} not found in {dict2}'
      )
      testcase.assertTrue(
          value_found,
          msg=f'Value for key {i} in {dict1} not matching that in {dict2}',
      )


def assert_dictionary_almost_equal(
    testcase: unittest.TestCase,
    dictionary1: _NumericMapping,
    dictionary2: _NumericMapping,
) -> None:
  """Check two dictionaries have almost equal values.

  Keys of type float are checked for almost equality. Values are always checked
  for almost equality.

  Keys corresponding to values close to 0 are ignored in this test.

  Args:
    testcase: unittestTestCase object to assert containment of dictionary.
    dictionary1: first dictionary
    dictionary2: second dictionary
  """
  assert_dictionary_contained(testcase, dictionary1, dictionary2)
  assert_dictionary_contained(testcase, dictionary2, dictionary1)


def assert_almost_greater_equal(
    testcase: unittest.TestCase,
    a: float,
    b: float,
    msg: str | None = None,
) -> None:
  """Asserts that first value is greater or almost equal to second value."""
  msg = f'{a} is less than {b}' if msg is None else msg
  testcase.assertTrue(a >= b or np.isclose(a, b), msg=msg)

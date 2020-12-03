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

"""Common classes and functions for the accounting library."""

import math
import typing
import dataclasses


@dataclasses.dataclass
class DifferentialPrivacyParameters(object):
  """Representation of the differential privacy parameters of a mechanism.

  Attributes:
    epsilon: the epsilon in (epsilon, delta)-differential privacy.
    delta: the delta in (epsilon, delta)-differential privacy.
  """
  epsilon: float
  delta: float = 0


@dataclasses.dataclass
class BinarySearchParameters(object):
  """Parameters used for binary search.

  Attributes:
    upper_bound: An upper bound on the binary search range.
    lower_bound: A lower bound on the binary search range.
    initial_guess: An initial guess to start the search with. Must be positive.
      When this guess is close to the true value, it can help make the binary
      search faster.
    tolerance: An acceptable error on the returned value.
  """
  lower_bound: float
  upper_bound: float
  initial_guess: typing.Optional[float] = None
  tolerance: float = 1e-7


def inverse_monotone_function(
    func: typing.Callable[[float], float], value: float,
    search_parameters: BinarySearchParameters) -> typing.Union[float, None]:
  """Inverse a monotonically decreasing function.

  Args:
    func: The function to be inversed.
    value: The desired value of the function.
    search_parameters: Parameters used for binary search.

  Returns:
    x such that f(x) is no more than value, when such x exists; it is guaranteed
    that the returned x is within search_parameters.tolerance of the smallest
    such x. When no such x exists within the given range, returns None.
  """
  lower_x = search_parameters.lower_bound
  upper_x = search_parameters.upper_bound
  initial_guess_x = search_parameters.initial_guess

  if upper_x != math.inf and func(upper_x) > value:
    return None

  if initial_guess_x is not None:
    while (initial_guess_x < upper_x and func(initial_guess_x) > value):
      lower_x = initial_guess_x
      initial_guess_x *= 2
    upper_x = min(upper_x, initial_guess_x)

  while upper_x - lower_x > search_parameters.tolerance:
    mid_x = (upper_x + lower_x) / 2
    if func(mid_x) > value:
      lower_x = mid_x
    else:
      upper_x = mid_x

  return upper_x

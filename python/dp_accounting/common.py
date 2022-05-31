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
import abc
import dataclasses
import itertools
import math
from typing import Callable, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
from scipy import fft
from scipy import signal

ArrayLike = Union[np.ndarray, List[float]]


@dataclasses.dataclass
class DifferentialPrivacyParameters(object):
  """Representation of the differential privacy parameters of a mechanism.

  Attributes:
    epsilon: the epsilon in (epsilon, delta)-differential privacy.
    delta: the delta in (epsilon, delta)-differential privacy.
  """
  epsilon: float
  delta: float = 0

  def __post_init__(self):
    if self.epsilon < 0:
      raise ValueError(f'epsilon should be positive: {self.epsilon}')
    if self.delta < 0 or self.delta > 1:
      raise ValueError(f'delta should be between 0 and 1: {self.delta}')


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
    discrete: Whether the search is over integers.
  """
  lower_bound: float
  upper_bound: float
  initial_guess: Optional[float] = None
  tolerance: float = 1e-7
  discrete: bool = False


def inverse_monotone_function(
    func: Callable[[float], float],
    value: float,
    search_parameters: BinarySearchParameters,
    increasing: bool = False) -> Optional[float]:
  """Inverse a monotone function.

  Args:
    func: The function to be inversed.
    value: The desired value of the function.
    search_parameters: Parameters used for binary search.
    increasing: Whether the function is monotonically increasing.

  Returns:
    x such that func(x) is no more than value, when such x exists. It is
    guaranteed that the returned x is within search_parameters.tolerance of the
    smallest (for monotonically decreasing func) or the largest (for
    monotonically increasing func) such x. When no such x exists within the
    given range, returns None.
  """
  lower_x = search_parameters.lower_bound
  upper_x = search_parameters.upper_bound
  initial_guess_x = search_parameters.initial_guess

  if increasing:
    check = lambda func_value, target_value: func_value <= target_value
    if lower_x != -math.inf and func(lower_x) > value:
      return None
  else:
    check = lambda func_value, target_value: func_value > target_value
    if upper_x != math.inf and func(upper_x) > value:
      return None

  if initial_guess_x is not None:
    while initial_guess_x < upper_x and check(func(initial_guess_x), value):
      lower_x = initial_guess_x
      initial_guess_x *= 2
    upper_x = min(upper_x, initial_guess_x)

  if search_parameters.discrete:
    tolerance = 1
  else:
    tolerance = search_parameters.tolerance

  while upper_x - lower_x > tolerance:
    if search_parameters.discrete:
      mid_x = (upper_x + lower_x) // 2
    else:
      mid_x = (upper_x + lower_x) / 2

    if check(func(mid_x), value):
      lower_x = mid_x
    else:
      upper_x = mid_x

  if increasing:
    return lower_x
  else:
    return upper_x


def dictionary_to_list(
    input_dictionary: Mapping[int, float]) -> Tuple[int, List[float]]:
  """Converts an integer-keyed dictionary into an list.

  Args:
    input_dictionary: A dictionary whose keys are integers.

  Returns:
    A tuple of an integer offset and a list result_list. The offset is the
    minimum value of the input dictionary. result_list has length equal to the
    difference between the maximum and minimum values of the input dictionary.
    result_list[i] is equal to dictionary[offset + i] and is zero if offset + i
    is not a key in the input dictionary.
  """
  offset = min(input_dictionary)
  max_val = max(input_dictionary)
  result_list = [input_dictionary.get(i, 0) for i in range(offset, max_val + 1)]
  return (offset, result_list)


def list_to_dictionary(
    input_list: List[float],
    offset: int,
    tail_mass_truncation: float = 0) -> Mapping[int, float]:
  """Converts a list into an integer-keyed dictionary, with a specified offset.

  Args:
    input_list: An input list.
    offset: The offset in the key of the output dictionary
    tail_mass_truncation: an upper bound on the tails of the input list that
      might be truncated.

  Returns:
    A dictionary whose value at key is equal to input_list[key - offset]. If
    input_list[key - offset] is less than or equal to zero, it is not included
    in the dictionary.
  """
  lower_truncation_index = 0
  lower_truncation_mass = 0
  while lower_truncation_index < len(input_list):
    lower_truncation_mass += input_list[lower_truncation_index]
    if lower_truncation_mass > tail_mass_truncation / 2:
      break
    lower_truncation_index += 1

  upper_truncation_index = len(input_list) - 1
  upper_truncation_mass = 0
  while upper_truncation_index >= 0:
    upper_truncation_mass += input_list[upper_truncation_index]
    if upper_truncation_mass > tail_mass_truncation / 2:
      break
    upper_truncation_index -= 1

  result_dictionary = {}
  for i in range(lower_truncation_index, upper_truncation_index + 1):
    if input_list[i] > 0:
      result_dictionary[i + offset] = input_list[i]
  return result_dictionary


def convolve_dictionary(
    dictionary1: Mapping[int, float],
    dictionary2: Mapping[int, float],
    tail_mass_truncation: float = 0) -> Mapping[int, float]:
  """Computes a convolution of two dictionaries.

  Args:
    dictionary1: The first dictionary whose keys are integers.
    dictionary2: The second dictionary whose keys are integers.
    tail_mass_truncation: an upper bound on the tails of the output that might
      be truncated.

  Returns:
    The dictionary where for each key its corresponding value is the sum, over
    all key1, key2 such that key1 + key2 = key, of dictionary1[key1] times
    dictionary2[key2]
  """

  # Convert the dictionaries to lists.
  min1, list1 = dictionary_to_list(dictionary1)
  min2, list2 = dictionary_to_list(dictionary2)

  # Compute the convolution of the two lists.
  result_list = signal.fftconvolve(list1, list2)

  # Convert the list back to a dictionary and return
  return list_to_dictionary(
      result_list, min1 + min2, tail_mass_truncation=tail_mass_truncation)


def compute_self_convolve_bounds(
    input_list: List[float],
    num_times: int,
    tail_mass_truncation: float = 0,
    orders: Optional[List[float]] = None) -> Tuple[int, int]:
  """Computes truncation bounds for convolution using Chernoff bound.

  Args:
    input_list: The input list to be convolved.
    num_times: The number of times the list is to be convolved with itself.
    tail_mass_truncation: an upper bound on the tails of the output that might
      be truncated.
    orders: a list of orders on which the Chernoff bound is applied.

  Returns:
    A pair of upper and lower bounds for which the mass of the result of
    convolution outside of this range is at most tail_mass_truncation.
  """
  upper_bound = (len(input_list) - 1) * num_times
  lower_bound = 0

  if tail_mass_truncation == 0:
    return lower_bound, upper_bound

  if orders is None:
    # Set orders so whose absolute values are not too large; otherwise, we may
    # run into numerical issues.
    orders = (np.concatenate((np.arange(-20, 0), np.arange(1, 21)))
              / len(input_list))

  # Compute log of the moment generating function at the specified orders.
  log_mgfs = np.log([
      np.dot(np.exp(np.arange(len(input_list)) * order), input_list)
      for order in orders
  ])

  for order, log_mgf_value in zip(orders, log_mgfs):
    # Use Chernoff bound to update the upper/lower bound. See equation (5) in
    # the supplementary material.
    bound = (num_times * log_mgf_value +
             math.log(2 / tail_mass_truncation)) / order
    if order > 0:
      upper_bound = min(upper_bound, math.ceil(bound))
    if order < 0:
      lower_bound = max(lower_bound, math.floor(bound))

  return lower_bound, upper_bound


def self_convolve(input_list: ArrayLike,
                  num_times: int,
                  tail_mass_truncation: float = 0) -> Tuple[int, List[float]]:
  """Computes a convolution of the input list with itself num_times times.

  Args:
    input_list: The input list to be convolved.
    num_times: The number of times the list is to be convolved with itself.
    tail_mass_truncation: an upper bound on the tails of the output that might
      be truncated.

  Returns:
    A pair of truncation_lower_bound, output_list, where the i-th entry of
    output_list is approximately the sum, over all i_1, i_2, ..., i_num_times
    such that i_1 + i_2 + ... + i_num_times = i + truncation_lower_bound,
    of input_list[i_1] * input_list[i_2] * ... * input_list[i_num_times].
  """
  truncation_lower_bound, truncation_upper_bound = compute_self_convolve_bounds(
      input_list, num_times, tail_mass_truncation)

  # Use FFT to compute the convolution
  fast_len = fft.next_fast_len(truncation_upper_bound -
                                          truncation_lower_bound + 1)
  truncated_convolution_output = np.real(
      fft.ifft(fft.fft(input_list, fast_len)**num_times))

  # Discrete Fourier Transform wraps around modulo fast_len. Extract the output
  # values in the range of interest.
  output_list = [
      truncated_convolution_output[i % fast_len]
      for i in range(truncation_lower_bound, truncation_upper_bound + 1)
  ]

  return truncation_lower_bound, output_list


def self_convolve_dictionary(
    input_dictionary: Mapping[int, float],
    num_times: int,
    tail_mass_truncation: float = 0) -> Mapping[int, float]:
  """Computes a convolution of the input dictionary with itself num_times times.

  Args:
    input_dictionary: The input dictionary whose keys are integers.
    num_times: The number of times the dictionary is to be convolved with
      itself.
    tail_mass_truncation: an upper bound on the tails of the output that might
      be truncated.

  Returns:
    The dictionary where for each key its corresponding value is the sum, over
    all key1, key2, ..., key_num_times such that key1 + key2 + ... +
    key_num_times = key, of input_dictionary[key1] * input_dictionary[key2] *
    ... * input_dictionary[key_num_times]
  """
  min_val, input_list = dictionary_to_list(input_dictionary)
  min_val_convolution, output_list = self_convolve(
      input_list, num_times, tail_mass_truncation=tail_mass_truncation)
  return list_to_dictionary(output_list,
                            min_val * num_times + min_val_convolution)


def _get_delta_for_epsilon(infinity_mass: float,
                           reversed_losses: Iterable[float],
                           probs: Iterable[float], epsilon: float) -> float:
  """Computes the epsilon-hockey stick divergence.

  Args:
    infinity_mass: the probability of the infinite loss.
    reversed_losses: privacy losses, assumed to be sorted in descending order.
    probs: probabilities corresponding to losses.
    epsilon: the epsilon in the epsilon-hockey stick divergence.

  Returns:
    The epsilon-hockey stick divergence.
  """
  delta = 0
  for loss, prob in zip(reversed_losses, probs):
    if loss <= epsilon:
      break
    delta += (1 - np.exp(epsilon - loss)) * prob
  return delta + infinity_mass


def _get_epsilon_for_delta(infinity_mass: float,
                           reversed_losses: Iterable[float],
                           probs: Iterable[float], delta: float) -> float:
  """Computes epsilon for which hockey stick divergence is at most delta.

  Args:
    infinity_mass: the probability of the infinite loss.
    reversed_losses: privacy losses, assumed to be sorted in descending order.
    probs: probabilities corresponding to losses.
    delta: the target epsilon-hockey stick divergence..

  Returns:
     The smallest epsilon such that the epsilon-hockey stick divergence is at
     most delta. When no such finite epsilon exists, return math.inf.
  """
  if infinity_mass > delta:
    return math.inf

  mass_upper, mass_lower = infinity_mass, 0

  for loss, prob in zip(reversed_losses, probs):
    if (mass_upper > delta and mass_lower > 0 and math.log(
        (mass_upper - delta) / mass_lower) >= loss):
      # Epsilon is greater than or equal to loss.
      break

    mass_upper += prob
    mass_lower += math.exp(-loss) * prob

    if mass_upper >= delta and mass_lower == 0:
      # This only occurs when loss is very large, which results in exp(-loss)
      # being treated as zero.
      return max(0, loss)

  if mass_upper <= mass_lower + delta:
    return 0
  return math.log((mass_upper - delta) / mass_lower)


def _truncate_tails(probs: ArrayLike, tail_mass_truncation: float,
                    pessimistic_estimate: bool) -> Tuple[int, ArrayLike, float]:
  """Truncates an array from both sides by not more than tail_mass_truncation.

  It truncates the maximum prefix and suffix from probs, each of which have
  sum <= tail_mass_truncation/2.

  Args:
    probs: array to truncate.
    tail_mass_truncation: an upper bound on the tails of the probability mass of
      the PMF that might be truncated.
    pessimistic_estimate: if true then the left truncated sum is added to 0th
      element of the truncated array and the right truncated returned as it goes
      to infinity. If false then the right truncated sum is added to the last of
      the truncated array and the left truncated sum is discarded.

  Returns:
    Tuple of (size of truncated prefix, truncated array, mass that goes to
    infinity).
  """
  if tail_mass_truncation == 0:
    return 0, probs, 0

  def _find_prefix_to_truncate(arr: np.ndarray, threshold: float) -> int:
    # Find the max size of array prefix, with the sum of elements less than
    # threshold.
    s = 0
    for i, val in enumerate(arr):
      s += val
      if s > threshold:
        return i
    return len(arr)

  left_idx = _find_prefix_to_truncate(probs, tail_mass_truncation / 2)
  right_idx = len(probs) - _find_prefix_to_truncate(
      np.flip(probs), tail_mass_truncation / 2)
  # Be sure that left_idx <= right_idx. left_idx > right_idx might be when
  # tail_mass_truncation is too large or if probs has too small mass
  # (i.e. if a few truncations were operated on it already).
  right_idx = max(right_idx, left_idx)

  left_mass = np.sum(probs[:left_idx])
  right_mass = np.sum(probs[right_idx:])

  truncated_probs = probs[left_idx:right_idx]
  if pessimistic_estimate:
    # put truncated the left mass to the 0th element.
    truncated_probs[0] += left_mass
    return left_idx, truncated_probs, right_mass
  # This is rounding to left case. Put truncated the right mass to the last
  # element.
  truncated_probs[-1] += right_mass
  return left_idx, truncated_probs, 0


class PLDPmf(abc.ABC):
  """Base class for probability mass functions for privacy loss distributions."""

  def __init__(self, discretization: float, pessimistic_estimate: bool):
    self._discretization = discretization
    self._pessimistic_estimate = pessimistic_estimate

  @abc.abstractmethod
  def compose(self,
              other: 'PLDPmf',
              tail_mass_truncation: float = 0) -> 'PLDPmf':
    """Computes a PMF resulting from composing two PMFs.

    Args:
      other: the privacy loss distribution PMF to be composed. The two must have
        the same value_discretization_interval and pessimistic_estimate.
      tail_mass_truncation: an upper bound on the tails of the probability mass
        of the PMF that might be truncated.

    Returns:
      A PMF which is the result of convolving (composing) the two.
    """

  @abc.abstractmethod
  def self_compose(self,
                   num_times: int,
                   tail_mass_truncation: float = 0) -> 'PLDPmf':
    """Computes PMF resulting from repeated composing the PMF with itself.

    Args:
      num_times: the number of times to compose this PMF with itself.
      tail_mass_truncation: an upper bound on the tails of the probability mass
        of the PMF that might be truncated.

    Returns:
      A privacy loss distribution PMF which is the result of the composition.
    """

  @abc.abstractmethod
  def get_delta_for_epsilon(self, epsilon: float) -> float:
    """Computes the epsilon-hockey stick divergence."""

  @abc.abstractmethod
  def get_epsilon_for_delta(self, delta: float) -> float:
    """Computes epsilon for which hockey stick divergence is at most delta."""

  def _validate_composable(self, other: 'PLDPmf'):
    """Checks whether 'self' and 'other' can be composed."""
    if not isinstance(self, type(other)):
      raise ValueError(f'Only PMFs of the same type can be composed:'
                       f'{type(self).__name__} != {type(other).__name__}.')
    # pylint: disable=protected-access
    if self._discretization != other._discretization:
      raise ValueError(f'Discretization intervals are different: '
                       f'{self._discretization} != '
                       f'{other._discretization}.')
    if self._pessimistic_estimate != other._pessimistic_estimate:
      raise ValueError(f'Estimation types are different: '
                       f'{self._pessimistic_estimate} != '
                       f'{other._pessimistic_estimate}.')  # pylint: disable=protected-access
    # pylint: enable=protected-access


class DensePLDPmf(PLDPmf):
  """Class for dense probability mass function.

  It represents a discrete probability distribution on a grid of privacy losses.
  The grid contains numbers multiple of 'discretization', starting from
  lower_loss * discretization.
  """

  def __init__(self, discretization: float, lower_loss: int, probs: np.ndarray,
               infinity_mass: float, pessimistic_estimate: bool):
    super().__init__(discretization, pessimistic_estimate)
    self._lower_loss = lower_loss
    self._probs = probs
    self._infinity_mass = infinity_mass

  def compose(self,
              other: 'DensePLDPmf',
              tail_mass_truncation: float = 0) -> 'DensePLDPmf':
    """Computes a PMF resulting from composing two PMFs. See base class."""
    self._validate_composable(other)

    # pylint: disable=protected-access
    lower_loss = self._lower_loss + other._lower_loss
    probs = signal.fftconvolve(self._probs, other._probs)
    infinity_mass = 1 - (1 - self._infinity_mass) * (1 - other._infinity_mass)
    offset, probs, right_tail = _truncate_tails(probs, tail_mass_truncation,
                                                self._pessimistic_estimate)
    # pylint: enable=protected-access
    return DensePLDPmf(self._discretization, lower_loss + offset, probs,
                       infinity_mass + right_tail, self._pessimistic_estimate)

  def self_compose(self,
                   num_times: int,
                   tail_mass_truncation: float = 0) -> 'DensePLDPmf':
    """See base class."""
    if num_times <= 0:
      raise ValueError(f'num_times should be >= 1, num_times={num_times}')
    lower_loss = self._lower_loss * num_times
    truncation_lower_bound, probs = self_convolve(self._probs, num_times,
                                                  tail_mass_truncation)
    lower_loss += truncation_lower_bound
    probs = np.array(probs)
    inf_prob = 1 - (1 - self._infinity_mass)**num_times
    offset, probs, right_tail = _truncate_tails(probs, tail_mass_truncation,
                                                self._pessimistic_estimate)
    return DensePLDPmf(self._discretization, lower_loss + offset, probs,
                       inf_prob + right_tail, self._pessimistic_estimate)

  def get_delta_for_epsilon(self, epsilon: float) -> float:
    """Computes the epsilon-hockey stick divergence."""
    upper_loss = (self._lower_loss + len(self._probs) -
                  1) * self._discretization
    reversed_losses = itertools.count(upper_loss, -self._discretization)
    return _get_delta_for_epsilon(self._infinity_mass, reversed_losses,
                                  np.flip(self._probs), epsilon)

  def get_epsilon_for_delta(self, delta: float) -> float:
    """Computes epsilon for which hockey stick divergence is at most delta."""
    upper_loss = (self._lower_loss + len(self._probs) -
                  1) * self._discretization
    reversed_losses = itertools.count(upper_loss, -self._discretization)
    return _get_epsilon_for_delta(self._infinity_mass, reversed_losses,
                                  np.flip(self._probs), delta)


class SparsePLDPmf(PLDPmf):
  """Class for sparse probability mass function.

  It represents a discrete probability distribution on a grid of 1d losses with
  a dictionary. The grid contains numbers multiples of 'discretization'.
  """

  def __init__(self, loss_probs: Mapping[int, float], discretization: float,
               infinity_mass: float, pessimistic_estimate: bool):
    super().__init__(discretization, pessimistic_estimate)
    self._loss_probs = loss_probs
    self._infinity_mass = infinity_mass

  @property
  def size(self) -> int:
    return len(self._loss_probs)

  def compose(self,
              other: 'SparsePLDPmf',
              tail_mass_truncation: float = 0) -> 'SparsePLDPmf':
    """Computes a PMF resulting from composing two PMFs. See base class."""
    self._validate_composable(other)
    # Assumed small number of points, so simple quadratic algorithm is fine.
    convolution = {}
    # pylint: disable=protected-access
    for key1, value1 in self._loss_probs.items():
      for key2, value2 in other._loss_probs.items():
        key = key1 + key2
        convolution[key] = convolution.get(key, 0.0) + value1 * value2
    infinity_mass = 1 - (1 - self._infinity_mass) * (1 - other._infinity_mass)
    # pylint: enable=protected-access
    # Do truncation.
    sorted_losses = sorted(convolution.keys())
    probs = [convolution[loss] for loss in sorted_losses]
    offset, probs, right_mass = _truncate_tails(probs, tail_mass_truncation,
                                                self._pessimistic_estimate)
    sorted_losses = sorted_losses[offset:offset + len(probs)]
    truncated_convolution = dict(zip(sorted_losses, probs))
    return SparsePLDPmf(truncated_convolution, self._discretization,
                        infinity_mass + right_mass, self._pessimistic_estimate)

  def self_compose(self,
                   num_times: int,
                   tail_mass_truncation: float = 0) -> 'SparsePLDPmf':
    """See base class."""
    if num_times <= 0:
      raise ValueError(f'num_times should be >= 1, num_times={num_times}')
    if num_times == 1:
      return self
    result = self
    for i in range(2, num_times + 1):
      # To truncate only on the last composition.
      mass_truncation = 0 if i != num_times else tail_mass_truncation
      result = result.compose(self, mass_truncation)

    return result

  def _get_reversed_losses_probs(self) -> Tuple[List[float], List[float]]:
    """Returns losses, sorted in reverse order and respective probabilities."""
    reversed_losses = sorted(list(self._loss_probs.keys()), reverse=True)
    reversed_probs = [self._loss_probs[loss] for loss in reversed_losses]
    reversed_losses = [loss * self._discretization for loss in reversed_losses]
    return reversed_losses, reversed_probs

  def get_delta_for_epsilon(self, epsilon: float) -> float:
    """Computes the epsilon-hockey stick divergence."""
    reversed_losses, reversed_probs = self._get_reversed_losses_probs()
    return _get_delta_for_epsilon(self._infinity_mass, reversed_losses,
                                  reversed_probs, epsilon)

  def get_epsilon_for_delta(self, delta: float) -> float:
    """Computes epsilon for which hockey stick divergence is at most delta."""
    reversed_losses, reversed_probs = self._get_reversed_losses_probs()
    return _get_epsilon_for_delta(self._infinity_mass, reversed_losses,
                                  reversed_probs, delta)

  def to_dense_pmf(self) -> DensePLDPmf:
    """"Converts to dense PMF."""
    lower_loss, probs = dictionary_to_list(self._loss_probs)
    return DensePLDPmf(self._discretization, lower_loss, np.array(probs),
                       self._infinity_mass, self._pessimistic_estimate)

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
"""Helper functions for privacy accounting across queries."""

import math
import typing

import dataclasses

import privacy_loss_distribution
import privacy_loss_mechanism


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

  if func(upper_x) > value:
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


def get_smallest_parameter(
    privacy_parameters: privacy_loss_mechanism.DifferentialPrivacyParameters,
    num_queries: int, privacy_loss_distribution_constructor: typing.Callable[
        [float], privacy_loss_distribution.PrivacyLossDistribution],
    search_parameters: BinarySearchParameters) -> typing.Union[float, None]:
  """Find smallest parameter for which the mechanism satisfies desired privacy.

  This function computes the smallest "parameter" for which the corresponding
  mechanism, when run a specified number of times, satisfies a given privacy
  level. It is assumed that, when the parameter increases, the mechanism becomes
  more private.

  Args:
    privacy_parameters: The desired privacy guarantee.
    num_queries: Number of times the mechanism will be invoked.
    privacy_loss_distribution_constructor: A function that takes in a parameter
      and returns the privacy loss distribution for the corresponding mechanism
      for the given parameter.
    search_parameters: Parameters used for binary search.

  Returns:
    Smallest parameter for which the corresponding mechanism with that
    parameter, when applied the given number of times, satisfies the desired
    privacy guarantee. When no parameter in the given range satisfies this,
    return None.
  """

  def get_delta_for_parameter(parameter):
    pld_single_query = privacy_loss_distribution_constructor(parameter)
    pld_all_queries = pld_single_query.self_compose(num_queries)
    return pld_all_queries.get_delta_for_epsilon(privacy_parameters.epsilon)

  return inverse_monotone_function(get_delta_for_parameter,
                                   privacy_parameters.delta, search_parameters)


def get_smallest_laplace_noise(
    privacy_parameters: privacy_loss_mechanism.DifferentialPrivacyParameters,
    num_queries: int,
    sensitivity: float = 1) -> float:
  """Find smallest Laplace noise for which the mechanism satisfies desired privacy.

  Args:
    privacy_parameters: The desired privacy guarantee.
    num_queries: Number of times the mechanism will be invoked.
    sensitivity: The l1 sensitivity of each query.

  Returns:
    Smallest parameter for which the Laplace mechanism with this parameter, when
    applied the given number of times, satisfies the desired privacy guarantee.
  """

  def privacy_loss_distribution_constructor(parameter):
    # Setting value_discretization_interval equal to 0.001 * epsilon ensures
    # that the resulting parameter is not (epsilon', delta)-DP for epsilon' less
    # than  0.999 * epsilon. This is a heuristic for getting a reasonable
    # pessimistic estimate for the noise parameter.
    return (privacy_loss_distribution.PrivacyLossDistribution
            .from_laplace_mechanism(
                parameter,
                sensitivity=sensitivity,
                value_discretization_interval=0.001 *
                privacy_parameters.epsilon))

  # Laplace mechanism with parameter sensitivity * num_queries / epsilon is
  # epsilon-DP (for num_queries queries).
  search_parameters = BinarySearchParameters(
      0, num_queries * sensitivity / privacy_parameters.epsilon)

  parameter = get_smallest_parameter(privacy_parameters, num_queries,
                                     privacy_loss_distribution_constructor,
                                     search_parameters)
  if parameter is None:
    parameter = num_queries * sensitivity / privacy_parameters.epsilon
  return parameter


def get_smallest_discrete_laplace_noise(
    privacy_parameters: privacy_loss_mechanism.DifferentialPrivacyParameters,
    num_queries: int,
    sensitivity: int = 1) -> float:
  """Find smallest discrete Laplace noise for which the mechanism satisfies desired privacy.

  Note that from the way discrete Laplace distribution is defined, the amount of
  noise decreases as the parameter increases. (In other words, the mechanism
  becomes less private as the parameter increases.) As a result, the output will
  be the largest parameter (instead of smallest as in Laplace).

  Args:
    privacy_parameters: The desired privacy guarantee.
    num_queries: Number of times the mechanism will be invoked.
    sensitivity: The l1 sensitivity of each query.

  Returns:
    Largest parameter for which the discrete Laplace mechanism with this
    parameter, when applied the given number of times, satisfies the desired
    privacy guarantee.
  """

  # Search for inverse of the parameter instead of the parameter itself.
  def privacy_loss_distribution_constructor(inverse_parameter):
    parameter = 1 / inverse_parameter
    # Setting value_discretization_interval equal to parameter because the
    # privacy loss of discrete Laplace mechanism is always divisible by the
    # parameter.
    return (privacy_loss_distribution.PrivacyLossDistribution
            .from_discrete_laplace_mechanism(
                parameter,
                sensitivity=sensitivity,
                value_discretization_interval=parameter))

  # discrete Laplace mechanism with parameter
  # epsilon / (sensitivity * num_queries) is epsilon-DP (for num_queries
  # queries).
  search_parameters = BinarySearchParameters(
      0, num_queries * sensitivity / privacy_parameters.epsilon)

  inverse_parameter = get_smallest_parameter(
      privacy_parameters, num_queries, privacy_loss_distribution_constructor,
      search_parameters)
  if inverse_parameter is None:
    parameter = privacy_parameters.epsilon / (num_queries * sensitivity)
  else:
    parameter = 1 / inverse_parameter
  return parameter


def get_smallest_gaussian_noise(
    privacy_parameters: privacy_loss_mechanism.DifferentialPrivacyParameters,
    num_queries: int,
    sensitivity: float = 1) -> float:
  """Find smallest Gaussian noise for which the mechanism satisfies desired privacy.

  Args:
    privacy_parameters: The desired privacy guarantee.
    num_queries: Number of times the mechanism will be invoked.
    sensitivity: The l2 sensitivity of each query.

  Returns:
    Smallest standard deviation for which the Gaussian mechanism with this std,
    when applied the given number of times, satisfies the desired privacy
    guarantee.
  """
  # The l2 sensitivity grows as square root of the number of queries
  return privacy_loss_mechanism.GaussianPrivacyLoss.from_privacy_guarantee(
      privacy_parameters,
      sensitivity=sensitivity * math.sqrt(num_queries)).standard_deviation

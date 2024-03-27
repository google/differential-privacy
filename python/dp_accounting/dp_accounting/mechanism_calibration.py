# Copyright 2022, The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Library for calibration of differentially private mechanisms.

Algorithms to optimize some quantity while remaining within a specified privacy
budget.
"""

from typing import Callable, Optional, Union

import attr
from scipy import optimize

from dp_accounting import dp_event
from dp_accounting import privacy_accountant


class BracketInterval(object):
  pass


@attr.define(frozen=True)
class ExplicitBracketInterval(BracketInterval):
  endpoint_1: float
  endpoint_2: float


@attr.define(frozen=True)
class LowerEndpointAndGuess(BracketInterval):
  lower_endpoint: float
  initial_guess: float


class NoBracketIntervalFoundError(Exception):
  """Error raised when explicit bracket interval cannot be found."""


class NonEmptyAccountantError(Exception):
  """Error raised when result of make_fresh_accountant has nonempty ledger."""


def _search_for_explicit_bracket_interval(
    bracket_interval: LowerEndpointAndGuess,
    epsilon_gap: Callable[[float], float]) -> ExplicitBracketInterval:
  """Explores exponentially increasing interval to find an explicit bracket.

  Args:
    bracket_interval: A LowerEndpointAndGuess which will be expanded to find
      an explicit interval.
    epsilon_gap: Function computing the epsilon at the provided value minus
      the target epsilon. It is assumed that this function is monotonic with
      respect to its parameter, otherwise the search could fail.

  Returns:
    A valid ExplicitBracketInterval.

  Raises:
    NoBracketIntervalFoundError: if no valid bracketing interval is found
      within a factor of 2**30 of the initial guess.
  """
  lower, upper = attr.astuple(bracket_interval)
  if lower >= upper:
    raise ValueError(
        f'bracket_interval.lower_endpoint ({bracket_interval.lower_endpoint}) '
        f'must be less than bracket_interval.initial_guess '
        f'({bracket_interval.initial_guess}).')

  lower_value = epsilon_gap(lower)
  upper_value = epsilon_gap(upper)

  gap = upper - lower
  num_tries = 0

  while lower_value * upper_value > 0:
    num_tries += 1
    if num_tries > 30:
      raise NoBracketIntervalFoundError(
          'Unable to find bracketing interval within 2**30 of initial guess. '
          'Consider providing an ExplicitBracketInterval.')

    gap *= 2  # Loop invariant: gap = initial_gap * (2 ** num_tries).
    lower, upper = upper, upper + gap
    lower_value, upper_value = upper_value, epsilon_gap(upper)

  return ExplicitBracketInterval(lower, upper)


def _bisect(
    function: Callable[[float], float],
    lower: float,
    upper: float,
    tol: float,
    lower_value: Optional[float] = None,
    upper_value: Optional[float] = None,
) -> float:
  """Bisection search to find approximate root with non-positive value.

  Args:
    function: Function to find approximate root of. Conceptually, the function
      should be continuous, although really the only requirement is that it has
      opposite signs at the endpoints.
    lower: Lower endpoint.
    upper: Upper endpoint.
    tol: Terminate when endpoints are within tol of each other.
    lower_value: Value at lower endpoint.
    upper_value: Value at upper endpoint.

  Returns:
    A point with non-positive value within tol of root.

  Raises:
    ValueError: If values at lower and upper do not have opposite signs.
  """
  if lower_value is None:
    lower_value = function(lower)
  if upper_value is None:
    upper_value = function(upper)

  if lower_value == 0:
    return lower
  if upper_value == 0:
    return upper

  if lower_value * upper_value > 0:
    raise ValueError('Values must have opposite signs.')

  if upper - lower <= tol:
    return lower if lower_value < 0 else upper

  middle = (lower + upper) / 2
  middle_value = function(middle)

  if middle_value == 0:
    return middle
  elif lower_value * middle_value < 0:
    return _bisect(function, lower, middle, tol, lower_value, middle_value)
  else:
    return _bisect(function, middle, upper, tol, middle_value, upper_value)


def calibrate_dp_mechanism(
    make_fresh_accountant: Callable[[], privacy_accountant.PrivacyAccountant],
    make_event_from_param: Union[Callable[[float], dp_event.DpEvent],
                                 Callable[[int], dp_event.DpEvent]],
    target_epsilon: float,
    target_delta: float,
    bracket_interval: Optional[BracketInterval] = None,
    discrete: bool = False,
    tol: Optional[float] = None) -> Union[float, int]:
  """Searches for optimal mechanism parameter value within privacy budget.

  The procedure searches over the space of parameters by creating, for each
  sample value, a DpEvent representing the mechanism generated from that value,
  and a freshly initialized PrivacyAccountant. Then the accountant is applied to
  the event to determine its epsilon at the target delta. Brent's method is used
  to determine the value of the parameter at which the target epsilon is
  achieved.

  Args:
    make_fresh_accountant: A callable with no parameters that returns an
      initialized PrivacyAccountant. The accountants that are returned across
      multiple calls are assumed to be initialized identically. It is an error
      for the initialized accountant's `ledger` property to return anything
      besides `NoOpDpEvent`.
    make_event_from_param: A callable that takes a parameter value as an
      argument and creates a `DpEvent` representing the mechanism defined using
      that value.
    target_epsilon: The target epsilon value.
    target_delta: The target delta value.
    bracket_interval: A BracketInterval used to determine the upper and lower
      endpoints of the interval within which Brent's method will search. If
      None, searches for a non-negative bracket starting from [0, 1].
    discrete: A bool determining whether the parameter is continuous or discrete
      valued. If True, the parameter is assumed to take only integer values.
      Concretely, `discrete=True` has three effects. 1) ints, not floats are
      passed to `make_event_from_param`. 2) The minimum optimization tolerance
      is 0.5. 3) An integer is returned.
    tol: The tolerance, in parameter space. If the maximum (or minimum) value of
      the parameter that meets the privacy requirements is x*,
      calibrate_dp_mechanism is guaranteed to return a value x such that |x -
      x*| <= tol. If `None`, tol is set to 1e-6 for continuous parameters or 0.5
      for discrete parameters.

  Returns:
    A value of the parameter within tol of the optimum subject to the privacy
    constraint. If discrete=True, the returned value will be an integer.
    Otherwise it will be a float.

  Raises:
    NoBracketIntervalFoundError: if bracket_interval is LowerEndpointAndGuess
      and no upper bound can be found within a factor of 2**30 of the original
      guess.
    NonEmptyAccountantError: if make_fresh_accountant returns an accountant with
      nonempty ledger.
  """

  if not callable(make_fresh_accountant):
    raise TypeError(f'make_fresh_accountant must be callable. '
                    f'found {type(make_fresh_accountant)}.')

  if not callable(make_event_from_param):
    raise TypeError(f'make_event_from_param must be callable. '
                    f'found {type(make_event_from_param)}.')

  if target_epsilon < 0:
    raise ValueError(f'target_epsilon must be nonnegative. Found '
                     f'{target_epsilon}.')

  if not 0 <= target_delta <= 1:
    raise ValueError(f'target_delta must be in range [0, 1]. Found '
                     f'{target_delta}.')

  if bracket_interval is None:
    bracket_interval = LowerEndpointAndGuess(0, 1)

  if tol is None:
    tol = 1.0 if discrete else 1e-6
  elif discrete:
    tol = max(tol, 1.0)
  elif tol <= 0:
    raise ValueError(f'tol must be positive. Found {tol}.')

  def epsilon_gap(x: float) -> float:
    if discrete:
      x = round(x)
    event = make_event_from_param(x)
    accountant = make_fresh_accountant()
    if not isinstance(accountant.ledger, dp_event.NoOpDpEvent):
      raise NonEmptyAccountantError()
    return accountant.compose(event).get_epsilon(target_delta) - target_epsilon

  if isinstance(bracket_interval, LowerEndpointAndGuess):
    bracket_interval = _search_for_explicit_bracket_interval(
        bracket_interval, epsilon_gap)
  elif not isinstance(bracket_interval, ExplicitBracketInterval):
    raise TypeError(f'Unrecognized bracket_interval type: '
                    f'{type(bracket_interval)}')

  try:
    root, result = optimize.brentq(
        epsilon_gap,
        bracket_interval.endpoint_1,
        bracket_interval.endpoint_2,
        xtol=tol,
        full_output=True)
  except ValueError as err:
    raise ValueError(
        '`brentq` raised ValueError. This often means the supplied bracket '
        f'interval {bracket_interval} did not bracket a solution.') from err

  if not result.converged:
    root = None
  else:
    # We need to ensure that gap is not positive, guaranteeing the returned
    # parameter gives no less privacy than was requested. Since epsilon_gap can
    # be expensive to compute, we first try values near the returned root.
    # Considering brentq's contract that there exists a root within tol of the
    # returned value, in most cases adding +/- tol will suffice.
    if epsilon_gap(root) > 0:
      if epsilon_gap(root + tol) < 0:
        root += tol
      elif epsilon_gap(root - tol) < 0:
        root -= tol
      else:
        root = None

  if root is None:
    # Fallback to custom bisection that guarantees root with non-positive value.
    root = _bisect(
        epsilon_gap,
        bracket_interval.endpoint_1,
        bracket_interval.endpoint_2,
        tol,
    )

  if discrete:
    root = round(root)

  return root

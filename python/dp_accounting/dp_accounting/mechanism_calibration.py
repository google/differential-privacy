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
import numpy as np
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


def _find_auto_bracket(
    epsilon_gap: Callable[[float], float],
) -> ExplicitBracketInterval:
  """Finds a bracket interval by probing geometrically spaced points.

  Uses a two-phase approach:
  1. Coarse scan: evaluates epsilon_gap at points 10^k for k in [-12, 12]
     (stepping by 2) to find a sign change across ~13 probe points.
  2. Refinement: once a coarse bracket [10^a, 10^b] is found (b = a + 2),
     probes the midpoint 10^((a+b)/2) to narrow the bracket.

  Unlike _search_for_explicit_bracket_interval, a failure at one point does
  not disable further search — each point is evaluated independently.

  Args:
    epsilon_gap: Function computing epsilon(param) - target_epsilon.

  Returns:
    An ExplicitBracketInterval bracketing a root.

  Raises:
    NoBracketIntervalFoundError: if no sign change is found.
  """
  # Phase 1: Coarse scan with 13 points.
  exponents = np.arange(-12, 13, 2)
  probe_points = 10.0**exponents

  last_valid_point = None
  last_valid_sign = None

  for point in probe_points:
    try:
      value = epsilon_gap(point)
      if np.isnan(value) or np.isinf(value):
        continue
      sign = np.sign(value)
      if sign == 0:
        return ExplicitBracketInterval(point, point)
      if last_valid_sign is not None and sign != last_valid_sign:
        # Phase 2: Refine by probing the geometric midpoint.
        lo, hi = last_valid_point, point
        mid = np.sqrt(lo * hi)
        try:
          mid_value = epsilon_gap(mid)
          if not (np.isnan(mid_value) or np.isinf(mid_value)):
            mid_sign = np.sign(mid_value)
            if mid_sign == last_valid_sign:
              lo = mid
            elif mid_sign != 0:
              hi = mid
        except Exception:  # pylint: disable=broad-except
          pass
        return ExplicitBracketInterval(lo, hi)
      last_valid_point = point
      last_valid_sign = sign
    except Exception:  # pylint: disable=broad-except
      continue

  raise NoBracketIntervalFoundError(
      'Unable to find bracket interval. The function epsilon_gap did not '
      'change sign over probe points in [1e-12, 1e12]. Consider providing '
      'an ExplicitBracketInterval or LowerEndpointAndGuess.'
  )


def _search_for_explicit_bracket_interval(
    bracket_interval: LowerEndpointAndGuess,
    epsilon_gap: Callable[[float], float]) -> ExplicitBracketInterval:
  """Explores exponentially sized intervals to find an explicit bracket.

  Args:
    bracket_interval: A LowerEndpointAndGuess which will be expanded to find an
      explicit interval.
    epsilon_gap: Function computing the epsilon at the provided value minus the
      target epsilon. It is assumed that this function is monotonic with respect
      to its parameter, otherwise the search could fail.

  Returns:
    A valid ExplicitBracketInterval.

  Raises:
    ValueError: if `bracket_interval.lower_endpoint` is not less than
      `bracket_interval.initial_guess`.
    NoBracketIntervalFoundError: if no valid bracketing interval is found
      within a factor of 2**30 of the initial guess.
  """

  endpoint = bracket_interval.lower_endpoint
  guess = bracket_interval.initial_guess
  if endpoint >= guess:
    raise ValueError(
        f'bracket_interval.lower_endpoint ({endpoint}) must be less than '
        f'bracket_interval.initial_guess ({guess}).'
    )

  scale = guess - endpoint
  upper = lower = guess
  search_up = search_down = True
  orig_sign = np.sign(epsilon_gap(guess))

  # For i in (0, -1, 1, -2, 2, -3, 3, ...) try scaled interval (2**i, 2**(i+1)).

  for power in range(1, 31):
    if search_up:
      next_upper = endpoint + scale * (2**power)
      try:
        next_upper_value = epsilon_gap(next_upper)
        if np.isnan(next_upper_value):
          raise ValueError('Got NaN for epsilon gap.')
        elif np.sign(next_upper_value) != orig_sign:
          return ExplicitBracketInterval(upper, next_upper)
        upper = next_upper
      except Exception:  # pylint: disable=broad-except
        search_up = False

    if search_down:
      next_lower = endpoint + scale * (2**-power)
      try:
        next_lower_value = epsilon_gap(next_lower)
        if np.isnan(next_lower_value):
          raise ValueError('Got NaN for epsilon gap.')
        elif np.sign(next_lower_value) != orig_sign:
          return ExplicitBracketInterval(next_lower, lower)
        lower = next_lower
      except Exception:  # pylint: disable=broad-except
        search_down = False

    if not search_up and not search_down:
      break

  raise NoBracketIntervalFoundError(
      'Unable to find bracketing interval within 2**30 of initial guess. '
      'Consider providing an ExplicitBracketInterval.'
  )


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

  If ``make_fresh_accountant`` is a ``PrivacyAccountant`` *class* (rather than
  an already-bound factory function), the calibrator calls
  ``make_fresh_accountant.for_calibration(target_epsilon, target_delta)`` to
  create each fresh accountant instance.  This allows accountant subclasses to
  automatically tune internal parameters — e.g., PLD discretization interval or
  RDP orders — for the target privacy budget.  To opt out, pass a lambda::

      make_fresh_accountant=lambda: pld.PLDAccountant(my_custom_interval)

  Args:
    make_fresh_accountant: A callable with no parameters that returns an
      initialized PrivacyAccountant, or a PrivacyAccountant class.  If a class
      is provided, its ``for_calibration`` classmethod is used to create
      instances tuned for the target privacy parameters.  The accountants that
      are returned across multiple calls are assumed to be initialized
      identically. It is an error for the initialized accountant's ``ledger``
      property to return anything besides ``NoOpDpEvent``.
    make_event_from_param: A callable that takes a parameter value as an
      argument and creates a `DpEvent` representing the mechanism defined using
      that value.
    target_epsilon: The target epsilon value.
    target_delta: The target delta value.
    bracket_interval: A BracketInterval used to determine the upper and lower
      endpoints of the interval within which Brent's method will search. If
      None, automatically determines a bracket by probing epsilon_gap at
      geometrically spaced points in [1e-12, 1e12].
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

  if tol is None:
    tol = 1.0 if discrete else 1e-6
  elif discrete:
    tol = max(tol, 1.0)
  elif tol <= 0:
    raise ValueError(f'tol must be positive. Found {tol}.')

  # If a PrivacyAccountant class is provided (rather than a factory function),
  # use its for_calibration classmethod to create instances tuned for the
  # target privacy parameters.
  if isinstance(make_fresh_accountant, type) and issubclass(
      make_fresh_accountant, privacy_accountant.PrivacyAccountant
  ):
    accountant_class = make_fresh_accountant
    make_fresh_accountant = lambda: accountant_class.for_calibration(
        target_epsilon, target_delta
    )

  def epsilon_gap(x: float) -> float:
    if discrete:
      x = round(x)
    event = make_event_from_param(x)
    accountant = make_fresh_accountant()
    if not isinstance(accountant.ledger, dp_event.NoOpDpEvent):
      raise NonEmptyAccountantError()
    return accountant.compose(event).get_epsilon(target_delta) - target_epsilon

  if bracket_interval is None:
    bracket_interval = _find_auto_bracket(epsilon_gap)
  elif isinstance(bracket_interval, LowerEndpointAndGuess):
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

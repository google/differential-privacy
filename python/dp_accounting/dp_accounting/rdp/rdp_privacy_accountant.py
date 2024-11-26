# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Privacy accountant that uses Renyi differential privacy."""

import math
from typing import Callable, Optional, Sequence, Tuple, Union

from absl import logging
import numpy as np
from scipy import special

from dp_accounting import dp_event
from dp_accounting import privacy_accountant

NeighborRel = privacy_accountant.NeighboringRelation
CompositionErrorDetails = (
    privacy_accountant.PrivacyAccountant.CompositionErrorDetails
)


def _log_add(logx: float, logy: float) -> float:
  """Adds two numbers in the log space."""
  a, b = min(logx, logy), max(logx, logy)
  if a == -np.inf:  # adding 0
    return b
  # Use exp(a) + exp(b) = (exp(a - b) + 1) * exp(b)
  return math.log1p(math.exp(a - b)) + b  # log1p(x) = log(x + 1)


def _log_sub_sign(logx: float, logy: float) -> Tuple[bool, float]:
  """Returns log(exp(logx)-exp(logy)) and its sign."""
  if logx > logy:
    s = True
    mag = logx + math.log1p(-np.exp(logy - logx))
  elif logx < logy:
    s = False
    mag = logy + np.log1p(-np.exp(logx - logy))
  else:
    s = True
    mag = -np.inf

  return s, mag


def _log_comb(n: float, k: float) -> float:
  """Computes log of binomial coefficient."""
  return (
      special.gammaln(n + 1)
      - special.gammaln(k + 1)
      - special.gammaln(n - k + 1)
  )


def _compute_log_a_int(q: float, sigma: float, alpha: int) -> float:
  """Computes log(A_alpha) for integer alpha, 0 < q < 1."""

  # Initialize with 0 in the log space.
  log_a = -np.inf
  log1mq = math.log1p(-q)

  for i in range(alpha + 1):
    log_coef_i = _log_comb(alpha, i) + i * math.log(q) + (alpha - i) * log1mq
    s = log_coef_i + (i * i - i) / (2 * (sigma**2))
    log_a = _log_add(log_a, s)

  return log_a


_MAX_STEPS_LOG_A_FRAC = 1000


def _compute_log_a_frac(q: float, sigma: float, alpha: float) -> float:
  """Computes log(A_alpha) for fractional alpha, 0 < q < 1."""
  # Computation derived in Sec 3.3 of https://arxiv.org/pdf/1908.10530.
  # The two parts of A_alpha, integrals over (-inf,z0] and [z0, +inf), are
  # initialized to 0 in the log space:
  log_a0, log_a1 = -np.inf, -np.inf
  z0 = sigma**2 * math.log(1 / q - 1) + 0.5
  log1mq = math.log1p(-q)

  last_s0 = last_s1 = -np.inf

  for i in range(_MAX_STEPS_LOG_A_FRAC):
    log_coef = _log_comb(alpha, i)
    j = alpha - i

    log_t0 = log_coef + i * math.log(q) + j * log1mq
    log_t1 = log_coef + j * math.log(q) + i * log1mq

    log_e0 = math.log(0.5) + _log_erfc((i - z0) / (math.sqrt(2) * sigma))
    log_e1 = math.log(0.5) + _log_erfc((z0 - j) / (math.sqrt(2) * sigma))

    log_s0 = log_t0 + (i * i - i) / (2 * (sigma**2)) + log_e0
    log_s1 = log_t1 + (j * j - j) / (2 * (sigma**2)) + log_e1

    log_a0 = _log_add(log_a0, log_s0)
    log_a1 = _log_add(log_a1, log_s1)

    total = _log_add(log_a0, log_a1)

    # Terminate when both s0 and s1 are decreasing and sufficiently small
    # relative to total.
    if (
        log_s0 < last_s0
        and log_s1 < last_s1
        and max(log_s0, log_s1) < total - 30
    ):
      return total

    last_s0 = log_s0
    last_s1 = log_s1

  logging.warning(
      '_compute_log_a_frac failed to converge after %d iterations with q=%f'
      ', sigma=%f, alpha=%f. Excluding this order from the epsilon '
      'computation.',
      _MAX_STEPS_LOG_A_FRAC,
      q,
      sigma,
      alpha,
  )
  return np.inf


def _log_erfc(x: float) -> float:
  """Computes log(erfc(x)) with high accuracy for large x."""
  try:
    return math.log(2) + special.log_ndtr(-x * 2**0.5)
  except NameError:
    # If log_ndtr is not available, approximate as follows:
    r = special.erfc(x)
    if r == 0.0:
      # Using the Laurent series at infinity for the tail of the erfc function:
      #     erfc(x) ~ exp(-x^2-.5/x^2+.625/x^4)/(x*pi^.5)
      # To verify in Mathematica:
      #     Series[Log[Erfc[x]] + Log[x] + Log[Pi]/2 + x^2, {x, Infinity, 6}]
      return (
          -math.log(math.pi) / 2
          - math.log(x)
          - x**2
          - 0.5 * x**-2
          + 0.625 * x**-4
          - 37.0 / 24.0 * x**-6
          + 353.0 / 64.0 * x**-8
      )
    else:
      return math.log(r)


def compute_delta(
    orders: Sequence[float], rdp: Sequence[float], epsilon: float
) -> Tuple[float, float]:
  """Computes delta given a list of RDP values and target epsilon.

  Args:
    orders: An array of orders.
    rdp: An array of RDP guarantees.
    epsilon: The target epsilon.

  Returns:
    2-tuple containing optimal delta and the optimal order.

  Raises:
    ValueError: If input is malformed.
  """
  if epsilon < 0:
    raise ValueError(f'Epsilon cannot be negative. Found {epsilon}.')
  if len(orders) != len(rdp):
    raise ValueError('Input lists must have the same length.')

  # Basic bound (see https://arxiv.org/abs/1702.07476 Proposition 3 in v3):
  #   delta = min( np.exp((rdp - epsilon) * (orders - 1)) )

  # Improved bound from https://arxiv.org/abs/2004.00010 Proposition 12 (in v4):
  logdeltas = []  # work in log space to avoid overflows
  for a, r in zip(orders, rdp):
    if a < 1:
      raise ValueError(f'Renyi divergence order must be at least 1. Found {a}.')
    if r < 0:
      logging.warning(
          'Negative Renyi divergence of %s, probably caused by numerical '
          'instability from extreme DpEvents. Returning a delta of zero.',
          r,
      )
      logdelta = -np.inf
    # For small alpha, we are better of with bound via KL divergence:
    # delta <= sqrt(1-exp(-KL)).
    # Take a min of the two bounds.
    elif r == 0:
      logdelta = -np.inf
    else:
      logdelta = 0.5 * math.log1p(-math.exp(-r))
    if a > 1.01:
      # This bound is not numerically stable as alpha->1.
      # Thus we have a min value for alpha.
      # The bound is also not useful for small alpha, so doesn't matter.
      rdp_bound = (a - 1) * (r - epsilon + math.log1p(-1 / a)) - math.log(a)
      logdelta = min(logdelta, rdp_bound)

    logdeltas.append(logdelta)

  optimal_index = np.argmin(logdeltas)
  return min(math.exp(logdeltas[optimal_index]), 1.0), orders[optimal_index]


def compute_epsilon(
    orders: Sequence[float], rdp: Sequence[float], delta: float
) -> Tuple[float, float]:
  """Computes epsilon given a list of RDP values and target delta.

  Args:
    orders: An array of orders.
    rdp: An array of RDP guarantees.
    delta: The target delta. Must be >= 0.

  Returns:
    2-tuple containing optimal epsilon and the optimal order.

  Raises:
    ValueError: If input is malformed.
  """
  if delta < 0:
    raise ValueError(f'Delta cannot be negative. Found {delta}.')

  if delta == 0:
    if all(r == 0 for r in rdp):
      return 0, 0
    else:
      return np.inf, 0

  if len(orders) != len(rdp):
    raise ValueError('Input lists must have the same length.')

  # Basic bound (see https://arxiv.org/abs/1702.07476 Proposition 3 in v3):
  #   epsilon = min( rdp - math.log(delta) / (orders - 1) )

  # Improved bound from https://arxiv.org/abs/2004.00010 Proposition 12 (in v4).
  # Also appears in https://arxiv.org/abs/2001.05990 Equation 20 (in v1).
  eps = []
  for a, r in zip(orders, rdp):
    if a < 1:
      raise ValueError(f'Renyi divergence order must be at least 1. Found {a}.')
    if r < 0:
      logging.warning(
          'Negative Renyi divergence of %s, probably caused by numerical '
          'instability from extreme DpEvents. Returning an epsilon of zero.',
          r,
      )
      epsilon = 0.0
    elif delta**2 + math.expm1(-r) > 0:
      # In this case, we can simply bound via KL divergence:
      # delta <= sqrt(1-exp(-KL)).
      epsilon = 0  # No need to try further computation if we have epsilon = 0.
    elif a > 1.01:
      # This bound is not numerically stable as alpha->1.
      # Thus we have a min value of alpha.
      # The bound is also not useful for small alpha, so doesn't matter.
      epsilon = r + math.log1p(-1 / a) - math.log(delta * a) / (a - 1)
    else:
      # In this case we can't do anything. E.g., asking for delta = 0.
      epsilon = np.inf
    eps.append(epsilon)

  optimal_index = np.argmin(eps)
  return max(0, eps[optimal_index]), orders[optimal_index]


def _stable_inplace_diff_in_log(
    vec: np.ndarray, signs: np.ndarray, n: Optional[int] = None
):
  """Replaces the first n-1 dims of vec with the log of abs difference operator.

  Args:
    vec: numpy array of floats with size larger than 'n'
    signs: Optional numpy array of bools with the same size as vec in case one
      needs to compute partial differences vec and signs jointly describe a
      vector of real numbers' sign and abs in log scale.
    n: Optonal upper bound on number of differences to compute. If None, all
      differences are computed.

  Returns:
    The first n-1 dimension of vec and signs will store the log-abs and sign of
    the difference.

  Raises:
    ValueError: If input is malformed.
  """

  if vec.shape != signs.shape:
    raise ValueError('Shape of vec and signs do not match.')
  if signs.dtype != bool:
    raise ValueError('signs must be of type bool')
  if n is None:
    n = np.max(vec.shape) - 1
  else:
    assert np.max(vec.shape) >= n + 1
  for j in range(0, n, 1):
    if signs[j] == signs[j + 1]:  # When the signs are the same
      # if the signs are both positive, then we can just use the standard one
      signs[j], vec[j] = _log_sub_sign(vec[j + 1], vec[j])
      # otherwise, we do that but toggle the sign
      if not signs[j + 1]:
        signs[j] = ~signs[j]
    else:  # When the signs are different.
      vec[j] = _log_add(vec[j], vec[j + 1])
      signs[j] = signs[j + 1]


def _get_forward_diffs(
    fun: Callable[[float], float], n: int
) -> Tuple[np.ndarray, np.ndarray]:
  """Computes up to nth order forward difference evaluated at 0.

  See Theorem 27 of https://arxiv.org/pdf/1808.00087.pdf

  Args:
    fun: Function to compute forward differences of.
    n: Number of differences to compute.

  Returns:
    Pair (deltas, signs_deltas) of the log deltas and their signs.
  """
  func_vec = np.zeros(n + 3)
  signs_func_vec = np.ones(n + 3, dtype=bool)

  # ith coordinate of deltas stores log(abs(ith order discrete derivative))
  deltas = np.zeros(n + 2)
  signs_deltas = np.zeros(n + 2, dtype=bool)
  for i in range(1, n + 3, 1):
    func_vec[i] = fun(1.0 * (i - 1))
  for i in range(0, n + 2, 1):
    # Diff in log scale
    _stable_inplace_diff_in_log(func_vec, signs_func_vec, n=n + 2 - i)
    deltas[i] = func_vec[0]
    signs_deltas[i] = signs_func_vec[0]
  return deltas, signs_deltas


def _compute_log_a(
    q: float, noise_multiplier: float, alpha: Union[int, float]
) -> float:
  if float(alpha).is_integer():
    return _compute_log_a_int(q, noise_multiplier, int(alpha))
  else:
    return _compute_log_a_frac(q, noise_multiplier, alpha)


def _compute_rdp_poisson_subsampled_gaussian(
    q: float, noise_multiplier: float, orders: Sequence[float]
) -> Union[float, np.ndarray]:
  """Computes RDP of the Poisson sampled Gaussian mechanism.

  Args:
    q: The sampling rate.
    noise_multiplier: The ratio of the standard deviation of the Gaussian noise
      to the l2-sensitivity of the function to which it is added.
    orders: An array of RDP orders.

  Returns:
    The RDPs at all orders. Can be `np.inf`.

  Raises:
    ValueError: If q is not in [0, 1] or noise_multiplier is negative.
  """
  if not 0 <= q <= 1:
    raise ValueError(f'Sampling rate must be in [0, 1]. Found {q}.')

  if noise_multiplier < 0:
    raise ValueError(
        f'Noise multiplier must be non-negative: {noise_multiplier}'
    )

  def compute_one_order(q, alpha):
    if q == 0:
      return 0

    if np.isinf(alpha) or noise_multiplier == 0:
      return np.inf

    if q == 1.0:
      return alpha / (2 * noise_multiplier**2)

    return _compute_log_a(q, noise_multiplier, alpha) / (alpha - 1)

  return np.array([compute_one_order(q, order) for order in orders])


def _compute_rdp_sample_wor_gaussian(
    q: float, noise_multiplier: float, orders: Sequence[float]
) -> Union[float, np.ndarray]:
  """Computes RDP of Gaussian mechanism using sampling without replacement.

  This function applies to the following schemes:
  1. Sampling w/o replacement: Sample a uniformly random subset of size m = q*n.
  2. ``Replace one data point'' version of differential privacy, i.e., n is
     considered public information.

  Reference: Theorem 27 of https://arxiv.org/pdf/1808.00087.pdf (A strengthened
  version applies subsampled-Gaussian mechanism.)
  - Wang, Balle, Kasiviswanathan. "Subsampled Renyi Differential Privacy and
  Analytical Moments Accountant." AISTATS'2019.

  Args:
    q: The sampling proportion =  m / n.  Assume m is an integer <= n.
    noise_multiplier: The ratio of the standard deviation of the Gaussian noise
      to the l2-sensitivity of the function to which it is added.
    orders: An array of RDP orders.

  Returns:
    The RDPs at all orders, can be np.inf.
  """
  return np.array([
      _compute_rdp_sample_wor_gaussian_scalar(q, noise_multiplier, order)
      for order in orders
  ])


def _compute_rdp_sample_wor_gaussian_scalar(
    q: float, sigma: float, alpha: Union[float, int]
) -> float:
  """Computes RDP of the Sampled Gaussian mechanism at order alpha.

  Args:
    q: The sampling proportion =  m / n.  Assume m is an integer <= n.
    sigma: The std of the additive Gaussian noise.
    alpha: The order at which RDP is computed.

  Returns:
    RDP at alpha, can be np.inf.
  """

  assert (q <= 1) and (q >= 0) and (alpha >= 1)

  if q == 0:
    return 0

  if q == 1.0:
    return alpha / (2 * sigma**2)

  if np.isinf(alpha):
    return np.inf

  if float(alpha).is_integer():
    return _compute_rdp_sample_wor_gaussian_int(q, sigma, int(alpha)) / (
        alpha - 1
    )
  else:
    # When alpha not an integer, we apply Corollary 10 of [WBK19] to interpolate
    # the CGF and obtain an upper bound
    alpha_f = math.floor(alpha)
    alpha_c = math.ceil(alpha)

    x = _compute_rdp_sample_wor_gaussian_int(q, sigma, alpha_f)
    y = _compute_rdp_sample_wor_gaussian_int(q, sigma, alpha_c)
    t = alpha - alpha_f
    return ((1 - t) * x + t * y) / (alpha - 1)


def _compute_rdp_sample_wor_gaussian_int(
    q: float, sigma: float, alpha: int
) -> float:
  """Computes log(A_alpha) for integer alpha, subsampling without replacement.

  When alpha is smaller than max_alpha, compute the bound Theorem 27 exactly,
    otherwise compute the bound with Stirling approximation.

  Args:
    q: The sampling proportion = m / n.  Assume m is an integer <= n.
    sigma: The std of the additive Gaussian noise.
    alpha: The order at which RDP is computed.

  Returns:
    RDP at alpha, can be np.inf.
  """

  max_alpha = 256

  if np.isinf(alpha):
    return np.inf
  elif alpha == 1:
    return 0

  def cgf(x):
    # Return rdp(x+1)*x, the rdp of Gaussian mechanism is alpha/(2*sigma**2)
    return x * 1.0 * (x + 1) / (2.0 * sigma**2)

  def func(x):
    # Return the rdp of Gaussian mechanism
    return 1.0 * x / (2.0 * sigma**2)

  # Initialize with 1 in the log space.
  log_a = 0
  # Calculates the log term when alpha = 2
  log_f2m1 = func(2.0) + math.log1p(-np.exp(-func(2.0)))
  if alpha <= max_alpha:
    # We need forward differences of exp(cgf)
    # The following line is the numerically stable way of implementing it.
    # The output is in polar form with logarithmic magnitude
    deltas, _ = _get_forward_diffs(cgf, alpha)
    # Compute the bound exactly requires book keeping of O(alpha**2)
    for i in range(2, alpha + 1):
      if i == 2:
        s = (
            2 * np.log(q)
            + _log_comb(alpha, 2)
            + np.minimum(np.log(4) + log_f2m1, func(2.0) + np.log(2))
        )
      else:  # i > 2
        delta_lo = deltas[int(2 * np.floor(i / 2.0)) - 1]
        delta_hi = deltas[int(2 * np.ceil(i / 2.0)) - 1]
        s = np.log(4) + 0.5 * (delta_lo + delta_hi)
        s = np.minimum(s, np.log(2) + cgf(i - 1))
        s += i * np.log(q) + _log_comb(alpha, i)
      log_a = _log_add(log_a, s)
    return float(log_a)
  else:
    # Compute the bound with stirling approximation. Everything is O(x) now.
    for i in range(2, alpha + 1):
      if i == 2:
        s = (
            2 * np.log(q)
            + _log_comb(alpha, 2)
            + np.minimum(np.log(4) + log_f2m1, func(2.0) + np.log(2))
        )
      else:
        s = np.log(2) + cgf(i - 1) + i * np.log(q) + _log_comb(alpha, i)
      log_a = _log_add(log_a, s)

    return log_a


def _effective_gaussian_noise_multiplier(
    event: dp_event.DpEvent,
) -> Union[float, dp_event.DpEvent]:
  """Determines the effective noise multiplier of nested structure of Gaussians.

  A series of Gaussian queries on the same data can be reexpressed as a single
  query with pre- and post- processing. For details, see section 3 of
  https://arxiv.org/pdf/1812.06210.pdf.

  Args:
    event: A `dp_event.DpEvent`. In order for conversion to be successful it
      must consist of a single `dp_event.GaussianDpEvent`, or a nested structure
      of `dp_event.ComposedDpEvent` and/or `dp_event.SelfComposedDpEvent`
      bottoming out in `dp_event.GaussianDpEvent`s.

  Returns:
    The noise multiplier of the equivalent `dp_event.GaussianDpEvent`. If the
    input event was not a `dp_event.GaussianDpEvent` or a nested structure of
    `dp_event.ComposedDpEvent` and/or `dp_event.SelfComposedDpEvent` bottoming
    out in `dp_event.GaussianDpEvent`s, returns offending subevent.
  """
  if isinstance(event, dp_event.GaussianDpEvent):
    return event.noise_multiplier
  elif isinstance(event, dp_event.ComposedDpEvent):
    sum_sigma_inv_sq = 0
    for e in event.events:
      sigma = _effective_gaussian_noise_multiplier(e)
      if not isinstance(sigma, float):
        return sigma
      sum_sigma_inv_sq += sigma**-2
    return sum_sigma_inv_sq**-0.5
  elif isinstance(event, dp_event.SelfComposedDpEvent):
    sigma = _effective_gaussian_noise_multiplier(event.event)
    if not isinstance(sigma, float):
      return sigma
    return sigma * event.count**-0.5
  else:
    return event


def _compute_rdp_single_epoch_tree_aggregation(
    noise_multiplier: float,
    step_counts: Union[int, Sequence[int]],
    orders: Sequence[float],
) -> Union[float, np.ndarray]:
  """Computes RDP of the Tree Aggregation Protocol for Gaussian Mechanism.

  This function implements the accounting when the tree is periodically
  restarted and no record occurs twice across all trees. See appendix D of
  "Practical and Private (Deep) Learning without Sampling or Shuffling"
  https://arxiv.org/abs/2103.00039.

  Args:
    noise_multiplier: A non-negative float representing the ratio of the
      standard deviation of the Gaussian noise to the l2-sensitivity of the
      function to which it is added.
    step_counts: A scalar or a list of non-negative integers representing the
      number of steps per epoch (between two restarts).
    orders: An array of RDP orders.

  Returns:
    The RDPs at all orders. Can be `np.inf`.
  """
  if noise_multiplier < 0:
    raise ValueError(
        f'noise_multiplier must be non-negative. Got {noise_multiplier}.'
    )
  if noise_multiplier == 0:
    return np.inf

  if not step_counts:
    raise ValueError(
        'steps_list must be a non-empty list, or a non-zero scalar. Got '
        f'{step_counts}.'
    )

  if np.isscalar(step_counts):
    step_counts = [step_counts]

  for steps in step_counts:
    if steps < 0:
      raise ValueError(f'Steps must be non-negative. Got {step_counts}')

  max_depth = math.ceil(math.log2(max(step_counts) + 1))
  return np.array([a * max_depth / (2 * noise_multiplier**2) for a in orders])


def _expm1_over_x(x: float) -> float:
  """Computes (exp(x)-1)/x in a numerically stable manner.

  Args:
    x: float

  Returns:
    (exp(x)-1)/x
  """
  if x < -0.1 or x > 0.1:
    return math.expm1(x) / x
  # exp(x) = sum_{k>=0} x^k / k!
  # (exp(x)-1)/x = sum_{k>=1} x^{k-1} / k!
  terms = []
  y = 1  # = x^{k-1}/k!
  for k in range(1, 100):
    y = y / k
    terms.append(y)
    y = y * x
  return math.fsum(terms)
  # Dropped terms: sum_{k>=100} x^{k-1} / k!
  # Since |x|<= 0.1, this sum is < 10^-100.
  # Note that 0.9 < (exp(x)-1)/x < 1.1, so this is also a small relative error.


def _logx_over_xm1(x: float) -> float:
  """Computes log(x)/(x-1) in a numerically stable manner.

  Here log is the natural logarithm.

  Args:
    x: float

  Returns:
    log(x)/(x-1)
  """
  if x < 0.9 or x > 1.1:
    return math.log(x) / (x - 1)
  # Denote y = 1-x. Then we have a Taylor series for the natural logarithm:
  # log(x) = log(1-y) = - sum_{k>=1} y^k / k
  # Thus log(x)/(x-1) = -log(1-y)/y = sum_{k>=1} y^{k-1}/k
  return math.fsum((1 - x) ** (k - 1) / k for k in range(1, 100))
  # Dropped terms: sum_{k>=100} y^{k-1}/k.
  # Since |y|<=0.1, this sum is < 10^-100.
  # Since 0.9 <= log(x)/(x-1) <= 1.1, this is also a small relative error.


def _truncated_negative_binomial_mean(gamma: float, shape: float) -> float:
  """Computes the mean of the truncated negative binomial Distribution.

  See Definition 1 in https://arxiv.org/pdf/2110.03620v2.pdf#page=5

  Args:
    gamma: Halting probability parameter of the distribution. Must be >0 and
      <=1.
    shape: Shape parameter of the Distribution. Must be >=0.

  Returns:
    The mean of the distribution.
  """
  if shape < 0:
    raise ValueError(f'shape must be non-negative. Got {shape}')
  if gamma <= 0 or gamma > 1:
    raise ValueError(f'gamma must be in (0,1]. Got {gamma}')

  if shape == 1:  # Geometric Distribution
    return 1 / gamma
  elif shape == 0:  # Logarithmic distribution
    # answer = (1 - 1 / gamma) / log(gamma)
    #        = 1/(gamma*log(gamma)/(gamma-1))
    return 1 / (gamma * _logx_over_xm1(gamma))
  else:  # Truncated Negative Binomial
    # answer = shape * (1 / gamma - 1) / (1 - gamma**shape)
    #        = 1 / ( (exp(shape*log(gamma))-1)/(shape*log(gamma)) *
    #                                              log(gamma)/(gamma-1) * gamma)
    a = _expm1_over_x(shape * math.log(gamma))
    b = _logx_over_xm1(gamma)
    return 1 / (gamma * a * b)


def _gamma_truncated_negative_binomial(
    shape: float, mean: float, tolerance: float = 1e-9
) -> float:
  """Computes gamma parameter of truncated negative binomial from mean.

  Args:
    shape: shape parameter of truncated negative binomial distribution. Must be
      >= 0.
    mean: the expectation of the Distribution
    tolerance: relative (i.e. multiplicative) accuracy bound for gamma. Default
      tolerance is 10^-9.

  Returns:
    The gamma parameter = success probability of the distribution.
  """
  if shape < 0:
    raise ValueError(f'shape must be non-negative. Got {shape}')
  if mean < 1:
    raise ValueError(f'mean must be at least 1. Got {mean}')

  if shape == 1:
    return 1 / mean  # Geometric distribution
  # Otherwise we invert the _truncated_negative_binomial_mean function.
  gamma_min = 0  # gamma=0 corresponds to mean=infinity.
  gamma_max = min(1, 2 * (shape + 1) / mean)  # gamma=1 corresponds to mean=1.
  # Also max{shape,1/ln(1/gamma)}*(1/gamma-1) <= mean <= 2*(shape+1)/gamma,
  # which implies gamma <= 2*(shape+1)/mean
  while gamma_max > gamma_min * (1 + tolerance):
    gamma = (gamma_min + gamma_max) / 2
    gamma_mean = _truncated_negative_binomial_mean(gamma, shape)
    if gamma_mean < mean:
      gamma_max = gamma
    else:
      gamma_min = gamma
  return gamma_min  # The conservative estimate is returned.


def _compute_rdp_repeat_and_select(
    orders: Sequence[float], rdp: Sequence[float], mean: float, shape: float
) -> Sequence[float]:
  # pyformat: disable
  """Computes RDP of repeating and selecting best run.

  Inputs orders & rdp represent RDP of a single run.
  Output represents RDP of running multiple times and returning the best run;
  outputs of other runs are not returned.

  The total number of runs is randomized and drawn from a distribution
  with the given parameters. Poisson (shape=infinity), Geometric (shape=1),
  Logarithmic (shape=0), or Tuncated Negative binomial (0<shape<infinity).

  See https://arxiv.org/abs/2110.03620 for details.

  Args:
    orders: List of Renyi orders considered. Each order should be >= 1.
    rdp: List of RDPs for a single run of the mechanism.
    mean: The mean of the distribution of the random number of repetitions.
    shape: Shape/type of the distribution. Should be >= 0.
       * shape == 0 is the logarithmic distribution.
       * shape == 1 is the geometric distribution.
       * shape == infinity is the Poisson Distribution
       * shape in (0, infinity) is the truncated negative binomial.

  Returns:
    The RDPs at all orders.
  """
  # pyformat: enable
  if math.isnan(shape) or shape < 0:
    raise ValueError(f'Distribution of repetitions must be >=0. Got {shape}.')
  if math.isnan(mean) or mean < 1:
    raise ValueError(f'Mean of number of repetitions must be >=1. Got {mean}.')
  if len(orders) != len(rdp):
    raise ValueError(
        f'orders and rdp must be same length, got {len(orders)} & {len(rdp)}.'
    )

  orders = np.asarray(orders)
  rdp_out = np.zeros_like(orders, dtype=np.float64)  # This will be the output.
  rdp_out += np.inf  # Initialize to infinity.

  if shape == np.inf:  # Poisson Distribution
    for i in range(len(orders)):
      # orders[i]=lambda and rdp[i]=epsilon in the language of
      # Theorem 6 of https://arxiv.org/pdf/2110.03620v2.pdf#page=7
      if orders[i] <= 1:
        continue  # Our formula is not applicable in this case.
      epshat = math.log1p(1 / (orders[i] - 1))
      deltahat, _ = compute_delta(orders, rdp, epshat)
      rdp_out[i] = rdp[i] + mean * deltahat + math.log(mean) / (orders[i] - 1)
  else:  # Truncated Negative Binomial (includes Logarithmic & Geometric)
    # First we map mean parameter to gamma parameter of TNB.
    gamma = _gamma_truncated_negative_binomial(shape, mean)

    # Next we apply the formula.
    # Theorem 2 of https://arxiv.org/pdf/2110.03620v2.pdf#page=5
    # orders[i] = lambda, rdp[i] = epsilon,
    # orders[j] = lambdahat, rdp[j] = epsilonhat
    # First compute constant term
    c = (1 + shape) * np.min((1 - 1 / orders) * rdp - math.log(gamma) / orders)
    for i in range(len(orders)):
      if orders[i] > 1:  # Otherwise our formula is invalid.
        rdp_out[i] = rdp[i] + math.log(mean) / (orders[i] - 1) + c

    # Finally we apply monotonicity of Renyi DP
    # i.e. if orders[i] < orders[j], then rdp[i] < rdp[j].
    # We can use this to bound rdp for low orders.
    for i in range(len(orders)):
      rdp_out[i] = min(
          rdp_out[j] for j in range(len(orders)) if orders[i] <= orders[j]
      )
  return rdp_out


def _laplace_rdp(eps: float, order: float) -> float:
  """Computes RDP of Laplace noise addition.

  See Proposition 6 of Mironov (2017): https://arxiv.org/abs/1702.07476
  RDP = eps + log[ 1 + (a-1) * [exp( (1-2*a) * eps) - 1] / (2*a-1) ] / (a-1).
  In contrast, the naive bound is RDP <= eps, which is tight as order->infinity.
  For small order & eps, we can do better: In general, RDP <= order * eps^2 / 2.
  The above formula is exactly tight for the Laplace mechanism.

  Args:
    eps: The pure DP guarantee corresponding to the limit order->infinity.
    order: The Renyi divergence order (a.k.a. alpha).

  Returns:
    The RDP of Laplace noise addition
  """
  a = float(order)
  eps = float(eps)
  if not math.isfinite(eps) or eps < 0:
    raise ValueError(f'eps must be > 0. Got {eps}')
  if not math.isfinite(a) or a < 1:
    raise ValueError(f'order must be >= 1. Got {a}')

  if a == 1:  # KL divergence
    # Taking the limit as order->1 gives the following expression.
    return eps + math.expm1(-eps)
  elif a < 1.1:  # 1 < order <= 1.1
    # For numerical stability in this case, we use a Taylor series.
    # log(1+x) = sum_{k>=1} -(-x)^k/k
    # Thus log(1+c*x)/x = c * sum_{k>=1} (-c*x)^{k-1}/k
    # Since 0 <= -c*x <= 0.1, this Taylor series converges rapidly.
    c = math.expm1(eps * (1 - 2 * a)) / (2 * a - 1)
    v = -c * (a - 1)
    return eps + c * math.fsum(v ** (k - 1) / k for k in range(1, 100))
  else:  # order > 1.1
    return eps + math.log1p(
        (a - 1) * math.expm1(eps * (1 - 2 * a)) / (2 * a - 1)
    ) / (a - 1)


def _randomized_response_rdp_replace_special(
    p: float, k: int, alpha: float
) -> float:
  """Computes RDP of Randomized Response with REPLACE_SPECIAL relation.

  RDP = max{1/(a-1) * log((k * (1-p) + p)^a / k     + (k - 1) * p^a / k),
            1/(a-1) * log((k * (1-p) + p)^{1-a} / k + (k - 1) * p^{1-a} / k)}.

  Args:
    p: The noise parameter as defined in dp_event.RandomizedResponseDpEvent.
    k: The number of buckets for k-ary Randomized Response mechanism.
    alpha: The Renyi divergence order(s) (a.k.a. alpha).

  Returns:
    The RDP of Randomized Response at order alpha.
  """
  if alpha <= 1:
    raise ValueError(f'Renyi divergence order alpha must be > 1. Got {alpha}')
  if k == 1 or p == 1:
    return 0
  if p == 0:
    return np.inf
  log_1 = np.log(k * (1 - p) + p)
  log_2 = np.log(p)
  if np.isinf(alpha):
    # RDP infinite bound approaches epsilon-DP bound.
    return max(log_1, -log_2)

  bound1, bound2 = (
      special.logsumexp([x * log_1, x * log_2], b=[1 / k, 1 - 1 / k])
      for x in [alpha, 1 - alpha]
  )
  return max(bound1, bound2) / (alpha - 1)


def _randomized_response_rdp_replace_one(
    p: float, k: int, alpha: float
) -> float:
  """Computes RDP of Randomized Response with REPLACE_ONE relation.

  The RDP is the Renyi divergence of order alpha between the following
  distributions supported over {1, 2, ..., k}:
    P = [ 1 - p + p/k, p/k, p/k, ... , p/k ]
    Q = [ p/k, 1 - p + p/k, p/k, ... , p/k ]
  RDP = 1 / (a-1) * log(
      (1 - p + p / k)^a (p / k)^(1 - a) +
      (p / k)^a (1 - p + p/k)^(1 - a) +
      (k - 2) * p / k
    )

  Args:
    p: The noise parameter as defined in dp_event.RandomizedResponseDpEvent.
    k: The number of buckets for k-ary Randomized Response mechanism.
    alpha: The Renyi divergence order(s) (a.k.a. alpha).

  Returns:
    The RDP of Randomized Response at order alpha.
  """
  if alpha <= 1:
    raise ValueError(f'Renyi divergence order alpha must be > 1. Got {alpha}')
  if k == 1 or p == 1:
    return 0
  if p == 0:
    return np.inf

  log_1 = np.log(k / p - k + 1)
  if np.isinf(alpha):
    return log_1
  return special.logsumexp(
      a=[alpha * log_1, -alpha * log_1, 0],
      b=[p / k, 1 - p + p / k, (1 - 2 / k) * p],
  ) / (alpha - 1)


def _compute_randomized_response_rdp(
    noise_parameter: float,
    num_buckets: int,
    alphas: Sequence[float],
    neighboring_relation: NeighborRel,
) -> Union[float, np.ndarray]:
  """Computes RDP of Randomized Response with REPLACE_SPECIAL or REPLACE_ONE neighboring relation for an array of orders.

  Args:
    noise_parameter: The noise parameter as defined in
      dp_event.RandomizedResponseDpEvent.
    num_buckets: The number of buckets for k-ary Randomized Response mechanism.
    alphas: The array of orders.
    neighboring_relation: The neighboring relation.

  Returns:
    The RDP of Randomized Response at given orders.
  """
  if noise_parameter < 0 or noise_parameter > 1:
    raise ValueError(f'noise_parameter must be in [0,1]. Got {noise_parameter}')
  if num_buckets < 1:
    raise ValueError(f'num_buckets must be >= 1. Got {num_buckets}')
  rr_rdp_fn = (
      _randomized_response_rdp_replace_one
      if neighboring_relation == NeighborRel.REPLACE_ONE
      else _randomized_response_rdp_replace_special
  )
  return np.array([
      rr_rdp_fn(p=noise_parameter, k=num_buckets, alpha=alpha)
      for alpha in alphas
  ])


# Default orders chosen to give good coverage for Gaussian mechanism in
# the privacy regime of interest.
DEFAULT_RDP_ORDERS = (
    [1 + x / 10.0 for x in range(1, 100)]
    + list(range(11, 64))
    + [128, 256, 512, 1024]
)


class RdpAccountant(privacy_accountant.PrivacyAccountant):
  """Privacy accountant that uses Renyi differential privacy."""

  def __init__(
      self,
      orders: Optional[Sequence[float]] = None,
      neighboring_relation: NeighborRel = NeighborRel.ADD_OR_REMOVE_ONE,
  ):
    """Initializes the RDP accountant.

    Args:
      orders: The set of RDP orders.
      neighboring_relation: The neighboring relation.
    """
    super().__init__(neighboring_relation)
    if orders is None:
      orders = DEFAULT_RDP_ORDERS
    self._orders = np.array(orders)
    self._rdp = np.zeros_like(orders, dtype=np.float64)

  def _maybe_compose(
      self, event: dp_event.DpEvent, count: int, do_compose: bool
  ) -> Optional[CompositionErrorDetails]:
    """Traverses `event` and performs composition if `do_compose` is True."""

    if isinstance(event, dp_event.NoOpDpEvent):
      return None
    elif isinstance(event, dp_event.NonPrivateDpEvent):
      if do_compose:
        self._rdp += np.inf
      return None
    elif isinstance(event, dp_event.SelfComposedDpEvent):
      return self._maybe_compose(event.event, event.count * count, do_compose)
    elif isinstance(event, dp_event.ComposedDpEvent):
      for e in event.events:
        result = self._maybe_compose(e, count, do_compose)
        if result is not None:
          return result
      return None
    elif isinstance(event, dp_event.GaussianDpEvent):
      if do_compose:
        self._rdp += count * _compute_rdp_poisson_subsampled_gaussian(
            q=1.0, noise_multiplier=event.noise_multiplier, orders=self._orders
        )
      return None
    elif isinstance(event, dp_event.PoissonSampledDpEvent):
      if self._neighboring_relation not in [
          NeighborRel.ADD_OR_REMOVE_ONE,
          NeighborRel.REPLACE_SPECIAL,
      ]:
        error_msg = (
            'neighboring_relation must be `ADD_OR_REMOVE_ONE` or'
            ' `REPLACE_SPECIAL` for `PoissonSampledDpEvent`. Found'
            f' {self._neighboring_relation}.'
        )
        return CompositionErrorDetails(
            invalid_event=event, error_message=error_msg
        )
      sigma_or_bad_event = _effective_gaussian_noise_multiplier(event.event)
      if isinstance(sigma_or_bad_event, dp_event.DpEvent):
        return CompositionErrorDetails(
            invalid_event=event,
            error_message=(
                'Subevent of `PoissonSampledDpEvent` must be a'
                ' `GaussianDpEvent` or a nested structure of `ComposedDpEvent`'
                ' and/or `SelfComposedDpEvent` bottoming out in'
                f' `GaussianDpEvent`s. Found subevent {sigma_or_bad_event}.'
            ),
        )
      if do_compose:
        self._rdp += count * _compute_rdp_poisson_subsampled_gaussian(
            q=event.sampling_probability,
            noise_multiplier=sigma_or_bad_event,
            orders=self._orders,
        )
      return None
    elif isinstance(event, dp_event.SampledWithoutReplacementDpEvent):
      if self._neighboring_relation is not NeighborRel.REPLACE_ONE:
        error_msg = (
            'neighboring_relation must be `REPLACE_ONE` for '
            '`SampledWithoutReplacementDpEvent`. Found '
            f'{self._neighboring_relation}.'
        )
        return CompositionErrorDetails(
            invalid_event=event, error_message=error_msg
        )
      sigma_or_bad_event = _effective_gaussian_noise_multiplier(event.event)
      if isinstance(sigma_or_bad_event, dp_event.DpEvent):
        return CompositionErrorDetails(
            invalid_event=event,
            error_message=(
                'Subevent of `SampledWithoutReplacementDpEvent` must be a'
                ' `GaussianDpEvent` or a nested structure of `ComposedDpEvent`'
                ' and/or `SelfComposedDpEvent` bottoming out in'
                f' `GaussianDpEvent`s. Found subevent {sigma_or_bad_event}.'
            ),
        )
      if do_compose:
        self._rdp += count * _compute_rdp_sample_wor_gaussian(
            q=event.sample_size / event.source_dataset_size,
            noise_multiplier=sigma_or_bad_event,
            orders=self._orders,
        )
      return None
    elif isinstance(event, dp_event.SingleEpochTreeAggregationDpEvent):
      if self._neighboring_relation is not NeighborRel.REPLACE_SPECIAL:
        error_msg = (
            'neighboring_relation must be `REPLACE_SPECIAL` for '
            '`SingleEpochTreeAggregationDpEvent`. Found '
            f'{self._neighboring_relation}.'
        )
        return CompositionErrorDetails(
            invalid_event=event, error_message=error_msg
        )
      if do_compose:
        self._rdp += count * _compute_rdp_single_epoch_tree_aggregation(
            event.noise_multiplier, event.step_counts, self._orders
        )
      return None
    elif isinstance(event, dp_event.LaplaceDpEvent):
      if do_compose:
        eps = 1 / event.noise_multiplier
        self._rdp += count * np.array(
            [_laplace_rdp(eps, order) for order in self._orders]
        )
      return None
    elif isinstance(event, dp_event.RepeatAndSelectDpEvent):
      save_rdp = self._rdp
      if do_compose:
        # Save the RDP values from already composed DPEvents. These will
        # be added back after we process this RepeatAndSelectDpEvent.
        # Zero out self._rdp before computing the RDP of the underlying
        # DP event.
        self._rdp = np.zeros_like(self._orders, dtype=np.float64)
      composition_error = self._maybe_compose(event.event, 1, do_compose)
      if composition_error is None and do_compose:
        self._rdp = (
            count
            * _compute_rdp_repeat_and_select(
                self._orders, self._rdp, event.mean, event.shape
            )
            + save_rdp
        )
      return composition_error
    elif isinstance(event, dp_event.RandomizedResponseDpEvent):
      if self._neighboring_relation not in [
          NeighborRel.REPLACE_SPECIAL,
          NeighborRel.REPLACE_ONE,
      ]:
        error_msg = (
            'neighboring_relation must be `REPLACE_SPECIAL` or `REPLACE_ONE`'
            ' for `RandomizedResponseDpEvent`. Found'
            f' {self._neighboring_relation}.'
        )
        return CompositionErrorDetails(
            invalid_event=event, error_message=error_msg
        )
      if do_compose:
        self._rdp += count * _compute_randomized_response_rdp(
            event.noise_parameter,
            event.num_buckets,
            self._orders,
            self._neighboring_relation,
        )
      return None
    else:
      # Unsupported event (including `UnsupportedDpEvent`).
      return CompositionErrorDetails(
          invalid_event=event, error_message='Unsupported event.'
      )

  def get_epsilon_and_optimal_order(
      self, target_delta: float
  ) -> Tuple[float, float]:
    """Returns the current epsilon and the optimal RDP order.

    Args:
      target_delta: The target delta.

    Returns:
      A tuple containing the current epsilon, accounting for all composed
      `DpEvent`s, and the optimal order.
    """
    return compute_epsilon(self._orders, self._rdp, target_delta)

  def get_epsilon(self, target_delta: float) -> float:
    """Returns the current epsilon.

    Args:
      target_delta: The target delta.

    Returns:
      The current epsilon, accounting for all composed `DpEvent`s.
    """
    return compute_epsilon(self._orders, self._rdp, target_delta)[0]

  def get_delta_and_optimal_order(
      self, target_epsilon: float
  ) -> Tuple[float, float]:
    """Returns the current delta and the optimal RDP order.

    Args:
      target_epsilon: The target epsilon.

    Returns:
      A tuple containing the current delta, accounting for all composed
      `DpEvent`s, and the optimal order.
    """
    return compute_delta(self._orders, self._rdp, target_epsilon)

  def get_delta(self, target_epsilon: float) -> float:
    """Returns the current delta.

    Args:
      target_epsilon: The target epsilon.

    Returns:
      The current delta, accounting for all composed `DpEvent`s.
    """
    return compute_delta(self._orders, self._rdp, target_epsilon)[0]

  @property
  def rdp(self) -> np.ndarray:
    """Returns the current RDP values at each order."""
    return np.copy(self._rdp)

  @property
  def orders(self) -> np.ndarray:
    """Returns the set of RDP orders."""
    return np.copy(self._orders)

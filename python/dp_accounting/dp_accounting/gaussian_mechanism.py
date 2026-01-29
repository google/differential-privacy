"""Functions for exact calibration and accounting for the Gaussian mechanism.

This file contains functions for exact conversion between epsilon, delta, and
sigma for the Gaussian mechanism. Similar functionality could be achieved with
PLD accountant and mechanism_calibration, but these computations are more
accurate, faster, and have a more convenient API.
"""

import numpy as np
import scipy.optimize
import scipy.stats


_norm = scipy.stats.norm


def _get_log_delta(sigma: float, eps: float) -> float:
  # See https://arxiv.org/pdf/1805.06530, Eq. (6).
  t_star = eps * sigma + 1 / (2 * sigma)
  x = _norm.logcdf(1 / sigma - t_star)
  y = eps + _norm.logcdf(-t_star)
  return x + np.log1p(-np.exp(y - x)) if y <= x else -np.inf


def get_epsilon_gaussian(
    sigma: float, delta: float, tol: float = 1.0e-12
) -> float:
  """Compute the epsilon for the Gaussian mechanism with the given DP params.

  The analytical epsilon for the Gaussian mechanism can be computed using the
  method in https://arxiv.org/pdf/1805.06530.

  Args:
    sigma: The standard deviation of the Gaussian noise.
    delta: The target delta.
    tol: Error tolerance for search.

  Returns:
    The epsilon for the Gaussian mechanism with the given DP params.
  """
  if sigma < 0:
    raise ValueError(f'sigma must be non-negative, got sigma={sigma}.')
  if not 0 <= delta <= 1:
    raise ValueError(f'delta must be in [0, 1], got delta={delta}.')

  if delta == 1:
    return 0
  elif sigma == 0:
    return np.inf
  elif sigma == np.inf:
    return 0
  elif delta == 0:
    return np.inf

  log_delta = np.log(delta)
  if _get_log_delta(sigma, 0) < log_delta:
    # We have (0, delta)-DP.
    return 0

  eps_lo, eps_hi = 0.0, 1.0
  while _get_log_delta(sigma, eps_hi) > log_delta:
    eps_lo, eps_hi = eps_hi, eps_hi * 10

  return scipy.optimize.brentq(
      lambda eps: _get_log_delta(sigma, eps) - log_delta,
      eps_lo,
      eps_hi,
      xtol=tol,
  )


def get_sigma_gaussian(
    epsilon: float, delta: float, tol: float = 1.0e-12
) -> float:
  """Compute the noise std for the Gaussian mechanism with the given DP params.

  The optimal noise for the Gaussian mechanism can be computed using the method
  in https://arxiv.org/pdf/1805.06530.

  Args:
    epsilon: The target epsilon.
    delta: The target delta.
    tol: Error tolerance for search.

  Returns:
    The noise std for the Gaussian mechanism with the given DP params.
  """
  if epsilon < 0:
    raise ValueError(f'epsilon must be non-negative, got epsilon={epsilon}.')
  if not 0 <= delta <= 1:
    raise ValueError(f'delta must be in [0, 1], got delta={delta}.')

  if delta == 1:
    return 0
  elif epsilon == np.inf:
    return 0
  elif delta == 0:
    return np.inf

  log_delta = np.log(delta)
  sigma_lo, sigma_hi = 1e-1, 1e0
  while _get_log_delta(sigma_lo, epsilon) < log_delta:
    sigma_hi, sigma_lo = sigma_lo, sigma_lo / 10
  while _get_log_delta(sigma_hi, epsilon) > log_delta:
    sigma_lo, sigma_hi = sigma_hi, sigma_hi * 10

  return scipy.optimize.brentq(
      lambda sigma: _get_log_delta(sigma, epsilon) - log_delta,
      sigma_lo,
      sigma_hi,
      xtol=tol,
  )

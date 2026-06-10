"""Utility functions for distribution operations and numerical stability."""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from dp_accounting.pld.random_allocation import ra_distributions
from dp_accounting.pld.random_allocation import ra_types


# =============================================================================
# Boundary-Mass Convolution Utilities
# =============================================================================


def _convolve_boundary_masses(
    p_min_1: float,
    p_max_1: float,
    p_min_2: float,
    p_max_2: float,
    domain: ra_distributions.Domain,
) -> tuple[float, float]:
  """Compute boundary masses (p_min, p_max) for the convolution Z = X + Y.

  Domain semantics differ for the lower boundary:
  - REALS    (−∞ absorbing): P(Z=−∞) = 1 − (1−p_min_1)(1−p_min_2)
  - POSITIVES (0 neutral):   P(Z=0)  = p_min_1 · p_min_2

  The upper boundary (+∞ absorbing) uses the same formula for both domains:
    P(Z=+∞) = 1 − (1−p_max_1)(1−p_max_2)

  Both inputs must share the same domain.
  """

  # p_max: +∞ is always absorbing
  p_max = float(
      np.clip(-np.expm1(np.log1p(-p_max_1) + np.log1p(-p_max_2)), 0.0, 1.0)
  )

  # p_min: depends on domain
  if domain == ra_distributions.Domain.POSITIVES:
    # 0 is neutral: Z=0 only when both X=0 and Y=0
    p_min = p_min_1 * p_min_2
  else:
    # −∞ is absorbing: Z=−∞ when either X=−∞ or Y=−∞
    p_min = float(
        np.clip(-np.expm1(np.log1p(-p_min_1) + np.log1p(-p_min_2)), 0.0, 1.0)
    )

  return p_min, p_max


def _self_convolve_boundary_masses(
    dist: ra_distributions.DiscreteDistBase,
    num_convolutions: int,
) -> tuple[float, float]:
  """Compute boundary masses after ``num_convolutions`` self-convolutions."""
  # p_max: absorbing in both domains
  p_max = float(
      np.clip(-np.expm1(num_convolutions * np.log1p(-dist.p_max)), 0.0, 1.0)
  )

  if dist.domain == ra_distributions.Domain.POSITIVES:
    # 0 is neutral: Z=0 only when all num_convolutions factors are 0
    p_min = dist.p_min**num_convolutions
  else:
    # −∞ is absorbing: Z=−∞ when any copy is −∞
    p_min = float(
        np.clip(-np.expm1(num_convolutions * np.log1p(-dist.p_min)), 0.0, 1.0)
    )

  return p_min, p_max


# =============================================================================
# Helpers used by _combine_distributions
# =============================================================================


@ra_types._optional_njit()
def _kahan_reverse_exclusive_cumsum(
    padded_probs: NDArray[np.float64],
) -> NDArray[np.float64]:
  """Compute exclusive CCDF using Kahan summation for numerical stability.

  Computes exclusive reverse cumulative sum: CCDF[i] = sum(padded_probs[i+1:]).
  Uses Kahan compensated summation to minimize floating-point rounding errors.
  """
  n = len(padded_probs)
  ccdf = np.zeros(n, dtype=np.float64)

  # Start from the right (highest index) and accumulate backwards
  running_sum = 0.0
  compensation = 0.0

  for i in range(n - 1, -1, -1):
    # Store the running sum BEFORE adding current element (exclusive)
    ccdf[i] = running_sum

    # Kahan summation: compensated addition of current element
    y = padded_probs[i] - compensation
    t = running_sum + y
    compensation = (t - running_sum) - y
    running_sum = t

  return ccdf


def _ccdf_from_pmf(
    dist: ra_distributions.DiscreteDistBase,
) -> NDArray[np.float64]:
  """Convert distribution PMF to padded complementary CDF.

  Returns CCDF over [−∞/0, l_0, l_1, ..., +∞]:
    CCDF[i] = P(X > position[i])

  Includes both boundary atoms (p_min at the left, p_max at the right).
  """
  padded_probs = np.concatenate(([dist.p_min], dist.prob_arr, [dist.p_max]))
  return _kahan_reverse_exclusive_cumsum(
      padded_probs=padded_probs,
  )


def _expand_to_grid(
    *,
    dist: ra_distributions.DiscreteDistBase,
    grid: NDArray[np.float64],
) -> ra_distributions.SparseDiscreteDist:
  """Insert zero-mass points for missing support values."""
  x = dist._x_array
  pmf = dist.prob_arr
  expanded_pmf = np.zeros_like(grid, dtype=np.float64)
  indices = np.searchsorted(grid, x)
  if not np.all(grid[indices] == x):
    raise ValueError("Target grid must contain all original support points")
  expanded_pmf[indices] = pmf
  return ra_distributions.SparseDiscreteDist(
      x_array=grid,
      prob_arr=expanded_pmf,
      p_min=dist.p_min,
      p_max=dist.p_max,
  )


def _align_distributions_to_union_grid(
    *,
    dist_1: ra_distributions.DiscreteDistBase,
    dist_2: ra_distributions.DiscreteDistBase,
) -> tuple[
    ra_distributions.SparseDiscreteDist, ra_distributions.SparseDiscreteDist
]:
  """Return distributions on a shared grid by inserting zero-mass points."""
  x_union = np.unique(np.concatenate((dist_1._x_array, dist_2._x_array)))
  return (
      _expand_to_grid(
          dist=dist_1,
          grid=x_union,
      ),
      _expand_to_grid(
          dist=dist_2,
          grid=x_union,
      ),
  )


# =============================================================================
# Public Convolution Utilities
# =============================================================================


def _binary_self_convolve(
    *,
    dist: ra_distributions.DenseDiscreteDist,
    T: int,  # pylint: disable=invalid-name  # T matches paper notation
    tail_truncation: float,
    bound_type: ra_types.BoundType,
    convolve: Callable[..., ra_distributions.DenseDiscreteDist],
) -> ra_distributions.DenseDiscreteDist:
  """Self-convolve via exponentiation by squaring, using a convolve callback.

  Algorithm 3 (`self-conv`) in Appendix C of https://arxiv.org/abs/2602.17284.
  """
  if T < 1:
    raise ValueError(f"T must be >= 1, got {T}")
  if T == 1:
    return dist

  base_dist = dist
  acc_dist = None
  tail_truncation /= 4
  while T > 0:
    if T & 1:
      if acc_dist is None:
        acc_dist = base_dist
      else:
        acc_dist = convolve(
            dist_1=acc_dist,
            dist_2=base_dist,
            tail_truncation=tail_truncation / T,
            bound_type=bound_type,
        )
    T >>= 1
    if T > 0:
      base_dist = convolve(
          dist_1=base_dist,
          dist_2=base_dist,
          tail_truncation=tail_truncation / T,
          bound_type=bound_type,
      )
  # If T is a power of two, acc_dist is never set; return the final squared
  # base_dist.
  return acc_dist if acc_dist is not None else base_dist


def _combine_distributions(
    *,
    dist_1: ra_distributions.DiscreteDistBase,
    dist_2: ra_distributions.DiscreteDistBase,
    bound_type: ra_types.BoundType,
) -> ra_distributions.SparseDiscreteDist:
  """Combine two distributions by tightening bounds via CCDF min/max.

  For DOMINATES: tighter dominating distribution via pointwise min CCDF.
  For IS_DOMINATED: tighter dominated distribution via pointwise max CCDF.
  """
  ccdf_op: Any
  if bound_type == ra_types.BoundType.DOMINATES:
    ccdf_op = np.minimum
  elif bound_type == ra_types.BoundType.IS_DOMINATED:
    ccdf_op = np.maximum
  else:
    raise ValueError(f"Unknown BoundType: {bound_type}")

  if ra_distributions._stable_array_equal(a=dist_1._x_array, b=dist_2._x_array):
    dist_1_aligned, dist_2_aligned = dist_1, dist_2
  else:
    dist_1_aligned, dist_2_aligned = _align_distributions_to_union_grid(
        dist_1=dist_1,
        dist_2=dist_2,
    )

  x_array = dist_1_aligned._x_array
  ccdf_1 = _ccdf_from_pmf(dist_1_aligned)
  ccdf_2 = _ccdf_from_pmf(dist_2_aligned)
  combined_ccdf = ccdf_op(ccdf_1, ccdf_2)
  prob_arr = combined_ccdf[:-2] - combined_ccdf[1:-1]

  prob_arr, p_min, p_max = ra_distributions._enforce_mass_conservation(
      prob_arr=prob_arr,
      expected_p_min=max(dist_1_aligned.p_min, dist_2_aligned.p_min),
      expected_p_max=max(dist_1_aligned.p_max, dist_2_aligned.p_max),
      bound_type=bound_type,
  )

  return ra_distributions.SparseDiscreteDist(
      x_array=x_array,
      prob_arr=prob_arr,
      p_min=p_min,
      p_max=p_max,
  )


# =============================================================================
# Distribution Transform Utilities
# =============================================================================


def _exp_linear_to_geometric(
    dist: ra_distributions.DenseDiscreteDist,
) -> ra_distributions.DenseDiscreteDist:
  """Map a linear-grid distribution to a geometric-grid one via exp(.).

  Maps REALS domain → POSITIVES domain.
  The −∞ atom (p_min in REALS) maps to the 0 atom (p_min in POSITIVES).
  """
  if dist.spacing_type != ra_types.SpacingType.LINEAR:
    raise ValueError(
        "exp_linear_to_geometric requires LINEAR spacing input, got "
        f"{dist.spacing_type}"
    )
  x_min_exp = float(np.exp(dist.x_min))

  return ra_distributions.DenseDiscreteDist(
      x_min=x_min_exp,
      step=dist.step,
      prob_arr=dist.prob_arr.copy(),
      p_min=dist.p_min,  # −∞ atom → 0 atom (p_min identity preserved)
      p_max=dist.p_max,  # +∞ atom unchanged
      spacing_type=ra_types.SpacingType.GEOMETRIC,
      domain=ra_distributions.Domain.POSITIVES,
  )


def _log_geometric_to_linear(
    dist: ra_distributions.DenseDiscreteDist,
) -> ra_distributions.DenseDiscreteDist:
  """Map a geometric-grid distribution to a linear-grid one via log(.).

  Maps POSITIVES domain → REALS domain.
  The 0 atom (p_min in POSITIVES) maps to the −∞ atom (p_min in REALS).
  """
  if dist.spacing_type != ra_types.SpacingType.GEOMETRIC:
    raise ValueError(
        "log_geometric_to_linear requires GEOMETRIC spacing input, got "
        f"{dist.spacing_type}"
    )
  x_min_log = float(np.log(dist.x_min))

  return ra_distributions.DenseDiscreteDist(
      x_min=x_min_log,
      step=dist.step,
      prob_arr=dist.prob_arr.copy(),
      p_min=dist.p_min,  # 0 atom → −∞ atom (p_min identity preserved)
      p_max=dist.p_max,
      spacing_type=ra_types.SpacingType.LINEAR,
      domain=ra_distributions.Domain.REALS,
  )


def _negate_reverse_linear_distribution(
    dist: ra_distributions.DenseDiscreteDist,
) -> ra_distributions.DenseDiscreteDist:
  """Map X -> -X, reverse PMF order, and swap boundary atoms."""
  n = dist.prob_arr.size
  return ra_distributions.DenseDiscreteDist(
      x_min=-(dist.x_min + dist.step * (n - 1)),
      step=dist.step,
      prob_arr=np.flip(dist.prob_arr),
      p_min=dist.p_max,
      p_max=dist.p_min,
  )


def _calc_pld_dual(
    realization: ra_distributions.PLDRealization,
) -> ra_distributions.PLDRealization:
  """Compute the paper PLD dual ``D(L)`` (Definition 3.1).

  Algorithm 7 (`PLD-dual`) in Appendix C of https://arxiv.org/abs/2602.17284.

  For a PLD realization ``L`` with support ``l`` and mass ``f_L(l)``, the
  dual has:
  - finite mass ``f_D(-l) = f_L(l) * exp(-l)``,
  - support reflected to ``-l``,
  - residual mass at ``+inf``.
  """
  if not isinstance(realization, ra_distributions.PLDRealization):
    raise TypeError(
        f"calc_pld_dual requires PLDRealization, got {type(realization)}"
    )

  dual_probs_aligned = np.zeros_like(realization.prob_arr)
  mask = realization.prob_arr > 0
  dual_probs_aligned[mask] = np.exp(
      np.log(realization.prob_arr[mask]) - realization._x_array[mask]
  )
  dual_probs = np.flip(dual_probs_aligned)

  sum_prob = math.fsum(map(float, dual_probs))
  if sum_prob > 1.0:
    dual_probs *= 1.0 / sum_prob
    sum_prob = 1.0

  return ra_distributions.PLDRealization(
      x_min=-(
          realization.x_min + realization.step * (realization.prob_arr.size - 1)
      ),
      step=realization.step,
      prob_arr=dual_probs,
      p_max=max(0.0, 1.0 - sum_prob),
      p_min=0.0,
  )


# =============================================================================
# Internal Helper Functions
# =============================================================================


def _validate_dense_linear_dist(dist: object) -> None:
  """Raise TypeError if dist is not a LINEAR DenseDiscreteDist."""
  if not (
      isinstance(dist, ra_distributions.DenseDiscreteDist)
      and dist.spacing_type == ra_types.SpacingType.LINEAR
  ):
    _st = getattr(dist, "spacing_type", "?")
    raise TypeError(
        f"Expected DenseDiscreteDist with LINEAR spacing, "
        f"got {type(dist).__name__} with spacing {_st}"
    )


def _validate_dense_geometric_dist(dist: object) -> None:
  """Raise TypeError if dist is not a GEOMETRIC DenseDiscreteDist."""
  if not (
      isinstance(dist, ra_distributions.DenseDiscreteDist)
      and dist.spacing_type == ra_types.SpacingType.GEOMETRIC
  ):
    _st = getattr(dist, "spacing_type", "?")
    raise TypeError(
        f"Expected DenseDiscreteDist with GEOMETRIC spacing, "
        f"got {type(dist).__name__} with spacing {_st}"
    )


# =============================================================================
# Privacy Parameter Validation
# =============================================================================


def _validate_privacy_params(
    params: ra_types.PrivacyParams,
    *,
    require_delta: bool = False,
    require_epsilon: bool = False,
) -> None:
  """Validate PrivacyParams object.

  Args:
      params: Privacy parameters to validate.
      require_delta: If True, validate that delta is set and in valid range
          (0, 1).
      require_epsilon: If True, validate that epsilon is set and positive.

  Raises:
      TypeError: If params is not a PrivacyParams instance.
      ValueError: If any parameter value is invalid.

  """
  if not isinstance(params, ra_types.PrivacyParams):
    raise TypeError(f"params must be PrivacyParams, got {type(params)}")
  _validate_gaussian_params(
      params.sigma, params.num_steps, params.num_selected, params.num_epochs
  )
  if require_delta:
    _validate_delta(params.delta)
  if require_epsilon:
    _validate_epsilon(params.epsilon)


def _validate_gaussian_params(
    sigma: float,
    num_steps: int,
    num_selected: int,
    num_epochs: int,
) -> None:
  """Validate Gaussian allocation parameters.

  Args:
      sigma: Gaussian noise scale.
      num_steps: Total number of random-allocation steps.
      num_selected: Number of selections per epoch.
      num_epochs: Number of epochs.

  Raises:
      ValueError: If any parameter value is invalid.

  """
  if sigma <= 0:
    raise ValueError(f"sigma must be positive, got {sigma}")
  _validate_allocation_params(num_steps, num_selected, num_epochs)


def _validate_allocation_params(
    num_steps: int,
    num_selected: int,
    num_epochs: int,
) -> None:
  """Validate allocation parameters.

  Args:
      num_steps: Total number of random-allocation steps.
      num_selected: Number of selections per epoch.
      num_epochs: Number of epochs.

  Raises:
      ValueError: If any parameter value is invalid.

  """
  if num_steps < 1 or num_selected < 1 or num_epochs < 1:
    raise ValueError(
        f"num_steps (={num_steps}), num_selected (={num_selected}), "
        f"and num_epochs (={num_epochs}) must be >= 1"
    )
  if num_selected > num_steps:
    raise ValueError(
        f"num_selected ({num_selected}) cannot exceed num_steps ({num_steps})"
    )


def _validate_delta(delta: float | None) -> None:
  """Validate delta value.

  Args:
      delta: Delta value for differential privacy.

  Raises:
      ValueError: If delta is None or not in the valid range (0, 1).

  """
  if delta is None or not 0 < delta < 1:
    raise ValueError(f"delta must be in (0, 1), got {delta}")


def _validate_epsilon(epsilon: float | None) -> None:
  """Validate epsilon value.

  Args:
      epsilon: Epsilon value for differential privacy.

  Raises:
      ValueError: If epsilon is None or not positive.

  """
  if epsilon is None or epsilon <= 0:
    raise ValueError(f"epsilon must be positive, got {epsilon}")


# =============================================================================
# Bound Type Validation
# =============================================================================


def _validate_bound_type(bound_type: ra_types.BoundType) -> None:
  """Validate BoundType enum value.

  Args:
      bound_type: The bound type to validate.

  Raises:
      ValueError: If bound_type is not DOMINATES or IS_DOMINATED.

  """
  if bound_type not in (
      ra_types.BoundType.DOMINATES,
      ra_types.BoundType.IS_DOMINATED,
  ):
    raise ValueError(f"Invalid bound_type: {bound_type}")


# =============================================================================
# Discretization Parameter Validation
# =============================================================================


def _validate_discretization_params(
    loss_discretization: float,
    tail_truncation: float,
) -> None:
  """Validate discretization parameters.

  Args:
      loss_discretization: Loss discretization interval.
      tail_truncation: Tail truncation threshold.

  Raises:
      ValueError: If any parameter is invalid.

  """
  if loss_discretization <= 0:
    raise ValueError(
        f"loss_discretization must be positive, got {loss_discretization}"
    )
  if tail_truncation <= 0:
    raise ValueError(f"tail_truncation must be positive, got {tail_truncation}")


def _validate_allocation_scheme_config(
    config: ra_types.AllocationSchemeConfig,
) -> None:
  """Validate AllocationSchemeConfig fields.

  Args:
      config: Configuration to validate.

  Raises:
      TypeError: If config is not an AllocationSchemeConfig instance.
      ValueError: If any field value is out of range.

  """
  if not isinstance(config, ra_types.AllocationSchemeConfig):
    raise TypeError(
        f"config must be AllocationSchemeConfig, got {type(config)}"
    )
  _validate_discretization_params(
      config.value_discretization_interval, config.tail_truncation
  )
  if config.max_grid_mult != -1 and config.max_grid_mult <= 0:
    raise ValueError(
        f"max_grid_mult must be -1 (no limit) or a positive integer, "
        f"got {config.max_grid_mult}"
    )


def _validate_optional_discretization_params(
    initial_discretization: float | None = None,
    initial_tail_truncation: float | None = None,
) -> None:
  """Validate optional discretization parameters.

  Args:
      initial_discretization: Optional initial loss discretization interval.
      initial_tail_truncation: Optional initial tail truncation threshold.

  Raises:
      ValueError: If any provided parameter is invalid.

  """
  if initial_discretization is not None and initial_discretization <= 0:
    raise ValueError(
        f"initial_discretization must be positive, got {initial_discretization}"
    )
  if initial_tail_truncation is not None and initial_tail_truncation <= 0:
    raise ValueError(
        "initial_tail_truncation must be positive, got "
        f"{initial_tail_truncation}"
    )

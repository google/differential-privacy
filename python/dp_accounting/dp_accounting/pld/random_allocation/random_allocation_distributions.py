"""Discrete distributions, PMF utilities, and grid discretization."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy import stats
from scipy.stats._distn_infrastructure import rv_frozen
from typing_extensions import Self

from .random_allocation_types import (
    BoundType,
    RegularGrid,
    SpacingType,
    validate_discrete_pmf_and_boundaries,
)

PMF_MASS_TOL = 10 * np.finfo(float).eps  # total-mass tolerance (10× machine epsilon)
RENORMALIZATION_THRESHOLD = 10 * np.finfo(float).eps
REALIZATION_MOMENT_TOL = 1e-12
SPACING_ATOL = 1e-12
SPACING_RTOL = 1e-6
MIN_GRID_SIZE = 100  # Minimum number of points in a  discretization grid.
MAX_SAFE_EXP_ARG = math.log(np.finfo(np.float64).max)
TAIL_SWITCH = 1e-10


class Domain(Enum):
    """Domain of a discrete distributsion's support."""

    REALS = "reals"  # p_min = mass at −∞, p_max = mass at +∞
    POSITIVES = "positives"  # p_min = mass at 0,  p_max = mass at +∞


# =============================================================================
# ABSTRACT BASE
# =============================================================================


class DiscreteDistBase(ABC):
    """Abstract base for discrete PMF representations with boundary masses.

    Attributes:
        prob_arr: probability mass on finite support
        p_min: mass at the lower boundary (−∞ for REALS, 0 for POSITIVES)
        p_max: mass at +∞
        domain: whether the support is over the reals or positive numbers
    """

    def __init__(
        self,
        prob_arr: NDArray[np.float64],
        p_min: float = 0.0,
        p_max: float = 0.0,
        domain: Domain = Domain.REALS,
    ) -> None:
        """Initialize discrete distribution with PMF array and boundary masses."""
        self.prob_arr = np.asarray(prob_arr, dtype=np.float64)
        self.p_min = float(p_min)
        self.p_max = float(p_max)
        self.domain = domain
        self._validate_basic()

    @abstractmethod
    def get_x_array(self) -> NDArray[np.float64]:
        """Return materialized support points."""

    @property
    def x_array(self) -> NDArray[np.float64]:
        """Materialized support."""
        return self.get_x_array()

    def _validate_basic(self) -> None:
        validate_discrete_pmf_and_boundaries(
            self.prob_arr,
            self.p_min,
            self.p_max,
        )

        pmf_sum = math.fsum(map(float, self.prob_arr))
        total_mass = pmf_sum + self.p_min + self.p_max
        mass_error = abs(total_mass - 1.0)
        if mass_error > PMF_MASS_TOL:
            error_msg = "MASS CONSERVATION ERROR"
            error_msg += f": Error={mass_error:.2e} (tolerance={PMF_MASS_TOL:.2e})"
            error_msg += f", PMF sum={pmf_sum:.15f}"
            error_msg += f", min={self.p_min:.2e}"
            error_msg += f", max={self.p_max:.2e}"
            error_msg += f", Total mass={total_mass:.15f}"
            raise ValueError(error_msg)

        # REALS domain: both boundaries being non-zero is not allowed.
        if self.domain == Domain.REALS and self.p_min > PMF_MASS_TOL and self.p_max > PMF_MASS_TOL:
            raise ValueError("REALS domain: p_min and p_max cannot both be non-zero")

    def truncate_edges(self, tail_truncation: float, bound_type: BoundType) -> Self:
        """Truncate distribution edges. Computation lives in distribution_utils."""
        new_prob_arr, new_p_min, new_p_max, min_ind, max_ind = compute_truncation(
            self.prob_arr, self.p_min, self.p_max, tail_truncation, bound_type
        )
        return self._create_truncated(new_prob_arr, new_p_min, new_p_max, min_ind, max_ind)

    @abstractmethod
    def _create_truncated(
        self,
        new_prob_arr: NDArray[np.float64],
        new_p_min: float,
        new_p_max: float,
        min_ind: int,
        max_ind: int,
    ) -> Self:
        """Create truncated instance preserving representation semantics."""

    @abstractmethod
    def copy(self) -> Self:
        """Deep-copy this distribution while preserving representation type."""


# =============================================================================
# GENERAL (EXPLICIT) DISTRIBUTION
# =============================================================================


class SparseDiscreteDist(DiscreteDistBase):
    """General discrete distribution with explicit support values."""

    def __init__(
        self,
        x_array: NDArray[np.float64],
        prob_arr: NDArray[np.float64],
        p_min: float = 0.0,
        p_max: float = 0.0,
        domain: Domain = Domain.REALS,
    ) -> None:
        """Initialize general discrete distribution with explicit support points."""
        self._x_array = np.asarray(x_array, dtype=np.float64)
        super().__init__(prob_arr, p_min, p_max, domain)
        self._validate_x_array()

    def _validate_x_array(self) -> None:
        if self._x_array.ndim != 1 or self._x_array.shape != self.prob_arr.shape:
            raise ValueError("x and PMF must be 1-D arrays of equal length")
        if not np.all(np.diff(self._x_array) > 0):
            raise ValueError("x must be strictly increasing")

    def get_x_array(self) -> NDArray[np.float64]:
        """Return materialized support points."""
        return self._x_array

    def _create_truncated(
        self,
        new_prob_arr: NDArray[np.float64],
        new_p_min: float,
        new_p_max: float,
        min_ind: int,
        max_ind: int,
    ) -> SparseDiscreteDist:
        return SparseDiscreteDist(
            x_array=self._x_array[slice(min_ind, max_ind + 1)].copy(),
            prob_arr=new_prob_arr,
            p_min=new_p_min,
            p_max=new_p_max,
            domain=self.domain,
        )

    def copy(self) -> SparseDiscreteDist:
        """Create a deep copy of this distribution."""
        return SparseDiscreteDist(
            x_array=self._x_array.copy(),
            prob_arr=self.prob_arr.copy(),
            p_min=self.p_min,
            p_max=self.p_max,
            domain=self.domain,
        )


# =============================================================================
# UNIFIED REGULAR-GRID DISTRIBUTION
# =============================================================================


class DenseDiscreteDist(DiscreteDistBase):
    """Discrete distribution on a regular (linear or geometric) grid.

    spacing_type = LINEAR:    x[i] = x_min + i * step   (step = additive gap > 0)
    spacing_type = GEOMETRIC: x[i] = x_min * exp(i * step) (step = log-ratio > 0)

    For geometric grids the domain is always POSITIVES (x_min > 0 enforces positivity).
    """

    def __init__(
        self,
        x_min: float,
        step: float,
        prob_arr: NDArray[np.float64],
        p_min: float = 0.0,
        p_max: float = 0.0,
        spacing_type: SpacingType = SpacingType.LINEAR,
        domain: Domain = Domain.REALS,
        grid: RegularGrid | None = None,
    ) -> None:
        """Initialize regular-grid discrete distribution."""
        prob_arr = np.asarray(prob_arr, dtype=np.float64)
        if grid is None:
            grid = RegularGrid(
                x_min=float(x_min),
                step=float(step),
                size=prob_arr.size,
                spacing_type=spacing_type,
            )
        elif grid.size != prob_arr.size:
            raise ValueError(
                f"Grid size must match PMF size, got grid.size={grid.size}, "
                f"prob_arr.size={prob_arr.size}"
            )
        self.grid = grid
        super().__init__(prob_arr, p_min, p_max, domain)
        self._validate_grid()

    @classmethod
    def from_grid(
        cls,
        *,
        grid: RegularGrid,
        prob_arr: NDArray[np.float64],
        p_min: float = 0.0,
        p_max: float = 0.0,
        domain: Domain = Domain.REALS,
    ) -> "DenseDiscreteDist":
        """Create DenseDiscreteDist from exact regular-grid metadata."""
        return cls(
            x_min=grid.x_min,
            step=grid.step,
            prob_arr=prob_arr,
            p_min=p_min,
            p_max=p_max,
            spacing_type=grid.spacing_type,
            domain=domain,
            grid=grid,
        )

    @property
    def x_min(self) -> float:
        return self.grid.x_min

    @property
    def step(self) -> float:
        return self.grid.step

    @property
    def spacing_type(self) -> SpacingType:
        return self.grid.spacing_type

    def _validate_grid(self) -> None:
        if self.spacing_type == SpacingType.LINEAR:
            if self.step <= 0.0:
                raise ValueError("step must be positive for linear grid")
        elif self.spacing_type == SpacingType.GEOMETRIC:
            if self.x_min <= 0.0:
                raise ValueError("x_min must be positive for geometric grid")
            if self.step <= 0.0:
                raise ValueError("step must be positive for geometric grid")
            if self.domain != Domain.POSITIVES:
                raise ValueError("Geometric spacing requires domain=Domain.POSITIVES")
        else:
            raise ValueError(f"Unknown SpacingType: {self.spacing_type}")

    @classmethod
    def from_x_array(
        cls,
        x_array: NDArray[np.float64],
        prob_arr: NDArray[np.float64],
        p_min: float = 0.0,
        p_max: float = 0.0,
        spacing_type: SpacingType = SpacingType.LINEAR,
        domain: Domain = Domain.REALS,
    ) -> "DenseDiscreteDist":
        """Create DenseDiscreteDist from x_array by extracting x_min and step."""
        if spacing_type == SpacingType.LINEAR:
            step = compute_bin_width(x_array)
        else:
            step = compute_bin_log_ratio(x_array)
        return cls(
            x_min=float(x_array[0]),
            step=step,
            prob_arr=prob_arr,
            p_min=p_min,
            p_max=p_max,
            spacing_type=spacing_type,
            domain=domain,
        )

    def get_x_array(self) -> NDArray[np.float64]:
        """Return materialized support points."""
        return self.grid.x_array

    def _create_truncated(
        self,
        new_prob_arr: NDArray[np.float64],
        new_p_min: float,
        new_p_max: float,
        min_ind: int,
        max_ind: int,
    ) -> "DenseDiscreteDist":
        if self.spacing_type == SpacingType.LINEAR:
            new_grid = RegularGrid(
                x_min=self.x_min + min_ind * self.step,
                step=self.step,
                size=new_prob_arr.size,
                spacing_type=self.spacing_type,
            )
        else:
            new_grid = RegularGrid(
                x_min=self.x_min * float(np.exp(self.step * min_ind)),
                step=self.step,
                size=new_prob_arr.size,
                spacing_type=self.spacing_type,
            )
        return self.__class__(
            x_min=new_grid.x_min,
            step=new_grid.step,
            prob_arr=new_prob_arr,
            p_min=new_p_min,
            p_max=new_p_max,
            spacing_type=self.spacing_type,
            domain=self.domain,
            grid=new_grid,
        )

    def copy(self) -> "DenseDiscreteDist":
        """Create a deep copy of this distribution."""
        return self.__class__(
            x_min=self.grid.x_min,
            step=self.grid.step,
            prob_arr=self.prob_arr.copy(),
            p_min=self.p_min,
            p_max=self.p_max,
            spacing_type=self.spacing_type,
            domain=self.domain,
            grid=self.grid,
        )


# =============================================================================
# PLD REALIZATION
# =============================================================================


class PLDRealization(DenseDiscreteDist):
    """Linear-grid PLD realization in loss space."""

    def __init__(
        self,
        x_min: float,
        step: float,
        prob_arr: NDArray[np.float64],
        p_min: float = 0.0,
        p_max: float = 0.0,
    ) -> None:
        """Initialize PLD realization with privacy loss values and probabilities."""
        super().__init__(
            x_min=x_min,
            step=step,
            prob_arr=prob_arr,
            p_min=float(p_min),
            p_max=float(p_max),
            spacing_type=SpacingType.LINEAR,
            domain=Domain.REALS,
        )
        self._validate_pld_realization()

    @classmethod
    def from_linear_dist(cls, dist: DenseDiscreteDist) -> "PLDRealization":
        """Build a validated PLD realization from a linear-grid DenseDiscreteDist."""
        if not isinstance(dist, DenseDiscreteDist) or dist.spacing_type != SpacingType.LINEAR:
            raise TypeError(
                f"from_linear_dist requires DenseDiscreteDist with LINEAR spacing, got {type(dist)}"
            )
        return cls(
            x_min=dist.x_min,
            step=dist.step,
            prob_arr=dist.prob_arr,
            p_max=dist.p_max,
            p_min=dist.p_min,
        )

    def _validate_pld_realization(self) -> None:
        """Validate the properties of PLD-realization.

        1. p(-inf) = 0 (p_min = 0).
        2. E[e^(-X)] <= 1.
        """
        # PLD realizations must have zero mass at negative-infinity loss.
        if self.p_min > PMF_MASS_TOL:
            raise ValueError(f"PLD realization requires p_min = 0, got {self.p_min:.2e}")

        exp_moment_val = exp_moment_terms(prob_arr=self.prob_arr, x_vals=self.x_array)
        if np.any(np.isinf(exp_moment_val)):
            raise ValueError(
                "Exponential moment E[exp(-L)] is infinite, not a valid PLD realization"
            )
        exp_moment_total = math.fsum(map(float, exp_moment_val))
        if exp_moment_total > 1.0 + REALIZATION_MOMENT_TOL:
            raise ValueError(
                f"Exponential moment E[exp(-L)] = {exp_moment_total:.15f} > 1.0, "
                "not a valid PLD realization"
            )

    def copy(self) -> "PLDRealization":
        """Create a deep copy of this PLD realization."""
        return PLDRealization(
            x_min=self.x_min,
            step=self.step,
            prob_arr=self.prob_arr.copy(),
            p_max=self.p_max,
            p_min=self.p_min,
        )

    def truncate_edges(  # type: ignore[override]
        self, tail_truncation: float, bound_type: BoundType
    ) -> DenseDiscreteDist:
        if bound_type == BoundType.IS_DOMINATED:
            # IS_DOMINATED can set p_min > 0, violating PLDRealization.p_min = 0.
            # Delegate through a plain DenseDiscreteDist so the result is not a PLDRealization.
            return DenseDiscreteDist(
                x_min=self.x_min,
                step=self.step,
                prob_arr=self.prob_arr.copy(),
                p_min=self.p_min,
                p_max=self.p_max,
            ).truncate_edges(tail_truncation, bound_type)
        return super().truncate_edges(tail_truncation, bound_type)

    def _create_truncated(
        self,
        new_prob_arr: NDArray[np.float64],
        new_p_min: float,
        new_p_max: float,
        min_ind: int,
        max_ind: int,
    ) -> "PLDRealization":
        """Create a truncated PLD realization while preserving linear-loss semantics."""
        del max_ind
        return PLDRealization(
            x_min=self.x_min + min_ind * self.step,
            step=self.step,
            prob_arr=new_prob_arr,
            p_min=new_p_min,
            p_max=new_p_max,
        )

# =============================================================================
# Public Utility Functions
# =============================================================================


def enforce_mass_conservation(
    *,
    prob_arr: NDArray[np.float64],
    expected_p_min: float,
    expected_p_max: float,
    bound_type: BoundType,
) -> tuple[NDArray[np.float64], float, float]:
    """Enforce total mass with one bound-type-selected boundary held fixed.

    - ``DOMINATES`` enforces ``expected_p_max``.
    - ``IS_DOMINATED`` enforces ``expected_p_min``.

    Excess mass is removed directionally over an extended array that includes the
    opposite boundary, matching the truncation logic:
    - ``DOMINATES`` trims from the left over ``[p_min, *prob_arr]``.
    - ``IS_DOMINATED`` trims from the right over ``[*prob_arr, p_max]``.

    Any remaining slack is assigned to the enforced boundary.
    """
    prob_arr = np.asarray(prob_arr, dtype=np.float64).copy()
    validate_discrete_pmf_and_boundaries(
        prob_arr,
        expected_p_min,
        expected_p_max,
    )

    total_mass = math.fsum(map(float, prob_arr)) + expected_p_min + expected_p_max
    if total_mass <= 0.0:
        raise ValueError("Cannot enforce mass conservation with zero total mass")

    if bound_type == BoundType.DOMINATES:
        if expected_p_max > 1.0:
            raise ValueError("Expected p_max cannot exceed 1")
        extended = np.concatenate(([expected_p_min], prob_arr))
        target_mass = 1.0 - expected_p_max
        current_mass = math.fsum(map(float, extended))
        excess = current_mass - target_mass
        if excess > 0:
            if excess < RENORMALIZATION_THRESHOLD:
                # Tiny excess (numerical noise): renormalize instead of trimming bins
                extended = extended * (target_mass / current_mass)
            else:
                extended = _zero_mass(values=extended, mass=excess, from_left=True, exact=True)
        current_mass = math.fsum(map(float, extended))
        return (
            extended[1:].copy(),
            float(extended[0]),
            expected_p_max + max(0.0, target_mass - current_mass),
        )

    if bound_type == BoundType.IS_DOMINATED:
        if expected_p_min > 1.0:
            raise ValueError("Expected p_min cannot exceed 1")
        extended = np.concatenate((prob_arr, [expected_p_max]))
        target_mass = 1.0 - expected_p_min
        current_mass = math.fsum(map(float, extended))
        excess = current_mass - target_mass
        if excess > 0:
            if excess < RENORMALIZATION_THRESHOLD:
                # Tiny excess (numerical noise): renormalize instead of trimming bins
                extended = extended * (target_mass / current_mass)
            else:
                extended = _zero_mass(values=extended, mass=excess, from_left=False, exact=True)
        current_mass = math.fsum(map(float, extended))
        return (
            extended[:-1].copy(),
            expected_p_min + max(0.0, target_mass - current_mass),
            float(extended[-1]),
        )

    raise ValueError(
        f"Invalid bound_type: {bound_type}. Must be BoundType.DOMINATES or BoundType.IS_DOMINATED."
    )


def compute_bin_width_two_arrays(
    *, x_array_1: NDArray[np.float64], x_array_2: NDArray[np.float64]
) -> float:
    """Compute linear spacing width for two grids and return their average."""
    w1 = compute_bin_width(x_array_1)
    w2 = compute_bin_width(x_array_2)
    if not stable_isclose(a=w1, b=w2):
        raise ValueError(f"Grid spacing must match: w1={w1:.12g} vs w2={w2:.12g}")
    return (w1 + w2) / 2


# =============================================================================
# Grid Spacing Utilities
# =============================================================================


def compute_bin_log_ratio(x_array: NDArray[np.float64]) -> float:
    """Compute geometric log-ratio spacing for a grid."""
    if x_array.size < 2:
        raise ValueError("Cannot compute geometric bin ratio with less than 2 bins")
    if np.any(x_array <= 0):
        raise ValueError("Cannot compute geometric bin ratio for non-positive values")
    log_x = np.log(x_array)
    log_ratio = float((log_x[-1] - log_x[0]) / (x_array.size - 1))
    diffs = np.diff(log_x)
    if not np.allclose(log_ratio, diffs, rtol=SPACING_RTOL, atol=SPACING_ATOL):
        max_diff = np.max(np.abs(log_ratio - diffs))
        raise ValueError(
            "Distribution has non-uniform bin widths: "
            f"log_ratio={log_ratio}, max_diff={max_diff}"
        )
    return log_ratio


def compute_bin_width(x_array: NDArray[np.float64]) -> float:
    """Compute linear spacing width for a grid."""
    if x_array.size < 2:
        raise ValueError("Cannot compute width with less than 2 bins")
    diffs = np.diff(x_array)
    median_diff = np.median(diffs)
    if not np.allclose(median_diff, diffs, rtol=SPACING_RTOL, atol=SPACING_ATOL):
        max_diff = np.max(np.abs(median_diff - diffs))
        raise ValueError(
            "Distribution has non-uniform bin widths: "
            f"median_diff={median_diff}, max diff={max_diff}"
        )
    return float(median_diff)


def stable_isclose(*, a: float, b: float) -> bool:
    """Consistent closeness check using shared spacing tolerances."""
    return bool(np.isclose(a, b, rtol=SPACING_RTOL, atol=SPACING_ATOL))


def stable_array_equal(*, a: NDArray[np.float64], b: NDArray[np.float64]) -> bool:
    """Consistent array closeness check using shared spacing tolerances."""
    return a.shape == b.shape and np.allclose(a, b, rtol=SPACING_RTOL, atol=SPACING_ATOL)


def exp_moment_terms(
    *,
    prob_arr: NDArray[np.float64],
    x_vals: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return per-bin contributions to ``E[exp(-X)]``.

    For very negative ``x_vals`` the naive product ``p * exp(-x)`` can overflow
    even when the combined term is representable. In that regime we evaluate the
    contribution as ``exp(log(p) - x)`` instead.

    Terms that still exceed float64 range are returned as ``inf``.
    """
    prob_arr = np.asarray(prob_arr, dtype=np.float64)
    x_vals = np.asarray(x_vals, dtype=np.float64)
    if prob_arr.shape != x_vals.shape:
        raise ValueError("prob_arr and x_vals must have the same shape")

    terms = np.zeros_like(prob_arr, dtype=np.float64)
    positive_mask = prob_arr > 0.0
    safe_mask = positive_mask & (x_vals >= -MAX_SAFE_EXP_ARG)
    if np.any(safe_mask):
        terms[safe_mask] = prob_arr[safe_mask] * np.exp(-x_vals[safe_mask])

    extreme_mask = positive_mask & (x_vals < -MAX_SAFE_EXP_ARG)
    if np.any(extreme_mask):
        log_terms = np.log(prob_arr[extreme_mask]) - x_vals[extreme_mask]
        terms_extreme = np.exp(np.minimum(log_terms, MAX_SAFE_EXP_ARG))
        terms_extreme[log_terms > MAX_SAFE_EXP_ARG] = np.inf
        terms[extreme_mask] = terms_extreme

    return terms


# =============================================================================
# Distribution Edge Truncation
# =============================================================================


def compute_truncation(
    prob_arr: NDArray[np.float64],
    p_min: float,
    p_max: float,
    tail_truncation: float,
    bound_type: BoundType,
) -> tuple[NDArray[np.float64], float, float, int, int]:
    """Compute truncated distribution parameters without creating objects.

    Algorithm:
      A. Remove leading/trailing zeros from PMF (always done).
      B. If tail_truncation > 0:
         - Compute how much to consume from each side (up to tail_truncation).
         - For DOMINATES:    Operate over the [p_min, *prob_arr] range.
                             Left tail folds into first remaining element;
                             right tail goes to p_max.
         - For IS_DOMINATED: Operate over the [*prob_arr, p_max] range.
                             Right tail folds into last remaining element;
                             left tail goes to p_min;
      C. Apply step A again to remove any newly created leading/trailing zeros.

    Returns:
        (new_prob_arr, new_p_min, new_p_max, min_ind, max_ind) where min_ind and
        max_ind are indices into the original prob_arr.
    """
    # Remove zero probability tails to reduce unnecessary computations
    inner_min, inner_max = _strip_zero_edges(prob_arr)
    trimmed_prob_arr = prob_arr[slice(inner_min, inner_max + 1)].copy()

    if tail_truncation == 0.0:
        return trimmed_prob_arr.copy(), p_min, p_max, inner_min, inner_max

    if bound_type == BoundType.DOMINATES:
        extended_prob = np.concatenate([[p_min], trimmed_prob_arr])
        original_mass = math.fsum(map(float, extended_prob))
        # Truncate left tail and add its mass to the next finite bin
        extended_prob = _zero_mass(
            values=extended_prob, mass=tail_truncation, from_left=True, exact=False
        )
        shifted_mass = original_mass - math.fsum(map(float, extended_prob))
        extended_prob[np.nonzero(extended_prob)[0][0]] += shifted_mass
        p_min_out = extended_prob[0]
        # Truncate right tail and add its mass to to p_max
        extended_prob = _zero_mass(
            values=extended_prob, mass=tail_truncation, from_left=False, exact=False
        )
        shifted_mass = original_mass - math.fsum(map(float, extended_prob))
        p_max_out = p_max + shifted_mass
        prob_arr_out = extended_prob[1:]
    elif bound_type == BoundType.IS_DOMINATED:
        extended_prob = np.concatenate((trimmed_prob_arr, [p_max]))
        original_mass = math.fsum(map(float, extended_prob))
        # Truncate right tail and add its mass to the next finite bin
        extended_prob = _zero_mass(
            values=extended_prob, mass=tail_truncation, from_left=False, exact=False
        )
        shifted_mass = original_mass - math.fsum(map(float, extended_prob))
        extended_prob[np.nonzero(extended_prob)[0][-1]] += shifted_mass
        p_max_out = extended_prob[-1]
        # Truncate left tail and add its mass to to p_min
        extended_prob = _zero_mass(
            values=extended_prob, mass=tail_truncation, from_left=True, exact=False
        )
        shifted_mass = original_mass - math.fsum(map(float, extended_prob))
        p_min_out = p_min + shifted_mass
        prob_arr_out = extended_prob[:-1]
    else:
        raise ValueError(f"Unknown BoundType: {bound_type}")

    # Remove zero probability tails to reduce unnecessary computations
    inner_min_new, inner_max_new = _strip_zero_edges(prob_arr_out)
    min_ind_new = inner_min + inner_min_new
    max_ind_new = inner_min + inner_max_new
    return (
        prob_arr_out[slice(inner_min_new, inner_max_new + 1)].copy(),
        p_min_out,
        p_max_out,
        min_ind_new,
        max_ind_new,
    )


def _strip_zero_edges(prob_arr: NDArray[np.float64]) -> tuple[int, int]:
    """Return (min_ind, max_ind) of the nonzero range in prob_arr.

    Raises ValueError if all mass is zero.
    """
    nonzero_indices = np.nonzero(prob_arr)[0]
    if nonzero_indices.size == 0:
        raise ValueError("Cannot truncate distribution with zero finite mass")
    return int(nonzero_indices[0]), int(nonzero_indices[-1])


def _zero_mass(
    *,
    values: NDArray[np.float64],
    mass: float,
    from_left: bool,
    exact: bool,
) -> NDArray[np.float64]:
    """Remove mass probability from values from one of the side, based on ``from_left``.

    If ``exact`` is true, partially consume the pivot bin so that exactly
    ``mass`` is removed. Otherwise, consume only complete bins whose cumulative
    mass does not exceed ``mass`` and leave the pivot bin unchanged.
    """
    if mass <= 0.0:
        return values
    total_mass = math.fsum(map(float, values))
    if mass >= total_mass:
        raise ValueError(
            "mass must be smaller than total array mass, "
            f"got mass={mass:.12g}, total={total_mass:.12g}"
        )

    # When removing from the right, we just flip the array before and after the calculation
    if not from_left:
        values = values[::-1]

    # Find the pivot index
    cumsum = np.cumsum(values, dtype=np.float64)
    if exact:
        pivot = int(np.searchsorted(cumsum, mass, side="left"))
    else:
        pivot = int(np.searchsorted(cumsum, mass, side="right"))

    # Remove the probability mass below the pivot
    removed_before = float(cumsum[pivot - 1]) if pivot > 0 else 0.0
    if pivot > 0:
        values[:pivot] = 0.0
    # Remove the additional probability mass from the pivot if needed
    if exact:
        values[pivot] = max(0.0, values[pivot] - (mass - removed_before))

    if not from_left:
        values = values[::-1]
    return values


# =============================================================================
# Public API: Continuous Distribution Discretization
# =============================================================================


def discretize_continuous_distribution(
    *,
    dist: stats.rv_continuous | rv_frozen[Any, Any],
    tail_truncation: float,
    bound_type: BoundType,
    spacing_type: SpacingType,
    step: float,
    align_to_multiples: bool,
    domain: Domain = Domain.REALS,
) -> DenseDiscreteDist:
    """Discretize a continuous distribution to a typed structured representation.

    Args:
        dist: Continuous distribution to discretize.
        tail_truncation: Tail mass budget used to define quantile bounds and bin increment floor.
        bound_type: Tie-breaking direction for interval mass assignment.
        spacing_type: Output grid spacing family (linear or geometric).
        step: Linear bin width or geometric log-ratio.
        align_to_multiples: Whether to align the quantile-derived bounds to integer step multiples.
        domain: Domain semantics for boundary masses in the output discrete distribution.

    Returns:
        Discretized distribution on a structured dense grid.
    """

    grid = _discretize_continuous_to_grid(
        dist=dist,
        tail_truncation=tail_truncation,
        spacing_type=spacing_type,
        step=step,
        align_to_multiples=align_to_multiples,
    )
    x_array = grid.x_array
    if x_array[0] <= 0 and domain == Domain.POSITIVES:
        dist_name = getattr(dist, "name", type(dist).__name__)
        raise ValueError(
            f"Cannot discretize {dist_name} to a positive range, got x_min={x_array[0]}"
        )

    # 2. Map density to PMF with semantics.
    return discretize_continuous_grid(
        dist=dist,
        grid=grid,
        bound_type=bound_type,
        PMF_min_increment=tail_truncation,
        domain=domain,
    )


def discretize_continuous_dist(
    *,
    dist: stats.rv_continuous | rv_frozen[Any, Any],
    x_array: NDArray[np.float64],
    bound_type: BoundType,
    PMF_min_increment: float,
    spacing_type: SpacingType,
    domain: Domain = Domain.REALS,
) -> DenseDiscreteDist:
    """Convert continuous distribution to discrete PMF with bounding semantics."""
    prob_arr, p_min, p_max = _discretize_continuous_prob_arr(
        dist=dist,
        x_array=x_array,
        bound_type=bound_type,
        PMF_min_increment=PMF_min_increment,
    )

    if spacing_type == SpacingType.LINEAR:
        return DenseDiscreteDist.from_x_array(
            x_array=x_array,
            prob_arr=prob_arr,
            p_min=p_min,
            p_max=p_max,
            domain=domain,
        )

    if spacing_type == SpacingType.GEOMETRIC:
        return DenseDiscreteDist.from_x_array(
            x_array=x_array,
            prob_arr=prob_arr,
            p_min=p_min,
            p_max=p_max,
            spacing_type=SpacingType.GEOMETRIC,
            domain=Domain.POSITIVES,
        )

    raise ValueError(f"Invalid spacing_type: {spacing_type}")


def discretize_continuous_grid(
    *,
    dist: stats.rv_continuous | rv_frozen[Any, Any],
    grid: RegularGrid,
    bound_type: BoundType,
    PMF_min_increment: float,
    domain: Domain = Domain.REALS,
) -> DenseDiscreteDist:
    """Convert continuous distribution to a discrete PMF on a known regular grid."""
    prob_arr, p_min, p_max = _discretize_continuous_prob_arr(
        dist=dist,
        x_array=grid.x_array,
        bound_type=bound_type,
        PMF_min_increment=PMF_min_increment,
    )
    if grid.spacing_type == SpacingType.LINEAR:
        return DenseDiscreteDist.from_grid(
            grid=grid,
            prob_arr=prob_arr,
            p_min=p_min,
            p_max=p_max,
            domain=domain,
        )
    if grid.spacing_type == SpacingType.GEOMETRIC:
        return DenseDiscreteDist.from_grid(
            grid=grid,
            prob_arr=prob_arr,
            p_min=p_min,
            p_max=p_max,
            domain=Domain.POSITIVES,
        )
    raise ValueError(f"Invalid spacing_type: {grid.spacing_type}")


def _discretize_continuous_prob_arr(
    *,
    dist: stats.rv_continuous | rv_frozen[Any, Any],
    x_array: NDArray[np.float64],
    bound_type: BoundType,
    PMF_min_increment: float,
) -> tuple[NDArray[np.float64], float, float]:
    """Compute discrete PMF and boundary masses on a materialized grid."""
    bin_probs, p_left, p_right = _compute_discrete_prob(
        dist=dist, x_array=x_array, bound_type=bound_type, PMF_min_increment=PMF_min_increment
    )

    n = x_array.size
    prob_arr = np.zeros(n)

    if bound_type == BoundType.DOMINATES:
        # Shift mass right: left tail (-inf, x0) -> x0,
        # each interval [x_i, x_{i+1}) -> x_{i+1}, right tail (x_n, inf) -> inf.
        prob_arr[0] = p_left
        prob_arr[1:] = bin_probs
        return prob_arr, 0.0, p_right

    if bound_type == BoundType.IS_DOMINATED:
        # Shift mass left: left tail (-inf, x0) -> -inf,
        # each interval [x_i, x_{i+1}) -> x_i, right tail (x_n, inf) -> x_n.
        prob_arr[:-1] = bin_probs
        prob_arr[-1] = p_right
        return prob_arr, p_left, 0.0

    raise ValueError(f"Unknown BoundType: {bound_type}")


def rediscretize_dist(
    *,
    dist: DiscreteDistBase,
    tail_truncation: float,
    loss_discretization: float,
    spacing_type: SpacingType,
    bound_type: BoundType,
) -> DenseDiscreteDist:
    """Rediscretize a distribution onto a requested grid spacing.

    Remaps PMF onto a new grid with the requested spacing and discretization.
    Implementation trims zero/tail regions, computes new grid size, then remaps
    using domination-aware rounding (e.g., linear grids for dp_accounting output).

    Algorithm 6 (`disc-dist`) in Appendix C.
    """

    # Support for rediscritizing a dominating distribution into a dominated one and vice versa
    working_dist = dist.copy()
    if bound_type == BoundType.IS_DOMINATED and working_dist.p_max > 0.0:
        working_dist.prob_arr[-1] += working_dist.p_max
        working_dist.p_max = 0.0
    elif (
        bound_type == BoundType.DOMINATES
        and spacing_type == SpacingType.LINEAR
        and working_dist.p_min > 0.0
    ):
        working_dist.prob_arr[0] += working_dist.p_min
        working_dist.p_min = 0.0

    # Quantile-truncation
    trunc_dist = working_dist.truncate_edges(
        tail_truncation=tail_truncation / 2, bound_type=bound_type
    )

    x_array = trunc_dist.x_array
    x_min = x_array[0]
    x_max = x_array[-1]

    grid_out = discretize_aligned_grid(
        x_min=x_min,
        x_max=x_max,
        spacing_type=spacing_type,
        align_to_multiples=True,
        discretization=loss_discretization,
    )
    x_array_out = grid_out.x_array

    prob_arr_out = rediscretize_prob(
        x_array=x_array,
        prob_arr=trunc_dist.prob_arr,
        x_array_out=x_array_out,
        dominates=(bound_type == BoundType.DOMINATES),
    )

    prob_arr_out, p_min, p_max = enforce_mass_conservation(
        prob_arr=prob_arr_out,
        expected_p_min=working_dist.p_min,
        expected_p_max=working_dist.p_max,
        bound_type=bound_type,
    )

    if spacing_type == SpacingType.LINEAR:
        return DenseDiscreteDist.from_grid(
            grid=grid_out,
            prob_arr=prob_arr_out,
            p_min=p_min,
            p_max=p_max,
        )

    if spacing_type == SpacingType.GEOMETRIC:
        return DenseDiscreteDist.from_grid(
            grid=grid_out,
            prob_arr=prob_arr_out,
            p_min=p_min,
            p_max=p_max,
            domain=Domain.POSITIVES,
        )

    raise ValueError(f"Invalid spacing_type: {spacing_type}")


def discretize_aligned_grid(
    *,
    x_min: float,
    x_max: float,
    spacing_type: SpacingType,
    align_to_multiples: bool,
    discretization: float,
) -> RegularGrid:
    """Return regular grid metadata covering [x_min, x_max].

    Args:
        x_min: Minimum value of the range.
        x_max: Maximum value of the range.
        spacing_type: Type of spacing (LINEAR or GEOMETRIC).
        align_to_multiples: If True, align range to whole multiples of discretization.
                           If False, use x_min and x_max directly without alignment.
        discretization: Grid spacing parameter (step size for LINEAR, log ratio for GEOMETRIC).

    """
    # Validate inputs
    if spacing_type not in (SpacingType.GEOMETRIC, SpacingType.LINEAR):
        raise ValueError(f"Unsupported spacing_type: {spacing_type}")
    if x_max <= x_min:
        raise ValueError(f"x_max must be greater than x_min, got x_min={x_min}, x_max={x_max}")
    if spacing_type == SpacingType.GEOMETRIC and x_min <= 0:
        raise ValueError(
            f"Geometric spacing requires positive values, got x_min={x_min}, x_max={x_max}"
        )

    if discretization <= 0:
        raise ValueError("discretization must be positive")

    d = float(discretization)

    if spacing_type == SpacingType.LINEAR:
        if align_to_multiples:
            k_lo = int(np.floor(x_min / d))
            k_hi = int(np.ceil(x_max / d))
            grid = RegularGrid(
                x_min=float(d * k_lo),
                step=d,
                size=k_hi - k_lo + 1,
                spacing_type=SpacingType.LINEAR,
            )
            # It is possible that aligned bounds miss by a small float64 roundoff.
            if grid.x_array[0] > x_min:
                k_lo -= 1
            grid = RegularGrid(
                x_min=float(d * k_lo),
                step=d,
                size=k_hi - k_lo + 1,
                spacing_type=SpacingType.LINEAR,
            )
            if grid.x_array[-1] < x_max:
                k_hi += 1
            return RegularGrid(
                x_min=float(d * k_lo),
                step=d,
                size=k_hi - k_lo + 1,
                spacing_type=SpacingType.LINEAR,
            )
        span = x_max - x_min
        n = int(np.ceil(span / d)) + 1
        grid = RegularGrid(
            x_min=float(x_min),
            step=d,
            size=n,
            spacing_type=SpacingType.LINEAR,
        )
        if grid.x_array[-1] < x_max:
            n += 1
        return RegularGrid(
            x_min=float(x_min),
            step=d,
            size=n,
            spacing_type=SpacingType.LINEAR,
        )

    # GEOMETRIC: discretization is log-ratio per step; grid x = exp(d * k).
    if align_to_multiples:
        k_lo = int(np.floor(np.log(x_min) / d))
        k_hi = int(np.ceil(np.log(x_max) / d))
        grid = RegularGrid(
            x_min=float(np.exp(d * k_lo)),
            step=d,
            size=k_hi - k_lo + 1,
            spacing_type=SpacingType.GEOMETRIC,
        )
        # It is possible that aligned bounds miss by a small float64 roundoff.
        if grid.x_array[0] > x_min:
            k_lo -= 1
        grid = RegularGrid(
            x_min=float(np.exp(d * k_lo)),
            step=d,
            size=k_hi - k_lo + 1,
            spacing_type=SpacingType.GEOMETRIC,
        )
        if grid.x_array[-1] < x_max:
            k_hi += 1
        return RegularGrid(
            x_min=float(np.exp(d * k_lo)),
            step=d,
            size=k_hi - k_lo + 1,
            spacing_type=SpacingType.GEOMETRIC,
        )
    n = int(np.ceil(np.log(x_max / x_min) / d)) + 1
    grid = RegularGrid(
        x_min=float(x_min),
        step=d,
        size=n,
        spacing_type=SpacingType.GEOMETRIC,
    )
    if grid.x_array[-1] < x_max:
        n += 1
    return RegularGrid(
        x_min=float(x_min),
        step=d,
        size=n,
        spacing_type=SpacingType.GEOMETRIC,
    )


def discretize_aligned_range(
    *,
    x_min: float,
    x_max: float,
    spacing_type: SpacingType,
    align_to_multiples: bool,
    discretization: float,
) -> NDArray[np.float64]:
    """Return a grid covering [x_min, x_max]."""
    return discretize_aligned_grid(
        x_min=x_min,
        x_max=x_max,
        spacing_type=spacing_type,
        align_to_multiples=align_to_multiples,
        discretization=discretization,
    ).x_array


@njit(cache=True)
def rediscretize_prob(
    x_array: NDArray[np.float64],
    prob_arr: NDArray[np.float64],
    x_array_out: NDArray[np.float64],
    dominates: bool,
) -> NDArray[np.float64]:
    """Remap PMF onto new grid with domination-aware rounding.

    Maps each probability mass to output grid position based on domination semantics.
    Implementation: dominates=True uses ceil (pessimistic), False uses floor (optimistic).
    Uses Kahan summation for numerical accuracy.
    Returns: remapped PMF array.

    """
    n_out = x_array_out.size
    prob_arr_out = np.zeros(n_out)
    compensations = np.zeros(n_out)

    # single pointer into x_array_out since x_array is strictly increasing
    j = 0

    if dominates:
        # ceil: bin = first index with x_array_out[j] >= z; overflow right -> p_max
        for i in range(x_array.size):
            z = x_array[i]
            mass = prob_arr[i]
            # Skip only zero-mass bins, not small-mass bins
            if mass <= 0:
                continue

            # advance while x_array_out[j] < z
            while j < n_out and x_array_out[j] < z:
                j += 1

            if j >= n_out:
                # overflow to the right: discard mass (goes to p_max via enforce_mass_conservation)
                continue
            # include values below x_array_out[0] in the first bin (ceil behavior)
            y = mass - compensations[j]
            t = prob_arr_out[j] + y
            compensations[j] = (t - prob_arr_out[j]) - y
            prob_arr_out[j] = t

    else:
        # floor: bin = last index with x_array_out[j] <= z; underflow left -> p_min
        for i in range(x_array.size):
            z = x_array[i]
            mass = prob_arr[i]
            # Skip only zero-mass bins, not small-mass bins
            if mass <= 0:
                continue

            # advance while x_array_out[j] <= z
            while j < n_out and x_array_out[j] <= z:
                j += 1

            idx = j - 1
            if idx < 0:
                # underflow to the left: discard mass (goes to p_min via enforce_mass_conservation)
                continue
            # include values above x_array_out[-1] in the last bin (floor behavior)
            y = mass - compensations[idx]
            t = prob_arr_out[idx] + y
            compensations[idx] = (t - prob_arr_out[idx]) - y
            prob_arr_out[idx] = t

    return prob_arr_out


# =============================================================================
# Helper Functions
# =============================================================================


@njit(cache=True)
def _adaptive_bins_from_cdf(
    *,
    cdf: NDArray[np.float64],
    tail_truncation: float,
) -> NDArray[np.float64]:
    """Adaptive binning from CDF with mass accumulation.

    Accumulates mass from CDF increments until threshold is reached, then assigns
    accumulated mass to current bin. All mass is conserved - no mass is discarded.
    """
    n = cdf.size
    bin_probs = np.zeros(n - 1, dtype=np.float64)
    accumulated_mass = 0.0

    for i in range(n - 1):
        # Current increment in CDF
        current_increment = cdf[i + 1] - cdf[i]
        accumulated_mass += current_increment

        if accumulated_mass >= tail_truncation:
            # Assign accumulated mass to this bin
            bin_probs[i] = accumulated_mass
            accumulated_mass = 0.0

    # Assign any remaining accumulated mass to the last bin
    if accumulated_mass > 0.0:
        bin_probs[n - 2] += accumulated_mass

    return bin_probs


@njit(cache=True)
def _adaptive_bins_from_sf(
    *,
    sf: NDArray[np.float64],
    tail_truncation: float,
) -> NDArray[np.float64]:
    """Adaptive binning from survival function with mass accumulation.

    Accumulates mass from SF increments until threshold is reached, then assigns
    accumulated mass to current bin. All mass is conserved - no mass is discarded.
    Processes from right to left (high to low x values).
    """
    n = sf.size
    bin_probs = np.zeros(n - 1, dtype=np.float64)
    accumulated_mass = 0.0

    for i in range(n - 2, -1, -1):
        # Current increment in SF (going backwards)
        current_increment = sf[i] - sf[i + 1]
        accumulated_mass += current_increment

        if accumulated_mass >= tail_truncation:
            # Assign accumulated mass to this bin
            bin_probs[i] = accumulated_mass
            accumulated_mass = 0.0

    # Assign any remaining accumulated mass to the first bin
    if accumulated_mass > 0.0:
        bin_probs[0] += accumulated_mass

    return bin_probs


def _stable_cdf_and_sf(
    *,
    dist: stats.rv_continuous | rv_frozen[Any, Any],
    x_array: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    median = dist.median()
    cdf = np.empty_like(x_array, dtype=np.float64)
    sf = np.empty_like(x_array, dtype=np.float64)

    mask_left = x_array < median
    if np.any(mask_left):
        logcdf_vals = dist.logcdf(x_array[mask_left])
        cdf[mask_left] = np.exp(logcdf_vals)
        sf[mask_left] = -np.expm1(logcdf_vals)

    mask_right = ~mask_left
    if np.any(mask_right):
        logsf_vals = dist.logsf(x_array[mask_right])
        sf[mask_right] = np.exp(logsf_vals)
        cdf[mask_right] = -np.expm1(logsf_vals)

    cdf = np.clip(cdf, 0.0, 1.0)
    sf = np.clip(sf, 0.0, 1.0)
    return cdf, sf


def _compute_discrete_prob(
    *,
    dist: stats.rv_continuous | rv_frozen[Any, Any],
    x_array: NDArray[np.float64],
    bound_type: BoundType,
    PMF_min_increment: float,
) -> tuple[NDArray[np.float64], float, float]:
    """Compute bin probabilities using adaptive CDF/SF increments with logcdf/logsf stability.

    PMF_min_increment controls the minimum CDF/SF increment that becomes a bin mass.

    """
    cdf, sf = _stable_cdf_and_sf(
        dist=dist,
        x_array=x_array,
    )
    p_left = cdf[0]
    p_right = sf[-1]
    PMF_min_increment = max(0.0, PMF_min_increment)

    if bound_type == BoundType.DOMINATES:
        # Suppress intermediate debug logging.
        bin_probs = _adaptive_bins_from_cdf(
            cdf=cdf,
            tail_truncation=PMF_min_increment,
        )
    elif bound_type == BoundType.IS_DOMINATED:
        # Suppress intermediate debug logging.
        bin_probs = _adaptive_bins_from_sf(
            sf=sf,
            tail_truncation=PMF_min_increment,
        )
    else:
        raise ValueError(f"Unknown BoundType: {bound_type}")

    return bin_probs, p_left, p_right


def _discretize_continuous_to_grid(
    *,
    dist: stats.rv_continuous | rv_frozen[Any, Any],
    tail_truncation: float,
    spacing_type: SpacingType,
    step: float,
    align_to_multiples: bool,
) -> RegularGrid:
    """Generate grid covering the quantile range defined by tail_truncation."""
    # Determine support bounds via quantiles
    x_min = float(dist.ppf(tail_truncation))
    x_max = float(dist.isf(tail_truncation))
    if not np.isfinite(x_min) or not np.isfinite(x_max):
        raise ValueError(f"Quantiles not finite: x_min={x_min}, x_max={x_max}")

    if spacing_type == SpacingType.GEOMETRIC:
        if step <= 0.0:
            raise ValueError(f"Geometric step must be positive, got {step}")
        discretization = float(step)
    else:
        if step <= 0.0:
            raise ValueError(f"Linear step must be positive, got {step}")
        discretization = float(step)

    return discretize_aligned_grid(
        x_min=x_min,
        x_max=x_max,
        spacing_type=spacing_type,
        align_to_multiples=align_to_multiples,
        discretization=discretization,
    )

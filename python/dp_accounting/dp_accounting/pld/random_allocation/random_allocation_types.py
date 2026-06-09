"""Core type definitions and low-level PMF validation for random-allocation."""

from __future__ import annotations

import warnings as _warnings
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

# numba is an optional dependency: if installed, JIT-compiled functions run
# significantly faster; if not, callers can dispatch to NumPy fallbacks. _NJIT
# is imported as a private name so callers must use _optional_njit() and cannot
# accidentally bypass the fallback by using @njit directly.
_NJIT = None
try:
    from numba import njit as _NJIT  # type: ignore[assignment]

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    _warnings.warn(
        "numba is not installed; some operations will use NumPy fallbacks "
        "and may be slower. Install numba for full performance.",
        ImportWarning,
        stacklevel=2,
    )


def _optional_njit():
    """Return a cached numba njit decorator, or the identity decorator.

    Use as a decorator at function-definition time for kernels whose Python
    body is valid (just slow) without JIT compilation.  The function runs
    correctly either way; numba only accelerates it.
    """
    if _HAS_NUMBA:
        assert _NJIT is not None
        return _NJIT(cache=True)
    return lambda x: x


def has_numba() -> bool:
    """Return whether numba JIT compilation is available.

    Use at call time to dispatch between a JIT kernel and a separate NumPy
    fallback when the two implementations differ structurally (e.g. a
    scalar loop vs. a vectorised array operation).  Unlike _optional_njit,
    this lets callers choose a fundamentally different code path at runtime.
    """
    return _HAS_NUMBA


# =============================================================================
# Discrete Distribution Types
# =============================================================================


class BoundType(Enum):
    """Tie-breaking bound_type for discretization."""

    DOMINATES = "DOMINATES"
    IS_DOMINATED = "IS_DOMINATED"


class SpacingType(Enum):
    """Grid spacing_type strategy."""

    LINEAR = "linear"
    GEOMETRIC = "geometric"


@dataclass(frozen=True)
class RegularGrid:
    """Regular grid metadata with exact spacing semantics."""

    x_min: float
    step: float
    size: int
    spacing_type: SpacingType

    @property
    def x_array(self) -> NDArray[np.float64]:
        indices = np.arange(self.size, dtype=np.float64)
        if self.spacing_type == SpacingType.LINEAR:
            return self.x_min + self.step * indices
        if self.spacing_type == SpacingType.GEOMETRIC:
            return self.x_min * np.exp(self.step * indices)
        raise ValueError(f"Unknown SpacingType: {self.spacing_type}")


class Direction(Enum):
    """Enum for direction of privacy analysis."""

    ADD = "add"
    REMOVE = "remove"


# Defaults for AllocationSchemeConfig (independent of REALIZATION_MOMENT_TOL;
# tail budget is a modeling choice).
DEFAULT_VALUE_DISCRETIZATION_INTERVAL = 1e-4
DEFAULT_TAIL_TRUNCATION = 1e-12


@dataclass
class PrivacyParams:
    """Parameters common to all privacy schemes."""

    sigma: float
    num_steps: int
    num_selected: int = 1
    num_epochs: int = 1
    epsilon: float | None = None
    delta: float | None = None


@dataclass
class AllocationSchemeConfig:
    """Configuration for privacy schemes."""

    value_discretization_interval: float = DEFAULT_VALUE_DISCRETIZATION_INTERVAL
    tail_truncation: float = DEFAULT_TAIL_TRUNCATION
    max_grid_mult: int = -1  # -1 means no upper limit on grid size


# =============================================================================
# Discrete PMF validation
# =============================================================================


def _validate_discrete_pmf_and_boundaries(
    prob_arr: NDArray[np.float64],
    p_min: float,
    p_max: float,
) -> None:
    """Validate 1-D nonnegative PMF and nonnegative boundary masses.

    Args:
        prob_arr: Finite-support probability masses.
        p_min: Lower boundary mass (e.g. mass at ``-∞`` or 0).
        p_max: Upper boundary mass (e.g. mass at ``+∞``).

    Raises:
        ValueError: If shape or nonnegativity checks fail.

    """
    prob_arr = np.asarray(prob_arr, dtype=np.float64)
    if prob_arr.ndim != 1:
        raise ValueError("PMF must be 1-D array")
    if np.any(prob_arr < 0.0):
        raise ValueError("PMF must be nonnegative")
    if p_min < 0.0:
        raise ValueError(f"min must be nonnegative, got {p_min:.2e}")
    if p_max < 0.0:
        raise ValueError(f"max must be nonnegative, got {p_max:.2e}")

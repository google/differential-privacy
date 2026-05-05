"""FFT and geometric-grid convolution for privacy loss distributions."""

from __future__ import annotations

import math
import os
import warnings

import numpy as np
from dp_accounting.pld.common import compute_self_convolve_bounds
from numba import njit
from numpy.typing import NDArray
from scipy.fft import irfft, next_fast_len, rfft

from .random_allocation_distributions import DenseDiscreteDist, Domain
from .random_allocation_distributions import enforce_mass_conservation, stable_isclose
from .random_allocation_types import BoundType, SpacingType
from .random_allocation_utils import (
    binary_self_convolve,
    convolve_boundary_masses,
    self_convolve_boundary_masses,
    validate_bound_type,
)

# Maximum bytes for a single FFT allocation (default 8 GB, override via MAX_FFT_BYTES env var)
MAX_FFT_BYTES = int(os.environ.get("MAX_FFT_BYTES", 8 * 1024**3))


def fft_convolve(
    *,
    dist_1: DenseDiscreteDist,
    dist_2: DenseDiscreteDist,
    tail_truncation: float,
    bound_type: BoundType,
) -> DenseDiscreteDist:
    """Convolve two linear-grid distributions via FFT."""
    if not (
        isinstance(dist_1, DenseDiscreteDist) and dist_1.spacing_type == SpacingType.LINEAR
    ) or not (isinstance(dist_2, DenseDiscreteDist) and dist_2.spacing_type == SpacingType.LINEAR):
        raise TypeError(
            "fft_convolve requires linear DenseDiscreteDist inputs; "
            f"got dist_1={type(dist_1).__name__} (spacing={dist_1.spacing_type}), "
            f"dist_2={type(dist_2).__name__} (spacing={dist_2.spacing_type})"
        )
    if dist_1.domain != dist_2.domain:
        raise ValueError(f"Input domains must be identical, got {dist_1.domain} vs {dist_2.domain}")
    if not np.any(dist_1.prob_arr) or not np.any(dist_2.prob_arr):
        raise ValueError("FFT convolution requires nonzero finite mass in both inputs")
    if not stable_isclose(a=dist_1.step, b=dist_2.step):
        raise ValueError(f"Grid spacing must match: w1={dist_1.step:.12g} vs w2={dist_2.step:.12g}")

    width = dist_1.step
    conv_x_min = dist_1.x_min + dist_2.x_min

    # --- Manual rfft/irfft with in-place multiply (saves one complex128 buffer) ---
    conv_full_len = dist_1.prob_arr.size + dist_2.prob_arr.size - 1
    fft_size = next_fast_len(conv_full_len)
    _check_fft_memory(fft_size, label="fft_convolve")

    # Capture ghost-mass bounds and normalization factors before FFT buffers are allocated
    nz1 = np.nonzero(dist_1.prob_arr)[0]
    nz2 = np.nonzero(dist_2.prob_arr)[0]
    min_idx = int(nz1[0] + nz2[0])
    max_idx = int(nz1[-1] + nz2[-1])
    finite_prob_1 = math.fsum(map(float, dist_1.prob_arr))
    finite_prob_2 = math.fsum(map(float, dist_2.prob_arr))

    # Self-squaring optimization: if both inputs are the same object,
    # compute rfft once and square in-place (saves one complex buffer)
    is_self_convolve = dist_1 is dist_2 or dist_1.prob_arr is dist_2.prob_arr
    fft1 = rfft(dist_1.prob_arr, n=fft_size)
    if is_self_convolve:
        fft1 *= fft1  # in-place square
    else:
        fft2 = rfft(dist_2.prob_arr, n=fft_size)
        fft1 *= fft2  # in-place multiply
        del fft2  # free second complex buffer immediately
    conv_full = irfft(fft1, n=fft_size, overwrite_x=True)
    del fft1  # free complex buffer
    conv_pmf = conv_full[:conv_full_len].copy()  # copy needed portion
    del conv_full  # free full irfft output

    # Zero negative roundoff and ghost mass outside reachable support
    conv_pmf[conv_pmf < 0] = 0.0
    conv_pmf[:min_idx] = 0.0
    max_idx_plus_one = max_idx + 1
    if max_idx_plus_one < conv_pmf.size:
        conv_pmf[max_idx_plus_one:] = 0.0

    current_finite_mass = math.fsum(map(float, conv_pmf))
    if current_finite_mass <= 0.0:
        raise ValueError("FFT convolution produced zero finite mass")
    # Renormalize finite mass before reattaching the analytically computed
    # infinity masses. This corrects small drift from FFT arithmetic/clipping.
    conv_pmf *= finite_prob_1 * finite_prob_2 / current_finite_mass

    expected_p_min, expected_p_max = convolve_boundary_masses(
        dist_1.p_min, dist_1.p_max, dist_2.p_min, dist_2.p_max, dist_1.domain
    )
    conv_pmf, p_min, p_max = enforce_mass_conservation(
        prob_arr=conv_pmf,
        expected_p_min=expected_p_min,
        expected_p_max=expected_p_max,
        bound_type=bound_type,
    )

    return DenseDiscreteDist(
        x_min=conv_x_min,
        step=width,
        prob_arr=conv_pmf,
        p_min=p_min,
        p_max=p_max,
        domain=dist_1.domain,
    ).truncate_edges(tail_truncation, bound_type)


def fft_self_convolve(
    *,
    dist: DenseDiscreteDist,
    T: int,
    tail_truncation: float,
    bound_type: BoundType,
    use_direct: bool,
) -> DenseDiscreteDist:
    """T-fold self-convolution via FFT with optional direct exponentiation path."""
    if not (isinstance(dist, DenseDiscreteDist) and dist.spacing_type == SpacingType.LINEAR):
        raise TypeError("fft_self_convolve requires DenseDiscreteDist input")

    if use_direct:
        try:
            return _fft_self_convolve_direct(
                dist=dist,
                T=T,
                tail_truncation=tail_truncation,
                bound_type=bound_type,
            )
        except MemoryError:
            warnings.warn(
                f"fft_self_convolve: direct method exceeded {MAX_FFT_BYTES / 1024**3:.0f} GB "
                f"memory limit for T={T}, pmf_size={dist.prob_arr.size:,}. "
                f"Falling back to binary self-convolution."
            )

    self_conv = binary_self_convolve(
        dist=dist,
        T=T,
        tail_truncation=tail_truncation,
        bound_type=bound_type,
        convolve=fft_convolve,
    )
    if not (
        isinstance(self_conv, DenseDiscreteDist) and self_conv.spacing_type == SpacingType.LINEAR
    ):
        raise TypeError(
            f"Expected DenseDiscreteDist from FFT self-convolution, got {type(self_conv)}"
        )
    return self_conv


def _fft_self_convolve_direct(
    *,
    dist: DenseDiscreteDist,
    T: int,
    tail_truncation: float,
    bound_type: BoundType,
) -> DenseDiscreteDist:
    # Budget split: the input tail_truncation is divided into three equal thirds.
    #   _calc_fft_window_size: Chernoff-based window determines the one-sided tail
    #          cutoff (right-tail for DOMINATES, folded-back mass bound for IS_DOMINATED).
    #   explicit opposite-side trim: left_tail_ind for DOMINATES (zeroes left bins,
    #          pushes mass to p_max) / right_tail_ind for IS_DOMINATED (zeroes right bins,
    #          pushes mass to p_min).
    #   final truncate_edges: trims actual near-zero edge bins the Chernoff window
    #          conservatively included on the remaining untrimmed side, reducing output
    #          bin count without sacrificing accuracy.
    # Total: 3 * (tail_truncation / 3) = tail_truncation
    tail_truncation /= 3

    finite_mass = math.fsum(map(float, dist.prob_arr))
    # The Chernoff window calculation expects a normalized finite PMF, so the
    # tail target must be rescaled when some mass already sits at infinity.
    normalized_pmf = dist.prob_arr / finite_mass
    tail_truncation_rescaled = tail_truncation / finite_mass

    shift_left, window_size = _calc_fft_window_size(
        pmf=normalized_pmf, num_convolutions=T, tail_truncation=tail_truncation_rescaled
    )

    fft_size = next_fast_len(max(window_size, dist.prob_arr.size))
    _check_fft_memory(fft_size, label=f"_fft_self_convolve_direct(T={T})")
    fft_data = rfft(dist.prob_arr, n=fft_size)
    fft_data **= T  # in-place power: avoids allocating a second complex buffer
    raw_conv = np.asarray(irfft(fft_data, n=fft_size, overwrite_x=True), dtype=np.float64)
    del fft_data  # free complex buffer
    raw_conv[raw_conv < 0] = 0.0
    # ``shift_left`` is the left edge of the retained convolution window. Rolling aligns
    # that window to index 0 so truncation logic can work in-place.
    rolled_conv = np.roll(raw_conv, -shift_left)

    conv_p_min, conv_p_max = self_convolve_boundary_masses(dist, num_convolutions=T)
    if bound_type == BoundType.DOMINATES:
        # For an upper bound, any dropped left-tail mass is pushed to +inf.
        cumsum = np.cumsum(rolled_conv)
        left_tail_ind = int(np.searchsorted(cumsum, tail_truncation, side="right"))
        shifted_mass = math.fsum(map(float, rolled_conv[:left_tail_ind]))
        rolled_conv[:left_tail_ind] = 0.0
        right_tail_mass = math.fsum(map(float, rolled_conv[window_size:]))
        conv_p_max += shifted_mass + right_tail_mass
    elif bound_type == BoundType.IS_DOMINATED:
        # For a lower bound, dropped right-tail mass moves to -inf, while any
        # overflow beyond the retained FFT window is folded onto the last kept
        # finite bin to preserve domination direction.
        cumsum = np.cumsum(rolled_conv[::-1])
        right_tail_ind = (
            rolled_conv.size - 1 - int(np.searchsorted(cumsum, tail_truncation, side="right"))
        )
        after_right_tail = right_tail_ind + 1
        shifted_mass = math.fsum(map(float, rolled_conv[after_right_tail:]))
        rolled_conv[after_right_tail:] = 0.0
        conv_p_min += shifted_mass

        right_tail_mass = math.fsum(map(float, rolled_conv[window_size:]))
        rolled_conv[min(window_size, right_tail_ind) - 1] += right_tail_mass
    else:
        raise ValueError(f"Unknown BoundType: {bound_type}")

    x_min = dist.x_min * T + shift_left * dist.step
    pmf_conv = rolled_conv[:window_size]
    pmf_conv, p_min_final, p_max_final = enforce_mass_conservation(
        prob_arr=pmf_conv,
        expected_p_min=conv_p_min,
        expected_p_max=conv_p_max,
        bound_type=bound_type,
    )

    return DenseDiscreteDist(
        x_min=x_min,
        step=dist.step,
        prob_arr=pmf_conv,
        p_min=p_min_final,
        p_max=p_max_final,
        domain=dist.domain,
    ).truncate_edges(tail_truncation, bound_type)


def _calc_fft_window_size(
    *, pmf: np.ndarray, num_convolutions: int, tail_truncation: float
) -> tuple[int, int]:
    """Calculate FFT window bounds for ``num_convolutions`` self-convolutions with fallback."""
    # ``compute_self_convolve_bounds`` gives a Chernoff-style window [lower, upper] that
    # should contain all but ``tail_truncation`` mass after ``num_convolutions`` convolutions.
    lower_idx, upper_idx = compute_self_convolve_bounds(pmf, num_convolutions, tail_truncation)
    window_size = upper_idx - lower_idx + 1

    if not 0 < window_size < float("inf"):
        lower_idx = 0
        n = len(pmf)
        # Fallback to the exact full-support FFT length when the bound becomes
        # numerically unusable for extreme truncation parameters.
        window_size = num_convolutions * (n - 1) + 1
        warnings.warn(
            "calc_fft_window_size: Chernoff bounds failed "
            f"(tail_truncation={tail_truncation:.3e}, num_convolutions={num_convolutions}). "
            f"Using fallback lower_idx=0, window_size={window_size:,} (n={n})."
        )

    return int(lower_idx), int(window_size)


def _check_fft_memory(fft_size: int, label: str = "FFT") -> None:
    """Raise MemoryError if an FFT of this size would exceed the safety limit.

    rfft produces complex128 output (~16 bytes per element) and the input is
    float64 (~8 bytes), so peak usage is roughly 24 * fft_size bytes.
    """
    estimated_bytes = 24 * fft_size
    if estimated_bytes > MAX_FFT_BYTES:
        raise MemoryError(
            f"{label}: estimated {estimated_bytes / 1024**3:.1f} GB for "
            f"fft_size={fft_size:,} exceeds safety limit of "
            f"{MAX_FFT_BYTES / 1024**3:.1f} GB. "
            f"Reduce grid size, increase loss_discretization, or raise MAX_FFT_BYTES."
        )

# Rounding tolerance for grid bin mapping — must stay at machine-epsilon scale
# to avoid misrouting mass between bins.
_GRID_ROUNDING_TOL = 10 * np.finfo(np.float64).eps

# =============================================================================
# PUBLIC API
# =============================================================================


def geometric_convolve(
    *,
    dist_1: DenseDiscreteDist,
    dist_2: DenseDiscreteDist,
    tail_truncation: float,
    bound_type: BoundType,
) -> DenseDiscreteDist:
    """Convolve two geometric-grid distributions.

    Algorithm 4 (`conv`) in Appendix C wrapper.
    For POSITIVES-domain distributions the 0 atom is neutral (not absorbing),
    so cross-terms (0 + finite and finite + 0) are added to the finite PMF.
    """
    # Input validation
    if not (
        isinstance(dist_1, DenseDiscreteDist)
        and dist_1.spacing_type == SpacingType.GEOMETRIC
        and dist_1.domain == Domain.POSITIVES
    ) or not (
        isinstance(dist_2, DenseDiscreteDist)
        and dist_2.spacing_type == SpacingType.GEOMETRIC
        and dist_2.domain == Domain.POSITIVES
    ):
        raise TypeError(
            "geometric_convolve requires geometric DenseDiscreteDist inputs on "
            f"Domain.POSITIVES; got dist_1={type(dist_1).__name__} "
            f"(spacing={dist_1.spacing_type}, domain={dist_1.domain}), "
            f"dist_2={type(dist_2).__name__} "
            f"(spacing={dist_2.spacing_type}, domain={dist_2.domain})"
        )
    if tail_truncation < 0:
        raise ValueError(f"tail_truncation must be non-negative, got {tail_truncation}")

    # Ensure both inputs share the same geometric log step.
    if not stable_isclose(a=dist_1.step, b=dist_2.step):
        raise ValueError(
            "Geometric log steps must match: "
            f"step_1={dist_1.step:.12g}, step_2={dist_2.step:.12g}"
        )
    geom_step = dist_1.step

    # Core Numeric Convolution
    x_out, pmf_conv = _compute_geometric_convolution(
        x1=dist_1.x_array,
        p1=dist_1.prob_arr,
        x2=dist_2.x_array,
        p2=dist_2.prob_arr,
        geom_step=geom_step,
        bound_type=bound_type,
    )

    # Add cross-terms from the 0 atom
    x_out_0 = float(x_out[0])
    pmf_conv = _add_single_zero_atom_cross_term(
        pmf_conv=pmf_conv,
        x_arr=dist_2.x_array,
        prob_arr=dist_2.prob_arr,
        zero_prob=dist_1.p_min,
        x_out_0=x_out_0,
        geom_step=geom_step,
        bound_type=bound_type,
    )
    pmf_conv = _add_single_zero_atom_cross_term(
        pmf_conv=pmf_conv,
        x_arr=dist_1.x_array,
        prob_arr=dist_1.prob_arr,
        zero_prob=dist_2.p_min,
        x_out_0=x_out_0,
        geom_step=geom_step,
        bound_type=bound_type,
    )

    expected_p_min, expected_p_max = convolve_boundary_masses(
        dist_1.p_min, dist_1.p_max, dist_2.p_min, dist_2.p_max, dist_1.domain
    )

    pmf_conv, p_min, p_max = enforce_mass_conservation(
        prob_arr=pmf_conv,
        expected_p_min=expected_p_min,
        expected_p_max=expected_p_max,
        bound_type=bound_type,
    )

    return DenseDiscreteDist(
        x_min=float(x_out[0]),
        step=geom_step,
        prob_arr=pmf_conv,
        p_min=p_min,
        p_max=p_max,
        spacing_type=SpacingType.GEOMETRIC,
        domain=Domain.POSITIVES,
    ).truncate_edges(tail_truncation, bound_type)


def geometric_self_convolve(
    *,
    dist: DenseDiscreteDist,
    T: int,
    tail_truncation: float,
    bound_type: BoundType,
) -> DenseDiscreteDist:
    """Self-convolve distribution T times using binary exponentiation."""
    # Input validation
    if not (isinstance(dist, DenseDiscreteDist) and dist.spacing_type == SpacingType.GEOMETRIC):
        raise TypeError(f"dist must be DenseDiscreteDist, got {type(dist)}")
    validate_bound_type(bound_type)
    if T < 1:
        raise ValueError(f"T must be >= 1, got {T}")
    if tail_truncation < 0:
        raise ValueError(f"tail_truncation must be non-negative, got {tail_truncation}")

    self_conv = binary_self_convolve(
        dist=dist,
        T=T,
        tail_truncation=tail_truncation,
        bound_type=bound_type,
        convolve=geometric_convolve,
    )
    if not (
        isinstance(self_conv, DenseDiscreteDist) and self_conv.spacing_type == SpacingType.GEOMETRIC
    ):
        raise TypeError(f"Expected DenseDiscreteDist from self-convolution, got {type(self_conv)}")
    return self_conv


# =============================================================================
# INTERNAL KERNEL IMPLEMENTATION
# =============================================================================


def _compute_geometric_convolution(
    *,
    x1: NDArray[np.float64],
    p1: NDArray[np.float64],
    x2: NDArray[np.float64],
    p2: NDArray[np.float64],
    geom_step: float,
    bound_type: BoundType,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Align grids, compute bin mapping parameters, and invoke the Numba kernel.

    Algorithm 4 (`conv`) with internal Algorithm 5 (`range-renorm`) in Appendix C.
    """
    # --- A. Standardization (Swap & Pad) ---
    # We normalize such that x_base (x1) starts at the lower value.
    # This ensures scale = x2[0]/x1[0] >= 1, simplifying log calculations.

    # 1. Swap if necessary so x1[0] <= x2[0]
    if x1[0] > x2[0]:
        x1, p1, x2, p2 = x2, p2, x1, p1

    # 2. Calculate Scale (Relative Offset)
    scale = x2[0] / x1[0]

    # 3. Equalize Lengths (Right-Padding)
    # The Numba kernel assumes arrays of equal length 'n'.
    target_n = max(x1.size, x2.size)
    if x1.size < target_n:
        x1, p1 = _pad_right_geometric(
            x=x1,
            p=p1,
            geom_step=geom_step,
            target_n=target_n,
        )
    elif x2.size < target_n:
        x2, p2 = _pad_right_geometric(
            x=x2,
            p=p2,
            geom_step=geom_step,
            target_n=target_n,
        )

    # Convert to float64 for Numba compatibility
    x_base = x1.astype(np.float64, copy=False)
    pmf_base = p1.astype(np.float64, copy=False)
    pmf_scaled = p2.astype(np.float64, copy=False)

    # --- B. Grid Mapping Parameters ---
    n = x_base.size

    # Edge case: Single point
    if n == 1:
        mass = pmf_base[0] * pmf_scaled[0]
        x_out = np.array([(scale + 1.0) * x_base[0]], dtype=np.float64)
        pmf_out = np.array([mass], dtype=np.float64)
        return x_out, pmf_out

    # Calculate shift parameters (delta)
    log_r = geom_step
    log_scale = np.log(scale)
    log_ap1 = np.log(scale + 1.0)

    # Vectorized calculation for d=1..n-1
    d_vec = np.arange(n, dtype=np.float64)
    log_r_d = d_vec * log_r

    log_lohi = np.logaddexp(0.0, log_scale + log_r_d)  # log(1 + scale*r^d)
    tau_lohi = (log_lohi - log_ap1) / log_r

    log_hilo = np.logaddexp(log_scale, log_r_d)  # log(scale + r^d)
    tau_hilo = (log_hilo - log_ap1) / log_r

    # Rounding strategy
    delta_lohi = np.zeros(n, dtype=np.int64)
    delta_hilo = np.zeros(n, dtype=np.int64)
    rounding_eps = _GRID_ROUNDING_TOL

    if bound_type == BoundType.DOMINATES:
        # Pessimistic: Round UP
        delta_lohi[1:] = np.ceil(tau_lohi[1:] - rounding_eps).astype(np.int64)
        delta_hilo[1:] = np.ceil(tau_hilo[1:] - rounding_eps).astype(np.int64)
    elif bound_type == BoundType.IS_DOMINATED:
        # Optimistic: Round DOWN
        delta_lohi[1:] = np.floor(tau_lohi[1:] + rounding_eps).astype(np.int64)
        delta_hilo[1:] = np.floor(tau_hilo[1:] + rounding_eps).astype(np.int64)
    else:
        raise ValueError(f"Unknown BoundType: {bound_type}")

    # --- C. Kernel Execution ---
    pmf_out = _numba_geometric_kernel(
        PMF_base=pmf_base,
        PMF_scaled=pmf_scaled,
        delta_lohi=delta_lohi,
        delta_hilo=delta_hilo,
    )

    # Construct output X grid: x_out = x_base * (1 + scale)
    x_out = x_base * (scale + 1.0)

    return x_out, pmf_out


def _pad_right_geometric(
    *,
    x: NDArray[np.float64],
    p: NDArray[np.float64],
    geom_step: float,
    target_n: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Extend grid to the right to reach target_n using geometric log step."""
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    n = x.size
    if n >= target_n:
        return x, p

    k = target_n - n
    tail = x[-1] * np.exp(geom_step * np.arange(1, k + 1, dtype=np.float64))

    x_ext = np.concatenate([x, tail])
    p_ext = np.pad(p, (0, k), mode="constant")
    return x_ext, p_ext


@njit(cache=True)
def _numba_geometric_kernel(
    *,
    PMF_base: NDArray[np.float64],
    PMF_scaled: NDArray[np.float64],
    delta_lohi: NDArray[np.int64],
    delta_hilo: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Core convolution loop.

    Calculates Z = X + Y by iterating over the difference 'd' between indices.

    """
    n = PMF_base.size
    pmf_out = np.zeros(n, dtype=np.float64)
    comp = np.zeros(n, dtype=np.float64)

    for i in range(n):
        mass = PMF_base[i] * PMF_scaled[i]
        y = mass - comp[i]
        t = pmf_out[i] + y
        comp[i] = (t - pmf_out[i]) - y
        pmf_out[i] = t

    for d in range(1, n):
        imax = n - d
        kshift1 = int(delta_lohi[d])
        kshift2 = int(delta_hilo[d])

        for i in range(imax):
            k1 = i + kshift1
            mass1 = PMF_base[i] * PMF_scaled[i + d]
            if 0 <= k1 < n:
                y = mass1 - comp[k1]
                t = pmf_out[k1] + y
                comp[k1] = (t - pmf_out[k1]) - y
                pmf_out[k1] = t

            k2 = i + kshift2
            mass2 = PMF_base[i + d] * PMF_scaled[i]
            if 0 <= k2 < n:
                y = mass2 - comp[k2]
                t = pmf_out[k2] + y
                comp[k2] = (t - pmf_out[k2]) - y
                pmf_out[k2] = t

    return pmf_out


def _add_single_zero_atom_cross_term(
    *,
    pmf_conv: NDArray[np.float64],
    x_arr: NDArray[np.float64],
    prob_arr: NDArray[np.float64],
    zero_prob: float,
    x_out_0: float,
    geom_step: float,
    bound_type: BoundType,
) -> NDArray[np.float64]:
    """Map one family of 0+finite cross-terms onto the fixed output grid."""
    if zero_prob == 0.0:
        return pmf_conv

    return _numba_add_single_zero_atom_cross_term(
        pmf_out=pmf_conv,
        x_vals=np.asarray(x_arr, dtype=np.float64),
        prob_arr=np.asarray(prob_arr, dtype=np.float64),
        zero_prob=float(zero_prob),
        x_out_0=float(x_out_0),
        log_r=float(geom_step),
        dominates=(bound_type == BoundType.DOMINATES),
    )


@njit(cache=True)
def _numba_add_single_zero_atom_cross_term(
    *,
    pmf_out: NDArray[np.float64],
    x_vals: NDArray[np.float64],
    prob_arr: NDArray[np.float64],
    zero_prob: float,
    x_out_0: float,
    log_r: float,
    dominates: bool,
) -> NDArray[np.float64]:
    """Core loop for one 0+finite cross-term family."""
    n = pmf_out.size

    for i in range(prob_arr.size):
        weight = prob_arr[i] * zero_prob
        if weight == 0.0:
            continue

        x = x_vals[i]
        if x <= 0.0:
            continue

        frac_k = math.log(x / x_out_0) / log_r
        if dominates:
            k = int(math.ceil(frac_k - _GRID_ROUNDING_TOL))
        else:
            k = int(math.floor(frac_k + _GRID_ROUNDING_TOL))

        if 0 <= k < n:
            pmf_out[k] += weight
        elif k < 0 and dominates:
            # Upper-bound rounding maps sub-grid mass to the first finite bin.
            pmf_out[0] += weight

    return pmf_out

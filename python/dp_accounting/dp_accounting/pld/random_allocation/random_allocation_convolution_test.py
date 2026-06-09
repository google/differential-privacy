"""Tests for binary self-convolution on linear and geometric grids."""

import math

import numpy as np
import pytest
from .random_allocation_convolution import (
    _geometric_kernel,
    _numba_geometric_kernel,
    _numpy_geometric_kernel,
    fft_convolve,
    fft_self_convolve,
    geometric_convolve,
)
from .random_allocation_distributions import DenseDiscreteDist, Domain
from .random_allocation_types import BoundType, SpacingType
from .random_allocation_utils import binary_self_convolve, log_geometric_to_linear
from . import random_allocation_types


def _linear_dist(n: int = 5) -> DenseDiscreteDist:
    x = np.linspace(0.0, 1.0, n)
    pmf = np.ones(n, dtype=np.float64) / n
    return DenseDiscreteDist.from_x_array(x_array=x, prob_arr=pmf)


def _geometric_dist(n: int = 6) -> DenseDiscreteDist:
    x = np.geomspace(0.1, 1.0, n)
    pmf = np.ones(n, dtype=np.float64) / n
    return DenseDiscreteDist.from_x_array(
        x_array=x,
        prob_arr=pmf,
        spacing_type=SpacingType.GEOMETRIC,
        domain=Domain.POSITIVES,
    )


def test_binary_self_convolve_rejects_invalid_t():
    dist = _linear_dist()
    with pytest.raises(ValueError, match="T must be >= 1"):
        binary_self_convolve(
            dist=dist,
            T=0,
            tail_truncation=0.0,
            bound_type=BoundType.DOMINATES,
            convolve=fft_convolve,
        )


def test_binary_self_convolve_t1_identity():
    dist = _linear_dist()
    result = binary_self_convolve(
        dist=dist, T=1, tail_truncation=0.0, bound_type=BoundType.DOMINATES, convolve=fft_convolve
    )
    assert np.allclose(result.x_array, dist.x_array)
    assert np.allclose(result.prob_arr, dist.prob_arr)


def test_binary_self_convolve_matches_direct_fft_t2():
    dist = _linear_dist()
    result = binary_self_convolve(
        dist=dist, T=2, tail_truncation=0.0, bound_type=BoundType.DOMINATES, convolve=fft_convolve
    )
    direct = fft_convolve(dist_1=dist, dist_2=dist, tail_truncation=0.0, bound_type=BoundType.DOMINATES)
    assert np.allclose(result.x_array, direct.x_array)
    assert np.allclose(result.prob_arr, direct.prob_arr, atol=1e-12)


def test_binary_self_convolve_matches_repeated_geometric():
    dist = _geometric_dist()
    result = binary_self_convolve(
        dist=dist,
        T=3,
        tail_truncation=0.0,
        bound_type=BoundType.DOMINATES,
        convolve=geometric_convolve,
    )
    repeated = geometric_convolve(
        dist_1=dist, dist_2=dist, tail_truncation=0.0, bound_type=BoundType.DOMINATES
    )
    repeated = geometric_convolve(
        dist_1=repeated, dist_2=dist, tail_truncation=0.0, bound_type=BoundType.DOMINATES
    )
    assert np.allclose(result.x_array, repeated.x_array)
    assert np.allclose(result.prob_arr, repeated.prob_arr, atol=1e-12)


def test_geometric_convolve_preserves_step_for_linear_round_trip():
    step = 1e-4
    dist = DenseDiscreteDist(
        x_min=1.0,
        step=step,
        prob_arr=np.array([0.5, 0.5], dtype=np.float64),
        spacing_type=SpacingType.GEOMETRIC,
        domain=Domain.POSITIVES,
    )

    result = geometric_convolve(
        dist_1=dist,
        dist_2=dist,
        tail_truncation=0.0,
        bound_type=BoundType.DOMINATES,
    )

    assert result.step == step
    assert log_geometric_to_linear(result).step == step


def test_binary_self_convolve_preserves_mass_fft():
    dist = _linear_dist()
    result = binary_self_convolve(
        dist=dist, T=5, tail_truncation=0.0, bound_type=BoundType.DOMINATES, convolve=fft_convolve
    )
    total = math.fsum([*map(float, result.prob_arr), result.p_min, result.p_max])
    assert np.isclose(total, 1.0, atol=1e-10)


def test_fft_self_convolve_direct_vs_binary():
    dist = _linear_dist(n=9)
    direct = fft_self_convolve(
        dist=dist, T=7, tail_truncation=0.0, bound_type=BoundType.DOMINATES, use_direct=True
    )
    binary = fft_self_convolve(
        dist=dist, T=7, tail_truncation=0.0, bound_type=BoundType.DOMINATES, use_direct=False
    )

    direct_mass = math.fsum([*map(float, direct.prob_arr), direct.p_min, direct.p_max])
    binary_mass = math.fsum([*map(float, binary.prob_arr), binary.p_min, binary.p_max])
    assert np.isclose(direct_mass, 1.0, atol=1e-10)
    assert np.isclose(binary_mass, 1.0, atol=1e-10)

    assert direct.x_array.size >= 2
    assert binary.x_array.size >= 2


def test_numpy_geometric_kernel_matches_numba_kernel():
    pmf_base = np.array([0.2, 0.0, 0.3, 0.1, 0.4], dtype=np.float64)
    pmf_scaled = np.array([0.1, 0.25, 0.0, 0.15, 0.5], dtype=np.float64)
    delta_lohi = np.array([0, -1, 0, 2, 8], dtype=np.int64)
    delta_hilo = np.array([0, 0, 1, -2, 3], dtype=np.int64)

    expected = _numba_geometric_kernel(
        PMF_base=pmf_base,
        PMF_scaled=pmf_scaled,
        delta_lohi=delta_lohi,
        delta_hilo=delta_hilo,
    )
    actual = _numpy_geometric_kernel(
        PMF_base=pmf_base,
        PMF_scaled=pmf_scaled,
        delta_lohi=delta_lohi,
        delta_hilo=delta_hilo,
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)


def test_geometric_kernel_uses_numpy_fallback_without_numba():
    pmf_base = np.array([0.2, 0.3, 0.5], dtype=np.float64)
    pmf_scaled = np.array([0.4, 0.1, 0.5], dtype=np.float64)
    delta_lohi = np.array([0, 0, 1], dtype=np.int64)
    delta_hilo = np.array([0, 1, -1], dtype=np.int64)

    original_has_numba = random_allocation_types._HAS_NUMBA
    try:
        random_allocation_types._HAS_NUMBA = False
        expected = _numpy_geometric_kernel(
            PMF_base=pmf_base,
            PMF_scaled=pmf_scaled,
            delta_lohi=delta_lohi,
            delta_hilo=delta_hilo,
        )
        actual = _geometric_kernel(
            PMF_base=pmf_base,
            PMF_scaled=pmf_scaled,
            delta_lohi=delta_lohi,
            delta_hilo=delta_hilo,
        )
    finally:
        random_allocation_types._HAS_NUMBA = original_has_numba

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)

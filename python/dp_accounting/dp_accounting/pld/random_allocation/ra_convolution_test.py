"""Tests for binary self-convolution on linear and geometric grids."""

import math

import numpy as np
from absl.testing import absltest
from dp_accounting.pld.random_allocation import ra_convolution
from dp_accounting.pld.random_allocation import ra_distributions
from dp_accounting.pld.random_allocation import ra_types
from dp_accounting.pld.random_allocation import ra_utils


def _linear_dist(n: int = 5) -> ra_distributions.DenseDiscreteDist:
    x = np.linspace(0.0, 1.0, n)
    pmf = np.ones(n, dtype=np.float64) / n
    return ra_distributions.DenseDiscreteDist.from_x_array(x_array=x, prob_arr=pmf)


def _geometric_dist(n: int = 6) -> ra_distributions.DenseDiscreteDist:
    x = np.geomspace(0.1, 1.0, n)
    pmf = np.ones(n, dtype=np.float64) / n
    return ra_distributions.DenseDiscreteDist.from_x_array(
        x_array=x,
        prob_arr=pmf,
        spacing_type=ra_types.SpacingType.GEOMETRIC,
        domain=ra_distributions.Domain.POSITIVES,
    )


class BinarySelfConvolveTest(absltest.TestCase):

    def test_rejects_invalid_t(self):
        dist = _linear_dist()
        with self.assertRaisesRegex(ValueError, "T must be >= 1"):
            ra_utils._binary_self_convolve(
                dist=dist,
                T=0,
                tail_truncation=0.0,
                bound_type=ra_types.BoundType.DOMINATES,
                convolve=ra_convolution._fft_convolve,
            )

    def test_t1_identity(self):
        dist = _linear_dist()
        result = ra_utils._binary_self_convolve(
            dist=dist,
            T=1,
            tail_truncation=0.0,
            bound_type=ra_types.BoundType.DOMINATES,
            convolve=ra_convolution._fft_convolve,
        )
        np.testing.assert_allclose(result._x_array, dist._x_array)
        np.testing.assert_allclose(result.prob_arr, dist.prob_arr)

    def test_matches_direct_fft_t2(self):
        dist = _linear_dist()
        result = ra_utils._binary_self_convolve(
            dist=dist,
            T=2,
            tail_truncation=0.0,
            bound_type=ra_types.BoundType.DOMINATES,
            convolve=ra_convolution._fft_convolve,
        )
        direct = ra_convolution._fft_convolve(
            dist_1=dist,
            dist_2=dist,
            tail_truncation=0.0,
            bound_type=ra_types.BoundType.DOMINATES,
        )
        np.testing.assert_allclose(result._x_array, direct._x_array)
        np.testing.assert_allclose(result.prob_arr, direct.prob_arr, atol=1e-12)

    def test_matches_repeated_geometric(self):
        dist = _geometric_dist()
        result = ra_utils._binary_self_convolve(
            dist=dist,
            T=3,
            tail_truncation=0.0,
            bound_type=ra_types.BoundType.DOMINATES,
            convolve=ra_convolution._geometric_convolve,
        )
        repeated = ra_convolution._geometric_convolve(
            dist_1=dist,
            dist_2=dist,
            tail_truncation=0.0,
            bound_type=ra_types.BoundType.DOMINATES,
        )
        repeated = ra_convolution._geometric_convolve(
            dist_1=repeated,
            dist_2=dist,
            tail_truncation=0.0,
            bound_type=ra_types.BoundType.DOMINATES,
        )
        np.testing.assert_allclose(result._x_array, repeated._x_array)
        np.testing.assert_allclose(result.prob_arr, repeated.prob_arr, atol=1e-12)

    def test_preserves_mass_fft(self):
        dist = _linear_dist()
        result = ra_utils._binary_self_convolve(
            dist=dist,
            T=5,
            tail_truncation=0.0,
            bound_type=ra_types.BoundType.DOMINATES,
            convolve=ra_convolution._fft_convolve,
        )
        total = math.fsum([*map(float, result.prob_arr), result.p_min, result.p_max])
        np.testing.assert_allclose(total, 1.0, atol=1e-10, rtol=0)


class GeometricConvolveTest(absltest.TestCase):

    def test_preserves_step_for_linear_round_trip(self):
        step = 1e-4
        dist = ra_distributions.DenseDiscreteDist(
            x_min=1.0,
            step=step,
            prob_arr=np.array([0.5, 0.5], dtype=np.float64),
            spacing_type=ra_types.SpacingType.GEOMETRIC,
            domain=ra_distributions.Domain.POSITIVES,
        )

        result = ra_convolution._geometric_convolve(
            dist_1=dist,
            dist_2=dist,
            tail_truncation=0.0,
            bound_type=ra_types.BoundType.DOMINATES,
        )

        self.assertEqual(result.step, step)
        self.assertEqual(ra_utils._log_geometric_to_linear(result).step, step)


class FftSelfConvolveTest(absltest.TestCase):

    def test_direct_vs_binary(self):
        dist = _linear_dist(n=9)
        direct = ra_convolution._fft_self_convolve(
            dist=dist,
            T=7,
            tail_truncation=0.0,
            bound_type=ra_types.BoundType.DOMINATES,
            use_direct=True,
        )
        binary = ra_convolution._fft_self_convolve(
            dist=dist,
            T=7,
            tail_truncation=0.0,
            bound_type=ra_types.BoundType.DOMINATES,
            use_direct=False,
        )

        direct_mass = math.fsum(
            [*map(float, direct.prob_arr), direct.p_min, direct.p_max]
        )
        binary_mass = math.fsum(
            [*map(float, binary.prob_arr), binary.p_min, binary.p_max]
        )
        np.testing.assert_allclose(direct_mass, 1.0, atol=1e-10, rtol=0)
        np.testing.assert_allclose(binary_mass, 1.0, atol=1e-10, rtol=0)

        self.assertGreaterEqual(direct._x_array.size, 2)
        self.assertGreaterEqual(binary._x_array.size, 2)


class GeometricKernelTest(absltest.TestCase):

    def test_numpy_kernel_matches_numba_kernel(self):
        pmf_base = np.array([0.2, 0.0, 0.3, 0.1, 0.4], dtype=np.float64)
        pmf_scaled = np.array([0.1, 0.25, 0.0, 0.15, 0.5], dtype=np.float64)
        delta_lohi = np.array([0, -1, 0, 2, 8], dtype=np.int64)
        delta_hilo = np.array([0, 0, 1, -2, 3], dtype=np.int64)

        expected = ra_convolution._numba_geometric_kernel(
            PMF_base=pmf_base,
            PMF_scaled=pmf_scaled,
            delta_lohi=delta_lohi,
            delta_hilo=delta_hilo,
        )
        actual = ra_convolution._numpy_geometric_kernel(
            PMF_base=pmf_base,
            PMF_scaled=pmf_scaled,
            delta_lohi=delta_lohi,
            delta_hilo=delta_hilo,
        )

        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)

    def test_uses_numpy_fallback_without_numba(self):
        pmf_base = np.array([0.2, 0.3, 0.5], dtype=np.float64)
        pmf_scaled = np.array([0.4, 0.1, 0.5], dtype=np.float64)
        delta_lohi = np.array([0, 0, 1], dtype=np.int64)
        delta_hilo = np.array([0, 1, -1], dtype=np.int64)

        original_has_numba = ra_types._HAS_NUMBA
        try:
            ra_types._HAS_NUMBA = False
            expected = ra_convolution._numpy_geometric_kernel(
                PMF_base=pmf_base,
                PMF_scaled=pmf_scaled,
                delta_lohi=delta_lohi,
                delta_hilo=delta_hilo,
            )
            actual = ra_convolution._geometric_kernel(
                PMF_base=pmf_base,
                PMF_scaled=pmf_scaled,
                delta_lohi=delta_lohi,
                delta_hilo=delta_hilo,
            )
        finally:
            ra_types._HAS_NUMBA = original_has_numba

        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    absltest.main()

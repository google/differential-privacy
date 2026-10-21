from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from dp_accounting import dp_event
from dp_accounting import gaussian_mechanism
from dp_accounting.pld import pld_privacy_accountant


class GaussianTest(parameterized.TestCase):

  @parameterized.product(
      eps=[1e-5, 1e-3, 0.1, 1.0, 10.0, 1e3, 1e5],
      delta=[0.0, 1e-10, 1e-5, 0.1, 0.5, 0.9, 1.0],
  )
  def test_get_epsilon_gaussian_inverts_get_sigma_gaussian(self, eps, delta):
    sigma = gaussian_mechanism.get_sigma_gaussian(eps, delta, tol=1e-15)
    recovered_eps = gaussian_mechanism.get_epsilon_gaussian(
        sigma, delta, tol=1e-15
    )
    np.testing.assert_allclose(recovered_eps, eps if (0 < delta < 1) else 0)

  @parameterized.product(
      sigma=[1e-5, 1e-3, 0.1, 1.0, 10.0, 1e3, 1e5],
      delta=[0.0, 1e-10, 1e-5, 0.1, 0.5, 0.9, 1.0],
  )
  def test_get_sigma_gaussian_inverts_get_epsilon_gaussian(self, sigma, delta):
    eps = gaussian_mechanism.get_epsilon_gaussian(sigma, delta)
    recovered_sigma = gaussian_mechanism.get_sigma_gaussian(eps, delta)
    if eps == 0:
      # We have (0, delta)-DP at the original noise level. get_sigma_gaussian
      # will return the smallest sigma consistent with (0, delta)-DP.
      self.assertLessEqual(recovered_sigma, sigma)
    else:
      np.testing.assert_allclose(recovered_sigma, sigma if delta > 0 else 0)

  @parameterized.product(sigma=[0.5, 1, 3], delta=[1e-1, 1e-2, 1e-3])
  def test_get_epsilon_gaussian_pld(self, sigma, delta):
    expected_eps = (
        pld_privacy_accountant.PLDAccountant()
        .compose(dp_event.GaussianDpEvent(sigma))
        .get_epsilon(delta)
    )
    np.testing.assert_allclose(
        gaussian_mechanism.get_epsilon_gaussian(sigma, delta), expected_eps
    )

  @parameterized.product(
      eps=[0.0, 1e-5, 1e-3, 0.1, 1.0, 10.0, 1e3, 1e5, np.inf],
      delta=[0.0, 1e-10, 1e-5, 0.1, 0.5, 0.9, 1.0],
  )
  def test_get_delta_gaussian_inverts_get_sigma_gaussian(self, eps, delta):
    sigma = gaussian_mechanism.get_sigma_gaussian(eps, delta, tol=1e-15)
    recovered_delta = gaussian_mechanism.get_delta_gaussian(eps, sigma)
    np.testing.assert_allclose(
        recovered_delta, delta if eps < np.inf else 0, atol=1e-12
    )

  @parameterized.product(
      sigma=[1e-5, 1e-3, 0.1, 1.0, 10.0, 1e3, 1e5, np.inf],
      delta=[0.0, 1e-10, 1e-5, 0.1, 0.5, 0.9, 1.0],
  )
  def test_get_delta_gaussian_inverts_get_epsilon_gaussian(self, sigma, delta):
    eps = gaussian_mechanism.get_epsilon_gaussian(sigma, delta, tol=1e-15)
    recovered_delta = gaussian_mechanism.get_delta_gaussian(eps, sigma)
    if eps == 0:
      # If we calibrated to epsilon = 0, delta is not necessarily the minimum
      # achieving epsilon = 0, but recovered_delta should be.
      self.assertLessEqual(recovered_delta, delta)
    else:
      np.testing.assert_allclose(recovered_delta, delta)

  @parameterized.product(
      eps=[0.0, 1e-3, 0.1, 1.0, np.inf],
      sigma=[0.5, 1.0, 3.0],
  )
  def test_get_epsilon_gaussian_inverts_get_delta_gaussian(self, eps, sigma):
    delta = gaussian_mechanism.get_delta_gaussian(eps, sigma)
    recovered_eps = gaussian_mechanism.get_epsilon_gaussian(
        sigma, delta, tol=1e-15
    )
    np.testing.assert_allclose(recovered_eps, eps, atol=1e-12)

  @parameterized.product(
      eps=[0.0, 1e-3, 0.1, 1.0, np.inf],
      sigma=[0.0, 0.5, 1.0, 3.0, np.inf],
  )
  def test_get_sigma_gaussian_inverts_get_delta_gaussian(self, eps, sigma):
    delta = gaussian_mechanism.get_delta_gaussian(eps, sigma)
    recovered_sigma = gaussian_mechanism.get_sigma_gaussian(
        eps, delta, tol=1e-15
    )
    np.testing.assert_allclose(recovered_sigma, sigma if eps < np.inf else 0)

  @parameterized.product(sigma=[0.5, 1, 3], eps=[0.0, 0.1, 0.5, 1.0])
  def test_get_delta_gaussian_pld(self, sigma, eps):
    expected_delta = (
        pld_privacy_accountant.PLDAccountant()
        .compose(dp_event.GaussianDpEvent(sigma))
        .get_delta(eps)
    )
    np.testing.assert_allclose(
        gaussian_mechanism.get_delta_gaussian(eps, sigma), expected_delta
    )

  @parameterized.parameters(
      (0.0, 0.0, 1.0),
      (1.0, 0.0, 1.0),
      (0.0, np.inf, 0.0),
      (1.0, np.inf, 0.0),
      (np.inf, 0.0, 0.0),
      (np.inf, 1.0, 0.0),
      (np.inf, np.inf, 0.0),
  )
  def test_get_delta_gaussian_boundary(self, eps, sigma, expected_delta):
    self.assertEqual(
        gaussian_mechanism.get_delta_gaussian(eps, sigma), expected_delta
    )

  @parameterized.parameters(
      (-1.0, 1.0, 'epsilon must be non-negative'),
      (1.0, -1.0, 'sigma must be non-negative'),
  )
  def test_get_delta_gaussian_invalid_args(self, eps, sigma, error_msg):
    with self.assertRaisesRegex(ValueError, error_msg):
      gaussian_mechanism.get_delta_gaussian(eps, sigma)


if __name__ == '__main__':
  absltest.main()

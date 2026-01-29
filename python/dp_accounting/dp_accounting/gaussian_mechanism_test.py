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


if __name__ == '__main__':
  absltest.main()

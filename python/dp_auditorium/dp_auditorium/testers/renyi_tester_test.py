# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for RenyiTester."""

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from dp_auditorium.configs import privacy_property
from dp_auditorium.configs import property_tester_config
from dp_auditorium.testers import renyi_tester


def _compute_error_param(error: float) -> float:
  """Inverse error parameter."""
  return (np.exp(error) - 1) / (np.exp(error) + 1)


def _compute_renyi_tester_num_samples_from_error(
    alpha: float,
    nn_model_function_bound: float,
    failure_probability: float,
    error: float,
) -> int:
  """Computes number of samples required to achieve given error."""
  gamma = _compute_error_param(error)
  num_samples_1 = (
      3
      * np.exp(2 * (alpha - 1) * nn_model_function_bound)
      * np.log(2 / failure_probability)
      / (gamma**2)
  )
  num_samples_2 = (
      2
      * np.exp(2 * alpha * nn_model_function_bound)
      * np.log(2 / failure_probability)
      / (gamma**2)
  )
  num_samples = int(max(num_samples_1, num_samples_2))

  return num_samples


def _renyi_laplace(
    mu1: float, mu2: float, lam1: float, lam2: float, alpha: float
):
  """Compute Renyi divergence for two Laplace distributions.

  Expression taken from
  https://mast.queensu.ca/~communications/Papers/GiAlLi13.pdf.

  Args:
    mu1: mean of first Laplace distribution.
    mu2: mean of second Laplace distribution.
    lam1: scale of first Laplace distribution.
    lam2: scale of second Laplace distribution.
    alpha: Renyi divergence parameter.

  Returns:
    Renyi divergence between Laplace(mu1,lam1) and Laplace(mu2,lam2).

  Raises:
    ValueError if `alpha != lam1 / (lam1 + lam2)` and
    `alpha * lam2 + (1 - alpha) * lam1 <= 0`.
  """
  t1 = np.log(lam2 / lam1)
  if alpha == lam1 / (lam1 + lam2):
    t2 = np.abs(mu1 - mu2) / lam2
    m = (lam1 + lam2) / lam2
    t3 = m * np.log(2 * lam1 / (lam1 + lam2 + np.abs(mu1 - mu2)))
    divergence = t1 + t2 + t3
  else:
    if alpha * lam2 + (1 - alpha) * lam1 > 0:
      e1 = -(1 - alpha) * np.abs(mu1 - mu2) / lam2
      e2 = -alpha * np.abs(mu1 - mu2) / lam1
      g_term = alpha * np.exp(e1) / lam1 - (1 - alpha) * np.exp(e2) / lam2
      log_num = lam1 * lam2 * lam2 * g_term
      log_den = (alpha**2 * lam2**2) - ((1 - alpha) ** 2 * lam1**2)
      t2 = np.log(log_num / log_den) / (alpha - 1)
      divergence = t1 + t2
    else:
      raise ValueError('Divergence not defined')
  return divergence


class RenyiTesterTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()

    tf.keras.utils.set_random_seed(12345)
    self.rng = np.random.default_rng(12345)
    epsilon = 0.1
    self.privacy_property = privacy_property.PrivacyProperty(
        pure_dp=privacy_property.PureDp(epsilon=epsilon)
    )
    self.training_config = property_tester_config.TrainingConfig(
        training_epochs=5,
        batch_size=128,
        optimizer_learning_rate=0.001,
        model_output_coordinate_bound=16 * epsilon,
    )
    self.renyi_tester_config = property_tester_config.RenyiPropertyTesterConfig(
        training_config=self.training_config,
        privacy_property=self.privacy_property,
        alpha=3.0,
    )
    self.renyi_tester = renyi_tester.RenyiPropertyTester(
        self.renyi_tester_config,
        base_model=renyi_tester.make_default_renyi_base_model(),
    )

  @parameterized.parameters(1.1, 1.5)
  def test_returns_lower_bound_gaussian(self, alpha: float):
    """Test estimated divergence on x and y lower bounds the expected divergence."""

    num_samples = 100000
    mu = 2.71
    sigma = 3.1415
    x = self.rng.normal(0, sigma, (num_samples, 1))
    y = self.rng.normal(mu, sigma, (num_samples, 1))

    expected_divergence = (alpha * mu**2) / (2 * sigma * sigma)

    x_test = self.rng.normal(0, sigma, (num_samples, 1))
    y_test = self.rng.normal(mu, sigma, (num_samples, 1))
    self.renyi_tester_config.alpha = alpha
    tester = renyi_tester.RenyiPropertyTester(
        self.renyi_tester_config,
        base_model=renyi_tester.make_default_renyi_base_model(),
    )

    model = tester._get_optimized_divergence_estimation_model(x, y)
    divergence_test = tester._compute_divergence_on_samples(
        model,
        x_test,
        y_test,
        failure_probability=0.1,
    )
    logging.info('Result divergence test: %.3f', divergence_test)
    logging.info('Expected divergence: %.3f', expected_divergence)
    self.assertLess(divergence_test, expected_divergence)

  @parameterized.parameters(1.1, 1.5)
  def test_returns_lower_bound_uniform(self, alpha: float):
    """Test estimated divergence on x and y lower bounds the expected divergence."""

    num_samples = 100000

    low_1 = 0.5
    low_2 = 0.1
    high_1 = 1.3
    high_2 = 2.1
    x = self.rng.uniform(low_1, high_1, (num_samples, 1))
    y = self.rng.uniform(low_2, high_2, (num_samples, 1))

    # Divergence between two Gaussians borrowed from
    # https://mast.queensu.ca/~communications/Papers/GiAlLi13.pdf
    expected_divergence = np.log((high_2 - low_2) / (high_1 - low_1))

    x_test = self.rng.uniform(low_1, high_1, (num_samples, 1))
    y_test = self.rng.uniform(low_2, high_2, (num_samples, 1))
    self.renyi_tester_config.alpha = alpha

    tester = renyi_tester.RenyiPropertyTester(
        self.renyi_tester_config,
        base_model=renyi_tester.make_default_renyi_base_model(),
    )

    model = tester._get_optimized_divergence_estimation_model(x, y)
    divergence_test = tester._compute_divergence_on_samples(
        model,
        x_test,
        y_test,
        failure_probability=0.1,
    )
    logging.info('Result divergence test: %.3f', divergence_test)
    logging.info('Expected divergence: %.3f', expected_divergence)
    self.assertLess(divergence_test, expected_divergence)

  @parameterized.parameters(1.1, 1.5)
  def test_returns_lower_bound_exponential(self, alpha: float):
    """Test estimated divergence on x and y lower bounds the expected divergence."""

    num_samples = 100000

    lambda_1 = 7.0
    lambda_2 = 3.0

    x = self.rng.exponential(lambda_1, (num_samples, 1))
    y = self.rng.exponential(lambda_2, (num_samples, 1))

    # Divergence between two uniforms borrowed from
    # https://mast.queensu.ca/~communications/Papers/GiAlLi13.pdf
    lambda_alpha = alpha * lambda_1 + (1 - alpha) * lambda_2
    expected_divergence = (
        np.log(lambda_1 / lambda_2)
        + np.log(lambda_1 / lambda_alpha) / lambda_alpha
    )

    x_test = self.rng.exponential(lambda_1, (num_samples, 1))
    y_test = self.rng.exponential(lambda_2, (num_samples, 1))

    self.renyi_tester_config.alpha = alpha
    tester = renyi_tester.RenyiPropertyTester(
        self.renyi_tester_config,
        base_model=renyi_tester.make_default_renyi_base_model(),
    )

    model = tester._get_optimized_divergence_estimation_model(x, y)
    divergence_test = tester._compute_divergence_on_samples(
        model,
        x_test,
        y_test,
        failure_probability=0.1,
    )
    logging.info('Result divergence test: %.3f', divergence_test)
    logging.info('Expected divergence: %.3f', expected_divergence)
    self.assertLess(divergence_test, expected_divergence)

  @parameterized.parameters(1.1, 1.5)
  def test_returns_lower_bound_laplace(self, alpha: float):
    """Test estimated divergence on x and y lower bounds the expected divergence."""

    num_samples = 100000

    mu_1 = 0.0
    mu_2 = 1.1
    scale_1 = 1.3
    scale_2 = 2.1
    x = self.rng.laplace(mu_1, scale_1, (num_samples, 1))
    y = self.rng.laplace(mu_2, scale_2, (num_samples, 1))

    # Divergence between two uniforms borrowed from
    # https://mast.queensu.ca/~communications/Papers/GiAlLi13.pdf
    expected_divergence = _renyi_laplace(mu_1, mu_2, scale_1, scale_2, alpha)

    x_test = self.rng.laplace(mu_1, scale_1, (num_samples, 1))
    y_test = self.rng.laplace(mu_2, scale_2, (num_samples, 1))
    self.renyi_tester_config.alpha = alpha
    tester = renyi_tester.RenyiPropertyTester(
        self.renyi_tester_config,
        base_model=renyi_tester.make_default_renyi_base_model(),
    )

    model = tester._get_optimized_divergence_estimation_model(x, y)
    divergence_test = tester._compute_divergence_on_samples(
        model,
        x_test,
        y_test,
        failure_probability=0.1,
    )
    logging.info('Result divergence test: %.3f', divergence_test)
    logging.info('Expected divergence: %.3f', expected_divergence)
    self.assertLess(divergence_test, expected_divergence)


class RenyiTesterUtilsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.training_config = property_tester_config.TrainingConfig(
        training_epochs=1,
        batch_size=4,
        optimizer_learning_rate=0.01,
        model_output_coordinate_bound=1.5,
    )
    self.privacy_property = privacy_property.PrivacyProperty(
        renyi_dp=privacy_property.RenyiDp(epsilon=0.1, alpha=3.0)
    )

  def test_renyi_model_parameters_initializer_wrong_property(self):
    approx_dp_privacy_property = privacy_property.PrivacyProperty(
        approximate_dp=privacy_property.ApproximateDp(epsilon=0.1, delta=0.01)
    )
    renyi_tester_config = property_tester_config.RenyiPropertyTesterConfig(
        alpha=1.0,
        training_config=self.training_config,
        privacy_property=approx_dp_privacy_property,
    )
    with self.assertRaises(ValueError):
      _ = renyi_tester.RenyiPropertyTester(
          config=renyi_tester_config,
          base_model=renyi_tester.make_default_renyi_base_model(),
      )

  @parameterized.parameters(
      (
          privacy_property.PrivacyProperty(
              pure_dp=privacy_property.PureDp(epsilon=0.1)
          ),
          2.0,
          0.04,
      ),
      (
          privacy_property.PrivacyProperty(
              pure_dp=privacy_property.PureDp(epsilon=0.1)
          ),
          6.0,
          0.1,
      ),
      (
          privacy_property.PrivacyProperty(
              renyi_dp=privacy_property.RenyiDp(epsilon=0.1, alpha=2.0)
          ),
          2.0,
          0.1,
      ),
  )
  def test_renyi_model_parameters_initializer_sets_params(
      self, tested_property, alpha, threshold
  ):
    renyi_tester_config = property_tester_config.RenyiPropertyTesterConfig(
        alpha=alpha,
        training_config=self.training_config,
        privacy_property=tested_property,
    )
    tester = renyi_tester.RenyiPropertyTester(
        config=renyi_tester_config,
        base_model=renyi_tester.make_default_renyi_base_model(),
    )
    self.assertAlmostEqual(tester._test_threshold, threshold, places=6)
    self.assertAlmostEqual(tester._alpha, alpha, places=6)

  def test_computes_error_from_samples(self):
    alpha = 2
    bound = 1.0
    failure_probability = 0.05
    expected_error = 0.1
    num_samples = _compute_renyi_tester_num_samples_from_error(
        alpha=alpha,
        nn_model_function_bound=bound,
        failure_probability=failure_probability,
        error=expected_error,
    )

    result_error = renyi_tester._compute_error_from_samples(
        num_samples=num_samples,
        failure_probability=failure_probability,
        model_output_coordinate_bound=bound,
        alpha=alpha,
    )
    self.assertAlmostEqual(result_error, expected_error, places=6)


if __name__ == '__main__':
  absltest.main()

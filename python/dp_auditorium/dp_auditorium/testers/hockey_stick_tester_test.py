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
"""Test for hockey stick divergence estimator."""

from absl.testing import absltest
import numpy as np
from tensorflow import keras
from dp_auditorium.configs import privacy_property
from dp_auditorium.configs import property_tester_config as config
from dp_auditorium.testers import hockey_stick_tester as hst


_SEED = 123456
_RNG = np.random.default_rng(seed=_SEED)
_ESTIMATION_RANGE_BOUND = 1.0


class HockeyStickDivergenceTest(absltest.TestCase):

  def test_confidence_bound(self):
    n_samples = 100
    n_experiments = 1000
    cb = hst._get_accuracy_confidence_bound(
        _ESTIMATION_RANGE_BOUND, n_samples, confidence=0.95
    )
    # Generate 1000 sums of bernoulli random variables.
    sample = _RNG.binomial(n_samples, 0.3, n_experiments)
    errors = np.abs(sample / n_samples - 0.3)
    errors_above_cb = np.sum(errors > cb)
    self.assertLess(errors_above_cb, 0.05 * n_experiments)
    self.assertGreater(errors_above_cb, 1)

  def dummy_mechanism(self, data: np.ndarray, n_samples: int) -> np.ndarray:
    if data[0] == -1:
      return _RNG.uniform(-1, -0.1, (n_samples, 2))
    else:
      return _RNG.uniform(0.1, 1, (n_samples, 2))

  def make_privacy_property(self, epsilon: float, delta: float):
    return privacy_property.ApproximateDp(epsilon=epsilon, delta=delta)

  def test_get_model_inputs(self):
    model = keras.Sequential([keras.layers.Dense(1)])
    training_config = hst.make_default_hs_training_config()
    hs_config = config.HockeyStickPropertyTesterConfig(
        approximate_dp=self.make_privacy_property(epsilon=1.0, delta=0.001),
        training_config=training_config, evaluation_batch_size=1000,
    )
    div_estimator = hst.HockeyStickPropertyTester(
        config=hs_config, base_model=model
    )
    samples1 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    samples2 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    features, labels = div_estimator._generate_inputs_to_model(
        samples1, samples2
    )
    expected_features = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1, 2])
    expected_features = expected_features.reshape((-1, 1))
    expected_labels = np.array([1, 1, 1, 1, 1, 1, 0, 0])
    expected_labels = expected_labels.reshape((-1, 1))

    features_sort_ix = features[:, 0].argsort()

    np.testing.assert_array_equal(
        features[features_sort_ix, ...], expected_features
    )
    np.testing.assert_array_equal(
        labels[features_sort_ix, ...], expected_labels
    )

  def test_get_model_inputs_higher_dim(self):
    model = keras.Sequential([keras.layers.Dense(1)])
    training_config = hst.make_default_hs_training_config()
    hs_config = config.HockeyStickPropertyTesterConfig(
        approximate_dp=self.make_privacy_property(epsilon=1.0, delta=0.001),
        training_config=training_config, evaluation_batch_size=1000,
    )
    div_estimator = hst.HockeyStickPropertyTester(
        config=hs_config, base_model=model
    )
    samples1 = np.array(
        [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8]]
    )
    samples2 = np.array([
        [0.1, 0.1],
        [0.2, 0.2],
        [0.3, 0.3],
        [0.4, 0.4],
        [0.5, 0.5],
        [0.6, 0.6],
        [0.7, 0.7],
        [0.8, 0.8],
    ])
    features, labels = div_estimator._generate_inputs_to_model(
        samples1, samples2
    )
    expected_features = np.array([
        [0.3, 0.3],
        [0.4, 0.4],
        [0.5, 0.5],
        [0.6, 0.6],
        [0.7, 0.7],
        [0.8, 0.8],
        [1, 1],
        [2, 2],
    ])

    expected_labels = np.array([1, 1, 1, 1, 1, 1, 0, 0])
    expected_labels = expected_labels.reshape((-1, 1))

    features_sort_ix = features[:, 0].argsort()
    np.testing.assert_array_equal(
        features[features_sort_ix, ...], expected_features
    )
    np.testing.assert_array_equal(
        labels[features_sort_ix, ...], expected_labels
    )

  def test_compute_divergence_on_samples(self):
    data1 = np.array([1])
    data2 = np.array([-1])
    samples1_train = self.dummy_mechanism(data1, 2000)
    samples2_train = self.dummy_mechanism(data2, 2000)

    samples1_test = self.dummy_mechanism(data1, 2000)
    samples2_test = self.dummy_mechanism(data2, 2000)
    model = keras.Sequential([keras.layers.Dense(1, use_bias=False)])
    training_options = hst.make_default_hs_training_config()
    training_options.training_epochs = 1000
    hs_config = config.HockeyStickPropertyTesterConfig(
        approximate_dp=self.make_privacy_property(epsilon=0.5, delta=0.1),
        training_config=training_options, evaluation_batch_size=1000,
    )
    div_estimator = hst.HockeyStickPropertyTester(
        config=hs_config, base_model=model
    )

    model = div_estimator._get_optimized_divergence_estimation_model(
        samples1_train, samples2_train
    )
    div = div_estimator._compute_divergence_on_samples(
        model, samples1_test, samples2_test, 0.05
    )
    self.assertGreater(div, 0.5)

  def laplace_mechanism(self, x, n_samples):
    return _RNG.laplace(0, 1.0, n_samples) + x

  def test_compute_divergence_private_mechanism(self):
    data1 = np.array([1])
    data2 = np.array([0])
    model = keras.Sequential([keras.layers.Dense(1)])
    samples1 = self.laplace_mechanism(data1, 2000)
    samples2 = self.laplace_mechanism(data2, 2000)
    training_config = hst.make_default_hs_training_config()
    training_config.training_epochs = 1000
    hs_config = config.HockeyStickPropertyTesterConfig(
        approximate_dp=self.make_privacy_property(epsilon=1.0, delta=0.0),
        training_config=training_config, evaluation_batch_size=1000,
    )
    div_estimator = hst.HockeyStickPropertyTester(
        config=hs_config, base_model=model
    )
    model = div_estimator._get_optimized_divergence_estimation_model(
        samples1[:1000], samples2[:1000]
    )
    div = div_estimator._compute_divergence_on_samples(
        model, samples1, samples2, 0.05
    )

    self.assertLess(div, 0.0)

  def mechanism_with_different_output_shapes(self, data, n_samples):
    if data[0] == 0:
      return np.ones((n_samples, 2))
    if data[0] == 1:
      return np.ones((n_samples, 3, 4))

  def test_fails_on_mechanism_with_different_output_shapes(self):
    model = keras.Sequential([keras.layers.Dense(1)])
    training_options = hst.make_default_hs_training_config()
    hs_config = config.HockeyStickPropertyTesterConfig(
        approximate_dp=self.make_privacy_property(epsilon=1.0, delta=0.1),
        training_config=training_options, evaluation_batch_size=1000,
    )
    div_estimator = hst.HockeyStickPropertyTester(
        config=hs_config, base_model=model
    )
    samples1 = self.mechanism_with_different_output_shapes(np.array([0]), 100)
    samples2 = self.mechanism_with_different_output_shapes(np.array([1]), 100)
    with self.assertRaises(ValueError) as context:
      div_estimator._generate_inputs_to_model(samples1, samples2)
    self.assertIn("rank", str(context.exception))

  def private_mechanism_for_testing(self, data, n_samples):
    del data
    return _RNG.laplace(0, 1.0, n_samples)

  def non_private_mechanism_for_testing(self, data, n_samples):
    if len(data) == 1:
      return np.zeros(n_samples)
    return np.ones(n_samples)

  def test_hockey_stick_privacy_tester(self):
    model = keras.Sequential([keras.layers.Dense(1, use_bias=True)])
    epsilon = 1.0
    delta = 0.0
    training_options = hst.make_default_hs_training_config()
    training_options.training_epochs = 1000
    hs_config = config.HockeyStickPropertyTesterConfig(
        approximate_dp=self.make_privacy_property(epsilon=epsilon, delta=delta),
        training_config=training_options, evaluation_batch_size=1000,
    )
    hsdt = hst.HockeyStickPropertyTester(config=hs_config, base_model=model)

    samples1 = self.private_mechanism_for_testing(np.array([0, 1]), 200)
    samples2 = self.private_mechanism_for_testing(np.array([0]), 200)

    failure_probability = 0.05
    divergence = hsdt.estimate_lower_bound(
        samples1, samples2, failure_probability
    )
    with self.subTest("divergence_less_than_threshold"):
      self.assertLess(divergence, 0.0)
    found_privacy_violation = hsdt.reject_property(divergence)
    with self.subTest("found_privacy_violation"):
      self.assertFalse(found_privacy_violation)

  def test_hockey_stick_non_private_mechanism(self):
    model = keras.Sequential([keras.layers.Dense(1, use_bias=True)])
    epsilon = 1.0
    delta = 0.0
    training_options = hst.make_default_hs_training_config()
    training_options.training_epochs = 1000
    hs_config = config.HockeyStickPropertyTesterConfig(
        approximate_dp=self.make_privacy_property(epsilon=epsilon, delta=delta),
        training_config=training_options, evaluation_batch_size=1000,
    )
    hsdt = hst.HockeyStickPropertyTester(config=hs_config, base_model=model)
    samples1 = self.non_private_mechanism_for_testing(np.array([0, 1]), 200)
    samples2 = self.non_private_mechanism_for_testing(np.array([0]), 200)
    divergence = hsdt.estimate_lower_bound(samples1, samples2, 0.05)
    with self.subTest("divergence_greater_than_threshold"):
      self.assertLess(0.5, divergence)

    found_privacy_violation = hsdt.reject_property(divergence)
    with self.subTest("found_privacy_violation"):
      self.assertTrue(found_privacy_violation)

  def test_assert_privacy_violation(self):
    training_options = hst.make_default_hs_training_config()
    hs_config = config.HockeyStickPropertyTesterConfig(
        approximate_dp=self.make_privacy_property(epsilon=1.0, delta=0.1),
        training_config=training_options, evaluation_batch_size=1000,
    )
    hsdt = hst.HockeyStickPropertyTester(
        config=hs_config,
        base_model=keras.Sequential([keras.layers.Dense(1)]),
    )
    self.assertTrue(hsdt.reject_property(0.2))
    self.assertTrue(hsdt.reject_property(0.11))
    self.assertFalse(hsdt.reject_property(0.09))

  def test_privacy_property(self):
    training_options = hst.make_default_hs_training_config()
    hs_config = config.HockeyStickPropertyTesterConfig(
        approximate_dp=self.make_privacy_property(epsilon=1.0, delta=0.1),
        training_config=training_options, evaluation_batch_size=1000,
    )
    hs_tester = hst.HockeyStickPropertyTester(
        config=hs_config, base_model=hst.make_default_hs_base_model()
    )
    self.assertEqual(
        hs_config.approximate_dp, hs_tester.privacy_property.approximate_dp
    )


if __name__ == "__main__":
  absltest.main()

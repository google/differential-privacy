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
"""Tests for main PrivacyTestRunner class."""

from typing import Any

from absl.testing import absltest
import numpy as np
import tensorflow as tf

from dp_auditorium import interfaces
from dp_auditorium import privacy_test_runner
from dp_auditorium.configs import privacy_property
from dp_auditorium.configs import privacy_test_runner_config

_LOWER_BOUND = 0.0


def _stub_dataset_generator(_):
  return np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0])


def _stub_mechanism(_, num_samples):
  return np.ones(num_samples)


class StubPropertyTester(interfaces.PropertyTester):

  def __init__(self, rejects_property: bool):
    self.rejects_property = rejects_property

  @property
  def privacy_property(self) -> privacy_property.PrivacyProperty:
    return privacy_property.PrivacyProperty(
        pure_dp=privacy_property.PureDp(epsilon=1.0)
    )

  def estimate_lower_bound(
      self,
      samples1: np.ndarray[Any, Any],
      samples2: np.ndarray[Any, Any],
      failure_probability: float,
  ) -> float:
    return _LOWER_BOUND

  def reject_property(self, lower_bound: float) -> bool:
    return self.rejects_property


class PrivacyTestRunnerTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.mechanism_name = "stub_mechanism"

    self.runner_config = privacy_test_runner_config.PrivacyTestRunnerConfig(
        property_tester=privacy_test_runner_config.PropertyTester.UNSPECIFIED_TESTER,
        max_num_trials=3,
        failure_probability=0.1,
        num_samples=10,
        post_processing=privacy_test_runner_config.PostProcessing.NONE,
    )

  def test_test_privacy_private_mechanism(self):
    stub_private_mechanism_property_tester = StubPropertyTester(
        rejects_property=False
    )
    test_privacy_tester = privacy_test_runner.PrivacyTestRunner(
        config=self.runner_config,
        dataset_generator=_stub_dataset_generator,
        property_tester=stub_private_mechanism_property_tester,
    )
    expected_results = privacy_test_runner_config.PrivacyTestRunnerResults(
        mechanism_name=self.mechanism_name,
        property_tester=privacy_test_runner_config.PropertyTester.UNSPECIFIED_TESTER,
        privacy_property=stub_private_mechanism_property_tester.privacy_property,
        max_num_trials=self.runner_config.max_num_trials,
        num_inspected_trials=self.runner_config.max_num_trials,
        lower_bound_divergence_estimates=[],
        termination_reason=privacy_test_runner_config.TerminationReason.TRIAL_LIMIT_REACHED,
    )
    expected_results.lower_bound_divergence_estimates.extend(
        [_LOWER_BOUND for _ in range(self.runner_config.max_num_trials)]
    )

    results = test_privacy_tester.test_privacy(
        mechanism=_stub_mechanism, mechanism_name=self.mechanism_name
    )

    self.assertEqual(results, expected_results)

  def test_test_privacy_non_private_mechanism(self):
    stub_non_private_mechanism_property_tester = StubPropertyTester(
        rejects_property=True
    )
    test_privacy_tester = privacy_test_runner.PrivacyTestRunner(
        config=self.runner_config,
        dataset_generator=_stub_dataset_generator,
        property_tester=stub_non_private_mechanism_property_tester,
    )
    expected_results = privacy_test_runner_config.PrivacyTestRunnerResults(
        mechanism_name=self.mechanism_name,
        property_tester=privacy_test_runner_config.PropertyTester.UNSPECIFIED_TESTER,
        privacy_property=stub_non_private_mechanism_property_tester.privacy_property,
        max_num_trials=self.runner_config.max_num_trials,
        lower_bound_divergence_estimates=[],
        num_inspected_trials=1,
        termination_reason=privacy_test_runner_config.TerminationReason.FOUND_PRIVACY_VIOLATION,
        found_privacy_violation=privacy_test_runner_config.FoundPrivacyViolation(
            failure_probability=self.runner_config.failure_probability
        ),
    )
    expected_results.lower_bound_divergence_estimates.append(_LOWER_BOUND)

    results = test_privacy_tester.test_privacy(
        mechanism=_stub_mechanism, mechanism_name=self.mechanism_name
    )

    self.assertEqual(results, expected_results)

  def test_test_privacy_apply_tanh(self):
    stub_private_mechanism_property_tester = StubPropertyTester(
        rejects_property=False
    )
    self.runner_config.post_processing = (
        privacy_test_runner_config.PostProcessing.TANH
    )
    test_privacy_tester = privacy_test_runner.PrivacyTestRunner(
        config=self.runner_config,
        dataset_generator=_stub_dataset_generator,
        property_tester=stub_private_mechanism_property_tester,
    )
    stub_samples = np.array([[-10.4], [5.0], [11.5]])
    expected_results = np.tanh(stub_samples)

    results = test_privacy_tester.maybe_postprocess(stub_samples)
    self.assertAllClose(results, expected_results)

  def test_test_privacy_apply_chebyshev_d5(self):
    stub_private_mechanism_property_tester = StubPropertyTester(
        rejects_property=False
    )
    self.runner_config.post_processing = (
        privacy_test_runner_config.PostProcessing.CHEBYSHEV_POLYNOMIALS_D5
    )
    test_privacy_tester = privacy_test_runner.PrivacyTestRunner(
        config=self.runner_config,
        dataset_generator=_stub_dataset_generator,
        property_tester=stub_private_mechanism_property_tester,
    )
    stub_samples = np.array([[-1.0], [0.0], [0.5]])
    expected_results = np.array([
        [+1.000000, -0.761594, +0.160051, +0.517806, -0.948767, +0.927345],
        [+1.000000, +0.000000, -1.000000, -0.000000, +1.000000, +0.000000],
        [+1.000000, +0.462117, -0.572895, -0.991607, -0.343582, +0.674057],
    ])

    results = test_privacy_tester.maybe_postprocess(stub_samples)
    self.assertAllClose(results, expected_results)


if __name__ == "__main__":
  absltest.main()

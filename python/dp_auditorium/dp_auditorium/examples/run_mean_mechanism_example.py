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
"""Example binary running privacy tests for a DP mean mechanism."""

from collections.abc import Sequence
import os

from absl import app
from absl import flags
from absl import logging
import numpy as np

from dp_auditorium import privacy_test_runner
from dp_auditorium.configs import dataset_generator_config
from dp_auditorium.configs import mechanism_config
from dp_auditorium.configs import privacy_property
from dp_auditorium.configs import privacy_test_runner_config
from dp_auditorium.configs import property_tester_config
from dp_auditorium.generators import vizier_dataset_generator
from dp_auditorium.mechanisms import mean
from dp_auditorium.testers import hockey_stick_tester


_OUTPUT_DIR = flags.DEFINE_string("output_dir", "", "Output directory")
_EXPERIMENT_NAME = flags.DEFINE_string("experiment_name", "", "Experiment name")
_EPSILON = flags.DEFINE_float("epsilon", 0.01, "Privacy parameter")
_DELTA = flags.DEFINE_float("delta", 0.0, "Privacy parameter")
_SEED = flags.DEFINE_integer(
    "seed", 0, "Seed to initialize random numbers generator."
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  rng = np.random.default_rng(seed=_SEED.value)

  # Configuration for a non-private mean mechanism that uses the true number of
  # points to calculate the average.
  mech_config = mechanism_config.MeanMechanismConfig(
      epsilon=_EPSILON.value,
      delta=_DELTA.value,
      use_noised_counts_for_calculating_mean=False,
      use_noised_counts_for_calculating_noise_scale=True,
      min_value=0.0,
      max_value=1.0,
  )
  # Initialize the mechanism.
  mechanism = mean.MeanMechanism(mech_config, rng)

  # Configuration for a Hockey-Stick property tester. Given arrays s1 and s2
  # with samples two distributions it will estimate the hockey-stick divergence
  # from the underlying distributions. It checks if the divergence is bounded by
  # delta.
  tester_config = property_tester_config.HockeyStickPropertyTesterConfig(
      training_config=hockey_stick_tester.make_default_hs_training_config(),
      approximate_dp=privacy_property.ApproximateDp(
          epsilon=_EPSILON.value,
          delta=_DELTA.value,
      ),
  )
  # Initialize a classifier model for the Hockey-Stick property tester.
  base_model = hockey_stick_tester.make_default_hs_base_model()
  # Initialize a property tester.
  property_tester = hockey_stick_tester.HockeyStickPropertyTester(
      config=tester_config,
      base_model=base_model,
  )

  # Configuration for dataset generator. It generates neighboring datasets under
  # the add/remove definition.
  generator_config = dataset_generator_config.VizierDatasetGeneratorConfig(
      study_name="non-private-mean-hockey-stick-test",
      study_owner="owner",
      num_vizier_parameters=2,
      data_type=dataset_generator_config.DataType.DATA_TYPE_FLOAT,
      min_value=-1.0,
      max_value=1.0,
      search_algorithm="GAUSSIAN_PROCESS_BANDIT",
      metric_name="hockey_stick_divergence",
  )
  # Initialize the dataset generator.
  dataset_generator = (
      vizier_dataset_generator.VizierScalarDataAddRemoveGenerator(
          config=generator_config,
      )
  )

  # Configuration for the test runner.
  test_runner_config = privacy_test_runner_config.PrivacyTestRunnerConfig(
      property_tester=privacy_test_runner_config.PropertyTester.HOCKEY_STICK_TESTER,
      max_num_trials=5,
      failure_probability=0.05,
      num_samples=10_000,
      # Apply a hyperbolic tangent function to the output of the mechanism
      post_processing=privacy_test_runner_config.PostProcessing.TANH,
  )
  # Initialize the test runner.
  test_runner = privacy_test_runner.PrivacyTestRunner(
      config=test_runner_config,
      dataset_generator=dataset_generator,
      property_tester=property_tester,
  )

  results = test_runner.test_privacy(mechanism, "non-private-mean-mechanism")

  logging.info("\nResults: \n")
  logging.info(results)

  # Write results to the output directory.
  if not os.path.exists(_OUTPUT_DIR.value):
    os.makedirs(_OUTPUT_DIR.value)
    logging.info("Directory %s created successfully", _OUTPUT_DIR.value)
  else:
    print(f"Directory '{_OUTPUT_DIR.value}' already exists")

  output_file_name = os.path.join(_OUTPUT_DIR.value, _EXPERIMENT_NAME.value)
  with open(output_file_name, mode="w") as f:
    f.write(str(results))


if __name__ == "__main__":
  app.run(main)

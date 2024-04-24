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
"""Helper functions for Renyi Tester."""

from typing import Optional

import numpy as np

from dp_auditorium.configs import privacy_property
from dp_auditorium.configs import property_tester_config


def split_train_test_samples(
    samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """Splits a sample into training and testing samples.

  Args:
    samples: Original sample. Assumes that the first dimension in the array is
      the number of examples.

  Returns:
    Two samples of the same rank as samples but with the first and second
    half of the data respectively. When the number of samples has an odd length,
    the first half will have one fewer record.


  Raises:
    ValueError if samples has less than 2 elements.
  """
  n = samples.shape[0]
  if n == 1:
    raise ValueError(
        'Must have at least 2 elements in sample to generate '
        'training and test samples'
    )
  return samples[0 : n // 2, ...], samples[n // 2 :, ...]


def validate_training_config(
    training_config: property_tester_config.TrainingConfig,
):
  """Returns true if the training_params parameters are valid."""
  if (
      not training_config.optimizer_learning_rate
      or training_config.optimizer_learning_rate <= 0
  ):
    raise ValueError('Learning rate should be positive.')
  if (
      not training_config.training_epochs
      or training_config.training_epochs <= 0
  ):
    raise ValueError('Training epochs should be positive.')
  if not training_config.batch_size or training_config.batch_size <= 0:
    raise ValueError('Batch size should be positive.')
  if (
      not training_config.model_output_coordinate_bound
      or training_config.model_output_coordinate_bound <= 0
  ):
    raise ValueError('Model output coordinate bound should be positive.')


def _validate_epsilon_parameter(epsilon: Optional[float]):
  """Checks that the epsilon parameter is valid."""
  if not epsilon:
    raise ValueError('`epsilon` should be specified.')
  if epsilon <= 0:
    raise ValueError('`epsilon` should be positive.')


def validate_approximate_dp_property(
    approximate_dp: privacy_property.ApproximateDp,
):
  """Checks that the approximate differential privacy parameters are valid.

  Args:
    approximate_dp: An approximate differential privacy configuration.

  Raises:
    ValueError: epsilon is None or non-positive.
    ValueError: delta is None or negative.
  """
  _validate_epsilon_parameter(approximate_dp.epsilon)
  if approximate_dp.delta is None:
    raise ValueError('`delta` should be specified for approximate DP.')

  if approximate_dp.delta < 0:
    raise ValueError('`delta` should be non-negative.')


def validate_pure_dp_property(pure_dp: privacy_property.PureDp):
  """Checks that the pure differential privacy parameters are valid.

  Args:
    pure_dp: A pure differential privacy configuration.

  Raises:
    ValueError: epsilon is None or non-positive.
  """
  _validate_epsilon_parameter(pure_dp.epsilon)


def validate_renyi_dp_property(renyi_dp: privacy_property.RenyiDp):
  """Checks that Renyi differential privacy parameters are valid.

  Args:
    renyi_dp: A pure differential privacy configuration.

  Raises:
    ValueError: epsilon is None or non-positive.
    ValueError: alpha is None or less or equal than one.
  """
  _validate_epsilon_parameter(renyi_dp.epsilon)
  if not renyi_dp.alpha:
    raise ValueError(
        'The order Renyi divergence `alpha` should be specified for Renyi DP.'
    )
  if renyi_dp.alpha <= 1:
    raise ValueError(
        'The order of Renyi divergence `alpha` should be greater than one.'
    )

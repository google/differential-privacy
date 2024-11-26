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
"""Renyi divergence calculators.

Functions to estimate Renyi divergence between samples of two distributions.
"""

from typing import Dict

import numpy as np
import tensorflow as tf
from typing_extensions import override

from dp_auditorium.configs import privacy_property
from dp_auditorium.configs import property_tester_config
from dp_auditorium.testers import divergence_tester
from dp_auditorium.testers import property_tester_utils


def make_default_renyi_base_model() -> tf.keras.Model:
  return tf.keras.models.Sequential([
      tf.keras.layers.Dense(100, activation=tf.keras.activations.tanh),
      tf.keras.layers.Dense(100, activation=tf.keras.activations.tanh),
      tf.keras.layers.Dense(1),
  ])


def _compute_error_from_samples(
    num_samples: int,
    failure_probability: float,
    model_output_coordinate_bound: float,
    alpha: float,
) -> float:
  """Returns error tolerance from number of samples.

  Args:
    num_samples: Number of used samples to estimate divergence.
    failure_probability: Probability of failing the Chernoff bound.
    model_output_coordinate_bound: Constant that bounds function class to
      estimate Renyi divergence.
    alpha: Order of Renyi divergence.
  """
  error_1 = np.sqrt(
      3
      * np.log(2 / failure_probability)
      * np.exp(2 * (alpha - 1) * model_output_coordinate_bound)
      / num_samples
  )
  error_2 = np.sqrt(
      2
      * np.log(2 / failure_probability)
      * np.exp(2 * alpha * model_output_coordinate_bound)
      / num_samples
  )
  gamma = max(error_1, error_2)
  error_from_gamma = np.log((1 + gamma) / (1 - gamma))
  return error_from_gamma


class RenyiModel(tf.keras.Model):
  """Model to estimate Renyi Divergence using variational formulation."""

  def __init__(self, nn_model, alpha):
    super().__init__()
    self.nn_model = nn_model
    self.alpha = alpha

  def train_step(
      self, data: tuple[np.ndarray, np.ndarray]
  ) -> Dict[str, tf.Tensor]:
    with tf.GradientTape() as tape:
      divergence = self(data, training=True)
      loss = -divergence

    trainable_vars = self.nn_model.trainable_variables
    d_loss = tape.gradient(loss, trainable_vars)
    self.optimizer.apply_gradients(zip(d_loss, trainable_vars))
    return {'divergence': divergence}

  def call(  # pytype: disable=annotation-type-mismatch
      self, data: tuple[np.ndarray, np.ndarray], training: bool = None
  ) -> tf.Tensor:
    """Estimate renyi divergence from samples and current nn_model.

    This function unpacks samples x_p and x_q from two distributions and uses a
    function parametrized by nn_model to estimate the renyi divergence following
    the variational representation in https://arxiv.org/abs/2007.03814. Letting
    t1 an t2 be the average of exp(nn_model(x_p) and exp(nn_model(x_q))
    respectively, we estimate the divergence as alpha/(alpha-1)log(t1) -
    log(t2).

    Args:
      data: tuple of two float arrays with samples from two distributions.
      training: indicates if the model is being used for training or inference.

    Returns:
      estimated divergence with current nn_model parameters.
    """
    x_p, x_q = data
    g_p = self.nn_model(x_p, training=training)
    g_q = self.nn_model(x_q, training=training)
    t1 = tf.math.reduce_mean(tf.exp((self.alpha - 1) * g_p))
    t2 = tf.math.reduce_mean(tf.exp(self.alpha * g_q))
    divergence = (self.alpha / (self.alpha - 1)) * tf.math.log(
        t1
    ) - tf.math.log(t2)
    return divergence


class RenyiPropertyTester(divergence_tester.DivergencePropertyTester):
  """Renyi tester main class.

  RenyiTester computes a lower bound for the Renyi divergence using Algorithm 2
  in https://arxiv.org/abs/2307.05608. It computes a lower bound for the Renyi
  divergence first using train samples to find a suitable function parametrized
  by `renyi_model`. Then it uses test samples to estimate the lower end point of
  a confidence interval for the divergence.
  """

  def __init__(
      self,
      config: property_tester_config.RenyiPropertyTesterConfig,
      base_model: tf.keras.Model,
  ):
    # Get privacy parameters
    if config.privacy_property.renyi_dp is not None:
      property_tester_utils.validate_renyi_dp_property(
          config.privacy_property.renyi_dp
      )
      privacy_type = 'renyi_dp'
      epsilon = config.privacy_property.renyi_dp.epsilon
      alpha = config.privacy_property.renyi_dp.alpha
      if config.alpha != alpha:
        raise ValueError(
            'Alpha parameter for Renyi DP should be specified in'
            ' privacy_tester_config.privacy_property. It was specified in'
            ' config.alpha which is only used for Pure DP tests.'
        )
    elif config.privacy_property.pure_dp is not None:
      property_tester_utils.validate_pure_dp_property(
          config.privacy_property.pure_dp
      )
      privacy_type = 'pure_dp'
      epsilon = config.privacy_property.pure_dp.epsilon
      alpha = config.alpha
    else:
      raise ValueError(
          'The specified privacy_property is not supported by'
          ' RenyiPropertyTester.'
      )
    property_tester_utils.validate_training_config(config.training_config)

    if privacy_type == 'renyi_dp':
      self._initial_test_threshold = epsilon
    else:
      self._initial_test_threshold = min(epsilon, 2 * alpha * epsilon**2)

    self._tested_property = config.privacy_property
    self._alpha = alpha

    self._training_config = config.training_config

    self._model_output_coordinate_bound = (
        config.training_config.model_output_coordinate_bound
    )

    def scaled_tanh(x):
      return self._model_output_coordinate_bound * tf.keras.activations.tanh(x)

    base_model.add(tf.keras.layers.Activation(scaled_tanh))

    self._renyi_model = RenyiModel(base_model, self._alpha)
    self._renyi_model.compile(
        optimizer=tf.keras.optimizers.Adam(
            config.training_config.optimizer_learning_rate
        ),
    )
    self._divergence_train = []

  @property
  def _test_threshold(self) -> float:
    return self._initial_test_threshold

  @property
  def privacy_property(self) -> privacy_property.PrivacyProperty:
    return self._tested_property

  def _reset_model_weights(self):
    for layer in self._renyi_model.nn_model.layers:
      if hasattr(layer, 'kernel'):
        if layer.kernel is not None and hasattr(layer, 'kernel_initializer'):
          layer.kernel.assign(layer.kernel_initializer(tf.shape(layer.kernel)))
      if hasattr(layer, 'bias'):
        if layer.bias is not None and hasattr(layer, 'bias_initializer'):
          layer.bias.assign(layer.bias_initializer(tf.shape(layer.bias)))

  @override
  def _get_optimized_divergence_estimation_model(
      self,
      samples_first_distribution: np.ndarray,
      samples_second_distribution: np.ndarray,
  ) -> tf.keras.Model:
    self._reset_model_weights()
    self._renyi_model.fit(
        samples_first_distribution,
        samples_second_distribution,
        batch_size=self._training_config.batch_size,
        epochs=self._training_config.training_epochs,
        verbose=self._training_config.verbose,
    )
    return self._renyi_model

  @override
  def _compute_divergence_on_samples(
      self,
      model: tf.keras.Model,
      samples1_test: np.ndarray,
      samples2_test: np.ndarray,
      failure_probability: float,
  ) -> float:
    divergence_test = model((samples1_test, samples2_test))

    # Calculate lower end of confidence interval.
    num_samples = min(samples1_test.shape[0], samples2_test.shape[0])
    error = _compute_error_from_samples(
        num_samples=num_samples,
        failure_probability=failure_probability,
        model_output_coordinate_bound=self._model_output_coordinate_bound,
        alpha=self._alpha,
    )
    divergence_test_lower_bound = divergence_test - error

    return divergence_test_lower_bound

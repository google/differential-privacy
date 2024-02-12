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

from typing import Dict, Optional, Union

import numpy as np
import tensorflow as tf
from typing_extensions import override

from dp_auditorium import interfaces
from dp_auditorium.configs import privacy_property
from dp_auditorium.configs import property_tester_config
from dp_auditorium.testers import property_tester_utils


def _renyi_model_parameters_initializer(
    config: property_tester_config.RenyiPropertyTesterConfig,
    base_model: Optional[tf.keras.Model] = None,
) -> dict[str, Union[float, int, None, tf.keras.Model]]:
  """Initializes attributes for RenyiPropertyTester.

  This function processes `config` to extract privacy parameters
  and initialize the model parametrizing the Renyi divergence approximation. See
  section 4.1. of https://arxiv.org/pdf/2307.05608.pdf for more details.

  Args:
    config: A RenyiPropertyTester configuration.
    base_model: A keras model to use to parametrize the variational formulation
      of the Renyi divergence.

  Returns:
    A dictionary with relevant attributes to initialize a RenyiPropertyTester.
    The dictionary contains (1) a value `alpha` for the order of the Renyi
    divergence being estimated, (2) the `test_threshold`, and (3) a `base_model`
    keras model parametrizing the function space to estimate the Renyi
    divergence.

  Raises:
    ValueError if the config sets two different alpha parameters
    when testing Renyi DP or if the privacy property is different than pure or
    Renyi DP.
  """

  if config.privacy_property.renyi_dp is not None:
    privacy_type = 'renyi_dp'
    alpha = config.privacy_property.renyi_dp.alpha
    epsilon = config.privacy_property.renyi_dp.epsilon
    if config.alpha != alpha:
      raise ValueError(
          'Alpha parameter for Renyi DP should be specified in'
          ' privacy_tester_config.privacy_property. It was specified in'
          ' config.alpha which is only used for Pure DP tests.'
      )
  elif config.privacy_property.pure_dp is not None:
    privacy_type = 'pure_dp'
    epsilon = config.privacy_property.pure_dp.epsilon
    alpha = config.alpha
  else:
    raise ValueError(
        'The specified privacy_property is not supported by RenyiTester.'
    )

  model_output_coordinate_bound = (
      config.training_config.model_output_coordinate_bound
  )

  def scaled_tanh(x):
    return model_output_coordinate_bound * tf.keras.activations.tanh(x)

  if base_model is None:
    base_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(100, activation=scaled_tanh),
        tf.keras.layers.Dense(100, activation=scaled_tanh),
        tf.keras.layers.Dense(1),
    ])

  base_model.add(tf.keras.layers.Activation(scaled_tanh))

  if privacy_type == 'renyi_dp':
    threshold = epsilon
  else:
    threshold = min(epsilon, 2 * alpha * epsilon**2)
  return {
      'alpha': alpha,
      'test_threshold': threshold,
      'base_model': base_model,
  }


def _compute_error_from_gamma(gamma: float) -> float:
  """Returns additive error from convenience variable gamma.

  To estimate number of samples we allow for a multiplicative error `gamma` from
  chernoff bound in https://arxiv.org/abs/2307.05608. This function converts the
  multiplicative error to additive error.

  Args:
    gamma: Multiplicative error.
  """
  return np.log((1 + gamma) / (1 - gamma))


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
  return _compute_error_from_gamma(gamma)


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
    return {'renyi': divergence}

  def call(
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


class RenyiPropertyTester(interfaces.PropertyTester):
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
      base_model: Optional[tf.keras.Model] = None,
  ):
    # Get privacy parameters
    if config.privacy_property.renyi_dp is not None:
      property_tester_utils.validate_renyi_dp_property(
          config.privacy_property.renyi_dp
      )
    elif config.privacy_property.pure_dp is not None:
      property_tester_utils.validate_pure_dp_property(
          config.privacy_property.pure_dp
      )
    else:
      raise ValueError(
          'The specified privacy_property is not supported by'
          ' RenyiPropertyTester.'
      )
    property_tester_utils.validate_training_params(config.training_config)
    params = _renyi_model_parameters_initializer(
        config=config,
        base_model=base_model,
    )

    # Privacy test parameters.
    self._tested_property = config.privacy_property
    self._model_output_coordinate_bound = (
        config.training_config.model_output_coordinate_bound
    )
    self._alpha = params['alpha']
    self._test_threshold = params['test_threshold']

    # Optimization parameters.
    self._training_config = config.training_config

    self._renyi_model = RenyiModel(params['base_model'], self._alpha)
    self._renyi_model.compile(
        optimizer=tf.keras.optimizers.Adam(
            config.training_config.optimizer_learning_rate
        ),
    )
    self._divergence_train = []

  @property
  def privacy_property(self) -> privacy_property.PrivacyProperty:
    return self._tested_property

  def _reinitialize_nn_model(self):
    for layer in self._renyi_model.nn_model.layers:
      if hasattr(layer, 'kernel'):
        if layer.kernel is not None and hasattr(layer, 'kernel_initializer'):
          layer.kernel.assign(layer.kernel_initializer(tf.shape(layer.kernel)))
      if hasattr(layer, 'bias'):
        if layer.bias is not None and hasattr(layer, 'bias_initializer'):
          layer.bias.assign(layer.bias_initializer(tf.shape(layer.bias)))

  def _optimize_renyi_divergence(
      self,
      samples_first_distribution: np.ndarray,
      samples_second_distribution: np.ndarray,
      verbose: int = 0,
  ) -> tf.Tensor:
    """Renyi divergence computation.

    Args:
      samples_first_distribution: one dimensional array with samples.
      samples_second_distribution: one dimensional arrays with samples and same
        shape as as p.
      verbose: whether to print training evolution, for details see
        `tf.keras.mode.fit`.

    Returns:
      Estimated Renyi divergence on train samples.
    """
    self._reinitialize_nn_model()
    self._renyi_model.fit(
        samples_first_distribution,
        samples_second_distribution,
        batch_size=self._training_config.batch_size,
        epochs=self._training_config.training_epochs,
        verbose=verbose,
    )
    train_renyi = self._renyi_model.history.history['renyi'][-1]

    return train_renyi

  def estimate_divergence_from_samples(
      self,
      samples_1_train: np.ndarray,
      samples_2_train: np.ndarray,
      samples_1_test: np.ndarray,
      samples_2_test: np.ndarray,
      failure_probability: float,
      verbose: int,
  ) -> tuple[tf.Tensor, tf.Tensor]:
    """Estimates Renyi divergence from samples.

    This method estimates the Renyi divergence beween two distributions. First
    it optimizes over a function space determined by the RenyiModel and
    then uses the learned function to estimate the Renyi divergence over test
    samples.

    Args:
      samples_1_train: Samples from the first distribution used to find a
        suitable set of parameters for `renyi_model`.
      samples_2_train: Samples from the second distribution used to find a
        suitable set of parameters for `renyi_model`.
      samples_1_test: Samples from the first distribution used to estimate
        divergence.
      samples_2_test: Samples from the second distribution used to estimate
        divergence.
      failure_probability: P
      verbose: integer passed to `fit` method for logging.

    Returns:
      A tuple where the first element is the train divergence and the second is
      the estimated divergence lower bound.
    """
    # Find suitable model parameters.
    divergence_train = self._optimize_renyi_divergence(
        samples_1_train, samples_2_train, verbose=verbose
    )

    divergence_test = self._renyi_model((samples_1_test, samples_2_test))

    # Calculate lower end of confidence interval.
    num_samples = samples_1_test.shape[0]
    error = _compute_error_from_samples(
        num_samples=num_samples,
        failure_probability=failure_probability,
        model_output_coordinate_bound=self._model_output_coordinate_bound,
        alpha=self._alpha,
    )
    divergence_test_lower_bound = divergence_test - error

    return divergence_train, divergence_test_lower_bound

  @override
  def estimate_lower_bound(
      self,
      samples1: np.ndarray,
      samples2: np.ndarray,
      failure_probability: float,
  ) -> float:
    samples1_train, samples1_test = (
        property_tester_utils.split_train_test_samples(samples1)
    )
    samples2_train, samples2_test = (
        property_tester_utils.split_train_test_samples(samples2)
    )

    divergence_train, divergence_test = self.estimate_divergence_from_samples(
        samples1_train,
        samples2_train,
        samples1_test,
        samples2_test,
        failure_probability,
        verbose=0,
    )
    self._divergence_train.append(divergence_train)

    return divergence_test.numpy()

  @override
  def reject_property(self, lower_bound: float) -> bool:
    return lower_bound > self._test_threshold

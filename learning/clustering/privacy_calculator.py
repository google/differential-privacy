# Copyright 2022 Google LLC.
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
"""Calculates average and count privacy params."""

import functools

from absl import logging
import numpy as np

from clustering import central_privacy_utils
from clustering import clustering_params
from dp_accounting import dp_event
from dp_accounting import dp_event_builder
from dp_accounting import mechanism_calibration
from dp_accounting.pld import accountant
from dp_accounting.pld import common
from dp_accounting.pld import pld_privacy_accountant


def make_clustering_event(sum_std_dev: float, count_laplace_param: float,
                          sensitivity: float,
                          max_depth: int) -> dp_event.DpEvent:
  """Returns a DpEvent for clustering."""
  builder = dp_event_builder.DpEventBuilder()

  if sum_std_dev == 0:
    builder.compose(dp_event.NonPrivateDpEvent())
  else:
    builder.compose(dp_event.GaussianDpEvent(sum_std_dev / sensitivity))

  # Depth is based on the number of edges in the path, so add one to the depth
  # to get the number of levels.
  if count_laplace_param == np.inf:
    builder.compose(dp_event.NonPrivateDpEvent())
  else:
    builder.compose(
        dp_event.LaplaceDpEvent(1 / count_laplace_param), max_depth + 1)
  return builder.build()


def make_clustering_event_from_param(
    multipliers: clustering_params.PrivacyCalculatorMultiplier,
    sensitivity: float, max_depth: int, alpha: float) -> dp_event.DpEvent:
  """Returns a DpEvent for clustering with the parameter alpha.

  Args:
    multipliers: multipliers to calculate the noise parameters given alpha.
    sensitivity: sensitivity of the dataset for the sum operations.
    max_depth: max depth of the prefix tree for generating the coreset.
    alpha: parameter varied in mechanism calibration.
  """
  logging.debug('Mechanism Calibration: Testing param alpha = %s', alpha)
  return make_clustering_event(
      sum_std_dev=multipliers.get_gaussian_std_dev(alpha, sensitivity),
      count_laplace_param=multipliers.get_laplace_param(alpha),
      sensitivity=sensitivity,
      max_depth=max_depth)


def get_alpha_interval(
    privacy_param: clustering_params.DifferentialPrivacyParam, radius: float,
    multipliers: clustering_params.PrivacyCalculatorMultiplier
) -> mechanism_calibration.BracketInterval:
  """Returns an interval for alpha used in mechanism calibration.

  Args:
    privacy_param: privacy parameters, epsilon must not be infinite, and delta
      must be less than 1.
    radius: radius of the dataset.
    multipliers: multipliers for noise parameters.
  """
  if privacy_param.epsilon == np.inf or privacy_param.delta >= 1:
    raise ValueError(
        'get_alpha_interval should not be called for nonprivate parameters.')

  # To pick a lower bound, check what the Gaussian std dev would be if we
  # used the entire privacy budget on the Gaussian operation.
  all_eps_std_dev = accountant.get_smallest_gaussian_noise(
      privacy_parameters=common.DifferentialPrivacyParameters(
          privacy_param.epsilon, privacy_param.delta),
      num_queries=1,
      sensitivity=radius)
  lower_bound_alpha = multipliers.get_alpha(all_eps_std_dev, radius)
  return mechanism_calibration.LowerEndpointAndGuess(lower_bound_alpha,
                                                     2 * lower_bound_alpha)


class PrivacyCalculator():
  """Calculates and returns privacy parameters.

  Attributes:
    average_privacy_param: Privacy parameters for calculating the private
      average.
    count_privacy_param: Privacy parameters for calculating private counts.
  """
  average_privacy_param: central_privacy_utils.AveragePrivacyParam
  count_privacy_param: central_privacy_utils.CountPrivacyParam

  def __init__(
      self,
      privacy_param: clustering_params.DifferentialPrivacyParam,
      radius: float,
      max_depth: int,
      multipliers: clustering_params.PrivacyCalculatorMultiplier,
  ):
    """Initializes privacy calculator.

    Uses mechanism calibration to calculate noise parameters.

    Args:
      privacy_param: Privacy parameter for the clustering algorithm.
      radius: Radius of the dataset being clustered.
      max_depth: Maximum depth of the LSH prefix tree.
      multipliers: Noise multipliers to use for mechanism calibration.
    """
    if privacy_param.privacy_model != clustering_params.PrivacyModel.CENTRAL:
      raise NotImplementedError(
          f'Currently unsupported privacy model: {privacy_param.privacy_model}'
      )

    if privacy_param.epsilon == np.inf or privacy_param.delta >= 1:
      # No noise.
      self.average_privacy_param = central_privacy_utils.AveragePrivacyParam(
          0, radius
      )
      self.count_privacy_param = central_privacy_utils.CountPrivacyParam(np.inf)
      return

    interval = get_alpha_interval(privacy_param, radius, multipliers)
    alpha = mechanism_calibration.calibrate_dp_mechanism(
        pld_privacy_accountant.PLDAccountant,
        make_event_from_param=functools.partial(
            make_clustering_event_from_param, multipliers, radius, max_depth
        ),
        target_epsilon=privacy_param.epsilon,
        target_delta=privacy_param.delta,
        bracket_interval=interval,
    )

    self.average_privacy_param = central_privacy_utils.AveragePrivacyParam(
        multipliers.get_gaussian_std_dev(alpha, radius), radius
    )
    self.count_privacy_param = central_privacy_utils.CountPrivacyParam(
        multipliers.get_laplace_param(alpha)
    )

  def validate_accounting(
      self,
      privacy_param: clustering_params.DifferentialPrivacyParam,
      max_depth: int,
  ):
    """Errors if the params exceed the privacy budget."""
    if privacy_param.epsilon == np.inf or privacy_param.delta >= 1:
      return

    clustering_event = make_clustering_event(
        self.average_privacy_param.gaussian_standard_deviation,
        self.count_privacy_param.laplace_param,
        self.average_privacy_param.sensitivity,
        max_depth,
    )

    acct = pld_privacy_accountant.PLDAccountant()
    acct.compose(clustering_event)
    calculated_epsilon = acct.get_epsilon(privacy_param.delta)
    calculated_delta = acct.get_delta(privacy_param.epsilon)

    logging.debug('Accounted epsilon: %s', calculated_epsilon)
    logging.debug('Accounted delta: %s', calculated_delta)

    if (
        calculated_epsilon > privacy_param.epsilon
        or calculated_delta > privacy_param.delta
    ):
      raise ValueError(
          'Accounted privacy params greater than allowed: '
          f'({calculated_epsilon}, {calculated_delta}) > '
          f'({privacy_param.epsilon}, {privacy_param.delta})'
      )

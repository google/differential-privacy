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
"""Helper methods for mechanisms."""

from dp_auditorium.configs import mechanism_config


def get_wrong_one_step_gd_budget(
    noise_multiplier: float, alpha: float
) -> float:
  """Wrong privacy budget unless l2_norm_clip is per-sample clip.

  This corresponds to a Gaussian mechanism with sensitivity
  2*l2_norm_clip*noise_multiplier, The correct budget should be
  epsilon = alpha * mu ** 2 / (2 * sigma ** 2)
  = alpha * (2 * clip) ** 2 / (2 * (nm * clip) ** 2) = 2 * alpha / nm ** 2
  i.e., it is off by a factor of 4 (see https://arxiv.org/pdf/1702.07476.pdf).

  Args:
    noise_multiplier: Ratio of l2_norm_clip and standard deviation of the noise
      applied to gradients.
    alpha: Rényi divergence order for privacy accounting.

  Returns:
    privacy budget epsilon.
  """
  epsilon = alpha / (2.0 * (noise_multiplier) ** 2)
  return epsilon


def default_mean_mechanism_config_generator(
    mechanism_name, epsilon, delta=0.0
) -> mechanism_config.MeanMechanismConfig:
  """Returns a config to initialize some common implementations.

  Args:
    mechanism_name: One of "non_private_mean_v1", "non_private_mean_v2",
      "private_mean".
    epsilon: The epsilon value to instantiate the mechanism.
    delta: The delta value to instantiate the mechanism.

  Returns:
    A MeanMechanismConfig

  Raises:
    AttributeError if passed an unkown mechanism.
  """
  if mechanism_name == "non_private_mean_v1":
    use_noised_counts_for_calculating_mean = False
    use_noised_counts_for_calculating_noise_scale = False
  elif mechanism_name == "non_private_mean_v2":
    use_noised_counts_for_calculating_mean = False
    use_noised_counts_for_calculating_noise_scale = True
  elif mechanism_name == "private_mean":
    use_noised_counts_for_calculating_mean = True
    use_noised_counts_for_calculating_noise_scale = True
  else:
    raise AttributeError("Unknown mechanism %s" % mechanism_name)
  return mechanism_config.MeanMechanismConfig(
      epsilon=epsilon,
      delta=delta,
      use_noised_counts_for_calculating_mean=use_noised_counts_for_calculating_mean,
      use_noised_counts_for_calculating_noise_scale=use_noised_counts_for_calculating_noise_scale,
      max_value=1.0,
      min_value=0.0,
  )

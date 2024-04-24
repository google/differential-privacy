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
import dataclasses
import enum
from typing import Optional

from dp_auditorium.configs import privacy_property


@dataclasses.dataclass
class TrainingConfig:
  """Training config for testers.

  It contains parameters for testers that require training models over samples
  of a mechanism to learn a model that estimates a divergence between
  distributions, e.g., HockeyStick tester and RenyiTester.

  Attributes:
    training_epochs: Number of train epochs over train examples.
    batch_size: Batch size used for training.
    optimizer_learning_rate: Learning rate for the optimizer.
    model_output_coordinate_bound: Constant bounding the magnitude of the
      trained model outputs coordinate-wise.
    verbose: Verbose mode for tensorflow keras model fit method.
  """

  training_epochs: int
  batch_size: int
  optimizer_learning_rate: float
  model_output_coordinate_bound: Optional[float] = None
  verbose: int = 0


@dataclasses.dataclass
class RenyiPropertyTesterConfig:
  """Configuration for Renyi divergence based property tester.

  Attributes:
    alpha: Order of Renyi divergence. Only used when testing for pure DP. When
      testing for Renyi DP the tester will use the alpha specified in the
      privacy_property in PrivacyTesterConfig.
    training_config: Required training parameters.
    privacy_property: Privacy guarantee the property tester is testing for.
  """

  alpha: float
  training_config: TrainingConfig
  privacy_property: privacy_property.PrivacyProperty


@dataclasses.dataclass
class HockeyStickPropertyTesterConfig:
  """Configuration for HockeyStick divergence based property tester.

  Attributes:
    training_config: Required training parameters.
    approximate_dp: Approximate DP privacy parameters to be tested.
    evaluation_batch_size: Batch size for computing accuracy of classifier
      distinguishing two distributions for Hockey Stick divergence. See
      `HockeyStickPropertyTester` class for details.
  """

  training_config: TrainingConfig
  approximate_dp: privacy_property.ApproximateDp
  evaluation_batch_size: int = 1000


@dataclasses.dataclass
class HistogramPropertyTesterConfig:
  """Configuration for histogram based property tester.

  Attributes:
    test_discrete_mechanism: Whether the output of the tested mechanism is
      discrete.
    histogram_size: Number of bins to construct the histogram.
    min_value: Lower end value for the histogram.
    max_value: Upper end value for the histogram.
    approximate_dp: Approximate DP privacy parameters to be tested.
    use_original_tester: Whether to use the original version of the tester due
      to Gilbert and McMillan (2018), or a new version developed for
      DP-Auditorium. The new version generally improves over the original
      verison, but the original version is retained for comparison purposes.
  """

  test_discrete_mechanism: bool
  histogram_size: int
  min_value: float
  max_value: float
  approximate_dp: privacy_property.ApproximateDp
  use_original_tester: bool = False


class Kernel(enum.Enum):
  """Possible kernel functions."""

  KERNEL_UNSPECIFIED = 0
  KERNEL_RBF = 1
  KERNEL_LAPLACIAN = 2


@dataclasses.dataclass
class MMDPropertyTesterConfig:
  """Configuration for maximum mean discrepancy (MMD) property tester.

  Attributes:
    kernel: Kernel function of the reproducing kernel Hilbert space over which
      the mean discrepancy is maximized.
    bandwidth: Bandwidth of the kernel function.
    approximate_dp: Approximate DP privacy parameters to be tested.
  """

  kernel: Kernel
  bandwidth: float
  approximate_dp: privacy_property.ApproximateDp

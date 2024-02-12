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
"""Privacy tester manager config and results."""

import dataclasses
import enum
from typing import Optional

from dp_auditorium.configs import privacy_property


class PropertyTester(enum.Enum):
  """Available property testers."""

  UNSPECIFIED_TESTER = 0
  HISTOGRAM_TESTER = 1
  HOCKEY_STICK_TESTER = 2
  MMD_TESTER = 3
  RENYI_TESTER = 4


class PostProcessing(enum.Enum):
  """Available postprocessing functions to apply to the samples of a mechanism

  before passing the samples to the PropertyTester.
  """

  NONE = 0
  TANH = 1
  CHEBYSHEV_POLYNOMIALS_D5 = 2


@dataclasses.dataclass
class PrivacyTestRunnerConfig:
  """Privacy tester manager class config.

  Attributes:
    property_tester: Required PropertyTester that will be used for the test.
    max_num_trials: Required maximum number of datasets inspected during a
      privacy test.
    failure_probability: Required probability of test failure for each trial.
    num_samples: Required number of samples from the tested mechanism to use for
      testing.
    post_processing: Optional field indicating if a postprocesssing is used for
      the samples of the mechanism.
  """

  property_tester: PropertyTester
  max_num_trials: int
  failure_probability: float
  num_samples: int
  post_processing: PostProcessing


class TerminationReason(enum.Enum):
  """Reasons for the test to finish.

  Either the PropertyTester finds a privacy violation at trial `t`,
  `0<=t<=max_num_trials` or it evaluates `max_num_trials` without finding
  privacy violations.
  """

  UNKNOWN = 0
  TRIAL_LIMIT_REACHED = 1
  FOUND_PRIVACY_VIOLATION = 2


@dataclasses.dataclass
class FoundPrivacyViolation:
  """Message used to record when a privacy violation is found and the corresponding probability.

  Attributes:
    failure_probability: Probability of test failure in case a privacy violation
      is found.
  """

  failure_probability: float


@dataclasses.dataclass
class PrivacyTestRunnerResults:
  """Client-facing representation of a privacy tester results.

  Attributes:
    mechanism_name: Required field indicating the name of the evaluated
      mechanism.
    privacy_property: Required field indicating the property the mechanism is
      being evaluated for.
    property_tester: Required PropertyTester that will be used for the test.
    max_num_trials: Required field indicating the maximum number of datasets
      over which the mechanism could have been evaluated during the test.
    lower_bound_divergence_estimates: Values of estimated divergence lower
      bound. When a privacy violation is found the privacy test manager returns
      the results before `max_num_trials` is reached. There is one divergence
      value for each trial, so the number of values in this field indicates the
      number of effective trials evaluated during the test.
    termination_reason: Required field indicating the reason why the test
      terminated.
    found_privacy_violation: Optional field indicating if the PropertyTester
      found that the mechanism violates privacy with probability
      `1-failure_probability`.
    num_inspected_trials: Number of datasets inspected before the test returned
      the results.
  """

  mechanism_name: str
  privacy_property: privacy_property.PrivacyProperty
  property_tester: PropertyTester
  max_num_trials: int
  lower_bound_divergence_estimates: list[float]
  termination_reason: TerminationReason = TerminationReason.UNKNOWN
  found_privacy_violation: Optional[FoundPrivacyViolation] = None
  num_inspected_trials: int = 0

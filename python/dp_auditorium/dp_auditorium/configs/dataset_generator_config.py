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
"""Configuration for Vizier dataset generators."""

import dataclasses
import enum
from vizier.service import pyvizier as vz


class DataType(enum.Enum):
  DATA_TYPE_UNSPECIFIED = 0
  DATA_TYPE_INT32 = 1
  DATA_TYPE_FLOAT = 2


@dataclasses.dataclass
class VizierDatasetGeneratorConfig:
  """Configuration for Vizier dataset generators.

  Attributes:
    study_name: String passed to Vizier to identify the study.
    study_owner: String determining the owner of the study. A Vizier client
      organizes studies by `owner` and `study_id`. For details see the class
      `Study` in `vizier/_src/service/clients.py`
      (https://github.com/google/vizier/tree/main).
    num_vizier_parameters: Number of parameters created by vizier.
    data_type: Data type of the Vizier parameters.
    min_value: Minimum value of the Vizier parameters.
    max_value: Maximum value of the Vizier parameters.
    search_algorithm: Search algorithm used by Vizier to generate parameters.
      See `vizier/_src/pyvizier/oss/study_config.py` for possible values.
    metric_name: User-specifed key, which is passed to Vizier for the purpose of
      storing metric values to be optimized. For instance, in a test using the
      `RenyiPropertyTester`, this can be denoted as `renyi_divergence`.
  """

  study_name: str
  study_owner: str
  num_vizier_parameters: int
  data_type: DataType
  min_value: float
  max_value: float
  search_algorithm: str
  metric_name: str


@dataclasses.dataclass(frozen=True)
class ClassificationDatasetGeneratorConfig:
  """Configuration for classification dataset generators supported by Vizier.

  Configuration to generate pairs of add/remove-neighboring datasets, where each
  record has the form `(features, label)`. `features` is a float array and label
  an integer indicating a class.

  Attributes:
    study_name: String passed to Vizier to identify the study.
    study_owner: String determining the owner of the study. A Vizier client
      organizes studies by `owner` and `study_id`. For details see the class
      `Study` in `vizier/_src/service/clients.py`
      (https://github.com/google/vizier/tree/main).
    num_vizier_parameters: Number of parameters created by vizier.
    data_type: Data type of the Vizier parameters.
    min_value: Minimum value of the Vizier parameters.
    max_value: Maximum value of the Vizier parameters.
    search_algorithm: Search algorithm used by Vizier to generate parameters.
      See `vizier/_src/pyvizier/oss/study_config.py` for possible values.
    metric_name: User-specifed key, which is passed to Vizier for the purpose of
      storing metric values to be optimized. For instance, in a test using the
      `RenyiPropertyTester`, this can be denoted as `renyi_divergence`.
    sample_dim: dimension of the feature space.
    num_samples: Number of samples in the largest dataset.
    num_classes: Number of classes in the classification task.
  """

  study_name: str
  study_owner: str
  min_value: float
  max_value: float
  search_algorithm: vz.Algorithm
  metric_name: str
  sample_dim: int
  num_samples: int
  num_classes: int

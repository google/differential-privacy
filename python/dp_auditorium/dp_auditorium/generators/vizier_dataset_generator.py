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
"""DatasetGenerator that connects with Vizier to suggest trials."""

import abc
from typing import Optional

from absl import logging
import numpy as np
from vizier.service import clients
from vizier.service import pyvizier as vz

from dp_auditorium import interfaces
from dp_auditorium.configs import dataset_generator_config


def _get_trial(
    study_client: clients.Study,
) -> tuple[Optional[clients.Trial], bool]:
  """Generates a new trial from a Vizier study client.

  Args:
    study_client: Vizier client study.

  Returns:
    A tuple (trial, loaded_new_trial) where loaded_new_trial is a boolean taking
    the value false if Vizier did not suggest a new trial and true otherwise.
  """
  suggestions = study_client.suggest(count=1)
  new_trial = suggestions[-1]
  loaded_new_trial = new_trial is not None
  return new_trial, loaded_new_trial


def _get_params_name_mapping(
    num_params: int,
) -> tuple[dict[int, str], dict[str, int]]:
  """Returns lookup dictionary for index to string and string to index."""
  idx_to_str = {idx: f'x{idx}' for idx in range(num_params)}
  str_to_idx = {
      name: int(name.split('x', 2)[1]) for name in idx_to_str.values()
  }
  return idx_to_str, str_to_idx


def _add_params_to_vizier_problem(
    vizier_problem: vz.ProblemStatement,
    idx_to_str: dict[int, str],
    input_config: dataset_generator_config.VizierDatasetGeneratorConfig,
) -> vz.ProblemStatement:
  """Adds variables to optimize be optimized to a vizier problem."""
  if (
      input_config.data_type
      == dataset_generator_config.DataType.DATA_TYPE_FLOAT
  ):
    for i in range(input_config.num_vizier_parameters):
      vizier_problem.search_space.root.add_float_param(
          idx_to_str[i],
          min_value=input_config.min_value,
          max_value=input_config.max_value,
      )
  elif (
      input_config.data_type
      == dataset_generator_config.DataType.DATA_TYPE_INT32
  ):
    for i in range(input_config.num_vizier_parameters):
      logging.info(
          'When using `DATA_TYPE_INT32`, `min_value` and `max_value` of type'
          ' float will be converted to `int` type.'
      )
      vizier_problem.search_space.root.add_int_param(
          idx_to_str[i],
          min_value=int(input_config.min_value),
          max_value=int(input_config.max_value),
      )
  else:
    raise NotImplementedError(
        'Unsupported data type: %s' % input_config.data_type
    )
  return vizier_problem


class VizierDatasetGenerator(interfaces.DatasetGenerator):
  """Data generator that generates neighboring datasets using OSS Vizier.

  Abstract dataset generator class that allows loading parameters from Vizier.
  Each use case has to implement the
  `get_neighboring_datasets_from_vizier_params` method that receives a one
  dimensional array of parameters generated from vizier and outputs two
  neighboring datasets.

  Attributes:
    study_client: Vizier client study.
    metric_name: Name of the metric being optimized by Vizier.
  """

  def __init__(
      self, config: dataset_generator_config.VizierDatasetGeneratorConfig
  ):
    """Initializes a `VizierDatasetGenerator` instance.

    Args:
      config: A configuration proto for Vizier dataset generator.
    """
    # Get indices to parameter names mapping assigned in Vizier and its inverse.
    idx_to_str, str_to_idx = _get_params_name_mapping(
        config.num_vizier_parameters
    )
    self._str_to_idx = str_to_idx
    # Define problem parameters
    problem = vz.ProblemStatement()
    problem = _add_params_to_vizier_problem(
        vizier_problem=problem,
        idx_to_str=idx_to_str,
        input_config=config,
    )

    # Define metric.
    self._metric_name = config.metric_name
    problem.metric_information = [
        vz.MetricInformation(
            name=config.metric_name, goal=vz.ObjectiveMetricGoal.MAXIMIZE
        )
    ]

    study_config = vz.StudyConfig.from_problem(problem)
    study_config.algorithm = config.search_algorithm

    self._study_client = clients.Study.from_study_config(
        study_config, owner=config.study_owner, study_id=config.study_name
    )

    self._num_vizier_params = config.num_vizier_parameters

    self._trial_loaded = False
    self._last_trial = None

  def _load_trial(self) -> None:
    if not self._trial_loaded:
      # Request a new trial from Vizier.
      logging.info('Loading new trial')
      self._last_trial, self._trial_loaded = _get_trial(self._study_client)
    if self._last_trial is None:
      raise ValueError('Vizier study did not generate a new trial.')

  def _complete_trial(self, last_trial_result: float) -> None:
    """Reports objective value of last trial to Vizier.

    Args:
      last_trial_result: objective value for the current trial that will be
        reported to Vizier.
    """
    if self._last_trial is None:
      logging.info(
          'The provided result to update last trial will not be used because '
          'last_trial does not exist.'
      )
    else:
      result_to_measurement = vz.Measurement(
          {self._metric_name: last_trial_result}
      )
      self._last_trial.complete(result_to_measurement)
      self._trial_loaded = False

  def _extract_params_from_trial(self) -> np.ndarray:
    """Returns numpy array of data from Vizier trial.

    This method updates Vizier with new information from last trial if any and
    generates a new data array x[0],..., x[num_vizier_params] with a new trial
    suggested by Vizier; neighboring datasets should be created as
    `D1= {x[0],..., x[num_vizier_params-1]} and D2={x[0],...,
    x[num_vizier_params]}.

    Returns:
      Array with Vizier suggestions for new neighboring datasets.
    """
    if not self._last_trial:
      raise ValueError(
          'Trying to extract parameters from trial but no trial is loaded.'
      )
    data = np.zeros(self._num_vizier_params)
    for point_id, val in self._last_trial.parameters.items():
      i = self._str_to_idx[point_id]
      data[i] = val
    return data

  @abc.abstractmethod
  def get_neighboring_datasets_from_vizier_params(
      self, vizier_params: np.ndarray
  ) -> interfaces.NeighboringDatasetsType:
    """Transforms a one-dimensional numpy array to neighboring datasets."""

  def __call__(
      self,
      last_trial_result: Optional[float],
  ) -> interfaces.NeighboringDatasetsType:
    """Returns numpy array of data from Vizier trial.

    This method updates Vizier with new information from last trial if any and
    generates a new data array x[0],..., x[num_vizier_params] with a new trial
    suggested by Vizier; neighboring datasets should be created as
    `D1= {x[0],..., x[num_vizier_params-1]} and D2={x[0],...,
    x[num_vizier_params]}.

    Args:
      last_trial_result: objective value from last trial (if any).

    Returns:
      Array with Vizier suggestions for new neighboring datasets.
    """
    if last_trial_result:
      self._complete_trial(last_trial_result)

    self._load_trial()

    # Load suggestion parameters into data array.
    vizier_params = self._extract_params_from_trial()
    neighboring_datasets = self.get_neighboring_datasets_from_vizier_params(
        vizier_params
    )
    return neighboring_datasets


class VizierScalarDataAddRemoveGenerator(VizierDatasetGenerator):
  """Vizier dataset generator for scalar records and add/remove neighboring.

  Vizier dataset generator that generates pairs of datasets (D1, D2) where
  D=[x_1, ..., x_n] and D2=[x_1, ..., x_{n-1}, x{n}]
  """

  def get_neighboring_datasets_from_vizier_params(
      self, vizier_params: np.ndarray
  ) -> interfaces.NeighboringDatasetsType:
    """Transforms a one-dimensional numpy array to neighboring datasets."""
    return vizier_params[:-1], vizier_params

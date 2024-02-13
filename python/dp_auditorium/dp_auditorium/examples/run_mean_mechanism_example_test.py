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

from absl.testing import absltest
import numpy as np
from vizier.service import clients

from dp_auditorium import interfaces
from dp_auditorium.configs import dataset_generator_config
from dp_auditorium.examples.run_mean_mechanism_example import mean_mechanism_report
from dp_auditorium.generators import vizier_dataset_generator


class StubVizierGenerator(
    vizier_dataset_generator.VizierScalarDataAddRemoveGenerator
):

  def get_neighboring_datasets_from_vizier_params(
      self, vizier_params: np.ndarray
  ) -> interfaces.NeighboringDatasetsType:
    return np.ones(2), np.ones(2)


class RunMeanMechanismExampleTest(absltest.TestCase):

  def test_generates_result(self):
    clients.environment_variables.servicer_use_sql_ram()
    output = mean_mechanism_report(
        0.1,
        0.1,
        1,
        lambda config: vizier_dataset_generator.VizierScalarDataAddRemoveGenerator(
            config=config
        ),
    )
    self.assertNotEmpty(str(output))


if __name__ == "__main__":
  absltest.main()

# Copyright 2023, The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl.testing import absltest
from absl.testing import parameterized
import attrs
import tree

from dp_accounting import dp_event


def assert_not_contains_attrs(structure):
  def _fn(structure):
    if attrs.has(type(structure)):
      raise AssertionError(
          'Expected structure to not contain `attrs` decorated classes, '
          f'found {structure}.'
      )
    return None

  tree.traverse(_fn, structure)


def assert_not_contains_named_tuples(structure):
  def _fn(structure):
    if isinstance(structure, dp_event.DpEventNamedTuple):
      raise AssertionError(
          'Expected structure to not contain `dp_event.DpEventNamedTuple`s, '
          f'found {structure}.'
      )
    return None

  tree.traverse(_fn, structure)


class DpEventTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('base_class', dp_event.DpEvent()),
      ('no_op', dp_event.NoOpDpEvent()),
      ('non_private', dp_event.NonPrivateDpEvent()),
      ('unsupported', dp_event.UnsupportedDpEvent()),
      ('gaussian', dp_event.GaussianDpEvent(1.0)),
      ('laplace', dp_event.LaplaceDpEvent(1.0)),
      (
          'self_composed',
          dp_event.SelfComposedDpEvent(dp_event.GaussianDpEvent(1.0), 10),
      ),
      (
          'composed',
          dp_event.ComposedDpEvent(
              [dp_event.GaussianDpEvent(1.0), dp_event.LaplaceDpEvent(1.0)]
          ),
      ),
      (
          'poisson',
          dp_event.PoissonSampledDpEvent(0.1, dp_event.GaussianDpEvent(1.0)),
      ),
      (
          'sampled_with_replacement',
          dp_event.SampledWithReplacementDpEvent(
              1000, 10, dp_event.GaussianDpEvent(1.0)
          ),
      ),
      (
          'sampled_without_replacement',
          dp_event.SampledWithoutReplacementDpEvent(
              1000, 10, dp_event.GaussianDpEvent(1.0)
          ),
      ),
      ('tree_int', dp_event.SingleEpochTreeAggregationDpEvent(1.0, 5)),
      ('tree_list', dp_event.SingleEpochTreeAggregationDpEvent(1.0, [5, 10])),
      (
          'repeat_and_select',
          dp_event.RepeatAndSelectDpEvent(
              dp_event.GaussianDpEvent(1.0), 30.0, 1.0
          ),
      ),
      (
          'complex',
          dp_event.ComposedDpEvent([
              dp_event.SingleEpochTreeAggregationDpEvent(1.0, 5),
              dp_event.PoissonSampledDpEvent(0.1, dp_event.LaplaceDpEvent(1.0)),
              dp_event.SelfComposedDpEvent(
                  dp_event.SampledWithReplacementDpEvent(
                      1000, 10, dp_event.GaussianDpEvent(1.0)
                  ),
                  50,
              ),
          ]),
      ),
      (
          'mixture_gaussian',
          dp_event.MixtureOfGaussiansDpEvent(1.0, [0, 1, 2], [0.25, 0.5, 0.25]),
      ),
  )
  def test_to_from_named_tuple(self, event):
    named_tuple = event.to_named_tuple()
    self.assertIsInstance(named_tuple, tuple)
    self.assertIsInstance(named_tuple, dp_event.DpEventNamedTuple)
    assert_not_contains_attrs(named_tuple)

    reconstructed = dp_event.DpEvent.from_named_tuple(named_tuple)
    assert_not_contains_named_tuples(reconstructed)
    self.assertEqual(event, reconstructed)


if __name__ == '__main__':
  absltest.main()

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
from collections.abc import Iterator, Sequence
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import attrs
import tree

from dp_accounting import dp_event


@attrs.frozen
class ContainerDpEvent(dp_event.DpEvent):
  """Event with `str`, non-`list` sequence and mapping fields, for testing."""

  name: str
  events: tuple[dp_event.DpEvent, ...]
  table: dict[str, dp_event.DpEvent]


@attrs.frozen
class _ArgSpec:
  default: Any
  valid: Sequence[Any] = ()
  invalid: Sequence[Any] = ()


_EVENT = _ArgSpec(default=dp_event.NoOpDpEvent())
_NONNEG = _ArgSpec(default=1, valid=(0,), invalid=(-0.1, float('nan')))
_GEQ_ONE = _ArgSpec(default=1, valid=(1,), invalid=(0, 0.9, -1, float('nan')))
_ZERO_TO_ONE = _ArgSpec(
    default=0.5, valid=(0.0, 1.0), invalid=(-0.1, 1.1, float('nan'))
)
_BOUNDED_BY_ONE = _ArgSpec(
    default=0, valid=(0, 1), invalid=(-1, 2, float('nan'))
)

_EVENT_ARG_SPECS: Sequence[
    tuple[type[dp_event.DpEvent], Sequence[_ArgSpec]]
] = (
    (dp_event.RandomizedResponseDpEvent, (_ZERO_TO_ONE, _GEQ_ONE)),
    (dp_event.EpsilonDeltaDpEvent, (_NONNEG, _ZERO_TO_ONE)),
    (dp_event.GaussianDpEvent, (_NONNEG,)),
    (dp_event.LaplaceDpEvent, (_NONNEG,)),
    (dp_event.DiscreteLaplaceDpEvent, (_NONNEG, _NONNEG)),
    (
        dp_event.DiscreteGaussianDpEvent,
        (
            _NONNEG,
            _NONNEG,
            _ArgSpec(
                default=None, valid=(None, 1), invalid=(0, -1, float('nan'))
            ),
        ),
    ),
    (dp_event.SelfComposedDpEvent, (_EVENT, _NONNEG)),
    (dp_event.PoissonSampledDpEvent, (_ZERO_TO_ONE, _EVENT)),
    (
        dp_event.SampledWithReplacementDpEvent,
        (
            _GEQ_ONE,
            _ArgSpec(default=1, valid=(0, 5), invalid=(-1, float('nan'))),
            _EVENT,
        ),
    ),
    (
        dp_event.SampledWithoutReplacementDpEvent,
        (_GEQ_ONE, _BOUNDED_BY_ONE, _EVENT),
    ),
    (
        dp_event.SingleEpochTreeAggregationDpEvent,
        (
            _NONNEG,
            _ArgSpec(
                default=1,
                valid=(0, [0]),
                invalid=(-1, float('nan'), [], [5, -1], [5, float('nan')]),
            ),
        ),
    ),
    (
        dp_event.RepeatAndSelectDpEvent,
        (
            _EVENT,
            _GEQ_ONE,
            _ArgSpec(
                default=1.0,
                valid=(0.0, float('inf')),
                invalid=(-0.1, float('nan')),
            ),
        ),
    ),
    (
        dp_event.MixtureOfGaussiansDpEvent,
        (
            _NONNEG,
            _ArgSpec(
                default=[0.0, 1.0],
                valid=([0.0, 0.0],),
                invalid=([], [1.0], [-1.0, 1.0], [float('nan'), 1.0]),
            ),
            _ArgSpec(
                default=[0.5, 0.5],
                valid=([0.0, 1.0],),
                invalid=([], [-0.1, 1.1], [float('nan'), 0.5], [0.3, 0.3]),
            ),
        ),
    ),
    (dp_event.ZCDpEvent, (_NONNEG, _NONNEG)),
    (dp_event.ExponentialMechanismDpEvent, (_NONNEG,)),
    (dp_event.PermuteAndFlipDpEvent, (_NONNEG,)),
    (
        dp_event.TruncatedSubsampledGaussianDpEvent,
        (_NONNEG, _ZERO_TO_ONE, _NONNEG, _NONNEG),
    ),
    (dp_event.RandomAllocationDpEvent, (_EVENT, _BOUNDED_BY_ONE, _GEQ_ONE)),
)


def _generate_test_cases(
    valid: bool,
) -> Iterator[tuple[str, type[dp_event.DpEvent], tuple[Any, ...]]]:
  for event_cls, specs in _EVENT_ARG_SPECS:
    fields = attrs.fields(event_cls)
    defaults = [s.default for s in specs]
    for i, (field, spec) in enumerate(zip(fields, specs, strict=True)):
      values = spec.valid if valid else spec.invalid
      for val in values:
        args = tuple(val if j == i else d for j, d in enumerate(defaults))
        name = f'{event_cls.__name__}_{field.name}_{val}'
        yield (name, event_cls, args)


def assert_not_contains_attrs(structure: Any) -> None:
  def _fn(structure: Any) -> None:
    if attrs.has(type(structure)):
      raise AssertionError(
          'Expected structure to not contain `attrs` decorated classes, '
          f'found {structure}.'
      )

  tree.traverse(_fn, structure)


def assert_not_contains_named_tuples(structure: Any) -> None:
  def _fn(structure: Any) -> None:
    if isinstance(structure, dp_event.DpEventNamedTuple):
      raise AssertionError(
          'Expected structure to not contain `dp_event.DpEventNamedTuple`s, '
          f'found {structure}.'
      )

  tree.traverse(_fn, structure)


class DpEventTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('base_class', dp_event.DpEvent()),
      ('no_op', dp_event.NoOpDpEvent()),
      ('non_private', dp_event.NonPrivateDpEvent()),
      ('unsupported', dp_event.UnsupportedDpEvent()),
      ('epsilon_delta', dp_event.EpsilonDeltaDpEvent(1.0, 0.1)),
      ('gaussian', dp_event.GaussianDpEvent(1.0)),
      ('laplace', dp_event.LaplaceDpEvent(1.0)),
      ('dlaplace', dp_event.DiscreteLaplaceDpEvent(0.5, 1)),
      ('dgaussian', dp_event.DiscreteGaussianDpEvent(sigma=10.0)),
      (
          'dgaussian_dim',
          dp_event.DiscreteGaussianDpEvent(
              sigma=10.0, sensitivity=2.0, dimension=1
          ),
      ),
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
          'parallel_composed',
          dp_event.ParallelComposedDpEvent(
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
      (
          'exponential_mechanism',
          dp_event.ExponentialMechanismDpEvent(1.0),
      ),
      (
          'permute_and_flip',
          dp_event.PermuteAndFlipDpEvent(1.0),
      ),
      (
          'zcdp',
          dp_event.ZCDpEvent(1.0, 2.0),
      ),
      (
          'truncated_subsampled_gaussian',
          dp_event.TruncatedSubsampledGaussianDpEvent(
              10000,
              0.1,
              1000,
              1.0,
          ),
      ),
      (
          'random_allocation',
          dp_event.RandomAllocationDpEvent(
              event=dp_event.GaussianDpEvent(2.0),
              num_selected=10,
              num_steps=100,
          ),
      ),
      (
          'containers',
          ContainerDpEvent(
              name='containers',
              events=(dp_event.GaussianDpEvent(1.0), dp_event.NoOpDpEvent()),
              table={'laplace': dp_event.LaplaceDpEvent(1.0)},
          ),
      ),
  )
  def test_to_from_named_tuple(self, event: dp_event.DpEvent):
    named_tuple = event.to_named_tuple()
    self.assertIsInstance(named_tuple, tuple)
    self.assertIsInstance(named_tuple, dp_event.DpEventNamedTuple)
    assert_not_contains_attrs(named_tuple)

    reconstructed = dp_event.DpEvent.from_named_tuple(named_tuple)
    assert_not_contains_named_tuples(reconstructed)
    self.assertEqual(event, reconstructed)

  def test_to_named_tuple_preserves_container_types(self):
    event = ContainerDpEvent(
        name='containers',
        events=(dp_event.GaussianDpEvent(1.0),),
        table={'laplace': dp_event.LaplaceDpEvent(1.0)},
    )
    named_tuple: Any = event.to_named_tuple()
    self.assertEqual(named_tuple.name, 'containers')
    self.assertIsInstance(named_tuple.events, tuple)
    self.assertIsInstance(named_tuple.table, dict)
    self.assertEqual(named_tuple.events[0].class_name, 'GaussianDpEvent')
    self.assertEqual(named_tuple.table['laplace'].class_name, 'LaplaceDpEvent')

  @parameterized.named_parameters(*_generate_test_cases(valid=True))
  def test_valid_boundary_parameters(self, event_cls, args):
    event_cls(*args)

  @parameterized.named_parameters(*_generate_test_cases(valid=False))
  def test_invalid_parameters_raise_value_error(self, event_cls, args):
    with self.assertRaises(ValueError):
      event_cls(*args)


if __name__ == '__main__':
  absltest.main()

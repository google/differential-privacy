# Copyright 2021, The TensorFlow Authors.
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
from dp_accounting import dp_event
from dp_accounting import dp_event_builder
from dp_accounting.pld import pld_privacy_accountant
from dp_accounting.rdp import rdp_privacy_accountant

_gaussian_event = dp_event.GaussianDpEvent(1.0)
_laplace_event = dp_event.LaplaceDpEvent(1.0)
_poisson_event = dp_event.PoissonSampledDpEvent(0.1, _gaussian_event)
_self_composed_event = dp_event.SelfComposedDpEvent(_gaussian_event, 3)


class DpEventBuilderTest(absltest.TestCase):

  def test_no_op(self):
    builder = dp_event_builder.DpEventBuilder()
    self.assertEqual(dp_event.NoOpDpEvent(), builder.build())

  def test_single_gaussian(self):
    builder = dp_event_builder.DpEventBuilder()
    builder.compose(_gaussian_event)
    self.assertEqual(_gaussian_event, builder.build())

  def test_single_laplace(self):
    builder = dp_event_builder.DpEventBuilder()
    builder.compose(_laplace_event)
    self.assertEqual(_laplace_event, builder.build())

  def test_compose_no_op(self):
    builder = dp_event_builder.DpEventBuilder()
    builder.compose(dp_event.NoOpDpEvent())
    builder.compose(_gaussian_event)
    builder.compose(dp_event.NoOpDpEvent())
    self.assertEqual(_gaussian_event, builder.build())

  def test_compose_self(self):
    builder = dp_event_builder.DpEventBuilder()
    builder.compose(_gaussian_event)
    builder.compose(_gaussian_event, 2)
    self.assertEqual(_self_composed_event, builder.build())

  def test_compose_heterogenous(self):
    builder = dp_event_builder.DpEventBuilder()
    builder.compose(_poisson_event)
    builder.compose(_gaussian_event)
    builder.compose(_gaussian_event, 2)
    builder.compose(_poisson_event)
    expected_event = dp_event.ComposedDpEvent(
        [_poisson_event, _self_composed_event, _poisson_event])
    self.assertEqual(expected_event, builder.build())

  def test_compose_composed(self):
    builder = dp_event_builder.DpEventBuilder()
    composed_event = dp_event.ComposedDpEvent(
        [_gaussian_event, _poisson_event, _self_composed_event])
    builder.compose(_gaussian_event)
    builder.compose(composed_event)
    builder.compose(composed_event, 2)
    builder.compose(_poisson_event)
    builder.compose(_poisson_event)
    expected_event = dp_event.ComposedDpEvent([
        _gaussian_event,
        dp_event.SelfComposedDpEvent(composed_event, 3),
        dp_event.SelfComposedDpEvent(_poisson_event, 2)
    ])
    self.assertEqual(expected_event, builder.build())


class CanonicalizeDpEventTest(absltest.TestCase):

  _ACCOUNTANT_FACTORIES = (
      lambda: rdp_privacy_accountant.RdpAccountant(orders=[2, 5, 10, 50]),
      pld_privacy_accountant.PLDAccountant,
  )
  _DELTA = 1e-5

  def _assert_canonical(self, event, expected, *, merge_gaussians=False):
    """Assert canonical form matches expected and preserves epsilon."""
    result = dp_event_builder.canonicalize(
        event, merge_gaussians=merge_gaussians
    )
    self.assertEqual(expected, result)
    for make_accountant in self._ACCOUNTANT_FACTORIES:
      accountant = make_accountant()
      if not accountant.supports(event):
        continue
      eps_orig = make_accountant().compose(event).get_epsilon(self._DELTA)
      eps_canon = make_accountant().compose(result).get_epsilon(self._DELTA)
      self.assertAlmostEqual(
          eps_orig,
          eps_canon,
          places=6,
          msg=(
              f'{type(accountant).__name__}: epsilon mismatch for '
              f'{event!r} vs {result!r}'
          ),
      )

  def test_leaf_unchanged(self):
    """Leaf events are returned as-is."""
    for event in [
        dp_event.NoOpDpEvent(),
        dp_event.NonPrivateDpEvent(),
        _gaussian_event,
        _laplace_event,
    ]:
      self.assertEqual(event, dp_event_builder.canonicalize(event))

  def test_flatten_nested_composed(self):
    event = dp_event.ComposedDpEvent([
        _gaussian_event,
        dp_event.ComposedDpEvent([_laplace_event, _gaussian_event]),
    ])
    expected = dp_event.ComposedDpEvent([
        dp_event.SelfComposedDpEvent(_gaussian_event, 2),
        _laplace_event,
    ])
    self._assert_canonical(event, expected)

  def test_unwrap_single_element_composed(self):
    event = dp_event.ComposedDpEvent([_gaussian_event])
    self._assert_canonical(event, _gaussian_event)

  def test_unwrap_self_composed_count_one(self):
    event = dp_event.SelfComposedDpEvent(_gaussian_event, 1)
    self._assert_canonical(event, _gaussian_event)

  def test_collapse_nested_self_composed(self):
    event = dp_event.SelfComposedDpEvent(
        dp_event.SelfComposedDpEvent(_gaussian_event, 3), 5
    )
    expected = dp_event.SelfComposedDpEvent(_gaussian_event, 15)
    self._assert_canonical(event, expected)

  def test_drop_noop_from_composition(self):
    event = dp_event.ComposedDpEvent([
        dp_event.NoOpDpEvent(),
        _gaussian_event,
        dp_event.NoOpDpEvent(),
    ])
    self._assert_canonical(event, _gaussian_event)

  def test_all_noops_becomes_noop(self):
    event = dp_event.ComposedDpEvent(
        [dp_event.NoOpDpEvent(), dp_event.NoOpDpEvent()]
    )
    self._assert_canonical(event, dp_event.NoOpDpEvent())

  def test_self_composed_noop(self):
    event = dp_event.SelfComposedDpEvent(dp_event.NoOpDpEvent(), 100)
    self._assert_canonical(event, dp_event.NoOpDpEvent())

  def test_group_non_adjacent_identical(self):
    """Non-adjacent identical events are grouped together."""
    event = dp_event.ComposedDpEvent(
        [_gaussian_event, _laplace_event, _gaussian_event]
    )
    expected = dp_event.ComposedDpEvent([
        dp_event.SelfComposedDpEvent(_gaussian_event, 2),
        _laplace_event,
    ])
    self._assert_canonical(event, expected)

  def test_group_adjacent_identical(self):
    event = dp_event.ComposedDpEvent(
        [_gaussian_event, _gaussian_event, _gaussian_event]
    )
    expected = dp_event.SelfComposedDpEvent(_gaussian_event, 3)
    self._assert_canonical(event, expected)

  def test_canonical_sort_order(self):
    """Events are sorted by (type_name, repr) for canonical ordering."""
    # Laplace sorts before Poisson by type name.
    event = dp_event.ComposedDpEvent([_poisson_event, _laplace_event])
    expected = dp_event.ComposedDpEvent([_laplace_event, _poisson_event])
    self._assert_canonical(event, expected)

  def test_idempotent(self):
    event = dp_event.ComposedDpEvent([
        _poisson_event,
        _gaussian_event,
        dp_event.ComposedDpEvent([_laplace_event, _gaussian_event]),
        _poisson_event,
    ])
    once = dp_event_builder.canonicalize(event)
    twice = dp_event_builder.canonicalize(once)
    self.assertEqual(once, twice)

  def test_recurse_into_poisson_sampled(self):
    """Canonicalizes the inner event of PoissonSampledDpEvent."""
    inner = dp_event.SelfComposedDpEvent(_gaussian_event, 1)
    event = dp_event.PoissonSampledDpEvent(0.1, inner)
    expected = dp_event.PoissonSampledDpEvent(0.1, _gaussian_event)
    self._assert_canonical(event, expected)

  def test_recurse_into_repeat_and_select(self):
    inner = dp_event.ComposedDpEvent([_gaussian_event])
    event = dp_event.RepeatAndSelectDpEvent(inner, mean=10.0, shape=1.0)
    expected = dp_event.RepeatAndSelectDpEvent(
        _gaussian_event, mean=10.0, shape=1.0
    )
    self._assert_canonical(event, expected)

  def test_merge_gaussians_basic(self):
    event = dp_event.ComposedDpEvent([
        dp_event.GaussianDpEvent(2.0),
        dp_event.GaussianDpEvent(3.0),
    ])
    # sigma_eff = 1 / sqrt(1/4 + 1/9) = 6 / sqrt(13)
    expected = dp_event.GaussianDpEvent((1 / 4 + 1 / 9) ** -0.5)
    self._assert_canonical(event, expected, merge_gaussians=True)

  def test_merge_gaussians_with_counts(self):
    event = dp_event.ComposedDpEvent([
        dp_event.SelfComposedDpEvent(dp_event.GaussianDpEvent(2.0), 3),
        dp_event.GaussianDpEvent(4.0),
    ])
    # sigma_eff = 1 / sqrt(3/4 + 1/16)
    expected = dp_event.GaussianDpEvent((3 / 4 + 1 / 16) ** -0.5)
    self._assert_canonical(event, expected, merge_gaussians=True)

  def test_merge_gaussians_self_composed(self):
    """A bare SelfComposed Gaussian is collapsed to a single Gaussian."""
    event = dp_event.SelfComposedDpEvent(dp_event.GaussianDpEvent(2.0), 4)
    # sigma_eff = 2.0 / sqrt(4) = 1.0
    expected = dp_event.GaussianDpEvent(1.0)
    self._assert_canonical(event, expected, merge_gaussians=True)

  def test_merge_gaussians_preserves_non_gaussians(self):
    event = dp_event.ComposedDpEvent([
        dp_event.GaussianDpEvent(1.0),
        _laplace_event,
        dp_event.GaussianDpEvent(1.0),
    ])
    result = dp_event_builder.canonicalize(event, merge_gaussians=True)
    self.assertIsInstance(result, dp_event.ComposedDpEvent)
    self.assertLen(result.events, 2)  # merged Gaussian + Laplace

  def test_merge_gaussians_does_not_cross_sampling_boundary(self):
    """Gaussians inside PoissonSampledDpEvent are not merged at outer level."""
    event = dp_event.ComposedDpEvent([
        dp_event.PoissonSampledDpEvent(0.1, dp_event.GaussianDpEvent(1.0)),
        dp_event.PoissonSampledDpEvent(0.1, dp_event.GaussianDpEvent(1.0)),
    ])
    expected = dp_event.SelfComposedDpEvent(
        dp_event.PoissonSampledDpEvent(0.1, dp_event.GaussianDpEvent(1.0)), 2
    )
    self._assert_canonical(event, expected, merge_gaussians=True)

  def test_merge_gaussians_off_by_default(self):
    event = dp_event.ComposedDpEvent([
        dp_event.GaussianDpEvent(1.0),
        dp_event.GaussianDpEvent(2.0),
    ])
    result = dp_event_builder.canonicalize(event)
    # Without merge_gaussians, distinct Gaussians remain separate.
    self.assertIsInstance(result, dp_event.ComposedDpEvent)
    self.assertLen(result.events, 2)

  def test_no_gaussians_merge_is_noop(self):
    event = dp_event.ComposedDpEvent([_laplace_event, _poisson_event])
    result_without = dp_event_builder.canonicalize(event)
    result_with = dp_event_builder.canonicalize(event, merge_gaussians=True)
    self.assertEqual(result_without, result_with)

  def test_deeply_nested(self):
    """Deeply nested composition is fully flattened and grouped."""
    event = dp_event.ComposedDpEvent([
        dp_event.ComposedDpEvent([
            _gaussian_event,
            dp_event.ComposedDpEvent([_laplace_event, _gaussian_event]),
        ]),
        _gaussian_event,
    ])
    expected = dp_event.ComposedDpEvent([
        dp_event.SelfComposedDpEvent(_gaussian_event, 3),
        _laplace_event,
    ])
    self._assert_canonical(event, expected)


if __name__ == '__main__':
  absltest.main()

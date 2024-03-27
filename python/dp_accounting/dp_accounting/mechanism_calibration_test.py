# Copyright 2022, The TensorFlow Authors.
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
"""Tests for mechanism_calibration."""

from absl.testing import absltest
from absl.testing import parameterized
import attr
import numpy as np

from dp_accounting import dp_event
from dp_accounting import mechanism_calibration
from dp_accounting import privacy_accountant


@attr.define
class FakeEvent(dp_event.DpEvent):
  param: float


class FakeAccountant(privacy_accountant.PrivacyAccountant):

  def __init__(self, value_to_epsilon):
    super().__init__(
        privacy_accountant.NeighboringRelation.ADD_OR_REMOVE_ONE)
    self._value = 0.0
    self._value_to_epsilon = value_to_epsilon

  def _maybe_compose(self, event: dp_event.DpEvent, count: int,
                     do_compose: bool):
    self._value = event.param

  def get_epsilon(self, target_delta: float) -> float:
    return self._value_to_epsilon(self._value)


class MechanismCalibrationTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('identity', lambda x: x, 2.0),
      ('4_minus_x', lambda x: 4 - x, 2.0),
      ('square', np.square, np.sqrt(2)),
      ('cbrt', np.cbrt, 8.0),
      ('cubic', lambda x: (x - 5) ** 3 + 2, 5),
      ('trig_one', lambda x: np.cos(x / 3) + 2, 3 * np.pi / 2),
      ('trig_two', lambda x: np.sin(x - 5) + (x + 3) / 4, 5),
      ('trig_three', lambda x: (13 - x) / 4 - np.sin(x - 5), 5),
  )
  def test_basic_inversion(self, eps_fn, expected):

    value = mechanism_calibration.calibrate_dp_mechanism(
        make_fresh_accountant=lambda: FakeAccountant(eps_fn),
        make_event_from_param=FakeEvent,
        target_epsilon=2,
        target_delta=0,
        bracket_interval=mechanism_calibration.ExplicitBracketInterval(0, 10),
        tol=1e-12,
    )

    self.assertIsInstance(value, float)
    self.assertAlmostEqual(value, expected)

    accountant = FakeAccountant(eps_fn)
    accountant.compose(FakeEvent(value))
    epsilon = accountant.get_epsilon(0)
    self.assertLessEqual(epsilon, 2)

  @parameterized.named_parameters(
      ('neg_one_pos_one', lambda x: -1 if x < 0 else 1),
      ('pos_one_neg_one', lambda x: 1 if x < 0 else -1),
      ('sawtooth', lambda x: x - 1 if x < 0 else x + 1),
      ('neg_sawtooth', lambda x: -2 - x if x < 0 else 2 - x),
      ('x_plus_two', lambda x: x + 2 if x < 0 else x - 2),
      ('neg_one_minus_x', lambda x: 1 - x if x < 0 else -1 - x),
  )
  def test_discontinuous(self, eps_fn):
    value = mechanism_calibration.calibrate_dp_mechanism(
        make_fresh_accountant=lambda: FakeAccountant(eps_fn),
        make_event_from_param=FakeEvent,
        target_epsilon=0,
        target_delta=0,
        bracket_interval=mechanism_calibration.ExplicitBracketInterval(-1, 1),
        tol=1e-12,
    )

    self.assertIsInstance(value, float)
    self.assertAlmostEqual(value, 0)

    accountant = FakeAccountant(eps_fn)
    accountant.compose(FakeEvent(value))
    epsilon = accountant.get_epsilon(0)
    self.assertLessEqual(epsilon, 0)

  @parameterized.named_parameters(
      ('x_minus_2', lambda x: x - 2, 0),
      ('x_minus_2_1', lambda x: x - 2.1, -0.1),
      ('x_minus_2_9', lambda x: x - 2.9, -0.9),
      ('2_minus_x', lambda x: 2 - x, 0),
      ('1_9_minus_x', lambda x: 1.9 - x, -0.1),
      ('1_1_minus_x', lambda x: 1.1 - x, -0.9),
  )
  def test_discrete(self, eps_fn, expected_eps):
    value = mechanism_calibration.calibrate_dp_mechanism(
        make_fresh_accountant=lambda: FakeAccountant(eps_fn),
        make_event_from_param=FakeEvent,
        target_epsilon=0,
        target_delta=0,
        bracket_interval=mechanism_calibration.ExplicitBracketInterval(0, 5),
        discrete=True,
    )

    self.assertIsInstance(value, int)
    self.assertEqual(value, 2)

    accountant = FakeAccountant(eps_fn)
    accountant.compose(FakeEvent(value))
    epsilon = accountant.get_epsilon(0)
    self.assertAlmostEqual(epsilon, expected_eps)

  @parameterized.named_parameters(
      ('identity', lambda x: x, -1, -0.5),
      ('neg_x', lambda x: -x, -1, -0.5),
      ('exp_minus_2', lambda x: np.exp(x) - 2, 0, 0.1),
      ('1_minus_sqrt_x', lambda x: 1 - np.sqrt(x), 0, 0.1),
      ('log_minus_20', lambda x: np.log(x) - 20, 1, 2),
  )
  def test_search_for_explicit_bracket_interval(
      self, epsilon_gap, lower, guess
  ):
    interval = mechanism_calibration._search_for_explicit_bracket_interval(
        mechanism_calibration.LowerEndpointAndGuess(lower, guess),
        epsilon_gap,
    )
    lower_value = epsilon_gap(interval.endpoint_1)
    upper_value = epsilon_gap(interval.endpoint_2)
    self.assertLessEqual(lower_value * upper_value, 0)

  def test_raises_unknown_bracket_interval_type(self):
    class UnknownBracketInterval(mechanism_calibration.BracketInterval):
      pass

    with self.assertRaisesRegex(TypeError, 'Unrecognized bracket_interval'):
      mechanism_calibration.calibrate_dp_mechanism(
          make_fresh_accountant=lambda: FakeAccountant(lambda x: x),
          make_event_from_param=FakeEvent,
          target_epsilon=1.0,
          target_delta=0,
          bracket_interval=UnknownBracketInterval(),
      )

  @parameterized.named_parameters(
      ('square', np.square, np.sqrt),
      ('cbrt', np.cbrt, lambda x: x**3),
      ('80th_power', lambda x: x**80, lambda x: x ** (1 / 80)),
      ('80th_root', lambda x: x ** (1 / 80), lambda x: x**80),
      (
          'cos',
          lambda x: np.cos(np.pi / 2 * x),
          lambda x: 2 / np.pi * np.arccos(x),
      ),
  )
  def test_bisect_finds_root_with_negative_value(self, function, inv_function):
    # Repeat the test many times to ensure negative value always returned.
    repetitions = 20
    generator = np.random.default_rng(seed=0xBAD5EED)
    for target in generator.uniform(size=repetitions):
      # pylint: disable=cell-var-from-loop
      root = mechanism_calibration._bisect(
          lambda x: function(x) - target, 0, 1, 1e-12
      )
      # pylint: enable=cell-var-from-loop
      expected = inv_function(target)
      self.assertAlmostEqual(root, expected)
      self.assertLessEqual(function(root), target)

  @parameterized.named_parameters(
      ('both_positive', lambda x: 1),
      ('both_negative', lambda x: -1),
  )
  def test_bisect_raises_values_same_sign(self, function):
    with self.assertRaisesRegex(ValueError, 'opposite signs'):
      mechanism_calibration._bisect(function, 0, 1, 1e-12)

  def test_raises_mfa_not_callable(self):
    with self.assertRaisesRegex(
        TypeError, 'make_fresh_accountant must be callable'
    ):
      mechanism_calibration.calibrate_dp_mechanism(
          make_fresh_accountant='not a callable',
          make_event_from_param=FakeEvent,
          target_epsilon=1.0,
          target_delta=0,
          bracket_interval=mechanism_calibration.ExplicitBracketInterval(0, 5),
      )

  def test_raises_mefp_not_callable(self):
    with self.assertRaisesRegex(
        TypeError, 'make_event_from_param must be callable'
    ):
      mechanism_calibration.calibrate_dp_mechanism(
          make_fresh_accountant=lambda: FakeAccountant(lambda x: x),
          make_event_from_param='not a callable',
          target_epsilon=1.0,
          target_delta=0,
          bracket_interval=mechanism_calibration.ExplicitBracketInterval(0, 5),
      )

  def test_raises_target_epsilon_negative(self):
    with self.assertRaisesRegex(ValueError, 'nonnegative'):
      mechanism_calibration.calibrate_dp_mechanism(
          make_fresh_accountant=lambda: FakeAccountant(lambda x: x),
          make_event_from_param=FakeEvent,
          target_epsilon=-1.0,
          target_delta=0,
          bracket_interval=mechanism_calibration.ExplicitBracketInterval(0, 5),
      )

  def test_raises_target_delta_out_of_range(self):
    with self.assertRaisesRegex(ValueError, 'in range'):
      mechanism_calibration.calibrate_dp_mechanism(
          make_fresh_accountant=lambda: FakeAccountant(lambda x: x),
          make_event_from_param=FakeEvent,
          target_epsilon=0.0,
          target_delta=-0.1,
          bracket_interval=mechanism_calibration.ExplicitBracketInterval(0, 5),
      )

    with self.assertRaisesRegex(ValueError, 'in range'):
      mechanism_calibration.calibrate_dp_mechanism(
          make_fresh_accountant=lambda: FakeAccountant(lambda x: x),
          make_event_from_param=FakeEvent,
          target_epsilon=0.0,
          target_delta=1.1,
          bracket_interval=mechanism_calibration.ExplicitBracketInterval(0, 5),
      )

  def test_bad_bracket_interval(self):
    with self.assertRaisesRegex(ValueError, 'did not bracket a solution'):
      mechanism_calibration.calibrate_dp_mechanism(
          make_fresh_accountant=lambda: FakeAccountant(lambda x: x),
          make_event_from_param=FakeEvent,
          target_epsilon=1.0,
          target_delta=0.0,
          bracket_interval=mechanism_calibration.ExplicitBracketInterval(2, 5),
      )

    with self.assertRaisesRegex(ValueError, 'did not bracket a solution'):
      mechanism_calibration.calibrate_dp_mechanism(
          make_fresh_accountant=lambda: FakeAccountant(lambda x: x),
          make_event_from_param=FakeEvent,
          target_epsilon=1.0,
          target_delta=0.0,
          bracket_interval=mechanism_calibration.ExplicitBracketInterval(-2, 0),
      )

    with self.assertRaisesRegex(ValueError, 'must be less than'):
      mechanism_calibration.calibrate_dp_mechanism(
          make_fresh_accountant=lambda: FakeAccountant(lambda x: x),
          make_event_from_param=FakeEvent,
          target_epsilon=1.0,
          target_delta=0.0,
          bracket_interval=mechanism_calibration.LowerEndpointAndGuess(2, 0),
      )

  def test_negative_tol(self):
    with self.assertRaisesRegex(ValueError, 'tol'):
      mechanism_calibration.calibrate_dp_mechanism(
          make_fresh_accountant=lambda: FakeAccountant(lambda x: x),
          make_event_from_param=FakeEvent,
          target_epsilon=1.0,
          target_delta=0.0,
          bracket_interval=mechanism_calibration.LowerEndpointAndGuess(0, 1),
          tol=-1,
      )

  def test_no_bracket_interval_found(self):
    with self.assertRaises(mechanism_calibration.NoBracketIntervalFoundError):
      mechanism_calibration.calibrate_dp_mechanism(
          make_fresh_accountant=lambda: FakeAccountant(lambda x: x),
          make_event_from_param=FakeEvent,
          target_epsilon=1.0e10,
          target_delta=0.0,
          bracket_interval=mechanism_calibration.LowerEndpointAndGuess(0, 1),
      )

  def test_nonempty_accountant(self):
    def make_fresh_accountant():
      accountant = FakeAccountant(lambda x: x)
      accountant.compose(FakeEvent(1.0))
      return accountant

    with self.assertRaises(mechanism_calibration.NonEmptyAccountantError):
      mechanism_calibration.calibrate_dp_mechanism(
          make_fresh_accountant=make_fresh_accountant,
          make_event_from_param=FakeEvent,
          target_epsilon=0.5,
          target_delta=0.0,
          bracket_interval=mechanism_calibration.ExplicitBracketInterval(0, 1),
      )


if __name__ == '__main__':
  absltest.main()

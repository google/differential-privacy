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
      ('log_minus_20', lambda x: np.log(x) - 20, 0, 1),
      ('log_plus_20', lambda x: np.log(x) + 20, 0, 1)
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

  @parameterized.named_parameters(
      ('bad_on_right', lambda x: np.log((1.5 - x) / 1.4)),  #  zero at 0.1.
      ('bad_on_left', lambda x: 1 / (x - 0.5) - 1 / 20),  # zero at 20.5.
  )
  def test_search_for_explicit_bracket_interval_bad_on_one_side(
      self, epsilon_gap
  ):
    interval = mechanism_calibration._search_for_explicit_bracket_interval(
        mechanism_calibration.LowerEndpointAndGuess(0, 1),
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

  # -------- Auto bracket tests --------

  @parameterized.named_parameters(
      ('identity_target_5', lambda x: x - 5),
      ('decreasing_target_5', lambda x: 5 - x),
      ('log_target_20', lambda x: np.log(x) - 20),
      ('reciprocal_target_0', lambda x: 1 / x - 1),
  )
  def test_find_auto_bracket(self, epsilon_gap):
    interval = mechanism_calibration._find_auto_bracket(epsilon_gap)
    lower_value = epsilon_gap(interval.endpoint_1)
    upper_value = epsilon_gap(interval.endpoint_2)
    self.assertLessEqual(lower_value * upper_value, 0)

  def test_find_auto_bracket_no_sign_change(self):
    with self.assertRaises(mechanism_calibration.NoBracketIntervalFoundError):
      mechanism_calibration._find_auto_bracket(lambda x: 1.0)

  def test_auto_bracket_skips_inf(self):
    """Auto bracket skips inf values and finds bracket at finite points."""

    def epsilon_gap(x):
      if x < 0.01:
        return float('inf')
      return 10 / x - 5  # root at x=2

    interval = mechanism_calibration._find_auto_bracket(epsilon_gap)
    lower_value = epsilon_gap(interval.endpoint_1)
    upper_value = epsilon_gap(interval.endpoint_2)
    self.assertLessEqual(lower_value * upper_value, 0)

  def test_auto_bracket_skips_exceptions(self):
    """Auto bracket skips points that raise and continues searching."""

    def epsilon_gap(x):
      if x < 0.1:
        raise ValueError('invalid')
      return 10 / x - 5  # root at x=2

    interval = mechanism_calibration._find_auto_bracket(epsilon_gap)
    lower_value = epsilon_gap(interval.endpoint_1)
    upper_value = epsilon_gap(interval.endpoint_2)
    self.assertLessEqual(lower_value * upper_value, 0)

  # -------- Auto bracket with fake accountant --------

  @parameterized.named_parameters(
      ('identity_target_1', lambda x: x, 1.0),
      ('identity_target_0_1', lambda x: x, 0.1),
      ('identity_target_100', lambda x: x, 100),
      ('decreasing_target_1', lambda x: 10 / x, 1.0),
      ('decreasing_target_100', lambda x: 10 / x, 100),
  )
  def test_calibrate_auto_bracket_fake(self, eps_fn, target_eps):
    """Test calibration with auto-bracket (no explicit interval)."""
    value = mechanism_calibration.calibrate_dp_mechanism(
        make_fresh_accountant=lambda: FakeAccountant(eps_fn),
        make_event_from_param=FakeEvent,
        target_epsilon=target_eps,
        target_delta=0,
    )
    accountant = FakeAccountant(eps_fn)
    accountant.compose(FakeEvent(value))
    self.assertLessEqual(accountant.get_epsilon(0), target_eps)

  # -------- Auto bracket with extreme epsilon values --------
  # These tests simulate the behavior of real accountants (epsilon decreasing
  # as noise parameter increases) at extreme target epsilon values, which is
  # the scenario that previously required manual bracket specification.

  @parameterized.named_parameters(
      ('very_small_eps', 1 / 1e6),
      ('small_eps', 0.01),
      ('medium_eps', 1.0),
      ('large_eps', 100.0),
      ('very_large_eps', 1e6),
  )
  def test_calibrate_auto_bracket_decreasing_extreme(self, target_eps):
    """Auto-bracket with decreasing epsilon = 1/x at extreme values."""
    value = mechanism_calibration.calibrate_dp_mechanism(
        make_fresh_accountant=lambda: FakeAccountant(lambda x: 1 / x),
        make_event_from_param=FakeEvent,
        target_epsilon=target_eps,
        target_delta=0,
    )
    accountant = FakeAccountant(lambda x: 1 / x)
    accountant.compose(FakeEvent(value))
    self.assertLessEqual(accountant.get_epsilon(0), target_eps)

  @parameterized.named_parameters(
      ('very_small_eps', 1 / 1e6),
      ('small_eps', 0.01),
      ('medium_eps', 1.0),
      ('large_eps', 100.0),
      ('very_large_eps', 1e6),
  )
  def test_calibrate_auto_bracket_increasing_extreme(self, target_eps):
    """Auto-bracket with increasing epsilon = x at extreme values."""
    value = mechanism_calibration.calibrate_dp_mechanism(
        make_fresh_accountant=lambda: FakeAccountant(lambda x: x),
        make_event_from_param=FakeEvent,
        target_epsilon=target_eps,
        target_delta=0,
    )
    accountant = FakeAccountant(lambda x: x)
    accountant.compose(FakeEvent(value))
    self.assertLessEqual(accountant.get_epsilon(0), target_eps)

  @parameterized.named_parameters(
      ('small_eps', 0.01),
      ('medium_eps', 1.0),
      ('large_eps', 100.0),
  )
  def test_calibrate_auto_bracket_gaussian_like(self, target_eps):
    """Auto-bracket with epsilon = 1/(2*x^2), mimicking Gaussian."""
    value = mechanism_calibration.calibrate_dp_mechanism(
        make_fresh_accountant=lambda: FakeAccountant(lambda x: 1 / (2 * x**2)),
        make_event_from_param=FakeEvent,
        target_epsilon=target_eps,
        target_delta=0,
    )
    accountant = FakeAccountant(lambda x: 1 / (2 * x**2))
    accountant.compose(FakeEvent(value))
    self.assertLessEqual(accountant.get_epsilon(0), target_eps)

  def test_auto_bracket_consistency_with_explicit(self):
    """Auto-bracket and explicit bracket give consistent results."""
    eps_fn = lambda x: 1 / x

    auto_value = mechanism_calibration.calibrate_dp_mechanism(
        make_fresh_accountant=lambda: FakeAccountant(eps_fn),
        make_event_from_param=FakeEvent,
        target_epsilon=1.0,
        target_delta=0,
    )

    explicit_value = mechanism_calibration.calibrate_dp_mechanism(
        make_fresh_accountant=lambda: FakeAccountant(eps_fn),
        make_event_from_param=FakeEvent,
        target_epsilon=1.0,
        target_delta=0,
        bracket_interval=mechanism_calibration.ExplicitBracketInterval(
            0.01, 100
        ),
    )

    self.assertAlmostEqual(auto_value, explicit_value, places=5)

  # -------- for_calibration protocol tests --------

  def test_class_uses_for_calibration(self):
    """Passing a class invokes for_calibration with target params."""
    call_log = []

    class TrackedAccountant(FakeAccountant):

      @classmethod
      def for_calibration(cls, target_epsilon, target_delta):
        call_log.append((target_epsilon, target_delta))
        return cls(lambda x: 1 / x)

    mechanism_calibration.calibrate_dp_mechanism(
        make_fresh_accountant=TrackedAccountant,
        make_event_from_param=FakeEvent,
        target_epsilon=1.0,
        target_delta=0.5,
    )

    # for_calibration should have been called at least once with correct args.
    self.assertNotEmpty(call_log)
    for eps, delta in call_log:
      self.assertEqual(eps, 1.0)
      self.assertEqual(delta, 0.5)

  def test_lambda_bypasses_for_calibration(self):
    """Passing a lambda does NOT invoke for_calibration."""
    call_log = []

    class TrackedAccountant(FakeAccountant):

      @classmethod
      def for_calibration(cls, target_epsilon, target_delta):
        call_log.append((target_epsilon, target_delta))
        return cls(lambda x: 1 / x)

    mechanism_calibration.calibrate_dp_mechanism(
        make_fresh_accountant=lambda: TrackedAccountant(lambda x: 1 / x),
        make_event_from_param=FakeEvent,
        target_epsilon=1.0,
        target_delta=0,
    )

    # for_calibration should NOT be called when a lambda is used.
    self.assertEmpty(call_log)

  def test_default_for_calibration_returns_cls(self):
    """Base class for_calibration with **kwargs returns cls(**kwargs)."""
    # FakeAccountant requires value_to_epsilon; the default for_calibration
    # forwards **kwargs, so we can pass it through.
    accountant = FakeAccountant.for_calibration(
        1.0, 1e-6, value_to_epsilon=lambda x: x
    )
    self.assertIsInstance(accountant, FakeAccountant)


if __name__ == '__main__':
  absltest.main()

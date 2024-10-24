# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Abstract base class for tests of `PrivacyAccountant` classes.

Checks that a class derived from `PrivacyAccountant` has the correct behavior
for standard `DpEvent` classes.
"""

from typing import Collection

from absl.testing import absltest
from absl.testing import parameterized

from dp_accounting import dp_event
from dp_accounting import privacy_accountant


class UnknownDpEvent(dp_event.DpEvent):
  pass


@absltest.skipThisClass('only intended to be run by subclasses')
class PrivacyAccountantTest(parameterized.TestCase):

  def _make_test_accountants(
      self) -> Collection[privacy_accountant.PrivacyAccountant]:
    """Makes a list of accountants to test.

    Subclasses should define this to return a list of accountants to be tested.

    Returns:
      A list of accountants to test.
    """
    return []

  def test_make_test_accountants(self):
    self.assertNotEmpty(self._make_test_accountants())

  @parameterized.product(
      unsupported_event=(dp_event.UnsupportedDpEvent(), UnknownDpEvent()),
      nest_fn=(
          lambda event_: event_,
          lambda event_: dp_event.ComposedDpEvent([event_]),
          lambda event_: dp_event.SelfComposedDpEvent(event_, 10),
      ),
  )
  def test_unsupported(self, unsupported_event, nest_fn):
    event_ = nest_fn(unsupported_event)
    for accountant in self._make_test_accountants():
      composition_error = accountant._maybe_compose(
          event_, count=1, do_compose=False
      )
      self.assertIsNotNone(composition_error)
      self.assertEqual(composition_error.invalid_event, unsupported_event)
      self.assertFalse(accountant.supports(event_))
      with self.assertRaisesRegex(
          privacy_accountant.UnsupportedEventError, 'caused by subevent'
      ):
        accountant.compose(event_)

  def test_no_events(self):
    for accountant in self._make_test_accountants():
      self.assertEqual(accountant.get_epsilon(1e-12), 0)
      self.assertEqual(accountant.get_epsilon(0), 0)
      self.assertEqual(accountant.get_epsilon(1), 0)
      try:
        self.assertEqual(accountant.get_delta(1e-12), 0)
        self.assertEqual(accountant.get_delta(0), 0)
        self.assertEqual(accountant.get_delta(float('inf')), 0)
      except NotImplementedError:
        # Implementing `get_delta` is optional.
        pass

  def test_no_op(self):
    for accountant in self._make_test_accountants():
      event = dp_event.NoOpDpEvent()
      self.assertTrue(accountant.supports(event))
      accountant.compose(event)
      self.assertEqual(accountant.get_epsilon(1e-12), 0)
      self.assertEqual(accountant.get_epsilon(0), 0)
      self.assertEqual(accountant.get_epsilon(1), 0)
      try:
        self.assertEqual(accountant.get_delta(1e-12), 0)
        self.assertEqual(accountant.get_delta(0), 0)
        self.assertEqual(accountant.get_delta(float('inf')), 0)
      except NotImplementedError:
        # Implementing `get_delta` is optional.
        pass

  @parameterized.parameters(
      dp_event.NonPrivateDpEvent(),
      dp_event.ComposedDpEvent([dp_event.NonPrivateDpEvent()]),
      dp_event.ComposedDpEvent(
          [dp_event.NoOpDpEvent(), dp_event.NonPrivateDpEvent()]
      ),
      dp_event.SelfComposedDpEvent(dp_event.NonPrivateDpEvent(), 10),
  )
  def test_non_private(self, non_private_event):
    for accountant in self._make_test_accountants():
      self.assertTrue(accountant.supports(non_private_event))
      accountant.compose(non_private_event)
      self.assertEqual(accountant.get_epsilon(0.99), float('inf'))
      self.assertEqual(accountant.get_epsilon(0), float('inf'))
      self.assertEqual(accountant.get_epsilon(1), float('inf'))
      try:
        self.assertEqual(accountant.get_delta(100), 1)
        self.assertEqual(accountant.get_delta(0), 1)
        self.assertEqual(accountant.get_delta(float('inf')), 1)
      except NotImplementedError:
        # Implementing `get_delta` is optional.
        pass

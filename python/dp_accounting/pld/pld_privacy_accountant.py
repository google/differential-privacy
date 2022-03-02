# Copyright 2021 Google LLC.
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
"""Privacy accountant that uses Privacy Loss Distributions."""

import math

from dp_accounting import dp_event
from dp_accounting import privacy_accountant
from dp_accounting import privacy_loss_distribution

NeighborRel = privacy_accountant.NeighboringRelation
PLD = privacy_loss_distribution


class PLDAccountant(privacy_accountant.PrivacyAccountant):
  """Privacy accountant that uses Privacy Loss Distributions."""

  def __init__(
      self,
      neighboring_relation: NeighborRel = NeighborRel.ADD_OR_REMOVE_ONE,
      value_discretization_interval: float = 1e-4,
  ):
    super(PLDAccountant, self).__init__(neighboring_relation)
    self._contains_non_dp_event = False
    self._pld = PLD.identity(
        value_discretization_interval=value_discretization_interval)
    self._value_discretization_interval = value_discretization_interval

  def supports(self, event: dp_event.DpEvent) -> bool:
    return self._maybe_compose(event, 0, False)

  def _compose(self, event: dp_event.DpEvent, count: int = 1):
    self._maybe_compose(event, count, True)

  def _maybe_compose(self, event: dp_event.DpEvent, count: int,
                     do_compose: bool) -> bool:
    """Traverses `event` and performs composition if `do_compose` is True.

    If `do_compose` is False, can be used to check whether composition is
    supported.

    Args:
      event: A `DpEvent` to process.
      count: The number of times to compose the event.
      do_compose: Whether to actually perform the composition.

    Returns:
      True if event is supported, otherwise False.
    """

    if isinstance(event, dp_event.NoOpDpEvent):
      return True
    elif isinstance(event, dp_event.NonPrivateDpEvent):
      if do_compose:
        self._contains_non_dp_event = True
      return True
    elif isinstance(event, dp_event.SelfComposedDpEvent):
      return self._maybe_compose(event.event, event.count * count, do_compose)
    elif isinstance(event, dp_event.ComposedDpEvent):
      return all(
          self._maybe_compose(e, count, do_compose) for e in event.events)
    elif isinstance(event, dp_event.GaussianDpEvent):
      if do_compose:
        gaussian_pld = PLD.from_gaussian_mechanism(
            standard_deviation=event.noise_multiplier / math.sqrt(count),
            value_discretization_interval=self._value_discretization_interval)
        self._pld = self._pld.compose(gaussian_pld)
      return True
    elif isinstance(event, dp_event.PoissonSampledDpEvent):
      if not (self.neighboring_relation == NeighborRel.ADD_OR_REMOVE_ONE and
              isinstance(event.event, dp_event.GaussianDpEvent)):
        return False
      if do_compose:
        subsampled_gaussian_pld = PLD.from_gaussian_mechanism(
            standard_deviation=event.event.noise_multiplier,
            value_discretization_interval=self._value_discretization_interval,
            sampling_prob=event.sampling_probability).self_compose(count)
        self._pld = self._pld.compose(subsampled_gaussian_pld)
      return True
    else:
      # Unsupported event (including `UnsupportedDpEvent`).
      return False

  def get_epsilon(self, target_delta: float) -> float:
    if self._contains_non_dp_event:
      return math.inf
    return self._pld.get_epsilon_for_delta(target_delta)

  def get_delta(self, target_epsilon: float) -> float:
    if self._contains_non_dp_event:
      return 1
    return self._pld.get_delta_for_epsilon(target_epsilon)

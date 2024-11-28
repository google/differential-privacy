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
from typing import Optional

from dp_accounting import dp_event
from dp_accounting import privacy_accountant
from dp_accounting.pld import privacy_loss_distribution

NeighborRel = privacy_accountant.NeighboringRelation
CompositionErrorDetails = privacy_accountant.PrivacyAccountant.CompositionErrorDetails
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

  def _maybe_compose(self, event: dp_event.DpEvent, count: int,
                     do_compose: bool) -> Optional[CompositionErrorDetails]:
    if isinstance(event, dp_event.NoOpDpEvent):
      return None
    elif isinstance(event, dp_event.NonPrivateDpEvent):
      if do_compose:
        self._contains_non_dp_event = True
      return None
    elif isinstance(event, dp_event.SelfComposedDpEvent):
      return self._maybe_compose(event.event, event.count * count, do_compose)
    elif isinstance(event, dp_event.ComposedDpEvent):
      for e in event.events:
        result = self._maybe_compose(e, count, do_compose)
        if result is not None:
          return result
      return None
    elif isinstance(event, dp_event.RandomizedResponseDpEvent):
      if self.neighboring_relation not in [NeighborRel.REPLACE_ONE,
                                           NeighborRel.REPLACE_SPECIAL]:
        error_msg = (
            'neighboring_relation must be `REPLACE_ONE` or '
            '`REPLACE_SPECIAL` for `RandomizedResponseDpEvent`. Found '
            f'{self._neighboring_relation}.')
        return CompositionErrorDetails(
            invalid_event=event, error_message=error_msg)
      if do_compose:
        if event.num_buckets == 1:
          # This is a NoOp event, even when noise_parameter is zero.
          pass
        elif event.noise_parameter == 0:
          self._contains_non_dp_event = True
        else:
          rr_pld = PLD.from_randomized_response(
              noise_parameter=event.noise_parameter,
              num_buckets=event.num_buckets,
              value_discretization_interval=self._value_discretization_interval,
              neighboring_relation=self._neighboring_relation,
          )
          self._pld = self._pld.compose(rr_pld)
      return None
    elif isinstance(event, dp_event.GaussianDpEvent):
      if do_compose:
        if event.noise_multiplier == 0:
          self._contains_non_dp_event = True
        else:
          gaussian_pld = PLD.from_gaussian_mechanism(
              standard_deviation=event.noise_multiplier / math.sqrt(count),
              value_discretization_interval=self._value_discretization_interval)
          self._pld = self._pld.compose(gaussian_pld)
      return None
    elif isinstance(event, dp_event.LaplaceDpEvent):
      if do_compose:
        if event.noise_multiplier == 0:
          self._contains_non_dp_event = True
        else:
          laplace_pld = PLD.from_laplace_mechanism(
              parameter=event.noise_multiplier,
              value_discretization_interval=self._value_discretization_interval
          ).self_compose(count)
          self._pld = self._pld.compose(laplace_pld)
      return None
    elif isinstance(event, dp_event.MixtureOfGaussiansDpEvent):
      if do_compose:
        if len(event.sensitivities) == 1 and event.sensitivities[0] == 0.0:
          pass
        elif event.standard_deviation == 0:
          self._contains_non_dp_event = True
        else:
          mog_pld = PLD.from_mixture_gaussian_mechanism(
              standard_deviation=event.standard_deviation,
              sensitivities=event.sensitivities,
              sampling_probs=event.sampling_probs,
              value_discretization_interval=self._value_discretization_interval,
          ).self_compose(count)
          self._pld = self._pld.compose(mog_pld)
      return None
    elif isinstance(event, dp_event.PoissonSampledDpEvent):
      if self.neighboring_relation not in [
          NeighborRel.ADD_OR_REMOVE_ONE, NeighborRel.REPLACE_SPECIAL
      ]:
        error_msg = (
            'neighboring_relation must be `ADD_OR_REMOVE_ONE` or '
            '`REPLACE_SPECIAL` for `PoissonSampledDpEvent`. Found '
            f'{self._neighboring_relation}.')
        return CompositionErrorDetails(
            invalid_event=event, error_message=error_msg)
      if isinstance(event.event, dp_event.GaussianDpEvent):
        if do_compose:
          if event.sampling_probability == 0:
            pass
          elif event.event.noise_multiplier == 0:
            self._contains_non_dp_event = True
          else:
            subsampled_gaussian_pld = PLD.from_gaussian_mechanism(
                standard_deviation=event.event.noise_multiplier,
                value_discretization_interval=self
                ._value_discretization_interval,
                sampling_prob=event.sampling_probability).self_compose(count)
            self._pld = self._pld.compose(subsampled_gaussian_pld)
        return None
      elif isinstance(event.event, dp_event.LaplaceDpEvent):
        if do_compose:
          if event.sampling_probability == 0:
            pass
          elif event.event.noise_multiplier == 0:
            self._contains_non_dp_event = True
          else:
            subsampled_laplace_pld = PLD.from_laplace_mechanism(
                parameter=event.event.noise_multiplier,
                value_discretization_interval=self
                ._value_discretization_interval,
                sampling_prob=event.sampling_probability).self_compose(count)
            self._pld = self._pld.compose(subsampled_laplace_pld)
        return None
      else:
        return CompositionErrorDetails(
            invalid_event=event,
            error_message=(
                'Subevent of `PoissonSampledEvent` must be either '
                f'`GaussianDpEvent` or `LaplaceDpEvent`. Found {event.event}.'))
    else:
      # Unsupported event (including `UnsupportedDpEvent`).
      return CompositionErrorDetails(
          invalid_event=event, error_message='Unsupported event.')

  def get_epsilon(self, target_delta: float) -> float:
    if self._contains_non_dp_event:
      return math.inf
    return self._pld.get_epsilon_for_delta(target_delta)

  def get_delta(self, target_epsilon: float) -> float:
    if self._contains_non_dp_event:
      return 1
    return self._pld.get_delta_for_epsilon(target_epsilon)  # pytype: disable=bad-return-type

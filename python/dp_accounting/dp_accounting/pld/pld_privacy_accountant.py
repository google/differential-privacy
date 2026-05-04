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
import numbers
from typing import Optional, Union

import numpy as np

from dp_accounting import dp_event
from dp_accounting import privacy_accountant
from dp_accounting.pld import common
from dp_accounting.pld import privacy_loss_distribution


NeighborRel = privacy_accountant.NeighboringRelation
CompositionErrorDetails = (
    privacy_accountant.PrivacyAccountant.CompositionErrorDetails
)
PLD = privacy_loss_distribution


class PLDAccountant(privacy_accountant.PrivacyAccountant):
  """Privacy accountant that uses Privacy Loss Distributions."""

  def __init__(
      self,
      neighboring_relation: NeighborRel = NeighborRel.ADD_OR_REMOVE_ONE,
      value_discretization_interval: float = 1e-4,
  ):
    super().__init__(neighboring_relation)
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
    elif isinstance(event, dp_event.EpsilonDeltaDpEvent):
      if do_compose:
        self._pld = self._pld.compose(
            PLD.from_privacy_parameters(
                privacy_parameters=common.DifferentialPrivacyParameters(
                    epsilon=event.epsilon, delta=event.delta
                ),
                value_discretization_interval=self._value_discretization_interval,
            ).self_compose(count)
        )
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
              value_discretization_interval=self._value_discretization_interval,
              neighboring_relation=self.neighboring_relation)
          self._pld = self._pld.compose(gaussian_pld)
      return None
    elif isinstance(event, dp_event.LaplaceDpEvent):
      if self.neighboring_relation not in [
          NeighborRel.ADD_OR_REMOVE_ONE, NeighborRel.REPLACE_SPECIAL
      ]:
        return CompositionErrorDetails(
            invalid_event=event,
            error_message=(
                'neighboring_relation must be `ADD_OR_REMOVE_ONE` or '
                '`REPLACE_SPECIAL` for `LaplaceDpEvent`. Found '
                f'{self._neighboring_relation}.'
            ),
        )
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
    elif isinstance(event, dp_event.DiscreteLaplaceDpEvent):
      if self.neighboring_relation not in [
          NeighborRel.ADD_OR_REMOVE_ONE, NeighborRel.REPLACE_SPECIAL
      ]:
        return CompositionErrorDetails(
            invalid_event=event,
            error_message=(
                'neighboring_relation must be `ADD_OR_REMOVE_ONE` or '
                '`REPLACE_SPECIAL` for `DiscreteLaplaceDpEvent`. Found '
                f'{self._neighboring_relation}.'
            ),
        )
      if do_compose:
        if event.noise_parameter == 0:
          self._contains_non_dp_event = True
        else:
          discrete_laplace_pld = PLD.from_discrete_laplace_mechanism(
              parameter=event.noise_parameter,
              sensitivity=event.sensitivity,
              value_discretization_interval=self._value_discretization_interval
          ).self_compose(count)
          self._pld = self._pld.compose(discrete_laplace_pld)
      return None
    elif isinstance(event, dp_event.MixtureOfGaussiansDpEvent):
      if self.neighboring_relation not in [
          NeighborRel.ADD_OR_REMOVE_ONE, NeighborRel.REPLACE_SPECIAL
      ]:
        return CompositionErrorDetails(
            invalid_event=event,
            error_message=(
                'neighboring_relation must be `ADD_OR_REMOVE_ONE` or '
                '`REPLACE_SPECIAL` for `MixtureOfGaussiansDpEvent`. Found '
                f'{self._neighboring_relation}.'
            ),
        )
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
    elif isinstance(event, dp_event.ExponentialMechanismDpEvent):
      if do_compose:
        if event.epsilon < 0:
          raise ValueError(f'epsilon must be >= 0. Got {event.epsilon}')
      if do_compose:
        # We use a worst-case PLD for any epsilon-DP mechanism, which is a
        # (loose) upper bound for the PLD of the exponental mechanism.
        eps_dp_pld = PLD.from_privacy_parameters(
            privacy_parameters=common.DifferentialPrivacyParameters(
                epsilon=event.epsilon, delta=0
            ),
            value_discretization_interval=self._value_discretization_interval,
        ).self_compose(count)
        self._pld = self._pld.compose(eps_dp_pld)
      return None
    elif isinstance(event, dp_event.PoissonSampledDpEvent):
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
                sampling_prob=event.sampling_probability,
                neighboring_relation=self.neighboring_relation,
            ).self_compose(count)
            self._pld = self._pld.compose(subsampled_gaussian_pld)
        return None
      elif isinstance(event.event, dp_event.LaplaceDpEvent):
        if self.neighboring_relation not in [
            NeighborRel.ADD_OR_REMOVE_ONE, NeighborRel.REPLACE_SPECIAL
        ]:
          return CompositionErrorDetails(
              invalid_event=event,
              error_message=(
                  'neighboring_relation must be `ADD_OR_REMOVE_ONE` or '
                  '`REPLACE_SPECIAL` for `PoissonSampledDpEvent` of a '
                  f'`LaplaceDpEvent`. Found {self._neighboring_relation}.'
              ),
          )
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
                f'`GaussianDpEvent` or `LaplaceDpEvent`. Found {event.event}.'
            ),
        )
    elif isinstance(event, dp_event.TruncatedSubsampledGaussianDpEvent):
      if do_compose:
        if (
            event.sampling_probability == 0
            or event.truncated_batch_size == 0
            or event.dataset_size == 0
        ):
          pass
        elif event.noise_multiplier == 0:
          self._contains_non_dp_event = True
        else:
          truncated_subsampled_gaussian_pld = PLD.from_truncated_subsampled_gaussian_mechanism(
              dataset_size=event.dataset_size,
              sampling_probability=event.sampling_probability,
              truncated_batch_size=event.truncated_batch_size,
              noise_multiplier=event.noise_multiplier,
              value_discretization_interval=self._value_discretization_interval,
              neighboring_relation=self.neighboring_relation,
          ).self_compose(
              count
          )
          self._pld = self._pld.compose(truncated_subsampled_gaussian_pld)
      return None
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

  def get_true_positive_rates(
      self,
      false_positive_rates: Union[float, np.ndarray],
      deltas: Optional[np.ndarray] = None,
  ) -> Union[float, np.ndarray]:
    """Computes an upper bound on the true positive rate (TPR).

    In particular, each (epsilon, delta) pair implied by the PLD also implies an
    upper bound on the TPR for a given false positive rate (FPR). This function
    computes this upper bound for a range of deltas (either user-specified, or a
    default range) and then returns the minimum TPR across all deltas. See
    Section 3.1 of the supplementary material for details.

    Note that this implementation reports a TPR-FPR curve which is symmetric
    with respect to the line y=1-x, which is not true for the true TPR-FPR curve
    for asymmetric mechanisms (e.g., subsampled Gaussian under add-remove). In
    this case the curve is still a valid upper bound on the true TPR-FPR curve,
    but perhaps overly pessimistic.

    Args:
      false_positive_rates: the FPR or list of FPRs at which to compute the TPR.
      deltas: the list of deltas to use for the computation. If None, the
        default deltas `np.logspace(np.log10(1e-13), np.log10(1), num=3000)` and
        0 will be used. A denser and wider range of deltas will yield a more
        accurate estimate, at the cost of increased run-time.

    Returns:
      A float or array of floats representing the upper bound on the TPR at the
      given FPR or list of FPRs.
    """
    if self._contains_non_dp_event:
      if isinstance(false_positive_rates, numbers.Number):
        return 1.0
      else:
        return np.ones_like(false_positive_rates)
    return self._pld.get_true_positive_rates(false_positive_rates, deltas)

  def get_gdp_parameter_estimate(
      self,
      false_positive_rates: Optional[np.ndarray] = None,
      deltas: Optional[np.ndarray] = None,
  ) -> float:
    """Computes an estimate of the mu-GDP parameter implied by the PLD.

    Specifically, we upper bound the true positive rate (TPR) at a given range
    of false positive rates (FPRs), and then find the minimum mu-GDP value that
    upper bounds all of the TPR upper bounds. This is pessimistic in that we are
    using upper bounds on the TPRs, but optimistic in that we are using a finite
    grid of TPRs, which is not guaranteed to contain the point at which the true
    mu-GDP parameter is tight. See Section 3.2 of the supplementary material for
    details.

    If the privacy loss is infinite with probability greater than
    min(false_positive_rates), then this function will return infinity.
    This is so that (i) when a PLD has large infinity mass, we correctly report
    infinite mu-GDP, but simultaneously (ii) the small infinity masses
    introduced by truncating a PLD to finite support do not result in infinite
    mu-GDP. We recommend reporting the minimum FPR used when reporting mu-GDP
    values computed using this function.

    Args:
      false_positive_rates: The list of FPRs to use for the computation. If
        None, the default FPRs `np.logspace(np.log10(1e-12), np.log10(0.5),
        num=500)` will be used. A denser and wider range of FPRs will reduce
        optimism of the estimate, at the cost of increased run-time.
      deltas: The list of deltas to use for the computation. If None, the
        default deltas `np.logspace(np.log10(1e-13), np.log10(1), num=3000)` and
        0 will be used. A denser and wider range of deltas will reduce pessimism
        of the estimate, at the cost of increased run-time.

    Returns:
      The estimated mu-GDP parameter. Note that this is not guaranteed to be an
      upper or lower bound on the true mu-GDP parameter (but can be made
      arbitrarily close to the true mu-GDP parameter by increasing the precision
      of the PLD, deltas, and FPRs).
    """
    if self._contains_non_dp_event:
      return math.inf
    return self._pld.get_gdp_parameter_estimate(false_positive_rates, deltas)

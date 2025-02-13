# Copyright 2020 Google LLC.
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

"""Implementing Privacy Loss of Mechanisms.

This file implements privacy loss of several additive noise mechanisms,
including Gaussian Mechanism, Laplace Mechanism and Discrete Laplace Mechanism.
Please refer to the supplementary material below for more details:
../docs/Privacy_Loss_Distributions.pdf
"""

import abc
import dataclasses
import enum
import functools
import math
import numbers
from typing import Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import scipy
from scipy import stats

from dp_accounting.pld import common


class AdjacencyType(enum.Enum):
  """Designates the type of adjacency for computing privacy loss distributions.

  ADD: the 'add' adjacency type specifies that the privacy loss distribution
    for a mechanism M is to be computed with mu_upper = M(D) and mu_lower =
    M(D'), where D' contains one more datapoint than D.
  REMOVE: the 'remove' adjacency type specifies that the privacy loss
    distribution for a mechanism M is to be computed with mu_upper = M(D) and
    mu_lower = M(D'), where D' contains one less datapoint than D.

  Note: The rest of code currently assumes existence of only these two adjacency
  types. If a new adjacency type is added and used, the API in this file will
  pretend that it is same as REMOVE.
  """
  ADD = 'ADD'
  REMOVE = 'REMOVE'


@dataclasses.dataclass(frozen=True)
class TailPrivacyLossDistribution:
  """Representation of the tail of privacy loss distribution.

  Attributes:
    lower_x_truncation: the minimum value of x that should be considered after
      the tail is discarded.
    upper_x_truncation: the maximum value of x that should be considered after
      the tail is discarded.
    tail_probability_mass_function: the probability mass of the privacy loss
      distribution that has to be added due to the discarded tail; each key is a
      privacy loss value and the corresponding value is the probability mass
      that the value occurs.
  """
  lower_x_truncation: float
  upper_x_truncation: float
  tail_probability_mass_function: Mapping[float, float]


@dataclasses.dataclass(frozen=True)
class ConnectDotsBounds:
  """Upper & lower bounds on epsilon or x to use for Connect-the-Dots algorithm.

  For continuous noise mechanisms, connect-the-dots algorithm will use all
  epsilons that are integer multiples of discretization interval that are
  between the specified lower and upper bounds on epsilons.

  For discrete noise mechanisms, connect-the-dots algorithm will use all
  epsilons that are obtained by rounding (both floor and ceiling) of the privacy
  loss at x values between the specified lower and upper bounds on x.

  Attributes:
    epsilon_upper: largest epsilon value to use in connect-the-dots algorithm
      for continuous additive noise mechanisms.
    epsilon_lower: smallest epsilon value to use in connect-the-dots algorithm
      for continuous additive noise mechanisms.
    lower_x: smallest x value to be considered in the connect-the-dots algorithm
      for discrete additive noise mechanisms.
    upper_x: largest x value to be considered in the connect-the-dots algorithm
      for discrete additive noise mechanisms.
  """
  epsilon_upper: Optional[float] = None
  epsilon_lower: Optional[float] = None
  lower_x: Optional[int] = None
  upper_x: Optional[int] = None


class MonotonePrivacyLoss(metaclass=abc.ABCMeta):
  """Superclass for privacy loss distributions that are monotone in x.

  Given distributions mu_upper and mu_lower over real numbers, the privacy loss
  function is defined as: privacy_loss(x) := ln(mu_upper(x) / mu_lower(x)).
  The privacy loss distribution is generated as follows:
  - Sample x ~ mu_upper and let the privacy loss be privacy_loss(x).

  This class assumes that privacy_loss(x) is non-increasing as x increases.

  Attributes:
    is_discrete: a value indicating whether the privacy loss distribution is
      discrete. If this is True, then it is assumed that mu_upper and mu_lower
      are supported over integer values. If False, then it is assumed that
      mu_upper and mu_lower are supported over real numbers.
  """

  def __init__(self, is_discrete: bool):
    self.is_discrete = is_discrete

  @abc.abstractmethod
  def mu_upper_cdf(
      self, x: Union[float, Iterable[float]]) -> Union[float, np.ndarray]:
    """Computes the cumulative density function of the mu_upper distribution.

    Args:
      x: the point or points at which the cumulative density function is to be
        calculated.

    Returns:
      The cumulative density function of the mu_upper distribution at x, i.e.,
      the probability that mu_upper is less than or equal to x.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def mu_lower_log_cdf(
      self, x: Union[float, Iterable[float]]) -> Union[float, np.ndarray]:
    """Computes log cumulative density function of the mu_lower distribution.

    Args:
      x: the point or points at which the log of the cumulative density function
        is to be calculated.

    Returns:
      The log of the cumulative density function of the mu_lower distribution at
      x, i.e., the log of the probability that mu_lower is less than or equal to
      x.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def get_delta_for_epsilon(
      self, epsilon: Union[float, Sequence[float]],
  ) -> Union[float, Sequence[float]]:
    """Computes the epsilon-hockey stick divergence of the mechanism.

    The epsilon-hockey stick divergence of the mechanism is the value of delta
    for which the mechanism is (epsilon, delta)-differentially private. (See
    Observation 1 in the supplementary material.)

    Args:
      epsilon: the epsilon, or list-like object of epsilon values, in
      epsilon-hockey stick divergence.

    Returns:
      A non-negative real number which is the epsilon-hockey stick divergence of
      the mechanism, or a numpy array if epsilon is list-like.
    """

  @abc.abstractmethod
  def privacy_loss_tail(self) -> TailPrivacyLossDistribution:
    """Computes the privacy loss at the tail of the distribution.

    Returns:
      A TailPrivacyLossDistribution instance representing the tail of the
      privacy loss distribution.

    Raises:
      NotImplementedError: If not implemented by the subclass.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def connect_dots_bounds(self) -> ConnectDotsBounds:
    """Computes the bounds on epsilon values to use in connect-the-dots algorithm.

    Returns:
      A ConnectDotsBounds instance containing either
      - upper and lower values of epsilon for continuous privacy loss
        distributions, or
      - lower and upper values of x for discrete privacy loss distributions.
      These values are to be used in connect-the-dots algorithm.
    """

  @abc.abstractmethod
  def privacy_loss(self, x: float) -> float:
    """Computes the privacy loss at a given point.

    Args:
      x: the point at which the privacy loss is computed.

    Returns:
      The privacy loss at point x.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def inverse_privacy_loss(self, privacy_loss: float) -> float:
    """Computes the inverse of a given privacy loss.

    Args:
      privacy_loss: the privacy loss value.

    Returns:
      The largest float x such that the privacy loss at x is at least
      privacy_loss.
    """
    raise NotImplementedError


class AdditiveNoisePrivacyLoss(MonotonePrivacyLoss, metaclass=abc.ABCMeta):
  """Superclass for privacy loss of additive noise mechanisms.

  An additive noise mechanism for computing a scalar-valued function f is a
  mechanism that outputs the sum of the true value of the function and a noise
  drawn from a certain distribution mu. This class allows one to compute several
  quantities related to the privacy loss of additive noise mechanisms.

  We assume that the noise mu is such that the algorithm is more private as the
  sensitivity of f decreases. (Recall that the sensitivity of f is the maximum
  absolute change in f when an input to a single user changes.) Under this
  assumption, the privacy loss distribution of the mechanism is exactly
  generated as follows:
  - Let mu_lower(x) := mu(x - sensitivity), i.e., right shifted by sensitivity
  - Sample x ~ mu_upper = mu and let the privacy loss be
    ln(mu_upper(x) / mu_lower(x)).
  When mu is discrete, mu(x) refers to the probability mass of mu at x, and when
  mu is continuous, mu(x) is the probability density of mu at x; mu_upper and
  mu_lower are defined analogously.

  Support for sub-sampling (Refer to supplementary material for more details):
  An additive noise mechanism with Poisson sub-sampling first samples a subset
  of data points including each data point independently with probability q,
  and outputs the sum of the true value of the function and a noise drawn from
  a certain distribution mu. Here, we consider differential privacy with
  respect to the addition/removal relation.

  With sub-sampling probability of q, the privacy loss distribution is
  generated as follows:
  For ADD adjacency type:
  - Let mu_lower(x) := q * mu(x - sensitivity) + (1-q) * mu(x)
  - Sample x ~ mu_upper = mu and let the privacy loss be
    ln(mu_upper(x) / mu_lower(x)).
  For REMOVE adjacency type:
  - Let mu_upper(x) := q * mu(x + sensitivity) + (1-q) * mu(x)
  - Sample x ~ mu_lower = mu and let the privacy loss be
    ln(mu_upper(x) / mu_lower(x)).

  Note: When q = 1, the result privacy loss distributions for both ADD and
    REMOVE adjacency types are identical.

  This class also assumes the privacy loss is non-increasing as x increases.

  Attributes:
    sensitivity: the sensitivity of function f. (i.e. the maximum absolute
      change in f when an input to a single user changes.)
    is_discrete: a value indicating whether the noise is discrete. If this
        is True, then it is assumed that the noise can only take integer values.
        If False, then it is assumed that the noise is continuous, i.e., the
        probability mass at any given point is zero.
    sampling_prob: sub-sampling probability, a value in (0,1].
    adjacency_type: type of adjacency relation to used for defining the privacy
        loss distribution.
  """

  def __init__(self,
               sensitivity: float,
               is_discrete: bool,
               sampling_prob: float = 1.0,
               adjacency_type: AdjacencyType = AdjacencyType.REMOVE):
    if sensitivity <= 0:
      raise ValueError(
          f'Sensitivity is not a positive real number: {sensitivity}')
    if sampling_prob <= 0 or sampling_prob > 1:
      raise ValueError(
          f'Sampling probability is not in (0,1] : {sampling_prob}')
    self.sensitivity = sensitivity
    self.sampling_prob = sampling_prob
    self.adjacency_type = adjacency_type
    super().__init__(is_discrete=is_discrete)

  def mu_upper_cdf(
      self, x: Union[float, Iterable[float]]) -> Union[float, np.ndarray]:
    """Computes the cumulative density function of the mu_upper distribution.

    For ADD adjacency type, for any sub-sampling probability:
      mu_upper(x) := mu
    For REMOVE adjacency type, with sub-sampling probability q:
      mu_upper(x) := (1-q) * mu(x) + q * mu(x + sensitivity)

    Args:
      x: the point or points at which the cumulative density function is to be
        calculated.

    Returns:
      The cumulative density function of the mu_upper distribution at x, i.e.,
      the probability that mu_upper is less than or equal to x.
    """

    if self.adjacency_type == AdjacencyType.ADD:
      return self.noise_cdf(x)
    else:  # Case: self.adjacency_type == AdjacencyType.REMOVE
      # For performance, the case of sampling_prob=1 is handled separately.
      if self.sampling_prob == 1.0:
        return self.noise_cdf(np.add(x, self.sensitivity))
      return ((1 - self.sampling_prob) * self.noise_cdf(x) +
              self.sampling_prob * self.noise_cdf(np.add(x, self.sensitivity)))

  def mu_lower_log_cdf(
      self, x: Union[float, Iterable[float]]) -> Union[float, np.ndarray]:
    """Computes log cumulative density function of the mu_lower distribution.

    For ADD adjacency type, with sub-sampling probability q:
      mu_lower(x) := (1-q) * mu(x) + q * mu(x - sensitivity)
    For REMOVE adjacency type, for any sub-sampling probability:
      mu_lower(x) := mu(x)

    Args:
      x: the point or points at which the log of the cumulative density function
        is to be calculated.

    Returns:
      The log of the cumulative density function of the mu_lower distribution at
      x, i.e., the log of the probability that mu_lower is less than or equal to
      x.
    """
    if self.adjacency_type == AdjacencyType.ADD:
      # For performance, the case of sampling_prob=1 is handled separately.
      if self.sampling_prob == 1.0:
        return self.noise_log_cdf(np.add(x, -self.sensitivity))
      return np.logaddexp(
          np.log1p(-self.sampling_prob) + self.noise_log_cdf(x),
          np.log(self.sampling_prob) +
          self.noise_log_cdf(np.add(x, -self.sensitivity))
      )

    else:  # Case: self.adjacency_type == AdjacencyType.REMOVE
      return self.noise_log_cdf(x)

  def get_delta_for_epsilon(
      self, epsilon: Union[float, Sequence[float]],
  ) -> Union[float, Sequence[float]]:
    """Computes the epsilon-hockey stick divergence of the mechanism.

    The epsilon-hockey stick divergence of the mechanism is the value of delta
    for which the mechanism is (epsilon, delta)-differentially private. (See
    Observation 1 in the supplementary material.)

    This function assumes the privacy loss is non-increasing as x increases.
    Under this assumption, the hockey stick divergence is simply
    mu_upper_cdf(inverse_privacy_loss(epsilon)) - exp(epsilon) *
    mu_lower_cdf(inverse_privacy_loss(epsilon) - sensitivity), because the
    privacy loss at a point x is at least epsilon iff
    x <= inverse_privacy_loss(epsilon).

    When adjacency_type is ADD and epsilon >= -log(1 - sampling_prob),
      the hockey stick divergence is 0,
      since mu_lower_cdf*exp(epsilon) is pointwise greater than mu_upper_cdf.
    When adjacency_type is REMOVE and epsilon <= log(1 - sampling_prob),
      the hockey stick divergence is 1-exp(epsilon),
      since mu_lower_cdf*exp(epsilon) is pointwise lower than mu_upper_cdf.

    Args:
      epsilon: the epsilon, or list-like object of epsilon values, in
      epsilon-hockey stick divergence.

    Returns:
      A non-negative real number which is the epsilon-hockey stick divergence of
      the mechanism, or a numpy array if epsilon is list-like.
    """
    is_scalar = isinstance(epsilon, numbers.Number)
    epsilons = np.array([epsilon]) if is_scalar else np.asarray(epsilon)
    deltas = np.zeros_like(epsilons, dtype=float)
    if self.sampling_prob == 1.0:
      inverse_indices = np.full_like(epsilons, True, dtype=bool)
    elif self.adjacency_type == AdjacencyType.ADD:
      inverse_indices = epsilons < -math.log1p(-self.sampling_prob)
    else:  # Case: self.adjacency_type == AdjacencyType.REMOVE
      inverse_indices = epsilons > math.log1p(-self.sampling_prob)
      other_indices = np.logical_not(inverse_indices)
      deltas[other_indices] = -np.expm1(epsilons[other_indices])
    x_cutoffs = np.array([
        self.inverse_privacy_loss(eps) for eps in epsilons[inverse_indices]
    ])
    deltas[inverse_indices] = (
        self.mu_upper_cdf(x_cutoffs) -
        np.exp(epsilons[inverse_indices] + self.mu_lower_log_cdf(x_cutoffs)))
    # Clip delta values to lie in [0,1] (to avoid numerical errors)
    deltas = np.clip(deltas, 0, 1)
    return float(deltas[0]) if is_scalar else deltas

  @abc.abstractmethod
  def privacy_loss_tail(self) -> TailPrivacyLossDistribution:
    """Computes the privacy loss at the tail of the distribution.

    Returns:
      A TailPrivacyLossDistribution instance representing the tail of the
      privacy loss distribution.

    Raises:
      NotImplementedError: If not implemented by the subclass.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def connect_dots_bounds(self) -> ConnectDotsBounds:
    """Computes the bounds on epsilon values to use in connect-the-dots algorithm.

    Returns:
      A ConnectDotsBounds instance containing either
      - upper and lower values of epsilon for continuous noise mechanisms, or
      - lower and upper values of x for discrete noise mechanisms.
      These values are to be used in connect-the-dots algorithm.
    """

  def privacy_loss(self, x: float) -> float:
    """Computes the privacy loss at a given point.

    For ADD adjacency type, with sub-sampling probability of q:
    the privacy loss at x is
    - log(1-q + q*exp(-privacy_loss_without_subsampling(x))).

    For REMOVE adjacency type, with sub-sampling probability of q:
    the privacy loss at x is
    log(1-q + q*exp(privacy_loss_without_subsampling(x))).

    Args:
      x: the point at which the privacy loss is computed.

    Returns:
      The privacy loss at point x.

    Raises:
      NotImplementedError: If privacy_loss_without_subsampling is not
        implemented by the subclass.
      ValueError: If privacy loss is undefined at x.
    """
    privacy_loss_without_subsampling = self.privacy_loss_without_subsampling(x)
    # For performance, the case of sampling_prob=1 is handled separately.
    if self.sampling_prob == 1.0:
      return privacy_loss_without_subsampling
    if self.adjacency_type == AdjacencyType.ADD:
      # Privacy loss is
      # -log(1 - sampling_prob +
      #      sampling_prob * exp(-privacy_loss_without_subsampling)).
      return -common.log_a_times_exp_b_plus_c(self.sampling_prob,
                                              -privacy_loss_without_subsampling,
                                              1 - self.sampling_prob)
    else:  # Case: self.adjacency_type == AdjacencyType.REMOVE
      # Privacy loss is
      # log(1 - sampling_prob +
      #      sampling_prob * exp(privacy_loss_without_subsampling)).
      return common.log_a_times_exp_b_plus_c(self.sampling_prob,
                                             privacy_loss_without_subsampling,
                                             1 - self.sampling_prob)

  @abc.abstractmethod
  def privacy_loss_without_subsampling(self, x: float) -> float:
    """Computes the privacy loss at a given point without sub-sampling.

    Args:
      x: the point at which the privacy loss is computed.

    Returns:
      The privacy loss at point x without sub-sampling, which is given as:
      For ADD adjacency type: ln(mu(x - sensitivity) / mu(x)).
      If mu(x - sensitivity) == 0 and mu(x) > 0, this is -infinity.
      If mu(x - sensitivity) > 0  and mu(x) == 0, this is +infinity.
      If mu(x - sensitivity) == 0 and mu(x) == 0, this is undefined
        (ValueError is raised in this case).

      For REMOVE adjacency type: ln(mu(x + sensitivity) / mu(x)).
      Similar conventions (regarding corner cases) apply as above.

    Raises:
      NotImplementedError: If not implemented by the subclass.
    """
    raise NotImplementedError

  def inverse_privacy_loss(self, privacy_loss: float) -> float:
    """Computes the inverse of a given privacy loss.

    Args:
      privacy_loss: the privacy loss value.

    Returns:
      The largest float x such that the privacy loss at x is at least
      privacy_loss.

      For the ADD adjacency type, with sub-sampling probability of q:
      the inverse privacy loss is given as
       inverse_privacy_loss_without_subsampling(-log(1 +
                                                     (exp(-privacy_loss)-1)/q)),
      When privacy_loss >= -log(1-q), the inverse privacy loss is
        inverse_privacy_loss_without_subsampling(+infinity),
      When privacy_loss == -infinity, the inverse privacy loss is
        inverse_privacy_loss_without_subsampling(-infinity).

      For the REMOVE adjacency type, with sub-sampling probability of q:
      the inverse privacy loss is given as
        inverse_privacy_loss_without_subsampling(log(1 +
                                                     (exp(privacy_loss)-1)/q)),
      When privacy_loss <= log(1-q), the inverse privacy loss is
        inverse_privacy_loss_without_subsampling(-infinity),
      When privacy_loss == infinity, the inverse privacy loss is
        inverse_privacy_loss_without_subsampling(+infinity).

    Raises:
      NotImplementedError: If inverse_privacy_loss_without_subsampling is not
        implemented by the subclass.
      ValueError: If inverse_privacy_loss_without_subsampling raises a
        ValueError
    """
    # For performance, the case of sampling_prob=1 is handled separately.
    if self.sampling_prob == 1.0:
      return self.inverse_privacy_loss_without_subsampling(privacy_loss)

    if self.adjacency_type == AdjacencyType.ADD:
      if math.isclose(privacy_loss, - math.log(1 - self.sampling_prob)):
        return self.inverse_privacy_loss_without_subsampling(math.inf)
      if privacy_loss > - math.log(1 - self.sampling_prob):
        raise ValueError(f'privacy_loss ({privacy_loss}) is larger than '
                         f'-log(1 - sampling_prob) '
                         f'({-math.log(1 - self.sampling_prob)}')
      # Privacy loss without subsampling is
      # -log(1 + (exp(-privacy_loss) - 1) / sampling_prob).
      privacy_loss_without_subsampling = -common.log_a_times_exp_b_plus_c(
          1 / self.sampling_prob, -privacy_loss, 1 - 1 / self.sampling_prob)
      return self.inverse_privacy_loss_without_subsampling(
          privacy_loss_without_subsampling)

    else:  # Case: self.adjacency_type == AdjacencyType.REMOVE
      if math.isclose(privacy_loss, math.log(1 - self.sampling_prob)):
        return self.inverse_privacy_loss_without_subsampling(-math.inf)
      if privacy_loss <= math.log(1 - self.sampling_prob):
        raise ValueError(f'privacy_loss ({privacy_loss}) is smaller than '
                         f'log(1 - sampling_prob) '
                         f'({math.log(1 - self.sampling_prob)}')
      # Privacy loss without subsampling is
      # log(1 + (exp(privacy_loss) - 1) / sampling_prob).
      privacy_loss_without_subsampling = common.log_a_times_exp_b_plus_c(
          1 / self.sampling_prob, privacy_loss, 1 - 1 / self.sampling_prob)
      return self.inverse_privacy_loss_without_subsampling(
          privacy_loss_without_subsampling)

  @abc.abstractmethod
  def inverse_privacy_loss_without_subsampling(self,
                                               privacy_loss: float) -> float:
    """Computes the inverse of a given privacy loss without sub-sampling.

    Args:
      privacy_loss: the privacy loss value.

    Returns:
      The largest float x such that the privacy loss at x without sub-sampling,
      is at least privacy_loss.

    Raises:
      NotImplementedError: If not implemented by the subclass.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def noise_cdf(self, x: Union[float,
                               Iterable[float]]) -> Union[float, np.ndarray]:
    """Computes the cumulative density function of the noise distribution mu.

    Args:
      x: the point or points at which the cumulative density function is to be
        calculated.

    Returns:
      The cumulative density function of that noise at x, i.e., the probability
      that mu is less than or equal to x.

    Raises:
      NotImplementedError: If not implemented by the subclass.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def noise_log_cdf(
      self, x: Union[float, Iterable[float]]) -> Union[float, np.ndarray]:
    """Computes log of cumulative density function of the noise distribution mu.

    Args:
      x: the point or points at which the log cumulative density function is to
        be calculated.

    Returns:
      The log cumulative density function of that noise at x, i.e., the log of
      the probability that mu is less than or equal to x.

    Raises:
      NotImplementedError: If not implemented by the subclass.
    """
    raise NotImplementedError

  @classmethod
  @abc.abstractmethod
  def from_privacy_guarantee(
      cls,
      privacy_parameters: common.DifferentialPrivacyParameters,
      sensitivity: float = 1,
      pessimistic_estimate: bool = True,
      sampling_prob: float = 1.0,
      adjacency_type: AdjacencyType = AdjacencyType.REMOVE
  ) -> 'AdditiveNoisePrivacyLoss':
    """Creates the privacy loss for the mechanism with a given privacy.

    Computes parameters achieving given privacy with REMOVE relation,
    irrespective of adjacency_type, since for all epsilon > 0, the hockey-stick
    divergence for PLD with respect to the REMOVE adjacency type is at least
    that for PLD with respect to ADD adjacency type.

    The returned object has the specified adjacency_type.

    Args:
      privacy_parameters: the desired privacy guarantee of the mechanism.
      sensitivity: the sensitivity of function f. (i.e. the maximum absolute
        change in f when an input to a single user changes.)
      pessimistic_estimate: a value indicating whether the rounding is done in
        such a way that the resulting epsilon-hockey stick divergence
        computation gives an upper estimate to the real value.
      sampling_prob: sub-sampling probability, a value in (0,1].
      adjacency_type: type of adjacency relation to used for defining the
        privacy loss distribution.

    Returns:
      The privacy loss of the mechanism with the given privacy guarantee.

    Raises:
      NotImplementedError: If not implemented by the subclass.
    """
    raise NotImplementedError


class LaplacePrivacyLoss(AdditiveNoisePrivacyLoss):
  """Privacy loss of the Laplace mechanism.

  The Laplace mechanism for computing a scalar-valued function f simply outputs
  the sum of the true value of the function and a noise drawn from the Laplace
  distribution. Recall that the Laplace distribution with parameter b has
  probability density function 0.5/b * exp(-|x|/b) at x for any real number x.

  The privacy loss distribution of the Laplace mechanism is equivalent to the
  privacy loss distribution between the Laplace distribution and the same
  distribution but shifted by the sensitivity of f. Specifically, the privacy
  loss distribution of the Laplace mechanism is generated as follows:
  - Let mu = Lap(0, b) be the Laplace noise PDF as given above.
  - Let mu_lower(x) := mu(x - sensitivity), i.e., right shifted by sensitivity
  - Sample x ~ mu_upper = mu and let the privacy loss be
    ln(mu_upper(x) / mu_lower(x)), which is equal to
    (|x - sensitivity| - |x|) / parameter.

  Case of sub-sampling (Refer to supplementary material for more details):
  The Laplace mechanism with sub-sampling for computing a scalar-valued function
  f, first samples a subset of data points including each data point
  independently with probability q, and returns the sum of the true values and a
  noise drawn from the Laplace distribution. Here, we consider differential
  privacy with respect to the addition/removal relation.

  When the sub-sampling probability is q, the worst-case privacy loss
  distribution is generated as follows:
  For ADD adjacency type:
  - Let mu_lower(x) := q * mu(x - sensitivity) + (1-q) * mu(x)
  - Sample x ~ mu_upper = mu and let the privacy loss be
    ln(mu_upper(x) / mu_lower(x)).
  For REMOVE adjacency type:
  - Let mu_upper(x) := q * mu(x + sensitivity) + (1-q) * mu(x)
  - Sample x ~ mu_lower = mu and let the privacy loss be
    ln(mu_upper(x) / mu_lower(x)).

  Note: When q = 1, the result privacy loss distributions for both ADD and
    REMOVE adjacency types are identical.
  """

  def __init__(self,
               parameter: float,
               sensitivity: float = 1,
               sampling_prob: float = 1.0,
               adjacency_type: AdjacencyType = AdjacencyType.REMOVE) -> None:
    """Initializes the privacy loss of the Laplace mechanism.

    Args:
      parameter: the parameter of the Laplace distribution.
      sensitivity: the sensitivity of function f. (i.e. the maximum absolute
        change in f when an input to a single user changes.)
      sampling_prob: sub-sampling probability, a value in (0,1].
      adjacency_type: type of adjacency relation to used for defining the
        privacy loss distribution.
    """
    if parameter <= 0:
      raise ValueError(f'Parameter is not a positive real number: {parameter}')

    self._parameter = parameter
    self._laplace_random_variable = stats.laplace(scale=parameter)
    super().__init__(sensitivity, False, sampling_prob, adjacency_type)

  def privacy_loss_tail(self) -> TailPrivacyLossDistribution:
    """Computes the privacy loss at the tail of the Laplace distribution.

    For ADD adjacency type:
    lower_x_truncation = 0 and upper_x_truncation = sensitivity

    For REMOVE adjacency type:
    lower_x_truncation = -sensitivity and upper_x_truncation = 0

    The probability masses below lower_x_truncation and above upper_x_truncation
    are computed using mu_upper_cdf.

    Returns:
      A TailPrivacyLossDistribution instance representing the tail of the
      privacy loss distribution.
    """
    if self.adjacency_type == AdjacencyType.ADD:
      lower_x_truncation, upper_x_truncation = 0.0, self.sensitivity
    else:  # Case: self.adjacency_type == AdjacencyType.REMOVE
      lower_x_truncation, upper_x_truncation = -self.sensitivity, 0.0

    return TailPrivacyLossDistribution(
        lower_x_truncation, upper_x_truncation, {
            self.privacy_loss(lower_x_truncation):
                self.mu_upper_cdf(lower_x_truncation),
            self.privacy_loss(upper_x_truncation):
                1 - self.mu_upper_cdf(upper_x_truncation)
        })

  def connect_dots_bounds(self) -> ConnectDotsBounds:
    """Computes the bounds on epsilon values to use in connect-the-dots algorithm.

    With sub-sampling probability of q,
    For ADD adjacency type:
      epsilon_upper = - log(1 - q + q * e^{-sensitivity / parameter})
      epsilon_lower = - log(1 - q + q * e^{sensitivity / parameter})

    For REMOVE adjacency type:
      epsilon_upper = log(1 - q + q * e^{sensitivity / parameter})
      epsilon_lower = log(1 - q + q * e^{-sensitivity / parameter})

    Returns:
      A ConnectDotsBounds instance containing upper and lower values of
      epsilon to use in connect-the-dots algorithm.
    """
    max_epsilon = self.sensitivity / self._parameter
    if self.sampling_prob == 1.0:
      # For efficiency this case is handled separately.
      return ConnectDotsBounds(epsilon_upper=max_epsilon,
                               epsilon_lower=-max_epsilon)
    elif self.adjacency_type == AdjacencyType.ADD:
      return ConnectDotsBounds(
          epsilon_upper=- math.log(1 - self.sampling_prob +
                                   self.sampling_prob * math.exp(-max_epsilon)),
          epsilon_lower=- math.log(1 - self.sampling_prob +
                                   self.sampling_prob * math.exp(max_epsilon)))
    else:  # Case: self.adjacency_type == AdjacencyType.REMOVE
      return ConnectDotsBounds(
          epsilon_upper=math.log(1 - self.sampling_prob +
                                 self.sampling_prob * math.exp(max_epsilon)),
          epsilon_lower=math.log(1 - self.sampling_prob +
                                 self.sampling_prob * math.exp(-max_epsilon)))

  def privacy_loss_without_subsampling(self, x: float) -> float:
    """Computes the privacy loss of the Laplace mechanism without sub-sampling at a given point.

    Args:
      x: the point at which the privacy loss is computed.

    Returns:
      The privacy loss of the Laplace mechanism without sub-sampling at point x,
      which is given as
      For ADD adjacency type:    (|x - sensitivity| - |x|) / parameter.
      For REMOVE adjacency type: (|x| - |x + sensitivity|) / parameter.
    """
    if self.adjacency_type == AdjacencyType.ADD:
      return (abs(x - self.sensitivity) - abs(x)) / self._parameter
    else:  # Case: self.adjacency_type == AdjacencyType.REMOVE
      return (abs(x) - abs(x + self.sensitivity)) / self._parameter

  def inverse_privacy_loss_without_subsampling(self,
                                               privacy_loss: float) -> float:
    """Computes the inverse of a given privacy loss for the Laplace mechanism without sub-sampling.

    Args:
      privacy_loss: the privacy loss value.

    Returns:
      The largest float x such that the privacy loss at x is at least
      privacy_loss.
      For ADD adjacency type:
        If privacy_loss <= - sensitivity / parameter, x is equal to infinity.
        If - sensitivity / parameter < privacy_loss <= sensitivity / parameter,
          x is equal to 0.5 * (sensitivity - privacy_loss * parameter).
        If privacy_loss > sensitivity / parameter, no such x exists and the
          function returns -infinity.
      For REMOVE adjacency type:
        For any value of privacy_loss, x is equal to the corresponding value for
          ADD adjacency type decreased by sensitivity.
    """
    loss_threshold = privacy_loss * self._parameter
    if loss_threshold > self.sensitivity:
      return -math.inf
    if loss_threshold <= -self.sensitivity:
      return math.inf
    if self.adjacency_type == AdjacencyType.ADD:
      return 0.5 * (self.sensitivity - loss_threshold)
    else:  # Case: self.adjacency_type == AdjacencyType.REMOVE
      return 0.5 * (-self.sensitivity - loss_threshold)

  def noise_cdf(self, x: Union[float,
                               Iterable[float]]) -> Union[float, np.ndarray]:
    """Computes the cumulative density function of the Laplace distribution.

    Args:
      x: the point or points at which the cumulative density function is to be
        calculated.

    Returns:
      The cumulative density function of the Laplace noise at x, i.e., the
      probability that the Laplace noise is less than or equal to x.
    """
    return self._laplace_random_variable.cdf(x)

  def noise_log_cdf(
      self, x: Union[float, Iterable[float]]) -> Union[float, np.ndarray]:
    """Computes log of cumulative density function of the Laplace distribution.

    Args:
      x: the point or points at which the log cumulative density function is to
        be calculated.

    Returns:
      The log cumulative density function of the Laplace noise at x, i.e., the
      log of the probability that the Laplace noise is less than or equal to x.
    """
    return self._laplace_random_variable.logcdf(x)

  @classmethod
  def from_privacy_guarantee(
      cls,
      privacy_parameters: common.DifferentialPrivacyParameters,
      sensitivity: float = 1,
      pessimistic_estimate: bool = True,
      sampling_prob: float = 1.0,
      adjacency_type: AdjacencyType = AdjacencyType.REMOVE
  ) -> 'LaplacePrivacyLoss':
    """Creates the privacy loss for Laplace mechanism with given privacy.

    Without sub-sampling, the parameter of the Laplace mechanism is simply
      sensitivity / epsilon.
    With sub-sampling probability of q, the parameter is given as
      sensitivity / log(1 + (exp(epsilon) - 1)/q).
    Note: Only the REMOVE adjacency type is used in determining the parameter,
      since for all epsilon > 0, the hockey-stick divergence for PLD with
      respect to the REMOVE adjacency type is at least that for PLD with respect
      to ADD adjacency type.

    Args:
      privacy_parameters: the desired privacy guarantee of the mechanism.
      sensitivity: the sensitivity of function f. (i.e. the maximum absolute
        change in f when an input to a single user changes.)
      pessimistic_estimate: a value indicating whether the rounding is done in
        such a way that the resulting epsilon-hockey stick divergence
        computation gives an upper estimate to the real value.
      sampling_prob: sub-sampling probability, a value in (0,1].
      adjacency_type: type of adjacency relation to used for defining the
        privacy loss distribution.

    Returns:
      The privacy loss of the Laplace mechanism with the given privacy
        guarantee.
    """
    parameter = (
        sensitivity /
        np.log(1 + (np.exp(privacy_parameters.epsilon) - 1) / sampling_prob))
    return LaplacePrivacyLoss(
        parameter,
        sensitivity=sensitivity,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)

  @property
  def parameter(self) -> float:
    """The parameter of the corresponding Laplace noise."""
    return self._parameter


class GaussianPrivacyLoss(AdditiveNoisePrivacyLoss):
  """Privacy loss of the Gaussian mechanism.

  The Gaussian mechanism for computing a scalar-valued function f simply
  outputs the sum of the true value of the function and a noise drawn from the
  Gaussian distribution. Recall that the (centered) Gaussian distribution with
  standard deviation sigma has probability density function
  1/(sigma * sqrt(2 * pi)) * exp(-0.5 x^2/sigma^2) at x for any real number x.

  The privacy loss distribution of the Gaussian mechanism is equivalent to the
  privacy loss distribution between the Gaussian distribution and the same
  distribution but shifted by the sensitivity of f. Specifically, the privacy
  loss distribution of the Gaussian mechanism is generated as follows:
  - Let mu = N(0, sigma^2) be the Gaussian noise PDF as given above.
  - Let mu_lower(x) := mu(x - sensitivity), i.e., right shifted by sensitivity
  - Sample x ~ mu_upper = mu and let the privacy loss be
    ln(mu_upper(x) / mu_lower(x)).

  Case of sub-sampling (Refer to supplementary material for more details):
  The Gaussian mechanism with sub-sampling for computing a scalar-valued
  function f, first samples a subset of data points including each data point
  independently with probability q, and returns the sum of the true values and a
  noise drawn from the Gaussian distribution. Here, we consider differential
  privacy with respect to the addition/removal relation.

  When the sub-sampling probability is q, the worst-case privacy loss
  distribution is generated as follows:
  For ADD adjacency type:
  - Let mu_lower(x) := q * mu(x - sensitivity) + (1-q) * mu(x)
  - Sample x ~ mu_upper = mu and let the privacy loss be
    ln(mu_upper(x) / mu_lower(x)).
  For REMOVE adjacency type:
  - Let mu_upper(x) := q * mu(x + sensitivity) + (1-q) * mu(x)
  - Sample x ~ mu_lower = mu and let the privacy loss be
    ln(mu_upper(x) / mu_lower(x)).

  Note: When q = 1, the result privacy loss distributions for both ADD and
    REMOVE adjacency types are identical.
  """

  def __init__(self,
               standard_deviation: float,
               sensitivity: float = 1,
               pessimistic_estimate: bool = True,
               log_mass_truncation_bound: float = -50,
               sampling_prob: float = 1.0,
               adjacency_type: AdjacencyType = AdjacencyType.REMOVE) -> None:
    """Initializes the privacy loss of the Gaussian mechanism.

    Args:
      standard_deviation: the standard_deviation of the Gaussian distribution.
      sensitivity: the sensitivity of function f. (i.e. the maximum absolute
        change in f when an input to a single user changes.)
      pessimistic_estimate: a value indicating whether the rounding is done in
        such a way that the resulting epsilon-hockey stick divergence
        computation gives an upper estimate to the real value.
      log_mass_truncation_bound: the ln of the probability mass that might be
        discarded from the noise distribution. The larger this number, the more
        error it may introduce in divergence calculations.
      sampling_prob: sub-sampling probability, a value in (0,1].
      adjacency_type: type of adjacency relation to used for defining the
        privacy loss distribution.
    """
    if standard_deviation <= 0:
      raise ValueError(f'Standard deviation is not a positive real number: '
                       f'{standard_deviation}')
    if log_mass_truncation_bound > 0:
      raise ValueError(f'Log mass truncation bound is not a non-positive real '
                       f'number: {log_mass_truncation_bound}')

    self._standard_deviation = standard_deviation
    self._gaussian_random_variable = stats.norm(scale=standard_deviation)
    self._pessimistic_estimate = pessimistic_estimate
    self._log_mass_truncation_bound = log_mass_truncation_bound
    super().__init__(sensitivity, False, sampling_prob, adjacency_type)

  def privacy_loss_tail(self) -> TailPrivacyLossDistribution:
    """Computes the privacy loss at the tail of the Gaussian distribution.

    For REMOVE adjacency type: lower_x_truncation is set such that
      CDF(lower_x_truncation) = 0.5 * exp(log_mass_truncation_bound), and
      upper_x_truncation is set to be -lower_x_truncation. Finally,
      lower_x_truncation is shifted by -1 * sensitivity.
      Recall that here mu_upper(x) := (1-q).mu(x) + q.mu(x + sensitivity),
      where q=sampling_prob. The truncations chosen above ensure that the tails
      of both mu(x) and mu(x+sensitivity) are smaller than 0.5 *
      exp(log_mass_truncation_bound). This ensures that the considered tails of
      mu_upper are no larger than exp(log_mass_truncation_bound). This is
      computationally cheaper than computing exact tail thresholds for mu_upper.

    For ADD adjacency type: lower_x_truncation is set such that
      CDF(lower_x_truncation) = 0.5 * exp(log_mass_truncation_bound), and
      upper_x_truncation is set to be -lower_x_truncation. Finally,
      upper_x_truncation is shifted by +1 * sensitivity.
      Recall that here mu_upper(x) := mu(x) for any value of sampling_prob.
      The truncations chosen ensures that the tails of mu(x) (and hence of
      mu_upper) are no larger than 0.5 * exp(log_mass_truncation_bound).
      While it was not strictly necessary to shift upper_x_truncation by +1 *
      sensitivity in this case, this choice leads to the same discretized
      privacy loss distribution for both ADD and REMOVE adjacency
      types, in the case where sampling_prob = 1.

    If pessimistic_estimate is True, the privacy losses for
    x < lower_x_truncation and x > upper_x_truncation are rounded up and added
    to tail_probability_mass_function. In the case x < lower_x_truncation,
    the privacy loss is rounded up to infinity. In the case
    x > upper_x_truncation, it is rounded up to the privacy loss at
    upper_x_truncation.

    On the other hand, if pessimistic_estimate is False, the privacy losses for
    x < lower_x_truncation and x > upper_x_truncation are rounded down and added
    to tail_probability_mass_function. In the case x < lower_x_truncation, the
    privacy loss is rounded down to the privacy loss at lower_x_truncation.
    In the case x > upper_x_truncation, it is rounded down to -infinity and
    hence not included in tail_probability_mass_function,

    Returns:
      A TailPrivacyLossDistribution instance representing the tail of the
      privacy loss distribution.
    """
    lower_x_truncation = self._gaussian_random_variable.ppf(
        0.5 * math.exp(self._log_mass_truncation_bound))
    upper_x_truncation = -lower_x_truncation
    if self.adjacency_type == AdjacencyType.ADD:
      upper_x_truncation += self.sensitivity
    else:  # Case: self.adjacency_type == AdjacencyType.REMOVE
      lower_x_truncation -= self.sensitivity
    if self._pessimistic_estimate:
      tail_probability_mass_function = {
          math.inf:
              self.mu_upper_cdf(lower_x_truncation),
          self.privacy_loss(upper_x_truncation):
              1 - self.mu_upper_cdf(upper_x_truncation)
      }
    else:
      tail_probability_mass_function = {
          self.privacy_loss(lower_x_truncation):
              self.mu_upper_cdf(lower_x_truncation),
      }
    return TailPrivacyLossDistribution(lower_x_truncation, upper_x_truncation,
                                       tail_probability_mass_function)

  def connect_dots_bounds(self) -> ConnectDotsBounds:
    """Computes the bounds on epsilon values to use in connect-the-dots algorithm.

    epsilon_upper = privacy_loss(lower_x_truncation)
    epsilon_lower = privacy_loss(upper_x_truncation)

    where lower_x_truncation and upper_x_truncation are the lower and upper
    values of trunction as given by privacy_loss_tail().

    Returns:
      A ConnectDotsBounds instance containing upper and lower values of
      epsilon to use in connect-the-dots algorithm.
    """
    tail_pld = self.privacy_loss_tail()

    return ConnectDotsBounds(
        epsilon_upper=self.privacy_loss(tail_pld.lower_x_truncation),
        epsilon_lower=self.privacy_loss(tail_pld.upper_x_truncation))

  def privacy_loss_without_subsampling(self, x: float) -> float:
    """Computes the privacy loss of the Gaussian mechanism without sub-sampling at a given point.

    Args:
      x: the point at which the privacy loss is computed.

    Returns:
      The privacy loss of the Laplace mechanism at point x, which is given as
      For ADD adjacency type: (|x - sensitivity| - |x|) / parameter.
      For REMOVE adjacency type: (|x| - |x + sensitivity|) / parameter.
      The privacy loss of the Gaussian mechanism without sub-sampling at point
      x, which is given as
      For ADD adjacency type:
        sensitivity * (0.5 * sensitivity - x) / standard_deviation^2.
      For REMOVE adjacency type:
        sensitivity * (- 0.5 * sensitivity - x) / standard_deviation^2.
    """
    if self.adjacency_type == AdjacencyType.ADD:
      return (self.sensitivity * (0.5 * self.sensitivity - x) /
              (self._standard_deviation**2))
    else:  # Case: self.adjacency_type == AdjacencyType.REMOVE
      return (self.sensitivity * (-0.5 * self.sensitivity - x) /
              (self._standard_deviation**2))

  def inverse_privacy_loss_without_subsampling(self,
                                               privacy_loss: float) -> float:
    """Computes the inverse of a given privacy loss for the Gaussian mechanism without sub-sampling.

    Args:
      privacy_loss: the privacy loss value.

    Returns:
      The largest float x such that the privacy loss at x is at least
      privacy_loss. This is equal to
      For ADD adjacency type:
        0.5 * sensitivity - privacy_loss * standard_deviation^2 / sensitivity.
      For REMOVE adjacency type:
        -0.5 * sensitivity - privacy_loss * standard_deviation^2 / sensitivity.
    """
    if self.adjacency_type == AdjacencyType.ADD:
      return (0.5 * self.sensitivity - privacy_loss *
              (self._standard_deviation**2) / self.sensitivity)
    else:  # Case: self.adjacency_type == AdjacencyType.REMOVE
      return (-0.5 * self.sensitivity - privacy_loss *
              (self._standard_deviation**2) / self.sensitivity)

  def noise_cdf(self, x: Union[float,
                               Iterable[float]]) -> Union[float, np.ndarray]:
    """Computes the cumulative density function of the Gaussian distribution.

    Args:
     x: the point or points at which the cumulative density function is to be
       calculated.

    Returns:
      The cumulative density function of the Gaussian noise at x, i.e., the
      probability that the Gaussian noise is less than or equal to x.
    """
    return self._gaussian_random_variable.cdf(x)

  def noise_log_cdf(
      self, x: Union[float, Iterable[float]]) -> Union[float, np.ndarray]:
    """Computes log of cumulative density function of the Gaussian distribution.

    Args:
      x: the point or points at which the log cumulative density function is to
        be calculated.

    Returns:
      The log cumulative density function of the Gaussian noise at x, i.e., the
      log of the probability that the Gaussian noise is less than or equal to x.
    """
    return self._gaussian_random_variable.logcdf(x)

  @classmethod
  def from_privacy_guarantee(
      cls,
      privacy_parameters: common.DifferentialPrivacyParameters,
      sensitivity: float = 1,
      pessimistic_estimate: bool = True,
      sampling_prob: float = 1.0,
      adjacency_type: AdjacencyType = AdjacencyType.REMOVE
  ) -> 'GaussianPrivacyLoss':
    """Creates the privacy loss for Gaussian mechanism with desired privacy.

    Uses binary search to find the smallest possible standard deviation of the
    Gaussian noise for which the mechanism is (epsilon, delta)-differentially
    private, with respect to the REMOVE relation.

    Note: Only the REMOVE adjacency type is used in determining the parameter,
      since for all epsilon > 0, the hockey-stick divergence for PLD with
      respect to the REMOVE adjacency type is at least that for PLD with respect
      to ADD adjacency type.

    Args:
      privacy_parameters: the desired privacy guarantee of the mechanism.
      sensitivity: the sensitivity of function f. (i.e. the maximum absolute
        change in f when an input to a single user changes.)
      pessimistic_estimate: a value indicating whether the rounding is done in
        such a way that the resulting epsilon-hockey stick divergence
        computation gives an upper estimate to the real value.
      sampling_prob: sub-sampling probability, a value in (0,1].
      adjacency_type: type of adjacency relation to used for defining the
        privacy loss distribution.

    Returns:
      The privacy loss of the Gaussian mechanism with the given privacy
      guarantee.
    """
    if privacy_parameters.delta == 0:
      raise ValueError('delta=0 is not allowed for the Gaussian mechanism')

    # The initial standard deviation is set to
    # sqrt(2 * ln(1.5/delta)) * sensitivity / epsilon. It is known that, when
    # epsilon is no more than one, the Gaussian mechanism with this standard
    # deviation is (epsilon, delta)-DP. See e.g. Appendix A in Dwork and Roth
    # book, "The Algorithmic Foundations of Differential Privacy".
    search_parameters = common.BinarySearchParameters(
        0,
        math.inf,
        initial_guess=math.sqrt(2 * math.log(1.5 / privacy_parameters.delta)) *
        sensitivity / privacy_parameters.epsilon)

    def _get_delta_for_standard_deviation(current_standard_deviation):
      return GaussianPrivacyLoss(
          current_standard_deviation,
          sensitivity=sensitivity,
          sampling_prob=sampling_prob,
          adjacency_type=AdjacencyType.REMOVE).get_delta_for_epsilon(
              privacy_parameters.epsilon)

    standard_deviation = common.inverse_monotone_function(
        _get_delta_for_standard_deviation, privacy_parameters.delta,
        search_parameters)

    return GaussianPrivacyLoss(
        standard_deviation,
        sensitivity=sensitivity,
        pessimistic_estimate=pessimistic_estimate,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)

  @property
  def standard_deviation(self) -> float:
    """The standard deviation of the corresponding Gaussian noise."""
    return self._standard_deviation


class DiscreteLaplacePrivacyLoss(AdditiveNoisePrivacyLoss):
  """Privacy loss of the discrete Laplace mechanism.

  The discrete Laplace mechanism for computing an integer-valued function f
  simply outputs the sum of the true value of the function and a noise drawn
  from the discrete Laplace distribution. Recall that the discrete Laplace
  distribution with parameter a > 0 has probability mass function
  Z * exp(-a * |x|) at x for any integer x, where Z = (e^a - 1) / (e^a + 1).

  This class represents the privacy loss for the aforementioned
  discrete Laplace mechanism with a given parameter, and a given sensitivity of
  the function f. It is assumed that the function f only outputs an integer.
  The privacy loss distribution of the discrete Laplace mechanism is equivalent
  to that between the discrete Laplace distribution and the same distribution
  but shifted by the sensitivity. Specifically, the privacy loss
  distribution of the discrete Laplace mechanism is generated as follows:
  - Let mu = DLap(0, a) be the discrete Laplace noise PMF as given above.
  - Let mu_lower(x) := mu(x - sensitivity), i.e., right shifted by sensitivity
  - Sample x ~ mu_upper = mu and let the privacy loss be
    ln(mu_upper(x) / mu_lower(x)), which is equal to
    parameter * (|x - sensitivity| - |x|).

  Case of sub-sampling (Refer to supplementary material for more details):
  The discrete Laplace mechanism with sub-sampling for computing a scalar
  integer-valued function f, first samples a subset of data points including
  each data point independently with probability q, and returns the sum of the
  true values and a noise drawn from the discrete Laplace distribution. Here, we
  consider differential privacy with respect to the addition/removal relation.

  When the sub-sampling probability is q, the worst-case privacy loss
  distribution is generated as follows:
  For ADD adjacency type:
  - Let mu_lower(x) := q * mu(x - sensitivity) + (1-q) * mu(x)
  - Sample x ~ mu_upper = mu and let the privacy loss be
    ln(mu_upper(x) / mu_lower(x)).
  For REMOVE adjacency type:
  - Let mu_upper(x) := q * mu(x + sensitivity) + (1-q) * mu(x)
  - Sample x ~ mu_lower = mu and let the privacy loss be
    ln(mu_upper(x) / mu_lower(x)).

  Note: When q = 1, the result privacy loss distributions for both ADD and
    REMOVE adjacency types are identical.
  """

  def __init__(self,
               parameter: float,
               sensitivity: int = 1,
               sampling_prob: float = 1.0,
               adjacency_type: AdjacencyType = AdjacencyType.REMOVE) -> None:
    """Initializes the privacy loss of the discrete Laplace mechanism.

    Args:
      parameter: the parameter of the discrete Laplace distribution.
      sensitivity: the sensitivity of function f. (i.e. the maximum absolute
        change in f when an input to a single user changes.)
      sampling_prob: sub-sampling probability, a value in (0,1].
      adjacency_type: type of adjacency relation to used for defining the
        privacy loss distribution.
    """
    if parameter <= 0:
      raise ValueError(f'Parameter is not a positive real number: {parameter}')

    if not isinstance(sensitivity, int):
      raise ValueError(f'Sensitivity is not an integer : {sensitivity}')

    self._parameter = parameter
    self._discrete_laplace_random_variable = stats.dlaplace(parameter)
    super().__init__(sensitivity, True, sampling_prob, adjacency_type)

  def privacy_loss_tail(self) -> TailPrivacyLossDistribution:
    """Computes privacy loss at the tail of the discrete Laplace distribution.

    For ADD adjacency type:
    lower_x_truncation = 1 and upper_x_truncation = sensitivity-1

    For REMOVE adjacency type:
    lower_x_truncation = -sensitivity+1 and upper_x_truncation = -1

    The probability mass below lower_x_truncation and above upper_x_truncation
    are computed using mu_upper_cdf.

    Returns:
      A TailPrivacyLossDistribution instance representing the tail of the
      privacy loss distribution.
    """
    if self.adjacency_type == AdjacencyType.ADD:
      lower_x_truncation, upper_x_truncation = 1, self.sensitivity - 1
    else:  # Case: self.adjacency_type == AdjacencyType.REMOVE
      lower_x_truncation, upper_x_truncation = 1 - self.sensitivity, -1
    return TailPrivacyLossDistribution(
        lower_x_truncation, upper_x_truncation, {
            self.privacy_loss(lower_x_truncation - 1):
                self.mu_upper_cdf(lower_x_truncation - 1),
            self.privacy_loss(upper_x_truncation + 1):
                1 - self.mu_upper_cdf(upper_x_truncation)
        })

  def connect_dots_bounds(self) -> ConnectDotsBounds:
    """Computes the bounds on epsilon values to use in connect-the-dots algorithm.

    With sub-sampling probability of q,
    For ADD adjacency type:
      lower_x = 0 and upper_x = sensitivity

    For REMOVE adjacency type:
      lower_x = -sensitivity and upper_x = 0

    Returns:
      A ConnectDotsBounds instance containing lower and upper values of x
      to use in connect-the-dots algorithm.
    """
    if self.adjacency_type == AdjacencyType.ADD:
      lower_x, upper_x = 0, int(self.sensitivity)
    else:  # Case: self.adjacency_type == AdjacencyType.REMOVE
      lower_x, upper_x = -int(self.sensitivity), 0

    return ConnectDotsBounds(lower_x=lower_x, upper_x=upper_x)

  def privacy_loss_without_subsampling(self, x: float) -> float:
    """Computes privacy loss of the discrete Laplace mechanism without sub-sampling at a given point.

    Args:
      x: the point at which the privacy loss is computed.

    Returns:
      The privacy loss of the discrete Laplace mechanism without sub-sampling at
      integer value x, which is given as
      For ADD adjacency type:    parameter * (|x - sensitivity| - |x|).
      For REMOVE adjacency type: parameter * (|x| - |x + sensitivity|).
    """
    if not isinstance(x, int):
      raise ValueError(f'Privacy loss at x is undefined for x = {x}')

    if self.adjacency_type == AdjacencyType.ADD:
      return (abs(x - self.sensitivity) - abs(x)) * self._parameter
    else:  # Case: self.adjacency_type == AdjacencyType.REMOVE
      return (abs(x) - abs(x + self.sensitivity)) * self._parameter

  def inverse_privacy_loss_without_subsampling(self,
                                               privacy_loss: float) -> float:
    """Computes the inverse of a given privacy loss for the discrete Laplace mechanism.

    Args:
      privacy_loss: the privacy loss value.

    Returns:
      The largest float x such that the privacy loss at x is at least
      privacy_loss.
      For ADD adjacency type:
        If privacy_loss <= - sensitivity * parameter, x is equal to infinity.
        If - sensitivity * parameter < privacy_loss <= sensitivity * parameter,
          x is equal to floor(0.5 * (sensitivity - privacy_loss / parameter)).
        If privacy_loss > sensitivity * parameter, no such x exists and the
          function returns -infinity.
      For REMOVE adjacency type:
        For any value of privacy_loss, x is equal to the corresponding value for
          ADD adjacency type decreased by sensitivity.
    """
    loss_threshold = privacy_loss / self._parameter
    if loss_threshold > self.sensitivity:
      return -math.inf
    if loss_threshold <= -self.sensitivity:
      return math.inf
    if self.adjacency_type == AdjacencyType.ADD:
      return math.floor(0.5 * (self.sensitivity - loss_threshold))
    else:  # Case: self.adjacency_type == AdjacencyType.REMOVE
      return math.floor(0.5 * (-self.sensitivity - loss_threshold))

  def noise_cdf(self, x: Union[float,
                               Iterable[float]]) -> Union[float, np.ndarray]:
    """Computes cumulative density function of the discrete Laplace distribution.

    Args:
      x: the point or points at which the cumulative density function is to be
        calculated.

    Returns:
      The cumulative density function of the discrete Laplace noise at x, i.e.,
      the probability that the discrete Laplace noise is less than or equal to
      x.
    """
    return self._discrete_laplace_random_variable.cdf(x)

  def noise_log_cdf(
      self, x: Union[float, Iterable[float]]) -> Union[float, np.ndarray]:
    """Computes log of the CDF of the discrete Laplace distribution.

    Args:
      x: the point or points at which the log cumulative density function is to
        be calculated.

    Returns:
      The log cumulative density function of the discrete Laplace noise at x,
      i.e., the log of the probability that the discrete Laplace noise is less
      than or equal to x.
    """
    return self._discrete_laplace_random_variable.logcdf(x)

  @classmethod
  def from_privacy_guarantee(
      cls,
      privacy_parameters: common.DifferentialPrivacyParameters,
      sensitivity: int = 1,
      sampling_prob: float = 1.0,
      adjacency_type: AdjacencyType = AdjacencyType.REMOVE
  ) -> 'DiscreteLaplacePrivacyLoss':
    """Creates privacy loss for discrete Laplace mechanism with desired privacy.

    Without sub-sampling, the parameter of the Laplace mechanism is simply
      epsilon / sensitivity.
    With sub-sampling probability of q, the parameter is given as below.
      log(1 + (exp(epsilon) - 1)/q) / sensitivity,
    Note: Only the REMOVE adjacency type is used in determining the parameter,
      since for all epsilon > 0, the hockey-stick divergence for PLD with
      respect to the REMOVE adjacency type is at least that for PLD with respect
      to ADD adjacency type.

    Args:
      privacy_parameters: the desired privacy guarantee of the mechanism.
      sensitivity: the sensitivity of function f. (i.e. the maximum absolute
        change in f when an input to a single user changes.)
      sampling_prob: sub-sampling probability, a value in (0,1].
      adjacency_type: type of adjacency relation to used for defining the
        privacy loss distribution.

    Returns:
      The privacy loss of the discrete Laplace mechanism with the given privacy
      guarantee.
    """
    if not isinstance(sensitivity, int):
      raise ValueError(f'Sensitivity is not an integer : {sensitivity}')
    if sensitivity <= 0:
      raise ValueError(
          f'Sensitivity is not a positive real number: {sensitivity}')
    if sampling_prob <= 0 or math.isclose(sampling_prob, 0):
      raise ValueError(
          f'Sampling probability ({sampling_prob}) is equal or too close to 0.')
    parameter = (
        np.log(1 + (np.exp(privacy_parameters.epsilon) - 1) / sampling_prob) /
        sensitivity)

    return DiscreteLaplacePrivacyLoss(
        parameter,
        sensitivity=sensitivity,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)

  @property
  def parameter(self) -> float:
    """The parameter of the corresponding Discrete Laplace noise."""
    return self._parameter


class DiscreteGaussianPrivacyLoss(AdditiveNoisePrivacyLoss):
  """Privacy loss of the discrete Gaussian mechanism.

  The discrete Gaussian mechanism for computing a scalar-valued function f
  simply outputs the sum of the true value of the function and a noise drawn
  from the discrete Gaussian distribution. Recall that the (centered) discrete
  Gaussian distribution with parameter sigma has probability mass function
  proportional to exp(-0.5 x^2/sigma^2) at x for any integer x. Since its
  normalization factor and cumulative density function do not have a closed
  form, we will instead consider the truncated version where the noise x is
  restricted to only be in [-truncated_bound, truncated_bound].

  The privacy loss distribution of the discrete Gaussian mechanism is equivalent
  to the privacy loss distribution between the discrete Gaussian distribution
  and the same distribution but shifted by the sensitivity of f. Specifically,
  the privacy loss distribution of the discrete Gaussian mechanism is generated
  as follows:
  - Let mu = N_Z(0, sigma^2, truncation_bound) be the discrete Gaussian noise
    PMF as given above.
  - Let mu_lower(x) := mu(x - sensitivity), i.e., right shifted by sensitivity
  - Sample x ~ mu_upper = mu and let the privacy loss be
    ln(mu_upper(x) / mu_lower(x)).
  Note that since we consider the truncated version of the noise, we set the
  privacy loss to infinity when x < -truncation_bound + sensitivity.

  Case of sub-sampling (Refer to supplementary material for more details):
  The discrete Gaussian mechanism with sub-sampling for computing a scalar
  integer-valued function f, first samples a subset of data points including
  each data point independently with probability q, and returns the sum of the
  true values and a noise drawn from the discrete Gaussian distribution. Here,
  we consider differential privacy with respect to the
  addition/removal relation.

  When the sub-sampling probability is q, the worst-case privacy loss
  distribution is generated as follows:
  For ADD adjacency type:
  - Let mu_lower(x) := q * mu(x - sensitivity) + (1-q) * mu(x)
  - Sample x ~ mu_upper = mu and let the privacy loss be
    ln(mu_upper(x) / mu_lower(x)).
  For REMOVE adjacency type:
  - Let mu_upper(x) := q * mu(x + sensitivity) + (1-q) * mu(x)
  - Sample x ~ mu_lower = mu and let the privacy loss be
    ln(mu_upper(x) / mu_lower(x)).

  Note: When q = 1, the result privacy loss distributions for both ADD and
    REMOVE adjacency types are identical.

  Reference:
  Canonne, Kamath, Steinke. "The Discrete Gaussian for Differential Privacy".
  In NeurIPS 2020.
  """

  def __init__(self,
               sigma: float,
               sensitivity: int = 1,
               truncation_bound: Optional[int] = None,
               sampling_prob: float = 1.0,
               adjacency_type: AdjacencyType = AdjacencyType.REMOVE) -> None:
    """Initializes the privacy loss of the discrete Gaussian mechanism.

    Args:
      sigma: the parameter of the discrete Gaussian distribution. Note that
        unlike the (continuous) Gaussian distribution this is not equal to the
        standard deviation of the noise.
      sensitivity: the sensitivity of function f. (i.e. the maximum absolute
        change in f when an input to a single user changes.)
      truncation_bound: bound for truncating the noise, i.e. the noise will only
        have a support in [-truncation_bound, truncation_bound]. When not
        specified, truncation_bound will be chosen in such a way that the mass
        of the noise outside of this range is at most 1e-30.
      sampling_prob: sub-sampling probability, a value in (0,1].
      adjacency_type: type of adjacency relation to used for defining the
        privacy loss distribution.
    """
    if sigma <= 0:
      raise ValueError(f'Sigma is not a positive real number: {sigma}')
    if not isinstance(sensitivity, int):
      raise ValueError(f'Sensitivity is not an integer : {sensitivity}')

    self._sigma = sigma
    if truncation_bound is None:
      # Tail bound from Canonne et al. ensures that the mass that gets truncated
      # is at most 1e-30. (See Proposition 1 in the supplementary material.)
      self._truncation_bound = math.ceil(11.6 * sigma)
    else:
      self._truncation_bound = truncation_bound

    if 2 * self._truncation_bound < sensitivity:
      raise ValueError(f'Truncation bound ({truncation_bound}) is smaller '
                       f'than 0.5 * sensitivity (0.5 * {sensitivity})')

    # Create the PMF and CDF.
    self._offset = -1 * self._truncation_bound - 1
    indices = np.arange(-1 * self._truncation_bound, self._truncation_bound + 1)
    self._log_pmf_array = np.append(-np.inf, -0.5 * indices**2 / (sigma**2))
    self._log_cdf_array = np.logaddexp.accumulate(self._log_pmf_array)
    self._log_pmf_array -= self._log_cdf_array[-1]
    self._log_cdf_array -= self._log_cdf_array[-1]
    self._pmf_array = np.exp(self._log_pmf_array)
    self._cdf_array = np.exp(self._log_cdf_array)

    super().__init__(sensitivity, True, sampling_prob, adjacency_type)

  def privacy_loss_tail(self) -> TailPrivacyLossDistribution:
    """Computes the privacy loss at the tail of the discrete Gaussian distribution.

    The lower_x_truncation and upper_x_truncation are chosen such that for any
    x < lower_x_truncation, the privacy loss is +infinity (or undefined), and
    for any
    x > upper_x_truncation, the privacy loss is -infinity (or undefined).

    With sampling probability of q, the privacy loss tail is given as
    For ADD adjacency type:
    (if q == 1) lower_x_truncation = sensitivity - truncation_bound
    (if q < 1)  lower_x_truncation = - truncation_bound
    In either case, upper_x_truncation = truncation_bound

    For REMOVE adjacency type:
    (if q == 1) upper_x_truncation = truncation_bound - sensitivity
    (if q < 1)  upper_x_truncation = truncation_bound
    In either case, lower_x_truncation = - truncation_bound

    Returns:
      A TailPrivacyLossDistribution instance representing the tail of the
      privacy loss distribution.
    """
    if self.adjacency_type == AdjacencyType.ADD:
      upper_x_truncation = self._truncation_bound
      if self.sampling_prob == 1.0:
        lower_x_truncation = self.sensitivity - self._truncation_bound
      else:
        lower_x_truncation = -1 * self._truncation_bound
    else:  # Case: self.adjacency_type == AdjacencyType.REMOVE
      lower_x_truncation = -1 * self._truncation_bound
      if self.sampling_prob == 1.0:
        upper_x_truncation = self._truncation_bound - self.sensitivity
      else:
        upper_x_truncation = self._truncation_bound

    return TailPrivacyLossDistribution(
        lower_x_truncation, upper_x_truncation,
        {math.inf: self.mu_upper_cdf(lower_x_truncation - 1)})

  def connect_dots_bounds(self) -> ConnectDotsBounds:
    """Computes the bounds on epsilon values to use in connect-the-dots algorithm.

    lower_x and upper_x are same as lower_x_truncation and upper_x_truncation
    as given by privacy_loss_tail().

    Returns:
      A ConnectDotsBounds instance containing lower and upper values of x
      to use in connect-the-dots algorithm.
    """
    tail_pld = self.privacy_loss_tail()

    return ConnectDotsBounds(lower_x=int(tail_pld.lower_x_truncation),
                             upper_x=int(tail_pld.upper_x_truncation))

  def privacy_loss_without_subsampling(self, x: float) -> float:
    """Computes the privacy loss of the discrete Gaussian mechanism without sub-sampling at a given point.

    Args:
      x: the point at which the privacy loss is computed.

    Returns:
      The privacy loss of the discrete Gaussian mechanism at integer value x,
      which is given as

      For ADD adjacency type:
      If x lies in [-truncation_bound + sensitivity, truncation_bound],
        it is equal to sensitivity * (0.5 * sensitivity - x) / sigma^2.
      If x lies in [-truncation_bound, -truncation_bound + sensitivity),
        it is equal to infinity.
      If x lies in (truncation_bound, trunction_bound + sensitivity],
        it is equal to -infinity.
      Otherwise, the privacy loss is undefined (ValueError is raised).

      For REMOVE adjacency type:
       Same as the case of ADD with x replaced by x + sensitivity.

    Raises:
      ValueError: if the privacy loss is undefined.
    """
    def privacy_loss_without_subsampling_for_add(x: float) -> float:
      if (not isinstance(x, int) or x < -1 * self._truncation_bound or
          x > self._truncation_bound + self.sensitivity):
        actual_x = (
            x if self.adjacency_type == AdjacencyType.ADD else
            x - self.sensitivity)
        raise ValueError(f'Privacy loss at x is undefined for x = {actual_x}')
      if x > self._truncation_bound:
        return -math.inf
      if x < self.sensitivity - self._truncation_bound:
        return math.inf
      return self.sensitivity * (0.5 * self.sensitivity - x) / (self._sigma**2)

    if self.adjacency_type == AdjacencyType.ADD:
      return privacy_loss_without_subsampling_for_add(x)
    else:  # Case: self.adjacency_type == AdjacencyType.REMOVE
      return privacy_loss_without_subsampling_for_add(x + self.sensitivity)

  def inverse_privacy_loss_without_subsampling(self,
                                               privacy_loss: float) -> float:
    """Computes the inverse of a given privacy loss for the discrete Gaussian mechanism without sub-sampling.

    Args:
      privacy_loss: the privacy loss value.

    Returns:
      The largest int x such that the privacy loss at x is at least
      privacy_loss, which is given as
      For ADD adjacency type:
        floor(0.5 * sensitivity - privacy_loss * sigma^2 / sensitivity) clipped
        to the interval [sensitivity - truncation_bound - 1, truncation_bound].
      For REMOVE adjacency type:
        Same as that for ADD decreased by sensitivity.
    """

    def inverse_privacy_loss_without_subsampling_for_add(
        privacy_loss: float) -> float:
      if privacy_loss == -math.inf:
        return self._truncation_bound
      return math.floor(
          np.clip(
              0.5 * self.sensitivity - privacy_loss * (self._sigma**2) /
              self.sensitivity,
              self.sensitivity - self._truncation_bound - 1,
              self._truncation_bound))

    if self.adjacency_type == AdjacencyType.ADD:
      return inverse_privacy_loss_without_subsampling_for_add(privacy_loss)
    else:  # Case: self.adjacency_type == AdjacencyType.REMOVE
      return (inverse_privacy_loss_without_subsampling_for_add(privacy_loss) -
              self.sensitivity)

  def noise_cdf(self, x: Union[float,
                               Iterable[float]]) -> Union[float, np.ndarray]:
    """Computes the CDF of the discrete Gaussian distribution.

    Args:
      x: the point or points at which the cumulative density function is to be
        calculated.

    Returns:
      The cumulative density function of the discrete Gaussian noise at x, i.e.,
      the probability that the discrete Gaussian noise is less than or equal to
      x.
    """
    clipped_x = np.clip(x, -1 * self._truncation_bound - 1,
                        self._truncation_bound)
    indices = np.floor(clipped_x).astype('int') - self._offset
    return self._cdf_array[indices]

  def noise_log_cdf(
      self, x: Union[float, Iterable[float]]) -> Union[float, np.ndarray]:
    """Computes log of the CDF of the discrete Gaussian distribution.

    Args:
      x: the point or points at which the log cumulative density function is to
        be calculated.

    Returns:
      The log cumulative density function of the discrete Gaussian noise at x,
      i.e., the log of the probability that the discrete Gaussian noise is less
      than or equal to x.
    """
    clipped_x = np.clip(x, -1 * self._truncation_bound - 1,
                        self._truncation_bound)
    indices = np.floor(clipped_x).astype('int') - self._offset
    return self._log_cdf_array[indices]

  @classmethod
  def from_privacy_guarantee(
      cls,
      privacy_parameters: common.DifferentialPrivacyParameters,
      sensitivity: int = 1,
      sampling_prob: float = 1.0,
      adjacency_type: AdjacencyType = AdjacencyType.REMOVE
  ) -> 'DiscreteGaussianPrivacyLoss':
    """Creates the privacy loss for discrete Gaussian mechanism with desired privacy.

    Uses binary search to find the smallest possible standard deviation of the
    discrete Gaussian noise for which the protocol is (epsilon, delta)-DP.

    Note: Only the REMOVE adjacency type is used in determining the parameter,
      since for all epsilon > 0, the hockey-stick divergence for PLD with
      respect to the REMOVE adjacency type is at least that for PLD with respect
      to ADD adjacency type.

    Args:
      privacy_parameters: the desired privacy guarantee of the mechanism.
      sensitivity: the sensitivity of function f. (i.e. the maximum absolute
        change in f when an input to a single user changes.)
      sampling_prob: sub-sampling probability, a value in (0,1].
      adjacency_type: type of adjacency relation to used for defining the
        privacy loss distribution.

    Returns:
      The privacy loss of the discrete Gaussian mechanism with the given privacy
      guarantee.
    """
    if not isinstance(sensitivity, int):
      raise ValueError(f'Sensitivity is not an integer : {sensitivity}')
    if privacy_parameters.delta == 0:
      raise ValueError('delta=0 is not allowed for discrete Gaussian mechanism')

    # The initial standard deviation is set to
    # sqrt(2 * ln(1.5/delta)) * sensitivity / epsilon. It is known that, when
    # epsilon is no more than one, the (continuous) Gaussian mechanism with this
    # standard deviation is (epsilon, delta)-DP. See e.g. Appendix A in Dwork
    # and Roth book, "The Algorithmic Foundations of Differential Privacy".
    search_parameters = common.BinarySearchParameters(
        0,
        math.inf,
        initial_guess=math.sqrt(2 * math.log(1.5 / privacy_parameters.delta)) *
        sensitivity / privacy_parameters.epsilon)

    def _get_delta_for_sigma(current_sigma):
      return DiscreteGaussianPrivacyLoss(
          current_sigma,
          sensitivity=sensitivity,
          sampling_prob=sampling_prob,
          adjacency_type=AdjacencyType.REMOVE).get_delta_for_epsilon(
              privacy_parameters.epsilon)

    sigma = common.inverse_monotone_function(_get_delta_for_sigma,
                                             privacy_parameters.delta,
                                             search_parameters)

    return DiscreteGaussianPrivacyLoss(
        sigma,
        sensitivity=sensitivity,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)

  def standard_deviation(self) -> float:
    """The standard deviation of the corresponding discrete Gaussian noise."""
    return math.sqrt(
        sum(((i + self._offset)**2) * probability_mass
            for i, probability_mass in enumerate(self._pmf_array)))


@np.vectorize
def _pl_without_sampling_add(pl, sampling_prob):
  return -common.log_a_times_exp_b_plus_c(
      1 / (sampling_prob), -pl, 1 - 1 / (sampling_prob)
  )


@np.vectorize
def _pl_without_sampling_remove(pl, sampling_prob):
  return -_pl_without_sampling_add(-pl, sampling_prob)


class DoubleMixturePrivacyLoss(MonotonePrivacyLoss):
  """Privacy loss of a Double Mixture mechanism.

  It is the privacy loss distribution between two mixtures, each of which
  is some base continuous distribution convolved with
  a discrete distribution specified by sensitivities and sampling_probs.
  That is, let mu be some continuous PDF.
  The privacy loss distribution is generated as follows:
  For ADD adjacency type:
  - Let mu_upper(x) := sum over i of sampling_probs_upper[i] *
    mu(x + sensitivities_upper[i])
  - Let mu_lower(x) := sum over j of sampling_probs_lower[j] *
    mu(x - sensitivities_lower[j])
  - Sample x ~ mu_upper and let the privacy loss be
    ln(mu_upper(x) / mu_lower(x)).

  For example, when mu is a zero-mean Gaussian distribution:
    - sensitivities_upper = [1.0], sampling_probs_upper = [1.0],
      sensitivities_lower = [0.0], sampling_probs_lower = [1.0]
      captures the sensitivity-1 Gaussian mechanism.
    - sensitivities_upper = [0.0, 1.0], sampling_probs_upper = [0.9, 0.1]
      sensitivities_lower = [0.0], sampling_probs_lower = [1.0]
      captures the sensitivity-1 subsampled Gaussian mechanism.
    - sensitivities_upper = [0.0, 1.0], sampling_probs_upper = [0.9, 0.1]
      sensitivities_lower = [0.0, 1.0, 3.0],
      sampling_probs_lower = [0.7, 0.1, 0.2]
      captures a generalization where mu_upper and mu_lower have different sensitivities,
      where mu_lower samples sensitivity 1 with probability 0.1
      and samples sensitivity 3 with probability 0.2.

  The concrete methods in this base class make the following assumptions
  about privacy loss l(x) = log(mu_upper(x) / mu_lower(x)):
  - l is continuous
  - l is strictly decreasing on some interval (a, b) with a < b
  - l is constant outside this interval
  These assumptions hold for common mu (Gausian, Laplace, ...),
  but methods may have to be overriden for atypical distributions.

  Attributes:
    sensitivities_upper: the support of the upper sensitivity distribution.
    sensitivities_lower: the support of the lower sensitivity distribution.
    sampling_probs_upper: the probabilities associated with sensitivities_upper.
    sampling_probs_lower: the probabilities associated with sensitivities_upper.
  """
  def __init__(  # pylint: disable=super-init-not-called
      self,
      sensitivities_upper: Sequence[float],
      sensitivities_lower: Sequence[float],
      sampling_probs_upper: Sequence[float],
      sampling_probs_lower: Sequence[float],
      pessimistic_estimate: bool = True,
      log_mass_truncation_bound: float = -50
  ) -> None:
    """Initializes the privacy loss of a DoubleMixture mechanism.

    Args:
      sensitivities_upper: The support of the upper sensitivity distribution.
        Must be the ame length as sampling_probs_upper, and both should be 1D.
      sampling_probs_upper: Probabilities associated with sensitivities_upper.
      sensitivities_lower: The support of the lower sensitivity distribution.
        Must be the ame length as sampling_probs_lower, and both should be 1D.
      sampling_probs_lower: Probabilities associated with sensitivities_lower.
      pessimistic_estimate: A value indicating whether the rounding is done in
        such a way that the resulting epsilon-hockey stick divergence
        computation gives an upper estimate to the real value.
      log_mass_truncation_bound: The ln of the probability mass that might be
        discarded from the noise distribution. The larger this number, the more
        error it may introduce in divergence calculations.

    Raises:
      ValueError: If args are invalid, e.g. sensitivities and sampling_probs
        are different lengths.
    """

    if log_mass_truncation_bound > 0:
      raise ValueError(
          'Log mass truncation bound is not a non-positive real '
          f'number: {log_mass_truncation_bound}'
      )

    if len(sampling_probs_upper) != len(sensitivities_upper):
      raise ValueError(
          'sensitivities and sampling_probs must have the same '
          f'length. Got {sampling_probs_upper=} '
          f'of length {len(sampling_probs_upper)}, '
          f'{sensitivities_upper=} of length {len(sensitivities_upper)}.'
      )

    if len(sampling_probs_lower) != len(sampling_probs_lower):
      raise ValueError(
          'sensitivities and sampling_probs must have the same '
          f'length. Got {sampling_probs_upper=} '
          f'of length {len(sampling_probs_upper)}, '
          f'{sampling_probs_lower=} of length {len(sampling_probs_lower)}.'
      )

    super().__init__(is_discrete=False)

    non_zero_indices_upper = np.asarray(sampling_probs_upper) != 0.0
    sensitivities_upper = np.asarray(sensitivities_upper)[
                              non_zero_indices_upper]
    sampling_probs_upper = np.asarray(sampling_probs_upper)[
                              non_zero_indices_upper]

    non_zero_indices_lower = np.asarray(sampling_probs_lower) != 0.0
    sensitivities_lower = np.asarray(sensitivities_lower)[
                              non_zero_indices_lower]
    sampling_probs_lower = np.asarray(sampling_probs_lower)[
                              non_zero_indices_lower]

    if np.any(sensitivities_upper < 0) or np.any(sensitivities_lower < 0):
        raise ValueError(
            'Sensitivities contain a negative number. '
            f'Got {sensitivities_upper=}, '
            f'{sensitivities_lower=}.'
        )

    if not (math.isclose(sum(sampling_probs_upper), 1)
            and
            math.isclose(sum(sampling_probs_lower), 1)):
        raise ValueError(
            'Probabilities do not add up to 1. '
            f'sum(sampling_probs_upper)={sum(sampling_probs_upper)}, '
            f'sum(sampling_probs_lower)={sum(sampling_probs_lower)}.'
        )

    if (np.any((sampling_probs_upper <= 0) | (sampling_probs_upper > 1))
        or np.any((sampling_probs_lower <= 0) | (sampling_probs_lower > 1))):

        raise ValueError(
            'Sampling probabilities are not in (0,1].'
        )

    self.sampling_probs_upper = sampling_probs_upper
    self.sensitivities_upper = sensitivities_upper
    self.sampling_probs_lower = sampling_probs_lower
    self.sensitivities_lower = sensitivities_lower

    self._pessimistic_estimate = pessimistic_estimate
    self._log_mass_truncation_bound = log_mass_truncation_bound

    # Constant properties.
    self._log_sampling_probs_upper = np.log(self.sampling_probs_upper)
    self._pos_sampling_probs_upper = self.sampling_probs_upper[
                                        self.sensitivities_upper > 0.0]
    self._sampling_prob_upper = np.clip(self._pos_sampling_probs_upper.sum(),
                                        0, 1)
    self._max_sens_upper = self.sensitivities_upper.max()

    self._log_sampling_probs_lower = np.log(self.sampling_probs_lower)
    self._pos_sampling_probs_lower = self.sampling_probs_lower[
                                        self.sensitivities_lower > 0.0]
    self._sampling_prob_lower = np.clip(self._pos_sampling_probs_lower.sum(),
                                        0, 1)
    self._max_sens_lower = self.sensitivities_lower.max()

    if (self._max_sens_upper <= 0) and (self._max_sens_lower <= 0):
      raise ValueError('Must have at least one positive sensitivity, '
                       f'but {self._max_sens_upper=} and '
                       f'{self._max_sens_lower=}.')

  @property
  @abc.abstractmethod
  def _strictly_decreasing_interval(self) -> Optional[Tuple[float, float]]:
    """Interval on which privacy loss is strictly decreasing.

    Returns:
        Optional[Tuple[float, float]]: Left and right boundary of the interval.
          None if no such interval exists or privacy loss is not
          constant outside this interval.
    """
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def _privacy_loss_at_boundaries(self) -> Optional[Tuple[float, float]]:
    """Privacy loss at left and right boundary of _strictly_decreasing_interval.

    Returns:
        Optional[Tuple[float, float]]: Privacy loss l(a) and l(b)
          when _strictly_decreasing_interval=(a, b).
          None if _strictly_decreasing_interval=None.
    """
    raise NotImplementedError

  def _verify_monotonicity(self) -> None:
    """Verifies that privacy loss fulfills monotonicity requirements.

    These requiremnts are:
    - l is strictly decreasing on some interval (a, b) with a < b
    - l is constant outside this interval

    Raises:
        ValueError: If monotonicity requirements are violated.
    """
    if ((self._strictly_decreasing_interval is None)
        or (self._strictly_decreasing_interval[1]
            <= self._strictly_decreasing_interval[0])):
      raise ValueError('Privacy loss must be strictly  '
                       'decreasing on some non-empty interval, but '
                       f'{self._strictly_decreasing_interval=}.')

  def mu_upper_cdf(
      self, x: Union[float, Iterable[float]]
  ) -> Union[float, np.ndarray]:
    """Computes the cumulative density function of the mu_upper distribution.

      mu_upper(x) := sum of sampling_probs_upper[i] *
                      mu(x + sensitivities_upper[i])

    Args:
      x: the point or points at which the cumulative density function is to be
        calculated.

    Returns:
      The cumulative density function of the mu_upper distribution at x, i.e.,
      the probability that mu_upper is less than or equal to x.
    """
    points_per_sens = np.add.outer(np.atleast_1d(x), self.sensitivities_upper)
    output = (self.noise_cdf(points_per_sens) * self.sampling_probs_upper).sum(
        axis=1
    )
    if isinstance(x, numbers.Number):
      return output[0]
    else:
      return output

  def mu_lower_log_cdf(
      self, x: Union[float, Iterable[float]]
  ) -> Union[float, np.ndarray]:
    """Computes log cumulative density function of the mu_lower distribution.

      mu_lower(x) := sum of sampling_probs_lower[i] *
                      mu(x - sensitivities_lower[i])

    Args:
      x: the point or points at which the log of the cumulative density function
        is to be calculated.

    Returns:
      The log of the cumulative density function of the mu_lower distribution at
      x, i.e., the log of the probability that mu_lower is less than or equal to
      x.
    """
    points_per_sens = np.add.outer(np.atleast_1d(x), -self.sensitivities_lower)
    logcdf_per_sens = self.noise_log_cdf(points_per_sens)
    output = scipy.special.logsumexp(
        logcdf_per_sens, axis=1, b=self.sampling_probs_lower
    )
    if isinstance(x, numbers.Number):
      return output[0]
    else:
      return output

  def get_delta_for_epsilon(
      self, epsilon: Union[float, Sequence[float]],
  ) -> Union[float, Sequence[float]]:
    """Computes the epsilon-hockey stick divergence of the mechanism.

    Args:
      epsilon: the epsilon, or list-like object of epsilon values, in
      epsilon-hockey stick divergence.

    Returns:
      A non-negative real number which is the epsilon-hockey stick divergence of
      the mechanism, or a numpy array if epsilon is list-like.

    Raises:
      ValueError: If monotonicity requirements for privacy loss or epsilons
        are violated.
    """
    self._verify_monotonicity()

    is_scalar = isinstance(epsilon, numbers.Number)
    epsilons = np.array([epsilon]) if is_scalar else np.asarray(epsilon)
    if not np.all(epsilons[1:] >= epsilons[:-1]):
      raise ValueError(f'Epsilon values must be non-decreasing: {epsilons}')
    deltas = np.zeros_like(epsilons, dtype=float)

    # Computing delta requires that we determine (log)-probabilitites of
    # pre-images of the set {r | r > epsilon} under privacy loss l.

    # Preimage = {}
    beyond_left_boundary = (epsilons >= self._privacy_loss_at_boundaries[0])

    # Preimage = (-infty, infty)
    beyond_right_boundary = (epsilon < self._privacy_loss_at_boundaries[1])
    # In case of asymptotic bounds (strictly_decreasing_interval[1]=infty),
    # privacy loss is always marginally larger than
    # self._privacy_loss_at_boundaries[1]
    if self._strictly_decreasing_interval[1] == np.infty:
        beyond_right_boundary = np.logical_or(
          beyond_right_boundary,
          epsilon == self._privacy_loss_at_boundaries[1])
    deltas[beyond_right_boundary] = -np.expm1(epsilons[beyond_right_boundary])

    # Preimage = (-infty, b)
    # In case of non-asymptotic bounds, privacy loss is only larger than
    # self._privacy_loss_at_boundaries[1] for x < b
    # where b = self._strictly_decreasing_interval[1]
    if self._strictly_decreasing_interval[1] < np.infty:
      at_right_boundary = (epsilon == self._privacy_loss_at_boundaries[1])
      deltas[at_right_boundary] = (
          self.mu_upper_cdf(self._strictly_decreasing_interval[1]) -
          np.exp(epsilons[at_right_boundary] +
                 self.mu_lower_log_cdf(self._strictly_decreasing_interval[1])))
    else:
      at_right_boundary = np.full_like(epsilons, False, dtype=bool)

    # Preimage = (-infty, c) with some c s.t. a  < c < b. This c can be found
    # via the inverse privacy loss on the strictly decreasing interval (a, b).
    inverse_indices = np.logical_not(
                        np.logical_or(beyond_left_boundary,
                                      np.logical_or(beyond_right_boundary,
                                                    at_right_boundary)))

    x_cutoffs = self.inverse_privacy_losses(epsilons[inverse_indices])
    deltas[inverse_indices] = (
        self.mu_upper_cdf(x_cutoffs) -
        np.exp(epsilons[inverse_indices] + self.mu_lower_log_cdf(x_cutoffs)))
    # Clip delta values to lie in [0,1] (to avoid numerical errors)
    deltas = np.clip(deltas, 0, 1)
    if isinstance(epsilon, numbers.Number):
      return float(deltas)
    else:
      # For numerical stability reasons, deltas may not be non-increasing. This
      # is fixed post-hoc at small cost in accuracy.
      for i in reversed(range(deltas.shape[0] - 1)):
        deltas[i] = max(deltas[i], deltas[i + 1])
      return deltas

  def privacy_loss_tail(
        self, precision: float = 1e-4
  ) -> TailPrivacyLossDistribution:
    """Computes the privacy loss at the tail of the random-sensitivity Gaussian.

    If max(sensitivity_upper) = 0: The upper distribution has a single component
      and we can exactly compute the tails easily.

    Otherwise:  We set upper_x_truncation such that
      CDF(upper_x_truncation) = 1 - 0.5 * exp(log_mass_truncation_bound). It is
      worthwhile to spend some up-front computation getting a more precise value
      for lower_x_truncation to save computation later on. So we binary search
      over the interval [-upper_x_truncation - max(sensitivities),
      -upper_x_truncation] for the point where the cdf of mu_upper is
      0.5 * exp(log_mass_truncation_bound). Since we're binary searching over a
      continuous domain, we proceed until the width of the binary search
      interval is at most some small precision, and then set lower_x_truncation
      to be the left endpoint of this interval.

    Args:
      precision: The additive error we will compute the truncation values
        within. That is, we terminate the binary search when the interval has
        length at most precision, and then use the more conservative endpoint
        of the interval as our truncation value.

    Returns:
      A TailPrivacyLossDistribution instance representing the tail of the
      privacy loss distribution.
    """
    tail_mass = 0.5 * np.exp(self._log_mass_truncation_bound)
    z_value = self.noise_ppf(tail_mass)
    upper_x_truncation = -z_value
    if self._max_sens_upper == 0.0:
      lower_x_truncation = z_value
    else:
      lower_x_truncation = common.inverse_monotone_function(
          self.mu_upper_cdf,
          tail_mass,
          common.BinarySearchParameters(
              z_value - self._max_sens_upper,
              z_value,
              tolerance=precision
          ),
          increasing=True,
      )
    if self._pessimistic_estimate:
      tail_probability_mass_function = {
          math.inf: self.mu_upper_cdf(lower_x_truncation),
          self.privacy_loss(upper_x_truncation): 1 - self.mu_upper_cdf(
              upper_x_truncation
          ),
      }
    else:
      tail_probability_mass_function = {
          self.privacy_loss(lower_x_truncation): self.mu_upper_cdf(
              lower_x_truncation
          ),
      }
    return TailPrivacyLossDistribution(
        lower_x_truncation, upper_x_truncation, tail_probability_mass_function
    )

  def connect_dots_bounds(self) -> ConnectDotsBounds:
    """Computes bounds on epsilon values to use in connect-the-dots algorithm.

    Returns:
      A ConnectDotsBounds instance containing upper and lower values of
      epsilon to use in connect-the-dots algorithm.
    """
    tail_pld = self.privacy_loss_tail()

    return ConnectDotsBounds(
        epsilon_upper=self.privacy_loss(tail_pld.lower_x_truncation),
        epsilon_lower=self.privacy_loss(tail_pld.upper_x_truncation),
    )

  def privacy_loss(self, x: float) -> float:
    """Computes the privacy loss at a given point `x`."""
    p_upper = scipy.special.logsumexp(
                self.noise_log_pdf(x + self.sensitivities_upper),
                b=self.sampling_probs_upper)

    p_lower = scipy.special.logsumexp(
                self.noise_log_pdf(x - self.sensitivities_lower),
                b=self.sampling_probs_lower)

    return p_upper - p_lower

  def inverse_privacy_loss(
      self, privacy_loss: float, precision: float = 1e-6
  ) -> float:
    """(Approximately) computes the inverse of a given privacy loss.

    Technically, this method can be sped up by rewriting the logic in
    inverse_privacy_losses to take advantage of the fact that we have a
    single privacy loss rather than a list. However, this method is only written
    to complete the abstract class, and the process of generating a PLD from
    this class won't ever call this method. So, we have chosen the simple but
    inefficient implementation of calling inverse_privacy_losses.

    Args:
      privacy_loss: the privacy loss value.
      precision: Precision of the output.

    Returns:
      The largest float x such that the privacy loss at x is at least
      privacy_loss, rounded down to the nearest multiple of precision if
      we are using pessimistic estimates, and otherwise rounded up.
    """
    return float(
        self.inverse_privacy_losses(np.atleast_1d(privacy_loss), precision)[0]
    )

  def inverse_privacy_losses(
      self,
      privacy_losses: np.ndarray,
      precision: float = 1e-6,
  ) -> np.ndarray:
    """(Approximately) computes the inverse of a list of privacy losses.

    Unlike subsampled Gaussians, the privacy loss generally does not have
    a closed-form inverse. So, we use binary search.
    This is the main bottleneck in this library, so we optimize
    it by doing one binary search for all values in privacy losses rather than a
    separate binary search for each. This way, we avoid recomputing the privacy
    loss at the same point across different binary searches.

    Args:
      privacy_losses: the privacy losses we wish to invert, in increasing order.
      precision: Precision of the output. In particular, for each entry l in
        privacy_losses, we output the smallest multiple of precision, x, such
        that the privacy loss at x is at most l. This ensures (i) given a
        monotonic privacy_losses, we return a monotonic list of xs, and (ii) the
        approximation results in an overestimate of epsilon, i.e. the final
        epsilon reported is valid.

    Returns:
      For each l in privacy_losses, the smallest multiple of precision, x, such
      that the privacy loss at x is at most l.

    Raises:
      ValueError: If monotonicity requirements for inversion of the privacy loss
        via binary search are violated.
    """
    self._verify_monotonicity()

    if not (np.diff(privacy_losses) >= 0).all():
      raise ValueError(
          f'Expected non-decreasing privacy_losses, got: {privacy_losses}.'
      )
    if len(privacy_losses) == 0:  # pylint: disable=g-explicit-length-test
      return np.ndarray([])

    # Some privacy losses might be close to the privacy loss at x=a or a=b,
    # where (a, b) is the interval on which the privacy loss is strictly
    # monotonic, in which case we report the corresponding a or infinity.
    max_pl = privacy_losses[-1]
    min_pl = privacy_losses[0]
    output = np.empty_like(privacy_losses, dtype=float)

    if max_pl > self._privacy_loss_at_boundaries[0]:
        raise ValueError(
            f'max of privacy_losses ({max_pl}) is larger than '
            f'{self._privacy_loss_at_boundaries[0]=}.'
        )

    left_boundary_mask = np.isclose(privacy_losses,
                                    self._privacy_loss_at_boundaries[0])
    output[left_boundary_mask] = self._strictly_decreasing_interval[0]

    if min_pl <= self._privacy_loss_at_boundaries[1]:
        raise ValueError(
            f'min of privacy_losses ({min_pl}) is smaller than '
            f'{self._privacy_loss_at_boundaries[1]=}'
        )

    right_boundary_mask = np.isclose(privacy_losses,
                                     self._privacy_loss_at_boundaries[1])
    # Privacy loss is constant for x >= b,
    # so we pessimistically assume that we attain this constant via x=infty,
    # which maximizes delta in get_delta_for_epsilon(...).
    output[right_boundary_mask] = np.inf

    within_interval_mask = np.logical_not(
                              np.logical_or(
                                left_boundary_mask, right_boundary_mask))
    max_pl = np.max(privacy_losses[within_interval_mask])
    min_pl = np.min(privacy_losses[within_interval_mask])

    search_bounds = self._binary_search_bounds(min_pl, max_pl, precision)
    output[within_interval_mask] = self._inverse_privacy_losses_with_range(
        privacy_losses[within_interval_mask], search_bounds, precision
    )
    return output

  def _binary_search_bounds(
      self, min_pl: float, max_pl: float,
      precision: float = 1e-6,) -> Tuple[float, float]:
    """Determines interval s.t. privacy loss contains max_pl and min_pl.

    Since have no additional assumptions about mu and sensitivities,
    we choose pessimistic bounds and then refine them by searching
    for min_pl and max_pl
    (which is preferable to performing binary search with pessimistic bounds
    for many privacy losses in inverse_privacy_losses).

    Args:
        min_pl: Smallest privacy loss that must be attained within
          the search bounds.
        max_pl: Largest privacy loss that must be attained within
          the search bounds.
        precision: Precision of the output. In particular, for each entry l in
          privacy_losses, we output the smallest multiple of precision, x, such
          that the privacy loss at x is at most l. This ensures (i) given a
          monotonic privacy_losses, we return a monotonic list of xs, and (ii) the
          approximation results in an overestimate of epsilon, i.e. the final
          epsilon reported is valid.

    Returns:
        Tuple[float, float]: _description_
    """
    left_bound = max(-1, self._strictly_decreasing_interval[0])
    while (left_bound < 0) and (self.privacy_loss(left_bound) < max_pl):
      left_bound *= 2
    if left_bound < self._strictly_decreasing_interval[0]:
      left_bound = self._strictly_decreasing_interval[0]

    right_bound = min(1, self._strictly_decreasing_interval[1])
    while (right_bound > 0) and (self.privacy_loss(right_bound) > min_pl):
      right_bound *= 2
    if right_bound > self._strictly_decreasing_interval[1]:
      right_bound = self._strictly_decreasing_interval[1]

    refined_bounds = tuple(self._inverse_privacy_losses_with_range(
      np.array([min_pl, max_pl]),
      [left_bound, right_bound],
      precision*0.5,  # Higher precision to avoid off-by-one rounding errors.
    ))

    # Must revert order, because inverse of max_pl is smaller.
    return refined_bounds[1], refined_bounds[0]

  def _inverse_privacy_losses_with_range(
      self,
      privacy_losses: np.ndarray,
      bounds: tuple[float, float],
      precision: float = 1e-6,
  ) -> Iterable[float]:
    """Helper method for performing binary search in inverse_privacy_losses.

    Args:
      privacy_losses: the privacy losses we wish to invert.
      bounds: Range to search over, i.e. the inverses are in the range
        [bounds[0], bounds[1]].
      precision: Precision of the output; in particular, for each entry l in
        privacy_losses, we output the smallest multiple of precision, x, such
        that the privacy loss at x is at most l. This ensures (i) given a
        monotonic privacy_losses, we return a monotonic list of xs, and (ii) the
        approximation results in an overestimate of epsilon, i.e. the final
        epsilon we report is a valid epsilon.

    Returns:
      For each l in privacy_losses, the smallest multiple of precision, x, such
      that the privacy loss at x is at most l.
    """
    if len(privacy_losses) == 0:  # pylint: disable=g-explicit-length-test
      return []
    if bounds[1] - bounds[0] <= precision:
      return np.repeat(
          np.floor(bounds[1] / precision) * precision, len(privacy_losses)
      )

    mid = (bounds[0] + bounds[1]) / 2
    pl_split = self.privacy_loss(mid)
    lower_indices = privacy_losses < pl_split
    higher_indices = privacy_losses >= pl_split
    output = np.zeros_like(privacy_losses)
    output[lower_indices] = self._inverse_privacy_losses_with_range(
        privacy_losses[lower_indices], (mid, bounds[1]), precision
    )
    output[higher_indices] = self._inverse_privacy_losses_with_range(
        privacy_losses[higher_indices], (bounds[0], mid), precision
    )
    return output

  @abc.abstractmethod
  def noise_cdf(self, x: Union[float,
                               Iterable[float]]) -> Union[float, np.ndarray]:
    """Computes the cumulative density function of base distribution mu.

    Args:
     x: the point or points at which the cumulative density function is to be
       calculated.

    Returns:
      The cumulative density function of the base noise at x, i.e., the
      probability that the base noise is less than or equal to x.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def noise_log_pdf(
    self, x: Union[float, Iterable[float]]
  ) -> Union[float, np.ndarray]:
    """Computes the probability desnsity function of base distribution mu.

    Args:
     x: the point or points at which the cumulative density function is to be
       calculated.

    Returns:
      The cumulative density function of the base noise at x, i.e., the
      probability that the base noise is less than or equal to x.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def noise_ppf(self, p: Union[float,
                               Iterable[float]]) -> Union[float, np.ndarray]:
    """Computes the probability point function of base distribution mu.

    Args:
     x: the point or points at which the probability point function, i.e.,
      the inverse cumulative density function is to be evaluated.

    Returns:
      The probability point function of the base noise at p, i.e., an x
      such that the interval (-infty, x] has probability p.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def noise_log_cdf(
      self, x: Union[float, Iterable[float]]) -> Union[float, np.ndarray]:
    """Computes log of cumulative density function of the base distribution mu.

    Args:
      x: the point or points at which the log cumulative density function is to
        be calculated.

    Returns:
      The log cumulative density function of the base noise at x, i.e., the
      log of the probability that the base noise is less than or equal to x.
    """
    raise NotImplementedError

class MixtureGaussianPrivacyLoss(MonotonePrivacyLoss):
  """Privacy loss of the Mixture of Gaussians mechanism.

  This class gives the privacy loss for a scalar Gaussian mechanism where the
  sensitivity is a random variable equal to sensitivities[i] with probability
  sampling_probs[i].

  The privacy loss distribution of the Mixture of Gaussians mechanism is
  equivalent to the privacy loss distribution between the Gaussian distribution
  and the same distribution convolved with the discrete distribution specified
  by sensitivities and sampling_probs. That is, let mu be the Gaussian noise PDF
  with sigma = standard_deviation. The privacy loss distribution is generated as
  follows:
  For ADD adjacency type:
  - Let mu_lower(x) := sum over i of sampling_probs[i] *
    mu(x - sensitivities[i])
  - Sample x ~ mu_upper = mu and let the privacy loss be
    ln(mu_upper(x) / mu_lower(x)).
  For REMOVE adjacency type:
  - Let mu_upper(x) := sum over i of sampling_probs[i] *
    mu(x + sensitivities[i])
  - Sample x ~ mu_lower = mu and let the privacy loss be
    ln(mu_upper(x) / mu_lower(x)).

  For example:
    sensitivities = [1.0], sampling_probs = [1.0] captures the sensitivity-1
      Gaussian mechanism.
    sensitivities = [0.0, 1.0], sampling_probs = [0.9, 0.1] captures the
      sensitivity-1 subsampled Gaussian mechanism with sampling probability
      0.1.
    sensitivities = [0.0, 1.0, 2.0], sampling_probs = [0.81, 0.18, 0.01]
      captures a generalization of the subsampled Gaussian mechanism where
      we can potentially sample the sensitive example twice, each time
      with sampling probability 0.1.

  This class can be used for vector-valued mechanisms via Corollary 4.7 from the
  following paper:
    Title: Privacy Amplification for Matrix Mechanisms
    Authors: C. A. Choquette-Choo, A. Ganesh, T. Steinke, A. Thakurta
    Link: https://arxiv.org/abs/2310.15526
  In particular, this corollary shows the vector-valued mixture of Gaussians
  mechanism is dominated by the scalar-valued mechanism given by replacing each
  sensitivity vector with its norm.

  For almost all methods (the exception is inverse_privacy_losses, for reasons
  discussed in the docstring for that method), we reimplement the corresponding
  method in GaussianPrivacyLoss, but with some minor changes to account for the
  fact that this class is more general.

  Attributes:
    adjacency_type: type of adjacency relation to used for defining the privacy
        loss distribution.
    sensitivities: the support of the sensitivity distribution.
    sampling_probs: the probabilities associated with the sensitivities.
  """

  def __init__(
      self,
      standard_deviation: float,
      sensitivities: Sequence[float],
      sampling_probs: Sequence[float],
      pessimistic_estimate: bool = True,
      log_mass_truncation_bound: float = -50,
      adjacency_type: AdjacencyType = AdjacencyType.REMOVE,
  ) -> None:
    """Initializes the privacy loss of the MoG mechanism.

    Args:
      standard_deviation: The standard_deviation of the Gaussian distribution.
      sensitivities: The support of the sensitivity distribution. Must be the
        same length as sampling_probs, and both should be 1D.
      sampling_probs: The probabilities associated with the sensitivities.
      pessimistic_estimate: A value indicating whether the rounding is done in
        such a way that the resulting epsilon-hockey stick divergence
        computation gives an upper estimate to the real value.
      log_mass_truncation_bound: The ln of the probability mass that might be
        discarded from the noise distribution. The larger this number, the more
        error it may introduce in divergence calculations.
      adjacency_type: Type of adjacency relation to be used for defining the
        privacy loss distribution.

    Raises:
      ValueError: If args are invalid, e.g. standard_deviation is negative or
      sensitivities and sampling_probs are different lengths.
    """
    if standard_deviation <= 0:
      raise ValueError(
          'Standard deviation is not a positive real number: '
          f'{standard_deviation}'
      )
    if log_mass_truncation_bound > 0:
      raise ValueError(
          'Log mass truncation bound is not a non-positive real '
          f'number: {log_mass_truncation_bound}'
      )
    if len(sampling_probs) != len(sensitivities):
      raise ValueError(
          'sensitivities and sampling_probs must have the same '
          f'length. Got {sampling_probs=} of length {len(sampling_probs)}, '
          f'{sensitivities=} of length {len(sensitivities)}.'
      )

    non_zero_indices = np.asarray(sampling_probs) != 0.0
    sensitivities = np.asarray(sensitivities)[non_zero_indices]
    sampling_probs = np.asarray(sampling_probs)[non_zero_indices]
    if np.any(sensitivities < 0):
      raise ValueError(
          f'Sensitivities contains a negative number: {sensitivities}.'
      )
    if sensitivities.max() == 0.0:
      raise ValueError('Must have at least one positive sensitivity.')
    if not math.isclose(sum(sampling_probs), 1):
      raise ValueError(
          f'Probabilities do not add up to 1: {sum(sampling_probs)}'
      )
    for sampling_prob in sampling_probs:
      if sampling_prob <= 0 or sampling_prob > 1:
        raise ValueError(
            f'Sampling probability is not in (0,1] : {sampling_prob}'
        )
    super().__init__(is_discrete=False)
    self.adjacency_type = adjacency_type
    self.sampling_probs = sampling_probs
    self.sensitivities = sensitivities
    self._standard_deviation = standard_deviation
    self._variance = standard_deviation**2
    self._pessimistic_estimate = pessimistic_estimate
    self._log_mass_truncation_bound = log_mass_truncation_bound

    # Constant properties.
    self._log_sampling_probs = np.log(self.sampling_probs)
    self._pos_sampling_probs = self.sampling_probs[self.sensitivities > 0.0]
    self._sampling_prob = np.clip(self._pos_sampling_probs.sum(), 0, 1)
    nonzero_sens = self.sensitivities[self.sensitivities > 0.0]
    self._min_sens = np.min(nonzero_sens)
    self._max_sens = np.max(nonzero_sens)
    self._gaussian_random_variable = stats.norm(scale=standard_deviation)

  def mu_upper_cdf(
      self, x: Union[float, Iterable[float]]
  ) -> Union[float, np.ndarray]:
    """Computes the cumulative density function of the mu_upper distribution.

    For ADD adjacency type:
      mu_upper(x) := mu
    For REMOVE adjacency type:
      mu_upper(x) := sum of sampling_probs[i] * mu(x + sensitivities[i])

    Args:
      x: the point or points at which the cumulative density function is to be
        calculated.

    Returns:
      The cumulative density function of the mu_upper distribution at x, i.e.,
      the probability that mu_upper is less than or equal to x.
    """
    if self.adjacency_type == AdjacencyType.ADD:
      return self.noise_cdf(x)
    else:  # Case: self.adjacency_type == AdjacencyType.REMOVE
      points_per_sens = np.add.outer(np.atleast_1d(x), self.sensitivities)
      output = (self.noise_cdf(points_per_sens) * self.sampling_probs).sum(
          axis=1
      )
      if isinstance(x, numbers.Number):
        return output[0]
      else:
        return output

  def mu_lower_log_cdf(
      self, x: Union[float, Iterable[float]]
  ) -> Union[float, np.ndarray]:
    """Computes log cumulative density function of the mu_lower distribution.

    For ADD adjacency type:
      mu_lower(x) := sum of sampling_probs[i] * mu(x - sensitivities[i])
    For REMOVE adjacency type:
      mu_lower(x) := mu

    Args:
      x: the point or points at which the log of the cumulative density function
        is to be calculated.

    Returns:
      The log of the cumulative density function of the mu_lower distribution at
      x, i.e., the log of the probability that mu_lower is less than or equal to
      x.
    """
    if self.adjacency_type == AdjacencyType.ADD:
      points_per_sens = np.add.outer(np.atleast_1d(x), -self.sensitivities)
      logcdf_per_sens = self.noise_log_cdf(points_per_sens)
      output = scipy.special.logsumexp(
          logcdf_per_sens, axis=1, b=self.sampling_probs
      )
      if isinstance(x, numbers.Number):
        return output[0]
      else:
        return output
    else:  # Case: self.adjacency_type == AdjacencyType.REMOVE
      return self.noise_log_cdf(x)

  def get_delta_for_epsilon(
      self, epsilon: Union[float, Sequence[float]]
  ) -> Union[float, list[float]]:
    """Computes the epsilon-hockey stick divergence of the mechanism.

    Args:
      epsilon: the epsilon, or list-like object of epsilon values, in
        epsilon-hockey stick divergence. Should be non-decreasing if list-like.

    Returns:
      A non-negative real number which is the epsilon-hockey stick divergence of
      the mechanism, or a numpy array if epsilon is list-like.
    """
    epsilons = np.atleast_1d(epsilon)
    if not np.all(epsilons[1:] >= epsilons[:-1]):
      raise ValueError(f'Epsilon values must be non-decreasing: {epsilons}')
    deltas = np.zeros_like(epsilons, dtype=float)
    if self._sampling_prob == 1.0:
      inverse_indices = np.full_like(epsilons, True, dtype=bool)
    elif self.adjacency_type == AdjacencyType.ADD:
      inverse_indices = epsilons < -np.log1p(-self._sampling_prob)
    else:  # Case: self.adjacency_type == AdjacencyType.REMOVE
      inverse_indices = epsilons > np.log1p(-self._sampling_prob)
      other_indices = np.logical_not(inverse_indices)
      deltas[other_indices] = -np.expm1(epsilons[other_indices])
    x_cutoffs = self.inverse_privacy_losses(epsilons[inverse_indices])
    deltas[inverse_indices] = self.mu_upper_cdf(x_cutoffs) - np.exp(
        epsilons[inverse_indices] + self.mu_lower_log_cdf(x_cutoffs)
    )
    # Clip delta values to lie in [0,1] (to avoid numerical errors)
    deltas = np.clip(deltas, 0, 1)
    if isinstance(epsilon, numbers.Number):
      return float(deltas)
    else:
      # For numerical stability reasons, deltas may not be non-increasing. This
      # is fixed post-hoc at small cost in accuracy.
      for i in reversed(range(deltas.shape[0] - 1)):
        deltas[i] = max(deltas[i], deltas[i + 1])
      return deltas

  def privacy_loss_tail(
      self, precision: float = 1e-4
  ) -> TailPrivacyLossDistribution:
    """Computes the privacy loss at the tail of the random-sensitivity Gaussian.

    For ADD adjacency type: The upper distribution is a single Gaussian and we
      can exactly compute the tails easily.

    For REMOVE adjacency type:  We set upper_x_truncation such that
      CDF(upper_x_truncation) = 1 - 0.5 * exp(log_mass_truncation_bound). It is
      worthwhile to spend some up-front computation getting a more precise value
      for lower_x_truncation to save computation later on. So we binary search
      over the interval [-upper_x_truncation - max(sensitivities),
      -upper_x_truncation] for the point where the cdf of mu_upper is
      0.5 * exp(log_mass_truncation_bound). Since we're binary searching over a
      continuous domain, we proceed until the width of the binary search
      interval is at most some small precision, and then set lower_x_truncation
      to be the left endpoint of this interval.

    Args:
      precision: The additive error we will compute the truncation values
        within. That is, when we binary search for log_mass_truncation_bound in
        the REMOVE case, we terminate the binary search when the interval has
        length at most precision, and then use the more conservative endpoint
        of the interval as our truncation value.

    Returns:
      A TailPrivacyLossDistribution instance representing the tail of the
      privacy loss distribution.
    """
    tail_mass = 0.5 * np.exp(self._log_mass_truncation_bound)
    z_value = self._gaussian_random_variable.ppf(tail_mass)
    upper_x_truncation = -z_value
    if self.adjacency_type == AdjacencyType.ADD:
      lower_x_truncation = z_value
    else:  # Case: self.adjacency_type == AdjacencyType.REMOVE
      lower_x_truncation = common.inverse_monotone_function(
          self.mu_upper_cdf,
          tail_mass,
          common.BinarySearchParameters(
              z_value - self._max_sens,
              z_value,
              tolerance=precision
          ),
          increasing=True,
      )
    if self._pessimistic_estimate:
      tail_probability_mass_function = {
          math.inf: self.mu_upper_cdf(lower_x_truncation),
          self.privacy_loss(upper_x_truncation): 1 - self.mu_upper_cdf(
              upper_x_truncation
          ),
      }
    else:
      tail_probability_mass_function = {
          self.privacy_loss(lower_x_truncation): self.mu_upper_cdf(
              lower_x_truncation
          ),
      }
    return TailPrivacyLossDistribution(
        lower_x_truncation, upper_x_truncation, tail_probability_mass_function
    )

  def connect_dots_bounds(self) -> ConnectDotsBounds:
    """Computes the bounds on epsilon values to use in connect-the-dots algorithm.

    Returns:
      A ConnectDotsBounds instance containing upper and lower values of
      epsilon to use in connect-the-dots algorithm.
    """
    tail_pld = self.privacy_loss_tail()

    return ConnectDotsBounds(
        epsilon_upper=self.privacy_loss(tail_pld.lower_x_truncation),
        epsilon_lower=self.privacy_loss(tail_pld.upper_x_truncation),
    )

  @functools.cached_property
  def _precompute_privacy_loss_constants(self) -> np.ndarray:
    """(Pre-)computes the constants in the expression for the privacy loss."""
    sens_loss = self.privacy_loss_for_single_gaussian(
        np.repeat(0, len(self.sensitivities)), self.sensitivities
    )
    if self.adjacency_type == AdjacencyType.ADD:
      return self._log_sampling_probs - sens_loss
    else:  # Case: self.adjacency_type == AdjacencyType.REMOVE
      return self._log_sampling_probs + sens_loss

  def privacy_loss(self, x: float) -> float:
    """Computes the privacy loss at a given point `x`."""
    # Because this method is called many times, we opt to precompute as much
    # as possible. It can be see that the computation below is equivalent to:
    #
    # x_rep = np.repeat(x, len(self.sensitivities))
    # privacy_losses_without_subsampling = (
    #     self.privacy_loss_for_single_gaussian(x_rep, self.sensitivities)
    # )
    # if self.adjacency_type == AdjacencyType.ADD:
    #   summands = self._log_sampling_probs - privacy_losses_without_subsampling
    #   return -np.logaddexp.reduce(summands)
    #
    # and similarly for .REMOVE.
    x_loss = self.sensitivities * x / (self._variance)
    if self.adjacency_type == AdjacencyType.ADD:
      summands = self._precompute_privacy_loss_constants + x_loss
      return -np.logaddexp.reduce(summands)
    else:  # Case: self.adjacency_type == AdjacencyType.REMOVE
      summands = self._precompute_privacy_loss_constants - x_loss
      return np.logaddexp.reduce(summands)

  def privacy_loss_for_single_gaussian(
      self,
      x: Union[float, np.ndarray],
      sensitivity: Union[float, np.ndarray] = 1.0
  ) -> np.ndarray:
    """Computes the privacy loss of the Gaussian mechanism.

    Args:
      x: the point(s) at which the privacy loss is computed.
      sensitivity: The sensitivity/sensitivities of interest.

    Returns:
      The privacy loss of the Gaussian mechanism without sub-sampling at
      point(s) x with the given sensitivity, which is given as
      For ADD adjacency type:
        sensitivity * (0.5 * sensitivity - x) / standard_deviation^2.
      For REMOVE adjacency type:
        sensitivity * (- 0.5 * sensitivity - x) / standard_deviation^2.
    """
    x = np.atleast_1d(x)
    if self.adjacency_type == AdjacencyType.ADD:
      scale = 0.5
    else:  # Case: self.adjacency_type == AdjacencyType.REMOVE
      scale = -0.5
    return sensitivity * (scale * sensitivity - x) / (self._variance)

  def inverse_privacy_loss_for_single_gaussian(
      self,
      privacy_loss: Union[float, np.ndarray],
      sensitivity: Union[float, np.ndarray] = 1.0,
  ) -> Union[float, np.ndarray]:
    """Computes the inverse privacy loss for the Gaussian mechanism.

    Args:
      privacy_loss: the privacy loss value(s).
      sensitivity: sensitivity/sensitivies of the Gaussian.

    Returns:
      The largest float(s) x such that the privacy loss at x is at least
      privacy_loss. REMOVE is the same as ADD, minus sensitivity.
    """
    add_inverse_loss = (
        0.5 * sensitivity - privacy_loss * (self._variance) / sensitivity
    )
    if self.adjacency_type == AdjacencyType.ADD:
      return add_inverse_loss
    else:  # Case: self.adjacency_type == AdjacencyType.REMOVE
      return add_inverse_loss - sensitivity

  def inverse_privacy_loss(
      self, privacy_loss: float, precision: float = 1e-6
  ) -> float:
    """(Approximately) computes the inverse of a given privacy loss.

    Technically, this method can be sped up by rewriting the logic in
    inverse_privacy_losses to take advantage of the fact that we have a
    single privacy loss rather than a list. However, this method is only written
    to complete the abstract class, and the process of generating a PLD from
    this class won't ever call this method. So, we have chosen the simple but
    inefficient implementation of calling inverse_privacy_losses.

    Args:
      privacy_loss: the privacy loss value.
      precision: Precision of the output.

    Returns:
      The largest float x such that the privacy loss at x is at least
      privacy_loss, rounded down to the nearest multiple of precision if
      we are using pessimistic estimates, and otherwise rounded up.
    """
    return float(
        self.inverse_privacy_losses(np.atleast_1d(privacy_loss), precision)[0]
    )

  def inverse_privacy_losses(
      self,
      privacy_losses: np.ndarray,
      precision: float = 1e-6,
  ) -> np.ndarray:
    """(Approximately) computes the inverse of a list of privacy losses.

    Unlike subsampled Gaussians, for mixture Gaussians the privacy loss does
    not have a closed-form inverse, to the best of our knowledge. So, we use
    binary search. This is the main bottleneck in this library, so we optimize
    it by doing one binary search for all values in privacy losses rather than a
    separate binary search for each. This way, we avoid recomputing the privacy
    loss at the same point across different binary searches.

    Args:
      privacy_losses: the privacy losses we wish to invert, in increasing order.
      precision: Precision of the output. In particular, for each entry l in
        privacy_losses, we output the smallest multiple of precision, x, such
        that the privacy loss at x is at most l. This ensures (i) given a
        monotonic privacy_losses, we return a monotonic list of xs, and (ii) the
        approximation results in an overestimate of epsilon, i.e. the final
        epsilon reported is valid.

    Returns:
      For each l in privacy_losses, the smallest multiple of precision, x, such
      that the privacy loss at x is at most l.
    """
    if not (np.diff(privacy_losses) >= 0).all():
      raise ValueError(
          f'Expected non-decreasing privacy_losses, got: {privacy_losses}.'
      )
    if len(privacy_losses) == 0:  # pylint: disable=g-explicit-length-test
      return np.ndarray([])

    # If we have a non-zero probability of choosing sensitivity = 0, then the
    # privacy loss does not take on all values in [-inf, inf], and so we need to
    # make sure all values in privacy_losses are in the proper range for the
    # given adjacency type.

    # Some privacy losses might be close to the privacy loss at x = +/-inf, in
    # which case we report the corresponding infinity for them.
    min_pl = privacy_losses[0]
    max_pl = privacy_losses[-1]
    log_1m_prob = (
        math.log1p(-self._sampling_prob) if self._sampling_prob < 1 else -np.inf
    )
    if self.adjacency_type == AdjacencyType.ADD:
      if max_pl > -log_1m_prob:
        raise ValueError(
            f'max of privacy_losses ({max_pl}) is larger than '
            f'-log(1 - sampling_prob)={-log_1m_prob}.'
        )
      finite_indices = np.logical_not(np.isclose(privacy_losses, -log_1m_prob))
      max_pl = np.max(privacy_losses[finite_indices])
    else:  # Case: self.adjacency_type == AdjacencyType.REMOVE
      if min_pl <= log_1m_prob:
        raise ValueError(
            f'min of privacy_losses ({min_pl}) is smaller than '
            f'log(1 - sampling_prob)={log_1m_prob}'
        )
      finite_indices = np.logical_not(np.isclose(privacy_losses, log_1m_prob))
      min_pl = np.min(privacy_losses[finite_indices])

    # Now, we need to determine a suitable range to binary search over. To do
    # this, we consider the subsampled Gaussian mechanisms given by moving
    # all the probability mass on non-zero sensitivities to either the
    # smallest or largest sensitivity. We can show the privacy loss of the
    # mixture is contained between these two mechanisms' privacy losses, and
    # the subsampled Gaussian privacy loss is easy to invert. Then, we can
    # compute the inverse privacy loss at min_pl and max_pl to get four
    # candidate bounds, and take the min/max of these bounds.
    loss_bounds = np.array([min_pl, min_pl, max_pl, max_pl])
    sens_bounds = np.array([
        self._min_sens,
        self._max_sens,
        self._min_sens,
        self._max_sens,
    ])
    if self.adjacency_type == AdjacencyType.ADD:
      pl_without_sampling = _pl_without_sampling_add
    else:  # Case: self.adjacency_type == AdjacencyType.REMOVE
      pl_without_sampling = _pl_without_sampling_remove

    possible_bounds = self.inverse_privacy_loss_for_single_gaussian(
        pl_without_sampling(loss_bounds, self._sampling_prob), sens_bounds
    )
    bounds = (
        np.floor(np.min(possible_bounds) / precision) * precision,
        np.ceil(np.max(possible_bounds) / precision) * precision,
    )
    if self.adjacency_type == AdjacencyType.ADD:
      output = self.privacy_loss_for_single_gaussian(
          np.full_like(privacy_losses, np.inf)
      )
    else:
      output = self.privacy_loss_for_single_gaussian(
          np.full_like(privacy_losses, -np.inf)
      )
    output[finite_indices] = self._inverse_privacy_losses_with_range(
        privacy_losses[finite_indices], bounds, precision
    )
    return output

  def _inverse_privacy_losses_with_range(
      self,
      privacy_losses: np.ndarray,
      bounds: tuple[float, float],
      precision: float = 1e-6,
  ) -> Iterable[float]:
    """Helper method for performing binary search in inverse_privacy_losses.

    Args:
      privacy_losses: the privacy losses we wish to invert.
      bounds: Range to search over, i.e. the inverses are in the range
        [bounds[0], bounds[1]].
      precision: Precision of the output; in particular, for each entry l in
        privacy_losses, we output the smallest multiple of precision, x, such
        that the privacy loss at x is at most l. This ensures (i) given a
        monotonic privacy_losses, we return a monotonic list of xs, and (ii) the
        approximation results in an overestimate of epsilon, i.e. the final
        epsilon we report is a valid epsilon.

    Returns:
      For each l in privacy_losses, the smallest multiple of precision, x, such
      that the privacy loss at x is at most l.
    """
    if len(privacy_losses) == 0:  # pylint: disable=g-explicit-length-test
      return []
    if bounds[1] - bounds[0] <= precision:
      return np.repeat(
          np.floor(bounds[1] / precision) * precision, len(privacy_losses)
      )

    mid = (bounds[0] + bounds[1]) / 2
    pl_split = self.privacy_loss(mid)
    lower_indices = privacy_losses < pl_split
    higher_indices = privacy_losses >= pl_split
    output = np.zeros_like(privacy_losses)
    output[lower_indices] = self._inverse_privacy_losses_with_range(
        privacy_losses[lower_indices], (mid, bounds[1]), precision
    )
    output[higher_indices] = self._inverse_privacy_losses_with_range(
        privacy_losses[higher_indices], (bounds[0], mid), precision
    )
    return output

  def noise_cdf(
      self, x: Union[float, Iterable[float]]
  ) -> Union[float, np.ndarray]:
    """Computes the cumulative density function of the Gaussian distribution.

    Args:
     x: the point or points at which the cumulative density function is to be
       calculated.

    Returns:
      The cumulative density function of the Gaussian noise at x, i.e., the
      probability that the Gaussian noise is less than or equal to x.
    """
    return self._gaussian_random_variable.cdf(x)

  def noise_log_cdf(
      self, x: Union[float, Iterable[float]]
  ) -> Union[float, np.ndarray]:
    """Computes log of cumulative density function of the Gaussian distribution.

    Args:
      x: the point or points at which the log cumulative density function is to
        be calculated.

    Returns:
      The log cumulative density function of the Gaussian noise at x, i.e., the
      log of the probability that the Gaussian noise is less than or equal to x.
    """
    return self._gaussian_random_variable.logcdf(x)

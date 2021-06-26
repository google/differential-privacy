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
import math
import typing
from typing import Iterable, Optional, Union
import dataclasses
import numpy as np
from scipy import stats

from dp_accounting import common


@dataclasses.dataclass
class TailPrivacyLossDistribution(object):
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
  tail_probability_mass_function: typing.Mapping[float, float]


class AdditiveNoisePrivacyLoss(metaclass=abc.ABCMeta):
  """Superclass for privacy loss of additive noise mechanisms.

  An additive noise mechanism for computing a scalar-valued function f is a
  mechanism that outputs the sum of the true value of the function and a noise
  drawn from a certain distribution mu. This class allows one to compute several
  quantities related to the privacy loss of additive noise mechanisms.

  We assume that the noise mu is such that the algorithm is more private as the
  sensitivity of f decreases. (Recall that the sensitivity of f is the maximum
  absolute change in f when an input to a single user changes.) Under this
  assumption, the privacy loss distribution of the mechanism is exactly
  generated as follows: pick x from mu and let the privacy loss be
  ln(P(x) / P(x - sensitivity)). Note that when mu is discrete, P(x) and
  P(x - sensitivity) are the probability masses of mu at x and x - sensitivity
  respectively. When mu is continuous, P(x) and P(x - sensitivity) are the
  probability densities of mu at x and x - sensitivity respectively.

  This class also assumes the privacy loss is non-increasing as x increases.

  Attributes:
    sensitivity: the sensitivity of function f. (i.e. the maximum absolute
      change in f when an input to a single user changes.)
    discrete_noise: a value indicating whether the noise is discrete. If this
        is True, then it is assumed that the noise can only take integer values.
        If False, then it is assumed that the noise is continuous, i.e., the
        probability mass at any given point is zero.
  """

  def __init__(self, sensitivity, discrete_noise):
    if sensitivity <= 0:
      raise ValueError(
          f'Sensitivity is not a positive real number: {sensitivity}')
    self.sensitivity = sensitivity
    self.discrete_noise = discrete_noise

  def get_delta_for_epsilon(self, epsilon):
    """Computes the epsilon-hockey stick divergence of the mechanism.

    The epsilon-hockey stick divergence of the mechanism is the value of delta
    for which the mechanism is (epsilon, delta)-differentially private. (See
    Observation 1 in the supplementary material.)

    This function assumes the privacy loss is non-increasing as x increases.
    Under this assumption, the hockey stick divergence is simply
    CDF(inverse_privacy_loss(epsilon)) - exp(epsilon) *
    CDF(inverse_privacy_loss(epsilon) - sensitivity), because the privacy loss
    at a point x is at least epsilon iff x <= inverse_privacy_loss(epsilon).

    Args:
      epsilon: the epsilon in epsilon-hockey stick divergence.

    Returns:
      A non-negative real number which is the epsilon-hockey stick divergence
      of the mechanism.
    """
    x_cutoff = self.inverse_privacy_loss(epsilon)
    return self.noise_cdf(x_cutoff) - math.exp(epsilon) * self.noise_cdf(
        x_cutoff - self.sensitivity)

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
  def privacy_loss(self, x: float) -> float:
    """Computes the privacy loss at a given point.

    Args:
      x: the point at which the privacy loss is computed.

    Returns:
      The privacy loss at point x, which is equal to
      ln(P(x) / P(x - sensitivity)).

    Raises:
      NotImplementedError: If not implemented by the subclass.
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

    Raises:
      NotImplementedError: If not implemented by the subclass.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def noise_cdf(self, x: Union[float,
                               Iterable[float]]) -> Union[float, np.ndarray]:
    """Computes the cumulative density function of the noise distribution.

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

  @classmethod
  @abc.abstractmethod
  def from_privacy_guarantee(
      cls,
      privacy_parameters: common.DifferentialPrivacyParameters,
      sensitivity: float = 1,
      pessimistic_estimate: bool = True
  ) -> 'AdditiveNoisePrivacyLoss':
    """Creates the privacy loss for the mechanism with a given privacy.

    Args:
      privacy_parameters: the desired privacy guarantee of the mechanism.
      sensitivity: the sensitivity of function f. (i.e. the maximum absolute
        change in f when an input to a single user changes.)
      pessimistic_estimate: a value indicating whether the rounding is done in
        such a way that the resulting epsilon-hockey stick divergence
        computation gives an upper estimate to the real value.

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
  loss distribution of the Laplace mechanism is generated as follows: first pick
  x according to the Laplace noise. Then, let the privacy loss be
  ln(PDF(x) / PDF(x - sensitivity)) which is equal to
  (|x - sensitivity| - |x|) / parameter.
  """

  def __init__(self,
               parameter: float,
               sensitivity: float = 1) -> None:
    """Initializes the privacy loss of the Laplace mechanism.

    Args:
      parameter: the parameter of the Laplace distribution.
      sensitivity: the sensitivity of function f. (i.e. the maximum absolute
        change in f when an input to a single user changes.)
    """
    if parameter <= 0:
      raise ValueError(f'Parameter is not a positive real number: {parameter}')

    self._parameter = parameter
    self._laplace_random_variable = stats.laplace(scale=parameter)
    super(LaplacePrivacyLoss, self).__init__(sensitivity, False)

  def privacy_loss_tail(self) -> TailPrivacyLossDistribution:
    """Computes the privacy loss at the tail of the Laplace distribution.

    When x <= 0, the privacy loss is simply sensitivity / parameter; this
    happens with probability 0.5. When x >= sensitivity, the privacy loss is
    simply - sensitivity / parameter; this happens with probability
    1 - CDF(sensitivity) = CDF(-sensitivity).

    Returns:
      A TailPrivacyLossDistribution instance representing the tail of the
      privacy loss distribution.
    """
    return TailPrivacyLossDistribution(
        0.0, self.sensitivity, {
            self.sensitivity / self._parameter:
                0.5,
            -self.sensitivity / self._parameter:
                self._laplace_random_variable.cdf(-self.sensitivity)
        })

  def privacy_loss(self, x: float) -> float:
    """Computes the privacy loss of the Laplace mechanism at a given point.

    Args:
      x: the point at which the privacy loss is computed.

    Returns:
      The privacy loss of the Laplace mechanism at point x, which is equal to
      (|x - sensitivity| - |x|) / parameter.
    """
    return (abs(x - self.sensitivity) - abs(x)) / self._parameter

  def inverse_privacy_loss(self, privacy_loss: float) -> float:
    """Computes the inverse of a given privacy loss for the Laplace mechanism.

    Args:
      privacy_loss: the privacy loss value.

    Returns:
      The largest float x such that the privacy loss at x is at least
      privacy_loss. When privacy_loss is at most - sensitivity / parameter, x is
      equal to infinity. When - sensitivity / parameter < privacy_loss <=
      sensitivity / parameter, x is equal to
      0.5 * (sensitivity - privacy_loss * parameter). When privacy_loss >
      sensitivity / parameter, no such x exists and the function returns
      -infinity.
    """
    if privacy_loss > self.sensitivity / self._parameter:
      return -math.inf
    if privacy_loss <= -self.sensitivity / self._parameter:
      return math.inf
    return 0.5 * (self.sensitivity - privacy_loss * self._parameter)

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

  @classmethod
  def from_privacy_guarantee(
      cls,
      privacy_parameters: common.DifferentialPrivacyParameters,
      sensitivity: float = 1,
      pessimistic_estimate: bool = True
  ) -> 'LaplacePrivacyLoss':
    """Creates the privacy loss for Laplace mechanism with given privacy.

    The parameter of the Laplace mechanism is simply sensitivity / epsilon.

    Args:
      privacy_parameters: the desired privacy guarantee of the mechanism.
      sensitivity: the sensitivity of function f. (i.e. the maximum absolute
        change in f when an input to a single user changes.)
      pessimistic_estimate: a value indicating whether the rounding is done in
        such a way that the resulting epsilon-hockey stick divergence
        computation gives an upper estimate to the real value.

    Returns:
      The privacy loss of the Laplace mechanism with the given privacy
        guarantee.
    """
    parameter = sensitivity / privacy_parameters.epsilon
    return LaplacePrivacyLoss(parameter, sensitivity=sensitivity)

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
  loss distribution of the Gaussian mechanism is generated as follows: first
  pick x according to the Gaussian noise. Then, let the privacy loss be
  ln(PDF(x) / PDF(x - sensitivity)) which is equal to
  0.5 * sensitivity * (sensitivity - 2 * x) / sigma^2.
  """

  def __init__(self,
               standard_deviation: float,
               sensitivity: float = 1,
               pessimistic_estimate: bool = True,
               log_mass_truncation_bound: float = -50) -> None:
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
    super(GaussianPrivacyLoss, self).__init__(sensitivity, False)

  def privacy_loss_tail(self) -> TailPrivacyLossDistribution:
    """Computes the privacy loss at the tail of the Gaussian distribution.

    We set lower_x_truncation so that CDF(lower_x_truncation) =
    0.5 * exp(log_mass_truncation_bound), and then set upper_x_truncation to be
    -lower_x_truncation.

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
    if self._pessimistic_estimate:
      tail_probability_mass_function = {
          math.inf:
              0.5 * math.exp(self._log_mass_truncation_bound),
          self.privacy_loss(upper_x_truncation):
              0.5 * math.exp(self._log_mass_truncation_bound)
      }
    else:
      tail_probability_mass_function = {
          self.privacy_loss(lower_x_truncation):
              0.5 * math.exp(self._log_mass_truncation_bound),
      }
    return TailPrivacyLossDistribution(lower_x_truncation, upper_x_truncation,
                                       tail_probability_mass_function)

  def privacy_loss(self, x: float) -> float:
    """Computes the privacy loss of the Gaussian mechanism at a given point.

    Args:
      x: the point at which the privacy loss is computed.

    Returns:
      The privacy loss of the Gaussian mechanism at point x, which is equal to
      0.5 * sensitivity * (sensitivity - 2 * x) / standard_deviation^2.
    """
    return (0.5 * self.sensitivity * (self.sensitivity - 2 * x) /
            (self._standard_deviation**2))

  def inverse_privacy_loss(self, privacy_loss: float) -> float:
    """Computes the inverse of a given privacy loss for the Gaussian mechanism.

    Args:
      privacy_loss: the privacy loss value.

    Returns:
      The largest float x such that the privacy loss at x is at least
      privacy_loss. This is equal to
      0.5 * sensitivity - privacy_loss * standard_deviation^2 / sensitivity.
    """
    return (0.5 * self.sensitivity - privacy_loss *
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

  @classmethod
  def from_privacy_guarantee(
      cls,
      privacy_parameters: common.DifferentialPrivacyParameters,
      sensitivity: float = 1,
      pessimistic_estimate: bool = True,
  ) -> 'GaussianPrivacyLoss':
    """Creates the privacy loss for Gaussian mechanism with desired privacy.

    Uses binary search to find the smallest possible standard deviation of the
    Gaussian noise for which the protocol is (epsilon, delta)-differentially
    private.

    Args:
      privacy_parameters: the desired privacy guarantee of the mechanism.
      sensitivity: the sensitivity of function f. (i.e. the maximum absolute
        change in f when an input to a single user changes.)
      pessimistic_estimate: a value indicating whether the rounding is done in
        such a way that the resulting epsilon-hockey stick divergence
        computation gives an upper estimate to the real value.

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
          sensitivity=sensitivity).get_delta_for_epsilon(
              privacy_parameters.epsilon)

    standard_deviation = common.inverse_monotone_function(
        _get_delta_for_standard_deviation, privacy_parameters.delta,
        search_parameters)

    return GaussianPrivacyLoss(
        standard_deviation,
        sensitivity=sensitivity,
        pessimistic_estimate=pessimistic_estimate)

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
  but shifted by the sensitivity. More specifically, the privacy loss
  distribution of the discrete Laplace mechanism is generated as follows: first
  pick x according to the discrete Laplace noise. Then, let the privacy loss be
  ln(PMF(x) / PMF(x - sensitivity)) which is equal to
  parameter * (|x - sensitivity| - |x|).
  """

  def __init__(self,
               parameter: float,
               sensitivity: int = 1) -> None:
    """Initializes the privacy loss of the discrete Laplace mechanism.

    Args:
      parameter: the parameter of the discrete Laplace distribution.
      sensitivity: the sensitivity of function f. (i.e. the maximum absolute
        change in f when an input to a single user changes.)
    """
    if parameter <= 0:
      raise ValueError(f'Parameter is not a positive real number: {parameter}')

    if not isinstance(sensitivity, int):
      raise ValueError(f'Sensitivity is not an integer : {sensitivity}')

    self.sensitivity = sensitivity
    self._parameter = parameter
    self._discrete_laplace_random_variable = stats.dlaplace(parameter)
    super(DiscreteLaplacePrivacyLoss, self).__init__(sensitivity, True)

  def privacy_loss_tail(self) -> TailPrivacyLossDistribution:
    """Computes privacy loss at the tail of the discrete Laplace distribution.

    When x <= 0, the privacy loss is simply sensitivity * parameter; this
    happens with probability CDF(0). When x >= sensitivity, the privacy loss is
    simply - sensitivity * parameter; this happens with probability
    1 - CDF(sensitivity - 1) = CDF(-sensitivity).

    Returns:
      A TailPrivacyLossDistribution instance representing the tail of the
      privacy loss distribution.
    """
    return TailPrivacyLossDistribution(
        1, self.sensitivity - 1, {
            self.sensitivity * self._parameter:
                self._discrete_laplace_random_variable.cdf(0),
            -self.sensitivity * self._parameter:
                self._discrete_laplace_random_variable.cdf(-self.sensitivity)
        })

  def privacy_loss(self, x: float) -> float:
    """Computes privacy loss of the discrete Laplace mechanism at a given point.

    Args:
      x: the point at which the privacy loss is computed.

    Returns:
      The privacy loss of the discrete Laplace mechanism at point x, which is
      equal to (|x - sensitivity| - |x|) * parameter for any integer x.
    """
    if not isinstance(x, int):
      raise ValueError(f'Privacy loss at x is undefined for x = {x}')

    return (abs(x - self.sensitivity) - abs(x)) * self._parameter

  def inverse_privacy_loss(self, privacy_loss: float) -> float:
    """Computes the inverse of a given privacy loss for the discrete Laplace mechanism.

    Args:
      privacy_loss: the privacy loss value.

    Returns:
      The largest float x such that the privacy loss at x is at least
      privacy_loss. When privacy_loss is at most - sensitivity * parameter, x is
      equal to infinity. When - sensitivity * parameter < privacy_loss <=
      sensitivity * parameter, x is equal to
      floor(0.5 * (sensitivity - privacy_loss / parameter)). When privacy_loss >
      sensitivity * parameter, no such x exists and the function returns
      -infinity.
    """
    if privacy_loss > self.sensitivity * self._parameter:
      return -math.inf
    if privacy_loss <= -self.sensitivity * self._parameter:
      return math.inf
    return math.floor(0.5 *
                      (self.sensitivity - privacy_loss / self._parameter))

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

  @classmethod
  def from_privacy_guarantee(
      cls,
      privacy_parameters: common.DifferentialPrivacyParameters,
      sensitivity: int = 1
  ) -> 'DiscreteLaplacePrivacyLoss':
    """Creates privacy loss for discrete Laplace mechanism with desired privacy.

    The parameter of the discrete Laplace mechanism is simply
    epsilon / sensitivity.

    Args:
      privacy_parameters: the desired privacy guarantee of the mechanism.
      sensitivity: the sensitivity of function f. (i.e. the maximum absolute
        change in f when an input to a single user changes.)

    Returns:
      The privacy loss of the discrete Laplace mechanism with the given privacy
      guarantee.
    """
    if not isinstance(sensitivity, int):
      raise ValueError(f'Sensitivity is not an integer : {sensitivity}')
    if sensitivity <= 0:
      raise ValueError(
          f'Sensitivity is not a positive real number: {sensitivity}')

    return DiscreteLaplacePrivacyLoss(
        privacy_parameters.epsilon / sensitivity,
        sensitivity=sensitivity)

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
  as follows: first pick x according to the discrete Gaussian noise. Then, let
  the privacy loss be ln(PMF(x) / PMF(x - sensitivity)) which is equal to
  0.5 * sensitivity * (sensitivity - 2 * x) / sigma^2. Note that since we
  consider the truncated version of the noise, we set the privacy loss to
  infinity when x < -truncation_bound + sensitivity.

  Reference:
  Canonne, Kamath, Steinke. "The Discrete Gaussian for Differential Privacy".
  In NeurIPS 2020.
  """

  def __init__(self,
               sigma: float,
               sensitivity: int = 1,
               truncation_bound: Optional[int] = None) -> None:
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
    self._pmf_array = np.array(
        list(range(-1 * self._truncation_bound, self._truncation_bound + 1)))
    self._pmf_array = np.exp(-0.5 * (self._pmf_array)**2 / (sigma**2))
    self._pmf_array = np.insert(self._pmf_array, 0, 0)
    self._cdf_array = np.add.accumulate(self._pmf_array)
    self._pmf_array /= self._cdf_array[-1]
    self._cdf_array /= self._cdf_array[-1]

    super(DiscreteGaussianPrivacyLoss, self).__init__(sensitivity, True)

  def privacy_loss_tail(self) -> TailPrivacyLossDistribution:
    """Computes the privacy loss at the tail of the discrete Gaussian distribution.

    When x < -truncation_bound + sensitivity, the privacy loss is infinity.
    Due to truncation, x > truncation_bound never occurs.

    Returns:
      A TailPrivacyLossDistribution instance representing the tail of the
      privacy loss distribution.
    """
    return TailPrivacyLossDistribution(
        self.sensitivity - self._truncation_bound, self._truncation_bound, {
            math.inf:
                self.noise_cdf(self.sensitivity - self._truncation_bound - 1)
        })

  def privacy_loss(self, x: float) -> float:
    """Computes the privacy loss of the discrete Gaussian mechanism at a given point.

    Args:
      x: the point at which the privacy loss is computed.

    Returns:
      The privacy loss of the discrete Gaussian mechanism at point x. If x is an
      integer in the range [-truncation_bound + sensitivity, truncation_bound],
      it is equal to 0.5 * sensitivity * (sensitivity - 2 * x) / sigma^2. If x
      is an integer in the range [-truncation_bound,
      truncation_bound + sensitivity), then it is equal to infinity. Otherwise,
      the privacy loss is undefined
    """
    if (not isinstance(x, int)
        or x > self._truncation_bound or x < -1 * self._truncation_bound):
      raise ValueError(f'Privacy loss at x is undefined for x = {x}')

    if x >= self.sensitivity - self._truncation_bound:
      return (0.5 * self.sensitivity * (self.sensitivity - 2 * x) /
              (self._sigma**2))
    return math.inf

  def inverse_privacy_loss(self, privacy_loss: float) -> float:
    """Computes the inverse of a given privacy loss for the discrete Gaussian mechanism.

    Args:
      privacy_loss: the privacy loss value.

    Returns:
      The largest int x such that the privacy loss at x is at least
      privacy_loss. This is equal to
      floor(0.5 * sensitivity - privacy_loss * sigma^2 / sensitivity).
    """
    return math.floor(0.5 * self.sensitivity - privacy_loss *
                      (self._sigma**2) / self.sensitivity)

  def noise_cdf(self, x: Union[float,
                               Iterable[float]]) -> Union[float, np.ndarray]:
    """Computes the cumulative density function of the discrete Gaussian distribution.

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

  @classmethod
  def from_privacy_guarantee(
      cls,
      privacy_parameters: common.DifferentialPrivacyParameters,
      sensitivity: int = 1,
  ) -> 'DiscreteGaussianPrivacyLoss':
    """Creates the privacy loss for discrete Gaussian mechanism with desired privacy.

    Uses binary search to find the smallest possible standard deviation of the
    discrete Gaussian noise for which the protocol is (epsilon, delta)-DP.

    Args:
      privacy_parameters: the desired privacy guarantee of the mechanism.
      sensitivity: the sensitivity of function f. (i.e. the maximum absolute
        change in f when an input to a single user changes.)

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
          sensitivity=sensitivity).get_delta_for_epsilon(
              privacy_parameters.epsilon)

    sigma = common.inverse_monotone_function(
        _get_delta_for_sigma, privacy_parameters.delta, search_parameters)

    return DiscreteGaussianPrivacyLoss(sigma, sensitivity=sensitivity)

  def standard_deviation(self) -> float:
    """The standard deviation of the corresponding discrete Gaussian noise."""
    return math.sqrt(
        sum([
            ((i + self._offset)**2) * probability_mass
            for i, probability_mass in enumerate(self._pmf_array)
        ]))

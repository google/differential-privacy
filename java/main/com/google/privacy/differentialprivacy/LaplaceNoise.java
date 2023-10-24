//
// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

package com.google.privacy.differentialprivacy;

import static com.google.common.base.Preconditions.checkArgument;
import static java.lang.Math.exp;

import com.google.privacy.differentialprivacy.proto.SummaryOuterClass.MechanismType;
import java.security.SecureRandom;
import java.util.Random;
import javax.annotation.Nullable;

/**
 * Generates and adds Laplace noise to a raw piece of numerical data such that the result is
 * securely differentially private.
 *
 * <p>The Laplace noise is generated according to the geometric sampling mechanism described <a
 * href="https://github.com/google/differential-privacy/blob/main/common_docs/Secure_Noise_Generation.pdf">here</a>.
 * This approach is robust against unintentional privacy leaks due to artifacts of floating point
 * arithmetic.
 */
public class LaplaceNoise implements Noise {
  /**
   * This parameter determines the resolution of the numerical noise that is being generated
   * relative to the L_inf sensitivity and privacy parameter epsilon. More precisely, the
   * granularity parameter corresponds to the value 2^k as described <a
   * href="https://github.com/google/differential-privacy/blob/main/common_docs/Secure_Noise_Generation.pdf">here</a>.
   * Larger values result in more fine grained noise, but increase the chance of sampling
   * inaccuracies due to overflows. The probability of an overflow is less than 2^-1000, if the
   * granularity parameter is set to a value of 2^40 or less and the epsilon passed to addNoise is
   * at least 2^-50.
   *
   * <p>This parameter should be a power of 2.
   */
  private static final double GRANULARITY_PARAM = (double) (1L << 40);

  private final Random random;

  /** Returns a Noise instance initialized with a secure randomness source. */
  public LaplaceNoise() {
    this(new SecureRandom());
  }

  private LaplaceNoise(Random random) {
    this.random = random;
  }

  /**
   * Returns a Noise instance initialized with a specified randomness source. This should only be
   * used for testing and may only be called via the static methods in {@link TestNoiseFactory}.
   *
   * <p>This method is package-private for use by the factory.
   */
  static LaplaceNoise createForTesting(Random random) {
    return new LaplaceNoise(random);
  }

  /**
   * Adds Laplace noise to {@code x} such that the output is {@code epsilon}-differentially private,
   * with respect to the specified L_0 and L_inf sensitivities. Note that {@code delta} must be set
   * to {@code null} because it does not parameterize Laplace noise. Moreover, {@code epsilon} must
   * be at least 2^-50.
   */
  @Override
  public double addNoise(
      double x, int l0Sensitivity, double lInfSensitivity, double epsilon, double delta) {
    DpPreconditions.checkSensitivities(l0Sensitivity, lInfSensitivity);

    return addNoise(x, Noise.getL1Sensitivity(l0Sensitivity, lInfSensitivity), epsilon, delta);
  }

  /**
   * @deprecated Use {@link #addNoise(double, double, double, double)} instead. Set delta to 0.0.
   */
  @Deprecated
  public double addNoise(double x, double l1Sensitivity, double epsilon, @Nullable Double delta) {
    double primitiveDelta = delta == null ? 0.0 : delta;
    return addNoise(x, l1Sensitivity, epsilon, primitiveDelta);
  }

  /**
   * See {@link #addNoise(double, int, double, double, double)}.
   *
   * <p>As opposed to the latter method, this accepts the L_1 sensitivity of {@code x} directly
   * instead of the L_0 and L_Inf proxies. This should be used in settings where it is feasible or
   * more convenient to calculate the L_1 sensitivity directly.
   */
  public double addNoise(double x, double l1Sensitivity, double epsilon, double delta) {
    checkParameters(l1Sensitivity, epsilon, delta);

    double granularity = getGranularity(l1Sensitivity, epsilon);
    long twoSidedGeomericSample =
        SamplingUtil.sampleTwoSidedGeometric(
            random, granularity * epsilon / (l1Sensitivity + granularity));
    return SecureNoiseMath.roundToMultipleOfPowerOfTwo(x, granularity)
        + twoSidedGeomericSample * granularity;
  }

  /**
   * Adds Laplace noise to the integer {@code x} such that the output is {@code
   * epsilon}-differentially private, with respect to the specified L_0 and L_inf sensitivities.
   * Note that {@code delta} must be set to {@code null} because it does not parameterize Laplace
   * noise. Moreover, {@code epsilon} must be at least 2^-50.
   */
  @Override
  public long addNoise(
      long x, int l0Sensitivity, long lInfSensitivity, double epsilon, double delta) {
    DpPreconditions.checkSensitivities(l0Sensitivity, lInfSensitivity);

    return addNoise(
        x, (long) Noise.getL1Sensitivity(l0Sensitivity, lInfSensitivity), epsilon, delta);
  }

  /**
   * @deprecated Use {@link #addNoise(long, long, double, double)} instead. Set delta to 0.0.
   */
  @Deprecated
  public long addNoise(long x, long l1Sensitivity, double epsilon, @Nullable Double delta) {
    double primitiveDelta = delta == null ? 0.0 : delta;
    return addNoise(x, l1Sensitivity, epsilon, primitiveDelta);
  }

  /**
   * See {@link #addNoise(long, int, long, double, double)}.
   *
   * <p>As opposed to the latter method, this accepts the L_1 sensitivity of {@code x} directly
   * instead of the L_0 and L_Inf proxies. This should be used in settings where it is feasible or
   * more convenient to calculate the L_1 sensitivity directly.
   */
  public long addNoise(long x, long l1Sensitivity, double epsilon, double delta) {
    checkParameters(l1Sensitivity, epsilon, delta);

    double granularity = getGranularity(l1Sensitivity, epsilon);
    long twoSidedGeomericSample =
        SamplingUtil.sampleTwoSidedGeometric(
            random, granularity * epsilon / (l1Sensitivity + granularity));
    if (granularity <= 1.0) {
      return x + Math.round(twoSidedGeomericSample * granularity);
    } else {
      return SecureNoiseMath.roundToMultiple(x, (long) granularity)
          + twoSidedGeomericSample * (long) granularity;
    }
  }

  @Override
  public MechanismType getMechanismType() {
    return MechanismType.LAPLACE;
  }

  /**
   * Computes a confidence interval that contains the raw value {@code x} passed to {@link
   * #addNoise(double, int, double, double, double)} with a probability equal to {@code 1 - alpha}
   * based on the specified {@code noisedX} and noise parameters. Note that {@code delta} must be
   * set to {@code null} because it does not parameterize Laplace noise. Moreover, {@code epsilon}
   * must be at least 2^-50.
   *
   * <p>Refer to <a
   * href="https://github.com/google/differential-privacy/tree/main/common_docs/confidence_intervals.md">this</a> doc for
   * more information.
   */
  @Override
  public ConfidenceInterval computeConfidenceInterval(
      double noisedX,
      int l0Sensitivity,
      double lInfSensitivity,
      double epsilon,
      @Nullable Double delta,
      double alpha) {
    checkConfidenceIntervalParameters(l0Sensitivity, lInfSensitivity, epsilon, delta, alpha);
    double z =
        computeQuantile(alpha / 2.0, noisedX, l0Sensitivity, lInfSensitivity, epsilon, delta);
    // Because of the symmetry of the Laplace distribution, 2 * noisedX - z corresponds to the
    // (1 - alpha/2)-quantile of the distribution, meaning that the interval [z, 2 * noisedX - z]
    // contains 1-alpha of the probability mass. Deriving the (1 - alpha/2)-quantile from the
    // (alpha/2)-quantile and not vice versa is a deliberate choice. The reason is that alpha tends
    // to be very small. Consequently, alpha/2 is more accurately representable as a double than
    // 1 - alpha/2, facilitating numerical computations.
    return ConfidenceInterval.create(z, 2.0 * noisedX - z);
  }

  /**
   * Computes a confidence interval that contains the raw integer value {@code x} passed to {@link
   * #addNoise(long, int, long, double, double)} with a probability greater or equal to {@code 1 -
   * alpha} based on the specified {@code noisedX} and noise parameters. Note that {@code delta}
   * must be set to {@code null} because it does not parameterize Laplace noise. Moreover, {@code
   * epsilon} must be at least 2^-50.
   *
   * <p>Refer to <a
   * href="https://github.com/google/differential-privacy/tree/main/common_docs/confidence_intervals.md">this</a> doc for
   * more information.
   */
  @Override
  public ConfidenceInterval computeConfidenceInterval(
      long noisedX,
      int l0Sensitivity,
      long lInfSensitivity,
      double epsilon,
      @Nullable Double delta,
      double alpha) {
    // Computing the confidence interval around zero rather than nosiedX helps represent the
    // interval bounds more accurately. The reason is that the resolution of double values is most
    // fine grained around zero.
    ConfidenceInterval confIntAroundZero =
        computeConfidenceInterval(0.0, l0Sensitivity, lInfSensitivity, epsilon, delta, alpha);
    // Adding noisedX after converting the interval bounds to long ensures that no precision is lost
    // due to the coarse resolution of double values for large instances of noisedX.
    return ConfidenceInterval.create(
        SecureNoiseMath.nextSmallerDouble(Math.round(confIntAroundZero.lowerBound()) + noisedX),
        SecureNoiseMath.nextLargerDouble(Math.round(confIntAroundZero.upperBound()) + noisedX));
  }

  /**
   * Computes the cumulative density function, i.e. Pr[Y <= z] for a Laplace random variable Y whose
   * distribution is given by applying the Laplace mechanism to the raw value {@code x} using the
   * specified privacy parameters {@code epsilon}, {@code delta}, and {@code l1Sensitivity}.
   *
   * <p>This is inverse to {@link #computeQuantile(double, double, double, double)} with the same
   * parameters.
   */
  public static double cumulativeDensity(double z, double x, double l1Sensitivity, double epsilon) {
    DpPreconditions.checkL1Sensitivity(l1Sensitivity);
    DpPreconditions.checkEpsilon(epsilon);

    double scale = l1Sensitivity / epsilon;
    double y = z - x;
    if (y > 0) {
      return 1 - 0.5 * exp(-y / scale);
    }
    return 0.5 * exp(y / scale);
  }

  /**
   * Computes the quantile z satisfying Pr[Y <= z] = {@code rank} for a Laplace random variable Y
   * whose distribution is given by applying the Laplace mechanism to the raw value {@code x} using
   * the specified privacy parameters {@code epsilon}, {@code delta}, {@code l0Sensitivity}, and
   * {@code lInfSensitivity}.
   */
  @Override
  public double computeQuantile(
      double rank,
      double x,
      int l0Sensitivity,
      double lInfSensitivity,
      double epsilon,
      @Nullable Double delta) {
    DpPreconditions.checkNoiseComputeQuantileArguments(
        this, rank, l0Sensitivity, lInfSensitivity, epsilon, delta);
    double l1Sensitivity = Noise.getL1Sensitivity(l0Sensitivity, lInfSensitivity);
    return computeQuantile(rank, x, l1Sensitivity, epsilon);
  }

  /**
   * Computes the quantile z satisfying Pr[Y <= z] = {@code rank} for a Laplace random variable Y
   * whose distribution is given by applying the Laplace mechanism to the raw value {@code x} using
   * the specified privacy parameters {@code epsilon}, {@code delta}, and {@code l1Sensitivity}.
   *
   * <p>This is inverse to {@link #cumulativeDensity} with the same parameters.
   */
  public static double computeQuantile(
      double rank, double x, double l1Sensitivity, double epsilon) {
    DpPreconditions.checkL1Sensitivity(l1Sensitivity);
    DpPreconditions.checkEpsilon(epsilon);
    checkArgument(rank > 0 && rank < 1, "rank must be > 0 and < 1. Provided value: %s", rank);

    double lambda = l1Sensitivity / epsilon;
    if (rank < 0.5) {
      return x + lambda * Math.log(2 * rank);
    }
    return x - lambda * Math.log(2 * (1 - rank));
  }

  private void checkParameters(double l1Sensitivity, double epsilon, @Nullable Double delta) {
    DpPreconditions.checkEpsilon(epsilon);
    DpPreconditions.checkNoiseDelta(delta, this);
    DpPreconditions.checkL1Sensitivity(l1Sensitivity);
  }

  private void checkConfidenceIntervalParameters(
      int l0Sensitivity,
      double lInfSensitivity,
      double epsilon,
      @Nullable Double delta,
      double alpha) {
    DpPreconditions.checkAlpha(alpha);
    DpPreconditions.checkSensitivities(l0Sensitivity, lInfSensitivity);
    checkParameters(Noise.getL1Sensitivity(l0Sensitivity, lInfSensitivity), epsilon, delta);
  }

  /**
   * Determines the granularity of the output of {@link #addNoise} based on the epsilon and
   * l1Sensitivity of the Laplace mechanism.
   */
  private static double getGranularity(double l1Sensitivity, double epsilon) {
    return SecureNoiseMath.ceilPowerOfTwo((l1Sensitivity / epsilon) / GRANULARITY_PARAM);
  }
}

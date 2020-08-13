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

import com.google.common.annotations.VisibleForTesting;
import com.google.differentialprivacy.SummaryOuterClass.MechanismType;
import java.security.SecureRandom;
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
   * href="https://github.com/google/differential-privacy/blob/main/common_docs/Secure_Noise_Generation.pdf">here</a>..
   * Larger values result in more fine grained noise, but increase the chance of sampling
   * inaccuracies due to overflows. The probability of an overflow is less than 2^-1000, if the
   * granularity parameter is set to a value of 2^40 or less and the epsilon passed to addNoise is
   * at least 2^-50.
   *
   * <p>This parameter should be a power of 2.
   */
  private static final double GRANULARITY_PARAM = (double) (1L << 40);

  private final SecureRandom random;

  /** Returns a Noise instance initialized with a secure randomness source. */
  public LaplaceNoise() {
    random = new SecureRandom();
  }

  /**
   * Adds Laplace noise to {@code x} such that the output is {@code epsilon}-differentially private,
   * with respect to the specified L_0 and L_inf sensitivities. Note that {@code delta} must be set
   * to {@code null} because it does not parameterize Laplace noise. Moreover, {@code epsilon} must
   * be at least 2^-50.
   */
  @Override
  public double addNoise(
      double x, int l0Sensitivity, double lInfSensitivity, double epsilon, @Nullable Double delta) {
    DpPreconditions.checkSensitivities(l0Sensitivity, lInfSensitivity);

    return addNoise(x, Noise.getL1Sensitivity(l0Sensitivity, lInfSensitivity), epsilon, delta);
  }

  /**
   * See {@link #addNoise(double, int, double, double, Double)}.
   *
   * <p>As opposed to the latter method, this accepts the L_1 sensitivity of {@code x} directly
   * instead of the L_0 and L_Inf proxies. This should be used in settings where it is feasible or
   * more convenient to calculate the L_1 sensitivity directly.
   */
  public double addNoise(double x, double l1Sensitivity, double epsilon, @Nullable Double delta) {
    checkParameters(l1Sensitivity, epsilon, delta);

    double granularity =
        SecureNoiseMath.ceilPowerOfTwo((l1Sensitivity / epsilon) / GRANULARITY_PARAM);
    long twoSidedGeomericSample =
        sampleTwoSidedGeometric(granularity * epsilon / (l1Sensitivity + granularity));
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
      long x, int l0Sensitivity, long lInfSensitivity, double epsilon, @Nullable Double delta) {
    // Calling addNoise on 0.0 avoids casting x to a double value, which is not secure from a
    // privacy perspective as it can have unforeseen effects on the sensitivity of x. Rounding and
    // adding the resulting noise to x in a post processing step is a secure operation (for noise of
    // moderate magnitude, i.e. < 2^53).
    return Math.round(addNoise(0.0, l0Sensitivity, (double) lInfSensitivity, epsilon, delta)) + x;
  }

  @Override
  public MechanismType getMechanismType() {
    return MechanismType.LAPLACE;
  }

  /**
   * See {@link #addNoise(long, int, long, double, Double)}.
   *
   * <p>As opposed to the latter method, this accepts the L_1 sensitivity of {@code x} directly
   * instead of the L_0 and L_Inf proxies. This should be used in settings where it's more
   * convenient (or feasible) to calculate the L_1 sensitivity directly.
   */
  public long addNoise(long x, int l1Sensitivity, double epsilon, @Nullable Double delta) {
    return Math.round(addNoise((double) x, l1Sensitivity, epsilon, delta));
  }

  /**
   * Computes a confidence interval that contains the raw value {@code x} passed to {@link
   * #addNoise(double, int, double, double, Double)} with a probability equal to {@code
   * confidenceLevel} based on the specified {@code noisedX} and noise parameters. Note that {@code
   * delta} must be set to {@code null} because it does not parameterize Laplace noise.
   */
  @Override
  public ConfidenceInterval computeConfidenceInterval(
      double noisedX,
      int l0Sensitivity,
      double lInfSensitivity,
      double epsilon,
      @Nullable Double delta,
      double confidenceLevel) {
    // TODO: Implement confidence interval computation.
    throw new UnsupportedOperationException("Not implemented yet.");
  }

  /**
   * Computes a confidence interval that contains the raw value integer {@code x} passed to {@link
   * #addNoise(long, int, long, double, Double)} with a probability greater or equal to {@code
   * confidenceLevel} based on the specified {@code noisedX} and noise parameters. Note that {@code
   * delta} must be set to {@code null} because it does not parameterize Laplace noise.
   */
  @Override
  public ConfidenceInterval computeConfidenceInterval(
      long noisedX,
      int l0Sensitivity,
      long lInfSensitivity,
      double epsilon,
      @Nullable Double delta,
      double confidenceLevel) {
    // TODO: Implement confidence interval computation.
    throw new UnsupportedOperationException("Not implemented yet.");
  }

  private void checkParameters(double l1Sensitivity, double epsilon, @Nullable Double delta) {
    DpPreconditions.checkEpsilon(epsilon);
    DpPreconditions.checkNoiseDelta(delta, this);
    DpPreconditions.checkL1Sensitivity(l1Sensitivity);
  }

  /**
   * Returns a sample drawn from the geometric distribution of parameter {@code p = 1 - e^-lambda},
   * i.e., the number of Bernoulli trials until the first success where the success probability is
   * {@code 1 - e^-lambda}. The returned sample is truncated to the max long value. To ensure that a
   * truncation happens with probability less than 10^-6, {@code lambda} must be greater than 2^-59.
   */
  @VisibleForTesting
  long sampleGeometric(double lambda) {
    checkArgument(
        lambda > 1.0 / (1L << 59),
        "The parameter lambda must be at least 2^-59. Provided value: %s",
        lambda);

    // Return truncated sample in the case that the sample exceeds the max long value.
    if (random.nextDouble() > -1.0 * Math.expm1(-1.0 * lambda * Long.MAX_VALUE)) {
      return Long.MAX_VALUE;
    }

    // Perform a binary search for the sample in the interval from 1 to max long. Each iteration
    // splits the interval in two and randomly keeps either the left or the right subinterval
    // depending on the respective probability of the sample being contained in them. The search
    // ends once the interval only contains a single sample.
    long left = 0; // exclusive bound
    long right = Long.MAX_VALUE; // inclusive bound

    while (left + 1 < right) {
      // Compute a midpoint that divides the probability mass of the current interval approximately
      // evenly between the left and right subinterval. The resulting midpoint will be less or equal
      // to the arithmetic mean of the interval. This reduces the expected number of iterations of
      // the binary search compared to a search that uses the arithmetic mean as a midpoint. The
      // speed up is more pronounced, the higher the success probability p is.
      long mid =
          (long)
              Math.ceil(
                  (left
                      - (Math.log(0.5) + Math.log1p(Math.exp(lambda * (left - right)))) / lambda));
      // Ensure that mid is contained in the search interval. This is a safeguard to account for
      // potential mathematical inaccuracies due to finite precision arithmetic.
      mid = Math.min(Math.max(mid, left + 1), right - 1);

      // Probability that the sample is at most mid, i.e., q = Pr[X ≤ mid | left < X ≤ right] where
      // X denotes the sample. The value of q should be approximately one half.
      double q = Math.expm1(lambda * (left - mid)) / Math.expm1(lambda * (left - right));
      if (random.nextDouble() <= q) {
        right = mid;
      } else {
        left = mid;
      }
    }
    return right;
  }

  /**
   * Returns a sample drawn from a geometric distribution that is mirrored at 0. The non-negative
   * part of the distribution's PDF matches the PDF of a geometric distribution of parameter {@code
   * p = 1 - e^-lambda} that is shifted to the left by 1 and scaled accordingly.
   */
  private long sampleTwoSidedGeometric(double lambda) {
    long geometricSample = 0;
    boolean sign = false;
    // Keep a sample of 0 only if the sign is positive. Otherwise, the probability of 0 would be
    // twice as high as it should be.
    while (geometricSample == 0 && !sign) {
      geometricSample = sampleGeometric(lambda) - 1;
      sign = random.nextBoolean();
    }
    return sign ? geometricSample : -geometricSample;
  }
}

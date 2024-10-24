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
import com.google.privacy.differentialprivacy.proto.SummaryOuterClass.MechanismType;
import java.security.SecureRandom;
import java.util.Random;
import javax.annotation.Nullable;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.special.Erf;

/**
 * Generates and adds Gaussian noise to a raw piece of numerical data such that the result is
 * securely differentially private.
 *
 * <p>The Gaussian noise is generated according to the binomial sampling mechanism described <a
 * href="https://github.com/google/differential-privacy/blob/main/common_docs/Secure_Noise_Generation.pdf">here</a>.
 * This approach is robust against unintentional privacy leaks due to artifacts of floating point
 * arithmetic.
 */
public class GaussianNoise implements Noise {
  /**
   * The square root of the maximum number n of Bernoulli trials from which a binomial sample is
   * drawn. Larger values result in more fine grained noise, but increase the chance of sampling
   * inaccuracies due to overflows. The probability of such an event will be roughly 2^-45 or less,
   * if the square root is set to 2^57.
   */
  private static final double BINOMIAL_BOUND = (double) (1L << 57);
  /**
   * The absolute bound of the two sided geometric samples k that are used for creating a binomial
   * sample m + n / 2. For performance reasons, m is not composed of n Bernoulli trials. Instead m
   * is obtained via a rejection sampling technique, which sets m = (k + l) * (sqrt(2 * n) + 1),
   * where l is a uniform random sample between 0 and 1. Bounding k is therefore necessary to
   * prevent m from overflowing.
   *
   * <p>The probability of a single sample k being bounded is 2^-45. The overall privacy loss
   * resulting from this bound is minor and can safely be accounted for in the delta parameter.
   */
  private static final long GEOMETRIC_BOUND =
      (Long.MAX_VALUE / Math.round(Math.sqrt(2) * BINOMIAL_BOUND + 1.0)) - 1;

  /**
   * The standard normal distribution, of mean 0 and variance 1. Since we don't need to sample from
   * this distribution but only use its cumulative distribution function, we initialize it with a
   * null random generator as recommended by its documentation.
   */
  private static final NormalDistribution NORMAL_DISTRIBUTION = new NormalDistribution(null, 0, 1);
  /**
   * The relative accuracy at which to stop the binary search to find the tightest sigma such that
   * Gaussian noise satisfies (epsilon, delta)-differential privacy given the sensitivities.
   */
  private static final double GAUSSIAN_SIGMA_ACCURACY = 1e-3;

  private final Random random;

  /** Returns a Noise instance initialized with a secure randomness source. */
  public GaussianNoise() {
    this(new SecureRandom());
  }

  private GaussianNoise(Random random) {
    this.random = random;
  }

  /**
   * Returns a Noise instance initialized with a specified randomness source. This should only be
   * used for testing and may only be called via the static methods in {@link TestNoiseFactory}.
   *
   * <p>This method is package-private for use by the factory.
   */
  static GaussianNoise createForTesting(Random random) {
    return new GaussianNoise(random);
  }

  /**
   * Adds Gaussian noise to {@code x} such that the output is {@code (epsilon,
   * delta)}-differentially private, with respect to the specified L_0 and L_inf sensitivities.
   */
  @Override
  public double addNoise(
      double x, int l0Sensitivity, double lInfSensitivity, double epsilon, double delta) {
    checkParameters(l0Sensitivity, lInfSensitivity, epsilon, delta);

    double l2Sensitivity = Noise.getL2Sensitivity(l0Sensitivity, lInfSensitivity);
    return addNoiseDefinedBySigma(x, getSigma(l2Sensitivity, epsilon, delta));
  }

  /**
   * Adds Gaussian noise to the integer {@code x} such that the output is {@code (epsilon,
   * delta)}-differentially private, with respect to the specified L_0 and L_inf sensitivities.
   */
  @Override
  public long addNoise(
      long x, int l0Sensitivity, long lInfSensitivity, double epsilon, double delta) {
    checkParameters(l0Sensitivity, lInfSensitivity, epsilon, delta);

    double l2Sensitivity = Noise.getL2Sensitivity(l0Sensitivity, lInfSensitivity);
    return addNoiseDefinedBySigma(x, getSigma(l2Sensitivity, epsilon, delta));
  }

  /**
   * Adds Gaussian noise to {@code x} such that the output is {@code rho}-zero Concentrated DP
   * (zCDP) with respect to the specified L_2 sensitivity. For more details on rho-zCDP see
   * https://eprint.iacr.org/2016/816.pdf
   */
  public double addNoiseDefinedByRho(double x, double l2Sensitivity, double rho) {
    checkParametersForRhozCDP(l2Sensitivity, rho);
    return addNoiseDefinedBySigma(x, getSigmaForRho(l2Sensitivity, rho));
  }

  /**
   * Adds Gaussian noise to {@code x} such that the output is {@code rho}-zCDP with respect to the
   * specified L_2 sensitivity.
   */
  public long addNoiseDefinedByRho(long x, double l2Sensitivity, double rho) {
    checkParametersForRhozCDP(l2Sensitivity, rho);
    return addNoiseDefinedBySigma(x, getSigmaForRho(l2Sensitivity, rho));
  }

  private double addNoiseDefinedBySigma(double x, double noiseSigma) {
    double granularity = getGranularity(noiseSigma);

    // The square root of n is chosen in a way that places it in the interval between BINOMIAL_BOUND
    // and BINOMIAL_BOUND / 2. This ensures that the respective binomial distribution consists of
    // enough Bernoulli samples to closely approximate a Gaussian distribution.
    double sqrtN = 2.0 * noiseSigma / granularity;
    long binomialSample = sampleSymmetricBinomial(sqrtN);
    return SecureNoiseMath.roundToMultipleOfPowerOfTwo(x, granularity)
        + binomialSample * granularity;
  }

  private long addNoiseDefinedBySigma(long x, double noiseSigma) {
    double granularity = getGranularity(noiseSigma);

    // The square root of n is chosen in a way that places it in the interval between BINOMIAL_BOUND
    // and BINOMIAL_BOUND / 2. This ensures that the respective binomial distribution consists of
    // enough Bernoulli samples to closely approximate a Gaussian distribution.
    double sqrtN = 2.0 * noiseSigma / granularity;
    long binomialSample = sampleSymmetricBinomial(sqrtN);
    if (granularity <= 1.0) {
      return x + Math.round(binomialSample * granularity);
    }
    return SecureNoiseMath.roundToMultiple(x, (long) granularity)
        + binomialSample * (long) granularity;
  }


  @Override
  public MechanismType getMechanismType() {
    return MechanismType.GAUSSIAN;
  }

  /**
   * Computes a confidence interval that contains the raw value {@code x} passed to {@link
   * #addNoise(double, int, double, double, Double)} with a probability equal to {@code 1 - alpha}
   * based on the specified {@code noisedX} and noise parameters.
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
      Double delta,
      double alpha) {
    checkConfidenceIntervalParameters(l0Sensitivity, lInfSensitivity, epsilon, delta, alpha);
    double z =
        computeQuantile(alpha / 2.0, noisedX, l0Sensitivity, lInfSensitivity, epsilon, delta);
    // Because of the symmetry of the Gaussian distribution, 2 * noisedX - z corresponds to the
    // (1 - alpha/2)-quantile of the distribution, meaning that the interval [z, 2 * noisedX - z]
    // contains 1-alpha of the probability mass. Deriving the (1 - alpha/2)-quantile from the
    // (alpha/2)-quantile and not vice versa is a deliberate choice. The reason is that alpha tends
    // to be very small. Consequently, alpha/2 is more accurately representable as a double than
    // 1 - alpha/2, facilitating numerical computations.
    return ConfidenceInterval.create(z, 2.0 * noisedX - z);
  }

  /**
   * Computes a confidence interval that contains the raw integer value {@code x} passed to {@link
   * #addNoise(long, int, long, double, Double)} with a probability greater or equal to {@code 1 -
   * alpha} based on the specified {@code noisedX} and noise parameters.
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
      Double delta,
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
   * Computes the quantile z satisfying Pr[Y <= z] = {@code rank} for a Gaussian random variable Y
   * whose distribution is given by applying the Gaussian mechanism to the raw value {@code x} using
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

    double l2Sensitivity = Noise.getL2Sensitivity(l0Sensitivity, lInfSensitivity);
    double sigma = getSigma(l2Sensitivity, epsilon, delta);
    return x - sigma * Math.sqrt(2) * Erf.erfcInv(2 * rank);
  }

  private void checkParameters(
      int l0Sensitivity, double lInfSensitivity, double epsilon, Double delta) {
    DpPreconditions.checkSensitivities(l0Sensitivity, lInfSensitivity);
    DpPreconditions.checkEpsilon(epsilon);
    DpPreconditions.checkNoiseDelta(delta, this);

    // The secure Gaussian noise implementation will fail if 2 * lInfSensitivity is infinite.
    double twoLInf = 2.0 * lInfSensitivity;
    checkArgument(
        Double.isFinite(twoLInf), "2 * lInfSensitivity must be finite but is %s", twoLInf);
  }

  private void checkParametersForRhozCDP(double l2Sensitivity, double rho) {
    DpPreconditions.checkL2Sensitivity(l2Sensitivity);
    DpPreconditions.checkRho(rho);
  }

  private void checkConfidenceIntervalParameters(
      int l0Sensitivity, double lInfSensitivity, double epsilon, Double delta, double alpha) {
    DpPreconditions.checkAlpha(alpha);
    checkParameters(l0Sensitivity, lInfSensitivity, epsilon, delta);
  }

  /**
   * Returns the standard deviation of the Gaussian noise necessary to obtain {@code (epsilon,
   * delta)}-differential privacy for the given L_2 sensitivity. The result will deviate from the
   * tightest possible value sigma_tight by at most GAUSSIAN_SIGMA_ACCURACY * sigma_tight.
   *
   * <p>This implementation uses a binary search. Its runtime is rougly log(GAUSSIAN_SIGMA_ACCURACY)
   * + log(max{sigma_tight / l2sensitivity, l2sensitivity / sigma_tight}).
   *
   * <p>The calculation is based on <a href="https://arxiv.org/abs/1805.06530v2">Balle and Wang's
   * "Improving the Gaussian Mechanism for Differential Privacy: Analytical Calibration and Optimal
   * Denoising"</a>. The paper states that the lower bound on sigma from the original analysis of
   * the Gaussian mechanism (sigma ≥ sqrt(2 * l2_sensitivity^2 * log(1.25/𝛿) / 𝜖^2)) is far from
   * tight and binary search can give us a better lower bound.
   */
  public static double getSigma(double l2Sensitivity, double epsilon, double delta) {
    // We use l2sensitivity as a starting guess for the upper bound, since the required noise grows
    // linearly with sensitivity.
    double upperBound = l2Sensitivity;
    double lowerBound = 0;

    // Increase lowerBound and upperBound until upperBound is actually an upper bound of
    // sigma_tight, using exponential search.
    while (getDelta(upperBound, l2Sensitivity, epsilon) > delta) {
      lowerBound = upperBound;
      upperBound = upperBound * 2;
    }

    // Binary search [lowerBound, upperBound] to find a good enough approximation of sigma_tight.
    while (upperBound - lowerBound > GAUSSIAN_SIGMA_ACCURACY * lowerBound) {
      double middle = lowerBound * 0.5 + upperBound * 0.5;
      if (getDelta(middle, l2Sensitivity, epsilon) > delta) {
        lowerBound = middle;
      } else {
        upperBound = middle;
      }
    }

    // Return the over-approximation to err on the safe side.
    return upperBound;
  }

  /*
   * Computes sigma of Gaissian noise to satisify rho Zero Concentrated DP (rho-zCDP).
   * For more details on rho-zCDP see https://eprint.iacr.org/2016/816.pdf
   */
  public static double getSigmaForRho(double l2Sensitivity, double rho) {
    // From https://eprint.iacr.org/2016/816.pdf Propositon 6.
    return l2Sensitivity / Math.sqrt(2 * rho);
  }

  /**
   * Returns the smallest delta such that the Gaussian mechanism with standard deviation {@code
   * sigma} obtains {@code (epsilon, delta)}-differential privacy with respect to the provided L_2
   * sensitivity. The calculation is based on Theorem 8 of Balle and Wang's "Improving the Gaussian
   * Mechanism for Differential Privacy: Analytical Calibration and Optimal Denoising", available <a
   * href="https://arxiv.org/abs/1805.06530v2">here</a>.
   */
  private static double getDelta(double sigma, double l2Sensitivity, double epsilon) {
    // Denoting by CDF the CDF function of the standard Gaussian distribution (mean 0, variance 1),
    // and s the L2 sensitivity, the tight choice of delta is:
    //    CDF(s/(2*sigma) - epsilon*sigma/s) - exp(epsilon)*CDF(-s/(2*sigma) - epsilon*sigma/s)
    // To simplify the reasoning floating-point underflow/overflows, we rewrite this formula into:
    //    CDF(a - b) - c * CDF(-a - b)
    // where a = s / (2 * sigma), b = epsilon * sigma / s, and c = exp(epsilon).
    double a = l2Sensitivity / (2 * sigma);
    double b = epsilon * sigma / l2Sensitivity;
    double c = Math.exp(epsilon);

    if (b == Double.POSITIVE_INFINITY || c == Double.POSITIVE_INFINITY) {
      // If either l2Sensitivity goes to 0 or e^epsilon goes to infinity, delta goes to 0.
      return 0;
    }
    return NORMAL_DISTRIBUTION.cumulativeProbability(a - b)
        - c * NORMAL_DISTRIBUTION.cumulativeProbability(-a - b);
  }

  /**
   * Determines the granularity of the output of {@link addNoise} based on the sigma of the Gaussian
   * noise.
   */
  private static double getGranularity(double sigma) {
    return SecureNoiseMath.ceilPowerOfTwo(2.0 * sigma / BINOMIAL_BOUND);
  }

  /**
   * Returns a random sample m where {@code m + n / 2} is drawn from a binomial distribution of
   * {@code n} Bernoulli trials that have a success probability of 1 / 2 each. The sampling
   * technique is based on Bringmann et al.'s rejection sampling approach proposed in "Internal DLA:
   * Efficient Simulation of a Physical Growth Model", available <a
   * href="https://people.mpi-inf.mpg.de/~kbringma/paper/2014ICALP.pdf">here</a>.
   *
   * <p>The square root of {@code n} must be at least 10^6. This is to ensure an accurate
   * approximation of a Gaussian distribution.
   */
  @VisibleForTesting
  long sampleSymmetricBinomial(double sqrtN) {
    checkArgument(sqrtN >= 1000000.0, "Input must be at least 10^6. Provided value: %s", sqrtN);
    checkArgument(Double.isFinite(sqrtN), "Input must be finite. Provided value: %s", sqrtN);

    long stepSize = Math.round(Math.sqrt(2) * sqrtN + 1.0);
    while (true) {
      long geometricSample = sampleBoundedGeometric();
      long twoSidedGeometricSample = random.nextBoolean() ? geometricSample : -geometricSample - 1;
      long result = stepSize * twoSidedGeometricSample + sampleUniform(stepSize);

      double resultProbability = approximateBinomialProbability(sqrtN, result);
      double rejectProbability = random.nextDouble();
      if (resultProbability > 0.0
          && rejectProbability > 0.0
          && rejectProbability
              < resultProbability * stepSize * Math.pow(2.0, geometricSample) / 4.0) {
        return result;
      }
    }
  }

  /**
   * Returns a sample drawn from the geometric distribution with success probability 1 / 2, i.e.,
   * the number of unsuccessful Bernoulli trials until the first success. The sample is capped
   * should it exceed the geometric bound.
   */
  private long sampleBoundedGeometric() {
    long result = 0;
    while (random.nextBoolean() && result < GEOMETRIC_BOUND) {
      result++;
    }
    return result;
  }

  /**
   * Draws an integer greater or equal to 0 and strictly less than {@code n} uniformly at random.
   * This custom implementation is necessary because SecureRandom provides such functionality only
   * for int but not for long.
   */
  private long sampleUniform(long n) {
    long largestMultipleOfN = (Long.MAX_VALUE / n) * n;

    while (true) {
      long signMask = 0x7fffffffffffffffL;
      long uniformNonNegativeLong = signMask & random.nextLong();
      if (uniformNonNegativeLong < largestMultipleOfN) {
        return uniformNonNegativeLong % n;
      }
    }
  }

  /**
   * Approximates the probability of a random sample {@code m + n / 2} drawn from a binomial
   * distribution of n Bernoulli trials that have a success probability of 1 / 2 each. The
   * approximation is taken from Lemma 7 of the noise generation documentation, available <a
   * href="https://github.com/google/differential-privacy/blob/main/common_docs/Secure_Noise_Generation.pdf">here</a>.
   *
   * <p>Note that m might be very large and m * m might not be representable as long.
   */
  private static double approximateBinomialProbability(double sqrtN, long m) {
    if (Math.abs(m) > sqrtN * Math.sqrt(Math.log(sqrtN) / 2)) {
      return 0.0;
    } else {
      return (Math.sqrt(2.0 / Math.PI) / sqrtN)
          * Math.exp(-2.0 * Math.pow(m / sqrtN, 2))
          * (1 - (0.4 * Math.pow(2.0 * Math.log(sqrtN), 1.5) / sqrtN));
    }
  }
}

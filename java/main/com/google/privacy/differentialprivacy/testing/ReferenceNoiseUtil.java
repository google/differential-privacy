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

package com.google.privacy.differentialprivacy.testing;

import static com.google.common.base.Preconditions.checkArgument;

import java.security.SecureRandom;
import org.apache.commons.math3.distribution.NormalDistribution;

/**
 * Utility class providing tools for drawing random samples that may serve as a reference when
 * evaluating statistical tests. Note that these samples do not secure against privacy attacks and
 * should not be used as a source of noise for differentially private mechanisms.
 */
public final class ReferenceNoiseUtil {
  private static final double GAUSSIAN_VARIANCE_ACCURACY = 1e-3;
  private static final NormalDistribution NORMAL_DISTRIBUTION = new NormalDistribution(null, 0, 1);
  private static final SecureRandom RANDOM = new SecureRandom();

  private ReferenceNoiseUtil() {}

  /**
   * Returns a random sample drawn from a Laplace distribution with the specified {@code mean} and
   * {@code variance}. The {@code variance} must be a positive value.
   */
  public static double sampleLaplace(double mean, double variance) {
    checkArgument(variance > 0.0, "variance must be positive");
    // Draw a random sample from the interval (0,1) uniformly at random and transform it to the
    // Laplace distribution via the inverse transform method. The values 0 and 1 are excluded as
    // they are discontinuities of the transformation function.
    double randomDouble = 0;
    while (randomDouble <= 0 || 1 <= randomDouble) {
      randomDouble = RANDOM.nextDouble();
    }
    if (randomDouble < 0.5) {
      return mean + Math.sqrt(variance / 2) * Math.log(2 * randomDouble);
    } else {
      return mean - Math.sqrt(variance / 2) * Math.log(2 - 2 * randomDouble);
    }
  }

  /**
   * Returns a random sample drawn from a Laplace distribution centered around the {@code rawInput}
   * such that the sample is {@code epsilon}-differentially private for the specified {@code
   * l1Sensitivity}. The {@code l1Sensitivity} and {@code epsilon} must be a positive.
   */
  public static double sampleLaplace(double rawInput, double epsilon, double l1Sensitivity) {
    return sampleLaplace(rawInput, getLaplaceVariance(epsilon, l1Sensitivity));
  }

  /**
   * Returns a random sample drawn from a Gaussian distribution with the specified mean and
   * variance. The variance must be a positive value.
   */
  public static double sampleGaussian(double mean, double variance) {
    checkArgument(variance > 0.0, "variance must be positive");
    return mean + RANDOM.nextGaussian() * Math.sqrt(variance);
  }

  /**
   * Returns a random sample drawn from a Gaussian distribution centered around the {@code rawInput}
   * such that the sample is ({@code epsilon}, {@code delta})-differentially private for the
   * specified {@code l2Sensitivity}. The {@code l2Sensitivity}, {@code epsilon} and {@code delta}
   * must be a positive.
   */
  public static double sampleGaussian(
      double rawInput, double epsilon, double delta, double l2Sensitivity) {
    return sampleGaussian(rawInput, getGaussianVariance(epsilon, delta, l2Sensitivity));
  }

  /**
   * Returns the smallest variance for which a Laplace distribution is {@code
   * epsilon}-differentially private with respect to the provided {@code l1Sensitivity}.
   */
  public static double getLaplaceVariance(double epsilon, double l1Sensitivity) {
    checkArgument(epsilon > 0.0, "epsilon must be positive");
    checkArgument(l1Sensitivity > 0.0, "l1Sensitivity must be positive");

    return 2.0 * Math.pow(l1Sensitivity / epsilon, 2.0);
  }

  /**
   * Returns the smallest variance for which a Gaussian distribution is ({@code epsilon}, {@code
   * delta})-differentially private with respect to the provided {@code l2Sensitivity}.
   */
  public static double getGaussianVariance(double epsilon, double delta, double l2Sensitivity) {
    checkArgument(epsilon > 0.0, "epsilon must be positive");
    checkArgument(delta > 0.0, "delta must be positive");
    checkArgument(l2Sensitivity > 0.0, "l2Sensitivity must be positive");

    // We use l2sensitivity as a starting guess for the upper bound, since the required noise grows
    // linearly with sensitivity.
    double upperBound = l2Sensitivity;
    double lowerBound = 0;

    // Increase lowerBound and upperBound until upperBound is actually an upper bound of
    // sigma_tight, using exponential search.
    while (getGaussianDelta(upperBound, l2Sensitivity, epsilon) > delta) {
      lowerBound = upperBound;
      upperBound = upperBound * 2;
    }

    // Binary search [lowerBound,upperBound] to find a good enough approximation of sigma_tight.
    while (upperBound - lowerBound > GAUSSIAN_VARIANCE_ACCURACY * lowerBound) {
      double middle = lowerBound * 0.5 + upperBound * 0.5;
      if (getGaussianDelta(middle, l2Sensitivity, epsilon) > delta) {
        lowerBound = middle;
      } else {
        upperBound = middle;
      }
    }

    // Return the over-approximation to err on the safe side.
    return upperBound * upperBound;
  }

  /**
   * Returns the smallest delta such that the Gaussian mechanism with standard deviation {@code
   * sigma} obtains {@code (epsilon, delta)}-differential privacy with respect to the provided
   * {@code l2Sensitivity}. The calculation is based on Theorem 8 of Balle and Wang's "Improving the
   * Gaussian Mechanism for Differential Privacy: Analytical Calibration and Optimal Denoising",
   * available at https://arxiv.org/abs/1805.06530v2.
   */
  private static double getGaussianDelta(double sigma, double l2Sensitivity, double epsilon) {
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
   * Returns an upper bound on the l1Sensitivity based on the provided {@code l0Sensitivity} and
   * {@code lInfSensitivity}. The {@code l0Sensitivity} and {@code lInfSensitivity} must be
   * positive.
   */
  public static double getL1Sensitivity(int l0Sensitivity, double lInfSensitivity) {
    checkArgument(l0Sensitivity > 0.0, "l0Sensitivity must be positive");
    checkArgument(lInfSensitivity > 0.0, "lInfSensitivity must be positive");

    return l0Sensitivity * lInfSensitivity;
  }

  /**
   * Returns an upper bound on the l2Senistvity based on the provided {@code l0Sensitivity} and
   * {@code lInfSensitivity}. The {@code l0Sensitivity} and {@code lInfSensitivity} must be
   * positive.
   */
  public static double getL2Sensitivity(int l0Sensitivity, double lInfSensitivity) {
    checkArgument(l0Sensitivity > 0.0, "l0Sensitivity must be positive");
    checkArgument(lInfSensitivity > 0.0, "lInfSensitivity must be positive");

    return Math.sqrt(l0Sensitivity) * lInfSensitivity;
  }
}

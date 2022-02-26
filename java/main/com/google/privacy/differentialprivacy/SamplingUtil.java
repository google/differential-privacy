//
// Copyright 2022 Google LLC
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
import static java.lang.Math.max;
import static java.lang.Math.min;

import java.util.Random;

/** Function for sampling from several distributions. */
final class SamplingUtil {

  private SamplingUtil() {}

  /**
   * Returns a sample drawn from the geometric distribution of parameter {@code p = 1 - e^-lambda},
   * i.e., the number of Bernoulli trials until the first success where the success probability is
   * {@code 1 - e^-lambda}. The returned sample is truncated to the max long value. To ensure that a
   * truncation happens with probability less than 10^-6, {@code lambda} must be greater than 2^-59.
   */
  static long sampleGeometric(Random random, double lambda) {
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
      mid = min(max(mid, left + 1), right - 1);

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
  public static long sampleTwoSidedGeometric(Random random, double lambda) {
    long geometricSample = 0;
    boolean sign = false;
    // Keep a sample of 0 only if the sign is positive. Otherwise, the probability of 0 would be
    // twice as high as it should be.
    while (geometricSample == 0 && !sign) {
      geometricSample = sampleGeometric(random, lambda) - 1;
      sign = random.nextBoolean();
    }
    return sign ? geometricSample : -geometricSample;
  }
}

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
import static java.lang.Math.max;
import static java.lang.Math.min;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

/** Utility class providing tests for evaluating statistical properties of anonymization tools. */
public final class StatisticalTestsUtil {

  // TODO: Move Java Doc to README file to improve clarity and structure.

  private StatisticalTestsUtil() {}

  /**
   * Decides whether two sets of random samples were likely drawn from similar discrete
   * distributions.
   *
   * <p>The distributions are considered similar if the l2 distance between them is less than half
   * the specified l2 tolerance t. Otherwise, if the distance is greater than t, they are considered
   * dissimilar. The error probability is at most 4014 / (n * t^2), where n is the number of samples
   * contained in one of the sets. See (broken link) for more information.
   *
   * @param samplesA a non-empty collection of independent random samples drawn from one of the
   *     distributions that are being compared.
   * @param samplesB a collection of independent random samples of the same size as the previous
   *     collection drawn from the other distribution.
   * @param l2Tolerance the l2 distance beyond which the two distributions are considered
   *     dissimilar. The lower the tolerance, the more samples are required to obtain acceptable
   *     error bounds. As a general guideline, this parameter must be positive and should be less
   *     than 1 with values greater than 10^-4 still being computationally feasible.
   * @return true if the samples are likely drawn from similar distributions and false otherwise.
   */
  public static <T> boolean verifyCloseness(T[] samplesA, T[] samplesB, double l2Tolerance) {
    checkArgument(samplesA.length == samplesB.length, "The sample sets must be of equal size.");
    checkArgument(samplesA.length > 0, "The sample sets must not be empty");
    checkArgument(l2Tolerance > 0, "The l2 tolerance must be positive");
    checkArgument(l2Tolerance < 1, "The l2 tolerance should be less than 1");

    Map<T, Long> histogramA = buildHistogram(samplesA);
    Map<T, Long> histogramB = buildHistogram(samplesB);

    long selfCollisionCountA = 0;
    long selfCollisionCountB = 0;
    long crossCollisionCount = 0;
    for (long count : histogramA.values()) {
      selfCollisionCountA += (count * (count - 1)) / 2;
    }
    for (long count : histogramB.values()) {
      selfCollisionCountB += (count * (count - 1)) / 2;
    }
    for (T sample : histogramA.keySet()) {
      if (histogramB.containsKey(sample)) {
        crossCollisionCount += histogramA.get(sample) * histogramB.get(sample);
      }
    }

    double testValue =
        selfCollisionCountA
            + selfCollisionCountB
            - ((samplesA.length - 1.0) / samplesA.length) * crossCollisionCount;
    double threshold =
        (l2Tolerance * (samplesA.length - 1)) * (l2Tolerance * samplesA.length) / 4.0;
    return testValue < threshold;
  }

  /**
   * Decides whether two sets of random samples were likely drawn from a pair of discrete
   * distributions that approximately satisfy (ε, δ) differential privacy.
   *
   * <p>The two distributions are considered to be (ε, δ) differentially private if the likelihood
   * of any event with respect to the first distribution is at most δ plus e^ε times the likelihood
   * of the same event in the second distribution and vice versa. Moreover, the distributions are
   * considered approximately (ε, δ) differentially private if there exists a δ' such that the
   * distributions are (ε, δ') differentially private and |δ' - δ| is less than half of a given
   * tolerance α. Otherwise if no δ' exists such that |δ' - δ| is less than α, the distributions are
   * not considered approximately (ε, δ) differentially private. Assuming that α > (m / n)^0.5 * (1
   * + e^(2 * ε)), the error probability is at most (1 + e^(2 * ε)) / (n * (α - (m / n)^0.5 * (1 +
   * e^(2 * ε)))^2), where m is the size of the support of the distributions and n is the expected
   * value of a Poisson distribution from which the number of samples is drawn. See
   * (broken link) for more information.
   *
   * @param samplesA a non-empty collection of independent random samples drawn from one of the
   *     distributions that are being compared.
   * @param samplesB a collection of independent random samples of the same size as the previous
   *     collection drawn from the other distribution.
   * @param epsilon the ε privacy parameter for which approximate differential privacy is assessed.
   *     The value of this parameter must not be negative.
   * @param delta the δ privacy parameter for which approximate differential privacy is assessed.
   *     The value of this parameter must not be negative and should be less than 1.
   * @param deltaTolerance the threshold α on the absolute difference |δ' - δ| beyond which the
   *     distributions are not considered approximately (ε, δ) differentially private anymore. The
   *     lower the tolerance, the more samples are required to obtain acceptable error bounds. This
   *     parameter must be positive and should be less than 1.
   * @return true if the samples are likely drawn from approximately (ε, δ) differentially private
   *     distributions and false otherwise.
   */
  public static <T> boolean verifyApproximateDp(
      T[] samplesA, T[] samplesB, double epsilon, double delta, double deltaTolerance) {
    checkArgument(samplesA.length == samplesB.length, "The sample sets must be of equal size.");
    checkArgument(samplesA.length > 0, "The sample sets must not be empty");
    checkArgument(deltaTolerance > 0, "The delta tolerance must be positive");
    checkArgument(deltaTolerance < 1, "The delta tolerance should be less than 1");
    checkArgument(epsilon >= 0, "Epsilon must not be negative");
    checkArgument(delta >= 0, "Delta must not be negative");
    checkArgument(delta < 1, "Delta should be less than 1");

    Map<T, Long> histogramA = buildHistogram(samplesA);
    Map<T, Long> histogramB = buildHistogram(samplesB);

    double testValueA =
        computeAproximateDpTestValue(histogramA, histogramB, epsilon, samplesA.length);
    double testValueB =
        computeAproximateDpTestValue(histogramB, histogramA, epsilon, samplesA.length);
    return testValueA < delta + deltaTolerance && testValueB < delta + deltaTolerance;
  }

  /**
   * Rounds a numerical {@code sample} to the next multiple of the specified {@code granularity}.
   * This is intended as a preprocessing step for continous samples before evaluating them based on
   * a discrete statistical test. Note that {@code granularity} must be a positive value.
   */
  public static Double discretize(double sample, double granularity) {
    checkArgument(granularity > 0, "granularity must be positive");
    double scaledSample = sample / granularity;
    if (Math.abs(scaledSample) >= 1L << 54) {
      // Rounding scaledSample to a long value may result in an overflow. However, since its
      // absolute value is greater or equal to 2^54, no rounding is required in the first place.
      // This is because the mantissa of double values only has 52 bits of precision, which in turn
      // implies that a double greater or equal to 2^54 has no fractional digits.
      return scaledSample * granularity;
    } else {
      // The absolute value of scaledSample is less than 2^54 and therefore it can be rounded to a
      // long value without risking an overflow.
      return Math.round(scaledSample) * granularity;
    }
  }

  /**
   * Rounds a numerical {@code sample} to the next multiple of the specified {@code granularity}.
   * This is intended as a preprocessing step for continous samples before evaluating them based on
   * a discrete statistical test. Note that {@code granularity} must be a positive value.
   */
  public static Long discretize(Long sample, double granularity) {
    double doubleResult = StatisticalTestsUtil.discretize(sample.doubleValue(), granularity);
    checkArgument(doubleResult <= Long.MAX_VALUE, "sample does overflow");
    return Math.round(doubleResult);
  }

  /**
   * Partitions the interval between {@code lower} and {@code upper} into {@code numberOfBuckets}
   * subintervals of equal size and returns the index (from 0 to {@code numberOfBuckets} - 1) of the
   * subinterval that contains the specified {@code sample}.
   *
   * <p>Note that {@code lower} <= {@code sample} <= {@code upper}, {@code lower} < {@code upper}
   * and {@code numberOfBuckets} > 0.
   */
  public static int bucketize(double sample, double lower, double upper, int numberOfBuckets) {
    checkArgument(sample >= lower, "sample must be greater or equal to lower");
    checkArgument(sample <= upper, "sample must be less or equal to upper");
    checkArgument(lower < upper, "lower must be strictly less than upper");
    checkArgument(numberOfBuckets > 0, "numberOfBuckets must be greater than 0");
    return min(
        numberOfBuckets - 1,
        (int) Math.floor(((sample - lower) / (upper - lower)) * numberOfBuckets));
  }

  /**
   * Run {@code sampleGenerator} {@code numberOfSamples} times using {@code numThreads} threads and
   * return the results in a list.
   */
  public static <T> List<T> generateSamplesParallel(
      Supplier<T> sampleGenerator, int numberOfSamples, int numThreads)
      throws InterruptedException {
    List<T> samples = new ArrayList<T>(numberOfSamples);

    List<Thread> threads = new ArrayList<>();
    for (int i = 0; i < numThreads; i++) {
      threads.add(
          new Thread(
              () -> {
                while (true) {
                  T sample = sampleGenerator.get();
                  synchronized (samples) {
                    if (samples.size() >= numberOfSamples) {
                      break;
                    }
                    samples.add(sample);
                  }
                }
              }));
    }
    for (Thread thread : threads) {
      thread.start();
    }
    for (Thread thread : threads) {
      thread.join();
    }
    return samples;
  }

  private static <T> double computeAproximateDpTestValue(
      Map<T, Long> histogramA, Map<T, Long> histogramB, double epsilon, int numOfSamples) {
    double testValue = 0;
    for (T sample : histogramA.keySet()) {
      double sampleCountA = histogramA.get(sample);
      if (histogramB.containsKey(sample)) {
        double sampleCountB = histogramB.get(sample);
        testValue += max(0.0, (sampleCountA - Math.exp(epsilon) * sampleCountB) / numOfSamples);
      } else {
        testValue += sampleCountA / numOfSamples;
      }
    }
    return testValue;
  }

  private static <T> Map<T, Long> buildHistogram(T[] samples) {
    Map<T, Long> histogram = new HashMap<>();
    for (T sample : samples) {
      if (histogram.containsKey(sample)) {
        histogram.put(sample, histogram.get(sample) + 1);
      } else {
        histogram.put(sample, 1L);
      }
    }
    return histogram;
  }
}

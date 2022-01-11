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

import static com.google.common.truth.Truth.assertThat;
import static com.google.privacy.differentialprivacy.proto.testing.StatisticalTests.NoiseType.GAUSSIAN;
import static com.google.privacy.differentialprivacy.proto.testing.StatisticalTests.NoiseType.LAPLACE;
import static java.lang.Math.max;
import static java.lang.Math.sqrt;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.Supplier;
import com.google.privacy.differentialprivacy.proto.testing.StatisticalTests.BoundedStdvDpTestCase;
import com.google.privacy.differentialprivacy.proto.testing.StatisticalTests.BoundedStdvDpTestCaseCollection;
import com.google.privacy.differentialprivacy.proto.testing.StatisticalTests.BoundedStdvSamplingParameters;
import com.google.privacy.differentialprivacy.proto.testing.StatisticalTests.DpTestParameters;
import com.google.privacy.differentialprivacy.proto.testing.StatisticalTests.NoiseType;
import com.google.protobuf.TextFormat;
import java.io.IOException;
import java.io.InputStreamReader;
import java.security.SecureRandom;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(Parameterized.class)
public final class BoundedStdvDpTestCasesValidityTest {
  private static final String TEST_CASES_FILE_PATH =

  "external/com_google_differential_privacy/proto/testing/bounded_stdv_dp_test_cases.textproto";

  // The fractions of the privacy budget that are allocated to the count, sum and sum of squares
  // aggregation respectively. The three fractions should sum up to 1.0.
  private static final double COUNT_PRIVACY_BUDGET_FRACTION = 0.333333;
  private static final double SUM_PRIVACY_BUDGET_FRACTION = 0.333333;
  private static final double SUM_OF_SQUARES_PRIVACY_BUDGET_FRACTION = 0.333333;
  private final int numberOfVotes =
      getTestCaseCollectionFromFile().getVotingParameters().getNumberOfVotes();
  private final double distanceSpecificity =
      getTestCaseCollectionFromFile().getValidityParameters().getDistanceSpecificity();
  private final double failureSpecificity =
      getTestCaseCollectionFromFile().getValidityParameters().getFailureSpecificity();
  private final BoundedStdvDpTestCase testCase;

  public BoundedStdvDpTestCasesValidityTest(BoundedStdvDpTestCase testCase) {
    this.testCase = testCase;
  }

  @Parameterized.Parameters
  public static Iterable<? extends Object> testCases() {
    return getTestCaseCollectionFromFile().getBoundedStdvDpTestCaseList();
  }

  @Test
  public void boundedStdvDpTest_acceptsStdvsDifferingByTheSensitivity() {
    BoundedStdvSamplingParameters samplingParameters = testCase.getBoundedStdvSamplingParameters();
    DpTestParameters dpTestParameters = testCase.getDpTestParameters();
    double epsilon = samplingParameters.getEpsilon();
    double delta = samplingParameters.getDelta();
    int maxPartitionsContributed = samplingParameters.getMaxPartitionsContributed();
    int maxContributionsPerPartition = samplingParameters.getMaxContributionsPerPartition();
    double distanceBetweenBounds =
        samplingParameters.getUpperBound() - samplingParameters.getLowerBound();
    NoiseType noiseType = samplingParameters.getNoiseType();

    double countVariance =
        getCountVariance(
            epsilon * COUNT_PRIVACY_BUDGET_FRACTION,
            delta * COUNT_PRIVACY_BUDGET_FRACTION,
            maxPartitionsContributed,
            maxContributionsPerPartition,
            noiseType);
    double sumVariance =
        getSumVariance(
            epsilon * SUM_PRIVACY_BUDGET_FRACTION,
            delta * SUM_PRIVACY_BUDGET_FRACTION,
            maxPartitionsContributed,
            maxContributionsPerPartition,
            distanceBetweenBounds,
            noiseType);
    double sumOfSquaresVariance =
        getSumVariance(
            epsilon * SUM_OF_SQUARES_PRIVACY_BUDGET_FRACTION,
            delta * SUM_OF_SQUARES_PRIVACY_BUDGET_FRACTION,
            maxPartitionsContributed,
            maxContributionsPerPartition,
            // The distance between the lower and upper bound of an entry of the sum of square is
            // (distanceBetweenBounds * 0.5)^2
            (distanceBetweenBounds * 0.5) * (distanceBetweenBounds * 0.5),
            noiseType);
    double sensitivity =
        getCountSensitivity(maxPartitionsContributed, maxContributionsPerPartition, noiseType);
    double normalizedBound = distanceBetweenBounds * 0.5;

    // To check that a DP test reliably accepts DP samples drawn from two neighbouring datasets
    // according to the specified privacy parameters, we construct the datasets in a way that
    // maximizes the difference in standard deviation. For this purpose, we assume that both
    // datasets contain n entries of value l where n ≤ sensitivity and l is the normalized lower
    // bound. Moreover, we assume that the second dataset contains n entries of value u where u is
    // the normalized upper bound.
    Supplier<Double> sampleGenerator =
        () ->
            computeNoisyStdv(
                /*noisyCount =*/ sensitivity + sampleReferenceNoise(countVariance, noiseType),
                /*noisySum =*/ sensitivity * -1.0 * normalizedBound
                    + sampleReferenceNoise(sumVariance, noiseType),
                /*noisySumOfSquares =*/ sensitivity * normalizedBound * normalizedBound
                    + sampleReferenceNoise(sumOfSquaresVariance, noiseType));
    Supplier<Double> neighbouringSampleGenerator =
        () ->
            computeNoisyStdv(
                /*noisyCount =*/ 2.0 * sensitivity + sampleReferenceNoise(countVariance, noiseType),
                /*noisySum =*/ 0.0 + sampleReferenceNoise(sumVariance, noiseType),
                /*noisySumOfSquares =*/ 2.0 * sensitivity * normalizedBound * normalizedBound
                    + sampleReferenceNoise(sumOfSquaresVariance, noiseType));

    assertThat(
            VotingUtil.runBallot(
                () ->
                    generateVote(
                        sampleGenerator,
                        neighbouringSampleGenerator,
                        samplingParameters.getNumberOfSamples(),
                        dpTestParameters.getEpsilon(),
                        dpTestParameters.getDelta(),
                        dpTestParameters.getDeltaTolerance(),
                        dpTestParameters.getGranularity()),
                numberOfVotes))
        .isTrue();
  }

  @Test
  public void boundedStdvDpTest_rejectsStdvsDifferingByMoreThanTheSensitivity() {
    BoundedStdvSamplingParameters samplingParameters = testCase.getBoundedStdvSamplingParameters();
    DpTestParameters dpTestParameters = testCase.getDpTestParameters();
    double epsilon = samplingParameters.getEpsilon();
    double delta = samplingParameters.getDelta();
    int maxPartitionsContributed = samplingParameters.getMaxPartitionsContributed();
    int maxContributionsPerPartition = samplingParameters.getMaxContributionsPerPartition();
    double distanceBetweenBounds =
        samplingParameters.getUpperBound() - samplingParameters.getLowerBound();
    NoiseType noiseType = samplingParameters.getNoiseType();

    double countVariance =
        getCountVariance(
            epsilon * COUNT_PRIVACY_BUDGET_FRACTION,
            delta * COUNT_PRIVACY_BUDGET_FRACTION,
            maxPartitionsContributed,
            maxContributionsPerPartition,
            noiseType);
    double sumVariance =
        getSumVariance(
            epsilon * SUM_PRIVACY_BUDGET_FRACTION,
            delta * SUM_PRIVACY_BUDGET_FRACTION,
            maxPartitionsContributed,
            maxContributionsPerPartition,
            distanceBetweenBounds,
            noiseType);
    double sumOfSquaresVariance =
        getSumVariance(
            epsilon * SUM_OF_SQUARES_PRIVACY_BUDGET_FRACTION,
            delta * SUM_OF_SQUARES_PRIVACY_BUDGET_FRACTION,
            maxPartitionsContributed,
            maxContributionsPerPartition,
            // The distance between the lower and upper bound of an entry of the sum of square is
            // (distanceBetweenBounds * 0.5)^2
            (distanceBetweenBounds * 0.5) * (distanceBetweenBounds * 0.5),
            noiseType);
    double scaledSensitivity =
        getCountSensitivity(maxPartitionsContributed, maxContributionsPerPartition, noiseType)
            * distanceSpecificity;
    double normalizedBound = distanceBetweenBounds * 0.5;

    // To check that a DP test rejects samples drawn from two neighbouring datasets that violate the
    // specified sensitivity, we construct the datasets in a way that maximizes the difference in
    // standard deviation for the given sensitivity violation. For this purpose, we assume that both
    // datasets contain n entries of value l where n ≥ sensitivity * distanceSpecificity and l is
    // the normalized lower bound. Moreover, we assume that the second dataset contains n entries of
    // value u where u is the normalized upper bound.
    Supplier<Double> sampleGenerator =
        () ->
            computeNoisyStdv(
                /*noisyCount=*/ scaledSensitivity + sampleReferenceNoise(countVariance, noiseType),
                /*noisySum=*/ scaledSensitivity * -1.0 * normalizedBound
                    + sampleReferenceNoise(sumVariance, noiseType),
                /*noisySumOfSquares=*/ scaledSensitivity * normalizedBound * normalizedBound
                    + sampleReferenceNoise(sumOfSquaresVariance, noiseType));
    Supplier<Double> neighbouringSampleGenerator =
        () ->
            computeNoisyStdv(
                /*noisyCount=*/ 2.0 * scaledSensitivity
                    + sampleReferenceNoise(countVariance, noiseType),
                /*noisySum=*/ 0.0 + sampleReferenceNoise(sumVariance, noiseType),
                /*noisySumOfSquares=*/ 2.0 * scaledSensitivity * normalizedBound * normalizedBound
                    + sampleReferenceNoise(sumOfSquaresVariance, noiseType));

    assertThat(
            VotingUtil.runBallot(
                () ->
                    generateVote(
                        sampleGenerator,
                        neighbouringSampleGenerator,
                        samplingParameters.getNumberOfSamples(),
                        dpTestParameters.getEpsilon(),
                        dpTestParameters.getDelta(),
                        dpTestParameters.getDeltaTolerance(),
                        dpTestParameters.getGranularity()),
                numberOfVotes))
        .isFalse();
  }

  @Test
  public void boundedStdvsDpTest_rejectsCriticallyFailingStdvs() {
    BoundedStdvSamplingParameters samplingParameters = testCase.getBoundedStdvSamplingParameters();
    DpTestParameters dpTestParameters = testCase.getDpTestParameters();
    SecureRandom random = new SecureRandom();
    double epsilon = samplingParameters.getEpsilon();
    double delta = samplingParameters.getDelta();
    int maxPartitionsContributed = samplingParameters.getMaxPartitionsContributed();
    int maxContributionsPerPartition = samplingParameters.getMaxContributionsPerPartition();
    double distanceBetweenBounds =
        samplingParameters.getUpperBound() - samplingParameters.getLowerBound();
    NoiseType noiseType = samplingParameters.getNoiseType();

    double countVariance =
        getCountVariance(
            epsilon * COUNT_PRIVACY_BUDGET_FRACTION,
            delta * COUNT_PRIVACY_BUDGET_FRACTION,
            maxPartitionsContributed,
            maxContributionsPerPartition,
            noiseType);
    double sumVariance =
        getSumVariance(
            epsilon * SUM_PRIVACY_BUDGET_FRACTION,
            delta * SUM_PRIVACY_BUDGET_FRACTION,
            maxPartitionsContributed,
            maxContributionsPerPartition,
            distanceBetweenBounds,
            noiseType);
    double sumOfSquaresVariance =
        getSumVariance(
            epsilon * SUM_OF_SQUARES_PRIVACY_BUDGET_FRACTION,
            delta * SUM_OF_SQUARES_PRIVACY_BUDGET_FRACTION,
            maxPartitionsContributed,
            maxContributionsPerPartition,
            // The distance between the lower and upper bound of an entry of the sum of square is
            // (distanceBetweenBounds * 0.5)^2
            (distanceBetweenBounds * 0.5) * (distanceBetweenBounds * 0.5),
            noiseType);
    // Assume an empty set of entries.
    Supplier<Double> sampleGenerator =
        () ->
            computeNoisyStdv(
                /*noisyCount=*/ sampleReferenceNoise(countVariance, noiseType),
                /*noisySum=*/ sampleReferenceNoise(sumVariance, noiseType),
                /*noisySumOfSquares=*/ sampleReferenceNoise(sumOfSquaresVariance, noiseType));
    Supplier<Double> criticallyFailingSampleGenerator =
        () ->
            random.nextDouble() > dpTestParameters.getDeltaTolerance() * failureSpecificity
                ? sampleGenerator.get()
                : Double.NaN;

    assertThat(
            VotingUtil.runBallot(
                () ->
                    generateVote(
                        sampleGenerator,
                        criticallyFailingSampleGenerator,
                        samplingParameters.getNumberOfSamples(),
                        dpTestParameters.getEpsilon(),
                        dpTestParameters.getDelta(),
                        dpTestParameters.getDeltaTolerance(),
                        dpTestParameters.getGranularity()),
                numberOfVotes))
        .isFalse();
  }

  private static double computeNoisyStdv(
      double noisyCount, double noisySum, double noisySumOfSquares) {
    double correctedNoisyCount = max(1.0, noisyCount);
    double noisyVariance =
        (noisySumOfSquares / correctedNoisyCount)
            - ((noisySum * noisySum) / (correctedNoisyCount * correctedNoisyCount));
    return sqrt(max(0.0, noisyVariance));
  }

  private static BoundedStdvDpTestCaseCollection getTestCaseCollectionFromFile() {
    BoundedStdvDpTestCaseCollection.Builder testCaseCollectionBuilder =
        BoundedStdvDpTestCaseCollection.newBuilder();
    try {
      TextFormat.merge(
          new InputStreamReader(
              BoundedStdvDpTestCasesValidityTest.class
                  .getClassLoader()
                  .getResourceAsStream(TEST_CASES_FILE_PATH),
              UTF_8),
          testCaseCollectionBuilder);
    } catch (IOException e) {
      throw new RuntimeException("Unable to read input.", e);
    } catch (NullPointerException e) {
      throw new RuntimeException("Unable to find input file.", e);
    }
    return testCaseCollectionBuilder.build();
  }

  private static boolean generateVote(
      Supplier<Double> sampleGeneratorA,
      Supplier<Double> sampleGeneratorB,
      int numberOfSamples,
      double epsilon,
      double delta,
      double deltaTolerance,
      double granularity) {
    Double[] samplesA = new Double[numberOfSamples];
    Double[] samplesB = new Double[numberOfSamples];
    for (int i = 0; i < numberOfSamples; i++) {
      samplesA[i] = discretize(sampleGeneratorA.get(), granularity);
      samplesB[i] = discretize(sampleGeneratorB.get(), granularity);
    }
    return StatisticalTestsUtil.verifyApproximateDp(
        samplesA, samplesB, epsilon, delta, deltaTolerance);
  }

  private static double sampleReferenceNoise(double variance, NoiseType noiseType) {
    switch (noiseType) {
      case LAPLACE:
        return ReferenceNoiseUtil.sampleLaplace(/* mean= */ 0.0, variance);
      case GAUSSIAN:
        return ReferenceNoiseUtil.sampleGaussian(/* mean= */ 0.0, variance);
      default:
        return 0;
    }
  }

  private static double getCountSensitivity(
      int maxPartitionsContributed, int maxContributionsPerPartition, NoiseType noiseType) {
    int l0Sensitivity = maxPartitionsContributed;
    double lInfSensitivity = maxContributionsPerPartition;
    switch (noiseType) {
      case LAPLACE:
        return ReferenceNoiseUtil.getL1Sensitivity(l0Sensitivity, lInfSensitivity);
      case GAUSSIAN:
        return ReferenceNoiseUtil.getL2Sensitivity(l0Sensitivity, lInfSensitivity);
      default:
        return 0.0;
    }
  }

  private static double getSumSensitivity(
      int maxPartitionsContributed,
      int maxContributionsPerPartition,
      double distanceBetweenBounds,
      NoiseType noiseType) {
    int l0Sensitivity = maxPartitionsContributed;
    double lInfSensitivity = maxContributionsPerPartition * distanceBetweenBounds * 0.5;
    switch (noiseType) {
      case LAPLACE:
        return ReferenceNoiseUtil.getL1Sensitivity(l0Sensitivity, lInfSensitivity);
      case GAUSSIAN:
        return ReferenceNoiseUtil.getL2Sensitivity(l0Sensitivity, lInfSensitivity);
      default:
        return 0.0;
    }
  }

  private static double getCountVariance(
      double epsilon,
      double delta,
      int maxPartitionsContributed,
      int maxContributionsPerPartition,
      NoiseType noiseType) {
    switch (noiseType) {
      case LAPLACE:
        return ReferenceNoiseUtil.getLaplaceVariance(
            epsilon,
            getCountSensitivity(maxPartitionsContributed, maxContributionsPerPartition, LAPLACE));
      case GAUSSIAN:
        return ReferenceNoiseUtil.getGaussianVariance(
            epsilon,
            delta,
            getCountSensitivity(maxPartitionsContributed, maxContributionsPerPartition, GAUSSIAN));
      default:
        return 0.0;
    }
  }

  private static double getSumVariance(
      double epsilon,
      double delta,
      int maxPartitionsContributed,
      int maxContributionsPerPartition,
      double distanceBetweenBounds,
      NoiseType noiseType) {
    switch (noiseType) {
      case LAPLACE:
        return ReferenceNoiseUtil.getLaplaceVariance(
            epsilon,
            getSumSensitivity(
                maxPartitionsContributed,
                maxContributionsPerPartition,
                distanceBetweenBounds,
                LAPLACE));
      case GAUSSIAN:
        return ReferenceNoiseUtil.getGaussianVariance(
            epsilon,
            delta,
            getSumSensitivity(
                maxPartitionsContributed,
                maxContributionsPerPartition,
                distanceBetweenBounds,
                GAUSSIAN));
      default:
        return 0.0;
    }
  }

  private static Double discretize(Double sample, double granularity) {
    return Double.isNaN(sample) ? sample : StatisticalTestsUtil.discretize(sample, granularity);
  }
}

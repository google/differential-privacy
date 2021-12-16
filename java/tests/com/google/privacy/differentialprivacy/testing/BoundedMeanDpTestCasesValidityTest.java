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
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.Supplier;
import com.google.privacy.differentialprivacy.proto.testing.StatisticalTests.BoundedMeanDpTestCase;
import com.google.privacy.differentialprivacy.proto.testing.StatisticalTests.BoundedMeanDpTestCaseCollection;
import com.google.privacy.differentialprivacy.proto.testing.StatisticalTests.BoundedMeanSamplingParameters;
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
public final class BoundedMeanDpTestCasesValidityTest {
  private static final String TEST_CASES_FILE_PATH =

  "external/com_google_differential_privacy/proto/testing/bounded_mean_dp_test_cases.textproto";

  // The fractions of the privacy budget that are allocated to the count and sum aggregation
  // respectively. The two fractions should sum up to 1.0.
  private static final double COUNT_PRIVACY_BUDGET_FRACTION = 0.5;
  private static final double SUM_PRIVACY_BUDGET_FRACTION = 0.5;
  private final int numberOfVotes =
      getTestCaseCollectionFromFile().getVotingParameters().getNumberOfVotes();
  private final double distanceSpecificity =
      getTestCaseCollectionFromFile().getValidityParameters().getDistanceSpecificity();
  private final double failureSpecificity =
      getTestCaseCollectionFromFile().getValidityParameters().getFailureSpecificity();
  private final BoundedMeanDpTestCase testCase;

  public BoundedMeanDpTestCasesValidityTest(BoundedMeanDpTestCase testCase) {
    this.testCase = testCase;
  }

  @Parameterized.Parameters
  public static Iterable<?> testCases() {
    return getTestCaseCollectionFromFile().getBoundedMeanDpTestCaseList();
  }

  @Test
  public void boundedMeanDpTest_acceptsMeansDifferingByTheSensitivity() {
    BoundedMeanSamplingParameters samplingParameters = testCase.getBoundedMeanSamplingParameters();
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
    double sensitivity =
        getCountSensitivity(maxPartitionsContributed, maxContributionsPerPartition, noiseType);
    double normalizedBound = distanceBetweenBounds * 0.5;

    // To check that a DP test reliably accepts DP samples drawn from two neighbouring datasets
    // according to the specified privacy parameters, we construct the datasets in a way that
    // maximizes the difference in their mean. For this purpose, we assume that both datasets
    // contain 1 entry of value l where l is the normalized lower bound. Moreover, we assume that
    // the second dataset contains n entries of value u where n ≤ sensitivity and u is the
    // normalized upper bound.
    Supplier<Double> sampleGenerator =
        () ->
            (normalizedBound + sampleReferenceNoise(sumVariance, noiseType))
                / max(1 + sampleReferenceNoise(countVariance, noiseType), 1.0);
    Supplier<Double> neighbouringSampleGenerator =
        () ->
            ((1 - sensitivity) * normalizedBound + sampleReferenceNoise(sumVariance, noiseType))
                / max(1 + sensitivity + sampleReferenceNoise(countVariance, noiseType), 1.0);

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
  public void boundedMeanDpTest_rejectsMeansDifferingByMoreThanTheSensitivity() {
    BoundedMeanSamplingParameters samplingParameters = testCase.getBoundedMeanSamplingParameters();
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
    double scaledSensitivity =
        getCountSensitivity(maxPartitionsContributed, maxContributionsPerPartition, noiseType)
            * distanceSpecificity;
    double normalizedBound = distanceBetweenBounds * 0.5;

    // To check that a DP test rejects samples drawn from two neighbouring datasets that violate the
    // specified sensitivity, we construct the datasets in a way that maximizes the difference in
    // standard deviation for the given sensitivity violation. For this purpose, we assume that both
    // datasets contain 1 entry of value l where l is the normalized lower bound. Moreover, we
    // assume that the second dataset contains n entries of value u where n ≥
    // sensitivity * distanceSpecificity and u is the normalized upper bound.
    Supplier<Double> sampleGenerator =
        () ->
            (normalizedBound + sampleReferenceNoise(sumVariance, noiseType))
                / max(1 + sampleReferenceNoise(countVariance, noiseType), 1.0);
    Supplier<Double> neighbouringSampleGenerator =
        () ->
            ((1 - scaledSensitivity) * normalizedBound
                    + sampleReferenceNoise(sumVariance, noiseType))
                / max(1 + scaledSensitivity + sampleReferenceNoise(countVariance, noiseType), 1.0);

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
  public void boundedMeanDpTest_rejectsCriticallyFailingMeans() {
    BoundedMeanSamplingParameters samplingParameters = testCase.getBoundedMeanSamplingParameters();
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
    // Assume an empty set of entries.
    Supplier<Double> sampleGenerator =
        () ->
            sampleReferenceNoise(sumVariance, noiseType)
                / max(sampleReferenceNoise(countVariance, noiseType), 1.0);
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

  private static BoundedMeanDpTestCaseCollection getTestCaseCollectionFromFile() {
    BoundedMeanDpTestCaseCollection.Builder testCaseCollectionBuilder =
        BoundedMeanDpTestCaseCollection.newBuilder();
    try {
      TextFormat.merge(
          new InputStreamReader(
              BoundedMeanDpTestCasesValidityTest.class
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

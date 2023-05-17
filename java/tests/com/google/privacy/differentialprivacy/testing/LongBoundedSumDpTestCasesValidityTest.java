//
// Copyright 2023 Google LLC
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
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.Supplier;
import com.google.privacy.differentialprivacy.proto.testing.StatisticalTests.DpTestParameters;
import com.google.privacy.differentialprivacy.proto.testing.StatisticalTests.LongBoundedSumDpTestCase;
import com.google.privacy.differentialprivacy.proto.testing.StatisticalTests.LongBoundedSumDpTestCaseCollection;
import com.google.privacy.differentialprivacy.proto.testing.StatisticalTests.LongBoundedSumSamplingParameters;
import com.google.privacy.differentialprivacy.proto.testing.StatisticalTests.NoiseType;
import com.google.protobuf.TextFormat;
import java.io.IOException;
import java.io.InputStreamReader;
import java.security.SecureRandom;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(Parameterized.class)
public final class LongBoundedSumDpTestCasesValidityTest {
   private static final String TEST_CASES_FILE_PATH =

  "external/com_google_differential_privacy/proto/testing/long_bounded_sum_dp_test_cases.textproto";
  private final int numberOfVotes =
      getTestCaseCollectionFromFile().getVotingParameters().getNumberOfVotes();
  private final double distanceSpecificity =
      getTestCaseCollectionFromFile().getValidityParameters().getDistanceSpecificity();
  private final double failureSpecificity =
      getTestCaseCollectionFromFile().getValidityParameters().getFailureSpecificity();
  private final LongBoundedSumDpTestCase testCase;

  public LongBoundedSumDpTestCasesValidityTest(LongBoundedSumDpTestCase testCase) {
    this.testCase = testCase;
  }

  @Parameterized.Parameters
  public static Iterable<? extends Object> testCases() {
    return getTestCaseCollectionFromFile().getLongBoundedSumDpTestCaseList();
  }

  @Test
  public void boundedSumDpTest_acceptsSumsDifferingByTheSensitivity() {
    LongBoundedSumSamplingParameters samplingParameters =
        testCase.getLongBoundedSumSamplingParameters();
    DpTestParameters dpTestParameters = testCase.getDpTestParameters();
    double epsilon = samplingParameters.getEpsilon();
    double delta = samplingParameters.getDelta();
    int maxPartitionsContributed = samplingParameters.getMaxPartitionsContributed();
    double maxAbsBound =
        Math.max(
            Math.abs(samplingParameters.getLowerBound()),
            Math.abs(samplingParameters.getUpperBound()));
    NoiseType noiseType = samplingParameters.getNoiseType();

    double variance = getVariance(epsilon, delta, maxPartitionsContributed, maxAbsBound, noiseType);
    Supplier<Double> sampleGenerator = () -> sampleReferenceNoise(variance, noiseType);
    Supplier<Double> shiftedSampleGenerator =
        () ->
            sampleReferenceNoise(variance, noiseType)
                + getSensitivity(maxPartitionsContributed, maxAbsBound, noiseType);

    assertThat(
            VotingUtil.runBallot(
                () ->
                    generateVote(
                        sampleGenerator,
                        shiftedSampleGenerator,
                        samplingParameters.getNumberOfSamples(),
                        dpTestParameters.getEpsilon(),
                        dpTestParameters.getDelta(),
                        dpTestParameters.getDeltaTolerance(),
                        dpTestParameters.getGranularity()),
                numberOfVotes))
        .isTrue();
  }

  @Test
  public void boundedSumDpTest_rejectsSumsDifferingByMoreThanTheSensitivity() {
    LongBoundedSumSamplingParameters samplingParameters =
        testCase.getLongBoundedSumSamplingParameters();
    DpTestParameters dpTestParameters = testCase.getDpTestParameters();
    double epsilon = samplingParameters.getEpsilon();
    double delta = samplingParameters.getDelta();
    int maxPartitionsContributed = samplingParameters.getMaxPartitionsContributed();
    double maxAbsBound =
        Math.max(
            Math.abs(samplingParameters.getLowerBound()),
            Math.abs(samplingParameters.getUpperBound()));
    NoiseType noiseType = samplingParameters.getNoiseType();

    double variance = getVariance(epsilon, delta, maxPartitionsContributed, maxAbsBound, noiseType);
    Supplier<Double> sampleGenerator = () -> sampleReferenceNoise(variance, noiseType);
    Supplier<Double> shiftedSampleGenerator =
        () ->
            sampleReferenceNoise(variance, noiseType)
                + distanceSpecificity
                    * getSensitivity(maxPartitionsContributed, maxAbsBound, noiseType);

    assertThat(
            VotingUtil.runBallot(
                () ->
                    generateVote(
                        sampleGenerator,
                        shiftedSampleGenerator,
                        samplingParameters.getNumberOfSamples(),
                        dpTestParameters.getEpsilon(),
                        dpTestParameters.getDelta(),
                        dpTestParameters.getDeltaTolerance(),
                        dpTestParameters.getGranularity()),
                numberOfVotes))
        .isFalse();
  }

  @Test
  public void boundedSumDpTest_rejectsCriticallyFailingSums() {
    LongBoundedSumSamplingParameters samplingParameters =
        testCase.getLongBoundedSumSamplingParameters();
    DpTestParameters dpTestParameters = testCase.getDpTestParameters();
    SecureRandom random = new SecureRandom();
    double epsilon = samplingParameters.getEpsilon();
    double delta = samplingParameters.getDelta();
    int maxPartitionsContributed = samplingParameters.getMaxPartitionsContributed();
    double maxAbsBound =
        Math.max(
            Math.abs(samplingParameters.getLowerBound()),
            Math.abs(samplingParameters.getUpperBound()));
    NoiseType noiseType = samplingParameters.getNoiseType();

    double variance = getVariance(epsilon, delta, maxPartitionsContributed, maxAbsBound, noiseType);
    Supplier<Double> sampleGenerator = () -> sampleReferenceNoise(variance, noiseType);
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

  private static LongBoundedSumDpTestCaseCollection getTestCaseCollectionFromFile() {
    LongBoundedSumDpTestCaseCollection.Builder testCaseCollectionBuilder =
        LongBoundedSumDpTestCaseCollection.newBuilder();
    try {
      TextFormat.merge(
          new InputStreamReader(
              LongBoundedSumDpTestCasesValidityTest.class
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

  private static double getSensitivity(
      int maxPartitionsContributed, double maxAbsBound, NoiseType noiseType) {
    int l0Sensitivity = maxPartitionsContributed;
    double lInfSensitivity = maxAbsBound;
    switch (noiseType) {
      case LAPLACE:
        return ReferenceNoiseUtil.getL1Sensitivity(l0Sensitivity, lInfSensitivity);
      case GAUSSIAN:
        return ReferenceNoiseUtil.getL2Sensitivity(l0Sensitivity, lInfSensitivity);
      default:
        return 0.0;
    }
  }

  private static double getVariance(
      double epsilon,
      double delta,
      int maxPartitionsContributed,
      double maxAbsBound,
      NoiseType noiseType) {
    switch (noiseType) {
      case LAPLACE:
        return ReferenceNoiseUtil.getLaplaceVariance(
            epsilon, getSensitivity(maxPartitionsContributed, maxAbsBound, LAPLACE));
      case GAUSSIAN:
        return ReferenceNoiseUtil.getGaussianVariance(
            epsilon, delta, getSensitivity(maxPartitionsContributed, maxAbsBound, GAUSSIAN));
      default:
        return 0.0;
    }
  }

  private static Double discretize(Double sample, double granularity) {
    return Double.isNaN(sample) ? sample : StatisticalTestsUtil.discretize(sample, granularity);
  }
}

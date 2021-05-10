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
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.Supplier;
import com.google.privacy.differentialprivacy.proto.testing.StatisticalTests.CountDpTestCase;
import com.google.privacy.differentialprivacy.proto.testing.StatisticalTests.CountDpTestCaseCollection;
import com.google.privacy.differentialprivacy.proto.testing.StatisticalTests.CountSamplingParameters;
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
public final class CountDpTestCasesValidityTest {
  private static final String TEST_CASES_FILE_PATH =
      "external/com_google_differential_privacy/proto/testing/count_dp_test_cases.textproto";
  private final int numberOfVotes =
      getTestCaseCollectionFromFile().getVotingParameters().getNumberOfVotes();
  private final double distanceSpecificity =
      getTestCaseCollectionFromFile().getValidityParameters().getDistanceSpecificity();
  private final double failureSpecificity =
      getTestCaseCollectionFromFile().getValidityParameters().getFailureSpecificity();
  private final CountDpTestCase testCase;

  public CountDpTestCasesValidityTest(CountDpTestCase testCase) {
    this.testCase = testCase;
  }

  @Parameterized.Parameters
  public static Iterable<? extends Object> testCases() {
    return getTestCaseCollectionFromFile().getCountDpTestCaseList();
  }

  @Test
  public void countDpTest_acceptsCountsDifferingByTheSensitivity() {
    CountSamplingParameters samplingParameters = testCase.getCountSamplingParameters();
    DpTestParameters dpTestParameters = testCase.getDpTestParameters();
    double epsilon = samplingParameters.getEpsilon();
    double delta = samplingParameters.getDelta();
    int maxPartitionsContributed = samplingParameters.getMaxPartitionsContributed();
    NoiseType noiseType = samplingParameters.getNoiseType();

    double variance = getVariance(epsilon, delta, maxPartitionsContributed, noiseType);
    Supplier<Long> sampleGenerator = () -> sampleReferenceNoise(variance, noiseType);
    Supplier<Long> shiftedSampleGenerator =
        () ->
            sampleReferenceNoise(variance, noiseType)
                + (long) Math.floor(getSensitivity(maxPartitionsContributed, noiseType));

    assertThat(
            VotingUtil.runBallot(
                () ->
                    generateVote(
                        sampleGenerator,
                        shiftedSampleGenerator,
                        samplingParameters.getNumberOfSamples(),
                        dpTestParameters.getEpsilon(),
                        dpTestParameters.getDelta(),
                        dpTestParameters.getDeltaTolerance()),
                numberOfVotes))
        .isTrue();
  }

  @Test
  public void countDpTest_rejectsCountsDifferingByMoreThanTheSensitivity() {
    CountSamplingParameters samplingParameters = testCase.getCountSamplingParameters();
    DpTestParameters dpTestParameters = testCase.getDpTestParameters();
    double epsilon = samplingParameters.getEpsilon();
    double delta = samplingParameters.getDelta();
    int maxPartitionsContributed = samplingParameters.getMaxPartitionsContributed();
    NoiseType noiseType = samplingParameters.getNoiseType();

    double variance = getVariance(epsilon, delta, maxPartitionsContributed, noiseType);
    Supplier<Long> sampleGenerator = () -> sampleReferenceNoise(variance, noiseType);
    Supplier<Long> shiftedSampleGenerator =
        () ->
            sampleReferenceNoise(variance, noiseType)
                + (long)
                    Math.ceil(
                        distanceSpecificity * getSensitivity(maxPartitionsContributed, noiseType));

    assertThat(
            VotingUtil.runBallot(
                () ->
                    generateVote(
                        sampleGenerator,
                        shiftedSampleGenerator,
                        samplingParameters.getNumberOfSamples(),
                        dpTestParameters.getEpsilon(),
                        dpTestParameters.getDelta(),
                        dpTestParameters.getDeltaTolerance()),
                numberOfVotes))
        .isFalse();
  }

  @Test
  public void countDpTest_rejectsCriticallyFailingCounts() {
    CountSamplingParameters samplingParameters = testCase.getCountSamplingParameters();
    DpTestParameters dpTestParameters = testCase.getDpTestParameters();
    SecureRandom random = new SecureRandom();
    double epsilon = samplingParameters.getEpsilon();
    double delta = samplingParameters.getDelta();
    int maxPartitionsContributed = samplingParameters.getMaxPartitionsContributed();
    NoiseType noiseType = samplingParameters.getNoiseType();

    double variance = getVariance(epsilon, delta, maxPartitionsContributed, noiseType);
    Supplier<Long> sampleGenerator = () -> sampleReferenceNoise(variance, noiseType);
    Supplier<Long> criticallyFailingSampleGenerator =
        () ->
            random.nextDouble() > dpTestParameters.getDeltaTolerance() * failureSpecificity
                ? sampleGenerator.get()
                : Long.MIN_VALUE;

    assertThat(
            VotingUtil.runBallot(
                () ->
                    generateVote(
                        sampleGenerator,
                        criticallyFailingSampleGenerator,
                        samplingParameters.getNumberOfSamples(),
                        dpTestParameters.getEpsilon(),
                        dpTestParameters.getDelta(),
                        dpTestParameters.getDeltaTolerance()),
                numberOfVotes))
        .isFalse();
  }

  private static CountDpTestCaseCollection getTestCaseCollectionFromFile() {
    CountDpTestCaseCollection.Builder testCaseCollectionBuilder =
        CountDpTestCaseCollection.newBuilder();
    try {
      TextFormat.merge(
          new InputStreamReader(
              CountDpTestCasesValidityTest.class
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
      Supplier<Long> sampleGeneratorA,
      Supplier<Long> sampleGeneratorB,
      int numberOfSamples,
      double epsilon,
      double delta,
      double deltaTolerance) {
    Long[] samplesA = new Long[numberOfSamples];
    Long[] samplesB = new Long[numberOfSamples];
    for (int i = 0; i < numberOfSamples; i++) {
      samplesA[i] = sampleGeneratorA.get();
      samplesB[i] = sampleGeneratorB.get();
    }
    return StatisticalTestsUtil.verifyApproximateDp(
        samplesA, samplesB, epsilon, delta, deltaTolerance);
  }

  private static long sampleReferenceNoise(double variance, NoiseType noiseType) {
    switch (noiseType) {
      case LAPLACE:
        return Math.round(ReferenceNoiseUtil.sampleLaplace(/* mean= */ 0.0, variance));
      case GAUSSIAN:
        return Math.round(ReferenceNoiseUtil.sampleGaussian(/* mean= */ 0.0, variance));
      default:
        return 0;
    }
  }

  private static double getSensitivity(int maxPartitionsContributed, NoiseType noiseType) {
    int l0Sensitivity = maxPartitionsContributed;
    double lInfSensitivity = 1.0;
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
      double epsilon, double delta, int maxPartitionsContributed, NoiseType noiseType) {
    switch (noiseType) {
      case LAPLACE:
        return ReferenceNoiseUtil.getLaplaceVariance(
            epsilon, getSensitivity(maxPartitionsContributed, LAPLACE));
      case GAUSSIAN:
        return ReferenceNoiseUtil.getGaussianVariance(
            epsilon, delta, getSensitivity(maxPartitionsContributed, GAUSSIAN));
      default:
        return 0.0;
    }
  }
}

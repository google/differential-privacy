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
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.Supplier;
import com.google.differentialprivacy.testing.StatisticalTests.ClosenessTestParameters;
import com.google.differentialprivacy.testing.StatisticalTests.DistributionClosenessTestCase;
import com.google.differentialprivacy.testing.StatisticalTests.DistributionClosenessTestCaseCollection;
import com.google.protobuf.TextFormat;
import java.io.IOException;
import java.io.InputStreamReader;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;


@RunWith(Parameterized.class)
public final class LaplaceClosenessTestCasesValidityTest {
  private static final String TEST_CASES_FILE_PATH =
     "external/com_google_differential_privacy/proto/testing/laplace_closeness_test_cases.textproto";
  private final int numberOfVotes =
      getTestCaseCollectionFromFile().getVotingParameters().getNumberOfVotes();
  private final double shiftSpecificity =
      getTestCaseCollectionFromFile().getValidityParameters().getShiftSpecificity();
  private final double scaleSpecificity =
      getTestCaseCollectionFromFile().getValidityParameters().getScaleSpecificity();
  private final DistributionClosenessTestCase testCase;

  public LaplaceClosenessTestCasesValidityTest(DistributionClosenessTestCase testCase) {
    this.testCase = testCase;
  }

  @Parameterized.Parameters
  public static Iterable<? extends Object> testCases() {
    return getTestCaseCollectionFromFile().getDistributionClosenessTestCaseList();
  }

  @Test
  public void closenessTest_acceptsIdenticalLaplaceDistributions() {
    ClosenessTestParameters closenessTestParameters = testCase.getClosenessTestParameters();
    double mean = closenessTestParameters.getMean();
    double variance = closenessTestParameters.getVariance();

    Supplier<Double> laplaceSampleGenerator =
        () -> ReferenceNoiseUtil.sampleLaplace(mean, variance);

    assertThat(
            VotingUtil.runBallot(
                () ->
                    generateVote(
                        laplaceSampleGenerator,
                        laplaceSampleGenerator,
                        testCase.getNoiseSamplingParameters().getNumberOfSamples(),
                        closenessTestParameters.getL2Tolerance(),
                        closenessTestParameters.getGranularity()),
                numberOfVotes))
        .isTrue();
  }

  @Test
  public void closenessTest_rejectsShiftedLaplaceDistributions() {
    ClosenessTestParameters closenessTestParameters = testCase.getClosenessTestParameters();
    double mean = closenessTestParameters.getMean();
    double variance = closenessTestParameters.getVariance();

    Supplier<Double> laplaceSampleGenerator =
        () -> ReferenceNoiseUtil.sampleLaplace(mean, variance);
    Supplier<Double> shiftedLaplaceSampleGenerator =
        () ->
            ReferenceNoiseUtil.sampleLaplace(
                mean + Math.sqrt(variance) * shiftSpecificity, variance);

    assertThat(
            VotingUtil.runBallot(
                () ->
                    generateVote(
                        laplaceSampleGenerator,
                        shiftedLaplaceSampleGenerator,
                        testCase.getNoiseSamplingParameters().getNumberOfSamples(),
                        closenessTestParameters.getL2Tolerance(),
                        closenessTestParameters.getGranularity()),
                numberOfVotes))
        .isFalse();
  }

  @Test
  public void closenessTest_rejectsScaledLaplaceDistributions() {
    ClosenessTestParameters closenessTestParameters = testCase.getClosenessTestParameters();
    double mean = closenessTestParameters.getMean();
    double variance = closenessTestParameters.getVariance();

    Supplier<Double> laplaceSampleGenerator =
        () -> ReferenceNoiseUtil.sampleLaplace(mean, variance);
    Supplier<Double> scaledLaplaceSampleGenerator =
        () -> ReferenceNoiseUtil.sampleLaplace(mean, variance * scaleSpecificity);

    assertThat(
            VotingUtil.runBallot(
                () ->
                    generateVote(
                        laplaceSampleGenerator,
                        scaledLaplaceSampleGenerator,
                        testCase.getNoiseSamplingParameters().getNumberOfSamples(),
                        closenessTestParameters.getL2Tolerance(),
                        closenessTestParameters.getGranularity()),
                numberOfVotes))
        .isFalse();
  }

  @Test
  public void closenessTest_rejectsDifferentTypesOfDistributions() {
    ClosenessTestParameters closenessTestParameters = testCase.getClosenessTestParameters();
    double mean = closenessTestParameters.getMean();
    double variance = closenessTestParameters.getVariance();

    Supplier<Double> laplaceSampleGenerator =
        () -> ReferenceNoiseUtil.sampleLaplace(mean, variance);
    Supplier<Double> gaussianSampleGenerator =
        () -> ReferenceNoiseUtil.sampleGaussian(mean, variance);

    assertThat(
            VotingUtil.runBallot(
                () ->
                    generateVote(
                        laplaceSampleGenerator,
                        gaussianSampleGenerator,
                        testCase.getNoiseSamplingParameters().getNumberOfSamples(),
                        closenessTestParameters.getL2Tolerance(),
                        closenessTestParameters.getGranularity()),
                numberOfVotes))
        .isFalse();
  }

  private static DistributionClosenessTestCaseCollection getTestCaseCollectionFromFile() {
    DistributionClosenessTestCaseCollection.Builder testCaseCollectionBuilder =
        DistributionClosenessTestCaseCollection.newBuilder();
    try {
      TextFormat.merge(
          new InputStreamReader(
              LaplaceClosenessTestCasesValidityTest.class.getClassLoader().getResourceAsStream(TEST_CASES_FILE_PATH), UTF_8),
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
      double l2Tolerance,
      double granularity) {
    Double[] samplesA = new Double[numberOfSamples];
    Double[] samplesB = new Double[numberOfSamples];
    for (int i = 0; i < numberOfSamples; i++) {
      samplesA[i] = StatisticalTestsUtil.discretize(sampleGeneratorA.get(), granularity);
      samplesB[i] = StatisticalTestsUtil.discretize(sampleGeneratorB.get(), granularity);
    }
    return StatisticalTestsUtil.verifyCloseness(samplesA, samplesB, l2Tolerance);
  }
}

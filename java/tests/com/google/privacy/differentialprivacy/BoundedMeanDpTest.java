//
// Copyright 2019 Google LLC
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

import static com.google.common.truth.Truth.assertThat;
import static com.google.differentialprivacy.testing.StatisticalTests.NoiseType.GAUSSIAN;
import static com.google.differentialprivacy.testing.StatisticalTests.NoiseType.LAPLACE;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.Supplier;
import com.google.differentialprivacy.testing.StatisticalTests.BoundedMeanDpTestCase;
import com.google.differentialprivacy.testing.StatisticalTests.BoundedMeanDpTestCaseCollection;
import com.google.differentialprivacy.testing.StatisticalTests.BoundedMeanSamplingParameters;
import com.google.differentialprivacy.testing.StatisticalTests.DpTestParameters;
import com.google.privacy.differentialprivacy.testing.StatisticalTestsUtil;
import com.google.privacy.differentialprivacy.testing.VotingUtil;
import com.google.protobuf.TextFormat;
import java.io.IOException;
import java.io.InputStreamReader;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

/**
 * Tests that {@link BoundedMean} conforms to the specified privacy parameters epsilon and delta.
 */
@RunWith(Parameterized.class)
public final class BoundedMeanDpTest {
  private static final String TEST_CASES_FILE_PATH =

  "external/com_google_differential_privacy/proto/testing/bounded_mean_dp_test_cases.textproto";

  private final BoundedMeanDpTestCase testCase;

  public BoundedMeanDpTest(BoundedMeanDpTestCase testCase) {
    this.testCase = testCase;
  }

  @Parameterized.Parameters
  public static Iterable<? extends Object> testCases() {
    return getTestCaseCollectionFromFile().getBoundedMeanDpTestCaseList();
  }

  @Test
  public void boundedMeanDpTest() {

    BoundedMeanSamplingParameters samplingParameters = testCase.getBoundedMeanSamplingParameters();
    DpTestParameters dpTestParameters = testCase.getDpTestParameters();

    Noise noise;
    Double delta;
    switch (samplingParameters.getNoiseType()) {
      case LAPLACE:
        noise = new LaplaceNoise();
        delta = null;
        break;
      case GAUSSIAN:
        noise = new GaussianNoise();
        delta = samplingParameters.getDelta();
        break;
      default:
        throw new IllegalArgumentException(
            "Noise type " + samplingParameters.getNoiseType() + " is not supported");
    }

    BoundedMean.Params.Builder boundedMeanBuilder =
        BoundedMean.builder()
            .epsilon(samplingParameters.getEpsilon())
            .delta(delta)
            .maxPartitionsContributed(samplingParameters.getMaxPartitionsContributed())
            .maxContributionsPerPartition(samplingParameters.getMaxContributionsPerPartition())
            .lower(samplingParameters.getLowerBound())
            .upper(samplingParameters.getUpperBound())
            .noise(noise);

    Supplier<Double> boundedMeanGenerator =
        () -> {
          BoundedMean boundedMean = boundedMeanBuilder.build();
          for (double entry : samplingParameters.getRawEntryList()) {
            boundedMean.addEntry(entry);
          }
          return boundedMean.computeResult();
        };
    Supplier<Double> neighbourBoundedMeanGenerator =
        () -> {
          BoundedMean boundedMean = boundedMeanBuilder.build();
          for (double entry : samplingParameters.getNeighbourRawEntryList()) {
            boundedMean.addEntry(entry);
          }
          return boundedMean.computeResult();
        };

    assertThat(
            VotingUtil.runBallot(
                () ->
                    generateVote(
                        boundedMeanGenerator,
                        neighbourBoundedMeanGenerator,
                        samplingParameters.getNumberOfSamples(),
                        dpTestParameters.getEpsilon(),
                        dpTestParameters.getDelta(),
                        dpTestParameters.getDeltaTolerance(),
                        dpTestParameters.getGranularity()),
                getNumberOfVotesFromFile()))
        .isTrue();
  }

  private int getNumberOfVotesFromFile() {
    return getTestCaseCollectionFromFile().getVotingParameters().getNumberOfVotes();
  }

  private static BoundedMeanDpTestCaseCollection getTestCaseCollectionFromFile() {
    BoundedMeanDpTestCaseCollection.Builder testCaseCollectionBuilder =
        BoundedMeanDpTestCaseCollection.newBuilder();
    try {
      TextFormat.merge(
          new InputStreamReader(
              BoundedMeanDpTest.class.getClassLoader().getResourceAsStream(TEST_CASES_FILE_PATH),
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
      double l2Tolerance,
      double granularity) {
    Double[] samplesA = new Double[numberOfSamples];
    Double[] samplesB = new Double[numberOfSamples];
    for (int i = 0; i < numberOfSamples; i++) {
      samplesA[i] = StatisticalTestsUtil.discretize(sampleGeneratorA.get(), granularity);
      samplesB[i] = StatisticalTestsUtil.discretize(sampleGeneratorB.get(), granularity);
    }
    return StatisticalTestsUtil.verifyApproximateDp(
        samplesA, samplesB, epsilon, delta, l2Tolerance);
  }
}

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

package com.google.privacy.differentialprivacy.statistical;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.differentialprivacy.testing.StatisticalTests.BoundedSumDpTestCase;
import com.google.differentialprivacy.testing.StatisticalTests.BoundedSumDpTestCaseCollection;
import com.google.differentialprivacy.testing.StatisticalTests.BoundedSumSamplingParameters;
import com.google.differentialprivacy.testing.StatisticalTests.DpTestParameters;
import com.google.privacy.differentialprivacy.BoundedSum;
import com.google.privacy.differentialprivacy.GaussianNoise;
import com.google.privacy.differentialprivacy.LaplaceNoise;
import com.google.privacy.differentialprivacy.Noise;
import com.google.privacy.differentialprivacy.testing.StatisticalTestsUtil;
import com.google.privacy.differentialprivacy.testing.VotingUtil;
import com.google.protobuf.TextFormat;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.function.Supplier;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

/** Tests that {@link BoundedSum} conforms to the specified privacy parameters epsilon and delta. */
@RunWith(Parameterized.class)
public final class BoundedSumDpTest {
  private static final String TEST_CASES_FILE_PATH =
      "external/com_google_differential_privacy/proto/testing/bounded_sum_dp_test_cases.textproto";

  private final BoundedSumDpTestCase testCase;

  public BoundedSumDpTest(BoundedSumDpTestCase testCase) {
    this.testCase = testCase;
  }

  @Parameterized.Parameters
  public static Iterable<?> testcases() {
    return getTestCaseCollectionFromFile().getBoundedSumDpTestCaseList();
  }

  @Test
  public void boundedSumDpTest() {

    BoundedSumSamplingParameters samplingParameters = testCase.getBoundedSumSamplingParameters();
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

    BoundedSum.Params.Builder boundedSumBuilder =
        BoundedSum.builder()
            .epsilon(samplingParameters.getEpsilon())
            .delta(delta)
            .maxPartitionsContributed(samplingParameters.getMaxPartitionsContributed())
            .lower(samplingParameters.getLowerBound())
            .upper(samplingParameters.getUpperBound())
            .noise(noise);

    Supplier<Double> boundedSumGenerator =
        () -> {
          BoundedSum boundedSum = boundedSumBuilder.build();
          for (double entry : samplingParameters.getRawEntryList()) {
            boundedSum.addEntry(entry);
          }
          return boundedSum.computeResult();
        };
    Supplier<Double> neighbourBoundedSumGenerator =
        () -> {
          BoundedSum boundedSum = boundedSumBuilder.build();
          for (double entry : samplingParameters.getNeighbourRawEntryList()) {
            boundedSum.addEntry(entry);
          }
          return boundedSum.computeResult();
        };

    assertThat(
            VotingUtil.runBallot(
                () ->
                    generateVote(
                        boundedSumGenerator,
                        neighbourBoundedSumGenerator,
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

  private static BoundedSumDpTestCaseCollection getTestCaseCollectionFromFile() {
    BoundedSumDpTestCaseCollection.Builder testCaseCollectionBuilder =
        BoundedSumDpTestCaseCollection.newBuilder();
    try {
      TextFormat.merge(
          new InputStreamReader(
              BoundedSumDpTest.class.getClassLoader().getResourceAsStream(TEST_CASES_FILE_PATH), UTF_8),
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

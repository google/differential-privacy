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

import com.google.privacy.differentialprivacy.Count;
import com.google.privacy.differentialprivacy.Noise;
import com.google.privacy.differentialprivacy.TestNoiseFactory;
import com.google.privacy.differentialprivacy.proto.testing.StatisticalTests.CountDpTestCase;
import com.google.privacy.differentialprivacy.proto.testing.StatisticalTests.CountDpTestCaseCollection;
import com.google.privacy.differentialprivacy.proto.testing.StatisticalTests.CountSamplingParameters;
import com.google.privacy.differentialprivacy.proto.testing.StatisticalTests.DpTestParameters;
import com.google.privacy.differentialprivacy.testing.StatisticalTestsUtil;
import com.google.privacy.differentialprivacy.testing.VotingUtil;
import com.google.protobuf.TextFormat;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Random;
import java.util.function.Supplier;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

/** Tests that {@link Count} conforms to the specified privacy parameters epsilon and delta. */
@RunWith(Parameterized.class)
public final class CountDpTest {
  private static final String TEST_CASES_FILE_PATH =
      "external/com_google_differential_privacy/proto/testing/count_dp_test_cases.textproto";

  private final CountDpTestCase testCase;

  public CountDpTest(CountDpTestCase testCase) {
    this.testCase = testCase;
  }

  @Parameterized.Parameters
  public static Iterable<?> testCases() {
    return getTestCaseCollectionFromFile().getCountDpTestCaseList();
  }

  @Test
  public void countDpTest() {

    CountSamplingParameters samplingParameters = testCase.getCountSamplingParameters();
    DpTestParameters dpTestParameters = testCase.getDpTestParameters();

    Noise noise;
    double delta;
    switch (samplingParameters.getNoiseType()) {
      case LAPLACE:
        noise = TestNoiseFactory.createLaplaceNoise(new Random());
        delta = 0.0;
        break;
      case GAUSSIAN:
        noise = TestNoiseFactory.createGaussianNoise(new Random());
        delta = samplingParameters.getDelta();
        break;
      default:
        throw new IllegalArgumentException(
            "Noise type " + samplingParameters.getNoiseType() + " is not supported");
    }

    Count.Params.Builder countBuilder =
        Count.builder()
            .epsilon(samplingParameters.getEpsilon())
            .delta(delta)
            .maxPartitionsContributed(samplingParameters.getMaxPartitionsContributed())
            .noise(noise);

    Supplier<Long> countGenerator =
        () -> {
          Count count = countBuilder.build();
          for (long increment : samplingParameters.getRawIncrementByList()) {
            count.incrementBy((int) increment);
          }
          return count.computeResult();
        };
    Supplier<Long> neighbourCountGenerator =
        () -> {
          Count count = countBuilder.build();
          for (long increment : samplingParameters.getNeighbourRawIncrementByList()) {
            count.incrementBy((int) increment);
          }
          return count.computeResult();
        };

    assertThat(
            VotingUtil.runBallot(
                () ->
                    generateVote(
                        countGenerator,
                        neighbourCountGenerator,
                        samplingParameters.getNumberOfSamples(),
                        dpTestParameters.getEpsilon(),
                        dpTestParameters.getDelta(),
                        dpTestParameters.getDeltaTolerance()),
                getNumberOfVotesFromFile()))
        .isTrue();
  }

  private static int getNumberOfVotesFromFile() {
    return getTestCaseCollectionFromFile().getVotingParameters().getNumberOfVotes();
  }

  private static CountDpTestCaseCollection getTestCaseCollectionFromFile() {
    CountDpTestCaseCollection.Builder testCaseCollectionBuilder =
        CountDpTestCaseCollection.newBuilder();
    try {
      TextFormat.merge(
          new InputStreamReader(
              CountDpTest.class.getClassLoader().getResourceAsStream(TEST_CASES_FILE_PATH), UTF_8),
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
}

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

import com.google.privacy.differentialprivacy.LongBoundedSum;
import com.google.privacy.differentialprivacy.Noise;
import com.google.privacy.differentialprivacy.TestNoiseFactory;
import com.google.privacy.differentialprivacy.proto.testing.StatisticalTests.DpTestParameters;
import com.google.privacy.differentialprivacy.proto.testing.StatisticalTests.LongBoundedSumDpTestCase;
import com.google.privacy.differentialprivacy.proto.testing.StatisticalTests.LongBoundedSumDpTestCaseCollection;
import com.google.privacy.differentialprivacy.proto.testing.StatisticalTests.LongBoundedSumSamplingParameters;
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

/**
 * Tests that {@link LongBoundedSum} conforms to the specified privacy parameters epsilon and delta.
 */
@RunWith(Parameterized.class)
public final class LongBoundedSumDpTest {
  private static final String TEST_CASES_FILE_PATH =

  "external/com_google_differential_privacy/proto/testing/long_bounded_sum_dp_test_cases.textproto";

  private final LongBoundedSumDpTestCase testCase;

  public LongBoundedSumDpTest(LongBoundedSumDpTestCase testCase) {
    this.testCase = testCase;
  }

  @Parameterized.Parameters
  public static Iterable<?> testcases() {
    return getTestCaseCollectionFromFile().getLongBoundedSumDpTestCaseList();
  }

  @Test
  public void longBoundedSumDpTest() {

    LongBoundedSumSamplingParameters samplingParameters =
        testCase.getLongBoundedSumSamplingParameters();
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

    LongBoundedSum.Params.Builder longBoundedSumBuilder =
        LongBoundedSum.builder()
            .epsilon(samplingParameters.getEpsilon())
            .delta(delta)
            .maxPartitionsContributed(samplingParameters.getMaxPartitionsContributed())
            .lower(samplingParameters.getLowerBound())
            .upper(samplingParameters.getUpperBound())
            .noise(noise);

    Supplier<Long> longBoundedSumGenerator =
        () -> {
          LongBoundedSum longBoundedSum = longBoundedSumBuilder.build();
          for (long entry : samplingParameters.getRawEntryList()) {
            longBoundedSum.addEntry(entry);
          }
          return longBoundedSum.computeResult();
        };
    Supplier<Long> neighbourLongBoundedSumGenerator =
        () -> {
          LongBoundedSum longBoundedSum = longBoundedSumBuilder.build();
          for (long entry : samplingParameters.getNeighbourRawEntryList()) {
            longBoundedSum.addEntry(entry);
          }
          return longBoundedSum.computeResult();
        };

    assertThat(
            VotingUtil.runBallot(
                () ->
                    generateVote(
                        longBoundedSumGenerator,
                        neighbourLongBoundedSumGenerator,
                        samplingParameters.getNumberOfSamples(),
                        dpTestParameters.getEpsilon(),
                        dpTestParameters.getDelta(),
                        dpTestParameters.getDeltaTolerance(),
                        dpTestParameters.getGranularity()),
                getNumberOfVotesFromFile()))
        .isTrue();
  }

  private static int getNumberOfVotesFromFile() {
    return getTestCaseCollectionFromFile().getVotingParameters().getNumberOfVotes();
  }

  private static LongBoundedSumDpTestCaseCollection getTestCaseCollectionFromFile() {
    LongBoundedSumDpTestCaseCollection.Builder testCaseCollectionBuilder =
        LongBoundedSumDpTestCaseCollection.newBuilder();
    try {
      TextFormat.merge(
          new InputStreamReader(
              LongBoundedSumDpTest.class.getClassLoader().getResourceAsStream(TEST_CASES_FILE_PATH),
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
      double deltaTolerance,
      double granularity) {
    Long[] samplesA = new Long[numberOfSamples];
    Long[] samplesB = new Long[numberOfSamples];
    for (int i = 0; i < numberOfSamples; i++) {
      samplesA[i] = StatisticalTestsUtil.discretize(sampleGeneratorA.get(), granularity);
      samplesB[i] = StatisticalTestsUtil.discretize(sampleGeneratorB.get(), granularity);
    }
    return StatisticalTestsUtil.verifyApproximateDp(
        samplesA, samplesB, epsilon, delta, deltaTolerance);
  }
}

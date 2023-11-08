//
// Copyright 2021 Google LLC
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
import static java.lang.Math.sqrt;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.fail;

import com.google.privacy.differentialprivacy.BoundedVariance;
import com.google.privacy.differentialprivacy.Noise;
import com.google.privacy.differentialprivacy.TestNoiseFactory;
import com.google.privacy.differentialprivacy.proto.testing.StatisticalTests.BoundedStdvDpTestCase;
import com.google.privacy.differentialprivacy.proto.testing.StatisticalTests.BoundedStdvDpTestCaseCollection;
import com.google.privacy.differentialprivacy.proto.testing.StatisticalTests.BoundedStdvSamplingParameters;
import com.google.privacy.differentialprivacy.proto.testing.StatisticalTests.DpTestParameters;
import com.google.privacy.differentialprivacy.testing.StatisticalTestsUtil;
import com.google.privacy.differentialprivacy.testing.VotingUtil;
import com.google.protobuf.TextFormat;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Supplier;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

/**
 * Tests that {@link BoundedVariance} conforms to the specified privacy parameters epsilon and
 * delta.
 */
@RunWith(Parameterized.class)
public final class BoundedVarianceDpTest {
  private static final int NUM_SAMPLE_GENERATION_THREADS = 4;
  private static final String TEST_CASES_FILE_PATH =

  "external/com_google_differential_privacy/proto/testing/bounded_stdv_dp_test_cases.textproto";
  private final BoundedStdvDpTestCase testCase;
  public BoundedVarianceDpTest(BoundedStdvDpTestCase testCase) {
    this.testCase = testCase;
  }
  @Parameterized.Parameters
  public static Iterable<?> testCases() {
    return getTestCaseCollectionFromFile().getBoundedStdvDpTestCaseList();
  }
  /* Gets a new BoundedQuantiles with a thread-local source of randomness */
  private static BoundedVariance getBoundedVariance(
      BoundedStdvSamplingParameters samplingParameters) {
    Noise noise;
    double delta;
    switch (samplingParameters.getNoiseType()) {
      case LAPLACE:
        noise = TestNoiseFactory.createLaplaceNoise(ThreadLocalRandom.current());
        delta = 0.0;
        break;
      case GAUSSIAN:
        noise = TestNoiseFactory.createGaussianNoise(ThreadLocalRandom.current());
        delta = samplingParameters.getDelta();
        break;
      default:
        throw new IllegalArgumentException(
            "Noise type " + samplingParameters.getNoiseType() + " is not supported");
    }
    return BoundedVariance.builder()
        .epsilon(samplingParameters.getEpsilon())
        .delta(delta)
        .maxPartitionsContributed(samplingParameters.getMaxPartitionsContributed())
        .maxContributionsPerPartition(samplingParameters.getMaxContributionsPerPartition())
        .lower(samplingParameters.getLowerBound())
        .upper(samplingParameters.getUpperBound())
        .noise(noise)
        .build();
  }

  @Test
  public void boundedVarianceDpTest() {
    BoundedStdvSamplingParameters samplingParameters = testCase.getBoundedStdvSamplingParameters();
    DpTestParameters dpTestParameters = testCase.getDpTestParameters();
    Supplier<Double> boundedStdvGenerator =
        () -> {
          BoundedVariance boundedVariance = getBoundedVariance(samplingParameters);
          for (double entry : samplingParameters.getRawEntryList()) {
            boundedVariance.addEntry(entry);
          }
          // Returning the square root as the sample since the test cases are for
          // standard deviation.
          return sqrt(boundedVariance.computeResult());
        };
    Supplier<Double> neighbourBoundedStdvGenerator =
        () -> {
          BoundedVariance boundedVariance = getBoundedVariance(samplingParameters);
          for (double entry : samplingParameters.getNeighbourRawEntryList()) {
            boundedVariance.addEntry(entry);
          }
          // Returning the square root as the sample since the test cases are for
          // standard deviation.
          return sqrt(boundedVariance.computeResult());
        };
    assertThat(
            VotingUtil.runBallot(
                () ->
                    generateVote(
                        boundedStdvGenerator,
                        neighbourBoundedStdvGenerator,
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
  private static BoundedStdvDpTestCaseCollection getTestCaseCollectionFromFile() {
    BoundedStdvDpTestCaseCollection.Builder testCaseCollectionBuilder =
        BoundedStdvDpTestCaseCollection.newBuilder();
    try {
      TextFormat.merge(
          new InputStreamReader(
              BoundedVarianceDpTest.class
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
    try {
      List<Double> samplesA =
          StatisticalTestsUtil.generateSamplesParallel(
              () -> StatisticalTestsUtil.discretize(sampleGeneratorA.get(), granularity),
              numberOfSamples,
              NUM_SAMPLE_GENERATION_THREADS);
      List<Double> samplesB =
          StatisticalTestsUtil.generateSamplesParallel(
              () -> StatisticalTestsUtil.discretize(sampleGeneratorB.get(), granularity),
              numberOfSamples,
              NUM_SAMPLE_GENERATION_THREADS);
      return StatisticalTestsUtil.verifyApproximateDp(
          samplesA.toArray(), samplesB.toArray(), epsilon, delta, deltaTolerance);
    } catch (InterruptedException e) {
      fail(e.getMessage());
      return false;
    }
  }
}

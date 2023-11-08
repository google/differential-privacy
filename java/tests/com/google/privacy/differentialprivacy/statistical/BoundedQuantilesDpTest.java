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
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.fail;

import com.google.privacy.differentialprivacy.BoundedQuantiles;
import com.google.privacy.differentialprivacy.Noise;
import com.google.privacy.differentialprivacy.TestNoiseFactory;
import com.google.privacy.differentialprivacy.proto.testing.StatisticalTests.BoundedQuantilesDpTestCase;
import com.google.privacy.differentialprivacy.proto.testing.StatisticalTests.BoundedQuantilesDpTestCaseCollection;
import com.google.privacy.differentialprivacy.proto.testing.StatisticalTests.BoundedQuantilesSamplingParameters;
import com.google.privacy.differentialprivacy.proto.testing.StatisticalTests.DpTestParameters;
import com.google.privacy.differentialprivacy.testing.StatisticalTestsUtil;
import com.google.privacy.differentialprivacy.testing.VotingUtil;
import com.google.protobuf.TextFormat;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Supplier;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

/**
 * Tests that {@link BoundedQuantiles} conforms to the specified privacy parameters epsilon and
 * delta.
 */
@RunWith(Parameterized.class)
public final class BoundedQuantilesDpTest {
  private static final int NUM_SAMPLE_GENERATION_THREADS = 4;

  private static final String TEST_CASES_FILE_PATH =

  "external/com_google_differential_privacy/proto/testing/bounded_quantiles_dp_test_cases.textproto";

  private final BoundedQuantilesDpTestCase testCase;

  public BoundedQuantilesDpTest(BoundedQuantilesDpTestCase testCase) {
    this.testCase = testCase;
  }

  @Parameterized.Parameters
  public static Iterable<?> testcases() {
    return getTestCaseCollectionFromFile().getBoundedQuantilesDpTestCaseList();
  }

  /* Gets a new BoundedQuantiles with a thread-local source of randomness */
  private static BoundedQuantiles getBoundedQuantiles(
      BoundedQuantilesSamplingParameters samplingParameters) {
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
    return BoundedQuantiles.builder()
        .epsilon(samplingParameters.getEpsilon())
        .delta(delta)
        .maxContributionsPerPartition(samplingParameters.getMaxContributionsPerPartition())
        .maxPartitionsContributed(samplingParameters.getMaxPartitionsContributed())
        .lower(samplingParameters.getLowerBound())
        .upper(samplingParameters.getUpperBound())
        .noise(noise)
        .treeHeight(samplingParameters.getTreeHeight())
        .branchingFactor(samplingParameters.getBranchingFactor())
        .build();
  }

  @Test
  public void boundedQuantilesDpTest() {

    BoundedQuantilesSamplingParameters samplingParameters =
        testCase.getBoundedQuantilesSamplingParameters();
    DpTestParameters dpTestParameters = testCase.getDpTestParameters();

    Supplier<List<Double>> boundedQuantilesGenerator =
        () -> {
          BoundedQuantiles boundedQuantiles = getBoundedQuantiles(samplingParameters);
          for (double entry : samplingParameters.getRawEntryList()) {
            boundedQuantiles.addEntry(entry);
          }
          List<Double> result = new ArrayList<>(samplingParameters.getRankCount());
          for (double rank : samplingParameters.getRankList()) {
            result.add(boundedQuantiles.computeResult(rank));
          }
          return result;
        };
    Supplier<List<Double>> neighbourBoundedQuantilesGenerator =
        () -> {
          BoundedQuantiles boundedQuantiles = getBoundedQuantiles(samplingParameters);
          for (double entry : samplingParameters.getNeighbourRawEntryList()) {
            boundedQuantiles.addEntry(entry);
          }
          List<Double> result = new ArrayList<>(samplingParameters.getRankCount());
          for (double rank : samplingParameters.getRankList()) {
            result.add(boundedQuantiles.computeResult(rank));
          }
          return result;
        };

    assertThat(
            VotingUtil.runBallot(
                () ->
                    generateVote(
                        boundedQuantilesGenerator,
                        neighbourBoundedQuantilesGenerator,
                        samplingParameters.getNumberOfSamples(),
                        samplingParameters.getRankCount(),
                        samplingParameters.getLowerBound(),
                        samplingParameters.getUpperBound(),
                        dpTestParameters.getEpsilon(),
                        dpTestParameters.getDelta(),
                        dpTestParameters.getDeltaTolerance(),
                        dpTestParameters.getNumOfBuckets()),
                getNumberOfVotesFromFile()))
        .isTrue();
  }

  private static int getNumberOfVotesFromFile() {
    return getTestCaseCollectionFromFile().getVotingParameters().getNumberOfVotes();
  }

  private static BoundedQuantilesDpTestCaseCollection getTestCaseCollectionFromFile() {
    BoundedQuantilesDpTestCaseCollection.Builder testCaseCollectionBuilder =
        BoundedQuantilesDpTestCaseCollection.newBuilder();
    try {
      TextFormat.merge(
          new InputStreamReader(
              BoundedQuantilesDpTest.class
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
      Supplier<List<Double>> sampleGeneratorA,
      Supplier<List<Double>> sampleGeneratorB,
      int numberOfSamples,
      int numberOfRanks,
      double lower,
      double upper,
      double epsilon,
      double delta,
      double deltaTolerance,
      int numberOfBuckets) {

    List<List<Double>> generatedSamplesA;
    List<List<Double>> generatedSamplesB;
    try {
      generatedSamplesA =
          StatisticalTestsUtil.generateSamplesParallel(
              sampleGeneratorA, numberOfSamples, NUM_SAMPLE_GENERATION_THREADS);
      generatedSamplesB =
          StatisticalTestsUtil.generateSamplesParallel(
              sampleGeneratorB, numberOfSamples, NUM_SAMPLE_GENERATION_THREADS);
    } catch (InterruptedException e) {
      fail(e.getMessage());
      return false;
    }

    // Each sample consists of a vector of quantiles, one for each rank.
    List<Integer[]> samplesByRankA = new ArrayList<>(numberOfRanks);
    List<Integer[]> samplesByRankB = new ArrayList<>(numberOfRanks);
    for (int j = 0; j < numberOfRanks; j++) {
      samplesByRankA.add(new Integer[numberOfSamples]);
      samplesByRankB.add(new Integer[numberOfSamples]);
    }
    for (int i = 0; i < numberOfSamples; i++) {
      for (int j = 0; j < numberOfRanks; j++) {
        samplesByRankA.get(j)[i] =
            StatisticalTestsUtil.bucketize(
                generatedSamplesA.get(i).get(j), lower, upper, numberOfBuckets);
        samplesByRankB.get(j)[i] =
            StatisticalTestsUtil.bucketize(
                generatedSamplesB.get(i).get(j), lower, upper, numberOfBuckets);
      }
    }

    // Only cast an accept vote if all quantiles pass the test.
    for (int j = 0; j < numberOfRanks; j++) {
      if (!StatisticalTestsUtil.verifyApproximateDp(
          samplesByRankA.get(j), samplesByRankB.get(j), epsilon, delta, deltaTolerance)) {
        return false;
      }
    }
    return true;
  }
}

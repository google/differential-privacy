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
import static com.google.common.truth.Truth.assertWithMessage;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.privacy.differentialprivacy.ApproximateBounds;
import com.google.privacy.differentialprivacy.proto.testing.StatisticalTests.ApproximateBoundsDpTestCase;
import com.google.privacy.differentialprivacy.proto.testing.StatisticalTests.ApproximateBoundsDpTestCaseCollection;
import com.google.privacy.differentialprivacy.proto.testing.StatisticalTests.ApproximateBoundsSamplingParameters;
import com.google.privacy.differentialprivacy.proto.testing.StatisticalTests.DpTestParameters;
import com.google.privacy.differentialprivacy.proto.testing.StatisticalTests.NoiseType;
import com.google.privacy.differentialprivacy.testing.StatisticalTestsUtil;
import com.google.privacy.differentialprivacy.testing.VotingUtil;
import com.google.protobuf.TextFormat;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.List;
import java.util.Optional;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

/**
 * Tests that {@link ApproximateBounds} conforms to the specified privacy parameters epsilon and
 * delta.
 */
@RunWith(Parameterized.class)
public final class ApproximateBoundsDpTest {
  private static final String TEST_CASES_FILE_PATH =

  "external/com_google_differential_privacy/proto/testing/approximate_bounds_dp_test_cases.textproto";

  private final ApproximateBoundsDpTestCase testCase;

  public ApproximateBoundsDpTest(ApproximateBoundsDpTestCase testCase) {
    this.testCase = testCase;
  }

  @Test
  public void approximateBoundsDpTest() {
    ApproximateBoundsSamplingParameters samplingParameters =
        testCase.getApproximateBoundsSamplingParameters();
    ApproximateBounds.Params.Builder approximateBoundsBuilder =
        ApproximateBounds.builder()
            .inputType(ApproximateBounds.Params.InputType.TEST)
            .epsilon(samplingParameters.getEpsilon())
            .maxContributions(samplingParameters.getMaxContributions());
    DpTestParameters dpTestParameters = testCase.getDpTestParameters();
    verifyParametersMatch(ApproximateBounds.Params.InputType.TEST, samplingParameters);
    SampleGenerator generatorA =
        getSampleGenerator(samplingParameters.getRawEntryList(), approximateBoundsBuilder);
    SampleGenerator generatorB =
        getSampleGenerator(samplingParameters.getNeighbourRawEntryList(), approximateBoundsBuilder);

    assertThat(
            VotingUtil.runBallot(
                () ->
                    generateVote(
                        generatorA,
                        generatorB,
                        samplingParameters.getNumberOfSamples(),
                        dpTestParameters.getEpsilon(),
                        dpTestParameters.getDelta(),
                        dpTestParameters.getDeltaTolerance()),
                getNumberOfVotesFromFile()))
        .isTrue();
  }

  /**
   * Creates a SampleGenerator that runs on the given sample.
   *
   * <p>Calling {@link SampleGenerator#sample} on the returned generator will calculate bounds for
   * the dataset {@code dataset} using the parameters in {@code approximateBoundsBuilder}.
   */
  private SampleGenerator getSampleGenerator(
      List<Double> dataset, ApproximateBounds.Params.Builder approximateBoundsBuilder) {
    return () -> {
      ApproximateBounds approximateBounds = approximateBoundsBuilder.build();
      for (double entry : dataset) {
        approximateBounds.addEntry(entry);
      }
      try {
        return Optional.of(approximateBounds.computeResult());
      } catch (IllegalArgumentException e) {
        assertThat(e).hasMessageThat().contains("Bin count threshold was too large");
        return Optional.empty();
      }
    };
  }

  private static boolean generateVote(
      SampleGenerator sampleGeneratorA,
      SampleGenerator sampleGeneratorB,
      int numberOfSamples,
      double epsilon,
      double delta,
      double deltaTolerance) {
    Integer[] samplesA = new Integer[numberOfSamples];
    Integer[] samplesB = new Integer[numberOfSamples];
    for (int i = 0; i < numberOfSamples; i++) {
      samplesA[i] = sampleGeneratorA.sample().hashCode();
      samplesB[i] = sampleGeneratorB.sample().hashCode();
    }
    return StatisticalTestsUtil.verifyApproximateDp(
        samplesA, samplesB, epsilon, delta, deltaTolerance);
  }

  private static int getNumberOfVotesFromFile() {
    return getTestCaseCollectionFromFile().getVotingParameters().getNumberOfVotes();
  }

  private void verifyParametersMatch(
      ApproximateBounds.Params.InputType inputType,
      ApproximateBoundsSamplingParameters samplingParameters) {
    String header =
        "Consistency check failed: the sampling parameters for this test case were not expected.";
    assertWithMessage(header)
        .that(samplingParameters.getNumberOfBins())
        .isEqualTo(inputType.numPositiveBins);
    assertWithMessage(header)
        .that(samplingParameters.getNumberOfBins())
        .isEqualTo(inputType.numNegativeBins);
    assertWithMessage(header).that(samplingParameters.getScale()).isEqualTo(inputType.scale);
    assertWithMessage(header).that(samplingParameters.getBase()).isEqualTo(inputType.base);
    assertWithMessage(header).that(samplingParameters.getNoiseType()).isEqualTo(NoiseType.LAPLACE);
  }

  @Parameterized.Parameters
  public static List<ApproximateBoundsDpTestCase> testCases() {
    return getTestCaseCollectionFromFile().getApproximateBoundsDpTestCaseList();
  }

  private static ApproximateBoundsDpTestCaseCollection getTestCaseCollectionFromFile() {
    ApproximateBoundsDpTestCaseCollection.Builder testCaseCollectionBuilder =
        ApproximateBoundsDpTestCaseCollection.newBuilder();
    try {
      TextFormat.merge(
          new InputStreamReader(
              ApproximateBoundsDpTest.class
                  .getClassLoader()
                  .getResourceAsStream(TEST_CASES_FILE_PATH),
              UTF_8),
          testCaseCollectionBuilder);
    } catch (IOException | NullPointerException e) {
      throw new RuntimeException("Unable to read input file.", e);
    }
    return testCaseCollectionBuilder.build();
  }

  /**
   * Generates bounds for a fixed sample with fixed DP parameters.
   *
   * <p>This is just a friendlier name for {@code Supplier<Optional<ApproximateBounds.Result>>}.
   */
  private static interface SampleGenerator {
    Optional<ApproximateBounds.Result> sample();
  }
}

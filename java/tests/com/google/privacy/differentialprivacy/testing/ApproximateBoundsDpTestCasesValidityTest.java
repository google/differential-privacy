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

import static com.google.common.truth.Truth.assertWithMessage;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.Supplier;
import com.google.privacy.differentialprivacy.proto.testing.StatisticalTests.ApproximateBoundsDpTestCase;
import com.google.privacy.differentialprivacy.proto.testing.StatisticalTests.ApproximateBoundsDpTestCaseCollection;
import com.google.privacy.differentialprivacy.proto.testing.StatisticalTests.ApproximateBoundsSamplingParameters;
import com.google.privacy.differentialprivacy.proto.testing.StatisticalTests.DpTestParameters;
import com.google.protobuf.TextFormat;
import java.io.IOException;
import java.io.InputStreamReader;
import java.security.SecureRandom;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(Parameterized.class)
public final class ApproximateBoundsDpTestCasesValidityTest {
  private static final String TEST_CASES_FILE_PATH =

  "external/com_google_differential_privacy/proto/testing/approximate_bounds_dp_test_cases.textproto";
  private static final short BOUND_RETURNED = 1;
  private static final short BOUND_NOT_RETURNED = 0;
  private static final short FAILED = -1;

  private final int numberOfVotes =
      getTestCaseCollectionFromFile().getVotingParameters().getNumberOfVotes();
  private final double epsilonSpecificity =
      getTestCaseCollectionFromFile().getValidityParameters().getEpsilonSpecificity();
  private final double failureSpecificity =
      getTestCaseCollectionFromFile().getValidityParameters().getFailureSpecificity();
  private final ApproximateBoundsDpTestCase testCase;

  public ApproximateBoundsDpTestCasesValidityTest(ApproximateBoundsDpTestCase testCase) {
    this.testCase = testCase;
  }

  @Parameterized.Parameters
  public static List<ApproximateBoundsDpTestCase> testCases() {
    return getTestCaseCollectionFromFile().getApproximateBoundsDpTestCaseList();
  }

  @Test
  public void approximateBoundsDpTest_acceptsProbabilityDifferenceWithinEpsilonParameter() {
    ApproximateBoundsSamplingParameters samplingParameters =
        testCase.getApproximateBoundsSamplingParameters();
    DpTestParameters dpTestParameters = testCase.getDpTestParameters();
    SecureRandom random = new SecureRandom();
    double epsilon = samplingParameters.getEpsilon();
    double delta = samplingParameters.getDelta();

    // Let p be the probability that approximate bounds returns a bound for a given input.
    // Similarly let p' be the probability that approximate bounds returns that bound for a
    // neighbouring input. To satisfy DP, it is necessary (but not sufficient) that
    //    p * e^ε + δ >= p'.
    // Thus p = e^-ε * (p' - delta) is the smallest value of p that should be accepted.
    // Note that we intentionally don't adjust for maxContributions here, because we're
    // describing the distribution of the mechanism output, not applying the Laplace mechanism.
    double pPrime = 0.5; // arbitrary choice
    double p = Math.exp(-epsilon) * (pPrime - delta);

    Supplier<Short> sampleGenerator =
        () -> random.nextDouble() > p ? BOUND_RETURNED : BOUND_NOT_RETURNED;
    Supplier<Short> neighbourSampleGenerator =
        () -> random.nextDouble() > pPrime ? BOUND_RETURNED : BOUND_NOT_RETURNED;

    assertWithMessage(
            "Expected ballot to accept '%s' when using correct epsilon.", testCase.getName())
        .that(
            VotingUtil.runBallot(
                () ->
                    generateVote(
                        sampleGenerator,
                        neighbourSampleGenerator,
                        samplingParameters.getNumberOfSamples(),
                        dpTestParameters.getEpsilon(),
                        dpTestParameters.getDelta(),
                        dpTestParameters.getDeltaTolerance()),
                numberOfVotes))
        .isTrue();
  }

  @Test
  public void approximateBoundsDpTest_rejectsProbabilityDifferenceGreaterThanEpsilonParameter() {
    ApproximateBoundsSamplingParameters samplingParameters =
        testCase.getApproximateBoundsSamplingParameters();
    DpTestParameters dpTestParameters = testCase.getDpTestParameters();
    SecureRandom random = new SecureRandom();
    double epsilon = samplingParameters.getEpsilon();
    double delta = samplingParameters.getDelta();

    // Let p be the probability that approximate bounds returns a bound for a given input. Similarly
    // let p' be the probability that approximate bounds returns that bound for a neighbouring
    // input. To satisfy DP, it is necessary (but not sufficient) that
    //     p * e^ε + δ >= p'.
    // Let s > 1 be the test specificity. If we set p = e^(-ε * s) * (p' - delta), this is below
    // threshold e^(-ε) * (p' - delta) that is the minimum that we expect the test to accept, so we
    // expect the test to fail. Note that we intentionally don't adjust for maxContributions here,
    // because we're describing the distribution of the output, not applying the Laplace mechanism.
    double pPrime = 0.5; // arbitrary choice
    double p = Math.exp(-epsilon * epsilonSpecificity) * (pPrime - delta);

    assertWithMessage(
            "Chosen value of p is not expected to violate DP: please increase epsilon specificity")
        .that(p * Math.exp(epsilon) + delta + dpTestParameters.getDeltaTolerance())
        .isLessThan(pPrime);
    Supplier<Short> sampleGenerator =
        () -> random.nextDouble() > p ? BOUND_RETURNED : BOUND_NOT_RETURNED;
    Supplier<Short> neighbourSampleGenerator =
        () -> random.nextDouble() > pPrime ? BOUND_RETURNED : BOUND_NOT_RETURNED;

    assertWithMessage(
            "Expected ballot to reject '%s' when using incorrect epsilon.", testCase.getName())
        .that(
            VotingUtil.runBallot(
                () ->
                    generateVote(
                        sampleGenerator,
                        neighbourSampleGenerator,
                        samplingParameters.getNumberOfSamples(),
                        dpTestParameters.getEpsilon(),
                        dpTestParameters.getDelta(),
                        dpTestParameters.getDeltaTolerance()),
                numberOfVotes))
        .isFalse();
  }

  @Test
  public void approximateBoundsDpTest_rejectsCriticallyFailingBounds() {
    ApproximateBoundsSamplingParameters samplingParameters =
        testCase.getApproximateBoundsSamplingParameters();
    DpTestParameters dpTestParameters = testCase.getDpTestParameters();
    SecureRandom random = new SecureRandom();

    Supplier<Short> sampleGenerator =
        () -> random.nextBoolean() ? BOUND_RETURNED : BOUND_NOT_RETURNED;
    Supplier<Short> neighbourSampleGenerator =
        () ->
            random.nextDouble() > dpTestParameters.getDeltaTolerance() * failureSpecificity
                ? (random.nextBoolean() ? BOUND_RETURNED : BOUND_NOT_RETURNED)
                : FAILED;

    assertWithMessage(
            "Expected ballot to reject '%s' when critical failures present.", testCase.getName())
        .that(
            VotingUtil.runBallot(
                () ->
                    generateVote(
                        sampleGenerator,
                        neighbourSampleGenerator,
                        samplingParameters.getNumberOfSamples(),
                        dpTestParameters.getEpsilon(),
                        dpTestParameters.getDelta(),
                        dpTestParameters.getDeltaTolerance()),
                numberOfVotes))
        .isFalse();
  }

  private static ApproximateBoundsDpTestCaseCollection getTestCaseCollectionFromFile() {
    ApproximateBoundsDpTestCaseCollection.Builder testCaseCollectionBuilder =
        ApproximateBoundsDpTestCaseCollection.newBuilder();
    try {
      TextFormat.merge(
          new InputStreamReader(
              ApproximateBoundsDpTestCasesValidityTest.class
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
      Supplier<Short> sampleGeneratorA,
      Supplier<Short> sampleGeneratorB,
      int numberOfSamples,
      double epsilon,
      double delta,
      double deltaTolerance) {
    Short[] samplesA = new Short[numberOfSamples];
    Short[] samplesB = new Short[numberOfSamples];
    for (int i = 0; i < numberOfSamples; i++) {
      samplesA[i] = sampleGeneratorA.get();
      samplesB[i] = sampleGeneratorB.get();
    }
    return StatisticalTestsUtil.verifyApproximateDp(
        samplesA, samplesB, epsilon, delta, deltaTolerance);
  }
}

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

package com.google.privacy.differentialprivacy.testing;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.Supplier;
import com.google.privacy.differentialprivacy.proto.testing.StatisticalTests.BoundedQuantilesDpTestCase;
import com.google.privacy.differentialprivacy.proto.testing.StatisticalTests.BoundedQuantilesDpTestCaseCollection;
import com.google.privacy.differentialprivacy.proto.testing.StatisticalTests.BoundedQuantilesSamplingParameters;
import com.google.privacy.differentialprivacy.proto.testing.StatisticalTests.DpTestParameters;
import com.google.protobuf.TextFormat;
import java.io.IOException;
import java.io.InputStreamReader;
import java.security.SecureRandom;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(Parameterized.class)
public final class BoundedQuantilesDpTestCasesValidityTest {
   private static final String TEST_CASES_FILE_PATH =

  "external/com_google_differential_privacy/proto/testing/bounded_quantiles_dp_test_cases.textproto";
  private final int numberOfVotes =
      getTestCaseCollectionFromFile().getVotingParameters().getNumberOfVotes();
  private final double epsilonSpecificity =
      getTestCaseCollectionFromFile().getValidityParameters().getEpsilonSpecificity();
  private final double failureSpecificity =
      getTestCaseCollectionFromFile().getValidityParameters().getFailureSpecificity();
  private final BoundedQuantilesDpTestCase testCase;

  public BoundedQuantilesDpTestCasesValidityTest(BoundedQuantilesDpTestCase testCase) {
    this.testCase = testCase;
  }

  @Parameterized.Parameters
  public static Iterable<? extends Object> testCases() {
    return getTestCaseCollectionFromFile().getBoundedQuantilesDpTestCaseList();
  }

  @Test
  public void boundedQuantilesDpTest_acceptsDistributionsSatisfyingEpsilonDP() {
    // Note that we are testing for epsilon-DP rather than (epsilon, delta)-DP. This is stricter
    // than necessary. However, a positive outcome implies that (epsilon, delta)-DP will be accepted
    // as well for any delta > 0.
    BoundedQuantilesSamplingParameters samplingParameters =
        testCase.getBoundedQuantilesSamplingParameters();
    DpTestParameters dpTestParameters = testCase.getDpTestParameters();
    SecureRandom random = new SecureRandom();
    double epsilon = samplingParameters.getEpsilon();

    Supplier<Integer> sampleGenerator =
        () ->
            random.nextDouble()
                    <= getDistinguishingProb(epsilon, dpTestParameters.getNumOfBuckets())
                ? 0
                : 1 + random.nextInt(dpTestParameters.getNumOfBuckets() - 1);
    Supplier<Integer> neighbouringSampleGenerator =
        () ->
            random.nextDouble()
                    <= getDistinguishingProb(epsilon, dpTestParameters.getNumOfBuckets())
                ? dpTestParameters.getNumOfBuckets() - 1
                : random.nextInt(dpTestParameters.getNumOfBuckets() - 1);

    assertThat(
            VotingUtil.runBallot(
                () ->
                    generateVote(
                        sampleGenerator,
                        neighbouringSampleGenerator,
                        samplingParameters.getNumberOfSamples(),
                        dpTestParameters.getEpsilon(),
                        dpTestParameters.getDelta(),
                        dpTestParameters.getDeltaTolerance()),
                numberOfVotes))
        .isTrue();
  }

  @Test
  public void boundedQuantilesDpTest_rejectsDistributionsViolatingScaledEpsilonDP() {
    BoundedQuantilesSamplingParameters samplingParameters =
        testCase.getBoundedQuantilesSamplingParameters();
    DpTestParameters dpTestParameters = testCase.getDpTestParameters();
    SecureRandom random = new SecureRandom();
    double epsilon = samplingParameters.getEpsilon();

    Supplier<Integer> sampleGenerator =
        () ->
            random.nextDouble()
                    <= epsilonSpecificity
                        * getDistinguishingProb(epsilon, dpTestParameters.getNumOfBuckets())
                ? 0
                : 1 + random.nextInt(dpTestParameters.getNumOfBuckets() - 1);
    Supplier<Integer> neighbouringSampleGenerator =
        () ->
            random.nextDouble()
                    <= epsilonSpecificity
                        * getDistinguishingProb(epsilon, dpTestParameters.getNumOfBuckets())
                ? dpTestParameters.getNumOfBuckets() - 1
                : random.nextInt(dpTestParameters.getNumOfBuckets() - 1);

    assertThat(
            VotingUtil.runBallot(
                () ->
                    generateVote(
                        sampleGenerator,
                        neighbouringSampleGenerator,
                        samplingParameters.getNumberOfSamples(),
                        dpTestParameters.getEpsilon(),
                        dpTestParameters.getDelta(),
                        dpTestParameters.getDeltaTolerance()),
                numberOfVotes))
        .isFalse();
  }

  @Test
  public void boundedQuantilesDpTest_rejectsCriticallyFailingQuantiles() {

    BoundedQuantilesSamplingParameters samplingParameters =
        testCase.getBoundedQuantilesSamplingParameters();
    DpTestParameters dpTestParameters = testCase.getDpTestParameters();
    SecureRandom random = new SecureRandom();
    double epsilon = samplingParameters.getEpsilon();

    Supplier<Integer> sampleGenerator =
        () ->
            random.nextDouble()
                    <= getDistinguishingProb(epsilon, dpTestParameters.getNumOfBuckets())
                ? 0
                : 1 + random.nextInt(dpTestParameters.getNumOfBuckets() - 1);
    Supplier<Integer> criticallyFailingSampleGenerator =
        () ->
            random.nextDouble() > dpTestParameters.getDeltaTolerance() * failureSpecificity
                ? sampleGenerator.get()
                : -1;

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

  private static BoundedQuantilesDpTestCaseCollection getTestCaseCollectionFromFile() {
    BoundedQuantilesDpTestCaseCollection.Builder testCaseCollectionBuilder =
        BoundedQuantilesDpTestCaseCollection.newBuilder();
    try {
      TextFormat.merge(
          new InputStreamReader(
              BoundedQuantilesDpTestCasesValidityTest.class
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
      Supplier<Integer> sampleGeneratorA,
      Supplier<Integer> sampleGeneratorB,
      int numberOfSamples,
      double epsilon,
      double delta,
      double deltaTolerance) {
    Integer[] samplesA = new Integer[numberOfSamples];
    Integer[] samplesB = new Integer[numberOfSamples];
    for (int i = 0; i < numberOfSamples; i++) {
      samplesA[i] = sampleGeneratorA.get();
      samplesB[i] = sampleGeneratorB.get();
    }
    return StatisticalTestsUtil.verifyApproximateDp(
        samplesA, samplesB, epsilon, delta, deltaTolerance);
  }

  /**
   * Let Omega be a discrete proability space with n elements, such that all of its elements have
   * the same probability except for one distinguishing element E, which is more likely then the
   * rest. Given a DP paramter epsilon, this method returns the probability Pr[E] of E such that
   *     Pr[E] = e^epsilon * Pr[E']
   * for all elements E' in Omega / {E}.
   *
   * <p>The idea is that switching the distinguishing element E, results in two discrete proability
   * spaces, that tightly satisfy epsilon-DP.
   */
  private static double getDistinguishingProb(double epsilon, int n) {
    return Math.exp(epsilon) / (n - 1.0 + Math.exp(epsilon));
  }
}

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

package com.google.privacy.differentialprivacy;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static java.lang.Math.abs;
import static java.lang.Math.max;
import static org.junit.Assert.assertThrows;

import com.google.auto.value.AutoValue;
import com.google.privacy.differentialprivacy.proto.SummaryOuterClass.PreAggSelectPartitionSummary;
import com.google.protobuf.InvalidProtocolBufferException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.runners.Enclosed;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.junit.runners.Parameterized;

@RunWith(Enclosed.class)
public class PreAggSelectPartitionTest {
  private static final double EPSILON = Math.log(2);
  private static final double LOW_EPSILON = Math.log(1.5);
  private static final double HIGH_EPSILON = 50;
  private static final double HIGH_DELTA = 1 - 1e-15;
  private static final double DELTA = 0.1;
  private static final double LOW_DELTA = 1e-200;
  private static final int ONE_PARTITION_CONTRIBUTED = 1;

  @RunWith(JUnit4.class)
  public static final class NonParameterizedTests {

    private PreAggSelectPartition preAggSelectPartition;

    @Before
    public void setUp() {
      preAggSelectPartition = getPreAggSelectPartitionBuilderWithFields().build();
    }

    @Test
    public void increment_calledAfterShouldKeepPartition_throwsException() {
      preAggSelectPartition.shouldKeepPartition();
      assertThrows(IllegalStateException.class, () -> preAggSelectPartition.increment());
    }

    @Test
    public void increment_calledAfterSerialize_throwsException() {
      preAggSelectPartition.getSerializableSummary();
      assertThrows(IllegalStateException.class, () -> preAggSelectPartition.increment());
    }

    @Test
    public void incrementBy_allowsNegativeValues() {
      PreAggSelectPartition largeDeltaPreAggSelectPartition =
          PreAggSelectPartition.builder()
              .epsilon(HIGH_EPSILON)
              .delta(HIGH_DELTA)
              .maxPartitionsContributed(1)
              .build();
      // We can't access the value of the count, so instead we test that adding and subtracting the
      // same large count results in a deterministic false when shouldKeepPartition is called.
      // If negative values are ignored, shouldKeepPartition would return true.
      largeDeltaPreAggSelectPartition.incrementBy(100);
      largeDeltaPreAggSelectPartition.incrementBy(-100);

      assertThat(largeDeltaPreAggSelectPartition.shouldKeepPartition()).isFalse();
    }

    // An attempt to compute the result several times should throw an exception.
    @Test
    public void computeResult_multipleCalls_throwsException() {
      preAggSelectPartition.shouldKeepPartition();
      assertThrows(IllegalStateException.class, preAggSelectPartition::shouldKeepPartition);
    }

    @Test
    public void getSerializableSummary_copiesIdsCountCorrectly() {
      preAggSelectPartition.increment();
      preAggSelectPartition.increment();
      preAggSelectPartition.increment();

      PreAggSelectPartitionSummary summary = getSummary(preAggSelectPartition);
      assertThat(summary.getIdsCount()).isEqualTo(3);
    }

    @Test
    public void incrementByAndGetSerializableSummary_copiesIdsCountCorrectly() {
      preAggSelectPartition.incrementBy(5);
      preAggSelectPartition.incrementBy(5);

      PreAggSelectPartitionSummary summary = getSummary(preAggSelectPartition);
      assertThat(summary.getIdsCount()).isEqualTo(10);
    }

    @Test
    public void getSerializableSummary_copiesZeroIdsCountCorrectly() {
      PreAggSelectPartitionSummary summary = getSummary(preAggSelectPartition);
      assertThat(summary.getIdsCount()).isEqualTo(0);
    }

    @Test
    public void getSerializableSummary_copiesPreThresholdCorrectly() {
      PreAggSelectPartitionSummary summary =
          getSummary(getPreAggSelectPartitionBuilderWithFields().preThreshold(2).build());
      assertThat(summary.getPreThreshold()).isEqualTo(2);
    }

    @Test
    public void getSerializableSummary_copiesDefaultPreThresholdCorrectly() {
      PreAggSelectPartitionSummary summary = getSummary(preAggSelectPartition);
      assertThat(summary.getPreThreshold()).isEqualTo(1);
    }

    @Test
    public void getSerializableSummary_calledAfterComputeResult_throwsException() {
      preAggSelectPartition.shouldKeepPartition();
      assertThrows(
          IllegalStateException.class, () -> preAggSelectPartition.getSerializableSummary());
    }

    @Test
    public void getSerializableSummary_multipleCalls_returnsSameSummary() {
      preAggSelectPartition =
          PreAggSelectPartition.builder()
              .epsilon(EPSILON)
              .delta(DELTA)
              .maxPartitionsContributed(1)
              .build();
      preAggSelectPartition.increment();
      byte[] summary1 = preAggSelectPartition.getSerializableSummary();
      byte[] summary2 = preAggSelectPartition.getSerializableSummary();
      assertThat(summary1).isEqualTo(summary2);
    }

    @Test
    public void computeResult_calledAfterSerialize_throwsException() {
      preAggSelectPartition.getSerializableSummary();
      assertThrows(IllegalStateException.class, () -> preAggSelectPartition.shouldKeepPartition());
    }

    @Test
    public void getSerializableSummary_copiesEpsilonCorrectly() {
      preAggSelectPartition = getPreAggSelectPartitionBuilderWithFields().epsilon(EPSILON).build();
      PreAggSelectPartitionSummary summary = getSummary(preAggSelectPartition);
      assertThat(summary.getEpsilon()).isEqualTo(EPSILON);
    }

    @Test
    public void getSerializableSummary_copiesDeltaCorrectly() {
      preAggSelectPartition = getPreAggSelectPartitionBuilderWithFields().delta(DELTA).build();
      PreAggSelectPartitionSummary summary = getSummary(preAggSelectPartition);
      assertThat(summary.getDelta()).isEqualTo(DELTA);
    }

    @Test
    public void getSerializableSummary_copiesMaxPartitionsContributedCorrectly() {
      int maxPartitionsContributed = 150;
      preAggSelectPartition =
          getPreAggSelectPartitionBuilderWithFields()
              .maxPartitionsContributed(maxPartitionsContributed)
              .build();
      PreAggSelectPartitionSummary summary = getSummary(preAggSelectPartition);
      assertThat(summary.getMaxPartitionsContributed()).isEqualTo(maxPartitionsContributed);
    }

    @Test
    public void merge_basicExample_sumsIdsCounts() {
      PreAggSelectPartition targetPreAggSelectPartition =
          getPreAggSelectPartitionBuilderWithFields().delta(0.3).build();
      targetPreAggSelectPartition.increment();
      PreAggSelectPartition sourcePreAggSelectPartition =
          getPreAggSelectPartitionBuilderWithFields().delta(0.3).build();
      sourcePreAggSelectPartition.increment();
      sourcePreAggSelectPartition.increment();
      sourcePreAggSelectPartition.increment();

      targetPreAggSelectPartition.mergeWith(sourcePreAggSelectPartition.getSerializableSummary());
      // We expect it always be true because the total number of ids in the merged object will be
      // equal to 4 and for that params and that count the probability of keeping that partition is
      // equal to 1.
      assertThat(targetPreAggSelectPartition.shouldKeepPartition()).isTrue();
    }

    @Test
    public void merge_calledTwice_sumsIdsCounts() {
      PreAggSelectPartition targetPreAggSelectPartition =
          getPreAggSelectPartitionBuilderWithFields().delta(0.3).build();
      targetPreAggSelectPartition.increment();
      PreAggSelectPartition sourcePreAggSelectPartition1 =
          getPreAggSelectPartitionBuilderWithFields().delta(0.3).build();
      sourcePreAggSelectPartition1.increment();
      sourcePreAggSelectPartition1.increment();
      PreAggSelectPartition sourcePreAggSelectPartition2 =
          getPreAggSelectPartitionBuilderWithFields().delta(0.3).build();
      sourcePreAggSelectPartition2.increment();

      targetPreAggSelectPartition.mergeWith(sourcePreAggSelectPartition1.getSerializableSummary());
      targetPreAggSelectPartition.mergeWith(sourcePreAggSelectPartition2.getSerializableSummary());
      // We expect it always be true because the total number of ids in the merged object will be
      // equal to 4 and for that params and that count the probability of keeping that partition is
      // equal to 1.
      assertThat(targetPreAggSelectPartition.shouldKeepPartition()).isTrue();
    }

    @Test
    public void merge_differentEpsilon_throwsException() {
      PreAggSelectPartition targetPreAggSelectPartition =
          getPreAggSelectPartitionBuilderWithFields().epsilon(EPSILON).build();
      PreAggSelectPartition sourcePreAggSelectPartition =
          getPreAggSelectPartitionBuilderWithFields().epsilon(2 * EPSILON).build();
      assertThrows(
          IllegalArgumentException.class,
          () ->
              targetPreAggSelectPartition.mergeWith(
                  sourcePreAggSelectPartition.getSerializableSummary()));
    }

    @Test
    public void merge_differentDelta_throwsException() {
      PreAggSelectPartition targetPreAggSelectPartition =
          getPreAggSelectPartitionBuilderWithFields().delta(DELTA).build();
      PreAggSelectPartition sourcePreAggSelectPartition =
          getPreAggSelectPartitionBuilderWithFields().delta(2 * DELTA).build();
      assertThrows(
          IllegalArgumentException.class,
          () ->
              targetPreAggSelectPartition.mergeWith(
                  sourcePreAggSelectPartition.getSerializableSummary()));
    }

    @Test
    public void merge_differentMaxPartitionsContributed_throwsException() {
      PreAggSelectPartition targetPreAggSelectPartition =
          getPreAggSelectPartitionBuilderWithFields().maxPartitionsContributed(1).build();
      PreAggSelectPartition sourcePreAggSelectPartition =
          getPreAggSelectPartitionBuilderWithFields().maxPartitionsContributed(2).build();
      assertThrows(
          IllegalArgumentException.class,
          () ->
              targetPreAggSelectPartition.mergeWith(
                  sourcePreAggSelectPartition.getSerializableSummary()));
    }

    @Test
    public void merge_differentPreThreshold_throwsException() {
      PreAggSelectPartition targetPreAggSelectPartition =
          getPreAggSelectPartitionBuilderWithFields().preThreshold(1).build();
      PreAggSelectPartition sourcePreAggSelectPartition =
          getPreAggSelectPartitionBuilderWithFields().preThreshold(2).build();
      assertThrows(
          IllegalArgumentException.class,
          () ->
              targetPreAggSelectPartition.mergeWith(
                  sourcePreAggSelectPartition.getSerializableSummary()));
    }

    @Test
    public void merge_calledAfterComputeResult_onTargetCount_throwsException() {
      PreAggSelectPartition targetPreAggSelectPartition =
          getPreAggSelectPartitionBuilderWithFields().build();
      PreAggSelectPartition sourcePreAggSelectPartition =
          getPreAggSelectPartitionBuilderWithFields().build();

      targetPreAggSelectPartition.shouldKeepPartition();
      assertThrows(
          IllegalStateException.class,
          () ->
              targetPreAggSelectPartition.mergeWith(
                  sourcePreAggSelectPartition.getSerializableSummary()));
    }

    @Test
    public void merge_calledAfterComputeResult_onSourceCount_throwsException() {
      PreAggSelectPartition targetPreAggSelectPartition =
          getPreAggSelectPartitionBuilderWithFields().build();
      PreAggSelectPartition sourcePreAggSelectPartition =
          getPreAggSelectPartitionBuilderWithFields().build();

      sourcePreAggSelectPartition.shouldKeepPartition();
      assertThrows(
          IllegalStateException.class,
          () ->
              targetPreAggSelectPartition.mergeWith(
                  sourcePreAggSelectPartition.getSerializableSummary()));
    }

    @Test
    public void sumExpPowers_oneValue() {
      double expectedResult = 2.718281828459045;
      double actualResult = preAggSelectPartition.sumExpPowers(1, 1, 1);
      assertThat(actualResult)
          .isWithin(getTolerance(expectedResult, actualResult))
          .of(expectedResult);
    }

    @Test
    public void sumExpPowers_threeValues() {
      double expectedResult = 11.107337927389695;
      double actualResult = preAggSelectPartition.sumExpPowers(1, 0, 3);
      assertThat(actualResult)
          .isWithin(getTolerance(expectedResult, actualResult))
          .of(expectedResult);
    }

    @Test
    public void sumExpPowers_negativeMinPower() {
      double expectedResult = 0.5032147244080551;
      double actualResult = preAggSelectPartition.sumExpPowers(1, -2, 2);
      assertThat(actualResult)
          .isWithin(getTolerance(expectedResult, actualResult))
          .of(expectedResult);
    }

    @Test
    public void sumExpPowers_nonIntegerEpsilon() {
      double expectedResult = 13;
      double actualResult = preAggSelectPartition.sumExpPowers(Math.log(3), 0, 3);
      assertThat(actualResult)
          .isWithin(getTolerance(expectedResult, actualResult))
          .of(expectedResult);
    }

    @Test
    public void sumExpPowers_largeEpsilon_positivePowers() {
      assertThat(preAggSelectPartition.sumExpPowers(Double.MAX_VALUE, 1, 5)).isPositiveInfinity();
    }

    @Test
    public void sumExpPowers_largeEpsilon_negativePowers() {
      double expectedResult = 0;
      double actualResult = preAggSelectPartition.sumExpPowers(Double.MAX_VALUE, -5, 3);
      assertThat(actualResult)
          .isWithin(getTolerance(expectedResult, actualResult))
          .of(expectedResult);
    }

    @Test
    public void sumExpPowers_verySmallEpsilon() {
      double expectedResult = 100;
      // exp(-epsilon) = 0
      double actualResult = preAggSelectPartition.sumExpPowers(1e-100, 0, 100);
      assertThat(actualResult)
          .isWithin(getTolerance(expectedResult, actualResult))
          .of(expectedResult);
    }

    @Test
    public void shouldSelectPartition_zeroIds_neverTrue() {
      assertThat(preAggSelectPartition.getKeepPartitionProbability()).isEqualTo(0);
    }

    /**
     * This test is non-deterministic. The probability of keeping a partition with that parameters
     * is equal to 0.3. The number of trials is equal to 100000. The binomial distribution with
     * parameters (100000, 0.3) yields a value in the interval (29017, 30989) with probability at
     * least 1 - 1e-11. Dividing the interval endpoints by 100000, we see that the average is within
     * 0.3 +/- 0.01 with high probability. Running this test has a 1e-11 flakiness rate, so we retry
     * up to 2 times upon failure to drive the flakiness rate down to the a truly negligeable:
     * (1e-11)^3 = 1e-33 flakiness rate.
     */
    @Test
    public void shouldKeepPartition_oneId_sometimesTrue() {
      int numTrials = 100_000;
      double expectedSelectionRate = 0.3;
      double tolerance = 0.01;

      List<Double> actualSelectionRates = new ArrayList<>();
      for (int retry = 0; retry < 2; ++retry) {
        int selections = 0;
        for (int i = 0; i < numTrials; ++i) {
          PreAggSelectPartition preAggSelectPartition =
              getPreAggSelectPartitionBuilderWithFields().delta(0.3).build();

          preAggSelectPartition.increment();
          if (preAggSelectPartition.shouldKeepPartition()) {
            selections++;
          }
        }

        double actualSelectionRate = (double) selections / (double) numTrials;
        actualSelectionRates.add(actualSelectionRate);
      }

      // anyMatch is used to make the test pass if at least one of the retries succeeds.
      assertThat(
              actualSelectionRates.stream()
                  .anyMatch(
                      actualSelectionRate ->
                          abs(expectedSelectionRate - actualSelectionRate) <= tolerance))
          .isTrue();
    }

    /**
     * This test is non-deterministic. The keep partition probability with that parameters is equal
     * to 0.3. The number of trials is equal to 100000. The binomial distribution with parameters
     * (100000, 0.3) yields a value in the interval (29017, 30989) with probability at least 1 -
     * 1e-11. Dividing the interval endpoints by 100000, we see that the average is within 0.3 +/-
     * 0.01 with high probability. Running this test has a 1e-11 flakiness rate, so we retry up to 2
     * times upon failure to drive the flakiness rate down to the a truly negligeable: (1e-11)^3 =
     * 1e-33 flakiness rate.
     */
    @Test
    public void shouldKeepPartition_oneId_twoPartitionsContributed_sometimesTrue() {
      int numTrials = 100_000;
      double expectedSelectionRate = 0.3;
      double tolerance = 0.01;

      List<Double> actualSelectionRates = new ArrayList<>();
      for (int retry = 0; retry < 2; ++retry) {
        int selections = 0;
        for (int i = 0; i < numTrials; ++i) {
          PreAggSelectPartition preAggSelectPartition =
              getPreAggSelectPartitionBuilderWithFields()
                  .delta(0.6)
                  .maxPartitionsContributed(2)
                  .build();

          preAggSelectPartition.increment();
          if (preAggSelectPartition.shouldKeepPartition()) {
            selections++;
          }
        }

        double actualSelectionRate = (double) selections / (double) numTrials;
        actualSelectionRates.add(actualSelectionRate);
      }

      // anyMatch is used to make the test pass if at least one of the retries succeeds.
      assertThat(
              actualSelectionRates.stream()
                  .anyMatch(
                      actualSelectionRate ->
                          abs(expectedSelectionRate - actualSelectionRate) <= tolerance))
          .isTrue();
    }

    /** This test is non-deterministic. It might fail with probability at most 1e-10. */
    @Test
    public void shouldKeepPartition_gaussianPartitionSelection_keepsLargePartitions() {
      PreAggSelectPartition preAggSelectPartition =
          PreAggSelectPartition.builder()
              .epsilon(5)
              // delta is split equally between noise and threshold
              .delta(2e-15)
              // maxPartitionsContributed >= 3 triggers Gaussian thresholding
              .maxPartitionsContributed(5)
              .build();
      preAggSelectPartition.incrementBy(55);

      // For epsilon = 5, noiseDelta = thresholdDelta = 1e-15, l0Sensitivity = 5
      // the noise is at most +-24 with probability (1 - 1e-10) and the threshold is 30.
      // Hence, in the majority of the cases, a partition with >= 55 privacy IDs should be
      // kept after noise addition.
      // To compute the threshold value, use GaussianNoise.computeQuantile(...).
      assertThat(preAggSelectPartition.shouldKeepPartition()).isTrue();
    }

    /**
     * Similar to {@link #shouldKeepPartition_gaussianPartitionSelection_keepsLargePartitions} but
     * this sets a larger initial number of unique contributions and a threshold value such that the
     * thresholded number of contributions is the same. This test is non-deterministic. It might
     * fail with probability at most 1e-10.
     */
    @Test
    public void shouldKeepPartition_gaussianPartitionSelectionPreThreshold_keepsLargePartitions() {
      PreAggSelectPartition preAggSelectPartition =
          PreAggSelectPartition.builder()
              .epsilon(5)
              // delta is split equally between noise and threshold
              .delta(2e-15)
              // maxPartitionsContributed >= 3 triggers Gaussian thresholding
              .maxPartitionsContributed(5)
              .preThreshold(6)
              .build();
      preAggSelectPartition.incrementBy(60);

      // For epsilon = 5, noiseDelta = thresholdDelta = 1e-15, l0Sensitivity = 5
      // the noise is at most +-24 with probability (1 - 1e-10) and the threshold is 30.
      // Hence, in the majority of the cases, a partition with >= 55 privacy IDs should be
      // kept after noise addition.
      // With pre-thresholding = 6, contributions = (60 - (6 - 1)) = 55.
      // To compute the threshold value, use GaussianNoise.computeQuantile(...).
      assertThat(preAggSelectPartition.shouldKeepPartition()).isTrue();
    }

    /** This test is non-deterministic. It might fail with probability at most 1e-10. */
    @Test
    public void shouldKeepPartition_gaussianPartitionSelection_dropsSmallPartitions() {
      PreAggSelectPartition preAggSelectPartition =
          PreAggSelectPartition.builder()
              .epsilon(5)
              // delta is split equally between noise and threshold
              .delta(2e-15)
              // maxPartitionsContributed >= 3 triggers Gaussian thresholding
              .maxPartitionsContributed(5)
              .build();
      preAggSelectPartition.incrementBy(5);

      // For epsilon = 5, noiseDelta = thresholdDelta = 1e-15, l0Sensitivity = 5
      // the noise is at most +-24 with probability (1 - 1e-10) and the threshold is 30.
      // Hence, in the majority of the cases, a partition with <= 5 privacy IDs should be
      // dropped after noise addition.
      // To compute the threshold value, use GaussianNoise.computeQuantile(...).
      assertThat(preAggSelectPartition.shouldKeepPartition()).isFalse();
    }

    /**
     * Similar to {@link #shouldKeepPartition_gaussianPartitionSelection_dropsSmallPartitions} but
     * this sets a larger initial number of unique contributions and a threshold value such that the
     * thresholded number of contributions is the same. This test is non-deterministic. It might
     * fail with probability at most 1e-10.
     */
    @Test
    public void shouldKeepPartition_gaussianPartitionSelectionPreThreshold_dropsSmallPartitions() {
      PreAggSelectPartition preAggSelectPartition =
          PreAggSelectPartition.builder()
              .epsilon(5)
              // delta is split equally between noise and threshold
              .delta(2e-15)
              // maxPartitionsContributed >= 3 triggers Gaussian thresholding
              .maxPartitionsContributed(5)
              .preThreshold(6)
              .build();
      preAggSelectPartition.incrementBy(10);

      // For epsilon = 5, noiseDelta = thresholdDelta = 1e-15, l0Sensitivity = 5
      // the noise is at most +-24 with probability (1 - 1e-10) and the threshold is 30.
      // Hence, in the majority of the cases, a partition with <= 5 privacy IDs should be
      // dropped after noise addition.
      // With pre-thresholding = 6, contributions = (10 - (6 - 1)) = 5.
      // To compute the threshold value, use GaussianNoise.computeQuantile(...).
      assertThat(preAggSelectPartition.shouldKeepPartition()).isFalse();
    }

    /**
     * Keep partition probability for that parameters is equal to 1. So the algorithm should always
     * return true for such partitions.
     */
    @Test
    public void shouldKeepPartition_fourIds_alwaysTrue() {
      PreAggSelectPartition preAggSelectPartition =
          getPreAggSelectPartitionBuilderWithFields().delta(0.3).build();

      preAggSelectPartition.increment();
      preAggSelectPartition.increment();
      preAggSelectPartition.increment();
      preAggSelectPartition.increment();

      assertThat(preAggSelectPartition.shouldKeepPartition()).isTrue();
    }

    /**
     * Set pre-thresholding larger than the number of contributions, it should return false
     * deterministically.
     */
    @Test
    public void shouldKeepPartition_inputValueOneLessThanPreThreshold_returnsFalse() {
      PreAggSelectPartition preAggSelectPartition =
          getPreAggSelectPartitionBuilderWithFields().preThreshold(2).build();

      preAggSelectPartition.increment();

      assertThat(preAggSelectPartition.shouldKeepPartition()).isFalse();
    }

    /**
     * Similar logic to {@link #shouldKeepPartition_oneId_sometimesTrue} but this tests
     * preThresholding on 100 unique user contributions because pre-thresholding decrements the
     * number of unique user contributions.
     */
    @Test
    public void shouldKeepPartition_nonDefaultPreThreshold_hasExpectedSelectionRate() {
      int numTrials = 100_000;
      double expectedSelectionRate = 0.3;
      double tolerance = 0.01;

      List<Double> actualSelectionRates = new ArrayList<>();
      for (int retry = 0; retry < 2; ++retry) {
        int selections = 0;
        for (int i = 0; i < numTrials; ++i) {
          PreAggSelectPartition preAggSelectPartition =
              getPreAggSelectPartitionBuilderWithFields().delta(0.3).preThreshold(100).build();

          preAggSelectPartition.incrementBy(100);
          // With pre-thresholding = 100, contributions = (100 - (100 - 1)) = 1
          if (preAggSelectPartition.shouldKeepPartition()) {
            selections++;
          }
        }

        double actualSelectionRate = (double) selections / (double) numTrials;
        actualSelectionRates.add(actualSelectionRate);
      }

      // anyMatch is used to make the test pass if at least one of the retries succeeds.
      assertThat(
              actualSelectionRates.stream()
                  .anyMatch(
                      actualSelectionRate ->
                          abs(expectedSelectionRate - actualSelectionRate) <= tolerance))
          .isTrue();
    }
  }

  @RunWith(Parameterized.class)
  public static class KeepPartitionProbabilityTests {
    private final KeepPartitionProbabilityTestCase testCase;

    public KeepPartitionProbabilityTests(KeepPartitionProbabilityTestCase testCase) {
      this.testCase = testCase;
    }

    @Test
    public void keepPartitionProbability() {
      PreAggSelectPartition preAggSelectPartition =
          PreAggSelectPartition.builder()
              .epsilon(testCase.epsilon())
              .delta(testCase.delta())
              .maxPartitionsContributed(testCase.maxPartitionsContributed())
              .build();

      for (int i = 0; i < testCase.idsCount(); ++i) {
        preAggSelectPartition.increment();
      }

      double actualProbability = preAggSelectPartition.getKeepPartitionProbability();
      double tolerance = getTolerance(actualProbability, testCase.expectedProbability());
      assertWithMessage(
              "Pr[ε = %s, δ = %s, maxPartitionsContributed = %s, idsCount = %s] = %s",
              testCase.epsilon(),
              testCase.delta(),
              testCase.maxPartitionsContributed(),
              testCase.idsCount(),
              actualProbability)
          .that(actualProbability)
          .isWithin(tolerance)
          .of(testCase.expectedProbability());
    }

    @Parameterized.Parameters(name = "{index}: = failed")
    public static Iterable<Object> getKeepPartitionProbabilityTestCases() {
      // seed is equal to 1 to make it deterministic.
      Random random = new Random(1);

      return Arrays.asList(
          // ε = ln2, δ = 0.1, 1 partition contributed, idsCount = 0 .. 5 .. (for counts >= 5 the
          // probability is always = 1)
          KeepPartitionProbabilityTestCase.create(EPSILON, DELTA, ONE_PARTITION_CONTRIBUTED, 0, 0),
          KeepPartitionProbabilityTestCase.create(
              EPSILON, DELTA, ONE_PARTITION_CONTRIBUTED, 1, 0.1),
          KeepPartitionProbabilityTestCase.create(
              EPSILON, DELTA, ONE_PARTITION_CONTRIBUTED, 2, 0.3),
          KeepPartitionProbabilityTestCase.create(
              EPSILON, DELTA, ONE_PARTITION_CONTRIBUTED, 3, 0.7),
          KeepPartitionProbabilityTestCase.create(
              EPSILON, DELTA, ONE_PARTITION_CONTRIBUTED, 4, 0.9),
          KeepPartitionProbabilityTestCase.create(EPSILON, DELTA, ONE_PARTITION_CONTRIBUTED, 5, 1),
          KeepPartitionProbabilityTestCase.create(
              EPSILON, DELTA, ONE_PARTITION_CONTRIBUTED, random.nextInt(1000) + 5, 1),
          // ε = 2 * ln2, δ = 0.2, 2 partitions contributed, idsCount = 0 .. 5 .. (for counts >= 5
          // the probability is always = 1)
          KeepPartitionProbabilityTestCase.create(
              2 * EPSILON, 2 * DELTA, 2 * ONE_PARTITION_CONTRIBUTED, 0, 0),
          KeepPartitionProbabilityTestCase.create(
              2 * EPSILON, 2 * DELTA, 2 * ONE_PARTITION_CONTRIBUTED, 1, 0.1),
          KeepPartitionProbabilityTestCase.create(
              2 * EPSILON, 2 * DELTA, 2 * ONE_PARTITION_CONTRIBUTED, 2, 0.3),
          KeepPartitionProbabilityTestCase.create(
              2 * EPSILON, 2 * DELTA, 2 * ONE_PARTITION_CONTRIBUTED, 3, 0.7),
          KeepPartitionProbabilityTestCase.create(
              2 * EPSILON, 2 * DELTA, 2 * ONE_PARTITION_CONTRIBUTED, 4, 0.9),
          KeepPartitionProbabilityTestCase.create(
              2 * EPSILON, 2 * DELTA, 2 * ONE_PARTITION_CONTRIBUTED, 5, 1),
          KeepPartitionProbabilityTestCase.create(
              2 * EPSILON, 2 * DELTA, 2 * ONE_PARTITION_CONTRIBUTED, random.nextInt(1000) + 5, 1),
          // ε = ln(1.5), δ = 0.1, 1 partition contributed, idsCount = 0 .. 7 .. (for counts >= 7
          // the probability is always = 1)
          KeepPartitionProbabilityTestCase.create(
              LOW_EPSILON, DELTA, ONE_PARTITION_CONTRIBUTED, 0, 0),
          KeepPartitionProbabilityTestCase.create(
              LOW_EPSILON, DELTA, ONE_PARTITION_CONTRIBUTED, 1, 0.1),
          KeepPartitionProbabilityTestCase.create(
              LOW_EPSILON, DELTA, ONE_PARTITION_CONTRIBUTED, 2, 0.25),
          KeepPartitionProbabilityTestCase.create(
              LOW_EPSILON, DELTA, ONE_PARTITION_CONTRIBUTED, 3, 0.475),
          KeepPartitionProbabilityTestCase.create(
              LOW_EPSILON, DELTA, ONE_PARTITION_CONTRIBUTED, 4, 0.716666666667),
          KeepPartitionProbabilityTestCase.create(
              LOW_EPSILON, DELTA, ONE_PARTITION_CONTRIBUTED, 5, 0.877777777778),
          KeepPartitionProbabilityTestCase.create(
              LOW_EPSILON, DELTA, ONE_PARTITION_CONTRIBUTED, 6, 0.985185185185),
          KeepPartitionProbabilityTestCase.create(
              LOW_EPSILON, DELTA, ONE_PARTITION_CONTRIBUTED, 7, 1),
          KeepPartitionProbabilityTestCase.create(
              LOW_EPSILON, DELTA, ONE_PARTITION_CONTRIBUTED, random.nextInt(1000) + 7, 1),
          // ε = 50, δ = 1e-200, 1 partition contributed, idsCount = 0 .. 11 .. (for counts >= 11
          // the probability is always = 1)
          KeepPartitionProbabilityTestCase.create(
              HIGH_EPSILON, LOW_DELTA, ONE_PARTITION_CONTRIBUTED, 0, 0),
          KeepPartitionProbabilityTestCase.create(
              HIGH_EPSILON, LOW_DELTA, ONE_PARTITION_CONTRIBUTED, 1, 1e-200),
          KeepPartitionProbabilityTestCase.create(
              HIGH_EPSILON, LOW_DELTA, ONE_PARTITION_CONTRIBUTED, 2, 5.184706e-179),
          KeepPartitionProbabilityTestCase.create(
              HIGH_EPSILON, LOW_DELTA, ONE_PARTITION_CONTRIBUTED, 3, 2.688117e-157),
          KeepPartitionProbabilityTestCase.create(
              HIGH_EPSILON, LOW_DELTA, ONE_PARTITION_CONTRIBUTED, 4, 1.393710e-135),
          KeepPartitionProbabilityTestCase.create(
              HIGH_EPSILON, LOW_DELTA, ONE_PARTITION_CONTRIBUTED, 5, 7.225974e-114),
          KeepPartitionProbabilityTestCase.create(
              HIGH_EPSILON, LOW_DELTA, ONE_PARTITION_CONTRIBUTED, 6, 3.746455e-92),
          KeepPartitionProbabilityTestCase.create(
              HIGH_EPSILON, LOW_DELTA, ONE_PARTITION_CONTRIBUTED, 7, 1.942426e-70),
          KeepPartitionProbabilityTestCase.create(
              HIGH_EPSILON, LOW_DELTA, ONE_PARTITION_CONTRIBUTED, 8, 1.007091e-48),
          KeepPartitionProbabilityTestCase.create(
              HIGH_EPSILON, LOW_DELTA, ONE_PARTITION_CONTRIBUTED, 9, 5.221470e-27),
          KeepPartitionProbabilityTestCase.create(
              HIGH_EPSILON, LOW_DELTA, ONE_PARTITION_CONTRIBUTED, 10, 2.707178e-05),
          KeepPartitionProbabilityTestCase.create(
              HIGH_EPSILON, LOW_DELTA, ONE_PARTITION_CONTRIBUTED, 11, 1),
          KeepPartitionProbabilityTestCase.create(
              HIGH_EPSILON, LOW_DELTA, ONE_PARTITION_CONTRIBUTED, random.nextInt(1000) + 11, 1),
          // ε = Double.MAX_VALUE, δ = 1e-200, 1 partition contributed, idsCount = 0 .. 2 .. (for
          // counts >= 2 the probability is always = 1)
          KeepPartitionProbabilityTestCase.create(
              Double.MAX_VALUE, LOW_DELTA, ONE_PARTITION_CONTRIBUTED, 0, 0),
          KeepPartitionProbabilityTestCase.create(
              Double.MAX_VALUE, LOW_DELTA, ONE_PARTITION_CONTRIBUTED, 1, 1e-200),
          KeepPartitionProbabilityTestCase.create(
              Double.MAX_VALUE, LOW_DELTA, ONE_PARTITION_CONTRIBUTED, 2, 1),
          KeepPartitionProbabilityTestCase.create(
              Double.MAX_VALUE, LOW_DELTA, ONE_PARTITION_CONTRIBUTED, random.nextInt(1000) + 2, 1));
    }

    @AutoValue
    public abstract static class KeepPartitionProbabilityTestCase implements Serializable {
      static KeepPartitionProbabilityTestCase create(
          double epsilon,
          double delta,
          int maxPartitionsContributed,
          int idsCount,
          double expectedProbability) {
        return new AutoValue_PreAggSelectPartitionTest_KeepPartitionProbabilityTests_KeepPartitionProbabilityTestCase(
            epsilon, delta, maxPartitionsContributed, idsCount, expectedProbability);
      }

      abstract double epsilon();

      abstract double delta();

      abstract int maxPartitionsContributed();

      abstract int idsCount();

      abstract double expectedProbability();
    }
  }

  /**
   * Note that {@link PreAggSelectPartitionSummary} isn't visible to the actual clients, who only
   * see an opaque {@code byte[]} blob. Here, we parse said blob to perform whitebox testing, to
   * verify some expectations of the blob's content. We do this because achieving good coverage with
   * pure behaviour testing (i.e., blackbox testing) isn't possible.
   */
  private static PreAggSelectPartitionSummary getSummary(
      PreAggSelectPartition preAggSelectPartition) {
    byte[] nonParsedSummary = preAggSelectPartition.getSerializableSummary();
    try {
      // We are deliberately ignoring the warning from JavaCodeClarity because
      // ExtensionRegistry.getGeneratedRegistry() breaks kokoro tests, is not open-sourced, and
      // there is no simple external alternative. However, we don't (and it is unlikely we will) use
      // extensions in Summary protos, so we do not expect this to be a problem.
      return PreAggSelectPartitionSummary.parseFrom(nonParsedSummary);
    } catch (InvalidProtocolBufferException pbe) {
      throw new IllegalArgumentException(pbe);
    }
  }

  private static PreAggSelectPartition.Params.Builder getPreAggSelectPartitionBuilderWithFields() {
    return PreAggSelectPartition.builder()
        .epsilon(EPSILON)
        .delta(DELTA)
        .maxPartitionsContributed(1);
  }

  private static double getTolerance(double x, double y) {
    double maxMagnitude = max(abs(x), abs(y));
    return 1e-6 * maxMagnitude;
  }
}

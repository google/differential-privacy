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
import static com.google.differentialprivacy.SummaryOuterClass.MechanismType.GAUSSIAN;
import static com.google.differentialprivacy.SummaryOuterClass.MechanismType.LAPLACE;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;
import static org.mockito.ArgumentMatchers.anyDouble;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.math.Stats;
import com.google.differentialprivacy.SummaryOuterClass.CountSummary;
import com.google.protobuf.InvalidProtocolBufferException;
import java.util.Collection;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnit;
import org.mockito.junit.MockitoRule;

/**
 * Tests behavior of {@link Count}. The test mocks a {@link Noise} instance to always generate zero
 * noise.
 *
 * <p>Statistical and DP properties of the algorithm are tested in {@link CountDpTest}.
 */
@RunWith(JUnit4.class)
public class CountTest {

  private static final double EPSILON = 0.123;
  private static final double DELTA = 0.123;
  private static final int NUM_SAMPLES = 100000;
  private static final double LN_3 = Math.log(3.0);

  @Mock private Noise noise;
  @Mock private Collection<Double> hugeCollection;
  @Rule public final MockitoRule mocks = MockitoJUnit.rule();

  private Count count;

  @Before
  public void setUp() {
    // Mock the noise mechanism so that it does not add any noise.
    when(noise.addNoise(anyLong(), anyInt(), anyLong(), anyDouble(), anyDouble()))
        .thenAnswer(invocation -> invocation.getArguments()[0]);
    // Tests that use serialization need to access to the type of the noise they use. Because the
    // tests don't rely on a specific noise type, we arbitrarily return Gaussian.
    when(noise.getMechanismType()).thenReturn(GAUSSIAN);

    count =
        Count.builder()
            .epsilon(EPSILON)
            .delta(DELTA)
            .noise(noise)
            .maxPartitionsContributed(1)
            .build();

    when(hugeCollection.size()).thenReturn(Integer.MAX_VALUE);
  }

  @Test
  public void increment() {
    int entriesCount = 10000;
    for (int i = 0; i < entriesCount; ++i) {
      count.increment();
    }

    assertThat(count.computeResult()).isEqualTo(entriesCount);
  }

  @Test
  public void incrementBy() {
    count.incrementBy(9);

    assertThat(count.computeResult()).isEqualTo(9);
  }

  @Test
  public void incrementAndIncrementByAllTogether() {
    count.increment();
    count.incrementBy(7);
    count.increment();

    assertThat(count.computeResult()).isEqualTo(9);
  }

  @Test
  public void computeResult_noIncrements_returnsZero() {
    assertThat(count.computeResult()).isEqualTo(0);
  }

  // An attempt to compute the count several times should result in an exception.
  @Test
  public void computeResult_multipleCalls_throwsException() {
    count.increment();

    count.computeResult();
    assertThrows(IllegalStateException.class, count::computeResult);
  }

  @Test
  public void incrementBy_hugeValues_dontOverflow() {
    count.incrementBy(Integer.MAX_VALUE);
    count.incrementBy(Integer.MAX_VALUE);
    count.increment();
    count.increment();
    count.increment();

    long expected = Integer.MAX_VALUE * 2L + 3L;
    long actualResult = count.computeResult();
    assertThat(actualResult).isEqualTo(expected);
  }

  @Test
  public void computeResult_callsNoiseCorrectly() {
    int l0Sensitivity = 1;
    count =
        Count.builder()
            .epsilon(EPSILON)
            .delta(DELTA)
            .noise(noise)
            .maxPartitionsContributed(l0Sensitivity)
            .maxContributionsPerPartition(5)
            .build();
    count.increment();
    count.computeResult();

    verify(noise)
        .addNoise(
            eq(1L), // count of added entries = 1
            eq(l0Sensitivity),
            eq(/* lInfSensitivity = maxContributionsPerPartition = 5 */ 5L),
            eq(EPSILON),
            eq(DELTA));
  }

  @Test
  public void computeResult_addsNoise() {
    // Mock the noise mechanism so that it always generates 100.0.
    when(noise.addNoise(anyLong(), anyInt(), anyLong(), anyDouble(), anyDouble()))
        .thenAnswer(invocation -> (long) invocation.getArguments()[0] + 100);
    count =
        Count.builder()
            .epsilon(EPSILON)
            .delta(DELTA)
            .noise(noise)
            .maxPartitionsContributed(1)
            .build();

    count.increment();
    assertThat(count.computeResult()).isEqualTo(101); // value (1) + noise (100) = 101
  }

  @Test
  public void getSerializableSummary_copiesCountCorrectly() {
    count.increment();
    count.incrementBy(9);

    CountSummary summary = getSummary(count);
    assertThat(summary.getCount()).isEqualTo(10);
  }

  @Test
  public void getSerializableSummary_copiesZeroCountCorrectly() {
    CountSummary summary = getSummary(count);
    assertThat(summary.getCount()).isEqualTo(0);
  }

  @Test
  public void getSerializableSummary_copiesMaxIntCountCorrectly() {
    count.incrementBy(Integer.MAX_VALUE);

    CountSummary summary = getSummary(count);
    assertThat(summary.getCount()).isEqualTo(Integer.MAX_VALUE);
  }

  @Test
  public void getSerializableSummary_copiesMinIntCountCorrectly() {
    count.incrementBy(Integer.MIN_VALUE);

    CountSummary summary = getSummary(count);
    assertThat(summary.getCount()).isEqualTo(Integer.MIN_VALUE);
  }

  @Test
  public void getSerializableSummary_calledAfterComputeResult_throwsException() {
    count.computeResult();
    assertThrows(IllegalStateException.class, () -> count.getSerializableSummary());
  }

  @Test
  public void getSerializableSummary_twoCalls_throwsException() {
    count.getSerializableSummary();
    assertThrows(IllegalStateException.class, () -> count.getSerializableSummary());
  }

  @Test
  public void computeResult_calledAfterSerialize_throwsException() {
    count.getSerializableSummary();
    assertThrows(IllegalStateException.class, () -> count.computeResult());
  }

  @Test
  public void getSerializableSummary_copiesEpsilonCorrectly() {
    count = getCountBuilderWithFields().epsilon(EPSILON).build();
    CountSummary summary = getSummary(count);
    assertThat(summary.getEpsilon()).isEqualTo(EPSILON);
  }

  @Test
  public void getSerializableSummary_copiesDeltaCorrectly() {
    count = getCountBuilderWithFields().delta(DELTA).build();
    CountSummary summary = getSummary(count);
    assertThat(summary.getDelta()).isEqualTo(DELTA);
  }

  @Test
  public void getSerializableSummary_copiesGaussianNoiseCorrectly() {
    count = getCountBuilderWithFields().noise(new GaussianNoise()).build();
    CountSummary summary = getSummary(count);
    assertThat(summary.getMechanismType()).isEqualTo(GAUSSIAN);
  }

  @Test
  public void getSerializableSummary_copiesLaplaceNoiseCorrectly() {
    count = getCountBuilderWithFields().noise(new LaplaceNoise()).delta(null).build();
    CountSummary summary = getSummary(count);
    assertThat(summary.getMechanismType()).isEqualTo(LAPLACE);
  }

  @Test
  public void getSerializableSummary_copiesMaxPartitionsContributedCorrectly() {
    int maxPartitionsContributed = 150;
    count = getCountBuilderWithFields().maxPartitionsContributed(maxPartitionsContributed).build();
    CountSummary summary = getSummary(count);
    assertThat(summary.getMaxPartitionsContributed()).isEqualTo(maxPartitionsContributed);
  }

  @Test
  public void getSerializableSummary_copiesMaxContributionsPerPartitionCorrectly() {
    int maxContributionsPerPartition = 150;
    count =
        getCountBuilderWithFields()
            .maxContributionsPerPartition(maxContributionsPerPartition)
            .build();
    CountSummary summary = getSummary(count);
    assertThat(summary.getMaxContributionsPerPartition()).isEqualTo(maxContributionsPerPartition);
  }

  @Test
  public void merge_basicExample_sumsCounts() {
    Count targetCount = getCountBuilderWithFields().build();
    Count sourceCount = getCountBuilderWithFields().build();

    targetCount.increment();
    sourceCount.increment();

    targetCount.mergeWith(sourceCount.getSerializableSummary());

    assertThat(targetCount.computeResult()).isEqualTo(2);
  }

  @Test
  public void merge_calledTwice_sumsCounts() {
    Count targetCount = getCountBuilderWithFields().build();
    Count sourceCount1 = getCountBuilderWithFields().build();
    Count sourceCount2 = getCountBuilderWithFields().build();

    targetCount.increment();
    sourceCount1.incrementBy(2);
    sourceCount2.incrementBy(3);

    targetCount.mergeWith(sourceCount1.getSerializableSummary());
    targetCount.mergeWith(sourceCount2.getSerializableSummary());

    assertThat(targetCount.computeResult()).isEqualTo(6);
  }

  @Test
  public void merge_nullDelta_noException() {
    Count targetCount = getCountBuilderWithFields().noise(new LaplaceNoise()).delta(null).build();
    Count sourceCount = getCountBuilderWithFields().noise(new LaplaceNoise()).delta(null).build();
    // no exception is thrown
    targetCount.mergeWith(sourceCount.getSerializableSummary());
  }

  @Test
  public void merge_differentEpsilon_throwsException() {
    Count targetCount = getCountBuilderWithFields().epsilon(EPSILON).build();
    Count sourceCount = getCountBuilderWithFields().epsilon(2 * EPSILON).build();
    assertThrows(
        IllegalArgumentException.class,
        () -> targetCount.mergeWith(sourceCount.getSerializableSummary()));
  }

  @Test
  public void merge_differentDelta_throwsException() {
    Count targetCount = getCountBuilderWithFields().delta(DELTA).build();
    Count sourceCount = getCountBuilderWithFields().delta(2 * DELTA).build();
    assertThrows(
        IllegalArgumentException.class,
        () -> targetCount.mergeWith(sourceCount.getSerializableSummary()));
  }

  @Test
  public void merge_differentNoise_throwsException() {
    Count targetCount = getCountBuilderWithFields().noise(new LaplaceNoise()).delta(null).build();
    Count sourceCount = getCountBuilderWithFields().noise(new GaussianNoise()).build();
    assertThrows(
        IllegalArgumentException.class,
        () -> targetCount.mergeWith(sourceCount.getSerializableSummary()));
  }

  @Test
  public void merge_differentMaxPartitionsContributed_throwsException() {
    Count targetCount = getCountBuilderWithFields().maxPartitionsContributed(1).build();
    Count sourceCount = getCountBuilderWithFields().maxPartitionsContributed(2).build();
    assertThrows(
        IllegalArgumentException.class,
        () -> targetCount.mergeWith(sourceCount.getSerializableSummary()));
  }

  @Test
  public void merge_differentMaxContributionsPerPartition_throwsException() {
    Count targetCount = getCountBuilderWithFields().maxContributionsPerPartition(1).build();
    Count sourceCount = getCountBuilderWithFields().maxContributionsPerPartition(2).build();
    assertThrows(
        IllegalArgumentException.class,
        () -> targetCount.mergeWith(sourceCount.getSerializableSummary()));
  }

  @Test
  public void merge_calledAfterComputeResult_onTargetCount_throwsException() {
    Count targetCount = getCountBuilderWithFields().build();
    Count sourceCount = getCountBuilderWithFields().build();

    targetCount.computeResult();
    assertThrows(
        IllegalStateException.class,
        () -> targetCount.mergeWith(sourceCount.getSerializableSummary()));
  }

  @Test
  public void merge_calledAfterComputeResult_onSourceCount_throwsException() {
    Count targetCount = getCountBuilderWithFields().build();
    Count sourceCount = getCountBuilderWithFields().build();

    sourceCount.computeResult();
    assertThrows(
        IllegalStateException.class,
        () -> targetCount.mergeWith(sourceCount.getSerializableSummary()));
  }

  @Test
  public void addNoise_gaussianNoiseDefaultParametersEmptyCount_isUnbiased() {
    Count.Params.Builder countBuilder =
        Count.builder()
            .epsilon(LN_3)
            .delta(0.00001)
            .maxPartitionsContributed(1)
            .noise(new GaussianNoise());

    testForBias(countBuilder, /* rawCount */ 0, /* (over) approximation of variance */ 11.9);
  }

  @Test
  public void addNoise_gaussianNoiseDifferentEpsilonEmptyCount_isUnbiased() {
    Count.Params.Builder countBuilder =
        Count.builder()
            .epsilon(2.0 * LN_3)
            .delta(0.00001)
            .maxPartitionsContributed(1)
            .noise(new GaussianNoise());

    testForBias(countBuilder, /* rawCount */ 0, /* (over) approximation of variance */ 3.5);
  }

  @Test
  public void addNoise_gaussianNoiseDifferentDeltaEmptyCount_isUnbiased() {
    Count.Params.Builder countBuilder =
        Count.builder()
            .epsilon(LN_3)
            .delta(0.01)
            .maxPartitionsContributed(1)
            .noise(new GaussianNoise());

    testForBias(countBuilder, /* rawCount */ 0, /* (over) approximation of variance */ 3.2);
  }

  @Test
  public void addNoise_gaussianNoiseDifferentContributionBoundEmptyCount_isUnbiased() {
    Count.Params.Builder countBuilder =
        Count.builder()
            .epsilon(LN_3)
            .delta(0.00001)
            .maxPartitionsContributed(25)
            .noise(new GaussianNoise());

    testForBias(countBuilder, /* rawCount */ 0, /* (over) approximation of variance */ 295.0);
  }

  @Test
  public void addNoise_gaussianNoiseDefaultParameters_isUnbiased() {
    Count.Params.Builder countBuilder =
        Count.builder()
            .epsilon(LN_3)
            .delta(0.00001)
            .maxPartitionsContributed(1)
            .noise(new GaussianNoise());

    testForBias(countBuilder, /* rawCount */ 3380636, /* (over) approximation of variance */ 11.9);
  }

  @Test
  public void addNoise_laplaceNoiseDefaultParametersEmptyCount_isUnbiased() {
    Count.Params.Builder countBuilder =
        Count.builder().epsilon(LN_3).maxPartitionsContributed(1).noise(new LaplaceNoise());

    testForBias(countBuilder, /* rawCount */ 0, /* (over) approximation of variance */ 1.8);
  }

  @Test
  public void addNoise_laplaceNoiseDifferentEpsilonEmptyCount_isUnbiased() {
    Count.Params.Builder countBuilder =
        Count.builder().epsilon(2.0 * LN_3).maxPartitionsContributed(1).noise(new LaplaceNoise());

    testForBias(countBuilder, /* rawCount */ 0, /* (over) approximation of variance */ 0.5);
  }

  @Test
  public void addNoise_laplaceNoiseDifferentContributionBoundEmptyCount_isUnbiased() {
    Count.Params.Builder countBuilder =
        Count.builder().epsilon(LN_3).maxPartitionsContributed(25).noise(new LaplaceNoise());

    testForBias(countBuilder, /* rawCount */ 0, /* (over) approximation of variance */ 1035.0);
  }

  @Test
  public void addNoise_laplaceNoiseDefaultParameters_isUnbiased() {
    Count.Params.Builder countBuilder =
        Count.builder().epsilon(LN_3).maxPartitionsContributed(1).noise(new LaplaceNoise());

    testForBias(countBuilder, /* rawCount */ 3380636, /* (over) approximation of variance */ 1.8);
  }

  private Count.Params.Builder getCountBuilderWithFields() {
    return Count.builder()
        .epsilon(EPSILON)
        .delta(DELTA)
        .noise(noise)
        .maxPartitionsContributed(1)
        // maxContributionsPerPartition is arbitrarily chosen
        .maxContributionsPerPartition(10);
  }

  /**
   * Note that {@link CountSummary} isn't visible to the actual clients, who only see an opaque
   * {@code byte[]} blob. Here, we parse said blob to perform whitebox testing, to verify some
   * expectations of the blob's content. We do this because achieving good coverage with pure
   * behaviour testing (i.e., blackbox testing) isn't possible.
   */
  private static CountSummary getSummary(Count count) {
    byte[] nonParsedSummary = count.getSerializableSummary();
    try {
      // We are deliberately ignoring the warning from JavaCodeClarity because
      // ExtensionRegistry.getGeneratedRegistry() breaks kokoro tests, is not open-sourced, and
      // there is no simple external alternative. However, we don't (and it is unlikely we will) use
      // extensions in Summary protos, so we do not expect this to be a problem.
      return CountSummary.parseFrom(nonParsedSummary);
    } catch (InvalidProtocolBufferException pbe) {
      throw new IllegalArgumentException(pbe);
    }
  }

  private static void testForBias(
      Count.Params.Builder countBuilder, int rawCount, double variance) {
    ImmutableList.Builder<Double> samples = ImmutableList.builder();
    for (int i = 0; i < NUM_SAMPLES; i++) {
      Count count = countBuilder.build();
      count.incrementBy(rawCount);
      samples.add((double) count.computeResult());
    }
    Stats stats = Stats.of(samples.build());

    // The tolerance is chosen according to the 99.9995% quantile of the anticipated distributions
    // of the sample mean. Thus, the test falsely rejects with a probability of 10^-5.
    double sampleTolerance = 4.41717 * Math.sqrt(variance / NUM_SAMPLES);
    // The DP count is considered unbiased if the expeted value (approximated by stats.mean()) is
    // equal to the raw count.
    assertThat(stats.mean()).isWithin(sampleTolerance).of((double) rawCount);
  }

  @Test
  public void computeConfidenceInterval_negativeBounds() {
    when(noise.computeConfidenceInterval(
            anyLong(), anyInt(), anyLong(), anyDouble(), anyDouble(), anyDouble()))
            .thenReturn(ConfidenceInterval.create(-5,-3));
    count.computeResult();

    // The result interval = (-5, -3), but count can't be negative, then it should be clamped to (0, 0).
    assertThat(count.computeConfidenceInterval(0.152145599))
            .isEqualTo(ConfidenceInterval.create(0, 0));
  }

  @Test
  public void computeConfidenceInterval_positiveBounds() {
    when(noise.computeConfidenceInterval(
            anyLong(), anyInt(), anyLong(), anyDouble(), anyDouble(), anyDouble()))
            .thenReturn(ConfidenceInterval.create(5,3));
    count.computeResult();

    // The result interval is positive, then it should not be clamped.
    assertThat(count.computeConfidenceInterval(0.152145599))
            .isEqualTo(ConfidenceInterval.create(5, 3));
  }

  @Test
  public void computeConfidenceInterval_negativeLowerBound() {
    when(noise.computeConfidenceInterval(
            anyLong(), anyInt(), anyLong(), anyDouble(), anyDouble(), anyDouble()))
            .thenReturn(ConfidenceInterval.create(-5,8));
    count.computeResult();

    // The result interval = (-5, 8), but count can't be equal to -5 then it should be clamped to (0, 8).
    assertThat(count.computeConfidenceInterval(0.152145599))
            .isEqualTo(ConfidenceInterval.create(0, 8));
  }

  @Test
  public void computeConfidenceInterval_infiniteBounds() {
    when(noise.computeConfidenceInterval(
            anyLong(), anyInt(), anyLong(), anyDouble(), anyDouble(), anyDouble()))
            .thenReturn(ConfidenceInterval.create(Double.NEGATIVE_INFINITY,Double.POSITIVE_INFINITY));
    count.computeResult();

    // The result interval = (0, POSITIVE_INFINITY), but because of the long type, it will be (0, Long.MAX_VALUE).
    assertThat(count.computeConfidenceInterval(0.152145599))
            .isEqualTo(ConfidenceInterval.create(0, 9.223372036854776E18));
  }

  @Test
  public void computeConfidenceInterval_gaussianTest() {
    when(noise.computeConfidenceInterval(anyLong(), anyInt(), anyLong(), anyDouble(), anyDouble(), anyDouble()))
            .thenAnswer(invocation -> new GaussianNoise().computeConfidenceInterval(
                    (Long) invocation.getArguments()[0],  (Integer) invocation.getArguments()[1], (Long) invocation.getArguments()[2],
                    (Double) invocation.getArguments()[3],  (Double) invocation.getArguments()[4],  (Double) invocation.getArguments()[5]));
    // Mock the noise mechanism.
    count =
            Count.builder()
                    .epsilon(0.5)
                    .delta(0.9)
                    .noise(noise)
                    .maxPartitionsContributed(15)
                    .build();
    count.increment();
    count.computeResult();

    // The result interval = (-1, 3), but count can't be equal to  0 then it should be clamped to 0.
    assertThat(count.computeConfidenceInterval(0.152145599)).isEqualTo(ConfidenceInterval.create(0,3));
  }

  @Test
  public void computeConfidenceInterval_laplaceTest() {
    when(noise.computeConfidenceInterval(anyLong(), anyInt(), anyLong(), anyDouble(), anyDouble(), anyDouble()))
            .thenAnswer(invocation -> new LaplaceNoise().computeConfidenceInterval(
                    (Long) invocation.getArguments()[0],  (Integer) invocation.getArguments()[1], (Long) invocation.getArguments()[2],
                    (Double) invocation.getArguments()[3],  null,  (Double) invocation.getArguments()[5]));
    // Mock the noise mechanism. Since noise is not Laplace, nor Gaussian, delta will be passed as a value instead of null, in order to pass the checks.
    count =
            Count.builder()
                    .epsilon(0.1)
                    .noise(noise)
                    .delta(0.5)
                    .maxPartitionsContributed(1)
                    .build();
    count.incrementBy(10);
    count.computeResult();

    assertThat(count.computeConfidenceInterval(0.5)).isEqualTo(ConfidenceInterval.create(3,17));
  }

  @Test
  public void throwError_Long_whenComputeResultNotCalled() {
    when(noise.computeConfidenceInterval(anyDouble(), anyInt(), anyDouble(), anyDouble(), anyDouble(), anyDouble()))
            .thenAnswer(invocation -> new LaplaceNoise().computeConfidenceInterval(
                    (Long) invocation.getArguments()[0],  (Integer) invocation.getArguments()[1], (Long) invocation.getArguments()[2],
                    (Double) invocation.getArguments()[3], null,  (Double) invocation.getArguments()[5]));
    // Mock the noise mechanism. Since noise is not Laplace, nor Gaussian, delta will be passed as a value instead of null, in order to pass the checks.
    count = Count.builder()
            .epsilon(0.1)
            .noise(noise)
            .delta(0.5)
            .maxPartitionsContributed(1)
            .build();
    count.incrementBy(10);
    Exception exception = assertThrows(IllegalStateException.class, () -> {
      count.computeConfidenceInterval(0.1554684);
    });

    String expectedMessage = "Noised count must be computed before calling this function.";
    String actualMessage = exception.getMessage();

    assertTrue(actualMessage.contains(expectedMessage));
  }

  @Test
  public void computeResult_callsNoiseCorrectly_ForConfidenceIntervals() {
    double alpha = 0.1524;
    when(noise.computeConfidenceInterval(anyLong(), anyInt(), anyLong(), anyDouble(), anyDouble(), anyDouble()))
            .thenAnswer(invocation -> new GaussianNoise().computeConfidenceInterval(
                    (Long) invocation.getArguments()[0],  (Integer) invocation.getArguments()[1], (Long) invocation.getArguments()[2],
                    (Double) invocation.getArguments()[3],  (Double) invocation.getArguments()[4],  (Double) invocation.getArguments()[5]));
    // Mock the noise mechanism.
    count = Count.builder()
            .epsilon(EPSILON)
            .noise(noise)
            .delta(DELTA)
            .maxPartitionsContributed(1)
            .maxContributionsPerPartition(1)
            .build();
    count.incrementBy(10);
    count.computeResult();
    ConfidenceInterval confInt = count.computeConfidenceInterval(alpha);

    verify(noise)
            .computeConfidenceInterval(
                    eq(10L),
                    eq(1),
                    eq(1L),
                    eq(EPSILON),
                    eq(DELTA),
                    eq(alpha));
  }
}
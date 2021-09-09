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
import static com.google.privacy.differentialprivacy.proto.SummaryOuterClass.MechanismType.GAUSSIAN;
import static com.google.privacy.differentialprivacy.proto.SummaryOuterClass.MechanismType.LAPLACE;
import static org.junit.Assert.assertThrows;
import static org.mockito.ArgumentMatchers.anyDouble;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.google.privacy.differentialprivacy.proto.SummaryOuterClass.CountSummary;
import com.google.privacy.differentialprivacy.proto.SummaryOuterClass.MechanismType;
import com.google.protobuf.InvalidProtocolBufferException;
import java.util.Collection;
import java.util.Optional;
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
 * <p>Statistical and DP properties of the algorithm are tested in {@link
 * com.google.privacy.differentialprivacy.statistical.CountDpTest}.
 */
@RunWith(JUnit4.class)
public class CountTest {

  private static final double EPSILON = 0.123;
  private static final double DELTA = 0.123;
  private static final double THRESHOLD_DELTA = 1e-10;

  @Mock private Noise noise;
  @Mock private Collection<Double> hugeCollection;
  @Rule public final MockitoRule mocks = MockitoJUnit.rule();

  private Count count;

  @Before
  public void setUp() {
    // Mock the noise mechanism so that it does not add any noise.
    when(noise.addNoise(anyLong(), anyInt(), anyLong(), anyDouble(), anyDouble()))
        .thenAnswer(invocation -> invocation.getArguments()[0]);
    when(noise.addNoise(anyLong(), anyInt(), anyLong(), anyDouble(), isNull()))
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
  public void increment_calledAfterComputeResult_throwsException() {
    count.computeResult();
    assertThrows(IllegalStateException.class, () -> count.increment());
  }

  @Test
  public void increment_calledAfterSerialize_throwsException() {
    count.getSerializableSummary();
    assertThrows(IllegalStateException.class, () -> count.increment());
  }

  @Test
  public void incrementBy() {
    count.incrementBy(9);

    assertThat(count.computeResult()).isEqualTo(9);
  }

  @Test
  public void incrementBy_ignoresNegativeValues() {
    count.incrementBy(-100);

    assertThat(count.computeResult()).isEqualTo(0);
  }

  @Test
  public void incrementBy_calledAfterComputeResult_throwsException() {
    count.computeResult();
    assertThrows(IllegalStateException.class, () -> count.incrementBy(1));
  }

  @Test
  public void incrementBy_calledAfterSerialize_throwsException() {
    count.getSerializableSummary();
    assertThrows(IllegalStateException.class, () -> count.incrementBy(1));
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

  @Test
  public void computeResult_multipleCalls_throwsException() {
    count.increment();

    count.computeResult();
    assertThrows(IllegalStateException.class, count::computeResult);
  }

  @Test
  public void computeResult_calledAfterSerialize_throwsException() {
    count.getSerializableSummary();
    assertThrows(IllegalStateException.class, () -> count.computeResult());
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
  public void getSerializableSummary_calledAfterComputeResult_throwsException() {
    count.computeResult();
    assertThrows(IllegalStateException.class, () -> count.getSerializableSummary());
  }

  @Test
  public void getSerializableSummary_multipleCalls_returnsSameSummary() {
    count =
        Count.builder()
            .epsilon(EPSILON)
            .noise(new LaplaceNoise())
            .maxPartitionsContributed(1)
            .maxContributionsPerPartition(1)
            .build();
    count.increment();
    byte[] summary1 = count.getSerializableSummary();
    byte[] summary2 = count.getSerializableSummary();
    assertThat(summary1).isEqualTo(summary2);
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
  public void mergeWith_basicExample_sumsCounts() {
    Count targetCount = getCountBuilderWithFields().build();
    Count sourceCount = getCountBuilderWithFields().build();

    targetCount.increment();
    sourceCount.increment();

    targetCount.mergeWith(sourceCount.getSerializableSummary());

    assertThat(targetCount.computeResult()).isEqualTo(2);
  }

  @Test
  public void mergeWith_calledTwice_sumsCounts() {
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
  public void mergeWith_epsilonMismatch_throwsException() {
    Count targetCount = getCountBuilderWithFields().epsilon(EPSILON).build();
    Count sourceCount = getCountBuilderWithFields().epsilon(2 * EPSILON).build();
    assertThrows(
        IllegalArgumentException.class,
        () -> targetCount.mergeWith(sourceCount.getSerializableSummary()));
  }

  @Test
  public void mergeWith_nullDelta_mergesWithoutException() {
    Count targetCount = getCountBuilderWithFields().noise(new LaplaceNoise()).delta(null).build();
    Count sourceCount = getCountBuilderWithFields().noise(new LaplaceNoise()).delta(null).build();
    // no exception is thrown
    targetCount.mergeWith(sourceCount.getSerializableSummary());
  }

  @Test
  public void mergeWith_deltaMismatch_throwsException() {
    Count targetCount = getCountBuilderWithFields().delta(DELTA).build();
    Count sourceCount = getCountBuilderWithFields().delta(2 * DELTA).build();
    assertThrows(
        IllegalArgumentException.class,
        () -> targetCount.mergeWith(sourceCount.getSerializableSummary()));
  }

  @Test
  public void mergeWith_noiseMismatch_throwsException() {
    Count targetCount = getCountBuilderWithFields().noise(new LaplaceNoise()).delta(null).build();
    Count sourceCount = getCountBuilderWithFields().noise(new GaussianNoise()).build();
    assertThrows(
        IllegalArgumentException.class,
        () -> targetCount.mergeWith(sourceCount.getSerializableSummary()));
  }

  @Test
  public void mergeWith_maxPartitionsContributedMismatch_throwsException() {
    Count targetCount = getCountBuilderWithFields().maxPartitionsContributed(1).build();
    Count sourceCount = getCountBuilderWithFields().maxPartitionsContributed(2).build();
    assertThrows(
        IllegalArgumentException.class,
        () -> targetCount.mergeWith(sourceCount.getSerializableSummary()));
  }

  @Test
  public void mergeWith_maxContributionsPerPartitionMismatch_throwsException() {
    Count targetCount = getCountBuilderWithFields().maxContributionsPerPartition(1).build();
    Count sourceCount = getCountBuilderWithFields().maxContributionsPerPartition(2).build();
    assertThrows(
        IllegalArgumentException.class,
        () -> targetCount.mergeWith(sourceCount.getSerializableSummary()));
  }

  @Test
  public void mergeWith_calledAfterComputeResult_throwsException() {
    Count targetCount = getCountBuilderWithFields().build();
    Count sourceCount = getCountBuilderWithFields().build();

    targetCount.computeResult();
    byte[] summary = sourceCount.getSerializableSummary();
    assertThrows(IllegalStateException.class, () -> targetCount.mergeWith(summary));
  }

  @Test
  public void mergeWith_calledAfterSerialization_throwsException() {
    Count targetCount = getCountBuilderWithFields().build();
    Count sourceCount = getCountBuilderWithFields().build();

    targetCount.getSerializableSummary();
    byte[] summary = sourceCount.getSerializableSummary();
    assertThrows(IllegalStateException.class, () -> targetCount.mergeWith(summary));
  }

  @Test
  public void computeThresholdedResult_computeResultAlreadyCalled_throwsException() {
    count.computeResult();
    IllegalStateException exception =
        assertThrows(
            IllegalStateException.class,
            () -> {
              count.computeThresholdedResult(THRESHOLD_DELTA);
            });
    assertThat(exception).hasMessageThat().contains("DP count cannot be computed.");
  }

  @Test
  public void computeThresholdedResult_negativeThresholdDelta_throwsException() {
    IllegalArgumentException exception =
        assertThrows(IllegalArgumentException.class, () -> count.computeThresholdedResult(-2.0));
    assertThat(exception).hasMessageThat().startsWith("delta must be > 0 and < 1.");
  }

  @Test
  public void count_computeThresholdedResult_thresholdDeltaNaN_throwsException() {
    Exception e =
        assertThrows(
            IllegalArgumentException.class, () -> count.computeThresholdedResult(Double.NaN));
    assertThat(e).hasMessageThat().startsWith("delta must be");
  }

  @Test
  public void computeThresholdedResult_thresholdDeltaZero_throwsException() {
    IllegalArgumentException exception =
        assertThrows(IllegalArgumentException.class, () -> count.computeThresholdedResult(0.0));
    assertThat(exception).hasMessageThat().startsWith("delta must be");
  }

  @Test
  public void computeThresholdedResult_thresholdDeltaOne_throwsException() {
    IllegalArgumentException exception =
        assertThrows(IllegalArgumentException.class, () -> count.computeThresholdedResult(1.0));
    assertThat(exception).hasMessageThat().startsWith("delta must be");
  }

  @Test
  public void computeThresholdedResult_thresholdDeltaGreaterThanOne_throwsException() {
    IllegalArgumentException exception =
        assertThrows(IllegalArgumentException.class, () -> count.computeThresholdedResult(2.0));
    assertThat(exception).hasMessageThat().startsWith("delta must be");
  }

  @Test
  public void computeThresholdedResult_unknownNoiseType_throwsException() {
    when(noise.getMechanismType()).thenReturn(MechanismType.EMPTY);

    IllegalStateException exception =
        assertThrows(
            IllegalStateException.class, () -> count.computeThresholdedResult(THRESHOLD_DELTA));
    assertThat(exception).hasMessageThat().contains("unknown mechanism type");
  }

  @Test
  public void computeThresholdedResult_defaultParams_callsNoiseCorrectly() {
    count.computeThresholdedResult(THRESHOLD_DELTA);

    verify(noise)
        .computeQuantile(
            eq(THRESHOLD_DELTA), // rank = thresholdDelta / lInfSensitivity = THRESHOLD_DELTA / 1
            eq(/* x = mean = */ 0.0),
            eq(/* l0Sensitivity = maxPartitionsContributed = */ 1),
            eq(/* lInfSensitivity = maxContributionsPerPartition = */ 1.0),
            eq(EPSILON),
            eq(DELTA));
  }

  @Test
  public void computeThresholdedResult_scaledLinfSensitivity_callsNoiseCorrectly() {
    count = getCountBuilderWithFields().maxContributionsPerPartition(10).build();

    count.computeThresholdedResult(THRESHOLD_DELTA);

    verify(noise)
        .computeQuantile(
            eq(
                THRESHOLD_DELTA
                    / 10.0), // rank = thresholdDelta / lInfSensitivity = THRESHOLD_DELTA / 10
            eq(/* x = mean = */ 0.0),
            eq(/* l0Sensitivity = maxPartitionsContributed = */ 1),
            eq(/* lInfSensitivity = maxContributionsPerPartition = */ 10.0),
            eq(EPSILON),
            eq(DELTA));
  }

  @Test
  public void computeThresholdedResult_countBelowThreshold_returnsEmptyResult() {
    double quantile = -10.0;
    when(noise.computeQuantile(
            anyDouble(), anyDouble(), anyInt(), anyDouble(), anyDouble(), anyDouble()))
        .thenReturn(quantile);

    // threshold is equal to -1 * quantile + maxContributionsPerPartition = 11;
    // the result count is equal to 1 which doesn't pass the
    // threshold of 11 and therefore the empty result should be returned.
    count.increment();
    Optional<Long> actualResult = count.computeThresholdedResult(THRESHOLD_DELTA);
    assertThat(actualResult.isPresent()).isFalse();
  }

  @Test
  public void computeThresholdedResult_countGreaterThanThreshold_returnsComputedResult() {
    double quantile = -10.0;
    when(noise.computeQuantile(
            anyDouble(), anyDouble(), anyInt(), anyDouble(), anyDouble(), anyDouble()))
        .thenReturn(quantile);

    // threshold is equal to -1 * quantile + maxContributionsPerPartition = 11;
    // the result count is equal to 15 which passes the
    // threshold of 11 and therefore the computed result should be returned.
    count.incrementBy(15);
    Optional<Long> actualResult = count.computeThresholdedResult(THRESHOLD_DELTA);
    assertThat(actualResult.get()).isEqualTo(15);
  }

  @Test
  public void computeThresholdedResult_forLaplace_rawCountAsFloorThreshold_returnsEmptyResult() {
    when(noise.getMechanismType()).thenReturn(LAPLACE);
    when(noise.computeQuantile(
            anyDouble(), anyDouble(), anyInt(), anyDouble(), anyDouble(), isNull()))
        .thenAnswer(
            invocation ->
                new LaplaceNoise()
                    .computeQuantile(
                        (double) invocation.getArguments()[0],
                        (double) invocation.getArguments()[1],
                        (int) invocation.getArguments()[2],
                        (double) invocation.getArguments()[3],
                        (double) invocation.getArguments()[4],
                        null));

    Count count =
        Count.builder()
            .epsilon(Math.log(3))
            .maxContributionsPerPartition(1)
            .maxPartitionsContributed(1)
            .noise(noise)
            .build();

    count.incrementBy((long) Math.floor(21.33));
    // threshold is equal to 21.33
    Optional<Long> actualResult = count.computeThresholdedResult(1e-10);
    assertThat(actualResult.isPresent()).isFalse();
  }

  @Test
  public void computeThresholdedResult_forLaplace_rawCountAsCeilThreshold_returnsComputedResult() {
    when(noise.getMechanismType()).thenReturn(LAPLACE);
    when(noise.computeQuantile(
            anyDouble(), anyDouble(), anyInt(), anyDouble(), anyDouble(), isNull()))
        .thenAnswer(
            invocation ->
                new LaplaceNoise()
                    .computeQuantile(
                        (double) invocation.getArguments()[0],
                        (double) invocation.getArguments()[1],
                        (int) invocation.getArguments()[2],
                        (double) invocation.getArguments()[3],
                        (double) invocation.getArguments()[4],
                        null));

    Count count =
        Count.builder()
            .epsilon(Math.log(3))
            .maxContributionsPerPartition(1)
            .maxPartitionsContributed(1)
            .noise(noise)
            .build();

    long rawCount = (long) Math.ceil(21.33);
    count.incrementBy(rawCount);
    // threshold is equal to 21.33
    Optional<Long> actualResult = count.computeThresholdedResult(1e-10);
    assertThat(actualResult.get()).isEqualTo(rawCount);
  }

  @Test
  public void computeThresholdedResult_forGaussian_rawCountAsFloorThreshold_returnsEmptyResult() {
    when(noise.computeQuantile(
            anyDouble(), anyDouble(), anyInt(), anyDouble(), anyDouble(), anyDouble()))
        .thenAnswer(
            invocation ->
                new GaussianNoise()
                    .computeQuantile(
                        (double) invocation.getArguments()[0],
                        (double) invocation.getArguments()[1],
                        (int) invocation.getArguments()[2],
                        (double) invocation.getArguments()[3],
                        (double) invocation.getArguments()[4],
                        (double) invocation.getArguments()[5]));

    Count count =
        Count.builder()
            .epsilon(Math.log(3))
            .delta(0.26546844106038714)
            .maxContributionsPerPartition(1)
            .maxPartitionsContributed(2)
            .noise(noise)
            .build();

    long rawCount = (long) Math.floor(2.9997099087500634);
    count.incrementBy(rawCount);
    // threshold is equal to 2.9997099087500634
    Optional<Long> actualResult = count.computeThresholdedResult(0.022828893856);
    assertThat(actualResult.isPresent()).isFalse();
  }

  @Test
  public void computeThresholdedResult_forGaussian_rawCountAsCeilThreshold_returnsComputedResult() {
    when(noise.computeQuantile(
            anyDouble(), anyDouble(), anyInt(), anyDouble(), anyDouble(), anyDouble()))
        .thenAnswer(
            invocation ->
                new GaussianNoise()
                    .computeQuantile(
                        (double) invocation.getArguments()[0],
                        (double) invocation.getArguments()[1],
                        (int) invocation.getArguments()[2],
                        (double) invocation.getArguments()[3],
                        (double) invocation.getArguments()[4],
                        (double) invocation.getArguments()[5]));

    Count count =
        Count.builder()
            .epsilon(Math.log(3))
            .delta(0.26546844106038714)
            .maxContributionsPerPartition(1)
            .maxPartitionsContributed(2)
            .noise(noise)
            .build();

    long rawCount = (long) Math.ceil(2.9997099087500634);
    count.incrementBy(rawCount);
    // threshold is equal to 2.9997099087500634
    Optional<Long> actualResult = count.computeThresholdedResult(0.022828893856);
    assertThat(actualResult.get()).isEqualTo(rawCount);
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
}

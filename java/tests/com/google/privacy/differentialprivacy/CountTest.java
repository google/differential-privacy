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
import static org.mockito.ArgumentMatchers.doubleThat;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.google.privacy.differentialprivacy.proto.SummaryOuterClass.CountSummary;
import com.google.privacy.differentialprivacy.proto.SummaryOuterClass.MechanismType;
import com.google.protobuf.InvalidProtocolBufferException;
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
  public void incrementBy_allowsNegativeValues() {
    count.incrementBy(-100);

    assertThat(count.computeResult()).isEqualTo(-100);
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
  public void incrementBy_hugeIntegerValues_dontOverflow() {
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
  public void incrementBy_hugeLongValues_doOverflow() {
    count.incrementBy(Long.MAX_VALUE);
    count.increment();
    count.increment();

    assertThat(count.computeResult()).isEqualTo(Long.MIN_VALUE + 1);
  }

  @Test
  public void incrementBy_hugeLongValues_doOverflowByNoise() {
    // Mock the noise mechanism so that it always generates 1.
    when(noise.addNoise(anyLong(), anyInt(), anyLong(), anyDouble(), anyDouble()))
        .thenAnswer(invocation -> (long) invocation.getArguments()[0] + 1);
    count.incrementBy(Long.MAX_VALUE);

    assertThat(count.computeResult()).isEqualTo(Long.MIN_VALUE);
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
    Count targetCount = getCountBuilderWithFields().noise(new LaplaceNoise()).delta(0.0).build();
    Count sourceCount = getCountBuilderWithFields().noise(new LaplaceNoise()).delta(0.0).build();
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
    Count targetCount = getCountBuilderWithFields().noise(new LaplaceNoise()).delta(0.0).build();
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
  public void computeThresholdedResult_usesAccurateThresholdDeltaPerPartitionValue() {
    // The computation of thresholdDeltaPerPartition can be vulnerable to numerical imprecisions for
    // very small values of thresholdDelta if done naively. To check the accuracy of
    // thresholdDeltaPerPartition, we look at the rank argument used in the internal quantile
    // computation (because thresholdDeltaPerPartition is used as the rank argument).

    // The choice of maxContributionsPerPartition and thresholdDelta is arbitrary. It is also
    // extreme in the sense that it would result in numerical inaccuracies if
    // thresholdDeltaPerPartition is computed naively.
    count = getCountBuilderWithFields().maxPartitionsContributed(12345).build();
    count.computeThresholdedResult(/* thresholdDelta= */ 5.4321e-60);

    // The anticipated value of thresholdDeltaPerPartition is
    // thresholdDeltaPerPartition = 1 - (1 - thresholdDelta)^(1 / maxPartitionsContributed)
    //                            = 1 - (1 - 5.4321e-60)^(1 / 12345)
    //                            ≈ 4.4002430134e-64
    verify(noise)
        .computeQuantile(
            doubleThat(d -> d > 4.4002430133e-64 && d < 4.4002430135e-64),
            anyDouble(),
            anyInt(),
            anyDouble(),
            anyDouble(),
            anyDouble());
  }

  @Test
  public void computeThresholdedResult_countBelowThreshold_returnsEmptyResult() {
    double quantile = -10.0;
    when(noise.computeQuantile(
            anyDouble(), anyDouble(), anyInt(), anyDouble(), anyDouble(), anyDouble()))
        .thenReturn(quantile);

    // threshold is equal to -1 * quantile + maxContributionsPerPartition = 11. Moreover, the (zero-
    // noise) count is equal to 10, which doesn't pass the threshold and therefore the empty result
    // should be returned.
    count.incrementBy(10);
    Optional<Long> actualResult = count.computeThresholdedResult(THRESHOLD_DELTA);
    assertThat(actualResult.isPresent()).isFalse();
  }

  @Test
  public void computeThresholdedResult_countGreaterThanThreshold_returnsCount() {
    double quantile = -10.0;
    when(noise.computeQuantile(
            anyDouble(), anyDouble(), anyInt(), anyDouble(), anyDouble(), anyDouble()))
        .thenReturn(quantile);

    // threshold is equal to -1 * quantile + maxContributionsPerPartition = 11. Moreover, the (zero-
    // noise) count is equal to 12, which passes the threshold and therefore 12 should be returned.
    count.incrementBy(12);
    Optional<Long> actualResult = count.computeThresholdedResult(THRESHOLD_DELTA);
    assertThat(actualResult).hasValue(12);
  }

  @Test
  public void computeThresholdedResult_forLaplace_appliesCorrectThreshold() {
    when(noise.getMechanismType()).thenReturn(LAPLACE);
    when(noise.computeQuantile(
            anyDouble(), anyDouble(), anyInt(), anyDouble(), anyDouble(), eq(0.0)))
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

    Count.Params.Builder builder =
        Count.builder()
            .epsilon(Math.log(3))
            .maxContributionsPerPartition(3)
            .maxPartitionsContributed(11)
            .noise(noise);

    // For the given parameters and a thresholdDelta of 0.1, the threshold should be
    //    k = mu - lambda * log(2 * thresholdDeltaPerPartition) ≈ 121.94708
    // where
    //    mu = maxContributionsPerPartition
    //    lambda = (maxContributionsPerPartition * maxPartitionsContributed) / epsilon
    //    thresholdDeltaPerPartition = 1 - (1 - thresholdDelta)^(1 / maxPartitionsContributed).
    //
    // Using the floor and the ceil of this threshold as input to a zero-noise count, we expect
    // the floor to be thresholded but the ceil to pass.
    Count count = builder.build();
    count.incrementBy(121);
    Optional<Long> flooredThreshold = count.computeThresholdedResult(0.1);
    count = builder.build();
    count.incrementBy(122);
    Optional<Long> ceiledThreshold = count.computeThresholdedResult(0.1);

    assertThat(flooredThreshold).isEmpty();
    assertThat(ceiledThreshold).hasValue(122);
  }

  @Test
  public void computeThresholdedResult_forGaussian_appliesCorrectThreshold() {
    when(noise.getMechanismType()).thenReturn(GAUSSIAN);
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

    Count.Params.Builder builder =
        Count.builder()
            .epsilon(Math.log(3))
            .delta(0.0001)
            .maxContributionsPerPartition(3)
            .maxPartitionsContributed(11)
            .noise(noise);

    // For the given parameters and a thresholdDelta of 0.1, the threshold should be
    //    k ≈ 71.42627.
    //
    // Using the floor and the ceil of this threshold as input to a zero-noise count, we expect
    // the floor to be thresholded but the ceil to pass.
    Count count = builder.build();
    count.incrementBy(71);
    Optional<Long> flooredThreshold = count.computeThresholdedResult(0.1);
    count = builder.build();
    count.incrementBy(72);
    Optional<Long> ceiledThreshold = count.computeThresholdedResult(0.1);

    assertThat(flooredThreshold).isEmpty();
    assertThat(ceiledThreshold).hasValue(72);
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

//
// Copyright 2023 Google LLC
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
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.google.privacy.differentialprivacy.proto.SummaryOuterClass.LongBoundedSumSummary;
import com.google.protobuf.ExtensionRegistry;
import com.google.protobuf.InvalidProtocolBufferException;
import java.util.Arrays;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnit;
import org.mockito.junit.MockitoRule;

/**
 * Tests the accuracy of {@link LongBoundedSum}. The test mocks {@link Noise} instance which
 * generates zero noise.
 *
 * <p>Statistical and DP properties of the algorithm are tested in {@link
 * com.google.privacy.differentialprivacy.statistical.LongBoundedSumDpTest}.
 */
@RunWith(JUnit4.class)
public class LongBoundedSumTest {
  private static final double EPSILON = 0.123;
  private static final double DELTA = 0.456;

  @Mock private Noise noise;
  private LongBoundedSum sum;

  @Rule public final MockitoRule mocks = MockitoJUnit.rule();

  @Before
  public void setUp() {
    // Mock the noise mechanism so that it does not add any noise.
    when(noise.addNoise(anyLong(), anyInt(), anyLong(), anyDouble(), anyDouble()))
        .thenAnswer(invocation -> invocation.getArguments()[0]);
    // Tests that use serialization need to access to the type of the noise they use. Because the
    // tests don't rely on a specific noise type, we arbitrarily return Gaussian.
    when(noise.getMechanismType()).thenReturn(GAUSSIAN);

    sum =
        LongBoundedSum.builder()
            .epsilon(EPSILON)
            .delta(DELTA)
            .noise(noise)
            .maxPartitionsContributed(1)
            .maxContributionsPerPartition(1)
            // The lower and upper bounds are arbitrarily chosen negative and positive values.
            .lower(Long.MIN_VALUE)
            .upper(Long.MAX_VALUE)
            .build();
  }

  @Test
  public void addEntry() {
    sum.addEntry(1);
    sum.addEntry(2);
    sum.addEntry(3);
    sum.addEntry(4);

    assertThat(sum.computeResult()).isEqualTo(10);
  }

  @Test
  public void addEntries() {
    sum.addEntries(Arrays.asList(1L, 2L, 3L, 4L));
    assertThat(sum.computeResult()).isEqualTo(10);
  }

  @Test
  public void addEntry_calledAfterComputeResult_throwsException() {
    var unused = sum.computeResult();
    assertThrows(IllegalStateException.class, () -> sum.addEntry(0));
  }

  @Test
  public void addEntry_calledAfterSerialize_throwsException() {
    var unused = sum.getSerializableSummary();
    assertThrows(IllegalStateException.class, () -> sum.addEntry(0));
  }

  @Test
  public void addEntries_calledAfterComputeResult_throwsException() {
    var unused = sum.computeResult();
    assertThrows(IllegalStateException.class, () -> sum.addEntry(0L));
  }

  @Test
  public void addEntries_calledAfterSerialize_throwsException() {
    var unused = sum.getSerializableSummary();
    assertThrows(IllegalStateException.class, () -> sum.addEntry(0L));
  }

  @Test
  public void computeResult_multipleCalls_throwsException() {
    var unused = sum.computeResult();
    assertThrows(IllegalStateException.class, () -> sum.computeResult());
  }

  @Test
  public void computeResult_calledAfterSerialize_throwsException() {
    var unused = sum.getSerializableSummary();
    assertThrows(IllegalStateException.class, () -> sum.computeResult());
  }

  // Input values should be clamped to the upper and lower bounds.
  @Test
  public void addEntry_clampsInput() {
    sum =
        LongBoundedSum.builder()
            .epsilon(EPSILON)
            .delta(DELTA)
            .noise(noise)
            .maxPartitionsContributed(1)
            .lower(0)
            .upper(1)
            .build();

    sum.addEntry(-1); // should be clamped to 0
    sum.addEntry(1); // should not be clamped
    sum.addEntry(10); // should be clamped to 1

    // 0 + 1 + 1
    assertThat(sum.computeResult()).isEqualTo(2);
  }

  @Test
  public void computeResult_callsNoiseCorrectly() {
    long value = 5;
    int l0Sensitivity = 1;
    sum =
        LongBoundedSum.builder()
            .epsilon(EPSILON)
            .delta(DELTA)
            .noise(noise)
            .maxPartitionsContributed(l0Sensitivity)
            .maxContributionsPerPartition(5)
            .lower(0)
            .upper(100)
            .build();
    sum.addEntry(value);
    var unused = sum.computeResult();

    verify(noise)
        .addNoise(
            eq(value),
            eq(l0Sensitivity),
            eq(/* lower = 0, upper = 100, maxContributionsPerPartition = 5 =>
             lInfSensitivity = max(abs(0), abs(100)) * 5 = 500 */ 500L),
            eq(EPSILON),
            eq(DELTA));
  }

  @Test
  public void computeResult_addsNoise() {
    // Mock the noise mechanism so that it always generates 100.
    when(noise.addNoise(anyLong(), anyInt(), anyLong(), anyDouble(), anyDouble()))
        .thenAnswer(invocation -> (long) invocation.getArguments()[0] + 100);
    sum =
        LongBoundedSum.builder()
            .epsilon(EPSILON)
            .delta(DELTA)
            .noise(noise)
            .maxPartitionsContributed(1)
            .lower(0)
            .upper(1000)
            .build();

    sum.addEntry(10);
    assertThat(sum.computeResult()).isEqualTo(110); // value (10) + noise (100) = 110
  }

  // The current implementation of LongBoundedSum only supports double as input.
  // This test verifies that, if the lower bound is the smallest possible integer (represented as
  // double), then the L_Inf sensitivity calculation does not overflow.
  @Test
  public void lowerBoundMinInteger_doesntOverflow() {
    sum =
        LongBoundedSum.builder()
            .epsilon(EPSILON)
            .delta(DELTA)
            .noise(noise)
            .maxPartitionsContributed(1)
            .maxContributionsPerPartition(1)
            .lower(Integer.MIN_VALUE)
            .upper(0)
            .build();

    var unused = sum.computeResult();
    // LongBoundedSum first calculates L_Inf sensitivity and then passes it to the noise.
    // Verify that L_Inf sensitivity does not overflow and that
    // the noise generation is called with
    // L_Inf sensitivity == lowerBound * maxContributionsPerPartition ==
    // -(double)Integer.MIN_VALUE.
    // More precisely:
    // L_Inf sensitivity =
    // max(abs(lower), abs(upper)) * maxContributionsPerPartition =
    // max(-Integer.MIN_VALUE, 0) = -Integer.MIN_VALUE.
    verify(noise)
        .addNoise(anyLong(), anyInt(), eq(-(long) Integer.MIN_VALUE), anyDouble(), anyDouble());
  }

  @Test
  public void getSerializableSummary_copiesPartialSumCorrectly() {
    sum.addEntry(10);
    sum.addEntry(10);

    LongBoundedSumSummary summary = getSummary(sum);
    assertThat(summary.getPartialSum().getIntValue()).isEqualTo(20);
  }

  @Test
  public void getSerializableSummary_copiesZeroSumCorrectly() {
    LongBoundedSumSummary summary = getSummary(sum);
    assertThat(summary.getPartialSum().getIntValue()).isEqualTo(0);
  }

  @Test
  public void getSerializableSummary_copiesMaxDoubleSumCorrectly() {
    sum.addEntry(Long.MAX_VALUE);

    LongBoundedSumSummary summary = getSummary(sum);
    assertThat(summary.getPartialSum().getIntValue()).isEqualTo(Long.MAX_VALUE);
  }

  @Test
  public void getSerializableSummary_copiesMinDoubleSumCorrectly() {
    sum.addEntry(Long.MIN_VALUE);

    LongBoundedSumSummary summary = getSummary(sum);
    assertThat(summary.getPartialSum().getIntValue()).isEqualTo(Long.MIN_VALUE);
  }

  @Test
  public void getSerializableSummary_copiesNegativeSumCorrectly() {
    sum.addEntry(-5);
    LongBoundedSumSummary summary = getSummary(sum);
    assertThat(summary.getPartialSum().getIntValue()).isEqualTo(-5);
  }

  @Test
  public void getSerializableSummary_copiesPositiveSumCorrectly() {
    sum.addEntry(5);
    LongBoundedSumSummary summary = getSummary(sum);
    assertThat(summary.getPartialSum().getIntValue()).isEqualTo(5);
  }

  @Test
  public void getSerializableSummary_calledAfterComputeResult_throwsException() {
    var unused = sum.computeResult();
    assertThrows(IllegalStateException.class, () -> sum.getSerializableSummary());
  }

  @Test
  public void getSerializableSummary_multipleCalls_returnsSameSummary() {
    sum =
        LongBoundedSum.builder()
            .epsilon(EPSILON)
            .noise(new LaplaceNoise())
            .maxPartitionsContributed(1)
            .lower(0)
            .upper(1)
            .build();
    sum.addEntry(1);
    byte[] summary1 = sum.getSerializableSummary();
    byte[] summary2 = sum.getSerializableSummary();
    assertThat(summary1).isEqualTo(summary2);
  }

  @Test
  public void getSerializableSummary_copiesEpsilonCorrectly() {
    sum = getLongBoundedSumBuilderWithFields().epsilon(EPSILON).build();
    LongBoundedSumSummary summary = getSummary(sum);
    assertThat(summary.getEpsilon()).isEqualTo(EPSILON);
  }

  @Test
  public void getSerializableSummary_copiesDeltaCorrectly() {
    sum = getLongBoundedSumBuilderWithFields().delta(DELTA).build();
    LongBoundedSumSummary summary = getSummary(sum);
    assertThat(summary.getDelta()).isEqualTo(DELTA);
  }

  @Test
  public void getSerializableSummary_copiesGaussianNoiseCorrectly() {
    sum = getLongBoundedSumBuilderWithFields().noise(new GaussianNoise()).build();
    LongBoundedSumSummary summary = getSummary(sum);
    assertThat(summary.getMechanismType()).isEqualTo(GAUSSIAN);
  }

  @Test
  public void getSerializableSummary_copiesLaplaceNoiseCorrectly() {
    sum = getLongBoundedSumBuilderWithFields().noise(new LaplaceNoise()).delta(0.0).build();
    LongBoundedSumSummary summary = getSummary(sum);
    assertThat(summary.getMechanismType()).isEqualTo(LAPLACE);
  }

  @Test
  public void getSerializableSummary_copiesMaxPartitionsContributedCorrectly() {
    int maxPartitionsContributed = 150;
    sum =
        getLongBoundedSumBuilderWithFields()
            .maxPartitionsContributed(maxPartitionsContributed)
            .build();
    LongBoundedSumSummary summary = getSummary(sum);
    assertThat(summary.getMaxPartitionsContributed()).isEqualTo(maxPartitionsContributed);
  }

  @Test
  public void getSerializableSummary_copiesMaxContributionsPerPartitionCorrectly() {
    int maxContributionsPerPartition = 150;
    sum =
        getLongBoundedSumBuilderWithFields()
            .maxContributionsPerPartition(maxContributionsPerPartition)
            .build();
    LongBoundedSumSummary summary = getSummary(sum);
    assertThat(summary.getMaxContributionsPerPartition()).isEqualTo(maxContributionsPerPartition);
  }

  @Test
  public void getSerializableSummary_copiesLowerCorrectly() {
    long lower = -1;
    sum = getLongBoundedSumBuilderWithFields().lower(lower).build();
    LongBoundedSumSummary summary = getSummary(sum);
    assertThat(summary.getLower()).isEqualTo(lower);
  }

  @Test
  public void getSerializableSummary_copiesUpperCorrectly() {
    long upper = 1;
    sum = getLongBoundedSumBuilderWithFields().upper(upper).build();
    LongBoundedSumSummary summary = getSummary(sum);
    assertThat(summary.getUpper()).isEqualTo(upper);
  }

  @Test
  public void mergeWith_basicExample_sumsValues() {
    LongBoundedSum targetSum = getLongBoundedSumBuilderWithFields().build();
    LongBoundedSum sourceSum = getLongBoundedSumBuilderWithFields().build();

    targetSum.addEntry(1);
    sourceSum.addEntry(1);

    targetSum.mergeWith(sourceSum.getSerializableSummary());

    assertThat(targetSum.computeResult()).isEqualTo(2);
  }

  @Test
  public void mergeWith_calledTwice_sumsValues() {
    LongBoundedSum targetSum = getLongBoundedSumBuilderWithFields().build();
    LongBoundedSum sourceSum1 = getLongBoundedSumBuilderWithFields().build();
    LongBoundedSum sourceSum2 = getLongBoundedSumBuilderWithFields().build();

    targetSum.addEntry(1);
    sourceSum1.addEntry(2);
    sourceSum2.addEntry(3);

    targetSum.mergeWith(sourceSum1.getSerializableSummary());
    targetSum.mergeWith(sourceSum2.getSerializableSummary());

    assertThat(targetSum.computeResult()).isEqualTo(6);
  }

  @Test
  public void mergeWith_epsilonMismatch_throwsException() {
    LongBoundedSum targetSum = getLongBoundedSumBuilderWithFields().epsilon(EPSILON).build();
    LongBoundedSum sourceSum = getLongBoundedSumBuilderWithFields().epsilon(2 * EPSILON).build();
    assertThrows(
        IllegalArgumentException.class,
        () -> targetSum.mergeWith(sourceSum.getSerializableSummary()));
  }

  @Test
  public void mergeWith_nullDelta_mergesWithoutException() {
    LongBoundedSum targetSum =
        getLongBoundedSumBuilderWithFields().noise(new LaplaceNoise()).delta(0.0).build();
    LongBoundedSum sourceSum =
        getLongBoundedSumBuilderWithFields().noise(new LaplaceNoise()).delta(0.0).build();
    // No exception should be thrown.
    targetSum.mergeWith(sourceSum.getSerializableSummary());
  }

  @Test
  public void mergeWith_deltaMismatch_throwsException() {
    LongBoundedSum targetSum = getLongBoundedSumBuilderWithFields().delta(DELTA).build();
    LongBoundedSum sourceSum = getLongBoundedSumBuilderWithFields().delta(2 * DELTA).build();
    assertThrows(
        IllegalArgumentException.class,
        () -> targetSum.mergeWith(sourceSum.getSerializableSummary()));
  }

  @Test
  public void mergeWith_noiseMismatch_throwsException() {
    LongBoundedSum targetSum =
        getLongBoundedSumBuilderWithFields().noise(new LaplaceNoise()).delta(0.0).build();
    LongBoundedSum sourceSum =
        getLongBoundedSumBuilderWithFields().noise(new DiscreteLaplaceNoise()).delta(0.0).build();
    assertThrows(
        IllegalArgumentException.class,
        () -> targetSum.mergeWith(sourceSum.getSerializableSummary()));
  }

  @Test
  public void mergeWith_maxPartitionsContributedMismatch_throwsException() {
    LongBoundedSum targetSum =
        getLongBoundedSumBuilderWithFields().maxPartitionsContributed(1).build();
    LongBoundedSum sourceSum =
        getLongBoundedSumBuilderWithFields().maxPartitionsContributed(2).build();
    assertThrows(
        IllegalArgumentException.class,
        () -> targetSum.mergeWith(sourceSum.getSerializableSummary()));
  }

  @Test
  public void mergeWith_maxContributionsPerPartitionMismatch_throwsException() {
    LongBoundedSum targetSum =
        getLongBoundedSumBuilderWithFields().maxContributionsPerPartition(1).build();
    LongBoundedSum sourceSum =
        getLongBoundedSumBuilderWithFields().maxContributionsPerPartition(2).build();
    assertThrows(
        IllegalArgumentException.class,
        () -> targetSum.mergeWith(sourceSum.getSerializableSummary()));
  }

  @Test
  public void mergeWith_lowerBoundsMismatch_throwsException() {
    LongBoundedSum targetSum = getLongBoundedSumBuilderWithFields().lower(-1).build();
    LongBoundedSum sourceSum = getLongBoundedSumBuilderWithFields().lower(-100).build();
    assertThrows(
        IllegalArgumentException.class,
        () -> targetSum.mergeWith(sourceSum.getSerializableSummary()));
  }

  @Test
  public void mergeWith_upperBoundsMismatch_throwsException() {
    LongBoundedSum targetSum = getLongBoundedSumBuilderWithFields().upper(1).build();
    LongBoundedSum sourceSum = getLongBoundedSumBuilderWithFields().upper(100).build();
    assertThrows(
        IllegalArgumentException.class,
        () -> targetSum.mergeWith(sourceSum.getSerializableSummary()));
  }

  @Test
  public void mergeWith_calledAfterComputeResult_throwsException() {
    LongBoundedSum targetSum = getLongBoundedSumBuilderWithFields().build();
    LongBoundedSum sourceSum = getLongBoundedSumBuilderWithFields().build();

    var unused = targetSum.computeResult();
    byte[] summary = sourceSum.getSerializableSummary();
    assertThrows(IllegalStateException.class, () -> targetSum.mergeWith(summary));
  }

  @Test
  public void mergeWith_calledAfterSerialization_throwsException() {
    LongBoundedSum targetSum = getLongBoundedSumBuilderWithFields().build();
    LongBoundedSum sourceSum = getLongBoundedSumBuilderWithFields().build();

    var unused = targetSum.getSerializableSummary();
    byte[] summary = sourceSum.getSerializableSummary();
    assertThrows(IllegalStateException.class, () -> targetSum.mergeWith(summary));
  }

  private LongBoundedSum.Params.Builder getLongBoundedSumBuilderWithFields() {
    return LongBoundedSum.builder()
        .epsilon(EPSILON)
        .delta(DELTA)
        .noise(noise)
        .maxPartitionsContributed(1)
        // lower, upper and, maxContributionsPerPartition have arbitrarily chosen values.
        .maxContributionsPerPartition(10)
        .lower(-10)
        .upper(10);
  }

  /**
   * Note that {@link LongBoundedSumSummary} isn't visible to the actual clients, who only see an
   * opaque {@code byte[]} blob. Here, we parse said blob to perform whitebox testing, to verify
   * some expectations of the blob's content. We do this because achieving good coverage with pure
   * behaviour testing (i.e., blackbox testing) isn't possible.
   */
  private static LongBoundedSumSummary getSummary(LongBoundedSum sum) {
    byte[] nonParsedSummary = sum.getSerializableSummary();
    try {
      return LongBoundedSumSummary.parseFrom(nonParsedSummary, ExtensionRegistry.newInstance());
    } catch (InvalidProtocolBufferException pbe) {
      throw new IllegalArgumentException(pbe);
    }
  }
}

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
import static java.lang.Double.NaN;
import static org.junit.Assert.assertThrows;
import static org.mockito.ArgumentMatchers.anyDouble;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.google.privacy.differentialprivacy.proto.SummaryOuterClass.BoundedSumSummary;
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
 * Tests for {@link BoundedSum}.
 *
 * <p>Statistical and DP properties of the algorithm are tested in {@link
 * com.google.privacy.differentialprivacy.statistical.BoundedSumDpTest}.
 */
@RunWith(JUnit4.class)
public class BoundedSumTest {
  private static final double EPSILON = 0.123;
  private static final double DELTA = 0.456;

  @Mock private Noise mockNoise;

  @Rule public final MockitoRule mocks = MockitoJUnit.rule();

  @Before
  public void setUp() {
    when(mockNoise.getMechanismType()).thenReturn(LAPLACE);
  }

  @Test
  public void addEntry() {
    BoundedSum sum = getBoundedSumBuilder().noise(new ZeroNoise()).build();

    sum.addEntry(1.0);
    sum.addEntry(2.0);
    sum.addEntry(3.0);
    sum.addEntry(4.0);

    assertThat(sum.computeResult()).isEqualTo(10.0);
  }

  @Test
  public void addEntries() {
    BoundedSum sum = getBoundedSumBuilder().noise(new ZeroNoise()).build();

    sum.addEntries(Arrays.asList(1.0, 2.0, 3.0, 4.0));

    assertThat(sum.computeResult()).isEqualTo(10.0);
  }

  @Test
  public void addEntry_Nan_ignored() {
    BoundedSum sum = getBoundedSumBuilder().noise(new ZeroNoise()).build();

    sum.addEntry(NaN);
    sum.addEntry(2);

    assertThat(sum.computeResult()).isEqualTo(2.0);
  }

  @Test
  public void addEntry_calledAfterComputeResult_throwsException() {
    BoundedSum sum = buildBoundedSum();

    sum.computeResult();

    assertThrows(IllegalStateException.class, () -> sum.addEntry(0.0));
  }

  @Test
  public void addEntry_calledAfterSerialize_throwsException() {
    BoundedSum sum = buildBoundedSum();

    sum.getSerializableSummary();

    assertThrows(IllegalStateException.class, () -> sum.addEntry(0.0));
  }

  @Test
  public void addEntries_calledAfterComputeResult_throwsException() {
    BoundedSum sum = buildBoundedSum();

    sum.computeResult();

    assertThrows(IllegalStateException.class, () -> sum.addEntries(Arrays.asList(0.0)));
  }

  @Test
  public void addEntries_calledAfterSerialize_throwsException() {
    BoundedSum sum = buildBoundedSum();

    sum.getSerializableSummary();

    assertThrows(IllegalStateException.class, () -> sum.addEntries(Arrays.asList(0.0)));
  }

  @Test
  public void computeResult_multipleCalls_throwsException() {
    BoundedSum sum = buildBoundedSum();

    sum.computeResult();

    assertThrows(IllegalStateException.class, () -> sum.computeResult());
  }

  @Test
  public void computeResult_calledAfterSerialize_throwsException() {
    BoundedSum sum = buildBoundedSum();

    sum.getSerializableSummary();

    assertThrows(IllegalStateException.class, () -> sum.computeResult());
  }

  // Input values should be clamped to the upper and lower bounds.
  @Test
  public void addEntry_clampsInput() {
    BoundedSum sum = getBoundedSumBuilder().noise(new ZeroNoise()).lower(0).upper(1).build();

    sum.addEntry(-1.0); // should be clamped to 0
    sum.addEntry(1.0); // should not be clamped
    sum.addEntry(10.0); // should be clamped to 1

    // 0 + 1 + 1
    assertThat(sum.computeResult()).isEqualTo(2);
  }

  @Test
  public void computeResult_callsNoiseCorrectly() {
    when(mockNoise.getMechanismType()).thenReturn(GAUSSIAN);
    int l0Sensitivity = 11;
    BoundedSum sum =
        getBoundedSumBuilder()
            .epsilon(EPSILON)
            .delta(DELTA)
            .noise(mockNoise)
            .maxPartitionsContributed(l0Sensitivity)
            .maxContributionsPerPartition(5)
            .lower(0)
            .upper(100)
            .build();
    double value = 0.5;

    sum.addEntry(value);
    sum.computeResult();

    verify(mockNoise)
        .addNoise(
            eq(value),
            eq(l0Sensitivity),
            eq(/* lower = 0, upper = 100, maxContributionsPerPartition = 5 =>
             lInfSensitivity = max(abs(0), abs(100)) * 5 = 500 */ 500.0),
            eq(EPSILON),
            eq(DELTA));
  }

  @Test
  public void computeResult_addsNoise() {
    // Mock the noise mechanism so that it always generates 100.0.
    when(mockNoise.addNoise(anyDouble(), anyInt(), anyDouble(), anyDouble(), anyDouble()))
        .thenAnswer(invocation -> (double) invocation.getArguments()[0] + 100.0);
    BoundedSum sum = getBoundedSumBuilder().noise(mockNoise).build();
    sum.addEntry(10);

    assertThat(sum.computeResult()).isEqualTo(110); // value (10) + noise (100) = 110
  }

  // The current implementation of BoundedSum only supports double as input.
  // This test verifies that, if the lower bound is the smallest possible integer (represented as
  // double), then the L_Inf sensitivity calculation does not overflow.
  @Test
  public void lowerBoundMinInteger_doesntOverflow() {
    BoundedSum sum =
        getBoundedSumBuilder()
            .noise(mockNoise)
            .lower(Integer.MIN_VALUE)
            .upper(0)
            .maxContributionsPerPartition(1)
            .build();

    sum.computeResult();

    // BoundedSum first calculates L_Inf sensitivity and then passes it to the noise.
    // Verify that L_Inf sensitivity does not overflow and that
    // the noise generation is called with
    // L_Inf sensitivity == lowerBound * maxContributionsPerPartition ==
    // -(double)Integer.MIN_VALUE.
    // More precisely:
    // L_Inf sensitivity =
    // max(abs(lower), abs(upper)) * maxContributionsPerPartition =
    // max(-Integer.MIN_VALUE, 0) = -Integer.MIN_VALUE.
    verify(mockNoise)
        .addNoise(anyDouble(), anyInt(), eq(-(double) Integer.MIN_VALUE), anyDouble(), anyDouble());
  }

  @Test
  public void getSerializableSummary_copiesPartialSumCorrectly() {
    BoundedSum sum = buildBoundedSum();
    sum.addEntry(10.0);
    sum.addEntry(10.0);

    BoundedSumSummary summary = getSummary(sum);

    assertThat(summary.getPartialSum().getFloatValue()).isEqualTo(20.0);
  }

  @Test
  public void getSerializableSummary_copiesZeroSumCorrectly() {
    BoundedSum sum = buildBoundedSum();

    BoundedSumSummary summary = getSummary(sum);

    assertThat(summary.getPartialSum().getFloatValue()).isEqualTo(0.0);
  }

  @Test
  public void getSerializableSummary_copiesMaxDoubleSumCorrectly() {
    BoundedSum sum =
        getBoundedSumBuilder()
            .lower(-Double.MAX_VALUE)
            .upper(Double.MAX_VALUE)
            .maxContributionsPerPartition(1)
            .build();
    sum.addEntry(Double.MAX_VALUE);

    BoundedSumSummary summary = getSummary(sum);

    assertThat(summary.getPartialSum().getFloatValue()).isEqualTo(Double.MAX_VALUE);
  }

  @Test
  public void getSerializableSummary_copiesMinDoubleSumCorrectly() {
    BoundedSum sum =
        getBoundedSumBuilder()
            .lower(-Double.MAX_VALUE)
            .upper(Double.MAX_VALUE)
            .maxContributionsPerPartition(1)
            .build();
    sum.addEntry(Double.MIN_VALUE);

    BoundedSumSummary summary = getSummary(sum);

    assertThat(summary.getPartialSum().getFloatValue()).isEqualTo(Double.MIN_VALUE);
  }

  @Test
  public void getSerializableSummary_copiesNegativeSumCorrectly() {
    BoundedSum sum = buildBoundedSum();
    sum.addEntry(-5.0);

    BoundedSumSummary summary = getSummary(sum);

    assertThat(summary.getPartialSum().getFloatValue()).isEqualTo(-5);
  }

  @Test
  public void getSerializableSummary_copiesPositiveSumCorrectly() {
    BoundedSum sum = buildBoundedSum();
    sum.addEntry(5);

    BoundedSumSummary summary = getSummary(sum);

    assertThat(summary.getPartialSum().getFloatValue()).isEqualTo(5.0);
  }

  @Test
  public void getSerializableSummary_calledAfterComputeResult_throwsException() {
    BoundedSum sum = buildBoundedSum();
    sum.computeResult();

    assertThrows(IllegalStateException.class, sum::getSerializableSummary);
  }

  @Test
  public void getSerializableSummary_multipleCalls_returnsSameSummary() {
    BoundedSum sum = buildBoundedSum();
    sum.addEntry(0.5);

    byte[] summary1 = sum.getSerializableSummary();
    byte[] summary2 = sum.getSerializableSummary();

    assertThat(summary1).isEqualTo(summary2);
  }

  @Test
  public void getSerializableSummary_copiesEpsilonCorrectly() {
    BoundedSum sum = getBoundedSumBuilder().epsilon(EPSILON).build();

    BoundedSumSummary summary = getSummary(sum);

    assertThat(summary.getEpsilon()).isEqualTo(EPSILON);
  }

  @Test
  public void getSerializableSummary_copiesDeltaCorrectly() {
    BoundedSum sum = getBoundedSumBuilder().noise(new GaussianNoise()).delta(DELTA).build();

    BoundedSumSummary summary = getSummary(sum);

    assertThat(summary.getDelta()).isEqualTo(DELTA);
  }

  @Test
  public void getSerializableSummary_copiesGaussianNoiseCorrectly() {
    BoundedSum sum = getBoundedSumBuilder().noise(new GaussianNoise()).delta(DELTA).build();

    BoundedSumSummary summary = getSummary(sum);

    assertThat(summary.getMechanismType()).isEqualTo(GAUSSIAN);
  }

  @Test
  public void getSerializableSummary_copiesLaplaceNoiseCorrectly() {
    BoundedSum sum = getBoundedSumBuilder().noise(new LaplaceNoise()).build();

    BoundedSumSummary summary = getSummary(sum);

    assertThat(summary.getMechanismType()).isEqualTo(LAPLACE);
  }

  @Test
  public void getSerializableSummary_copiesMaxPartitionsContributedCorrectly() {
    int maxPartitionsContributed = 150;
    BoundedSum sum =
        getBoundedSumBuilder().maxPartitionsContributed(maxPartitionsContributed).build();

    BoundedSumSummary summary = getSummary(sum);

    assertThat(summary.getMaxPartitionsContributed()).isEqualTo(maxPartitionsContributed);
  }

  @Test
  public void getSerializableSummary_copiesMaxContributionsPerPartitionCorrectly() {
    int maxContributionsPerPartition = 150;
    BoundedSum sum =
        getBoundedSumBuilder().maxContributionsPerPartition(maxContributionsPerPartition).build();

    BoundedSumSummary summary = getSummary(sum);

    assertThat(summary.getMaxContributionsPerPartition()).isEqualTo(maxContributionsPerPartition);
  }

  @Test
  public void getSerializableSummary_copiesLowerCorrectly() {
    double lower = -0.1;
    BoundedSum sum = getBoundedSumBuilder().lower(lower).build();

    BoundedSumSummary summary = getSummary(sum);

    assertThat(summary.getLower()).isEqualTo(lower);
  }

  @Test
  public void getSerializableSummary_copiesUpperCorrectly() {
    double upper = 0.1;
    BoundedSum sum = getBoundedSumBuilder().upper(upper).build();

    BoundedSumSummary summary = getSummary(sum);

    assertThat(summary.getUpper()).isEqualTo(upper);
  }

  @Test
  public void mergeWith_basicExample_sumsValues() {
    BoundedSum targetSum = getBoundedSumBuilder().noise(new ZeroNoise()).build();
    targetSum.addEntry(1);
    BoundedSum sourceSum = getBoundedSumBuilder().noise(new ZeroNoise()).build();
    sourceSum.addEntry(1);

    targetSum.mergeWith(sourceSum.getSerializableSummary());

    assertThat(targetSum.computeResult()).isEqualTo(2);
  }

  @Test
  public void mergeWith_calledTwice_sumsValues() {
    BoundedSum targetSum = getBoundedSumBuilder().noise(new ZeroNoise()).build();
    targetSum.addEntry(1);
    BoundedSum sourceSum1 = getBoundedSumBuilder().noise(new ZeroNoise()).build();
    sourceSum1.addEntry(2);
    BoundedSum sourceSum2 = getBoundedSumBuilder().noise(new ZeroNoise()).build();
    sourceSum2.addEntry(3);

    targetSum.mergeWith(sourceSum1.getSerializableSummary());
    targetSum.mergeWith(sourceSum2.getSerializableSummary());

    assertThat(targetSum.computeResult()).isEqualTo(6);
  }

  @Test
  public void mergeWith_epsilonMismatch_throwsException() {
    BoundedSum targetSum = getBoundedSumBuilder().epsilon(EPSILON).build();
    BoundedSum sourceSum = getBoundedSumBuilder().epsilon(2 * EPSILON).build();

    assertThrows(
        IllegalArgumentException.class,
        () -> targetSum.mergeWith(sourceSum.getSerializableSummary()));
  }

  @Test
  public void mergeWith_nullDelta_mergesWithoutException() {
    BoundedSum targetSum = getBoundedSumBuilder().noise(new LaplaceNoise()).build();
    BoundedSum sourceSum = getBoundedSumBuilder().noise(new LaplaceNoise()).build();

    // No exception should be thrown.
    targetSum.mergeWith(sourceSum.getSerializableSummary());
  }

  @Test
  public void mergeWith_deltaMismatch_throwsException() {
    BoundedSum targetSum = getBoundedSumBuilder().noise(new GaussianNoise()).delta(DELTA).build();
    BoundedSum sourceSum =
        getBoundedSumBuilder().noise(new GaussianNoise()).delta(2 * DELTA).build();

    assertThrows(
        IllegalArgumentException.class,
        () -> targetSum.mergeWith(sourceSum.getSerializableSummary()));
  }

  @Test
  public void mergeWith_noiseMismatch_throwsException() {
    BoundedSum targetSum = getBoundedSumBuilder().noise(new LaplaceNoise()).build();
    BoundedSum sourceSum = getBoundedSumBuilder().noise(new GaussianNoise()).delta(DELTA).build();

    assertThrows(
        IllegalArgumentException.class,
        () -> targetSum.mergeWith(sourceSum.getSerializableSummary()));
  }

  @Test
  public void mergeWith_maxPartitionsContributedMismatch_throwsException() {
    BoundedSum targetSum = getBoundedSumBuilder().maxPartitionsContributed(1).build();
    BoundedSum sourceSum = getBoundedSumBuilder().maxPartitionsContributed(2).build();

    assertThrows(
        IllegalArgumentException.class,
        () -> targetSum.mergeWith(sourceSum.getSerializableSummary()));
  }

  @Test
  public void mergeWith_maxContributionsPerPartitionMismatch_throwsException() {
    BoundedSum targetSum = getBoundedSumBuilder().maxContributionsPerPartition(1).build();
    BoundedSum sourceSum = getBoundedSumBuilder().maxContributionsPerPartition(2).build();

    assertThrows(
        IllegalArgumentException.class,
        () -> targetSum.mergeWith(sourceSum.getSerializableSummary()));
  }

  @Test
  public void mergeWith_lowerBoundsMismatch_throwsException() {
    BoundedSum targetSum = getBoundedSumBuilder().lower(-1).build();
    BoundedSum sourceSum = getBoundedSumBuilder().lower(-100).build();

    assertThrows(
        IllegalArgumentException.class,
        () -> targetSum.mergeWith(sourceSum.getSerializableSummary()));
  }

  @Test
  public void mergeWith_upperBoundsMismatch_throwsException() {
    BoundedSum targetSum = getBoundedSumBuilder().upper(1).build();
    BoundedSum sourceSum = getBoundedSumBuilder().upper(100).build();

    assertThrows(
        IllegalArgumentException.class,
        () -> targetSum.mergeWith(sourceSum.getSerializableSummary()));
  }

  @Test
  public void mergeWith_calledAfterComputeResult_throwsException() {
    BoundedSum targetSum = buildBoundedSum();
    BoundedSum sourceSum = buildBoundedSum();
    targetSum.computeResult();
    byte[] summary = sourceSum.getSerializableSummary();

    assertThrows(IllegalStateException.class, () -> targetSum.mergeWith(summary));
  }

  @Test
  public void mergeWith_calledAfterSerialization_throwsException() {
    BoundedSum targetSum = buildBoundedSum();
    BoundedSum sourceSum = buildBoundedSum();
    targetSum.getSerializableSummary();
    byte[] summary = sourceSum.getSerializableSummary();

    assertThrows(IllegalStateException.class, () -> targetSum.mergeWith(summary));
  }

  private BoundedSum buildBoundedSum() {
    return getBoundedSumBuilder().noise(new LaplaceNoise()).build();
  }

  private BoundedSum.Params.Builder getBoundedSumBuilder() {
    return BoundedSum.builder()
        .epsilon(EPSILON)
        .maxPartitionsContributed(1)
        // lower, upper and, maxContributionsPerPartition have arbitrarily chosen values.
        .maxContributionsPerPartition(10)
        .lower(-10)
        .upper(10);
  }

  /**
   * Note that {@link BoundedSumSummary} isn't visible to the actual clients, who only see an opaque
   * {@code byte[]} blob. Here, we parse said blob to perform whitebox testing, to verify some
   * expectations of the blob's content. We do this because achieving good coverage with pure
   * behaviour testing (i.e., blackbox testing) isn't possible.
   */
  private static BoundedSumSummary getSummary(BoundedSum sum) {
    byte[] nonParsedSummary = sum.getSerializableSummary();
    try {
      return BoundedSumSummary.parseFrom(nonParsedSummary);
    } catch (InvalidProtocolBufferException pbe) {
      throw new IllegalArgumentException(pbe);
    }
  }
}

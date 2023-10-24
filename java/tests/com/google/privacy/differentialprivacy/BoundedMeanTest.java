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

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static java.lang.Double.NaN;
import static java.util.stream.Collectors.joining;
import static org.junit.Assert.assertThrows;
import static org.mockito.ArgumentMatchers.anyDouble;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.google.common.collect.Range;
import com.google.privacy.differentialprivacy.proto.SummaryOuterClass.MechanismType;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnit;
import org.mockito.junit.MockitoRule;

/**
 * Tests for {@link BoundedMean}.
 *
 * <p>Statistical and DP properties of the algorithm are tested in {@link
 * com.google.privacy.differentialprivacy.statistical.BoundedMeanDpTest}.
 */
@RunWith(JUnit4.class)
public class BoundedMeanTest {
  private static final double EPSILON = 1.0;
  private static final double DELTA = 0.123;
  @Rule public final MockitoRule mocks = MockitoJUnit.rule();
  @Mock private Noise mockNoise;

  @Before
  public void setUp() {
    when(mockNoise.getMechanismType()).thenReturn(MechanismType.GAUSSIAN);
  }

  @Test
  public void addEntry() {
    BoundedMean mean = getMeanBuilder().noise(new ZeroNoise()).build();

    mean.addEntry(2.0);
    mean.addEntry(4.0);
    mean.addEntry(6.0);
    mean.addEntry(8.0);

    assertThat(mean.computeResult()).isEqualTo(5.0);
  }

  @Test
  public void addEntry_Nan_ignored() {
    BoundedMean mean = getMeanBuilder().noise(new ZeroNoise()).build();

    // Add NaN - no exception is thrown.
    mean.addEntry(NaN);
    // Add any values (let's say 7 and 9). Verify that the result is equal to their mean.
    mean.addEntry(7);
    mean.addEntry(9);

    assertThat(mean.computeResult()).isEqualTo(8.0);
  }

  @Test
  public void addEntry_calledAfterComputeResult_throwsException() {
    BoundedMean mean = getMeanBuilder().build();

    mean.computeResult();

    assertThrows(IllegalStateException.class, () -> mean.addEntry(0.0));
  }

  @Test
  public void addEntry_calledAfterSerialize_throwsException() {
    BoundedMean mean = getMeanBuilder().build();

    mean.getSerializableSummary();

    assertThrows(IllegalStateException.class, () -> mean.addEntry(0.0));
  }

  @Test
  public void addEntries() {
    BoundedMean mean = getMeanBuilder().noise(new ZeroNoise()).build();

    mean.addEntries(Arrays.asList(2.0, 4.0, 6.0, 8.0));

    assertThat(mean.computeResult()).isEqualTo(5.0);
  }

  @Test
  public void addEntries_calledAfterComputeResult_throwsException() {
    BoundedMean mean = getMeanBuilder().build();

    mean.computeResult();

    assertThrows(IllegalStateException.class, () -> mean.addEntries(List.of(0.0)));
  }

  @Test
  public void addEntries_calledAfterSerialize_throwsException() {
    BoundedMean mean = getMeanBuilder().build();

    mean.getSerializableSummary();

    assertThrows(IllegalStateException.class, () -> mean.addEntries(List.of(0.0)));
  }

  @Test
  public void computeResult_multipleCalls_throwsException() {
    BoundedMean mean = getMeanBuilder().build();

    mean.computeResult();

    assertThrows(IllegalStateException.class, mean::computeResult);
  }

  @Test
  public void computeResult_calledAfterSerialize_throwsException() {
    BoundedMean mean = getMeanBuilder().build();

    mean.getSerializableSummary();

    assertThrows(IllegalStateException.class, mean::computeResult);
  }

  // Input values are clamped to the upper and lower bounds.
  @Test
  public void addEntry_clampsInput() {
    BoundedMean mean = getMeanBuilder().lower(0.0).upper(2.0).build();

    mean.addEntry(-1.0); // will be clamped to 0
    mean.addEntry(1.0); // will not be clamped
    mean.addEntry(10.0); // will be clamped to 2

    assertThat(mean.computeResult()).isEqualTo(/* (0 + 1 + 2) / 3 */ 1.0);
  }

  @Test
  public void computeResult_singleInput_returnsInput() {
    BoundedMean mean = getMeanBuilder().lower(0).upper(5).noise(new ZeroNoise()).build();

    mean.addEntry(3.0);

    assertThat(mean.computeResult()).isEqualTo(3.0);
  }

  @Test
  public void computeResult_noInput_returnsMidpoint() {
    BoundedMean mean = getMeanBuilder().lower(1).upper(9).build();

    double midpoint = 5.0; // (1 + 9) / 2
    assertThat(mean.computeResult()).isEqualTo(midpoint);
  }

  @Test
  public void computeResult_callsNoiseCorrectly() {
    int maxPartitionsContributed = 1;
    int maxContributionsPerPartition = 3;
    BoundedMean mean =
        BoundedMean.builder()
            .epsilon(EPSILON)
            .delta(DELTA)
            .noise(mockNoise)
            .maxPartitionsContributed(maxPartitionsContributed)
            .maxContributionsPerPartition(maxContributionsPerPartition)
            .lower(1.0)
            .upper(9.0)
            .build();
    mean.addEntry(2.0);
    mean.addEntry(4.0);

    mean.computeResult();

    // Noising normalized sum.
    verify(mockNoise)
        .addNoise(
            eq(/* x1 + x2 - midpoint * count = 2 + 4 - 5 * 2*/ -4.0),
            eq(maxPartitionsContributed),
            eq(/* maxContributionsPerPartition * (upper - lower) / 2 = 3 * (9 - 1) / 2 */ 12.0),
            eq(EPSILON / 2.0),
            eq(DELTA / 2.0));
    // Noising count.
    verify(mockNoise)
        .addNoise(
            eq(/* count */ 2L),
            eq(maxPartitionsContributed),
            eq(
                /* sensitivity of count  = maxContributionsPerPartition*/ (long)
                    maxContributionsPerPartition),
            eq(EPSILON / 2.0),
            eq(DELTA / 2.0));
  }

  @Test
  public void computeResult_addsNoiseToSum() {
    // Mock the noise mechanism so that it adds noise to the sum == 10.0.
    mockDoubleNoise(10);
    // Mock the noise mechanism so that it adds noise to the count == 0.
    mockLongNoise(0);
    BoundedMean mean =
        BoundedMean.builder()
            .epsilon(EPSILON)
            .delta(DELTA)
            .noise(mockNoise)
            .maxPartitionsContributed(1)
            .maxContributionsPerPartition(1)
            .lower(-100)
            .upper(100)
            .build();
    mean.addEntry(20);
    mean.addEntry(20);

    // midpoint = (lower + upper) / 2 = 0,
    // noised_normalized_sum = (x1 + x2) - midpoint * count + noise = 20 + 20 - 0 + 10 = 50,
    // noised_count = count + noise = 2 + 0 = 2,
    // BoundedMean.computeResult() = noised_normalized_sum / noised_count + midpoint = 50 / 2 + 0.
    assertThat(mean.computeResult()).isEqualTo(25);
  }

  @Test
  public void computeResult_addsNoiseToCount() {
    // Mock the noise mechanism so that it adds noise to the sum == 0.0.
    mockDoubleNoise(0);
    // Mock the noise mechanism so that it adds noise to the count == 2.
    mockLongNoise(2);
    BoundedMean mean =
        BoundedMean.builder()
            .epsilon(EPSILON)
            .delta(DELTA)
            .noise(mockNoise)
            .maxPartitionsContributed(1)
            .maxContributionsPerPartition(1)
            .lower(-100)
            .upper(100)
            .build();
    mean.addEntry(20);
    mean.addEntry(20);

    // midpoint = (lower + upper) / 2 = 0,
    // noised_normalized_sum = (x1 + x2) - midpoint * count + noise = 20 + 20 - 0 + 0 = 40,
    // noised_count = count + noise = 2 + 2 = 4,
    // BoundedMean.computeResult() = noised_normalized_sum / noised_count + midpoint = 40 / 4 + 0.
    assertThat(mean.computeResult()).isEqualTo(10);
  }

  @Test
  public void computeResult_clampsTooHighAverage() {
    // We need to have non-zero noise to sum in order to get average which needs to be clamped
    // (i.e., outside of the bounds). If no noise is added then the average will always be within
    // the bounds because the input values are clamped.
    // The noise added to sum is 100.
    mockDoubleNoise(100);
    // The noise added to sum is 0.
    mockLongNoise(0);
    BoundedMean mean =
        BoundedMean.builder()
            .epsilon(EPSILON)
            .delta(DELTA)
            .noise(mockNoise)
            .maxPartitionsContributed(1)
            .maxContributionsPerPartition(1)
            .lower(0)
            .upper(10)
            .build();
    mean.addEntry(5.0);
    mean.addEntry(5.0);

    // midpoint = (lower + upper) / 2 = 5,
    // noised_normalized_sum = (x1 + x2) - midpoint * count + noise = 5 + 5 - 5 * 2 + 100 = 100,
    // noised_count = count + noise = 2 + 0 = 2,
    // non_clamped_average = noised_normalized_sum / noised_count + midpoint = 50 + 5 = 55,
    // BoundedMean.computeResult = clamp(non_clamped_average) = 10 (upper bound).
    assertThat(mean.computeResult()).isEqualTo(10);
  }

  @Test
  public void computeResult_clampsTooLowAverage() {
    // We need to add non-zero noise to sum in order to get average which needs to be clamped
    // (i.e., outside of the bounds). If no noise is added then the average will always be within
    // the bounds because the input values are clamped.
    // The noise added to sum is 100.
    mockDoubleNoise(-100);
    // The noise added to sum is 0.
    mockLongNoise(0);
    BoundedMean mean =
        BoundedMean.builder()
            .epsilon(EPSILON)
            .delta(DELTA)
            .noise(mockNoise)
            .maxPartitionsContributed(1)
            .maxContributionsPerPartition(1)
            .lower(0)
            .upper(10)
            .build();
    mean.addEntry(5.0);
    mean.addEntry(5.0);

    // midpoint = (lower + upper) / 2 = 5,
    // noised_normalized_sum = (x1 + x2) - midpoint * count + noise = 5 + 5 - 5 * 2 - 100 = -100,
    // noised_count = count + noise = 2 + 0 = 2,
    // non_clamped_average = noised_normalized_sum / noised_count + midpoint = -50 + 5 = -45,
    // BoundedMean.computeResult = clamp(non_clamped_average) = 0 (lower bound).
    assertThat(mean.computeResult()).isEqualTo(0);
  }

  /**
   * This test was designed to be not deterministic. It goes along with deterministic analogues in
   * order to ensure that they don't miss something.
   */
  @Test
  public void computeResult_resultAlwaysInsideProvidedBoundaries() {
    int datasetSize = 10;
    for (int i = 0; i < 100; ++i) {
      Random random = new Random();
      double lower = random.nextDouble() * 100;
      double upper = lower + random.nextDouble() * 100;
      BoundedMean mean =
          BoundedMean.builder()
              .epsilon(EPSILON)
              .noise(new LaplaceNoise())
              .maxPartitionsContributed(1)
              .maxContributionsPerPartition(1)
              .lower(lower)
              .upper(upper)
              .build();

      List<Double> dataset =
          random
              .doubles()
              .map(x -> x * 300 * getRandomSign(random))
              .limit(datasetSize)
              .boxed()
              .collect(toImmutableList());

      mean.addEntries(dataset);

      assertWithMessage(
              "lower = %s\nupper = %s\ndataset = [%s]",
              lower, upper, dataset.stream().map(x -> Double.toString(x)).collect(joining(",\n")))
          .that(mean.computeResult())
          .isIn(Range.closed(lower, upper));
    }
  }

  @Test
  public void computeResult_lowLInfSensitivity_noiseAdded() {
    BoundedMean mean =
        BoundedMean.builder()
            .noise(new LaplaceNoise())
            .epsilon(0.123)
            .lower(0)
            .upper(10e-21)
            .maxContributionsPerPartition(1)
            .maxPartitionsContributed(1)
            .build();
    for (int i = 0; i < 100; i++) {
      mean.addEntry(5e-21);
    }

    double result = mean.computeResult();

    assertThat(result).isNotEqualTo(5e-21);
  }

  @Test
  public void getSerializableSummary_calledAfterComputeResult_throwsException() {
    BoundedMean mean = getMeanBuilder().build();

    mean.computeResult();

    assertThrows(IllegalStateException.class, mean::getSerializableSummary);
  }

  @Test
  public void getSerializableSummary_multipleCalls_returnsSameSummary() {
    BoundedMean mean = getMeanBuilder().build();
    mean.addEntry(0.5);

    byte[] summary1 = mean.getSerializableSummary();
    byte[] summary2 = mean.getSerializableSummary();

    assertThat(summary1).isEqualTo(summary2);
  }

  @Test
  public void mergeWith_basicExample_meansValues() {
    BoundedMean targetMean = getMeanBuilder().noise(new ZeroNoise()).build();
    BoundedMean sourceMean = getMeanBuilder().noise(new ZeroNoise()).build();
    targetMean.addEntry(1);
    sourceMean.addEntry(9);

    targetMean.mergeWith(sourceMean.getSerializableSummary());

    assertThat(targetMean.computeResult()).isEqualTo(5);
  }

  @Test
  public void mergeWith_calledTwice_meansValues() {
    BoundedMean targetMean = getMeanBuilder().noise(new ZeroNoise()).build();
    BoundedMean sourceMean1 = getMeanBuilder().noise(new ZeroNoise()).build();
    BoundedMean sourceMean2 = getMeanBuilder().noise(new ZeroNoise()).build();
    targetMean.addEntry(1);
    sourceMean1.addEntry(2);
    sourceMean2.addEntry(3);

    targetMean.mergeWith(sourceMean1.getSerializableSummary());
    targetMean.mergeWith(sourceMean2.getSerializableSummary());

    assertThat(targetMean.computeResult()).isEqualTo(2);
  }

  @Test
  public void mergeWith_epsilonMismatch_throwsException() {
    BoundedMean targetMean = getMeanBuilder().epsilon(EPSILON).build();
    BoundedMean sourceMean = getMeanBuilder().epsilon(2 * EPSILON).build();

    assertThrows(
        IllegalArgumentException.class,
        () -> targetMean.mergeWith(sourceMean.getSerializableSummary()));
  }

  @Test
  public void mergeWith_nullDelta_mergesWithoutException() {
    BoundedMean mean = getMeanBuilder().noise(new LaplaceNoise()).delta(0.0).build();
    BoundedMean sameMean = getMeanBuilder().noise(new LaplaceNoise()).delta(0.0).build();

    // No exception should be thrown.
    mean.mergeWith(sameMean.getSerializableSummary());
  }

  @Test
  public void mergeWith_deltaMismatch_throwsException() {
    BoundedMean mean = getMeanBuilder().delta(DELTA).build();
    BoundedMean meanDifferentDelta = getMeanBuilder().delta(2 * DELTA).build();

    assertThrows(
        IllegalArgumentException.class,
        () -> mean.mergeWith(meanDifferentDelta.getSerializableSummary()));
  }

  @Test
  public void mergeWith_noiseMismatch_throwsException() {
    BoundedMean mean = getMeanBuilder().noise(new LaplaceNoise()).delta(0.0).build();
    BoundedMean meanDifferentNoise = getMeanBuilder().noise(new GaussianNoise()).build();

    assertThrows(
        IllegalArgumentException.class,
        () -> mean.mergeWith(meanDifferentNoise.getSerializableSummary()));
  }

  @Test
  public void mergeWith_maxPartitionsContributedMismatch_throwsException() {
    BoundedMean mean = getMeanBuilder().maxPartitionsContributed(1).build();
    BoundedMean meanDifferentContributionBound =
        getMeanBuilder().maxPartitionsContributed(2).build();

    assertThrows(
        IllegalArgumentException.class,
        () -> mean.mergeWith(meanDifferentContributionBound.getSerializableSummary()));
  }

  @Test
  public void mergeWith_differentMaxContributionsPerPartitionMismatch_throwsException() {
    BoundedMean mean = getMeanBuilder().maxContributionsPerPartition(1).build();
    BoundedMean meanDifferentContributionBound =
        getMeanBuilder().maxContributionsPerPartition(2).build();

    assertThrows(
        IllegalArgumentException.class,
        () -> mean.mergeWith(meanDifferentContributionBound.getSerializableSummary()));
  }

  @Test
  public void mergeWith_lowerBoundsMismatch_throwsException() {
    BoundedMean mean = getMeanBuilder().lower(-1).build();
    BoundedMean meanDifferentContributionBound = getMeanBuilder().lower(-100).build();

    assertThrows(
        IllegalArgumentException.class,
        () -> mean.mergeWith(meanDifferentContributionBound.getSerializableSummary()));
  }

  @Test
  public void mergeWith_upperBoundsMismatch_throwsException() {
    BoundedMean mean = getMeanBuilder().upper(1).build();
    BoundedMean meanDifferentContributionBound = getMeanBuilder().upper(100).build();

    assertThrows(
        IllegalArgumentException.class,
        () -> mean.mergeWith(meanDifferentContributionBound.getSerializableSummary()));
  }

  @Test
  public void mergeWith_calledAfterComputeResult_throwsException() {
    BoundedMean mean = getMeanBuilder().build();
    BoundedMean otherMean = getMeanBuilder().build();
    byte[] summary = mean.getSerializableSummary();
    otherMean.computeResult();

    assertThrows(IllegalStateException.class, () -> otherMean.mergeWith(summary));
  }

  @Test
  public void mergeWith_calledAfterSerializationOnTargetMean_throwsException() {
    BoundedMean mean = getMeanBuilder().build();
    BoundedMean otherMean = getMeanBuilder().build();
    byte[] summary = otherMean.getSerializableSummary();
    mean.getSerializableSummary();

    assertThrows(IllegalStateException.class, () -> mean.mergeWith(summary));
  }

  private BoundedMean.Params.Builder getMeanBuilder() {
    return BoundedMean.builder()
        .epsilon(EPSILON)
        .delta(DELTA)
        .noise(mockNoise)
        .maxPartitionsContributed(1)
        // lower, upper and, maxContributionsPerPartition have arbitrarily chosen values.
        .maxContributionsPerPartition(10)
        .lower(-10)
        .upper(10);
  }

  private void mockDoubleNoise(double value) {
    when(mockNoise.addNoise(anyDouble(), anyInt(), anyDouble(), anyDouble(), anyDouble()))
        .thenAnswer(invocation -> (double) invocation.getArguments()[0] + value);
  }

  private void mockLongNoise(long value) {
    when(mockNoise.addNoise(anyLong(), anyInt(), anyLong(), anyDouble(), anyDouble()))
        .thenAnswer(invocation -> (long) invocation.getArguments()[0] + value);
  }

  private static int getRandomSign(Random random) {
    return random.nextBoolean() ? 1 : -1;
  }
}

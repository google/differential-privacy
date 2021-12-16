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
 * Tests the accuracy of {@link BoundedMean}. The test mocks {@link Noise} instance which generates
 * zero noise.
 *
 * <p>Statistical and DP properties of the algorithm are tested in {@link
 * com.google.privacy.differentialprivacy.statistical.BoundedMeanDpTest}.
 */
@RunWith(JUnit4.class)
public class BoundedMeanTest {
  private static final double EPSILON = 1.0;
  private static final double DELTA = 0.123;
  @Rule public final MockitoRule mocks = MockitoJUnit.rule();
  @Mock private Noise noise;
  private BoundedMean mean;

  @Before
  public void setUp() {
    when(noise.getMechanismType()).thenReturn(MechanismType.GAUSSIAN);
    // Mock the noise mechanism so that it does not add any noise.
    mockDoubleNoise(0);
    mockLongNoise(0);

    mean =
        BoundedMean.builder()
            .epsilon(EPSILON)
            .delta(DELTA)
            .noise(noise)
            .maxPartitionsContributed(1)
            .maxContributionsPerPartition(1)
            .lower(1.0)
            .upper(9.0)
            .build();
  }

  @Test
  public void addEntry() {
    mean.addEntry(2.0);
    mean.addEntry(4.0);
    mean.addEntry(6.0);
    mean.addEntry(8.0);

    assertThat(mean.computeResult()).isEqualTo(5.0);
  }

  @Test
  public void addEntry_Nan_ignored() {
    // Add NaN - no exception is thrown.
    mean.addEntry(NaN);
    // Add any values (let's say 7 and 9). Verify that the result is equal to their mean.
    mean.addEntry(7);
    mean.addEntry(9);
    assertThat(mean.computeResult()).isEqualTo(8.0);
  }

  @Test
  public void addEntry_calledAfterComputeResult_throwsException() {
    mean.computeResult();
    assertThrows(IllegalStateException.class, () -> mean.addEntry(0.0));
  }

  @Test
  public void addEntry_calledAfterSerialize_throwsException() {
    mean.getSerializableSummary();
    assertThrows(IllegalStateException.class, () -> mean.addEntry(0.0));
  }

  @Test
  public void addEntries() {
    mean.addEntries(Arrays.asList(2.0, 4.0, 6.0, 8.0));
    assertThat(mean.computeResult()).isEqualTo(5.0);
  }

  @Test
  public void addEntries_calledAfterComputeResult_throwsException() {
    mean.computeResult();
    assertThrows(IllegalStateException.class, () -> mean.addEntries(Arrays.asList(0.0)));
  }

  @Test
  public void addEntries_calledAfterSerialize_throwsException() {
    mean.getSerializableSummary();
    assertThrows(IllegalStateException.class, () -> mean.addEntries(Arrays.asList(0.0)));
  }

  @Test
  public void computeResult_multipleCalls_throwsException() {
    mean.computeResult();
    assertThrows(IllegalStateException.class, () -> mean.computeResult());
  }

  @Test
  public void computeResult_calledAfterSerialize_throwsException() {
    mean.getSerializableSummary();
    assertThrows(IllegalStateException.class, () -> mean.computeResult());
  }

  // Input values are clamped to the upper and lower bounds.
  @Test
  public void addEntry_clampsInput() {
    mean =
        BoundedMean.builder()
            .epsilon(EPSILON)
            .delta(DELTA)
            .noise(noise)
            .maxPartitionsContributed(1)
            .maxContributionsPerPartition(1)
            .lower(0.0)
            .upper(2.0)
            .build();

    mean.addEntry(-1.0); // will be clamped to 0
    mean.addEntry(1.0); // will not be clamped
    mean.addEntry(10.0); // will be clamped to 2

    assertThat(mean.computeResult()).isEqualTo(/* (0 + 1 + 2) / 3 */ 1.0);
  }

  @Test
  public void computeResult_singleInput_returnsInput() {
    mean =
        BoundedMean.builder()
            .epsilon(EPSILON)
            .delta(DELTA)
            .noise(noise)
            .maxPartitionsContributed(1)
            .maxContributionsPerPartition(1)
            .lower(1.0)
            .upper(9.0)
            .build();

    mean.addEntry(3.0);
    assertThat(mean.computeResult()).isEqualTo(3.0);
  }

  @Test
  public void computeResult_noInput_returnsMidpoint() {
    mean =
        BoundedMean.builder()
            .epsilon(EPSILON)
            .delta(DELTA)
            .noise(noise)
            .maxPartitionsContributed(1)
            .maxContributionsPerPartition(1)
            .lower(1.0)
            .upper(9.0)
            .build();

    assertThat(mean.computeResult()).isEqualTo(/* midpoint = (1 + 9) / 2 */ 5.0);
  }

  @Test
  public void computeResult_callsNoiseCorrectly() {
    int maxPartitionsContributed = 1;
    int maxContributionsPerPartition = 3;
    mean =
        BoundedMean.builder()
            .epsilon(EPSILON)
            .delta(DELTA)
            .noise(noise)
            .maxPartitionsContributed(maxPartitionsContributed)
            .maxContributionsPerPartition(maxContributionsPerPartition)
            .lower(1.0)
            .upper(9.0)
            .build();
    mean.addEntry(2.0);
    mean.addEntry(4.0);
    mean.computeResult();

    // Noising normalized sum.
    verify(noise)
        .addNoise(
            eq(/* x1 + x2 - midpoint * count = 2 + 4 - 5 * 2*/ -4.0),
            eq(maxPartitionsContributed),
            eq(/* maxContributionsPerPartition * (upper - lower) / 2 = 3 * (9 - 1) / 2 */ 12.0),
            eq(EPSILON / 2.0),
            eq(DELTA / 2.0));

    // Noising count.
    verify(noise)
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

    mean =
        BoundedMean.builder()
            .epsilon(EPSILON)
            .delta(DELTA)
            .noise(noise)
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

    mean =
        BoundedMean.builder()
            .epsilon(EPSILON)
            .delta(DELTA)
            .noise(noise)
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

    mean =
        BoundedMean.builder()
            .epsilon(EPSILON)
            .delta(DELTA)
            .noise(noise)
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

    mean =
        BoundedMean.builder()
            .epsilon(EPSILON)
            .delta(DELTA)
            .noise(noise)
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

      mean =
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
  public void getSerializableSummary_calledAfterComputeResult_throwsException() {
    mean.computeResult();
    assertThrows(IllegalStateException.class, () -> mean.getSerializableSummary());
  }

  @Test
  public void getSerializableSummary_multipleCalls_returnsSameSummary() {
    mean =
        BoundedMean.builder()
            .epsilon(EPSILON)
            .noise(new LaplaceNoise())
            .maxPartitionsContributed(1)
            .maxContributionsPerPartition(1)
            .lower(0.0)
            .upper(1.0)
            .build();
    mean.addEntry(0.5);
    byte[] summary1 = mean.getSerializableSummary();
    byte[] summary2 = mean.getSerializableSummary();
    assertThat(summary1).isEqualTo(summary2);
  }

  @Test
  public void mergeWith_basicExample_meansValues() {
    BoundedMean targetMean = getMeanBuilder().build();
    BoundedMean sourceMean = getMeanBuilder().build();

    targetMean.addEntry(1);
    sourceMean.addEntry(9);

    targetMean.mergeWith(sourceMean.getSerializableSummary());

    assertThat(targetMean.computeResult()).isEqualTo(5);
  }

  @Test
  public void mergeWith_calledTwice_meansValues() {
    BoundedMean targetMean = getMeanBuilder().build();
    BoundedMean sourceMean1 = getMeanBuilder().build();
    BoundedMean sourceMean2 = getMeanBuilder().build();

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
    BoundedMean targetMean = getMeanBuilder().noise(new LaplaceNoise()).delta(null).build();
    BoundedMean sourceMean = getMeanBuilder().noise(new LaplaceNoise()).delta(null).build();
    // No exception should be thrown.
    targetMean.mergeWith(sourceMean.getSerializableSummary());
  }

  @Test
  public void mergeWith_deltaMismatch_throwsException() {
    BoundedMean targetMean = getMeanBuilder().delta(DELTA).build();
    BoundedMean sourceMean = getMeanBuilder().delta(2 * DELTA).build();
    assertThrows(
        IllegalArgumentException.class,
        () -> targetMean.mergeWith(sourceMean.getSerializableSummary()));
  }

  @Test
  public void mergeWith_noiseMismatch_throwsException() {
    BoundedMean targetMean = getMeanBuilder().noise(new LaplaceNoise()).delta(null).build();
    BoundedMean sourceMean = getMeanBuilder().noise(new GaussianNoise()).build();
    assertThrows(
        IllegalArgumentException.class,
        () -> targetMean.mergeWith(sourceMean.getSerializableSummary()));
  }

  @Test
  public void mergeWith_maxPartitionsContributedMismatch_throwsException() {
    BoundedMean targetMean = getMeanBuilder().maxPartitionsContributed(1).build();
    BoundedMean sourceMean = getMeanBuilder().maxPartitionsContributed(2).build();
    assertThrows(
        IllegalArgumentException.class,
        () -> targetMean.mergeWith(sourceMean.getSerializableSummary()));
  }

  @Test
  public void mergeWith_differentMaxContributionsPerPartitionMismatch_throwsException() {
    BoundedMean targetMean = getMeanBuilder().maxContributionsPerPartition(1).build();
    BoundedMean sourceMean = getMeanBuilder().maxContributionsPerPartition(2).build();
    assertThrows(
        IllegalArgumentException.class,
        () -> targetMean.mergeWith(sourceMean.getSerializableSummary()));
  }

  @Test
  public void mergeWith_lowerBoundsMismatch_throwsException() {
    BoundedMean targetMean = getMeanBuilder().lower(-1).build();
    BoundedMean sourceMean = getMeanBuilder().lower(-100).build();
    assertThrows(
        IllegalArgumentException.class,
        () -> targetMean.mergeWith(sourceMean.getSerializableSummary()));
  }

  @Test
  public void mergeWith_upperBoundsMismatch_throwsException() {
    BoundedMean targetMean = getMeanBuilder().upper(1).build();
    BoundedMean sourceMean = getMeanBuilder().upper(100).build();
    assertThrows(
        IllegalArgumentException.class,
        () -> targetMean.mergeWith(sourceMean.getSerializableSummary()));
  }

  @Test
  public void mergeWith_calledAfterComputeResult_throwsException() {
    BoundedMean targetMean = getMeanBuilder().build();
    BoundedMean sourceMean = getMeanBuilder().build();

    targetMean.computeResult();
    byte[] summary = sourceMean.getSerializableSummary();
    assertThrows(IllegalStateException.class, () -> targetMean.mergeWith(summary));
  }

  @Test
  public void mergeWith_calledAfterSerializationOnTargetMean_throwsException() {
    BoundedMean targetMean = getMeanBuilder().build();
    BoundedMean sourceMean = getMeanBuilder().build();

    targetMean.getSerializableSummary();
    byte[] summary = sourceMean.getSerializableSummary();
    assertThrows(IllegalStateException.class, () -> targetMean.mergeWith(summary));
  }

  private BoundedMean.Params.Builder getMeanBuilder() {
    return BoundedMean.builder()
        .epsilon(EPSILON)
        .delta(DELTA)
        .noise(noise)
        .maxPartitionsContributed(1)
        // lower, upper and, maxContributionsPerPartition have arbitrarily chosen values.
        .maxContributionsPerPartition(10)
        .lower(-10)
        .upper(10);
  }

  private void mockDoubleNoise(double value) {
    when(noise.addNoise(anyDouble(), anyInt(), anyDouble(), anyDouble(), anyDouble()))
        .thenAnswer(invocation -> (double) invocation.getArguments()[0] + value);
  }

  private void mockLongNoise(long value) {
    when(noise.addNoise(anyLong(), anyInt(), anyLong(), anyDouble(), anyDouble()))
        .thenAnswer(invocation -> (long) invocation.getArguments()[0] + value);
  }

  private static int getRandomSign(Random random) {
    return random.nextBoolean() ? 1 : -1;
  }
}

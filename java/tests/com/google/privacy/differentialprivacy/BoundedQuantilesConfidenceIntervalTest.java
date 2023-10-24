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

package com.google.privacy.differentialprivacy;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;
import static org.mockito.ArgumentMatchers.anyDouble;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.privacy.differentialprivacy.proto.SummaryOuterClass.MechanismType;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnit;
import org.mockito.junit.MockitoRule;

/** Tests the confidence intervals provided by {@link BoundedQuantiles}. */
@RunWith(JUnit4.class)
public class BoundedQuantilesConfidenceIntervalTest {
  private static final double ARBITRARY_EPSILON = 5.5;
  private static final int ARBITRARY_MAX_CONTRIBUTIONS_PER_PARTITION = 1;
  private static final int ARBITRARY_MAX_PARTITIONS_CONTRIBUTED = 1;
  private static final double ARBITRARY_LOWER = -2.68545;
  private static final double ARBITRARY_UPPER = 2.68545;
  private static final int ARBITRARY_TREE_HEIGHT = 10;
  private static final int ARBITRARY_BRANCHING_FACTOR = 3;
  private static final double ARBITRARY_RANK = 0.54321;
  private static final double ARBITRARY_ALPHA = 0.23645;
  private static final ImmutableList<Double> ARBITRARY_DISTRIBUTION =
      ImmutableList.of(
          -3.0, -3.0, -2.98, -2.97, -2.95, -2.95, -1.99, -1.94, -1.89, -0.5, -0.43, -0.43, 0.02,
          0.02, 0.021, 0.22, 0.22, 0.29, 0.32, 0.38, 0.68, 0.69, 0.81, 0.86, 0.99, 1.32, 22.1);
  // Number of samples drawn for the statistical evaluation of the confidence intervals.
  private static final int NUMBER_OF_SAMPLES = 2500;
  private static final ImmutableList<Double> RANKS =
      ImmutableList.of(0.0, 0.005, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.995, 1.0);
  private static final ImmutableList<Double> ALPHAS =
      ImmutableList.of(0.0001, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99);
  // Minimum number of times the raw value needs to be contained in the confidence interval for a
  // given alpha so that the statistical test accepts. The failure probability is less than 10^-6.
  private static final ImmutableMap<Double, Integer> ACCEPT_THRESHOLDS =
      new ImmutableMap.Builder<Double, Integer>()
          .put(0.0001, 2494)
          .put(0.001, 2486)
          .put(0.01, 2447)
          .put(0.05, 2319)
          .put(0.1, 2175)
          .put(0.25, 1769)
          .put(0.5, 1130)
          .put(0.75, 523)
          .put(0.9, 181)
          .put(0.95, 76)
          .put(0.99, 4)
          .build();

  @Rule public final MockitoRule mocks = MockitoJUnit.rule();

  private BoundedQuantiles.Params.Builder builder;
  @Mock private Noise mockNoise;

  @Before
  public void setUp() {
    builder =
        BoundedQuantiles.builder()
            .epsilon(ARBITRARY_EPSILON)
            .maxPartitionsContributed(ARBITRARY_MAX_PARTITIONS_CONTRIBUTED)
            .maxContributionsPerPartition(ARBITRARY_MAX_CONTRIBUTIONS_PER_PARTITION)
            .lower(ARBITRARY_LOWER)
            .upper(ARBITRARY_UPPER)
            .treeHeight(ARBITRARY_TREE_HEIGHT)
            .branchingFactor(ARBITRARY_BRANCHING_FACTOR);
  }

  @Test
  public void computeConfidenceInterval_noNoise_intervalMatchesQuantile() {
    // Mock a noise mechanism that adds no noise and returns confidence intervals of size 0.
    when(mockNoise.addNoise(anyDouble(), anyInt(), anyDouble(), anyDouble(), anyDouble()))
        .thenAnswer(invocation -> invocation.getArguments()[0]);
    when(mockNoise.computeConfidenceInterval(
            anyDouble(), anyInt(), anyDouble(), anyDouble(), anyDouble(), anyDouble()))
        .thenAnswer(
            invocation ->
                ConfidenceInterval.create(
                    (double) invocation.getArguments()[0], (double) invocation.getArguments()[0]));

    BoundedQuantiles quantiles = builder.noise(mockNoise).build();
    quantiles.addEntries(ARBITRARY_DISTRIBUTION);

    for (double rank : RANKS) {
      double rawValue = quantiles.computeResult(rank);
      ConfidenceInterval confidenceInterval =
          quantiles.computeConfidenceInterval(rank, ARBITRARY_ALPHA);

      assertThat(confidenceInterval.lowerBound()).isEqualTo(rawValue);
      assertThat(confidenceInterval.upperBound()).isEqualTo(rawValue);
    }
  }

  @Test
  public void computeConfidenceInterval_zeroConfidenceLevel_intervalMatchesNoisedQuantile() {
    // In general it is not possible to specify a confidence level of zero. To simulate this, we
    // mock a noise mechanism that adds noise, but returns confidence intervals of size 0.
    Noise noise = new GaussianNoise();
    when(mockNoise.addNoise(anyDouble(), anyInt(), anyDouble(), anyDouble(), anyDouble()))
        .thenAnswer(
            invocation ->
                noise.addNoise(
                    (double) invocation.getArguments()[0],
                    (int) invocation.getArguments()[1],
                    (double) invocation.getArguments()[2],
                    (double) invocation.getArguments()[3],
                    (double) invocation.getArguments()[4]));
    when(mockNoise.computeConfidenceInterval(
            anyDouble(), anyInt(), anyDouble(), anyDouble(), anyDouble(), anyDouble()))
        .thenAnswer(
            invocation ->
                ConfidenceInterval.create(
                    (double) invocation.getArguments()[0], (double) invocation.getArguments()[0]));

    BoundedQuantiles quantiles = builder.noise(mockNoise).delta(1e-5).build();
    quantiles.addEntries(ARBITRARY_DISTRIBUTION);

    for (double rank : RANKS) {
      double rawValue = quantiles.computeResult(rank);
      ConfidenceInterval confidenceInterval =
          quantiles.computeConfidenceInterval(rank, ARBITRARY_ALPHA);

      assertThat(confidenceInterval.lowerBound()).isEqualTo(rawValue);
      assertThat(confidenceInterval.upperBound()).isEqualTo(rawValue);
    }
  }

  @Test
  public void computeConfidenceInterval_emptyDistribution_lowerBoundLessThanUpperBound() {
    for (int i = 0; i < 1000; i++) {
      BoundedQuantiles quantiles = builder.build();
      // Compute an arbitrary quantile to apply noise and enable confidence interval queries.
      quantiles.computeResult(ARBITRARY_RANK);

      // Use a large alpha to increase the chance of a violation.
      for (double rank : RANKS) {
        assertThat(quantiles.computeConfidenceInterval(rank, /*alpha=*/ 0.99).lowerBound())
            .isAtMost(quantiles.computeConfidenceInterval(rank, /*alpha=*/ 0.99).upperBound());
      }
    }
  }

  @Test
  public void computeConfidenceInterval_arbitraryDistribution_lowerBoundLessThanUpperBound() {
    for (int i = 0; i < 1000; i++) {
      BoundedQuantiles quantiles = builder.build();
      quantiles.addEntries(ARBITRARY_DISTRIBUTION);
      // Compute an arbitrary quantile to apply noise and enable confidence interval queries.
      quantiles.computeResult(ARBITRARY_RANK);

      // Use a large alpha to increase the chance of a violation.
      for (double rank : RANKS) {
        assertThat(quantiles.computeConfidenceInterval(rank, /*alpha=*/ 0.99).lowerBound())
            .isAtMost(quantiles.computeConfidenceInterval(rank, /*alpha=*/ 0.99).upperBound());
      }
    }
  }

  @Test
  public void computeConfidenceInterval_resultWithinBounds() {
    builder = builder.lower(1.0).upper(2.0);

    for (int i = 0; i < 1000; i++) {
      BoundedQuantiles quantiles = builder.build();
      quantiles.addEntries(ImmutableList.of(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0));
      quantiles.addEntries(ImmutableList.of(2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0));
      // Compute an arbitrary quantile to apply noise and enable confidence interval queries.
      quantiles.computeResult(ARBITRARY_RANK);

      // To increase the chance of a violation, we use a small alpha and consider the confidence
      // intervals of the min and max quantile, which match the bounds of the input range.
      assertThat(quantiles.computeConfidenceInterval(/*rank=*/ 0.0, /*alpha=*/ 0.01).lowerBound())
          .isAtLeast(1.0);
      assertThat(quantiles.computeConfidenceInterval(/*rank=*/ 1.0, /*alpha=*/ 0.01).upperBound())
          .isAtMost(2.0);
    }
  }

  @Test
  public void computeConfidenceInterval_gaussianNoise_calledTwiceForSameAlpha_returnsSameResult() {
    BoundedQuantiles quantiles = builder.noise(new GaussianNoise()).delta(0.1).build();
    quantiles.addEntries(ARBITRARY_DISTRIBUTION);
    // Compute an arbitrary quantile to apply noise and enable confidence interval queries.
    quantiles.computeResult(ARBITRARY_RANK);

    for (double rank : RANKS) {
      assertThat(quantiles.computeConfidenceInterval(rank, ARBITRARY_ALPHA))
          .isEqualTo(quantiles.computeConfidenceInterval(rank, ARBITRARY_ALPHA));
    }
  }

  @Test
  public void computeConfidenceInterval_laplaceNoise_calledTwiceForSameAlpha_returnsSameResult() {
    BoundedQuantiles quantiles = builder.noise(new LaplaceNoise()).delta(0.0).build();
    quantiles.addEntries(ARBITRARY_DISTRIBUTION);
    // Compute an arbitrary quantile to apply noise and enable confidence interval queries.
    quantiles.computeResult(ARBITRARY_RANK);

    for (double rank : RANKS) {
      assertThat(quantiles.computeConfidenceInterval(rank, ARBITRARY_ALPHA))
          .isEqualTo(quantiles.computeConfidenceInterval(rank, ARBITRARY_ALPHA));
    }
  }

  @Test
  public void
      computeConfidenceInterval_gaussianNoise_resultForSmallAlphaContainedInResultForLargeAlpha() {
    BoundedQuantiles quantiles = builder.noise(new GaussianNoise()).delta(0.1).build();
    quantiles.addEntries(ARBITRARY_DISTRIBUTION);
    // Compute an arbitrary quantile to apply noise and enable confidence interval queries.
    quantiles.computeResult(ARBITRARY_RANK);

    for (double rank : RANKS) {
      assertThat(quantiles.computeConfidenceInterval(rank, ARBITRARY_ALPHA * 0.5).lowerBound())
          .isAtMost(quantiles.computeConfidenceInterval(rank, ARBITRARY_ALPHA).lowerBound());
      assertThat(quantiles.computeConfidenceInterval(rank, ARBITRARY_ALPHA * 0.5).upperBound())
          .isAtLeast(quantiles.computeConfidenceInterval(rank, ARBITRARY_ALPHA).upperBound());
    }
  }

  @Test
  public void
      computeConfidenceInterval_laplaceNoise_resultForSmallAlphaContainedInResultForLargeAlpha() {
    BoundedQuantiles quantiles = builder.noise(new LaplaceNoise()).delta(0.0).build();
    quantiles.addEntries(ARBITRARY_DISTRIBUTION);
    // Compute an arbitrary quantile to apply noise and enable confidence interval queries.
    quantiles.computeResult(ARBITRARY_RANK);

    for (double rank : RANKS) {
      assertThat(quantiles.computeConfidenceInterval(rank, ARBITRARY_ALPHA * 0.5).lowerBound())
          .isAtMost(quantiles.computeConfidenceInterval(rank, ARBITRARY_ALPHA).lowerBound());
      assertThat(quantiles.computeConfidenceInterval(rank, ARBITRARY_ALPHA * 0.5).upperBound())
          .isAtLeast(quantiles.computeConfidenceInterval(rank, ARBITRARY_ALPHA).upperBound());
    }
  }

  @Test
  public void computeConfidenceInterval_oneEntry_gaussianNoise_satisfiesConfidenceLevel() {
    statisticallyAssertConfidenceLevel(
        /* entries= */ ImmutableList.of(0.0), builder, new GaussianNoise());
  }

  @Test
  public void computeConfidenceInterval_oneEntry_laplaceNoise_satisfiesConfidenceLevel() {
    statisticallyAssertConfidenceLevel(
        /* entries= */ ImmutableList.of(0.0), builder, new LaplaceNoise());
  }

  @Test
  public void computeConfidenceInterval_uniformEntries_gaussianNoise_satisfiesConfidenceLevel() {
    ImmutableList.Builder<Double> entriesBuilder = new ImmutableList.Builder<>();
    for (int j = 0; j <= 500; j++) {
      entriesBuilder.add(j / 500.0);
    }
    ImmutableList<Double> entries = entriesBuilder.build();

    statisticallyAssertConfidenceLevel(entries, builder, new GaussianNoise());
  }

  @Test
  public void computeConfidenceInterval_uniformEntries_laplaceNoise_satisfiesConfidenceLevel() {
    ImmutableList.Builder<Double> entriesBuilder = new ImmutableList.Builder<>();
    for (int j = 0; j <= 500; j++) {
      entriesBuilder.add(j / 500.0);
    }
    ImmutableList<Double> entries = entriesBuilder.build();

    statisticallyAssertConfidenceLevel(entries, builder, new LaplaceNoise());
  }

  @Test
  public void computeConfidenceInterval_constantEntries_gaussianNoise_satisfiesConfidenceLevel() {
    ImmutableList.Builder<Double> entriesBuilder = new ImmutableList.Builder<>();
    for (int j = 0; j <= 20; j++) {
      entriesBuilder.add(1.5);
    }
    ImmutableList<Double> entries = entriesBuilder.build();

    statisticallyAssertConfidenceLevel(entries, builder, new GaussianNoise());
  }

  @Test
  public void computeConfidenceInterval_constantEntries_laplaceNoise_satisfiesConfidenceLevel() {
    ImmutableList.Builder<Double> entriesBuilder = new ImmutableList.Builder<>();
    for (int j = 0; j <= 20; j++) {
      entriesBuilder.add(1.5);
    }
    ImmutableList<Double> entries = entriesBuilder.build();

    statisticallyAssertConfidenceLevel(entries, builder, new LaplaceNoise());
  }

  @Test
  public void computeConfidenceInterval_bernoulliEntries_gaussianNoise_satisfiesConfidenceLevel() {
    ImmutableList.Builder<Double> entriesBuilder = new ImmutableList.Builder<>();
    for (int j = 0; j <= 100; j++) {
      entriesBuilder.add(1.0);
      entriesBuilder.add(-1.0);
    }
    ImmutableList<Double> entries = entriesBuilder.build();

    statisticallyAssertConfidenceLevel(entries, builder, new GaussianNoise());
  }

  @Test
  public void computeConfidenceInterval_bernoulliEntries_laplaceNoise_satisfiesConfidenceLevel() {
    ImmutableList.Builder<Double> entriesBuilder = new ImmutableList.Builder<>();
    for (int j = 0; j <= 100; j++) {
      entriesBuilder.add(1.0);
      entriesBuilder.add(-1.0);
    }
    ImmutableList<Double> entries = entriesBuilder.build();

    statisticallyAssertConfidenceLevel(entries, builder, new LaplaceNoise());
  }

  @Test
  public void computeConfidenceInterval_calledBeforeComputeResult_throwsException() {
    BoundedQuantiles quantiles = builder.build();
    assertThrows(
        IllegalStateException.class,
        () -> quantiles.computeConfidenceInterval(ARBITRARY_RANK, ARBITRARY_ALPHA));
  }

  @Test
  public void computeConfidenceInterval_calledAfterSerialization_throwsException() {
    BoundedQuantiles quantiles = builder.build();
    quantiles.getSerializableSummary();
    assertThrows(
        IllegalStateException.class,
        () -> quantiles.computeConfidenceInterval(ARBITRARY_RANK, ARBITRARY_ALPHA));
  }

  private void statisticallyAssertConfidenceLevel(
      List<Double> entries, BoundedQuantiles.Params.Builder builder, Noise noise) {
    // Prepare a hit counter that counts the number of times a confidence interval contains the raw
    // value keyed by rank and alpha.
    Map<Double, Map<Double, Integer>> hitCounter = new HashMap<>();
    for (double rank : RANKS) {
      hitCounter.put(rank, new HashMap<>());
      for (double alpha : ALPHAS) {
        hitCounter.get(rank).put(alpha, /* hit count */ 0);
      }
    }

    // Mock a noise mechanism that adds no noise.
    when(mockNoise.addNoise(anyDouble(), anyInt(), anyDouble(), anyDouble(), anyDouble()))
        .thenAnswer(invocation -> invocation.getArguments()[0]);

    // Approximate the raw quantile values by querying a zero noise instance of the quantiles
    // mechanism.
    Map<Double, Double> rawValues = new HashMap<>();
    BoundedQuantiles zeroNoiseQuantiles = builder.noise(mockNoise).build();
    zeroNoiseQuantiles.addEntries(entries);
    for (double rank : RANKS) {
      rawValues.put(rank, zeroNoiseQuantiles.computeResult(rank));
    }

    builder.noise(noise);
    if (noise.getMechanismType() == MechanismType.GAUSSIAN) {
      builder.delta(1e-5);
    }

    // Sample the hit frequencies.
    for (int i = 0; i < NUMBER_OF_SAMPLES; i++) {
      BoundedQuantiles quantiles = builder.build();
      quantiles.addEntries(entries);
      // Compute an arbitrary quantile to apply noise and enable confidence interval queries.
      quantiles.computeResult(ARBITRARY_RANK);

      // Check whether the confidence intervals contain the respective raw value for all ranks and
      // alphas.
      for (double rank : RANKS) {
        for (double alpha : ALPHAS) {
          if (quantiles.computeConfidenceInterval(rank, alpha).lowerBound() <= rawValues.get(rank)
              && rawValues.get(rank)
                  <= quantiles.computeConfidenceInterval(rank, alpha).upperBound()) {
            hitCounter.get(rank).put(alpha, hitCounter.get(rank).get(alpha) + 1);
          }
        }
      }
    }

    // Assert that the hit frequency was sufficiently large for all ranks and alphas.
    for (double rank : RANKS) {
      for (double alpha : ALPHAS) {
        assertThat(hitCounter.get(rank).get(alpha)).isAtLeast(ACCEPT_THRESHOLDS.get(alpha));
      }
    }
  }
}

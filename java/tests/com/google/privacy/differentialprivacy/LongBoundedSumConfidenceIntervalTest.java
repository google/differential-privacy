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
import static org.junit.Assert.assertThrows;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests the confidence intervals provided by {@link LongBoundedSum}. */
@RunWith(JUnit4.class)
public class LongBoundedSumConfidenceIntervalTest {
  private static final double ARBITRARY_EPSILON = 0.5;
  private static final double ARBITRARY_DELTA = 0.00001;
  private static final int ARBITRARY_MAX_CONTRIBUTIONS_PER_PARTITION = 1;
  private static final int ARBITRARY_MAX_PARTITIONS_CONTRIBUTED = 1;
  private static final long ARBITRARY_LOWER = -3;
  private static final long ARBITRARY_UPPER = 3;
  private static final double ARBITRARY_ALPHA = 0.23645;

  private LongBoundedSum.Params.Builder builder;

  @Before
  public void setUp() {
    builder =
        LongBoundedSum.builder()
            .epsilon(ARBITRARY_EPSILON)
            .delta(ARBITRARY_DELTA)
            .noise(new GaussianNoise())
            .maxPartitionsContributed(ARBITRARY_MAX_PARTITIONS_CONTRIBUTED)
            .maxContributionsPerPartition(ARBITRARY_MAX_CONTRIBUTIONS_PER_PARTITION)
            .lower(ARBITRARY_LOWER)
            .upper(ARBITRARY_UPPER);
  }

  @Test
  public void computeConfidenceInterval_nonPositiveBounds_clampsPositiveSubinterval() {
    builder = builder.lower(-1).upper(0);

    for (int i = 0; i < 1000; i++) {
      LongBoundedSum sum = builder.build();
      var unused = sum.computeResult();

      // Using a large alpha to get a small confidence interval. This increases the chance of both
      // the lower and the upper bound being clamped.
      assertThat(sum.computeConfidenceInterval(/* alpha= */ 0.99).lowerBound())
          .isAtMost(sum.computeConfidenceInterval(/* alpha= */ 0.99).upperBound());
      assertThat(sum.computeConfidenceInterval(/* alpha= */ 0.99).upperBound()).isAtMost(0.0);

      // Using a small alpha to get a large confidence interval. This increases the chance of only
      // the upper bound being clamped.
      assertThat(sum.computeConfidenceInterval(/* alpha= */ 0.01).lowerBound())
          .isAtMost(sum.computeConfidenceInterval(/* alpha= */ 0.01).upperBound());
      assertThat(sum.computeConfidenceInterval(/* alpha= */ 0.01).upperBound()).isAtMost(0.0);
    }
  }

  @Test
  public void computeConfidenceInterval_nonNegativeBounds_clampsNegativeSubinterval() {
    builder = builder.lower(0).upper(1);

    for (int i = 0; i < 1000; i++) {
      LongBoundedSum sum = builder.build();
      var unused = sum.computeResult();

      // Using a large alpha to get a small confidence interval. This increases the chance of both
      // the lower and the upper bound being clamped.
      assertThat(sum.computeConfidenceInterval(/* alpha= */ 0.99).lowerBound())
          .isAtMost(sum.computeConfidenceInterval(/* alpha= */ 0.99).upperBound());
      assertThat(sum.computeConfidenceInterval(/* alpha= */ 0.99).lowerBound()).isAtLeast(0.0);

      // Using a small alpha to get a large confidence interval. This increases the chance of only
      // the the upper bound being clamped.
      assertThat(sum.computeConfidenceInterval(/* alpha= */ 0.01).lowerBound())
          .isAtMost(sum.computeConfidenceInterval(/* alpha= */ 0.01).upperBound());
      assertThat(sum.computeConfidenceInterval(/* alpha= */ 0.01).lowerBound()).isAtLeast(0.0);
    }
  }

  @Test
  public void computeConfidenceInterval_gaussianNoise_calledTwiceForSameAlpha_returnsSameResult() {
    LongBoundedSum sum = builder.noise(new GaussianNoise()).build();
    var unused = sum.computeResult();

    assertThat(sum.computeConfidenceInterval(ARBITRARY_ALPHA))
        .isEqualTo(sum.computeConfidenceInterval(ARBITRARY_ALPHA));
  }

  @Test
  public void computeConfidenceInterval_laplaceNoise_calledTwiceForSameAlpha_returnsSameResult() {
    LongBoundedSum sum = builder.noise(new LaplaceNoise()).delta(0.0).build();
    var unused = sum.computeResult();

    assertThat(sum.computeConfidenceInterval(ARBITRARY_ALPHA))
        .isEqualTo(sum.computeConfidenceInterval(ARBITRARY_ALPHA));
  }

  @Test
  public void
      computeConfidenceInterval_gaussianNoise_resultForSmallAlphaContainedInResultForLargeAlpha() {
    LongBoundedSum sum = builder.noise(new GaussianNoise()).build();
    var unused = sum.computeResult();

    assertThat(sum.computeConfidenceInterval(ARBITRARY_ALPHA * 0.5).lowerBound())
        .isLessThan(sum.computeConfidenceInterval(ARBITRARY_ALPHA).lowerBound());
    assertThat(sum.computeConfidenceInterval(ARBITRARY_ALPHA * 0.5).upperBound())
        .isGreaterThan(sum.computeConfidenceInterval(ARBITRARY_ALPHA).upperBound());
  }

  @Test
  public void
      computeConfidenceInterval_laplaceNoise_resultForSmallAlphaContainedInResultForLargeAlpha() {
    LongBoundedSum sum = builder.noise(new LaplaceNoise()).delta(0.0).build();
    var unused = sum.computeResult();

    assertThat(sum.computeConfidenceInterval(ARBITRARY_ALPHA * 0.5).lowerBound())
        .isLessThan(sum.computeConfidenceInterval(ARBITRARY_ALPHA).lowerBound());
    assertThat(sum.computeConfidenceInterval(ARBITRARY_ALPHA * 0.5).upperBound())
        .isGreaterThan(sum.computeConfidenceInterval(ARBITRARY_ALPHA).upperBound());
  }

  @Test
  public void
      computeConfidenceInterval_gaussianNoise_resultMatchesConfidenceIntervalOfGaussianNoise() {
    // Lower and upper bounds have different signs, so no clamping should occur.
    LongBoundedSum sum =
        builder
            .epsilon(ARBITRARY_EPSILON)
            .delta(ARBITRARY_DELTA)
            .noise(new GaussianNoise())
            .maxPartitionsContributed(ARBITRARY_MAX_PARTITIONS_CONTRIBUTED)
            .maxContributionsPerPartition(ARBITRARY_MAX_CONTRIBUTIONS_PER_PARTITION)
            .lower(-875)
            .upper(38)
            .build();
    long result = sum.computeResult();

    assertThat(sum.computeConfidenceInterval(ARBITRARY_ALPHA))
        .isEqualTo(
            new GaussianNoise()
                .computeConfidenceInterval(
                    result,
                    /* l0Sensitivity= */ ARBITRARY_MAX_PARTITIONS_CONTRIBUTED,
                    /* lInfSensitivity= */ ARBITRARY_MAX_CONTRIBUTIONS_PER_PARTITION * 875,
                    ARBITRARY_EPSILON,
                    ARBITRARY_DELTA,
                    ARBITRARY_ALPHA));
  }

  @Test
  public void
      computeConfidenceInterval_laplaceNoise_resultMatchesConfidenceIntervalOfLaplaceNoise() {
    // Lower and upper bounds have different signs, so no clamping should occur.
    LongBoundedSum sum =
        builder
            .epsilon(ARBITRARY_EPSILON)
            .delta(0.0)
            .noise(new LaplaceNoise())
            .maxPartitionsContributed(ARBITRARY_MAX_PARTITIONS_CONTRIBUTED)
            .maxContributionsPerPartition(ARBITRARY_MAX_CONTRIBUTIONS_PER_PARTITION)
            .lower(-875)
            .upper(38)
            .build();
    long result = sum.computeResult();

    assertThat(sum.computeConfidenceInterval(ARBITRARY_ALPHA))
        .isEqualTo(
            new LaplaceNoise()
                .computeConfidenceInterval(
                    result,
                    /* l0Sensitivity= */ ARBITRARY_MAX_PARTITIONS_CONTRIBUTED,
                    /* lInfSensitivity= */ ARBITRARY_MAX_CONTRIBUTIONS_PER_PARTITION * 875,
                    ARBITRARY_EPSILON,
                    /* delta= */ 0.0,
                    ARBITRARY_ALPHA));
  }

  @Test
  public void computeConfidenceInterval_gaussianNoise_smallAlpha_satisfiesConfidenceLevel() {
    builder.noise(new GaussianNoise());
    long rawValue = 1;

    int hits = 0;
    for (int i = 0; i < 100000; i++) {
      LongBoundedSum sum = builder.build();
      sum.addEntry(rawValue);
      var unused = sum.computeResult();

      if (sum.computeConfidenceInterval(/* alpha= */ 0.1).lowerBound() <= rawValue
          && rawValue <= sum.computeConfidenceInterval(/* alpha= */ 0.1).upperBound()) {
        hits++;
      }
    }
    // Assuming that the true alpha of the confidence interval mechanism is 0.1, i.e., the raw value
    // is within the confidence interval with probability of at least 0.9, then the hits count will
    // be at least 89546 with probability greater than 1 - 10^-6.
    assertThat(hits).isAtLeast(89546);
  }

  @Test
  public void computeConfidenceInterval_laplaceNoise_smallAlpha_satisfiesConfidenceLevel() {
    builder.noise(new LaplaceNoise()).delta(0.0);
    long rawValue = 1;

    int hits = 0;
    for (int i = 0; i < 100000; i++) {
      LongBoundedSum sum = builder.build();
      sum.addEntry(rawValue);
      var unused = sum.computeResult();

      if (sum.computeConfidenceInterval(/* alpha= */ 0.1).lowerBound() <= rawValue
          && rawValue <= sum.computeConfidenceInterval(/* alpha= */ 0.1).upperBound()) {
        hits++;
      }
    }
    // Assuming that the true alpha of the confidence interval mechanism is 0.1, i.e., the raw value
    // is within the confidence interval with probability of at least 0.9, then the hits count will
    // be at least 89546 with probability greater than 1 - 10^-6.
    assertThat(hits).isAtLeast(89546);
  }

  @Test
  public void computeConfidenceInterval_gaussianNoise_largeAlpha_satisfiesConfidenceLevel() {
    builder.noise(new GaussianNoise());
    long rawValue = 1;

    int hits = 0;
    for (int i = 0; i < 100000; i++) {
      LongBoundedSum sum = builder.build();
      sum.addEntry(rawValue);
      var unused = sum.computeResult();

      if (sum.computeConfidenceInterval(/* alpha= */ 0.9).lowerBound() <= rawValue
          && rawValue <= sum.computeConfidenceInterval(/* alpha= */ 0.9).upperBound()) {
        hits++;
      }
    }
    // Assuming that the true alpha of the confidence interval mechanism is 0.9, i.e., the raw value
    // is within the confidence interval with probability of at least 0.1, then the hits count will
    // be at least 9552 with probability greater than 1 - 10^-6.
    assertThat(hits).isAtLeast(9552);
  }

  @Test
  public void computeConfidenceInterval_laplaceNoise_largeAlpha_satisfiesConfidenceLevel() {
    builder.noise(new LaplaceNoise()).delta(0.0);
    long rawValue = 1;

    int hits = 0;
    for (int i = 0; i < 100000; i++) {
      LongBoundedSum sum = builder.build();
      sum.addEntry(rawValue);
      var unused = sum.computeResult();

      if (sum.computeConfidenceInterval(/* alpha= */ 0.9).lowerBound() <= rawValue
          && rawValue <= sum.computeConfidenceInterval(/* alpha= */ 0.9).upperBound()) {
        hits++;
      }
    }
    // Assuming that the true alpha of the confidence interval mechanism is 0.9, i.e., the raw value
    // is within the confidence interval with probability of at least 0.1, then the hits count will
    // be at least 9552 with probability greater than 1 - 10^-6.
    assertThat(hits).isAtLeast(9552);
  }

  @Test
  public void computeConfidenceInterval_calledBeforeComputeResult_throwsException() {
    LongBoundedSum sum = builder.build();
    assertThrows(IllegalStateException.class, () -> sum.computeConfidenceInterval(ARBITRARY_ALPHA));
  }

  @Test
  public void computeConfidenceInterval_calledAfterSerialization_throwsException() {
    LongBoundedSum sum = builder.build();
    var unused = sum.getSerializableSummary();
    assertThrows(IllegalStateException.class, () -> sum.computeConfidenceInterval(ARBITRARY_ALPHA));
  }
}

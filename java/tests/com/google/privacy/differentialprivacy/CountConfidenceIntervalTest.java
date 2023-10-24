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

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests the confidence intervals provided by {@link Count}. */
@RunWith(JUnit4.class)
public class CountConfidenceIntervalTest {
  private static final double ARBITRARY_EPSILON = 0.5;
  private static final double ARBITRARY_DELTA = 0.00001;
  private static final int ARBITRARY_MAX_PARTITIONS_CONTRIBUTED = 1;
  private static final double ARBITRARY_ALPHA = 0.23645;

  private Count.Params.Builder builder;

  @Before
  public void setUp() {
    builder =
        Count.builder()
            .epsilon(ARBITRARY_EPSILON)
            .delta(ARBITRARY_DELTA)
            .noise(new GaussianNoise())
            .maxPartitionsContributed(ARBITRARY_MAX_PARTITIONS_CONTRIBUTED);
  }

  @Test
  public void computeConfidenceInterval_clampsNegativeSubinterval() {
    for (int i = 0; i < 1000; i++) {
      Count count = builder.build();
      count.computeResult();

      // Using a large alpha to get a small confidence interval. This increases the chance of both
      // the lower and the upper bound being clamped.
      assertThat(count.computeConfidenceInterval(/*alpha=*/ 0.99).lowerBound()).isAtLeast(0.0);
      assertThat(count.computeConfidenceInterval(/*alpha=*/ 0.99).upperBound()).isAtLeast(0.0);

      // Using a small alpha to get a large confidence interval. This increases the chance of only
      // the the upper bound being clamped.
      assertThat(count.computeConfidenceInterval(/*alpha=*/ 0.01).lowerBound()).isAtLeast(0.0);
      assertThat(count.computeConfidenceInterval(/*alpha=*/ 0.01).upperBound()).isAtLeast(0.0);
    }
  }

  @Test
  public void computeConfidenceInterval_gaussianNoise_calledTwiceForSameAlpha_returnsSameResult() {
    Count count = builder.noise(new GaussianNoise()).build();
    count.incrementBy(100000000); // incrementing by large number to prevent clamping
    count.computeResult();

    assertThat(count.computeConfidenceInterval(ARBITRARY_ALPHA))
        .isEqualTo(count.computeConfidenceInterval(ARBITRARY_ALPHA));
  }

  @Test
  public void computeConfidenceInterval_laplaceNoise_calledTwiceForSameAlpha_returnsSameResult() {
    Count count = builder.noise(new LaplaceNoise()).delta(null).build();
    count.incrementBy(100000000); // incrementing by large number to prevent clamping
    count.computeResult();

    assertThat(count.computeConfidenceInterval(ARBITRARY_ALPHA))
        .isEqualTo(count.computeConfidenceInterval(ARBITRARY_ALPHA));
  }

  @Test
  public void
      computeConfidenceInterval_gaussianNoise_resultForSmallAlphaContainedInResultForLargeAlpha() {
    Count count = builder.noise(new GaussianNoise()).build();
    count.incrementBy(100000000); // incrementing by large number to prevent clamping
    count.computeResult();

    assertThat(count.computeConfidenceInterval(ARBITRARY_ALPHA * 0.5).lowerBound())
        .isLessThan(count.computeConfidenceInterval(ARBITRARY_ALPHA).lowerBound());
    assertThat(count.computeConfidenceInterval(ARBITRARY_ALPHA * 0.5).upperBound())
        .isGreaterThan(count.computeConfidenceInterval(ARBITRARY_ALPHA).upperBound());
  }

  @Test
  public void
      computeConfidenceInterval_laplaceNoise_resultForSmallAlphaContainedInResultForLargeAlpha() {
    Count count = builder.noise(new LaplaceNoise()).delta(null).build();
    count.incrementBy(100000000); // incrementing by large number to prevent clamping
    count.computeResult();

    assertThat(count.computeConfidenceInterval(ARBITRARY_ALPHA * 0.5).lowerBound())
        .isLessThan(count.computeConfidenceInterval(ARBITRARY_ALPHA).lowerBound());
    assertThat(count.computeConfidenceInterval(ARBITRARY_ALPHA * 0.5).upperBound())
        .isGreaterThan(count.computeConfidenceInterval(ARBITRARY_ALPHA).upperBound());
  }

  @Test
  public void
      computeConfidenceInterval_gaussianNoise_resultMatchesConfidenceIntervalOfGaussianNoise() {
    Count count =
        builder
            .epsilon(ARBITRARY_EPSILON)
            .delta(ARBITRARY_DELTA)
            .noise(new GaussianNoise())
            .maxPartitionsContributed(ARBITRARY_MAX_PARTITIONS_CONTRIBUTED)
            .build();
    count.incrementBy(100000000); // incrementing by large number to prevent clamping
    long result = count.computeResult();

    assertThat(count.computeConfidenceInterval(ARBITRARY_ALPHA))
        .isEqualTo(
            new GaussianNoise()
                .computeConfidenceInterval(
                    result,
                    /*l0Sensitivity=*/ ARBITRARY_MAX_PARTITIONS_CONTRIBUTED,
                    /*lInfSensitivity=*/ 1,
                    ARBITRARY_EPSILON,
                    ARBITRARY_DELTA,
                    ARBITRARY_ALPHA));
  }

  @Test
  public void
      computeConfidenceInterval_laplaceNoise_resultMatchesConfidenceIntervalOfLaplaceNoise() {
    Count count =
        builder
            .epsilon(ARBITRARY_EPSILON)
            .delta(null)
            .noise(new LaplaceNoise())
            .maxPartitionsContributed(ARBITRARY_MAX_PARTITIONS_CONTRIBUTED)
            .build();
    count.incrementBy(100000000); // incrementing by large number to prevent clamping
    long result = count.computeResult();

    assertThat(count.computeConfidenceInterval(ARBITRARY_ALPHA))
        .isEqualTo(
            new LaplaceNoise()
                .computeConfidenceInterval(
                    result,
                    /*l0Sensitivity=*/ ARBITRARY_MAX_PARTITIONS_CONTRIBUTED,
                    /*lInfSensitivity=*/ 1,
                    ARBITRARY_EPSILON,
                    /*delta=*/ null,
                    ARBITRARY_ALPHA));
  }

  @Test
  public void computeConfidenceInterval_gaussianNoise_smallAlpha_satisfiesConfidenceLevel() {
    builder.noise(new GaussianNoise());
    int rawCount = 14523;

    int hits = 0;
    for (int i = 0; i < 100000; i++) {
      Count count = builder.build();
      count.incrementBy(rawCount);
      count.computeResult();

      if (count.computeConfidenceInterval(/*alpha=*/ 0.1).lowerBound() <= rawCount
          && rawCount <= count.computeConfidenceInterval(/*alpha=*/ 0.1).upperBound()) {
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
    int rawCount = 14523;

    int hits = 0;
    for (int i = 0; i < 100000; i++) {
      Count count = builder.build();
      count.incrementBy(rawCount);
      count.computeResult();

      if (count.computeConfidenceInterval(/*alpha=*/ 0.1).lowerBound() <= rawCount
          && rawCount <= count.computeConfidenceInterval(/*alpha=*/ 0.1).upperBound()) {
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
    int rawCount = 14523;

    int hits = 0;
    for (int i = 0; i < 100000; i++) {
      Count count = builder.build();
      count.incrementBy(rawCount);
      count.computeResult();

      if (count.computeConfidenceInterval(/*alpha=*/ 0.9).lowerBound() <= rawCount
          && rawCount <= count.computeConfidenceInterval(/*alpha=*/ 0.9).upperBound()) {
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
    int rawCount = 14523;

    int hits = 0;
    for (int i = 0; i < 100000; i++) {
      Count count = builder.build();
      count.incrementBy(rawCount);
      count.computeResult();

      if (count.computeConfidenceInterval(/*alpha=*/ 0.9).lowerBound() <= rawCount
          && rawCount <= count.computeConfidenceInterval(/*alpha=*/ 0.9).upperBound()) {
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
    Count count = builder.build();
    assertThrows(
        IllegalStateException.class, () -> count.computeConfidenceInterval(ARBITRARY_ALPHA));
  }

  @Test
  public void computeConfidenceInterval_calledAfterSerialization_throwsException() {
    Count count = builder.build();
    count.getSerializableSummary();
    assertThrows(
        IllegalStateException.class, () -> count.computeConfidenceInterval(ARBITRARY_ALPHA));
  }
}

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

/** Tests the confidence intervals provided by {@link BoundedMean}. */
@RunWith(JUnit4.class)
public class BoundedMeanConfidenceIntervalTest {
  private static final double ARBITRARY_EPSILON = 0.5;
  private static final double ARBITRARY_DELTA = 0.00001;
  private static final int ARBITRARY_MAX_CONTRIBUTIONS_PER_PARTITION = 1;
  private static final int ARBITRARY_MAX_PARTITIONS_CONTRIBUTED = 1;
  private static final double ARBITRARY_LOWER = -2.68545;
  private static final double ARBITRARY_UPPER = 2.68545;
  private static final double ARBITRARY_ALPHA = 0.23645;

  private BoundedMean.Params.Builder builder;

  @Before
  public void setUp() {
    builder =
        BoundedMean.builder()
            .epsilon(ARBITRARY_EPSILON)
            .delta(ARBITRARY_DELTA)
            .noise(new GaussianNoise())
            .maxPartitionsContributed(ARBITRARY_MAX_PARTITIONS_CONTRIBUTED)
            .maxContributionsPerPartition(ARBITRARY_MAX_CONTRIBUTIONS_PER_PARTITION)
            .lower(ARBITRARY_LOWER)
            .upper(ARBITRARY_UPPER);
  }

  @Test
  public void computeConfidenceInterval_emptyMean_clampsToBounds() {
    // For empty instances of mean, the confidence interval of the denominator is likely to contain
    // negative values. This should not cause the mean's confidence interval to exceed the bounds.
    builder = builder.lower(-1.0).upper(1.0);

    for (int i = 0; i < 1000; i++) {
      BoundedMean mean = builder.build();
      mean.computeResult();

      // Using a large alpha to get small confidence intervals. This increases the chance of the
      // denominator's confidence interval to be completely negative.
      ConfidenceInterval ci = mean.computeConfidenceInterval(/*alpha=*/ 0.99);
      assertThat(ci.lowerBound()).isAtMost(ci.upperBound());
      assertThat(ci.lowerBound()).isAtLeast(-1.0);
      assertThat(ci.upperBound()).isAtMost(1.0);

      // Using a small alpha to get large confidence intervals. This increases the chance of the
      // denominator's confidence interval to be partially negative.
      ci = mean.computeConfidenceInterval(/*alpha=*/ 0.01);
      assertThat(ci.lowerBound()).isAtMost(ci.upperBound());
      assertThat(ci.lowerBound()).isAtLeast(-1.0);
      assertThat(ci.upperBound()).isAtMost(1.0);
    }
  }

  @Test
  public void computeConfidenceInterval_emptyMean_positiveBounds_clampsToBounds() {
    // For empty instances of mean, the confidence interval of the denominator is likely to contain
    // negative values. This should not cause the mean's confidence interval to exceed the bounds.
    builder = builder.lower(1.0).upper(2.0);

    for (int i = 0; i < 1000; i++) {
      BoundedMean mean = builder.build();
      mean.computeResult();

      // Using a large alpha to get small confidence intervals. This increases the chance of the
      // denominator's confidence interval to be completely negative.
      ConfidenceInterval ci = mean.computeConfidenceInterval(/*alpha=*/ 0.99);
      assertThat(ci.lowerBound()).isAtMost(ci.upperBound());
      assertThat(ci.lowerBound()).isAtLeast(1.0);
      assertThat(ci.upperBound()).isAtMost(2.0);

      // Using a small alpha to get large confidence intervals. This increases the chance of the
      // denominator's confidence interval to be partially negative.
      ci = mean.computeConfidenceInterval(/*alpha=*/ 0.01);
      assertThat(ci.lowerBound()).isAtMost(ci.upperBound());
      assertThat(ci.lowerBound()).isAtLeast(1.0);
      assertThat(ci.upperBound()).isAtMost(2.0);
    }
  }

  @Test
  public void computeConfidenceInterval_emptyMean_negativeBounds_clampsToBounds() {
    // For empty instances of mean, the confidence interval of the denominator is likely to contain
    // negative values. This should not cause the mean's confidence interval to exceed the bounds.
    builder = builder.lower(-2.0).upper(-1.0);

    for (int i = 0; i < 1000; i++) {
      BoundedMean mean = builder.build();
      mean.computeResult();

      // Using a large alpha to get small confidence intervals. This increases the chance of the
      // denominator's confidence interval to be completely negative.
      ConfidenceInterval ci = mean.computeConfidenceInterval(/*alpha=*/ 0.99);
      assertThat(ci.lowerBound()).isAtMost(ci.upperBound());
      assertThat(ci.lowerBound()).isAtLeast(-2.0);
      assertThat(ci.upperBound()).isAtMost(-1.0);

      // Using a small alpha to get large confidence intervals. This increases the chance of the
      // denominator's confidence interval to be partially negative.
      ci = mean.computeConfidenceInterval(/*alpha=*/ 0.01);
      assertThat(ci.lowerBound()).isAtMost(ci.upperBound());
      assertThat(ci.lowerBound()).isAtLeast(-2.0);
      assertThat(ci.upperBound()).isAtMost(-1.0);
    }
  }

  @Test
  public void computeConfidenceInterval_rawValueAtUpperBound_clampsToBounds() {
    builder = builder.lower(-1.0).upper(1.0);

    for (int i = 0; i < 1000; i++) {
      BoundedMean mean = builder.build();
      mean.addEntry(1.0);
      mean.computeResult();

      // Using a large alpha to get small confidence intervals. This increases the chance of the
      // denominator's confidence interval to be completely negative.
      ConfidenceInterval ci = mean.computeConfidenceInterval(/*alpha=*/ 0.99);
      assertThat(ci.lowerBound()).isAtMost(ci.upperBound());
      assertThat(ci.lowerBound()).isAtLeast(-1.0);
      assertThat(ci.upperBound()).isAtMost(1.0);

      // Using a small alpha to get large confidence intervals. This increases the chance of the
      // denominator's confidence interval to be partially negative.
      ci = mean.computeConfidenceInterval(/*alpha=*/ 0.01);
      assertThat(ci.lowerBound()).isAtMost(ci.upperBound());
      assertThat(ci.lowerBound()).isAtLeast(-1.0);
      assertThat(ci.upperBound()).isAtMost(1.0);
    }
  }

  @Test
  public void computeConfidenceInterval_rawValueAtLowerBound_clampsToBounds() {
    builder = builder.lower(-1.0).upper(1.0);

    for (int i = 0; i < 1000; i++) {
      BoundedMean mean = builder.build();
      mean.addEntry(-1.0);
      mean.computeResult();

      // Using a large alpha to get small confidence intervals. This increases the chance of the
      // denominator's confidence interval to be completely negative.
      ConfidenceInterval ci = mean.computeConfidenceInterval(/*alpha=*/ 0.99);
      assertThat(ci.lowerBound()).isAtMost(ci.upperBound());
      assertThat(ci.lowerBound()).isAtLeast(-1.0);
      assertThat(ci.upperBound()).isAtMost(1.0);

      // Using a small alpha to get large confidence intervals. This increases the chance of the
      // denominator's confidence interval to be partially negative.
      ci = mean.computeConfidenceInterval(/*alpha=*/ 0.01);
      assertThat(ci.lowerBound()).isAtMost(ci.upperBound());
      assertThat(ci.lowerBound()).isAtLeast(-1.0);
      assertThat(ci.upperBound()).isAtMost(1.0);
    }
  }

  @Test
  public void computeConfidenceInterval_gaussianNoise_calledTwiceForSameAlpha_returnsSameResult() {
    BoundedMean mean = builder.noise(new GaussianNoise()).build();
    mean.computeResult();

    assertThat(mean.computeConfidenceInterval(ARBITRARY_ALPHA))
        .isEqualTo(mean.computeConfidenceInterval(ARBITRARY_ALPHA));
  }

  @Test
  public void computeConfidenceInterval_laplaceNoise_calledTwiceForSameAlpha_returnsSameResult() {
    BoundedMean mean = builder.noise(new LaplaceNoise()).delta(0.0).build();
    mean.computeResult();

    assertThat(mean.computeConfidenceInterval(ARBITRARY_ALPHA))
        .isEqualTo(mean.computeConfidenceInterval(ARBITRARY_ALPHA));
  }

  @Test
  public void
      computeConfidenceInterval_gaussianNoise_resultForSmallAlphaContainedInResultForLargeAlpha() {
    BoundedMean mean = builder.noise(new GaussianNoise()).build();
    // Adding many entries to prevent clamping.
    for (int i = 0; i < 1000; i++) {
      mean.addEntry(0.5);
    }
    mean.computeResult();

    assertThat(mean.computeConfidenceInterval(ARBITRARY_ALPHA * 0.5).lowerBound())
        .isLessThan(mean.computeConfidenceInterval(ARBITRARY_ALPHA).lowerBound());
    assertThat(mean.computeConfidenceInterval(ARBITRARY_ALPHA * 0.5).upperBound())
        .isGreaterThan(mean.computeConfidenceInterval(ARBITRARY_ALPHA).upperBound());
  }

  @Test
  public void
      computeConfidenceInterval_laplaceNoise_resultForSmallAlphaContainedInResultForLargeAlpha() {
    BoundedMean mean = builder.noise(new LaplaceNoise()).delta(0.0).build();
    // Adding many entries to prevent clamping.
    for (int i = 0; i < 1000; i++) {
      mean.addEntry(0.5);
    }
    mean.computeResult();

    assertThat(mean.computeConfidenceInterval(ARBITRARY_ALPHA * 0.5).lowerBound())
        .isLessThan(mean.computeConfidenceInterval(ARBITRARY_ALPHA).lowerBound());
    assertThat(mean.computeConfidenceInterval(ARBITRARY_ALPHA * 0.5).upperBound())
        .isGreaterThan(mean.computeConfidenceInterval(ARBITRARY_ALPHA).upperBound());
  }

  @Test
  public void
      computeConfidenceInterval_gaussianNoise_smallAlpha_emptyMean_satisfiesConfidenceLevel() {
    builder.noise(new GaussianNoise()).lower(1.0).upper(2.0);
    // When empty, the raw mean is assumed to be the midpoint between lower and upper.
    double rawValue = 1.5;

    int hits = 0;
    for (int i = 0; i < 2500; i++) {
      BoundedMean mean = builder.build();
      mean.computeResult();

      ConfidenceInterval ci = mean.computeConfidenceInterval(/*alpha=*/ 0.1);
      if (ci.lowerBound() <= rawValue && rawValue <= ci.upperBound()) {
        hits++;
      }
    }
    // Assuming that the true alpha of the confidence interval mechanism is 0.1, i.e., the raw value
    // is within the confidence interval with probability of at least 0.9, then the hits count will
    // be at least 2176 with probability greater than 1 - 10^-6.
    assertThat(hits).isAtLeast(2176);
  }

  @Test
  public void
      computeConfidenceInterval_laplaceNoise_smallAlpha_emptyMean_satisfiesConfidenceLevel() {
    builder.noise(new LaplaceNoise()).delta(0.0).lower(1.0).upper(2.0);
    // When empty, the raw mean is assumed to be the midpoint between lower and upper.
    double rawValue = 1.5;

    int hits = 0;
    for (int i = 0; i < 2500; i++) {
      BoundedMean mean = builder.build();
      mean.computeResult();

      ConfidenceInterval ci = mean.computeConfidenceInterval(/*alpha=*/ 0.1);
      if (ci.lowerBound() <= rawValue && rawValue <= ci.upperBound()) {
        hits++;
      }
    }
    // Assuming that the true alpha of the confidence interval mechanism is 0.1, i.e., the raw value
    // is within the confidence interval with probability of at least 0.9, then the hits count will
    // be at least 2176 with probability greater than 1 - 10^-6.
    assertThat(hits).isAtLeast(2176);
  }

  @Test
  public void
      computeConfidenceInterval_gaussianNoise_largeAlpha_emptyMean_satisfiesConfidenceLevel() {
    builder.noise(new GaussianNoise()).lower(1.0).upper(2.0);
    // When empty, the raw mean is assumed to be the midpoint between lower and upper.
    double rawValue = 1.5;

    int hits = 0;
    for (int i = 0; i < 2500; i++) {
      BoundedMean mean = builder.build();
      mean.computeResult();

      ConfidenceInterval ci = mean.computeConfidenceInterval(/*alpha=*/ 0.9);
      if (ci.lowerBound() <= rawValue && rawValue <= ci.upperBound()) {
        hits++;
      }
    }
    // Assuming that the true alpha of the confidence interval mechanism is 0.9, i.e., the raw value
    // is within the confidence interval with probability of at least 0.1, then the hits count will
    // be at least 182 with probability greater than 1 - 10^-6.
    assertThat(hits).isAtLeast(182);
  }

  @Test
  public void
      computeConfidenceInterval_laplaceNoise_largeAlpha_emptyMean_satisfiesConfidenceLevel() {
    builder.noise(new LaplaceNoise()).delta(0.0).lower(1.0).upper(2.0);
    // When empty, the raw mean is assumed to be the midpoint between lower and upper.
    double rawValue = 1.5;

    int hits = 0;
    for (int i = 0; i < 2500; i++) {
      BoundedMean mean = builder.build();
      mean.computeResult();

      ConfidenceInterval ci = mean.computeConfidenceInterval(/*alpha=*/ 0.9);
      if (ci.lowerBound() <= rawValue && rawValue <= ci.upperBound()) {
        hits++;
      }
    }
    // Assuming that the true alpha of the confidence interval mechanism is 0.9, i.e., the raw value
    // is within the confidence interval with probability of at least 0.1, then the hits count will
    // be at least 182 with probability greater than 1 - 10^-6.
    assertThat(hits).isAtLeast(182);
  }

  @Test
  public void
      computeConfidenceInterval_gaussianNoise_smallAlpha_oneEntry_satisfiesConfidenceLevel() {
    builder.noise(new GaussianNoise()).lower(1.0).upper(2.0);
    // Choosing the midpoint between lower and upper to maximize the variance of the result. This
    // should increase the likelihood of detecting potential violations of the confidence level.
    double rawValue = 1.5;

    int hits = 0;
    for (int i = 0; i < 2500; i++) {
      BoundedMean mean = builder.build();
      mean.addEntry(rawValue);
      mean.computeResult();

      ConfidenceInterval ci = mean.computeConfidenceInterval(/*alpha=*/ 0.1);
      if (ci.lowerBound() <= rawValue && rawValue <= ci.upperBound()) {
        hits++;
      }
    }
    // Assuming that the true alpha of the confidence interval mechanism is 0.1, i.e., the raw value
    // is within the confidence interval with probability of at least 0.9, then the hits count will
    // be at least 2176 with probability greater than 1 - 10^-6.
    assertThat(hits).isAtLeast(2176);
  }

  @Test
  public void
      computeConfidenceInterval_laplaceNoise_smallAlpha_oneEntry_satisfiesConfidenceLevel() {
    builder.noise(new LaplaceNoise()).delta(0.0).lower(1.0).upper(2.0);
    // Choosing the midpoint between lower and upper to maximize the variance of the result. This
    // should increase the likelihood of detecting potential violations of the confidence level.
    double rawValue = 1.5;

    int hits = 0;
    for (int i = 0; i < 2500; i++) {
      BoundedMean mean = builder.build();
      mean.addEntry(rawValue);
      mean.computeResult();

      ConfidenceInterval ci = mean.computeConfidenceInterval(/*alpha=*/ 0.1);
      if (ci.lowerBound() <= rawValue && rawValue <= ci.upperBound()) {
        hits++;
      }
    }
    // Assuming that the true alpha of the confidence interval mechanism is 0.1, i.e., the raw value
    // is within the confidence interval with probability of at least 0.9, then the hits count will
    // be at least 2176 with probability greater than 1 - 10^-6.
    assertThat(hits).isAtLeast(2176);
  }

  @Test
  public void
      computeConfidenceInterval_gaussianNoise_largeAlpha_oneEntry_satisfiesConfidenceLevel() {
    builder.noise(new GaussianNoise()).lower(1.0).upper(2.0);
    // Choosing the midpoint between lower and upper to maximize the variance of the result. This
    // should increase the likelihood of detecting potential violations of the confidence level.
    double rawValue = 1.5;

    int hits = 0;
    for (int i = 0; i < 2500; i++) {
      BoundedMean mean = builder.build();
      mean.addEntry(rawValue);
      mean.computeResult();

      ConfidenceInterval ci = mean.computeConfidenceInterval(/*alpha=*/ 0.9);
      if (ci.lowerBound() <= rawValue && rawValue <= ci.upperBound()) {
        hits++;
      }
    }
    // Assuming that the true alpha of the confidence interval mechanism is 0.9, i.e., the raw value
    // is within the confidence interval with probability of at least 0.1, then the hits count will
    // be at least 182 with probability greater than 1 - 10^-6.
    assertThat(hits).isAtLeast(182);
  }

  @Test
  public void
      computeConfidenceInterval_laplaceNoise_largeAlpha_oneEntry_satisfiesConfidenceLevel() {
    builder.noise(new LaplaceNoise()).delta(0.0).lower(1.0).upper(2.0);
    // Choosing the midpoint between lower and upper to maximize the variance of the result. This
    // should increase the likelihood of detecting potential violations of the confidence level.
    double rawValue = 1.5;

    int hits = 0;
    for (int i = 0; i < 2500; i++) {
      BoundedMean mean = builder.build();
      mean.addEntry(rawValue);
      mean.computeResult();

      ConfidenceInterval ci = mean.computeConfidenceInterval(/*alpha=*/ 0.9);
      if (ci.lowerBound() <= rawValue && rawValue <= ci.upperBound()) {
        hits++;
      }
    }
    // Assuming that the true alpha of the confidence interval mechanism is 0.9, i.e., the raw value
    // is within the confidence interval with probability of at least 0.1, then the hits count will
    // be at least 182 with probability greater than 1 - 10^-6.
    assertThat(hits).isAtLeast(182);
  }

  @Test
  public void
      computeConfidenceInterval_gaussianNoise_smallAlpha_manyEntries_satisfiesConfidenceLevel() {
    builder.noise(new GaussianNoise()).lower(1.0).upper(2.0);
    // Choosing the midpoint between lower and upper to maximize the variance of the result. This
    // should increase the likelihood of detecting potential violations of the confidence level.
    double rawValue = 1.5;

    int hits = 0;
    for (int i = 0; i < 2500; i++) {
      BoundedMean mean = builder.build();
      for (int j = 0; j < 100; j++) {
        mean.addEntry(rawValue);
      }
      mean.computeResult();

      ConfidenceInterval ci = mean.computeConfidenceInterval(/*alpha=*/ 0.1);
      if (ci.lowerBound() <= rawValue && rawValue <= ci.upperBound()) {
        hits++;
      }
    }
    // Assuming that the true alpha of the confidence interval mechanism is 0.1, i.e., the raw value
    // is within the confidence interval with probability of at least 0.9, then the hits count will
    // be at least 2176 with probability greater than 1 - 10^-6.
    assertThat(hits).isAtLeast(2176);
  }

  @Test
  public void
      computeConfidenceInterval_laplaceNoise_smallAlpha_manyEntries_satisfiesConfidenceLevel() {
    builder.noise(new LaplaceNoise()).delta(0.0).lower(1.0).upper(2.0);
    // Choosing the midpoint between lower and upper to maximize the variance of the result. This
    // should increase the likelihood of detecting potential violations of the confidence level.
    double rawValue = 1.5;

    int hits = 0;
    for (int i = 0; i < 2500; i++) {
      BoundedMean mean = builder.build();
      for (int j = 0; j < 100; j++) {
        mean.addEntry(rawValue);
      }
      mean.computeResult();

      ConfidenceInterval ci = mean.computeConfidenceInterval(/*alpha=*/ 0.1);
      if (ci.lowerBound() <= rawValue && rawValue <= ci.upperBound()) {
        hits++;
      }
    }
    // Assuming that the true alpha of the confidence interval mechanism is 0.1, i.e., the raw value
    // is within the confidence interval with probability of at least 0.9, then the hits count will
    // be at least 2176 with probability greater than 1 - 10^-6.
    assertThat(hits).isAtLeast(2176);
  }

  @Test
  public void
      computeConfidenceInterval_gaussianNoise_largeAlpha_manyEntries_satisfiesConfidenceLevel() {
    builder.noise(new GaussianNoise()).lower(1.0).upper(2.0);
    // Choosing the midpoint between lower and upper to maximize the variance of the result. This
    // should increase the likelihood of detecting potential violations of the confidence level.
    double rawValue = 1.5;

    int hits = 0;
    for (int i = 0; i < 2500; i++) {
      BoundedMean mean = builder.build();
      for (int j = 0; j < 100; j++) {
        mean.addEntry(rawValue);
      }
      mean.computeResult();

      ConfidenceInterval ci = mean.computeConfidenceInterval(/*alpha=*/ 0.9);
      if (ci.lowerBound() <= rawValue && rawValue <= ci.upperBound()) {
        hits++;
      }
    }
    // Assuming that the true alpha of the confidence interval mechanism is 0.9, i.e., the raw value
    // is within the confidence interval with probability of at least 0.1, then the hits count will
    // be at least 182 with probability greater than 1 - 10^-6.
    assertThat(hits).isAtLeast(182);
  }

  @Test
  public void
      computeConfidenceInterval_laplaceNoise_largeAlpha_manyEntries_satisfiesConfidenceLevel() {
    builder.noise(new LaplaceNoise()).delta(0.0).lower(1.0).upper(2.0);
    // Choosing the midpoint between lower and upper to maximize the variance of the result. This
    // should increase the likelihood of detecting potential violations of the confidence level.
    double rawValue = 1.5;

    int hits = 0;
    for (int i = 0; i < 2500; i++) {
      BoundedMean mean = builder.build();
      for (int j = 0; j < 100; j++) {
        mean.addEntry(rawValue);
      }
      mean.computeResult();

      ConfidenceInterval ci = mean.computeConfidenceInterval(/*alpha=*/ 0.9);
      if (ci.lowerBound() <= rawValue && rawValue <= ci.upperBound()) {
        hits++;
      }
    }
    // Assuming that the true alpha of the confidence interval mechanism is 0.9, i.e., the raw value
    // is within the confidence interval with probability of at least 0.1, then the hits count will
    // be at least 182 with probability greater than 1 - 10^-6.
    assertThat(hits).isAtLeast(182);
  }

  @Test
  public void computeConfidenceInterval_calledBeforeComputeResult_throwsException() {
    BoundedMean mean = builder.build();
    assertThrows(
        IllegalStateException.class, () -> mean.computeConfidenceInterval(ARBITRARY_ALPHA));
  }

  @Test
  public void computeConfidenceInterval_calledAfterSerialization_throwsException() {
    BoundedMean mean = builder.build();
    mean.getSerializableSummary();
    assertThrows(
        IllegalStateException.class, () -> mean.computeConfidenceInterval(ARBITRARY_ALPHA));
  }
}

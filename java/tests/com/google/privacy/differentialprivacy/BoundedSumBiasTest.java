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

import com.google.common.collect.ImmutableList;
import com.google.common.math.Stats;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Statistical tests verifying that the {@link BoundedSum} mechanism is unbiased, i.e., the expected
 * value of the result returned by {@link BoundedSum} is equal to the raw bounded sum.
 */
@RunWith(JUnit4.class)
public class BoundedSumBiasTest {
  private static final int NUM_SAMPLES = 100000;
  private static final double LN_3 = Math.log(3.0);

  @Test
  public void computeResult_gaussianNoiseEmptySum_isUnbiased() {
    BoundedSum.Params.Builder sumBuilder =
        BoundedSum.builder()
            .epsilon(LN_3)
            .delta(0.00001)
            .maxPartitionsContributed(1)
            .lower(0.0)
            .upper(1.0)
            .noise(new GaussianNoise());

    // The sample variance is (over) approximated based on the privacy paramters.
    testForBias(sumBuilder, /* rawEntry= */ 0.0, /* sampleVariance= */ 11.8);
  }

  @Test
  public void computeResult_gaussianNoiseDifferentEpsilonEmptySum_isUnbiased() {
    BoundedSum.Params.Builder sumBuilder =
        BoundedSum.builder()
            .epsilon(2.0 * LN_3)
            .delta(0.00001)
            .maxPartitionsContributed(1)
            .lower(0.0)
            .upper(1.0)
            .noise(new GaussianNoise());

    // The sample variance is (over) approximated based on the privacy paramters.
    testForBias(sumBuilder, /* rawEntry= */ 0.0, /* sampleVariance= */ 3.4);
  }

  @Test
  public void computeResult_gaussianNoiseDifferentDeltaEmptySum_isUnbiased() {
    BoundedSum.Params.Builder sumBuilder =
        BoundedSum.builder()
            .epsilon(LN_3)
            .delta(0.01)
            .maxPartitionsContributed(1)
            .lower(0.0)
            .upper(1.0)
            .noise(new GaussianNoise());

    // The sample variance is (over) approximated based on the privacy paramters.
    testForBias(sumBuilder, /* rawEntry= */ 0.0, /* sampleVariance= */ 3.1);
  }

  @Test
  public void computeResult_gaussianNoiseDifferentContributionBoundEmptySum_isUnbiased() {
    BoundedSum.Params.Builder sumBuilder =
        BoundedSum.builder()
            .epsilon(LN_3)
            .delta(0.00001)
            .maxPartitionsContributed(25)
            .lower(0.0)
            .upper(1.0)
            .noise(new GaussianNoise());

    // The sample variance is (over) approximated based on the privacy paramters.
    testForBias(sumBuilder, /* rawEntry= */ 0.0, /* sampleVariance= */ 293.4);
  }

  @Test
  public void computeResult_gaussianNoiseDifferentEntryBoundsEmptySum_isUnbiased() {
    BoundedSum.Params.Builder sumBuilder =
        BoundedSum.builder()
            .epsilon(LN_3)
            .delta(0.00001)
            .maxPartitionsContributed(1)
            .lower(-0.5)
            .upper(0.0)
            .noise(new GaussianNoise());

    // The sample variance is (over) approximated based on the privacy paramters.
    testForBias(sumBuilder, /* rawEntry= */ 0.0, /* sampleVariance= */ 3.0);
  }

  @Test
  public void computeResult_gaussianNoisePositiveEntry_isUnbiased() {
    BoundedSum.Params.Builder sumBuilder =
        BoundedSum.builder()
            .epsilon(LN_3)
            .delta(0.00001)
            .maxPartitionsContributed(1)
            .lower(0.0)
            .upper(1.0)
            .noise(new GaussianNoise());

    // The sample variance is (over) approximated based on the privacy paramters.
    testForBias(sumBuilder, /* rawEntry= */ 1.0, /* sampleVariance= */ 11.8);
  }

  @Test
  public void computeResult_gaussianNoiseNegativeEntry_isUnbiased() {
    BoundedSum.Params.Builder sumBuilder =
        BoundedSum.builder()
            .epsilon(LN_3)
            .delta(0.00001)
            .maxPartitionsContributed(1)
            .lower(-1.0)
            .upper(0.0)
            .noise(new GaussianNoise());

    // The sample variance is (over) approximated based on the privacy paramters.
    testForBias(sumBuilder, /* rawEntry= */ -1.0, /* sampleVariance= */ 11.8);
  }

  @Test
  public void computeResult_laplaceNoiseEmptySum_isUnbiased() {
    BoundedSum.Params.Builder sumBuilder =
        BoundedSum.builder()
            .epsilon(LN_3)
            .maxPartitionsContributed(1)
            .lower(0.0)
            .upper(1.0)
            .noise(new LaplaceNoise());

    // The variance is derived from the privacy paramters.
    testForBias(sumBuilder, /* rawEntry= */ 0.0, /* sampleVariance= */ 1.7);
  }

  @Test
  public void computeResult_laplaceNoiseDifferentEpsilonEmptySum_isUnbiased() {
    BoundedSum.Params.Builder sumBuilder =
        BoundedSum.builder()
            .epsilon(2.0 * LN_3)
            .maxPartitionsContributed(1)
            .lower(0.0)
            .upper(1.0)
            .noise(new LaplaceNoise());

    // The variance is derived from the privacy paramters.
    testForBias(sumBuilder, /* rawEntry= */ 0.0, /* sampleVariance= */ 0.5);
  }

  @Test
  public void computeResult_laplaceNoiseDifferentContributionBoundEmptySum_isUnbiased() {
    BoundedSum.Params.Builder sumBuilder =
        BoundedSum.builder()
            .epsilon(LN_3)
            .maxPartitionsContributed(25)
            .lower(0.0)
            .upper(1.0)
            .noise(new LaplaceNoise());

    // The variance is derived from the privacy paramters.
    testForBias(sumBuilder, /* rawEntry= */ 0.0, /* sampleVariance= */ 1035.7);
  }

  @Test
  public void computeResult_laplaceNoiseDifferentEntryBoundsEmptySum_isUnbiased() {
    BoundedSum.Params.Builder sumBuilder =
        BoundedSum.builder()
            .epsilon(LN_3)
            .maxPartitionsContributed(1)
            .lower(-0.5)
            .upper(0.0)
            .noise(new LaplaceNoise());

    // The variance is derived from the privacy paramters.
    testForBias(sumBuilder, /* rawEntry */ 0.0, /* sampleVariance */ 0.5);
  }

  @Test
  public void computeResult_laplaceNoisePositiveEntry_isUnbiased() {
    BoundedSum.Params.Builder sumBuilder =
        BoundedSum.builder()
            .epsilon(LN_3)
            .maxPartitionsContributed(1)
            .lower(0.0)
            .upper(1.0)
            .noise(new LaplaceNoise());

    // The variance is derived from the privacy paramters.
    testForBias(sumBuilder, /* rawEntry= */ 1.0, /* sampleVariance= */ 1.7);
  }

  @Test
  public void computeResult_laplaceNoiseNegativeEntry_isUnbiased() {
    BoundedSum.Params.Builder sumBuilder =
        BoundedSum.builder()
            .epsilon(LN_3)
            .maxPartitionsContributed(1)
            .lower(-1.0)
            .upper(0.0)
            .noise(new LaplaceNoise());

    // The variance is derived from the privacy paramters.
    testForBias(sumBuilder, /* rawEntry= */ -1.0, /* sampleVariance= */ 1.7);
  }

  private static void testForBias(
      BoundedSum.Params.Builder sumBuilder, double rawEntry, double sampleVariance) {
    ImmutableList.Builder<Double> samples = ImmutableList.builder();
    for (int i = 0; i < NUM_SAMPLES; i++) {
      BoundedSum sum = sumBuilder.build();
      sum.addEntry(rawEntry);
      samples.add(sum.computeResult());
    }
    Stats stats = Stats.of(samples.build());

    // The tolerance is chosen according to the 99.9995% quantile of the anticipated distributions
    // of the sample mean. Thus, the test falsely rejects with a probability of 10^-5.
    double sampleTolerance = 4.41717 * Math.sqrt(sampleVariance / NUM_SAMPLES);
    // The DP count is considered unbiased if the expeted value (approximated by stats.mean()) is
    // equal to the raw count.
    assertThat(stats.mean()).isWithin(sampleTolerance).of(rawEntry);
  }
}

//
// Copyright 2022 Google LLC
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

import com.google.common.collect.ImmutableList;
import com.google.common.math.Stats;
import java.util.Random;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class SamplingUtilTest {
  private static final Random random = new Random();
  private static final int NUM_SAMPLES = 100000;

  @Test
  public void sampleGeometric_lowSuccessProbability_hasAccurateStatisticalProperties() {
    ImmutableList.Builder<Double> samples = ImmutableList.builder();

    for (int i = 0; i < NUM_SAMPLES; i++) {
      // Samples are drawn from a geometric distribution with success probability of p = 10^-6.
      samples.add((double) SamplingUtil.sampleGeometric(random, Math.log(1000000.0 / 999999.0)));
    }

    double mean = 1000000.0;
    double variance = 1E12;
    // The tolerance is chosen according to the 99.9995% quantile of the anticipated distributions
    // of the sample mean and variance. Thus, the test falsely rejects with a probability of 10^-5.
    double sampleMeanTolerance = 4.41717 * Math.sqrt(variance / NUM_SAMPLES);
    double sampleVarianceTolerance = 4.41717 * Math.sqrt(8E24 / NUM_SAMPLES);
    Stats stats = Stats.of(samples.build());
    assertThat(stats.mean()).isWithin(sampleMeanTolerance).of(mean);
    assertThat(stats.populationVariance()).isWithin(sampleVarianceTolerance).of(variance);
  }

  @Test
  public void sampleGeometric_highSuccessProbability_hasAccurateStatisticalProperties() {
    ImmutableList.Builder<Double> samples = ImmutableList.builder();

    for (int i = 0; i < NUM_SAMPLES; i++) {
      // Samples are drawn from a geometric distribution with success probability of p = 0.5.
      samples.add((double) SamplingUtil.sampleGeometric(random, Math.log(2.0)));
    }

    double mean = 2.0;
    double variance = 2.0;
    // The tolerance is chosen according to the 99.9995% quantile of the anticipated distributions
    // of the sample mean and variance. Thus, the test falsely rejects with a probability of 10^-5.
    double sampleMeanTolerance = 4.41717 * Math.sqrt(variance / NUM_SAMPLES);
    double sampleVarianceTolerance = 4.41717 * Math.sqrt(34.0 / NUM_SAMPLES);
    Stats stats = Stats.of(samples.build());
    assertThat(stats.mean()).isWithin(sampleMeanTolerance).of(mean);
    assertThat(stats.populationVariance()).isWithin(sampleVarianceTolerance).of(variance);
  }

  @Test
  public void sampleGeometric_tooSmallLambda_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () -> SamplingUtil.sampleGeometric(random, 1.0 / (1L << 59)));
  }

  @Test
  public void sampleTwoSidedGeometric_lowSuccessProbability_hasAccurateStatisticalProperties() {
    ImmutableList.Builder<Double> samples = ImmutableList.builder();

    for (int i = 0; i < NUM_SAMPLES; i++) {
      // Samples are drawn from a two-sided geometric distribution with success probability of p =
      // 10^-6.
      samples.add(
          (double) SamplingUtil.sampleTwoSidedGeometric(random, Math.log(1000000.0 / 999999.0)));
    }

    double mean = 0;
    double variance = 2E12;
    // The tolerance is chosen according to the 99.9995% quantile of the anticipated distributions
    // of the sample mean and variance. Thus, the test falsely rejects with a probability of 10^-5.
    double sampleMeanTolerance = 4.41717 * Math.sqrt(variance / NUM_SAMPLES);
    double sampleVarianceTolerance = 4.41717 * Math.sqrt(16E24 / NUM_SAMPLES);
    Stats stats = Stats.of(samples.build());
    assertThat(stats.mean()).isWithin(sampleMeanTolerance).of(mean);
    assertThat(stats.populationVariance()).isWithin(sampleVarianceTolerance).of(variance);
  }

  @Test
  public void sampleTwoSidedGeometric_highSuccessProbability_hasAccurateStatisticalProperties() {
    ImmutableList.Builder<Double> samples = ImmutableList.builder();

    for (int i = 0; i < NUM_SAMPLES; i++) {
      // Samples are drawn from a two-sided geometric distribution with success probability of p =
      // 0.5.
      samples.add((double) SamplingUtil.sampleTwoSidedGeometric(random, Math.log(2.0)));
    }

    double mean = 0;
    double variance = 4.0;
    // The tolerance is chosen according to the 99.9995% quantile of the anticipated distributions
    // of the sample mean and variance. Thus, the test falsely rejects with a probability of 10^-5.
    double sampleMeanTolerance = 4.41717 * Math.sqrt(variance / NUM_SAMPLES);
    double sampleVarianceTolerance = 4.41717 * Math.sqrt(68.0 / NUM_SAMPLES);
    Stats stats = Stats.of(samples.build());
    assertThat(stats.mean()).isWithin(sampleMeanTolerance).of(mean);
    assertThat(stats.populationVariance()).isWithin(sampleVarianceTolerance).of(variance);
  }
}

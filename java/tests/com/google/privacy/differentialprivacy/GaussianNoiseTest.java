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
import static com.google.differentialprivacy.SummaryOuterClass.MechanismType.GAUSSIAN;
import static java.lang.Double.NaN;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.math.Stats;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class GaussianNoiseTest {
  private static final GaussianNoise NOISE = new GaussianNoise();
  private static final int NUM_SAMPLES = 100000;
  private static final double LN_3 = Math.log(3);
  private static final double DEFAULT_MEAN = 0.0;
  private static final double DEFAULT_EPSILON = LN_3;
  private static final double DEFAULT_DELTA = 0.00001;
  private static final int DEFAULT_L_0_SENSITIVITY = 1;
  private static final double DEFAULT_L_INF_SENSITIVITY = 1.0;
  private static final double DEFAULT_RANK = 0.00001;

  @Test
  public void addNoise_hasAccurateStatisticalProperties() {
    ImmutableList.Builder<Double> samples = ImmutableList.builder();
    for (int i = 0; i < NUM_SAMPLES; i++) {
      samples.add(
          NOISE.addNoise(
              DEFAULT_MEAN,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              DEFAULT_DELTA));
    }
    Stats stats = Stats.of(samples.build());

    double variance = 11.735977;
    // The tolerance is chosen according to the 99.9995% quantile of the anticipated distributions
    // of the sample mean and variance. Thus, the test falsely rejects with a probability of 10^-5.
    double sampleMeanTolerance = 4.41717 * Math.sqrt(variance / NUM_SAMPLES);
    double sampleVarianceTolerance = 4.41717 * variance * Math.sqrt(2.0 / NUM_SAMPLES);
    assertThat(stats.mean()).isWithin(sampleMeanTolerance).of(DEFAULT_MEAN);
    assertThat(stats.populationVariance()).isWithin(sampleVarianceTolerance).of(variance);
  }

  @Test
  public void addNoise_differentMean_hasAccurateStatisticalProperties() {
    ImmutableList.Builder<Double> samples = ImmutableList.builder();
    for (int i = 0; i < NUM_SAMPLES; i++) {
      samples.add(
          NOISE.addNoise(
              /* mean */ 42.0,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              DEFAULT_DELTA));
    }
    Stats stats = Stats.of(samples.build());

    double mean = 42.0;
    double variance = 11.735977;
    // The tolerance is chosen according to the 99.9995% quantile of the anticipated distributions
    // of the sample mean and variance. Thus, the test falsely rejects with a probability of 10^-5.
    double sampleMeanTolerance = 4.41717 * Math.sqrt(variance / NUM_SAMPLES);
    double sampleVarianceTolerance = 4.41717 * variance * Math.sqrt(2.0 / NUM_SAMPLES);
    assertThat(stats.mean()).isWithin(sampleMeanTolerance).of(mean);
    assertThat(stats.populationVariance()).isWithin(sampleVarianceTolerance).of(variance);
  }

  @Test
  public void addNoise_differentSensitivity_hasAccurateStatisticalProperties() {
    ImmutableList.Builder<Double> samples = ImmutableList.builder();
    for (int i = 0; i < NUM_SAMPLES; i++) {
      samples.add(
          NOISE.addNoise(
              DEFAULT_MEAN,
              DEFAULT_L_0_SENSITIVITY,
              /* lInfSensitivity */ 0.5,
              DEFAULT_EPSILON,
              DEFAULT_DELTA));
    }
    Stats stats = Stats.of(samples.build());

    double variance = 2.9339943;
    // The tolerance is chosen according to the 99.9995% quantile of the anticipated distributions
    // of the sample mean and variance. Thus, the test falsely rejects with a probability of 10^-5.
    double sampleMeanTolerance = 4.41717 * Math.sqrt(variance / NUM_SAMPLES);
    double sampleVarianceTolerance = 4.41717 * variance * Math.sqrt(2.0 / NUM_SAMPLES);
    assertThat(stats.mean()).isWithin(sampleMeanTolerance).of(DEFAULT_MEAN);
    assertThat(stats.populationVariance()).isWithin(sampleVarianceTolerance).of(variance);
  }

  @Test
  public void addNoise_differentEpsilon_hasAccurateStatisticalProperties() {
    ImmutableList.Builder<Double> samples = ImmutableList.builder();
    for (int i = 0; i < NUM_SAMPLES; i++) {
      samples.add(
          NOISE.addNoise(
              DEFAULT_MEAN,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              /* epsilon */ 2 * LN_3,
              DEFAULT_DELTA));
    }
    Stats stats = Stats.of(samples.build());

    double variance = 3.3634987;
    // The tolerance is chosen according to the 99.9995% quantile of the anticipated distributions
    // of the sample mean and variance. Thus, the test falsely rejects with a probability of 10^-5.
    double sampleMeanTolerance = 4.41717 * Math.sqrt(variance / NUM_SAMPLES);
    double sampleVarianceTolerance = 4.41717 * variance * Math.sqrt(2.0 / NUM_SAMPLES);
    assertThat(stats.mean()).isWithin(sampleMeanTolerance).of(DEFAULT_MEAN);
    assertThat(stats.populationVariance()).isWithin(sampleVarianceTolerance).of(variance);
  }

  @Test
  public void addNoise_differentDelta_hasAccurateStatisticalProperties() {
    ImmutableList.Builder<Double> samples = ImmutableList.builder();
    for (int i = 0; i < NUM_SAMPLES; i++) {
      samples.add(
          NOISE.addNoise(
              DEFAULT_MEAN,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              /* delta */ 0.01));
    }
    Stats stats = Stats.of(samples.build());

    double variance = 3.0625;
    // The tolerance is chosen according to the 99.9995% quantile of the anticipated distributions
    // of the sample mean and variance. Thus, the test falsely rejects with a probability of 10^-5.
    double sampleMeanTolerance = 4.41717 * Math.sqrt(variance / NUM_SAMPLES);
    double sampleVarianceTolerance = 4.41717 * variance * Math.sqrt(2.0 / NUM_SAMPLES);
    assertThat(stats.mean()).isWithin(sampleMeanTolerance).of(DEFAULT_MEAN);
    assertThat(stats.populationVariance()).isWithin(sampleVarianceTolerance).of(variance);
  }

  @Test
  public void addNoise_integralMean_hasAccurateStatisticalProperties() {
    ImmutableList.Builder<Long> samples = ImmutableList.builder();
    for (int i = 0; i < NUM_SAMPLES; i++) {
      samples.add(
          NOISE.addNoise(
              /* mean */ 0L,
              DEFAULT_L_0_SENSITIVITY,
              /* lInfSensitivity */ 1L,
              DEFAULT_EPSILON,
              DEFAULT_DELTA));
    }
    Stats stats = Stats.of(samples.build());

    double mean = 0.0;
    double approxVariance = 11.85;
    // The tolerance is chosen according to the 99.9995% quantile of the anticipated distributions
    // of the sample mean and variance. Thus, the test falsely rejects with a probability of 10^-5.
    double sampleMeanTolerance = 4.41717 * Math.sqrt(approxVariance / NUM_SAMPLES);
    assertThat(stats.mean()).isWithin(sampleMeanTolerance).of(mean);
    // Not testing for the variance because it is not clear what variance should be expected.
  }

  @Test
  public void addNoise_epsilonTooSmall_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.addNoise(
                DEFAULT_MEAN,
                DEFAULT_L_0_SENSITIVITY,
                DEFAULT_L_INF_SENSITIVITY,
                /* epsilon */ 1.0 / (1L << 51),
                DEFAULT_DELTA));
  }

  @Test
  public void addNoise_epsilonPosInfinity_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.addNoise(
                DEFAULT_MEAN,
                DEFAULT_L_0_SENSITIVITY,
                DEFAULT_L_INF_SENSITIVITY,
                /* epsilon */ Double.POSITIVE_INFINITY,
                DEFAULT_DELTA));
  }

  @Test
  public void addNoise_epsilonNan_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.addNoise(
                DEFAULT_MEAN,
                DEFAULT_L_0_SENSITIVITY,
                DEFAULT_L_INF_SENSITIVITY,
                /* epsilon */ Double.NaN,
                DEFAULT_DELTA));
  }

  @Test
  public void addNoise_deltaNegative_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.addNoise(
                DEFAULT_MEAN,
                DEFAULT_L_0_SENSITIVITY,
                DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                /* delta */ -1.0));
  }

  @Test
  public void addNoise_deltaOne_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.addNoise(
                DEFAULT_MEAN,
                DEFAULT_L_0_SENSITIVITY,
                DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                /* delta */ 1.0));
  }

  @Test
  public void addNoise_deltaGreaterThanOne_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.addNoise(
                DEFAULT_MEAN,
                DEFAULT_L_0_SENSITIVITY,
                DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                /* delta */ 2.0));
  }

  @Test
  public void addNoise_deltaNull_throwsException() {
    assertThrows(
        NullPointerException.class,
        () ->
            NOISE.addNoise(
                DEFAULT_MEAN,
                DEFAULT_L_0_SENSITIVITY,
                DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                /* delta */ null));
  }

  @Test
  public void addNoise_deltaNan_throwsException() {
    assertThrows(IllegalArgumentException.class, () -> NOISE.addNoise(10, 1, 1, LN_3, NaN));
  }

  @Test
  public void addNoise_l0SensitivityNegative_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.addNoise(
                DEFAULT_MEAN,
                /* l0Sensitivity */ -1,
                DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                DEFAULT_DELTA));
  }

  @Test
  public void addNoise_l0SensitivityZero_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.addNoise(
                DEFAULT_MEAN,
                /* lInfSensitivity */ 0,
                DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                DEFAULT_DELTA));
  }

  @Test
  public void addNoise_lInfSensitivityNegative_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.addNoise(
                DEFAULT_MEAN,
                DEFAULT_L_0_SENSITIVITY,
                /* lInfSensitivity */ -1.0,
                DEFAULT_EPSILON,
                DEFAULT_DELTA));
  }

  @Test
  public void addNoise_lInfSensitivityZero_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.addNoise(
                DEFAULT_MEAN,
                DEFAULT_L_0_SENSITIVITY,
                /* lInfSensitivity */ 0.0,
                DEFAULT_EPSILON,
                DEFAULT_DELTA));
  }

  @Test
  public void addNoise_lInfSensitivityTooHigh_throwsException() {
    IllegalArgumentException thrown =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                NOISE.addNoise(
                    DEFAULT_MEAN,
                    DEFAULT_L_0_SENSITIVITY,
                    /* lInfSensitivity */ Double.MAX_VALUE,
                    DEFAULT_EPSILON,
                    DEFAULT_DELTA));
    assertThat(thrown)
        .hasMessageThat()
        .startsWith("2 * lInfSensitivity must be finite but is Infinity");
  }

  @Test
  public void addNoise_lInfSensitivityNan_throwsException() {
    assertThrows(IllegalArgumentException.class, () -> NOISE.addNoise(0, 1, NaN, 1, DEFAULT_DELTA));
  }

  @Test
  public void sampleSymmetricBinomial_tooSmallN_throwsException() {
    assertThrows(IllegalArgumentException.class, () -> NOISE.sampleSymmetricBinomial(999999.999));
  }

  @Test
  public void sampleSymmetricBinomial_hasAccurateStatisticalProperties() {
    ImmutableList.Builder<Double> samples = ImmutableList.builder();
    for (int i = 0; i < NUM_SAMPLES; i++) {
      samples.add((double) NOISE.sampleSymmetricBinomial(2000000.0));
    }
    Stats stats = Stats.of(samples.build());

    double mean = 0.0;
    double stdDev = 1000000.0;
    // The tolerance is chosen according to the 99.9995% quantile of the anticipated distributions
    // of the sample mean and variance. Thus, the test falsely rejects with a probability of 10^-5.
    double sampleMeanTolerance = 4.41717 * stdDev * Math.sqrt(1.0 / NUM_SAMPLES);
    double sampleVarianceTolerance = 4.41717 * stdDev * stdDev * Math.sqrt(2.0 / NUM_SAMPLES);
    assertThat(stats.mean()).isWithin(sampleMeanTolerance).of(mean);
    assertThat(stats.populationVariance()).isWithin(sampleVarianceTolerance).of(stdDev * stdDev);
  }

  @Test
  public void getMechanismType_returnsGaussian() {
    assertThat(NOISE.getMechanismType()).isEqualTo(GAUSSIAN);
  }

  @Test
  public void computeQuantile_epsilonNegative_throwsException() {
    Exception e =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                NOISE.computeQuantile(
                    DEFAULT_RANK,
                    DEFAULT_MEAN,
                    DEFAULT_L_0_SENSITIVITY,
                    DEFAULT_L_INF_SENSITIVITY,
                    /* epsilon */ -0.1,
                    /* delta */ null));
    assertThat(e).hasMessageThat().startsWith("epsilon must be");
  }

  @Test
  public void computeQuantile_epsilonZero_throwsException() {
    Exception e =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                NOISE.computeQuantile(
                    DEFAULT_RANK,
                    DEFAULT_MEAN,
                    DEFAULT_L_0_SENSITIVITY,
                    DEFAULT_L_INF_SENSITIVITY,
                    /* epsilon */ 0.0,
                    /* delta */ null));
    assertThat(e).hasMessageThat().startsWith("epsilon must be");
  }

  @Test
  public void computeThreshold_epsilonTooSmall_throwsException() {
    Exception e =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                NOISE.computeQuantile(
                    DEFAULT_RANK,
                    DEFAULT_MEAN,
                    DEFAULT_L_0_SENSITIVITY,
                    DEFAULT_L_INF_SENSITIVITY,
                    /* epsilon */ 1.0 / (1L << 51),
                    /* delta */ null));
    assertThat(e).hasMessageThat().startsWith("epsilon must be");
  }

  @Test
  public void computeQuantile_epsilonPosInfinity_throwsException() {
    Exception e =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                NOISE.computeQuantile(
                    DEFAULT_RANK,
                    DEFAULT_MEAN,
                    DEFAULT_L_0_SENSITIVITY,
                    DEFAULT_L_INF_SENSITIVITY,
                    /* epsilon */ Double.POSITIVE_INFINITY,
                    /* delta */ null));
    assertThat(e).hasMessageThat().startsWith("epsilon must be");
  }

  @Test
  public void computeQuantile_epsilonNan_throwsException() {
    Exception e =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                NOISE.computeQuantile(
                    DEFAULT_RANK,
                    DEFAULT_MEAN,
                    DEFAULT_L_0_SENSITIVITY,
                    DEFAULT_L_INF_SENSITIVITY,
                    /* epsilon */ Double.NaN,
                    /* delta */ null));
    assertThat(e).hasMessageThat().startsWith("epsilon must be");
  }

  @Test
  public void computeQuantile_deltaNull_throwsException() {
    assertThrows(
        NullPointerException.class,
        () ->
            NOISE.computeQuantile(
                DEFAULT_RANK,
                DEFAULT_MEAN,
                DEFAULT_L_0_SENSITIVITY,
                DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                /* delta */ null));
  }

  @Test
  public void computeQuantile_deltaNegative_throwsException() {
    Exception e =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                NOISE.computeQuantile(
                    DEFAULT_RANK,
                    DEFAULT_MEAN,
                    DEFAULT_L_0_SENSITIVITY,
                    DEFAULT_L_INF_SENSITIVITY,
                    DEFAULT_EPSILON,
                    /* delta */ -0.1));
    assertThat(e).hasMessageThat().startsWith("delta must be");
  }

  @Test
  public void computeQuantile_deltaZero_throwsException() {
    Exception e =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                NOISE.computeQuantile(
                    DEFAULT_RANK,
                    DEFAULT_MEAN,
                    DEFAULT_L_0_SENSITIVITY,
                    DEFAULT_L_INF_SENSITIVITY,
                    DEFAULT_EPSILON,
                    /* delta */ 0.0));
    assertThat(e).hasMessageThat().startsWith("delta must be");
  }

  @Test
  public void computeQuantile_deltaOne_throwsException() {
    Exception e =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                NOISE.computeQuantile(
                    DEFAULT_RANK,
                    DEFAULT_MEAN,
                    DEFAULT_L_0_SENSITIVITY,
                    DEFAULT_L_INF_SENSITIVITY,
                    DEFAULT_EPSILON,
                    /* delta */ 1.0));
    assertThat(e).hasMessageThat().startsWith("delta must be");
  }

  @Test
  public void computeQuantile_deltaGreaterThanOne_throwsException() {
    Exception e =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                NOISE.computeQuantile(
                    DEFAULT_RANK,
                    DEFAULT_MEAN,
                    DEFAULT_L_0_SENSITIVITY,
                    DEFAULT_L_INF_SENSITIVITY,
                    DEFAULT_EPSILON,
                    /* delta */ 2.0));
    assertThat(e).hasMessageThat().startsWith("delta must be");
  }

  @Test
  public void computeQuantile_deltaNaN_throwsException() {
    Exception e =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                NOISE.computeQuantile(
                    DEFAULT_RANK,
                    DEFAULT_MEAN,
                    DEFAULT_L_0_SENSITIVITY,
                    DEFAULT_L_INF_SENSITIVITY,
                    DEFAULT_EPSILON,
                    /* delta */ NaN));
    assertThat(e).hasMessageThat().startsWith("delta must be");
  }

  @Test
  public void computeQuantile_rankNegative_throwsException() {
    Exception e =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                NOISE.computeQuantile(
                    /* rank */ -1.0,
                    DEFAULT_MEAN,
                    DEFAULT_L_0_SENSITIVITY,
                    DEFAULT_L_INF_SENSITIVITY,
                    DEFAULT_EPSILON,
                    DEFAULT_DELTA));
    assertThat(e).hasMessageThat().startsWith("rank must be");
  }

  @Test
  public void computeQuantile_rankZero_throwsException() {
    Exception e =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                NOISE.computeQuantile(
                    /* rank */ 0.0,
                    DEFAULT_MEAN,
                    DEFAULT_L_0_SENSITIVITY,
                    DEFAULT_L_INF_SENSITIVITY,
                    DEFAULT_EPSILON,
                    DEFAULT_DELTA));
    assertThat(e).hasMessageThat().startsWith("rank must be");
  }

  @Test
  public void computeQuantile_rankOne_throwsException() {
    Exception e =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                NOISE.computeQuantile(
                    /* rank */ 1.0,
                    DEFAULT_MEAN,
                    DEFAULT_L_0_SENSITIVITY,
                    DEFAULT_L_INF_SENSITIVITY,
                    DEFAULT_EPSILON,
                    DEFAULT_DELTA));
    assertThat(e).hasMessageThat().startsWith("rank must be");
  }

  @Test
  public void computeQuantile_rankGreaterThanOne_throwsException() {
    Exception e =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                NOISE.computeQuantile(
                    /* rank */ 2.0,
                    DEFAULT_MEAN,
                    DEFAULT_L_0_SENSITIVITY,
                    DEFAULT_L_INF_SENSITIVITY,
                    DEFAULT_EPSILON,
                    DEFAULT_DELTA));
    assertThat(e).hasMessageThat().startsWith("rank must be");
  }

  @Test
  public void computeQuantile_rankNaN_throwsException() {
    Exception e =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                NOISE.computeQuantile(
                    /* rank */ NaN,
                    DEFAULT_MEAN,
                    DEFAULT_L_0_SENSITIVITY,
                    DEFAULT_L_INF_SENSITIVITY,
                    DEFAULT_EPSILON,
                    DEFAULT_DELTA));
    assertThat(e).hasMessageThat().startsWith("rank must be");
  }

  @Test
  public void computeQuantile_lInfSensitivityNan_throwsException() {
    Exception e =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                NOISE.computeQuantile(
                    DEFAULT_RANK,
                    DEFAULT_MEAN,
                    DEFAULT_L_0_SENSITIVITY,
                    /* lInfSensitvity */ Double.NaN,
                    /* epsilon */ 1,
                    DEFAULT_DELTA));
    assertThat(e).hasMessageThat().startsWith("lInfSensitivity must be");
  }

  @Test
  public void computeQuantile_lInfSensitivityNegative_throwsException() {
    Exception e =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                NOISE.computeQuantile(
                    DEFAULT_RANK,
                    DEFAULT_MEAN,
                    DEFAULT_L_0_SENSITIVITY,
                    /* lInfSensitivity */ -1.0,
                    DEFAULT_EPSILON,
                    DEFAULT_DELTA));
    assertThat(e).hasMessageThat().startsWith("lInfSensitivity must be");
  }

  @Test
  public void computeQuantile_lInfSensitivityZero_throwsException() {
    Exception e =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                NOISE.computeQuantile(
                    DEFAULT_RANK,
                    DEFAULT_MEAN,
                    DEFAULT_L_0_SENSITIVITY,
                    /* lInfSensitivity */ 0.0,
                    DEFAULT_EPSILON,
                    DEFAULT_DELTA));
    assertThat(e).hasMessageThat().startsWith("lInfSensitivity must be");
  }

  @Test
  public void computeQuantile_l0SensitivityNegative_throwsException() {
    Exception e =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                NOISE.computeQuantile(
                    DEFAULT_RANK,
                    DEFAULT_MEAN,
                    /* l0Sensitivity */ -1,
                    DEFAULT_L_INF_SENSITIVITY,
                    DEFAULT_EPSILON,
                    DEFAULT_DELTA));
    assertThat(e).hasMessageThat().startsWith("l0Sensitivity must be");
  }

  @Test
  public void computeQuantile_l0SensitivityZero_throwsException() {
    Exception e =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                NOISE.computeQuantile(
                    DEFAULT_RANK,
                    DEFAULT_MEAN,
                    /* lInfSensitivity */ 0,
                    DEFAULT_L_INF_SENSITIVITY,
                    DEFAULT_EPSILON,
                    DEFAULT_DELTA));
    assertThat(e).hasMessageThat().startsWith("l0Sensitivity must be");
  }
}

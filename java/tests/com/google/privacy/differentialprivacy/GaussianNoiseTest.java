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
import static java.lang.Double.NaN;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.math.Stats;
import java.security.SecureRandom;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class GaussianNoiseTest {
  private static final GaussianNoise NOISE = new GaussianNoise();
  private static final int NUM_SAMPLES = 100000;
  private static final double LN_3 = Math.log(3);
  private static final double DEFAULT_X = 0.0;
  private static final double DEFAULT_EPSILON = LN_3;
  private static final double DEFAULT_DELTA = 0.00001;
  private static final int DEFAULT_L_0_SENSITIVITY = 1;
  private static final double DEFAULT_L_INF_SENSITIVITY = 1.0;

  @Test
  public void addNoise_hasAccurateStatisticalProperties() {
    ImmutableList.Builder<Double> samples = ImmutableList.builder();
    for (int i = 0; i < NUM_SAMPLES; i++) {
      samples.add(
          NOISE.addNoise(
              DEFAULT_X,
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
    assertThat(stats.mean()).isWithin(sampleMeanTolerance).of(DEFAULT_X);
    assertThat(stats.populationVariance()).isWithin(sampleVarianceTolerance).of(variance);
  }

  @Test
  public void addNoise_differentMean_hasAccurateStatisticalProperties() {
    ImmutableList.Builder<Double> samples = ImmutableList.builder();
    for (int i = 0; i < NUM_SAMPLES; i++) {
      samples.add(
          NOISE.addNoise(
              /* x= */ 42.0,
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
              DEFAULT_X,
              DEFAULT_L_0_SENSITIVITY,
              /* lInfSensitivity= */ 0.5,
              DEFAULT_EPSILON,
              DEFAULT_DELTA));
    }
    Stats stats = Stats.of(samples.build());

    double variance = 2.9339943;
    // The tolerance is chosen according to the 99.9995% quantile of the anticipated distributions
    // of the sample mean and variance. Thus, the test falsely rejects with a probability of 10^-5.
    double sampleMeanTolerance = 4.41717 * Math.sqrt(variance / NUM_SAMPLES);
    double sampleVarianceTolerance = 4.41717 * variance * Math.sqrt(2.0 / NUM_SAMPLES);
    assertThat(stats.mean()).isWithin(sampleMeanTolerance).of(DEFAULT_X);
    assertThat(stats.populationVariance()).isWithin(sampleVarianceTolerance).of(variance);
  }

  @Test
  public void addNoise_differentEpsilon_hasAccurateStatisticalProperties() {
    ImmutableList.Builder<Double> samples = ImmutableList.builder();
    for (int i = 0; i < NUM_SAMPLES; i++) {
      samples.add(
          NOISE.addNoise(
              DEFAULT_X,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              /* epsilon= */ 2 * LN_3,
              DEFAULT_DELTA));
    }
    Stats stats = Stats.of(samples.build());

    double variance = 3.3634987;
    // The tolerance is chosen according to the 99.9995% quantile of the anticipated distributions
    // of the sample mean and variance. Thus, the test falsely rejects with a probability of 10^-5.
    double sampleMeanTolerance = 4.41717 * Math.sqrt(variance / NUM_SAMPLES);
    double sampleVarianceTolerance = 4.41717 * variance * Math.sqrt(2.0 / NUM_SAMPLES);
    assertThat(stats.mean()).isWithin(sampleMeanTolerance).of(DEFAULT_X);
    assertThat(stats.populationVariance()).isWithin(sampleVarianceTolerance).of(variance);
  }

  @Test
  public void addNoise_differentDelta_hasAccurateStatisticalProperties() {
    ImmutableList.Builder<Double> samples = ImmutableList.builder();
    for (int i = 0; i < NUM_SAMPLES; i++) {
      samples.add(
          NOISE.addNoise(
              DEFAULT_X,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              /* delta= */ 0.01));
    }
    Stats stats = Stats.of(samples.build());

    double variance = 3.0625;
    // The tolerance is chosen according to the 99.9995% quantile of the anticipated distributions
    // of the sample mean and variance. Thus, the test falsely rejects with a probability of 10^-5.
    double sampleMeanTolerance = 4.41717 * Math.sqrt(variance / NUM_SAMPLES);
    double sampleVarianceTolerance = 4.41717 * variance * Math.sqrt(2.0 / NUM_SAMPLES);
    assertThat(stats.mean()).isWithin(sampleMeanTolerance).of(DEFAULT_X);
    assertThat(stats.populationVariance()).isWithin(sampleVarianceTolerance).of(variance);
  }

  @Test
  public void addNoise_integralMean_hasAccurateStatisticalProperties() {
    ImmutableList.Builder<Long> samples = ImmutableList.builder();
    for (int i = 0; i < NUM_SAMPLES; i++) {
      samples.add(
          NOISE.addNoise(
              /* x= */ 0L,
              DEFAULT_L_0_SENSITIVITY,
              /* lInfSensitivity= */ 1L,
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
                DEFAULT_X,
                DEFAULT_L_0_SENSITIVITY,
                DEFAULT_L_INF_SENSITIVITY,
                /* epsilon= */ 1.0 / (1L << 51),
                DEFAULT_DELTA));
  }

  @Test
  public void addNoise_epsilonPosInfinity_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.addNoise(
                DEFAULT_X,
                DEFAULT_L_0_SENSITIVITY,
                DEFAULT_L_INF_SENSITIVITY,
                /* epsilon= */ Double.POSITIVE_INFINITY,
                DEFAULT_DELTA));
  }

  @Test
  public void addNoise_epsilonNan_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.addNoise(
                DEFAULT_X,
                DEFAULT_L_0_SENSITIVITY,
                DEFAULT_L_INF_SENSITIVITY,
                /* epsilon= */ Double.NaN,
                DEFAULT_DELTA));
  }

  @Test
  public void addNoise_deltaNegative_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.addNoise(
                DEFAULT_X,
                DEFAULT_L_0_SENSITIVITY,
                DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                /* delta= */ -1.0));
  }

  @Test
  public void addNoise_deltaOne_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.addNoise(
                DEFAULT_X,
                DEFAULT_L_0_SENSITIVITY,
                DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                /* delta= */ 1.0));
  }

  @Test
  public void addNoise_deltaGreaterThanOne_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.addNoise(
                DEFAULT_X,
                DEFAULT_L_0_SENSITIVITY,
                DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                /* delta= */ 2.0));
  }

  @Test
  public void addNoise_deltaNull_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.addNoise(
                DEFAULT_X,
                DEFAULT_L_0_SENSITIVITY,
                DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                /* delta= */ 0.0));
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
                DEFAULT_X,
                /* l0Sensitivity= */ -1,
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
                DEFAULT_X,
                /* l0Sensitivity= */ 0,
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
                DEFAULT_X,
                DEFAULT_L_0_SENSITIVITY,
                /* lInfSensitivity= */ -1.0,
                DEFAULT_EPSILON,
                DEFAULT_DELTA));
  }

  @Test
  public void addNoise_lInfSensitivityZero_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.addNoise(
                DEFAULT_X,
                DEFAULT_L_0_SENSITIVITY,
                /* lInfSensitivity= */ 0.0,
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
                    DEFAULT_X,
                    DEFAULT_L_0_SENSITIVITY,
                    /* lInfSensitivity= */ Double.MAX_VALUE,
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
  public void addNoiseDefinedByRho_Nan_rho_throwsException() {
    assertThrows(IllegalArgumentException.class, () -> NOISE.addNoiseDefinedByRho(0, 1, NaN));
  }

  @Test
  public void addNoiseDefinedByRho_zero_rho_throwsException() {
    assertThrows(IllegalArgumentException.class, () -> NOISE.addNoiseDefinedByRho(0, 1, 0));
  }

  @Test
  public void addNoiseDefinedByRho_negativeSensitivity_throwsException() {
    assertThrows(IllegalArgumentException.class, () -> NOISE.addNoiseDefinedByRho(0, -1, 1));
  }

  @Test
  public void addNoiseDefinedByRho_zeroSensitivity_throwsException() {
    assertThrows(IllegalArgumentException.class, () -> NOISE.addNoiseDefinedByRho(0, 0, 1));
  }

  @Test
  public void addNoise_returnsMultipleOfGranularity() {
    SecureRandom random = new SecureRandom();
    for (int i = 0; i < NUM_SAMPLES; i++) {
      // The rounding pricess should be independent of the value of x. Set x to a value between
      // -1*10^6 and 10^6 at random should covere a broad range of congruence classes.
      double x = random.nextDouble() * 2000000.0 - 1000000.0;

      // The following choice of epsilon, delta, l0 sensitivity and linf sensitivity should result
      // in a granularity of 2^-10
      double noisedX =
          NOISE.addNoise(
              x,
              /* l0Sensitivity= */ 1,
              /* lInfSensitivity= */ 1.0,
              /* epsilon= */ 1.0e-15,
              /* delta= */ 1.0e-14);
      assertThat(Math.floor(noisedX * 1024.0)).isEqualTo(noisedX * 1024.0);

      // The following choice of epsilon, delta, l0 sensitivity and linf sensitivity should result
      // in a granularity of 2^0
      noisedX =
          NOISE.addNoise(
              x,
              /* l0Sensitivity= */ 1,
              /* lInfSensitivity= */ 1024.0,
              /* epsilon= */ 1.0e-15,
              /* delta= */ 1.0e-14);
      assertThat(Math.floor(noisedX)).isEqualTo(noisedX);

      // The following choice of epsilon, delta, l0 sensitivity and linf sensitivity should result
      // in a granularity of 2^10
      noisedX =
          NOISE.addNoise(
              x,
              /* l0Sensitivity= */ 1,
              /* lInfSensitivity= */ 1048576.0,
              /* epsilon= */ 1.0e-15,
              /* delta= */ 1.0e-14);
      assertThat(Math.floor(noisedX / 1024.0)).isEqualTo(noisedX / 1024.0);
    }
  }

  @Test
  public void addNoise_integralX_returnsMultipleOfGranularity() {
    SecureRandom random = new SecureRandom();
    for (int i = 0; i < NUM_SAMPLES; i++) {
      // The rounding process should be independent of the value of x. Set x to a value between
      // -1*10^6 and 10^6 at random should cover a broad range of congruence classes.
      long x = (long) random.nextInt(2000000) - 1000000;

      // The following choice of epsilon, delta, l0 sensitivity and linf sensitivity should result
      // in a granularity of 2^1
      long noisedX =
          NOISE.addNoise(
              x,
              /* l0Sensitivity= */ 1,
              /* lInfSensitivity= */ 2048,
              /* epsilon= */ 1.0e-15,
              /* delta= */ 1.0e-14);
      assertThat(noisedX % 2).isEqualTo(0);

      // The following choice of epsilon, delta, l0 sensitivity and linf sensitivity should result
      // in a granularity of 2^10
      noisedX =
          NOISE.addNoise(
              x,
              /* l0Sensitivity= */ 1,
              /* lInfSensitivity= */ 1048576,
              /* epsilon= */ 1.0e-15,
              /* delta= */ 1.0e-14);
      assertThat(noisedX % 1024).isEqualTo(0);
    }
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
  public void getSigmaForRho_returnsCorrectly() {
    // sigma = l2Sensitivity / sqrt(2*rho)
    assertThat(GaussianNoise.getSigmaForRho(/* l2Sensitivity= */ 1.0, /* rho= */ 1))
        .isWithin(1e-12)
        .of(0.7071067811865475);
    assertThat(GaussianNoise.getSigmaForRho(/* l2Sensitivity= */ 3.0, /* rho= */ 1))
        .isWithin(1e-12)
        .of(2.1213203435596424);
    assertThat(GaussianNoise.getSigmaForRho(/* l2Sensitivity= */ 1.0, /* rho= */ 2))
        .isWithin(1e-12)
        .of(0.5);
    assertThat(GaussianNoise.getSigmaForRho(/* l2Sensitivity= */ 10.0, /* rho= */ 8))
        .isWithin(1e-12)
        .of(2.5);
  }

  @Test
  public void addNoiseDefinedByRho_hasAccurateStatisticalProperties() {
    ImmutableList.Builder<Double> samples = ImmutableList.builder();
    for (int i = 0; i < NUM_SAMPLES; i++) {
      samples.add(NOISE.addNoiseDefinedByRho(/* x */ 1.0, /* l2Sensitivity */ 10.0, /* rho */ 2));
    }
    Stats stats = Stats.of(samples.build());

    double variance = 25; // std_dev = 10/sqrt(2*2) = 5
    // The tolerance is chosen according to the 99.9995% quantile of the anticipated distributions
    // of the sample mean and variance. Thus, the test falsely rejects with a probability of 10^-5.
    final double stdNormQuantile = 4.41717; // 99.9995% quantile of the standard normal distribution
    double sampleMeanTolerance = stdNormQuantile * Math.sqrt(variance / NUM_SAMPLES);
    double sampleVarianceTolerance = stdNormQuantile * variance * Math.sqrt(2.0 / NUM_SAMPLES);
    assertThat(stats.mean()).isWithin(sampleMeanTolerance).of(1.0);
    assertThat(stats.populationVariance()).isWithin(sampleVarianceTolerance).of(variance);
  }
}

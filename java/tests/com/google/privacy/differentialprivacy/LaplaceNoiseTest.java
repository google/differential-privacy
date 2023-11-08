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
import static com.google.privacy.differentialprivacy.proto.SummaryOuterClass.MechanismType.LAPLACE;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.math.Stats;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import com.google.testing.junit.testparameterinjector.TestParameters;
import java.security.SecureRandom;
import org.junit.Test;
import org.junit.runner.RunWith;

@RunWith(TestParameterInjector.class)
public final class LaplaceNoiseTest {
  private static final LaplaceNoise NOISE = new LaplaceNoise();
  private static final int NUM_SAMPLES = 100000;
  private static final double LN_3 = Math.log(3);
  private static final double DEFAULT_X = 0.0;
  private static final double DEFAULT_EPSILON = LN_3;
  private static final int DEFAULT_L_0_SENSITIVITY = 1;
  private static final double DEFAULT_L_INF_SENSITIVITY = 1.0;

  /** Returns variance of Laplace noise for given parameters. */
  double getVariance(double epsilon, int l0sensitivty, double lInfSensitivty) {
    double l1Sensitivity = l0sensitivty * lInfSensitivty;
    return 2 * Math.pow(l1Sensitivity / epsilon, 2);
  }

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
              /* delta= */ 0.0));
    }
    Stats stats = Stats.of(samples.build());

    double variance =
        getVariance(DEFAULT_EPSILON, DEFAULT_L_0_SENSITIVITY, DEFAULT_L_INF_SENSITIVITY);
    // The tolerance is chosen according to the 99.9995% quantile of the anticipated distributions
    // of the sample mean and variance. Thus, the test falsely rejects with a probability of 10^-5.
    double sampleMeanTolerance = 4.41717 * Math.sqrt(variance / NUM_SAMPLES);
    double sampleVarianceTolerance = 4.41717 * Math.sqrt(5.0 * variance * variance / NUM_SAMPLES);
    assertThat(stats.mean()).isWithin(sampleMeanTolerance).of(DEFAULT_X);
    assertThat(stats.populationVariance()).isWithin(sampleVarianceTolerance).of(variance);
  }

  @Test
  public void addNoise_differentX_hasAccurateStatisticalProperties() {
    ImmutableList.Builder<Double> samples = ImmutableList.builder();
    for (int i = 0; i < NUM_SAMPLES; i++) {
      samples.add(
          NOISE.addNoise(
              /* x= */ 42.0,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              /* delta= */ 0.0));
    }
    Stats stats = Stats.of(samples.build());

    double mean = 42.0;
    double variance =
        getVariance(DEFAULT_EPSILON, DEFAULT_L_0_SENSITIVITY, DEFAULT_L_INF_SENSITIVITY);
    // The tolerance is chosen according to the 99.9995% quantile of the anticipated distributions
    // of the sample mean and variance. Thus, the test falsely rejects with a probability of 10^-5.
    double sampleMeanTolerance = 4.41717 * Math.sqrt(variance / NUM_SAMPLES);
    double sampleVarianceTolerance = 4.41717 * Math.sqrt(5.0 * variance * variance / NUM_SAMPLES);
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
              /* delta= */ 0.0));
    }
    Stats stats = Stats.of(samples.build());

    double variance = getVariance(DEFAULT_EPSILON, DEFAULT_L_0_SENSITIVITY, 0.5);
    // The tolerance is chosen according to the 99.9995% quantile of the anticipated distributions
    // of the sample mean and variance. Thus, the test falsely rejects with a probability of 10^-5.
    double sampleMeanTolerance = 4.41717 * Math.sqrt(variance / NUM_SAMPLES);
    double sampleVarianceTolerance = 4.41717 * Math.sqrt(5.0 * variance * variance / NUM_SAMPLES);
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
              /* delta= */ 0.0));
    }
    Stats stats = Stats.of(samples.build());

    double variance = getVariance(2 * LN_3, DEFAULT_L_0_SENSITIVITY, DEFAULT_L_INF_SENSITIVITY);
    // The tolerance is chosen according to the 99.9995% quantile of the anticipated distributions
    // of the sample mean and variance. Thus, the test falsely rejects with a probability of 10^-5.
    double sampleMeanTolerance = 4.41717 * Math.sqrt(variance / NUM_SAMPLES);
    double sampleVarianceTolerance = 4.41717 * Math.sqrt(5.0 * variance * variance / NUM_SAMPLES);
    assertThat(stats.mean()).isWithin(sampleMeanTolerance).of(DEFAULT_X);
    assertThat(stats.populationVariance()).isWithin(sampleVarianceTolerance).of(variance);
  }

  @Test
  public void addNoise_integralX_hasAccurateStatisticalProperties() {
    ImmutableList.Builder<Long> samples = ImmutableList.builder();
    for (int i = 0; i < NUM_SAMPLES; i++) {
      samples.add(
          NOISE.addNoise(
              /* x= */ 0L,
              DEFAULT_L_0_SENSITIVITY,
              /* lInfSensitivity= */ 1L,
              DEFAULT_EPSILON,
              /* delta= */ 0.0));
    }
    Stats stats = Stats.of(samples.build());

    double mean = 0.0;
    double approxVariance = getVariance(DEFAULT_EPSILON, DEFAULT_L_0_SENSITIVITY, 1);
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
                /* delta= */ 0.0));
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
                /* delta= */ 0.0));
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
                /* delta= */ 0.0));
  }

  @Test
  public void addNoise_deltaNonzero_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.addNoise(
                DEFAULT_X,
                DEFAULT_L_0_SENSITIVITY,
                DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                /* delta= */ 0.1));
  }

  @Test
  public void addNoise_lInfSensitivityTooHigh_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.addNoise(
                DEFAULT_X,
                DEFAULT_L_0_SENSITIVITY,
                /* lInfSensitivity= */ Double.MAX_VALUE,
                DEFAULT_EPSILON,
                /* delta= */ 0.0));
  }

  @Test
  public void addNoise_lInfSensitivityNan_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.addNoise(
                DEFAULT_X,
                DEFAULT_L_0_SENSITIVITY,
                /* lInfSensitivity= */ Double.NaN,
                /* epsilon= */ 1.0,
                /* delta= */ 0.0));
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
                /* delta= */ 0.0));
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
                /* delta= */ 0.0));
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
                /* delta= */ 0.0));
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
                /* delta= */ 0.0));
  }

  @Test
  public void addNoise_l1SensitivityNegative_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.addNoise(
                DEFAULT_X, /* l1Sensitivity= */ -1.0, DEFAULT_EPSILON, /* delta= */ 0.0));
  }

  @Test
  public void addNoise_l1SensitivityZero_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.addNoise(
                DEFAULT_X, /* l1Sensitivity= */ 0.0, DEFAULT_EPSILON, /* delta= */ 0.0));
  }

  @Test
  public void addNoise_l1SensitivityNan_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.addNoise(
                DEFAULT_X, /* l1Sensitivity= */ Double.NaN, DEFAULT_EPSILON, /* delta= */ 0.0));
  }

  @Test
  public void addNoise_returnsMultipleOfGranularity() {
    SecureRandom random = new SecureRandom();
    for (int i = 0; i < NUM_SAMPLES; i++) {
      // The rounding pricess should be independent of the value of x. Set x to a value between
      // -1*10^6 and 10^6 at random should covere a broad range of congruence classes.
      double x = random.nextDouble() * 2000000.0 - 1000000.0;

      // The following choice of epsilon, l0 sensitivity and linf sensitivity should result in a
      // granularity of 2^-10
      double noisedX =
          NOISE.addNoise(
              x,
              /* l0Sensitivity= */ 1,
              /* lInfSensitivity= */ 1.0,
              /* epsilon= */ 4.7e-10,
              /* delta= */ 0.0);
      assertThat(Math.floor(noisedX * 1024.0)).isEqualTo(noisedX * 1024.0);

      // The following choice of epsilon, l0 sensitivity and linf sensitivity should result in a
      // granularity of 2^0
      noisedX =
          NOISE.addNoise(
              x,
              /* l0Sensitivity= */ 1,
              /* lInfSensitivity= */ 1.0,
              /* epsilon= */ 9.1e-13,
              /* delta= */ 0.0);
      assertThat(Math.floor(noisedX)).isEqualTo(noisedX);

      // The following choice of epsilon, l0 sensitivity and linf sensitivity should result in a
      // granularity of 2^10
      noisedX =
          NOISE.addNoise(
              x,
              /* l0Sensitivity= */ 1,
              /* lInfSensitivity= */ 1.0,
              /* epsilon= */ 8.9e-16,
              /* delta= */ 0.0);
      assertThat(Math.floor(noisedX / 1024.0)).isEqualTo(noisedX / 1024.0);
    }
  }

  @Test
  public void addNoise_integralX_returnsMultipleOfGranularity() {
    SecureRandom random = new SecureRandom();
    for (int i = 0; i < NUM_SAMPLES; i++) {
      // The rounding pricess should be independent of the value of x. Set x to a value between
      // -1*10^6 and 10^6 at random should covere a broad range of congruence classes.
      long x = (long) random.nextInt(2000000) - 1000000;

      // The following choice of epsilon, l0 sensitivity and linf sensitivity should result in a
      // granularity of 2^1
      long noisedX =
          NOISE.addNoise(
              x,
              /* l0Sensitivity= */ 1,
              /* lInfSensitivity= */ 1,
              /* epsilon= */ 4.6e-13,
              /* delta= */ 0.0);
      assertThat(noisedX % 2).isEqualTo(0);

      // The following choice of epsilon, l0 sensitivity and linf sensitivity should result in a
      // granularity of 2^10
      noisedX =
          NOISE.addNoise(
              x,
              /* l0Sensitivity= */ 1,
              /* lInfSensitivity= */ 1,
              /* epsilon= */ 8.9e-16,
              /* delta= */ 0.0);
      assertThat(noisedX % 1024).isEqualTo(0);
    }
  }

  @Test
  public void getMechanismType_returnsLaplace() {
    assertThat(NOISE.getMechanismType()).isEqualTo(LAPLACE);
  }

  @Test
  // quantile < mean
  @TestParameters("{quantile: 0.6, mean: 1.0, l1Sensitivity: 4.9, epsilon: 2.7}")
  // quantile > mean
  @TestParameters("{quantile: 4.2, mean: 1.8, l1Sensitivity: 3.6, epsilon: 8.0}")
  public void computeQuantile_inverseToCumulativeDensity(
      double quantile, double mean, double l1Sensitivity, double epsilon) {
    double actualQuantile =
        LaplaceNoise.computeQuantile(
            LaplaceNoise.cumulativeDensity(quantile, mean, l1Sensitivity, epsilon),
            mean,
            l1Sensitivity,
            epsilon);

    assertThat(actualQuantile).isWithin(1e-10).of(quantile);
  }

  @Test
  // rank < 0.5
  @TestParameters("{rank: 0.2, mean: 7.7, l1Sensitivity: 1.3, epsilon: 0.8}")
  // rank > 0.5
  @TestParameters("{rank: 0.9, mean: 9.2, l1Sensitivity: 6.6, epsilon: 2.7}")
  public void cumulativeDensity_inverseToQuantile(
      double rank, double mean, double l1Sensitivity, double epsilon) {
    double actualRank =
        LaplaceNoise.cumulativeDensity(
            LaplaceNoise.computeQuantile(rank, mean, l1Sensitivity, epsilon),
            mean,
            l1Sensitivity,
            epsilon);

    assertThat(actualRank).isWithin(1e-10).of(rank);
  }

  @Test
  // z < mean
  @TestParameters("{z: -0.1, mean: 0.6, l1Sensitivity: 0.8, epsilon: 1.1, expectedP: 0.1909684}")
  // z == mean
  @TestParameters("{z: 4.2, mean: 4.2, l1Sensitivity: 6.1, epsilon: 1.5, expectedP: 0.5}")
  // z > mean
  @TestParameters("{z: 2.1, mean: 0.3, l1Sensitivity: 7.7, epsilon: 2.7, expectedP: 0.7340152}")
  public void cumulativeDensity_sampleValues_computesCorrectly(
      double z, double mean, double l1Sensitivity, double epsilon, double expectedP) {
    double actualP = LaplaceNoise.cumulativeDensity(z, mean, l1Sensitivity, epsilon);

    assertThat(actualP).isWithin(1e-6).of(expectedP);
  }
}

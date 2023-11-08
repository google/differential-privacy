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
import static com.google.privacy.differentialprivacy.proto.SummaryOuterClass.MechanismType.DISCRETE_LAPLACE;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.math.Stats;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import org.junit.Test;
import org.junit.runner.RunWith;

@RunWith(TestParameterInjector.class)
public final class DiscreteLaplaceNoiseTest {

  private static final DiscreteLaplaceNoise NOISE = new DiscreteLaplaceNoise();
  private static final int NUM_SAMPLES = 100000;
  private static final double LN_3 = Math.log(3);
  private static final long DEFAULT_X = 0;
  private static final double DEFAULT_EPSILON = LN_3;
  private static final int DEFAULT_L_0_SENSITIVITY = 1;
  private static final long DEFAULT_L_INF_SENSITIVITY = 1;

  /** Returns variance of Discrete Laplace noise for given parameters. */
  double getVariance(double epsilon, int l0sensitivty, double lInfSensitivty) {
    double l1Sensitivity = l0sensitivty * lInfSensitivty;
    double pGeometric = 1 - Math.exp(-epsilon / l1Sensitivity);
    return 2 * (1 - pGeometric) / Math.pow(pGeometric, 2);
  }

  @Test
  public void addNoise_hasAccurateStatisticalProperties() {
    ImmutableList.Builder<Long> samples = ImmutableList.builder();

    for (int i = 0; i < NUM_SAMPLES; i++) {
      samples.add(
          NOISE.addNoise(
              DEFAULT_X,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              /* delta= */ 0.0));
    }

    double variance =
        getVariance(DEFAULT_EPSILON, DEFAULT_L_0_SENSITIVITY, DEFAULT_L_INF_SENSITIVITY);
    // The tolerance is chosen according to the 99.9995% quantile of the anticipated distributions
    // of the sample mean and variance. Thus, the test falsely rejects with a probability of 10^-5.
    double sampleMeanTolerance = 4.41717 * Math.sqrt(variance / NUM_SAMPLES);
    double sampleVarianceTolerance = 4.41717 * Math.sqrt(5.0 * variance * variance / NUM_SAMPLES);
    Stats stats = Stats.of(samples.build());
    assertThat(stats.mean()).isWithin(sampleMeanTolerance).of(DEFAULT_X);
    assertThat(stats.populationVariance()).isWithin(sampleVarianceTolerance).of(variance);
  }

  @Test
  public void addNoise_differentX_hasAccurateStatisticalProperties() {
    ImmutableList.Builder<Long> samples = ImmutableList.builder();

    for (int i = 0; i < NUM_SAMPLES; i++) {
      samples.add(
          NOISE.addNoise(
              /* x= */ 42,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              /* delta= */ 0.0));
    }

    double mean = 42;
    double variance =
        getVariance(DEFAULT_EPSILON, DEFAULT_L_0_SENSITIVITY, DEFAULT_L_INF_SENSITIVITY);
    // The tolerance is chosen according to the 99.9995% quantile of the anticipated distributions
    // of the sample mean and variance. Thus, the test falsely rejects with a probability of 10^-5.
    double sampleMeanTolerance = 4.41717 * Math.sqrt(variance / NUM_SAMPLES);
    double sampleVarianceTolerance = 4.41717 * Math.sqrt(5.0 * variance * variance / NUM_SAMPLES);
    Stats stats = Stats.of(samples.build());
    assertThat(stats.mean()).isWithin(sampleMeanTolerance).of(mean);
    assertThat(stats.populationVariance()).isWithin(sampleVarianceTolerance).of(variance);
  }

  @Test
  public void addNoise_differentSensitivity_hasAccurateStatisticalProperties() {
    ImmutableList.Builder<Long> samples = ImmutableList.builder();

    for (int i = 0; i < NUM_SAMPLES; i++) {
      samples.add(
          NOISE.addNoise(
              DEFAULT_X,
              DEFAULT_L_0_SENSITIVITY,
              /* lInfSensitivity= */ 2,
              DEFAULT_EPSILON,
              /* delta= */ 0.0));
    }

    double variance = getVariance(DEFAULT_EPSILON, DEFAULT_L_0_SENSITIVITY, 2);
    // The tolerance is chosen according to the 99.9995% quantile of the anticipated distributions
    // of the sample mean and variance. Thus, the test falsely rejects with a probability of 10^-5.
    double sampleMeanTolerance = 4.41717 * Math.sqrt(variance / NUM_SAMPLES);
    double sampleVarianceTolerance = 4.41717 * Math.sqrt(5.0 * variance * variance / NUM_SAMPLES);
    Stats stats = Stats.of(samples.build());
    assertThat(stats.mean()).isWithin(sampleMeanTolerance).of(DEFAULT_X);
    assertThat(stats.populationVariance()).isWithin(sampleVarianceTolerance).of(variance);
  }

  @Test
  public void addNoise_differentEpsilon_hasAccurateStatisticalProperties() {
    ImmutableList.Builder<Long> samples = ImmutableList.builder();

    for (int i = 0; i < NUM_SAMPLES; i++) {
      samples.add(
          NOISE.addNoise(
              DEFAULT_X,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              /* epsilon= */ 0.5 * LN_3,
              /* delta= */ 0.0));
    }

    double variance = getVariance(0.5 * LN_3, DEFAULT_L_0_SENSITIVITY, DEFAULT_L_INF_SENSITIVITY);
    // The tolerance is chosen according to the 99.9995% quantile of the anticipated distributions
    // of the sample mean and variance. Thus, the test falsely rejects with a probability of 10^-5.
    double sampleMeanTolerance = 4.41717 * Math.sqrt(variance / NUM_SAMPLES);
    double sampleVarianceTolerance = 4.41717 * Math.sqrt(5.0 * variance * variance / NUM_SAMPLES);
    Stats stats = Stats.of(samples.build());
    assertThat(stats.mean()).isWithin(sampleMeanTolerance).of(DEFAULT_X);
    assertThat(stats.populationVariance()).isWithin(sampleVarianceTolerance).of(variance);
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
            NOISE.addNoise(DEFAULT_X, /* l1Sensitivity= */ -1, DEFAULT_EPSILON, /* delta= */ 0.0));
  }

  @Test
  public void addNoise_l1SensitivityZero_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.addNoise(DEFAULT_X, /* l1Sensitivity= */ 0, DEFAULT_EPSILON, /* delta= */ 0.0));
  }

  @Test
  public void getMechanismType_returnsDiscreteLaplace() {
    assertThat(NOISE.getMechanismType()).isEqualTo(DISCRETE_LAPLACE);
  }
}

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

package com.google.privacy.differentialprivacy.testing;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.math.Stats;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Collection of tests verifying basic properties of the statistical tests used for assesing the
 * Building Blocks library. Note that the statistical tests are only evaluated for deterministic
 * inputs. Evaluating their statistical properties is out of scope for this unit test.
 */
@RunWith(JUnit4.class)
public class ReferenceNoiseUtilTest {
  private static final double DEFAULT_MEAN = 0.0;
  private static final double DEFAULT_RAW_INPUT = 0.0;
  private static final double DEFAULT_VARIANCE = 2.0;
  private static final double DEFAULT_EPSILON = Math.log(3);
  private static final double DEFAULT_DELTA = 0.00001;
  private static final int DEFAULT_L_0_SENSITIVITY = 1;
  private static final double DEFAULT_L_1_SENSITIVITY = 1.0;
  private static final double DEFAULT_L_2_SENSITIVITY = 1.0;
  private static final double DEFAULT_L_INF_SENSITIVITY = 1.0;
  private static final int NUM_SAMPLES = 1000000;

  @Test
  public void sampleLaplace_hasAccurateStatisticalProperties() {
    ImmutableList.Builder<Double> samples = new ImmutableList.Builder<>();
    for (int i = 0; i < NUM_SAMPLES; i++) {
      samples.add(ReferenceNoiseUtil.sampleLaplace(DEFAULT_MEAN, DEFAULT_VARIANCE));
    }
    Stats stats = Stats.of(samples.build());
    assertThat(stats.mean()).isWithin(0.1).of(DEFAULT_MEAN);
    assertThat(stats.populationVariance()).isWithin(0.5).of(DEFAULT_VARIANCE);
  }

  @Test
  public void sampleLaplace_varianceLessOrEqualToZero_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () -> ReferenceNoiseUtil.sampleLaplace(DEFAULT_MEAN, /* variance= */ 0.0));
    assertThrows(
        IllegalArgumentException.class,
        () -> ReferenceNoiseUtil.sampleLaplace(DEFAULT_MEAN, -DEFAULT_MEAN));
  }

  @Test
  public void sampleLaplace_dpInputParameters_hasAccurateStatisticalProperties() {
    ImmutableList.Builder<Double> samples = new ImmutableList.Builder<>();
    for (int i = 0; i < NUM_SAMPLES; i++) {
      // The sensitivity is chosen so that the variance of the sample matches the default variance.
      samples.add(
          ReferenceNoiseUtil.sampleLaplace(
              DEFAULT_RAW_INPUT, DEFAULT_EPSILON, /* l1Sensitivity= */ Math.log(3)));
    }
    Stats stats = Stats.of(samples.build());
    assertThat(stats.mean()).isWithin(0.1).of(DEFAULT_MEAN);
    assertThat(stats.populationVariance()).isWithin(0.5).of(DEFAULT_VARIANCE);
  }

  @Test
  public void sampleLaplace_epsilonLessOrEqualToZero_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            ReferenceNoiseUtil.sampleLaplace(
                DEFAULT_RAW_INPUT, /* epsilon= */ 0.0, DEFAULT_L_1_SENSITIVITY));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            ReferenceNoiseUtil.sampleLaplace(
                DEFAULT_RAW_INPUT, -DEFAULT_EPSILON, DEFAULT_L_1_SENSITIVITY));
  }

  @Test
  public void sampleLaplace_l1SensitivityLessOrEqualToZero_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            ReferenceNoiseUtil.sampleLaplace(
                DEFAULT_RAW_INPUT, DEFAULT_EPSILON, /* l1Sensitivity= */ 0.0));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            ReferenceNoiseUtil.sampleLaplace(
                DEFAULT_RAW_INPUT, DEFAULT_EPSILON, -DEFAULT_L_1_SENSITIVITY));
  }

  @Test
  public void sampleGaussian_hasAccurateStatisticalProperties() {
    ImmutableList.Builder<Double> samples = new ImmutableList.Builder<>();
    for (int i = 0; i < NUM_SAMPLES; i++) {
      samples.add(ReferenceNoiseUtil.sampleGaussian(DEFAULT_MEAN, DEFAULT_VARIANCE));
    }
    Stats stats = Stats.of(samples.build());
    assertThat(stats.mean()).isWithin(0.1).of(DEFAULT_MEAN);
    assertThat(stats.populationVariance()).isWithin(0.5).of(DEFAULT_VARIANCE);
  }

  @Test
  public void sampleGaussian_varianceLessOrEqualToZero_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () -> ReferenceNoiseUtil.sampleGaussian(DEFAULT_MEAN, /* variance= */ 0.0));
    assertThrows(
        IllegalArgumentException.class,
        () -> ReferenceNoiseUtil.sampleGaussian(DEFAULT_MEAN, -DEFAULT_VARIANCE));
  }

  @Test
  public void sampleGaussian_dpInputParameters_hasAccurateStatisticalProperties() {
    ImmutableList.Builder<Double> samples = new ImmutableList.Builder<>();
    for (int i = 0; i < NUM_SAMPLES; i++) {
      // The sensitivity is chosen so that the variance of the sample matches the default variance.
      samples.add(
          ReferenceNoiseUtil.sampleGaussian(
              DEFAULT_RAW_INPUT, DEFAULT_EPSILON, DEFAULT_DELTA, /* l2Sensitivity= */ 0.41281));
    }
    Stats stats = Stats.of(samples.build());
    assertThat(stats.mean()).isWithin(0.1).of(DEFAULT_MEAN);
    assertThat(stats.populationVariance()).isWithin(0.5).of(DEFAULT_VARIANCE);
  }

  @Test
  public void sampleGaussian_epsilonLessOrEqualToZero_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            ReferenceNoiseUtil.sampleGaussian(
                DEFAULT_RAW_INPUT, /* epsilon= */ 0.0, DEFAULT_DELTA, DEFAULT_L_2_SENSITIVITY));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            ReferenceNoiseUtil.sampleGaussian(
                DEFAULT_RAW_INPUT, -DEFAULT_EPSILON, DEFAULT_DELTA, DEFAULT_L_2_SENSITIVITY));
  }

  @Test
  public void sampleGaussian_deltaLessOrEqualToZero_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            ReferenceNoiseUtil.sampleGaussian(
                DEFAULT_RAW_INPUT, DEFAULT_EPSILON, /* delta= */ 0.0, DEFAULT_L_2_SENSITIVITY));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            ReferenceNoiseUtil.sampleGaussian(
                DEFAULT_RAW_INPUT, DEFAULT_EPSILON, -DEFAULT_DELTA, DEFAULT_L_2_SENSITIVITY));
  }

  @Test
  public void sampleGaussian_l2SensitivityLessOrEqualToZero_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            ReferenceNoiseUtil.sampleGaussian(
                DEFAULT_RAW_INPUT, DEFAULT_EPSILON, DEFAULT_DELTA, /* l2Sensitivity= */ 0.0));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            ReferenceNoiseUtil.sampleGaussian(
                DEFAULT_RAW_INPUT, DEFAULT_EPSILON, DEFAULT_DELTA, -DEFAULT_L_2_SENSITIVITY));
  }

  @Test
  public void getL1Sensitivity_l0SensitivityLessOrEqualToZero_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            ReferenceNoiseUtil.getL1Sensitivity(/* l0Sensitivity= */ 0, DEFAULT_L_INF_SENSITIVITY));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            ReferenceNoiseUtil.getL1Sensitivity(
                -DEFAULT_L_0_SENSITIVITY, DEFAULT_L_INF_SENSITIVITY));
  }

  @Test
  public void getL1Sensitivity_lInfSensitivityLessOrEqualToZero_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            ReferenceNoiseUtil.getL1Sensitivity(
                DEFAULT_L_0_SENSITIVITY, /* lInfSensitivity= */ 0.0));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            ReferenceNoiseUtil.getL1Sensitivity(
                DEFAULT_L_0_SENSITIVITY, -DEFAULT_L_INF_SENSITIVITY));
  }

  @Test
  public void getL2Sensitivity_l0SensitivityLessOrEqualToZero_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            ReferenceNoiseUtil.getL2Sensitivity(/* l0Sensitivity= */ 0, DEFAULT_L_INF_SENSITIVITY));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            ReferenceNoiseUtil.getL2Sensitivity(
                -DEFAULT_L_0_SENSITIVITY, DEFAULT_L_INF_SENSITIVITY));
  }

  @Test
  public void getL2Sensitivity_lInfSensitivityLessOrEqualToZero_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            ReferenceNoiseUtil.getL2Sensitivity(
                DEFAULT_L_0_SENSITIVITY, /* lInfSensitivity= */ 0.0));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            ReferenceNoiseUtil.getL2Sensitivity(
                DEFAULT_L_0_SENSITIVITY, -DEFAULT_L_INF_SENSITIVITY));
  }

  @Test
  public void getLaplaceVariance_epsilonLessOrEqualToZero_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () -> ReferenceNoiseUtil.getLaplaceVariance(/* epsilon= */ 0.0, DEFAULT_L_1_SENSITIVITY));
    assertThrows(
        IllegalArgumentException.class,
        () -> ReferenceNoiseUtil.getLaplaceVariance(-DEFAULT_EPSILON, DEFAULT_L_1_SENSITIVITY));
  }

  @Test
  public void getLaplaceVariance_l1SensitivityLessOrEqualToZero_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () -> ReferenceNoiseUtil.getLaplaceVariance(DEFAULT_EPSILON, /* l1Sensitivity= */ 0.0));
    assertThrows(
        IllegalArgumentException.class,
        () -> ReferenceNoiseUtil.getLaplaceVariance(DEFAULT_EPSILON, -DEFAULT_L_1_SENSITIVITY));
  }

  @Test
  public void getGaussianVariance_epsilonLessOrEqualToZero_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            ReferenceNoiseUtil.getGaussianVariance(
                /* epsilon= */ 0.0, DEFAULT_DELTA, DEFAULT_L_2_SENSITIVITY));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            ReferenceNoiseUtil.getGaussianVariance(
                -DEFAULT_EPSILON, DEFAULT_DELTA, DEFAULT_L_2_SENSITIVITY));
  }

  @Test
  public void getGaussianVariance_deltaLessOrEqualToZero_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            ReferenceNoiseUtil.getGaussianVariance(
                DEFAULT_EPSILON, /* delta= */ 0.0, DEFAULT_L_2_SENSITIVITY));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            ReferenceNoiseUtil.getGaussianVariance(
                DEFAULT_EPSILON, -DEFAULT_DELTA, DEFAULT_L_2_SENSITIVITY));
  }

  @Test
  public void getGaussianVariance_l2SensitivityLessOrEqualToZero_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            ReferenceNoiseUtil.getGaussianVariance(
                DEFAULT_EPSILON, DEFAULT_DELTA, /* l2Sensitivity= */ 0.0));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            ReferenceNoiseUtil.getGaussianVariance(
                DEFAULT_EPSILON, DEFAULT_DELTA, -DEFAULT_L_2_SENSITIVITY));
  }
}

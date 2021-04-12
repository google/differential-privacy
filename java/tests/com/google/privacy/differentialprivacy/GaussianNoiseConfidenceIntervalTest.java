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

import static com.google.common.truth.Truth.assertWithMessage;
import static java.lang.Math.max;
import static org.junit.Assert.assertThrows;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class GaussianNoiseConfidenceIntervalTest {
  private static final Noise NOISE = new GaussianNoise();
  private static final double TOLERANCE = 1E-7;
  private static final double DEFAULT_EPSILON = Math.log(3);
  private static final int DEFAULT_L_0_SENSITIVITY = 1;
  private static final double DEFAULT_L_INF_SENSITIVITY = 1.0;
  private static final double DEFAULT_DELTA = 0.00001;
  private static final double DEFAULT_NOISED_X = 0.0;
  private static final double DEFAULT_ALPHA = 0.1;

  @Test
  public void computeConfidenceInterval_forDouble_arbitraryTest() {
    ConfidenceInterval actual =
        NOISE.computeConfidenceInterval(
            /* noisedX= */ 83.0,
            /* l0Sensitivity= */ 3,
            /* lInfSensitivity= */ 2.0,
            DEFAULT_EPSILON,
            DEFAULT_DELTA,
            /* alpha= */ 0.24);
    ConfidenceInterval expected = ConfidenceInterval.create(69.056133, 96.943867);
    verifyApproxEqual(actual, expected);
  }

  @Test
  public void computeConfidenceInterval_forDouble_defaultParameters() {
    ConfidenceInterval actual =
        NOISE.computeConfidenceInterval(
            DEFAULT_NOISED_X,
            DEFAULT_L_0_SENSITIVITY,
            DEFAULT_L_INF_SENSITIVITY,
            DEFAULT_EPSILON,
            DEFAULT_DELTA,
            DEFAULT_ALPHA);
    ConfidenceInterval expected = ConfidenceInterval.create(-5.6349087, 5.6349087);
    verifyApproxEqual(actual, expected);
  }

  @Test
  public void computeConfidenceInterval_forDouble_positiveNoisedX() {
    ConfidenceInterval actual =
        NOISE.computeConfidenceInterval(
            /* noisedX= */ 1.5865547456,
            DEFAULT_L_0_SENSITIVITY,
            DEFAULT_L_INF_SENSITIVITY,
            DEFAULT_EPSILON,
            DEFAULT_DELTA,
            DEFAULT_ALPHA);
    ConfidenceInterval expected = ConfidenceInterval.create(-4.0483540, 7.2214635);
    verifyApproxEqual(actual, expected);
  }

  @Test
  public void computeConfidenceInterval_forDouble_negativeNoisedX() {
    ConfidenceInterval actual =
        NOISE.computeConfidenceInterval(
            /* noisedX= */ -1.5865547456,
            DEFAULT_L_0_SENSITIVITY,
            DEFAULT_L_INF_SENSITIVITY,
            DEFAULT_EPSILON,
            DEFAULT_DELTA,
            DEFAULT_ALPHA);
    ConfidenceInterval expected = ConfidenceInterval.create(-7.2214635, 4.0483540);
    verifyApproxEqual(actual, expected);
  }

  @Test
  public void computeConfidenceInterval_forDouble_largePositiveNoisedX() {
    ConfidenceInterval actual =
        NOISE.computeConfidenceInterval(
            /* noisedX= */ 958655.4745,
            DEFAULT_L_0_SENSITIVITY,
            DEFAULT_L_INF_SENSITIVITY,
            DEFAULT_EPSILON,
            DEFAULT_DELTA,
            DEFAULT_ALPHA);
    ConfidenceInterval expected = ConfidenceInterval.create(958649.84, 958661.11);
    verifyApproxEqual(actual, expected);
  }

  @Test
  public void computeConfidenceInterval_forDouble_largeNegativeNoisedX() {
    ConfidenceInterval actual =
        NOISE.computeConfidenceInterval(
            /* noisedX= */ -958655.4745,
            DEFAULT_L_0_SENSITIVITY,
            DEFAULT_L_INF_SENSITIVITY,
            DEFAULT_EPSILON,
            DEFAULT_DELTA,
            DEFAULT_ALPHA);
    ConfidenceInterval expected = ConfidenceInterval.create(-958661.11, -958649.84);
    verifyApproxEqual(actual, expected);
  }

  @Test
  public void computeConfidenceInterval_forDouble_smallAlpha() {
    ConfidenceInterval actual =
        NOISE.computeConfidenceInterval(
            DEFAULT_NOISED_X,
            DEFAULT_L_0_SENSITIVITY,
            DEFAULT_L_INF_SENSITIVITY,
            DEFAULT_EPSILON,
            DEFAULT_DELTA,
            /* alpha= */ 7.856382354E-10);
    ConfidenceInterval expected = ConfidenceInterval.create(-21.061026, 21.061026);
    verifyApproxEqual(actual, expected);
  }

  @Test
  public void computeConfidenceInterval_forDouble_largeAlpha() {
    ConfidenceInterval actual =
        NOISE.computeConfidenceInterval(
            DEFAULT_NOISED_X,
            DEFAULT_L_0_SENSITIVITY,
            DEFAULT_L_INF_SENSITIVITY,
            DEFAULT_EPSILON,
            DEFAULT_DELTA,
            /* alpha= */ 1 - 7.856382354E-10);

    ConfidenceInterval expected = ConfidenceInterval.create(-3.3732006E-9, 3.3732006E-9);
    verifyApproxEqual(actual, expected);
  }

  @Test
  public void computeConfidenceInterval_forDouble_smallDelta() {
    ConfidenceInterval actual =
        NOISE.computeConfidenceInterval(
            DEFAULT_NOISED_X,
            DEFAULT_L_0_SENSITIVITY,
            DEFAULT_L_INF_SENSITIVITY,
            DEFAULT_EPSILON,
            /* delta= */ 1.78468549878E-10,
            DEFAULT_ALPHA);
    ConfidenceInterval expected = ConfidenceInterval.create(-8.6868832, 8.6868832);
    verifyApproxEqual(actual, expected);
  }

  @Test
  public void computeConfidenceInterval_forDouble_largeDelta() {
    ConfidenceInterval actual =
        NOISE.computeConfidenceInterval(
            DEFAULT_NOISED_X,
            DEFAULT_L_0_SENSITIVITY,
            DEFAULT_L_INF_SENSITIVITY,
            DEFAULT_EPSILON,
            /* delta= */ 1 - 1.78468549878E-10,
            DEFAULT_ALPHA);
    ConfidenceInterval expected = ConfidenceInterval.create(-0.12729946, 0.12729946);
    verifyApproxEqual(actual, expected);
  }

  @Test
  public void computeConfidenceInterval_forDouble_smallEpsilon() {
    ConfidenceInterval actual =
        NOISE.computeConfidenceInterval(
            DEFAULT_NOISED_X,
            DEFAULT_L_0_SENSITIVITY,
            DEFAULT_L_INF_SENSITIVITY,
            /* epsilon= */ 1.65463453425E-10,
            DEFAULT_DELTA,
            DEFAULT_ALPHA);
    ConfidenceInterval expected = ConfidenceInterval.create(-65636.239, 65636.239);
    verifyApproxEqual(actual, expected);
  }

  @Test
  public void computeConfidenceInterval_forDouble_largeEpsilon() {
    ConfidenceInterval actual =
        NOISE.computeConfidenceInterval(
            DEFAULT_NOISED_X,
            DEFAULT_L_0_SENSITIVITY,
            DEFAULT_L_INF_SENSITIVITY,
            /* epsilon= */ 1.65463453425E10,
            DEFAULT_DELTA,
            DEFAULT_ALPHA);
    ConfidenceInterval expected = ConfidenceInterval.create(0.0, 0.0);
    verifyApproxEqual(actual, expected);
  }

  @Test
  public void computeConfidenceInterval_forDouble_zeroL0Sensitivity_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                DEFAULT_NOISED_X,
                /* l0Sensitivity= */ 0,
                DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                DEFAULT_ALPHA));
  }

  @Test
  public void computeConfidenceInterval_forDouble_negativeL0Sensitivity_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                DEFAULT_NOISED_X,
                /* l0Sensitivity= */ -1,
                DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                DEFAULT_ALPHA));
  }

  @Test
  public void computeConfidenceInterval_forDouble_zeroLInfSensitivity_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                DEFAULT_NOISED_X,
                DEFAULT_L_0_SENSITIVITY,
                /* lInfSensitivity= */ 0,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                DEFAULT_ALPHA));
  }

  @Test
  public void computeConfidenceInterval_forDouble_negativeLInfSensitivity_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                DEFAULT_NOISED_X,
                DEFAULT_L_0_SENSITIVITY,
                /* lInfSensitivity= */ -1,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                DEFAULT_ALPHA));
  }

  @Test
  public void computeConfidenceInterval_forDouble_infiniteLInfSensitivity_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                DEFAULT_NOISED_X,
                DEFAULT_L_0_SENSITIVITY,
                /* lInfSensitivity= */ Double.POSITIVE_INFINITY,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                DEFAULT_ALPHA));
  }

  @Test
  public void computeConfidenceInterval_forDouble_lInfSensitivityNan_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                DEFAULT_NOISED_X,
                DEFAULT_L_0_SENSITIVITY,
                /* lInfSensitivity= */ Double.NaN,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                DEFAULT_ALPHA));
  }

  @Test
  public void computeConfidenceInterval_forDouble_tooSmallEpsilon_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                DEFAULT_NOISED_X,
                DEFAULT_L_0_SENSITIVITY,
                DEFAULT_L_INF_SENSITIVITY,
                /* epsilon= */ Math.exp(-51),
                DEFAULT_DELTA,
                DEFAULT_ALPHA));
  }

  @Test
  public void computeConfidenceInterval_forDouble_infiniteEpsilon_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                DEFAULT_NOISED_X,
                DEFAULT_L_0_SENSITIVITY,
                DEFAULT_L_INF_SENSITIVITY,
                /* epsilon= */ Double.POSITIVE_INFINITY,
                DEFAULT_DELTA,
                DEFAULT_ALPHA));
  }

  @Test
  public void computeConfidenceInterval_forDouble_epsilonNan_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                DEFAULT_NOISED_X,
                DEFAULT_L_0_SENSITIVITY,
                DEFAULT_L_INF_SENSITIVITY,
                /* epsilon= */ Double.NaN,
                DEFAULT_DELTA,
                DEFAULT_ALPHA));
  }

  @Test
  public void computeConfidenceInterval_forDouble_zeroAlpha_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                DEFAULT_NOISED_X,
                DEFAULT_L_0_SENSITIVITY,
                DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                /* alpha= */ 0));
  }

  @Test
  public void computeConfidenceInterval_forDouble_negativeAlpha_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                DEFAULT_NOISED_X,
                DEFAULT_L_0_SENSITIVITY,
                DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                /* alpha= */ -1));
  }

  @Test
  public void computeConfidenceInterval_forDouble_alphaEqualToOne_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                DEFAULT_NOISED_X,
                DEFAULT_L_0_SENSITIVITY,
                DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                /* alpha= */ 1));
  }

  @Test
  public void computeConfidenceInterval_forDouble_alphaGreaterThanOne_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                DEFAULT_NOISED_X,
                DEFAULT_L_0_SENSITIVITY,
                DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                /* alpha= */ 2));
  }

  @Test
  public void computeConfidenceInterval_forDouble_alphaNan_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                DEFAULT_NOISED_X,
                DEFAULT_L_0_SENSITIVITY,
                DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                /* alpha= */ Double.NaN));
  }

  @Test
  public void computeConfidenceInterval_forLong_arbitraryTest() {
    ConfidenceInterval actual =
        NOISE.computeConfidenceInterval(
            /* noisedX= */ 83L,
            /* l0Sensitivity= */ 3,
            /* lInfSensitivity= */ 2L,
            DEFAULT_EPSILON,
            DEFAULT_DELTA,
            /* alpha= */ 0.24);
    ConfidenceInterval expected = ConfidenceInterval.create(69.0, 97.0);
    verifyEqual(actual, expected);
  }

  @Test
  public void computeConfidenceInterval_forLong_defaultParameters() {
    ConfidenceInterval actual =
        NOISE.computeConfidenceInterval(
            (long) DEFAULT_NOISED_X,
            DEFAULT_L_0_SENSITIVITY,
            (long) DEFAULT_L_INF_SENSITIVITY,
            DEFAULT_EPSILON,
            DEFAULT_DELTA,
            DEFAULT_ALPHA);
    ConfidenceInterval expected = ConfidenceInterval.create(-6.0, 6.0);
    verifyEqual(actual, expected);
  }

  @Test
  public void computeConfidenceInterval_forLong_largePositiveNoisedX() {
    ConfidenceInterval actual =
        NOISE.computeConfidenceInterval(
            // Distance to neighbouring doubles is greater than half the size of the confidence
            // interval.
            /* noisedX= */ (1L << 58),
            DEFAULT_L_0_SENSITIVITY,
            (long) DEFAULT_L_INF_SENSITIVITY,
            DEFAULT_EPSILON,
            DEFAULT_DELTA,
            DEFAULT_ALPHA);
    ConfidenceInterval expected =
        ConfidenceInterval.create(
            Math.nextDown(Math.pow(2.0, 58.0)), Math.nextUp(Math.pow(2.0, 58.0)));
    verifyEqual(actual, expected);
  }

  @Test
  public void computeConfidenceInterval_forLong_largeNegativeNoisedX() {
    ConfidenceInterval actual =
        NOISE.computeConfidenceInterval(
            // Distance to neighbouring doubles is greater than half the size of the confidence
            // interval.
            /* noisedX= */ -(1L << 58),
            DEFAULT_L_0_SENSITIVITY,
            (long) DEFAULT_L_INF_SENSITIVITY,
            DEFAULT_EPSILON,
            DEFAULT_DELTA,
            DEFAULT_ALPHA);
    ConfidenceInterval expected =
        ConfidenceInterval.create(
            Math.nextDown(-Math.pow(2.0, 58.0)), Math.nextUp(-Math.pow(2.0, 58.0)));
    verifyEqual(actual, expected);
  }

  @Test
  public void computeConfidenceInterval_forLong_positiveNoisedX() {
    ConfidenceInterval actual =
        NOISE.computeConfidenceInterval(
            /* noisedX= */ 10,
            DEFAULT_L_0_SENSITIVITY,
            (long) DEFAULT_L_INF_SENSITIVITY,
            DEFAULT_EPSILON,
            DEFAULT_DELTA,
            DEFAULT_ALPHA);
    ConfidenceInterval expected = ConfidenceInterval.create(4.0, 16.0);
    verifyEqual(actual, expected);
  }

  @Test
  public void computeConfidenceInterval_forLong_negativeNoisedX() {
    ConfidenceInterval actual =
        NOISE.computeConfidenceInterval(
            /* noisedX= */ -10,
            DEFAULT_L_0_SENSITIVITY,
            (long) DEFAULT_L_INF_SENSITIVITY,
            DEFAULT_EPSILON,
            DEFAULT_DELTA,
            DEFAULT_ALPHA);
    ConfidenceInterval expected = ConfidenceInterval.create(-16.0, -4.0);
    verifyEqual(actual, expected);
  }

  @Test
  public void computeConfidenceInterval_forLong_smallAlpha() {
    ConfidenceInterval actual =
        NOISE.computeConfidenceInterval(
            (long) DEFAULT_NOISED_X,
            DEFAULT_L_0_SENSITIVITY,
            (long) DEFAULT_L_INF_SENSITIVITY,
            DEFAULT_EPSILON,
            DEFAULT_DELTA,
            /* alpha= */ 7.856382354E-10);
    ConfidenceInterval expected = ConfidenceInterval.create(-21.0, 21.0);
    verifyEqual(actual, expected);
  }

  @Test
  public void computeConfidenceInterval_forLong_largeAlpha() {
    ConfidenceInterval actual =
        NOISE.computeConfidenceInterval(
            (long) DEFAULT_NOISED_X,
            DEFAULT_L_0_SENSITIVITY,
            (long) DEFAULT_L_INF_SENSITIVITY,
            DEFAULT_EPSILON,
            DEFAULT_DELTA,
            /* alpha= */ 1 - 7.856382354E-10);
    ConfidenceInterval expected = ConfidenceInterval.create(0.0, 0.0);
    verifyEqual(actual, expected);
  }

  @Test
  public void computeConfidenceInterval_forLong_smallDelta() {
    ConfidenceInterval actual =
        NOISE.computeConfidenceInterval(
            (long) DEFAULT_NOISED_X,
            DEFAULT_L_0_SENSITIVITY,
            (long) DEFAULT_L_INF_SENSITIVITY,
            DEFAULT_EPSILON,
            /* delta= */ 1.78468549878E-10,
            DEFAULT_ALPHA);
    ConfidenceInterval expected = ConfidenceInterval.create(-9.0, 9.0);
    verifyEqual(actual, expected);
  }

  @Test
  public void computeConfidenceInterval_forLong_largeDelta() {
    ConfidenceInterval actual =
        NOISE.computeConfidenceInterval(
            (long) DEFAULT_NOISED_X,
            DEFAULT_L_0_SENSITIVITY,
            (long) DEFAULT_L_INF_SENSITIVITY,
            DEFAULT_EPSILON,
            /* delta= */ 1 - 1.78468549878E-10,
            DEFAULT_ALPHA);
    ConfidenceInterval expected = ConfidenceInterval.create(0.0, 0.0);
    verifyEqual(actual, expected);
  }

  @Test
  public void computeConfidenceInterval_forLong_smallEpsilon() {
    ConfidenceInterval actual =
        NOISE.computeConfidenceInterval(
            (long) DEFAULT_NOISED_X,
            DEFAULT_L_0_SENSITIVITY,
            (long) DEFAULT_L_INF_SENSITIVITY,
            /* epsilon= */ 1.65463453425E-10,
            DEFAULT_DELTA,
            DEFAULT_ALPHA);
    ConfidenceInterval expected = ConfidenceInterval.create(-65636.0, 65636.0);
    verifyEqual(actual, expected);
  }

  @Test
  public void computeConfidenceInterval_forLong_largeEpsilon() {
    ConfidenceInterval actual =
        NOISE.computeConfidenceInterval(
            (long) DEFAULT_NOISED_X,
            DEFAULT_L_0_SENSITIVITY,
            (long) DEFAULT_L_INF_SENSITIVITY,
            /* epsilon= */ 1.65463453425E10,
            DEFAULT_DELTA,
            DEFAULT_ALPHA);
    ConfidenceInterval expected = ConfidenceInterval.create(0.0, 0.0);
    verifyEqual(actual, expected);
  }

  @Test
  public void computeConfidenceInterval_forLong_zeroL0Sensitivity_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                (long) DEFAULT_NOISED_X,
                /* l0Sensitivity= */ 0,
                (long) DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                DEFAULT_ALPHA));
  }

  @Test
  public void computeConfidenceInterval_forLong_negativeL0Sensitivity_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                (long) DEFAULT_NOISED_X,
                /* l0Sensitivity= */ -1,
                (long) DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                DEFAULT_ALPHA));
  }

  @Test
  public void computeConfidenceInterval_forLong_zeroLInfSensitivity_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                (long) DEFAULT_NOISED_X,
                DEFAULT_L_0_SENSITIVITY,
                /* lInfSensitivity= */ 0L,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                DEFAULT_ALPHA));
  }

  @Test
  public void computeConfidenceInterval_forLong_negativeLInfSensitivity_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                (long) DEFAULT_NOISED_X,
                DEFAULT_L_0_SENSITIVITY,
                /* lInfSensitivity= */ -1L,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                DEFAULT_ALPHA));
  }

  @Test
  public void computeConfidenceInterval_forLong_tooSmallEpsilon_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                (long) DEFAULT_NOISED_X,
                DEFAULT_L_0_SENSITIVITY,
                (long) DEFAULT_L_INF_SENSITIVITY,
                /* epsilon= */ Math.exp(-51),
                DEFAULT_DELTA,
                DEFAULT_ALPHA));
  }

  @Test
  public void computeConfidenceInterval_forLong_infiniteEpsilon_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                (long) DEFAULT_NOISED_X,
                DEFAULT_L_0_SENSITIVITY,
                (long) DEFAULT_L_INF_SENSITIVITY,
                /* epsilon= */ Double.POSITIVE_INFINITY,
                DEFAULT_DELTA,
                DEFAULT_ALPHA));
  }

  @Test
  public void computeConfidenceInterval_forLong_epsilonNan_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                (long) DEFAULT_NOISED_X,
                DEFAULT_L_0_SENSITIVITY,
                (long) DEFAULT_L_INF_SENSITIVITY,
                /* epsilon= */ Double.NaN,
                DEFAULT_DELTA,
                DEFAULT_ALPHA));
  }

  @Test
  public void computeConfidenceInterval_forLong_zeroAlpha_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                (long) DEFAULT_NOISED_X,
                DEFAULT_L_0_SENSITIVITY,
                (long) DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                /* alpha= */ 0));
  }

  @Test
  public void computeConfidenceInterval_forLong_negativeAlpha_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                (long) DEFAULT_NOISED_X,
                DEFAULT_L_0_SENSITIVITY,
                (long) DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                /* alpha= */ -1));
  }

  @Test
  public void computeConfidenceInterval_forLong_alphaEqualToOne_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                (long) DEFAULT_NOISED_X,
                DEFAULT_L_0_SENSITIVITY,
                (long) DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                /* alpha= */ 1));
  }

  @Test
  public void computeConfidenceInterval_forLong_alphaGreaterThanOne_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                (long) DEFAULT_NOISED_X,
                DEFAULT_L_0_SENSITIVITY,
                (long) DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                /* alpha= */ 2));
  }

  @Test
  public void computeConfidenceInterval_forLong_alphaNan_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                (long) DEFAULT_NOISED_X,
                DEFAULT_L_0_SENSITIVITY,
                (long) DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                /* alpha= */ Double.NaN));
  }

  private static boolean approxEqual(double a, double b) {
    double maxMagnitude = max(Math.abs(a), Math.abs(b));
    return Math.abs(a - b) <= TOLERANCE * maxMagnitude;
  }

  private static void verifyApproxEqual(ConfidenceInterval actual, ConfidenceInterval expected) {
    assertWithMessage(
            "Lower bounds are not equal. Actual = %s, expected = %s",
            actual.lowerBound(), expected.lowerBound())
        .that(approxEqual(actual.lowerBound(), expected.lowerBound()))
        .isTrue();
    assertWithMessage(
            "Upper bounds are not equal. Actual = %s, expected = %s",
            actual.upperBound(), expected.upperBound())
        .that(approxEqual(actual.upperBound(), expected.upperBound()))
        .isTrue();
  }

  private static void verifyEqual(ConfidenceInterval actual, ConfidenceInterval expected) {
    assertWithMessage(
            "Lower bounds are not equal. Actual = %s, expected = %s",
            actual.lowerBound(), expected.lowerBound())
        .that(actual.lowerBound())
        .isEqualTo(expected.lowerBound());
    assertWithMessage(
            "Upper bounds are not equal. Actual = %s, expected = %s",
            actual.upperBound(), expected.upperBound())
        .that(actual.upperBound())
        .isEqualTo(expected.upperBound());
  }
}

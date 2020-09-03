package com.google.privacy.differentialprivacy;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import static com.google.common.truth.Truth.assertWithMessage;
import static org.junit.Assert.assertThrows;

@RunWith(JUnit4.class)
public class GaussianNoiseConfidenceIntervalTest {
  private static final Noise NOISE = new GaussianNoise();
  private static final double TOLERANCE = 1e-7;
  private static final double DEFAULT_EPSILON = 0.5;
  private static final int DEFAULT_L_0_SENSITIVITY = 1;
  private static final double DEFAULT_L_INF_SENSITIVITY = 1.0;
  private static final double DEFAULT_DELTA = 0.3;
  private static final double DEFAULT_NOISED_X = 70;
  private static final double DEFAULT_ALPHA = 0.1;

  private boolean approxEqual(double a, double b) {
    double mxMagnitude = Math.max(Math.abs(a), Math.abs(b));
    return Math.abs(a - b) <= TOLERANCE * mxMagnitude;
  }

  private void isEqual(ConfidenceInterval a, ConfidenceInterval b) {
    assertWithMessage("Lower bounds are not equal.")
        .that(a.lowerBound())
        .isWithin(TOLERANCE)
        .of(b.lowerBound());
    assertWithMessage("Upper bounds are not equal.")
        .that(a.upperBound())
        .isWithin(TOLERANCE)
        .of(b.upperBound());
  }

  private void isApproxEqual(ConfidenceInterval a, ConfidenceInterval b) {
    assertWithMessage("Lower bounds are not equal.")
        .that(approxEqual(a.lowerBound(), b.lowerBound()))
        .isTrue();
    assertWithMessage("Upper bounds are not equal.")
        .that(approxEqual(a.upperBound(), b.upperBound()))
        .isTrue();
  }

  @Test
  public void computeConfidenceInterval_double_arbitraryTest() {
    ConfidenceInterval confInt =
        NOISE.computeConfidenceInterval(
            /* noisedX */ 70.00,
            /* l0Sensitivity*/ 5,
            /* lInfSensitivity */ 36,
            /* epsilon */ 0.8,
            /* delta */ 0.8,
            /* alpha */ 0.2);
    ConfidenceInterval expected = ConfidenceInterval.create(35.26815080641682, 104.73184919358317);
    isEqual(confInt, expected);
  }

  @Test
  public void computeConfidenceInterval_double_highAlpha() {
    ConfidenceInterval confInt =
        NOISE.computeConfidenceInterval(
            /* noisedX */ 70.00,
            /* l0Sensitivity*/ 5,
            /* lInfSensitivity */ 36,
            /* epsilon */ 0.8,
            /* delta */ 0.8,
            /* alpha */ 1 - 7.856382354e-10);

    ConfidenceInterval expected = ConfidenceInterval.create(69.9999999733, 70.0000000267);
    isEqual(confInt, expected);
  }

  @Test
  public void computeConfidenceInterval_double_lowAlpha() {
    ConfidenceInterval confInt =
        NOISE.computeConfidenceInterval(
            DEFAULT_NOISED_X,
            /* l0Sensitivity*/ 5,
            /* lInfSensitivity */ 36,
            /* epsilon */ 0.8,
            /* delta */ 0.8,
            /* alpha */ 7.856382354e-10);
    ConfidenceInterval expected = ConfidenceInterval.create(-96.6140883158, 236.6140883158);
    isEqual(confInt, expected);
  }

  @Test
  public void computeConfidenceInterval_double_highNoisedX() {
    ConfidenceInterval confInt =
        NOISE.computeConfidenceInterval(
            /* noisedX */ 9.58655474558e10,
            DEFAULT_L_0_SENSITIVITY,
            DEFAULT_L_INF_SENSITIVITY,
            DEFAULT_EPSILON,
            DEFAULT_DELTA,
            DEFAULT_ALPHA);
    ConfidenceInterval expected = ConfidenceInterval.create(95865547454.4, 95865547457.2);
    // Relative error is used instead of absolute error since bounds values are less than TOLERANCE.
    isApproxEqual(confInt, expected);
  }

  @Test
  public void computeConfidenceInterval_double_lowNoisedX() {
    ConfidenceInterval confInt =
        NOISE.computeConfidenceInterval(
            /* noisedX */ 1.58655474558e-12,
            DEFAULT_L_0_SENSITIVITY,
            DEFAULT_L_INF_SENSITIVITY,
            DEFAULT_EPSILON,
            DEFAULT_DELTA,
            DEFAULT_ALPHA);
    ConfidenceInterval expected = ConfidenceInterval.create(-1.42479020225, 1.42479020225);
    isEqual(confInt, expected);
  }

  @Test
  public void computeConfidenceInterval_double_negativeNoisedX() {
    ConfidenceInterval confInt =
        NOISE.computeConfidenceInterval(
            /* noisedX */ -1.58655474558,
            DEFAULT_L_0_SENSITIVITY,
            DEFAULT_L_INF_SENSITIVITY,
            DEFAULT_EPSILON,
            DEFAULT_DELTA,
            DEFAULT_ALPHA);
    ConfidenceInterval expected =
        ConfidenceInterval.create(-3.0113449478319105345747175, -0.1617645433280894273764261);
    isEqual(confInt, expected);
  }

  @Test
  public void computeConfidenceInterval_double_lowDelta() {
    ConfidenceInterval confInt =
        NOISE.computeConfidenceInterval(
            DEFAULT_NOISED_X,
            DEFAULT_L_0_SENSITIVITY,
            DEFAULT_L_INF_SENSITIVITY,
            DEFAULT_EPSILON,
            /* delta */ 1.78468549878e-10,
            DEFAULT_ALPHA);
    ConfidenceInterval expected =
        ConfidenceInterval.create(51.4953966967959289036116388, 88.5046033032040782018157188);
    isEqual(confInt, expected);
  }

  @Test
  public void computeConfidenceInterval_double_highDelta() {
    ConfidenceInterval confInt =
        NOISE.computeConfidenceInterval(
            DEFAULT_NOISED_X,
            DEFAULT_L_0_SENSITIVITY,
            DEFAULT_L_INF_SENSITIVITY,
            DEFAULT_EPSILON,
            /* delta */ 1 - 1.78468549878e-10,
            DEFAULT_ALPHA);
    ConfidenceInterval expected =
        ConfidenceInterval.create(69.8717969920888037904660450, 70.1282030079111962095339550);
    isEqual(confInt, expected);
  }

  @Test
  public void computeConfidenceInterval_double_lowEpsilon() {
    ConfidenceInterval confInt =
        NOISE.computeConfidenceInterval(
            DEFAULT_NOISED_X,
            DEFAULT_L_0_SENSITIVITY,
            /* lInfSensitivity */ 2,
            /* epsilon */ 1.65463453425e-10,
            DEFAULT_DELTA,
            DEFAULT_ALPHA);
    ConfidenceInterval expected = ConfidenceInterval.create(65.73044830035448, 74.26955169964552);
    isEqual(confInt, expected);
  }

  @Test
  public void computeConfidenceInterval_double_highEpsilon() {
    ConfidenceInterval confInt =
        NOISE.computeConfidenceInterval(
            DEFAULT_NOISED_X,
            DEFAULT_L_0_SENSITIVITY,
            /* lInfSensitivity */ 2,
            /* epsilon */ 1.65463453425e10,
            DEFAULT_DELTA,
            DEFAULT_ALPHA);
    ConfidenceInterval expected = ConfidenceInterval.create(70, 70);
    isEqual(confInt, expected);
  }

  @Test
  public void computeConfidenceInterval_long_arbitraryTest() {
    ConfidenceInterval confInt =
        NOISE.computeConfidenceInterval(
            DEFAULT_NOISED_X,
            /* l0Sensitivity*/ 5,
            /* lInfSensitivity */ 36,
            /* epsilon */ 0.8,
            /* delta */ 0.8,
            /* alpha */ 0.2);
    ConfidenceInterval expected = ConfidenceInterval.create(35.26815080641682, 104.73184919358317);
    isEqual(confInt, expected);
  }

  @Test
  public void computeConfidenceInterval_long_highAlpha() {
    ConfidenceInterval confInt =
        NOISE.computeConfidenceInterval(
            DEFAULT_NOISED_X,
            /* l0Sensitivity*/ 5,
            /* lInfSensitivity */ 36,
            /* epsilon */ 0.8,
            /* delta */ 0.8,
            /* alpha */ 1 - 7.856382354e-1);
    ConfidenceInterval expected = ConfidenceInterval.create(36.34924852649089, 103.6507514735091);
    isEqual(confInt, expected);
  }

  @Test
  public void computeConfidenceInterval_long_lowAlpha() {
    ConfidenceInterval confInt =
        NOISE.computeConfidenceInterval(
            DEFAULT_NOISED_X,
            /* l0Sensitivity*/ 5,
            /* lInfSensitivity */ 36,
            /* epsilon */ 0.8,
            /* delta */ 0.8,
            /* alpha */ 7.856382354e-10);
    ConfidenceInterval expected = ConfidenceInterval.create(-96.61408831584356, 236.61408831584356);
    isEqual(confInt, expected);
  }

  @Test
  public void computeConfidenceInterval_long_highNoisedX() {
    ConfidenceInterval confInt =
        NOISE.computeConfidenceInterval(
            /* noisedX */ 9e9,
            DEFAULT_L_0_SENSITIVITY,
            DEFAULT_L_INF_SENSITIVITY,
            DEFAULT_EPSILON,
            DEFAULT_DELTA,
            DEFAULT_ALPHA);
    ConfidenceInterval expected = ConfidenceInterval.create(8.999999999E9, 9.000000001E9);
    // Relative error is used instead of absolute error since bounds values are less than TOLERANCE.
    isApproxEqual(confInt, expected);
  }

  @Test
  public void computeConfidenceInterval_long_lowNoisedX() {
    ConfidenceInterval confInt =
        NOISE.computeConfidenceInterval(
            /* noisedX */ 0,
            DEFAULT_L_0_SENSITIVITY,
            DEFAULT_L_INF_SENSITIVITY,
            DEFAULT_EPSILON,
            DEFAULT_DELTA,
            DEFAULT_ALPHA);
    ConfidenceInterval expected =
        ConfidenceInterval.create(-1.4247902022519105535991457, 1.4247902022519105535991457);
    // Relative error is used instead of absolute error since bounds values are less than TOLERANCE.
    isApproxEqual(confInt, expected);
  }

  @Test
  public void computeConfidenceInterval_long_negativeNoisedX() {
    ConfidenceInterval confInt =
        NOISE.computeConfidenceInterval(
            /* noisedX */ -10,
            DEFAULT_L_0_SENSITIVITY,
            DEFAULT_L_INF_SENSITIVITY,
            DEFAULT_EPSILON,
            DEFAULT_DELTA,
            DEFAULT_ALPHA);
    ConfidenceInterval expected =
        ConfidenceInterval.create(-11.424790202251911, -8.575209797748089);
    isEqual(confInt, expected);
  }

  @Test
  public void computeConfidenceInterval_long_lowDelta() {
    ConfidenceInterval confInt =
        NOISE.computeConfidenceInterval(
            DEFAULT_NOISED_X,
            DEFAULT_L_0_SENSITIVITY,
            DEFAULT_L_INF_SENSITIVITY,
            DEFAULT_EPSILON,
            /* delta */ 1.78468549878e-10,
            DEFAULT_ALPHA);
    ConfidenceInterval expected = ConfidenceInterval.create(51.49539669679593, 88.50460330320408);
    isEqual(confInt, expected);
  }

  @Test
  public void computeConfidenceInterval_long_highDelta() {
    ConfidenceInterval confInt =
        NOISE.computeConfidenceInterval(
            DEFAULT_NOISED_X,
            DEFAULT_L_0_SENSITIVITY,
            DEFAULT_L_INF_SENSITIVITY,
            DEFAULT_EPSILON,
            /* delta */ 1 - 1.78468549878e-10,
            DEFAULT_ALPHA);
    ConfidenceInterval expected =
        ConfidenceInterval.create(69.8717969920888037904660450, 70.1282030079111962095339550);
    isEqual(confInt, expected);
  }

  @Test
  public void computeConfidenceInterval_long_lowEpsilon() {
    ConfidenceInterval confInt =
        NOISE.computeConfidenceInterval(
            DEFAULT_NOISED_X,
            DEFAULT_L_0_SENSITIVITY,
            /* lInfSensitivity */ 2,
            /* epsilon */ 1.65463453425e-10,
            DEFAULT_DELTA,
            DEFAULT_ALPHA);
    ConfidenceInterval expected = ConfidenceInterval.create(65.73044830035448, 74.26955169964552);
    isEqual(confInt, expected);
  }

  @Test
  public void computeConfidenceInterval_long_highEpsilon() {
    ConfidenceInterval confInt =
        NOISE.computeConfidenceInterval(
            DEFAULT_NOISED_X,
            DEFAULT_L_0_SENSITIVITY,
            /* lInfSensitivity */ 2,
            /* epsilon */ 1.65463453425e10,
            DEFAULT_DELTA,
            DEFAULT_ALPHA);
    ConfidenceInterval expected = ConfidenceInterval.create(70, 70);
    isEqual(confInt, expected);
  }

  @Test
  public void computeConfidenceInterval_double_zeroL0Sensitivity_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                /* noisedX */ 0,
                /* l0Sensitivity*/ 0,
                DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                /* alpha */ 0.5));
  }

  @Test
  public void computeConfidenceInterval_double_negativeL0Sensitivity_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                /* noisedX */ 0,
                /* l0Sensitivity*/ -1,
                DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                /* alpha */ 0.5));
  }

  @Test
  public void computeConfidenceInterval_double_zeroLInfSensitivity_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                /* noisedX */ 0,
                DEFAULT_L_0_SENSITIVITY,
                /* lInfSensitivity*/ 0,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                /* alpha*/ 0.5));
  }

  @Test
  public void computeConfidenceInterval_double_negativeLInfSensitivity_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                /* noisedX */ 0,
                DEFAULT_L_0_SENSITIVITY,
                /* lInfSensitivity*/ -1,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                /* alpha*/ 0.5));
  }

  @Test
  public void computeConfidenceInterval_double_infiniteLInfSensitivity_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                /* noisedX */ 0,
                DEFAULT_L_0_SENSITIVITY,
                /* lInfSensitivity*/ Double.POSITIVE_INFINITY,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                /* alpha*/ 0.5));
  }

  @Test
  public void computeConfidenceInterval_double_NaNLInfSensitivity_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                /* noisedX */ 0,
                DEFAULT_L_0_SENSITIVITY,
                /* lInfSensitivity*/ Double.NaN,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                /* alpha*/ 0.5));
  }

  @Test
  public void computeConfidenceInterval_double_tooSmallEpsilon_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                /* noisedX */ 0,
                DEFAULT_L_0_SENSITIVITY,
                DEFAULT_L_INF_SENSITIVITY,
                /* epsilon*/ Math.exp(-51),
                DEFAULT_DELTA,
                /* alpha*/ 0));
  }

  @Test
  public void computeConfidenceInterval_double_infiniteEpsilon_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                /* noisedX */ 0,
                DEFAULT_L_0_SENSITIVITY,
                DEFAULT_L_INF_SENSITIVITY,
                /* epsilon*/ Double.POSITIVE_INFINITY,
                DEFAULT_DELTA,
                /* alpha*/ 0.5));
  }

  @Test
  public void computeConfidenceInterval_double_NaNEpsilon_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                /* noisedX */ 0,
                DEFAULT_L_0_SENSITIVITY,
                DEFAULT_L_INF_SENSITIVITY,
                /* epsilon*/ Double.NaN,
                DEFAULT_DELTA,
                /* alpha*/ 0));
  }

  @Test
  public void computeConfidenceInterval_double_zeroAlpha_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                DEFAULT_NOISED_X,
                DEFAULT_L_0_SENSITIVITY,
                DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                /* alpha */ 0));
  }

  @Test
  public void computeConfidenceInterval_double_negativeAlpha_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                /* noisedX */ 0,
                DEFAULT_L_0_SENSITIVITY,
                DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                /* alpha */ -1));
  }

  @Test
  public void computeConfidenceInterval_double_alphaEqualToOne_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                DEFAULT_NOISED_X,
                DEFAULT_L_0_SENSITIVITY,
                DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                /* alpha */ 1));
  }

  @Test
  public void computeConfidenceInterval_double_alphaGreaterThanOne_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                DEFAULT_NOISED_X,
                DEFAULT_L_0_SENSITIVITY,
                DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                /* alpha */ 2));
  }

  @Test
  public void computeConfidenceInterval_double_NaNAlpha_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                DEFAULT_NOISED_X,
                DEFAULT_L_0_SENSITIVITY,
                DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                /* alpha */ Double.NaN));
  }

  @Test
  public void computeConfidenceInterval_long_zeroL0Sensitivity_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                /* noisedX */ 0,
                /* l0Sensitivity */ 0,
                DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                /* alpha */ 0.5));
  }

  @Test
  public void computeConfidenceInterval_long_negativeL0Sensitivity_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                /* noisedX */ 0,
                /* l0Sensitivity */ -1,
                DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                /* alpha */ 0.5));
  }

  @Test
  public void computeConfidenceInterval_long_zeroLInfSensitivity_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                /* noisedX */ 0,
                DEFAULT_L_0_SENSITIVITY,
                /* lInfSensitivity */ 0,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                /* alpha */ 0.5));
  }

  @Test
  public void computeConfidenceInterval_long_negativeLInfSensitivity_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                /* noisedX */ 0,
                DEFAULT_L_0_SENSITIVITY,
                /* lInfSensitivity */ -1,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                /* alpha */ 0.5));
  }

  @Test
  public void computeConfidenceInterval_long_infiniteLInfSensitivity_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                /* noisedX */ 0,
                DEFAULT_L_0_SENSITIVITY,
                /* lInfSensitivity */ Double.POSITIVE_INFINITY,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                /* alpha */ 0.5));
  }

  @Test
  public void computeConfidenceInterval_long_NaNLInfSensitivity_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                /* noisedX */ 0,
                DEFAULT_L_0_SENSITIVITY,
                /* lInfSensitivity */ Double.NaN,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                /* alpha */ 0.5));
  }

  @Test
  public void computeConfidenceInterval_long_tooSmallEpsilon_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                /* noisedX */ 0,
                DEFAULT_L_0_SENSITIVITY,
                DEFAULT_L_INF_SENSITIVITY,
                /* epsilon */ Math.exp(-51),
                DEFAULT_DELTA,
                /* alpha */ 0));
  }

  @Test
  public void computeConfidenceInterval_long_infiniteEpsilon_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                /* noisedX */ 0,
                DEFAULT_L_0_SENSITIVITY,
                DEFAULT_L_INF_SENSITIVITY,
                /* epsilon */ Double.POSITIVE_INFINITY,
                DEFAULT_DELTA,
                /* alpha */ 0.5));
  }

  @Test
  public void computeConfidenceInterval_long_NaNEpsilon_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                /* noisedX */ 0,
                DEFAULT_L_0_SENSITIVITY,
                DEFAULT_L_INF_SENSITIVITY,
                /* epsilon */ Double.NaN,
                DEFAULT_DELTA,
                /* alpha */ 0));
  }

  @Test
  public void computeConfidenceInterval_long_zeroAlpha_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                (long) DEFAULT_NOISED_X,
                DEFAULT_L_0_SENSITIVITY,
                (long) DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                /* alpha */ 0));
  }

  @Test
  public void computeConfidenceInterval_long_negativeAlpha_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                (long) DEFAULT_NOISED_X,
                DEFAULT_L_0_SENSITIVITY,
                (long) DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                /* alpha */ -1));
  }

  @Test
  public void computeConfidenceInterval_long_alphaEqualToOne_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                (long) DEFAULT_NOISED_X,
                DEFAULT_L_0_SENSITIVITY,
                (long) DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                /* alpha */ 1));
  }

  @Test
  public void computeConfidenceInterval_long_alphaGreaterThanOne_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                (long) DEFAULT_NOISED_X,
                DEFAULT_L_0_SENSITIVITY,
                (long) DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                /* alpha */ 2));
  }

  @Test
  public void computeConfidenceInterval_long_NaNAlpha_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                (long) DEFAULT_NOISED_X,
                DEFAULT_L_0_SENSITIVITY,
                (long) DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                /* alpha */ Double.NaN));
  }
}

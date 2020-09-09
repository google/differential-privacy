package com.google.privacy.differentialprivacy;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import static com.google.common.truth.Truth.assertWithMessage;
import static java.lang.Math.max;
import static org.junit.Assert.assertThrows;

@RunWith(JUnit4.class)
public class LaplaceNoiseConfidenceIntervalTest {
  private static final Noise NOISE = new LaplaceNoise();
  private static final double TOLERANCE = 1e-6;
  private static final double DEFAULT_NOISEDX = 0.0;
  private static final int DEFAULT_L_0_SENSITIVITY = 1;
  private static final double DEFAULT_L_INF_SENSITIVITY = 1.0;
  private static final double DEFAULT_EPSILON = 0.1;
  private static final Double DEFAULT_DELTA = null;
  private static final double DEFAULT_ALPHA = 0.1;

  @Test
  public void computeConfidenceInterval_forDouble_arbitraryTest() {
    ConfidenceInterval actual =
        NOISE.computeConfidenceInterval(
            /* noisedX */ 13.0,
            DEFAULT_L_0_SENSITIVITY,
            DEFAULT_L_INF_SENSITIVITY,
            /* epsilon */ 0.3,
            DEFAULT_DELTA,
            /* alpha */ 0.05);
    ConfidenceInterval expected = ConfidenceInterval.create(3.014225755, 22.98577425);
    verifyApproxEqual(actual, expected);
  }

  @Test
  public void computeConfidenceInterval_forDouble_lowConfidenceLevel() {
    ConfidenceInterval actual =
        NOISE.computeConfidenceInterval(
            DEFAULT_NOISEDX,
            DEFAULT_L_0_SENSITIVITY,
            DEFAULT_L_INF_SENSITIVITY,
            DEFAULT_EPSILON,
            DEFAULT_DELTA,
            /* alpha */ 1 - 3.548957438e-10);
    ConfidenceInterval expected = ConfidenceInterval.create(-3.548957437370e-9, 3.548957437370e-9);
    verifyApproxEqual(actual, expected);
  }

  @Test
  public void computeConfidenceInterval_forDouble_highConfidenceLevel() {
    ConfidenceInterval actual =
        NOISE.computeConfidenceInterval(
            /* noisedX */ 50,
            DEFAULT_L_0_SENSITIVITY,
            DEFAULT_L_INF_SENSITIVITY,
            DEFAULT_EPSILON,
            DEFAULT_DELTA,
            /* alpha */ 7.856382354e-10);
    ConfidenceInterval expected = ConfidenceInterval.create(-159.645246897, 259.645246897);
    verifyApproxEqual(actual, expected);
  }

  @Test
  public void computeConfidenceInterval_forDouble_lowEpsilon() {
    ConfidenceInterval actual =
        NOISE.computeConfidenceInterval(
            DEFAULT_NOISEDX,
            DEFAULT_L_0_SENSITIVITY,
            DEFAULT_L_INF_SENSITIVITY,
            /* epsilon */ 1.567321563235e-10,
            DEFAULT_DELTA,
            DEFAULT_ALPHA);
    ConfidenceInterval expected = ConfidenceInterval.create(-14691210451.04132, 14691210451.04132);
    verifyApproxEqual(actual, expected);
  }

  @Test
  public void computeConfidenceInterval_forDouble_highEpsilon() {
    ConfidenceInterval actual =
        NOISE.computeConfidenceInterval(
            DEFAULT_NOISEDX,
            DEFAULT_L_0_SENSITIVITY,
            DEFAULT_L_INF_SENSITIVITY,
            /* epsilon */ 1.567321563235e10,
            DEFAULT_DELTA,
            DEFAULT_ALPHA);
    ConfidenceInterval expected = ConfidenceInterval.create(-1.4691210451E-10, 1.4691210451E-10);
    verifyApproxEqual(actual, expected);
  }

  @Test
  public void computeConfidenceInterval_forDouble_highPositiveNoisedX() {
    ConfidenceInterval actual =
        NOISE.computeConfidenceInterval(
            /* noisedX */ 3847569385690.0,
            DEFAULT_L_0_SENSITIVITY,
            DEFAULT_L_INF_SENSITIVITY,
            DEFAULT_EPSILON,
            DEFAULT_DELTA,
            DEFAULT_ALPHA);
    // Double precision should be accurate for abs(noisedX) < 2^54.
    ConfidenceInterval expected =
        ConfidenceInterval.create(3847569385666.974, 3847569385713.026);
    verifyApproxEqual(actual, expected);
  }

  @Test
  public void computeConfidenceInterval_forDouble_highNegativeNoisedX() {
    ConfidenceInterval actual =
        NOISE.computeConfidenceInterval(
            /* noisedX */ -3847569385690.0,
            DEFAULT_L_0_SENSITIVITY,
            (long) DEFAULT_L_INF_SENSITIVITY,
            DEFAULT_EPSILON,
            DEFAULT_DELTA,
            DEFAULT_ALPHA);
    // Double precision should be accurate for abs(noisedX) < 2^54.
    ConfidenceInterval expected =
        ConfidenceInterval.create(-3847569385713.026, -3847569385666.974);
    verifyApproxEqual(actual, expected);
  }

  @Test
  public void computeConfidenceInterval_forDouble_zeroL0Sensitivity_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                DEFAULT_NOISEDX,
                /* l0Sensitivity */ 0,
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
                DEFAULT_NOISEDX,
                /* l0Sensitivity */ -1,
                DEFAULT_L_0_SENSITIVITY,
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
                DEFAULT_NOISEDX,
                DEFAULT_L_0_SENSITIVITY,
                /* lInfSensitivity */ 0,
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
                DEFAULT_NOISEDX,
                DEFAULT_L_0_SENSITIVITY,
                /* lInfSensitivity */ -1,
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
                DEFAULT_NOISEDX,
                DEFAULT_L_0_SENSITIVITY,
                /* lInfSensitivity */ Double.POSITIVE_INFINITY,
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
                DEFAULT_NOISEDX,
                DEFAULT_L_0_SENSITIVITY,
                /* lInfSensitivity */ Double.NaN,
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
                DEFAULT_NOISEDX,
                DEFAULT_L_0_SENSITIVITY,
                DEFAULT_L_INF_SENSITIVITY,
                /* epsilon */ 1.0 / (1L << 51),
                DEFAULT_DELTA,
                DEFAULT_ALPHA));
  }

  @Test
  public void computeConfidenceInterval_forDouble_infiniteEpsilon_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                DEFAULT_NOISEDX,
                DEFAULT_L_0_SENSITIVITY,
                DEFAULT_L_INF_SENSITIVITY,
                /* epsilon */ Double.POSITIVE_INFINITY,
                DEFAULT_DELTA,
                DEFAULT_ALPHA));
  }

  @Test
  public void computeConfidenceInterval_forDouble_epsilonNan_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                DEFAULT_NOISEDX,
                DEFAULT_L_0_SENSITIVITY,
                DEFAULT_L_INF_SENSITIVITY,
                /* epsilon */ Double.NaN,
                DEFAULT_DELTA,
                DEFAULT_ALPHA));
  }

  @Test
  public void computeConfidenceInterval_forDouble_nonNullDelta_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                DEFAULT_NOISEDX,
                DEFAULT_L_0_SENSITIVITY,
                DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                /* delta */ 1.0,
                DEFAULT_ALPHA));
  }

  @Test
  public void computeConfidenceInterval_forDouble_zeroAlpha_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                DEFAULT_NOISEDX,
                DEFAULT_L_0_SENSITIVITY,
                DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                /* alpha */ 0));
  }

  @Test
  public void computeConfidenceInterval_forDouble_negativeAlpha_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                DEFAULT_NOISEDX,
                DEFAULT_L_0_SENSITIVITY,
                DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                /* alpha */ -1));
  }

  @Test
  public void computeConfidenceInterval_forDouble_alphaEqualToOne_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                DEFAULT_NOISEDX,
                DEFAULT_L_0_SENSITIVITY,
                DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                /* alpha */ 1));
  }

  @Test
  public void computeConfidenceInterval_forDouble_alphaGreaterThanOne_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                DEFAULT_NOISEDX,
                DEFAULT_L_0_SENSITIVITY,
                DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                /* alpha */ 2));
  }

  @Test
  public void computeConfidenceInterval_forDouble_alphaNan_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                DEFAULT_NOISEDX,
                DEFAULT_L_0_SENSITIVITY,
                DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                /* alpha */ Double.NaN));
  }

  @Test
  public void computeConfidenceInterval_forLong_arbitraryTest() {
    ConfidenceInterval actual =
        NOISE.computeConfidenceInterval(
            /* noisedX */ 83,
            /* l0Sensitivity */ 3,
            /* lInfSensitivity */ 2,
            DEFAULT_EPSILON,
            DEFAULT_DELTA,
            /* alpha */ 0.24);
    ConfidenceInterval expected = ConfidenceInterval.create(-3, 169);
    verifyEqual(actual, expected);
  }

  @Test
  public void computeConfidenceInterval_forLong_lowConfidenceLevel() {
    ConfidenceInterval actual =
        NOISE.computeConfidenceInterval(
            (long) DEFAULT_NOISEDX,
            DEFAULT_L_0_SENSITIVITY,
            (long) DEFAULT_L_INF_SENSITIVITY,
            DEFAULT_EPSILON,
            DEFAULT_DELTA,
            /* alpha */ 1 - 3.548957438e-10);
    ConfidenceInterval expected = ConfidenceInterval.create(0, 0);
    verifyEqual(actual, expected);
  }

  @Test
  public void computeConfidenceInterval_forLong_highConfidenceLevel() {
    ConfidenceInterval actual =
        NOISE.computeConfidenceInterval(
            /* noisedX */ 50L,
            DEFAULT_L_0_SENSITIVITY,
            (long) DEFAULT_L_INF_SENSITIVITY,
            DEFAULT_EPSILON,
            DEFAULT_DELTA,
            /* alpha */ 7.856382354e-10);
    ConfidenceInterval expected = ConfidenceInterval.create(-160, 260);
    verifyEqual(actual, expected);
  }

  @Test
  public void computeConfidenceInterval_forLong_lowEpsilon() {
    ConfidenceInterval actual =
        NOISE.computeConfidenceInterval(
            (long) DEFAULT_NOISEDX,
            DEFAULT_L_0_SENSITIVITY,
            (long) DEFAULT_L_INF_SENSITIVITY,
            /* epsilon */ 1.567321563235e-10,
            DEFAULT_DELTA,
            DEFAULT_ALPHA);
    ConfidenceInterval expected = ConfidenceInterval.create(-14691210451.0, 14691210451.0);
    verifyEqual(actual, expected);
  }

  @Test
  public void computeConfidenceInterval_forLong_highEpsilon() {
    ConfidenceInterval actual =
        NOISE.computeConfidenceInterval(
            (long) DEFAULT_NOISEDX,
            DEFAULT_L_0_SENSITIVITY,
            (long) DEFAULT_L_INF_SENSITIVITY,
            /* epsilon */ 1.567321563235e10,
            DEFAULT_DELTA,
            DEFAULT_ALPHA);
    ConfidenceInterval expected = ConfidenceInterval.create(0, 0);
    verifyEqual(actual, expected);
  }

  @Test
  public void computeConfidenceInterval_forLong_highPositiveNoisedX() {
    ConfidenceInterval actual =
        NOISE.computeConfidenceInterval(
            /* noisedX */ (1L << 58),
            DEFAULT_L_0_SENSITIVITY,
            (long) DEFAULT_L_INF_SENSITIVITY,
            DEFAULT_EPSILON,
            DEFAULT_DELTA,
            DEFAULT_ALPHA);
    // Z value = -23, upperBound = 1 << 58 + 23 = 288230376151711767 gets rounded
    // down to 288230376151711744, nextLargeDouble rounds it back up to 288230376151711808.
    // lowerBound = 1 << 58 - 23 = 288230376151711721 gets rounded down to 288230376151711712,
    // nextSmaller returns the same value.
    ConfidenceInterval expected =
        ConfidenceInterval.create(288230376151711712.0, 2.8823037615171181E+17);
    verifyEqual(actual, expected);
  }

  @Test
  public void computeConfidenceInterval_forLong_highNegativeNoisedX() {
    ConfidenceInterval actual =
        NOISE.computeConfidenceInterval(
            /* noisedX */ -(1L << 58),
            DEFAULT_L_0_SENSITIVITY,
            (long) DEFAULT_L_INF_SENSITIVITY,
            DEFAULT_EPSILON,
            DEFAULT_DELTA,
            DEFAULT_ALPHA);
    // Z value = -23, upperBound = -(1 << 58) + 23 = -288230376151711721 gets rounded
    // up to -288230376151711712, nextLargeDouble returns the same value.
    // lowerBound = -(1 << 58) - 23 = -288230376151711767 gets rounded up to -288230376151711744,
    // nextSmaller rounds it back down to -288230376151711808.
    ConfidenceInterval expected =
        ConfidenceInterval.create(-2.8823037615171181E+17, -288230376151711712.0);
    verifyEqual(actual, expected);
  }

  @Test
  public void computeConfidenceInterval_forLong_zeroL0Sensitivity_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                (long) DEFAULT_NOISEDX,
                /* l0Sensitivity */ 0,
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
                (long) DEFAULT_NOISEDX,
                /* l0Sensitivity */ -1,
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
                (long) DEFAULT_NOISEDX,
                DEFAULT_L_0_SENSITIVITY,
                /* lInfSensitivity */ 0,
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
                (long) DEFAULT_NOISEDX,
                DEFAULT_L_0_SENSITIVITY,
                /* lInfSensitivity */ -1,
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
                (long) DEFAULT_NOISEDX,
                DEFAULT_L_0_SENSITIVITY,
                (long) DEFAULT_L_INF_SENSITIVITY,
                /* epsilon */ 1.0 / (1L << 51),
                DEFAULT_DELTA,
                DEFAULT_ALPHA));
  }

  @Test
  public void computeConfidenceInterval_forLong_infiniteEpsilon_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                (long) DEFAULT_NOISEDX,
                DEFAULT_L_0_SENSITIVITY,
                (long) DEFAULT_L_INF_SENSITIVITY,
                /* epsilon */ Double.POSITIVE_INFINITY,
                DEFAULT_DELTA,
                DEFAULT_ALPHA));
  }

  @Test
  public void computeConfidenceInterval_forLong_epsilonNan_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                (long) DEFAULT_NOISEDX,
                DEFAULT_L_0_SENSITIVITY,
                (long) DEFAULT_L_INF_SENSITIVITY,
                /* epsilon */ Double.NaN,
                DEFAULT_DELTA,
                DEFAULT_ALPHA));
  }

  @Test
  public void computeConfidenceInterval_forLong_nonNullDelta_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                (long) DEFAULT_NOISEDX,
                DEFAULT_L_0_SENSITIVITY,
                (long) DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                /* delta */ 1.0,
                DEFAULT_ALPHA));
  }

  @Test
  public void computeConfidenceInterval_forLong_zeroAlpha_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                (long) DEFAULT_NOISEDX,
                DEFAULT_L_0_SENSITIVITY,
                (long) DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                /* alpha */ 0));
  }

  @Test
  public void computeConfidenceInterval_forLong_negativeAlpha_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                (long) DEFAULT_NOISEDX,
                DEFAULT_L_0_SENSITIVITY,
                (long) DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                /* alpha */ -1));
  }

  @Test
  public void computeConfidenceInterval_forLong_alphaEqualToOne_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                (long) DEFAULT_NOISEDX,
                DEFAULT_L_0_SENSITIVITY,
                (long) DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                /* alpha */ 1));
  }

  @Test
  public void computeConfidenceInterval_forLong_alphaGreaterThanOne_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                (long) DEFAULT_NOISEDX,
                DEFAULT_L_0_SENSITIVITY,
                (long) DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                /* alpha */ 2));
  }

  @Test
  public void computeConfidenceInterval_forLong_alphaNan_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            NOISE.computeConfidenceInterval(
                (long) DEFAULT_NOISEDX,
                DEFAULT_L_0_SENSITIVITY,
                (long) DEFAULT_L_INF_SENSITIVITY,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                /* alpha */ Double.NaN));
  }

  private static boolean approxEqual(double a, double b) {
    double maxMagnitude = max(Math.abs(a), Math.abs(b));
    return Math.abs(a - b) <= TOLERANCE * maxMagnitude;
  }

  private static void verifyApproxEqual(ConfidenceInterval a, ConfidenceInterval b) {
    assertWithMessage("Lower bounds are not equal.")
        .that(approxEqual(a.lowerBound(), b.lowerBound()))
        .isTrue();
    assertWithMessage("Upper bounds are not equal.")
        .that(approxEqual(a.upperBound(), b.upperBound()))
        .isTrue();
  }

  private static void verifyEqual(ConfidenceInterval a, ConfidenceInterval b) {
    assertWithMessage("Lower bounds are not equal.")
        .that(a.lowerBound())
        .isEqualTo(b.lowerBound());
    assertWithMessage("Upper bounds are not equal.")
        .that(a.upperBound())
        .isEqualTo(b.upperBound());
  }
}
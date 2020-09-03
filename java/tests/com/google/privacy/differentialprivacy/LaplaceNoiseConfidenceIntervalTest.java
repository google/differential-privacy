package com.google.privacy.differentialprivacy;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import static com.google.common.truth.Truth.assertWithMessage;
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
            /* noisedX */ 13.0,
            DEFAULT_L_0_SENSITIVITY,
            DEFAULT_L_INF_SENSITIVITY,
            /* epsilon */ 0.3,
            DEFAULT_DELTA,
            /* alpha */ 0.05);
    ConfidenceInterval expected = ConfidenceInterval.create(3.014225755, 22.98577425);
    isEqual(confInt, expected);
  }

  @Test
  public void computeConfidenceInterval_double_lowConfidenceLevel() {
    ConfidenceInterval confInt =
        NOISE.computeConfidenceInterval(
            DEFAULT_NOISEDX,
            DEFAULT_L_0_SENSITIVITY,
            DEFAULT_L_INF_SENSITIVITY,
            DEFAULT_EPSILON,
            DEFAULT_DELTA,
            /* alpha */ 1 - 3.548957438e-10);
    ConfidenceInterval expected = ConfidenceInterval.create(-3.548957437370e-9, 3.548957437370e-9);
    // Relative error is used instead of absolute error since bounds values are less than TOLERANCE.
    isApproxEqual(confInt, expected);
  }

  @Test
  public void computeConfidenceInterval_double_highConfidenceLevel() {
    ConfidenceInterval confInt =
        NOISE.computeConfidenceInterval(
            /* noisedX */ 50,
            DEFAULT_L_0_SENSITIVITY,
            DEFAULT_L_INF_SENSITIVITY,
            DEFAULT_EPSILON,
            DEFAULT_DELTA,
            /* alpha */ 7.856382354e-10);
    ConfidenceInterval expected = ConfidenceInterval.create(-159.645246897, 259.645246897);
    isEqual(confInt, expected);
  }

  @Test
  public void computeConfidenceInterval_double_lowEpsilon() {
    ConfidenceInterval confInt =
        NOISE.computeConfidenceInterval(
            DEFAULT_NOISEDX,
            DEFAULT_L_0_SENSITIVITY,
            DEFAULT_L_INF_SENSITIVITY,
            /* epsilon */ 1.567321563235e-10,
            DEFAULT_DELTA,
            DEFAULT_ALPHA);
    ConfidenceInterval expected = ConfidenceInterval.create(-14691210451.04132, 14691210451.04132);
    isEqual(confInt, expected);
  }

  @Test
  public void computeConfidenceInterval_double_highEpsilon() {
    ConfidenceInterval confInt =
        NOISE.computeConfidenceInterval(
            DEFAULT_NOISEDX,
            DEFAULT_L_0_SENSITIVITY,
            DEFAULT_L_INF_SENSITIVITY,
            /* epsilon */ 1.567321563235e10,
            DEFAULT_DELTA,
            DEFAULT_ALPHA);
    ConfidenceInterval expected = ConfidenceInterval.create(-1.4691210451E-10, 1.4691210451E-10);
    // Relative error is used instead of absolute error since bounds values are less than TOLERANCE.
    isApproxEqual(confInt, expected);
  }

  @Test
  public void computeConfidenceInterval_double_highPositiveNoisedX() {
    ConfidenceInterval confInt =
        NOISE.computeConfidenceInterval(
            /* noisedX */ 3847569385690.0,
            DEFAULT_L_0_SENSITIVITY,
            DEFAULT_L_INF_SENSITIVITY,
            DEFAULT_EPSILON,
            DEFAULT_DELTA,
            DEFAULT_ALPHA);
    // Double precision should be accurate for abs(noisedX) < 2^54.
    ConfidenceInterval expected =
        ConfidenceInterval.create(3847569385666.974149, 3847569385713.02585093);
    isEqual(confInt, expected);
  }

  @Test
  public void computeConfidenceInterval_double_highNegativeNoisedX() {
    ConfidenceInterval confInt =
        NOISE.computeConfidenceInterval(
            /* noisedX */ -3847569385690.0,
            DEFAULT_L_0_SENSITIVITY,
            (long) DEFAULT_L_INF_SENSITIVITY,
            DEFAULT_EPSILON,
            DEFAULT_DELTA,
            DEFAULT_ALPHA);
    // Double precision should be accurate for abs(noisedX) < 2^54.
    ConfidenceInterval expected =
        ConfidenceInterval.create(-3847569385713.02585093, -3847569385666.97414907);
    isEqual(confInt, expected);
  }

  @Test
  public void computeConfidenceInterval_double_zeroL0Sensitivity_throwsException() {
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
  public void computeConfidenceInterval_double_negativeL0Sensitivity_throwsException() {
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
  public void computeConfidenceInterval_double_zeroLInfSensitivity_throwsException() {
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
  public void computeConfidenceInterval_double_negativeLInfSensitivity_throwsException() {
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
  public void computeConfidenceInterval_double_infiniteLInfSensitivity_throwsException() {
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
  public void computeConfidenceInterval_double_NaNLInfSensitivity_throwsException() {
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
  public void computeConfidenceInterval_double_tooSmallEpsilon_throwsException() {
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
  public void computeConfidenceInterval_double_infiniteEpsilon_throwsException() {
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
  public void computeConfidenceInterval_double_NaNEpsilon_throwsException() {
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
  public void computeConfidenceInterval_double_nonNullDelta_throwsException() {
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
  public void computeConfidenceInterval_double_zeroAlpha_throwsException() {
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
  public void computeConfidenceInterval_double_negativeAlpha_throwsException() {
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
  public void computeConfidenceInterval_double_alphaEqualToOne_throwsException() {
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
  public void computeConfidenceInterval_double_alphaGreaterThanOne_throwsException() {
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
  public void computeConfidenceInterval_double_NaNAlpha_throwsException() {
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
  public void computeConfidenceInterval_long_arbitraryTest() {
    ConfidenceInterval confInt =
        NOISE.computeConfidenceInterval(
            /* noisedX */ 83,
            /* l0Sensitivity */ 3,
            /* lInfSensitivity */ 2,
            DEFAULT_EPSILON,
            DEFAULT_DELTA,
            /* alpha */ 0.24);
    ConfidenceInterval expected = ConfidenceInterval.create(-3, 169);
    isEqual(confInt, expected);
  }

  @Test
  public void computeConfidenceInterval_long_lowConfidenceLevel() {
    ConfidenceInterval confInt =
        NOISE.computeConfidenceInterval(
            (long) DEFAULT_NOISEDX,
            DEFAULT_L_0_SENSITIVITY,
            (long) DEFAULT_L_INF_SENSITIVITY,
            DEFAULT_EPSILON,
            DEFAULT_DELTA,
            /* alpha */ 1 - 3.548957438e-10);
    ConfidenceInterval expected = ConfidenceInterval.create(0, 0);
    isEqual(confInt, expected);
  }

  @Test
  public void computeConfidenceInterval_long_highConfidenceLevel() {
    ConfidenceInterval confInt =
        NOISE.computeConfidenceInterval(
            /* noisedX */ 50L,
            DEFAULT_L_0_SENSITIVITY,
            (long) DEFAULT_L_INF_SENSITIVITY,
            DEFAULT_EPSILON,
            DEFAULT_DELTA,
            /* alpha */ 7.856382354e-10);
    ConfidenceInterval expected = ConfidenceInterval.create(-160, 260);
    isEqual(confInt, expected);
  }

  @Test
  public void computeConfidenceInterval_long_lowEpsilon() {
    ConfidenceInterval confInt =
        NOISE.computeConfidenceInterval(
            (long) DEFAULT_NOISEDX,
            DEFAULT_L_0_SENSITIVITY,
            (long) DEFAULT_L_INF_SENSITIVITY,
            /* epsilon */ 1.567321563235e-10,
            DEFAULT_DELTA,
            DEFAULT_ALPHA);
    ConfidenceInterval expected = ConfidenceInterval.create(-14691210451.0, 14691210451.0);
    isEqual(confInt, expected);
  }

  @Test
  public void computeConfidenceInterval_long_highEpsilon() {
    ConfidenceInterval confInt =
        NOISE.computeConfidenceInterval(
            (long) DEFAULT_NOISEDX,
            DEFAULT_L_0_SENSITIVITY,
            (long) DEFAULT_L_INF_SENSITIVITY,
            /* epsilon */ 1.567321563235e10,
            DEFAULT_DELTA,
            DEFAULT_ALPHA);
    ConfidenceInterval expected = ConfidenceInterval.create(0, 0);
    isEqual(confInt, expected);
  }

  @Test
  public void computeConfidenceInterval_long_highPositiveNoisedX() {
    ConfidenceInterval confInt =
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
        ConfidenceInterval.create(288230376151711712.0, 288230376151711808.0);
    isEqual(confInt, expected);
  }

  @Test
  public void computeConfidenceInterval_long_highNegativeNoisedX() {
    ConfidenceInterval confInt =
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
        ConfidenceInterval.create(-288230376151711808.0, -288230376151711712.0);
    isEqual(confInt, expected);
  }

  @Test
  public void computeConfidenceInterval_long_zeroL0Sensitivity_throwsException() {
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
  public void computeConfidenceInterval_long_negativeL0Sensitivity_throwsException() {
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
  public void computeConfidenceInterval_long_zeroLInfSensitivity_throwsException() {
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
  public void computeConfidenceInterval_long_negativeLInfSensitivity_throwsException() {
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
  public void computeConfidenceInterval_long_tooSmallEpsilon_throwsException() {
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
  public void computeConfidenceInterval_long_infiniteEpsilon_throwsException() {
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
  public void computeConfidenceInterval_long_NaNEpsilon_throwsException() {
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
  public void computeConfidenceInterval_long_nonNullDelta_throwsException() {
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
  public void computeConfidenceInterval_long_zeroAlpha_throwsException() {
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
  public void computeConfidenceInterval_long_negativeAlpha_throwsException() {
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
  public void computeConfidenceInterval_long_alphaEqualToOne_throwsException() {
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
  public void computeConfidenceInterval_long_alphaGreaterThanOne_throwsException() {
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
  public void computeConfidenceInterval_long_NaNAlpha_throwsException() {
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
}

package com.google.privacy.differentialprivacy;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static org.junit.Assert.assertThrows;

import com.google.auto.value.AutoValue;
import org.junit.Test;
import org.junit.experimental.runners.Enclosed;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Collection;

@RunWith(Enclosed.class)
public class GaussianNoiseConfidenceIntervalTest {
  private static final Noise NOISE = new GaussianNoise();
  private static final double TOLERANCE = 1e-7;
  private static final double DEFAULT_EPSILON = 0.5;
  private static final int DEFAULT_L_0_SENSITIVITY = 1;
  private static final double DEFAULT_L_INF_SENSITIVITY = 1.0;
  private static final double DEFAULT_DELTA = 0.3;
  private static final double DEFAULT_NOISED_X = 70;
  private static final double DEFAULT_ALPHA = 0.1;

  @AutoValue
  public abstract static class ConfidenceIntervalTestCase implements Serializable {
    static ConfidenceIntervalTestCase create(
        double noisedX,
        int l0Sensitivity,
        double lInfSensitivity,
        double epsilon,
        Double delta,
        double alpha,
        ConfidenceInterval expected) {
      return new AutoValue_GaussianNoiseConfidenceIntervalTest_ConfidenceIntervalTestCase(
          noisedX, l0Sensitivity, lInfSensitivity, epsilon, delta, alpha, expected);
    }

    abstract double noisedX();

    abstract int l0Sensitivity();

    abstract double lInfSensitivity();

    abstract double epsilon();

    abstract Double delta();

    abstract double alpha();

    abstract ConfidenceInterval expected();
  }

  @RunWith(Parameterized.class)
  public static class computeConfidenceIntervalDoubleTests {
    private final ConfidenceIntervalTestCase testCase;

    public computeConfidenceIntervalDoubleTests(ConfidenceIntervalTestCase testCase) {
      this.testCase = testCase;
    }

    @Parameterized.Parameters(name = "{index}: = failed")
    public static Collection<Object> getTestCases() {
      return Arrays.asList(
          // Arbitrary Tests.
          ConfidenceIntervalTestCase.create(
              /* noisedX */ 70.00,
              /* l0Sensitivity*/ 5,
              /* lInfSensitivity */ 36,
              /* epsilon */ 0.8,
              /* delta */ 0.8,
              /* alpha */ 0.2,
              ConfidenceInterval.create(35.26815080641682, 104.73184919358317)),
          ConfidenceIntervalTestCase.create(
              /* noisedX */ 699.2402199905,
              /* l0Sensitivity*/ 1,
              /* lInfSensitivity */ 5,
              /* epsilon */ 0.333,
              /* delta */ 0.9,
              /* alpha */ 0.001256458,
              ConfidenceInterval.create(694.5583238637953, 703.9221161172047)),
          // High alpha.
          ConfidenceIntervalTestCase.create(
              /* noisedX */ 70.00,
              /* l0Sensitivity*/ 5,
              /* lInfSensitivity */ 36,
              /* epsilon */ 0.8,
              /* delta */ 0.8,
              /* alpha */ 1 - 7.856382354e-10,
              ConfidenceInterval.create(69.9999999733, 70.0000000267)),
          // Low alpha.
          ConfidenceIntervalTestCase.create(
              DEFAULT_NOISED_X,
              /* l0Sensitivity*/ 5,
              /* lInfSensitivity */ 36,
              /* epsilon */ 0.8,
              /* delta */ 0.8,
              /* alpha */ 7.856382354e-10,
              ConfidenceInterval.create(-96.6140883158, 236.6140883158)),
          // High noisedX.
          ConfidenceIntervalTestCase.create(
              9.58655474558e10,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              DEFAULT_ALPHA,
              ConfidenceInterval.create(95865547454.4, 95865547457.2)),
          // Low noisedX.
          ConfidenceIntervalTestCase.create(
              1.58655474558e-12,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              DEFAULT_ALPHA,
              ConfidenceInterval.create(-1.42479020225, 1.42479020225)),
          // Negative noisedX.
          ConfidenceIntervalTestCase.create(
              -1.58655474558,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              DEFAULT_ALPHA,
              ConfidenceInterval.create(
                  -3.0113449478319105345747175, -0.1617645433280894273764261)),
          // Low delta.
          ConfidenceIntervalTestCase.create(
              DEFAULT_NOISED_X,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              /* delta */ 1.78468549878e-10,
              DEFAULT_ALPHA,
              ConfidenceInterval.create(
                  51.4953966967959289036116388, 88.5046033032040782018157188)),
          // High delta.
          ConfidenceIntervalTestCase.create(
              DEFAULT_NOISED_X,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              /* delta */ 1 - 1.78468549878e-10,
              DEFAULT_ALPHA,
              ConfidenceInterval.create(
                  69.8717969920888037904660450, 70.1282030079111962095339550)),
          // Low epsilon.
          ConfidenceIntervalTestCase.create(
              DEFAULT_NOISED_X,
              DEFAULT_L_0_SENSITIVITY,
              /* lInfSensitivity */ 2,
              /* epsilon */ 1.65463453425e-10,
              DEFAULT_DELTA,
              DEFAULT_ALPHA,
              ConfidenceInterval.create(65.73044830035448, 74.26955169964552)),
          // High epsilon.
          ConfidenceIntervalTestCase.create(
              DEFAULT_NOISED_X,
              DEFAULT_L_0_SENSITIVITY,
              /* lInfSensitivity */ 2,
              /* epsilon */ 1.65463453425e10,
              DEFAULT_DELTA,
              DEFAULT_ALPHA,
              ConfidenceInterval.create(70, 70)));
    }

    @Test
    public void computeConfidenceInterval_hasAccurateResultsForDouble() {
      ConfidenceInterval confInt =
          NOISE.computeConfidenceInterval(
              testCase.noisedX(),
              testCase.l0Sensitivity(),
              testCase.lInfSensitivity(),
              testCase.epsilon(),
              testCase.delta(),
              testCase.alpha());
      double mxMagnitudeLower =
          Math.max(Math.abs(confInt.lowerBound()), Math.abs(testCase.expected().lowerBound()));
      assertWithMessage("Lower bounds are not equal")
          .that(Math.abs(confInt.lowerBound() - testCase.expected().lowerBound()))
          .isAtMost(TOLERANCE * mxMagnitudeLower);
      double mxMagnitudeUpper =
          Math.max(Math.abs(confInt.upperBound()), Math.abs(testCase.expected().upperBound()));
      assertWithMessage("Upper bounds are not equal")
          .that(Math.abs(confInt.upperBound() - testCase.expected().upperBound()))
          .isAtMost(TOLERANCE * mxMagnitudeUpper);
    }
  }

  @RunWith(Parameterized.class)
  public static class computeConfidenceIntervalLongTests {
    private final ConfidenceIntervalTestCase testCase;

    public computeConfidenceIntervalLongTests(ConfidenceIntervalTestCase testCase) {
      this.testCase = testCase;
    }

    @Parameterized.Parameters(name = "{index}: = failed")
    public static Collection<Object> getTestCases() {
      return Arrays.asList(
          // Arbitrary Tests.
          ConfidenceIntervalTestCase.create(
              DEFAULT_NOISED_X,
              /* l0Sensitivity*/ 5,
              /* lInfSensitivity */ 36,
              /* epsilon */ 0.8,
              /* delta */ 0.8,
              /* alpha */ 0.2,
              ConfidenceInterval.create(35, 105)),
          ConfidenceIntervalTestCase.create(
              /* noisedX */ 700,
              /* l0Sensitivity*/ 1,
              /* lInfSensitivity */ 5,
              /* epsilon */ 0.333,
              /* delta */ 0.9,
              /* alpha */ 0.001256458,
              ConfidenceInterval.create(695, 705)),
          // High alpha.
          ConfidenceIntervalTestCase.create(
              DEFAULT_NOISED_X,
              /* l0Sensitivity*/ 5,
              /* lInfSensitivity */ 36,
              /* epsilon */ 0.8,
              /* delta */ 0.8,
              /* alpha */ 1 - 7.856382354e-10,
              ConfidenceInterval.create(70, 70)),
          // Low alpha.
          ConfidenceIntervalTestCase.create(
              DEFAULT_NOISED_X,
              /* l0Sensitivity*/ 5,
              /* lInfSensitivity */ 36,
              /* epsilon */ 0.8,
              /* delta */ 0.8,
              /* alpha */ 7.856382354e-10,
              ConfidenceInterval.create(-97, 237)),
          ConfidenceIntervalTestCase.create(
              DEFAULT_NOISED_X,
              /* l0Sensitivity*/ 5,
              /* lInfSensitivity */ 36,
              /* epsilon */ 0.8,
              /* delta */ 0.8,
              /* alpha */ 7.856382354e-10,
              ConfidenceInterval.create(-97, 237)),
          // High noisedX.
          ConfidenceIntervalTestCase.create(
              /* noisedX */ 9e9,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              DEFAULT_ALPHA,
              ConfidenceInterval.create(8.999999999E9, 9.000000001E9)),
          // Low noisedX.
          ConfidenceIntervalTestCase.create(
              /* noisedX */ 0,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              DEFAULT_ALPHA,
              ConfidenceInterval.create(-1, 1)),
          // Negative noisedX.
          ConfidenceIntervalTestCase.create(
              -10,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              DEFAULT_ALPHA,
              ConfidenceInterval.create(-11, -9)),
          // Low delta.
          ConfidenceIntervalTestCase.create(
              DEFAULT_NOISED_X,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              /* delta */ 1.78468549878e-10,
              DEFAULT_ALPHA,
              ConfidenceInterval.create(51, 89)),
          // High delta.
          ConfidenceIntervalTestCase.create(
              DEFAULT_NOISED_X,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              /* delta */ 1 - 1.78468549878e-10,
              DEFAULT_ALPHA,
              ConfidenceInterval.create(70, 70)),
          // Low epsilon.
          ConfidenceIntervalTestCase.create(
              DEFAULT_NOISED_X,
              DEFAULT_L_0_SENSITIVITY,
              /* lInfSensitivity */ 2,
              /* epsilon */ 1.65463453425e-10,
              DEFAULT_DELTA,
              DEFAULT_ALPHA,
              ConfidenceInterval.create(66, 74)),
          // High epsilon.
          ConfidenceIntervalTestCase.create(
              DEFAULT_NOISED_X,
              DEFAULT_L_0_SENSITIVITY,
              /* lInfSensitivity */ 2,
              /* epsilon */ 1.65463453425e10,
              DEFAULT_DELTA,
              DEFAULT_ALPHA,
              ConfidenceInterval.create(70, 70)));
    }

    @Test
    public void computeConfidenceInterval_hasAccurateResultsForLong() {
      ConfidenceInterval confInt =
          NOISE.computeConfidenceInterval(
              (long) testCase.noisedX(),
              testCase.l0Sensitivity(),
              (long) testCase.lInfSensitivity(),
              testCase.epsilon(),
              testCase.delta(),
              testCase.alpha());
      assertWithMessage("Lower bound is not equal")
          .that(confInt.lowerBound())
          .isEqualTo(testCase.expected().lowerBound());
      assertWithMessage("Upper bound is not equal")
          .that(confInt.upperBound())
          .isEqualTo(testCase.expected().upperBound());
    }
  }

  @RunWith(Parameterized.class)
  public static class ArgumentCheckingConfidenceIntervalDoubleTests {
    private final ConfidenceIntervalTestCase testCase;

    public ArgumentCheckingConfidenceIntervalDoubleTests(ConfidenceIntervalTestCase testCase) {
      this.testCase = testCase;
    }

    @Parameterized.Parameters(name = "{index}: = failed")
    public static Collection<Object> getTestCases() {
      return Arrays.asList(
          // Zero l0Sensitivity
          ConfidenceIntervalTestCase.create(
              /* noisedX */ 0,
              /* l0Sensitivity*/ 0,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              /* alpha */ 0.5,
              ConfidenceInterval.create(0, 0)),
          // Negative l0Sensitivity
          ConfidenceIntervalTestCase.create(
              /* noisedX */ 0,
              /* l0Sensitivity*/ -1,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              /* alpha */ 0.5,
              ConfidenceInterval.create(0, 0)),
          // Zero lInfSensitivity
          ConfidenceIntervalTestCase.create(
              /* noisedX */ 0,
              DEFAULT_L_0_SENSITIVITY,
              /* lInfSensitivity*/ 0,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              /* alpha*/ 0.5,
              ConfidenceInterval.create(0, 0)),
          // Negative lInfSensitivity
          ConfidenceIntervalTestCase.create(
              /* noisedX */ 0,
              DEFAULT_L_0_SENSITIVITY,
              /* lInfSensitivity*/ -1,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              /* alpha*/ 0.5,
              ConfidenceInterval.create(0, 0)),
          // Infinite lInfSensitivity
          ConfidenceIntervalTestCase.create(
              /* noisedX */ 0,
              DEFAULT_L_0_SENSITIVITY,
              /* lInfSensitivity*/ Double.POSITIVE_INFINITY,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              /* alpha*/ 0.5,
              ConfidenceInterval.create(0, 0)),
          // NaN lInfSensitivity
          ConfidenceIntervalTestCase.create(
              /* noisedX */ 0,
              DEFAULT_L_0_SENSITIVITY,
              /* lInfSensitivity*/ Double.NaN,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              /* alpha*/ 0.5,
              ConfidenceInterval.create(0, 0)),
          // Infinite epsilon
          ConfidenceIntervalTestCase.create(
              /* noisedX */ 0,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              /* epsilon*/ Double.POSITIVE_INFINITY,
              DEFAULT_DELTA,
              /* alpha*/ 0.5,
              ConfidenceInterval.create(0, 0)),
          // Very small epsilon (less than 2^-50)
          ConfidenceIntervalTestCase.create(
              /* noisedX */ 0,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              /* epsilon*/ Math.exp(-51),
              DEFAULT_DELTA,
              /* alpha*/ 0,
              ConfidenceInterval.create(0, 0)),
          // NaN epsilon
          ConfidenceIntervalTestCase.create(
              /* noisedX */ 0,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              /* epsilon*/ Double.NaN,
              DEFAULT_DELTA,
              /* alpha*/ 0,
              ConfidenceInterval.create(0, 0)),
          // Negative confidence level
          ConfidenceIntervalTestCase.create(
              /* noisedX */ 0,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              /* alpha */ -1,
              ConfidenceInterval.create(0, 0)),
          // Greater than 1 confidence level
          ConfidenceIntervalTestCase.create(
              /* noisedX */ 0,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              /* alpha */ 2,
              ConfidenceInterval.create(0, 0)),
          // NaN confidence level
          ConfidenceIntervalTestCase.create(
              /* noisedX */ 0,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              /* alpha */ Double.NaN,
              ConfidenceInterval.create(0, 0)));
    }

    @Test
    public void argumentCheckingConfidenceIntervalDouble() {
      assertThrows(
          IllegalArgumentException.class,
          () ->
              NOISE.computeConfidenceInterval(
                  testCase.noisedX(),
                  testCase.l0Sensitivity(),
                  testCase.lInfSensitivity(),
                  testCase.epsilon(),
                  testCase.delta(),
                  testCase.alpha()));
    }
  }

  @RunWith(Parameterized.class)
  public static class ArgumentCheckingConfidenceIntervalIntTests {
    private final ConfidenceIntervalTestCase testCase;

    public ArgumentCheckingConfidenceIntervalIntTests(ConfidenceIntervalTestCase testCase) {
      this.testCase = testCase;
    }

    @Parameterized.Parameters(name = "{index}: = failed")
    public static Collection<Object> getTestCases() {
      return Arrays.asList(
          // Zero l0Sensitivity
          ConfidenceIntervalTestCase.create(
              /* noisedX */ 0,
              /* l0Sensitivity */ 0,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              /* alpha */ 0.5,
              ConfidenceInterval.create(0, 0)),
          // Negative l0Sensitivity
          ConfidenceIntervalTestCase.create(
              /* noisedX */ 0,
              /* l0Sensitivity */ -1,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              /* alpha */ 0.5,
              ConfidenceInterval.create(0, 0)),
          // Zero lInfSensitivity
          ConfidenceIntervalTestCase.create(
              /* noisedX */ 0,
              DEFAULT_L_0_SENSITIVITY,
              /* lInfSensitivity */ 0,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              /* alpha */ 0.5,
              ConfidenceInterval.create(0, 0)),
          // Negative lInfSensitivity
          ConfidenceIntervalTestCase.create(
              /* noisedX */ 0,
              DEFAULT_L_0_SENSITIVITY,
              /* lInfSensitivity */ -1,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              /* alpha */ 0.5,
              ConfidenceInterval.create(0, 0)),
          // Infinite lInfSensitivity
          ConfidenceIntervalTestCase.create(
              /* noisedX */ 0,
              DEFAULT_L_0_SENSITIVITY,
              /* lInfSensitivity */ Double.POSITIVE_INFINITY,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              /* alpha */ 0.5,
              ConfidenceInterval.create(0, 0)),
          // NaN lInfSensitivity
          ConfidenceIntervalTestCase.create(
              /* noisedX */ 0,
              DEFAULT_L_0_SENSITIVITY,
              /* lInfSensitivity */ Double.NaN,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              /* alpha */ 0.5,
              ConfidenceInterval.create(0, 0)),
          // Infinite epsilon
          ConfidenceIntervalTestCase.create(
              /* noisedX */ 0,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              /* epsilon */ Double.POSITIVE_INFINITY,
              DEFAULT_DELTA,
              /* alpha */ 0.5,
              ConfidenceInterval.create(0, 0)),
          // Very small epsilon (less than 2^-50)
          ConfidenceIntervalTestCase.create(
              /* noisedX */ 0,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              /* epsilon */ Math.exp(-51),
              DEFAULT_DELTA,
              /* alpha */ 0,
              ConfidenceInterval.create(0, 0)),
          // NaN epsilon
          ConfidenceIntervalTestCase.create(
              /* noisedX */ 0,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              /* epsilon */ Double.NaN,
              DEFAULT_DELTA,
              /* alpha */ 0,
              ConfidenceInterval.create(0, 0)),
          // Negative confidence level
          ConfidenceIntervalTestCase.create(
              /* noisedX */ 0,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              /* alpha */ -1,
              ConfidenceInterval.create(0, 0)),
          // Greater than 1 confidence level
          ConfidenceIntervalTestCase.create(
              /* noisedX */ 0,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              /* alpha */ 2,
              ConfidenceInterval.create(0, 0)),
          // NaN confidence level
          ConfidenceIntervalTestCase.create(
              /* noisedX */ 0,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              /* alpha */ Double.NaN,
              ConfidenceInterval.create(0, 0)));
    }

    @Test
    public void argumentCheckingConfidenceIntervalInt() {
      assertThrows(
          IllegalArgumentException.class,
          () ->
              NOISE.computeConfidenceInterval(
                  (long) testCase.noisedX(),
                  testCase.l0Sensitivity(),
                  testCase.lInfSensitivity(),
                  testCase.epsilon(),
                  testCase.delta(),
                  testCase.alpha()));
    }
  }
}
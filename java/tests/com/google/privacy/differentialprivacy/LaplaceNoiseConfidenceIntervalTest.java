package com.google.privacy.differentialprivacy;

import static com.google.common.truth.Truth.assertWithMessage;
import static org.junit.Assert.assertThrows;

import com.google.auto.value.AutoValue;
import org.junit.Test;
import org.junit.experimental.runners.Enclosed;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import javax.annotation.Nullable;
import java.io.Serializable;
import java.util.Arrays;
import java.util.Collection;

@RunWith(Enclosed.class)
public class LaplaceNoiseConfidenceIntervalTest {
  private static final Noise NOISE = new LaplaceNoise();
  private static final double TOLERANCE = 1e-6;
  private static final double DEFAULT_NOISEDX = 0.0;
  private static final int DEFAULT_L_0_SENSITIVITY = 1;
  private static final double DEFAULT_L_INF_SENSITIVITY = 1.0;
  private static final double DEFAULT_EPSILON = 0.1;
  private static final Double DEFAULT_DELTA = null;
  private static final double DEFAULT_ALPHA = 0.1;

  @AutoValue
  public abstract static class ConfidenceIntervalTestCase implements Serializable {
    static ConfidenceIntervalTestCase create(
        double noisedValue,
        int l0Sensitivity,
        double lInfSensitivity,
        double epsilon,
        @Nullable Double delta,
        double alpha,
        @Nullable ConfidenceInterval expected) {
      return new AutoValue_LaplaceNoiseConfidenceIntervalTest_ConfidenceIntervalTestCase(
          noisedValue, l0Sensitivity, lInfSensitivity, epsilon, delta, alpha, expected);
    }

    abstract double noisedValue();

    abstract int l0Sensitivity();

    abstract double lInfSensitivity();

    abstract double epsilon();

    abstract @Nullable Double delta();

    abstract double alpha();

    abstract @Nullable ConfidenceInterval expected();
  }

  @RunWith(Parameterized.class)
  public static class ComputeConfidenceIntervalDoubleTests {
    private final ConfidenceIntervalTestCase testCase;

    public ComputeConfidenceIntervalDoubleTests(ConfidenceIntervalTestCase testCase) {
      this.testCase = testCase;
    }

    @Test
    public void computeConfidenceIntervalDouble() {
      ConfidenceInterval confInt =
          NOISE.computeConfidenceInterval(
              testCase.noisedValue(),
              testCase.l0Sensitivity(),
              testCase.lInfSensitivity(),
              testCase.epsilon(),
              testCase.delta(),
              testCase.alpha());
      double mxMagnitudeLower =
          Math.max(Math.abs(confInt.lowerBound()), Math.abs(testCase.expected().lowerBound()));
      assertWithMessage("Lower bounds are not equal.")
          .that(Math.abs(confInt.lowerBound() - testCase.expected().lowerBound()))
          .isAtMost(TOLERANCE * mxMagnitudeLower);
      double mxMagnitudeUpper =
          Math.max(Math.abs(confInt.upperBound()), Math.abs(testCase.expected().upperBound()));
      assertWithMessage("Upper bounds are not equal.")
          .that(Math.abs(confInt.upperBound() - testCase.expected().upperBound()))
          .isAtMost(TOLERANCE * mxMagnitudeUpper);
    }

    @Parameterized.Parameters(name = "{index}: = failed")
    public static Collection<Object> getTestCases() {
      return Arrays.asList(
          // Arbitrary Tests
          ConfidenceIntervalTestCase.create(
              /* noisedX */ 13,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              /* epsilon */ 0.3,
              DEFAULT_DELTA,
              /* alpha */ 0.05,
              ConfidenceInterval.create(3.014225755, 22.98577425)),
          ConfidenceIntervalTestCase.create(
              /* noisedX */ 83.1235,
              /* l0Sensitivity */ 3,
              /* lInfSensitivity */ 2,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              /* alpha */ 0.24,
              ConfidenceInterval.create(-2.503481338, 168.7504813)),
          ConfidenceIntervalTestCase.create(
              /* noisedX */ 5,
              /* l0Sensitivity */ 10,
              /* lInfSensitivity */ 2,
              /* epsilon */ 3.0,
              DEFAULT_DELTA,
              /* alpha */ 0.6,
              ConfidenceInterval.create(1.594495842, 8.405504158)),
          ConfidenceIntervalTestCase.create(
              /* noisedX */ 65.4621,
              /* l0Sensitivity */ 7,
              /* lInfSensitivity */ 10,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              /* alpha */ 0.8,
              ConfidenceInterval.create(-90.73838592, 221.6625859)),
          // Extremely low confidence level
          ConfidenceIntervalTestCase.create(
              DEFAULT_NOISEDX,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              /* alpha */ 1 - 3.548957438e-10,
              ConfidenceInterval.create(-3.548957437370245055312e-9, 3.548957437370245055312e-9)),
          // Extremely high confidence level
          ConfidenceIntervalTestCase.create(
              /* noisedX */ 50,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              /* alpha */ 7.856382354e-10,
              ConfidenceInterval.create(-159.6452468975697118041, 259.6452468975697118041)),
          // Extremely low epsilon
          ConfidenceIntervalTestCase.create(
              DEFAULT_NOISEDX,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              /* epsilon */ 1.567321563235e-10,
              DEFAULT_DELTA,
              DEFAULT_ALPHA,
              ConfidenceInterval.create(-14691210451.04132, 14691210451.04132)),
          // Extremely high epsilon
          ConfidenceIntervalTestCase.create(
              DEFAULT_NOISEDX,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              /* epsilon */ 1.567321563235e10,
              DEFAULT_DELTA,
              DEFAULT_ALPHA,
              ConfidenceInterval.create(-1.46912104510413237E-10, 1.46912104510413237E-10)));
    }
  }

  @RunWith(Parameterized.class)
  public static class ComputeConfidenceIntervalIntTests {
    private final ConfidenceIntervalTestCase testCase;

    public ComputeConfidenceIntervalIntTests(ConfidenceIntervalTestCase testCase) {
      this.testCase = testCase;
    }

    @Test
    public void computeConfidenceIntervalInt() {
      ConfidenceInterval confInt =
          NOISE.computeConfidenceInterval(
              (long) testCase.noisedValue(),
              testCase.l0Sensitivity(),
              (long) testCase.lInfSensitivity(),
              testCase.epsilon(),
              testCase.delta(),
              testCase.alpha());
      assertWithMessage("Lower bounds are not equal.")
          .that(confInt.lowerBound())
          .isEqualTo(testCase.expected().lowerBound());
      assertWithMessage("Upper bounds are not equal.")
          .that(confInt.upperBound())
          .isEqualTo(testCase.expected().upperBound());
    }

    @Parameterized.Parameters(name = "{index}: = failed")
    public static Collection<Object> getTestCases() {
      return Arrays.asList(
          // Arbitrary tests
          ConfidenceIntervalTestCase.create(
              /* noisedX */ 13,
              /* l0Sensitivity */ 1,
              /* lInfSensitivity */ 1,
              /* epsilon */ 0.3,
              DEFAULT_DELTA,
              /* alpha */ 0.05,
              ConfidenceInterval.create(3, 23)),
          ConfidenceIntervalTestCase.create(
              /* noisedX */ 83,
              /* l0Sensitivity */ 3,
              /* lInfSensitivity */ 2,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              /* alpha */ 0.24,
              ConfidenceInterval.create(-3, 169)),
          ConfidenceIntervalTestCase.create(
              /* noisedX */ 5,
              /* l0Sensitivity */ 10,
              /* lInfSensitivity */ 2,
              /* epsilon */ 3,
              DEFAULT_DELTA,
              /* alpha */ 0.6,
              ConfidenceInterval.create(2, 8)),
          ConfidenceIntervalTestCase.create(
              /* noisedX */ 65,
              /* l0Sensitivity */ 7,
              /* lInfSensitivity */ 10,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              /* alpha */ 0.8,
              ConfidenceInterval.create(-91, 221)),
          // Extremely low confidence level
          ConfidenceIntervalTestCase.create(
              DEFAULT_NOISEDX,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              /* alpha */ 1 - 3.548957438e-10,
              ConfidenceInterval.create(0, 0)),
          // Extremely high confidence level
          ConfidenceIntervalTestCase.create(
              /* noisedX */ 50,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              /* alpha */ 7.856382354e-10,
              ConfidenceInterval.create(-160, 260)),
          // Extremely low epsilon
          ConfidenceIntervalTestCase.create(
              DEFAULT_NOISEDX,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              /* epsilon */ 1.567321563235e-10,
              DEFAULT_DELTA,
              DEFAULT_ALPHA,
              ConfidenceInterval.create(-14691210451.0, 14691210451.0)),
          // Extremely high epsilon
          ConfidenceIntervalTestCase.create(
              DEFAULT_NOISEDX,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              /* epsilon */ 1.567321563235e10,
              DEFAULT_DELTA,
              DEFAULT_ALPHA,
              ConfidenceInterval.create(0, 0)),
          // Extremely high positive noisedX
          ConfidenceIntervalTestCase.create(
              /* noisedX */ (1L << 58),
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              DEFAULT_ALPHA,
              /* Z value = -23, upperBound = 1 << 58 + 23 = 288230376151711767 gets rounded
               * down to 288230376151711744, nextLargeDouble rounds it back up to 288230376151711808.
               * lowerBound = 1 << 58 - 23 = 288230376151711721 gets rounded down to 288230376151711712,
               * nextSmaller returns the same value.*/
              ConfidenceInterval.create(288230376151711712.0, 288230376151711808.0)),
          // Extremely high negative noisedX
          ConfidenceIntervalTestCase.create(
              /* noisedX */ -(1L << 58),
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              DEFAULT_ALPHA,
              /* Z value = -23, upperBound = -(1 << 58) + 23 = -288230376151711721 gets rounded
               * up to -288230376151711712, nextLargeDouble returns the same value.
               * lowerBound = -(1 << 58) - 23 = -288230376151711767 gets rounded up to -288230376151711744,
               * nextSmaller rounds it back down to -288230376151711808. */
              ConfidenceInterval.create(-288230376151711808.0, -288230376151711712.0)));
    }
  }

  @RunWith(Parameterized.class)
  public static class ArgumentCheckingConfidenceIntervalDoubleTests {
    private final ConfidenceIntervalTestCase testCase;

    public ArgumentCheckingConfidenceIntervalDoubleTests(ConfidenceIntervalTestCase testCase) {
      this.testCase = testCase;
    }

    @Test
    public void argumentCheckingConfidenceIntervalDouble() {
      assertThrows(
          IllegalArgumentException.class,
          () ->
              NOISE.computeConfidenceInterval(
                  testCase.noisedValue(),
                  testCase.l0Sensitivity(),
                  testCase.lInfSensitivity(),
                  testCase.epsilon(),
                  testCase.delta(),
                  testCase.alpha()));
    }

    @Parameterized.Parameters(name = "{index}: = failed")
    public static Collection<Object> getTestCases() {
      return Arrays.asList(
          // Zero l0Sensitivity
          ConfidenceIntervalTestCase.create(
              DEFAULT_NOISEDX,
              /* l0Sensitivity */ 0,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              DEFAULT_ALPHA,
              null),
          // Negative l0Sensitivity
          ConfidenceIntervalTestCase.create(
              DEFAULT_NOISEDX,
              /* l0Sensitivity */ -1,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              DEFAULT_ALPHA,
              null),
          // Zero lInfSensitivity
          ConfidenceIntervalTestCase.create(
              DEFAULT_NOISEDX,
              DEFAULT_L_0_SENSITIVITY,
              /* lInfSensitivity */ 0,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              DEFAULT_ALPHA,
              null),
          // Negative lInfSensitivity
          ConfidenceIntervalTestCase.create(
              DEFAULT_NOISEDX,
              DEFAULT_L_0_SENSITIVITY,
              /* lInfSensitivity */ -1,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              DEFAULT_ALPHA,
              null),
          // Infinite lInfSensitivity
          ConfidenceIntervalTestCase.create(
              DEFAULT_NOISEDX,
              DEFAULT_L_0_SENSITIVITY,
              /* lInfSensitivity */ Double.POSITIVE_INFINITY,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              DEFAULT_ALPHA,
              null),
          // NaN lInfSensitivity
          ConfidenceIntervalTestCase.create(
              DEFAULT_NOISEDX,
              DEFAULT_L_0_SENSITIVITY,
              /* lInfSensitivity */ Double.NaN,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              DEFAULT_ALPHA,
              null),
          //  Very small epsilon (less than 2^-50)
          ConfidenceIntervalTestCase.create(
              DEFAULT_NOISEDX,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              /* epsilon */ 1.0 / (1L << 51),
              DEFAULT_DELTA,
              DEFAULT_ALPHA,
              null),
          // Infinite epsilon
          ConfidenceIntervalTestCase.create(
              DEFAULT_NOISEDX,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              /* epsilon */ Double.POSITIVE_INFINITY,
              DEFAULT_DELTA,
              DEFAULT_ALPHA,
              null),
          // NaN epsilon
          ConfidenceIntervalTestCase.create(
              DEFAULT_NOISEDX,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              /* epsilon */ Double.NaN,
              DEFAULT_DELTA,
              DEFAULT_ALPHA,
              null),
          // Non-null delta
          ConfidenceIntervalTestCase.create(
              DEFAULT_NOISEDX,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              /* delta */ 1.0,
              DEFAULT_ALPHA,
              null),
          // Zero alpha
          ConfidenceIntervalTestCase.create(
              DEFAULT_NOISEDX,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              /* alpha */ 0,
              null),
          // Negative alpha
          ConfidenceIntervalTestCase.create(
              DEFAULT_NOISEDX,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              /* alpha */ -1,
              null),
          // 1 alpha
          ConfidenceIntervalTestCase.create(
              DEFAULT_NOISEDX,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              /* alpha */ 1,
              null),
          // Greater than 1 alpha
          ConfidenceIntervalTestCase.create(
              DEFAULT_NOISEDX,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              /* alpha */ 2,
              null),
          // NaN alpha
          ConfidenceIntervalTestCase.create(
              DEFAULT_NOISEDX,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              /* alpha */ Double.NaN,
              null));
    }
  }

  @RunWith(Parameterized.class)
  public static class ArgumentCheckingConfidenceIntervalIntTests {
    private final ConfidenceIntervalTestCase testCase;

    public ArgumentCheckingConfidenceIntervalIntTests(ConfidenceIntervalTestCase testCase) {
      this.testCase = testCase;
    }

    @Test
    public void argumentCheckingConfidenceIntervalInt() {
      assertThrows(
          IllegalArgumentException.class,
          () ->
              NOISE.computeConfidenceInterval(
                  (int) testCase.noisedValue(),
                  testCase.l0Sensitivity(),
                  (long) testCase.lInfSensitivity(),
                  testCase.epsilon(),
                  testCase.delta(),
                  testCase.alpha()));
    }

    @Parameterized.Parameters(name = "{index}: = failed")
    public static Collection<Object> getTestCases() {
      return Arrays.asList(
          // Zero l0Sensitivity
          ConfidenceIntervalTestCase.create(
              DEFAULT_NOISEDX,
              /* l0Sensitivity */ 0,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              DEFAULT_ALPHA,
              null),
          // Negative l0Sensitivity
          ConfidenceIntervalTestCase.create(
              DEFAULT_NOISEDX,
              /* l0Sensitivity */ -1,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              DEFAULT_ALPHA,
              null),
          // Zero lInfSensitivity
          ConfidenceIntervalTestCase.create(
              DEFAULT_NOISEDX,
              DEFAULT_L_0_SENSITIVITY,
              /* lInfSensitivity */ 0,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              DEFAULT_ALPHA,
              null),
          // Negative lInfSensitivity
          ConfidenceIntervalTestCase.create(
              DEFAULT_NOISEDX,
              DEFAULT_L_0_SENSITIVITY,
              /* lInfSensitivity */ -1,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              DEFAULT_ALPHA,
              null),
          //  Very small epsilon (less than 2^-50)
          ConfidenceIntervalTestCase.create(
              DEFAULT_NOISEDX,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              /* epsilon */ 1.0 / (1L << 51),
              DEFAULT_DELTA,
              DEFAULT_ALPHA,
              null),
          // Infinite epsilon
          ConfidenceIntervalTestCase.create(
              DEFAULT_NOISEDX,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              /* epsilon */ Double.POSITIVE_INFINITY,
              DEFAULT_DELTA,
              DEFAULT_ALPHA,
              null),
          // NaN epsilon
          ConfidenceIntervalTestCase.create(
              DEFAULT_NOISEDX,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              /* epsilon */ Double.NaN,
              DEFAULT_DELTA,
              DEFAULT_ALPHA,
              null),
          // Non-null delta
          ConfidenceIntervalTestCase.create(
              DEFAULT_NOISEDX,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              /* delta */ 1.0,
              DEFAULT_ALPHA,
              null),
          // Zero alpha
          ConfidenceIntervalTestCase.create(
              DEFAULT_NOISEDX,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              /* alpha */ 0,
              null),
          // Negative alpha
          ConfidenceIntervalTestCase.create(
              DEFAULT_NOISEDX,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              /* alpha */ -1,
              null),
          // 1 alpha
          ConfidenceIntervalTestCase.create(
              DEFAULT_NOISEDX,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              /* alpha */ 1,
              null),
          // Greater than 1 alpha
          ConfidenceIntervalTestCase.create(
              DEFAULT_NOISEDX,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              /* alpha */ 2,
              null),
          // NaN alpha
          ConfidenceIntervalTestCase.create(
              DEFAULT_NOISEDX,
              DEFAULT_L_0_SENSITIVITY,
              DEFAULT_L_INF_SENSITIVITY,
              DEFAULT_EPSILON,
              DEFAULT_DELTA,
              /* alpha */ Double.NaN,
              null));
    }
  }
}

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
public class ConfidenceIntervalLaplaceTest {
  private static final Noise lap = new LaplaceNoise();
  private static final double TOLERANCE = 1e-6;

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
      return new AutoValue_ConfidenceIntervalLaplaceTest_ConfidenceIntervalTestCase(
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
    public void ComputeConfidenceIntervalDouble() {
      ConfidenceInterval confInt =
          lap.computeConfidenceInterval(
              testCase.noisedValue(),
              testCase.l0Sensitivity(),
              testCase.lInfSensitivity(),
              testCase.epsilon(),
              testCase.delta(),
              testCase.alpha());
      assertWithMessage("Lower bounds are not equal.")
          .that(confInt.lowerBound())
          .isWithin(TOLERANCE)
          .of(testCase.expected().lowerBound());
      assertWithMessage("Upper bounds are not equal.")
          .that(confInt.upperBound())
          .isWithin(TOLERANCE)
          .of(testCase.expected().upperBound());
    }

    @Parameterized.Parameters(name = "{index}: = failed")
    public static Collection<Object> getTestCases() {
      return Arrays.asList(
          // Random Tests
          ConfidenceIntervalTestCase.create(
              13, 1, 1, 0.3, null, 0.05, ConfidenceInterval.create(3.014225755, 22.98577425)),
          ConfidenceIntervalTestCase.create(
              83.1235, 3, 2, 0.1, null, 0.24, ConfidenceInterval.create(-2.503481338, 168.7504813)),
          ConfidenceIntervalTestCase.create(
              5, 10, 2, 3, null, 0.6, ConfidenceInterval.create(1.594495842, 8.405504158)),
          ConfidenceIntervalTestCase.create(
              65.4621, 7, 10, 0.1, null, 0.8, ConfidenceInterval.create(-90.73838592, 221.6625859)),
          // Extremely low confidence level
          ConfidenceIntervalTestCase.create(
              0,
              1,
              1,
              0.1,
              null,
              1 - 3.548957438e-10,
              ConfidenceInterval.create(-3.548957437370245055312e-9, 3.548957437370245055312e-9)),
          // Extremely high confidence level
          ConfidenceIntervalTestCase.create(
              50,
              1,
              1,
              0.1,
              null,
              7.856382354e-10,
              ConfidenceInterval.create(-159.6452468975697118041, 259.6452468975697118041)));
    }
  }

  @RunWith(Parameterized.class)
  public static class ComputeConfidenceIntervalIntTests {
    private final ConfidenceIntervalTestCase testCase;

    public ComputeConfidenceIntervalIntTests(ConfidenceIntervalTestCase testCase) {
      this.testCase = testCase;
    }

    @Test
    public void ComputeConfidenceIntervalInt() {
      ConfidenceInterval confInt =
          lap.computeConfidenceInterval(
              (int) testCase.noisedValue(),
              testCase.l0Sensitivity(),
              (long) testCase.lInfSensitivity(),
              testCase.epsilon(),
              testCase.delta(),
              testCase.alpha());
      assertWithMessage("Lower bounds are not equal.")
          .that(confInt.lowerBound())
          .isWithin(TOLERANCE)
          .of(testCase.expected().lowerBound());
      assertWithMessage("Upper bounds are not equal.")
          .that(confInt.upperBound())
          .isWithin(TOLERANCE)
          .of(testCase.expected().upperBound());
    }

    @Parameterized.Parameters(name = "{index}: = failed")
    public static Collection<Object> getTestCases() {
      return Arrays.asList(
          // Random tests
          ConfidenceIntervalTestCase.create(
              13, 1, 1, 0.3, null, 0.05, ConfidenceInterval.create(3, 23)),
          ConfidenceIntervalTestCase.create(
              83, 3, 2, 0.1, null, 0.24, ConfidenceInterval.create(-3, 169)),
          ConfidenceIntervalTestCase.create(
              5, 10, 2, 3, null, 0.6, ConfidenceInterval.create(2, 8)),
          ConfidenceIntervalTestCase.create(
              65, 7, 10, 0.1, null, 0.8, ConfidenceInterval.create(-91, 221)),
          // Extremely low confidence level
          ConfidenceIntervalTestCase.create(
              0, 1, 1, 0.1, null, 1 - 3.548957438e-10, ConfidenceInterval.create(0, 0)),
          // Extremely high confidence level
          ConfidenceIntervalTestCase.create(
              50, 1, 1, 0.1, null, 7.856382354e-10, ConfidenceInterval.create(-160, 260)));
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
              lap.computeConfidenceInterval(
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
          ConfidenceIntervalTestCase.create(0, 0, 1, 0.1, null, 0.1, null),
          // Negative l0Sensitivity
          ConfidenceIntervalTestCase.create(0, -1, 1, 0.1, null, 0.1, null),
          // Zero lInfSensitivity
          ConfidenceIntervalTestCase.create(0, 1, 0, 0.1, null, 0.1, null),
          // Negative lInfSensitivity
          ConfidenceIntervalTestCase.create(0, 1, -1, 0.1, null, 0.1, null),
          // Infinite lInfSensitivity
          ConfidenceIntervalTestCase.create(0, 1, Double.POSITIVE_INFINITY, 0.1, null, 0.5, null),
          // NaN lInfSensitivity
          ConfidenceIntervalTestCase.create(0, 1, Double.NaN, 0.1, null, 0.1, null),
          // Zero epsilon
          ConfidenceIntervalTestCase.create(0, 1, 1, 0, null, 0.1, null),
          // Negative epsilon
          ConfidenceIntervalTestCase.create(0, 1, 1, -1, null, 0.1, null),
          // Infinite epsilon
          ConfidenceIntervalTestCase.create(0, 1, 1, Double.POSITIVE_INFINITY, null, 0.1, null),
          // NaN epsilon
          ConfidenceIntervalTestCase.create(0, 1, 1, Double.NaN, null, 0.1, null),
          // Non-null delta
          ConfidenceIntervalTestCase.create(0, 1, 1, 0.1, 1.0, 0.1, null),
          // Zero alpha
          ConfidenceIntervalTestCase.create(0, 1, 1, 0.1, null, 0, null),
          // Negative alpha
          ConfidenceIntervalTestCase.create(0, 1, 1, 0.1, null, -1, null),
          // 1 alpha
          ConfidenceIntervalTestCase.create(0, 1, 1, 0.1, null, 1, null),
          // Greater than 1 alpha
          ConfidenceIntervalTestCase.create(0, 1, 1, 0.1, null, 2, null),
          // NaN alpha
          ConfidenceIntervalTestCase.create(0, 1, 1, 0.1, null, Double.NaN, null));
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
              lap.computeConfidenceInterval(
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
          ConfidenceIntervalTestCase.create(0, 0, 1, 0.1, null, 0.1, null),
          // Negative l0Sensitivity
          ConfidenceIntervalTestCase.create(0, -1, 1, 0.1, null, 0.1, null),
          // Zero lInfSensitivity
          ConfidenceIntervalTestCase.create(0, 1, 0, 0.1, null, 0.1, null),
          // Negative lInfSensitivity
          ConfidenceIntervalTestCase.create(0, 1, -1, 0.1, null, 0.1, null),
          // Zero epsilon
          ConfidenceIntervalTestCase.create(0, 1, 1, 0, null, 0.1, null),
          // Negative epsilon
          ConfidenceIntervalTestCase.create(0, 1, 1, -1, null, 0.1, null),
          // Infinite epsilon
          ConfidenceIntervalTestCase.create(0, 1, 1, Double.POSITIVE_INFINITY, null, 0.1, null),
          // NaN epsilon
          ConfidenceIntervalTestCase.create(0, 1, 1, Double.NaN, null, 0.1, null),
          // Non-null delta
          ConfidenceIntervalTestCase.create(0, 1, 1, 0.1, 1.0, 0.1, null),
          // Zero alpha
          ConfidenceIntervalTestCase.create(0, 1, 1, 0.1, null, 0, null),
          // Negative alpha
          ConfidenceIntervalTestCase.create(0, 1, 1, 0.1, null, -1, null),
          // 1 alpha
          ConfidenceIntervalTestCase.create(0, 1, 1, 0.1, null, 1, null),
          // Greater than 1 alpha
          ConfidenceIntervalTestCase.create(0, 1, 1, 0.1, null, 2, null),
          // NaN alpha
          ConfidenceIntervalTestCase.create(0, 1, 1, 0.1, null, Double.NaN, null));
    }
  }
}

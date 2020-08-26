package com.google.privacy.differentialprivacy;

import static com.google.common.truth.Truth.assertThat;
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
public class ConfidenceIntervalGaussianTest {
    private static final Noise gauss = new GaussianNoise();
    private static final double TOLERANCE = 1e-7;

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
            return new AutoValue_ConfidenceIntervalGaussianTest_ConfidenceIntervalTestCase(
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
                    // Random Tests.
                    ConfidenceIntervalTestCase.create(
                            70.00,
                            5,
                            36,
                            0.8,
                            0.8,
                            0.2,
                            ConfidenceInterval.create(35.26815080641682, 104.73184919358317)),
                    ConfidenceIntervalTestCase.create(
                            699.2402199905,
                            1,
                            5,
                            0.333,
                            0.9,
                            0.001256458,
                            ConfidenceInterval.create(694.5583238637953, 703.9221161172047)),
                    // High alpha.
                    ConfidenceIntervalTestCase.create(
                            70.00,
                            5,
                            36,
                            0.8,
                            0.8,
                            1 - 7.856382354e-10,
                            ConfidenceInterval.create(69.9999999733, 70.0000000267)),
                    // Low alpha.
                    ConfidenceIntervalTestCase.create(
                            70.00,
                            5,
                            36,
                            0.8,
                            0.8,
                            7.856382354e-10,
                            ConfidenceInterval.create(-96.6140883158, 236.6140883158)));
        }

        @Test
        public void computeConfidenceIntervalDouble() {
            ConfidenceInterval confInt =
                    gauss.computeConfidenceInterval(
                            testCase.noisedX(),
                            testCase.l0Sensitivity(),
                            testCase.lInfSensitivity(),
                            testCase.epsilon(),
                            testCase.delta(),
                            testCase.alpha());
            assertWithMessage("Lower bound is not equal")
                    .that(confInt.lowerBound())
                    .isWithin(TOLERANCE)
                    .of(testCase.expected().lowerBound());
            assertWithMessage("Upper bound is not equal")
                    .that(confInt.upperBound())
                    .isWithin(TOLERANCE)
                    .of(testCase.expected().upperBound());
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
                    // Random Tests.
                    ConfidenceIntervalTestCase.create(
                            70, 5, 36, 0.8, 0.8, 0.2, ConfidenceInterval.create(35, 105)),
                    ConfidenceIntervalTestCase.create(
                            700, 1, 5, 0.333, 0.9, 0.001256458, ConfidenceInterval.create(695, 705)),
                    // High alpha.
                    ConfidenceIntervalTestCase.create(
                            70.00, 5, 36, 0.8, 0.8, 1 - 7.856382354e-10, ConfidenceInterval.create(70, 70)),
                    // Low alpha.
                    ConfidenceIntervalTestCase.create(
                            70.00, 5, 36, 0.8, 0.8, 7.856382354e-10, ConfidenceInterval.create(-97, 237)));
        }

        @Test
        public void computeConfidenceIntervalLong() {
            ConfidenceInterval confInt =
                    gauss.computeConfidenceInterval(
                            (long) testCase.noisedX(),
                            testCase.l0Sensitivity(),
                            (long) testCase.lInfSensitivity(),
                            testCase.epsilon(),
                            testCase.delta(),
                            testCase.alpha());
            assertWithMessage("Lower bound is not equal")
                    .that(confInt.lowerBound())
                    .isWithin(TOLERANCE)
                    .of(testCase.expected().lowerBound());
            assertWithMessage("Upper bound is not equal")
                    .that(confInt.upperBound())
                    .isWithin(TOLERANCE)
                    .of(testCase.expected().upperBound());
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
                            0, 0, 1, 0.5, 0.3, 0.5, ConfidenceInterval.create(0, 0)),
                    // Negative l0Sensitivity
                    ConfidenceIntervalTestCase.create(
                            0, -1, 1, 0.5, 0.3, 0.5, ConfidenceInterval.create(0, 0)),
                    // Zero lInfSensitivity
                    ConfidenceIntervalTestCase.create(
                            0, 1, 0, 0.5, 0.3, 0.5, ConfidenceInterval.create(0, 0)),
                    // Negative lInfSensitivity
                    ConfidenceIntervalTestCase.create(
                            0, 1, -1, 0.5, 0.3, 0.5, ConfidenceInterval.create(0, 0)),
                    // Infinite lInfSensitivity
                    ConfidenceIntervalTestCase.create(
                            0, 1, Double.POSITIVE_INFINITY, 0.5, 0.3, 0.5, ConfidenceInterval.create(0, 0)),
                    // NaN lInfSensitivity
                    ConfidenceIntervalTestCase.create(
                            0, 1, Double.NaN, 0.5, 0.3, 0.5, ConfidenceInterval.create(0, 0)),
                    // Infinite epsilon
                    ConfidenceIntervalTestCase.create(
                            0, 1, 1, Double.POSITIVE_INFINITY, 0.3, 0.5, ConfidenceInterval.create(0, 0)),
                    // Very small epsilon (less than 2^-50)
                    ConfidenceIntervalTestCase.create(
                            0, 1, 1, Math.exp(-51), 0.3, 0, ConfidenceInterval.create(0, 0)),
                    // NaN epsilon
                    ConfidenceIntervalTestCase.create(
                            0, 1, 1, Double.NaN, 0.3, 0, ConfidenceInterval.create(0, 0)),
                    // Negative confidence level
                    ConfidenceIntervalTestCase.create(0, 1, 1, 0.5, 0.3, -1, ConfidenceInterval.create(0, 0)),
                    // Greater than 1 confidence level
                    ConfidenceIntervalTestCase.create(0, 1, 1, 0.5, 0.3, 2, ConfidenceInterval.create(0, 0)),
                    // NaN confidence level
                    ConfidenceIntervalTestCase.create(
                            0, 1, 1, 0.5, 0.3, Double.NaN, ConfidenceInterval.create(0, 0)));
        }

        @Test
        public void argumentCheckingConfidenceIntervalDouble() {
            assertThrows(
                    IllegalArgumentException.class,
                    () ->
                            gauss.computeConfidenceInterval(
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
                            0, 0, 1, 0.5, 0.3, 0.5, ConfidenceInterval.create(0, 0)),
                    // Negative l0Sensitivity
                    ConfidenceIntervalTestCase.create(
                            0, -1, 1, 0.5, 0.3, 0.5, ConfidenceInterval.create(0, 0)),
                    // Zero lInfSensitivity
                    ConfidenceIntervalTestCase.create(
                            0, 1, 0, 0.5, 0.3, 0.5, ConfidenceInterval.create(0, 0)),
                    // Negative lInfSensitivity
                    ConfidenceIntervalTestCase.create(
                            0, 1, -1, 0.5, 0.3, 0.5, ConfidenceInterval.create(0, 0)),
                    // Infinite lInfSensitivity
                    ConfidenceIntervalTestCase.create(
                            0, 1, Double.POSITIVE_INFINITY, 0.5, 0.3, 0.5, ConfidenceInterval.create(0, 0)),
                    // NaN lInfSensitivity
                    ConfidenceIntervalTestCase.create(
                            0, 1, Double.NaN, 0.5, 0.3, 0.5, ConfidenceInterval.create(0, 0)),
                    // Infinite epsilon
                    ConfidenceIntervalTestCase.create(
                            0, 1, 1, Double.POSITIVE_INFINITY, 0.3, 0.5, ConfidenceInterval.create(0, 0)),
                    // Very small epsilon (less than 2^-50)
                    ConfidenceIntervalTestCase.create(
                            0, 1, 1, Math.exp(-51), 0.3, 0, ConfidenceInterval.create(0, 0)),
                    // NaN epsilon
                    ConfidenceIntervalTestCase.create(
                            0, 1, 1, Double.NaN, 0.3, 0, ConfidenceInterval.create(0, 0)),
                    // Negative confidence level
                    ConfidenceIntervalTestCase.create(0, 1, 1, 0.5, 0.3, -1, ConfidenceInterval.create(0, 0)),
                    // Greater than 1 confidence level
                    ConfidenceIntervalTestCase.create(0, 1, 1, 0.5, 0.3, 2, ConfidenceInterval.create(0, 0)),
                    // NaN confidence level
                    ConfidenceIntervalTestCase.create(
                            0, 1, 1, 0.5, 0.3, Double.NaN, ConfidenceInterval.create(0, 0)));
        }

        @Test
        public void argumentCheckingConfidenceIntervalInt() {
            assertThrows(
                    IllegalArgumentException.class,
                    () ->
                            gauss.computeConfidenceInterval(
                                    (long) testCase.noisedX(),
                                    testCase.l0Sensitivity(),
                                    testCase.lInfSensitivity(),
                                    testCase.epsilon(),
                                    testCase.delta(),
                                    testCase.alpha()));
        }
    }
}
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

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Collection of tests verifying basic properties of the statistical tests used for assessing the
 * Building Blocks library. Note that the statistical tests are only evaluated for deterministic
 * inputs. Evaluating their statistical properties is out of scope for this unit test.
 */
@RunWith(JUnit4.class)
public class StatisticalTestsUtilTest {
  private static final double DEFAULT_L2_TOLERANCE = 0.001;
  private static final double LOW_L2_TOLERANCE = 0.000000001;
  private static final double HIGH_L2_TOLERANCE = 0.5;
  private static final double DEFAULT_EPSILON = 1.0;
  private static final double DEFAULT_DELTA = 0.00001;
  private static final double DEFAULT_DELTA_TOLERANCE = 0.00001;
  private static final double LOW_DELTA_TOLERANCE = 0.0000000001;
  private static final double HIGH_DELTA_TOLERANCE = 0.5;
  private static final Double[] SAMPLES_A =
      new Double[] {1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0};
  private static final Double[] SAMPLES_A_SHUFFLED =
      new Double[] {5.0, 3.0, 2.0, 3.0, 1.0, 2.0, 5.0, 3.0, 4.0, 5.0, 5.0, 4.0, 5.0, 4.0, 4.0};
  private static final Double[] SAMPLES_B =
      new Double[] {1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0};
  private static final Double[] SAMPLES_B_SHUFFLED =
      new Double[] {4.0, 1.0, 5.0, 1.0, 4.0, 2.0, 3.0, 3.0, 1.0, 2.0, 2.0, 1.0, 3.0, 1.0, 2.0};

  @Test
  public void verifyCloseness_sameSampleSets_accepts() {
    // Should accept independent of tolerance.
    assertThat(StatisticalTestsUtil.verifyCloseness(SAMPLES_A, SAMPLES_A, LOW_L2_TOLERANCE))
        .isTrue();
    assertThat(StatisticalTestsUtil.verifyCloseness(SAMPLES_A, SAMPLES_A, HIGH_L2_TOLERANCE))
        .isTrue();
  }

  @Test
  public void verifyCloseness_differentSampleSets_rejects() {
    assertThat(StatisticalTestsUtil.verifyCloseness(SAMPLES_A, SAMPLES_B, DEFAULT_L2_TOLERANCE))
        .isFalse();
  }

  @Test
  public void verifyCloseness_differentSampleSetsLowL2Tolerance_rejects() {
    assertThat(StatisticalTestsUtil.verifyCloseness(SAMPLES_A, SAMPLES_B, LOW_L2_TOLERANCE))
        .isFalse();
  }

  @Test
  public void verifyCloseness_differentSampleSetsHighL2Tolerance_accepts() {
    assertThat(StatisticalTestsUtil.verifyCloseness(SAMPLES_A, SAMPLES_B, HIGH_L2_TOLERANCE))
        .isTrue();
  }

  @Test
  public void verifyCloseness_invariantToSampleOrder() {
    assertThat(StatisticalTestsUtil.verifyCloseness(SAMPLES_A, SAMPLES_A, DEFAULT_L2_TOLERANCE))
        .isEqualTo(
            StatisticalTestsUtil.verifyCloseness(
                SAMPLES_A_SHUFFLED, SAMPLES_A_SHUFFLED, DEFAULT_L2_TOLERANCE));
    assertThat(StatisticalTestsUtil.verifyCloseness(SAMPLES_A, SAMPLES_B, DEFAULT_L2_TOLERANCE))
        .isEqualTo(
            StatisticalTestsUtil.verifyCloseness(
                SAMPLES_B_SHUFFLED, SAMPLES_A_SHUFFLED, DEFAULT_L2_TOLERANCE));
  }

  @Test
  public void verifyCloseness_emptySamples_throwsError() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            StatisticalTestsUtil.verifyCloseness(
                new Integer[0], new Integer[0], DEFAULT_L2_TOLERANCE));
  }

  @Test
  public void verifyCloseness_mismatchingNumberOfSamples_throwsError() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            StatisticalTestsUtil.verifyCloseness(
                new Integer[] {0}, new Integer[] {1, 2}, DEFAULT_L2_TOLERANCE));
  }

  @Test
  public void verifyCloseness_l2ToleranceLessOrEqualToZero_throwsError() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            StatisticalTestsUtil.verifyCloseness(
                new Integer[] {0}, new Integer[] {1}, /* l2 tolerance= */ 0.0));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            StatisticalTestsUtil.verifyCloseness(
                new Integer[] {0}, new Integer[] {1}, -LOW_L2_TOLERANCE));
  }

  @Test
  public void verifyCloseness_l2ToleranceGreaterOrEqualToOne_throwsError() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            StatisticalTestsUtil.verifyCloseness(
                new Integer[] {0}, new Integer[] {1}, /* l2 tolerance= */ 1.0));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            StatisticalTestsUtil.verifyCloseness(
                new Integer[] {0}, new Integer[] {1}, 1.0 + LOW_L2_TOLERANCE));
  }

  @Test
  public void verifyApproximateDp_sameSampleSets_accepts() {
    // Should accept equal sample sets independent of tolerance even if epsilon and delta are 0.
    assertThat(
            StatisticalTestsUtil.verifyApproximateDp(
                SAMPLES_A, SAMPLES_A, /* epsilon= */ 0.0, /* delta= */ 0.0, LOW_DELTA_TOLERANCE))
        .isTrue();
    assertThat(
            StatisticalTestsUtil.verifyApproximateDp(
                SAMPLES_A, SAMPLES_A, /* epsilon= */ 0.0, /* delta= */ 0.0, HIGH_DELTA_TOLERANCE))
        .isTrue();
  }

  @Test
  public void verifyApproximateDp_differentSampleSets_rejects() {
    assertThat(
            StatisticalTestsUtil.verifyApproximateDp(
                SAMPLES_A, SAMPLES_B, DEFAULT_EPSILON, DEFAULT_DELTA, DEFAULT_DELTA_TOLERANCE))
        .isFalse();
  }

  @Test
  public void verifyApproximateDp_differentSampleSetsLowDeltaTolerance_rejects() {
    assertThat(
            StatisticalTestsUtil.verifyApproximateDp(
                SAMPLES_A, SAMPLES_B, DEFAULT_EPSILON, DEFAULT_DELTA, LOW_DELTA_TOLERANCE))
        .isFalse();
  }

  @Test
  public void verifyApproximateDp_differentSampleSetsHighDeltaTolerance_accepts() {
    assertThat(
            StatisticalTestsUtil.verifyApproximateDp(
                SAMPLES_A, SAMPLES_B, DEFAULT_EPSILON, DEFAULT_DELTA, HIGH_DELTA_TOLERANCE))
        .isTrue();
  }

  @Test
  public void verifyApproximateDp_invariantToSampleOrder() {
    // Should accept equal sample sets even if epsilon and delta are set to 0.
    assertThat(
            StatisticalTestsUtil.verifyApproximateDp(
                SAMPLES_A,
                SAMPLES_A,
                /* epsilon= */ 0.0,
                /* delta= */ 0.0,
                DEFAULT_DELTA_TOLERANCE))
        .isEqualTo(
            StatisticalTestsUtil.verifyApproximateDp(
                SAMPLES_A_SHUFFLED,
                SAMPLES_A_SHUFFLED,
                /* epsilon= */ 0.0,
                /* delta= */ 0.0,
                DEFAULT_DELTA_TOLERANCE));
    assertThat(
            StatisticalTestsUtil.verifyApproximateDp(
                SAMPLES_A, SAMPLES_B, DEFAULT_EPSILON, DEFAULT_DELTA, DEFAULT_DELTA_TOLERANCE))
        .isEqualTo(
            StatisticalTestsUtil.verifyApproximateDp(
                SAMPLES_B_SHUFFLED,
                SAMPLES_A_SHUFFLED,
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                DEFAULT_DELTA_TOLERANCE));
  }

  @Test
  public void verifyApproximateDp_emptySamples_throwsError() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            StatisticalTestsUtil.verifyApproximateDp(
                new Integer[0],
                new Integer[0],
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                DEFAULT_DELTA_TOLERANCE));
  }

  @Test
  public void verifyApproximateDp_mismatchingNumberOfSamples_throwsError() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            StatisticalTestsUtil.verifyApproximateDp(
                new Integer[] {0},
                new Integer[] {1, 2},
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                DEFAULT_DELTA_TOLERANCE));
  }

  @Test
  public void verifyApproximateDp_deltaToleranceLessOrEqualToZero_throwsError() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            StatisticalTestsUtil.verifyApproximateDp(
                new Integer[] {0},
                new Integer[] {1},
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                /* delta tolerance= */ 0.0));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            StatisticalTestsUtil.verifyApproximateDp(
                new Integer[] {0},
                new Integer[] {1},
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                -LOW_DELTA_TOLERANCE));
  }

  @Test
  public void verifyApproximateDp_deltaToleranceGreaterOrEqualToOne_throwsError() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            StatisticalTestsUtil.verifyApproximateDp(
                new Integer[] {0},
                new Integer[] {1},
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                /* delta tolerance= */ 1.0));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            StatisticalTestsUtil.verifyApproximateDp(
                new Integer[] {0},
                new Integer[] {1},
                DEFAULT_EPSILON,
                DEFAULT_DELTA,
                1.0 + LOW_DELTA_TOLERANCE));
  }

  @Test
  public void verifyApproximateDp_epsilonLessThanZero_throwsError() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            StatisticalTestsUtil.verifyApproximateDp(
                new Integer[] {0},
                new Integer[] {1},
                -DEFAULT_EPSILON,
                DEFAULT_DELTA,
                DEFAULT_DELTA_TOLERANCE));
  }

  @Test
  public void verifyApproximateDp_deltaLessThanZero_throwsError() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            StatisticalTestsUtil.verifyApproximateDp(
                new Integer[] {0},
                new Integer[] {1},
                DEFAULT_EPSILON,
                -DEFAULT_DELTA,
                DEFAULT_DELTA_TOLERANCE));
  }

  @Test
  public void verifyApproximateDp_deltaGreaterOrEqualToZero_throwsError() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            StatisticalTestsUtil.verifyApproximateDp(
                new Integer[] {0},
                new Integer[] {1},
                DEFAULT_EPSILON,
                /* delta= */ 1.0,
                DEFAULT_DELTA_TOLERANCE));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            StatisticalTestsUtil.verifyApproximateDp(
                new Integer[] {0},
                new Integer[] {1},
                DEFAULT_EPSILON,
                1.0 + DEFAULT_DELTA,
                DEFAULT_DELTA_TOLERANCE));
  }

  @Test
  public void discretize_binaryGranularity() {
    assertThat(StatisticalTestsUtil.discretize(36.4621596072, Math.pow(2.0, -10.0)))
        .isEqualTo(36.4619140625);
    assertThat(StatisticalTestsUtil.discretize(36.4621596072, 0.5)).isEqualTo(36.5);
    assertThat(StatisticalTestsUtil.discretize(36.4621596072, 1.0)).isEqualTo(36.0);
    assertThat(StatisticalTestsUtil.discretize(36.4621596072, 2.0)).isEqualTo(36.0);
    assertThat(StatisticalTestsUtil.discretize(36.4621596072, Math.pow(2.0, 10.0))).isEqualTo(0.0);
  }

  @Test
  public void discretize_granularityCloseToZero() {
    // No rounding should happen.
    assertThat(StatisticalTestsUtil.discretize(36.4621596072, Math.pow(2.0, -100.0)))
        .isEqualTo(36.4621596072);
  }

  @Test
  public void discretize_sampleIsMultipleOfGranularity() {
    // No rounding should happen.
    assertThat(StatisticalTestsUtil.discretize(36.4621596072, 36.4621596072))
        .isEqualTo(36.4621596072);
  }

  @Test
  public void bucketize_sampleLessThanLower_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            StatisticalTestsUtil.bucketize(
                /*sample=*/ -1.0, /*lower=*/ 0.0, /*upper=*/ 1.0, /*numberOfBuckets=*/ 10));
  }

  @Test
  public void bucketize_sampleGreaterThanUpper_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            StatisticalTestsUtil.bucketize(
                /*sample=*/ 2.0, /*lower=*/ 0.0, /*upper=*/ 1.0, /*numberOfBuckets=*/ 10));
  }

  @Test
  public void bucketize_lowerEqualToUpper_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            StatisticalTestsUtil.bucketize(
                /*sample=*/ 0.0, /*lower=*/ 0.0, /*upper=*/ 0.0, /*numberOfBuckets=*/ 10));
  }

  @Test
  public void bucketize_numberOfBucketsLessThanOne_throwsException() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            StatisticalTestsUtil.bucketize(
                /*sample=*/ 0.5, /*lower=*/ 0.0, /*upper=*/ 1.0, /*numberOfBuckets=*/ 0));
  }

  @Test
  public void bucketize_sampleEqualToLower_returnsZero() {
    assertThat(
            StatisticalTestsUtil.bucketize(
                /*sample=*/ 0.0, /*lower=*/ 0.0, /*upper=*/ 1.0, /*numberOfBuckets=*/ 10))
        .isEqualTo(0);
  }

  @Test
  public void bucketize_sampleEqualUpper_returnsNumberOfBucketsMinusOne() {
    assertThat(
            StatisticalTestsUtil.bucketize(
                /*sample=*/ 1.0, /*lower=*/ 0.0, /*upper=*/ 1.0, /*numberOfBuckets=*/ 10))
        .isEqualTo(9);
  }

  @Test
  public void bucketize_roundsCorrectly() {
    assertThat(
            StatisticalTestsUtil.bucketize(
                /*sample=*/ 0.39999, /*lower=*/ 0.0, /*upper=*/ 1.0, /*numberOfBuckets=*/ 10))
        .isEqualTo(3);
    assertThat(
            StatisticalTestsUtil.bucketize(
                /*sample=*/ 0.4, /*lower=*/ 0.0, /*upper=*/ 1.0, /*numberOfBuckets=*/ 10))
        .isEqualTo(4);
  }

  @Test
  public void roundToNextMultipleOf_sampleIsNegative() {
    assertThat(StatisticalTestsUtil.discretize(-36.4621596072, Math.pow(2.0, -10.0)))
        .isEqualTo(-36.4619140625);
  }

  @Test
  public void roundToNextMultipleOf_invariantToChangesInLowDigits() {
    assertThat(StatisticalTestsUtil.discretize(36.4621596072, 0.00001))
        .isEqualTo(StatisticalTestsUtil.discretize(36.462158888, 0.00001));
  }

  @Test
  public void roundToNextMultipleOf_granularityIsZero_throwsException() {
    assertThrows(
        IllegalArgumentException.class, () -> StatisticalTestsUtil.discretize(36.4621596072, 0.0));
  }

  @Test
  public void roundToNextMultipleOf_granularityIsNegative_throwsException() {
    assertThrows(
        IllegalArgumentException.class, () -> StatisticalTestsUtil.discretize(36.4621596072, -1.0));
  }
}

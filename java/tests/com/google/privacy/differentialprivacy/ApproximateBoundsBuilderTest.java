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

import static com.google.common.truth.Truth.assertThat;
import static java.lang.Double.NaN;
import static java.lang.Double.POSITIVE_INFINITY;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import com.google.testing.junit.testparameterinjector.TestParameters;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests validations done by {@link ApproximateBounds#builder()}. */
@RunWith(TestParameterInjector.class)
public class ApproximateBoundsBuilderTest {
  private ApproximateBounds.Params.Builder builder;

  private static final double EPSILON = 1.0;

  @Before
  public void setUp() {
    builder =
        ApproximateBounds.builder()
            .epsilon(EPSILON)
            .inputType(ApproximateBounds.Params.InputType.DOUBLE)
            .maxContributions(1);
  }

  @Test
  public void epsilon_belowZero_throwsException() {
    builder.epsilon(-1);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void epsilon_zero_throwsException() {
    builder.epsilon(0);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void epsilon_posInfinity_throwsException() {
    builder.epsilon(POSITIVE_INFINITY);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void epsilon_Nan_throwsException() {
    builder.epsilon(NaN);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void builder_hasSensibleDefaults() {
    ApproximateBounds bounds =
        ApproximateBounds.builder().epsilon(EPSILON).maxContributions(1).build();

    assertThat(bounds.params.inputType()).isEqualTo(ApproximateBounds.Params.InputType.DOUBLE);
  }

  @Test
  public void builder_inputTypeMissing_usesDoubleBinBoundaries() {
    ApproximateBounds bounds = ApproximateBounds.builder().epsilon(100).maxContributions(1).build();

    bounds.addEntries(ImmutableList.of(-0.4, 60.0));
    ApproximateBounds.Result result = bounds.computeResult();

    assertThat(bounds.params.inputType()).isEqualTo(ApproximateBounds.Params.InputType.DOUBLE);
    assertThat(result.lowerBound()).isEqualTo(-0.5);
    assertThat(result.upperBound()).isEqualTo(64);
  }

  // This test checks that the builder computes the boundaries of the histogram correctly. It does
  // this by using a dataset consisting of a single point: the lower and upper bounds that are
  // computed are then the boundaries of the bin containing that data point. Tests that the
  // approximate bounds algorithm itself works correctly are found in {@link ApproximateBoundsTest}.
  @Test
  // Input type TEST
  @TestParameters("{inputType: TEST, input: -20, lower: -16, upper: -8}")
  @TestParameters("{inputType: TEST, input: -12, lower: -16, upper: -8}")
  @TestParameters("{inputType: TEST, input: -5, lower: -8, upper: -4}")
  @TestParameters("{inputType: TEST, input: -2.5, lower: -4, upper: -2}")
  @TestParameters("{inputType: TEST, input: -1.5, lower: -2, upper: -1}")
  @TestParameters("{inputType: TEST, input: -0.1, lower: -1, upper: 0}")
  @TestParameters("{inputType: TEST, input: 0.1, lower: 0, upper: 1}")
  @TestParameters("{inputType: TEST, input: 1.6, lower: 1, upper: 2}")
  @TestParameters("{inputType: TEST, input: 3, lower: 2, upper: 4}")
  @TestParameters("{inputType: TEST, input: 7, lower: 4, upper: 8}")
  @TestParameters("{inputType: TEST, input: 13, lower: 8, upper: 16}")
  // Input type DOUBLE
  @TestParameters("{inputType: DOUBLE, input: -20, lower: -32, upper: -16}")
  @TestParameters("{inputType: DOUBLE, input: -3, lower: -4, upper: -2}")
  @TestParameters("{inputType: DOUBLE, input: -0.9, lower: -1, upper: -0.5}")
  @TestParameters("{inputType: DOUBLE, input: -0.4, lower: -0.5, upper: -0.25}")
  @TestParameters("{inputType: DOUBLE, input: 0.4, lower: 0.25, upper: 0.5}")
  @TestParameters("{inputType: DOUBLE, input: 0.9, lower: 0.5, upper: 1}")
  @TestParameters("{inputType: DOUBLE, input: 3, lower: 2, upper: 4}")
  @TestParameters("{inputType: DOUBLE, input: 200, lower: 128, upper: 256}")
  // Input type INTEGER
  @TestParameters("{inputType: INTEGER, input: -20, lower: -32, upper: -16}")
  @TestParameters("{inputType: INTEGER, input: -3, lower: -4, upper: -2}")
  @TestParameters("{inputType: INTEGER, input: -0.9, lower: -1, upper: 0}")
  @TestParameters("{inputType: INTEGER, input: 0.4, lower: 0, upper: 1}")
  @TestParameters("{inputType: INTEGER, input: 3, lower: 2, upper: 4}")
  @TestParameters("{inputType: INTEGER, input: 200, lower: 128, upper: 256}")
  // Input type POSITIVE_INTEGER
  @TestParameters("{inputType: POSITIVE_INTEGER, input: -20, lower: 0, upper: 1}")
  @TestParameters("{inputType: POSITIVE_INTEGER, input: -0.9, lower: 0, upper: 1}")
  @TestParameters("{inputType: POSITIVE_INTEGER, input: 0.4, lower: 0, upper: 1}")
  @TestParameters("{inputType: POSITIVE_INTEGER, input: 3, lower: 2, upper: 4}")
  @TestParameters("{inputType: POSITIVE_INTEGER, input: 200, lower: 128, upper: 256}")
  public void builder_computesBinBoundariesCorrectly(
      ApproximateBounds.Params.InputType inputType, double input, double lower, double upper) {
    ApproximateBounds bounds =
        ApproximateBounds.builder().epsilon(100).inputType(inputType).maxContributions(1).build();

    bounds.addEntry(input);
    ApproximateBounds.Result result = bounds.computeResult();

    assertThat(result.lowerBound()).isWithin(1e-10).of(lower);
    assertThat(result.upperBound()).isWithin(1e-10).of(upper);
  }

  @Test
  public void builder_inputTypeDouble_handlesExtremeValues() {
    ApproximateBounds bounds =
        ApproximateBounds.builder()
            .epsilon(100)
            .inputType(ApproximateBounds.Params.InputType.DOUBLE)
            .maxContributions(1)
            .build();

    bounds.addEntry(Double.MAX_VALUE);
    bounds.addEntry(-Double.MAX_VALUE);
    ApproximateBounds.Result result = bounds.computeResult();

    assertThat(result.lowerBound()).isFinite();
    assertThat(result.upperBound()).isFinite();
    assertThat(result.lowerBound() * 2).isNegativeInfinity();
    assertThat(result.upperBound() * 2).isPositiveInfinity();
  }

  @Test
  public void builder_inputTypeInteger_handlesExtremeValues() {
    ApproximateBounds bounds =
        ApproximateBounds.builder()
            .epsilon(100)
            .inputType(ApproximateBounds.Params.InputType.INTEGER)
            .maxContributions(1)
            .build();

    bounds.addEntry(Integer.MAX_VALUE);
    bounds.addEntry(Integer.MIN_VALUE);
    ApproximateBounds.Result result = bounds.computeResult();

    assertThat(result.upperBound() / 2).isLessThan((double) Integer.MAX_VALUE);
    assertThat(result.upperBound()).isAtLeast((double) Integer.MAX_VALUE);
    assertThat(result.lowerBound() / 2).isGreaterThan((double) Integer.MIN_VALUE);
    assertThat(result.lowerBound()).isAtMost((double) Integer.MIN_VALUE);
  }

  @Test
  public void builder_inputTypePositiveInteger_handlesExtremeValues() {
    ApproximateBounds bounds =
        ApproximateBounds.builder()
            .epsilon(100)
            .inputType(ApproximateBounds.Params.InputType.POSITIVE_INTEGER)
            .maxContributions(1)
            .build();

    bounds.addEntry(Integer.MAX_VALUE);
    bounds.addEntry(Integer.MIN_VALUE);
    ApproximateBounds.Result result = bounds.computeResult();

    assertThat(result.upperBound() / 2).isLessThan((double) Integer.MAX_VALUE);
    assertThat(result.upperBound()).isAtLeast((double) Integer.MAX_VALUE);
    assertThat(result.lowerBound()).isWithin(1e-10).of(0);
  }

  @Test
  // Input type DOUBLE is tested separately below
  // Input type TEST
  @TestParameters("{inputType: TEST, input: 0, expectedBinNumber: 0}")
  @TestParameters("{inputType: TEST, input: 0.5, expectedBinNumber: 0}")
  @TestParameters("{inputType: TEST, input: 20, expectedBinNumber: 4}") // 20.0 is clamped to 16.0
  // Input type INTEGER
  @TestParameters("{inputType: INTEGER, input: 0, expectedBinNumber: 0}")
  @TestParameters("{inputType: INTEGER, input: 0.5, expectedBinNumber: 0}")
  @TestParameters("{inputType: INTEGER, input: 1.5, expectedBinNumber: 1}")
  @TestParameters("{inputType: INTEGER, input: 20, expectedBinNumber: 5}") // 20 < 2^5 = 32
  // Input type POSITIVE_INTEGER
  @TestParameters("{inputType: POSITIVE_INTEGER, input: 0, expectedBinNumber: 0}")
  @TestParameters("{inputType: POSITIVE_INTEGER, input: 0.5, expectedBinNumber: 0}")
  @TestParameters("{inputType: POSITIVE_INTEGER, input: 1.5, expectedBinNumber: 1}")
  @TestParameters("{inputType: POSITIVE_INTEGER, input: 20, expectedBinNumber: 5}") // 20 < 2^5 = 32
  public void getPositiveBinNumber(
      ApproximateBounds.Params.InputType inputType, double input, int expectedBinNumber) {
    assertThat(inputType.getPositiveBinNumber(input)).isEqualTo(expectedBinNumber);
  }

  @Test
  public void getPositiveBinNumber_inputTypeDouble() {
    ApproximateBounds.Params.InputType inputType = ApproximateBounds.Params.InputType.DOUBLE;

    assertThat(inputType.getPositiveBinNumber(0)).isEqualTo(0);
    assertThat(inputType.getPositiveBinNumber(Double.MIN_NORMAL)).isEqualTo(0);
    assertThat(inputType.getPositiveBinNumber(1.5 * Double.MIN_NORMAL)).isEqualTo(1);
    assertThat(inputType.getPositiveBinNumber(2.0 * Double.MIN_NORMAL)).isEqualTo(1);
    assertThat(inputType.getPositiveBinNumber(4.0))
        .isEqualTo(1 + inputType.getPositiveBinNumber(2.0));
    assertThat(inputType.getPositiveBinNumber(Double.MAX_VALUE))
        .isEqualTo(inputType.numPositiveBins - 1);
  }

  @Test
  public void getPositiveBinNumber_negativeInput_throws() {
    ApproximateBounds.Params.InputType inputType = ApproximateBounds.Params.InputType.DOUBLE;

    Throwable thrown =
        assertThrows(IllegalArgumentException.class, () -> inputType.getPositiveBinNumber(-2));
    assertThat(thrown).hasMessageThat().contains("Expected a positive input");
  }
}

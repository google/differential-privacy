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
import static java.lang.Math.pow;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.privacy.differentialprivacy.proto.SummaryOuterClass.ApproxBoundsSummary;
import com.google.protobuf.ExtensionRegistry;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import com.google.testing.junit.testparameterinjector.TestParameterValuesProvider;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for {@link ApproximateBounds}. */
@RunWith(TestParameterInjector.class)
public class ApproximateBoundsTest {
  private ApproximateBounds.Params.Builder params;

  @Before
  public void setup() {
    params =
        ApproximateBounds.builder()
            .epsilon(100)
            .inputType(ApproximateBounds.Params.InputType.TEST)
            .maxContributions(1);
  }

  @Test
  public void addEntry_afterComputeResult_throwsException() {
    ApproximateBounds bounds =
        ApproximateBounds.builder()
            .epsilon(100)
            .inputType(ApproximateBounds.Params.InputType.TEST)
            .maxContributions(1)
            .build();

    bounds.addEntry(1);
    bounds.computeResult();
    assertThrows(IllegalStateException.class, () -> bounds.addEntry(1));
  }

  @Test
  public void addEntry_computeResultMultipleTimes_throwsException() {
    ApproximateBounds bounds =
        ApproximateBounds.builder()
            .epsilon(100)
            .inputType(ApproximateBounds.Params.InputType.TEST)
            .maxContributions(1)
            .build();

    bounds.addEntry(1);
    bounds.computeResult();

    assertThrows(IllegalStateException.class, () -> bounds.computeResult());
  }

  /**
   * This test copies the logic of {@link #computeResult_highEpsilon_bothBoundsPositive()}, adds
   * arbitrary many NaNs to the input dataset, and verifies that NaNs do not impact the result of
   * the computation.
   */
  @Test
  public void addEntry_Nan_ignored() {
    ApproximateBounds boundsWithoutNans = createBoundsApproximatorWithData();
    ApproximateBounds.Result boundsWithoutNansResult = boundsWithoutNans.computeResult();

    ApproximateBounds boundsWithNans = createBoundsApproximatorWithData();
    // Add arbitrary many Nans to ensure that they will impact the result if they are not ignored.
    for (int i = 0; i < 1000; i++) {
      boundsWithNans.addEntry(NaN);
    }
    ApproximateBounds.Result boundsWithNansResult = boundsWithNans.computeResult();

    assertThat(boundsWithNansResult.lowerBound()).isEqualTo(boundsWithoutNansResult.lowerBound());
    assertThat(boundsWithNansResult.upperBound()).isEqualTo(boundsWithoutNansResult.upperBound());
  }

  private static ApproximateBounds createBoundsApproximatorWithData() {
    ApproximateBounds bounds =
        ApproximateBounds.builder()
            // A sufficiently high epsilon.
            .epsilon(100)
            .inputType(ApproximateBounds.Params.InputType.TEST)
            .maxContributions(1)
            .build();
    bounds.addEntries(ImmutableList.of(1.0, 2.0, 3.0, 4.0, 5.0));
    return bounds;
  }

  //
  // A set of "high epsilon" tests that test the logic and corner cases with no noise in the way.
  //

  @Test
  public void computeResult_highEpsilon_bothBoundsPositive() {
    ApproximateBounds bounds =
        ApproximateBounds.builder()
            // A sufficiently high epsilon.
            .epsilon(100)
            .inputType(ApproximateBounds.Params.InputType.TEST)
            .maxContributions(1)
            .build();

    bounds.addEntries(ImmutableList.of(1.5, 2.5, 3.5, 4.5, 5.5));
    ApproximateBounds.Result boundsResult = bounds.computeResult();

    // This is calculated by rounding the entries to the following bins
    // [..., 0, 1, 2, 4, 8, 16].
    assertThat(boundsResult.lowerBound()).isEqualTo(1.0);
    assertThat(boundsResult.upperBound()).isEqualTo(8.0);
  }

  @Test
  public void computeResult_highEpsilon_bothBoundsNegative() {
    ApproximateBounds bounds =
        ApproximateBounds.builder()
            // A sufficiently high epsilon.
            .epsilon(100)
            .inputType(ApproximateBounds.Params.InputType.TEST)
            .maxContributions(1)
            .build();

    bounds.addEntries(ImmutableList.of(-10.0, -1.5));
    ApproximateBounds.Result boundsResult = bounds.computeResult();

    // This is calculated by rounding the entries to the following bins
    // [-16, -8, -4, -2, -1, 0, 1, ....].
    assertThat(boundsResult.lowerBound()).isEqualTo(-16.0);
    assertThat(boundsResult.upperBound()).isEqualTo(-1.0);
  }

  @Test
  public void computeResult_highEpsilon_positiveAndNegativeBounds() {
    ApproximateBounds bounds =
        ApproximateBounds.builder()
            // A sufficiently high epsilon.
            .epsilon(100)
            .inputType(ApproximateBounds.Params.InputType.TEST)
            .maxContributions(1)
            .build();

    bounds.addEntries(ImmutableList.of(-10.0, 1.5));
    ApproximateBounds.Result boundsResult = bounds.computeResult();

    // This is calculated by rounding the entries to the following bins
    // [-16, -8, -4, -2, -1, 0, 1, 2, ...].
    assertThat(boundsResult.lowerBound()).isEqualTo(-16.0);
    assertThat(boundsResult.upperBound()).isEqualTo(2.0);
  }

  @Test
  public void computeResult_highEpsilon_boundsOutsideOfBins() {
    ApproximateBounds bounds =
        ApproximateBounds.builder()
            // A sufficiently high epsilon.
            .epsilon(100)
            .inputType(ApproximateBounds.Params.InputType.TEST)
            .maxContributions(1)
            .build();

    bounds.addEntries(ImmutableList.of(-50.0, 50.0));
    ApproximateBounds.Result boundsResult = bounds.computeResult();

    // Calculated based on clipping the values to the following bins: [-16, -8, ..., 8, 16].
    assertThat(boundsResult.lowerBound()).isEqualTo(-16.0);
    assertThat(boundsResult.upperBound()).isEqualTo(16.0);
  }

  @Test
  public void computeResult_highEpsilon_borderValues() {
    ApproximateBounds bounds =
        ApproximateBounds.builder()
            // A sufficiently high epsilon.
            .epsilon(100)
            .inputType(ApproximateBounds.Params.InputType.TEST)
            .maxContributions(1)
            .build();

    // Two values residing on boundaries of [..., 0, 1, 2, 4, 8, 16] bins.
    bounds.addEntries(ImmutableList.of(1.0, 8.0));
    ApproximateBounds.Result boundsResult = bounds.computeResult();

    // Boundary values are included to the "left" bin.
    // That's why the minimum value is rounded down to the previous bin left boundary,
    // and right boundary is reported accurately.
    assertThat(boundsResult.lowerBound()).isEqualTo(0.0);
    assertThat(boundsResult.upperBound()).isEqualTo(8.0);
  }

  @Test
  public void computeResult_zeroBound() {
    ApproximateBounds bounds =
        ApproximateBounds.builder()
            // A sufficiently high epsilon.
            .epsilon(100)
            .inputType(ApproximateBounds.Params.InputType.TEST)
            .maxContributions(1)
            .build();

    bounds.addEntries(ImmutableList.of(0.0, 7.0));
    ApproximateBounds.Result boundsResult = bounds.computeResult();

    // Calculated based on the following bins: [..., 0, 1, 2, 4, 8, 16]
    assertThat(boundsResult.lowerBound()).isEqualTo(0.0);
    assertThat(boundsResult.upperBound()).isEqualTo(8.0);
  }

  //
  // A "low epsilon" tests that assume some noise and thresholding. These tests
  // validates that low counts are being disregarded, but do not do any rigid statistical testing.
  //

  @Test
  public void computeResult_lowEpsilon_disregardsLowCountBins() {
    ApproximateBounds bounds =
        ApproximateBounds.builder()
            .epsilon(1)
            .inputType(ApproximateBounds.Params.InputType.TEST)
            .maxContributions(1)
            .build();

    bounds.addEntries(ImmutableList.of(-10.0, 10.0));
    // Add a lot of identical entries to make sure [-4, -2] and [2, 4] are over the threshold.
    for (int i = 0; i < 100; i++) {
      bounds.addEntry(3.0);
      bounds.addEntry(-3.0);
    }
    ApproximateBounds.Result boundsResult = bounds.computeResult();

    assertThat(boundsResult.lowerBound()).isEqualTo(-4);
    assertThat(boundsResult.upperBound()).isEqualTo(4);
  }

  @Test
  public void computeResult_lowEpsilon_fewDataPoints_throwsException() {
    ApproximateBounds bounds =
        ApproximateBounds.builder()
            .epsilon(1)
            .inputType(ApproximateBounds.Params.InputType.TEST)
            .maxContributions(1)
            .build();

    // Add just a few entries.
    bounds.addEntries(ImmutableList.of(-10.0, 10.0));
    assertThrows(IllegalArgumentException.class, bounds::computeResult);
  }

  @Test
  public void computeResult_initialThresholdTooLow_relaxesAndRetries() {
    ApproximateBounds bounds =
        ApproximateBounds.builder()
            // We need to add few enough entries that the initial bounding attempt will fail, and
            // the test only passes if we retry the bounding with a lower threshold. We carefully
            // choose epsilon such that the initial threshold computed is slightly greater than 1
            // (it's approximately 1.0151) then add a single entry to a bin.
            .epsilon(22)
            .inputType(ApproximateBounds.Params.InputType.TEST)
            .maxContributions(1)
            .build();

    bounds.addEntry(3);
    ApproximateBounds.Result result = bounds.computeResult();

    assertThat(result.lowerBound()).isEqualTo(2);
    assertThat(result.upperBound()).isEqualTo(4);
  }

  @Test
  public void computePerBinFailureProbability_computesCorrectly(
      @TestParameter(valuesProvider = SuccessProbabilityProvider.class) Double successProbability,
      @TestParameter(valuesProvider = NumBinsProvider.class) Integer numBins) {
    double actualPerBinFailureProbability =
        ApproximateBounds.computePerBinFailureProbability(1 - successProbability, numBins);

    // This uses a simpler formula for the per-bin probability that we expect to be less accurate
    // for low failure probabilities.
    double expectedPerBinSuccessProbability = pow(successProbability, 1.0 / numBins);
    double expectedPerBinFailureProbability = 1 - expectedPerBinSuccessProbability;
    assertThat(actualPerBinFailureProbability).isWithin(1e-9).of(expectedPerBinFailureProbability);
  }

  @Test
  public void getSerializableSummary_copiesBinCountsCorrectly() throws Exception {
    ApproximateBounds bounds = params.inputType(ApproximateBounds.Params.InputType.TEST).build();
    bounds.addEntries(ImmutableList.of(-0.1, 5.0, 5.0, 20.0));

    byte[] serializedSummary = bounds.getSerializableSummary();

    ApproxBoundsSummary actual =
        ApproxBoundsSummary.parseFrom(serializedSummary, ExtensionRegistry.newInstance());
    assertThat(actual.getPosBinCountList()).containsExactly(0L, 0L, 0L, 2L, 1L);
    assertThat(actual.getNegBinCountList()).containsExactly(1L, 0L, 0L, 0L, 0L);
  }

  @Test
  public void getSerializableSummary_calledAfterComputeResult_throws() {
    ApproximateBounds bounds = params.build();
    bounds.addEntries(ImmutableList.of(1.0, 2.0, 3.0, 4.0));
    bounds.computeResult();

    IllegalStateException thrown =
        assertThrows(IllegalStateException.class, bounds::getSerializableSummary);
    assertThat(thrown).hasMessageThat().contains("result was already computed");
  }

  @Test
  public void getSerializableSummary_getResultAfterSerialization_throws() {
    ApproximateBounds bounds = params.build();
    bounds.addEntries(ImmutableList.of(1.0, 2.0, 3.0, 4.0));

    bounds.getSerializableSummary();

    IllegalStateException thrown = assertThrows(IllegalStateException.class, bounds::computeResult);
    assertThat(thrown).hasMessageThat().contains("already serialized");
  }

  @Test
  public void mergeWith_basicExample_sumsBinCounts() throws Exception {
    ApproximateBounds targetBounds =
        params.inputType(ApproximateBounds.Params.InputType.TEST).build();
    targetBounds.addEntries(ImmutableList.of(5.0, 5.0, 20.0));
    ApproximateBounds sourceBounds =
        params.inputType(ApproximateBounds.Params.InputType.TEST).build();
    sourceBounds.addEntries(ImmutableList.of(-20.0, -5.0, -0.1, -0.1, 0.1, 0.1, 5.0, 20.0));

    targetBounds.mergeWith(sourceBounds.getSerializableSummary());

    // The API doesn't expose the bin counts directly, so we'll check the serialization instead.
    ApproxBoundsSummary summary =
        ApproxBoundsSummary.parseFrom(
            targetBounds.getSerializableSummary(), ExtensionRegistry.newInstance());
    assertThat(summary.getPosBinCountList()).containsExactly(2L, 0L, 3L, 0L, 2L);
    assertThat(summary.getNegBinCountList()).containsExactly(2L, 0L, 1L, 0L, 1L);
  }

  @Test
  public void mergeWith_differentNumberOfBins_throws() {
    ApproximateBounds targetBounds =
        params.inputType(ApproximateBounds.Params.InputType.DOUBLE).build();
    ApproximateBounds sourceBounds =
        params.inputType(ApproximateBounds.Params.InputType.TEST).build();

    IllegalArgumentException thrown =
        assertThrows(
            IllegalArgumentException.class,
            () -> targetBounds.mergeWith(sourceBounds.getSerializableSummary()));
    assertThat(thrown).hasMessageThat().contains("must have the same number of positive bins");
  }

  @Test
  public void mergeWith_calledAfterMergeWith_succeeds() {
    ApproximateBounds targetBounds = params.build();
    ApproximateBounds sourceBounds1 = params.build();
    ApproximateBounds sourceBounds2 = params.build();
    targetBounds.mergeWith(sourceBounds1.getSerializableSummary());

    // Should not throw:
    targetBounds.mergeWith(sourceBounds2.getSerializableSummary());
  }

  @Test
  public void mergeWith_calledAfterComputeResult_throws() {
    ApproximateBounds targetBounds = params.build();
    targetBounds.addEntry(10.0);
    ApproximateBounds sourceBounds = params.build();
    targetBounds.computeResult();

    IllegalStateException thrown =
        assertThrows(
            IllegalStateException.class,
            () -> targetBounds.mergeWith(sourceBounds.getSerializableSummary()));
    assertThat(thrown).hasMessageThat().contains("result was already computed");
  }

  @Test
  public void mergeWith_calledAfterSerialization_throws() {
    ApproximateBounds targetBounds = params.build();
    ApproximateBounds sourceBounds = params.build();
    targetBounds.getSerializableSummary();

    IllegalStateException thrown =
        assertThrows(
            IllegalStateException.class,
            () -> targetBounds.mergeWith(sourceBounds.getSerializableSummary()));
    assertThat(thrown).hasMessageThat().contains("already serialized");
  }

  private static class SuccessProbabilityProvider extends TestParameterValuesProvider {
    @Override
    public List<Double> provideValues(Context context) {
      return ImmutableList.of(1e-11, 1e-5, 0.1, 0.25, 0.5, 0.75, 0.9, 1 - 1e-5, 1 - 1e-15);
    }
  }

  private static class NumBinsProvider extends TestParameterValuesProvider {
    @Override
    public List<Integer> provideValues(Context context) {
      return ImmutableList.of(1, 2, 4, 1024, 2048);
    }
  }
}

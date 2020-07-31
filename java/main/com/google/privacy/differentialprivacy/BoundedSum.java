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

import static java.lang.Math.abs;
import static java.lang.Math.max;

import com.google.auto.value.AutoValue;
import com.google.common.base.Preconditions;
import com.google.differentialprivacy.Data.ValueType;
import com.google.differentialprivacy.SummaryOuterClass.BoundedSumSummary;
import com.google.protobuf.InvalidProtocolBufferException;
import java.util.Collection;
import javax.annotation.Nullable;

/**
 * Calculates a differentially private sum for a collection of values using the Laplace or Gaussian
 * mechanism.
 *
 * <p>This class allows a single privacy unit (e.g., an individual) to contribute data to multiple
 * different partitions. The class does not check whether the number of partitions is within the
 * specified bounds. This is the responsibility of the caller.
 *
 * <p>This class assumes that each privacy unit may contribute to a single partition only once
 * (i.e., only one data contribution per privacy unit per partition). Multiple contributions from a
 * single privacy unit should be pre-aggregated before they are passed to this class.
 *
 * <p>The user can provide a {@link Noise} instance which will be used to generate the noise. If no
 * instance is specified, {@link LaplaceNoise} is applied.
 *
 * <p>This class provides an unbiased estimator for the raw bounded sum meaning that the expected
 * value of the differentially private bounded sum is equal to the raw bounded sum.
 *
 * <p>Note: this class is not thread-safe.
 *
 * <p>For more implementation details, see {@link #computeResult()}.
 *
 * <p>For general details and key definitions, see <a href=
 * "https://github.com/google/differential-privacy/blob/main/differential_privacy.md#key-definitions">
 * this</a> introduction to Differential Privacy.
 */
public class BoundedSum {

  private final Params params;
  private double sum;

  // Was the sum returned to the user?
  private boolean resultReturned;

  private BoundedSum(Params params) {
    sum = 0.0;
    this.params = params;
  }

  public static Params.Builder builder() {
    return Params.Builder.newBuilder();
  }

  /** Clamps the input value and adds it to the sum. */
  public void addEntry(double e) {
    if (resultReturned) {
      throw new IllegalStateException(
          "The sum has already been calculated and returned. It cannot be amended.");
    }

    // NaN is ignored because introducing even a single NaN entry will result in a NaN sum
    // regardless of other entries, which would break the indistinguishability property required
    // for differential privacy.
    if (Double.isNaN(e)) {
      return;
    }

    sum += clamp(e);
  }

  /** Clamps the input values and adds them to the sum. */
  public void addEntries(Collection<Double> e) {
    e.forEach(this::addEntry);
  }

  private double clamp(double e) {
    if (e > params.upper()) {
      return params.upper();
    }

    if (e < params.lower()) {
      return params.lower();
    }

    return e;
  }

  /**
   * Computes and returns a differentially private sum of the elements added via {@link #addEntry}
   * and {@link #addEntries}. The method can be called only once for a given collection of elements.
   * All subsequent calls will throw an exception.
   *
   * <p>The returned value is an unbiased estimate of the raw bounded sum.
   *
   * <p>The returned value may sometimes be outside the set of possible raw bounded sums, e.g., the
   * differentially private bounded sum may be positive although neither the lower nor the upper
   * bound are positive. This can be corrected by the caller of this method, e.g., by snapping the
   * result to the closest value representing a bounded sum that is possible. Note that such post
   * processing introduces bias to the result.
   */
  public double computeResult() {
    if (resultReturned) {
      throw new IllegalStateException("The result can be calculated and returned only once.");
    }

    resultReturned = true;
    double lInfSensitivity =
        getLInfSensitivity(params.lower(), params.upper(), params.maxContributionsPerPartition());
    return params
        .noise()
        .addNoise(sum, getL0Sensitivity(), lInfSensitivity, params.epsilon(), params.delta());
  }

  /**
   * Returns a serializable version of the current state of {@link BoundedSum} and the parameters
   * used to calculate it. After calling this method, this instance of BoundedSum will be unusable,
   * since the result can only be output once.
   */
  public byte[] getSerializableSummary() {
    if (resultReturned) {
      throw new IllegalStateException(
          "The sum has already been returned. It cannot be returned again.");
    }

    ValueType sumValue = ValueType.newBuilder().setFloatValue(sum).build();
    BoundedSumSummary.Builder builder =
        BoundedSumSummary.newBuilder()
            .setPartialSum(sumValue)
            .setEpsilon(params.epsilon())
            .setLower(params.lower())
            .setUpper(params.upper())
            .setMaxPartitionsContributed(params.maxPartitionsContributed())
            .setMaxContributionsPerPartition(params.maxContributionsPerPartition())
            .setMechanismType(params.noise().getMechanismType());
    if (params.delta() != null) {
      builder.setDelta(params.delta());
    }

    // Record that this object is no longer suitable for producing a differentially private sum,
    // since serialization exposes the object's raw state.
    resultReturned = true;

    return builder.build().toByteArray();
  }

  /**
   * Merges this instance with the output of {@link #getSerializableSummary()} from a different
   * {@link BoundedSum} and stores the merged result in this instance. This is required in the
   * distributed calculations context for merging partial results.
   *
   * @throws IllegalArgumentException if not all config parameters (e.g., epsilon, contribution
   *     bounds) are equal or if the passed serialized sum is invalid.
   * @throws IllegalStateException if this sum has already been calculated or serialized.
   */
  public void mergeWith(byte[] otherBoundedSumSummary) {
    if (resultReturned) {
      throw new IllegalStateException(
          "The sum has already been calculated and returned. It cannot be merged.");
    }

    BoundedSumSummary otherSummaryParsed;
    try {
      otherSummaryParsed = BoundedSumSummary.parseFrom(otherBoundedSumSummary);
    } catch (InvalidProtocolBufferException pbe) {
      throw new IllegalArgumentException(pbe);
    }

    checkMergeParametersAreEqual(otherSummaryParsed);
    this.sum += otherSummaryParsed.getPartialSum().getFloatValue();
  }

  private void checkMergeParametersAreEqual(BoundedSumSummary otherSum) {
    DpPreconditions.checkMergeMechanismTypesAreEqual(
        params.noise().getMechanismType(), otherSum.getMechanismType());
    DpPreconditions.checkMergeEpsilonAreEqual(params.epsilon(), otherSum.getEpsilon());
    DpPreconditions.checkMergeDeltaAreEqual(params.delta(), otherSum.getDelta());
    DpPreconditions.checkMergeMaxPartitionsContributedAreEqual(
        params.maxPartitionsContributed(), otherSum.getMaxPartitionsContributed());
    DpPreconditions.checkMergeMaxContributionsPerPartitionAreEqual(
        params.maxContributionsPerPartition(), otherSum.getMaxContributionsPerPartition());
    DpPreconditions.checkMergeBoundsAreEqual(
        params.lower(), otherSum.getLower(), params.upper(), otherSum.getUpper());
  }

  private static double getLInfSensitivity(
      double lower, double upper, int maxContributionsPerPartition) {
    return max(abs(lower), abs(upper)) * maxContributionsPerPartition;
  }

  private int getL0Sensitivity() {
    // maxPartitionsContributed is the user-facing parameter, which is technically the same as
    // L_0 sensitivity used by the noise internally.
    return params.maxPartitionsContributed();
  }

  @AutoValue
  public abstract static class Params {
    abstract Noise noise();

    abstract double epsilon();

    @Nullable
    abstract Double delta();

    abstract int maxPartitionsContributed();

    abstract int maxContributionsPerPartition();

    abstract double lower();

    abstract double upper();

    @AutoValue.Builder
    public abstract static class Builder {
      private static void checkLInfSensitivityOverflow(
          double lower, double upper, int maxContributionsPerPartition) {
        double lInfSensitivity = getLInfSensitivity(lower, upper, maxContributionsPerPartition);
        Preconditions.checkArgument(
            Double.compare(lInfSensitivity, Double.MAX_VALUE) <= 0,
            "bounds and maxContributionsPerPartition are too high - the LInfSensitivity "
                + " overflows. Provided values: lower bound = %s, upper bound = %s,"
                + " maxContributionsPerPartition = %s",
            lower,
            upper,
            maxContributionsPerPartition);
      }

      private static void checkL1SensitivityOverflow(
          double lower,
          double upper,
          int maxContributionsPerPartition,
          int maxPartitionsContributed) {
        double lInfSensitivity = getLInfSensitivity(lower, upper, maxContributionsPerPartition);
        double l1Sensitivity = Noise.getL1Sensitivity(maxPartitionsContributed, lInfSensitivity);
        Preconditions.checkArgument(
            Double.compare(l1Sensitivity, Double.MAX_VALUE) <= 0,
            "bounds and maxContributionsPerPartition are too high - the L1Sensitivity "
                + " overflows. Provided values: lower bound = %s, upper bound = %s,"
                + " maxContributionsPerPartition = %s",
            lower,
            upper,
            maxContributionsPerPartition);
      }

      private static void checkL2SensitivityOverflow(
          double lower,
          double upper,
          int maxContributionsPerPartition,
          int maxPartitionsContributed) {
        double lInfSensitivity = getLInfSensitivity(lower, upper, maxContributionsPerPartition);
        double l2Sensitivity = Noise.getL2Sensitivity(maxPartitionsContributed, lInfSensitivity);
        Preconditions.checkArgument(
            Double.compare(l2Sensitivity, Double.MAX_VALUE) <= 0,
            "bounds and maxContributionsPerPartition are too high - the L2Sensitivity "
                + " overflows. Provided values: lower bound = %s, upper bound = %s,"
                + " maxContributionsPerPartition = %s",
            lower,
            upper,
            maxContributionsPerPartition);
      }

      private static Builder newBuilder() {
        Params.Builder builder = new AutoValue_BoundedSum_Params.Builder();
        // Provide LaplaceNoise as a default noise generator.
        builder.noise(new LaplaceNoise());
        // By default, assume that each user contributes to a given partition no more than once.
        builder.maxContributionsPerPartition(1);
        return builder;
      }

      /** Epsilon DP parameter. */
      public abstract Builder epsilon(double value);

      /**
       * Delta DP parameter.
       *
       * <p>Note that Laplace noise does not use delta. Hence, delta should not be set when Laplace
       * noise is used.
       */
      public abstract Builder delta(@Nullable Double value);

      /**
       * Maximum number of partitions to which a single privacy unit (i.e., an individual) is
       * allowed to contribute.
       */
      public abstract Builder maxPartitionsContributed(int value);

      /** Distribution from which the noise will be generated and added to the sum. */
      public abstract Builder noise(Noise value);

      /**
       * Lower bound for the entries added to the sum. Any data values below this value will be
       * clamped (i.e., set) to this bound.
       */
      public abstract Builder lower(double value);

      /**
       * Upper bound for the entries added to the sum. Any data values above this value will be
       * clamped (i.e., set) to this bound.
       */
      public abstract Builder upper(double value);

      /**
       * Maximum number of contributions associated with a single privacy unit (e.g., an individual)
       * to a single partition. This is used to calculate the sensitivity of the sum operation. This
       * is not public because it should only be used by other aggregation functions inside the
       * library. See {@link BoundedSum} for more details.
       */
      abstract Builder maxContributionsPerPartition(int value);

      abstract Params autoBuild();

      public BoundedSum build() {
        Params params = autoBuild();
        // No need to check if noise is null: Laplace noise is used by default.
        DpPreconditions.checkEpsilon(params.epsilon());
        DpPreconditions.checkNoiseDelta(params.delta(), params.noise());
        DpPreconditions.checkMaxPartitionsContributed(params.maxPartitionsContributed());
        DpPreconditions.checkMaxContributionsPerPartition(params.maxContributionsPerPartition());
        DpPreconditions.checkBounds(params.lower(), params.upper());

        switch (params.noise().getMechanismType()) {
          case LAPLACE:
            checkL1SensitivityOverflow(
                params.lower(),
                params.upper(),
                params.maxContributionsPerPartition(),
                params.maxPartitionsContributed());
            break;
          case GAUSSIAN:
            checkL2SensitivityOverflow(
                params.lower(),
                params.upper(),
                params.maxContributionsPerPartition(),
                params.maxPartitionsContributed());
            break;
          default:
            throw new IllegalArgumentException(
                "Unable to validate sensitivity overflow: unknown mechanism type: "
                    + params.noise().getMechanismType());
        }
        checkLInfSensitivityOverflow(
            params.lower(), params.upper(), params.maxContributionsPerPartition());

        return new BoundedSum(params);
      }
    }
  }
}

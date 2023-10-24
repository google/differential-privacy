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

import static com.google.common.base.Preconditions.checkArgument;
import static java.lang.Math.abs;
import static java.lang.Math.max;
import static java.lang.Math.min;

import com.google.auto.value.AutoValue;
import com.google.common.base.Preconditions;
import com.google.privacy.differentialprivacy.proto.Data.ValueType;
import com.google.privacy.differentialprivacy.proto.SummaryOuterClass.BoundedSumSummary;
import com.google.protobuf.InvalidProtocolBufferException;
import java.util.Collection;

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
  private double noisedSum;

  private AggregationState state = AggregationState.DEFAULT;

  private BoundedSum(Params params) {
    sum = 0.0;
    this.params = params;
  }

  public static Params.Builder builder() {
    return Params.Builder.newBuilder();
  }

  /**
   * Clamps the input value and adds it to the sum.
   *
   * @throws IllegalStateException if this instance of {@link BoundedSum} has already been queried
   *     or serialized.
   */
  public void addEntry(double e) {
    Preconditions.checkState(state.equals(AggregationState.DEFAULT), "Entry cannot be added.");

    // NaN is ignored because introducing even a single NaN entry will result in a NaN sum
    // regardless of other entries, which would break the indistinguishability property required
    // for differential privacy.
    if (Double.isNaN(e)) {
      return;
    }

    sum += clamp(e);
  }

  /**
   * Clamps the input values and adds them to the sum.
   *
   * @throws IllegalStateException if this instance of {@link BoundedSum} has already been queried
   *     or serialized.
   */
  public void addEntries(Collection<Double> e) {
    e.forEach(this::addEntry);
  }

  private double clamp(double value) {
    return max(min(value, params.upper()), params.lower());
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
   *
   * @throws IllegalStateException if this instance of {@link BoundedSum} has already been queried
   *     or serialized.
   */
  public double computeResult() {
    Preconditions.checkState(state.equals(AggregationState.DEFAULT), "DP sum cannot be computed.");

    state = AggregationState.RESULT_RETURNED;

    noisedSum =
        params
            .noise()
            .addNoise(
                sum, getL0Sensitivity(), getLInfSensitivity(), params.epsilon(), params.delta());

    return noisedSum;
  }

  /**
   * Computes a confidence interval that contains the raw bounded sum with a probability greater or
   * equal to {@code 1 - alpha}. The interval is exclusively based on the noised bounded sum
   * returned by {@link #computeResult}. Thus, no privacy budget is consumed by this operation.
   *
   * <p>Refer to <a
   * href="https://github.com/google/differential-privacy/tree/main/common_docs/confidence_intervals.md">this</a> doc for
   * more information.
   *
   * @throws IllegalStateException if this instance of {@link BoundedSum} has not been queried yet.
   */
  public ConfidenceInterval computeConfidenceInterval(double alpha) {
    Preconditions.checkState(
        state.equals(AggregationState.RESULT_RETURNED), "Confidence interval cannot be computed.");

    ConfidenceInterval confInt =
        params
            .noise()
            .computeConfidenceInterval(
                noisedSum,
                getL0Sensitivity(),
                getLInfSensitivity(),
                params.epsilon(),
                params.delta(),
                alpha);
    if (params.lower() >= 0.0) {
      confInt =
          ConfidenceInterval.create(max(0.0, confInt.lowerBound()), max(0.0, confInt.upperBound()));
    } else if (params.upper() <= 0.0) {
      confInt =
          ConfidenceInterval.create(min(0.0, confInt.lowerBound()), min(0.0, confInt.upperBound()));
    }
    return confInt;
  }

  /**
   * Returns a serializable summary of the current state of this {@link BoundedSum} instance and its
   * parameters. The summary can be used to merge this instance with another instance of {@link
   * BoundedSum}.
   *
   * <p>This method cannot be invoked if the sum has already been queried, i.e., {@link
   * #computeResult()} has been called. Moreover, after this instance of {@link BoundedSum} has been
   * serialized once, further modification and queries are not possible anymore.
   *
   * @throws IllegalStateException if this instance of {@link BoundedSum} has already been queried.
   */
  public byte[] getSerializableSummary() {
    Preconditions.checkState(
        !state.equals(AggregationState.RESULT_RETURNED), "Sum cannot be serialized.");

    state = AggregationState.SERIALIZED;

    ValueType sumValue = ValueType.newBuilder().setFloatValue(sum).build();
    return BoundedSumSummary.newBuilder()
        .setPartialSum(sumValue)
        .setEpsilon(params.epsilon())
        .setDelta(params.delta())
        .setLower(params.lower())
        .setUpper(params.upper())
        .setMaxPartitionsContributed(params.maxPartitionsContributed())
        .setMaxContributionsPerPartition(params.maxContributionsPerPartition())
        .setMechanismType(params.noise().getMechanismType())
        .build()
        .toByteArray();
  }

  /**
   * Merges the output of {@link #getSerializableSummary()} from a different instance of {@link
   * BoundedSum} with this instance. Intended to be used in the context of distributed computation.
   *
   * @throws IllegalArgumentException if the parameters of the two instances (epsilon, delta,
   *     contribution bounds, etc.) do not match or if the passed serialized summary is invalid.
   * @throws IllegalStateException if this instance of {@link BoundedSum} has already been queried
   *     or serialized.
   */
  public void mergeWith(byte[] otherBoundedSumSummary) {
    Preconditions.checkState(state.equals(AggregationState.DEFAULT), "Sums cannot be merged.");

    BoundedSumSummary otherSummaryParsed;
    try {
      otherSummaryParsed = BoundedSumSummary.parseFrom(otherBoundedSumSummary);
    } catch (InvalidProtocolBufferException pbe) {
      throw new IllegalArgumentException(pbe);
    }

    checkMergeParametersAreEqual(otherSummaryParsed);
    this.sum += otherSummaryParsed.getPartialSum().getFloatValue();
  }

  private void checkMergeParametersAreEqual(BoundedSumSummary summary) {
    DpPreconditions.checkMergeMechanismTypesAreEqual(
        params.noise().getMechanismType(), summary.getMechanismType());
    DpPreconditions.checkMergeEpsilonAreEqual(params.epsilon(), summary.getEpsilon());
    DpPreconditions.checkMergeDeltaAreEqual(params.delta(), summary.getDelta());
    DpPreconditions.checkMergeMaxPartitionsContributedAreEqual(
        params.maxPartitionsContributed(), summary.getMaxPartitionsContributed());
    DpPreconditions.checkMergeMaxContributionsPerPartitionAreEqual(
        params.maxContributionsPerPartition(), summary.getMaxContributionsPerPartition());
    DpPreconditions.checkMergeBoundsAreEqual(
        params.lower(), summary.getLower(), params.upper(), summary.getUpper());
  }

  private int getL0Sensitivity() {
    // maxPartitionsContributed is the user-facing parameter, which is technically the same as
    // L_0 sensitivity used by the noise internally.
    return params.maxPartitionsContributed();
  }

  private double getLInfSensitivity() {
    return getLInfSensitivity(
        params.lower(), params.upper(), params.maxContributionsPerPartition());
  }

  private static double getLInfSensitivity(
      double lower, double upper, int maxContributionsPerPartition) {
    return max(abs(lower), abs(upper)) * maxContributionsPerPartition;
  }

  @AutoValue
  public abstract static class Params {
    abstract Noise noise();

    abstract double epsilon();

    abstract double delta();

    abstract int maxPartitionsContributed();

    abstract int maxContributionsPerPartition();

    abstract double lower();

    abstract double upper();

    @AutoValue.Builder
    public abstract static class Builder {
      private static void checkLInfSensitivityOverflow(
          double lower, double upper, int maxContributionsPerPartition) {
        double lInfSensitivity = getLInfSensitivity(lower, upper, maxContributionsPerPartition);
        checkArgument(
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
        checkArgument(
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
        checkArgument(
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
        // Provide LaplaceNoise as a default noise generator. Since it doesn't reqyuire delta,
        // it's by default set to 0.0.
        builder.noise(new LaplaceNoise());
        builder.delta(0.0);
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
      public abstract Builder delta(double value);

      /**
       * @deprecated use {@link #delta(double)}.
       *
       * TODO: migrate clients and delete this method.
       */
      @Deprecated
      public Builder delta(Double value) {
        double primitiveDelta = value == null ? 0.0 : value;
        return delta(primitiveDelta);
      }

      /**
       * Maximum number of partitions to which a single privacy unit (i.e., an individual) is
       * allowed to contribute.
       */
      public abstract Builder maxPartitionsContributed(int value);

      /** Distribution from which the noise will be generated and added to the sum. */
      public abstract Builder noise(Noise value);

      /**
       * Lower bound for the entries added to the sum. Any entires smaller than this value will be
       * set to this value.
       */
      public abstract Builder lower(double value);

      /**
       * Upper bound for the entries added to the sum. Any entires greater than this value will be
       * set to this value.
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
            break;
        }
        checkLInfSensitivityOverflow(
            params.lower(), params.upper(), params.maxContributionsPerPartition());

        return new BoundedSum(params);
      }
    }
  }
}

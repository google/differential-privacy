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

import static java.lang.Math.max;

import com.google.auto.value.AutoValue;
import com.google.common.base.Preconditions;
import com.google.privacy.differentialprivacy.proto.SummaryOuterClass.CountSummary;
import com.google.privacy.differentialprivacy.proto.SummaryOuterClass.MechanismType;
import com.google.protobuf.InvalidProtocolBufferException;
import java.util.Optional;

/**
 * Calculates a differentially private count for a collection of values using the Laplace or
 * Gaussian mechanism.
 *
 * <p>This class allows a single privacy unit (e.g., an individual) to contribute data to multiple
 * different partitions. The class does not check whether the number of partitions is within the
 * specified bounds. This is the responsibility of the caller.
 *
 * <p>This class assumes that each privacy unit may contribute to a single partition only once
 * (i.e., only one data contribution per privacy unit per partition), it doesn't do clamping. For
 * datasets with multiple contributions from the same user to a single partition {@link BoundedSum}
 * should be used instead.
 *
 * <p>The user can provide a {@link Noise} instance which will be used to generate the noise. If no
 * instance is specified, {@link LaplaceNoise} is applied.
 *
 * <p>This class provides an unbiased estimator for the raw count meaning that the expected value of
 * the differentially private count is equal to the raw count.
 *
 * <p>Note: this class is not thread-safe.
 *
 * <p>For more implementation details, see {@link #computeResult()}.
 *
 * <p>For general details and key definitions, see <a
 * href="https://github.com/google/differential-privacy/blob/main/differential_privacy.md#key-definitions">
 * this</a> introduction to Differential Privacy.
 */
public class Count {
  private final Params params;
  private long rawCount;
  private long noisedCount;

  private AggregationState state = AggregationState.DEFAULT;

  private Count(Params params) {
    this.params = params;
  }

  public static Params.Builder builder() {
    return Params.Builder.newBuilder();
  }

  /**
   * Increments count by one.
   *
   * @throws IllegalStateException if this instance of {@link Count} has already been queried or
   *     serialized.
   */
  public void increment() {
    incrementBy(1);
  }

  /**
   * Increments count by the given value. Note, that this shouldn't be used to count multiple
   * contributions to a partition from the same user.
   *
   * <p>Note that decrementing counts by inputting a negative value is allowed, for example if you
   * want to remove some users you have previously added.
   *
   * @throws IllegalStateException if this instance of {@link Count} has already been queried or
   *     serialized.
   */
  public void incrementBy(long count) {
    Preconditions.checkState(
        state.equals(AggregationState.DEFAULT), "Count cannot be incremented.");
    this.rawCount += count;
  }

  /**
   * Calculates and returns a differentially private count of elements added using {@link
   * #increment} and {@link #incrementBy}. The method can be called only once for a given collection
   * of elements. All subsequent calls will throw an exception.
   *
   * <p>The returned value is an unbiased estimate of the raw count.
   *
   * <p>The returned value may sometimes be negative. This can be corrected by setting negative
   * results to 0. Note that such post processing introduces bias to the result.
   *
   * @throws IllegalStateException if this instance of {@link Count} has already been queried or
   *     serialized.
   */
  public long computeResult() {
    Preconditions.checkState(
        state.equals(AggregationState.DEFAULT), "DP count cannot be computed.");

    state = AggregationState.RESULT_RETURNED;

    noisedCount =
        params
            .noise()
            .addNoise(
                rawCount,
                params.maxPartitionsContributed(),
                params.maxContributionsPerPartition(),
                params.epsilon(),
                params.delta());
    return noisedCount;
  }

  /**
   * Computes a confidence interval that contains the raw count with a probability greater or equal
   * to {@code 1 - alpha}. The interval is exclusively based on the noised count returned by {@link
   * #computeResult}. Thus, no privacy budget is consumed by this operation.
   *
   * <p>Refer to <a
   * href="https://github.com/google/differential-privacy/tree/main/common_docs/confidence_intervals.md">this</a> doc for
   * more information.
   *
   * @throws IllegalStateException if this instance of {@link Count} has not been queried yet.
   */
  public ConfidenceInterval computeConfidenceInterval(double alpha) {
    Preconditions.checkState(
        state.equals(AggregationState.RESULT_RETURNED), "Confidence interval cannot be computed.");

    ConfidenceInterval confInt =
        params
            .noise()
            .computeConfidenceInterval(
                noisedCount,
                params.maxPartitionsContributed(),
                params.maxContributionsPerPartition(),
                params.epsilon(),
                params.delta(),
                alpha);
    return ConfidenceInterval.create(
        max(0.0, confInt.lowerBound()), max(0.0, confInt.upperBound()));
  }

  /**
   * Returns either of {@link #computeResult} or {@link Optional#empty}. The result is (epsilon,
   * noiseDelta + thresholdDelta)-differentially private assuming that empty counts are not
   * published. The method can be called only once for a given collection of elements. All
   * subsequent calls will throw an exception.
   *
   * <p>To ensure that the boolean signal of a count's publication satisfies (0,
   * thresholdDelta)-differential privacy, noised counts smaller than an appropriately set threshold
   * k > 0 are returned as {@link Optional#empty}. It is the responsibility of the caller of this
   * method to ensure that a count that returned empty is not published.
   *
   * @param thresholdDelta the privacy budget spent on publishing non-empty counts.
   */
  public Optional<Long> computeThresholdedResult(double thresholdDelta) {
    DpPreconditions.checkDelta(thresholdDelta);

    long noisyCount = computeResult();

    // The implementation will work only for symmetrical noise.
    Preconditions.checkState(
        params.noise().getMechanismType() == MechanismType.LAPLACE
            || params.noise().getMechanismType() == MechanismType.GAUSSIAN,
        "Unable to calculate the threshold for an unknown mechanism type %s",
        params.noise().getMechanismType());

    // Let C_1,...,C_n be the noised counts to which a given privacy unit u contributed and to which
    // no other privacy unit contributed. The privacy parameter thresholdDelta is the worst case
    // probability with which at least one C_i exceeds the threshold k (because that would be a
    // distinguishing outcome from a dataset without u). More formally, we have
    //    thresholdDelta ≥ P[C_1 ≥ k ∨ ... ∨ C_n ≥ k]
    // The probability on the right is maximized (and thus equal to thresholdDelta) when u
    // contributes to the maximum number of counts, i.e., when n = maxPartitionsContributed, and
    // when u contributes the maximum amount to each count, i.e., maxContributionsPerPartition. In
    // this case, all probabilities P[C_i ≥ k] become the same and we refer to them as
    // thresholdDeltaPerPartition.
    //
    // Based on this worst case analysis, thresholdDelta can be expressed as
    //    thresholdDelta = P[C_1 ≥ k ∨ ... ∨ C_n ≥ k]
    //                   = 1 - P[C_1 < k ∧ ... ∧ C_n < k]
    //                   = 1 - (P[C_1 < k] * ... * P[C_n < k])
    //                   = 1 - ((1 - P[C_1 ≥ k]) * ... * (1 - P[C_n ≥ k]))
    //                   = 1 - (1 - thresholdDeltaPerPartition)^n,
    // Solving for thresholdDeltaPerPartition yields
    //    thresholdDeltaPerPartition = 1 - (1 - thresholdDelta)^(1 / n)
    //
    // Computing thresholdDeltaPerPartition this way is numerically inaccurate for small values of
    // thresholdDelta (which is the input we usually expect). The reason is that 1 - thresholdDelta
    // is close to 1 and thus its floating point representation loses precision. To bypass this
    // issue we compute thresholdDeltaPerPartition as
    //    thresholdDeltaPerPartition = -expm1(log1p(-thresholdDelta) / n)
    // instead, which is mathematically equivalent because
    //    -expm1(log1p(-thresholdDelta) / n) = -expm1(log(1 - thresholdDelta) / n)
    //                                       = -expm1(log((1 - thresholdDelta)^(1 / n)))
    //                                       = 1 - exp(log((1 - thresholdDelta)^(1 / n)))
    //                                       = 1 - (1 - thresholdDelta)^(1 / n).
    double thresholdDeltaPerPartition =
        -1.0 * Math.expm1(Math.log1p(-thresholdDelta) / params.maxPartitionsContributed());

    // Computing the threshold k for which P[C_i ≥ k] = thresholdDeltaPerPartition is equivalent to
    // computing the quantile q_r of the distribution of C_i for rank
    // r = (1 - thresholdDeltaPerPartition).
    //
    // Since r is close to 1 for small values of thresholdDeltaPerPartition this approach may cause
    // numerical issues. Instead, we look directly at the noise
    // nu_i = C_i - maxContributionsPerPartition that is added to the count C_i and compute the
    // quantile q'_r' for rank r' = thresholdDeltaPerPartition. Because the distribution of nu_i is
    // symmetric around 0.0 it holds that
    //   k = q_r = maxContributionsPerPartition - q'_r'.
    double threshold =
        params.maxContributionsPerPartition()
            - params
                .noise()
                .computeQuantile(
                    /* rank= */ thresholdDeltaPerPartition,
                    /* x= */ 0.0,
                    params.maxPartitionsContributed(),
                    params.maxContributionsPerPartition(),
                    params.epsilon(),
                    params.delta());
    if (Double.compare((double) noisyCount, threshold) >= 0) {
      return Optional.of(noisyCount);
    } else {
      return Optional.empty();
    }
  }

  /**
   * Returns a serializable summary of the current state of this {@link Count} instance and its
   * parameters. The summary can be used to merge this instance with another instance of {@link
   * Count}.
   *
   * <p>This method cannot be invoked if the count has already been queried, i.e., {@link
   * #computeResult()} has been called. Moreover, after this instance of {@link Count} has been
   * serialized once, further modification and queries are not possible anymore.
   *
   * @throws IllegalStateException if this instance of {@link Count} has already been queried.
   */
  public byte[] getSerializableSummary() {
    Preconditions.checkState(
        state != AggregationState.RESULT_RETURNED, "Count cannot be serialized.");

    state = AggregationState.SERIALIZED;

    return CountSummary.newBuilder()
        .setCount(rawCount)
        .setEpsilon(params.epsilon())
        .setDelta(params.delta())
        .setMaxPartitionsContributed(params.maxPartitionsContributed())
        .setMaxContributionsPerPartition(params.maxContributionsPerPartition())
        .setMechanismType(params.noise().getMechanismType())
        .build()
        .toByteArray();
  }

  /**
   * Merges the output of {@link #getSerializableSummary()} from a different instance of {@link
   * Count} with this instance. Intended to be used in the context of distributed computation.
   *
   * @throws IllegalArgumentException if the parameters of the two instances (epsilon, delta,
   *     contribution bounds, etc.) do not match or if the passed serialized summary is invalid.
   * @throws IllegalStateException if this instance of {@link Count} has already been queried or
   *     serialized.
   */
  public void mergeWith(byte[] otherCountSummary) {
    Preconditions.checkState(state.equals(AggregationState.DEFAULT), "Counts cannot be merged.");

    CountSummary otherSummaryParsed;
    try {
      otherSummaryParsed = CountSummary.parseFrom(otherCountSummary);
    } catch (InvalidProtocolBufferException pbe) {
      throw new IllegalArgumentException(pbe);
    }

    checkMergeParametersAreEqual(otherSummaryParsed);
    this.rawCount += otherSummaryParsed.getCount();
  }

  private void checkMergeParametersAreEqual(CountSummary summary) {
    DpPreconditions.checkMergeMechanismTypesAreEqual(
        params.noise().getMechanismType(), summary.getMechanismType());
    DpPreconditions.checkMergeEpsilonAreEqual(params.epsilon(), summary.getEpsilon());
    DpPreconditions.checkMergeDeltaAreEqual(params.delta(), summary.getDelta());
    DpPreconditions.checkMergeMaxPartitionsContributedAreEqual(
        params.maxPartitionsContributed(), summary.getMaxPartitionsContributed());
    DpPreconditions.checkMergeMaxContributionsPerPartitionAreEqual(
        params.maxContributionsPerPartition(), summary.getMaxContributionsPerPartition());
  }

  @AutoValue
  public abstract static class Params {
    abstract Noise noise();

    abstract double epsilon();

    abstract double delta();

    abstract int maxPartitionsContributed();

    abstract int maxContributionsPerPartition();

    @AutoValue.Builder
    public abstract static class Builder {
      private static Builder newBuilder() {
        Builder builder = new AutoValue_Count_Params.Builder();
        // Provide LaplaceNoise as a default noise generator.
        return builder
            .noise(new LaplaceNoise())
            // Laplace noise doesn't use delta. Hence, set it to 0.
            .delta(0.0)
            // By default, assume that each user contributes to a given partition no more than once.
            .maxContributionsPerPartition(1);
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
       *     <p>TODO: migrate clients and delete this method.
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

      /** Distribution from which the noise will be generated and added to the count. */
      public abstract Builder noise(Noise value);

      /**
       * Maximum number of contributions associated with a single privacy unit (e.g., an individual)
       * to a single partition. This is used to calculate the sensitivity of the count operation.
       * This is not public because it should be used only by other aggregation functions inside the
       * library. See {@link Count} for more details.
       */
      abstract Builder maxContributionsPerPartition(int value);

      abstract Params autoBuild();

      public Count build() {
        Params params = autoBuild();
        // No need to check if noise is null: Laplace noise is used by default.
        DpPreconditions.checkEpsilon(params.epsilon());
        DpPreconditions.checkNoiseDelta(params.delta(), params.noise());
        DpPreconditions.checkMaxPartitionsContributed(params.maxPartitionsContributed());
        DpPreconditions.checkMaxContributionsPerPartition(params.maxContributionsPerPartition());

        return new Count(params);
      }
    }
  }
}

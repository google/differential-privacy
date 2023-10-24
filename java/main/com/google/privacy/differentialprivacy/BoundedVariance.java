//
// Copyright 2021 Google LLC
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

package com.google.privacy.differentialprivacy;

import static java.lang.Math.max;
import static java.lang.Math.min;
import static java.lang.Math.pow;

import com.google.auto.value.AutoValue;
import com.google.common.base.Preconditions;
import com.google.privacy.differentialprivacy.proto.SummaryOuterClass.BoundedSumSummary;
import com.google.privacy.differentialprivacy.proto.SummaryOuterClass.BoundedVarianceSummary;
import com.google.privacy.differentialprivacy.proto.SummaryOuterClass.CountSummary;
import com.google.protobuf.InvalidProtocolBufferException;
import java.util.Collection;

/**
 * Calculates differentially private variance for a collection of values.
 *
 * <p>The variance is a biased estimate and is computed as difference between the noisy variance of
 * squares and square of the noisy variance. To improve utility, all entries are normalized by
 * setting them to the difference between their actual value and the middle of the input range
 * before summation.
 *
 * <p>Supports contributions from a single privacy unit to multiple partitions as well as multiple
 * contributions from a single privacy unit to a given partition.
 *
 * <p>The user can provide a {@link Noise} instance which will be used to generate the noise. If no
 * instance is specified, {@link LaplaceNoise} is applied.
 *
 * <p>Note: the class is not thread-safe.
 *
 * <p>For more implementation details, see {@link #computeResult()}.
 *
 * <p>For general details and key definitions, see <a href=
 * "https://github.com/google/differential-privacy/blob/main/differential_privacy.md#key-definitions">
 * this</a> introduction to Differential Privacy.
 */
public final class BoundedVariance {
  private final Params params;
  private final Count count;
  private final BoundedSum normalizedSum;
  private final BoundedSum normalizedSumOfSquares;
  private final double midpoint;
  private AggregationState state = AggregationState.DEFAULT;

  private BoundedVariance(BoundedVariance.Params params) {
    this.params = params;

    // Note: we don't calculate the midpoint as "(lower + upper) / 2" to avoid overflow.
    midpoint = params.lower() * 0.5 + params.upper() * 0.5;

    double maxDistFromMidpoint = params.upper() - midpoint;

    // We split the budget equally in three to calculate the count, the normalized sum and
    // normalized sum of squares.
    // TODO: This can be optimized.
    double countEpsilon = params.epsilon() / 3;
    double sumEpsilon = params.epsilon() / 3;
    double sumOfSquaresEpsilon = params.epsilon() - countEpsilon - sumEpsilon;
    double countDelta = params.delta() / 3;
    double sumDelta = params.delta() / 3;
    double sumOfSquaresDelta = params.delta() - countDelta - sumDelta;

    // Check that the parameters are compatible with the noise chosen by calling
    // the noise on some dummy value.
    double unused1 = params.noise().addNoise(0, 1, 1, countEpsilon, countDelta);
    double unused2 = params.noise().addNoise(0, 1, 1, sumEpsilon, sumDelta);
    double unused3 = params.noise().addNoise(0, 1, 1, sumOfSquaresEpsilon, sumOfSquaresDelta);

    // normalizedSumOfSquares s2 yields a differentially private sum of squares of the position of
    // the entries e_i relative to the midpoint m = (lower + upper) / 2 of the range of the bounded
    // variance, i.e., s2 = Σ_i (e_i - m) (e_i - m).
    //
    // normalizedSum s yields a differentially private sum of the position of the entries e_i
    // relative to the midpoint m = (lower + upper) / 2 of the range of the bounded variance, i.e.,
    // s = Σ_i (e_i - m).
    //
    // count c yields a differentially private count of the entries.
    //
    // Given normalized sum of squares s2, normalized sum s and count c (all without noise), the
    // true variance can be computed as (since variance is invariant to translation):
    // variance = s2 / c - (s / c)^2
    //
    // the rest follows from the code.
    count =
        Count.builder()
            .noise(params.noise())
            .epsilon(countEpsilon)
            .delta(countDelta)
            .maxPartitionsContributed(params.maxPartitionsContributed())
            .maxContributionsPerPartition(params.maxContributionsPerPartition())
            .build();

    normalizedSum =
        BoundedSum.builder()
            .noise(params.noise())
            .epsilon(sumEpsilon)
            .delta(sumDelta)
            .maxPartitionsContributed(params.maxPartitionsContributed())
            .maxContributionsPerPartition(params.maxContributionsPerPartition())
            .lower(-maxDistFromMidpoint)
            .upper(maxDistFromMidpoint)
            .build();

    normalizedSumOfSquares =
        BoundedSum.builder()
            .noise(params.noise())
            .epsilon(sumOfSquaresEpsilon)
            .delta(sumOfSquaresDelta)
            .maxPartitionsContributed(params.maxPartitionsContributed())
            .maxContributionsPerPartition(params.maxContributionsPerPartition())
            .lower(0)
            .upper(Math.pow(maxDistFromMidpoint, 2))
            .build();
  }

  public static BoundedVariance.Params.Builder builder() {
    return BoundedVariance.Params.Builder.newBuilder();
  }

  /**
   * Clamps the input value and adds it to the variance.
   *
   * @throws IllegalStateException if this instance of {@link BoundedVariance} has already been
   *     queried or serialized.
   */
  public void addEntry(double e) {
    Preconditions.checkState(state.equals(AggregationState.DEFAULT), "Entry cannot be added.");

    if (Double.isNaN(e)) {
      return;
    }

    double clampedE = clamp(e);
    count.increment();
    normalizedSum.addEntry(clampedE - midpoint);
    normalizedSumOfSquares.addEntry(pow(clampedE - midpoint, 2));
  }

  /**
   * Clamps the input values and adds them to the variance.
   *
   * @throws IllegalStateException if this instance of {@link BoundedVariance} has already been
   *     queried or serialized.
   */
  public void addEntries(Collection<Double> e) {
    e.forEach(this::addEntry);
  }

  /**
   * Calculates and returns differentially private variance of elements added using {@link
   * #addEntry} and {@link #addEntries}. The method can be called only once for a given instance of
   * this class. All subsequent calls will result in throwing an exception.
   *
   * <p>Note that the returned value is not an unbiased estimate of the raw bounded variance.
   *
   * @throws IllegalStateException if this instance of {@link BoundedVariance} has already been
   *     queried or serialized.
   */
  public double computeResult() {
    Preconditions.checkState(
        state.equals(AggregationState.DEFAULT), "DP variance cannot be computed.");

    state = AggregationState.RESULT_RETURNED;
    long noisedCount = max(1, count.computeResult());
    double normalizedNoisedSum = normalizedSum.computeResult();
    double normalizedNoisedSumOfSquares = normalizedSumOfSquares.computeResult();
    double noisedVariance =
        normalizedNoisedSumOfSquares / noisedCount - pow(normalizedNoisedSum / noisedCount, 2);
    return clampVariance(noisedVariance);
  }

  /**
   * Returns a serializable summary of the current state of this {@link BoundedVariance} instance
   * and its parameters. The summary can be used to merge this instance with another instance of
   * {@link BoundedVariance}.
   *
   * <p>This method cannot be invoked if the variance has already been queried, i.e., {@link
   * #computeResult} has been called. Moreover, after this instance of {@link BoundedVariance} has
   * been serialized once, further modification and queries are not possible anymore.
   *
   * @throws IllegalStateException if this instance of {@link BoundedVariance} has already been
   *     queried.
   */
  public byte[] getSerializableSummary() {
    Preconditions.checkState(
        state != AggregationState.RESULT_RETURNED, "Variance cannot be serialized.");

    state = AggregationState.SERIALIZED;

    CountSummary deserializedCount;
    BoundedSumSummary deserializedNormalizedSum;
    BoundedSumSummary deserializedNormalizedSumOfSquares;
    try {
      deserializedCount = CountSummary.parseFrom(count.getSerializableSummary());
      deserializedNormalizedSum =
          BoundedSumSummary.parseFrom(normalizedSum.getSerializableSummary());
      deserializedNormalizedSumOfSquares =
          BoundedSumSummary.parseFrom(normalizedSumOfSquares.getSerializableSummary());
    } catch (InvalidProtocolBufferException e) {
      throw new IllegalStateException("Variance object cannot be serialized. Reason: " + e);
    }

    BoundedVarianceSummary serializedVariance =
        BoundedVarianceSummary.newBuilder()
            .setCountSummary(deserializedCount)
            .setSumSummary(deserializedNormalizedSum)
            .setSumOfSquaresSummary(deserializedNormalizedSumOfSquares)
            .build();

    return serializedVariance.toByteArray();
  }

  /**
   * Merges the output of {@link #getSerializableSummary()} from a different instance of {@link
   * BoundedVariance} with this instance. Intended to be used in the context of distributed
   * computation.
   *
   * @throws IllegalArgumentException if the parameters of the two instances (epsilon, delta,
   *     contribution bounds, etc.) do not match or if the passed serialized summary is invalid.
   * @throws IllegalStateException if this instance of {@link BoundedVariance} has already been
   *     queried or serialized.
   */
  public void mergeWith(byte[] otherBoundedVarianceSummary) {
    Preconditions.checkState(state.equals(AggregationState.DEFAULT), "Variances cannot be merged.");

    BoundedVarianceSummary otherSummaryParsed;
    try {
      otherSummaryParsed = BoundedVarianceSummary.parseFrom(otherBoundedVarianceSummary);
    } catch (InvalidProtocolBufferException pbe) {
      throw new IllegalArgumentException(pbe);
    }

    this.normalizedSumOfSquares.mergeWith(
        otherSummaryParsed.getSumOfSquaresSummary().toByteArray());
    this.normalizedSum.mergeWith(otherSummaryParsed.getSumSummary().toByteArray());
    this.count.mergeWith(otherSummaryParsed.getCountSummary().toByteArray());
  }

  /** Parameters of {@link BoundedVariance}. */
  @AutoValue
  public abstract static class Params {
    abstract Noise noise();

    abstract double epsilon();

    abstract double delta();

    abstract int maxPartitionsContributed();

    abstract int maxContributionsPerPartition();

    abstract double lower();

    abstract double upper();

    /** Builder for parameters of {@link BoundedVariance}. */
    @AutoValue.Builder
    public abstract static class Builder {
      private static BoundedVariance.Params.Builder newBuilder() {
        BoundedVariance.Params.Builder builder = new AutoValue_BoundedVariance_Params.Builder();
        // Provides LaplaceNoise as a default noise generator.
        builder.noise(new LaplaceNoise());
        // Since Laplace noise doesn't use delta, set it to 0.0.
        builder.delta(0.0);

        return builder;
      }

      /** Noise that will be used to make the variance differentially private. */
      public abstract BoundedVariance.Params.Builder noise(Noise value);

      /** Epsilon DP parameter. */
      public abstract BoundedVariance.Params.Builder epsilon(double value);

      /**
       * Delta DP parameter.
       *
       * <p>Note that Laplace noise does not use delta. Hence, delta should not be set when Laplace
       * noise is used.
       */
      public abstract BoundedVariance.Params.Builder delta(double value);

      /**
       * @deprecated use {@link #delta(double)}.
       *     <p>TODO: migrate clients and delete this method.
       */
      @Deprecated
      public BoundedVariance.Params.Builder delta(Double value) {
        double primitiveDelta = value == null ? 0.0 : value;
        return delta(primitiveDelta);
      }

      /**
       * Maximum number of partitions that a single privacy unit (e.g., an individual) is allowed to
       * contribute to.
       */
      public abstract BoundedVariance.Params.Builder maxPartitionsContributed(int value);

      /** Max contributions per partition from a single privacy unit (e.g., an individual). */
      public abstract BoundedVariance.Params.Builder maxContributionsPerPartition(int value);

      /**
       * Lower bound for the entries added to the variance. Any entries smaller than this value will
       * be set to this value.
       */
      public abstract BoundedVariance.Params.Builder lower(double value);

      /**
       * Upper bound for the entries added to the variance. Any entries greater than this value will
       * be set to this value.
       */
      public abstract BoundedVariance.Params.Builder upper(double value);

      abstract BoundedVariance.Params autoBuild();

      public BoundedVariance build() {
        BoundedVariance.Params params = autoBuild();
        // No need to check noise nullability: the noise is defaulted to Laplace noise.
        DpPreconditions.checkEpsilon(params.epsilon());
        DpPreconditions.checkNoiseDelta(params.delta(), params.noise());
        DpPreconditions.checkMaxPartitionsContributed(params.maxPartitionsContributed());
        DpPreconditions.checkMaxContributionsPerPartition(params.maxContributionsPerPartition());
        DpPreconditions.checkBounds(params.lower(), params.upper());
        DpPreconditions.checkBoundsNotEqual(params.lower(), params.upper());

        return new BoundedVariance(params);
      }
    }
  }

  private double clamp(double value) {
    return max(min(value, params.upper()), params.lower());
  }

  private double clampVariance(double value) {
    return max(min(value, maxVariance()), 0);
  }

  private double maxVariance() {
    return pow(params.upper() - params.lower(), 2) / 4;
  }
}

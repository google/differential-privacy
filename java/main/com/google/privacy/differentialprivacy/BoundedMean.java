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

import com.google.auto.value.AutoValue;
import com.google.common.annotations.VisibleForTesting;
import java.util.Collection;
import javax.annotation.Nullable;


/**
 * Calculates differentially private average for a collection of values.
 *
 * <p>The mean is computed by dividing a noisy sum of the entries by a noisy count of the entries.
 * To improve utility, all entries are normalized by setting them to the difference between their
 * actual value and the middle of the input range before summation. The original mean is recovered
 * by adding the midpoint in a post processing step. This idea is taken from Algorithm 2.4 of
 * "Differential Privacy: From Theory to Practice", by Ninghui Li, Min Lyu, Dong Su and Weining Yang
 * (section 2.5.5, page 28). In contrast to Algorithm 2.4, we do not return the midpoint if the
 * noisy count is less or equal to 1. Instead we set the noisy count to 1. Since this is a mere post
 * processing step, the DP bounds are preserved. Moreover, for small numbers of entries, this
 * approach will return results that are closer to the actual mean in expectation.
 *
 * <p>Ninghui Li, Min Lyu, Dong Su and Weining Yang also propose Algorithm 2.3 for computing private
 * means, which according to them yields better accuracy. However, the proof of the Algorithm 2.3 is
 * flawed and it is not actually DP.
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
public class BoundedMean {
  private final BoundedMean.Params params;
  private final BoundedSum normalizedSum;
  private final Count count;
  /**
   * The midpoint between lower and upper bounds. It cannot be set by the user: it will be
   * calculated based on the {@link Params#lower()} and {@link Params#upper()} values.
   */
  private final double midpoint;

  private AggregationState state = AggregationState.DEFAULT;

  private BoundedMean(BoundedMean.Params params) {
    this.params = params;

    // Note: we don't calculate the midpoint as "(lower + upper) / 2" to avoid overflow.
    midpoint = params.lower() * 0.5 + params.upper() * 0.5;

    double maxDistFromMidpoint = Math.abs(params.upper() - midpoint);

    // We split the budget in half to calculate count and noised normalized sum
    double halfEpsilon = params.epsilon() * 0.5;
    Double halfDelta = params.delta() == null ? null : params.delta() * 0.5;

    // normalizedSum yields a differentially private sum of the position of the entries e_i relative
    // to the midpoint m = (lower + upper) / 2 of the range of the bounded mean, i.e., Σ_i (e_i - m)
    //
    // count yields a differentially private count of the entries.
    //
    // Given a normalized sum s and count c (both without noise), the true mean can be computed
    // as: mean =
    //   s / c + m =
    //   (Σ_i (e_i - m)) / c + m =
    //   (Σ_i (e_i - m)) / c + (Σ_i m) / c =
    //   (Σ_i e_i) / c
    //
    // the rest follows from the code.
    normalizedSum =
        BoundedSum.builder()
            .noise(params.noise())
            .epsilon(halfEpsilon)
            // TODO: this can be optimized for the Gaussian noise
            .delta(halfDelta)
            .maxPartitionsContributed(params.maxPartitionsContributed())
            .maxContributionsPerPartition(params.maxContributionsPerPartition())
            .lower(-maxDistFromMidpoint)
            .upper(maxDistFromMidpoint)
            .build();
    // Noised count of the entities.
    count =
        Count.builder()
            .noise(params.noise())
            .epsilon(halfEpsilon)
            // TODO: this can be optimized for the Gaussian noise
            .delta(halfDelta)
            .maxPartitionsContributed(params.maxPartitionsContributed())
            .maxContributionsPerPartition(params.maxContributionsPerPartition())
            .build();
  }

  public static BoundedMean.Params.Builder builder() {
    return BoundedMean.Params.Builder.newBuilder();
  }

  /** Clamps the input value and adds it to the average. */
  public void addEntry(double e) {
    if (state != AggregationState.DEFAULT) {
      throw new IllegalStateException("Mean cannot be amended. Reason: " + state.getErrorMessage());
    }

    // NaN is ignored because introducing even a single NaN entry will result in a NaN mean
    // regardless of other entries, which would break the indistinguishability
    // property required for differential privacy.
    if (Double.isNaN(e)) {
      return;
    }

    // BoundedSum will also attempt to clamp the input value but we do it here for transparency.
    normalizedSum.addEntry(clamp(e) - midpoint);

    count.increment();
  }

  /** Clamps the input values and adds them to the average. */
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
   * Calculates and returns differentially private average of elements added using {@link #addEntry}
   * and {@link #addEntries}. The method can be called only once for a given collection of elements.
   * All subsequent calls will result in throwing an exception.
   *
   * <p>Note that the returned value is not an unbiased estimate of the raw bounded mean.
   */
  public double computeResult() {
    if (state != AggregationState.DEFAULT) {
      throw new IllegalStateException(
          "Mean's noised result cannot be computed. Reason: " + state.getErrorMessage());
    }

    state = AggregationState.RESULT_RETURNED;

    long noisedCount = Math.max(1, count.computeResult());
    double normalizedNoisedSum = normalizedSum.computeResult();

    // Clamp the average before returning it to ensure it does not exceed the lower and upper
    // bounds.
    return clamp(normalizedNoisedSum / noisedCount + midpoint);
  }

  /**
   * Computes a confidence interval that contains the true mean with a probability greater or equal
   * to {@code 1 - alpha}. The computation is based exclusively on the noised data and the privacy
   * parameters. Thus no privacy budget is consumed by this operation.
   *
   * <p>See <a href="https://github.com/google/differential-privacy/tree/main/common_docs/confidence_intervals.md">
   * the confidence intervals doc</a>.
   */
  public ConfidenceInterval computeConfidenceInterval(double alpha) {
    if (state != AggregationState.RESULT_RETURNED) {
      throw new IllegalStateException("computeResult() must be called before calling computeConfidenceInterval()");
    }
    // The confidence interval of bounded mean is derived from confidence intervals of the mean's
    // numerator and denominator. The respective confidence levels 1 - alphaNum and 1 - alphaDen can
    // be chosen arbitrarily as long as
    //   (1 - alphaNum) * (1 - alphaDen) = 1 - alpha.
    // The following is a brute force search for alphaNum that minimizes the size of the confidence
    // interval of bounded mean.
    double minSize = Double.POSITIVE_INFINITY;
    ConfidenceInterval tightestConfInt = null;
    for (int i = 1; i < 1000; i++) {
      double alphaNum = (i / 1000.0) * alpha;
      ConfidenceInterval confInt = computeConfidenceInterval(alpha, alphaNum);
      double size = confInt.upperBound() - confInt.lowerBound();
      if (size < minSize) {
        minSize = size;
        tightestConfInt = confInt;
      }
    }
    return tightestConfInt;
  }

  /**
   * Like {@link #computeConfidenceInterval(double)} with the additional constraint that the
   * confidence level of the mean's numerator is {@code 1 - alphaNum}.
   */
  @VisibleForTesting
  ConfidenceInterval computeConfidenceInterval(double alpha, double alphaNum) {
    // Setting alphaDen such that (1 - alpha) = (1 - alphaNum) * (1 - alphaDen).
    double alphaDen = (alpha - alphaNum) / (1 - alphaNum);
    ConfidenceInterval confIntNum = normalizedSum.computeConfidenceInterval(alphaNum);
    ConfidenceInterval confIntDen = count.computeConfidenceInterval(alphaDen);

    // Ensuring that the lower and upper bounds of the denominator are consistent with how
    // computeResult() processes the denominator.
    confIntDen =
        ConfidenceInterval.create(
            Math.max(1.0, confIntDen.lowerBound()), Math.max(1.0, confIntDen.upperBound()));

    double meanLowerBound;
    double meanUpperBound;
    if (confIntNum.lowerBound() >= 0.0) {
      meanLowerBound = confIntNum.lowerBound() / confIntDen.upperBound();
    } else {
      meanLowerBound = confIntNum.lowerBound() / confIntDen.lowerBound();
    }
    if (confIntNum.upperBound() >= 0.0) {
      meanUpperBound = confIntNum.upperBound() / confIntDen.lowerBound();
    } else {
      meanUpperBound = confIntNum.upperBound() / confIntDen.upperBound();
    }

    // Ensuring that the lower and upper bounds of the mean are consistent with how computeResult()
    // processes the mean.
    meanLowerBound = clamp(meanLowerBound + midpoint);
    meanUpperBound = clamp(meanUpperBound + midpoint);
    return ConfidenceInterval.create(meanLowerBound, meanUpperBound);
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

      private static BoundedMean.Params.Builder newBuilder() {
        BoundedMean.Params.Builder builder = new AutoValue_BoundedMean_Params.Builder();
        // Provides LaplaceNoise as a default noise generator.
        builder.noise(new LaplaceNoise());

        return builder;
      }

      /** Epsilon DP parameter. */
      public abstract BoundedMean.Params.Builder epsilon(double value);

      /**
       * Delta DP parameter.
       *
       * <p>Note that Laplace noise does not use delta. Hence, delta should not be set when Laplace
       * noise is used.
       */
      public abstract BoundedMean.Params.Builder delta(@Nullable Double value);

      /**
       * Maximum number of partitions that a single privacy unit (e.g., an individual) is allowed to
       * contribute to.
       */
      public abstract BoundedMean.Params.Builder maxPartitionsContributed(int value);

      /** Max contributions per partition from a single privacy unit (e.g., an individual). */
      public abstract BoundedMean.Params.Builder maxContributionsPerPartition(int value);

      /** Noise that will be used to make the mean differentially private. */
      public abstract BoundedMean.Params.Builder noise(Noise value);

      /**
       * Lower bound for the entries added to the mean. Lower values will be clamped to this bound.
       */
      public abstract BoundedMean.Params.Builder lower(double value);

      /**
       * Higher bound for the entries added to the mean. Greater values will be clamped to this
       * bound.
       */
      public abstract BoundedMean.Params.Builder upper(double value);

      abstract BoundedMean.Params autoBuild();

      public BoundedMean build() {
        BoundedMean.Params params = autoBuild();
        // No need to check noise nullability: the noise is defaulted to Laplace noise.
        DpPreconditions.checkEpsilon(params.epsilon());
        DpPreconditions.checkNoiseDelta(params.delta(), params.noise());
        DpPreconditions.checkL0Sensitivity(params.maxPartitionsContributed());
        DpPreconditions.checkMaxContributionsPerPartition(params.maxContributionsPerPartition());
        DpPreconditions.checkBounds(params.lower(), params.upper());

        return new BoundedMean(params);
      }
    }
  }
}

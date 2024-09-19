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
import static com.google.common.base.Preconditions.checkState;
import static java.lang.Math.abs;
import static java.lang.Math.ceil;
import static java.lang.Math.expm1;
import static java.lang.Math.log;
import static java.lang.Math.log1p;
import static java.lang.Math.max;
import static java.lang.Math.min;

import com.google.auto.value.AutoValue;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.primitives.Longs;
import com.google.privacy.differentialprivacy.proto.SummaryOuterClass.ApproxBoundsSummary;
import com.google.protobuf.ExtensionRegistry;
import com.google.protobuf.InvalidProtocolBufferException;
import java.security.SecureRandom;
import java.util.Collection;
import java.util.Optional;
import java.util.Random;

/**
 * Finds the approximate bounds of a set of numbers.
 *
 * <p>The algorithm uses two logarithmic histograms, one for positive values and one for negative
 * values. Histogram bin counts are thresholded to remove bins with too few counts. The threshold is
 * derived from the "failure probability", which is an upper bound on the probability of returning
 * bounds that are too loose.
 *
 * <p>We choose the rightmost bin that exceeds the threshold count and return its greater boundary
 * as the approximate upper bound. Similarly the leftmost bin that exceeds the threshold count is
 * chosen and its smaller boundary is returned as the approximate lower bound. If the failure
 * probability is too low, then it is possible that no bin exceeds the threshold, in which case we
 * increase the failure probability and try again. When the failure probability exceeds a fixed
 * value, we throw an exception rather than risk returning bounds that are primarily determined by
 * the noise. If this happens, try providing a larger sample to ApproximateBounds.
 *
 * <p>For general details and key definitions, see <a href=
 * "https://github.com/google/differential-privacy/blob/main/differential_privacy.md#key-definitions">
 * the introduction to Differential Privacy</a>.
 */
public class ApproximateBounds {
  @VisibleForTesting final Params params;

  /**
   * The failure probability is an upper bound on the probability of returning bounds that are too
   * loose.
   *
   * <p>More precisely, it is the probability that we (incorrectly) return bounds for an empty
   * dataset due to the noise added. It is used to determine the threshold for a bin to be
   * considered non-empty.
   *
   * <p>Setting the failure probability too low will result in higher thresholds and, consequently,
   * in dropping bins with too few entries or failing to find bounds at all. If this happens, we
   * repeatedly retry with a larger failure probability until the failure probability reaches {@link
   * ApproximateBounds.MAX_FAILURE_PROBABILITY}.
   */
  private static final double INITIAL_FAILURE_PROBABILITY = 1e-9;

  /** The factor by which the failure probability is increased when bounding fails. */
  private static final double FAILURE_PROBABILITY_INCREMENT_FACTOR = 2;

  /**
   * The maximum acceptable failure probability.
   *
   * <p>When increasing the failure probability in order to try to find bounds for a small sample,
   * if the failure probability increases above this value then we return an error: the bounds we
   * would get would too often be influenced by noise rather than the sample.
   */
  private static final double MAX_FAILURE_PROBABILITY = 1e-6;

  private final long[] positiveBins;
  private final long[] negativeBins;
  private final double[] rightBinBoundaries;

  private final Random random = new SecureRandom();

  private AggregationState aggregationState = AggregationState.DEFAULT;

  private ApproximateBounds(Params params) {
    this.params = params;

    positiveBins = new long[params.inputType().numPositiveBins];
    negativeBins = new long[params.inputType().numNegativeBins];

    // Cache bin boundaries for performance reasons.
    rightBinBoundaries = params.inputType().generateRightBinBoundaries();
  }

  public static Params.Builder builder() {
    return Params.Builder.newBuilder();
  }

  /** Adds a value to the approximate bounds calculation. */
  public void addEntry(double value) {
    checkState(
        aggregationState.equals(AggregationState.DEFAULT), aggregationState.getErrorMessage());

    // NaN is ignored because we cannot run logic on NaNs.
    if (Double.isNaN(value)) {
      return;
    }

    // If our histogram has no negative bins, clamp negative values to the 0th bin
    if (value < 0 && !params.inputType().hasNegativeBins) {
      positiveBins[0]++;
      return;
    }

    // If our histogram has negative bins, then they are symmetric with the positive ones.
    int binIndex = params.inputType().getPositiveBinNumber(abs(value));
    if (value >= 0) {
      positiveBins[binIndex]++;
    } else {
      negativeBins[binIndex]++;
    }
  }

  /** Adds a set of value to the approximate bounds calculation. */
  public void addEntries(Collection<Double> e) {
    e.forEach(this::addEntry);
  }

  public Result computeResult() {
    checkState(
        aggregationState.equals(AggregationState.DEFAULT), aggregationState.getErrorMessage());
    aggregationState = AggregationState.RESULT_RETURNED;

    double[] noisyPositiveBins = addNoise(positiveBins);
    double[] noisyNegativeBins = addNoise(negativeBins);

    double currentFailureProbability = INITIAL_FAILURE_PROBABILITY;
    while (currentFailureProbability <= MAX_FAILURE_PROBABILITY) {
      double perBinFailureProbability =
          computePerBinFailureProbability(
              currentFailureProbability, params.inputType().totalBins());

      double threshold =
          LaplaceNoise.computeQuantile(
              1 - perBinFailureProbability,
              /* x= */ 0,
              /* l1Sensitivity= */ params.maxContributions(),
              params.epsilon());

      Optional<Double> minimum = findMinimum(threshold, noisyNegativeBins, noisyPositiveBins);
      Optional<Double> maximum = findMaximum(threshold, noisyNegativeBins, noisyPositiveBins);

      if (minimum.isPresent() && maximum.isPresent()) {
        return Result.create(minimum.get(), maximum.get());
      }

      currentFailureProbability = FAILURE_PROBABILITY_INCREMENT_FACTOR * currentFailureProbability;
    }

    throw new IllegalArgumentException(
        "Bin count threshold was too large to find approximate bounds. Either run over a larger "
            + "dataset or decrease success_probability and try again.");
  }

  /**
   * Converts an overall failure probability into the corresponding per-bin failure probability.
   *
   * <p>If we were dealing with a "success probability" (i.e. 1 - failure probability) and had N
   * bins, then the per-bin success probability is the N^th root of the overall success probability.
   * Translated into failure probabilities: perBinFP = 1 - pow(1 - FP, 1/N)
   *
   * <p>Since in general the failure probability is very small, we use expm1 and log1p to calculate
   * this:
   *
   * <pre>{@code
   * perBinFP = 1 - exp(log(pow(1 - FP, 1/N)))
   *          = 1 - exp(log(1 - FP) / N)
   *          = -expm1(log1p(-FP) / N)
   * }</pre>
   */
  @VisibleForTesting
  static double computePerBinFailureProbability(double overallFailureProbability, int numBins) {
    return -expm1(log1p(-overallFailureProbability) / numBins);
  }

  /**
   * Finds the minimum by looking for bins with sufficient amount of entries in the noisy
   * histograms.
   */
  private Optional<Double> findMinimum(
      double threshold, double[] noisyNegativeBins, double[] noisyPositiveBins) {
    for (int i = noisyNegativeBins.length - 1; i >= 0; --i) {
      if (noisyNegativeBins[i] >= threshold) {
        return Optional.of(getNegativeLeftBinBoundary(i));
      }
    }

    for (int i = 0; i < noisyPositiveBins.length; i++) {
      if (noisyPositiveBins[i] >= threshold) {
        return Optional.of(getPositiveLeftBinBoundary(i));
      }
    }

    return Optional.empty();
  }

  /**
   * Finds the maximum by looking for bins with sufficient amount of entries in the noisy
   * histograms.
   */
  private Optional<Double> findMaximum(
      double threshold, double[] noisyNegativeBins, double[] noisyPositiveBins) {
    for (int i = noisyPositiveBins.length - 1; i >= 0; --i) {
      if (noisyPositiveBins[i] >= threshold) {
        return Optional.of(getPositiveRightBinBoundary(i));
      }
    }

    for (int i = 0; i < noisyNegativeBins.length; i++) {
      if (noisyNegativeBins[i] >= threshold) {
        return Optional.of(getNegativeRightBinBoundary(i));
      }
    }

    return Optional.empty();
  }

  private double getNegativeLeftBinBoundary(int bin) {
    return -1.0 * getPositiveRightBinBoundary(bin);
  }

  private double getNegativeRightBinBoundary(int bin) {
    return -getPositiveLeftBinBoundary(bin);
  }

  private double getPositiveLeftBinBoundary(int bin) {
    if (bin == 0) {
      return 0;
    }
    return getPositiveRightBinBoundary(bin - 1);
  }

  private double getPositiveRightBinBoundary(int bin) {
    return rightBinBoundaries[bin];
  }

  private double[] addNoise(long[] bins) {
    double[] noisyBins = new double[bins.length];
    for (int i = 0; i < bins.length; i++) {
      // Since we never release the raw noised values from this class, we don't need to defend
      // against attacks against the least significant bits of the noise (see "On Significance of
      // the Least Significant Bits For Differential Privacy" by Ilya Mironov,
      // https://www.microsoft.com/en-us/research/wp-content/uploads/2012/10/lsbs.pdf). Therefore we
      // can use a faster randomness generation method.
      noisyBins[i] =
          LaplaceNoise.computeQuantile(
              /* rank= */ random.nextDouble(),
              /* x= */ (double) bins[i],
              /* l1Sensitivity= */ params.maxContributions(),
              params.epsilon());
    }

    return noisyBins;
  }

  /**
   * Returns a serializable summary of the current state of this {@link ApproximateBounds} instance.
   * The summary can be used to merge this instance with another instance of {@link
   * ApproximateBounds}.
   *
   * <p>This method cannot be invoked if the result has already been computed using {@link
   * computeResult()}. Moreover, after this instance of {@link ApproximateBounds} has been
   * serialized once, further modification and queries are not possible anymore.
   *
   * @throws IllegalStateException if this instance of {@link ApproximateBounds} has already been
   *     queried.
   */
  public byte[] getSerializableSummary() {
    checkState(
        aggregationState != AggregationState.RESULT_RETURNED, aggregationState.getErrorMessage());
    aggregationState = AggregationState.SERIALIZED;

    return ApproxBoundsSummary.newBuilder()
        .addAllPosBinCount(Longs.asList(positiveBins))
        .addAllNegBinCount(Longs.asList(negativeBins))
        .build()
        .toByteArray();
  }

  /**
   * Merges the output of {@link #getSerializableSummary()} from a different instance of {@link
   * ApproximateBounds} with this instance. Intended to be used in the context of distributed
   * computation.
   *
   * @throws IllegalArgumentException if the passed serialized summary is invalid or has a different
   *     number of bins to this instance.
   * @throws IllegalStateException if this instance of {@link ApproximateBounds} has already been
   *     queried or serialized.
   */
  public void mergeWith(byte[] otherSummary) {
    checkState(
        aggregationState.equals(AggregationState.DEFAULT), aggregationState.getErrorMessage());
    ApproxBoundsSummary otherSummaryParsed;
    try {
      otherSummaryParsed =
          ApproxBoundsSummary.parseFrom(otherSummary, ExtensionRegistry.newInstance());
    } catch (InvalidProtocolBufferException e) {
      throw new IllegalArgumentException(e);
    }

    checkArgument(
        otherSummaryParsed.getPosBinCountCount() == positiveBins.length,
        "Both histograms must have the same number of positive bins (merge target has %s, "
            + "merge source has %s).",
        positiveBins.length,
        otherSummaryParsed.getPosBinCountCount());
    checkArgument(
        otherSummaryParsed.getNegBinCountCount() == negativeBins.length,
        "Both histograms must have the same number of negative bins (merge target has %s, "
            + "merge source has %s).",
        negativeBins.length,
        otherSummaryParsed.getNegBinCountCount());
    for (int i = 0; i < otherSummaryParsed.getPosBinCountCount(); i++) {
      positiveBins[i] += otherSummaryParsed.getPosBinCount(i);
    }
    for (int i = 0; i < otherSummaryParsed.getNegBinCountCount(); i++) {
      negativeBins[i] += otherSummaryParsed.getNegBinCount(i);
    }
  }

  /**
   * Parameters for calculation of differentially-private approximate bounds.
   *
   * <p>See the {@link Builder} for the explanation of each parameter.
   */
  @AutoValue
  public abstract static class Params {
    abstract double epsilon();

    abstract InputType inputType();

    abstract int maxContributions();

    public abstract Builder toBuilder();

    @AutoValue.Builder
    public abstract static class Builder {
      // Approximate bounds supports only Laplace noise due to the way sensitivities
      // are calculated in the internal logarithmic histogram. Therefore, the builder doesn't
      // accept noise or delta parameters.

      /** Epsilon to be used for the approximate bounds calculation. */
      public abstract Builder epsilon(double value);

      /**
       * Type of the inputs to {@link ApproximateBounds}.
       *
       * <p>This parameter is used to optimize the bounding algorithm based on expected properties
       * of the input values, e.g., the values have the range and granularity of a certain numerical
       * type such as double, long, etc. For example, when the input type is {@code
       * InputType.DOUBLE}, we include both positive and negative bins in the histogram, as well as
       * bins between 0 and 1. Note that all values passed to {@link ApproximateBounds} are doubles;
       * this parameter only specifies how the algorithm processes the values.
       */
      public abstract Builder inputType(InputType inputType);

      /** The maximum number of contributions each privacy unit can make to the dataset. */
      public abstract Builder maxContributions(int value);

      abstract Params autoBuild();

      private static Builder newBuilder() {
        Params.Builder builder = new AutoValue_ApproximateBounds_Params.Builder();
        builder.inputType(InputType.DOUBLE);
        return builder;
      }

      public ApproximateBounds build() {
        Params params = autoBuild();
        DpPreconditions.checkEpsilon(params.epsilon());
        return new ApproximateBounds(params);
      }
    }

    /**
     * The type of input that this algorithm will be provided with.
     *
     * <p>This determines the size and number of bins, as well as whether we should include negative
     * bins and bins between 0 and 1.
     *
     * <p>In general, bin i contains the number of inputs that lie in the range (scale * base^(i-1),
     * scale * base^i], where scale and base are determined by the InputType. The exception is
     * positive bin 0, which has boundaries [0, scale * base^0]. Negative bin 0 does not contain 0.
     * Values outside of the range covered by bins are clipped to the closest bin boundary.
     *
     * <p>For example, if scale = 1, base = 2, and num_bins = 4 then the positive histogram bins are
     * [0, 1], (1, 2], (2, 4], and (4, 8].
     */
    public enum InputType {
      // For doubles, we include both positive and negative bins, with successive bins doubling in
      // size. These parameters result in a rightmost bin boundary equal to 2^1023. Since
      // Double.MAX_VALUE is slightly less than 2^1024, values close to ±Double.MAX_VALUE will get
      // clamped. Sample bin boundaries:
      // ..., [-2, -1), [-1, -0.5), ..., (0.5, 1], (1, 2], ...
      DOUBLE(2046, /*hasNegativeBins=*/ true, Double.MIN_NORMAL),
      // For integers, the bins are the same as for doubles except we don't subdivide the bins
      // [-1, 0) and [0, 1]. These parameters result in a rightmost bin boundary equal to 2^31.
      // Note that this is greater than Integer.MAX_VALUE, so Integer.MAX_VALUE lies in the
      // interior of the rightmost bin. Sample bin boundaries:
      // ..., [-2, -1), [-1, 0), [0, 1], (1, 2], ...
      INTEGER(32, /*hasNegativeBins=*/ true, 1),
      // For positive integers, the bins are the same as for integers except we don't include
      // negative bins. Sample bin boundaries:
      // [0, 1], (1, 2], ...
      POSITIVE_INTEGER(32, /*hasNegativeBins=*/ false, 1),
      // In tests, we use a reduced number of bins for performance reasons and to keep assertions
      // simple. We only use 10 bins, with successive bins doubling in size. Sample bin boundaries:
      // [-16, -8), [-8, -4) ..., [0, 1], (1, 2], ..., (8, 16]
      TEST(5, /*hasNegativeBins=*/ true, 1);

      @VisibleForTesting public final int numPositiveBins;
      @VisibleForTesting public final int numNegativeBins;
      @VisibleForTesting public final boolean hasNegativeBins;
      @VisibleForTesting public final double scale;
      @VisibleForTesting public final double base = 2.0;

      private InputType(int numPositiveBins, boolean hasNegativeBins, double scale) {
        this.numPositiveBins = numPositiveBins;
        this.numNegativeBins = hasNegativeBins ? numPositiveBins : 0;
        this.hasNegativeBins = hasNegativeBins;
        this.scale = scale;
      }

      int totalBins() {
        return this.numPositiveBins + this.numNegativeBins;
      }

      double[] generateRightBinBoundaries() {
        double[] rightBinBoundaries = new double[numPositiveBins];
        double current = scale;
        for (int i = 0; i < numPositiveBins; i++) {
          rightBinBoundaries[i] = current;
          // The above values for numPositiveBins are chosen so that they do not cause an overflow
          // here.
          current *= base;
        }

        return rightBinBoundaries;
      }

      /** Returns the bin number where the provided positive value should be placed. */
      int getPositiveBinNumber(double value) {
        if (value == 0) {
          // Zero is special since log(0) is undefined
          return 0;
        }
        if (value < 0) {
          throw new IllegalArgumentException("Expected a positive input but got " + value);
        }

        // Calculate the most significant bit and clamp to a valid bin index.
        // x satisfies scale * base^x = abs(value)
        double x = (log(abs(value)) - log(scale)) / log(base);
        int msb = (int) ceil(x);
        return max(0, min(msb, numPositiveBins - 1));
      }
    }
  }

  /** A result of approximate bounds calculation. */
  @AutoValue
  public abstract static class Result {

    /** Lower approximate bound. */
    public abstract double lowerBound();

    /** Upper approximate bound. */
    public abstract double upperBound();

    public static Result create(double lower, double upper) {
      return new AutoValue_ApproximateBounds_Result(lower, upper);
    }
  }
}

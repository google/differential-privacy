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
import static java.lang.Double.POSITIVE_INFINITY;
import static java.lang.Math.min;

import com.google.auto.value.AutoValue;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.privacy.differentialprivacy.proto.SummaryOuterClass.PreAggSelectPartitionSummary;
import com.google.protobuf.InvalidProtocolBufferException;
import java.security.SecureRandom;

/**
 * PreAggSelectPartition is used to compute an (ε,δ)-differentially private decision of whether to
 * materialize a partition.
 *
 * <p>Many differential privacy mechanisms work by performing an aggregation and adding noise. They
 * achieve (ε_m, δ_m)-differential privacy under the assumption that partitions are chosen in
 * advance. In other words, they assume that even if no data is associated with a partition, noise
 * is added to the empty aggregation, and the noisy result is materialized. However, when only
 * partitions containing data are materialized, such mechanisms fail to protect privacy for
 * partitions containing data from a single privacy unit ID (e.g., user). To fix this, partitions
 * with small numbers of privacy unit IDs must sometimes be dropped in order to maintain privacy.
 * This process of partition selection incurs an additional (ε, δ) differential privacy budget
 * resulting in a total differential privacy budget of (ε + ε_m, δ + δ_m) being used for the
 * aggregation with partition selection.
 *
 * <p>Depending on the maxPartitionsContributed, the PreAggSelectPartition uses one of two
 * differentially private partition selection algorithms.
 *
 * <p>When maxPartitionsContributed ≤ 3, the partition selection process is made (ε,δ)
 * differentially private by applying the definition of differential privacy to the count of privacy
 * IDs. Supposing l0Sensitivity bounds the number of partitions a privacy ID may contribute to, we
 * define:<br>
 * pε := ε/l0Sensitivity <br>
 * pδ := δ/l0Sensitivity <br>
 * to be the per-partition differential privacy losses incurred by the partition selection process.
 * Letting n denote the number of privacy IDs in a partition, the probability of selecting a
 * partition is given by the following recurrence relation:<br>
 * keepPartitionProbability(n) = min( keepPartitionProbability(n-1) * exp(pε) + pδ, (1) 1 - exp(-pε)
 * * (1-keepPartitionProbability(n-1)-pδ), (2) 1 (3) ) <br>
 * with base case keepPartitionProbability(0) = 0. This formula is optimal in terms of maximizing
 * the likelihood of selecting a partition under (ε,δ)-differential privacy, with the caveat that
 * the input values for pε and pδ are lower bound approximations. For efficiency, we use a
 * closed-form solution to this recurrence relation. See <a
 * href="https://arxiv.org/pdf/2006.03684.pdf">Differentially private partition selection paper</a>
 * for details on the underlying mathematics.
 *
 * <p>When l0sensitivity > 3, the partition selection process is made (ε,δ) differentially private
 * by using {@link Count#computeThresholdedResult} with Gaussian noise. Count computes a (ε,δ/2)
 * differentially private count of the privacy IDs in a partition by adding Gaussian noise. Then, it
 * computes a threshold T for which the probability that a (ε,δ/2) differentially private count of a
 * single privacy ID can exceed T is δ/2. It keeps the partition iff differentially private count
 * exceeds the threshold.
 *
 * <p>The reason two different algorithms for deciding whether to keep a partition are used is
 * because the first algorithm ("magic partition selection") is optimal when l0sensitivity ≤ 3 but
 * is outperformed by Gaussian-based thresholding when l0sensitivity > 3.
 *
 * <p>PreAggSelectPartition is a utility for maintaining the count of IDs in a single partition and
 * then determining whether the partition should be materialized. Use {@link #increment()} to
 * increment the count of IDs and {@link #shouldKeepPartition()} to decide if the partition should
 * be materialized.
 */
public class PreAggSelectPartition {
  private final PreAggSelectPartition.Params params;
  private final SecureRandom random;
  // The count of unique privacy unit IDs in the partition.
  private long idsCount;

  private AggregationState state = AggregationState.DEFAULT;

  public static PreAggSelectPartition.Params.Builder builder() {
    return PreAggSelectPartition.Params.Builder.newBuilder();
  }

  private PreAggSelectPartition(PreAggSelectPartition.Params params) {
    this.params = params;
    random = new SecureRandom();
  }

  /**
   * Increments the ids count by one. This is a responsibility of the caller of the library to
   * ensure that for each privacy unit ID this method is called at most once.
   */
  public void increment() {
    incrementBy(1);
  }

  /**
   * Increments the ids count by a given value. This is a responsibility of the caller of the
   * library to ensure that for each privacy unit ID this method is called at most once and the each
   * privacy id is counted at most once in the input value.
   *
   * <p>Note that decrementing counts by inputting a negative value is allowed, for example if you
   * want to remove some users you have previously added.
   *
   * @throws IllegalStateException if this this instance of {@link PreAggSelectPartition} has
   *     already been queried or serialized.
   */
  public void incrementBy(long value) {
    Preconditions.checkState(state.equals(AggregationState.DEFAULT), "Cannot increment.");

    idsCount += value;
  }

  /**
   * Returns whether the partition should be materialized.
   *
   * @throws IllegalStateException if this instance of {@link PreAggSelectPartition} has already
   *     been queried or serialized.
   */
  public boolean shouldKeepPartition() {
    Preconditions.checkState(
        state.equals(AggregationState.DEFAULT), "DP shouldKeepPartition cannot be computed.");

    state = AggregationState.RESULT_RETURNED;

    int preThreshold = params.preThreshold();
    // Pre-thresholding guarantees that at least this number of unique contributions are in the
    // partition.
    if (idsCount < preThreshold) {
      return false;
    }
    // If 1 was set as the default, subtract it here so it has no effect.
    // This subtraction also ensures that idsCount will always be > 0 if preThreshold = idsCount.
    idsCount = idsCount - (preThreshold - 1);

    if (params.maxPartitionsContributed() > 3) {
      Count count =
          Count.builder()
              .epsilon(params.epsilon())
              .delta(params.delta() / 2)
              .maxPartitionsContributed(params.maxPartitionsContributed())
              .noise(new GaussianNoise())
              .build();
      count.incrementBy(idsCount);
      return count.computeThresholdedResult(params.delta() / 2).isPresent();
    }

    double x = getKeepPartitionProbability();
    return random.nextDouble() < x;
  }

  /**
   * Returns a serializable summary of the current state of this {@link PreAggSelectPartition}
   * instance and its parameters. The summary can be used to merge this instance with another
   * instance of {@link PreAggSelectPartition}.
   *
   * <p>This method cannot be invoked if partition selection has already been queried, i.e., {@link
   * shouldKeepPartition()} has been called. Moreover, after this instance of {@link
   * PreAggSelectPartition} has been serialized once, further modification and queries are not
   * possible anymore.
   *
   * @throws IllegalStateException if this instance of {@link PreAggSelectPartition} has already
   *     been queried.
   */
  public byte[] getSerializableSummary() {
    Preconditions.checkState(
        state != AggregationState.RESULT_RETURNED, "PreAggSelectPartition cannot be serialized.");

    state = AggregationState.SERIALIZED;

    return PreAggSelectPartitionSummary.newBuilder()
        .setIdsCount(idsCount)
        .setEpsilon(params.epsilon())
        .setDelta(params.delta())
        .setMaxPartitionsContributed(params.maxPartitionsContributed())
        .setPreThreshold(params.preThreshold())
        .build()
        .toByteArray();
  }

  /**
   * Merges this instance with the output of {@link #getSerializableSummary()} from a different
   * {@link PreAggSelectPartition} and stores the merged result in this instance. This is required
   * in the distributed calculations context for merging partial results.
   *
   * @throws IllegalArgumentException if not all config parameters (e.g., epsilon) are equal or if
   *     the passed serialized ids count is invalid.
   * @throws IllegalStateException if this instance of {@link PreAggSelectPartitions} has already
   *     been queried or serialized.
   */
  public void mergeWith(byte[] otherPreAggSelectPartitionSummary) {
    Preconditions.checkState(
        state.equals(AggregationState.DEFAULT), "PreAggSelectPartitions cannot be merged.");

    PreAggSelectPartitionSummary otherPreAggSelectPartitionParsed;
    try {
      otherPreAggSelectPartitionParsed =
          PreAggSelectPartitionSummary.parseFrom(otherPreAggSelectPartitionSummary);
    } catch (InvalidProtocolBufferException pbe) {
      throw new IllegalArgumentException(pbe);
    }

    checkMergeParametersAreEqual(otherPreAggSelectPartitionParsed);
    this.idsCount += otherPreAggSelectPartitionParsed.getIdsCount();
  }

  @VisibleForTesting
  double getKeepPartitionProbability() {
    if (idsCount <= 0) {
      return 0;
    }

    double pEpsilon = params.epsilon() / (double) params.maxPartitionsContributed();
    double pDelta = params.delta() / (double) params.maxPartitionsContributed();

    // In selectPartitionPr's recurrence formula (see Theorem 1 in the
    // [Differentially private partition selection paper]),
    // argument selection in the min operation has 3 distinct regions: min selects (1) on the
    // lowest region of the domain, (2) on the second region of the domain, and (3) (i.e., the
    // value 1) on the highest region of the domain. We denote by nCr the
    // crossover point in the domain from (1) to (2).
    long nCr =
        (long)
            (1
                + Math.floor(
                    (1 / pEpsilon)
                        * Math.log(
                            (1 + Math.exp(-pEpsilon) * (2 * pDelta - 1))
                                / (pDelta * (1 + Math.exp(-pEpsilon))))));

    if (idsCount <= nCr) {
      // Closed form solution of selectPartitionPr(n) on (0, nCr].
      return pDelta * sumExpPowers(pEpsilon, 0, idsCount);
    }

    double selectPartitionPrNCr = pDelta * sumExpPowers(pEpsilon, 0, nCr);
    // Compute form solution of selectPartitionPr(n) on the domain (nCr, ∞).
    long m = idsCount - nCr;
    return min(
        1
            + Math.exp(-(double) m * pEpsilon) * (selectPartitionPrNCr - 1)
            + sumExpPowers(pEpsilon, -m, m) * pDelta,
        1);
  }

  /**
   * Returns e^(minPower * ε) + e^((minPower + 1) * ε) + ... + e^((numPowers + minPower - 1) * ε).
   */
  @VisibleForTesting
  double sumExpPowers(double epsilon, long minPower, long numPowers) {
    checkArgument(numPowers > 0, "numPowers must be > 0. Provided value: %s", numPowers);

    if (Double.isInfinite(Math.exp(minPower * epsilon))) {
      return POSITIVE_INFINITY;
    }
    // In the case ε=0, sumExpPowers is simply numPowers. We use exp(-ε) = 1 to
    // identify this case because our closed form solutions would otherwise
    // result in division by 0 under finite precision arithmetic.
    if (Math.exp(-epsilon) == 1) {
      return (double) numPowers;
    }

    /*
    For the general case, we use a closed form solution to a geometric
    series. See https://en.wikipedia.org/wiki/Geometric_series#Sum.

    The typical closed form formula is: e^(minPower*ε) * (e^(numPowers*ε) - 1) / (e^(ε) - 1).

    In our setting, it is OK to return +∞ but not OK to return NaN. We use the following
    mathematically equivalent formulas to avoid returning NaN when using our finite precision
    arithmetic:

    e^((minPower - 1)*ε) * (e^(numPowers*ε) - 1) / (1 - e^(-ε))                     (1)
    (e^((numPowers+minPower - 1) * ε) - e^((minPower - 1) * ε)) / (1 - e^(-ε))      (2)

    We use (1) when minPower >= 1. In that case, (e^(numPowers*ε) - 1) is the only potentially
    infinite term. The other two terms satisfy e^((minPower-1) * ε) >= 1 and 0 < (1 - exp(-ε)) <= 1.
    The multiplication and division operations involving these other two terms increase the result.
    Thus, the result is never NaN, and +∞ is returned only when necessary.

    We use (2) when minPower < 1. In that case, 0 <= exp((minPower-1)*ε) < 1, so the numerator
    (e^((numPowers + minPower - 1) * ε) - exp((minPower - 1) * ε)) is never NaN, and is only +∞ when
    necessary. The denominator in (2) satisfies 0 < (1-exp(-ε)) <= 1. The result of (2) is never NaN
    and only achieves +∞ when necessary overall.
    */
    if (minPower >= 1) {
      return Math.exp((minPower - 1) * epsilon)
          * (Math.expm1(numPowers * epsilon))
          / (1 - Math.exp(-epsilon));
    }

    return (Math.exp((numPowers + minPower - 1) * epsilon) - Math.exp((minPower - 1) * epsilon))
        / (1 - Math.exp(-epsilon));
  }

  private void checkMergeParametersAreEqual(PreAggSelectPartitionSummary otherCount) {
    DpPreconditions.checkMergeEpsilonAreEqual(params.epsilon(), otherCount.getEpsilon());
    DpPreconditions.checkMergeDeltaAreEqual(params.delta(), otherCount.getDelta());
    DpPreconditions.checkMergeMaxPartitionsContributedAreEqual(
        params.maxPartitionsContributed(), otherCount.getMaxPartitionsContributed());
    DpPreconditions.checkMergePreThresholdAreEqual(
        params.preThreshold(), otherCount.getPreThreshold());
  }

  /**
   * Returns the minimum number of distinct privacy IDs that should contribute to a partition in
   * order for partition to be kept with probability 1.
   */
  public static long getHardThreshold(double epsilon, double delta, int maxPartitionsContributed) {
    PreAggSelectPartition ps =
        PreAggSelectPartition.builder()
            .epsilon(epsilon)
            .delta(delta)
            .maxPartitionsContributed(maxPartitionsContributed)
            .build();
    for (long i = 1; ; i++) {
      ps.increment();
      if (ps.getKeepPartitionProbability() >= 1) {
        return i;
      }
      if (i == Long.MAX_VALUE) {
        throw new IllegalArgumentException(
            "Hard threshold exceeded max long value for the configured input.");
      }
    }
  }

  @AutoValue
  public abstract static class Params {
    abstract double epsilon();

    abstract double delta();

    abstract int maxPartitionsContributed();

    abstract int preThreshold();

    @AutoValue.Builder
    public abstract static class Builder {

      public static final int DEFAULT_PRE_THRESHOLD = 1;

      private static PreAggSelectPartition.Params.Builder newBuilder() {
        return new AutoValue_PreAggSelectPartition_Params.Builder()
            .preThreshold(DEFAULT_PRE_THRESHOLD);
      }

      /** Epsilon DP parameter. */
      public abstract PreAggSelectPartition.Params.Builder epsilon(double value);

      /* Delta DP parameter. */
      public abstract PreAggSelectPartition.Params.Builder delta(double value);

      /**
       * Maximum number of partitions to which a single privacy unit (i.e., an individual) is
       * allowed to contribute.
       */
      public abstract PreAggSelectPartition.Params.Builder maxPartitionsContributed(int value);

      /**
       * Guarantees no partitions with fewer than preThreshold number of unique contributions are
       * released.
       */
      // TODO: Link to md document to release pre-thresholding externally.
      public abstract PreAggSelectPartition.Params.Builder preThreshold(int value);

      abstract PreAggSelectPartition.Params autoBuild();

      public PreAggSelectPartition build() {
        PreAggSelectPartition.Params params = autoBuild();
        DpPreconditions.checkEpsilon(params.epsilon());
        DpPreconditions.checkDelta(params.delta());
        DpPreconditions.checkMaxPartitionsContributed(params.maxPartitionsContributed());
        DpPreconditions.checkPreThreshold(params.preThreshold());
        return new PreAggSelectPartition(params);
      }
    }
  }
}

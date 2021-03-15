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
//

package com.google.privacy.differentialprivacy;

import static com.google.common.base.Preconditions.checkArgument;
import static java.lang.Math.max;
import static java.lang.Math.min;

import com.google.auto.value.AutoValue;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.differentialprivacy.SummaryOuterClass.BoundedQuantilesSummary;
import com.google.protobuf.InvalidProtocolBufferException;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Calculates differentially private quantiles for a collection of values using a quantile tree
 * mechanism.
 *
 * <p>Note: the class is not thread-safe.
 *
 * <p>For general details and key definitions, see <a href=
 * "https://github.com/google/differential-privacy/blob/main/differential_privacy.md#key-definitions">
 * this</a> introduction to Differential Privacy.
 */
public class BoundedQuantiles {
  public static final double NUMERICAL_TOLERANCE = 1.0e-6;
  public static final int DEFAULT_TREE_HEIGHT = 4;
  public static final int DEFAULT_BRANCHING_FACTOR = 16;

  private static final int ROOT_INDEX = 0;
  // Fraction a node needs to contribute to the total count of itself and its siblings to be
  // considered during the search for a particular quantile. The idea of alpha is to filter out
  // noisy empty nodes. This is a post processing parameter with no privacy implications.
  private static final double ALPHA = 0.005;

  private final Params params;

  private final Map<Integer, Long> tree;
  private final Map<Integer, Double> noisedTree;

  private final int numberOfLeaves;
  private final int leftmostLeafIndex;

  private AggregationState state = AggregationState.DEFAULT;

  private BoundedQuantiles(BoundedQuantiles.Params params) {
    this.params = params;
    tree = new HashMap<>();
    noisedTree = new HashMap<>();

    int numberOfNodes =
        (int)
            ((Math.pow(params.branchingFactor(), params.treeHeight() + 1) - 1)
                / (params.branchingFactor() - 1));
    numberOfLeaves = (int) Math.pow(params.branchingFactor(), params.treeHeight());
    // The following assumes that nodes are indexed in a breadth first fashion from left to right.
    leftmostLeafIndex = numberOfNodes - numberOfLeaves;
  }

  public static BoundedQuantiles.Params.Builder builder() {
    return BoundedQuantiles.Params.Builder.newBuilder();
  }

  /**
   * Clamps the input value and adds it to the distribution.
   *
   * @throws IllegalStateException if this this instance of {@link BoundedQuantiles} has already
   *     been queried or serialized.
   */
  public void addEntry(double e) {
    Preconditions.checkState(state == AggregationState.DEFAULT, "Entry cannot be added.");

    // NaN is ignored because we cannot aggregate NaNs.
    if (Double.isNaN(e)) {
      return;
    }

    // Increment all counts on the path from the leaf node where the value is inserted up to the
    // first level (root not included)
    int index = getIndex(clamp(e));
    while (index != ROOT_INDEX) {
      long count = tree.containsKey(index) ? tree.get(index) : 0;
      tree.put(index, count + 1);
      index = getParent(index);
    }
  }

  /**
   * Clamps the input values and adds them to the distribution.
   *
   * @throws IllegalStateException if this this instance of {@link BoundedQuantiles} has already
   *     been queried or serialized.
   */
  public void addEntries(Collection<Double> e) {
    e.forEach(this::addEntry);
  }

  private double clamp(double value) {
    return max(min(value, params.upper()), params.lower());
  }

  /**
   * Returns the index of the leaf node associated with the provided value, assuming that the leaf
   * nodes partition the range betwen lower and upper into intervals of equal size.
   */
  private int getIndex(double value) {
    return leftmostLeafIndex
        + (value == params.upper()
            ? numberOfLeaves - 1
            : (int)
                Math.floor(
                    (value - params.lower()) / (params.upper() - params.lower()) * numberOfLeaves));
  }

  /**
   * Returns the smallest value mapped to the subtree of the provided index, assuming that the leaf
   * nodes partition the range betwen lower and upper into intervals of equal size.
   */
  private double getLeftValue(int index) {
    // Traverse the tree towards the leaves starting at the provided index always taking the
    // leftmost branch.
    while (index < leftmostLeafIndex) {
      index = getLeftmostChild(index);
    }
    return (params.upper() - params.lower())
            * ((index - leftmostLeafIndex) / (double) numberOfLeaves)
        + params.lower();
  }

  /**
   * Returns the greatest value mapped to the subtree of the provided index, assuming that the leaf
   * nodes partition the range betwen lower and upper into intervals of equal size.
   */
  private double getRightValue(int index) {
    // Traverse the tree towards the leaves starting at the provided index always taking the
    // rightmost branch.
    while (index < leftmostLeafIndex) {
      index = getRightmostChild(index);
    }
    // The returned value bounds the range of values for which getIndex returns the specified index.
    // This bound is not itself contained in that range, i.e., getIndex will return the next index
    // when called for the bound.
    return (params.upper() - params.lower())
            * ((index - leftmostLeafIndex + 1) / (double) numberOfLeaves)
        + params.lower();
  }

  /**
   * Calculates and returns a differentially private quantile of the values added via {@link
   * #addEntry} and {@link #addEntries}. The specified rank must be between 0.0 and 1.0.
   *
   * <p>This method can be called multiple times to compute different quantiles. Privacy budget is
   * paid only once, on its first invocation. Calling this method repeatedly for the same rank will
   * return the same result. The results of repeated calls are guaranteed to be monotonically
   * increasing in the sense that r_1 < r_2 implies that computeResult(r_1) <= computeResult(r_2).
   *
   * <p>Note that the returned value is not an unbiased estimate of the raw bounded quantile.
   *
   * @throws IllegalStateException if this instance of {@link BoundedQuantiles} has already been
   *     serialized.
   */
  public double computeResult(double rank) {
    Preconditions.checkState(
        state == AggregationState.DEFAULT || state == AggregationState.RESULT_RETURNED,
        "DP quantile cannot be computed.");

    state = AggregationState.RESULT_RETURNED;

    checkArgument(
        rank >= 0.0 && rank <= 1.0, "rank must be >= 0 and <= 1. Provided value: %s", rank);

    rank = adjustRank(rank);

    int index = ROOT_INDEX;
    // Search for the index of the leaf node containg the specified quantile, starting at the root.
    while (index < leftmostLeafIndex) {
      int leftmostChildIndex = getLeftmostChild(index);
      int rightmostChildIndex = getRightmostChild(index);

      double totalCount = 0.0;
      for (int i = leftmostChildIndex; i <= rightmostChildIndex; i++) {
        totalCount += getNoisedCount(i);
      }
      if (totalCount <= 0.0) {
        // All child nodes appear to be empty. There is no need to proceed further down the tree.
        break;
      }

      double correctedTotalCount = 0.0;
      for (int i = leftmostChildIndex; i <= rightmostChildIndex; i++) {
        // Treat child nodes contrinbuting less than an alpha fraction to the total count as empty
        // subtrees.
        correctedTotalCount += getNoisedCount(i) >= totalCount * ALPHA ? getNoisedCount(i) : 0.0;
      }
      if (correctedTotalCount == 0.0) {
        // No child node contributes more than an alpha fraction to the total count (this can only
        // happen when alpha > 1 / branching factor, which is not the case for the default branching
        // factor). This means that all child nodes are considered empty and there is no need to
        // proceed further down the tree.
        break;
      }

      // Determine the child node whose subtree contains the quantile.
      double partialCount = 0.0;
      for (int i = leftmostChildIndex; true; i++) {
        double count = getNoisedCount(i);
        // Skip child nodes contributing less than alpha to the total count.
        if (count >= totalCount * ALPHA) {
          partialCount += count;
          // Check if the quantile is in the current child's subtree.
          if (partialCount / correctedTotalCount >= rank - NUMERICAL_TOLERANCE) {
            rank =
                (rank - (partialCount - count) / correctedTotalCount)
                    / (count / correctedTotalCount);
            // Clamping rank to a value between 0.0 and 1.0. Note that rank can become greater than
            // 1 because of the numerical tolerance. Values less than 0.0 should not occur. The
            // respective clamping is set in place to be on the safe side.
            rank = min(max(0.0, rank), 1.0);
            index = i;
            break;
          }
        }
      }
    }
    // Linearly interpolate between the smallest and largest value associated with the node of the
    // current index.
    return (1 - rank) * getLeftValue(index) + rank * getRightValue(index);
  }

  private int getLeftmostChild(int index) {
    return index * params.branchingFactor() + 1;
  }

  private int getRightmostChild(int index) {
    return (index + 1) * params.branchingFactor();
  }

  private int getParent(int index) {
    return (index - 1) / params.branchingFactor();
  }

  /**
   * Clamps the rank to a value between 0.005 and 0.995. The purpose of this adjustment is to
   * mitigate the inaccuracy of the quantile tree mechanism around the min and max, i.e., the 0 and
   * 1 rank.
   */
  @VisibleForTesting
  static double adjustRank(double rank) {
    return max(min(rank, 0.995), 0.005);
  }

  /**
   * Returns a noised version of the count associated with the respective index. If the count has
   * been noised before, the same value as before is returned.
   */
  private double getNoisedCount(int index) {
    if (noisedTree.containsKey(index)) {
      return noisedTree.get(index);
    }
    double rawCount = tree.containsKey(index) ? tree.get(index) : 0;
    // The l_1 sensitivity of a privacy unit's contribution is
    //    treeHeight * maxPartitionsContributed * maxContributionsPerPartition
    // while the l_2 sensitivity is
    //    sqrt(treeHeight * maxPartitionsContributed) * maxContributionsPerPartition
    // (the latter is realized if the privacy unit increments the exact same counters for each of
    // their contributions to a particular partition). Setting the l_0 and l_inf sensitivity as
    // follows yields the respective l_1 and l_2 values.
    double noisedCount =
        params
            .noise()
            .addNoise(
                /* x= */ rawCount,
                /* l0Sensitivity= */ params.treeHeight() * params.maxPartitionsContributed(),
                /* lInfSensitivity= */ params.maxContributionsPerPartition(),
                /* epsilon= */ params.epsilon(),
                /* delta= */ params.delta());
    noisedTree.put(index, noisedCount);
    return noisedCount;
  }

  /**
   * Returns a serializable summary of the current state of this {@link BoundedQuantiles} instance
   * and its parameters. The summary can be used to merge this instance with another instance of
   * {@link BoundedQuantiles}.
   *
   * <p>This method cannot be invoked if a quantile has already been queried, i.e., {@link
   * computeResult(double)} has been called. Moreover, after this instance of {@link
   * BoundedQuantiles} has been serialized once, no further modification, queries or serialization
   * is possible anymore.
   *
   * @throws IllegalStateException if this this instance of {@link BoundedQuantiles} has already
   *     been queried or serialized.
   */
  public byte[] getSerializableSummary() {
    Preconditions.checkState(
        state == AggregationState.DEFAULT, "Distribution cannot be serialized.");

    state = AggregationState.SERIALIZED;

    BoundedQuantilesSummary.Builder builder =
        BoundedQuantilesSummary.newBuilder()
            .putAllQuantileTree(tree)
            .setEpsilon(params.epsilon())
            .setLower(params.lower())
            .setUpper(params.upper())
            .setMaxPartitionsContributed(params.maxPartitionsContributed())
            .setMaxContributionsPerPartition(params.maxContributionsPerPartition())
            .setMechanismType(params.noise().getMechanismType())
            .setTreeHeight(params.treeHeight())
            .setBranchingFactor(params.branchingFactor());
    if (params.delta() != null) {
      builder.setDelta(params.delta());
    }

    return builder.build().toByteArray();
  }

  /**
   * Merges the output of {@link #getSerializableSummary()} from a different instance of {@link
   * BoundedQuantiles} with this instance. Intended to be used in the context of distributed
   * computation.
   *
   * @throws IllegalArgumentException if the parameters of the two instances (epsilon, delta,
   *     contribution bounds, etc.) do not match or if the passed serialized summary is invalid.
   * @throws IllegalStateException if this this instance of {@link BoundedQuantiles} has already
   *     been queried or serialized.
   */
  public void mergeWith(byte[] otherBoundedQuantilesSummary) {
    Preconditions.checkState(state == AggregationState.DEFAULT, "Distributions cannot be merged.");

    BoundedQuantilesSummary otherSummaryParsed;
    try {
      otherSummaryParsed = BoundedQuantilesSummary.parseFrom(otherBoundedQuantilesSummary);
    } catch (InvalidProtocolBufferException pbe) {
      throw new IllegalArgumentException(pbe);
    }

    checkMergeParametersAreEqual(otherSummaryParsed);
    for (int index : otherSummaryParsed.getQuantileTreeMap().keySet()) {
      long oldCount = tree.containsKey(index) ? tree.get(index) : 0;
      tree.put(index, oldCount + otherSummaryParsed.getQuantileTreeOrDefault(index, 0));
    }
  }

  private void checkMergeParametersAreEqual(BoundedQuantilesSummary summary) {
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

    checkArgument(
        params.treeHeight() == summary.getTreeHeight(),
        "Failed to merge: unequal values of treeheight. " + "treeHeight1 = %s, treeHeight2 = %s",
        params.treeHeight(),
        summary.getTreeHeight());
    checkArgument(
        params.branchingFactor() == summary.getBranchingFactor(),
        "Failed to merge: unequal values of branchingFactor. "
            + "branchingFactor1 = %s, branchingFactor2 = %s",
        params.branchingFactor(),
        summary.getBranchingFactor());
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

    abstract int treeHeight();

    abstract int branchingFactor();

    @AutoValue.Builder
    public abstract static class Builder {

      private static BoundedQuantiles.Params.Builder newBuilder() {
        BoundedQuantiles.Params.Builder builder = new AutoValue_BoundedQuantiles_Params.Builder();
        // Provides LaplaceNoise as a default noise generator.
        builder.noise(new LaplaceNoise());
        // A default tree height of 4 and branching factor of 16 devides the domain of possible
        // values into 65536 partitions.
        builder.treeHeight(DEFAULT_TREE_HEIGHT);
        builder.branchingFactor(DEFAULT_BRANCHING_FACTOR);

        return builder;
      }

      /** Epsilon DP parameter. */
      public abstract BoundedQuantiles.Params.Builder epsilon(double value);

      /**
       * Delta DP parameter.
       *
       * <p>Note that Laplace noise does not use delta. Hence, delta should not be set when Laplace
       * noise is used.
       */
      public abstract BoundedQuantiles.Params.Builder delta(@Nullable Double value);

      /**
       * Maximum number of partitions that a single privacy unit (e.g., an individual) is allowed to
       * contribute to.
       */
      public abstract BoundedQuantiles.Params.Builder maxPartitionsContributed(int value);

      /** Max contributions per partition from a single privacy unit (e.g., an individual). */
      public abstract BoundedQuantiles.Params.Builder maxContributionsPerPartition(int value);

      /** Noise that will be used to make the quantiles differentially private. */
      public abstract BoundedQuantiles.Params.Builder noise(Noise value);

      /**
       * Lower bound for the entries added to the distribution. Any entires smaller than this value
       * will be set to this value.
       */
      public abstract BoundedQuantiles.Params.Builder lower(double value);

      /**
       * Upper bound for the entries added to the distribution. Any entires greater than this value
       * will be set to this value.
       */
      public abstract BoundedQuantiles.Params.Builder upper(double value);

      /**
       * The height of the quantile tree used to store the entries. Should be at least 1. This
       * parameter is not public, it should be used only by other aggregation functions inside the
       * library.
       */
      @VisibleForTesting
      public abstract BoundedQuantiles.Params.Builder treeHeight(int value);

      /**
       * The number of children for every non-leaf node of the quantile tree used to store the
       * entries. Should be at least 2. This parameters is not public, it should be used only by
       * other aggregation functions inside the library.
       */
      @VisibleForTesting
      public abstract BoundedQuantiles.Params.Builder branchingFactor(int value);

      private static void checkTreeHeight(int treeHeight) {
        checkArgument(
            treeHeight >= 1, "Tree height must be at least 1. Provided value: %s", treeHeight);
      }

      private static void checkBranchingFactor(int branchingFactor) {
        checkArgument(
            branchingFactor >= 2,
            "Branching factor must be at least 2. Provided value: %s",
            branchingFactor);
      }

      private static void checkBoundsNotEqual(double lower, double upper) {
        checkArgument(lower != upper, "Lower and upper bounds must not be equal");
      }

      abstract BoundedQuantiles.Params autoBuild();

      public BoundedQuantiles build() {
        BoundedQuantiles.Params params = autoBuild();
        // No need to check noise nullability: the noise is defaulted to Laplace noise.
        DpPreconditions.checkEpsilon(params.epsilon());
        DpPreconditions.checkNoiseDelta(params.delta(), params.noise());
        DpPreconditions.checkMaxPartitionsContributed(params.maxPartitionsContributed());
        DpPreconditions.checkMaxContributionsPerPartition(params.maxContributionsPerPartition());
        DpPreconditions.checkBounds(params.lower(), params.upper());
        checkBoundsNotEqual(params.lower(), params.upper());
        checkTreeHeight(params.treeHeight());
        checkBranchingFactor(params.branchingFactor());

        return new BoundedQuantiles(params);
      }
    }
  }
}

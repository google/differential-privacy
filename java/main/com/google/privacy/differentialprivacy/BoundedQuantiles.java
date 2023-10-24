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
import com.google.privacy.differentialprivacy.proto.SummaryOuterClass.BoundedQuantilesSummary;
import com.google.protobuf.InvalidProtocolBufferException;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

/**
 * Calculates differentially private quantiles for a collection of values using a quantile tree
 * mechanism.
 *
 * <p>See https://github.com/google/differential-privacy/blob/main/common_docs/Differentially_Private_Quantile_Trees.pdf.
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
  // considered during the search for a particular quantile. The idea of gamma is to filter out
  // noisy empty nodes. This is a post processing parameter with no privacy implications.
  private static final double GAMMA = 0.0075;

  private enum ConfidenceIntervalBoundType {
    LOWER,
    UPPER
  };

  private static final ConfidenceIntervalBoundType LOWER = ConfidenceIntervalBoundType.LOWER;
  private static final ConfidenceIntervalBoundType UPPER = ConfidenceIntervalBoundType.UPPER;

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
   * @throws IllegalStateException if this instance of {@link BoundedQuantiles} has already been
   *     queried or serialized.
   */
  public void addEntry(double e) {
    Preconditions.checkState(state.equals(AggregationState.DEFAULT), "Entry cannot be added.");

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
   * @throws IllegalStateException if this instance of {@link BoundedQuantiles} has already been
   *     queried or serialized.
   */
  public void addEntries(Collection<Double> e) {
    e.forEach(this::addEntry);
  }

  private double clamp(double value) {
    return max(min(value, params.upper()), params.lower());
  }

  /**
   * Returns the index of the leaf node associated with the provided value, assuming that the leaf
   * nodes partition the range between lower and upper into intervals of equal size.
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
   * nodes partition the range between lower and upper into intervals of equal size.
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
        state.equals(AggregationState.DEFAULT) || state.equals(AggregationState.RESULT_RETURNED),
        "DP quantile cannot be computed.");

    state = AggregationState.RESULT_RETURNED;

    checkArgument(
        rank >= 0.0 && rank <= 1.0, "rank must be >= 0 and <= 1. Provided value: %s", rank);

    rank = adjustRank(rank);

    int index = ROOT_INDEX;
    // Search for the index of the node containing the specified quantile, starting at the root.
    while (index < leftmostLeafIndex) {
      int leftmostChildIndex = getLeftmostChild(index);
      int rightmostChildIndex = getRightmostChild(index);

      Map<Integer, Double> noisedCounts = new HashMap<>();
      for (int i = leftmostChildIndex; i <= rightmostChildIndex; i++) {
        noisedCounts.put(i, getNoisedCount(i));
      }

      IndexAndRank nextIndexAndRank =
          getNextIndexAndRank(rank, leftmostChildIndex, rightmostChildIndex, noisedCounts);

      if (nextIndexAndRank == null) {
        // All child nodes are considred empty. No need to proceed with the search.
        break;
      } else {
        index = nextIndexAndRank.index();
        rank = nextIndexAndRank.rank();
      }
    }
    // Linearly interpolate between the smallest and largest value associated with the node returned
    // by the search.
    return (1 - rank) * getLeftValue(index) + rank * getRightValue(index);
  }

  /**
   * Computes a confidence interval that contains the quantile of the specified rank with a
   * probability greater or equal to {@code 1 - alpha}. More precisely, the confidence interval
   * contains the value the mechanism returns when no noise is added with the specified probability.
   * Note that this value might be different from the raw quantile as a result of bounding and
   * internal processing.
   *
   * <p>The confidence interval is exclusively based on the noised bounded quantile returned by
   * {@link #computeResult}. Thus, no privacy budget is consumed by this operation.
   *
   * <p>Refer to <a
   * href="https://github.com/google/differential-privacy/tree/main/common_docs/confidence_intervals.md">this</a> doc for
   * more information.
   *
   * @throws IllegalStateException if this instance of {@link BoundedQuantiles} has not been queried
   *     yet.
   */
  public ConfidenceInterval computeConfidenceInterval(double rank, double alpha) {
    Preconditions.checkState(
        state.equals(AggregationState.RESULT_RETURNED), "Confidence interval cannot be computed.");

    checkArgument(
        rank >= 0.0 && rank <= 1.0, "rank must be >= 0 and <= 1. Provided value: %s", rank);

    rank = adjustRank(rank);

    return ConfidenceInterval.create(
        computeConfidenceIntervalBound(rank, alpha, LOWER),
        computeConfidenceIntervalBound(rank, alpha, UPPER));
  }

  // The following computation of a lower or upper interval bound is based on the same search
  // algorithm used to compute the respective quantile. The difference is that instead of using the
  // noised node counts to determine the direction of the search, the algorithm uses the confidence
  // intervals bounds of the node counts.
  private double computeConfidenceIntervalBound(
      double rank, double alpha, ConfidenceIntervalBoundType boundType) {
    // Let b be the branching factor and h the height of the tree. The search for a quantile queries
    // at most b * h node counts. Assigning a confidence interval with error probability
    //    alpha' = 1 - (1 - alpha)^(1 / (b * h))
    // to each of these counts guarantees that the true counts are contained within these
    // confidence intervals with error probability
    //    1 - (1 - alpha')^(b * h) = 1 - (1 - (1 - (1 - alpha)^(1 / (b * h))))^(b * h) = alpha,
    // which matches the specified error probability.
    double alphaPerCount =
        1 - Math.pow(1 - alpha, 1.0 / (params.branchingFactor() * params.treeHeight()));

    // Confidence interval of a node count of 0 with error probability alpha'. All other node count
    // confidence intervals are computed by shifting this interval, which is faster than calling
    // computeConfidenceInterval() for each node count individually.
    ConfidenceInterval zeroConfidenceInterval =
        params
            .noise()
            .computeConfidenceInterval(
                /* noisedX= */ 0.0,
                /* l0Sensitivity= */ params.treeHeight() * params.maxPartitionsContributed(),
                /* lInfSensitivity= */ params.maxContributionsPerPartition(),
                /* epsilon= */ params.epsilon(),
                /* delta= */ params.delta(),
                /* alpha= */ alphaPerCount);

    // Value of the bound that is being computed. The value is set to the tightest bound possible
    // and loosened successively as needed.
    double bound = boundType == LOWER ? params.upper() : params.lower();

    int index = ROOT_INDEX;
    // Search for the index of the leaf node containing the desired bound, starting at the root.
    while (index < leftmostLeafIndex) {
      int leftmostChildIndex = getLeftmostChild(index);
      int rightmostChildIndex = getRightmostChild(index);

      // Index of the node visited next in the search. The value is set to the tightest index
      // possible and loosened successively as needed.
      int nextIndex = boundType == LOWER ? Integer.MAX_VALUE : Integer.MIN_VALUE;
      // Rank used in the next iteration of the search. The value will be set with the first update
      // of nextIndex.
      double nextRank = Double.NaN;

      Map<Integer, ConfidenceInterval> childConfidenceIntervals = new HashMap<>();
      for (int i = leftmostChildIndex; i <= rightmostChildIndex; i++) {
        childConfidenceIntervals.put(
            i,
            ConfidenceInterval.create(
                getNoisedCount(i) + zeroConfidenceInterval.lowerBound(),
                getNoisedCount(i) + zeroConfidenceInterval.upperBound()));
      }

      // Let [l_i, u_i] denote the confidence interval of child node i. To find a lower bound b for
      // the quantiles that can be reached via a particular configuration of counts c_i such that
      // l_i ≤ c_i ≤ u_i, the counts to left of b should be as large as possible while the counts to
      // the right of b should be as small as possible. Thus, we set
      //    c_i = u_i if i <= j and c_i = l_i if i > j
      // for some index j. Similarly, an upper bound can be obtained by setting
      //    c_i = l_i if i <= j and c_i = u_i if i > j.
      //
      // Because we don't know the index j in advance, we go through all possible indices j and pick
      // whichever yields the smallest lower bound or largest upper bound.
      for (int j = leftmostChildIndex - 1; j <= rightmostChildIndex; j++) {
        Map<Integer, Double> countBounds = new HashMap<>();
        for (int i = leftmostChildIndex; i <= j; i++) {
          countBounds.put(
              i,
              boundType == LOWER
                  ? childConfidenceIntervals.get(i).upperBound()
                  : childConfidenceIntervals.get(i).lowerBound());
        }
        for (int i = j + 1; i <= rightmostChildIndex; i++) {
          countBounds.put(
              i,
              boundType == LOWER
                  ? childConfidenceIntervals.get(i).lowerBound()
                  : childConfidenceIntervals.get(i).upperBound());
        }

        IndexAndRank nextIndexAndRank =
            getNextIndexAndRank(rank, leftmostChildIndex, rightmostChildIndex, countBounds);

        if (nextIndexAndRank == null) {
          // All child nodes are considred empty. Update the bound with a linear interpolation of
          // the smallest and largest value associated with the current node if the result yields
          // a looser bound.
          if (boundType == LOWER) {
            bound = min(bound, (1 - rank) * getLeftValue(index) + rank * getRightValue(index));
          } else {
            bound = max(bound, (1 - rank) * getLeftValue(index) + rank * getRightValue(index));
          }
        } else {
          // Update nextIndex and nextRank if this results in a looser bound.
          if ((boundType == LOWER && nextIndexAndRank.index() <= nextIndex)
              || (boundType == UPPER && nextIndexAndRank.index() >= nextIndex)) {
            if (nextIndexAndRank.index() != nextIndex) {
              nextIndex = nextIndexAndRank.index();
              nextRank = nextIndexAndRank.rank();
            } else if ((boundType == LOWER && nextIndexAndRank.rank() < nextRank)
                || (boundType == UPPER && nextIndexAndRank.rank() > nextRank)) {
              nextRank = nextIndexAndRank.rank();
            }
          }
        }
      }

      // Check if the current node was considered empty for all values of j (this is the case when
      // nextRank has not been set). If so, the search can be stopped and the bound returned.
      // Otherwise continue the search in the next node with the respective new rank.
      if (Double.isNaN(nextRank)) {
        return bound;
      }
      index = nextIndex;
      rank = nextRank;
    }
    // The search has reached a leaf node. In this case we either return a linear interpolation
    // between the smallest and the largest value associated with the leaf node, or the bound
    // computed so far should it be looser.
    double linearInterpolation = (1 - rank) * getLeftValue(index) + rank * getRightValue(index);
    return boundType == LOWER ? min(bound, linearInterpolation) : max(bound, linearInterpolation);
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

  /*
   * Retruns a pair of the index of the child node visited next in the quantile search together with
   * the rank for the next iteration of the search. If all child nodes are considered empty, null is
   * returned.
   */
  private static IndexAndRank getNextIndexAndRank(
      double rank,
      int leftmostChildIndex,
      int rightmostChildIndex,
      Map<Integer, Double> nodeCounts) {

    double totalCount = 0.0;
    for (int i = leftmostChildIndex; i <= rightmostChildIndex; i++) {
      totalCount += max(0.0, nodeCounts.get(i));
    }

    double correctedTotalCount = 0.0;
    for (int i = leftmostChildIndex; i <= rightmostChildIndex; i++) {
      // Treat child nodes contributing less than a gamma fraction to the total count as empty
      // subtrees.
      correctedTotalCount += nodeCounts.get(i) >= totalCount * GAMMA ? nodeCounts.get(i) : 0.0;
    }
    if (correctedTotalCount == 0.0) {
      // Either all counts are 0.0 or no child node contributes more than a gamma fraction to
      // the total count (the latter can only happen when gamma > 1 / branching factor, which is
      // not the case for the default branching factor). This means that all child nodes are
      // considered empty.
      return null;
    }

    // Determine the child node whose subtree contains the bound.
    double partialCount = 0.0;
    for (int i = leftmostChildIndex; true; i++) {
      double count = nodeCounts.get(i);
      // Skip child nodes contributing less than gamma to the total count.
      if (count >= totalCount * GAMMA) {
        partialCount += count;
        // Check if the bound is in the current child's subtree.
        if (partialCount / correctedTotalCount >= rank - NUMERICAL_TOLERANCE) {
          double nextRank =
              (rank - (partialCount - count) / correctedTotalCount) / (count / correctedTotalCount);
          // Clamping rank to a value between 0.0 and 1.0. Note that rank can become greater than
          // 1 because of the numerical tolerance. Values less than 0.0 should not occur. The
          // respective clamping is set in place to be on the safe side.
          nextRank = min(max(0.0, nextRank), 1.0);
          return new AutoValue_BoundedQuantiles_IndexAndRank(i, nextRank);
        }
      }
    }
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
   * #computeResult(double)} has been called. Moreover, after this instance of {@link
   * BoundedQuantiles} has been serialized once, further modification and queries are not possible
   * anymore.
   *
   * @throws IllegalStateException if this instance of {@link BoundedQuantiles} has already been
   *     queried.
   */
  public byte[] getSerializableSummary() {
    Preconditions.checkState(
        state != AggregationState.RESULT_RETURNED, "Distribution cannot be serialized.");

    state = AggregationState.SERIALIZED;

    return BoundedQuantilesSummary.newBuilder()
        .putAllQuantileTree(tree)
        .setEpsilon(params.epsilon())
        .setDelta(params.delta())
        .setLower(params.lower())
        .setUpper(params.upper())
        .setMaxPartitionsContributed(params.maxPartitionsContributed())
        .setMaxContributionsPerPartition(params.maxContributionsPerPartition())
        .setMechanismType(params.noise().getMechanismType())
        .setTreeHeight(params.treeHeight())
        .setBranchingFactor(params.branchingFactor())
        .build()
        .toByteArray();
  }

  /**
   * Merges the output of {@link #getSerializableSummary()} from a different instance of {@link
   * BoundedQuantiles} with this instance. Intended to be used in the context of distributed
   * computation.
   *
   * @throws IllegalArgumentException if the parameters of the two instances (epsilon, delta,
   *     contribution bounds, etc.) do not match or if the passed serialized summary is invalid.
   * @throws IllegalStateException if this instance of {@link BoundedQuantiles} has already been
   *     queried or serialized.
   */
  public void mergeWith(byte[] otherBoundedQuantilesSummary) {
    Preconditions.checkState(
        state.equals(AggregationState.DEFAULT), "Distributions cannot be merged.");

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
        "Failed to merge: unequal values of tree height. " + "treeHeight1 = %s, treeHeight2 = %s",
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
  abstract static class IndexAndRank {

    abstract int index();

    abstract double rank();
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

    abstract int treeHeight();

    abstract int branchingFactor();

    @AutoValue.Builder
    public abstract static class Builder {

      private static BoundedQuantiles.Params.Builder newBuilder() {
        BoundedQuantiles.Params.Builder builder = new AutoValue_BoundedQuantiles_Params.Builder();
        // Provides LaplaceNoise as a default noise generator.
        builder.noise(new LaplaceNoise());
        // Since Laplace noise doesn't use delta, set it to 0.0.
        builder.delta(0.0);
        // A default tree height of 4 and branching factor of 16 divides the domain of possible
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
      public abstract BoundedQuantiles.Params.Builder delta(double value);

      /**
       * @deprecated use {@link #delta(double)}.
       *     <p>TODO: migrate clients and delete this method.
       */
      @Deprecated
      public BoundedQuantiles.Params.Builder delta(Double value) {
        double primitiveDelta = value == null ? 0.0 : value;
        return delta(primitiveDelta);
      }

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
       * Lower bound for the entries added to the distribution. Any entries smaller than this value
       * will be set to this value.
       */
      public abstract BoundedQuantiles.Params.Builder lower(double value);

      /**
       * Upper bound for the entries added to the distribution. Any entries greater than this value
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

      abstract BoundedQuantiles.Params autoBuild();

      public BoundedQuantiles build() {
        BoundedQuantiles.Params params = autoBuild();
        // No need to check noise nullability: the noise is defaulted to Laplace noise.
        DpPreconditions.checkEpsilon(params.epsilon());
        DpPreconditions.checkNoiseDelta(params.delta(), params.noise());
        DpPreconditions.checkMaxPartitionsContributed(params.maxPartitionsContributed());
        DpPreconditions.checkMaxContributionsPerPartition(params.maxContributionsPerPartition());
        DpPreconditions.checkBounds(params.lower(), params.upper());
        DpPreconditions.checkBoundsNotEqual(params.lower(), params.upper());
        checkTreeHeight(params.treeHeight());
        checkBranchingFactor(params.branchingFactor());

        return new BoundedQuantiles(params);
      }
    }
  }
}

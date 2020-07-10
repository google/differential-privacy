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
import com.google.differentialprivacy.SummaryOuterClass.CountSummary;
import com.google.protobuf.InvalidProtocolBufferException;
import javax.annotation.Nullable;

/**
 * Calculates a differentially private count for a collection of values.
 *
 * <p>This class allows an individual privacy unit (e.g., a single user) to contribute data to
 * multiple different partitions. The class does not check whether the number of partitions is
 * within the specified bounds. This is the responsibility of the caller.
 *
 * <p>This class assumes that each privacy unit may contribute to a single partition only once
 * (i.e., only one data contribution per privacy unit per partition), it doesn't do clamping. For
 * datasets with multiple contributions from the same user to a single partition {@link BoundedSum}
 * should be used instead.
 *
 * <p>The user can provide a {@link Noise} instance which will be used to generate the noise. If no
 * instance is specified, {@link LaplaceNoise} is applied.
 *
 * <p>Note: this class is not thread-safe.
 *
 * <p>For more implementation details, see {@link #computeResult()}.
 *
 * <p>For general details and key definitions, see <a
 * href="https://github.com/google/differential-privacy/blob/master/differential_privacy.md#key-definition">
 * the introduction to Differential Privacy</a>.
 */
public class Count {
  private final Params params;
  private long rawCount;

  // Was the count returned to the user?
  private boolean resultReturned;

  private Count(Params params) {
    this.params = params;
  }

  public static Params.Builder builder() {
    return Params.Builder.newBuilder();
  }

  /** Increments count by one. */
  public void increment() {
    incrementBy(1);
  }

  /**
   * Increments count by the given value. Note, that this shouldn't be used to count multiple
   * contributions to a partition from the same user.
   */
  public void incrementBy(long count) {
    if (resultReturned) {
      throw new IllegalStateException(
          "The count has already been calculated and returned. It cannot be amended.");
    }

    this.rawCount += count;
  }

  /**
   * Calculates and returns a differentially private count of elements added using
   * {@link #increment} and {@link #incrementBy}. The method can be called only once for a given
   * collection of elements. All subsequent calls will throw an exception.
   */
  public long computeResult() {
    if (resultReturned) {
      throw new IllegalStateException("The result can be calculated and returned only once.");
    }

    resultReturned = true;
    return params
        .noise()
        .addNoise(
            rawCount,
            params.maxPartitionsContributed(),
            params.maxContributionsPerPartition(),
            params.epsilon(),
            params.delta());
  }

  /**
   * Returns a serializable version of the current state of {@link Count} and the parameters used to
   * calculate it. After calling this method, this instance of Count will be unusable, since the
   * result can only be output once.
   */
  public byte[] getSerializableSummary() {
    if (resultReturned) {
      throw new IllegalStateException(
          "The count has already been returned. It cannot be returned again.");
    }

    CountSummary.Builder builder =
        CountSummary.newBuilder()
            .setCount(rawCount)
            .setEpsilon(params.epsilon())
            .setMaxPartitionsContributed(params.maxPartitionsContributed())
            .setMaxContributionsPerPartition(params.maxContributionsPerPartition())
            .setMechanismType(params.noise().getMechanismType());
    if (params.delta() != null) {
      builder.setDelta(params.delta());
    }

    // Record that this object is no longer suitable for producing a differentially private count,
    // since serialization exposes the object's raw state.
    resultReturned = true;

    return builder.build().toByteArray();
  }

  /**
   * Merges this instance with the output of {@link #getSerializableSummary()} from a different
   * {@link Count} and stores the merged result in this instance. This is required in the
   * distributed calculations context for merging partial results.
   *
   * @throws IllegalArgumentException if not all config parameters (e.g., epsilon)
   *     are equal or if the passed serialized count is invalid.
   * @throws IllegalStateException if this count has already been calculated or serialized.
   */
  public void mergeWith(byte[] otherCountSummary) {
    if (resultReturned) {
      throw new IllegalStateException(
          "The count has already been calculated and returned. It cannot be merged.");
    }

    CountSummary otherSummaryParsed;
    try {
      otherSummaryParsed = CountSummary.parseFrom(otherCountSummary);
    } catch (InvalidProtocolBufferException pbe) {
      throw new IllegalArgumentException(pbe);
    }

    checkMergeParametersAreEqual(otherSummaryParsed);
    this.rawCount += otherSummaryParsed.getCount();
  }

  private void checkMergeParametersAreEqual(CountSummary otherCount) {
    DpPreconditions.checkMergeMechanismTypesAreEqual(
        params.noise().getMechanismType(), otherCount.getMechanismType());
    DpPreconditions.checkMergeEpsilonAreEqual(
        params.epsilon(), otherCount.getEpsilon());
    DpPreconditions.checkMergeDeltaAreEqual(
        params.delta(), otherCount.getDelta());
    DpPreconditions.checkMergeMaxPartitionsContributedAreEqual(
        params.maxPartitionsContributed(),
        otherCount.getMaxPartitionsContributed());
    DpPreconditions.checkMergeMaxContributionsPerPartitionAreEqual(
        params.maxContributionsPerPartition(), otherCount.getMaxContributionsPerPartition());
  }



  @AutoValue
  public abstract static class Params {
    abstract Noise noise();

    abstract double epsilon();

    @Nullable
    abstract Double delta();

    abstract int maxPartitionsContributed();

    abstract int maxContributionsPerPartition();

    @AutoValue.Builder
    public abstract static class Builder {
      private static Builder newBuilder() {
        Builder builder = new AutoValue_Count_Params.Builder();
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

      /** Distribution from which the noise will be generated and added to the count. */
      public abstract Builder noise(Noise value);

      /**
       * Maximum number of contributions associated with a single privacy unit (e.g., an
       * individual) to a single partition. This is used to calculate the sensitivity of the count
       * operation. This is not public because it should be used only by other aggregation functions
       *  inside the library. See {@link Count} for more details.
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

/*
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.privacy.differentialprivacy.pipelinedp4j.core

import com.google.errorprone.annotations.Immutable
import com.google.privacy.differentialprivacy.BoundedQuantiles
import com.google.privacy.differentialprivacy.GaussianNoise
import com.google.privacy.differentialprivacy.LaplaceNoise
import com.google.privacy.differentialprivacy.Noise
import com.google.privacy.differentialprivacy.ZeroNoise
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.COUNT
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.MEAN
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.SUM
import com.google.privacy.differentialprivacy.pipelinedp4j.core.budget.AllocatedBudget
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.CompoundAccumulator
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.CountAccumulator
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.DpAggregates
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.MeanAccumulator
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.PerFeature
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.PrivacyIdContributions
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.PrivacyIdCountAccumulator
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.QuantilesAccumulator
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.SumAccumulator
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.VarianceAccumulator
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.VectorSumAccumulator
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.compoundAccumulator
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.copy
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.countAccumulator
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.dpAggregates
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.meanAccumulator
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.perFeature
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.privacyIdCountAccumulator
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.quantilesAccumulator
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.sumAccumulator
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.varianceAccumulator
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.vectorSumAccumulator
import com.google.protobuf.ByteString
import java.io.Serializable
import kotlin.math.abs
import kotlin.math.max
import org.apache.commons.math3.linear.ArrayRealVector
import org.apache.commons.math3.linear.RealVector

/**
 * An entity that aggregates input values and adds noise for differential privacy (DP).
 *
 * Concrete implementations handle specific metric types (e.g., counts, sums, means, quantiles).
 *
 * @param AccumulatorT type of accumulator used to store intermediate aggregation results.
 * @param OutputT type of output produced by the combiner, typically a noisy metric.
 */
sealed interface Combiner<AccumulatorT, OutputT> : Serializable {
  /**
   * Whether per-partition contribution bounding has to be performed before calling
   * [createAccumulator].
   *
   * Privacy levels without contribution bounding should not affect this value, just assume a
   * privacy level with contribution bounding.
   */
  val requiresPerPartitionBoundedInput: Boolean

  /**
   * Creates a new accumulator instance from the given privacy id contributions.
   *
   * @param contributions the privacy id contributions to initialize the accumulator with.
   * @return a new accumulator instance representing the aggregated contributions.
   */
  fun createAccumulator(contributions: PrivacyIdContributions): AccumulatorT

  /**
   * Merges two accumulators into one.
   *
   * @param accumulator1 the first accumulator to merge.
   * @param accumulator2 the second accumulator to merge.
   * @return a new accumulator containing the combined results of the two input accumulators.
   */
  fun mergeAccumulators(accumulator1: AccumulatorT, accumulator2: AccumulatorT): AccumulatorT

  /**
   * Returns an anonymized metric computed on the given accumulator.
   *
   * @param accumulator the accumulator containing the aggregated data.
   * @return the computed anonymized metrics of type `OutputT`.
   */
  fun computeMetrics(accumulator: AccumulatorT): OutputT

  /**
   * Creates an empty accumulator that holds no value.
   *
   * @return an empty accumulator of type `AccumulatorT`.
   */
  fun emptyAccumulator() = createAccumulator(PrivacyIdContributions.getDefaultInstance())
}

/**
 * A [Combiner] for the [MetricType.COUNT]. It returns a noisy count of input items.
 *
 * @property aggregationParams parameters that control the aggregation behavior.
 * @property budget the amount of privacy budget that can be used by the combiner.
 * @property noiseFactory allows for passing the noise generator as a parameter in order to be able
 *   to mock it in tests.
 */
class CountCombiner(
  private val aggregationParams: AggregationParams,
  private val budget: AllocatedBudget,
  private val noiseFactory: (NoiseKind) -> Noise,
  private val executionMode: ExecutionMode,
) : Combiner<CountAccumulator, Double> {
  override val requiresPerPartitionBoundedInput = false

  /**
   * Creates a new [CountAccumulator] initialized with the size of the input contributions,
   * potentially bounded based on the privacy level and aggregation parameters.
   *
   * @param contributions the privacy id contributions to initialize the accumulator with.
   * @return a new [CountAccumulator] instance.
   */
  override fun createAccumulator(contributions: PrivacyIdContributions): CountAccumulator =
    countAccumulator {
      count =
        contributions
          .size()
          .toLong()
          .coerceInIfContributionBoundingEnabled(
            0,
            aggregationParams.maxContributionsPerPartition!!.toLong(),
            aggregationParams,
            executionMode,
          )
    }

  /**
   * Merges two [CountAccumulator] instances by summing their counts.
   *
   * @param accumulator1 the first accumulator to merge.
   * @param accumulator2 the second accumulator to merge.
   * @return a new [CountAccumulator] instance with the combined count.
   */
  override fun mergeAccumulators(
    accumulator1: CountAccumulator,
    accumulator2: CountAccumulator,
  ): CountAccumulator = countAccumulator { count = accumulator1.count + accumulator2.count }

  /**
   * Computes a noisy count from the given [CountAccumulator].
   *
   * @param accumulator the accumulator containing the aggregated count.
   * @return a noisy count with added differential privacy guarantees.
   */
  override fun computeMetrics(accumulator: CountAccumulator): Double {
    val noise = noiseFactory(aggregationParams.noiseKind)
    return noise.addNoise(
      accumulator.count.toDouble(),
      aggregationParams.maxPartitionsContributed!!,
      aggregationParams.maxContributionsPerPartition!!.toDouble(),
      budget.epsilon(),
      budget.delta(),
    )
  }
}

/**
 * A [Combiner] for the [MetricType.PRIVACY_ID_COUNT].
 *
 * It returns a noisy count of privacy ids.
 *
 * @property aggregationParams parameters controlling the aggregation process.
 * @property budget the amount of privacy budget that can be used by the combiner.
 * @property noiseFactory allows for passing the noise generator as a parameter in order to be able
 *   to mock it in tests.
 */
class PrivacyIdCountCombiner(
  private val aggregationParams: AggregationParams,
  private val budget: AllocatedBudget,
  private val noiseFactory: (NoiseKind) -> Noise,
  private val executionMode: ExecutionMode,
) : Combiner<PrivacyIdCountAccumulator, Double> {
  override val requiresPerPartitionBoundedInput = false

  /**
   * Creates a [PrivacyIdCountAccumulator] initialized with the count of unique privacy ids in the
   * contributions. If the contributions are empty (representing an empty public partition), the
   * count is 0. Otherwise, the count is 1, as the contributions correspond to a single privacy id.
   *
   * @param contributions privacy id contributions to initialize the accumulator with.
   * @return a new [PrivacyIdCountAccumulator] instance.
   */
  override fun createAccumulator(contributions: PrivacyIdContributions): PrivacyIdCountAccumulator {
    val privacyIdCount =
      if (contributions.size() == 0) {
        // Empty public partition, no privacy ids.
        0
      } else {
        // `contributions` corresponds to 1 privacy id.
        1
      }

    return privacyIdCountAccumulator { count = privacyIdCount.toLong() }
  }

  /**
   * Merges two [PrivacyIdCountAccumulator] instances by summing their counts.
   *
   * @param accumulator1 the first accumulator to merge.
   * @param accumulator2 the second accumulator to merge.
   * @return a new [PrivacyIdCountAccumulator] instance with the combined count.
   */
  override fun mergeAccumulators(
    accumulator1: PrivacyIdCountAccumulator,
    accumulator2: PrivacyIdCountAccumulator,
  ): PrivacyIdCountAccumulator = privacyIdCountAccumulator {
    count = accumulator1.count + accumulator2.count
  }

  /**
   * Computes a noisy count of unique privacy ids from the given [PrivacyIdCountAccumulator].
   *
   * Noise is added using the specified noise mechanism (`noiseFactory`), privacy budget (`budget`),
   * and the maximum number of partitions contributed
   * (`aggregationParams.maxPartitionsContributed`). The lInfSensitivity is set to 1.0, as each
   * privacy id contributes at most 1 to the count.
   *
   * @param accumulator the accumulator containing the aggregated privacy id count.
   * @return a noisy count of unique privacy ids.
   */
  override fun computeMetrics(accumulator: PrivacyIdCountAccumulator): Double =
    noiseFactory(aggregationParams.noiseKind)
      .addNoise(
        accumulator.count.toDouble(),
        aggregationParams.maxPartitionsContributed!!,
        /* lInfSensitivity = */ 1.0,
        budget.epsilon(),
        budget.delta(),
      )
}

/**
 * A [Combiner] for the computing exact (i.e. not anonymized) privacy id count.
 *
 * The exact privacy id count is used for pre-aggregation partition selection. The exact privacy id
 * count is not anonymized and hence it cannot be returned in the in anonymized output. This
 * combiner should not be used together with PrivacyIdCountCombiner.
 */
class ExactPrivacyIdCountCombiner : Combiner<PrivacyIdCountAccumulator, Double> {
  override val requiresPerPartitionBoundedInput = false

  /**
   * Creates a [PrivacyIdCountAccumulator] initialized with a count of 1, assuming the contributions
   * represent a single privacy id.
   *
   * @param contributions the privacy id contributions to initialize the accumulator with. Must NOT
   *   be empty.
   * @return a new [PrivacyIdCountAccumulator] instance with a count of 1.
   */
  override fun createAccumulator(contributions: PrivacyIdContributions): PrivacyIdCountAccumulator {
    return privacyIdCountAccumulator { count = 1 }
  }

  /**
   * Merges two [PrivacyIdCountAccumulator] instances by summing their counts.
   *
   * @param accumulator1 the first accumulator to merge.
   * @param accumulator2 the second accumulator to merge.
   * @return a new [PrivacyIdCountAccumulator] instance with the combined count.
   */
  override fun mergeAccumulators(
    accumulator1: PrivacyIdCountAccumulator,
    accumulator2: PrivacyIdCountAccumulator,
  ): PrivacyIdCountAccumulator = privacyIdCountAccumulator {
    count = accumulator1.count + accumulator2.count
  }

  /**
   * This method is not supported for `ExactPrivacyIdCountCombiner`.
   *
   * Since the count is not anonymized, computing metrics would violate differential privacy.
   *
   * @throws UnsupportedOperationException always, as this method is not meant to be used.
   */
  override fun computeMetrics(accumulator: PrivacyIdCountAccumulator): Double =
    throw UnsupportedOperationException(
      "ExactPrivacyIdCountCombiner does not support compute_metrics."
    )
}

/**
 * A [Combiner] which computes [MetricType.PRIVACY_ID_COUNT] and performs post-aggregation partition
 * selection. When the noised privacy id count is smaller than the threshold, it returns null,
 * otherwise the noised privacy id count, which is anonymized. The threshold is computed from input
 * parameters to ensure Differential Privacy. This combiner should not be used together with
 * PrivacyIdCountCombiner.
 *
 * @property aggregationParams parameters controlling the aggregation process.
 * @property noiseBudget the amount of privacy budget that can be used by the combiner for the
 *   noise.
 * @property thresholdingBudget the amount of privacy budget that can be used by the combiner for
 *   the thresholding.
 * @property noiseFactory allows for passing the noise generator as a parameter in order to be able
 *   to mock it in tests.
 */
class PostAggregationPartitionSelectionCombiner(
  private val aggregationParams: AggregationParams,
  private val noiseBudget: AllocatedBudget,
  private val thresholdingBudget: AllocatedBudget,
  private val noiseFactory: (NoiseKind) -> Noise,
  private val executionMode: ExecutionMode,
) : Combiner<PrivacyIdCountAccumulator, Double?> {
  override val requiresPerPartitionBoundedInput = false

  /**
   * Creates a [PrivacyIdCountAccumulator] initialized with a count of 1, assuming there is at least
   * one contribution (representing one privacy id).
   *
   * @param contributions the privacy id contributions to initialize the accumulator with. Must NOT
   *   be empty.
   * @return a new [PrivacyIdCountAccumulator] instance with a count of 1.
   * @throws IllegalArgumentException if the contributions are empty.
   */
  override fun createAccumulator(contributions: PrivacyIdContributions): PrivacyIdCountAccumulator {
    require(contributions.size() > 0) {
      "There must be contributions for PostAggregationPartitionSelectionCombiner."
    }
    return privacyIdCountAccumulator { count = 1 }
  }

  /**
   * Merges two accumulators by summing their counts.
   *
   * @param accumulator1 the first accumulator to merge.
   * @param accumulator2 the second accumulator to merge.
   * @return a new accumulator with the combined count.
   */
  override fun mergeAccumulators(
    accumulator1: PrivacyIdCountAccumulator,
    accumulator2: PrivacyIdCountAccumulator,
  ): PrivacyIdCountAccumulator = privacyIdCountAccumulator {
    count = accumulator1.count + accumulator2.count
  }

  /**
   * Computes the noisy count of privacy ids and applies the post-aggregation partition selection
   * mechanism.
   *
   * @param accumulator the accumulator containing the aggregated privacy id count.
   * @return the noisy count of privacy ids if the partition is kept, or null if it is discarded.
   */
  override fun computeMetrics(accumulator: PrivacyIdCountAccumulator): Double? =
    getPartitionSelector().addNoiseIfShouldKeep(accumulator.count)

  /**
   * Returns a [PostAggregationPartitionSelector] which is used for post-aggregation partition
   * selection.
   *
   * @return a [PostAggregationPartitionSelector] instance.
   */
  internal fun getPartitionSelector(): PostAggregationPartitionSelector =
    PostAggregationPartitionSelectorImpl(
      aggregationParams.maxPartitionsContributed!!,
      aggregationParams.noiseKind,
      aggregationParams.preThreshold,
      noiseBudget,
      thresholdingBudget,
      noiseFactory,
    )
}

/**
 * A [Combiner] for the [MetricType.SUM].
 *
 * It returns a noisy sum of input items.
 *
 * @property aggregationParams parameters controlling the aggregation process.
 * @property budget the amount of privacy budget that can be used by the combiner.
 * @property noiseFactory allows for passing the noise generator as a parameter in order to be able
 *   to mock it in tests.
 */
class SumCombiner(
  private val aggregationParams: AggregationParams,
  private val budget: AllocatedBudget,
  private val noiseFactory: (NoiseKind) -> Noise,
  private val executionMode: ExecutionMode,
  private val featureSpec: ScalarFeatureSpec,
) : Combiner<SumAccumulator, Double>, Serializable {
  private val featureMinTotalValue =
    checkNotNull(featureSpec.minTotalValue) {
      "minTotalValue should be set when requesting SUM metric for feature ${featureSpec.featureId}"
    }
  private val featureMaxTotalValue =
    checkNotNull(featureSpec.maxTotalValue) {
      "maxTotalValue should be set when requesting SUM metric for feature ${featureSpec.featureId}"
    }

  override val requiresPerPartitionBoundedInput = false

  /**
   * Creates a [SumAccumulator] from the given privacy id contributions.
   *
   * **Important Note:** The `contributions` must contain all contributions by a single privacy id
   * to a single partition.
   *
   * @param contributions the contributions of a single privacy id to a single partition.
   * @return a new `SumAccumulator` initialized with the sum of the contributions, potentially
   *   bounded based on the privacy level.
   */
  override fun createAccumulator(contributions: PrivacyIdContributions): SumAccumulator {
    val featureContributions =
      contributions.featuresList.find { it.featureId == featureSpec.featureId }
    return sumAccumulator {
      featureId = featureSpec.featureId
      sum =
        if (featureContributions == null) {
          0.0
        } else {
          featureContributions.singleValueContributionsList
            .sum()
            .coerceInIfContributionBoundingEnabled(
              featureMinTotalValue,
              featureMaxTotalValue,
              aggregationParams,
              executionMode,
            )
        }
    }
  }

  /**
   * Merges two [SumAccumulator] instances by adding their sums.
   *
   * @param accumulator1 the first accumulator to merge.
   * @param accumulator2 the second accumulator to merge.
   * @return a new `SumAccumulator` with the combined sum.
   */
  override fun mergeAccumulators(
    accumulator1: SumAccumulator,
    accumulator2: SumAccumulator,
  ): SumAccumulator = sumAccumulator {
    featureId = accumulator1.featureId
    sum = accumulator1.sum + accumulator2.sum
  }

  /**
   * Computes a noisy sum from the given [SumAccumulator].
   *
   * @param accumulator the accumulator containing the aggregated sum.
   * @return a noisy sum with added differential privacy guarantees.
   */
  override fun computeMetrics(accumulator: SumAccumulator): Double {
    val noise = noiseFactory(aggregationParams.noiseKind)
    val lInfSensitivity = max(abs(featureMinTotalValue), abs(featureMaxTotalValue))

    return noise.addNoise(
      accumulator.sum,
      aggregationParams.maxPartitionsContributed!!,
      lInfSensitivity,
      budget.epsilon(),
      budget.delta(),
    )
  }
}

/**
 * A [Combiner] for the [MetricType.VECTOR_SUM].
 *
 * It returns a noisy sum of input vectors.
 *
 * @property aggregationParams parameters controlling the aggregation process.
 * @property budget the amount of privacy budget that can be used by the combiner.
 * @property noiseFactory allows for passing the noise generator as a parameter in order to be able
 *   to mock it in tests.
 */
class VectorSumCombiner(
  private val aggregationParams: AggregationParams,
  private val budget: AllocatedBudget,
  private val noiseFactory: (NoiseKind) -> Noise,
  private val executionMode: ExecutionMode,
  private val featureSpec: VectorFeatureSpec,
) : Combiner<VectorSumAccumulator, List<Double>>, Serializable {
  private val normKind = featureSpec.normKind
  private val vectorMaxTotalNorm = featureSpec.vectorMaxTotalNorm
  private val vectorSize = featureSpec.vectorSize

  override val requiresPerPartitionBoundedInput = false

  /**
   * Creates a [VectorSumAccumulator] from the given privacy id contributions.
   *
   * **Important Note:** The `contributions` must contain all contributions by a single privacy id
   * to a single partition.
   *
   * @param contributions the contributions of a single privacy id to a single partition.
   * @return a new `VectorSumAccumulator` initialized with the sum of the contributions, potentially
   *   bounded based on the privacy level.
   */
  override fun createAccumulator(contributions: PrivacyIdContributions) = vectorSumAccumulator {
    val featureContributions =
      contributions.featuresList.find { it.featureId == featureSpec.featureId }
    featureId = featureSpec.featureId
    sumsPerDimension +=
      featureContributions
        ?.multiValueContributionsList
        ?.map { contribution -> ArrayRealVector(contribution.valuesList.toDoubleArray()) }
        ?.reduceOrNull { acc, vector -> acc.add(vector) }
        ?.clipIfContributionBoundingEnabled(vectorMaxTotalNorm, normKind)
        ?.toArray()
        ?.asList() ?: List(vectorSize) { 0.0 }
  }

  /**
   * Merges two [VectorSumAccumulator] instances by summing their vectors.
   *
   * @param accumulator1 the first accumulator to merge.
   * @param accumulator2 the second accumulator to merge.
   * @return a new `VectorSumAccumulator` with the combined vector sum.
   */
  override fun mergeAccumulators(
    accumulator1: VectorSumAccumulator,
    accumulator2: VectorSumAccumulator,
  ) = vectorSumAccumulator {
    featureId = accumulator1.featureId
    sumsPerDimension +=
      accumulator1.sumsPerDimensionList.zip(accumulator2.sumsPerDimensionList).map { (e1, e2) ->
        e1 + e2
      }
  }

  /**
   * Computes a noisy vector sum from the given [VectorSumAccumulator].
   *
   * @param accumulator the accumulator containing the aggregated vector of sum.
   * @return a noisy vector sum with added differential privacy guarantees.
   */
  override fun computeMetrics(accumulator: VectorSumAccumulator): List<Double> {
    val noise = noiseFactory(aggregationParams.noiseKind)
    val vector = ArrayRealVector(accumulator.sumsPerDimensionList.toDoubleArray())
    val noisedVector =
      when (noise) {
        is LaplaceNoise -> {
          val l1Sensitivity =
            calculateL1Sensistivity(
              normKind,
              vectorMaxTotalNorm,
              vectorSize,
              aggregationParams.maxPartitionsContributed!!,
            )
          vector.map { noise.addNoise(it, l1Sensitivity, budget.epsilon(), budget.delta()) }
        }
        is GaussianNoise -> {
          val l2Sensitivity =
            calculateL2Sensistivity(
              normKind,
              vectorMaxTotalNorm,
              vectorSize,
              aggregationParams.maxPartitionsContributed!!,
            )
          vector.map { noise.addNoise(it, l2Sensitivity, budget.epsilon(), budget.delta()) }
        }
        is ZeroNoise -> vector.map { it }
        else -> throw IllegalArgumentException("Unsupported noise kind: ${noise.javaClass}.")
      }
    return noisedVector.toArray().asList()
  }

  private fun RealVector.clipIfContributionBoundingEnabled(
    maxNorm: Double,
    normKind: NormKind,
  ): RealVector {
    if (!aggregationParams.applyPerPartitionBounding(executionMode)) {
      return this.copy()
    }
    // L_INF is treated differently. When clamping, each component is clamped independently.
    if (normKind == NormKind.L_INF) {
      return this.map { it.coerceIn(-maxNorm, maxNorm) }
    }

    val currentNorm =
      when (normKind) {
        NormKind.L1 -> this.l1Norm
        NormKind.L2 -> this.norm
        // L_INF is handled above.
        else -> throw IllegalStateException("Unsupported norm kind: $normKind")
      }
    return if (currentNorm <= maxNorm) {
      this.copy()
    } else {
      this.mapMultiply(maxNorm / currentNorm)
    }
  }

  private companion object {
    fun calculateL1Sensistivity(
      normKind: NormKind,
      maxNorm: Double,
      vectorSize: Int,
      maxPartitionsContributed: Int,
    ): Double {
      val onePartitionSensitivity =
        when (normKind) {
          NormKind.L1 -> maxNorm
          NormKind.L2 ->
            throw IllegalStateException(
              "L2 norm vector norm is not supported for Laplace mechanism."
            )
          NormKind.L_INF -> maxNorm * vectorSize
        }
      return onePartitionSensitivity * maxPartitionsContributed
    }

    fun calculateL2Sensistivity(
      normKind: NormKind,
      maxNorm: Double,
      vectorSize: Int,
      maxPartitionsContributed: Int,
    ): Double {
      val onePartitionSensitivity =
        when (normKind) {
          NormKind.L1 ->
            throw IllegalStateException(
              "L1 norm vector norm is not supported for Gaussian mechanism."
            )
          NormKind.L2 -> maxNorm
          NormKind.L_INF -> maxNorm * Math.sqrt(vectorSize.toDouble())
        }
      return onePartitionSensitivity * Math.sqrt(maxPartitionsContributed.toDouble())
    }
  }
}

/**
 * A [Combiner] for the [MetricType.MEAN].
 *
 * It returns a noisy mean of input items. It can also return count and sum if requested by the
 * user.
 *
 * @property aggregationParams parameters controlling the aggregation process.
 * @property countBudget the allocated privacy budget for the count calculation.
 * @property sumBudget the allocated privacy budget for the sum calculation.
 * @property noiseFactory allows for passing the noise generator as a parameter in order to be able
 *   to mock it in tests.
 */
class MeanCombiner(
  private val aggregationParams: AggregationParams,
  private val countBudget: AllocatedBudget,
  private val sumBudget: AllocatedBudget,
  private val noiseFactory: (NoiseKind) -> Noise,
  private val executionMode: ExecutionMode,
  private val featureSpec: ScalarFeatureSpec,
) : Combiner<MeanAccumulator, MeanCombinerResult>, Serializable {
  private val featureMinValue =
    checkNotNull(featureSpec.minValue) {
      "minValue should be set when requesting mean metric for feature ${featureSpec.featureId}"
    }
  private val featureMaxValue =
    checkNotNull(featureSpec.maxValue) {
      "maxValue should be set when requesting mean metric for feature ${featureSpec.featureId}"
    }
  private val midValue = (featureMinValue + featureMaxValue) / 2
  private val returnCount = aggregationParams.nonFeatureMetrics.any { it.type == COUNT }
  private val returnSum = featureSpec.metrics.any { it.type == SUM }

  override val requiresPerPartitionBoundedInput = true

  /**
   * **Important Note:** the [contributions] passed to this function must all be contributions of a
   * particular privacy id into a particular partition **sub-sampled** to
   * **aggregationParams.maxContributionsPerPartition**
   *
   * @param contributions privacy id contributions for a specific privacy id and partition.
   * @return a new [MeanAccumulator] with the count and normalized sum of the contributions.
   */
  override fun createAccumulator(contributions: PrivacyIdContributions): MeanAccumulator =
    // All input values are normalized to be their difference from the middle of the
    // input range. That allows us to calculate the sum of all input values with
    // half the sensitivity it would otherwise take for better accuracy (as compared
    // to doing noisy sum / noisy count).
    meanAccumulator {
      val featureContributions =
        contributions.featuresList.find { it.featureId == featureSpec.featureId }
      val values = featureContributions?.singleValueContributionsList ?: emptyList()
      featureId = featureSpec.featureId
      count = values.size.toLong()
      normalizedSum =
        values
          .map {
            it.coerceInIfContributionBoundingEnabled(
              featureMinValue,
              featureMaxValue,
              aggregationParams,
              executionMode,
            ) - midValue
          }
          .sum()
    }

  /**
   * Merges two [MeanAccumulator] instances by summing their counts and normalized sums.
   *
   * @param accumulator1 the first accumulator to merge.
   * @param accumulator2 the second accumulator to merge.
   * @return a new [MeanAccumulator] with the combined counts and normalized sums.
   */
  override fun mergeAccumulators(
    accumulator1: MeanAccumulator,
    accumulator2: MeanAccumulator,
  ): MeanAccumulator = meanAccumulator {
    featureId = accumulator1.featureId
    count = accumulator1.count + accumulator2.count
    normalizedSum = accumulator1.normalizedSum + accumulator2.normalizedSum
  }

  /**
   * Computes the DP mean from the given [MeanAccumulator].
   *
   * @param accumulator the accumulator containing aggregated count and normalized sum.
   * @return a [MeanCombinerResult] containing the DP mean, and optionally DP sum and count.
   */
  override fun computeMetrics(accumulator: MeanAccumulator): MeanCombinerResult {
    val dpCount = getNoisedCount(accumulator.count, aggregationParams, countBudget, noiseFactory)
    val dpNormalizedSum =
      getNoisedNormalizedSum(
        accumulator.normalizedSum,
        featureMaxValue,
        midValue,
        aggregationParams,
        sumBudget,
        noiseFactory,
      )
    // Adding midValue denormalize mean to [minValue, maxValue] range.
    val dpMean = dpNormalizedSum / dpCount + midValue
    val outSum = if (returnSum) dpMean * dpCount else null
    val outCount = if (returnCount) dpCount else null
    return MeanCombinerResult(dpMean, outSum, outCount)
  }
}

/**
 * Represents the result of the [MeanCombiner].
 *
 * @property mean the differentially private mean.
 * @property sum the differentially private sum (if requested).
 * @property count the differentially private count (if requested).
 */
@Immutable
data class MeanCombinerResult(val mean: Double, val sum: Double?, val count: Double?) :
  Serializable

/**
 * A [Combiner] for the [MetricType.QUANTILES].
 *
 * It returns noisy quantiles for the requested ranks. The output quantiles are sorted by ranks.
 *
 * @property ranks a list of ranks for which quantiles will be computed. The ranks must be between 0
 *   (inclusive) and 1 (inclusive) and sorted in ascending order.
 * @property aggregationParams parameters controlling the aggregation process.
 * @property budget the amount of privacy budget that can be used by the combiner.
 * @property noiseFactory allows for passing the noise generator as a parameter in order to be able
 *   to mock it in tests.
 */
class QuantilesCombiner(
  private val sortedRanks: List<Double>,
  private val aggregationParams: AggregationParams,
  private val budget: AllocatedBudget,
  private val noiseFactory: (NoiseKind) -> Noise,
  private val executionMode: ExecutionMode,
  private val featureSpec: ScalarFeatureSpec,
) : Combiner<QuantilesAccumulator, List<Double>>, Serializable {
  private val featureMinValue =
    checkNotNull(featureSpec.minValue) {
      "minValue should be set when requesting quantiles metric for feature ${featureSpec.featureId}"
    }
  private val featureMaxValue =
    checkNotNull(featureSpec.maxValue) {
      "maxValue should be set when requesting quantiles metric for feature ${featureSpec.featureId}"
    }

  override val requiresPerPartitionBoundedInput = true

  /**
   * Creates a `QuantilesAccumulator` from privacy id contributions.
   *
   * **Important Note:** the [contributions] passed to this function must all be contributions of a
   * particular privacy id into a particular partition **sub-sampled** to
   * **aggregationParams.maxContributionsPerPartition**
   *
   * @param contributions privacy id contributions.
   * @return a new `QuantilesAccumulator` containing a serialized summary of the quantiles.
   */
  override fun createAccumulator(contributions: PrivacyIdContributions): QuantilesAccumulator =
    quantilesAccumulator {
      val featureContributions =
        contributions.featuresList.find { it.featureId == featureSpec.featureId }
      val values = featureContributions?.singleValueContributionsList ?: emptyList()
      featureId = featureSpec.featureId
      val boundedQuantiles = emptyBoundedQuantiles()
      boundedQuantiles.addEntries(values)
      serializedQuantilesSummary = ByteString.copyFrom(boundedQuantiles.serializableSummary)
    }

  /**
   * Merges two [QuantilesAccumulator] instances.
   *
   * @param accumulator1 the first accumulator to merge.
   * @param accumulator2 the second accumulator to merge.
   * @return a new `QuantilesAccumulator` containing the merged quantile summary.
   */
  override fun mergeAccumulators(
    accumulator1: QuantilesAccumulator,
    accumulator2: QuantilesAccumulator,
  ): QuantilesAccumulator = quantilesAccumulator {
    featureId = accumulator1.featureId
    val boundedQuantiles = emptyBoundedQuantiles()
    boundedQuantiles.mergeWith(accumulator1.serializedQuantilesSummary.toByteArray())
    boundedQuantiles.mergeWith(accumulator2.serializedQuantilesSummary.toByteArray())
    serializedQuantilesSummary = ByteString.copyFrom(boundedQuantiles.serializableSummary)
  }

  /**
   * Computes and returns a list of noisy quantiles for the specified ranks.
   *
   * The output quantiles are sorted in ascending order based on their ranks.
   *
   * @param accumulator the accumulator containing the aggregated data.
   * @return a list of noisy quantiles corresponding to the specified ranks.
   */
  override fun computeMetrics(accumulator: QuantilesAccumulator): List<Double> {
    val boundedQuantiles = emptyBoundedQuantiles()
    boundedQuantiles.mergeWith(accumulator.serializedQuantilesSummary.toByteArray())
    return sortedRanks.map { boundedQuantiles.computeResult(it) }
  }

  /**
   * Creates an empty `BoundedQuantiles` builder initialized with the necessary parameters.
   *
   * @return an empty `BoundedQuantiles.Builder` instance.
   */
  private fun emptyBoundedQuantiles() =
    BoundedQuantiles.builder()
      .epsilon(budget.epsilon())
      .delta(budget.delta())
      .noise(noiseFactory(aggregationParams.noiseKind))
      // TODO: reconsider the if clauses, contribution bounds here affect the noise but
      // contribution bounding with them is not performed therefore we should probably just use the
      // bounds as they are no matter what execution mode is. We should make it clear in the
      // documentation.
      .maxPartitionsContributed(
        if (aggregationParams.applyPartitionsContributedBounding(executionMode)) {
          aggregationParams.maxPartitionsContributed!!
        } else {
          1
        }
      )
      .maxContributionsPerPartition(
        if (
          perPartitionContributionBoundingShouldBeApplied(
            executionMode,
            aggregationParams.contributionBoundingLevel,
          )
        ) {
          aggregationParams.maxContributionsPerPartition!!
        } else {
          Int.MAX_VALUE
        }
      )
      // Min and max values aren't changed if there is no contribution bounding because the extreme
      // values aren't supported by the DP library.
      .lower(featureMinValue)
      .upper(featureMaxValue)
      .build()
}

/**
 * A [Combiner] for the [MetricType.VARIANCE].
 *
 * It returns a noisy variance of input items. It can also return count, sum, and mean if requested
 * by the user.
 *
 * @property aggregationParams parameters controlling the aggregation process (including whether to
 *   return count, sum, and mean).
 * @property countBudget the privacy budget for the count calculation.
 * @property sumBudget the privacy budget for the sum calculation.
 * @property sumSquaresBudget the privacy budget for the sum of squares calculation.
 * @property noiseFactory allows for passing the noise generator as a parameter in order to be able
 *   to mock it in tests.
 */
class VarianceCombiner(
  private val aggregationParams: AggregationParams,
  private val countBudget: AllocatedBudget,
  private val sumBudget: AllocatedBudget,
  private val sumSquaresBudget: AllocatedBudget,
  private val noiseFactory: (NoiseKind) -> Noise,
  private val executionMode: ExecutionMode,
  private val featureSpec: ScalarFeatureSpec,
) : Combiner<VarianceAccumulator, VarianceCombinerResult>, Serializable {
  private val featureMinValue =
    checkNotNull(featureSpec.minValue) {
      "minValue should be set when requesting variance metrics for feature ${featureSpec.featureId}"
    }
  private val featureMaxValue =
    checkNotNull(featureSpec.maxValue) {
      "maxValue should be set when requesting variance metrics for feature ${featureSpec.featureId}"
    }
  private val midValue = (featureMinValue + featureMaxValue) / 2
  private val returnCount = aggregationParams.nonFeatureMetrics.any { it.type == COUNT }
  private val returnSum = featureSpec.metrics.any { it.type == SUM }
  private val returnMean = featureSpec.metrics.any { it.type == MEAN }

  override val requiresPerPartitionBoundedInput = true

  /**
   * **Important Note:** the [contributions] passed to this function must all be contributions of a
   * particular privacy id into a particular partition **sub-sampled** to
   * **aggregationParams.maxContributionsPerPartition**.
   *
   * @param contributions sub-sampled privacy id contributions.
   * @return a new `VarianceAccumulator` containing normalized sum, sum of squares, and count of the
   *   values.
   */
  override fun createAccumulator(contributions: PrivacyIdContributions): VarianceAccumulator =
    // All input values are normalized to be their difference from the middle of the
    // input range. That allows us to calculate the sum of all input values with
    // half the sensitivity it would otherwise take for better accuracy (as compared
    // to doing noisy sum / noisy count).
    varianceAccumulator {
      val featureContributions =
        contributions.featuresList.find { it.featureId == featureSpec.featureId }
      val values = featureContributions?.singleValueContributionsList ?: emptyList()
      featureId = featureSpec.featureId
      val coercedValues =
        values.map {
          it.coerceInIfContributionBoundingEnabled(
            featureMinValue,
            featureMaxValue,
            aggregationParams,
            executionMode,
          ) - midValue
        }
      count = coercedValues.size.toLong()
      normalizedSum = coercedValues.sum()
      normalizedSumSquares = coercedValues.map { it * it }.sum()
    }

  /**
   * Merges two `VarianceAccumulator` instances by summing their counts, normalized sums, and
   * normalized sums of squares.
   *
   * @param accumulator1 the first accumulator to merge.
   * @param accumulator2 the second accumulator to merge.
   * @return a new `VarianceAccumulator` with the combined values.
   */
  override fun mergeAccumulators(
    accumulator1: VarianceAccumulator,
    accumulator2: VarianceAccumulator,
  ): VarianceAccumulator = varianceAccumulator {
    featureId = accumulator1.featureId
    count = accumulator1.count + accumulator2.count
    normalizedSum = accumulator1.normalizedSum + accumulator2.normalizedSum
    normalizedSumSquares = accumulator1.normalizedSumSquares + accumulator2.normalizedSumSquares
  }

  /**
   * Computes the DP variance, and optionally DP sum, count, and mean, from the given
   * `VarianceAccumulator`.
   *
   * @param accumulator the accumulator containing the aggregated data.
   * @return a [VarianceCombinerResult] containing the computed DP variance, and potentially DP sum,
   *   count, and mean.
   */
  override fun computeMetrics(accumulator: VarianceAccumulator): VarianceCombinerResult {
    val dpCount = getNoisedCount(accumulator.count, aggregationParams, countBudget, noiseFactory)
    val dpNormalizedSum =
      getNoisedNormalizedSum(
        accumulator.normalizedSum,
        featureMaxValue,
        midValue,
        aggregationParams,
        sumBudget,
        noiseFactory,
      )
    val dpNormalizedSumSquares =
      getNoisedNormalizedSumOfSquares(
        accumulator.normalizedSumSquares,
        featureMaxValue,
        midValue,
        aggregationParams,
        sumSquaresBudget,
        noiseFactory,
      )
    val dpNormalizedMean = dpNormalizedSum / dpCount
    val dpVariance = dpNormalizedSumSquares / dpCount - dpNormalizedMean * dpNormalizedMean
    val outCount = if (returnCount) dpCount else null
    // Mean uses post processing from COUNT and SUM operations, so it consumes no budget.
    val dpMean = dpNormalizedMean + midValue
    val outSum = if (returnSum) dpMean * dpCount else null
    val outMean = if (returnMean) dpMean else null
    return VarianceCombinerResult(dpVariance, outCount, outSum, outMean)
  }
}

/**
 * Represents the result of the [VarianceCombiner].
 *
 * @property variance the differentially private variance.
 * @property count the differentially private count of values (if requested).
 * @property sum the differentially private sum of values (if requested).
 * @property mean the differentially private mean of values (if requested).
 */
@Immutable
data class VarianceCombinerResult(
  val variance: Double,
  val count: Double?,
  val sum: Double?,
  val mean: Double?,
) : Serializable

/**
 * An assembly of [Combiner]s corresponding to the metrics being computed. Returns a combination of
 * results computed by all the provided combiners.
 *
 * It is caller's responsibility to ensure that no redundant combiners are passed. For example, if
 * MeanCombiner is passed then it is not necessary to pass [CountCombiner] and [SumCombiner], it
 * will just lead to unnecessary computations and privacy budget consumption.
 *
 * It is prohibited to pass [CompoundCombiner] in the [combiners] constructor argument.
 *
 * @property combiners the collection of [Combiner] instances to be used for aggregating different
 *   metrics.
 */
class CompoundCombiner(val combiners: Iterable<Combiner<*, *>>) :
  Combiner<CompoundAccumulator, DpAggregates> {
  init {
    check(combiners.none { it is CompoundCombiner }) {
      "Compound combiner cannot be passed into other compound combiner. " +
        "Passed combiners: $combiners."
    }
  }

  override val requiresPerPartitionBoundedInput =
    combiners.any { it.requiresPerPartitionBoundedInput }

  /**
   * Creates a [CompoundAccumulator] by invoking the `createAccumulator` method on each underlying
   * combiner.
   *
   * @param contributions the privacy id contributions to initialize the accumulators with.
   * @return a new [CompoundAccumulator] containing accumulators for each underlying combiner.
   */
  override fun createAccumulator(contributions: PrivacyIdContributions) = compoundAccumulator {
    for (combiner in combiners) {
      when (combiner) {
        is PrivacyIdCountCombiner ->
          privacyIdCountAccumulator = combiner.createAccumulator(contributions)
        is ExactPrivacyIdCountCombiner ->
          privacyIdCountAccumulator = combiner.createAccumulator(contributions)
        is PostAggregationPartitionSelectionCombiner ->
          privacyIdCountAccumulator = combiner.createAccumulator(contributions)
        is CountCombiner -> countAccumulator = combiner.createAccumulator(contributions)
        is SumCombiner -> sumAccumulators += combiner.createAccumulator(contributions)
        is VectorSumCombiner -> vectorSumAccumulators += combiner.createAccumulator(contributions)
        is MeanCombiner -> meanAccumulators += combiner.createAccumulator(contributions)
        is QuantilesCombiner -> quantilesAccumulators += combiner.createAccumulator(contributions)
        is VarianceCombiner -> varianceAccumulators += combiner.createAccumulator(contributions)
        is CompoundCombiner -> throwIfCompoundCombiner()
      }
    }
  }

  /**
   * Merges two [CompoundAccumulator] instances by merging the corresponding accumulators for each
   * underlying combiner.
   *
   * @param accumulator1 the first accumulator to merge.
   * @param accumulator2 the second accumulator to merge.
   * @return a new [CompoundAccumulator] with the merged results for each metric.
   */
  override fun mergeAccumulators(
    accumulator1: CompoundAccumulator,
    accumulator2: CompoundAccumulator,
  ) = compoundAccumulator {
    for (combiner in combiners) {
      when (combiner) {
        is PrivacyIdCountCombiner ->
          privacyIdCountAccumulator =
            combiner.mergeAccumulators(
              accumulator1.privacyIdCountAccumulator,
              accumulator2.privacyIdCountAccumulator,
            )
        is ExactPrivacyIdCountCombiner ->
          privacyIdCountAccumulator =
            combiner.mergeAccumulators(
              accumulator1.privacyIdCountAccumulator,
              accumulator2.privacyIdCountAccumulator,
            )
        is PostAggregationPartitionSelectionCombiner ->
          privacyIdCountAccumulator =
            combiner.mergeAccumulators(
              accumulator1.privacyIdCountAccumulator,
              accumulator2.privacyIdCountAccumulator,
            )
        is CountCombiner ->
          countAccumulator =
            combiner.mergeAccumulators(accumulator1.countAccumulator, accumulator2.countAccumulator)
        is SumCombiner -> {}
        is VectorSumCombiner -> {}
        is MeanCombiner -> {}
        is VarianceCombiner -> {}
        is QuantilesCombiner -> {}
        is CompoundCombiner -> throwIfCompoundCombiner()
      }
    }
    combiners.filterIsInstance<SumCombiner>().firstOrNull()?.let { combiner ->
      sumAccumulators +=
        (accumulator1.sumAccumulatorsList + accumulator2.sumAccumulatorsList)
          .groupBy { it.featureId }
          .values
          .map { it.reduce { acc1, acc2 -> combiner.mergeAccumulators(acc1, acc2) } }
    }
    combiners.filterIsInstance<VectorSumCombiner>().firstOrNull()?.let { combiner ->
      vectorSumAccumulators +=
        (accumulator1.vectorSumAccumulatorsList + accumulator2.vectorSumAccumulatorsList)
          .groupBy { it.featureId }
          .values
          .map { it.reduce { acc1, acc2 -> combiner.mergeAccumulators(acc1, acc2) } }
    }
    combiners.filterIsInstance<MeanCombiner>().firstOrNull()?.let { combiner ->
      meanAccumulators +=
        (accumulator1.meanAccumulatorsList + accumulator2.meanAccumulatorsList)
          .groupBy { it.featureId }
          .values
          .map { it.reduce { acc1, acc2 -> combiner.mergeAccumulators(acc1, acc2) } }
    }
    combiners.filterIsInstance<VarianceCombiner>().firstOrNull()?.let { combiner ->
      varianceAccumulators +=
        (accumulator1.varianceAccumulatorsList + accumulator2.varianceAccumulatorsList)
          .groupBy { it.featureId }
          .values
          .map { it.reduce { acc1, acc2 -> combiner.mergeAccumulators(acc1, acc2) } }
    }
    combiners.filterIsInstance<QuantilesCombiner>().firstOrNull()?.let { combiner ->
      quantilesAccumulators +=
        (accumulator1.quantilesAccumulatorsList + accumulator2.quantilesAccumulatorsList)
          .groupBy { it.featureId }
          .values
          .map { it.reduce { acc1, acc2 -> combiner.mergeAccumulators(acc1, acc2) } }
    }
  }

  /**
   * Computes the DP aggregates by invoking the `computeMetrics` method on each underlying combiner.
   *
   * @param accumulator the [CompoundAccumulator] containing the aggregated data.
   * @return a [DpAggregates] object containing the computed results for all metrics.
   */
  override fun computeMetrics(accumulator: CompoundAccumulator): DpAggregates {
    var resCount: Double? = null
    var resPidCount: Double? = null
    val featureMetrics = mutableMapOf<String, PerFeature>()

    fun getFeature(fid: String) = featureMetrics.getOrPut(fid) { perFeature { featureId = fid } }

    for (combiner in combiners) {
      when (combiner) {
        is PrivacyIdCountCombiner ->
          resPidCount = combiner.computeMetrics(accumulator.privacyIdCountAccumulator)
        is ExactPrivacyIdCountCombiner -> {} // no anonymized output
        is PostAggregationPartitionSelectionCombiner -> {
          combiner.computeMetrics(accumulator.privacyIdCountAccumulator)?.let { resPidCount = it }
        }
        is CountCombiner -> resCount = combiner.computeMetrics(accumulator.countAccumulator)
        is SumCombiner ->
          for (acc in accumulator.sumAccumulatorsList) {
            featureMetrics[acc.featureId] =
              getFeature(acc.featureId).copy { sum = combiner.computeMetrics(acc) }
          }
        is VectorSumCombiner ->
          for (acc in accumulator.vectorSumAccumulatorsList) {
            featureMetrics[acc.featureId] =
              getFeature(acc.featureId).copy { vectorSum += combiner.computeMetrics(acc) }
          }
        is MeanCombiner ->
          for (acc in accumulator.meanAccumulatorsList) {
            val meanResult = combiner.computeMetrics(acc)
            featureMetrics[acc.featureId] =
              getFeature(acc.featureId).copy {
                mean = meanResult.mean
                meanResult.sum?.let { sum = it }
              }
            meanResult.count?.let { resCount = it }
          }
        is QuantilesCombiner ->
          for (acc in accumulator.quantilesAccumulatorsList) {
            featureMetrics[acc.featureId] =
              getFeature(acc.featureId).copy { quantiles += combiner.computeMetrics(acc) }
          }
        is VarianceCombiner ->
          for (acc in accumulator.varianceAccumulatorsList) {
            val varianceResult = combiner.computeMetrics(acc)
            featureMetrics[acc.featureId] =
              getFeature(acc.featureId).copy {
                variance = varianceResult.variance
                varianceResult.sum?.let { sum = it }
                varianceResult.mean?.let { mean = it }
              }
            varianceResult.count?.let { resCount = it }
          }
        is CompoundCombiner -> throwIfCompoundCombiner()
      }
    }
    return dpAggregates {
      resCount?.let { this.count = it }
      resPidCount?.let { this.privacyIdCount = it }
      perFeature += featureMetrics.values
    }
  }

  /**
   * Returns whether it contains a combiner for post aggregation partition selection.
   *
   * @return True if it has post aggregation partition selection combiner.
   */
  fun hasPostAggregationCombiner() =
    combiners.any { combiner -> combiner is PostAggregationPartitionSelectionCombiner }

  companion object {
    private fun throwIfCompoundCombiner() {
      throw IllegalStateException("Should not be reached, verified in init section.")
    }
  }
}

/**
 * Clamp value to the range [minimumValue, maximumValue] if per partition contribution bounding is
 * required.
 */
private fun <T : Comparable<T>> T.coerceInIfContributionBoundingEnabled(
  minimumValue: T,
  maximumValue: T,
  params: AggregationParams,
  executionMode: ExecutionMode,
): T {
  // Per-pertition bounding implies clamping.
  return if (params.applyPerPartitionBounding(executionMode)) {
    coerceIn(minimumValue, maximumValue)
  } else {
    this
  }
}

private fun getNoisedCount(
  count: Long,
  aggregationParams: AggregationParams,
  countBudget: AllocatedBudget,
  noiseFactory: (NoiseKind) -> Noise,
): Double {
  val noise = noiseFactory(aggregationParams.noiseKind)

  return noise.addNoise(
    count.toDouble(),
    aggregationParams.maxPartitionsContributed!!,
    aggregationParams.maxContributionsPerPartition!!.toDouble(),
    countBudget.epsilon(),
    countBudget.delta(),
  )
}

private fun getNoisedNormalizedSum(
  normalizedSum: Double,
  featureMaxValue: Double,
  midValue: Double,
  aggregationParams: AggregationParams,
  sumBudget: AllocatedBudget,
  noiseFactory: (NoiseKind) -> Noise,
): Double {
  val noise = noiseFactory(aggregationParams.noiseKind)
  // All values were normalized to the symmetric range [minValue-midValue, maxValue-midValue].
  // So the linf sensitivity of 1 record is (maxValue-midValue).
  val lInfSensitivity =
    (featureMaxValue - midValue) * aggregationParams.maxContributionsPerPartition!!
  return noise.addNoise(
    normalizedSum,
    aggregationParams.maxPartitionsContributed!!,
    lInfSensitivity,
    sumBudget.epsilon(),
    sumBudget.delta(),
  )
}

private fun getNoisedNormalizedSumOfSquares(
  normalizedSumOfSquares: Double,
  featureMaxValue: Double,
  midValue: Double,
  aggregationParams: AggregationParams,
  sumOfSquaresBudget: AllocatedBudget,
  noiseFactory: (NoiseKind) -> Noise,
): Double {
  val noise = noiseFactory(aggregationParams.noiseKind)
  // All values were normalized to the symmetric range [minValue-midValue, maxValue-midValue] which
  // were then squared and summed up.
  // So the linf sensitivity of 1 record is (maxValue-midValue)^2 distributed across allowed
  // partition contributions.
  val distance = featureMaxValue - midValue
  val lInfSensitivity = distance * distance * aggregationParams.maxContributionsPerPartition!!
  return noise.addNoise(
    normalizedSumOfSquares,
    aggregationParams.maxPartitionsContributed!!,
    lInfSensitivity,
    sumOfSquaresBudget.epsilon(),
    sumOfSquaresBudget.delta(),
  )
}

private fun PrivacyIdContributions.size(): Int {
  if (featuresList.isEmpty()) {
    return 0
  }
  return featuresList
    .map { feature -> feature.singleValueContributionsCount + feature.multiValueContributionsCount }
    .maxOrNull() ?: 0
}

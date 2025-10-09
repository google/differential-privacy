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

import com.google.common.collect.ImmutableList
import com.google.errorprone.annotations.Immutable
import com.google.privacy.differentialprivacy.pipelinedp4j.core.ContributionBoundingLevel.DATASET_LEVEL
import com.google.privacy.differentialprivacy.pipelinedp4j.core.ContributionBoundingLevel.PARTITION_LEVEL
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.COUNT
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.MEAN
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.PRIVACY_ID_COUNT
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.QUANTILES
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.SUM
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.VARIANCE
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.VECTOR_SUM
import com.google.privacy.differentialprivacy.pipelinedp4j.core.budget.BudgetPerOpSpec
import java.io.Serializable
import kotlin.reflect.KClass

// Constant to limit the number of contributions per privacy unit for avoiding OOM and stucking
// on privacy units with too many contributions. Usually such privacy units are not actual
// privacy units, but rather a set of privacy units, e.g. all users w/o privacy id.
// Now it is implemented only for DpEngine.SelectPartitions().
// TODO: Implement this for DpEngine.Aggregate().
const val MAX_PROCESSED_CONTRIBUTIONS_PER_PRIVACY_ID: Int = 100_000_000

/** Contains shared parameters for validation. */
sealed interface Params {
  /** The contribution bounding level that determines the kind of bounding. */
  val contributionBoundingLevel: ContributionBoundingLevel
  /**
   * The maximum number of partitions that can be contributed by a privacy unit.
   *
   * Must be set to 1 if contributionBoundingLevel is PARTITION_LEVEL to disable cross-partition
   * bounding.
   */
  val maxPartitionsContributed: Int?
  /**
   * The pre-threshold to use for partition selection.
   *
   * Pre-threshold is the minimum number of unique contributors (privacy units) a partition must
   * have. Partitions with fewer contributors will be dropped. If set to 1, no pre-thresholding is
   * applied.
   */
  val preThreshold: Int
}

/**
 * Whether per-partition contribution bounding should be applied with respect to execution mode.
 *
 * It can be used outside of this class to determine whether per-partition contribution bounding
 * should be applied with respect to [contributionBoundingLevel] and [executionMode].
 *
 * This property should not be used for validation of [Params] because it accounts for the execution
 * mode and validation should always be performed for the production mode only.
 * [contributionBoundingLevel] represents the contribution bounding level that is used in production
 * therefore only its values should be used for validation.
 */
fun Params.applyPerPartitionBounding(executionMode: ExecutionMode): Boolean =
  perPartitionContributionBoundingShouldBeApplied(executionMode, contributionBoundingLevel)

/**
 * Whether cross-partition contribution bounding should be applied with respect to execution mode.
 *
 * It can be used outside of this class to determine whether cross-partition contribution bounding
 * should be applied with respect to [executionMode] and [contributionBoundingLevel].
 *
 * This property should not be used for validation of [Params] because it accounts for the execution
 * mode and validation should always be performed for the production mode only.
 * [contributionBoundingLevel] represents the contribution bounding level that is used in production
 * therefore only its values should be used for validation.
 */
fun Params.applyPartitionsContributedBounding(executionMode: ExecutionMode): Boolean =
  partitionsContributedBoundingShouldBeApplied(executionMode, contributionBoundingLevel)

/**
 * Determines whether per-partition contribution bounding should be applied given [executionMode]
 * and [contributionBoundingLevel].
 */
fun perPartitionContributionBoundingShouldBeApplied(
  executionMode: ExecutionMode,
  contributionBoundingLevel: ContributionBoundingLevel,
): Boolean =
  executionMode.appliesContributionBounding &&
    contributionBoundingLevel.withContributionsPerPartitionBounding

/**
 * Determines whether cross-partition contribution bounding should be applied given [executionMode]
 * and [contributionBoundingLevel].
 */
private fun partitionsContributedBoundingShouldBeApplied(
  executionMode: ExecutionMode,
  contributionBoundingLevel: ContributionBoundingLevel,
): Boolean =
  executionMode.appliesContributionBounding &&
    contributionBoundingLevel.withPartitionsContributedBounding

/**
 * The parameters of the metrics being anonymized: the metric types, the contribution bounds, etc.
 * This data-class contains a "bag" of all possible parameters that can be used for any combination
 * of the metrics being computed.
 */
@Immutable
data class AggregationParams(
  /** The metrics being anonymized. */
  val nonFeatureMetrics: ImmutableList<MetricDefinition>,
  val features: ImmutableList<FeatureSpec> = ImmutableList.of(),
  val noiseKind: NoiseKind,
  /**
   * The maximum number of partitions that can be contributed by a privacy unit. Used by all
   * metrics. Note this is mutually exclusive with maxContributions.
   */
  override val maxPartitionsContributed: Int? = null,
  /**
   * The maximum number of times a privacy unit can contribute to a partition. Used for COUNT, MEAN
   * and QUANTILES. Note this is mutually exclusive with maxContributions.
   */
  val maxContributionsPerPartition: Int? = null,
  /**
   * The maximum number of times a privacy unit can contribute to a dataset. Used by all metrics.
   * Note this is mutually exclusive with maxContributionsPerPartition.
   */
  val maxContributions: Int? = null,
  /**
   * The amount of budget used for partition selection.
   *
   * If [BudgetPerOpSpec] is null, [RelativeBudgetPerOpSpec] with weight = 1 is used, i.e. the
   * budget is split evenly among all DP operations (metrics and partition selection).
   */
  val partitionSelectionBudget: BudgetPerOpSpec? = null,
  /** The pre-threshold to use for partition selection. */
  override val preThreshold: Int = 1,
  /**
   * The contribution bounding level that determines the kind of contribution bounding in
   * aggregations and partition selection if partitions are private.
   */
  override val contributionBoundingLevel: ContributionBoundingLevel = DATASET_LEVEL,
  /**
   * The balance of partitions.
   *
   * Optional parameter that influences only public partitions processing and will be used as a hint
   * for the execution to make it more optimized.
   *
   * Processing unbalanced partitions might lead to not enough paralellisation and long processing
   * time. In case if it happens for public partitions processing, set to [UNBALANCED], and as a
   * result special processing for better paralellisation will be performed. See [PartitionsBalance]
   * for definition of balanced/unbalanced partitions.
   */
  val partitionsBalance: PartitionsBalance = PartitionsBalance.UNKNOWN,
) : Params, Serializable

/**
 * Validates [AggregationParams].
 *
 * @param usePublicPartitions indicates whether [DpEngine.aggregate()] was called with public
 *   partitions.
 * @param hasValueExtractor indicates whether [DpEngine.aggregate()] was called with a DataExtractor
 *   which contains a value extractor.
 */
fun validateAggregationParams(
  params: AggregationParams,
  usePublicPartitions: Boolean,
  hasValueExtractor: Boolean,
) {
  // Validate params shared between AggregationParams and SelectPartitionsParams.
  validateBaseParams(params)

  // Contribution bounding level and maxPartitionsContributed are in sync.
  if (params.contributionBoundingLevel.withPartitionsContributedBounding) {
    require(params.maxPartitionsContributed != null || params.maxContributions != null) {
      "maxPartitionsContributed or maxContributions must be set because specified ${params.contributionBoundingLevel} contribution bounding level requires cross partition bounding."
    }
  }

  // Metrics & features validation.
  require(params.nonFeatureMetrics.isNotEmpty() || params.features.isNotEmpty()) {
    "At least one of nonFeatureMetrics or features must be specified."
  }
  require(
    params.nonFeatureMetrics.all {
      it.type == MetricType.COUNT || it.type == MetricType.PRIVACY_ID_COUNT
    }
  ) {
    "Only COUNT and PRIVACY_ID_COUNT are allowed in AggregationParams.nonFeatureMetrics. Other metrics should be provided via AggregationParams.features."
  }
  val featureMetrics = params.features.flatMap { it.metrics }
  require(
    featureMetrics.none { it.type == MetricType.COUNT || it.type == MetricType.PRIVACY_ID_COUNT }
  ) {
    "COUNT and PRIVACY_ID_COUNT are not allowed in features. They should be provided via AggregationParams.nonFeatureMetrics."
  }

  require(
    params.nonFeatureMetrics.map { it.type }.distinct().size == params.nonFeatureMetrics.size
  ) {
    "nonFeatureMetrics must not contain duplicate metric types. Provided ${params.nonFeatureMetrics.map { it.type }}."
  }
  for (feature in params.features) {
    require(feature.metrics.map { it.type }.distinct().size == feature.metrics.size) {
      "feature ${feature.featureId} must not contain duplicate metric types. Provided ${feature.metrics.map { it.type }}"
    }
  }
  require(params.features.map { it.featureId }.distinct().size == params.features.size) {
    "featureId must be unique. Provided ${params.features.map { it.featureId }}"
  }

  // Max contributions per partition.
  require(isGreaterThanZeroIfSet(params.maxContributionsPerPartition)) {
    "maxContributionsPerPartition must be positive. Provided value: " +
      "${params.maxContributionsPerPartition}."
  }

  // Max contributions.
  require(isGreaterThanZeroIfSet(params.maxContributions)) {
    "maxContributions must be positive. Provided value: " + "${params.maxContributions}."
  }
  // Mutually exclusive partition bounds
  require(params.maxContributions == null || params.maxPartitionsContributed == null) {
    "maxContributions and maxPartitionsContributed are mutually exclusive. " +
      "Provided values: maxContributions=${params.maxContributions}, " +
      "maxPartitionsContributed=${params.maxPartitionsContributed}."
  }
  require(params.maxContributions == null || params.maxContributionsPerPartition == null) {
    "maxContributions and maxContributionsPerPartition are mutually exclusive. " +
      "Provided values: maxContributions=${params.maxContributions}, " +
      "maxContributionsPerPartition=${params.maxContributionsPerPartition}."
  }

  // Required parameters per each metric.
  if (params.contributionBoundingLevel.withContributionsPerPartitionBounding) {
    val perPartitionBoundsSet = params.maxContributionsPerPartition != null
    val crossPartitionBoundsSet = params.maxContributions != null
    val totalValueBoundsSet =
      params.features.any {
        (it is ScalarFeatureSpec && it.minTotalValue != null && it.maxTotalValue != null) ||
          it is VectorFeatureSpec
      }
    require(perPartitionBoundsSet || crossPartitionBoundsSet || totalValueBoundsSet) {
      "maxContributionsPerPartition or maxContributions or (minTotalValue, maxTotalValue) or vectorMaxTotalNorm must be set because specified ${params.contributionBoundingLevel} contribution bounding level requires per partition bounding"
    }
  }

  if (metricIsRequested(COUNT::class, params.nonFeatureMetrics)) {
    require(params.maxContributionsPerPartition != null || params.maxContributions != null) {
      "maxContributionsPerPartition or maxContributions must be set for COUNT metric."
    }
  }

  for (feature in params.features) {
    when (feature) {
      is ScalarFeatureSpec -> validateScalarFeature(params, feature)
      is VectorFeatureSpec -> validateVectorFeature(params, feature)
    }
  }

  // Partition selection
  if (usePublicPartitions) {
    require(params.partitionSelectionBudget == null) {
      "partitionSelectionBudget can not be set for public partitions."
    }
  }

  // ValueExtractor: only COUNT and PRIVACY_ID_COUNT can be computed w/o a value extractor.
  if (!hasValueExtractor) {
    require(featureMetrics.isEmpty()) {
      "Metrics ${featureMetrics.map { it.type }} require a value extractor."
    }
  }
}

private fun validateScalarFeature(params: AggregationParams, feature: ScalarFeatureSpec) {
  // Min/Max bounds
  require(sameNullability(feature.minValue, feature.maxValue)) {
    "minValue and maxValue must be simultaneously equal or not equal to null. Provided values: " +
      "minValue=${feature.minValue}, maxValue=${feature.maxValue}."
  }
  var areMinMaxValuesSet = false
  if (feature.minValue != null && feature.maxValue != null) {
    areMinMaxValuesSet = true
    require(feature.minValue < feature.maxValue) {
      "minValue must be less than maxValue. Provided values: " +
        "minValue=${feature.minValue}, maxValue=${feature.maxValue}."
    }
  }
  require(sameNullability(feature.minTotalValue, feature.maxTotalValue)) {
    "minTotalValue and maxTotalValue must be simultaneously equal or not equal to null. " +
      "Provided values: minTotalValue=${feature.minTotalValue}, " +
      "maxTotalValue=${feature.maxTotalValue}."
  }
  var areMinMaxTotalValuesSet = false
  if (feature.minTotalValue != null && feature.maxTotalValue != null) {
    areMinMaxTotalValuesSet = true
    require(feature.minTotalValue <= feature.maxTotalValue) {
      "minTotalValue must be less or equal to maxTotalValue. Provided values: " +
        "minTotalValue=${feature.minTotalValue}, maxTotalValue=${feature.maxTotalValue}."
    }
  }

  // When MEAN and SUM are set together, then contribution bounding with (minValue, maxValue)
  // is used. SUM and VARIANCE should not be set together.
  if (
    metricIsRequested(SUM::class, feature.metrics) &&
      !metricIsRequested(MEAN::class, feature.metrics) &&
      !metricIsRequested(VARIANCE::class, feature.metrics)
  ) {
    require(areMinMaxTotalValuesSet) {
      "(minTotalValue, maxTotalValue) must be set for SUM metrics."
    }
  }

  if (metricIsRequested(MEAN::class, feature.metrics)) {
    require(params.maxContributionsPerPartition != null || params.maxContributions != null) {
      "maxContributionsPerPartition or maxContributions must be set for MEAN metric."
    }
    require(areMinMaxValuesSet) { "(minValue, maxValue) must be set for MEAN metric." }
    require(!areMinMaxTotalValuesSet) {
      "(minTotalValue, maxTotalValue) should not be set if MEAN metric is requested."
    }
  }
  require(
    params.nonFeatureMetrics.find { it.type == COUNT }?.budgetSpec == null ||
      feature.metrics.find { it.type == MEAN }?.budgetSpec == null
  ) {
    "BudgetPerOpSpec can not be set for both COUNT and MEAN metrics."
  }
  require(
    feature.metrics.find { it.type == SUM }?.budgetSpec == null ||
      feature.metrics.find { it.type == MEAN }?.budgetSpec === null
  ) {
    "BudgetPerOpSpec can not be set for both SUM and MEAN metrics."
  }
  require(
    feature.metrics.find { it.type == MEAN }?.budgetSpec == null ||
      feature.metrics.find { it.type == VARIANCE }?.budgetSpec == null
  ) {
    "BudgetPerOpSpec can not be set for both MEAN and VARIANCE metrics."
  }
  // Validation for VARIANCE metric.
  if (metricIsRequested(VARIANCE::class, feature.metrics)) {
    require(params.maxContributionsPerPartition != null || params.maxContributions != null) {
      "maxContributionsPerPartition or maxContributions must be set for VARIANCE metric."
    }
    require(areMinMaxValuesSet) { "(minValue, maxValue) must be set for VARIANCE metric." }
    require(!areMinMaxTotalValuesSet) {
      "(minTotalValue, maxTotalValue) should not be set if VARIANCE metric is requested."
    }
  }
  require(
    feature.metrics.find { it.type == SUM }?.budgetSpec == null ||
      feature.metrics.find { it.type == VARIANCE }?.budgetSpec == null
  ) {
    "BudgetPerOpSpec can not be set for both SUM and VARIANCE metrics."
  }

  require(
    params.nonFeatureMetrics.find { it.type == COUNT }?.budgetSpec == null ||
      feature.metrics.find { it.type == VARIANCE }?.budgetSpec == null
  ) {
    "BudgetPerOpSpec can not be set for both COUNT and VARIANCE metrics."
  }
  // Validation for QUANTILES metric.
  if (metricIsRequested(QUANTILES::class, feature.metrics)) {
    require(params.maxContributionsPerPartition != null) {
      "maxContributionsPerPartition must be set for QUANTILES metric."
    }
    require(areMinMaxValuesSet) { "(minValue, maxValue) must be set for QUANTILES metric." }
  }
}

private fun validateVectorFeature(params: AggregationParams, feature: VectorFeatureSpec) {
  // Validation for VECTOR_SUM metric.
  if (metricIsRequested(VECTOR_SUM::class, feature.metrics)) {
    when (params.noiseKind) {
      NoiseKind.LAPLACE ->
        require(feature.normKind in listOf(NormKind.L_INF, NormKind.L1)) {
          "vectorNormKind must be L_INF or L1 for LAPLACE noise. Provided value: ${feature.normKind}."
        }
      NoiseKind.GAUSSIAN ->
        require(feature.normKind in listOf(NormKind.L_INF, NormKind.L2)) {
          "vectorNormKind must be L_INF or L2 for GAUSSIAN noise. Provided value: ${feature.normKind}."
        }
    }
  }
}

/** The parameters of [DPEngine.selectPartitions()]. */
@Immutable
data class SelectPartitionsParams(
  /** The maximum number of partitions that can be contributed by a privacy unit. */
  override val maxPartitionsContributed: Int,
  /**
   * The amount of budget that should be used for partition selection.
   *
   * If [BudgetPerOpSpec] is null, [RelativeBudgetPerOpSpec] with weight = 1 is used, i.e. the
   * budget is split evenly among all DP operations (metrics and partition selection).
   */
  val budget: BudgetPerOpSpec? = null,
  /** The pre-threshold to use for partition selection. */
  override val preThreshold: Int = 1,
  /**
   * The contribution bounding level that determines the kind of contribution bounding in partition
   * selection.
   */
  override val contributionBoundingLevel: ContributionBoundingLevel = DATASET_LEVEL,
) : Params, Serializable

/** Validates [SelectPartitionsParams]. */
fun validateSelectPartitionsParams(params: SelectPartitionsParams) {
  // Validate params shared between AggregationParams and SelectPartitionsParams.
  validateBaseParams(params)
}

/**
 * The balance of the partitions in the input dataset.
 *
 * Partitions are balanced if there is no partition which contribute > 1% of data. Otherwise, the
 * partitions are unbalanced.
 */
enum class PartitionsBalance {
  /** Use if you don't know the answer. */
  UNKNOWN,
  /** Use if you know that the partitions are balanced. */
  BALANCED,
  /** Use if you know that the partitions are unbalanced. */
  UNBALANCED,
}

// TODO: Move maxPartitionsContributed and maxContributionsPerPartition to
// ContributionBoundingLevel.
/** The type of contribution bounding to be applied. */
enum class ContributionBoundingLevel(
  internal val withPartitionsContributedBounding: Boolean,
  internal val withContributionsPerPartitionBounding: Boolean,
) {
  /** Enables contribution bounding across the whole dataset. */
  DATASET_LEVEL(
    withPartitionsContributedBounding = true,
    withContributionsPerPartitionBounding = true,
  ),
  /** Enables contribution bounding only within partitions. */
  PARTITION_LEVEL(
    withPartitionsContributedBounding = false,
    withContributionsPerPartitionBounding = true,
  ),
}

/**
 * The execution mode of the DP engine.
 *
 * @property appliesContributionBounding whether contribution bounding should be applied in this
 *   mode. If [appliesContributionBounding] is false, then [ContributionBoundingLevel] will be
 *   ignored. If true then contribution bounding will be applied normally according to
 *   [ContributionBoundingLevel].
 * @property appliesNoise whether noise should be applied in this mode. If [appliesNoise] is false,
 *   then at the time of noise addition a zero noise will be used, however the provided [NoiseKind]
 *   will remain the same and still be used. If true, then noise will be added normally according to
 *   [NoiseKind].
 * @property partitionSelectionIsNonDeterministic whether partition selection should be
 *   non-deterministic in this mode. If [partitionSelectionIsNonDeterministic] is false, then
 *   partition selection will be deterministic and output all partitions that are present in the
 *   data. If true, then partition selection will be non-deterministic and differentially private.
 */
enum class ExecutionMode(
  val appliesContributionBounding: Boolean,
  val appliesNoise: Boolean,
  val partitionSelectionIsNonDeterministic: Boolean,
) {
  /**
   * Production execution mode.
   *
   * Only this mode should be used in production code. It does not affect the execution in any form
   * and fully respects the other executon params (i.e. it does not affect contribution bounding,
   * noise, parititon selection, etc.).
   */
  PRODUCTION(
    appliesContributionBounding = true,
    appliesNoise = true,
    partitionSelectionIsNonDeterministic = true,
  ),
  /**
   * Test mode when contribution bounding is applied, but no noise is added and partition selection
   * is deterministic.
   */
  TEST_MODE_WITH_CONTRIBUTION_BOUNDING(
    appliesContributionBounding = true,
    appliesNoise = false,
    partitionSelectionIsNonDeterministic = false,
  ),
  /**
   * Test mode when no contribution bounding and no noise are applied and partition selection is
   * deterministic.
   */
  FULL_TEST_MODE(
    appliesContributionBounding = false,
    appliesNoise = false,
    partitionSelectionIsNonDeterministic = false,
  ),
}

/**
 * Represents a feature for which DP metrics are calculated.
 *
 * A feature is a characteristic of the input data. For example, in a dataset of user activities, a
 * feature could be "time spent on page" or "user embedding". This interface and its implementations
 * are used to specify parameters for metrics calculated on these features.
 */
@Immutable
sealed interface FeatureSpec : Serializable {
  /** A unique identifier for the feature. */
  val featureId: String
  /** The list of DP metrics to be computed for this feature. */
  val metrics: ImmutableList<MetricDefinition>
}

/**
 * A [FeatureSpec] for scalar-valued features.
 *
 * This is used for features where each data point is a single numerical value (e.g., a Double). It
 * is suitable for metrics like [MetricType.SUM], [MetricType.MEAN], [MetricType.VARIANCE], and
 * [MetricType.QUANTILES].
 *
 * @property minValue The minimum value that a single contribution can take.
 * @property maxValue The maximum value that a single contribution can take.
 * @property minTotalValue The minimum total value that contributions from a single privacy unit can
 *   sum up to per partition. Must be set if [MetricType.SUM] is requested and neither
 *   [MetricType.MEAN] nor [MetricType.VARIANCE] is requested; otherwise, [minValue] and [maxValue]
 *   must be set.
 * @property maxTotalValue The maximum total value that contributions from a single privacy unit can
 *   sum up to per partition. Must be set if [MetricType.SUM] is requested and neither
 *   [MetricType.MEAN] nor [MetricType.VARIANCE] is requested; otherwise, [minValue] and [maxValue]
 *   must be set.
 */
@Immutable
data class ScalarFeatureSpec(
  override val featureId: String,
  override val metrics: ImmutableList<MetricDefinition>,
  val minValue: Double? = null,
  val maxValue: Double? = null,
  val minTotalValue: Double? = null,
  val maxTotalValue: Double? = null,
) : FeatureSpec, Serializable

/**
 * A [FeatureSpec] for vector-valued features.
 *
 * This is used for features where each data point is a vector of numerical values (e.g., an
 * embedding). It is suitable for metrics like [MetricType.VECTOR_SUM].
 *
 * @property vectorSize The size of the vector.
 * @property normKind The type of norm to use for contribution bounding.
 * @property vectorMaxTotalNorm The maximum total norm of contributions from a single privacy unit
 *   per partition.
 */
@Immutable
data class VectorFeatureSpec(
  override val featureId: String,
  override val metrics: ImmutableList<MetricDefinition>,
  val vectorSize: Int,
  val normKind: NormKind,
  val vectorMaxTotalNorm: Double,
) : FeatureSpec, Serializable

/** The definition of the DP metric to compute. */
@Immutable
data class MetricDefinition(
  val type: MetricType,
  /**
   * The amount of privacy budget used to anonymize this metric.
   *
   * If [budgetSpec] is null, [RelativeBudgetPerOpSpec] with weight = 1 is used, i.e. the budget is
   * split evenly among all DP calculations (metrics and partition selection).
   */
  val budgetSpec: BudgetPerOpSpec? = null,
) : Serializable

// TODO: have 2 types of MetricType feature and non feature for better code
// readability and remove complicated checks.
/** The types of metrics that can be anonymized. */
@Immutable
sealed class MetricType : Serializable {
  data object PRIVACY_ID_COUNT : MetricType()

  data object COUNT : MetricType()

  data object SUM : MetricType()

  data object VECTOR_SUM : MetricType()

  data object MEAN : MetricType()

  data class QUANTILES(private val ranks: ImmutableList<Double>) : MetricType() {
    val sortedRanks = ImmutableList.copyOf(ranks.sorted())

    init {
      require(sortedRanks.all { it in 0.0..1.0 }) { "Ranks for quantiles must be all in [0, 1]." }
    }
  }

  data object VARIANCE : MetricType()
}

/** The kind of noise that can be applied to the data. */
enum class NoiseKind {
  LAPLACE,
  GAUSSIAN,
}

/**
 * The kind of vector norm.
 *
 * See https://en.wikipedia.org/wiki/Norm_%28mathematics%29#p-norm for definitions.
 */
enum class NormKind {
  L_INF,
  L1,
  L2,
}

private fun sameNullability(a: Double?, b: Double?): Boolean {
  return (a == null) == (b == null)
}

private fun metricIsRequested(
  metricTypeClass: KClass<out MetricType>,
  metrics: Collection<MetricDefinition>,
) = metrics.any { metricTypeClass.isInstance(it.type) }

private fun isGreaterThanZeroIfSet(value: Int?): Boolean = value == null || value > 0

private fun isLessOrEqualToIfSet(value: Int?, upperBound: Int): Boolean =
  value == null || value <= upperBound

private fun validateBaseParams(params: Params) {
  // Cross partition bounds.
  require(isGreaterThanZeroIfSet(params.maxPartitionsContributed)) {
    "maxPartitionsContributed must be positive. Provided value: ${params.maxPartitionsContributed}."
  }

  require(
    isLessOrEqualToIfSet(
      params.maxPartitionsContributed,
      MAX_PROCESSED_CONTRIBUTIONS_PER_PRIVACY_ID,
    )
  ) {
    "maxPartitionsContributed must be less than ${MAX_PROCESSED_CONTRIBUTIONS_PER_PRIVACY_ID} " +
      "Provided values: maxPartitionsContributed=${params.maxPartitionsContributed}."
  }
  // Contribution bounding level.
  require(
    params.contributionBoundingLevel != PARTITION_LEVEL ||
      (params.contributionBoundingLevel == PARTITION_LEVEL && params.maxPartitionsContributed == 1)
  ) {
    "maxPartitionsContributed must be 1 if partition level contribution bounding is set. " +
      "Provided value: ${params.maxPartitionsContributed}."
  }

  // Pre-threshold.
  require(params.preThreshold > 0) {
    "preThreshold must be positive. Provided value: ${params.preThreshold}."
  }
}

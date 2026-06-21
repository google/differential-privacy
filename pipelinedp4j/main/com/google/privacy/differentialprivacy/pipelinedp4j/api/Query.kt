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

package com.google.privacy.differentialprivacy.pipelinedp4j.api

import com.google.common.collect.ImmutableList
import com.google.privacy.differentialprivacy.pipelinedp4j.core.AggregationParams
import com.google.privacy.differentialprivacy.pipelinedp4j.core.DataExtractors
import com.google.privacy.differentialprivacy.pipelinedp4j.core.DpEngine
import com.google.privacy.differentialprivacy.pipelinedp4j.core.DpEngineBudgetSpec
import com.google.privacy.differentialprivacy.pipelinedp4j.core.Encoder
import com.google.privacy.differentialprivacy.pipelinedp4j.core.EncoderFactory
import com.google.privacy.differentialprivacy.pipelinedp4j.core.FeatureValuesExtractor
import com.google.privacy.differentialprivacy.pipelinedp4j.core.FrameworkCollection
import com.google.privacy.differentialprivacy.pipelinedp4j.core.FrameworkTable
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType
import com.google.privacy.differentialprivacy.pipelinedp4j.core.ScalarFeatureSpec
import com.google.privacy.differentialprivacy.pipelinedp4j.core.SelectPartitionsParams
import com.google.privacy.differentialprivacy.pipelinedp4j.core.VectorFeatureSpec
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.DpAggregates

sealed interface Query<ReturnT> {
  /** Executes the query (in production mode). */
  fun run() = run(TestMode.NONE)

  /**
   * Executes the query in test mode.
   *
   * @param testMode the test mode to use for the query. Do not use in production code.
   */
  fun run(testMode: TestMode): ReturnT
}

/**
 * Base class for all queries.
 *
 * It runs the query with the DP engine. The child classes are responsible for converting the DP
 * engine result to the final result type.
 */
abstract class BaseQueryImpl<DataRowT : Any, PrivacyUnitT : Any, GroupKeysT : Any, ReturnT : Any>
protected constructor(
  private val data: FrameworkCollection<DataRowT>,
  private val encoderFactory: EncoderFactory,
  private val privacyUnitExtractor: (DataRowT) -> PrivacyUnitT,
  private val privacyUnitEncoder: Encoder<PrivacyUnitT>,
  private val contributionBoundingLevel: ContributionBoundingLevel,
  private val groupKeyExtractor: (DataRowT) -> GroupKeysT,
  private val groupKeyEncoder: Encoder<GroupKeysT>,
  private val groupsType: GroupsType,
  private val groupByAdditionalParameters: GroupByAdditionalParameters,
  internal val aggregations: List<AggregationSpec>,
  private val totalBudget: TotalBudget,
  private val noiseKind: NoiseKind?,
) : Query<ReturnT> {
  init {
    validate()
  }

  protected fun runWithDpEngine(testMode: TestMode): FrameworkTable<GroupKeysT, DpAggregates> {
    val dpEngine =
      DpEngine.create(
        encoderFactory,
        DpEngineBudgetSpec(totalBudget.toInternalTotalBudget()),
        testMode.toExecutionMode(),
      )

    // 1. If aggregation is empty then we do partition selection.
    if (aggregations.isEmpty()) {
      val extractors = createDataExtractors(aggregations)
      val result = dpEngine.selectPartitions(data, createSelectPartitionsParams(), extractors)
      dpEngine.done()

      return result.mapToTable(
        "Add empty DpAggregates",
        groupKeyEncoder,
        encoderFactory.protos(DpAggregates::class),
        { it to DpAggregates.getDefaultInstance() },
      )
    }

    val partitions: FrameworkCollection<GroupKeysT>? = groupsType.getPublicGroups()
    val result = aggregateWithDpEngine(dpEngine, aggregations, partitions)
    dpEngine.done()
    return result
  }

  private fun validate() {
    validateAggregations()
    validateBudgets()
    validateGroupSelection()
  }

  private fun validateAggregations() {
    if (aggregations.isEmpty()) {
      return
    }

    require(noiseKind != null) { "Noise kind must be specified because aggregations are used." }

    requireNoDuplicateOutputColumnNames()

    require(aggregations.filter { it is PrivacyIdCount }.size <= 1) {
      "There can be at most one aggregation of counting distinct privacy units (i.e. countDistinctPrivacyUnits() should be called at most once) because calculating it more than once doesn't provide any value."
    }
    require(aggregations.filter { it is Count }.size <= 1) {
      "There can be at most one count aggregation (i.e. count() should be called at most once) because calculating it more than once doesn't provide any value."
    }

    // Validate aggregations of specific values.
    val aggregationsPerValues =
      aggregations.filter { it is ValueAggregations<*> }.map { it as ValueAggregations<*> }
    requireDistinctValueColumnNames(aggregationsPerValues)
    requireDistinctValueExtractors(aggregationsPerValues)
    for (aggregationsPerValue in aggregationsPerValues) {
      requireDistinctAggregationsPerValue(aggregationsPerValue)
      val aggregationByTypes: Map<MetricType, ValueAggregationSpec> =
        aggregationsPerValue.valueAggregationSpecs
          .groupBy { it.metricType }
          .mapValues { it.value.first() }

      // Check total value bounds.
      val totalValueBounds = aggregationsPerValue.contributionBounds.totalValueBounds
      if (MetricType.SUM in aggregationByTypes) {
        val aggregationsThatCalculateSum = listOf(MetricType.MEAN, MetricType.VARIANCE)
        if (aggregationsThatCalculateSum.any { it in aggregationByTypes }) {
          require(totalValueBounds == null) {
            "Total value bounds should not be set if SUM is calculated together with MEAN or VARIANCE. Provided total value bounds: ${totalValueBounds}"
          }
        } else {
          // TODO: the requirement is complicated, add link to the documentation in the
          // error message once the documentation is available.
          require(totalValueBounds != null) {
            "Total value bounds should be set if SUM is calculated without calculating MEAN or VARIANCE. This is because to make calculations more precise SUM uses the total value bounds, unlike other aggregations that use the value bounds."
          }
        }
      } else {
        require(totalValueBounds == null) {
          "Total value bounds should not be set because SUM is not calculated. Provided total value bounds: ${totalValueBounds}"
        }
      }
      // Check value bounds.
      val aggregationsThatRequireValueBounds =
        listOf(MetricType.MEAN::class, MetricType.VARIANCE::class, MetricType.QUANTILES::class)
      val valueBounds = aggregationsPerValue.contributionBounds.valueBounds
      if (
        aggregationsThatRequireValueBounds.any { metricTypeClass ->
          aggregationByTypes.keys.any { metricTypeClass.isInstance(it) }
        }
      ) {
        require(valueBounds != null) {
          "Total value bounds should be set if MEAN, VARIANCE or QUANTILES are calculated."
        }
      } else {
        require(valueBounds == null) {
          "Value bounds are not needed if MEAN, VARIANCE or QUANTILES are not calculated. Therefore they should not be set. Provided bounds: ${valueBounds}"
        }
      }
    }

    // Validate aggregations of specific vectors.
    val aggregationsPerVectors =
      aggregations.filter { it is VectorAggregations<*> }.map { it as VectorAggregations<*> }
    requireDistinctVectorColumnNames(aggregationsPerVectors)
    requireDistinctVectorExtractors(aggregationsPerVectors)
    for (aggregationsPerVector in aggregationsPerVectors) {
      requireDistinctAggregationsPerVector(aggregationsPerVector)

      val normKind = aggregationsPerVector.vectorContributionBounds.maxVectorTotalNorm.normKind
      when (noiseKind) {
        NoiseKind.LAPLACE ->
          require(normKind in listOf(NormKind.L_INF, NormKind.L1)) {
            "Norm kind must be L_INF or L1 when Laplace mechanism is used. Provided norm kind: $normKind."
          }
        NoiseKind.GAUSSIAN ->
          require(normKind in listOf(NormKind.L_INF, NormKind.L2)) {
            "Norm kind must be L_INF or L2 when Gaussian mechanism is used. Provided norm kind: $normKind."
          }
      }
    }
  }

  private fun requireNoDuplicateOutputColumnNames() {
    val outputColumnNameCounts = aggregations.outputColumnNames().groupingBy { it }.eachCount()
    val duplicates = outputColumnNameCounts.filter { it.value > 1 }.keys
    require(duplicates.isEmpty()) {
      "There are aggregations with duplicate output column names: ${duplicates}."
    }
  }

  private fun requireDistinctValueExtractors(aggregationsPerValue: List<ValueAggregations<*>>) {
    val valueExtractorCounts = aggregationsPerValue.groupingBy { it.valueExtractor }.eachCount()
    val duplicates = valueExtractorCounts.filter { it.value > 1 }.keys
    val valueAggregationsWithDuplicates =
      aggregationsPerValue.filter { it.valueExtractor in duplicates }
    require(duplicates.isEmpty()) {
      "There are the same (object reference equality) value extractors used in different aggregateValue() calls. Please merge them into one call." +
        "\nValue aggregations with duplicate value extractors:\n${
                        valueAggregationsWithDuplicates.map { "* ${it.valueAggregationSpecs}" }.joinToString("\n")
                    }"
    }
  }

  private fun requireDistinctVectorExtractors(aggregationsPerVector: List<VectorAggregations<*>>) {
    val vectorExtractorCounts = aggregationsPerVector.groupingBy { it.vectorExtractor }.eachCount()
    val duplicates = vectorExtractorCounts.filter { it.value > 1 }.keys
    val vectorAggregationsWithDuplicates =
      aggregationsPerVector.filter { it.vectorExtractor in duplicates }
    require(duplicates.isEmpty()) {
      "There are the same (object reference equality) vector extractors used in different aggregateVector() calls. Please merge them into one call." +
        "\nVector aggregations with duplicate vector extractors:\n${
                        vectorAggregationsWithDuplicates.map { "* ${it.vectorAggregationSpecs}" }.joinToString("\n")
                    }"
    }
  }

  private fun requireDistinctValueColumnNames(aggregationsPerValue: List<ValueAggregations<*>>) {
    val valueColumnNameCounts =
      aggregationsPerValue.groupingBy { it.valueColumnName }.eachCount().filterKeys { it != null }
    val duplicates = valueColumnNameCounts.filter { it.value > 1 }.keys
    require(duplicates.isEmpty()) {
      "The same value column is used in different aggregateValue() calls. Please merge them into one call." +
        "\nDuplicate value columns:\n${
                        duplicates.joinToString("\n") { "* $it" }
                    }"
    }
  }

  private fun requireDistinctVectorColumnNames(aggregationsPerVector: List<VectorAggregations<*>>) {
    val vectorColumnNamesCounts =
      aggregationsPerVector
        .filter { it.vectorColumnNames != null }
        .groupingBy { it.vectorColumnNames!!.names.toSet() }
        .eachCount()
    val duplicates = vectorColumnNamesCounts.filter { it.value > 1 }.keys
    require(duplicates.isEmpty()) {
      "The same vector column names are used in different aggregateVector() calls. Please merge them into one call." +
        "\nDuplicate vector columns:\n${
                        duplicates.joinToString("\n") { "* $it" }
                    }"
    }
  }

  private fun requireDistinctAggregationsPerValue(valueAggregations: ValueAggregations<*>) {
    val metricTypeCounts =
      valueAggregations.valueAggregationSpecs.map { it.metricType }.groupingBy { it }.eachCount()
    val duplicates = metricTypeCounts.filter { it.value > 1 }.keys
    require(duplicates.isEmpty()) {
      "There are duplicate aggregations for the same value: ${duplicates}. Aggregations: ${valueAggregations.valueAggregationSpecs}."
    }
  }

  private fun requireDistinctAggregationsPerVector(vectorAggregations: VectorAggregations<*>) {
    val metricTypeCounts =
      vectorAggregations.vectorAggregationSpecs.map { it.metricType }.groupingBy { it }.eachCount()
    val duplicates = metricTypeCounts.filter { it.value > 1 }.keys
    require(duplicates.isEmpty()) {
      "There are duplicate aggregations for the same vector: ${duplicates}. Aggregations: ${vectorAggregations.vectorAggregationSpecs}."
    }
  }

  private fun validateBudgets() {
    require(totalBudget.epsilon > 0.0) {
      "Epsilon in the total budget must be positive. Provided epsilon: ${totalBudget.epsilon}."
    }
    for ((outputColumnName, budget) in aggregations.budgets()) {
      if (budget is AbsoluteBudgetPerOpSpec) {
        require(budget.epsilon > 0.0) {
          "Epsilon in the aggregation budget must be positive. Aggregation output column name: ${outputColumnName}."
        }
      }
    }
    when (noiseKind) {
      NoiseKind.LAPLACE -> {
        // Delta can be zero or positive. Nothing to check.
      }
      NoiseKind.GAUSSIAN -> {
        // Then deltas must be positive.
        require(totalBudget.delta > 0.0) {
          "Delta in the total budget must be positive when Gaussian noise is used. Provided delta: ${totalBudget.delta}."
        }
        for ((outputColumnName, budget) in aggregations.budgets()) {
          if (budget is AbsoluteBudgetPerOpSpec) {
            require(budget.delta > 0.0) {
              "Delta in the aggregation budget must be positive when Gaussian noise is used. Aggregation output column name: ${outputColumnName}."
            }
          }
        }
      }
      null -> {}
    }
    // Check correctness of the budget for private group selection.
    if (groupsType is GroupsType.PrivateGroups) {
      if (groupsType.budget is AbsoluteBudgetPerOpSpec) {
        require(groupsType.budget.delta > 0.0) {
          "Delta in the budget for private group selection must be positive. Provided delta: ${groupsType.budget.delta}."
        }
        if (aggregations.any { it is PrivacyIdCount }) {
          // TODO: the requirement is complicated, add link to the documentation in the
          // error message once the documentation is available.
          require(groupsType.budget.epsilon == 0.0) {
            "Epsilon in the budget for private group selection must be zero when counting distinct privacy units because epsilon from the budget for counting distinct privacy units will be used. Provided epsilon: ${groupsType.budget.epsilon}."
          }
        } else {
          require(groupsType.budget.epsilon > 0.0) {
            "Epsilon in the budget for private group selection must be positive when not counting distinct privacy units. Provided epsilon: ${groupsType.budget.epsilon}."
          }
        }
      } else {
        require(totalBudget.delta > 0.0) {
          "Delta in the total budget must be positive when private group selection is used. Provided delta: ${totalBudget.delta}."
        }
      }
    }
    // Verification that there is enough total budget is done inside the DpEngine because it is
    // easier and more realiable to do it there.
  }

  private fun validateGroupSelection() {
    if (aggregations.isEmpty()) {
      require(groupsType !is GroupsType.PublicGroups<*>) {
        "There are no aggregations, therefore public groups do not make any sense. You will just get the same groups back."
      }
    }
    // Other validations are done inside the DpEngine or in constructors of repsective data classes.
  }

  private fun aggregateWithDpEngine(
    dpEngine: DpEngine,
    aggregationSpecs: List<AggregationSpec>,
    partitions: FrameworkCollection<GroupKeysT>?,
  ): FrameworkTable<GroupKeysT, DpAggregates> {
    val extractors = createDataExtractors(aggregationSpecs)
    val params = createAggregationParams(aggregationSpecs)
    return dpEngine.aggregate(data, params, extractors, partitions)
  }

  private fun createDataExtractors(
    aggregations: List<AggregationSpec>
  ): DataExtractors<DataRowT, PrivacyUnitT, GroupKeysT> {
    val featureValueExtractors: List<FeatureValuesExtractor<DataRowT>> =
      aggregations.mapNotNull {
        when (it) {
          is ValueAggregations<*> -> {
            @Suppress("UNCHECKED_CAST") val va = it as ValueAggregations<DataRowT>
            val featureId = va.getFeatureId()
            val valueExtractor = va.valueExtractor
            FeatureValuesExtractor<DataRowT>(featureId, { row -> listOf(valueExtractor(row)) })
          }
          is VectorAggregations<*> -> {
            @Suppress("UNCHECKED_CAST") val vea = it as VectorAggregations<DataRowT>
            val featureId = vea.getFeatureId()
            val vectorExtractor = vea.vectorExtractor
            FeatureValuesExtractor<DataRowT>(featureId, { row -> vectorExtractor(row) })
          }
          else -> null
        }
      }
    return if (featureValueExtractors.isEmpty()) {
      DataExtractors.from(
        privacyUnitExtractor,
        privacyUnitEncoder,
        groupKeyExtractor,
        groupKeyEncoder,
      )
    } else {
      DataExtractors.from(
        privacyUnitExtractor,
        privacyUnitEncoder,
        groupKeyExtractor,
        groupKeyEncoder,
        valuesExtractors = featureValueExtractors,
      )
    }
  }

  private fun createSelectPartitionsParams() =
    SelectPartitionsParams(
      maxPartitionsContributed = contributionBoundingLevel.getMaxPartitionsContributed(),
      budget = (groupsType as GroupsType.PrivateGroups).budget?.toInternalBudgetPerOpSpec(),
      preThreshold = groupsType.getPreThreshold(),
      contributionBoundingLevel = contributionBoundingLevel.toInternalContributionBoundingLevel(),
    )

  private fun createAggregationParams(aggregationSpecs: List<AggregationSpec>): AggregationParams {
    val nonFeatureMetrics =
      aggregationSpecs
        .filter { it is Count || it is PrivacyIdCount }
        .map { it.toNonFeatureMetricDefinition() }
    val features =
      aggregationSpecs.mapNotNull {
        when (it) {
          is ValueAggregations<*> -> {
            val valueContributionBounds = it.contributionBounds
            ScalarFeatureSpec(
              featureId = it.getFeatureId(),
              metrics = it.valueAggregationSpecs.map { it.toMetricDefinition() }.toImmutableList(),
              minValue = valueContributionBounds.valueBounds?.minValue,
              maxValue = valueContributionBounds.valueBounds?.maxValue,
              minTotalValue = valueContributionBounds.totalValueBounds?.minValue,
              maxTotalValue = valueContributionBounds.totalValueBounds?.maxValue,
            )
          }
          is VectorAggregations<*> -> {
            val vectorContributionBounds = it.vectorContributionBounds
            VectorFeatureSpec(
              featureId = it.getFeatureId(),
              metrics = it.vectorAggregationSpecs.map { it.toMetricDefinition() }.toImmutableList(),
              vectorSize = it.vectorSize,
              normKind = vectorContributionBounds.maxVectorTotalNorm.normKind.toInternalNormKind(),
              vectorMaxTotalNorm = vectorContributionBounds.maxVectorTotalNorm.value,
            )
          }
          else -> null
        }
      }

    return AggregationParams(
      nonFeatureMetrics = nonFeatureMetrics.toImmutableList(),
      features = features.toImmutableList(),
      noiseKind =
        checkNotNull(noiseKind) { "noiseKind cannot be null if there are aggregations." }
          .toInternalNoiseKind(),
      maxPartitionsContributed = contributionBoundingLevel.getMaxPartitionsContributed(),
      maxContributionsPerPartition = contributionBoundingLevel.getMaxContributionsPerPartition(),
      partitionSelectionBudget = groupsType.getBudget()?.toInternalBudgetPerOpSpec(),
      preThreshold = groupsType.getPreThreshold(),
      contributionBoundingLevel = contributionBoundingLevel.toInternalContributionBoundingLevel(),
      partitionsBalance = groupByAdditionalParameters.groupsBalance.toPartitionsBalance(),
    )
  }
}

private fun <T : Any> Iterable<T>.toImmutableList(): ImmutableList<T> = ImmutableList.copyOf(this)

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
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricDefinition
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType

/** An internal interface to specify aggregations. */
sealed interface AggregationSpec

/**
 * A privacy id count aggregation.
 *
 * @param outputColumnName the name of the column to write the result to.
 * @param budget the budget to use for the aggregation.
 */
internal data class PrivacyIdCount(val outputColumnName: String, val budget: BudgetPerOpSpec?) :
  AggregationSpec

/**
 * A count aggregation.
 *
 * @param outputColumnName the name of the column to write the result to.
 * @param budget the budget to use for the aggregation.
 */
internal data class Count(val outputColumnName: String, val budget: BudgetPerOpSpec?) :
  AggregationSpec

/**
 * Aggregations of specific value.
 *
 * @param valueExtractor a function to extract the value to aggregate from the input.
 * @param valueAggregationSpecs the aggregations to perform.
 * @param contributionBounds contribution bounds of the value.
 * @param valueColumnName in case of data frames and column based API.
 */
internal data class ValueAggregations<DataRowT : Any>(
  val valueExtractor: (DataRowT) -> Double,
  val valueAggregationSpecs: List<ValueAggregationSpec>,
  val contributionBounds: ContributionBounds,
  val valueColumnName: String? = null,
) : AggregationSpec

/**
 * Aggregations of specific vector.
 *
 * @param vectorExtractor a function to extract the vector to aggregate from the input.
 * @param vectorSize the size of the vectors that will be aggregated.
 * @param vectorAggregationSpecs the aggregations to perform.
 * @param vectorContributionBounds contribution bounds of the vectors.
 * @param vectorColumnNames column names that form the vector in case of data frames and column
 *   based API.
 */
internal data class VectorAggregations<DataRowT : Any>(
  val vectorExtractor: (DataRowT) -> List<Double>,
  val vectorSize: Int,
  val vectorAggregationSpecs: List<VectorAggregationSpec>,
  val vectorContributionBounds: VectorContributionBounds,
  val vectorColumnNames: ColumnNames? = null,
) : AggregationSpec

/** Internal representation of aggregations of specific value. */
@ConsistentCopyVisibility
internal data class ValueAggregationSpec
private constructor(
  val metricType: MetricType,
  val outputColumnName: String,
  val budget: BudgetPerOpSpec?,
) {
  companion object {
    fun sum(outputColumnName: String, budget: BudgetPerOpSpec?) =
      ValueAggregationSpec(MetricType.SUM, outputColumnName, budget)

    fun mean(outputColumnName: String, budget: BudgetPerOpSpec?) =
      ValueAggregationSpec(MetricType.MEAN, outputColumnName, budget)

    fun variance(outputColumnName: String, budget: BudgetPerOpSpec?) =
      ValueAggregationSpec(MetricType.VARIANCE, outputColumnName, budget)

    fun quantiles(ranks: List<Double>, outputColumnName: String, budget: BudgetPerOpSpec?) =
      ValueAggregationSpec(
        MetricType.QUANTILES(ImmutableList.copyOf(ranks)),
        outputColumnName,
        budget,
      )
  }
}

/** Internal representation of aggregations of specific vector. */
internal data class VectorAggregationSpec
private constructor(
  val metricType: MetricType,
  val outputColumnName: String,
  val budget: BudgetPerOpSpec?,
) {
  companion object {
    fun vectorSum(outputColumnName: String, budget: BudgetPerOpSpec?) =
      VectorAggregationSpec(MetricType.VECTOR_SUM, outputColumnName, budget)
  }
}

internal fun List<AggregationSpec>.budgets(): Map<String, BudgetPerOpSpec?> = buildMap {
  for (aggregation in this@budgets) {
    when (aggregation) {
      is PrivacyIdCount -> put(aggregation.outputColumnName, aggregation.budget)
      is Count -> put(aggregation.outputColumnName, aggregation.budget)
      is ValueAggregations<*> -> {
        for (valueAggregationSpec in aggregation.valueAggregationSpecs) {
          put(valueAggregationSpec.outputColumnName, valueAggregationSpec.budget)
        }
      }
      is VectorAggregations<*> -> {
        for (vectorAggregationSpec in aggregation.vectorAggregationSpecs) {
          put(vectorAggregationSpec.outputColumnName, vectorAggregationSpec.budget)
        }
      }
    }
  }
}

internal fun List<AggregationSpec>.metrics(): List<MetricDefinition> = buildList {
  for (aggregation in this@metrics) {
    when (aggregation) {
      // Count and PrivacyIdCount do not aggregate any specific value, therefore they are handled
      // differently.
      is PrivacyIdCount ->
        add(
          MetricDefinition(
            MetricType.PRIVACY_ID_COUNT,
            aggregation.budget?.toInternalBudgetPerOpSpec(),
          )
        )
      is Count ->
        add(MetricDefinition(MetricType.COUNT, aggregation.budget?.toInternalBudgetPerOpSpec()))
      is ValueAggregations<*> -> {
        for (valueAggregationSpec in aggregation.valueAggregationSpecs) {
          add(
            MetricDefinition(
              valueAggregationSpec.metricType,
              valueAggregationSpec.budget?.toInternalBudgetPerOpSpec(),
            )
          )
        }
      }
      is VectorAggregations<*> -> {
        for (vectorAggregationSpec in aggregation.vectorAggregationSpecs) {
          add(
            MetricDefinition(
              vectorAggregationSpec.metricType,
              vectorAggregationSpec.budget?.toInternalBudgetPerOpSpec(),
            )
          )
        }
      }
    }
  }
}

internal fun List<AggregationSpec>.outputColumnNamesWithMetricTypes():
  List<Pair<String, MetricType>> = buildList {
  for (aggregation in this@outputColumnNamesWithMetricTypes) {
    when (aggregation) {
      is PrivacyIdCount -> add(aggregation.outputColumnName to MetricType.PRIVACY_ID_COUNT)
      is Count -> add(aggregation.outputColumnName to MetricType.COUNT)
      is ValueAggregations<*> -> {
        for (valueAggregationSpec in aggregation.valueAggregationSpecs) {
          add(valueAggregationSpec.outputColumnName to valueAggregationSpec.metricType)
        }
      }
      is VectorAggregations<*> -> {
        for (vectorAggregationSpec in aggregation.vectorAggregationSpecs) {
          add(vectorAggregationSpec.outputColumnName to vectorAggregationSpec.metricType)
        }
      }
    }
  }
}

internal fun List<AggregationSpec>.outputColumnNames(): List<String> =
  outputColumnNamesWithMetricTypes().map { it.first }

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
import com.google.privacy.differentialprivacy.pipelinedp4j.core.budget.BudgetPerOpSpec

/**
 * An internal interface to specify an aggregation.
 *
 * The fields of the sealed class are the properties that any aggregation must have: in which column
 * to write the result, how much budget we can use, aggregation type, and how to extract the
 * aggregated value from the input.
 *
 * @param T the type of the elements in the collection.
 */
sealed class AggregationSpec<T>
private constructor(
  internal val outputColumnName: String,
  internal val budget: BudgetPerOpSpec?,
  metricType: MetricType,
  internal val valueExtractor: ((T) -> Double)? = null,
) {
  internal val metricDefinition = MetricDefinition(metricType, budget)

  /**
   * A privacy id count aggregation.
   *
   * @param outputColumnName the name of the column to write the result to.
   * @param budget the budget to use for the aggregation.
   */
  internal class PrivacyIdCount<T>(outputColumnName: String, budget: BudgetPerOpSpec?) :
    AggregationSpec<T>(
      outputColumnName,
      budget,
      MetricType.PRIVACY_ID_COUNT,
      valueExtractor = null,
    ) {}

  /**
   * A count aggregation.
   *
   * @param outputColumnName the name of the column to write the result to.
   * @param budget the budget to use for the aggregation.
   */
  internal class Count<T>(outputColumnName: String, budget: BudgetPerOpSpec?) :
    AggregationSpec<T>(outputColumnName, budget, MetricType.COUNT, valueExtractor = null) {}

  /**
   * A sum aggregation.
   *
   * @param outputColumnName the name of the column to write the result to.
   * @param budget the budget to use for the aggregation.
   * @param valueExtractor a function to extract the aggregated value from the input.
   * @param minTotalValue the minimum value of the sum.
   * @param maxTotalValue the maximum value of the sum.
   */
  internal class Sum<T>(
    outputColumnName: String,
    budget: BudgetPerOpSpec?,
    valueExtractor: (T) -> Double,
    internal val minTotalValue: Double?,
    internal val maxTotalValue: Double?,
  ) : AggregationSpec<T>(outputColumnName, budget, MetricType.SUM, valueExtractor) {}

  /**
   * A mean aggregation.
   *
   * @param outputColumnName the name of the column to write the result to.
   * @param budget the budget to use for the aggregation.
   * @param valueExtractor a function to extract the aggregated value from the input.
   * @param minValue the smallest possible value that a privacy unit can contribute.
   * @param maxValue the largest possible value that a privacy unit can contribute.
   */
  internal class Mean<T>(
    outputColumnName: String,
    budget: BudgetPerOpSpec?,
    valueExtractor: (T) -> Double,
    internal val minValue: Double,
    internal val maxValue: Double,
  ) : AggregationSpec<T>(outputColumnName, budget, MetricType.MEAN, valueExtractor) {}

  /**
   * A variance aggregation.
   *
   * @param outputColumnName the name of the column to write the result to.
   * @param budget the budget to use for the aggregation.
   * @param valueExtractor a function to extract the aggregated value from the input.
   * @param minValue the smallest possible value that a privacy unit can contribute.
   * @param maxValue the largest possible value that a privacy unit can contribute.
   */
  internal class Variance<T>(
    outputColumnName: String,
    budget: BudgetPerOpSpec?,
    valueExtractor: (T) -> Double,
    internal val minValue: Double,
    internal val maxValue: Double,
  ) : AggregationSpec<T>(outputColumnName, budget, MetricType.VARIANCE, valueExtractor) {}

  /**
   * A quantiles aggregation.
   *
   * @param outputColumnName the name of the column to write the result to.
   * @param budget the budget to use for the aggregation.
   * @param valueExtractor a function to extract the aggregated value from the input.
   * @param ranks the ranks of the quantiles to compute.
   * @param minValue the smallest possible value that a privacy unit can contribute.
   * @param maxValue the largest possible value that a privacy unit can contribute.
   */
  internal class Quantiles<T>(
    outputColumnName: String,
    budget: BudgetPerOpSpec?,
    valueExtractor: (T) -> Double,
    ranks: List<Double>,
    internal val minValue: Double,
    internal val maxValue: Double,
  ) :
    AggregationSpec<T>(
      outputColumnName,
      budget,
      MetricType.QUANTILES(ImmutableList.copyOf(ranks)),
      valueExtractor,
    ) {}
}

internal fun <T> List<AggregationSpec<T>>.minTotalValue(): Double? {
  val values =
    mapNotNull {
        when (it) {
          is AggregationSpec.PrivacyIdCount<*> -> null
          is AggregationSpec.Count<*> -> null
          is AggregationSpec.Sum<*> -> it.minTotalValue
          is AggregationSpec.Mean<*> -> null
          is AggregationSpec.Variance<*> -> null
          is AggregationSpec.Quantiles<*> -> null
        }
      }
      .toSet()
  require(values.size <= 1) {
    "Different minTotalValues: ${values}. minTotalValue can be specified only once because for now only aggregations of the same value are supported."
  }
  return values.singleOrNull()
}

internal fun <T> List<AggregationSpec<T>>.maxTotalValue(): Double? {
  val values =
    mapNotNull {
        when (it) {
          is AggregationSpec.PrivacyIdCount<*> -> null
          is AggregationSpec.Count<*> -> null
          is AggregationSpec.Sum<*> -> it.maxTotalValue
          is AggregationSpec.Mean<*> -> null
          is AggregationSpec.Variance<*> -> null
          is AggregationSpec.Quantiles<*> -> null
        }
      }
      .toSet()
  require(values.size <= 1) {
    "Different maxTotalValues: ${values}. maxTotalValue can be specified only once because for now only aggregations of the same value are supported."
  }
  return values.singleOrNull()
}

internal fun <T> List<AggregationSpec<T>>.minValue(): Double? {
  val values =
    mapNotNull {
        when (it) {
          is AggregationSpec.PrivacyIdCount<*> -> null
          is AggregationSpec.Count<*> -> null
          is AggregationSpec.Sum<*> -> null
          is AggregationSpec.Mean<*> -> it.minValue
          is AggregationSpec.Variance<*> -> it.minValue
          is AggregationSpec.Quantiles<*> -> it.minValue
        }
      }
      .toSet()
  require(values.size <= 1) {
    "Different minValues: ${values}. Only aggregations of the same value are supported for now and they must have the same bounds including minValue."
  }
  return values.singleOrNull()
}

internal fun <T> List<AggregationSpec<T>>.maxValue(): Double? {
  val values =
    mapNotNull {
        when (it) {
          is AggregationSpec.PrivacyIdCount<*> -> null
          is AggregationSpec.Count<*> -> null
          is AggregationSpec.Sum<*> -> null
          is AggregationSpec.Mean<*> -> it.maxValue
          is AggregationSpec.Variance<*> -> it.maxValue
          is AggregationSpec.Quantiles<*> -> it.maxValue
        }
      }
      .toSet()
  require(values.size <= 1) {
    "Different maxValues: ${values}. Only aggregations of the same value are supported for now and they must have the same bounds including maxValue."
  }
  return values.singleOrNull()
}

internal fun <T> List<AggregationSpec<T>>.outputColumnNamesWithMetricTypes() = map {
  it.outputColumnName to it.metricDefinition.type
}

internal fun <T> List<AggregationSpec<T>>.valueExtractors() =
  mapNotNull {
      when (it) {
        is AggregationSpec.PrivacyIdCount<*> -> null
        is AggregationSpec.Count<*> -> null
        is AggregationSpec.Sum<*> -> it.valueExtractor
        is AggregationSpec.Mean<*> -> it.valueExtractor
        is AggregationSpec.Variance<*> -> it.valueExtractor
        is AggregationSpec.Quantiles<*> -> it.valueExtractor
      }
    }
    .toSet()

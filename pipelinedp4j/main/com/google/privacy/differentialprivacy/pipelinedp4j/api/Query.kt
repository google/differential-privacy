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
import com.google.privacy.differentialprivacy.pipelinedp4j.core.FrameworkTable
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.DpAggregates

/**
 * A differentially-private query.
 *
 * You can get the instance of a query by calling [QueryBuilder.build].
 *
 * @param T the type of the elements in the collection.
 * @param R the type of the result.
 */
sealed class Query<T, R>
protected constructor(
  private val data: PipelineDpCollection<T>,
  private val privacyIdExtractor: (T) -> String,
  private val groupKeyExtractor: (T) -> String,
  private val maxGroupsContributed: Int,
  private val maxContributionsPerGroup: Int,
  private val publicKeys: PipelineDpCollection<String>?,
  protected val aggregations: List<AggregationSpec<T>>,
) {
  init {
    validate()
  }

  /**
   * Runs the query with the given total budget and noise kind.
   *
   * @param budget the budget to use for the query.
   * @param noiseKind the noise kind to use for the query.
   * @return the result of the query.
   */
  abstract fun run(budget: TotalBudget, noiseKind: NoiseKind): R

  protected fun runWithDpEngine(
    budget: TotalBudget,
    noiseKind: NoiseKind,
  ): FrameworkTable<String, DpAggregates> {
    val dpEngine =
      DpEngine.create(data.encoderFactory, DpEngineBudgetSpec(budget.toInternalTotalBudget()))

    val valueExtractor = aggregations.mapNotNull { it.valueExtractor }.toSet().singleOrNull()
    val extractors =
      if (valueExtractor != null)
        DataExtractors.from<T, String, String>(
          privacyIdExtractor,
          privacyIdEncoder = data.encoderFactory.strings(),
          partitionKeyExtractor = groupKeyExtractor,
          partitionKeyEncoder = data.encoderFactory.strings(),
          valueExtractor = valueExtractor,
        )
      else
        DataExtractors.from<T, String, String>(
          privacyIdExtractor,
          privacyIdEncoder = data.encoderFactory.strings(),
          partitionKeyExtractor = groupKeyExtractor,
          partitionKeyEncoder = data.encoderFactory.strings(),
        )
    val aggregateParams =
      AggregationParams(
        metrics = ImmutableList.copyOf(aggregations.map { it.metricDefinition }),
        noiseKind = noiseKind.toInternalNoiseKind(),
        maxPartitionsContributed = maxGroupsContributed,
        maxContributionsPerPartition = maxContributionsPerGroup,
        minTotalValue = aggregations.minTotalValue(),
        maxTotalValue = aggregations.maxTotalValue(),
        minValue = aggregations.minValue(),
        maxValue = aggregations.maxValue(),
      )
    val result =
      dpEngine.aggregate(
        data.toFrameworkCollection(),
        aggregateParams,
        extractors,
        publicKeys?.toFrameworkCollection(),
      )
    dpEngine.done()
    return result
  }

  private fun validate() {
    requireNoDuplicateAggregations()
    requireOneValue()
  }

  private fun requireNoDuplicateAggregations() {
    val outputColumnNameCounts =
      aggregations.map { it.outputColumnName }.groupingBy { it }.eachCount()
    val duplicates = outputColumnNameCounts.filter { it.value > 1 }.keys
    require(duplicates.isEmpty()) {
      "There aggregations with duplicate output column names: ${duplicates}."
    }
  }

  private fun requireOneValue() =
    require(aggregations.valueExtractors().size <= 1) {
      "Aggregation of different values is not supported yet. Please aggregate only one value. If you provide value extractors then it has to be the same instance."
    }
}

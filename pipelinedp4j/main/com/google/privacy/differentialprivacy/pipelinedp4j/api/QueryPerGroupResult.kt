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

import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.DpAggregates

/**
 * The result of a query for a single group.
 *
 * @property groupKey is the group key.
 * @property aggregationResults is mapping from column name to the result of the aggregation for
 *   this group. For quantiles there will be multiple entries in the map, one per rank, with names
 *   "aggregationName_rank".
 * @param GroupKeysT is the type of the group key.
 */
@ConsistentCopyVisibility
data class QueryPerGroupResult<GroupKeysT : Any>
internal constructor(
  private val nullableGroupKey: GroupKeysT?,
  val aggregationResults: Map<String, Double>,
) {
  val groupKey: GroupKeysT
    get() = nullableGroupKey!!

  // Necessary for Beam serialization.
  private constructor() : this(null, mapOf())

  companion object {
    internal fun <GroupKeysT : Any> create(
      groupKey: GroupKeysT,
      dpAggregates: DpAggregates,
      outputColumnNamesWithMetricTypes: List<Pair<String, MetricType>>,
    ): QueryPerGroupResult<GroupKeysT> {
      val aggregationsMap =
        constructAggregationResults(dpAggregates, outputColumnNamesWithMetricTypes)
      return QueryPerGroupResult<GroupKeysT>(groupKey, aggregationsMap)
    }

    internal fun constructAggregationResults(
      dpAggregates: DpAggregates,
      outputColumnNamesWithMetricTypes: List<Pair<String, MetricType>>,
    ) =
      buildMap<String, Double> {
        for ((outputColumnName, metricType) in outputColumnNamesWithMetricTypes) {
          when (metricType) {
            MetricType.PRIVACY_ID_COUNT -> put(outputColumnName, dpAggregates.privacyIdCount)
            MetricType.COUNT -> put(outputColumnName, dpAggregates.count)
            MetricType.SUM -> put(outputColumnName, dpAggregates.sum)
            MetricType.VECTOR_SUM ->
              throw IllegalArgumentException("Vector sum is not supported yet.")
            MetricType.MEAN -> put(outputColumnName, dpAggregates.mean)
            MetricType.VARIANCE -> put(outputColumnName, dpAggregates.variance)
            is MetricType.QUANTILES -> {
              // TODO: consider creating a data class or resuing copy of
              // DpAggregates proto and not allowing outputColumnName.
              for ((rank, value) in metricType.sortedRanks.zip(dpAggregates.quantilesList)) {
                put(outputColumnName.withRank(rank), value)
              }
            }
          }
        }
      }

    internal fun columnsNamesInAggregationResults(
      outputColumnNamesWithMetricTypes: List<Pair<String, MetricType>>
    ) =
      buildList<String> {
        for ((outputColumnName, metricType) in outputColumnNamesWithMetricTypes) {
          when (metricType) {
            MetricType.PRIVACY_ID_COUNT -> add(outputColumnName)
            MetricType.COUNT -> add(outputColumnName)
            MetricType.SUM -> add(outputColumnName)
            MetricType.VECTOR_SUM ->
              throw IllegalArgumentException("Vector sum is not supported yet.")
            MetricType.MEAN -> add(outputColumnName)
            MetricType.VARIANCE -> add(outputColumnName)
            is MetricType.QUANTILES -> {
              for (rank in metricType.sortedRanks) {
                add(outputColumnName.withRank(rank))
              }
            }
          }
        }
      }

    private fun String.withRank(rank: Double) = "${this}_${rank}"
  }
}

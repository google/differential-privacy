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
  val valueAggregationResults: Map<String, Double>,
  val vectorAggregationResults: Map<String, List<Double>>,
) {
  val groupKey: GroupKeysT
    get() = nullableGroupKey!!

  // Necessary for Beam serialization.
  private constructor() : this(null, mapOf(), mapOf())

  companion object {
    internal fun <GroupKeysT : Any> create(
      groupKey: GroupKeysT,
      dpAggregates: DpAggregates,
      outputColumnNamesWithMetricTypes: List<Pair<String, MetricType>>,
    ): QueryPerGroupResult<GroupKeysT> {
      val valueAggregationsMap =
        constructValueAggregationResults(dpAggregates, outputColumnNamesWithMetricTypes)
      val vectorAggregationsMap =
        constructVectorAggregationResults(dpAggregates, outputColumnNamesWithMetricTypes)
      return QueryPerGroupResult<GroupKeysT>(groupKey, valueAggregationsMap, vectorAggregationsMap)
    }

    internal fun valueColumnsNamesInAggregationResults(
      outputColumnNamesWithMetricTypes: List<Pair<String, MetricType>>
    ) =
      buildList<String> {
        for ((outputColumnName, metricType) in outputColumnNamesWithMetricTypes) {
          when (metricType) {
            MetricType.PRIVACY_ID_COUNT -> add(outputColumnName)
            MetricType.COUNT -> add(outputColumnName)
            MetricType.SUM -> add(outputColumnName)
            MetricType.VECTOR_SUM -> {} // not processed in this function.
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

    internal fun vectorColumnsNamesInAggregationResults(
      outputColumnNamesWithMetricTypes: List<Pair<String, MetricType>>
    ) =
      buildList<String> {
        for ((outputColumnName, metricType) in outputColumnNamesWithMetricTypes) {
          when (metricType) {
            MetricType.PRIVACY_ID_COUNT -> {} // not processed in this function.
            MetricType.COUNT -> {} // not processed in this function.
            MetricType.SUM -> {} // not processed in this function.
            MetricType.VECTOR_SUM -> add(outputColumnName)
            MetricType.MEAN -> {} // not processed in this function.
            MetricType.VARIANCE -> {} // not processed in this function.
            is MetricType.QUANTILES -> {} // not processed in this function.
          }
        }
      }

    private fun constructValueAggregationResults(
      dpAggregates: DpAggregates,
      outputColumnNamesWithMetricTypes: List<Pair<String, MetricType>>,
    ) =
      buildMap<String, Double> {
        for ((outputColumnName, metricType) in outputColumnNamesWithMetricTypes) {
          when (metricType) {
            MetricType.PRIVACY_ID_COUNT -> put(outputColumnName, dpAggregates.privacyIdCount)
            MetricType.COUNT -> put(outputColumnName, dpAggregates.count)
            MetricType.SUM -> put(outputColumnName, dpAggregates.sum)
            MetricType.VECTOR_SUM -> {} // not processed in this function.
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

    private fun constructVectorAggregationResults(
      dpAggregates: DpAggregates,
      outputColumnNamesWithMetricTypes: List<Pair<String, MetricType>>,
    ) =
      buildMap<String, List<Double>> {
        for ((outputColumnName, metricType) in outputColumnNamesWithMetricTypes) {
          when (metricType) {
            MetricType.PRIVACY_ID_COUNT -> {} // not processed in this function.
            MetricType.COUNT -> {} // not processed in this function.
            MetricType.SUM -> {} // not processed in this function.
            MetricType.VECTOR_SUM -> put(outputColumnName, dpAggregates.vectorSumList)
            MetricType.MEAN -> {} // not processed in this function.
            MetricType.VARIANCE -> {} // not processed in this function.
            is MetricType.QUANTILES -> {} // not processed in this function.
          }
        }
      }

    private fun String.withRank(rank: Double) = "${this}_${rank}"
  }
}

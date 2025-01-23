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

import com.google.privacy.differentialprivacy.pipelinedp4j.beam.BeamTable
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.DpAggregates
import org.apache.beam.sdk.extensions.avro.coders.AvroCoder
import org.apache.beam.sdk.transforms.MapElements
import org.apache.beam.sdk.transforms.SerializableFunction
import org.apache.beam.sdk.values.KV
import org.apache.beam.sdk.values.PCollection as BeamPCollection

/**
 * A differentially-private query to run on Beam.
 *
 * @param T the type of the elements in the collection.
 */
class BeamQuery<T>
internal constructor(
  data: PipelineDpCollection<T>,
  privacyIdExtractor: (T) -> String,
  groupKeyExtractor: (T) -> String,
  maxGroupsContributed: Int,
  maxContributionsPerGroup: Int,
  publicKeys: PipelineDpCollection<String>?,
  aggregations: List<AggregationSpec<T>>,
) :
  Query<T, BeamPCollection<QueryPerGroupResult>>(
    data,
    privacyIdExtractor,
    groupKeyExtractor,
    maxGroupsContributed,
    maxContributionsPerGroup,
    publicKeys,
    aggregations,
  ) {
  /**
   * Runs the query with the given total budget and noise kind.
   *
   * @param budget the budget to use for the query.
   * @param noiseKind the noise kind to use for the query.
   * @return the result of the query.
   */
  override fun run(
    budget: TotalBudget,
    noiseKind: NoiseKind,
  ): BeamPCollection<QueryPerGroupResult> {
    val result = (runWithDpEngine(budget, noiseKind) as BeamTable<String, DpAggregates>).data
    val outputColumnNamesWithMetricTypes = aggregations.outputColumnNamesWithMetricTypes()
    val coder = AvroCoder.of(QueryPerGroupResult::class.java)
    val mapToResultFn = { kv: KV<String, DpAggregates> ->
      val key = kv.key
      val dpAggregates = kv.value

      val aggregationsMap =
        buildMap<String, Double> {
          for ((outputColumnName, metricType) in outputColumnNamesWithMetricTypes) {
            when (metricType) {
              MetricType.PRIVACY_ID_COUNT -> put(outputColumnName, dpAggregates.privacyIdCount)
              MetricType.COUNT -> put(outputColumnName, dpAggregates.count)
              MetricType.SUM -> put(outputColumnName, dpAggregates.sum)
              MetricType.MEAN -> put(outputColumnName, dpAggregates.mean)
              MetricType.VARIANCE -> put(outputColumnName, dpAggregates.variance)
              is MetricType.QUANTILES -> {
                // TODO: consider creating a data class or resuing copy of DpAggregates
                // proto and not allowing outputColumnName.
                for ((rank, value) in metricType.sortedRanks.zip(dpAggregates.quantilesList)) {
                  put("${outputColumnName}_${rank}", value)
                }
              }
            }
          }
        }

      QueryPerGroupResult(key, aggregationsMap)
    }
    return result
      .apply(MapElements.into(coder.encodedTypeDescriptor).via(SerializableFunction(mapToResultFn)))
      .setCoder(coder)
  }
}

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


import scala.Tuple2

/**
 * The result of a query for a single partition.
 *
 * @param groupKey is the partition key.
 * @param aggregationResults is mapping from aggregation name to the result of the aggregation for
 *   this partition. For quantiles there will be multiple entries in the map, one per rank, with
 *   names "aggregationName_rank".
 */
data class QueryPerGroupResult(val groupKey: String, val aggregationResults: Map<String, Double>) {
  // Necessary for Beam serialization.
  private constructor() : this("", mapOf())
}

val queryPerGroupResultTuple = Tuple2("", java.util.HashMap<String, Double>())
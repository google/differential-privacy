/*
 * Copyright 2025 Google LLC
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

/** Builder to specify which aggregations to perform on a vector. */
class VectorAggregationsBuilder() {
  internal val aggregations = mutableListOf<VectorAggregationSpec>()

  /**
   * Instructs the query to calculate the sum of vectors in each group.
   *
   * @param outputColumnName the name of the key in [QueryPerGroupResult.aggregationResults] which
   *   will be the output of the query.
   * @param budget the budget to use for the aggregation, optional. If not specified, then the
   *   budget will be relative and have weight 1. See [RelativeBudgetPerOpSpec] for details on what
   *   weights mean.
   */
  @JvmOverloads
  fun vectorSum(
    outputColumnName: String,
    budget: BudgetPerOpSpec? = null,
  ): VectorAggregationsBuilder {
    aggregations.add(VectorAggregationSpec.vectorSum(outputColumnName, budget))
    return this
  }
}

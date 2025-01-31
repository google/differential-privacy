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

/** Builder to specify which aggregations to perform on a value. */
class ValueAggregationsBuilder() {
  internal val aggregations = mutableListOf<ValueAggregationSpec>()

  /**
   * Instructs the query to calculate the sum of values in each group.
   *
   * @param outputColumnName the name of the key in [QueryPerGroupResult.aggregationResults] which
   *   will be the output of the query.
   * @param budget the budget to use for the aggregation, optional. If not specified, then the
   *   budget will be relative and have weight 1. See [RelativeBudgetPerOpSpec] for details on what
   *   weights mean.
   */
  @JvmOverloads
  fun sum(outputColumnName: String, budget: BudgetPerOpSpec? = null): ValueAggregationsBuilder {
    aggregations.add(ValueAggregationSpec.sum(outputColumnName, budget))
    return this
  }

  /**
   * Instructs the query to calculate the mean of values in each group.
   *
   * @param outputColumnName the name of the key in [QueryPerGroupResult.aggregationResults] which
   *   will be the output of the query.
   * @param budget the budget to use for the aggregation, optional. If not specified, then the
   *   budget will be relative and have weight 1. See [RelativeBudgetPerOpSpec] for details on what
   *   weights mean.
   */
  @JvmOverloads
  fun mean(outputColumnName: String, budget: BudgetPerOpSpec? = null): ValueAggregationsBuilder {
    aggregations.add(ValueAggregationSpec.mean(outputColumnName, budget))
    return this
  }

  /**
   * Instructs the query to calculate the variance of values in each group.
   *
   * @param outputColumnName the name of the key in [QueryPerGroupResult.aggregationResults] which
   *   will be the output of the query.
   * @param budget the budget to use for the aggregation, optional. If not specified, then the
   *   budget will be relative and have weight 1. See [RelativeBudgetPerOpSpec] for details on what
   *   weights mean.
   */
  @JvmOverloads
  fun variance(
    outputColumnName: String,
    budget: BudgetPerOpSpec? = null,
  ): ValueAggregationsBuilder {
    aggregations.add(ValueAggregationSpec.variance(outputColumnName, budget))
    return this
  }

  /**
   * Instructs the query to calculate the quantiles of values in each group.
   *
   * @param ranks the ranks of the quantiles to calculate.
   * @param outputColumnName the prefix of the keys in [QueryPerGroupResult.aggregationResults]
   *   which contain the output of the query. For quantiles, the number of output column names will
   *   equal the number of ranks requested to calculate. The column names will be
   *   `outputColumnName + "_<rank>"`.
   * @param budget the budget to use for the aggregation, optional. If not specified, then the
   *   budget will be relative and have weight 1. See [RelativeBudgetPerOpSpec] for details on what
   *   weights mean.
   */
  @JvmOverloads
  fun quantiles(
    ranks: List<Double>,
    outputColumnName: String,
    budget: BudgetPerOpSpec? = null,
  ): ValueAggregationsBuilder {
    aggregations.add(ValueAggregationSpec.quantiles(ranks, outputColumnName, budget))
    return this
  }
}

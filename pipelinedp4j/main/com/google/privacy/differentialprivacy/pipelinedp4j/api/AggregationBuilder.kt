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

/**
 * Specification of the differentially private aggregations of the query.
 *
 * If you want only to select private groups then you can call [AggregationBuilder.build]
 * immediately without calling any of the aggregation functions.
 */
sealed class AggregationBuilder<DataRowT : Any, ReturnT : Any> {
  // TODO: think how to do it nicer, it can be val, we need it var only due to delegation
  internal var aggregations = mutableListOf<AggregationSpec>()

  /**
   * Instructs the query to count the number of distinct privacy units in each group.
   *
   * @param outputColumnName the name of the key in [QueryPerGroupResult.aggregationResults] which
   *   will be the output of the query.
   * @param budget the budget to use for the aggregation, optional. If not specified, then the
   *   budget will be relative and have weight 1. See [RelativeBudgetPerOpSpec] for details on what
   *   weights mean.
   */
  @JvmOverloads
  fun countDistinctPrivacyUnits(
    outputColumnName: String,
    budget: BudgetPerOpSpec? = null,
  ): AggregationBuilder<DataRowT, ReturnT> {
    aggregations.add(PrivacyIdCount(outputColumnName, budget))
    return this
  }

  /**
   * Instructs the query to count the number of rows in each group.
   *
   * @param outputColumnName the name of the key in [QueryPerGroupResult.aggregationResults] which
   *   will be the output of the query.
   * @param budget the budget to use for the aggregation, optional. If not specified, then the
   *   budget will be relative and have weight 1. See [RelativeBudgetPerOpSpec] for details on what
   *   weights mean.
   */
  @JvmOverloads
  fun count(
    outputColumnName: String,
    budget: BudgetPerOpSpec? = null,
  ): AggregationBuilder<DataRowT, ReturnT> {
    aggregations.add(Count(outputColumnName, budget))
    return this
  }

  /**
   * Instructs the query to aggregate a value in each group.
   *
   * @param valueExtractor a function to extract from the input row the value to aggregate. It must
   *   be serializable. In Java, this means it can't be a Java lambda, method referencs or anonymous
   *   class because they capture `this` and therefore are not serializable. A workaround is to
   *   create a static nested class or a standalone class.
   * @param valueAggregations the aggregations to perform on the value.
   * @param contributionBounds contribution bounds for the value, see [ContributionBounds] for
   *   details on which bounds to specify in which cases.
   */
  fun aggregateValue(
    valueExtractor: (DataRowT) -> Double,
    valueAggregations: ValueAggregationsBuilder,
    contributionBounds: ContributionBounds,
  ): AggregationBuilder<DataRowT, ReturnT> {
    aggregations.add(
      ValueAggregations(valueExtractor, valueAggregations.aggregations, contributionBounds)
    )
    return this
  }

  /**
   * Instructs the query to aggregate a value in each group.
   *
   * @param valueColumnName column that stores the values to aggregate, the values must be of
   *   [Double] type.
   * @param valueAggregations the aggregations to perform on the value.
   * @param contributionBounds contribution bounds for the value, see [ContributionBounds] for
   *   details on which bounds to specify in which cases.
   */
  open fun aggregateValue(
    valueColumnName: String,
    valueAggregations: ValueAggregationsBuilder,
    contributionBounds: ContributionBounds,
  ): AggregationBuilder<DataRowT, ReturnT> {
    throw UnsupportedOperationException(
      "Column-based operations are not supported in row-based collections."
    )
  }

  /**
   * Builds the query.
   *
   * Final operation. After calling this method you get a query that you can execute.
   *
   * @param totalBudget the total budget to use for the query, the query will fail if the scheduled
   *   operations will exceed the total budget.
   * @param noiseKind the noise kind to use for the query, not needed if only selecting groups.
   */
  abstract fun build(totalBudget: TotalBudget, noiseKind: NoiseKind? = null): Query<ReturnT>
}

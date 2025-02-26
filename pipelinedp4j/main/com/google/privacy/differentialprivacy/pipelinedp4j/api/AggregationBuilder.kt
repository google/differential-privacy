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
   * Instructs the query to aggregate a vector in each group.
   *
   * A vector is any list of values of any size, including one-dimensional vectors. During
   * aggregation and contribution bounding the provided list of n values is treated as a vector in
   * n-dimensional space (R^n). Only vector aggregations are supported here (e.g. vector sum).
   *
   * You should use vector aggregation instead of aggregating each dimension separately if you want
   * to have contribution bounding to be propotional across dimensions (proportional to the norm of
   * the vector) which results in less bias. Also, using L1 and L2 max norm as contribution bounds
   * will result in less noise than L_inf (usual contribution bounding used in count, sum and other
   * aggregations).
   *
   * Example: you want to calculate number of OK, CANCELLED and ERROR responses. You have two
   * options:
   * 1. Make response type an additional column in your data and group by it.
   * 2. Create a vector of size 3 with one-hot encoding of the response type, e.g. [1, 0, 0] for OK,
   *    [0, 1, 0] for CANCELLED, [0, 0, 1] for ERROR. Then sum up the vectors.
   *
   * The first approach will give you more biased results that can be on different scale because the
   * same contribution bounds will applied in each group. Let's say you set maxContributionsPerGroup
   * to 100 and in your data OK responses are much more common than CANCELLED and ERROR. Then only
   * OK responses will be clamped and the contribution to CANCELLED and ERROR will not be clamped at
   * all. The amount of noise will be proportional to maxContributionsPerGroup *
   * maxGroupsContributed = 100 * maxGroupsContributed.
   *
   * This is not the case with the second approach. The vector will be clamped with the same factor
   * across all dimensions which will keep the results per response type proportional. Let's say you
   * set the L2 maxVectorNorm to 100.0 then the vector [150, 80, 10] will be scaled down to L2 norm
   * of 100.0, i.e. approximately to [88, 47, 5.8] (v * 100 / ||v||). The amount of noise will be
   * proportional to L2_norm * sqrt(maxGroupsContributed) = 100 * sqrt(maxGroupsContributed) which
   * is better (i.e. less noise) than in the first approach.
   *
   * If you are going to calculate percentage of OK, CANCELLED and ERROR responses then you should
   * use the second approach, otherwise you will get biased incorrect results.
   *
   * @param vectorExtractor a function to extract from the input row the double values of the vector
   *   to aggregate. It must be serializable. In Java, this means it can't be a Java lambda, method
   *   referencs or anonymous class because they capture `this` and therefore are not serializable.
   *   A workaround is to create a static nested class or a standalone class.
   * @param vectorSize the size of the vectors that will be aggregated.
   * @param valuesAggregations the aggregations to perform on the vector.
   * @param contributionBounds contribution bounds for the vector, see [ContributionBounds] for
   *   details on which bounds to specify in which cases.
   */
  // TODO: make it public once it is covered by tests.
  internal fun aggregateVector(
    vectorExtractor: (DataRowT) -> List<Double>,
    vectorSize: Int,
    vectorAggregations: VectorAggregationsBuilder,
    vectorContributionBounds: VectorContributionBounds,
  ): AggregationBuilder<DataRowT, ReturnT> {
    aggregations.add(
      VectorAggregations(
        vectorExtractor,
        vectorSize,
        vectorAggregations.aggregations,
        vectorContributionBounds,
      )
    )
    return this
  }

  /**
   * Instructs the query to aggregate a vector in each group.
   *
   * @param vectorColumnNames columns that store the values to aggregate. A column can be either a
   *   single double value or a list of double values. The columns will be joined into a single
   *   vector where lists will be flattened, e.g. columns with values (1.0, [-1.0, 2.0], 4.0) will
   *   result in a vector [1.0, -1.0, 2.0, 4.0].
   * @param vectorSize the size of the vectors that will be aggregated.
   * @param vectorAggregations the aggregations to perform on the vector.
   * @param vectorContributionBounds contribution bounds for the vector, see
   *   [VectorContributionBounds] for details on which bounds to specify in which cases.
   */
  // TODO: make it public once it is covered by tests.
  open internal fun aggregateVector(
    vectorColumnNames: ColumnNames,
    vectorSize: Int,
    vectorAggregations: VectorAggregationsBuilder,
    vectorContributionBounds: VectorContributionBounds,
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

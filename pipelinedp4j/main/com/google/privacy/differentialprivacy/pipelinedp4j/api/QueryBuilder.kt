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

import org.apache.beam.sdk.values.PCollection as BeamPCollection

import org.apache.spark.sql.Dataset

/**
 * A builder for differentially-private queries.
 *
 * To create a builder use [QueryBuilder.from] method. Then you should call [QueryBuilder.groupBy]
 * to group the data by keys, and then call one or more aggregation functions to specify the
 * differentially-private aggregations to perform on the data. Once you have specified all the
 * aggregations you should call [QueryBuilder.build] to get the instance of a query that you can
 * run.
 *
 * @param T the type of the elements in the collection.
 * @param R the type of the result.
 */
sealed class QueryBuilder<T, R>
protected constructor(
  protected val data: PipelineDpCollection<T>,
  protected val privacyIdExtractor: ((T) -> String),
) {
  protected lateinit var groupKeyExtractor: (T) -> String
  protected var maxGroupsContributed: Int = -1
  protected var maxContributionsPerGroup: Int = -1
  protected var publicGroups: PipelineDpCollection<String>? = null
  protected val aggregations = mutableListOf<AggregationSpec<T>>()

  protected fun groupBy(
    groupKeyExtractor: (T) -> String,
    maxGroupsContributed: Int,
    maxContributionsPerGroup: Int,
    publicGroups: PipelineDpCollection<String>? = null,
  ): QueryBuilder<T, R> {
    this.groupKeyExtractor = groupKeyExtractor
    this.maxGroupsContributed = maxGroupsContributed
    this.maxContributionsPerGroup = maxContributionsPerGroup
    require(
      publicGroups == null ||
        publicGroups is LocalPipelineDpCollection ||
        publicGroups::class == data::class
    ) {
      "Public keys must be either stored in a Sequence object or in the collection of the same type as the data is stored."
    }
    this.publicGroups = publicGroups
    return this
  }

  /**
   * Schedule an aggregation to count distinct privacy units.
   *
   * @param outputColumnName if output is dataframe then it is the name of the column to write the
   *   result to, if collection then it is the name of the output field.
   * @param budget the budget to use for the aggregation.
   */
  @JvmOverloads
  fun countDistinctPrivacyUnits(
    outputColumnName: String,
    budget: BudgetPerOpSpec? = null,
  ): QueryBuilder<T, R> {
    aggregations.add(
      AggregationSpec.PrivacyIdCount(outputColumnName, budget?.toInternalBudgetPerOpSpec())
    )
    return this
  }

  /**
   * Schedule a count aggregation.
   *
   * @param outputColumnName if output is dataframe then it is the name of the column to write the
   *   result to, if collection then it is the name of the output field.
   * @param budget the budget to use for the aggregation.
   */
  @JvmOverloads
  fun count(outputColumnName: String, budget: BudgetPerOpSpec? = null): QueryBuilder<T, R> {
    aggregations.add(AggregationSpec.Count(outputColumnName, budget?.toInternalBudgetPerOpSpec()))
    return this
  }

  /**
   * Schedule a sum aggregation.
   *
   * @param valueExtractor a function to extract the aggregated value from the input.
   * @param minTotalValuePerPrivacyUnitInGroup minimum across all groups of the sums of the privacy
   *   unit contributions to a group. Don't specify it if you also caclulate either MEAN or VARIANCE
   *   because in this case this value is not used.
   * @param maxTotalValuePerPrivacyUnitInGroup the maximum value of the same sum. Don't specify it
   *   if you also caclulate either MEAN or VARIANCE because in this case this value is not used.
   * @param outputColumnName if output is dataframe then it is the name of the column to write the
   *   result to, if collection then it is the name of the output field.
   * @param budget the budget to use for the aggregation.
   */
  @JvmOverloads
  fun sum(
    valueExtractor: (T) -> Double,
    minTotalValuePerPrivacyUnitInGroup: Double? = null,
    maxTotalValuePerPrivacyUnitInGroup: Double? = null,
    outputColumnName: String,
    budget: BudgetPerOpSpec? = null,
  ): QueryBuilder<T, R> {
    aggregations.add(
      AggregationSpec.Sum(
        outputColumnName,
        budget?.toInternalBudgetPerOpSpec(),
        valueExtractor,
        minTotalValuePerPrivacyUnitInGroup,
        maxTotalValuePerPrivacyUnitInGroup,
      )
    )
    return this
  }

  /**
   * Schedule a mean aggregation.
   *
   * @param valueExtractor a function to extract the aggregated value from the input.
   * @param minValue the minimum value that a privacy unit can contribute.
   * @param maxValue the maximum value that a privacy unit can contribute.
   * @param outputColumnName if output is dataframe then it is the name of the column to write the
   *   result to, if collection then it is the name of the output field.
   * @param budget the budget to use for the aggregation.
   */
  @JvmOverloads
  fun mean(
    valueExtractor: (T) -> Double,
    minValue: Double,
    maxValue: Double,
    outputColumnName: String,
    budget: BudgetPerOpSpec? = null,
  ): QueryBuilder<T, R> {
    aggregations.add(
      AggregationSpec.Mean(
        outputColumnName,
        budget?.toInternalBudgetPerOpSpec(),
        valueExtractor,
        minValue,
        maxValue,
      )
    )
    return this
  }

  /**
   * Schedule a variance aggregation.
   *
   * @param valueExtractor a function to extract the aggregated value from the input.
   * @param minValue the minimum value that a privacy unit can contribute.
   * @param maxValue the maximum value that a privacy unit can contribute.
   * @param outputColumnName if output is dataframe then it is the name of the column to write the
   *   result to, if collection then it is the name of the output field.
   * @param budget the budget to use for the aggregation.
   */
  @JvmOverloads
  fun variance(
    valueExtractor: (T) -> Double,
    minValue: Double,
    maxValue: Double,
    outputColumnName: String,
    budget: BudgetPerOpSpec? = null,
  ): QueryBuilder<T, R> {
    aggregations.add(
      AggregationSpec.Variance(
        outputColumnName,
        budget?.toInternalBudgetPerOpSpec(),
        valueExtractor,
        minValue,
        maxValue,
      )
    )
    return this
  }

  /**
   * Schedule a quantiles aggregation.
   *
   * @param valueExtractor a function to extract the aggregated value from the input.
   * @param ranks the ranks of the quantiles to compute.
   * @param minValue the minimum value that a privacy unit can contribute.
   * @param maxValue the maximum value that a privacy unit can contribute.
   * @param outputColumnName if output is dataframe then it is the name of the column to write the
   *   result to. If collection then there will be multiple output fields, one per rank, with names
   *   "outputColumnName_rank" where rank is the rank of the quantile.
   * @param budget the budget to use for the aggregation.
   */
  @JvmOverloads
  fun quantiles(
    valueExtractor: (T) -> Double,
    ranks: List<Double>,
    minValue: Double,
    maxValue: Double,
    outputColumnName: String,
    budget: BudgetPerOpSpec? = null,
  ): QueryBuilder<T, R> {
    aggregations.add(
      AggregationSpec.Quantiles(
        outputColumnName,
        budget?.toInternalBudgetPerOpSpec(),
        valueExtractor,
        ranks,
        minValue,
        maxValue,
      )
    )
    return this
  }

  abstract fun build(): Query<T, R>

  companion object {
    @JvmStatic
    fun <T> from(data: BeamPCollection<T>, privacyIdExtractor: (T) -> String) =
      BeamQueryBuilder<T>(data, privacyIdExtractor)

    @JvmStatic
    fun <T> from(data: Dataset<T>, privacyIdExtractor: (T) -> String) =
      SparkQueryBuilder<T>(data, privacyIdExtractor)
  }
}

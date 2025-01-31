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

import com.google.privacy.differentialprivacy.pipelinedp4j.core.Encoder
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType
import com.google.privacy.differentialprivacy.pipelinedp4j.local.LocalCollection as LocalFrameworkCollection
import com.google.privacy.differentialprivacy.pipelinedp4j.local.LocalEncoderFactory
import com.google.privacy.differentialprivacy.pipelinedp4j.local.LocalTable as LocalFrameworkTable
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.DpAggregates

/**
 * A builder for a query to run locally using JVM collections like [Sequence], [List], etc.
 *
 * This API opertates on row level, not column level. It means the data is represented as a
 * collection of rows and that the builder expects functions that know how to extract a certain
 * thing (e.g. privacy unit) from each row.
 *
 * The query building process is similar to SQL. It consists of three steps:
 * 1. FROM (`from` function): The data to run the query on
 * 2. GROUP BY (`groupBy` function): The columns to group the data by
 * 3. Aggregate (optional): the aggregations to calculate on the grouped data
 *
 * If only steps 1 and 2 are specified, the query will return the groups. If step 3 is also
 * specified, the query will also return the aggregated data along with the groups.
 *
 * Here is an example of a query that counts the number of viewers, number of views and average
 * rating of each movie:
 * ```kotlin
 * // `MoviewView` is a data class that has `userId` (string), `movieId` (string) and `rating` (double) fields.
 * data: List<MovieView> = ...
 * val query =
 *   LocalQueryBuilder.from(
 *       data,
 *       { it.userId },
 *       ContributionBoundingLevel.DATASET_LEVEL(
 *         maxGroupsContributed = 3,
 *         maxContributionsPerGroup = 1,
 *       ),
 *     )
 *     .groupBy({ it.movieId }, GroupsType.PrivateGroups())
 *     .countDistinctPrivacyUnits(outputColumnName = "numberOfViewers")
 *     .count(outputColumnName = "numberOfViews")
 *     .aggregateValue(
 *       { it.rating },
 *       ValueAggregationsBuilder().mean(outputColumnName = "averageOfRatings"),
 *       ContributionBounds(valueBounds = Bounds(minValue = 1.0, maxValue = 5.0)),
 *     )
 *     .build(TotalBudget(epsilon = 1.1, delta = 1e-10), NoiseKind.LAPLACE)
 * val result: Sequence<QueryPerGroupResult<String>> = query.run()
 * ```
 *
 * The query will return a [Sequence] of [QueryPerGroupResult]s, one for each group (in our example
 * each movie). Each [QueryPerGroupResult] will contain the group key (in our example the movie id)
 * and the differentially private aggregated values. The aggregated values are represented as a map
 * from the output column name to the aggregated value represented as [Double]. In our example the
 * aggregated values will be `numberOfViewers`, `numberOfViews` and `averageOfRatings`, i.e. the
 * mapping will look like this:
 * ```kotlin
 * mapOf(
 *   "numberOfViewers" to 123.76,
 *   "numberOfViews" to 456.2,
 *   "averageOfRatings" to 3.23,
 * )
 * ```
 *
 * If only group selection is used (i.e. no aggregations, only steps 1 and 2) then the query will
 * return a [Sequence] of [QueryPerGroupResult]s with empty maps in [QueryPerGroupResult] that can
 * be ignored.
 */
object LocalQueryBuilder {
  /**
   * Specification of the differentially private FROM clause of the query.
   *
   * The FROM clause specifies the data to run the query on, the privacy unit to protect and
   * contribution bounding level to apply.
   *
   * @param DataRowT the type of the data row.
   * @param PrivacyUnitT the type of the privacy unit.
   * @param data the [Sequence] containing the data to run the query on.
   * @param privacyUnitExtractor a function that extracts the privacy unit from each data row.
   * @param privacyUnitTKlass the Java class of the privacy unit type, necessary for encoding the
   *   privacy units.
   * @param contributionBoundingLevel the contribution bounding level to use.
   */
  @JvmStatic
  fun <DataRowT : Any, PrivacyUnitT : Any> from(
    data: Sequence<DataRowT>,
    privacyUnitExtractor: (DataRowT) -> PrivacyUnitT,
    contributionBoundingLevel: ContributionBoundingLevel,
  ): LocalGroupByBuilder<DataRowT, PrivacyUnitT> {
    val encoder: Encoder<PrivacyUnitT> = LocalEncoderFactory().encoderForArbitraryType()
    return LocalGroupByBuilder<DataRowT, PrivacyUnitT>(
      data,
      privacyUnitExtractor,
      encoder,
      contributionBoundingLevel,
    )
  }

  /**
   * Specification of the differentially private FROM clause of the query when the input is
   * iterable.
   *
   * @see [LocalQueryBuilder.from] for detailed documentation.
   */
  @JvmStatic
  fun <DataRowT : Any, PrivacyUnitT : Any> from(
    data: Iterable<DataRowT>,
    privacyUnitExtractor: (DataRowT) -> PrivacyUnitT,
    contributionBoundingLevel: ContributionBoundingLevel,
  ) = from(data.asSequence(), privacyUnitExtractor, contributionBoundingLevel)
}

/** Specification of the differentially private GROUP BY clause of the query. */
class LocalGroupByBuilder<DataRowT : Any, PrivacyUnitT : Any>
internal constructor(
  private val data: Sequence<DataRowT>,
  private val privacyUnitExtractor: (DataRowT) -> PrivacyUnitT,
  private val privacyUnitEncoder: Encoder<PrivacyUnitT>,
  private val contributionBoundingLevel: ContributionBoundingLevel,
) {
  /**
   * Specification of the differentially private GROUP BY clause of the query.
   *
   * The GROUP BY clause specifies how the data should be grouped in a differentially private way.
   *
   * @param groupKeyExtractor a function that extracts the group key from each data row.
   * @param groupsType either private or public, see [GroupsType] for more details.
   * @param groupByAdditionalParameters additional parameters for the GROUP BY clause, see
   *   [GroupByAdditionalParameters] for more details.
   */
  @JvmOverloads
  fun <GroupKeysT : Any> groupBy(
    groupKeyExtractor: (DataRowT) -> GroupKeysT,
    groupsType: GroupsType,
    groupByAdditionalParameters: GroupByAdditionalParameters = GroupByAdditionalParameters(),
  ): LocalAggregationBuilder<DataRowT, PrivacyUnitT, GroupKeysT> {
    return LocalAggregationBuilder<DataRowT, PrivacyUnitT, GroupKeysT>(
      data,
      privacyUnitExtractor,
      privacyUnitEncoder,
      contributionBoundingLevel,
      groupKeyExtractor,
      LocalEncoderFactory().encoderForArbitraryType<GroupKeysT>(),
      groupsType,
      groupByAdditionalParameters,
    )
  }
}

/**
 * Specification of the differentially private aggregations of the query.
 *
 * See [AggregationBuilder] for detailed documentation.
 */
class LocalAggregationBuilder<DataRowT : Any, PrivacyUnitT : Any, GroupKeysT : Any>
internal constructor(
  private val data: Sequence<DataRowT>,
  private val privacyUnitExtractor: (DataRowT) -> PrivacyUnitT,
  private val privacyUnitEncoder: Encoder<PrivacyUnitT>,
  private val contributionBoundingLevel: ContributionBoundingLevel,
  private val groupKeyExtractor: (DataRowT) -> GroupKeysT,
  private val groupKeyEncoder: Encoder<GroupKeysT>,
  private val groupsType: GroupsType,
  private val groupByAdditionalParameters: GroupByAdditionalParameters,
) : AggregationBuilder<DataRowT, Sequence<QueryPerGroupResult<GroupKeysT>>>() {
  override fun build(
    totalBudget: TotalBudget,
    noiseKind: NoiseKind?,
  ): LocalQuery<DataRowT, PrivacyUnitT, GroupKeysT> {
    return LocalQuery(
      data,
      privacyUnitExtractor,
      privacyUnitEncoder,
      contributionBoundingLevel,
      groupKeyExtractor,
      groupKeyEncoder,
      groupsType,
      groupByAdditionalParameters,
      aggregations,
      totalBudget,
      noiseKind,
    )
  }
}

class LocalQuery<DataRowT : Any, PrivacyUnitT : Any, GroupKeysT : Any>
internal constructor(
  data: Sequence<DataRowT>,
  privacyUnitExtractor: (DataRowT) -> PrivacyUnitT,
  privacyUnitEncoder: Encoder<PrivacyUnitT>,
  contributionBoundingLevel: ContributionBoundingLevel,
  groupKeyExtractor: (DataRowT) -> GroupKeysT,
  groupKeyEncoder: Encoder<GroupKeysT>,
  groupsType: GroupsType,
  groupByAdditionalParameters: GroupByAdditionalParameters,
  aggregations: List<AggregationSpec>,
  totalBudget: TotalBudget,
  noiseKind: NoiseKind?,
) :
  BaseQueryImpl<DataRowT, PrivacyUnitT, GroupKeysT, Sequence<QueryPerGroupResult<GroupKeysT>>>(
    LocalFrameworkCollection(data),
    LocalEncoderFactory(),
    privacyUnitExtractor,
    privacyUnitEncoder,
    contributionBoundingLevel,
    groupKeyExtractor,
    groupKeyEncoder,
    groupsType,
    groupByAdditionalParameters,
    aggregations,
    totalBudget,
    noiseKind,
  ) {
  override fun run(testMode: TestMode): Sequence<QueryPerGroupResult<GroupKeysT>> {
    val localResult =
      (runWithDpEngine(testMode) as LocalFrameworkTable<GroupKeysT, DpAggregates>).data
    val mapToResultFn =
      createConvertDpAggregatesToQueryPerGroupResultFn(
        aggregations.outputColumnNamesWithMetricTypes()
      )
    return localResult.map(mapToResultFn)
  }

  private fun createConvertDpAggregatesToQueryPerGroupResultFn(
    outputColumnNamesWithMetricTypes: List<Pair<String, MetricType>>
  ): (Pair<GroupKeysT, DpAggregates>) -> QueryPerGroupResult<GroupKeysT> {
    return { perGroupAggregates: Pair<GroupKeysT, DpAggregates> ->
      QueryPerGroupResult.create(
        groupKey = perGroupAggregates.first,
        dpAggregates = perGroupAggregates.second,
        outputColumnNamesWithMetricTypes,
      )
    }
  }
}

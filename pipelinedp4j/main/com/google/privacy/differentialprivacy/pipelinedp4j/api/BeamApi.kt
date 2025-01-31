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

import com.google.privacy.differentialprivacy.pipelinedp4j.beam.BeamCollection as BeamFrameworkCollection
import com.google.privacy.differentialprivacy.pipelinedp4j.beam.BeamEncoder
import com.google.privacy.differentialprivacy.pipelinedp4j.beam.BeamEncoderFactory
import com.google.privacy.differentialprivacy.pipelinedp4j.beam.BeamTable as BeamFrameworkTable
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.DpAggregates
import java.io.InputStream
import java.io.OutputStream
import org.apache.beam.sdk.coders.Coder
import org.apache.beam.sdk.coders.CustomCoder
import org.apache.beam.sdk.coders.DoubleCoder
import org.apache.beam.sdk.coders.MapCoder
import org.apache.beam.sdk.coders.StringUtf8Coder
import org.apache.beam.sdk.transforms.MapElements
import org.apache.beam.sdk.transforms.SerializableFunction
import org.apache.beam.sdk.values.KV
import org.apache.beam.sdk.values.PCollection as BeamPCollection

/**
 * A builder for a query to run on Beam.
 *
 * Beam queries opertate on row level, not column level. It means the data is represented as
 * distributed collection of rows and that the builder expects functions that know how to extract a
 * certain thing (e.g. privacy unit) from each row.
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
 * data: PCollection<MovieView> = ...
 * val query =
 *   BeamQueryBuilder.from(
 *       data,
 *       StringExtractor { it.userId },
 *       ContributionBoundingLevel.DATASET_LEVEL(
 *         maxGroupsContributed = 3,
 *         maxContributionsPerGroup = 1,
 *       ),
 *     )
 *     .groupBy(StringExtractor { it.movieId }, GroupsType.PrivateGroups())
 *     .countDistinctPrivacyUnits(outputColumnName = "numberOfViewers")
 *     .count(outputColumnName = "numberOfViews")
 *     .aggregateValue(
 *       { it.rating },
 *       ValueAggregationsBuilder().mean(outputColumnName = "averageOfRatings"),
 *       ContributionBounds(valueBounds = Bounds(minValue = 1.0, maxValue = 5.0)),
 *     )
 *     .build(TotalBudget(epsilon = 1.1, delta = 1e-10), NoiseKind.LAPLACE)
 * val result: PCollection<QueryPerGroupResult<String>> = query.run()
 * ```
 *
 * The query will return a [BeamPCollection] of [QueryPerGroupResult]s, one for each group (in our
 * example each movie). Each [QueryPerGroupResult] will contain the group key (in our example the
 * movie id) and the differentially private aggregated values. The aggregated values are represented
 * as a map from the output column name to the aggregated value represented as [Double]. In our
 * example the aggregated values will be `numberOfViewers`, `numberOfViews` and `averageOfRatings`,
 * i.e. the mapping will look like this:
 * ```kotlin
 * mapOf(
 *   "numberOfViewers" to 123.76,
 *   "numberOfViews" to 456.2,
 *   "averageOfRatings" to 3.23,
 * )
 * ```
 *
 * If only group selection is used (i.e. no aggregations, only steps 1 and 2) then the query will
 * return a [BeamPCollection] of [QueryPerGroupResult]s with empty maps in [QueryPerGroupResult]
 * that can be ignored.
 */
object BeamQueryBuilder {
  /**
   * Specification of the differentially private FROM clause of the query.
   *
   * The FROM clause specifies the data to run the query on, the privacy unit to protect and
   * contribution bounding level to apply.
   *
   * @param DataRowT the type of the data row.
   * @param PrivacyUnitT the type of the privacy unit.
   * @param data the [BeamPCollection] containing the data to run the query on.
   * @param privacyUnitExtractor a function that extracts the privacy unit from each data row.
   * @param privacyUnitTKlass the Java class of the privacy unit type, necessary for encoding the
   *   privacy units.
   * @param contributionBoundingLevel the contribution bounding level to use.
   */
  @JvmStatic
  fun <DataRowT : Any, PrivacyUnitT : Any> from(
    data: BeamPCollection<DataRowT>,
    privacyUnitExtractor: (DataRowT) -> PrivacyUnitT,
    privacyUnitTKlass: Class<PrivacyUnitT>,
    contributionBoundingLevel: ContributionBoundingLevel,
  ): BeamGroupByBuilder<DataRowT, PrivacyUnitT> {
    val encoderFactory = BeamEncoderFactory()
    @Suppress("UNCHECKED_CAST")
    val encoder: BeamEncoder<PrivacyUnitT> =
      encoderFactory.recordsOfUnknownClass(privacyUnitTKlass) as BeamEncoder<PrivacyUnitT>
    return BeamGroupByBuilder<DataRowT, PrivacyUnitT>(
      data,
      privacyUnitExtractor,
      encoder,
      contributionBoundingLevel,
    )
  }

  /**
   * Specification of the differentially private FROM clause of the query when the privacy unit type
   * is a string.
   *
   * To provide a privacy unit extractor, you can just pass a lambda `(DataRowT) -> String`.
   * [StringExtractor] is essentially just a wrapper for `(DataRowT) -> String` function to avoid
   * conflicting function overloads.
   *
   * @see [BeamQueryBuilder.from] for detailed documentation.
   */
  @JvmStatic
  fun <DataRowT : Any> from(
    data: BeamPCollection<DataRowT>,
    privacyUnitExtractor: StringExtractor<DataRowT>,
    contributionBoundingLevel: ContributionBoundingLevel,
  ) = from(data, privacyUnitExtractor, String::class.java, contributionBoundingLevel)

  /**
   * Specification of the differentially private FROM clause of the query when the privacy unit type
   * is an integer.
   *
   * To provide a privacy unit extractor, you can just pass a lambda `(DataRowT) -> Int`.
   * [IntExtractor] is essentially just a wrapper for `(DataRowT) -> Int` function to avoid
   * conflicting function overloads.
   *
   * @see [BeamQueryBuilder.from] for detailed documentation.
   */
  @JvmStatic
  fun <DataRowT : Any> from(
    data: BeamPCollection<DataRowT>,
    privacyUnitExtractor: IntExtractor<DataRowT>,
    contributionBoundingLevel: ContributionBoundingLevel,
  ) = from(data, privacyUnitExtractor, Int::class.java, contributionBoundingLevel)

  /**
   * Specification of the differentially private FROM clause of the query when the privacy unit type
   * is a long integer.
   *
   * To provide a privacy unit extractor, you can just pass a lambda `(DataRowT) -> Long`.
   * [LongExtractor] is essentially just a wrapper for `(DataRowT) -> Long` function to avoid
   * conflicting function overloads.
   *
   * @see [BeamQueryBuilder.from] for detailed documentation.
   */
  @JvmStatic
  fun <DataRowT : Any> from(
    data: BeamPCollection<DataRowT>,
    privacyUnitExtractor: LongExtractor<DataRowT>,
    contributionBoundingLevel: ContributionBoundingLevel,
  ) = from(data, privacyUnitExtractor, Long::class.java, contributionBoundingLevel)
}

/** Specification of the differentially private GROUP BY clause of the query. */
class BeamGroupByBuilder<DataRowT : Any, PrivacyUnitT : Any>
internal constructor(
  private val data: BeamPCollection<DataRowT>,
  private val privacyUnitExtractor: (DataRowT) -> PrivacyUnitT,
  private val privacyUnitEncoder: BeamEncoder<PrivacyUnitT>,
  private val contributionBoundingLevel: ContributionBoundingLevel,
) {
  /**
   * Specification of the differentially private GROUP BY clause of the query.
   *
   * The GROUP BY clause specifies how the data should be grouped in a differentially private way.
   *
   * @param groupKeyExtractor a function that extracts the group key from each data row.
   * @param groupKeysTKlass the Java class of the group key type, necessary for encoding the group
   *   keys.
   * @param groupsType either private or public, see [GroupsType] for more details.
   * @param groupByAdditionalParameters additional parameters for the GROUP BY clause, see
   *   [GroupByAdditionalParameters] for more details.
   */
  @JvmOverloads
  fun <GroupKeysT : Any> groupBy(
    groupKeyExtractor: (DataRowT) -> GroupKeysT,
    groupKeysTKlass: Class<GroupKeysT>,
    groupsType: GroupsType,
    groupByAdditionalParameters: GroupByAdditionalParameters = GroupByAdditionalParameters(),
  ): BeamAggregationBuilder<DataRowT, PrivacyUnitT, GroupKeysT> {
    @Suppress("UNCHECKED_CAST")
    return BeamAggregationBuilder<DataRowT, PrivacyUnitT, GroupKeysT>(
      data,
      privacyUnitExtractor,
      privacyUnitEncoder,
      contributionBoundingLevel,
      groupKeyExtractor,
      BeamEncoderFactory().recordsOfUnknownClass(groupKeysTKlass) as BeamEncoder<GroupKeysT>,
      groupsType,
      groupByAdditionalParameters,
    )
  }

  /**
   * Specification of the differentially private GROUP BY clause of the query when group keys type
   * is a string.
   *
   * To provide a group key extractor, you can just pass a lambda `(DataRowT) -> String`.
   * [StringExtractor] is essentially just a wrapper for `(DataRowT) -> String` function to avoid
   * conflicting function overloads.
   *
   * @see [BeamGroupByBuilder.groupBy] for detailed documentation.
   */
  @JvmOverloads
  fun groupBy(
    groupKeyExtractor: StringExtractor<DataRowT>,
    groupsType: GroupsType,
    groupByAdditionalParameters: GroupByAdditionalParameters = GroupByAdditionalParameters(),
  ) = groupBy(groupKeyExtractor, String::class.java, groupsType, groupByAdditionalParameters)

  /**
   * Specification of the differentially private GROUP BY clause of the query when group keys type
   * is an integer.
   *
   * To provide a group key extractor, you can just pass a lambda `(DataRowT) -> Int`.
   * [IntExtractor] is essentially just a wrapper for `(DataRowT) -> Int` function to avoid
   * conflicting function overloads.
   *
   * @see [BeamGroupByBuilder.groupBy] for detailed documentation.
   */
  @JvmOverloads
  fun groupBy(
    groupKeyExtractor: IntExtractor<DataRowT>,
    groupsType: GroupsType,
    groupByAdditionalParameters: GroupByAdditionalParameters = GroupByAdditionalParameters(),
  ) = groupBy(groupKeyExtractor, Int::class.java, groupsType, groupByAdditionalParameters)

  /**
   * Specification of the differentially private GROUP BY clause of the query when group keys type
   * is a long.
   *
   * To provide a group key extractor, you can just pass a lambda `(DataRowT) -> Long`.
   * [LongExtractor] is essentially just a wrapper for `(DataRowT) -> Long` function to avoid
   * conflicting function overloads.
   *
   * @see [BeamGroupByBuilder.groupBy] for detailed documentation.
   */
  @JvmOverloads
  fun groupBy(
    groupKeyExtractor: LongExtractor<DataRowT>,
    groupsType: GroupsType,
    groupByAdditionalParameters: GroupByAdditionalParameters = GroupByAdditionalParameters(),
  ) = groupBy(groupKeyExtractor, Long::class.java, groupsType, groupByAdditionalParameters)
}

/**
 * Specification of the differentially private aggregations of the query.
 *
 * See [AggregationBuilder] for detailed documentation.
 */
class BeamAggregationBuilder<DataRowT : Any, PrivacyUnitT : Any, GroupKeysT : Any>
internal constructor(
  private val data: BeamPCollection<DataRowT>,
  private val privacyUnitExtractor: (DataRowT) -> PrivacyUnitT,
  private val privacyUnitEncoder: BeamEncoder<PrivacyUnitT>,
  private val contributionBoundingLevel: ContributionBoundingLevel,
  private val groupKeyExtractor: (DataRowT) -> GroupKeysT,
  private val groupKeyEncoder: BeamEncoder<GroupKeysT>,
  private val groupsType: GroupsType,
  private val groupByAdditionalParameters: GroupByAdditionalParameters,
) : AggregationBuilder<DataRowT, BeamPCollection<QueryPerGroupResult<GroupKeysT>>>() {
  override fun build(
    totalBudget: TotalBudget,
    noiseKind: NoiseKind?,
  ): BeamQuery<DataRowT, PrivacyUnitT, GroupKeysT> {
    return BeamQuery(
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

class BeamQuery<DataRowT : Any, PrivacyUnitT : Any, GroupKeysT : Any>
internal constructor(
  data: BeamPCollection<DataRowT>,
  privacyUnitExtractor: (DataRowT) -> PrivacyUnitT,
  privacyUnitEncoder: BeamEncoder<PrivacyUnitT>,
  contributionBoundingLevel: ContributionBoundingLevel,
  groupKeyExtractor: (DataRowT) -> GroupKeysT,
  private val groupKeyEncoder: BeamEncoder<GroupKeysT>,
  groupsType: GroupsType,
  groupByAdditionalParameters: GroupByAdditionalParameters,
  aggregations: List<AggregationSpec>,
  totalBudget: TotalBudget,
  noiseKind: NoiseKind?,
) :
  BaseQueryImpl<
    DataRowT,
    PrivacyUnitT,
    GroupKeysT,
    BeamPCollection<QueryPerGroupResult<GroupKeysT>>,
  >(
    BeamFrameworkCollection(data),
    BeamEncoderFactory(),
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
  override fun run(testMode: TestMode): BeamPCollection<QueryPerGroupResult<GroupKeysT>> {
    val beamResult =
      (runWithDpEngine(testMode) as BeamFrameworkTable<GroupKeysT, DpAggregates>).data
    val coder = QueryPerGroupResultCoder(groupKeyEncoder.coder)
    val mapToResultFn =
      createConvertDpAggregatesToQueryPerGroupResultFn(
        aggregations.outputColumnNamesWithMetricTypes()
      )
    return beamResult
      .apply(MapElements.into(coder.encodedTypeDescriptor).via(SerializableFunction(mapToResultFn)))
      .setCoder(coder)
  }

  private fun createConvertDpAggregatesToQueryPerGroupResultFn(
    outputColumnNamesWithMetricTypes: List<Pair<String, MetricType>>
  ): (KV<GroupKeysT, DpAggregates>) -> QueryPerGroupResult<GroupKeysT> {
    return { perGroupAggregates: KV<GroupKeysT, DpAggregates> ->
      QueryPerGroupResult.create(
        groupKey = perGroupAggregates.key,
        dpAggregates = perGroupAggregates.value,
        outputColumnNamesWithMetricTypes,
      )
    }
  }
}

internal class QueryPerGroupResultCoder<GroupKeysT : Any>(
  private val groupKeyCoder: Coder<GroupKeysT>
) : CustomCoder<QueryPerGroupResult<GroupKeysT>>() {

  override fun encode(value: QueryPerGroupResult<GroupKeysT>, outStream: OutputStream) {
    groupKeyCoder.encode(value.groupKey, outStream)
    MapCoder.of(StringUtf8Coder.of(), DoubleCoder.of()).encode(value.aggregationResults, outStream)
  }

  override fun decode(inStream: InputStream): QueryPerGroupResult<GroupKeysT> {
    val groupKey = groupKeyCoder.decode(inStream)
    val aggregationResults = MapCoder.of(StringUtf8Coder.of(), DoubleCoder.of()).decode(inStream)
    @Suppress("UNCHECKED_CAST")
    return QueryPerGroupResult(groupKey, aggregationResults as Map<String, Double>)
  }
}

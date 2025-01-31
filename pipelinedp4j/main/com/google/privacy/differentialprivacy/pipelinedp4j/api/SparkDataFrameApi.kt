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

import org.apache.spark.api.java.function.MapFunction
import org.apache.spark.sql.Dataset as SparkDataset
import org.apache.spark.sql.Encoders as SparkEncoders
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.Row as SparkRow
import org.apache.spark.sql.RowFactory as SparkRowFactory
import org.apache.spark.sql.types.DataTypes as SparkDataTypes
import org.apache.spark.sql.types.Metadata as SparkMetadata
import org.apache.spark.sql.types.StructField as SparkStructField
import org.apache.spark.sql.types.StructType
import scala.collection.JavaConverters

typealias SparkDataFrame = SparkDataset<SparkRow>

/**
 * A builder for a query to run on Spark using DataFrames.
 *
 * Spark DataFrame queries operate on column level, not row level. It means the data is represented
 * as collection of columns and that the builder expects list of columns that represent a certain
 * thing (e.g. privacy unit).
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
 * // data is a DataFrame with columns `userId` (string), `movieId` (string) and `rating` (double).
 * data: SparkDataFrame = ...
 * val query =
 *   SparkDataFrameQueryBuilder.from(
 *       data,
 *       ColumnNames("userId"),
 *       ContributionBoundingLevel.DATASET_LEVEL(
 *         maxGroupsContributed = 3,
 *         maxContributionsPerGroup = 1,
 *       ),
 *     )
 *     .groupBy(ColumnNames("movieId"), GroupsType.PrivateGroups())
 *     .countDistinctPrivacyUnits(outputColumnName = "numberOfViewers")
 *     .count(outputColumnName = "numberOfViews")
 *     .aggregateValue(
 *       "rating",
 *       ValueAggregationsBuilder().mean(outputColumnName = "averageOfRatings"),
 *       ContributionBounds(valueBounds = Bounds(minValue = 1.0, maxValue = 5.0)),
 *     )
 *     .build(TotalBudget(epsilon = 1.1, delta = 1e-10), NoiseKind.LAPLACE)
 * val result: Dataset<Row> = query.run()
 * ```
 *
 * The query will return a [SparkDataFrame] (i.e. Dataset<Row>). The columns will be the following:
 * 1. group key columns (in our example `movieId`)
 * 2. aggregation columns (in our example `numberOfViewers`, `numberOfViews` and `averageOfRatings`)
 *
 * The dataframe.show() of our example will look like this:
 * ```
 * +--------+-----------------+---------------+--------------+
 * |movie_id|number_of_viewers|number_of_views|average_rating|
 * +--------+-----------------+---------------+--------------+
 * |    4567|          3023.75|        4500.89|           4.3|
 * |    7890|          1500.12|        3500.67|           3.8|
 * |    1234|          4500.50|        8700.33|           4.7|
 * |    6789|          2000.75|        7200.12|           4.1|
 * |    3456|          3900.80|        5600.45|           3.5|
 * +--------+-----------------+---------------+--------------+
 * ```
 *
 * If only group selection is used (i.e. no aggregations, only steps 1 and 2) then the query will
 * return a [SparkDataFrame] with only the group key columns (in our example with one column
 * `movieId`).
 */
object SparkDataFrameQueryBuilder {
  /**
   * Specification of the differentially private FROM clause of the query.
   *
   * The FROM clause specifies the data to run the query on, the privacy unit to protect and
   * contribution bounding level to apply.
   *
   * @param data the [SparkDataFrame] containing the data to run the query on.
   * @param privacyUnitColumnNames names of the columns that form a privacy unit.
   * @param contributionBoundingLevel the contribution bounding level to use.
   */
  @JvmStatic
  fun from(
    data: SparkDataFrame,
    privacyUnitColumnNames: ColumnNames,
    contributionBoundingLevel: ContributionBoundingLevel,
  ): SparkDataFrameGroupByBuilder {
    val privacyUnitExtractor: (SparkRow) -> List<Any?> =
      extractorFromColumns(privacyUnitColumnNames.names)
    val sparkGroupByBuilder =
      SparkQueryBuilder.from(
        data,
        privacyUnitExtractor,
        List::class.java,
        contributionBoundingLevel,
      )
    return SparkDataFrameGroupByBuilder(sparkGroupByBuilder, data.schema())
  }
}

/** Specification of the differentially private GROUP BY clause of the query. */
class SparkDataFrameGroupByBuilder
internal constructor(
  private val sparkGroupByBuilder: SparkGroupByBuilder<SparkRow, List<Any?>>,
  private val schema: StructType,
) {
  /**
   * Specification of the differentially private GROUP BY clause of the query.
   *
   * The GROUP BY clause specifies how the data should be grouped in a differentially private way.
   *
   * @param groupKeyColumnNames column names that form the group key.
   * @param groupsType either private or public, see [GroupsType] for more details.
   * @param groupByAdditionalParameters additional parameters for the GROUP BY clause, see
   *   [GroupByAdditionalParameters] for more details.
   */
  @JvmOverloads
  fun groupBy(
    groupKeyColumnNames: ColumnNames,
    groupsType: GroupsType,
    groupByAdditionalParameters: GroupByAdditionalParameters = GroupByAdditionalParameters(),
  ): SparkDataFrameAggregationBuilder {
    val groupKeyExtractor: (SparkRow) -> List<Any?> =
      extractorFromColumns(groupKeyColumnNames.names)
    val sparkAggregationBuilder =
      sparkGroupByBuilder.groupBy(
        groupKeyExtractor,
        List::class.java,
        groupsType,
        groupByAdditionalParameters,
      )
    return SparkDataFrameAggregationBuilder(
      sparkAggregationBuilder,
      schema.getColumnStructFields(groupKeyColumnNames.names),
    )
  }

  private fun StructType.getColumnStructFields(columnNames: List<String>): List<SparkStructField> {
    return columnNames.map { columnName ->
      fields().find { it.name() == columnName }
        ?: throw IllegalArgumentException("Column $columnName not found in the dataset schema.")
    }
  }
}

/**
 * Specification of the differentially private aggregations of the query.
 *
 * See [AggregationBuilder] for detailed documentation.
 */
class SparkDataFrameAggregationBuilder
internal constructor(
  private val sparkAggregationBuilder: SparkAggregationBuilder<SparkRow, List<Any?>, List<Any?>>,
  private val groupKeyColumnStructFields: List<SparkStructField>,
) : AggregationBuilder<SparkRow, SparkDataFrame>() {
  override fun aggregateValue(
    valueColumnName: String,
    valueAggregations: ValueAggregationsBuilder,
    contributionBounds: ContributionBounds,
  ): AggregationBuilder<SparkRow, SparkDataFrame> {
    val valueExtractor: (SparkRow) -> Double = { dataRow: SparkRow ->
      dataRow.getAs(valueColumnName)
    }
    aggregations.add(
      ValueAggregations(
        valueExtractor,
        valueAggregations.aggregations,
        contributionBounds,
        valueColumnName,
      )
    )
    return this
  }

  override fun build(totalBudget: TotalBudget, noiseKind: NoiseKind?): SparkDataFrameQuery {
    sparkAggregationBuilder.aggregations = aggregations
    val sparkQuery = sparkAggregationBuilder.build(totalBudget, noiseKind)
    return SparkDataFrameQuery(sparkQuery, groupKeyColumnStructFields)
  }
}

class SparkDataFrameQuery
internal constructor(
  private val sparkQuery: SparkQuery<SparkRow, List<Any?>, List<Any?>>,
  private val groupKeyColumnStructFields: List<SparkStructField>,
) : Query<SparkDataFrame> {
  override fun run(testMode: TestMode): SparkDataFrame {
    val sparkQueryResult: SparkDataset<QueryPerGroupResult<List<Any?>>> = sparkQuery.run(testMode)

    val rows =
      sparkQueryResult.map(
        MapFunction { record: QueryPerGroupResult<List<Any?>> ->
          val groupKeyValues = record.groupKey.toTypedArray()
          val aggregationValues = record.aggregationResults.values.toTypedArray()
          SparkRowFactory.create(*groupKeyValues, *aggregationValues)
        },
        SparkEncoders.kryo(SparkRow::class.java),
      )

    val aggregationColumnStructFields =
      QueryPerGroupResult.columnsNamesInAggregationResults(
          sparkQuery.aggregations.outputColumnNamesWithMetricTypes()
        )
        .map { key ->
          SparkStructField(
            key,
            SparkDataTypes.DoubleType,
            /* nullable= */ false,
            SparkMetadata.empty(),
          )
        }

    val schema =
      StructType((groupKeyColumnStructFields + aggregationColumnStructFields).toTypedArray())
    return rows.sparkSession().createDataFrame(rows.javaRDD(), schema)
  }
}

internal fun SparkDataFrame.toSparkDataset(): SparkDataset<List<Any?>> {
  return map(
    MapFunction { row -> ColumnValuesListImplementation(JavaConverters.asJava(row.toSeq())) },
    Encoders.kryo(List::class.java),
  )
}

/**
 * Returns a function that extracts values from specified columns of a row and returns them as a
 * list.
 *
 * The interface still returns List as it is read-only and in case we change the implementation, we
 * will need to change only [ColumnValuesListImplementation] and not all the interfaces down the
 * stream.
 */
private fun extractorFromColumns(columnNames: List<String>): (SparkRow) -> List<Any?> {
  return { dataRow: SparkRow ->
    columnNames.map { dataRow.getAs<Any?>(it) }.toCollection(ColumnValuesListImplementation())
  }
}

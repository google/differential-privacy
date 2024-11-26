package com.google.privacy.differentialprivacy.pipelinedp4j.api

import org.apache.spark.sql.Dataset

/**
 * A builder for queries on Spark.
 *
 * @param T the type of the elements in the collection.
 */
class SparkQueryBuilder<T>
internal constructor(data: Dataset<T>, privacyIdExtractor: (T) -> String) :
  QueryBuilder<T, Dataset<QueryPerGroupResult>>(
    SparkPipelineDpCollection(data),
    privacyIdExtractor = privacyIdExtractor,
  ) {
  /**
   * Groups the data by keys (corresponds to groupBy operation in SQL).
   *
   * @param groupKeyExtractor a function to extract the group key from the input.
   * @param maxGroupsContributed the maximum number of groups that a single privacy unit can
   *   contribute to.
   * @param maxContributionsPerGroup the maximum number of contributions that a single privacy unit
   *   can make to a single group.
   * @param publicGroups a collection of publicly known keys. Read more about public groups in the
   *   documentation to the library.
   */

  @JvmOverloads
  fun groupBy(
    groupKeyExtractor: (T) -> String,
    maxGroupsContributed: Int,
    maxContributionsPerGroup: Int,
    publicGroups: Dataset<String>? = null,
  ) =
    groupBy(
      groupKeyExtractor,
      maxGroupsContributed,
      maxContributionsPerGroup,
      publicGroups?.let { SparkPipelineDpCollection(it) },
    )

  override fun build() =
    SparkQuery<T>(
      data,
      privacyIdExtractor,
      groupKeyExtractor,
      maxGroupsContributed,
      maxContributionsPerGroup,
      publicGroups,
      aggregations,
    )
}

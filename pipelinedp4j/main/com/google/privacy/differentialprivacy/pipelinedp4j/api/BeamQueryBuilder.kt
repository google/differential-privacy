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

/**
 * A builder for queries on Beam.
 *
 * @param T the type of the elements in the collection.
 */
class BeamQueryBuilder<T>
internal constructor(data: BeamPCollection<T>, privacyIdExtractor: (T) -> String) :
  QueryBuilder<T, BeamPCollection<QueryPerGroupResult>>(
    BeamPipelineDpCollection(data),
    privacyIdExtractor = privacyIdExtractor,
  ) {
  /**
   * Groups the data by keys (corresnponds to groupBy operation in SQL).
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
    publicGroups: BeamPCollection<String>? = null,
  ) =
    groupBy(
      groupKeyExtractor,
      maxGroupsContributed,
      maxContributionsPerGroup,
      publicGroups?.let { BeamPipelineDpCollection(it) },
    )

  override fun build() =
    BeamQuery<T>(
      data,
      privacyIdExtractor,
      groupKeyExtractor,
      maxGroupsContributed,
      maxContributionsPerGroup,
      publicGroups,
      aggregations,
    )
}

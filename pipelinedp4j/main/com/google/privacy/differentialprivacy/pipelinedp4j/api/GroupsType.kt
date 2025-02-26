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

import com.google.privacy.differentialprivacy.pipelinedp4j.spark.SparkCollection
import org.apache.spark.sql.Dataset as SparkDataset
import com.google.privacy.differentialprivacy.pipelinedp4j.beam.BeamCollection
import com.google.privacy.differentialprivacy.pipelinedp4j.core.FrameworkCollection
import com.google.privacy.differentialprivacy.pipelinedp4j.local.LocalCollection
import org.apache.beam.sdk.values.PCollection as BeamPCollection

/** Type of the groups to use in the query. */
sealed interface GroupsType {
  /**
   * Private groups.
   *
   * It means that the groups are not publicly know and therefore the query perform differentially
   * private group selection. As the result of this selection the output will contain a subset of
   * the groups that were present in the input data because some of the groups might be dropped.
   *
   * @param budget the privacy budget that will be used for the differentially private group
   *   selection. If not specified, then the budget will be relative and have weight 1. See
   *   [RelativeBudgetPerOpSpec] for details on what weights mean.
   * @param minPrivacyUnitsPerGroup the minimum number of privacy units per group. If group will
   *   have less than that number of privacy units that contribued to it, it will be dropped.
   */
  data class PrivateGroups(
    internal val budget: BudgetPerOpSpec? = null,
    internal val minPrivacyUnitsPerGroup: Int = 1,
  ) : GroupsType {
    init {
      require(minPrivacyUnitsPerGroup > 0) { "minPrivacyUnitsPerGroup must be positive" }
    }
  }

  /**
   * Public groups.
   *
   * It means that the groups are publicly known and therefore the query does not perform
   * differentially private group selection. The groups in the output will be exactly the same as
   * the public groups. I.e. if the input data contains a group that is not in [publicGroups], that
   * group will be dropped. If the input data doesn't contain a group that is in [publicGroups], it
   * will be still added into the output and will have noised default (e.g. 0) values for
   * aggregations. If the input data contains a group that is in [publicGroups], it will be kept and
   * its aggregations will be anonymized.
   */
  @ConsistentCopyVisibility
  data class PublicGroups<GroupKeysT : Any>
  internal constructor(internal val publicGroups: FrameworkCollection<GroupKeysT>) : GroupsType {
    companion object {

      /**
       * Constructor for [PublicGroups] when [publicGroups] are provided as a [BeamPCollection].
       *
       * It can only be used in BeamApi.
       */
      @JvmStatic
      fun <GroupKeysT : Any> create(publicGroups: BeamPCollection<GroupKeysT>) =
        PublicGroups(BeamCollection(publicGroups))

      /**
       * Constructor for [PublicGroups] when [publicGroups] are provided as an [Iterable].
       *
       * @param GroupsTypeT the type of container where group keys are stored. Note that for
       *   DataFrames (e.g. Spark DataFrames), the type must be [ArrayList<Any?>] where this list
       *   must contain keys from each column, in the same order as the columns were specified in
       *   groupBy call.
       */
      @JvmStatic
      fun <GroupKeysT : Any> create(publicGroups: Iterable<GroupKeysT>) =
        create(publicGroups.asSequence())

      /**
       * Constructor for [PublicGroups] when [publicGroups] are provided as a [Sequence].
       *
       * @param GroupKeysT see the note in [create] method that accepts [Iterable] if you use
       *   DataFrames API.
       */
      @JvmStatic
      fun <GroupKeysT : Any> create(publicGroups: Sequence<GroupKeysT>) =
        PublicGroups(LocalCollection(publicGroups))

      /**
       * Constructor for [PublicGroups] when [publicGroups] are provided as a [SparkDataset].
       *
       * @param GroupKeysT see the note in [create] method that accepts [Iterable] if you use
       *   DataFrames API.
       */
      @JvmStatic
      fun <GroupKeysT : Any> create(publicGroups: SparkDataset<GroupKeysT>) =
        PublicGroups(SparkCollection(publicGroups))

      /**
       * Constructor for [PublicGroups] when [publicGroups] are provided as a [SparkDataFrame]
       *
       * Note that the order of the columns in the [SparkDataFrame] must match the order of the
       * columns in the groupBy call.
       */
      @JvmStatic
      fun createForDataFrame(publicGroups: SparkDataFrame): PublicGroups<List<Any?>> =
        PublicGroups(SparkCollection(publicGroups.toSparkDataset()))
    }
  }
}

/**
 * Additional parameters for the GROUP BY clause.
 *
 * @param groupsBalance the balance of the groups, used as hint to optimize the query, optional,
 *   default is UNKNOWN, see [GroupsBalance] for more details on how to determine the correct value.
 */
data class GroupByAdditionalParameters(
  internal val groupsBalance: GroupsBalance = GroupsBalance.UNKNOWN
)

internal fun GroupsType.getBudget(): BudgetPerOpSpec? =
  when (this) {
    is GroupsType.PrivateGroups -> this.budget
    is GroupsType.PublicGroups<*> -> null
  }

internal fun GroupsType.getPreThreshold() =
  when (this) {
    is GroupsType.PrivateGroups -> this.minPrivacyUnitsPerGroup
    is GroupsType.PublicGroups<*> -> 1
  }

@Suppress("UNCHECKED_CAST")
internal fun <GroupKeysT : Any> GroupsType.getPublicGroups(): FrameworkCollection<GroupKeysT>? =
  when (this) {
    is GroupsType.PrivateGroups -> null
    is GroupsType.PublicGroups<*> -> this.publicGroups as FrameworkCollection<GroupKeysT>
  }

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

import com.google.privacy.differentialprivacy.pipelinedp4j.core.ContributionBoundingLevel as InternalContributionBoundingLevel

/**
 * The type of contribution bounding level that determines how user contributions are bounded.
 *
 * Be aware that the choice of contribution bounding level affects privacy guarantees because it
 * changes the effective privacy unit. Therefore choose it consciously and carefully.
 *
 * There are two contribution bounding levels that can be used in production:
 * 1. @property DATASET_LEVEL
 *
 *    Bounds the contributions of the privacy unit across the whole dataset, i.e. performs
 *    cross-group and per-group bounding. The cross-group bounding is controlled with the
 *    [maxGroupsContributed] parameter and the per-group bounding is controlled via various
 *    parameters like [maxContributionsPerGroup], min/maxValue, etc.
 *
 *    From privacy point of view this contribution bounding level protects the privacy unit across
 *    the whole dataset, i.e. in all groups where the privacy unit is present. It means that the
 *    possibility of understanding whether the privacy unit is present in the dataset is limited by
 *    the given Differential Privacy budget. This is the strongest privacy guarantee for a given
 *    privacy unit.
 *
 *    When you use the dataset contribution bounding level the effective privacy unit remains the
 *    same, i.e. you protect the privacy unit across the whole dataset.
 * 2. @property GROUP_LEVEL
 *
 *    Bounds the contributions of the privacy unit only within the groups, i.e. performs only
 *    per-group bounding. The bounding is controlled via various parameters like
 *    [maxContributionsPerGroup], min/maxValue, etc.
 *
 *    From privacy point of view this contribution bounding level protects the privacy unit only
 *    within the groups where it is present. It means that the possibility of understanding whether
 *    the privacy unit is present in any specific group is limited by the given Differential Privacy
 *    budget, but there are no guarantees for being able to understand whether the privacy unit is
 *    present in the whole dataset. This privacy guarantee for a given privacy unit is weaker than
 *    the one provided by the dataset contribution bounding level.
 *
 *    When you use the group contribution bounding level the effective privacy unit becomes
 *    (privacy_unit, group_key). It means that contributions to different groups from the same
 *    privacy unit are treated as if they were contributions from different privacy units (users).
 *    At the same time, contributions of the privacy unit to the same group are treated as
 *    contributions from the same privacy unit (user) and therefore are bounded.
 */
sealed interface ContributionBoundingLevel {
  data class DATASET_LEVEL(
    internal val maxGroupsContributed: Int,
    internal val maxContributionsPerGroup: Int,
  ) : ContributionBoundingLevel {
    init {
      require(maxGroupsContributed > 0) { "maxGroupsContributed must be positive" }
      require(maxContributionsPerGroup > 0) { "maxContributionsPerGroup must be positive" }
    }
  }

  data class GROUP_LEVEL(internal val maxContributionsPerGroup: Int) : ContributionBoundingLevel {
    init {
      require(maxContributionsPerGroup > 0) { "maxContributionsPerGroup must be positive" }
    }
  }
}

internal fun ContributionBoundingLevel.getMaxPartitionsContributed() =
  when (this) {
    is ContributionBoundingLevel.DATASET_LEVEL -> this.maxGroupsContributed
    is ContributionBoundingLevel.GROUP_LEVEL -> 1
  }

internal fun ContributionBoundingLevel.getMaxContributionsPerPartition() =
  when (this) {
    is ContributionBoundingLevel.DATASET_LEVEL -> this.maxContributionsPerGroup
    is ContributionBoundingLevel.GROUP_LEVEL -> this.maxContributionsPerGroup
  }

/**
 * Converts the [ContributionBoundingLevel] to the [InternalContributionBoundingLevel].
 *
 * We delibaretly do not expose the internal classes in the public API to limit the surface of the
 * API. This will give us more flexibility to change the implementation.
 */
internal fun ContributionBoundingLevel.toInternalContributionBoundingLevel() =
  when (this) {
    is ContributionBoundingLevel.DATASET_LEVEL -> InternalContributionBoundingLevel.DATASET_LEVEL
    is ContributionBoundingLevel.GROUP_LEVEL -> InternalContributionBoundingLevel.PARTITION_LEVEL
  }

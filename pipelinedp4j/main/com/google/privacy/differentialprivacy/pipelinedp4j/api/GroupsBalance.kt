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

import com.google.privacy.differentialprivacy.pipelinedp4j.core.PartitionsBalance

/**
 * The balance of the groups in the input dataset.
 *
 * Groups are balanced if there is no group which contribute > 1% of data. Otherwise, the groups are
 * unbalanced.
 */
enum class GroupsBalance {
  /** Use if you don't know the answer. */
  UNKNOWN,
  /** Use if you know that the groups are balanced according to the definition above. */
  BALANCED,
  /** Use if you know that the groups are unbalanced according to the definition above. */
  UNBALANCED,
}

/**
 * Converts the [GroupsBalance] to the [PartitionsBalance] which is used internally.
 *
 * We delibaretly do not expose the internal classes in the public API to limit the surface of the
 * API. This will give us more flexibility to change the implementation.
 */
internal fun GroupsBalance.toPartitionsBalance() =
  when (this) {
    GroupsBalance.UNKNOWN -> PartitionsBalance.UNKNOWN
    GroupsBalance.BALANCED -> PartitionsBalance.BALANCED
    GroupsBalance.UNBALANCED -> PartitionsBalance.UNBALANCED
  }

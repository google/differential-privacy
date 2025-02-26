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

import java.io.Serializable

@ConsistentCopyVisibility
data class ColumnNames internal constructor(internal val names: List<String>) : Serializable {
  constructor(vararg names: String) : this(names.toList())
}

/**
 * Type of List implementation where column values of the same row will be stored.
 *
 * It is important for list comparisons (e.g. in "group by" operations) to have the same
 * implementations of the list. It is not enough that these lists have the same elements in the same
 * order (i.e. l1.equals(l2)). This is because keys (which are lists in our case) in join/group by
 * operations are compared just as a sequence of bytes and different list implementations have
 * different serialized representation even if they are equal.
 */
internal typealias ColumnValuesListImplementation = ArrayList<Any?>

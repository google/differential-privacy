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

import com.google.privacy.differentialprivacy.pipelinedp4j.core.NormKind as InternalNormKind

/**
 * Contribution bounds of the value.
 *
 * @param totalValueBounds the minimum and maximum **value of the sum per privacy unit in a group**.
 *   Used for calculating sum. You don't have to specify it if you also calculate mean or variance.
 *   Note that it is not the bounds of one value that a privacy unit can contribute. Providing total
 *   values makes computation of sum more optimal.
 * @param valueBounds the bounds of one value that a privacy unit can contribute. You have to
 *   specify it if you calculate mean, variance or quantiles.
 */
data class ContributionBounds(
  internal val totalValueBounds: Bounds? = null,
  internal val valueBounds: Bounds? = null,
)

/**
 * Closed range of possible values.
 *
 * In differential privacy the value boundaries are used to enforce contribution bounding.
 *
 * @param minValue the minimum value of the range, inclusive.
 * @param maxValue the maximum value of the range, inclusive.
 */
data class Bounds(internal val minValue: Double, internal val maxValue: Double) {
  init {
    require(minValue <= maxValue) {
      "minValue must be <= maxValue, but minValue=$minValue > maxValue=$maxValue"
    }
  }
}

/**
 * Contribution bounds of the vector.
 *
 * @param maxVectorTotalNorm the maximum norm of the vector sum per privacy unit in a group. Used
 *   for vector sum. E.g. if a privacy unit contributes v1, v2 and v3 in a group "A" then the sum
 *   v1 + v2 + v3 will be scaled down so that its norm is <= maxVectorTotalNorm and not each of v1,
 *   v2 and v3 will scaled down individually. That's why there is "total" in the name.
 *
 * There is no min bound on the vector norm because the norm of a vector is always non-negative.
 */
data class VectorContributionBounds(internal val maxVectorTotalNorm: VectorNorm)

/**
 * The norm of the vector.
 *
 * @param normKind the kind of the vector norm.
 * @param value the value of the norm.
 */
data class VectorNorm(internal val normKind: NormKind, internal val value: Double) {
  init {
    require(value >= 0.0) { "value of the vector norm must be >= 0.0, but value=$value" }
  }
}

/**
 * The kind of the vector norm.
 *
 * See https://en.wikipedia.org/wiki/Norm_%28mathematics%29#p-norm for definitions.
 */
enum class NormKind {
  L_INF,
  L1,
  L2,
}

/**
 * Converts the [NormKind] to the [InternalNormKind].
 *
 * We delibaretly do not expose the internal classes in the public API to limit the surface of the
 * API. This will give us more flexibility to change the implementation.
 */
internal fun NormKind.toInternalNormKind() =
  when (this) {
    NormKind.L_INF -> InternalNormKind.L_INF
    NormKind.L1 -> InternalNormKind.L1
    NormKind.L2 -> InternalNormKind.L2
  }

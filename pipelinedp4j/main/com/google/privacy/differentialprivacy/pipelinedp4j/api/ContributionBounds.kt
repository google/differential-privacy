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

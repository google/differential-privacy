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

import com.google.common.truth.Truth.assertThat
import java.io.Serializable

/** Test data row to be used in tests for different backends. */
internal data class TestDataRow(val groupKey: String, val privacyUnit: String, val value: Double) {
  // Necessary for Beam serialization.
  private constructor() : this("", "", 0.0)
}

/**
 * A [QueryPerGroupResult] with a tolerance for the double values.
 *
 * This is necessary because the tolerance is not part of the [QueryPerGroupResult] and is only used
 * for testing.
 */
internal data class QueryPerGroupResultWithTolerance(
  val groupKey: String,
  val aggregationResults: Map<String, DoubleWithTolerance>,
) : Serializable

internal data class DoubleWithTolerance(val value: Double, val tolerance: Double) : Serializable

internal fun assertEquals(
  result: List<QueryPerGroupResult<String>>,
  expected: List<QueryPerGroupResultWithTolerance>,
) {
  assertThat(result).hasSize(expected.size)
  val queryPerGroupResults = result.groupBy { it.groupKey }
  val expectedQueryPerGroupResults = expected.groupBy { it.groupKey }
  assertThat(queryPerGroupResults.keys).isEqualTo(expectedQueryPerGroupResults.keys)
  for (groupKey in queryPerGroupResults.keys) {
    assertThat(queryPerGroupResults[groupKey]!!.size).isEqualTo(1)
    val aggregationResults = queryPerGroupResults[groupKey]!!.first().aggregationResults
    assertThat(expectedQueryPerGroupResults[groupKey]!!.size).isEqualTo(1)
    val expectedAggregationResults = expectedQueryPerGroupResults[groupKey]!![0].aggregationResults
    assertThat(aggregationResults.keys).isEqualTo(expectedAggregationResults.keys)
    for (aggregationKey in aggregationResults.keys) {
      val expectedAggregationResult = expectedAggregationResults[aggregationKey]!!
      assertThat(aggregationResults[aggregationKey])
        .isWithin(expectedAggregationResult.tolerance)
        .of(expectedAggregationResult.value)
    }
  }
}

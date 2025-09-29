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

package com.google.privacy.differentialprivacy.pipelinedp4j.core

import com.google.common.truth.Truth.assertThat
import com.google.privacy.differentialprivacy.pipelinedp4j.local.LocalEncoderFactory
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4

@RunWith(JUnit4::class)
class DataExtractorsTest {
  @Test
  fun from_withScalarValueExtractor_constructsDataExtractors() {
    val inputRow = InputRow(privacyId = "userId", partitionKey = "partitionKey", value = 10.0)
    val ef = LocalEncoderFactory()

    val dataExtractors =
      DataExtractors.from(
        { inputRow: InputRow -> inputRow.privacyId },
        ef.strings(),
        { inputRow: InputRow -> inputRow.partitionKey },
        ef.strings(),
        valuesExtractors =
          listOf(FeatureValuesExtractor("feature") { inputRow: InputRow -> listOf(inputRow.value) }),
      )
    val extractedContribution = dataExtractors.contributionExtractor.invoke(inputRow)

    assertThat(extractedContribution)
      .isEqualTo(
        multiFeatureContribution(
          "userId",
          "partitionKey",
          PerFeatureValues(featureId = "feature", values = listOf(10.0)),
        )
      )
  }

  @Test
  fun from_withVectorValueExtractor_constructsDataExtractors() {
    val inputRow = InputRow(privacyId = "userId", partitionKey = "partitionKey", value = 10.0)
    val ef = LocalEncoderFactory()

    val dataExtractors =
      DataExtractors.from(
        { inputRow: InputRow -> inputRow.privacyId },
        ef.strings(),
        { inputRow: InputRow -> inputRow.partitionKey },
        ef.strings(),
        valuesExtractors =
          listOf(
            FeatureValuesExtractor("feature") { inputRow: InputRow ->
              listOf(inputRow.value, inputRow.value, inputRow.value)
            }
          ),
      )
    val extractedContribution = dataExtractors.contributionExtractor.invoke(inputRow)

    assertThat(extractedContribution)
      .isEqualTo(
        multiFeatureContribution(
          "userId",
          "partitionKey",
          PerFeatureValues(featureId = "feature", values = listOf(10.0, 10.0, 10.0)),
        )
      )
  }

  @Test
  fun from_withoutValueExtractor_constructsDataExtractors() {
    val inputRow = InputRow(privacyId = "userId", partitionKey = "partitionKey", value = 20.0)
    val ef = LocalEncoderFactory()

    val dataExtractors =
      DataExtractors.from<InputRow, String, String>(
        { it.privacyId },
        ef.strings(),
        { it.partitionKey },
        ef.strings(),
      )
    val extractedContribution = dataExtractors.contributionExtractor.invoke(inputRow)

    assertThat(extractedContribution)
      .isEqualTo(
        multiFeatureContribution(
          "userId",
          "partitionKey",
          PerFeatureValues(featureId = "", values = listOf(0.0)),
        )
      )
  }

  @Test
  fun from_withMultipleFeatures_constructsDataExtractors() {
    val inputRow = InputRow(privacyId = "userId", partitionKey = "partitionKey", value = 10.0)
    val ef = LocalEncoderFactory()

    val dataExtractors =
      DataExtractors.from(
        { inputRow: InputRow -> inputRow.privacyId },
        ef.strings(),
        { inputRow: InputRow -> inputRow.partitionKey },
        ef.strings(),
        valuesExtractors =
          listOf(
            FeatureValuesExtractor("feature1") { inputRow: InputRow -> listOf(inputRow.value) },
            FeatureValuesExtractor("feature2") { inputRow: InputRow ->
              listOf(inputRow.value, inputRow.value)
            },
          ),
      )
    val extractedContribution = dataExtractors.contributionExtractor.invoke(inputRow)

    assertThat(extractedContribution)
      .isEqualTo(
        multiFeatureContribution(
          "userId",
          "partitionKey",
          listOf(
            PerFeatureValues(featureId = "feature1", values = listOf(10.0)),
            PerFeatureValues(featureId = "feature2", values = listOf(10.0, 10.0)),
          ),
        )
      )
  }
}

data class InputRow(val privacyId: String, val partitionKey: String, val value: Double)

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
  fun from_withValueExtractor_constructsDataExtractors() {
    val inputRow = InputRow(privacyId = "userId", partitionKey = "partitionKey", value = 10.0)
    val ef = LocalEncoderFactory()

    val dataExtractors =
      DataExtractors.from<InputRow, String, String>(
        { it.privacyId },
        ef.strings(),
        { it.partitionKey },
        ef.strings(),
        { it.value },
      )
    val extractedContribution = dataExtractors.contributionExtractor.invoke(inputRow)

    assertThat(extractedContribution)
      .isEqualTo(contributionWithPrivacyId("userId", "partitionKey", 10.0))
  }

  @Test
  fun forVectorfrom_withValuesExtractor_constructsDataExtractors() {
    val inputRow = InputRow(privacyId = "userId", partitionKey = "partitionKey", value = 10.0)
    val ef = LocalEncoderFactory()

    val dataExtractors =
      DataExtractors.forVectorFrom<InputRow, String, String>(
        { it.privacyId },
        ef.strings(),
        { it.partitionKey },
        ef.strings(),
        { listOf(it.value, it.value, it.value) },
      )
    val extractedContribution = dataExtractors.contributionExtractor.invoke(inputRow)

    assertThat(extractedContribution)
      .isEqualTo(contributionWithPrivacyId("userId", "partitionKey", listOf(10.0, 10.0, 10.0)))
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
      .isEqualTo(contributionWithPrivacyId("userId", "partitionKey", 0.0))
  }
}

data class InputRow(val privacyId: String, val partitionKey: String, val value: Double)

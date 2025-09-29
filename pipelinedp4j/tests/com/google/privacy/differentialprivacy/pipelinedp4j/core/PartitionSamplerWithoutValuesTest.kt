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
import com.google.privacy.differentialprivacy.pipelinedp4j.local.LocalCollection
import com.google.privacy.differentialprivacy.pipelinedp4j.local.LocalEncoderFactory
import com.google.privacy.differentialprivacy.pipelinedp4j.local.LocalTable
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.PrivacyIdContributions
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.privacyIdContributions
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4

@RunWith(JUnit4::class)
class PartitionSamplerWithoutValuesTest {
  @Test
  fun sampleContributions_returnsSubsampleOfContributedPartitions() {
    val contributedPks = listOf("red", "blue", "green", "orange")
    val inputData =
      LocalCollection(
        sequenceOf(
          multiFeatureContribution(
            "samePrivacyId",
            "red",
            PerFeatureValues(featureId = "", values = listOf(1.0)),
          ),
          multiFeatureContribution(
            "samePrivacyId",
            "blue",
            PerFeatureValues(featureId = "", values = listOf(1.0)),
          ),
          multiFeatureContribution(
            "samePrivacyId",
            "green",
            PerFeatureValues(featureId = "", values = listOf(1.0)),
          ),
          multiFeatureContribution(
            "samePrivacyId",
            "orange",
            PerFeatureValues(featureId = "", values = listOf(1.0)),
          ),
        )
      )
    val sampledData =
      PartitionSamplerWithoutValues(
          maxPartitionsContributed = 3,
          LOCAL_EF.strings(),
          LOCAL_EF.strings(),
          LOCAL_EF,
        )
        .sampleContributions(inputData) as LocalTable<String, PrivacyIdContributions>

    val returnedPks = sampledData.data.map { it.first }.toList()
    assertThat(returnedPks.count()).isEqualTo(3)
    // Returned partition keys are all in the list of the contributed partition keys.
    assertThat(contributedPks).containsAtLeastElementsIn(returnedPks)
  }

  @Test
  fun sampleContributions_doesntReturnContributedValues() {
    val inputData =
      LocalCollection(
        sequenceOf(
          multiFeatureContribution(
            "privacyId",
            "pk",
            PerFeatureValues(featureId = "", values = listOf(1.0)),
          ),
          multiFeatureContribution(
            "privacyId",
            "pk",
            PerFeatureValues(featureId = "", values = listOf(1.0)),
          ),
          multiFeatureContribution(
            "privacyId",
            "pk",
            PerFeatureValues(featureId = "", values = listOf(1.0)),
          ),
        )
      )

    val sampledData =
      PartitionSamplerWithoutValues(
          maxPartitionsContributed = 2,
          LOCAL_EF.strings(),
          LOCAL_EF.strings(),
          LOCAL_EF,
        )
        .sampleContributions(inputData) as LocalTable<String, PrivacyIdContributions>

    // Check that all values are dropped
    assertThat(sampledData.data.toMap().get("pk")!!.singleValueContributionsList.size).isEqualTo(0)
  }

  @Test
  fun sampleContributions_returnsResultPerPrivacyId() {
    val inputData =
      LocalCollection(
        sequenceOf(
          multiFeatureContribution(
            "privacyId",
            "pk",
            PerFeatureValues(featureId = "", values = listOf(1.0)),
          ),
          multiFeatureContribution(
            "privacyId",
            "pk",
            PerFeatureValues(featureId = "", values = listOf(1.0)),
          ),
          multiFeatureContribution(
            "anotherPrivacyId",
            "pk",
            PerFeatureValues(featureId = "", values = listOf(2.0)),
          ),
        )
      )

    val sampledData =
      PartitionSamplerWithoutValues(
          maxPartitionsContributed = 5,
          LOCAL_EF.strings(),
          LOCAL_EF.strings(),
          LOCAL_EF,
        )
        .sampleContributions(inputData) as LocalTable<String, PrivacyIdContributions>

    assertThat(sampledData.data.toList())
      .containsExactly(Pair("pk", privacyIdContributions {}), Pair("pk", privacyIdContributions {}))
  }

  @Test
  fun sampleContributions_samplesFromManyPartitions() {
    val contributedKeys = (0 until 100_000).map { it.toString() }
    val inputData =
      LocalCollection(
        contributedKeys
          .map {
            multiFeatureContribution(
              "privacyId",
              it,
              PerFeatureValues(featureId = "", values = listOf(1.0)),
            )
          }
          .asSequence()
      )

    val sampledData =
      PartitionSamplerWithoutValues(
          maxPartitionsContributed = 300,
          LOCAL_EF.strings(),
          LOCAL_EF.strings(),
          LOCAL_EF,
        )
        .sampleContributions(inputData) as LocalTable<String, PrivacyIdContributions>

    val returnedPks = sampledData.data.map { it.first }.toList()
    assertThat(returnedPks.count()).isEqualTo(300)
    assertThat(contributedKeys).containsAtLeastElementsIn(returnedPks)
  }

  private companion object {
    private val LOCAL_EF = LocalEncoderFactory()
  }
}

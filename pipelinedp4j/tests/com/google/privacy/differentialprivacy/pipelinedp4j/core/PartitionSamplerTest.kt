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
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.PrivacyIdContributionsKt.multiValueContribution
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.privacyIdContributions
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4

@RunWith(JUnit4::class)
class PartitionSamplerTest {
  @Test
  fun sampleContributions_returnsSubsampleOfContributedPartitions() {
    val contributedPks = listOf("red", "blue", "green", "orange")
    val inputData =
      LocalCollection(
        sequenceOf(
          contributionWithPrivacyId("samePrivacyId", "red", 1.0),
          contributionWithPrivacyId("samePrivacyId", "blue", 1.0),
          contributionWithPrivacyId("samePrivacyId", "green", 1.0),
          contributionWithPrivacyId("samePrivacyId", "orange", 1.0),
        )
      )
    val sampledData =
      PartitionSampler(
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
  fun sampleContributions_doesntSampleContributionsPerPartition() {
    val inputData =
      LocalCollection(
        sequenceOf(
          contributionWithPrivacyId("privacyId", "pk", 1.0),
          contributionWithPrivacyId("privacyId", "pk", 1.0),
          contributionWithPrivacyId("privacyId", "pk", 1.0),
        )
      )

    val sampledData =
      PartitionSampler(
          maxPartitionsContributed = 2,
          LOCAL_EF.strings(),
          LOCAL_EF.strings(),
          LOCAL_EF,
        )
        .sampleContributions(inputData) as LocalTable<String, PrivacyIdContributions>

    assertThat(sampledData.data.toMap().get("pk")!!.singleValueContributionsList.size).isEqualTo(3)
  }

  @Test
  fun sampleContributions_singleValueContributions_returnsResultPerPrivacyId() {
    val inputData =
      LocalCollection(
        sequenceOf(
          contributionWithPrivacyId("privacyId", "pk", 1.0),
          contributionWithPrivacyId("privacyId", "pk", 1.0),
          contributionWithPrivacyId("anotherPrivacyId", "pk", 2.0),
        )
      )

    val sampledData =
      PartitionSampler(
          maxPartitionsContributed = 5,
          LOCAL_EF.strings(),
          LOCAL_EF.strings(),
          LOCAL_EF,
        )
        .sampleContributions(inputData) as LocalTable<String, PrivacyIdContributions>

    assertThat(sampledData.data.toList())
      .containsExactly(
        Pair("pk", privacyIdContributions { singleValueContributions += listOf(1.0, 1.0) }),
        Pair("pk", privacyIdContributions { singleValueContributions += 2.0 }),
      )
  }

  @Test
  fun sampleContributions_multiValueContributions_returnsResultPerPrivacyId() {
    val inputData =
      LocalCollection(
        sequenceOf(
          contributionWithPrivacyId("privacyId", "pk", listOf(1.0, 2.0)),
          contributionWithPrivacyId("privacyId", "pk", listOf(3.0, 4.0)),
          contributionWithPrivacyId("anotherPrivacyId", "pk", listOf(5.0, 6.0)),
        )
      )

    val sampledData =
      PartitionSampler(
          maxPartitionsContributed = 5,
          LOCAL_EF.strings(),
          LOCAL_EF.strings(),
          LOCAL_EF,
        )
        .sampleContributions(inputData) as LocalTable<String, PrivacyIdContributions>

    assertThat(sampledData.data.toList())
      .containsExactly(
        Pair(
          "pk",
          privacyIdContributions {
            multiValueContributions +=
              listOf(
                multiValueContribution { values += listOf(1.0, 2.0) },
                multiValueContribution { values += listOf(3.0, 4.0) },
              )
          },
        ),
        Pair(
          "pk",
          privacyIdContributions {
            multiValueContributions += listOf(multiValueContribution { values += listOf(5.0, 6.0) })
          },
        ),
      )
  }

  @Test
  fun sampleContributions_samplesFromManyPartitions() {
    val contributedKeys = (0 until 100_000).map { it.toString() }
    val inputData =
      LocalCollection(
        contributedKeys.map { contributionWithPrivacyId("privacyId", it, value = 1.0) }.asSequence()
      )

    val sampledData =
      PartitionSampler(
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

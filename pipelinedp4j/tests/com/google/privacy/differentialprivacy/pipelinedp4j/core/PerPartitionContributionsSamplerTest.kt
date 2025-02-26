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

import com.google.common.collect.ImmutableList
import com.google.common.truth.Truth.assertThat
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.MEAN
import com.google.privacy.differentialprivacy.pipelinedp4j.core.NoiseKind.GAUSSIAN
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
class PerPartitionContributionsSamplerTest {
  @Test
  fun sampleContributions_singleValueContributions_returnsSubsampleOfContributionsPerPartitionSinglePrivacyId() {
    val inputData =
      LocalCollection(
        sequenceOf(
          contributionWithPrivacyId("privacyId1", "pk1", 1.0),
          contributionWithPrivacyId("privacyId1", "pk1", 2.0),
          contributionWithPrivacyId("privacyId1", "pk1", 3.0),
          contributionWithPrivacyId("privacyId1", "pk1", 4.0),
        )
      )

    val sampledData =
      PerPartitionContributionsSampler(
          maxContributionsPerPartition = 2,
          LOCAL_EF.strings(),
          LOCAL_EF.strings(),
          LOCAL_EF,
        )
        .sampleContributions(inputData) as LocalTable<String, PrivacyIdContributions>
    val returnedContributions = sampledData.data.toMap().get("pk1")!!
    assertThat(returnedContributions.multiValueContributionsList.count()).isEqualTo(0)
    val singleValueContributions = returnedContributions.singleValueContributionsList
    assertThat(singleValueContributions.count()).isEqualTo(2)
    // Returned values are all in the list of the contributed values.
    assertThat(listOf(1.0, 2.0, 3.0, 4.0)).containsAtLeastElementsIn(singleValueContributions)
  }

  @Test
  fun sampleContributions_multiValueContributions_returnsSubsampleOfContributionsPerPartitionSinglePrivacyId() {
    val inputData =
      LocalCollection(
        sequenceOf(
          contributionWithPrivacyId("samePrivacyId", "pk1", listOf(1.0, 2.0)),
          contributionWithPrivacyId("samePrivacyId", "pk1", listOf(3.0, 4.0)),
          contributionWithPrivacyId("samePrivacyId", "pk1", listOf(5.0, 6.0)),
          contributionWithPrivacyId("samePrivacyId", "pk1", listOf(7.0, 8.0)),
        )
      )

    val sampledData =
      PerPartitionContributionsSampler(
          maxContributionsPerPartition = 2,
          LOCAL_EF.strings(),
          LOCAL_EF.strings(),
          LOCAL_EF,
        )
        .sampleContributions(inputData) as LocalTable<String, PrivacyIdContributions>
    val returnedContributions = sampledData.data.toMap().get("pk1")!!
    assertThat(returnedContributions.singleValueContributionsList.count()).isEqualTo(0)
    val multiValueContributions = returnedContributions.multiValueContributionsList
    assertThat(multiValueContributions.count()).isEqualTo(2)
    // Returned values are all in the list of the contributed values.
    assertThat(
        listOf(
          multiValueContribution { values += listOf(1.0, 2.0) },
          multiValueContribution { values += listOf(3.0, 4.0) },
          multiValueContribution { values += listOf(5.0, 6.0) },
          multiValueContribution { values += listOf(7.0, 8.0) },
        )
      )
      .containsAtLeastElementsIn(multiValueContributions)
  }

  @Test
  fun sampleContributions_returnsOriginalContributionsPerPartition() {
    val sampledData =
      PerPartitionContributionsSampler(
          maxContributionsPerPartition = 10,
          LOCAL_EF.strings(),
          LOCAL_EF.strings(),
          LOCAL_EF,
        )
        .sampleContributions(multipleContributionsMultiplePrivacyIdInput)
        as LocalTable<String, PrivacyIdContributions>

    assertThat(sampledData.data.toList())
      .containsExactly(
        Pair("pk1", privacyIdContributions { singleValueContributions += listOf(1.0, 2.0, 3.0) }),
        Pair("pk2", privacyIdContributions { singleValueContributions += listOf(4.0, 5.0, 6.0) }),
        Pair("pk1", privacyIdContributions { singleValueContributions += listOf(7.0) }),
      )
  }

  @Test
  fun sampleContributions_returnsSubsampleOfContributionsPerPartition() {
    val sampledData =
      PerPartitionContributionsSampler(
          maxContributionsPerPartition = 1,
          LOCAL_EF.strings(),
          LOCAL_EF.strings(),
          LOCAL_EF,
        )
        .sampleContributions(multipleContributionsMultiplePrivacyIdInput)
        as LocalTable<String, PrivacyIdContributions>
    val returnedPks = sampledData.data.map { it.first }.toList()

    assertThat(returnedPks.count()).isEqualTo(3)

    // Returned partition keys should only have 1 contribution from each privacy ID.
    for (pk in returnedPks) {
      assertThat(sampledData.data.toMap().getValue(pk).singleValueContributionsList).hasSize(1)
    }
  }

  @Test
  fun sampleContributions_samplesFromManyContributions() {
    val inputData =
      LocalCollection(
        sequence {
            repeat(100_000) {
              yield(contributionWithPrivacyId("privacyId", "pk", value = it.toDouble()))
            }
          }
          .asSequence()
      )

    val sampledData =
      PerPartitionContributionsSampler(
          maxContributionsPerPartition = 300,
          LOCAL_EF.strings(),
          LOCAL_EF.strings(),
          LOCAL_EF,
        )
        .sampleContributions(inputData) as LocalTable<String, PrivacyIdContributions>
    val returnedContributions = sampledData.data.toMap().getValue("pk").singleValueContributionsList

    assertThat(returnedContributions.count()).isEqualTo(300)
  }

  private companion object {
    val aggParams =
      AggregationParams(
        metrics = ImmutableList.of(MetricDefinition(MEAN)),
        noiseKind = GAUSSIAN,
        maxPartitionsContributed = 1,
        maxContributionsPerPartition = 2,
      )
    val contributedPks = listOf("pk1", "pk2", "pk1")
    val multipleContributionsMultiplePrivacyIdInput =
      LocalCollection(
        sequenceOf(
          contributionWithPrivacyId("privacyId1", "pk1", value = 1.0),
          contributionWithPrivacyId("privacyId1", "pk1", value = 2.0),
          contributionWithPrivacyId("privacyId1", "pk1", value = 3.0),
          contributionWithPrivacyId("privacyId1", "pk2", value = 4.0),
          contributionWithPrivacyId("privacyId1", "pk2", value = 5.0),
          contributionWithPrivacyId("privacyId1", "pk2", value = 6.0),
          contributionWithPrivacyId("privacyId2", "pk1", value = 7.0),
        )
      )
    private val LOCAL_EF = LocalEncoderFactory()
  }
}

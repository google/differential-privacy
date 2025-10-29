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
import kotlin.Int.Companion.MAX_VALUE
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4

@RunWith(JUnit4::class)
class PartitionAndPerPartitionSamplerTest {
  val AGG_PARAMS =
    AggregationParams(
      nonFeatureMetrics = ImmutableList.of(),
      features =
        ImmutableList.of(
          ScalarFeatureSpec("value", ImmutableList.of(MetricDefinition(MEAN)), -1.0, 1.0)
        ),
      noiseKind = GAUSSIAN,
      maxPartitionsContributed = MAX_VALUE,
      maxContributionsPerPartition = MAX_VALUE,
    )

  @Test
  fun sampleContributions_returnsSubsampleOfContributedPartitions() {
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
      PartitionAndPerPartitionSampler(
          maxPartitionsContributed = 3,
          maxContributionsPerPartition = MAX_VALUE,
          LOCAL_EF.strings(),
          LOCAL_EF.strings(),
          LOCAL_EF,
        )
        .sampleContributions(inputData) as LocalTable<String, PrivacyIdContributions>

    val returnedPks = sampledData.data.map { it.first }.toList()
    assertThat(returnedPks.count()).isEqualTo(3)
    // Returned partition keys are all in the list of the contributed partition keys.
    assertThat(listOf("red", "blue", "green", "orange")).containsAtLeastElementsIn(returnedPks)
  }

  @Test
  fun sampleContributions_singleValueContributions_returnsSubsampleOfContributionsPerPartition() {
    val inputData =
      LocalCollection(
        sequenceOf(
          multiFeatureContribution(
            "samePrivacyId",
            "samePk",
            PerFeatureValues(featureId = "", values = listOf(1.0)),
          ),
          multiFeatureContribution(
            "samePrivacyId",
            "samePk",
            PerFeatureValues(featureId = "", values = listOf(2.0)),
          ),
          multiFeatureContribution(
            "samePrivacyId",
            "samePk",
            PerFeatureValues(featureId = "", values = listOf(3.0)),
          ),
          multiFeatureContribution(
            "samePrivacyId",
            "samePk",
            PerFeatureValues(featureId = "", values = listOf(4.0)),
          ),
        )
      )

    val sampledData =
      PartitionAndPerPartitionSampler(
          maxPartitionsContributed = MAX_VALUE,
          maxContributionsPerPartition = 3,
          LOCAL_EF.strings(),
          LOCAL_EF.strings(),
          LOCAL_EF,
        )
        .sampleContributions(inputData) as LocalTable<String, PrivacyIdContributions>

    val returnedContributions = sampledData.data.toMap().get("samePk")!!
    assertThat(returnedContributions.multiValueContributionsList.count()).isEqualTo(0)
    val singleValueContributions = returnedContributions.singleValueContributionsList
    assertThat(singleValueContributions.count()).isEqualTo(3)
    // Returned values are all in the list of the contributed values.
    assertThat(listOf(1.0, 2.0, 3.0, 4.0)).containsAtLeastElementsIn(singleValueContributions)
  }

  @Test
  fun sampleContributions_multValueContributions_returnsSubsampleOfContributionsPerPartition() {
    val inputData =
      LocalCollection(
        sequenceOf(
          multiFeatureContribution(
            "samePrivacyId",
            "samePk",
            PerFeatureValues(featureId = "", values = listOf(1.0, 2.0)),
          ),
          multiFeatureContribution(
            "samePrivacyId",
            "samePk",
            PerFeatureValues(featureId = "", values = listOf(3.0, 4.0)),
          ),
          multiFeatureContribution(
            "samePrivacyId",
            "samePk",
            PerFeatureValues(featureId = "", values = listOf(5.0, 6.0)),
          ),
          multiFeatureContribution(
            "samePrivacyId",
            "samePk",
            PerFeatureValues(featureId = "", values = listOf(7.0, 8.0)),
          ),
        )
      )

    val sampledData =
      PartitionAndPerPartitionSampler(
          maxPartitionsContributed = MAX_VALUE,
          maxContributionsPerPartition = 3,
          LOCAL_EF.strings(),
          LOCAL_EF.strings(),
          LOCAL_EF,
        )
        .sampleContributions(inputData) as LocalTable<String, PrivacyIdContributions>

    val returnedContributions = sampledData.data.toMap().get("samePk")!!
    assertThat(returnedContributions.singleValueContributionsList.count()).isEqualTo(0)
    val multiValueContributions = returnedContributions.multiValueContributionsList
    assertThat(multiValueContributions.count()).isEqualTo(3)
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
  fun sampleContributions_groupsResultPerPrivacyIdAndPartitionKey() {
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
            PerFeatureValues(featureId = "", values = listOf(2.0)),
          ),
          multiFeatureContribution(
            "privacyId",
            "anotherPk",
            PerFeatureValues(featureId = "", values = listOf(3.0)),
          ),
          multiFeatureContribution(
            "privacyId",
            "anotherPk",
            PerFeatureValues(featureId = "", values = listOf(4.0)),
          ),
          multiFeatureContribution(
            "anotherPrivacyId",
            "pk",
            PerFeatureValues(featureId = "", values = listOf(5.0)),
          ),
        )
      )

    val sampledData =
      PartitionAndPerPartitionSampler(
          maxPartitionsContributed = MAX_VALUE,
          maxContributionsPerPartition = MAX_VALUE,
          LOCAL_EF.strings(),
          LOCAL_EF.strings(),
          LOCAL_EF,
        )
        .sampleContributions(inputData) as LocalTable<String, PrivacyIdContributions>

    assertThat(sampledData.data.toList())
      .containsExactly(
        Pair("pk", privacyIdContributions { singleValueContributions += listOf(1.0, 2.0) }),
        Pair("anotherPk", privacyIdContributions { singleValueContributions += listOf(3.0, 4.0) }),
        Pair("pk", privacyIdContributions { singleValueContributions += 5.0 }),
      )
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
      PartitionAndPerPartitionSampler(
          maxPartitionsContributed = 300,
          maxContributionsPerPartition = MAX_VALUE,
          LOCAL_EF.strings(),
          LOCAL_EF.strings(),
          LOCAL_EF,
        )
        .sampleContributions(inputData) as LocalTable<String, PrivacyIdContributions>

    val returnedPks = sampledData.data.map { it.first }.toList()
    assertThat(returnedPks.count()).isEqualTo(300)
    assertThat(contributedKeys).containsAtLeastElementsIn(returnedPks)
  }

  @Test
  fun sampleContributions_samplesFromManyContributions() {
    val inputData =
      LocalCollection(
        (0 until 100_000)
          .map {
            multiFeatureContribution(
              "privacyId",
              "pk",
              PerFeatureValues(featureId = "", values = listOf(it.toDouble())),
            )
          }
          .asSequence()
      )

    val sampledData =
      PartitionAndPerPartitionSampler(
          maxPartitionsContributed = MAX_VALUE,
          maxContributionsPerPartition = 300,
          LOCAL_EF.strings(),
          LOCAL_EF.strings(),
          LOCAL_EF,
        )
        .sampleContributions(inputData) as LocalTable<String, PrivacyIdContributions>

    val returnedContributions = sampledData.data.toMap().get("pk")!!.singleValueContributionsList
    assertThat(returnedContributions.count()).isEqualTo(300)
  }

  private companion object {
    private val LOCAL_EF = LocalEncoderFactory()
  }
}

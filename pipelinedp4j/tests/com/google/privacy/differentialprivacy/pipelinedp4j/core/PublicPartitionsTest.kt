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
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.CompoundAccumulator
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.compoundAccumulator
import com.google.testing.junit.testparameterinjector.TestParameter
import com.google.testing.junit.testparameterinjector.TestParameterInjector
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(TestParameterInjector::class)
class PublicPartitionsTest {
  @Test
  fun dropNonPublicPartitions_keepsOnlyPublicPartitions(
    @TestParameter partitionsBalance: PartitionsBalance
  ) {
    val inputData =
      LocalCollection(
        sequenceOf(
          contributionWithPrivacyId("pid1", "privatePartition1", 1.0),
          contributionWithPrivacyId("pid1", "privatePartition2", 1.0),
          contributionWithPrivacyId("pid1", "publicPartition1", 1.0),
          contributionWithPrivacyId("pid2", "publicPartition1", 1.0),
          contributionWithPrivacyId("pid2", "publicPartition1", 1.0),
          contributionWithPrivacyId("pid3", "publicPartition2", 1.0),
          contributionWithPrivacyId("pid4", "privatePartition2", 1.0),
        )
      )
    val publicPartitions =
      LocalCollection(sequenceOf("publicPartition1", "publicPartition2", "publicPartition3"))

    val result =
      inputData.dropNonPublicPartitions(publicPartitions, LOCAL_EF.strings(), partitionsBalance)
        as LocalCollection<ContributionWithPrivacyId<String, String>>

    assertThat(result.data.toList())
      .containsExactly(
        contributionWithPrivacyId("pid1", "publicPartition1", 1.0),
        contributionWithPrivacyId("pid2", "publicPartition1", 1.0),
        contributionWithPrivacyId("pid2", "publicPartition1", 1.0),
        contributionWithPrivacyId("pid3", "publicPartition2", 1.0),
      )
  }

  @Test
  fun insertPublicPartitions_addsAllPublicPartitionsWithEmptyAccumulatorAsValues() {
    val inputData =
      LocalTable(
        sequenceOf("partition1" to compoundAccumulator {}, "partition3" to compoundAccumulator {})
      )
    val publicPartitions =
      LocalCollection(
        sequenceOf(
          "partition0",
          "partition1",
          "partition2",
          "partition3",
          "partition4",
          "partition5",
        )
      )
    val compoundCombiner = CompoundCombiner(combiners = emptyList())

    val result =
      inputData.insertPublicPartitions(
        publicPartitions,
        compoundCombiner,
        LOCAL_EF.strings(),
        LOCAL_EF,
      ) as LocalTable<String, CompoundAccumulator>

    assertThat(result.data.toList())
      .containsExactly(
        "partition1" to compoundAccumulator {},
        "partition3" to compoundAccumulator {},
        "partition0" to compoundCombiner.emptyAccumulator(),
        "partition1" to compoundCombiner.emptyAccumulator(),
        "partition2" to compoundCombiner.emptyAccumulator(),
        "partition3" to compoundCombiner.emptyAccumulator(),
        "partition4" to compoundCombiner.emptyAccumulator(),
        "partition5" to compoundCombiner.emptyAccumulator(),
      )
  }

  private companion object {
    private val LOCAL_EF = LocalEncoderFactory()
  }
}

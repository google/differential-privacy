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
class NoPrivacySamplerTest {
  @Test
  fun sampleContributions_noPrivacy_returnsOriginal() {
    val contributionsPk1 = listOf(1.0, 2.0, 3.0, 4.0)
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
      NoPrivacySampler(LOCAL_EF.strings(), LOCAL_EF.strings(), LOCAL_EF)
        .sampleContributions(inputData) as LocalTable<String, PrivacyIdContributions>
    val returnedContributionsPk1 =
      sampledData.data.toMap().getValue("pk1").singleValueContributionsList

    // Returned contributions are of the same size as the originals.
    assertThat(returnedContributionsPk1).hasSize(4)
    // Returned partition keys are all in the list of the contributed partition keys.
    assertThat(contributionsPk1).containsExactlyElementsIn(returnedContributionsPk1)
  }

  @Test
  fun sampleContributions_noPrivacy_returnsOriginalGroupsByPk() {
    val inputData =
      LocalCollection(
        sequenceOf(
          contributionWithPrivacyId("privacyId1", "pk1", 1.0),
          contributionWithPrivacyId("privacyId1", "pk1", 2.0),
          contributionWithPrivacyId("privacyId1", "pk1", 3.0),
          contributionWithPrivacyId("privacyId1", "pk1", 4.0),
          contributionWithPrivacyId("privacyId1", "pk2", 5.0),
          contributionWithPrivacyId("privacyId1", "pk2", 6.0),
          contributionWithPrivacyId("privacyId2", "pk2", 7.0),
          contributionWithPrivacyId("privacyId2", "pk2", 8.0),
        )
      )

    val sampledData =
      NoPrivacySampler(LOCAL_EF.strings(), LOCAL_EF.strings(), LOCAL_EF)
        .sampleContributions(inputData) as LocalTable<String, PrivacyIdContributions>
    val resultMap: Map<String, Set<PrivacyIdContributions>> =
      sampledData.data.groupBy({ (k, _) -> k }, { (_, v) -> v }).mapValues { (_, v) -> v.toSet() }

    assertThat(resultMap)
      .isEqualTo(
        mapOf(
          "pk1" to
            setOf(
              privacyIdContributions { singleValueContributions += listOf(1.0, 2.0, 3.0, 4.0) }
            ),
          "pk2" to
            setOf(
              privacyIdContributions { singleValueContributions += listOf(5.0, 6.0) },
              privacyIdContributions { singleValueContributions += listOf(7.0, 8.0) },
            ),
        )
      )
  }

  private companion object {
    private val LOCAL_EF = LocalEncoderFactory()
  }
}

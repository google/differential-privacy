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
import kotlin.test.assertFailsWith
import org.apache.beam.sdk.coders.DoubleCoder
import org.apache.beam.sdk.coders.KvCoder
import org.apache.beam.sdk.coders.StringUtf8Coder
import org.apache.beam.sdk.testing.TestPipeline
import org.apache.beam.sdk.transforms.Create
import org.apache.beam.sdk.values.KV
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4

@RunWith(JUnit4::class)
class BeamQueryBuilderTest {
  @get:Rule val testPipeline: TestPipeline = TestPipeline.create()

  @Test
  fun build_sameOutputColumnNames_throwsException() {
    val pCollection =
      testPipeline.apply(
        "Create input data",
        Create.of(listOf<KV<KV<String, String>, Double>>())
          .withCoder(
            KvCoder.of(KvCoder.of(StringUtf8Coder.of(), StringUtf8Coder.of()), DoubleCoder.of())
          ),
      )

    val queryBuilder =
      QueryBuilder.from(pCollection, { it.key.value })
        .groupBy({ it.key.key }, maxGroupsContributed = 1, maxContributionsPerGroup = 1)
        .sum(
          { it.value },
          minTotalValuePerPrivacyUnitInGroup = 1.0,
          maxTotalValuePerPrivacyUnitInGroup = 2.0,
          outputColumnName = "sameColumnName",
        )
        .count("sameColumnName")

    val e = assertFailsWith<IllegalArgumentException> { queryBuilder.build() }
    assertThat(e)
      .hasMessageThat()
      .contains("There aggregations with duplicate output column names: [sameColumnName]")

    testPipeline.run().waitUntilFinish()
  }

  @Test
  fun build_differentValues_throwsException() {
    val pCollection =
      testPipeline.apply(
        "Create input data",
        Create.of(listOf<KV<KV<String, String>, Double>>())
          .withCoder(
            KvCoder.of(KvCoder.of(StringUtf8Coder.of(), StringUtf8Coder.of()), DoubleCoder.of())
          ),
      )

    val queryBuilder =
      QueryBuilder.from(pCollection, { it.key.value })
        .groupBy({ it.key.key }, maxGroupsContributed = 1, maxContributionsPerGroup = 1)
        .sum(
          { it.value },
          minTotalValuePerPrivacyUnitInGroup = 1.0,
          maxTotalValuePerPrivacyUnitInGroup = 2.0,
          outputColumnName = "sameColumnName",
        )
        .sum(
          { it.value * 2.0 },
          minTotalValuePerPrivacyUnitInGroup = 1.0,
          maxTotalValuePerPrivacyUnitInGroup = 2.0,
          outputColumnName = "otherColumnName",
        )

    val e = assertFailsWith<IllegalArgumentException> { queryBuilder.build() }
    assertThat(e).hasMessageThat().contains("Aggregation of different values is not supported yet.")

    testPipeline.run().waitUntilFinish()
  }
}

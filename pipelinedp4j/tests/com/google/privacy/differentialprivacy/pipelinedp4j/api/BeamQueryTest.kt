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
import org.apache.beam.sdk.coders.DoubleCoder
import org.apache.beam.sdk.coders.KvCoder
import org.apache.beam.sdk.coders.StringUtf8Coder
import org.apache.beam.sdk.testing.PAssert
import org.apache.beam.sdk.testing.TestPipeline
import org.apache.beam.sdk.transforms.Create
import org.apache.beam.sdk.values.KV
import org.apache.beam.sdk.values.PCollection
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4

@RunWith(JUnit4::class)
class BeamQueryTest {
  @get:Rule val testPipeline: TestPipeline = TestPipeline.create()

  @Test
  fun run_onePublicGroupTwoDifferentContributions_allPossibleAggregations_calculatesStatisticsCorrectly() {
    val pCollection =
      testPipeline.apply(
        "Create input data",
        Create.of(
            listOf(
              KV.of(KV.of("group1", "pid1"), 1.0),
              KV.of(KV.of("group1", "pid1"), 1.5),
              KV.of(KV.of("group1", "pid2"), 2.0),
            )
          )
          .withCoder(
            KvCoder.of(KvCoder.of(StringUtf8Coder.of(), StringUtf8Coder.of()), DoubleCoder.of())
          ),
      )
    val publicGroups =
      testPipeline.apply(
        "Create public groups",
        Create.of(listOf("group1")).withCoder(StringUtf8Coder.of()),
      )
    val valueExtractor = { it: KV<KV<String, String>, Double> -> it.value }

    val result: PCollection<QueryPerGroupResult> =
      QueryBuilder.from(pCollection, { it.key.value })
        .groupBy(
          { it.key.key },
          maxGroupsContributed = 1,
          maxContributionsPerGroup = 2,
          publicGroups,
        )
        .countDistinctPrivacyUnits("pid_cnt")
        .count("cnt")
        .sum(valueExtractor, outputColumnName = "sumResult")
        .mean(valueExtractor, minValue = 1.0, maxValue = 2.0, "meanResult")
        .variance(valueExtractor, minValue = 1.0, maxValue = 2.0, "varianceResult")
        .quantiles(
          valueExtractor,
          ranks = listOf(0.5),
          minValue = 1.0,
          maxValue = 2.0,
          "quantilesResult",
        )
        .build()
        .run(TotalBudget(epsilon = 1000.0), NoiseKind.LAPLACE)

    PAssert.that(result).satisfies {
      assertThat(it).hasSize(1)
      val queryPerGroupResult = it.iterator().next()
      assertThat(queryPerGroupResult.groupKey).isEqualTo("group1")
      assertThat(queryPerGroupResult.aggregationResults).hasSize(6)
      assertThat(queryPerGroupResult.aggregationResults.keys)
        .containsExactly(
          "pid_cnt",
          "cnt",
          "sumResult",
          "meanResult",
          "varianceResult",
          "quantilesResult_0.5",
        )
      assertThat(queryPerGroupResult.aggregationResults["pid_cnt"]).isWithin(0.5).of(2.0)
      assertThat(queryPerGroupResult.aggregationResults["cnt"]).isWithin(0.5).of(3.0)
      assertThat(queryPerGroupResult.aggregationResults["sumResult"]).isWithin(0.5).of(4.5)
      assertThat(queryPerGroupResult.aggregationResults["meanResult"]).isWithin(0.5).of(1.5)
      // (1^2+(1.5)^2+2^2)/3-((1.0+1.5+2)/3)^2 = 0.1(6)
      assertThat(queryPerGroupResult.aggregationResults["varianceResult"]).isWithin(0.05).of(0.16)
      assertThat(queryPerGroupResult.aggregationResults["quantilesResult_0.5"])
        .isWithin(0.5)
        .of(1.5)
      null
    }

    testPipeline.run().waitUntilFinish()
  }

  @Test
  fun run_sumAndQuantiles_calculatesCorrectly() {
    val pCollection =
      testPipeline.apply(
        "Create input data",
        Create.of(
            listOf(
              KV.of(KV.of("group1", "pid1"), 1.0),
              KV.of(KV.of("group1", "pid1"), 1.5),
              KV.of(KV.of("group1", "pid2"), 2.0),
            )
          )
          .withCoder(
            KvCoder.of(KvCoder.of(StringUtf8Coder.of(), StringUtf8Coder.of()), DoubleCoder.of())
          ),
      )
    val publicGroups =
      testPipeline.apply(
        "Create public groups",
        Create.of(listOf("group1")).withCoder(StringUtf8Coder.of()),
      )
    val valueExtractor = { it: KV<KV<String, String>, Double> -> it.value }

    val result: PCollection<QueryPerGroupResult> =
      QueryBuilder.from(pCollection, { it.key.value })
        .groupBy(
          { it.key.key },
          maxGroupsContributed = 1,
          maxContributionsPerGroup = 2,
          publicGroups,
        )
        .sum(
          valueExtractor,
          minTotalValuePerPrivacyUnitInGroup = 2.0,
          maxTotalValuePerPrivacyUnitInGroup = 2.5,
          outputColumnName = "sumResult",
        )
        .quantiles(
          valueExtractor,
          ranks = listOf(0.5),
          minValue = 1.0,
          maxValue = 2.0,
          "quantilesResult",
        )
        .build()
        .run(TotalBudget(epsilon = 1000.0), NoiseKind.LAPLACE)

    PAssert.that(result).satisfies {
      assertThat(it).hasSize(1)
      val queryPerGroupResult = it.iterator().next()
      assertThat(queryPerGroupResult.groupKey).isEqualTo("group1")
      assertThat(queryPerGroupResult.aggregationResults).hasSize(2)
      assertThat(queryPerGroupResult.aggregationResults.keys)
        .containsExactly("sumResult", "quantilesResult_0.5")
      assertThat(queryPerGroupResult.aggregationResults["sumResult"]).isWithin(0.5).of(4.5)
      assertThat(queryPerGroupResult.aggregationResults["quantilesResult_0.5"])
        .isWithin(0.5)
        .of(1.5)
      null
    }

    testPipeline.run().waitUntilFinish()
  }
}

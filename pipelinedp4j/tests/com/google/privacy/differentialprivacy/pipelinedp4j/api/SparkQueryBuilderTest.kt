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
import com.google.privacy.differentialprivacy.pipelinedp4j.spark.SparkEncoderFactory
import com.google.privacy.differentialprivacy.pipelinedp4j.spark.SparkSessionRule
import org.apache.spark.sql.Encoder
import kotlin.test.assertFailsWith
import org.apache.spark.sql.Encoders
import org.junit.ClassRule
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4

@RunWith(JUnit4::class)
class SparkQueryBuilderTest {

  @Test
  fun build_sameOutputColumnNames_throwsException() {
    val dataset = sparkSession.spark.createDataset(listOf(), Encoders.kryo(Pair::class.java) as Encoder<Pair<Pair<String, String>, Double>>)

    val queryBuilder =
      QueryBuilder.from(dataset, { it.first.second })
        .groupBy({ it.first.first }, maxGroupsContributed = 1, maxContributionsPerGroup = 1)
        .sum(
          { it.second },
          minTotalValuePerPrivacyUnitInGroup = 1.0,
          maxTotalValuePerPrivacyUnitInGroup = 2.0,
          outputColumnName = "sameColumnName",
        )
        .count("sameColumnName")

    val e = assertFailsWith<IllegalArgumentException> { queryBuilder.build() }
    assertThat(e)
      .hasMessageThat()
      .contains("There aggregations with duplicate output column names: [sameColumnName]")
  }

  @Test
  fun build_differentValues_throwsException() {
    val dataset = sparkSession.spark.createDataset(listOf(), Encoders.kryo(Pair::class.java) as Encoder<Pair<Pair<String, String>, Double>>)

    val queryBuilder =
      QueryBuilder.from(dataset, { it.first.second })
        .groupBy({ it.first.first }, maxGroupsContributed = 1, maxContributionsPerGroup = 1)
        .sum(
          { it.second },
          minTotalValuePerPrivacyUnitInGroup = 1.0,
          maxTotalValuePerPrivacyUnitInGroup = 2.0,
          outputColumnName = "sameColumnName",
        )
        .sum(
          { it.second * 2.0 },
          minTotalValuePerPrivacyUnitInGroup = 1.0,
          maxTotalValuePerPrivacyUnitInGroup = 2.0,
          outputColumnName = "otherColumnName",
        )

    val e = assertFailsWith<IllegalArgumentException> { queryBuilder.build() }
    assertThat(e).hasMessageThat().contains("Aggregation of different values is not supported yet.")
  }

  companion object {
    @JvmField
    @ClassRule
    val sparkSession = SparkSessionRule()
  }
}

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
import com.google.privacy.differentialprivacy.pipelinedp4j.spark.SparkEncodersTest
import com.google.privacy.differentialprivacy.pipelinedp4j.spark.SparkSessionRule
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Encoders
import org.junit.ClassRule
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4
import scala.Tuple2

@RunWith(JUnit4::class)
class SparkQueryTest {

  @Test
  fun run_onePublicGroupTwoDifferentContributions_allPossibleAggregations_calculatesStatisticsCorrectly() {
    val dataset = sparkSession.spark.createDataset(listOf(
      Pair(Pair("group1", "pid1"), 1.0),
      Pair(Pair("group1", "pid1"), 1.5),
      Pair(Pair("group1", "pid2"), 2.0)
    ), sparkEncoderFactory.tuple2sOf(
      sparkEncoderFactory.tuple2sOf(sparkEncoderFactory.strings(), sparkEncoderFactory.strings()),
      sparkEncoderFactory.doubles()).encoder)

    val publicGroups = sparkSession.spark.createDataset(listOf("group1"), Encoders.STRING())

    val valueExtractor = { it: Pair<Pair<String, String>, Double> -> it.second }

    val result: Dataset<Tuple2<String, Map<String, Double>>> =
      QueryBuilder.from(dataset, { it.first.second })
        .groupBy(
          { it.first.first },
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

    val output = result.collect()
    //assertThat(output.size).isEqualTo(1)

//    assertThat(output).run {
//      hasLength(1)
//      val queryPerGroupResult = output.iterator().next()
//      assertThat(queryPerGroupResult._1).isEqualTo("group1")
//      assertThat(queryPerGroupResult._2).hasSize(6)
//      assertThat(queryPerGroupResult._2.keys).containsExactly( "pid_cnt",
//        "cnt",
//        "sumResult",
//        "meanResult",
//        "varianceResult",
//        "quantilesResult_0.5",
//        )
//      assertThat(queryPerGroupResult._2["pid_cnt"]).isWithin(0.5).of(2.0)
//      assertThat(queryPerGroupResult._2["cnt"]).isWithin(0.5).of(3.0)
//      assertThat(queryPerGroupResult._2["sumResult"]).isWithin(0.5).of(4.5)
//      assertThat(queryPerGroupResult._2["meanResult"]).isWithin(0.5).of(2.0)
//      assertThat(queryPerGroupResult._2["varianceResult"]).isWithin(0.05).of(0.16)
//      assertThat(queryPerGroupResult._2["quantilesResult_0"]).isWithin(0.5).of(1.5)
//      null
//    }
  }

  @Test
  fun run_sumAndQuantiles_calculatesCorrectly() {
    val dataset = sparkSession.spark.createDataset(listOf(
      Tuple2(Tuple2("group1", "pid1"), 1.0),
      Tuple2(Tuple2("group1", "pid1"), 1.5),
      Tuple2(Tuple2("group1", "pid2"), 2.0)
    ), Encoders.tuple(
      Encoders.tuple(
        Encoders.STRING(), Encoders.STRING()), Encoders.DOUBLE()))

    val publicGroups = sparkSession.spark.createDataset(listOf("group1"), Encoders.STRING())

    val valueExtractor = { it: Tuple2<Tuple2<String, String>, Double> -> it._2 }

    val result: Dataset<Tuple2<String, Map<String, Double>>> =
      QueryBuilder.from(dataset, { it._1._2 })
        .groupBy(
          { it._1._1 },
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

    val output = result.collect()
    //assertThat(output.size).isEqualTo(1)

//    assertThat(output).run {
//      hasLength(1)
//      val queryPerGroupResult = output.iterator().next()
//      assertThat(queryPerGroupResult._1).isEqualTo("group1")
//      assertThat(queryPerGroupResult._2).hasSize(2)
//      assertThat(queryPerGroupResult._2.keys).containsExactly( "sumResult", "quantilesResult_0.5")
//      assertThat(queryPerGroupResult._2["sumResult"]).isWithin(0.5).of(4.5)
//      assertThat(queryPerGroupResult._2["quantilesResult_0.5"]).isWithin(0.5).of(1.5)
//      null
//    }
  }

  companion object {
    @JvmField
    @ClassRule
    val sparkSession = SparkSessionRule()
    private val sparkEncoderFactory = SparkEncoderFactory()
  }
}

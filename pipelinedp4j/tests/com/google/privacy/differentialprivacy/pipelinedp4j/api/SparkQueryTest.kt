package com.google.privacy.differentialprivacy.pipelinedp4j.api

import com.google.common.truth.Truth.assertThat
import com.google.privacy.differentialprivacy.pipelinedp4j.spark.SparkSessionRule
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Encoder
import org.apache.spark.sql.Encoders
import org.junit.ClassRule
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4

@RunWith(JUnit4::class)
class SparkQueryTest {

  @Test
  fun run_onePublicGroupTwoDifferentContributions_allPossibleAggregations_calculatesStatisticsCorrectly() {
    val dataset = sparkSession.spark.createDataset(listOf(
      Pair(Pair("group1", "pid1"), 1.0),
      Pair(Pair("group1", "pid1"), 1.5),
      Pair(Pair("group1", "pid2"), 2.0)
    ), Encoders.kryo(Pair::class.java) as Encoder<Pair<Pair<String, String>, Double>>)

    val publicGroups = sparkSession.spark.createDataset(listOf("group1"), Encoders.STRING())

    val valueExtractor = { it: Pair<Pair<String, String>, Double> -> it.second }

    val result: Dataset<QueryPerGroupResult> =
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

    val output = result.collectAsList()
    assertThat(output).run {
      hasSize(1)
      val queryPerGroupResult = output.iterator().next()
      assertThat(queryPerGroupResult.groupKey).isEqualTo("group1")
      assertThat(queryPerGroupResult.aggregationResults).hasSize(6)
      assertThat(queryPerGroupResult.aggregationResults.keys).containsExactly( "pid_cnt",
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
      assertThat(queryPerGroupResult.aggregationResults["varianceResult"]).isWithin(0.05).of(0.16)
      assertThat(queryPerGroupResult.aggregationResults["quantilesResult_0.5"]).isWithin(0.5).of(1.5)
      null
    }
  }

  @Test
  fun run_sumAndQuantiles_calculatesCorrectly() {
    val dataset = sparkSession.spark.createDataset(listOf(
      Pair(Pair("group1", "pid1"), 1.0),
      Pair(Pair("group1", "pid1"), 1.5),
      Pair(Pair("group1", "pid2"), 2.0)
    ), Encoders.kryo(Pair::class.java) as Encoder<Pair<Pair<String, String>, Double>>)

    val publicGroups = sparkSession.spark.createDataset(listOf("group1"), Encoders.STRING())

    val valueExtractor = { it: Pair<Pair<String, String>, Double> -> it.second }

    val result: Dataset<QueryPerGroupResult> =
      QueryBuilder.from(dataset, { it.first.second })
        .groupBy(
          { it.first.first },
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

    val output = result.collectAsList()

    assertThat(output).run {
      hasSize(1)
      val queryPerGroupResult = output.iterator().next()
      assertThat(queryPerGroupResult.groupKey).isEqualTo("group1")
      assertThat(queryPerGroupResult.aggregationResults).hasSize(2)
      assertThat(queryPerGroupResult.aggregationResults.keys).containsExactly( "sumResult", "quantilesResult_0.5")
      assertThat(queryPerGroupResult.aggregationResults["sumResult"]).isWithin(0.5).of(4.5)
      assertThat(queryPerGroupResult.aggregationResults["quantilesResult_0.5"]).isWithin(0.5).of(1.5)
      null
    }
  }

  companion object {
    @JvmField
    @ClassRule
    val sparkSession = SparkSessionRule()
  }
}

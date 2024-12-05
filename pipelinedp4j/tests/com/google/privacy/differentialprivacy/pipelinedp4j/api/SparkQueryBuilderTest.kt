package com.google.privacy.differentialprivacy.pipelinedp4j.api

import com.google.common.truth.Truth.assertThat
import com.google.privacy.differentialprivacy.pipelinedp4j.spark.SparkSessionRule
import kotlin.test.assertFailsWith
import org.apache.spark.sql.Encoder
import org.apache.spark.sql.Encoders
import org.junit.ClassRule
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4

@RunWith(JUnit4::class)
class SparkQueryBuilderTest {

  @Test
  fun build_sameOutputColumnNames_throwsException() {
    val dataset =
      sparkSession.spark.createDataset(
        listOf(),
        Encoders.kryo(Pair::class.java) as Encoder<Pair<Pair<String, String>, Double>>,
      )

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
    val dataset =
      sparkSession.spark.createDataset(
        listOf(),
        Encoders.kryo(Pair::class.java) as Encoder<Pair<Pair<String, String>, Double>>,
      )

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
    @JvmField @ClassRule val sparkSession = SparkSessionRule()
  }
}

/*
 * Copyright 2025 Google LLC
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
import com.google.privacy.differentialprivacy.pipelinedp4j.spark.SparkSessionRule
import kotlin.test.assertFailsWith
import org.apache.spark.api.java.function.MapFunction
import org.apache.spark.sql.Encoder
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.Row
import org.junit.ClassRule
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4
import scala.collection.mutable.Seq as ScalaSeq
import scala.jdk.CollectionConverters.SeqHasAsJava

@RunWith(JUnit4::class)
class SparkDataFrameApiTest {
  // Validation tests.

  @Test
  fun build_noNoiseWhenAggregationScheduled_throwsException() {
    val data: SparkDataFrame = createEmptyInputData()
    val queryBuilder =
      SparkDataFrameQueryBuilder.from(
          data,
          ColumnNames("privacyUnit"),
          ContributionBoundingLevel.DATASET_LEVEL(
            maxGroupsContributed = 1,
            maxContributionsPerGroup = 1,
          ),
        )
        .groupBy(ColumnNames("groupKey"), GroupsType.PrivateGroups())
        .count(outputColumnName = "sameColumnName")

    val e =
      assertFailsWith<IllegalArgumentException> {
        queryBuilder.build(TotalBudget(epsilon = 1.1, delta = 0.001))
      }
    assertThat(e)
      .hasMessageThat()
      .contains("Noise kind must be specified because aggregations are used.")
  }

  @Test
  fun build_sameOutputColumnNames_throwsException() {
    val data: SparkDataFrame = createEmptyInputData()
    val queryBuilder =
      SparkDataFrameQueryBuilder.from(
          data,
          ColumnNames("privacyUnit"),
          ContributionBoundingLevel.DATASET_LEVEL(
            maxGroupsContributed = 1,
            maxContributionsPerGroup = 1,
          ),
        )
        .groupBy(ColumnNames("groupKey"), GroupsType.PrivateGroups())
        .count(outputColumnName = "sameColumnName")
        .aggregateValue(
          "value",
          ValueAggregationsBuilder().sum(outputColumnName = "sameColumnName"),
          ContributionBounds(totalValueBounds = Bounds(minValue = 0.0, maxValue = 100.0)),
        )

    val e =
      assertFailsWith<IllegalArgumentException> {
        queryBuilder.build(TotalBudget(epsilon = 1.1, delta = 0.001), NoiseKind.LAPLACE)
      }
    assertThat(e)
      .hasMessageThat()
      .contains("There are aggregations with duplicate output column names: [sameColumnName].")
  }

  @Test
  fun build_countingDistinctPrivacyUnitsMultipleTimes_throwsException() {
    val data: SparkDataFrame = createEmptyInputData()
    val queryBuilder =
      SparkDataFrameQueryBuilder.from(
          data,
          ColumnNames("privacyUnit"),
          ContributionBoundingLevel.DATASET_LEVEL(
            maxGroupsContributed = 1,
            maxContributionsPerGroup = 1,
          ),
        )
        .groupBy(ColumnNames("groupKey"), GroupsType.PrivateGroups())
        .countDistinctPrivacyUnits(outputColumnName = "column1")
        .countDistinctPrivacyUnits(outputColumnName = "column2")

    val e =
      assertFailsWith<IllegalArgumentException> {
        queryBuilder.build(TotalBudget(epsilon = 1.1, delta = 0.001), NoiseKind.LAPLACE)
      }
    assertThat(e)
      .hasMessageThat()
      .contains("There can be at most one aggregation of counting distinct privacy units")
  }

  @Test
  fun build_countingMultipleTimes_throwsException() {
    val data: SparkDataFrame = createEmptyInputData()
    val queryBuilder =
      SparkDataFrameQueryBuilder.from(
          data,
          ColumnNames("privacyUnit"),
          ContributionBoundingLevel.DATASET_LEVEL(
            maxGroupsContributed = 1,
            maxContributionsPerGroup = 1,
          ),
        )
        .groupBy(ColumnNames("groupKey"), GroupsType.PrivateGroups())
        .count(outputColumnName = "column1")
        .count(outputColumnName = "column2")

    val e =
      assertFailsWith<IllegalArgumentException> {
        queryBuilder.build(TotalBudget(epsilon = 1.1, delta = 0.001), NoiseKind.LAPLACE)
      }
    assertThat(e).hasMessageThat().contains("There can be at most one count aggregation")
  }

  @Test
  fun build_sameValueExtractors_throwsException() {
    val data: SparkDataFrame = createEmptyInputData()
    val valueExtractor: (Row) -> Double = { it.getAs("value") }
    val queryBuilder =
      SparkDataFrameQueryBuilder.from(
          data,
          ColumnNames("privacyUnit"),
          ContributionBoundingLevel.DATASET_LEVEL(
            maxGroupsContributed = 1,
            maxContributionsPerGroup = 1,
          ),
        )
        .groupBy(ColumnNames("groupKey"), GroupsType.PrivateGroups())
        .aggregateValue(
          valueExtractor,
          ValueAggregationsBuilder().sum(outputColumnName = "sum"),
          ContributionBounds(totalValueBounds = Bounds(minValue = 0.0, maxValue = 100.0)),
        )
        .aggregateValue(
          valueExtractor,
          ValueAggregationsBuilder().mean(outputColumnName = "mean"),
          ContributionBounds(valueBounds = Bounds(minValue = 0.0, maxValue = 10.0)),
        )

    val e =
      assertFailsWith<IllegalArgumentException> {
        queryBuilder.build(TotalBudget(epsilon = 1.1, delta = 0.001), NoiseKind.LAPLACE)
      }
    assertThat(e)
      .hasMessageThat()
      .contains(
        "There are the same (object reference equality) value extractors used in different aggregateValue() calls"
      )
  }

  @Test
  fun build_sameValueColumnNames_throwsException() {
    val data: SparkDataFrame = createEmptyInputData()
    val queryBuilder =
      SparkDataFrameQueryBuilder.from(
          data,
          ColumnNames("privacyUnit"),
          ContributionBoundingLevel.DATASET_LEVEL(
            maxGroupsContributed = 1,
            maxContributionsPerGroup = 1,
          ),
        )
        .groupBy(ColumnNames("groupKey"), GroupsType.PrivateGroups())
        .aggregateValue(
          "value",
          ValueAggregationsBuilder().sum(outputColumnName = "sum"),
          ContributionBounds(totalValueBounds = Bounds(minValue = 0.0, maxValue = 100.0)),
        )
        .aggregateValue(
          "value",
          ValueAggregationsBuilder().mean(outputColumnName = "mean"),
          ContributionBounds(valueBounds = Bounds(minValue = 0.0, maxValue = 10.0)),
        )

    val e =
      assertFailsWith<IllegalArgumentException> {
        queryBuilder.build(TotalBudget(epsilon = 1.1, delta = 0.001), NoiseKind.LAPLACE)
      }
    assertThat(e)
      .hasMessageThat()
      .contains("The same value column is used in different aggregateValue() calls")
  }

  @Test
  fun build_sameAggregationsPerValue_throwsException() {
    val data: SparkDataFrame = createEmptyInputData()
    val queryBuilder =
      SparkDataFrameQueryBuilder.from(
          data,
          ColumnNames("privacyUnit"),
          ContributionBoundingLevel.DATASET_LEVEL(
            maxGroupsContributed = 1,
            maxContributionsPerGroup = 1,
          ),
        )
        .groupBy(ColumnNames("groupKey"), GroupsType.PrivateGroups())
        .aggregateValue(
          "value",
          ValueAggregationsBuilder()
            .variance(outputColumnName = "variance")
            .variance(outputColumnName = "variance2"),
          ContributionBounds(valueBounds = Bounds(minValue = 0.0, maxValue = 10.0)),
        )

    val e =
      assertFailsWith<IllegalArgumentException> {
        queryBuilder.build(TotalBudget(epsilon = 1.1, delta = 0.001), NoiseKind.LAPLACE)
      }
    assertThat(e)
      .hasMessageThat()
      .contains("There are duplicate aggregations for the same value: [VARIANCE].")
  }

  @Test
  fun build_sumMeanAndVarianceWithTotalValueBounds_throwsException() {
    val data: SparkDataFrame = createEmptyInputData()
    val queryBuilder =
      SparkDataFrameQueryBuilder.from(
          data,
          ColumnNames("privacyUnit"),
          ContributionBoundingLevel.DATASET_LEVEL(
            maxGroupsContributed = 1,
            maxContributionsPerGroup = 1,
          ),
        )
        .groupBy(ColumnNames("groupKey"), GroupsType.PrivateGroups())
        .aggregateValue(
          "value",
          ValueAggregationsBuilder()
            .sum(outputColumnName = "sum")
            .mean(outputColumnName = "mean")
            .variance(outputColumnName = "variance"),
          ContributionBounds(
            totalValueBounds = Bounds(minValue = 0.0, maxValue = 100.0),
            valueBounds = Bounds(minValue = 0.0, maxValue = 10.0),
          ),
        )

    val e =
      assertFailsWith<IllegalArgumentException> {
        queryBuilder.build(TotalBudget(epsilon = 1.1, delta = 0.001), NoiseKind.LAPLACE)
      }
    assertThat(e)
      .hasMessageThat()
      .contains(
        "Total value bounds should not be set if SUM is calculated together with MEAN or VARIANCE."
      )
  }

  @Test
  fun build_sumWithoutMeanAndVarianceAndWithoutTotalValueBounds_throwsException() {
    val data: SparkDataFrame = createEmptyInputData()
    val queryBuilder =
      SparkDataFrameQueryBuilder.from(
          data,
          ColumnNames("privacyUnit"),
          ContributionBoundingLevel.DATASET_LEVEL(
            maxGroupsContributed = 1,
            maxContributionsPerGroup = 1,
          ),
        )
        .groupBy(ColumnNames("groupKey"), GroupsType.PrivateGroups())
        .aggregateValue(
          "value",
          ValueAggregationsBuilder().sum(outputColumnName = "sum"),
          ContributionBounds(),
        )

    val e =
      assertFailsWith<IllegalArgumentException> {
        queryBuilder.build(TotalBudget(epsilon = 1.1, delta = 0.001), NoiseKind.LAPLACE)
      }
    assertThat(e)
      .hasMessageThat()
      .contains(
        "Total value bounds should be set if SUM is calculated without calculating MEAN or VARIANCE."
      )
  }

  @Test
  fun build_noSumButTotalValueBoundsAreProvided_throwsException() {
    val data: SparkDataFrame = createEmptyInputData()
    val queryBuilder =
      SparkDataFrameQueryBuilder.from(
          data,
          ColumnNames("privacyUnit"),
          ContributionBoundingLevel.DATASET_LEVEL(
            maxGroupsContributed = 1,
            maxContributionsPerGroup = 1,
          ),
        )
        .groupBy(ColumnNames("groupKey"), GroupsType.PrivateGroups())
        .aggregateValue(
          "value",
          ValueAggregationsBuilder(),
          ContributionBounds(totalValueBounds = Bounds(minValue = 0.0, maxValue = 100.0)),
        )

    val e =
      assertFailsWith<IllegalArgumentException> {
        queryBuilder.build(TotalBudget(epsilon = 1.1, delta = 0.001), NoiseKind.LAPLACE)
      }
    assertThat(e)
      .hasMessageThat()
      .contains("Total value bounds should not be set because SUM is not calculated.")
  }

  @Test
  fun build_meanAndVarianceAndQuantilesWithoutValueBounds_throwsException() {
    val data: SparkDataFrame = createEmptyInputData()
    val queryBuilder =
      SparkDataFrameQueryBuilder.from(
          data,
          ColumnNames("privacyUnit"),
          ContributionBoundingLevel.DATASET_LEVEL(
            maxGroupsContributed = 1,
            maxContributionsPerGroup = 1,
          ),
        )
        .groupBy(ColumnNames("groupKey"), GroupsType.PrivateGroups())
        .aggregateValue(
          "value",
          ValueAggregationsBuilder()
            .mean(outputColumnName = "mean")
            .variance(outputColumnName = "variance")
            .quantiles(outputColumnName = "quantiles", ranks = listOf()),
          ContributionBounds(),
        )

    val e =
      assertFailsWith<IllegalArgumentException> {
        queryBuilder.build(TotalBudget(epsilon = 1.1, delta = 0.001), NoiseKind.LAPLACE)
      }
    assertThat(e)
      .hasMessageThat()
      .contains("Total value bounds should be set if MEAN, VARIANCE or QUANTILES are calculated.")
  }

  @Test
  fun build_noMeanNoVarianceAndNoQuantilesButValueBoundsProvided_throwsException() {
    val data: SparkDataFrame = createEmptyInputData()
    val queryBuilder =
      SparkDataFrameQueryBuilder.from(
          data,
          ColumnNames("privacyUnit"),
          ContributionBoundingLevel.DATASET_LEVEL(
            maxGroupsContributed = 1,
            maxContributionsPerGroup = 1,
          ),
        )
        .groupBy(ColumnNames("groupKey"), GroupsType.PrivateGroups())
        .aggregateValue(
          "value",
          ValueAggregationsBuilder().sum(outputColumnName = "sum"),
          ContributionBounds(
            totalValueBounds = Bounds(minValue = 0.0, maxValue = 100.0),
            valueBounds = Bounds(minValue = 0.0, maxValue = 10.0),
          ),
        )

    val e =
      assertFailsWith<IllegalArgumentException> {
        queryBuilder.build(TotalBudget(epsilon = 1.1, delta = 0.001), NoiseKind.LAPLACE)
      }
    assertThat(e)
      .hasMessageThat()
      .contains(
        "Value bounds are not needed if MEAN, VARIANCE or QUANTILES are not calculated. Therefore they should not be set."
      )
  }

  @Test
  fun build_sameVectorExtractors_throwsException() {
    val data: SparkDataFrame = createEmptyInputData()
    val vectorExtractor: (Row) -> List<Double> = {
      listOf(it.getAs("value"), it.getAs("anotherValue"))
    }
    val queryBuilder =
      SparkDataFrameQueryBuilder.from(
          data,
          ColumnNames("privacyUnit"),
          ContributionBoundingLevel.DATASET_LEVEL(
            maxGroupsContributed = 1,
            maxContributionsPerGroup = 1,
          ),
        )
        .groupBy(ColumnNames("groupKey"), GroupsType.PrivateGroups())
        .aggregateVector(
          vectorExtractor,
          vectorSize = 2,
          VectorAggregationsBuilder().vectorSum(outputColumnName = "vectorSum"),
          VectorContributionBounds(
            maxVectorTotalNorm = VectorNorm(normKind = NormKind.L2, value = 10.0)
          ),
        )
        .aggregateVector(
          vectorExtractor,
          vectorSize = 2,
          VectorAggregationsBuilder().vectorSum(outputColumnName = "vectorSum2"),
          VectorContributionBounds(
            maxVectorTotalNorm = VectorNorm(normKind = NormKind.L_INF, value = 100.0)
          ),
        )

    val e =
      assertFailsWith<IllegalArgumentException> {
        queryBuilder.build(TotalBudget(epsilon = 1.1, delta = 0.001), NoiseKind.LAPLACE)
      }
    assertThat(e)
      .hasMessageThat()
      .contains(
        "There are the same (object reference equality) vector extractors used in different aggregateVector() calls"
      )
  }

  @Test
  fun build_sameVectorColumnNames_throwsException() {
    val data: SparkDataFrame = createEmptyInputData()
    val queryBuilder =
      SparkDataFrameQueryBuilder.from(
          data,
          ColumnNames("privacyUnit"),
          ContributionBoundingLevel.DATASET_LEVEL(
            maxGroupsContributed = 1,
            maxContributionsPerGroup = 1,
          ),
        )
        .groupBy(ColumnNames("groupKey"), GroupsType.PrivateGroups())
        .aggregateVector(
          ColumnNames("value", "anotherValue"),
          vectorSize = 2,
          VectorAggregationsBuilder().vectorSum(outputColumnName = "vectorSum"),
          VectorContributionBounds(
            maxVectorTotalNorm = VectorNorm(normKind = NormKind.L2, value = 10.0)
          ),
        )
        .aggregateVector(
          ColumnNames("anotherValue", "value"),
          vectorSize = 2,
          VectorAggregationsBuilder().vectorSum(outputColumnName = "vectorSum2"),
          VectorContributionBounds(
            maxVectorTotalNorm = VectorNorm(normKind = NormKind.L_INF, value = 100.0)
          ),
        )

    val e =
      assertFailsWith<IllegalArgumentException> {
        queryBuilder.build(TotalBudget(epsilon = 1.1, delta = 0.001), NoiseKind.LAPLACE)
      }
    assertThat(e)
      .hasMessageThat()
      .contains("The same vector column names are used in different aggregateVector() calls")
  }

  @Test
  fun build_sameAggregationsPerVector_throwsException() {
    val data: SparkDataFrame = createEmptyInputData()
    val queryBuilder =
      SparkDataFrameQueryBuilder.from(
          data,
          ColumnNames("privacyUnit"),
          ContributionBoundingLevel.DATASET_LEVEL(
            maxGroupsContributed = 1,
            maxContributionsPerGroup = 1,
          ),
        )
        .groupBy(ColumnNames("groupKey"), GroupsType.PrivateGroups())
        .aggregateVector(
          ColumnNames("value", "anotherValue"),
          vectorSize = 2,
          VectorAggregationsBuilder()
            .vectorSum(outputColumnName = "vectorSum")
            .vectorSum(outputColumnName = "vectorSum2"),
          VectorContributionBounds(
            maxVectorTotalNorm = VectorNorm(normKind = NormKind.L_INF, value = 100.0)
          ),
        )

    val e =
      assertFailsWith<IllegalArgumentException> {
        queryBuilder.build(TotalBudget(epsilon = 1.1, delta = 0.001), NoiseKind.LAPLACE)
      }
    assertThat(e)
      .hasMessageThat()
      .contains("There are duplicate aggregations for the same vector: [VECTOR_SUM].")
  }

  @Test
  fun build_l2VectorNormKindWithLaplaceNoise_throwsException() {
    val data: SparkDataFrame = createEmptyInputData()
    val queryBuilder =
      SparkDataFrameQueryBuilder.from(
          data,
          ColumnNames("privacyUnit"),
          ContributionBoundingLevel.DATASET_LEVEL(
            maxGroupsContributed = 1,
            maxContributionsPerGroup = 1,
          ),
        )
        .groupBy(ColumnNames("groupKey"), GroupsType.PrivateGroups())
        .aggregateVector(
          ColumnNames("value", "anotherValue"),
          vectorSize = 2,
          VectorAggregationsBuilder().vectorSum(outputColumnName = "vectorSum"),
          VectorContributionBounds(
            maxVectorTotalNorm = VectorNorm(normKind = NormKind.L2, value = 100.0)
          ),
        )

    val e =
      assertFailsWith<IllegalArgumentException> {
        queryBuilder.build(TotalBudget(epsilon = 1.1, delta = 0.001), NoiseKind.LAPLACE)
      }
    assertThat(e)
      .hasMessageThat()
      .contains("Norm kind must be L_INF or L1 when Laplace mechanism is used.")
  }

  @Test
  fun build_l1VectorNormKindWithGaussianNoise_throwsException() {
    val data: SparkDataFrame = createEmptyInputData()
    val queryBuilder =
      SparkDataFrameQueryBuilder.from(
          data,
          ColumnNames("privacyUnit"),
          ContributionBoundingLevel.DATASET_LEVEL(
            maxGroupsContributed = 1,
            maxContributionsPerGroup = 1,
          ),
        )
        .groupBy(ColumnNames("groupKey"), GroupsType.PrivateGroups())
        .aggregateVector(
          ColumnNames("value", "anotherValue"),
          vectorSize = 2,
          VectorAggregationsBuilder().vectorSum(outputColumnName = "vectorSum"),
          VectorContributionBounds(
            maxVectorTotalNorm = VectorNorm(normKind = NormKind.L1, value = 100.0)
          ),
        )

    val e =
      assertFailsWith<IllegalArgumentException> {
        queryBuilder.build(TotalBudget(epsilon = 1.1, delta = 0.001), NoiseKind.GAUSSIAN)
      }
    assertThat(e)
      .hasMessageThat()
      .contains("Norm kind must be L_INF or L2 when Gaussian mechanism is used.")
  }

  @Test
  fun build_zeroEpsilonInTotalBudget_throwsException() {
    val data: SparkDataFrame = createEmptyInputData()
    val queryBuilder =
      SparkDataFrameQueryBuilder.from(
          data,
          ColumnNames("privacyUnit"),
          ContributionBoundingLevel.DATASET_LEVEL(
            maxGroupsContributed = 1,
            maxContributionsPerGroup = 1,
          ),
        )
        .groupBy(ColumnNames("groupKey"), GroupsType.PrivateGroups())

    val e =
      assertFailsWith<IllegalArgumentException> {
        queryBuilder.build(TotalBudget(epsilon = 0.0, delta = 0.001), NoiseKind.LAPLACE)
      }
    assertThat(e).hasMessageThat().contains("Epsilon in the total budget must be positive.")
  }

  @Test
  fun build_zeroEpsilonInAggregationBudget_throwsException() {
    val data: SparkDataFrame = createEmptyInputData()
    val queryBuilder =
      SparkDataFrameQueryBuilder.from(
          data,
          ColumnNames("privacyUnit"),
          ContributionBoundingLevel.DATASET_LEVEL(
            maxGroupsContributed = 1,
            maxContributionsPerGroup = 1,
          ),
        )
        .groupBy(ColumnNames("groupKey"), GroupsType.PrivateGroups())
        .count(
          outputColumnName = "count",
          budget = AbsoluteBudgetPerOpSpec(epsilon = 0.0, delta = 0.0),
        )

    val e =
      assertFailsWith<IllegalArgumentException> {
        queryBuilder.build(TotalBudget(epsilon = 1.1, delta = 0.001), NoiseKind.LAPLACE)
      }
    assertThat(e).hasMessageThat().contains("Epsilon in the aggregation budget must be positive.")
  }

  @Test
  fun build_zeroDeltaInTotalBudgetWithGaussianNoise_throwsException() {
    val data: SparkDataFrame = createEmptyInputData()
    val queryBuilder =
      SparkDataFrameQueryBuilder.from(
          data,
          ColumnNames("privacyUnit"),
          ContributionBoundingLevel.DATASET_LEVEL(
            maxGroupsContributed = 1,
            maxContributionsPerGroup = 1,
          ),
        )
        .groupBy(ColumnNames("groupKey"), GroupsType.PrivateGroups())

    val e =
      assertFailsWith<IllegalArgumentException> {
        queryBuilder.build(TotalBudget(epsilon = 1.1, delta = 0.0), NoiseKind.GAUSSIAN)
      }
    assertThat(e)
      .hasMessageThat()
      .contains("Delta in the total budget must be positive when Gaussian noise is used.")
  }

  @Test
  fun build_zeroDeltaInAggregationBudgetWithGaussianNoise_throwsException() {
    val data: SparkDataFrame = createEmptyInputData()
    val queryBuilder =
      SparkDataFrameQueryBuilder.from(
          data,
          ColumnNames("privacyUnit"),
          ContributionBoundingLevel.DATASET_LEVEL(
            maxGroupsContributed = 1,
            maxContributionsPerGroup = 1,
          ),
        )
        .groupBy(ColumnNames("groupKey"), GroupsType.PrivateGroups())
        .count(
          outputColumnName = "count",
          budget = AbsoluteBudgetPerOpSpec(epsilon = 1.1, delta = 0.0),
        )

    val e =
      assertFailsWith<IllegalArgumentException> {
        queryBuilder.build(TotalBudget(epsilon = 1.1, delta = 0.001), NoiseKind.GAUSSIAN)
      }
    assertThat(e)
      .hasMessageThat()
      .contains("Delta in the aggregation budget must be positive when Gaussian noise is used.")
  }

  @Test
  fun build_zeroDeltaInPrivateGroupSelectionAbsoluteBudget_throwsException() {
    val data: SparkDataFrame = createEmptyInputData()
    val queryBuilder =
      SparkDataFrameQueryBuilder.from(
          data,
          ColumnNames("privacyUnit"),
          ContributionBoundingLevel.DATASET_LEVEL(
            maxGroupsContributed = 1,
            maxContributionsPerGroup = 1,
          ),
        )
        .groupBy(
          ColumnNames("groupKey"),
          GroupsType.PrivateGroups(budget = AbsoluteBudgetPerOpSpec(epsilon = 1.1, delta = 0.0)),
        )

    val e =
      assertFailsWith<IllegalArgumentException> {
        queryBuilder.build(TotalBudget(epsilon = 1.1, delta = 0.001), NoiseKind.LAPLACE)
      }
    assertThat(e)
      .hasMessageThat()
      .contains("Delta in the budget for private group selection must be positive.")
  }

  @Test
  fun build_positiveEpsilonInPrivateGroupSelectionBudgetWhenCountingDistinctPrivacyUnits_throwsException() {
    val data: SparkDataFrame = createEmptyInputData()
    val queryBuilder =
      SparkDataFrameQueryBuilder.from(
          data,
          ColumnNames("privacyUnit"),
          ContributionBoundingLevel.DATASET_LEVEL(
            maxGroupsContributed = 1,
            maxContributionsPerGroup = 1,
          ),
        )
        .groupBy(
          ColumnNames("groupKey"),
          GroupsType.PrivateGroups(budget = AbsoluteBudgetPerOpSpec(epsilon = 0.5, delta = 0.001)),
        )
        .countDistinctPrivacyUnits(
          outputColumnName = "privacyIds",
          budget = AbsoluteBudgetPerOpSpec(epsilon = 0.6, delta = 0.0),
        )

    val e =
      assertFailsWith<IllegalArgumentException> {
        queryBuilder.build(TotalBudget(epsilon = 1.1, delta = 0.001), NoiseKind.LAPLACE)
      }
    assertThat(e)
      .hasMessageThat()
      .contains(
        "Epsilon in the budget for private group selection must be zero when counting distinct privacy units"
      )
  }

  @Test
  fun build_zeroEpsilonInPrivateGroupSelectionBudgetWhenNotCountingDistinctPrivacyUnits_throwsException() {
    val data: SparkDataFrame = createEmptyInputData()
    val queryBuilder =
      SparkDataFrameQueryBuilder.from(
          data,
          ColumnNames("privacyUnit"),
          ContributionBoundingLevel.DATASET_LEVEL(
            maxGroupsContributed = 1,
            maxContributionsPerGroup = 1,
          ),
        )
        .groupBy(
          ColumnNames("groupKey"),
          GroupsType.PrivateGroups(budget = AbsoluteBudgetPerOpSpec(epsilon = 0.0, delta = 0.001)),
        )

    val e =
      assertFailsWith<IllegalArgumentException> {
        queryBuilder.build(TotalBudget(epsilon = 1.1, delta = 0.001), NoiseKind.LAPLACE)
      }
    assertThat(e)
      .hasMessageThat()
      .contains(
        "Epsilon in the budget for private group selection must be positive when not counting distinct privacy units."
      )
  }

  @Test
  fun build_zeroDeltaInTotalBudgetWhenPrivateGroupSelectionIsUsed_throwsException() {
    val data: SparkDataFrame = createEmptyInputData()
    val queryBuilder =
      SparkDataFrameQueryBuilder.from(
          data,
          ColumnNames("privacyUnit"),
          ContributionBoundingLevel.DATASET_LEVEL(
            maxGroupsContributed = 1,
            maxContributionsPerGroup = 1,
          ),
        )
        .groupBy(ColumnNames("groupKey"), GroupsType.PrivateGroups())
        .count(outputColumnName = "count")

    val e =
      assertFailsWith<IllegalArgumentException> {
        queryBuilder.build(TotalBudget(epsilon = 1.1, delta = 0.0), NoiseKind.LAPLACE)
      }
    assertThat(e)
      .hasMessageThat()
      .contains("Delta in the total budget must be positive when private group selection is used.")
  }

  @Test
  fun build_groupSelectionOnlyAndGroupsArePublic_throwsException() {
    val data: SparkDataFrame = createEmptyInputData()
    val queryBuilder =
      SparkDataFrameQueryBuilder.from(
          data,
          ColumnNames("privacyUnit"),
          ContributionBoundingLevel.DATASET_LEVEL(
            maxGroupsContributed = 1,
            maxContributionsPerGroup = 1,
          ),
        )
        .groupBy(ColumnNames("groupKey"), GroupsType.PublicGroups.create(sequenceOf()))

    val e =
      assertFailsWith<IllegalArgumentException> {
        queryBuilder.build(TotalBudget(epsilon = 1.1, delta = 0.0), NoiseKind.LAPLACE)
      }
    assertThat(e)
      .hasMessageThat()
      .contains("There are no aggregations, therefore public groups do not make any sense.")
  }

  // Execution tests.

  @Test
  fun run_groupSelection_selectsGroupsCorrectly() {
    val data =
      createInputData(
        listOf(
          TestDataRow("group1", "pid1", 1.0),
          TestDataRow("group1", "pid1", 2.0),
          TestDataRow("group1", "pid2", 3.0),
          TestDataRow("group1", "pid2", 4.0),
          TestDataRow("group2", "pid1", 1.0),
          TestDataRow("group2", "pid1", 2.0),
          TestDataRow("group2", "pid2", 3.0),
          TestDataRow("group2", "pid1", 4.0),
        )
      )
    val query =
      SparkDataFrameQueryBuilder.from(
          data,
          ColumnNames("privacyUnit"),
          ContributionBoundingLevel.DATASET_LEVEL(
            maxGroupsContributed = 2,
            maxContributionsPerGroup = 2,
          ),
        )
        .groupBy(ColumnNames("groupKey"), GroupsType.PrivateGroups())
        .build(TotalBudget(epsilon = 10000000.0, delta = 0.99999999))

    val result: SparkDataFrame = query.run()

    val expected =
      listOf(
        QueryPerGroupResultWithTolerance(
          "group1",
          valueAggregationResults = mapOf(),
          vectorAggregationResults = mapOf(),
        ),
        QueryPerGroupResultWithTolerance(
          "group2",
          valueAggregationResults = mapOf(),
          vectorAggregationResults = mapOf(),
        ),
      )
    assertEquals(result, expected)
  }

  @Test
  fun run_publicGroups_allPossibleValueAggregations_calculatesStatisticsCorrectly() {
    val data =
      createInputData(
        listOf(
          TestDataRow("group1", "pid1", 1.0),
          TestDataRow("group1", "pid1", 1.5),
          TestDataRow("group1", "pid2", 2.0),
          TestDataRow("nonPublicGroup", "pid2", 2.0),
        )
      )
    val publicGroups = createPublicGroups(listOf("group1"))
    val query =
      SparkDataFrameQueryBuilder.from(
          data,
          ColumnNames("privacyUnit"),
          ContributionBoundingLevel.DATASET_LEVEL(
            maxGroupsContributed = 1,
            maxContributionsPerGroup = 2,
          ),
        )
        .groupBy(ColumnNames("groupKey"), GroupsType.PublicGroups.createForDataFrame(publicGroups))
        .countDistinctPrivacyUnits("pidCnt")
        .count("cnt")
        .aggregateValue(
          "value",
          ValueAggregationsBuilder()
            .sum("sumResult")
            .mean("meanResult")
            .variance("varianceResult")
            .quantiles(ranks = listOf(0.5), outputColumnName = "quantilesResult"),
          ContributionBounds(valueBounds = Bounds(minValue = 1.0, maxValue = 2.0)),
        )
        .build(TotalBudget(epsilon = 1000.0), NoiseKind.LAPLACE)

    val result: SparkDataFrame = query.run()

    val expected =
      listOf(
        QueryPerGroupResultWithTolerance(
          "group1",
          mapOf(
            "pidCnt" to DoubleWithTolerance(value = 2.0, tolerance = 0.5),
            "cnt" to DoubleWithTolerance(value = 3.0, tolerance = 0.5),
            "sumResult" to DoubleWithTolerance(value = 4.5, tolerance = 0.5),
            "meanResult" to DoubleWithTolerance(value = 1.5, tolerance = 0.5),
            // (1^2+(1.5)^2+2^2)/3-((1.0+1.5+2)/3)^2 = 0.1(6)
            "varianceResult" to DoubleWithTolerance(value = 0.16, tolerance = 0.05),
            "quantilesResult_0.5" to DoubleWithTolerance(value = 1.5, tolerance = 0.5),
          ),
          vectorAggregationResults = mapOf(),
        )
      )
    assertEquals(result, expected)
  }

  @Test
  fun run_publicGroups_allPossibleVectorAggregations_calculatesStatisticsCorrectly() {
    val data =
      createInputData(
        listOf(
          TestDataRow("group1", "pid1", 1.0, 2.0),
          TestDataRow("group1", "pid1", 0.5, 2.5),
          TestDataRow("group1", "pid2", 1.0, 0.0),
          TestDataRow("nonPublicGroup", "pid2", 3.0),
        )
      )
    val publicGroups = createPublicGroups(listOf("group1"))
    val query =
      SparkDataFrameQueryBuilder.from(
          data,
          ColumnNames("privacyUnit"),
          ContributionBoundingLevel.DATASET_LEVEL(
            maxGroupsContributed = 1,
            maxContributionsPerGroup = 2,
          ),
        )
        .groupBy(ColumnNames("groupKey"), GroupsType.PublicGroups.createForDataFrame(publicGroups))
        .countDistinctPrivacyUnits("pidCnt")
        .count("cnt")
        .aggregateVector(
          ColumnNames("value", "anotherValue"),
          vectorSize = 2,
          VectorAggregationsBuilder().vectorSum("vectorSumResult"),
          VectorContributionBounds(
            maxVectorTotalNorm = VectorNorm(normKind = NormKind.L_INF, value = 2.0)
          ),
        )
        .build(TotalBudget(epsilon = 1000.0), NoiseKind.LAPLACE)

    val result: SparkDataFrame = query.run()

    val expected =
      listOf(
        QueryPerGroupResultWithTolerance(
          "group1",
          mapOf(
            "pidCnt" to DoubleWithTolerance(value = 2.0, tolerance = 0.5),
            "cnt" to DoubleWithTolerance(value = 3.0, tolerance = 0.5),
          ),
          mapOf(
            // pid1: (1.0, 2.0) + (0.5, 2.5) = (1.5, 4.5), L_INF norm is 4.5 =>
            // clip it (1.5, 2.0).
            // pid2: (1.0, 0.0), L_INF norm is 1.0 => no clipping.
            // result: (1.5, 2.0) + (1.0, 0.0) = (2.5, 2.0)
            "vectorSumResult" to
              listOf(
                DoubleWithTolerance(value = 2.5, tolerance = 0.5),
                DoubleWithTolerance(value = 2.0, tolerance = 0.5),
              )
          ),
        )
      )
    assertEquals(result, expected)
  }

  @Test
  fun run_publicGroups_multipleValueAndVectorAggregations_calculatesStatisticsCorrectly() {
    val data =
      createInputData(
        listOf(
          TestDataRow("group1", "pid1", 1.0, 2.0),
          TestDataRow("group1", "pid1", 0.5, 2.5),
          TestDataRow("group1", "pid2", 1.0, 0.0),
          TestDataRow("nonPublicGroup", "pid2", 3.0),
        )
      )
    val publicGroups = createPublicGroups(listOf("group1"))
    val query =
      SparkDataFrameQueryBuilder.from(
          data,
          ColumnNames("privacyUnit"),
          ContributionBoundingLevel.DATASET_LEVEL(
            maxGroupsContributed = 1,
            maxContributionsPerGroup = 2,
          ),
        )
        .groupBy(ColumnNames("groupKey"), GroupsType.PublicGroups.createForDataFrame(publicGroups))
        .countDistinctPrivacyUnits("pidCnt")
        .count("cnt")
        .aggregateValue(
          "value",
          ValueAggregationsBuilder().sum("sumValue"),
          ContributionBounds(totalValueBounds = Bounds(1.0, 2.0)),
        )
        .aggregateValue(
          "anotherValue",
          ValueAggregationsBuilder().sum("sumAnotherValue"),
          ContributionBounds(totalValueBounds = Bounds(0.0, 3.0)),
        )
        .aggregateVector(
          ColumnNames("value", "anotherValue"),
          vectorSize = 2,
          VectorAggregationsBuilder().vectorSum("vectorSumResult"),
          VectorContributionBounds(
            maxVectorTotalNorm = VectorNorm(normKind = NormKind.L_INF, value = 2.0)
          ),
        )
        .build(TotalBudget(epsilon = 1000.0), NoiseKind.LAPLACE)

    val result: SparkDataFrame = query.run()

    // sumValue: pid1 contributes 1.5, pid2 contributes 1.0. Total 2.5
    // sumAnotherValue:
    // pid1 contributes 2.0 + 2.5 = 4.5. Bounded by [0.0, 3.0], so clipped to 3.0
    // pid2 contributes 0.0. Bounded by [0.0, 3.0], so it is 0.0
    // Total sumAnotherValue = 3.0 + 0.0 = 3.0

    val expected =
      listOf(
        QueryPerGroupResultWithTolerance(
          "group1",
          mapOf(
            "pidCnt" to DoubleWithTolerance(value = 2.0, tolerance = 0.5),
            "cnt" to DoubleWithTolerance(value = 3.0, tolerance = 0.5),
            "sumValue" to DoubleWithTolerance(value = 2.5, tolerance = 0.5),
            "sumAnotherValue" to DoubleWithTolerance(value = 3.0, tolerance = 0.5),
          ),
          mapOf(
            "vectorSumResult" to
              listOf(
                DoubleWithTolerance(value = 2.5, tolerance = 0.5),
                DoubleWithTolerance(value = 2.0, tolerance = 0.5),
              )
          ),
        )
      )
    assertEquals(result, expected)
  }

  @Test
  fun run_privateGroups_multipleValueAndVectorAggregations_calculatesStatisticsCorrectly() {
    val data =
      createInputData(
        listOf(
          TestDataRow("group1", "pid1", 1.0, 2.0),
          TestDataRow("group1", "pid1", 0.5, 2.5),
          TestDataRow("group1", "pid2", 1.0, 0.0),
          TestDataRow("nonPublicGroup", "pid2", 3.0),
        )
      )
    val query =
      SparkDataFrameQueryBuilder.from(
          data,
          ColumnNames("privacyUnit"),
          ContributionBoundingLevel.DATASET_LEVEL(
            maxGroupsContributed = 2,
            maxContributionsPerGroup = 2,
          ),
        )
        .groupBy(ColumnNames("groupKey"), GroupsType.PrivateGroups())
        .countDistinctPrivacyUnits("pidCnt")
        .count("cnt")
        .aggregateValue(
          "value",
          ValueAggregationsBuilder().sum("sumValue"),
          ContributionBounds(totalValueBounds = Bounds(1.0, 2.0)),
        )
        .aggregateValue(
          "anotherValue",
          ValueAggregationsBuilder().sum("sumAnotherValue"),
          ContributionBounds(totalValueBounds = Bounds(0.0, 3.0)),
        )
        .aggregateVector(
          ColumnNames("value", "anotherValue"),
          vectorSize = 2,
          VectorAggregationsBuilder().vectorSum("vectorSumResult"),
          VectorContributionBounds(
            maxVectorTotalNorm = VectorNorm(normKind = NormKind.L_INF, value = 2.0)
          ),
        )
        .build(TotalBudget(epsilon = 3500.0, delta = 0.001), NoiseKind.LAPLACE)

    val result: SparkDataFrame = query.run()

    // sumValue:
    // group1: pid1 contributes 1.5 (in [1,2]), pid2 contributes 1.0 (in [1,2]). Total 2.5.
    // nonPublicGroup: pid2 contributes 3.0, clipped to 2.0 by [1,2]. Total 2.0.
    // sumAnotherValue:
    // group1: pid1 contributes 4.5, clipped to 3.0 by [0,3]. pid2 contributes 0.0 (in [0,3]). Total
    // 3.0.
    // nonPublicGroup: pid2 contributes 0.0 (in [0,3]). Total 0.0.
    // vectorSumResult:
    // group1: pid1 contributes (1.5, 4.5), L_INF-clipped to (1.5, 2.0). pid2 contributes (1.0,
    // 0.0),
    // not clipped.
    // Total for group1: (1.5, 2.0) + (1.0, 0.0) = (2.5, 2.0).
    // nonPublicGroup: pid2 contributes (3.0, 0.0), L_INF-clipped to (2.0, 0.0).
    val expected =
      listOf(
        QueryPerGroupResultWithTolerance(
          "group1",
          mapOf(
            "pidCnt" to DoubleWithTolerance(value = 2.0, tolerance = 0.5),
            "cnt" to DoubleWithTolerance(value = 3.0, tolerance = 0.5),
            "sumValue" to DoubleWithTolerance(value = 2.5, tolerance = 0.5),
            "sumAnotherValue" to DoubleWithTolerance(value = 3.0, tolerance = 0.5),
          ),
          mapOf(
            "vectorSumResult" to
              listOf(
                DoubleWithTolerance(value = 2.5, tolerance = 0.5),
                DoubleWithTolerance(value = 2.0, tolerance = 0.5),
              )
          ),
        )
      )
    assertEquals(result, expected)
  }

  @Test
  fun run_privateGroups_noPidCount_multipleValueAndVectorAggregations_calculatesStatisticsCorrectly() {
    val data =
      createInputData(
        listOf(
          TestDataRow("group1", "pid1", 1.0, 2.0),
          TestDataRow("group1", "pid1", 0.5, 2.5),
          TestDataRow("group1", "pid2", 1.0, 0.0),
          TestDataRow("nonPublicGroup", "pid2", 3.0),
        )
      )
    val query =
      SparkDataFrameQueryBuilder.from(
          data,
          ColumnNames("privacyUnit"),
          ContributionBoundingLevel.DATASET_LEVEL(
            maxGroupsContributed = 2,
            maxContributionsPerGroup = 2,
          ),
        )
        .groupBy(ColumnNames("groupKey"), GroupsType.PrivateGroups())
        .count("cnt")
        .aggregateValue(
          "value",
          ValueAggregationsBuilder().sum("sumValue"),
          ContributionBounds(totalValueBounds = Bounds(1.0, 2.0)),
        )
        .aggregateValue(
          "anotherValue",
          ValueAggregationsBuilder().sum("sumAnotherValue"),
          ContributionBounds(totalValueBounds = Bounds(0.0, 3.0)),
        )
        .aggregateVector(
          ColumnNames("value", "anotherValue"),
          vectorSize = 2,
          VectorAggregationsBuilder().vectorSum("vectorSumResult"),
          VectorContributionBounds(
            maxVectorTotalNorm = VectorNorm(normKind = NormKind.L_INF, value = 2.0)
          ),
        )
        .build(TotalBudget(epsilon = 3500.0, delta = 0.001), NoiseKind.LAPLACE)

    val result: SparkDataFrame = query.run()

    // sumValue:
    // group1: pid1 contributes 1.5 (in [1,2]), pid2 contributes 1.0 (in [1,2]). Total 2.5.
    // nonPublicGroup: pid2 contributes 3.0, clipped to 2.0 by [1,2]. Total 2.0.
    // sumAnotherValue:
    // group1: pid1 contributes 4.5, clipped to 3.0 by [0,3]. pid2 contributes 0.0 (in [0,3]). Total
    // 3.0.
    // nonPublicGroup: pid2 contributes 0.0 (in [0,3]). Total 0.0.
    // vectorSumResult:
    // group1: pid1 contributes (1.5, 4.5), L_INF-clipped to (1.5, 2.0). pid2 contributes (1.0,
    // 0.0),
    // not clipped.
    // Total for group1: (1.5, 2.0) + (1.0, 0.0) = (2.5, 2.0).
    // nonPublicGroup: pid2 contributes (3.0, 0.0), L_INF-clipped to (2.0, 0.0).
    val expected =
      listOf(
        QueryPerGroupResultWithTolerance(
          "group1",
          mapOf(
            "cnt" to DoubleWithTolerance(value = 3.0, tolerance = 0.5),
            "sumValue" to DoubleWithTolerance(value = 2.5, tolerance = 0.5),
            "sumAnotherValue" to DoubleWithTolerance(value = 3.0, tolerance = 0.5),
          ),
          mapOf(
            "vectorSumResult" to
              listOf(
                DoubleWithTolerance(value = 2.5, tolerance = 0.5),
                DoubleWithTolerance(value = 2.0, tolerance = 0.5),
              )
          ),
        )
      )
    assertEquals(result, expected)
  }

  @Test
  fun run_privateGroups_allPossibleValueAggregations_calculatesStatisticsCorrectly() {
    val data =
      createInputData(
        listOf(
          TestDataRow("group1", "pid1", 1.0),
          TestDataRow("group1", "pid1", 1.5),
          TestDataRow("group1", "pid2", 2.0),
          TestDataRow("group2", "pid1", 1.0),
        )
      )
    val query =
      SparkDataFrameQueryBuilder.from(
          data,
          ColumnNames("privacyUnit"),
          ContributionBoundingLevel.DATASET_LEVEL(
            maxGroupsContributed = 2,
            maxContributionsPerGroup = 2,
          ),
        )
        .groupBy(ColumnNames("groupKey"), GroupsType.PrivateGroups())
        .countDistinctPrivacyUnits("pidCnt")
        .count("cnt")
        .aggregateValue(
          "value",
          ValueAggregationsBuilder()
            .sum("sumResult")
            .mean("meanResult")
            .variance("varianceResult")
            .quantiles(ranks = listOf(0.5), outputColumnName = "quantilesResult"),
          ContributionBounds(valueBounds = Bounds(minValue = 1.0, maxValue = 2.0)),
        )
        .build(TotalBudget(epsilon = 3500.0, delta = 0.001), NoiseKind.LAPLACE)

    val result: SparkDataFrame = query.run()

    val expected =
      listOf(
        QueryPerGroupResultWithTolerance(
          "group1",
          mapOf(
            "pidCnt" to DoubleWithTolerance(value = 2.0, tolerance = 0.5),
            "cnt" to DoubleWithTolerance(value = 3.0, tolerance = 0.5),
            "sumResult" to DoubleWithTolerance(value = 4.5, tolerance = 0.5),
            "meanResult" to DoubleWithTolerance(value = 1.5, tolerance = 0.5),
            // (1^2+(1.5)^2+2^2)/3-((1.0+1.5+2)/3)^2 = 0.1(6)
            "varianceResult" to DoubleWithTolerance(value = 0.16, tolerance = 0.05),
            "quantilesResult_0.5" to DoubleWithTolerance(value = 1.5, tolerance = 0.5),
          ),
          vectorAggregationResults = mapOf(),
        )
      )
    assertEquals(result, expected)
  }

  @Test
  fun run_privateGroups_allPossibleVectorAggregations_calculatesStatisticsCorrectly() {
    val data =
      createInputData(
        listOf(
          TestDataRow("group1", "pid1", 1.0, 2.0),
          TestDataRow("group1", "pid1", 1.5, 2.5),
          TestDataRow("group1", "pid2", 3.0, -1.0),
          TestDataRow("group2", "pid1", -1.0, -3.0),
        )
      )
    val query =
      SparkDataFrameQueryBuilder.from(
          data,
          ColumnNames("privacyUnit"),
          ContributionBoundingLevel.DATASET_LEVEL(
            maxGroupsContributed = 2,
            maxContributionsPerGroup = 2,
          ),
        )
        .groupBy(ColumnNames("groupKey"), GroupsType.PrivateGroups())
        .countDistinctPrivacyUnits("pidCnt")
        .count("cnt")
        .aggregateVector(
          ColumnNames("value", "anotherValue"),
          vectorSize = 2,
          VectorAggregationsBuilder().vectorSum("vectorSumResult"),
          VectorContributionBounds(
            maxVectorTotalNorm = VectorNorm(normKind = NormKind.L1, value = 5.0)
          ),
        )
        .build(TotalBudget(epsilon = 3500.0, delta = 0.001), NoiseKind.LAPLACE)

    val result: SparkDataFrame = query.run()

    val expected =
      listOf(
        QueryPerGroupResultWithTolerance(
          "group1",
          mapOf(
            "pidCnt" to DoubleWithTolerance(value = 2.0, tolerance = 0.5),
            "cnt" to DoubleWithTolerance(value = 3.0, tolerance = 0.5),
          ),
          mapOf(
            "vectorSumResult" to
              // pid1: (1.0, 2.0) + (1.5, 2.5) = (2.5, 4.5), L1 norm is 7 =>
              // clip it to (2.5, 4.5) * 5.0 / 7.0 = (1.8, 3.2)
              // pid2: (3.0, -1.0), L1 norm is 4.0 => no clipping.
              // result: (1.8, 3.2) + (3.0, -1.0) = (4.8, 2.2)
              listOf(
                DoubleWithTolerance(value = 4.8, tolerance = 0.5),
                DoubleWithTolerance(value = 2.2, tolerance = 0.5),
              )
          ),
        )
      )
    assertEquals(result, expected)
  }

  // When counting distinct privacy units different group selection mechanism is used.
  @Test
  fun run_privateGroups_noCountDistinctPrivacyUnits_calculatesStatisticsCorrectly() {
    val data =
      createInputData(
        listOf(
          TestDataRow("group1", "pid1", 1.0),
          TestDataRow("group1", "pid1", 1.5),
          TestDataRow("group1", "pid2", 2.0),
          TestDataRow("group2", "pid1", 1.0),
        )
      )
    val query =
      SparkDataFrameQueryBuilder.from(
          data,
          ColumnNames("privacyUnit"),
          ContributionBoundingLevel.DATASET_LEVEL(
            maxGroupsContributed = 2,
            maxContributionsPerGroup = 2,
          ),
        )
        .groupBy(ColumnNames("groupKey"), GroupsType.PrivateGroups())
        .count("cnt")
        .aggregateValue(
          "value",
          ValueAggregationsBuilder()
            .sum("sumResult")
            .mean("meanResult")
            .variance("varianceResult")
            .quantiles(ranks = listOf(0.5), outputColumnName = "quantilesResult"),
          ContributionBounds(valueBounds = Bounds(minValue = 1.0, maxValue = 2.0)),
        )
        .build(TotalBudget(epsilon = 3000.0, delta = 0.001), NoiseKind.GAUSSIAN)

    val result: SparkDataFrame = query.run()

    val expected =
      listOf(
        QueryPerGroupResultWithTolerance(
          "group1",
          mapOf(
            "cnt" to DoubleWithTolerance(value = 3.0, tolerance = 0.5),
            "sumResult" to DoubleWithTolerance(value = 4.5, tolerance = 0.5),
            "meanResult" to DoubleWithTolerance(value = 1.5, tolerance = 0.5),
            // (1^2+(1.5)^2+2^2)/3-((1.0+1.5+2)/3)^2 = 0.1(6)
            "varianceResult" to DoubleWithTolerance(value = 0.16, tolerance = 0.05),
            "quantilesResult_0.5" to DoubleWithTolerance(value = 1.5, tolerance = 0.5),
          ),
          vectorAggregationResults = mapOf(),
        )
      )
    assertEquals(result, expected)
  }

  @Test
  fun run_vectorSumOnly_calculatesStatisticsCorrectly() {
    val data =
      createInputData(
        listOf(
          TestDataRow("group1", "pid1", 1.0, 2.0),
          TestDataRow("group1", "pid1", 1.5, 2.5),
          TestDataRow("group1", "pid2", -2.0, 0.0),
        )
      )
    val publicGroups = createPublicGroups(listOf("group1"))
    val query =
      SparkDataFrameQueryBuilder.from(
          data,
          ColumnNames("privacyUnit"),
          ContributionBoundingLevel.DATASET_LEVEL(
            maxGroupsContributed = 1,
            maxContributionsPerGroup = 2,
          ),
        )
        .groupBy(ColumnNames("groupKey"), GroupsType.PublicGroups.createForDataFrame(publicGroups))
        .aggregateVector(
          ColumnNames("value", "anotherValue"),
          vectorSize = 2,
          VectorAggregationsBuilder().vectorSum("vectorSumResult"),
          VectorContributionBounds(
            maxVectorTotalNorm = VectorNorm(normKind = NormKind.L2, value = 3.0)
          ),
        )
        .build(TotalBudget(epsilon = 500.0, delta = 0.999), NoiseKind.GAUSSIAN)

    val result: SparkDataFrame = query.run()

    val expected =
      listOf(
        QueryPerGroupResultWithTolerance(
          "group1",
          valueAggregationResults = mapOf(),
          mapOf(
            "vectorSumResult" to
              // pid1: (1.0, 2.0) + (1.5, 2.5) = (2.5, 4.5), L2 norm is ~5.7 =>
              // clip it to (2.5, 4.5) * 3.0 / 5.1 = (1.5, 2.6).
              // pid2: (-2.0, 0.0), L2 norm is 2.0 => no clipping.
              // result: (1.5, 2.6) + (-2.0, 0.0) = (-0.5, 2.6)
              listOf(
                DoubleWithTolerance(value = -0.5, tolerance = 0.5),
                DoubleWithTolerance(value = 2.6, tolerance = 0.5),
              )
          ),
        )
      )
    assertEquals(result, expected)
  }

  // When sum without mean or variance is requested then total value bounds are used.
  @Test
  fun run_sumOnly_calculatesStatisticsCorrectly() {
    val data =
      createInputData(
        listOf(
          TestDataRow("group1", "pid1", 1.0),
          TestDataRow("group1", "pid1", 1.5),
          TestDataRow("group1", "pid2", 2.0),
        )
      )
    val publicGroups = createPublicGroups(listOf("group1"))
    val query =
      SparkDataFrameQueryBuilder.from(
          data,
          ColumnNames("privacyUnit"),
          ContributionBoundingLevel.DATASET_LEVEL(
            maxGroupsContributed = 1,
            maxContributionsPerGroup = 2,
          ),
        )
        .groupBy(ColumnNames("groupKey"), GroupsType.PublicGroups.createForDataFrame(publicGroups))
        .aggregateValue(
          "value",
          ValueAggregationsBuilder().sum("sumResult"),
          ContributionBounds(totalValueBounds = Bounds(minValue = 2.0, maxValue = 2.5)),
        )
        .build(TotalBudget(epsilon = 1000.0), NoiseKind.LAPLACE)

    val result: SparkDataFrame = query.run()

    val expected =
      listOf(
        QueryPerGroupResultWithTolerance(
          "group1",
          mapOf("sumResult" to DoubleWithTolerance(value = 4.5, tolerance = 0.5)),
          vectorAggregationResults = mapOf(),
        )
      )
    assertEquals(result, expected)
  }

  @Test
  fun run_sumAndQuantiles_bothBoundTypesAreUsed_calculatesStatisticsCorrectly() {
    val data =
      createInputData(
        listOf(
          TestDataRow("group1", "pid1", 1.0),
          TestDataRow("group1", "pid1", 1.5),
          TestDataRow("group1", "pid2", 2.0),
        )
      )
    val publicGroups = createPublicGroups(listOf("group1"))
    val query =
      SparkDataFrameQueryBuilder.from(
          data,
          ColumnNames("privacyUnit"),
          ContributionBoundingLevel.DATASET_LEVEL(
            maxGroupsContributed = 1,
            maxContributionsPerGroup = 2,
          ),
        )
        .groupBy(ColumnNames("groupKey"), GroupsType.PublicGroups.createForDataFrame(publicGroups))
        .aggregateValue(
          "value",
          ValueAggregationsBuilder()
            .sum("sumResult")
            .quantiles(ranks = listOf(0.5), outputColumnName = "quantilesResult"),
          ContributionBounds(
            valueBounds = Bounds(minValue = 1.0, maxValue = 2.0),
            totalValueBounds = Bounds(minValue = 2.0, maxValue = 2.5),
          ),
        )
        .build(TotalBudget(epsilon = 1000.0), NoiseKind.LAPLACE)

    val result: SparkDataFrame = query.run()

    val expected =
      listOf(
        QueryPerGroupResultWithTolerance(
          "group1",
          mapOf(
            "sumResult" to DoubleWithTolerance(value = 4.5, tolerance = 0.5),
            "quantilesResult_0.5" to DoubleWithTolerance(value = 1.5, tolerance = 0.5),
          ),
          vectorAggregationResults = mapOf(),
        )
      )
    assertEquals(result, expected)
  }

  @Test
  fun run_countSumAndMean_budgetIsUsedForMeanOnly() {
    val data =
      createInputData(
        listOf(
          TestDataRow("group1", "pid1", 1.0),
          TestDataRow("group1", "pid1", 1.5),
          TestDataRow("group1", "pid2", 2.0),
        )
      )
    val publicGroups = createPublicGroups(listOf("group1"))
    val query =
      SparkDataFrameQueryBuilder.from(
          data,
          ColumnNames("privacyUnit"),
          ContributionBoundingLevel.DATASET_LEVEL(
            maxGroupsContributed = 1,
            maxContributionsPerGroup = 2,
          ),
        )
        .groupBy(ColumnNames("groupKey"), GroupsType.PublicGroups.createForDataFrame(publicGroups))
        .count("cnt")
        .aggregateValue(
          "value",
          ValueAggregationsBuilder()
            .sum("sumResult")
            .mean(
              "meanResult",
              // We use all budget here for variance. Therefore if the budget was used for count,
              // sum or mean, we would go beyond the total budget and get an error.
              budget = AbsoluteBudgetPerOpSpec(epsilon = 1000.0, delta = 0.9999),
            ),
          ContributionBounds(valueBounds = Bounds(minValue = 1.0, maxValue = 2.0)),
        )
        .build(TotalBudget(epsilon = 1000.0, delta = 0.9999), NoiseKind.GAUSSIAN)

    val result: SparkDataFrame = query.run()

    val expected =
      listOf(
        QueryPerGroupResultWithTolerance(
          "group1",
          mapOf(
            "cnt" to DoubleWithTolerance(value = 3.0, tolerance = 0.5),
            "sumResult" to DoubleWithTolerance(value = 4.5, tolerance = 0.5),
            "meanResult" to DoubleWithTolerance(value = 1.5, tolerance = 0.5),
          ),
          vectorAggregationResults = mapOf(),
        )
      )
    assertEquals(result, expected)
  }

  @Test
  fun run_countSumMeanAndVariance_budgetIsUsedForVarianceOnly() {
    val data =
      createInputData(
        listOf(
          TestDataRow("group1", "pid1", 1.0),
          TestDataRow("group1", "pid1", 1.5),
          TestDataRow("group1", "pid2", 2.0),
        )
      )
    val publicGroups = createPublicGroups(listOf("group1"))
    val query =
      SparkDataFrameQueryBuilder.from(
          data,
          ColumnNames("privacyUnit"),
          ContributionBoundingLevel.DATASET_LEVEL(
            maxGroupsContributed = 1,
            maxContributionsPerGroup = 2,
          ),
        )
        .groupBy(ColumnNames("groupKey"), GroupsType.PublicGroups.createForDataFrame(publicGroups))
        .count("cnt")
        .aggregateValue(
          "value",
          ValueAggregationsBuilder()
            .sum("sumResult")
            .mean("meanResult")
            .variance(
              "varianceResult",
              // We use all budget here for variance. Therefore if the budget was used for count,
              // sum or mean, we would go beyond the total budget and get an error.
              budget = AbsoluteBudgetPerOpSpec(epsilon = 1000.0, delta = 0.9999),
            ),
          ContributionBounds(valueBounds = Bounds(minValue = 1.0, maxValue = 2.0)),
        )
        .build(TotalBudget(epsilon = 1000.0, delta = 0.9999), NoiseKind.GAUSSIAN)

    val result: SparkDataFrame = query.run()

    val expected =
      listOf(
        QueryPerGroupResultWithTolerance(
          "group1",
          mapOf(
            "cnt" to DoubleWithTolerance(value = 3.0, tolerance = 0.5),
            "sumResult" to DoubleWithTolerance(value = 4.5, tolerance = 0.5),
            "meanResult" to DoubleWithTolerance(value = 1.5, tolerance = 0.5),
            // (1^2+(1.5)^2+2^2)/3-((1.0+1.5+2)/3)^2 = 0.1(6)
            "varianceResult" to DoubleWithTolerance(value = 0.16, tolerance = 0.05),
          ),
          vectorAggregationResults = mapOf(),
        )
      )
    assertEquals(result, expected)
  }

  @Test
  fun run_quantileRanksNotSorted_mapsToCorrectColumnNames() {
    val data =
      createInputData(
        listOf(
          TestDataRow("group1", "pid1", 1.0),
          TestDataRow("group1", "pid1", 1.5),
          TestDataRow("group1", "pid2", 2.0),
        )
      )
    val publicGroups = createPublicGroups(listOf("group1"))
    val query =
      SparkDataFrameQueryBuilder.from(
          data,
          ColumnNames("privacyUnit"),
          ContributionBoundingLevel.DATASET_LEVEL(
            maxGroupsContributed = 1,
            maxContributionsPerGroup = 2,
          ),
        )
        .groupBy(ColumnNames("groupKey"), GroupsType.PublicGroups.createForDataFrame(publicGroups))
        .aggregateValue(
          "value",
          ValueAggregationsBuilder()
            .quantiles(ranks = listOf(1.0, 0.5, 0.0), outputColumnName = "quantilesResult"),
          ContributionBounds(valueBounds = Bounds(minValue = 1.0, maxValue = 2.0)),
        )
        .build(TotalBudget(epsilon = 1000.0), NoiseKind.LAPLACE)

    val result: SparkDataFrame = query.run()

    val expected =
      listOf(
        QueryPerGroupResultWithTolerance(
          "group1",
          mapOf(
            "quantilesResult_0.0" to DoubleWithTolerance(value = 1.0, tolerance = 0.5),
            "quantilesResult_0.5" to DoubleWithTolerance(value = 1.5, tolerance = 0.5),
            "quantilesResult_1.0" to DoubleWithTolerance(value = 2.0, tolerance = 0.5),
          ),
          vectorAggregationResults = mapOf(),
        )
      )
    assertEquals(result, expected)
  }

  @Test
  fun run_publicGroupsProvidedAsDatasetOfLists_respectsProvidedGroups() {
    val data =
      createInputData(
        listOf(
          TestDataRow("group1", "pid1", 1.0),
          TestDataRow("group1", "pid1", 1.5),
          TestDataRow("group1", "pid2", 2.0),
        )
      )
    val publicGroups =
      sparkSession.spark.createDataset(
        listOf(listOf("group1").toCollection(ArrayList())),
        Encoders.kryo(List::class.java) as Encoder<List<Any?>>,
      )
    val query =
      SparkDataFrameQueryBuilder.from(
          data,
          ColumnNames("privacyUnit"),
          ContributionBoundingLevel.DATASET_LEVEL(
            maxGroupsContributed = 1,
            maxContributionsPerGroup = 2,
          ),
        )
        .groupBy(ColumnNames("groupKey"), GroupsType.PublicGroups.create(publicGroups))
        .count("cnt")
        .build(TotalBudget(epsilon = 1000.0), NoiseKind.LAPLACE)

    val result: SparkDataFrame = query.run()

    val expected =
      listOf(
        QueryPerGroupResultWithTolerance(
          "group1",
          mapOf("cnt" to DoubleWithTolerance(value = 3.0, tolerance = 0.5)),
          vectorAggregationResults = mapOf(),
        )
      )
    assertEquals(result, expected)
  }

  @Test
  fun run_publicGroupsProvidedAsIterableOfLists_respectsProvidedGroups() {
    val data =
      createInputData(
        listOf(
          TestDataRow("group1", "pid1", 1.0),
          TestDataRow("group1", "pid1", 1.5),
          TestDataRow("group1", "pid2", 2.0),
        )
      )
    val publicGroups = listOf(listOf("group1").toCollection(ArrayList()))
    val query =
      SparkDataFrameQueryBuilder.from(
          data,
          ColumnNames("privacyUnit"),
          ContributionBoundingLevel.DATASET_LEVEL(
            maxGroupsContributed = 1,
            maxContributionsPerGroup = 2,
          ),
        )
        .groupBy(ColumnNames("groupKey"), GroupsType.PublicGroups.create(publicGroups))
        .count("cnt")
        .build(TotalBudget(epsilon = 1000.0), NoiseKind.LAPLACE)

    val result: SparkDataFrame = query.run()

    val expected =
      listOf(
        QueryPerGroupResultWithTolerance(
          "group1",
          mapOf("cnt" to DoubleWithTolerance(value = 3.0, tolerance = 0.5)),
          vectorAggregationResults = mapOf(),
        )
      )
    assertEquals(result, expected)
  }

  @Test
  fun run_withFullTestMode_addsNoNoiseAndDoesNotPerformContributionBounding() {
    val data =
      createInputData(
        listOf(
          TestDataRow("group1", "pid1", 1.0),
          TestDataRow("group1", "pid1", 1.5),
          TestDataRow("group1", "pid2", 2.0),
          TestDataRow("group2", "pid1", 5.0),
          TestDataRow("group2", "pid1", 6.0),
          TestDataRow("group2", "pid1", 7.0),
        )
      )
    val publicGroups = createPublicGroups(listOf("group1", "group2"))
    val query =
      SparkDataFrameQueryBuilder.from(
          data,
          ColumnNames("privacyUnit"),
          ContributionBoundingLevel.DATASET_LEVEL(
            maxGroupsContributed = 1,
            maxContributionsPerGroup = 1,
          ),
        )
        .groupBy(ColumnNames("groupKey"), GroupsType.PublicGroups.createForDataFrame(publicGroups))
        .count("cnt")
        .aggregateValue(
          "value",
          ValueAggregationsBuilder()
            .sum("sumResult")
            .mean("meanResult")
            .variance("varianceResult")
            .quantiles(ranks = listOf(1.0, 0.5, 0.0), outputColumnName = "quantilesResult"),
          ContributionBounds(valueBounds = Bounds(minValue = 1.0, maxValue = 2.0)),
        )
        .build(TotalBudget(epsilon = 0.0001, delta = 0.0001), NoiseKind.GAUSSIAN)

    val result: SparkDataFrame = query.run(testMode = TestMode.FULL)

    val valuesEqualTolerance = 1e-4
    val expected =
      listOf(
        QueryPerGroupResultWithTolerance(
          "group1",
          mapOf(
            "cnt" to DoubleWithTolerance(value = 3.0, tolerance = valuesEqualTolerance),
            "sumResult" to DoubleWithTolerance(value = 4.5, tolerance = valuesEqualTolerance),
            "meanResult" to DoubleWithTolerance(value = 1.5, tolerance = valuesEqualTolerance),
            "varianceResult" to
              DoubleWithTolerance(value = 0.16666, tolerance = valuesEqualTolerance),
            "quantilesResult_0.0" to
              DoubleWithTolerance(value = 1.0, tolerance = valuesEqualTolerance),
            "quantilesResult_0.5" to
              DoubleWithTolerance(value = 1.5, tolerance = valuesEqualTolerance),
            "quantilesResult_1.0" to
              DoubleWithTolerance(value = 2.0, tolerance = valuesEqualTolerance),
          ),
          vectorAggregationResults = mapOf(),
        ),
        QueryPerGroupResultWithTolerance(
          "group2",
          mapOf(
            "cnt" to DoubleWithTolerance(value = 3.0, tolerance = valuesEqualTolerance),
            "sumResult" to DoubleWithTolerance(value = 18.0, tolerance = valuesEqualTolerance),
            "meanResult" to DoubleWithTolerance(value = 6.0, tolerance = valuesEqualTolerance),
            "varianceResult" to
              DoubleWithTolerance(value = 0.66666, tolerance = valuesEqualTolerance),
            "quantilesResult_0.0" to
              DoubleWithTolerance(
                value = 2.0,
                tolerance = valuesEqualTolerance,
              ), // assert value should be 5.0 but FULL test mode does not currently perform
            // contribution bounding for quantiles.
            "quantilesResult_0.5" to
              DoubleWithTolerance(
                value = 2.0,
                tolerance = valuesEqualTolerance,
              ), // assert value should be 6.0 but FULL test mode does not currently perform
            // contribution bounding for quantiles.
            "quantilesResult_1.0" to
              DoubleWithTolerance(
                value = 2.0,
                tolerance = valuesEqualTolerance,
              ), // assert value should be 7.0 but FULL test mode does not currently perform
            // contribution bounding for quantiles.
          ),
          vectorAggregationResults = mapOf(),
        ),
      )
    assertEquals(result, expected)
  }

  private fun createEmptyInputData() = createInputData(listOf())

  private fun createInputData(data: List<TestDataRow>) =
    sparkSession.spark.createDataFrame(data, TestDataRow::class.java)

  private fun createPublicGroups(groupKeys: List<String>) =
    sparkSession.spark.createDataset(groupKeys, Encoders.STRING()).toDF("groupKey")

  private fun assertEquals(
    result: SparkDataFrame,
    expected: List<QueryPerGroupResultWithTolerance>,
  ) = assertEquals(result.toQueryPerGroupResultList(), expected)

  private fun SparkDataFrame.toQueryPerGroupResultList(): List<QueryPerGroupResult<String>> {
    @Suppress("UNCHECKED_CAST")
    return map(
        MapFunction { row: Row ->
          val aggregationsColumnNames = row.schema().fieldNames().drop(1)
          val aggregationResults: Map<String, Any?> =
            aggregationsColumnNames.associateWith { row.getAs(it) }
          QueryPerGroupResult(
            row.getAs("groupKey"),
            aggregationResults.filterValues { it is Double }.mapValues { it.value as Double },
            aggregationResults
              .filterValues { it is ScalaSeq<*> }
              .mapValues { SeqHasAsJava(it.value as ScalaSeq<Double>).asJava() },
          )
        },
        Encoders.kryo(QueryPerGroupResult::class.java) as Encoder<QueryPerGroupResult<String>>,
      )
      .collectAsList()
  }

  companion object {
    @JvmField @ClassRule val sparkSession = SparkSessionRule()
  }
}

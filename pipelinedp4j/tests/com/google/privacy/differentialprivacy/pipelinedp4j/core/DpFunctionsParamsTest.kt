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
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.COUNT
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.MEAN
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.PRIVACY_ID_COUNT
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.QUANTILES
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.SUM
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.VARIANCE
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.VECTOR_SUM
import com.google.privacy.differentialprivacy.pipelinedp4j.core.NoiseKind.LAPLACE
import com.google.privacy.differentialprivacy.pipelinedp4j.core.budget.AbsoluteBudgetPerOpSpec
import com.google.privacy.differentialprivacy.pipelinedp4j.core.budget.RelativeBudgetPerOpSpec
import com.google.testing.junit.testparameterinjector.TestParameter
import com.google.testing.junit.testparameterinjector.TestParameterInjector
import com.google.testing.junit.testparameterinjector.TestParameters
import kotlin.test.assertFailsWith
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(TestParameterInjector::class)
class DpFunctionsParamsTest {
  @Test
  fun validateAggregationParams_validParams_doesntThrow() {
    validateAggregationParams(
      AGGREGATION_PARAMS,
      usePublicPartitions = true,
      hasValueExtractor = true,
    )
    validateAggregationParams(
      AGGREGATION_PARAMS,
      usePublicPartitions = false,
      hasValueExtractor = true,
    )
    validateAggregationParams(
      AGGREGATION_PARAMS,
      usePublicPartitions = false,
      hasValueExtractor = false,
    )
    validateAggregationParams(
      AGGREGATION_PARAMS.copy(
        maxContributionsPerPartition = null,
        metrics = ImmutableList.of(MetricDefinition(SUM)),
        minValue = null,
        maxValue = null,
        minTotalValue = 1.0,
        maxTotalValue = 2.0,
      ),
      usePublicPartitions = true,
      hasValueExtractor = true,
    )
    validateAggregationParams(
      AGGREGATION_PARAMS.copy(
        maxContributionsPerPartition = null,
        metrics = ImmutableList.of(MetricDefinition(VECTOR_SUM)),
        minValue = null,
        maxValue = null,
        minTotalValue = null,
        maxTotalValue = null,
        vectorSize = 2,
        vectorMaxTotalNorm = 1.0,
        vectorNormKind = NormKind.L1,
      ),
      usePublicPartitions = false,
      hasValueExtractor = true,
    )
  }

  enum class InvalidAggregationParamsTestCase(
    val aggregationParams: AggregationParams,
    val publicPartitions: Boolean = false,
    val hasValueExtractor: Boolean = true,
    val exceptionMessage: String,
  ) {
    NOT_POSITIVE_MAX_PARTITION_CONTRIBUTED(
      aggregationParams = AGGREGATION_PARAMS.copy(maxPartitionsContributed = 0),
      exceptionMessage = "maxPartitionsContributed must be positive. Provided value: 0.",
    ),
    PARTITION_LEVEL_CONTRIBUTION_BOUNDING_MAX_PARTITIONS_CONTRIBUTED(
      aggregationParams =
        AGGREGATION_PARAMS.copy(
          contributionBoundingLevel = ContributionBoundingLevel.PARTITION_LEVEL,
          maxPartitionsContributed = 2,
        ),
      exceptionMessage =
        "maxPartitionsContributed must be 1 if partition level contribution bounding is set. Provided value: 2",
    ),
    MAX_PARTITIONS_CONTRIBUTED_NOT_SET_WHEN_CONTRIBUTION_BOUNDING_LEVEL_REQUIRES_CROSS_PARTITION_BOUNDING(
      aggregationParams =
        AGGREGATION_PARAMS.copy(
          contributionBoundingLevel = ContributionBoundingLevel.DATASET_LEVEL,
          maxPartitionsContributed = null,
          maxContributions = null,
        ),
      exceptionMessage =
        "maxPartitionsContributed or maxContributions must be set because specified DATASET_LEVEL contribution bounding level requires cross partition bounding",
    ),
    NOT_POSITIVE_PRETHRESHOLD(
      aggregationParams = AGGREGATION_PARAMS.copy(preThreshold = 0),
      exceptionMessage = "preThreshold must be positive. Provided value: 0",
    ),
    NO_METRICS(
      aggregationParams = AGGREGATION_PARAMS.copy(metrics = ImmutableList.of<MetricDefinition>()),
      exceptionMessage = "metrics must not be empty.",
    ),
    ZERO_MAX_CONTRIBUTIONS_PER_PARTITION(
      aggregationParams = AGGREGATION_PARAMS.copy(maxContributionsPerPartition = 0),
      exceptionMessage = "maxContributionsPerPartition must be positive. Provided value: 0.",
    ),
    NEGATIVE_MAX_CONTRIBUTIONS_PER_PARTITION(
      aggregationParams = AGGREGATION_PARAMS.copy(maxContributionsPerPartition = -1),
      exceptionMessage = "maxContributionsPerPartition must be positive. Provided value: -1.",
    ),
    MAX_CONTRIBUTIONS_PER_PARTITION_NOT_SET_WHEN_CONTRIBUTION_BOUNDING_LEVEL_REQUIRES_PER_PARTITION_BOUNDING(
      aggregationParams =
        AGGREGATION_PARAMS.copy(
          contributionBoundingLevel = ContributionBoundingLevel.PARTITION_LEVEL,
          maxContributionsPerPartition = null,
          maxContributions = null,
        ),
      exceptionMessage =
        "maxContributionsPerPartition or maxContributions or (minTotalValue, maxTotalValue) or vectorMaxTotalNorm must be set because specified PARTITION_LEVEL contribution bounding level requires per partition bounding",
    ),
    ZERO_MAX_CONTRIBUTIONS(
      aggregationParams = AGGREGATION_PARAMS.copy(maxContributions = 0),
      exceptionMessage = "maxContributions must be positive. Provided value: 0.",
    ),
    NEGATIVE_MAX_CONTRIBUTIONS(
      aggregationParams = AGGREGATION_PARAMS.copy(maxContributions = -1),
      exceptionMessage = "maxContributions must be positive. Provided value: -1.",
    ),
    MUTUALLY_EXCLUSIVE_MAX_CONTRIBUTIONS_PER_PARTITION_MAX_CONTRIBUTIONS(
      aggregationParams =
        AGGREGATION_PARAMS.copy(
          maxPartitionsContributed = null,
          maxContributionsPerPartition = 1,
          maxContributions = 1,
        ),
      exceptionMessage =
        "maxContributions and maxContributionsPerPartition are mutually exclusive. " +
          "Provided values: maxContributions=1, maxContributionsPerPartition=1",
    ),
    MUTUALLY_EXCLUSIVE_MAX_PARTITIONS_CONTRIBUTED_MAX_CONTRIBUTIONS(
      aggregationParams =
        AGGREGATION_PARAMS.copy(
          maxPartitionsContributed = 1,
          maxContributionsPerPartition = null,
          maxContributions = 1,
        ),
      exceptionMessage =
        "maxContributions and maxPartitionsContributed are mutually exclusive. " +
          "Provided values: maxContributions=1, maxPartitionsContributed=1",
    ),
    MUTUALLY_EXCLUSIVE_MAX_CONTRIBUTIONS_ALL_SET(
      aggregationParams =
        AGGREGATION_PARAMS.copy(
          maxPartitionsContributed = 1,
          maxContributionsPerPartition = 1,
          maxContributions = 1,
        ),
      exceptionMessage =
        "maxContributions and maxPartitionsContributed are mutually exclusive. Provided values: maxContributions=1, maxPartitionsContributed=1",
    ),
    MIN_VALUE_SET_MAX_VALUE_NOT_SET(
      aggregationParams = AGGREGATION_PARAMS.copy(minValue = 1.0, maxValue = null),
      exceptionMessage = "minValue and maxValue must be simultaneously equal or not equal to null.",
    ),
    MIN_VALUE_NOT_SET_MAX_VALUE_SET(
      aggregationParams = AGGREGATION_PARAMS.copy(minValue = null, maxValue = 2.0),
      exceptionMessage =
        "minValue and maxValue must be simultaneously equal or not equal to " +
          "null. Provided values: minValue=null, maxValue=2.0",
    ),
    MIN_VALUE_GREATER_THAN_MAX_VALUE(
      aggregationParams = AGGREGATION_PARAMS.copy(minValue = 1.5, maxValue = 1.0),
      exceptionMessage =
        "minValue must be less than maxValue. Provided values: " + "minValue=1.5, maxValue=1.0",
    ),
    MIN_VALUE_IS_EQUAL_TO_MAX_VALUE(
      aggregationParams = AGGREGATION_PARAMS.copy(minValue = 1.5, maxValue = 1.5),
      exceptionMessage =
        "minValue must be less than maxValue. Provided values: " + "minValue=1.5, maxValue=1.5",
    ),
    MIN_TOTAL_VALUE_SET_MAX_TOTAL_VALUE_NOT_SET(
      aggregationParams = AGGREGATION_PARAMS.copy(minTotalValue = 1.0, maxTotalValue = null),
      exceptionMessage =
        "minTotalValue and maxTotalValue must be simultaneously equal or not equal to null. " +
          "Provided values: minTotalValue=1.0, maxTotalValue=null",
    ),
    MIN_TOTAL_VALUE_NOT_SET_MAX_TOTAL_VALUE_SET(
      aggregationParams = AGGREGATION_PARAMS.copy(minTotalValue = null, maxTotalValue = 2.0),
      exceptionMessage =
        "minTotalValue and maxTotalValue must be simultaneously equal or not equal to null.",
    ),
    MIN_TOTAL_VALUE_GREATER_THAN_MAX_TOTAL_VALUE(
      aggregationParams = AGGREGATION_PARAMS.copy(minTotalValue = 2.0, maxTotalValue = 0.0),
      exceptionMessage =
        "minTotalValue must be less or equal to maxTotalValue. Provided values: " +
          "minTotalValue=2.0, maxTotalValue=0.0",
    ),
    MEAN_WITH_TOTAL_VALUE(
      aggregationParams =
        AGGREGATION_PARAMS.copy(
          metrics = ImmutableList.of(MetricDefinition(SUM), MetricDefinition(MEAN)),
          minValue = 0.0,
          maxValue = 3.0,
          minTotalValue = 1.5,
          maxTotalValue = 5.0,
        ),
      exceptionMessage =
        "(minTotalValue, maxTotalValue) should not be set if MEAN metric is requested",
    ),
    VARIANCE_WITH_TOTAL_VALUE(
      aggregationParams =
        AGGREGATION_PARAMS.copy(
          metrics = ImmutableList.of(MetricDefinition(SUM), MetricDefinition(VARIANCE)),
          minValue = 0.0,
          maxValue = 3.0,
          minTotalValue = 1.5,
          maxTotalValue = 5.0,
        ),
      exceptionMessage =
        "(minTotalValue, maxTotalValue) should not be set if VARIANCE metric is requested",
    ),
    MAX_CONTRIBUTIONS_PER_PARTITION_MAX_CONTRIBUTIONS_NOT_SET_FOR_COUNT(
      aggregationParams =
        AGGREGATION_PARAMS.copy(
          metrics = ImmutableList.of(MetricDefinition(COUNT), MetricDefinition(SUM)),
          maxContributionsPerPartition = null,
          maxContributions = null,
          minTotalValue = -1.0,
          maxTotalValue = 1.0,
          minValue = null,
          maxValue = null,
        ),
      exceptionMessage =
        "maxContributionsPerPartition or maxContributions must be set for COUNT metric.",
    ),
    MAX_CONTRIBUTIONS_PER_PARTITION_MAX_CONTRIBUTIONS_NOT_SET_FOR_MEAN(
      aggregationParams =
        AGGREGATION_PARAMS.copy(
          metrics = ImmutableList.of(MetricDefinition(MEAN), MetricDefinition(SUM)),
          maxContributionsPerPartition = null,
          maxContributions = null,
          minTotalValue = -1.0,
          maxTotalValue = 1.0,
          minValue = null,
          maxValue = null,
        ),
      exceptionMessage =
        "maxContributionsPerPartition or maxContributions must be set for MEAN metric.",
    ),
    MAX_CONTRIBUTIONS_PER_PARTITION_MAX_CONTRIBUTIONS_NOT_SET_FOR_QUANTILES(
      aggregationParams =
        AGGREGATION_PARAMS.copy(
          metrics =
            ImmutableList.of(
              MetricDefinition(QUANTILES(ranks = ImmutableList.of())),
              MetricDefinition(SUM),
            ),
          maxContributionsPerPartition = null,
          minTotalValue = -1.0,
          maxTotalValue = 1.0,
          minValue = null,
          maxValue = null,
        ),
      exceptionMessage = "maxContributionsPerPartition must be set for QUANTILES metric.",
    ),
    MIN_TOTAL_VALUE_NOT_SET_FOR_SUM(
      aggregationParams =
        AGGREGATION_PARAMS.copy(
          metrics = ImmutableList.of(MetricDefinition(SUM)),
          minTotalValue = null,
          maxTotalValue = null,
        ),
      exceptionMessage = "(minTotalValue, maxTotalValue) must be set for SUM metrics.",
    ),
    MIN_VALUE_NOT_SET_FOR_MEAN(
      aggregationParams =
        AGGREGATION_PARAMS.copy(
          metrics = ImmutableList.of(MetricDefinition(MEAN), MetricDefinition(SUM)),
          minTotalValue = 0.0,
          maxTotalValue = 1.0,
          minValue = null,
          maxValue = null,
        ),
      exceptionMessage = "(minValue, maxValue) must be set for MEAN metric.",
    ),
    VALUE_EXTRACTOR_NOT_SET_FOR_SUM_AND_MEAN(
      aggregationParams =
        AGGREGATION_PARAMS.copy(
          metrics =
            ImmutableList.of(MetricDefinition(COUNT), MetricDefinition(MEAN), MetricDefinition(SUM))
        ),
      hasValueExtractor = false,
      exceptionMessage = "Metrics [MEAN, SUM] require a value extractor.",
    ),
    MIN_VALUE_NOT_SET_FOR_QUANTILES(
      aggregationParams =
        AGGREGATION_PARAMS.copy(
          metrics = ImmutableList.of(MetricDefinition(QUANTILES(ranks = ImmutableList.of()))),
          minTotalValue = 0.0,
          maxTotalValue = 1.0,
          minValue = null,
          maxValue = null,
        ),
      exceptionMessage = "(minValue, maxValue) must be set for QUANTILES metric.",
    ),
    BUDGET_SPEC_SET_FOR_MEAN_AND_COUNT(
      aggregationParams =
        AGGREGATION_PARAMS.copy(
          metrics =
            ImmutableList.of(
              MetricDefinition(MEAN, RelativeBudgetPerOpSpec(weight = 1.0)),
              MetricDefinition(COUNT, AbsoluteBudgetPerOpSpec(epsilon = 2.0, delta = 1e-12)),
            )
        ),
      exceptionMessage = "BudgetPerOpSpec can not be set for both COUNT and MEAN metrics.",
    ),
    BUDGET_SPEC_SET_FOR_MEAN_AND_SUM(
      aggregationParams =
        AGGREGATION_PARAMS.copy(
          metrics =
            ImmutableList.of(
              MetricDefinition(MEAN, RelativeBudgetPerOpSpec(weight = 1.0)),
              MetricDefinition(SUM, RelativeBudgetPerOpSpec(weight = 2.0)),
            )
        ),
      exceptionMessage = "BudgetPerOpSpec can not be set for both SUM and MEAN metrics.",
    ),
    MAX_CONTRIBUTIONS_PER_PARTITION_MAX_CONTRIBUTIONS_NOT_SET_FOR_VARIANCE(
      aggregationParams =
        AGGREGATION_PARAMS.copy(
          metrics = ImmutableList.of(MetricDefinition(VARIANCE)),
          maxContributionsPerPartition = null,
          maxContributions = null,
          minTotalValue = -1.0,
          maxTotalValue = 1.0,
          minValue = null,
          maxValue = null,
        ),
      exceptionMessage =
        "maxContributionsPerPartition or maxContributions must be set for VARIANCE metric.",
    ),
    MIN_VALUE_NOT_SET_FOR_VARIANCE(
      aggregationParams =
        AGGREGATION_PARAMS.copy(
          metrics = ImmutableList.of(MetricDefinition(VARIANCE)),
          minTotalValue = 0.0,
          maxTotalValue = 1.0,
          minValue = null,
          maxValue = null,
        ),
      exceptionMessage = "(minValue, maxValue) must be set for VARIANCE metric.",
    ),
    MAX_VALUE_NOT_SET_FOR_VARIANCE(
      aggregationParams =
        AGGREGATION_PARAMS.copy(
          metrics = ImmutableList.of(MetricDefinition(VARIANCE)),
          minTotalValue = -1.0,
          maxTotalValue = 0.0,
          minValue = null,
          maxValue = null,
        ),
      exceptionMessage = "(minValue, maxValue) must be set for VARIANCE metric.",
    ),
    BUDGET_SPEC_SET_FOR_VARIANCE_AND_MEAN(
      aggregationParams =
        AGGREGATION_PARAMS.copy(
          metrics =
            ImmutableList.of(
              MetricDefinition(VARIANCE, RelativeBudgetPerOpSpec(weight = 1.0)),
              MetricDefinition(MEAN, RelativeBudgetPerOpSpec(weight = 1.0)),
            )
        ),
      exceptionMessage = "BudgetPerOpSpec can not be set for both MEAN and VARIANCE metrics.",
    ),
    BUDGET_SPEC_SET_FOR_VARIANCE_AND_COUNT(
      aggregationParams =
        AGGREGATION_PARAMS.copy(
          metrics =
            ImmutableList.of(
              MetricDefinition(VARIANCE, RelativeBudgetPerOpSpec(weight = 1.0)),
              MetricDefinition(COUNT, AbsoluteBudgetPerOpSpec(epsilon = 2.0, delta = 1e-12)),
            )
        ),
      exceptionMessage = "BudgetPerOpSpec can not be set for both COUNT and VARIANCE metrics.",
    ),
    BUDGET_SPEC_SET_FOR_VARIANCE_AND_SUM(
      aggregationParams =
        AGGREGATION_PARAMS.copy(
          metrics =
            ImmutableList.of(
              MetricDefinition(VARIANCE, RelativeBudgetPerOpSpec(weight = 1.0)),
              MetricDefinition(SUM, RelativeBudgetPerOpSpec(weight = 2.0)),
            )
        ),
      exceptionMessage = "BudgetPerOpSpec can not be set for both SUM and VARIANCE metrics.",
    ),
    PARTITION_SELECTION_BUDGET_FOR_PUBLIC_PARTITION(
      aggregationParams =
        AGGREGATION_PARAMS.copy(
          partitionSelectionBudget = AbsoluteBudgetPerOpSpec(epsilon = 1.0, delta = 1e-12)
        ),
      publicPartitions = true,
      exceptionMessage = "partitionSelectionBudget can not be set for public partitions.",
    ),
    DUPLICATE_METRIC_TYPES(
      aggregationParams =
        AGGREGATION_PARAMS.copy(
          metrics =
            ImmutableList.of(
              MetricDefinition(COUNT),
              MetricDefinition(PRIVACY_ID_COUNT),
              MetricDefinition(COUNT),
            )
        ),
      exceptionMessage =
        "metrics must not contain duplicate metric types. Provided " +
          "[COUNT, PRIVACY_ID_COUNT, COUNT].",
    ),
    NORM_KIND_NOT_SET_FOR_VECTOR_SUM(
      aggregationParams =
        AGGREGATION_PARAMS.copy(
          metrics = ImmutableList.of(MetricDefinition(VECTOR_SUM)),
          vectorNormKind = null,
          vectorMaxTotalNorm = 2.3,
          vectorSize = 2,
        ),
      exceptionMessage = "vectorNormKind must be set for VECTOR_SUM metric.",
    ),
    MAX_TOTAL_NORM_NOT_SET_FOR_VECTOR_SUM(
      aggregationParams =
        AGGREGATION_PARAMS.copy(
          metrics = ImmutableList.of(MetricDefinition(VECTOR_SUM)),
          vectorNormKind = NormKind.L2,
          vectorMaxTotalNorm = null,
          vectorSize = 2,
        ),
      exceptionMessage = "vectorMaxTotalNorm must be set for VECTOR_SUM metric.",
    ),
    VECTOR_SIZE_NOT_SET_FOR_VECTOR_SUM(
      aggregationParams =
        AGGREGATION_PARAMS.copy(
          metrics = ImmutableList.of(MetricDefinition(VECTOR_SUM)),
          vectorNormKind = NormKind.L_INF,
          vectorMaxTotalNorm = 1.0,
          vectorSize = null,
        ),
      exceptionMessage = "vectorSize must be set for VECTOR_SUM metric.",
    ),
    VECTOR_SUM_IS_REQUESTED_TOGETHER_WITH_SCALAR_METRICS(
      aggregationParams =
        AGGREGATION_PARAMS.copy(
          metrics =
            ImmutableList.of(
              MetricDefinition(VECTOR_SUM),
              MetricDefinition(SUM),
              MetricDefinition(MEAN),
              MetricDefinition(VARIANCE),
            ),
          vectorNormKind = NormKind.L_INF,
          vectorMaxTotalNorm = 1.0,
          vectorSize = 3,
        ),
      exceptionMessage =
        "VECTOR_SUM can not be computed together with scalar metrics such as SUM, MEAN, VARIANCE and QUANTILES.",
    ),
  }

  @Test
  fun validateAggregationParams_invalidParams_fails(
    @TestParameter testCase: InvalidAggregationParamsTestCase
  ) {
    val e =
      assertFailsWith<IllegalArgumentException> {
        validateAggregationParams(
          testCase.aggregationParams,
          testCase.publicPartitions,
          testCase.hasValueExtractor,
        )
      }
    assertThat(e).hasMessageThat().contains(testCase.exceptionMessage)
  }

  @Test
  fun validQuantiles_doesntThrow() {
    val unused = QUANTILES(ranks = ImmutableList.of(0.0, 0.0001, 0.5, 0.999, 1.0))
  }

  @Test
  fun validQuantiles_ranksAreSorted() {
    val ranks = QUANTILES(ranks = ImmutableList.of(1.0, 0.0, 0.5)).sortedRanks

    assertThat(ranks).containsExactly(0.0, 0.5, 1.0).inOrder()
  }

  @Test
  @TestParameters("{ranks: [-0.00001]}")
  @TestParameters("{ranks: [1.00001]}")
  fun invalidQuantiles_fail(ranks: List<Double>) {
    assertFailsWith<IllegalArgumentException>("in [0, 1]") {
      QUANTILES(ImmutableList.copyOf(ranks))
    }
  }

  @Test
  fun validateSelectPartitionsParams_validParams_doesntThrow() {
    validateSelectPartitionsParams(SELECT_PARTITIONS_PARAMS)
  }

  enum class InvalidSelectPartitionsParamsTestCase(
    val selectPartitionsParams: SelectPartitionsParams,
    val exceptionMessage: String,
  ) {
    NOT_POSITIVE_MAX_PARTITION_CONTRIBUTED(
      selectPartitionsParams = SELECT_PARTITIONS_PARAMS.copy(maxPartitionsContributed = 0),
      exceptionMessage = "maxPartitionsContributed must be positive. Provided value: 0.",
    ),
    PARTITION_LEVEL_CONTRIBUTION_BOUNDING_MAX_PARTITIONS_CONTRIBUTED(
      selectPartitionsParams =
        SELECT_PARTITIONS_PARAMS.copy(
          contributionBoundingLevel = ContributionBoundingLevel.PARTITION_LEVEL,
          maxPartitionsContributed = 2,
        ),
      exceptionMessage =
        "maxPartitionsContributed must be 1 if partition level contribution bounding is set. Provided value: 2",
    ),
    NOT_POSITIVE_PRETHRESHOLD(
      selectPartitionsParams = SELECT_PARTITIONS_PARAMS.copy(preThreshold = 0),
      exceptionMessage = "preThreshold must be positive. Provided value: 0",
    ),
    HUGE_MAX_PARTTIONS_CONTRIBUTED(
      selectPartitionsParams =
        SELECT_PARTITIONS_PARAMS.copy(maxPartitionsContributed = 110_000_000),
      exceptionMessage =
        "maxPartitionsContributed must be less than 100000000 Provided values: maxPartitionsContributed=110000000",
    ),
  }

  @Test
  fun validateSelectPartitionsParams_invalidParams_fails(
    @TestParameter testCase: InvalidSelectPartitionsParamsTestCase
  ) {
    val e =
      assertFailsWith<IllegalArgumentException> {
        validateSelectPartitionsParams(testCase.selectPartitionsParams)
      }
    assertThat(e).hasMessageThat().contains(testCase.exceptionMessage)
  }

  companion object {
    val AGGREGATION_PARAMS =
      AggregationParams(
        metrics = ImmutableList.of(MetricDefinition(COUNT), MetricDefinition(PRIVACY_ID_COUNT)),
        noiseKind = NoiseKind.LAPLACE,
        maxPartitionsContributed = 1,
        maxContributionsPerPartition = 1,
        maxContributions = null,
        minValue = 0.0,
        maxValue = 1.0,
      )

    val SELECT_PARTITIONS_PARAMS =
      SelectPartitionsParams(
        maxPartitionsContributed = 2,
        budget = AbsoluteBudgetPerOpSpec(epsilon = 1.0, delta = 1e-12),
        preThreshold = 10,
      )
  }
}

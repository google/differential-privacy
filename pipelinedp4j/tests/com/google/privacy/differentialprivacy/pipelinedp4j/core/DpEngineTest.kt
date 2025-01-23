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
import com.google.privacy.differentialprivacy.GaussianNoise
import com.google.privacy.differentialprivacy.Noise
import com.google.privacy.differentialprivacy.pipelinedp4j.core.ContributionBoundingLevel.DATASET_LEVEL
import com.google.privacy.differentialprivacy.pipelinedp4j.core.ContributionBoundingLevel.PARTITION_LEVEL
import com.google.privacy.differentialprivacy.pipelinedp4j.core.ExecutionMode.FULL_TEST_MODE
import com.google.privacy.differentialprivacy.pipelinedp4j.core.ExecutionMode.TEST_MODE_WITH_CONTRIBUTION_BOUNDING
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.COUNT
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.MEAN
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.PRIVACY_ID_COUNT
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.QUANTILES
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.SUM
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.VARIANCE
import com.google.privacy.differentialprivacy.pipelinedp4j.core.NoiseKind.GAUSSIAN
import com.google.privacy.differentialprivacy.pipelinedp4j.core.NoiseKind.LAPLACE
import com.google.privacy.differentialprivacy.pipelinedp4j.core.budget.AbsoluteBudgetPerOpSpec
import com.google.privacy.differentialprivacy.pipelinedp4j.core.budget.BudgetAccountingStrategy.NAIVE
import com.google.privacy.differentialprivacy.pipelinedp4j.core.budget.BudgetPerOpSpec
import com.google.privacy.differentialprivacy.pipelinedp4j.core.budget.RelativeBudgetPerOpSpec
import com.google.privacy.differentialprivacy.pipelinedp4j.core.budget.TotalBudget
import com.google.privacy.differentialprivacy.pipelinedp4j.dplibrary.ZeroNoiseFactory
import com.google.privacy.differentialprivacy.pipelinedp4j.local.LocalCollection
import com.google.privacy.differentialprivacy.pipelinedp4j.local.LocalEncoderFactory
import com.google.privacy.differentialprivacy.pipelinedp4j.local.LocalTable
import com.google.privacy.differentialprivacy.pipelinedp4j.local.createLocalEngine
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.DpAggregates
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.dpAggregates
import com.google.testing.junit.testparameterinjector.TestParameter
import com.google.testing.junit.testparameterinjector.TestParameterInjector
import com.google.testing.junit.testparameterinjector.TestParameters
import kotlin.test.assertFailsWith
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.ArgumentMatchers.anyDouble
import org.mockito.ArgumentMatchers.anyInt
import org.mockito.ArgumentMatchers.isA
import org.mockito.kotlin.any
import org.mockito.kotlin.anyOrNull
import org.mockito.kotlin.argThat
import org.mockito.kotlin.argumentCaptor
import org.mockito.kotlin.eq
import org.mockito.kotlin.isA
import org.mockito.kotlin.isNull
import org.mockito.kotlin.spy
import org.mockito.kotlin.verify

/**
 * Unit tests and behavioural (mocking) tests for DpEngine. All tests use the dpEngine instance
 * designed for testing which does not apply noise. Mocked tests verify that the proper internal
 * calls are made.
 */
@RunWith(TestParameterInjector::class)
class DpEngineTest {

  @Test
  fun aggregate_calledAfterDone_throws() {
    val dpEngine = DpEngine.createForTesting(LOCAL_EF, LARGE_BUDGET_SPEC, ZeroNoiseFactory())
    dpEngine.done()

    val e =
      assertFailsWith<IllegalStateException> {
        dpEngine.aggregate(
          LocalCollection(sequenceOf()),
          COUNT_PARAMS,
          testDataExtractors,
          LocalCollection(sequenceOf()),
        )
      }
    assertThat(e).hasMessageThat().contains("done() has already been called")
  }

  @Test
  fun done_calledTwice_throws() {
    val dpEngine = DpEngine.createForTesting(LOCAL_EF, LARGE_BUDGET_SPEC, ZeroNoiseFactory())
    dpEngine.done()

    val e = assertFailsWith<IllegalStateException> { dpEngine.done() }
    assertThat(e).hasMessageThat().contains("done() has already been called")
  }

  @Test
  fun aggregate_incorrectAggregateParams_throws() {
    val e =
      assertFailsWith<IllegalArgumentException> {
        DpEngine.createForTesting(LOCAL_EF, LARGE_BUDGET_SPEC, ZeroNoiseFactory())
          .aggregate(
            LocalCollection(sequenceOf()),
            // empty metrics are not allowed
            COUNT_PARAMS.copy(metrics = ImmutableList.of<MetricDefinition>()),
            testDataExtractors,
            LocalCollection(sequenceOf()),
          )
      }
    assertThat(e).hasMessageThat().contains("metrics must not be empty")
  }

  @Test
  fun aggregate_invalidDataExtactors_throws() {
    val dataExtractorsWithoutValueExtractor =
      DataExtractors.from<TestDataRow, String, String>(
        privacyIdExtractor = { row -> row.privacyId },
        LOCAL_EF.strings(),
        partitionKeyExtractor = { row -> row.partitionKey },
        LOCAL_EF.strings(),
      )

    val e =
      assertFailsWith<IllegalArgumentException> {
        DpEngine.createForTesting(LOCAL_EF, LARGE_BUDGET_SPEC, ZeroNoiseFactory())
          .aggregate(LocalCollection(sequenceOf()), SUM_PARAMS, dataExtractorsWithoutValueExtractor)
      }
    assertThat(e).hasMessageThat().contains("Metrics [SUM] require a value extractor")
  }

  @Test
  fun aggregate_partitionSelectionSetForPublicPartition_throws() {
    val e =
      assertFailsWith<IllegalArgumentException> {
        DpEngine.createForTesting(LOCAL_EF, LARGE_BUDGET_SPEC, ZeroNoiseFactory())
          .aggregate(
            LocalCollection(sequenceOf()),
            COUNT_PARAMS.copy(partitionSelectionBudget = AbsoluteBudgetPerOpSpec(1.0, 1e-5)),
            testDataExtractors,
            LocalCollection(sequenceOf()),
          )
      }
    assertThat(e)
      .hasMessageThat()
      .contains("partitionSelectionBudget can not be set for public partitions")
  }

  @Test
  fun aggregate_clampsCount() {
    val inputData =
      LocalCollection(
        sequenceOf(
          TestDataRow("Alice", "US", 10.0),
          TestDataRow("Alice", "US", 10.0),
          TestDataRow("Alice", "US", 10.0),
        )
      )
    val publicPartitions = LocalCollection(sequenceOf("US"))
    val dpEngine = DpEngine.createForTesting(LOCAL_EF, LARGE_BUDGET_SPEC, ZeroNoiseFactory())
    val params = COUNT_PARAMS.copy(maxContributionsPerPartition = 2)

    val dpAggregates =
      dpEngine.aggregate(inputData, params, testDataExtractors, publicPartitions)
        as LocalTable<String, DpAggregates>
    dpEngine.done()

    assertThat(dpAggregates.data.toList()).containsExactly(Pair("US", dpAggregates { count = 2.0 }))
  }

  /**
   * Count and sum clamping happen separately. Contributions discarded during count clamping are
   * still summed-up when sum is being computed. For example, if the client contributes
   * [10, 20, 30], maxContributionsPerPartition = 2 and maxTotalSum = 60, the user will contribute 2
   * to count and 60 to sum (i.e., all contributions will be counted towards sum as long as their
   * total sum doesn't exceed maxTotalSum). This behavior is tested below.
   */
  @Test
  fun aggregate_countClampingDoesntAffectSum() {
    val inputData =
      LocalCollection(
        sequenceOf(
          TestDataRow("Alice", "US", 10.0),
          TestDataRow("Alice", "US", 10.0),
          TestDataRow("Alice", "US", 10.0),
        )
      )
    val publicPartitions = LocalCollection(sequenceOf("US"))
    val dpEngine = DpEngine.createForTesting(LOCAL_EF, LARGE_BUDGET_SPEC, ZeroNoiseFactory())
    val params = COUNT_AND_SUM_PARAMS.copy(maxContributionsPerPartition = 2, maxTotalValue = 30.0)

    val dpAggregates =
      dpEngine.aggregate(inputData, params, testDataExtractors, publicPartitions)
        as LocalTable<String, DpAggregates>
    dpEngine.done()

    assertThat(dpAggregates.data.toList())
      .containsExactly(
        Pair(
          "US",
          dpAggregates {
            count = 2.0
            sum = 30.0
          },
        )
      )
  }

  @Test
  fun aggregate_clampsTotalSum() {
    val inputData =
      LocalCollection(
        sequenceOf(
          TestDataRow("Alice", "US", 10.0),
          TestDataRow("Alice", "US", 20.0),
          TestDataRow("Bob", "NL", -10.0),
          TestDataRow("Bob", "NL", -20.0),
        )
      )
    val publicPartitions = LocalCollection(sequenceOf("US", "NL"))
    val dpEngine = DpEngine.createForTesting(LOCAL_EF, LARGE_BUDGET_SPEC, ZeroNoiseFactory())
    val params = COUNT_AND_SUM_PARAMS.copy(minTotalValue = -25.0, maxTotalValue = 25.0)

    val dpAggregates =
      dpEngine.aggregate(inputData, params, testDataExtractors, publicPartitions)
        as LocalTable<String, DpAggregates>
    dpEngine.done()

    assertThat(dpAggregates.data.toList())
      .containsExactly(
        Pair(
          "US",
          dpAggregates {
            count = 2.0
            sum = 25.0
          },
        ),
        Pair(
          "NL",
          dpAggregates {
            count = 2.0
            sum = -25.0
          },
        ),
      )
  }

  @Test
  fun aggregate_addsNoise() {
    val inputData =
      LocalCollection(sequenceOf(TestDataRow("Alice", "US", 1.0), TestDataRow("Bob", "US", 2.0)))
    val publicPartitions = LocalCollection(sequenceOf("US"))
    val dpEngine = DpEngine.createLocalEngine(LARGE_BUDGET_SPEC)
    // Use low bounds to avoid sensitivity overflow when adding noise.
    val params =
      AggregationParams(
        metrics =
          ImmutableList.of(
            MetricDefinition(COUNT, AbsoluteBudgetPerOpSpec(0.1, 1e-5)),
            MetricDefinition(SUM, AbsoluteBudgetPerOpSpec(0.1, 1e-5)),
            MetricDefinition(PRIVACY_ID_COUNT, AbsoluteBudgetPerOpSpec(0.1, 1e-5)),
          ),
        noiseKind = GAUSSIAN,
        maxPartitionsContributed = 5,
        maxContributionsPerPartition = 5,
        minTotalValue = -5.0,
        maxTotalValue = 5.0,
      )

    val dpAggregates =
      dpEngine.aggregate(inputData, params, testDataExtractors, publicPartitions)
        as LocalTable<String, DpAggregates>
    val dpAggregatesAnotherRun =
      dpEngine.aggregate(inputData, params, testDataExtractors, publicPartitions)
        as LocalTable<String, DpAggregates>
    dpEngine.done()

    assertThat(dpAggregates.data.toMap()["US"]!!.count)
      .isNotEqualTo(dpAggregatesAnotherRun.data.toMap()["US"]!!.count)
    assertThat(dpAggregates.data.toMap()["US"]!!.sum)
      .isNotEqualTo(dpAggregatesAnotherRun.data.toMap()["US"]!!.sum)
    assertThat(dpAggregates.data.toMap()["US"]!!.privacyIdCount)
      .isNotEqualTo(dpAggregatesAnotherRun.data.toMap()["US"]!!.privacyIdCount)
  }

  @Test
  fun aggregate_aggregateReturnsMean() {
    val inputData =
      LocalCollection(sequenceOf(TestDataRow("Alice", "US", 1.0), TestDataRow("Bob", "US", 2.0)))
    val publicPartitions = LocalCollection(sequenceOf("US"))
    val dpEngine = DpEngine.createLocalEngine(LARGE_BUDGET_SPEC)
    // Use low bounds to avoid sensitivity overflow when adding noise.
    val params =
      AggregationParams(
        metrics =
          ImmutableList.of(MetricDefinition(COUNT), MetricDefinition(SUM), MetricDefinition(MEAN)),
        noiseKind = LAPLACE,
        maxPartitionsContributed = 1,
        maxContributionsPerPartition = 1,
        minValue = -2.0,
        maxValue = 2.0,
      )

    val dpAggregates =
      dpEngine.aggregate(inputData, params, testDataExtractors, publicPartitions)
        as LocalTable<String, DpAggregates>
    dpEngine.done()

    val partitionResult = dpAggregates.data.toMap()["US"]!!
    assertThat(partitionResult.count).isWithin(1e-1).of(2.0)
    assertThat(partitionResult.sum).isWithin(1e-1).of(3.0)
    assertThat(partitionResult.mean).isWithin(1e-10).of(partitionResult.sum / partitionResult.count)
  }

  @Test
  fun aggregate_aggregateReturnsVariance() {
    val inputData =
      LocalCollection(sequenceOf(TestDataRow("Alice", "US", 1.0), TestDataRow("Bob", "US", 2.0)))
    val publicPartitions = LocalCollection(sequenceOf("US"))
    val dpEngine = DpEngine.createLocalEngine(LARGE_BUDGET_SPEC)
    // Use low bounds to avoid sensitivity overflow when adding noise.
    val params =
      AggregationParams(
        metrics =
          ImmutableList.of(
            MetricDefinition(COUNT),
            MetricDefinition(SUM),
            MetricDefinition(MEAN),
            MetricDefinition(VARIANCE),
          ),
        noiseKind = LAPLACE,
        maxPartitionsContributed = 1,
        maxContributionsPerPartition = 1,
        minValue = -2.0,
        maxValue = 2.0,
      )
    val dpAggregates =
      dpEngine.aggregate(inputData, params, testDataExtractors, publicPartitions)
        as LocalTable<String, DpAggregates>
    dpEngine.done()

    val partitionResult = dpAggregates.data.toMap()["US"]!!
    assertThat(partitionResult.count).isWithin(1e-1).of(2.0)
    assertThat(partitionResult.sum).isWithin(1e-1).of(3.0)
    assertThat(partitionResult.mean).isWithin(1e-10).of(partitionResult.sum / partitionResult.count)
    assertThat(partitionResult.variance)
      .isWithin(1e-1)
      .of(((1.0 * 1.0) + (2.0 * 2.0)) / 2.0 - (3.0 / 2.0) * (3.0 / 2.0))
  }

  enum class CombinersTestCase(
    val params: AggregationParams,
    val countCombinerPresent: Boolean,
    val sumCombinerPresent: Boolean,
  ) {
    COUNT(params = COUNT_PARAMS, countCombinerPresent = true, sumCombinerPresent = false),
    SUM(params = SUM_PARAMS, countCombinerPresent = false, sumCombinerPresent = true),
    COUNT_AND_SUM(
      params = COUNT_AND_SUM_PARAMS,
      countCombinerPresent = true,
      sumCombinerPresent = true,
    ),
  }

  @Test
  fun aggregate_createsCombinersForMetrics(@TestParameter testCase: CombinersTestCase) {
    val graphFactorySpy: ComputationalGraphFactory = spy()
    val dpEngine =
      DpEngine.createForTesting(
        LOCAL_EF,
        LARGE_BUDGET_SPEC,
        computationalGraphFactory = graphFactorySpy,
      )

    val unused =
      dpEngine.aggregate(
        LocalCollection(sequenceOf()),
        testCase.params,
        testDataExtractors,
        LocalCollection(sequenceOf()),
      )

    val combinerCaptor = argumentCaptor<CompoundCombiner>()
    verify(graphFactorySpy)
      .createForPublicPartitions<TestDataRow, String, String>(
        any(),
        combinerCaptor.capture(),
        any(),
        any(),
        any(),
        any(),
      )
    val combiner = combinerCaptor.firstValue
    assertThat(combiner.combiners.any { it is CountCombiner })
      .isEqualTo(testCase.countCombinerPresent)
    assertThat(combiner.combiners.any { it is SumCombiner }).isEqualTo(testCase.sumCombinerPresent)
  }

  enum class ContributionSamplerTestCase(
    val aggregationParams: AggregationParams,
    val expectedContributionSampler: Class<out ContributionSampler<*, *>>,
  ) {
    DATASET_LEVEL_WITH_PRIVACY_ID_COUNT_METRIC(
      PRIVACY_ID_COUNT_PARAMS.copy(contributionBoundingLevel = DATASET_LEVEL),
      PartitionSampler::class.java,
    ),
    DATASET_LEVEL_WITH_COUNT_METRIC(
      COUNT_PARAMS.copy(contributionBoundingLevel = DATASET_LEVEL),
      PartitionSampler::class.java,
    ),
    DATASET_LEVEL_WITH_SUM_METRIC(
      SUM_PARAMS.copy(contributionBoundingLevel = DATASET_LEVEL),
      PartitionSampler::class.java,
    ),
    DATASET_LEVEL_WITH_MEAN_METRIC(
      MEAN_PARAMS.copy(contributionBoundingLevel = DATASET_LEVEL),
      PartitionAndPerPartitionSampler::class.java,
    ),
    DATASET_LEVEL_WITH_QUANTILES_METRIC(
      QUANTILES_PARAMS.copy(contributionBoundingLevel = DATASET_LEVEL),
      PartitionAndPerPartitionSampler::class.java,
    ),
    DATASET_LEVEL_WITH_ALL_METRICS(
      AggregationParams(
        contributionBoundingLevel = DATASET_LEVEL,
        noiseKind = GAUSSIAN,
        metrics =
          ImmutableList.of(
            MetricDefinition(PRIVACY_ID_COUNT),
            MetricDefinition(COUNT),
            MetricDefinition(SUM),
            MetricDefinition(MEAN),
            MetricDefinition(QUANTILES(ranks = ImmutableList.of())),
          ),
        maxPartitionsContributed = 10,
        maxContributionsPerPartition = 20,
        minValue = -10.0,
        maxValue = 10.0,
      ),
      PartitionAndPerPartitionSampler::class.java,
    ),
    PARTITION_LEVEL_WITH_PRIVACY_ID_COUNT_METRIC(
      PRIVACY_ID_COUNT_PARAMS.copy(
        contributionBoundingLevel = PARTITION_LEVEL,
        maxPartitionsContributed = 1,
      ),
      NoPrivacySampler::class.java,
    ),
    PARTITION_LEVEL_WITH_COUNT_METRIC(
      COUNT_PARAMS.copy(contributionBoundingLevel = PARTITION_LEVEL, maxPartitionsContributed = 1),
      NoPrivacySampler::class.java,
    ),
    PARTITION_LEVEL_WITH_SUM_METRIC(
      SUM_PARAMS.copy(contributionBoundingLevel = PARTITION_LEVEL, maxPartitionsContributed = 1),
      NoPrivacySampler::class.java,
    ),
    PARTITION_LEVEL_WITH_MEAN_METRIC(
      MEAN_PARAMS.copy(contributionBoundingLevel = PARTITION_LEVEL, maxPartitionsContributed = 1),
      PerPartitionContributionsSampler::class.java,
    ),
    PARTITION_LEVEL_WITH_QUANTILES_METRIC(
      QUANTILES_PARAMS.copy(
        contributionBoundingLevel = PARTITION_LEVEL,
        maxPartitionsContributed = 1,
      ),
      PerPartitionContributionsSampler::class.java,
    ),
    PARTITION_LEVEL_WITH_ALL_METRICS(
      AggregationParams(
        contributionBoundingLevel = PARTITION_LEVEL,
        noiseKind = GAUSSIAN,
        metrics =
          ImmutableList.of(
            MetricDefinition(PRIVACY_ID_COUNT),
            MetricDefinition(COUNT),
            MetricDefinition(SUM),
            MetricDefinition(MEAN),
            MetricDefinition(QUANTILES(ranks = ImmutableList.of())),
          ),
        maxPartitionsContributed = 1,
        maxContributionsPerPartition = 20,
        minValue = -10.0,
        maxValue = 10.0,
      ),
      PerPartitionContributionsSampler::class.java,
    ),
    // Count is an example of a metric that does not require per-partition bounded input.
    DATASET_LEVEL_COUNT_WITH_TEST_MODE_WITH_CONTRIBUTION_BOUNDING(
      COUNT_PARAMS.copy(
        contributionBoundingLevel = DATASET_LEVEL,
        executionMode = TEST_MODE_WITH_CONTRIBUTION_BOUNDING,
      ),
      PartitionSampler::class.java,
    ),
    // Mean is an example of a metric that requires per-partition bounded input.
    DATASET_LEVEL_MEAN_WITH_TEST_MODE_WITH_CONTRIBUTION_BOUNDING(
      MEAN_PARAMS.copy(
        contributionBoundingLevel = DATASET_LEVEL,
        executionMode = TEST_MODE_WITH_CONTRIBUTION_BOUNDING,
      ),
      PartitionAndPerPartitionSampler::class.java,
    ),
    // Count is an example of a metric that does not require per-partition bounded input.
    PARTITION_LEVEL_COUNT_WITH_TEST_MODE_WITH_CONTRIBUTION_BOUNDING(
      COUNT_PARAMS.copy(
        contributionBoundingLevel = PARTITION_LEVEL,
        maxPartitionsContributed = 1,
        executionMode = TEST_MODE_WITH_CONTRIBUTION_BOUNDING,
      ),
      NoPrivacySampler::class.java,
    ),
    // Mean is an example of a metric that requires per-partition bounded input.
    PARTITION_LEVEL_MEAN_WITH_TEST_MODE_WITH_CONTRIBUTION_BOUNDING(
      MEAN_PARAMS.copy(
        contributionBoundingLevel = PARTITION_LEVEL,
        maxPartitionsContributed = 1,
        executionMode = TEST_MODE_WITH_CONTRIBUTION_BOUNDING,
      ),
      PerPartitionContributionsSampler::class.java,
    ),
    DATASET_LEVEL_WITH_FULL_TEST_MODE(
      COUNT_PARAMS.copy(contributionBoundingLevel = DATASET_LEVEL, executionMode = FULL_TEST_MODE),
      NoPrivacySampler::class.java,
    ),
    PARTITION_LEVEL_WITH_FULL_TEST_MODE(
      COUNT_PARAMS.copy(
        contributionBoundingLevel = PARTITION_LEVEL,
        maxPartitionsContributed = 1,
        executionMode = FULL_TEST_MODE,
      ),
      NoPrivacySampler::class.java,
    ),
  }

  @Test
  fun aggregate_createsCorrectContributionSampler(
    @TestParameter testCase: ContributionSamplerTestCase
  ) {
    val graphFactorySpy: ComputationalGraphFactory = spy()
    val dpEngine =
      DpEngine.createForTesting(
        LOCAL_EF,
        LARGE_BUDGET_SPEC,
        computationalGraphFactory = graphFactorySpy,
      )

    val unused =
      dpEngine.aggregate(
        LocalCollection(sequenceOf()),
        testCase.aggregationParams,
        testDataExtractors,
      )

    val contributionSamplerCaptor = argumentCaptor<ContributionSampler<String, String>>()
    verify(graphFactorySpy)
      .createForPrivatePartitions<TestDataRow, String, String>(
        // isA(testCase.expectedContributionSampler) fails for some reason with message: isA(...)
        // must not be null.
        contributionSamplerCaptor.capture(),
        anyOrNull(),
        any(),
        any(),
        any(),
      )
    assertThat(contributionSamplerCaptor.firstValue)
      .isInstanceOf(testCase.expectedContributionSampler)
  }

  @TestParameters(
    "{executionMode: TEST_MODE_WITH_CONTRIBUTION_BOUNDING}",
    "{executionMode: FULL_TEST_MODE}",
  )
  @Test
  fun aggregate_testMode_createsNoPrivacyPartitionSelector(executionMode: ExecutionMode) {
    val graphFactorySpy: ComputationalGraphFactory = spy()
    val dpEngine =
      DpEngine.createForTesting(
        LOCAL_EF,
        LARGE_BUDGET_SPEC,
        computationalGraphFactory = graphFactorySpy,
      )

    val unused =
      dpEngine.aggregate(
        LocalCollection(sequenceOf()),
        PRIVACY_ID_COUNT_PARAMS.copy(executionMode = executionMode),
        testDataExtractors,
      )

    verify(graphFactorySpy)
      .createForPrivatePartitions<TestDataRow, String, String>(
        any(),
        isA<NoPrivacyPartitionSelector>(),
        any(),
        any(),
        any(),
      )
  }

  @Test
  fun aggregate_withPublicPartitions_createsGraphWithPublicPartitions() {
    val graphFactorySpy: ComputationalGraphFactory = spy()
    val dpEngine =
      DpEngine.createForTesting(
        LOCAL_EF,
        LARGE_BUDGET_SPEC,
        computationalGraphFactory = graphFactorySpy,
      )
    val publicPartitions = LocalCollection<String>(sequenceOf("Green", "White", "Green"))

    val unused =
      dpEngine.aggregate(
        LocalCollection(sequenceOf()),
        COUNT_AND_SUM_PARAMS,
        testDataExtractors,
        publicPartitions,
      )

    val publicPartitionsCaptor = argumentCaptor<FrameworkCollection<String>>()
    verify(graphFactorySpy)
      .createForPublicPartitions<TestDataRow, String, String>(
        any(),
        any(),
        any(),
        any(),
        publicPartitionsCaptor.capture(),
        any(),
      )
    assertThat(publicPartitionsCaptor.allValues).hasSize(1)
    assertThat(publicPartitionsCaptor.firstValue).isInstanceOf(LocalCollection::class.java)
    assertThat((publicPartitionsCaptor.firstValue as LocalCollection<String>).data.toList())
      .isEqualTo(listOf("Green", "White"))
  }

  @Test
  fun aggregate_passesDataExtractorsToComputationalGraph() {
    val graphFactorySpy: ComputationalGraphFactory = spy()
    val dpEngine =
      DpEngine.createForTesting(
        LOCAL_EF,
        LARGE_BUDGET_SPEC,
        computationalGraphFactory = graphFactorySpy,
      )

    val unused =
      dpEngine.aggregate(
        LocalCollection(sequenceOf()),
        COUNT_AND_SUM_PARAMS,
        testDataExtractors,
        LocalCollection(sequenceOf()),
      )

    verify(graphFactorySpy)
      .createForPublicPartitions(any(), any(), eq(testDataExtractors), any(), any(), any())
  }

  @Test
  fun aggregate_passesLocalFactoryToComputationalGraph() {
    val graphFactorySpy: ComputationalGraphFactory = spy()
    val dpEngine =
      DpEngine.createForTesting(
        LOCAL_EF,
        LARGE_BUDGET_SPEC,
        computationalGraphFactory = graphFactorySpy,
      )

    val unused =
      dpEngine.aggregate(
        LocalCollection(sequenceOf()),
        COUNT_AND_SUM_PARAMS,
        testDataExtractors,
        LocalCollection(sequenceOf()),
      )

    verify(graphFactorySpy)
      .createForPublicPartitions<TestDataRow, String, String>(
        any(),
        any(),
        any(),
        eq(LOCAL_EF),
        any(),
        any(),
      )
  }

  @Test
  fun aggregate_privatePartitions_createsGraphWithPrivatePartitions() {
    val graphFactorySpy: ComputationalGraphFactory = spy()
    val dpEngine =
      DpEngine.createForTesting(
        LOCAL_EF,
        LARGE_BUDGET_SPEC,
        computationalGraphFactory = graphFactorySpy,
      )

    val unused =
      dpEngine.aggregate(
        LocalCollection(sequenceOf()),
        COUNT_AND_SUM_PARAMS.copy(preThreshold = 10),
        testDataExtractors,
      )

    val combinerCaptor = argumentCaptor<CompoundCombiner>()
    verify(graphFactorySpy)
      .createForPrivatePartitions(
        isA<PartitionSampler<String, String>>(),
        argThat { (this as DpLibPreAggregationPartitionSelector).preThreshold == 10 },
        combinerCaptor.capture(),
        isA<DataExtractors<TestDataRow, String, String>>(),
        eq(LOCAL_EF),
      )
    val combiners = combinerCaptor.firstValue.combiners
    assertThat(combiners.count()).isEqualTo(3)
    assertThat(combiners.any { it is CountCombiner }).isTrue()
    assertThat(combiners.any { it is SumCombiner }).isTrue()
    // No PRIVACY_ID_COUNT in metrics, so ExactPrivacyIdCountCombiner should be used.
    assertThat(combiners.any { it is ExactPrivacyIdCountCombiner }).isTrue()
  }

  @Test
  fun aggregate_createsGraphWithPostAggregationPartitionSelection() {
    val graphFactorySpy: ComputationalGraphFactory = spy()
    val dpEngine =
      DpEngine.createForTesting(
        LOCAL_EF,
        LARGE_BUDGET_SPEC,
        computationalGraphFactory = graphFactorySpy,
      )

    val unused =
      dpEngine.aggregate(
        LocalCollection(sequenceOf()),
        PRIVACY_ID_COUNT_PARAMS.copy(preThreshold = 10),
        testDataExtractors,
      )

    val combinerCaptor = argumentCaptor<CompoundCombiner>()

    verify(graphFactorySpy)
      .createForPrivatePartitions<TestDataRow, String, String>(
        any(),
        preAggregationPartitionSelector = isNull(),
        combinerCaptor.capture(),
        any(),
        any(),
      )
    val compoundCombiner = combinerCaptor.firstValue
    // Post aggregation partition selection is performed.
    assertThat(compoundCombiner.hasPostAggregationCombiner()).isTrue()
    val combiners = compoundCombiner.combiners.toList()
    assertThat(combiners.size).isEqualTo(1)
    val partitionSelector =
      (combiners[0] as PostAggregationPartitionSelectionCombiner).getPartitionSelector()
    assertThat(partitionSelector).isInstanceOf(PostAggregationPartitionSelectorImpl::class.java)
    assertThat((partitionSelector as PostAggregationPartitionSelectorImpl).preThreshold)
      .isEqualTo(10)
  }

  @Test
  fun selectPartitons_computationalGraphIsCorrect() {
    val graphFactorySpy: ComputationalGraphFactory = spy()
    val dpEngine =
      DpEngine.createForTesting(
        LOCAL_EF,
        LARGE_BUDGET_SPEC,
        computationalGraphFactory = graphFactorySpy,
      )

    val unused =
      dpEngine.selectPartitions(
        LocalCollection(sequenceOf()),
        SelectPartitionsParams(maxPartitionsContributed = 5, preThreshold = 100),
        testDataExtractors,
      )

    val partitionSelectorCaptor = argumentCaptor<DpLibPreAggregationPartitionSelector>()
    verify(graphFactorySpy)
      .createForSelectPartitions(
        isA<PartitionSamplerWithoutValues<String, String>>(),
        partitionSelectorCaptor.capture(),
        isA<DataExtractors<TestDataRow, String, String>>(),
        any(),
      )
    assertThat(partitionSelectorCaptor.allValues.map { it.preThreshold }).isEqualTo(listOf(100))
  }

  @Test
  fun selectPartitons_fullTestMode_computationalGraphIsCorrect() {
    val graphFactorySpy: ComputationalGraphFactory = spy()
    val dpEngine =
      DpEngine.createForTesting(
        LOCAL_EF,
        LARGE_BUDGET_SPEC,
        computationalGraphFactory = graphFactorySpy,
      )

    val unused =
      dpEngine.selectPartitions(
        LocalCollection(sequenceOf()),
        SelectPartitionsParams(executionMode = FULL_TEST_MODE, maxPartitionsContributed = 5),
        testDataExtractors,
      )

    verify(graphFactorySpy)
      .createForSelectPartitions(
        isA<NoPrivacySampler<String, String>>(),
        isA<NoPrivacyPartitionSelector>(),
        isA<DataExtractors<TestDataRow, String, String>>(),
        any(),
      )
  }

  @Test
  fun selectPartitons_testModeWithContributionBounding_computationalGraphIsCorrect() {
    val graphFactorySpy: ComputationalGraphFactory = spy()
    val dpEngine =
      DpEngine.createForTesting(
        LOCAL_EF,
        LARGE_BUDGET_SPEC,
        computationalGraphFactory = graphFactorySpy,
      )

    val unused =
      dpEngine.selectPartitions(
        LocalCollection(sequenceOf()),
        SelectPartitionsParams(
          executionMode = TEST_MODE_WITH_CONTRIBUTION_BOUNDING,
          maxPartitionsContributed = 5,
        ),
        testDataExtractors,
      )

    verify(graphFactorySpy)
      .createForSelectPartitions(
        isA<PartitionSamplerWithoutValues<String, String>>(),
        isA<NoPrivacyPartitionSelector>(),
        isA<DataExtractors<TestDataRow, String, String>>(),
        any(),
      )
  }

  enum class CountSumBudgetTestCase(
    val totalMetricsBudget: TotalBudget,
    val requestedCountBudget: BudgetPerOpSpec,
    val requestedSumBudget: BudgetPerOpSpec,
    val countNoiseEpsilon: Double,
    val countNoiseDelta: Double,
    val sumNoiseEpsilon: Double,
    val sumNoiseDelta: Double,
  ) {
    ABSOLUTE(
      totalMetricsBudget = TotalBudget(3.0, 0.3),
      requestedCountBudget = AbsoluteBudgetPerOpSpec(1.0, 0.1),
      requestedSumBudget = AbsoluteBudgetPerOpSpec(2.0, 0.2),
      countNoiseEpsilon = 1.0,
      countNoiseDelta = 0.1,
      sumNoiseEpsilon = 2.0,
      sumNoiseDelta = 0.2,
    ),
    RELATIVE(
      totalMetricsBudget = TotalBudget(3.0, 0.3),
      requestedCountBudget = RelativeBudgetPerOpSpec(1.0),
      requestedSumBudget = RelativeBudgetPerOpSpec(2.0),
      countNoiseEpsilon = 1.0,
      countNoiseDelta = 0.1,
      sumNoiseEpsilon = 2.0,
      sumNoiseDelta = 0.2,
    ),
    ABSOLUTE_AND_RELATIVE(
      totalMetricsBudget = TotalBudget(3.0, 0.3),
      requestedCountBudget = AbsoluteBudgetPerOpSpec(1.0, 0.1),
      requestedSumBudget = RelativeBudgetPerOpSpec(1.0),
      countNoiseEpsilon = 1.0,
      countNoiseDelta = 0.1,
      sumNoiseEpsilon = 2.0,
      sumNoiseDelta = 0.2,
    ),
  }

  @Test
  fun aggregate_withRelativeNaiveBudgetSplit_allocatesBudgetAccordingToAggregationBudgetSpec(
    @TestParameter testCase: CountSumBudgetTestCase
  ) {
    val inputData =
      LocalCollection(sequenceOf(TestDataRow("Alice", "US", 1.0), TestDataRow("Bob", "US", 2.0)))
    val publicPartitions = LocalCollection(sequenceOf("US"))
    val gaussianNoiseSpy: GaussianNoise = spy()
    val noiseFactoryMock: (NoiseKind) -> Noise = { _ -> gaussianNoiseSpy }
    val budgetSpec =
      DpEngineBudgetSpec(budget = testCase.totalMetricsBudget, accountingStrategy = NAIVE)
    val dpEngine = DpEngine.createForTesting(LOCAL_EF, budgetSpec, noiseFactoryMock)
    // Use low bounds to avoid sensitivity overflow when adding noise.
    val params =
      AggregationParams(
        metrics =
          ImmutableList.of(
            MetricDefinition(COUNT, testCase.requestedCountBudget),
            MetricDefinition(SUM, testCase.requestedSumBudget),
          ),
        noiseKind = GAUSSIAN,
        // Choose large values to avoid contribution clamping but keep the values low enough to
        // avoid sensitivity overflow.
        maxPartitionsContributed = 100,
        maxContributionsPerPartition = 100,
        minTotalValue = -100.0,
        maxTotalValue = 100.0,
      )

    val result =
      dpEngine.aggregate(inputData, params, testDataExtractors, publicPartitions)
        as LocalTable<String, DpAggregates>
    dpEngine.done()

    // Access the result to trigger the computation.
    assertThat(result.data.toList()).isNotEmpty()
    // Check parameters of the noise addition to Count.
    val countDeltaCaptor = argumentCaptor<Double>()
    verify(gaussianNoiseSpy)
      .addNoise(
        anyDouble(),
        anyInt(),
        anyDouble(),
        eq(testCase.countNoiseEpsilon),
        countDeltaCaptor.capture(),
      )
    assertThat(countDeltaCaptor.firstValue).isWithin(1e-15).of(testCase.countNoiseDelta)
    // Check parameters of the noise addition to Sum.
    val sumDeltaCaptor = argumentCaptor<Double>()
    verify(gaussianNoiseSpy)
      .addNoise(
        anyDouble(),
        anyInt(),
        anyDouble(),
        eq(testCase.sumNoiseEpsilon),
        sumDeltaCaptor.capture(),
      )
    assertThat(sumDeltaCaptor.firstValue).isWithin(1e-15).of(testCase.sumNoiseDelta)
  }

  enum class MeanBudgetTestCase(
    val totalMetricsBudget: TotalBudget,
    val requestedCountBudget: BudgetPerOpSpec?,
    val requestedSumBudget: BudgetPerOpSpec?,
    val requestedMeanBudget: BudgetPerOpSpec?,
    val countNoiseEpsilon: Double,
    val countNoiseDelta: Double,
    val sumNoiseEpsilon: Double,
    val sumNoiseDelta: Double,
  ) {
    ABSOLUTE_COUNT_SUM(
      totalMetricsBudget = TotalBudget(3.0, 0.3),
      requestedCountBudget = AbsoluteBudgetPerOpSpec(1.0, 0.1),
      requestedSumBudget = AbsoluteBudgetPerOpSpec(2.0, 0.2),
      requestedMeanBudget = null,
      countNoiseEpsilon = 1.0,
      countNoiseDelta = 0.1,
      sumNoiseEpsilon = 2.0,
      sumNoiseDelta = 0.2,
    ),
    RELATIVE_COUNT_SUM(
      totalMetricsBudget = TotalBudget(3.0, 0.3),
      requestedCountBudget = RelativeBudgetPerOpSpec(1.0),
      requestedSumBudget = RelativeBudgetPerOpSpec(2.0),
      requestedMeanBudget = null,
      countNoiseEpsilon = 1.0,
      countNoiseDelta = 0.1,
      sumNoiseEpsilon = 2.0,
      sumNoiseDelta = 0.2,
    ),
    ABSOLUTE_AND_RELATIVE_COUNT_SUM(
      totalMetricsBudget = TotalBudget(3.0, 0.3),
      requestedCountBudget = AbsoluteBudgetPerOpSpec(1.0, 0.1),
      requestedSumBudget = RelativeBudgetPerOpSpec(1.0),
      requestedMeanBudget = null,
      countNoiseEpsilon = 1.0,
      countNoiseDelta = 0.1,
      sumNoiseEpsilon = 2.0,
      sumNoiseDelta = 0.2,
    ),
    ABSOLUTE_MEAN(
      totalMetricsBudget = TotalBudget(3.0, 0.3),
      requestedCountBudget = null,
      requestedSumBudget = null,
      requestedMeanBudget = AbsoluteBudgetPerOpSpec(2.0, 0.2),
      countNoiseEpsilon = 1.0,
      countNoiseDelta = 0.1,
      sumNoiseEpsilon = 1.0,
      sumNoiseDelta = 0.1,
    ),
    RELATIVE_MEAN(
      totalMetricsBudget = TotalBudget(2.0, 0.2),
      requestedCountBudget = null,
      requestedSumBudget = null,
      requestedMeanBudget = RelativeBudgetPerOpSpec(1.0),
      countNoiseEpsilon = 1.0,
      countNoiseDelta = 0.1,
      sumNoiseEpsilon = 1.0,
      sumNoiseDelta = 0.1,
    ),
    MEAN_DEFAULT(
      totalMetricsBudget = TotalBudget(2.0, 0.2),
      requestedCountBudget = null,
      requestedSumBudget = null,
      requestedMeanBudget = null,
      countNoiseEpsilon = 1.0,
      countNoiseDelta = 0.1,
      sumNoiseEpsilon = 1.0,
      sumNoiseDelta = 0.1,
    ),
    MEAN_COUNT_PROVIDED_SUM_DEFAULT(
      totalMetricsBudget = TotalBudget(2.0, 0.2),
      requestedCountBudget = AbsoluteBudgetPerOpSpec(1.5, 0.05),
      requestedSumBudget = null,
      requestedMeanBudget = null,
      countNoiseEpsilon = 1.5,
      countNoiseDelta = 0.05,
      sumNoiseEpsilon = 0.5,
      sumNoiseDelta = 0.15,
    ),
  }

  @Test
  fun aggregate_withAbsoluteNaiveBudgetSplit_allocatesBudgetAccordingToAggregationBudgetSpec(
    @TestParameter testCase: MeanBudgetTestCase
  ) {
    val inputData =
      LocalCollection(sequenceOf(TestDataRow("Alice", "US", 1.0), TestDataRow("Bob", "US", 2.0)))
    val publicPartitions = LocalCollection(sequenceOf("US"))
    val gaussianNoiseSpy: GaussianNoise = spy()
    val noiseFactoryMock: (NoiseKind) -> Noise = { _ -> gaussianNoiseSpy }
    val budgetSpec =
      DpEngineBudgetSpec(budget = testCase.totalMetricsBudget, accountingStrategy = NAIVE)
    val dpEngine = DpEngine.createForTesting(LOCAL_EF, budgetSpec, noiseFactoryMock)
    // Use low bounds to avoid sensitivity overflow when adding noise.
    val params =
      AggregationParams(
        metrics =
          ImmutableList.of(
            MetricDefinition(COUNT, testCase.requestedCountBudget),
            MetricDefinition(SUM, testCase.requestedSumBudget),
            MetricDefinition(MEAN, testCase.requestedMeanBudget),
          ),
        noiseKind = GAUSSIAN,
        maxPartitionsContributed = 10,
        maxContributionsPerPartition = 20,
        minValue = -10.0,
        maxValue = 10.0,
      )

    val result =
      dpEngine.aggregate(inputData, params, testDataExtractors, publicPartitions)
        as LocalTable<String, DpAggregates>
    dpEngine.done()

    // Access the result to trigger the computation.
    assertThat(result.data.toList()).isNotEmpty()
    // Check parameters of the noise addition to Count.
    val countDeltaCaptor = argumentCaptor<Double>()
    verify(gaussianNoiseSpy)
      .addNoise(
        eq(2.0), // True count
        anyInt(),
        anyDouble(),
        eq(testCase.countNoiseEpsilon),
        countDeltaCaptor.capture(),
      )
    assertThat(countDeltaCaptor.firstValue).isWithin(1e-15).of(testCase.countNoiseDelta)
    // Check parameters of the noise addition to Sum.
    val sumDeltaCaptor = argumentCaptor<Double>()
    verify(gaussianNoiseSpy)
      .addNoise(
        eq(3.0), // True sum
        anyInt(),
        anyDouble(),
        eq(testCase.sumNoiseEpsilon),
        sumDeltaCaptor.capture(),
      )
    assertThat(sumDeltaCaptor.firstValue).isWithin(1e-15).of(testCase.sumNoiseDelta)
  }

  enum class VarianceBudgetTestCase(
    val totalMetricsBudget: TotalBudget,
    val requestedVarianceBudget: BudgetPerOpSpec?,
    val countNoiseEpsilon: Double,
    val countNoiseDelta: Double,
    val sumNoiseEpsilon: Double,
    val sumNoiseDelta: Double,
    val sumSquaresNoiseEpsilon: Double,
    val sumSquaresNoiseDelta: Double,
  ) {
    DEFAULT_BUDGET(
      totalMetricsBudget = TotalBudget(3.0, 0.3),
      requestedVarianceBudget = null,
      countNoiseEpsilon = 1.0,
      countNoiseDelta = 0.1,
      sumNoiseEpsilon = 1.0,
      sumNoiseDelta = 0.1,
      sumSquaresNoiseEpsilon = 1.0,
      sumSquaresNoiseDelta = 0.1,
    ),
    VARIANCE_EVEN_SPLIT_ABSOLUTE(
      totalMetricsBudget = TotalBudget(6.0, 0.6),
      requestedVarianceBudget = AbsoluteBudgetPerOpSpec(6.0, 0.6),
      countNoiseEpsilon = 2.0,
      countNoiseDelta = 0.2,
      sumNoiseEpsilon = 2.0,
      sumNoiseDelta = 0.2,
      sumSquaresNoiseEpsilon = 2.0,
      sumSquaresNoiseDelta = 0.2,
    ),
    VARIANCE_EVEN_SPLIT_RELATIVE(
      totalMetricsBudget = TotalBudget(6.0, 0.6),
      requestedVarianceBudget = RelativeBudgetPerOpSpec(6.0),
      countNoiseEpsilon = 2.0,
      countNoiseDelta = 0.2,
      sumNoiseEpsilon = 2.0,
      sumNoiseDelta = 0.2,
      sumSquaresNoiseEpsilon = 2.0,
      sumSquaresNoiseDelta = 0.2,
    ),
  }

  @Test
  fun aggregate_withAbsoluteNaiveBudgetSplit_allocatesBudgetAccordingToAggregationBudgetSpec(
    @TestParameter testCase: VarianceBudgetTestCase
  ) {
    val inputData =
      LocalCollection(sequenceOf(TestDataRow("Alice", "US", 1.0), TestDataRow("Bob", "US", 2.0)))
    val publicPartitions = LocalCollection(sequenceOf("US"))
    val gaussianNoiseSpy: GaussianNoise = spy()
    val noiseFactoryMock: (NoiseKind) -> Noise = { _ -> gaussianNoiseSpy }
    val budgetSpec =
      DpEngineBudgetSpec(budget = testCase.totalMetricsBudget, accountingStrategy = NAIVE)
    val dpEngine = DpEngine.createForTesting(LOCAL_EF, budgetSpec, noiseFactoryMock)
    // Use low bounds to avoid sensitivity overflow when adding noise.
    val params =
      AggregationParams(
        metrics = ImmutableList.of(MetricDefinition(VARIANCE, testCase.requestedVarianceBudget)),
        noiseKind = GAUSSIAN,
        maxPartitionsContributed = 10,
        maxContributionsPerPartition = 20,
        minValue = -10.0,
        maxValue = 10.0,
      )

    val result =
      dpEngine.aggregate(inputData, params, testDataExtractors, publicPartitions)
        as LocalTable<String, DpAggregates>
    dpEngine.done()

    // Access the result to trigger the computation.
    assertThat(result.data.toList()).isNotEmpty()
    // Check parameters of the noise addition to Count.
    val countDeltaCaptor = argumentCaptor<Double>()
    verify(gaussianNoiseSpy)
      .addNoise(
        eq(2.0), // True count
        anyInt(),
        anyDouble(),
        eq(testCase.countNoiseEpsilon),
        countDeltaCaptor.capture(),
      )
    assertThat(countDeltaCaptor.firstValue).isWithin(1e-15).of(testCase.countNoiseDelta)
    // Check parameters of the noise addition to Sum.
    val sumDeltaCaptor = argumentCaptor<Double>()
    verify(gaussianNoiseSpy)
      .addNoise(
        eq(3.0), // True sum
        anyInt(),
        anyDouble(),
        eq(testCase.sumNoiseEpsilon),
        sumDeltaCaptor.capture(),
      )
    assertThat(sumDeltaCaptor.firstValue).isWithin(1e-15).of(testCase.sumNoiseDelta)
    // Check parameters of the noise addition to Sum of Squares.
    val sumOfSquaresDeltaCaptor = argumentCaptor<Double>()
    verify(gaussianNoiseSpy)
      .addNoise(
        eq(5.0), // True sum of squares
        anyInt(),
        anyDouble(),
        eq(testCase.sumSquaresNoiseEpsilon),
        sumOfSquaresDeltaCaptor.capture(),
      )
    assertThat(sumOfSquaresDeltaCaptor.firstValue).isWithin(1e-15).of(testCase.sumSquaresNoiseDelta)
  }

  companion object {
    // A DpEngineBudgetSpec with budget large enough to make sure that tests don't run out of it.
    private val LARGE_BUDGET_SPEC =
      DpEngineBudgetSpec(budget = TotalBudget(epsilon = 2000.0, delta = 0.999999))
    private val PRIVACY_ID_COUNT_PARAMS =
      AggregationParams(
        metrics = ImmutableList.of(MetricDefinition(PRIVACY_ID_COUNT)),
        noiseKind = GAUSSIAN,
        maxPartitionsContributed = 1_000_000,
        maxContributionsPerPartition = 1_000_000,
      )
    private val COUNT_PARAMS =
      AggregationParams(
        metrics = ImmutableList.of(MetricDefinition(COUNT)),
        noiseKind = GAUSSIAN,
        maxPartitionsContributed = 1_000_000,
        maxContributionsPerPartition = 1_000_000,
      )
    private val SUM_PARAMS =
      AggregationParams(
        metrics = ImmutableList.of(MetricDefinition(SUM)),
        noiseKind = GAUSSIAN,
        maxPartitionsContributed = 1_000_000,
        minTotalValue = -Double.MAX_VALUE,
        maxTotalValue = Double.MAX_VALUE,
      )
    private val COUNT_AND_SUM_PARAMS =
      AggregationParams(
        metrics = ImmutableList.of(MetricDefinition(COUNT), MetricDefinition(SUM)),
        noiseKind = GAUSSIAN,
        // Choose large values to avoid contribution clamping but keep the values low enough to
        // avoid sensitivity overflow.
        maxPartitionsContributed = 100,
        maxContributionsPerPartition = 100,
        minTotalValue = -100.0,
        maxTotalValue = 100.0,
      )
    private val MEAN_PARAMS =
      AggregationParams(
        metrics = ImmutableList.of(MetricDefinition(MEAN)),
        noiseKind = GAUSSIAN,
        maxPartitionsContributed = 10,
        maxContributionsPerPartition = 20,
        minValue = -10.0,
        maxValue = 10.0,
      )
    private val QUANTILES_PARAMS =
      AggregationParams(
        metrics =
          ImmutableList.of(
            MetricDefinition(QUANTILES(ranks = ImmutableList.of(0.0001, 0.0, 0.5, 0.999, 1.0)))
          ),
        noiseKind = GAUSSIAN,
        maxPartitionsContributed = 10,
        maxContributionsPerPartition = 20,
        minValue = -10.0,
        maxValue = 10.0,
      )
    private val LOCAL_EF = LocalEncoderFactory()
  }
}

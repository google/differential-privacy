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
import com.google.privacy.differentialprivacy.pipelinedp4j.core.ExecutionMode.FULL_TEST_MODE
import com.google.privacy.differentialprivacy.pipelinedp4j.core.ExecutionMode.TEST_MODE_WITH_CONTRIBUTION_BOUNDING
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.COUNT
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.MEAN
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.PRIVACY_ID_COUNT
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.SUM
import com.google.privacy.differentialprivacy.pipelinedp4j.core.NoiseKind.GAUSSIAN
import com.google.privacy.differentialprivacy.pipelinedp4j.core.NoiseKind.LAPLACE
import com.google.privacy.differentialprivacy.pipelinedp4j.core.budget.AbsoluteBudgetPerOpSpec
import com.google.privacy.differentialprivacy.pipelinedp4j.core.budget.TotalBudget
import com.google.privacy.differentialprivacy.pipelinedp4j.local.LocalCollection
import com.google.privacy.differentialprivacy.pipelinedp4j.local.LocalEncoderFactory
import com.google.privacy.differentialprivacy.pipelinedp4j.local.LocalTable
import com.google.privacy.differentialprivacy.pipelinedp4j.local.createLocalEngine
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.DpAggregates
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.dpAggregates
import com.google.testing.junit.testparameterinjector.TestParameterInjector
import org.junit.Test
import org.junit.runner.RunWith

/**
 * End-to-end tests for Kotlin CUJs starting with DP engine. Each of these tests creates a local DP
 * Engine that can apply noise to the result and evaluates the expected output.
 *
 * TODO: Add tests for all supported backends.
 */
@RunWith(TestParameterInjector::class)
class EndToEndTest {

  /** Tests for partition selection below */
  @Test
  fun selectPartitions_withManyContributions_keepsPartition() {
    // Create dataset with 1 partition and 100 privacy ids which contribute to this partition.
    val inputData =
      LocalCollection(List(100) { TestDataRow("PrivacyId$it", "Partition1", 2.0) }.asSequence())
    val dpEngine = DpEngine.createLocalEngine(LARGE_BUDGET_SPEC)
    // Ensure all contributions to the partition are kept.
    val params = SelectPartitionsParams(maxPartitionsContributed = 2)

    val dpAggregates =
      dpEngine.selectPartitions(inputData, params, testDataExtractors) as LocalCollection<String>
    dpEngine.done()

    val partitionResult = dpAggregates.data.toList()
    assertThat(partitionResult).containsExactly("Partition1")
  }

  @Test
  fun aggregate_preAggregationPartitionSelection_keepsPartitionAndCalculatesCorrectResult() {
    // Create dataset with 1 partition and 100 privacy ids which contribute to this partition.
    val inputData =
      LocalCollection(List(100) { TestDataRow("PrivacyId$it", "US", 2.0) }.asSequence())
    // Create a local DP engine with minimal noise so results are close to deterministic.
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
      dpEngine.aggregate(inputData, params, testDataExtractors) as LocalTable<String, DpAggregates>
    dpEngine.done()

    val partitionResult = dpAggregates.data.toMap()["US"]!!
    assertThat(partitionResult.count).isWithin(1e-1).of(100.0)
    assertThat(partitionResult.sum).isWithin(1e-1).of(200.0)
    assertThat(partitionResult.mean).isWithin(1e-10).of(partitionResult.sum / partitionResult.count)
  }

  @Test
  fun aggregate_postAggregationPartitionSelection_keepsPartitionAndCalculatesCorrectResult() {
    // Create dataset with 1 partition and 100 privacy ids which contribute to this partition.
    val inputData =
      LocalCollection(List(100) { TestDataRow("PrivacyId$it", "US", 2.0) }.asSequence())
    // Create a local DP engine with minimal noise so results are close to deterministic.
    val dpEngine = DpEngine.createLocalEngine(LARGE_BUDGET_SPEC)
    // Use low bounds to avoid sensitivity overflow when adding noise.
    val params =
      AggregationParams(
        metrics = ImmutableList.of(MetricDefinition(COUNT), MetricDefinition(PRIVACY_ID_COUNT)),
        noiseKind = LAPLACE,
        maxPartitionsContributed = 1,
        maxContributionsPerPartition = 1,
        minValue = -2.0,
        maxValue = 2.0,
      )

    val dpAggregates =
      dpEngine.aggregate(inputData, params, testDataExtractors) as LocalTable<String, DpAggregates>
    dpEngine.done()

    val partitionResult = dpAggregates.data.toMap()["US"]!!
    assertThat(partitionResult.privacyIdCount).isWithin(1e-1).of(100.0)
    assertThat(partitionResult.count).isWithin(1e-1).of(100.0)
  }

  /** Tests for no privacy below */
  @Test
  fun aggregate_fullTestMode_returnsNonDpResult() {
    // Create dataset with 2 partition and 100 privacy ids which contribute to each partition twice.
    val inputData =
      LocalCollection(
        (1..100)
          .flatMap {
            listOf(
              TestDataRow("PrivacyId$it", "US", value = 1.0),
              TestDataRow("PrivacyId$it", "US", value = 2.0),
              TestDataRow("PrivacyId$it", "Canada", value = 1.0),
              TestDataRow("PrivacyId$it", "Canada", value = 2.0),
            )
          }
          .asSequence()
      )
    val lowBudgetWithLotsOfNoise = TotalBudget(epsilon = 1e-10, delta = 1e-10)
    val dpEngine = DpEngine.createLocalEngine(DpEngineBudgetSpec(budget = lowBudgetWithLotsOfNoise))
    val params =
      AggregationParams(
        metrics = ImmutableList.of(MetricDefinition(COUNT)),
        noiseKind = LAPLACE,
        // Contribution bounding would be applied if it was not disabled.
        maxPartitionsContributed = 1,
        maxContributionsPerPartition = 1,
        executionMode = FULL_TEST_MODE,
      )

    val dpAggregates =
      dpEngine.aggregate(inputData, params, testDataExtractors) as LocalTable<String, DpAggregates>
    dpEngine.done()

    assertThat(dpAggregates.data.toList())
      .containsExactly(
        "US" to dpAggregates { count = 200.0 },
        "Canada" to dpAggregates { count = 200.0 },
      )
  }

  @Test
  fun aggregate_testModeWithContributionBounding_returnsBoundedNonDpResult() {
    // Create dataset with 2 partition and 100 privacy ids which contribute to each partition twice.
    val inputData =
      LocalCollection(
        (1..100)
          .flatMap {
            listOf(
              TestDataRow("PrivacyId$it", "US", value = 1.0),
              TestDataRow("PrivacyId$it", "US", value = 2.0),
              TestDataRow("PrivacyId$it", "Canada", value = 1.0),
              TestDataRow("PrivacyId$it", "Canada", value = 2.0),
            )
          }
          .asSequence()
      )

    val lowBudgetWithLotsOfNoise = TotalBudget(epsilon = 1e-10, delta = 1e-10)

    val dpEngine = DpEngine.createLocalEngine(DpEngineBudgetSpec(budget = lowBudgetWithLotsOfNoise))
    val params =
      AggregationParams(
        metrics = ImmutableList.of(MetricDefinition(COUNT)),
        noiseKind = LAPLACE,
        maxPartitionsContributed = 2, // Contributions to each of the two partitions are kept.
        maxContributionsPerPartition = 1, // Double contributions per partition are removed.
        executionMode = TEST_MODE_WITH_CONTRIBUTION_BOUNDING,
      )

    val dpAggregates =
      dpEngine.aggregate(inputData, params, testDataExtractors) as LocalTable<String, DpAggregates>
    dpEngine.done()

    // No noise is applied to the result but contributions are bounded in half per partition.
    assertThat(dpAggregates.data.toList())
      .containsExactly(
        "US" to dpAggregates { count = 100.0 },
        "Canada" to dpAggregates { count = 100.0 },
      )
  }

  /** Tests for correct metric calculations below */
  @Test
  fun aggregate_withPublicPartitions_calculatesCorrectResult() {
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
  fun aggregate_withPublicPartitions_calculatesDifferentResultsInDifferentRuns() {
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

  companion object {
    // A DpEngineBudgetSpec with budget large enough to make sure that tests don't run out of it.
    private val LARGE_BUDGET_SPEC =
      DpEngineBudgetSpec(budget = TotalBudget(epsilon = 2000.0, delta = 0.999999))
    private val LOCAL_EF = LocalEncoderFactory()
  }
}

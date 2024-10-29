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
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.PRIVACY_ID_COUNT
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.SUM
import com.google.privacy.differentialprivacy.pipelinedp4j.core.NoiseKind.GAUSSIAN
import com.google.privacy.differentialprivacy.pipelinedp4j.core.budget.AllocatedBudget
import com.google.privacy.differentialprivacy.pipelinedp4j.dplibrary.NoiseFactory
import com.google.privacy.differentialprivacy.pipelinedp4j.dplibrary.PreAggregationPartitionSelectionFactory
import com.google.privacy.differentialprivacy.pipelinedp4j.dplibrary.ZeroNoiseFactory
import com.google.privacy.differentialprivacy.pipelinedp4j.local.LocalCollection
import com.google.privacy.differentialprivacy.pipelinedp4j.local.LocalEncoderFactory
import com.google.privacy.differentialprivacy.pipelinedp4j.local.LocalTable
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.DpAggregates
import com.google.testing.junit.testparameterinjector.TestParameterInjector
import kotlin.test.assertFailsWith
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.kotlin.any
import org.mockito.kotlin.doReturn
import org.mockito.kotlin.mock
import org.mockito.kotlin.verify

@RunWith(TestParameterInjector::class)
class PrivatePartitionsComputationalGraphTest {
  @Test
  fun aggregate_appliesPreAggregationPartitionSelection_emptyResult() {
    val inputData =
      LocalCollection(
        sequenceOf(
          TestDataRow("Alice", "partition1", 1.0),
          TestDataRow("Alice", "partition2", 1.0),
          TestDataRow("Bob", "partition1", 1.0),
        )
      )
    val preAggregationPartitionSelector =
      mock<PreAggregationPartitionSelector>() {
        on { this.shouldKeep(any()) } doReturn false // Drop all partitions
      }
    val computationalGraph =
      PrivatePartitionsComputationalGraph(
        PartitionSampler(
          maxPartitionsContributed = 100,
          LOCAL_EF.strings(),
          LOCAL_EF.strings(),
          LOCAL_EF,
        ),
        preAggregationPartitionSelector,
        COUNT_SUM_AND_ID_COUNT_COMBINER_ZERO_NOISE,
        testDataExtractors,
        LOCAL_EF,
      )

    val dpAggregates = computationalGraph.aggregate(inputData) as LocalTable<String, DpAggregates>

    assertThat(dpAggregates.data.toList().isEmpty()).isTrue()
  }

  @Test
  fun aggregate_requestedMetricsComputed() {
    val inputData =
      LocalCollection(
        sequenceOf(
          TestDataRow("Alice", "partition1", 1.0),
          TestDataRow("Alice", "partition2", 2.0),
          TestDataRow("Bob", "partition1", 4.0),
        )
      )
    val partitionSelectorMock =
      mock<PreAggregationPartitionSelector>() {
        on { this.shouldKeep(any()) } doReturn true // Keep all partitions
      }
    val computationalGraph =
      PrivatePartitionsComputationalGraph(
        PartitionSampler(
          maxPartitionsContributed = 10,
          LOCAL_EF.strings(),
          LOCAL_EF.strings(),
          LOCAL_EF,
        ),
        partitionSelectorMock,
        COUNT_SUM_AND_ID_COUNT_COMBINER_ZERO_NOISE,
        testDataExtractors,
        LOCAL_EF,
      )

    val dpAggregates =
      (computationalGraph.aggregate(inputData) as LocalTable<String, DpAggregates>).data.toMap()

    // Assert
    verify(partitionSelectorMock).shouldKeep(1) // "partition1"
    verify(partitionSelectorMock).shouldKeep(2) // "partition2"

    assertThat(dpAggregates.keys).containsExactly("partition1", "partition2")
    assertThat(dpAggregates.get("partition1")!!.count).isEqualTo(2.0)
    assertThat(dpAggregates.get("partition1")!!.sum).isEqualTo(5.0)
    assertThat(dpAggregates.get("partition2")!!.sum).isEqualTo(2.0)
    assertThat(dpAggregates.get("partition2")!!.privacyIdCount).isEqualTo(1.0)
  }

  @Test
  fun aggregate_appliesPostAggregationPartitionSelection_emptyResult() {
    // Arrange.
    val inputData =
      LocalCollection(
        sequenceOf(TestDataRow("Alice", "partition1", 1.0), TestDataRow("Bob", "partition2", 1.0))
      )

    // The probability of publishing a partition is delta. Set small delta to make it the output
    // almost always empty.
    val thresholdBudget = AllocatedBudget().apply { initialize(0.0, 1e-15) }
    val preAggregationPartitionSelector =
      PostAggregationPartitionSelectionCombiner(
        PRIVACY_ID_COUNT_PARAMS,
        METRICS_ALLOCATED_BUDGET,
        thresholdBudget,
        NoiseFactory(),
      )

    val compoundCombinerWithThresholding = CompoundCombiner(listOf(preAggregationPartitionSelector))

    val computationalGraph =
      PrivatePartitionsComputationalGraph(
        PartitionSampler(
          maxPartitionsContributed = 100,
          LOCAL_EF.strings(),
          LOCAL_EF.strings(),
          LOCAL_EF,
        ),
        preAggregationPartitionSelector = null,
        compoundCombinerWithThresholding,
        testDataExtractors,
        LOCAL_EF,
      )

    // Act.
    val dpAggregates = computationalGraph.aggregate(inputData) as LocalTable<String, DpAggregates>

    // Assert.
    assertThat(dpAggregates.data.toList().isEmpty()).isTrue()
  }

  @Test
  fun aggregate_appliesPostAggregationPartitionSelection_partitionsKept() {
    // Arrange.
    // Creates a dataset with 100 privacy units, each contributes 1 record to the same partition.
    val inputData =
      LocalCollection((1..100).map { TestDataRow("PrivacyKey$it", "partition", 1.0) }.asSequence())

    // Set large budget, though the partition will be kept with the probability close to 1.
    val metricsBudget = AllocatedBudget().apply { initialize(5.0, 1e-2) }
    val thresholdBudget = AllocatedBudget().apply { initialize(0.0, 1e-2) }
    val preAggregationPartitionSelector =
      PostAggregationPartitionSelectionCombiner(
        PRIVACY_ID_COUNT_PARAMS,
        metricsBudget,
        thresholdBudget,
        NoiseFactory(),
      )

    val compoundCombinerWithThresholding = CompoundCombiner(listOf(preAggregationPartitionSelector))

    val computationalGraph =
      PrivatePartitionsComputationalGraph(
        PartitionSampler(
          maxPartitionsContributed = 1,
          LOCAL_EF.strings(),
          LOCAL_EF.strings(),
          LOCAL_EF,
        ),
        preAggregationPartitionSelector = null,
        compoundCombinerWithThresholding,
        testDataExtractors,
        LOCAL_EF,
      )

    // Act.
    val dpAggregates =
      (computationalGraph.aggregate(inputData) as LocalTable<String, DpAggregates>).data.toMap()

    // Assert.
    assertThat(dpAggregates.get("partition")!!.privacyIdCount).isWithin(10.0).of(100.0)
  }

  @Test
  fun aggregate_withPartitionSampler_appliesPartitionSampling() {
    val inputData =
      LocalCollection(
        sequenceOf(
          TestDataRow("Alice", "red", 10.0),
          TestDataRow("Alice", "green", 10.0),
          TestDataRow("Alice", "blue", 10.0),
        )
      )
    val partitionSelectorMock =
      mock<PreAggregationPartitionSelector>() {
        on { this.shouldKeep(any()) } doReturn true // Keep all partitions
      }
    val computationalGraph =
      PrivatePartitionsComputationalGraph(
        PartitionSampler(
          maxPartitionsContributed = 2,
          LOCAL_EF.strings(),
          LOCAL_EF.strings(),
          LOCAL_EF,
        ),
        partitionSelectorMock,
        COUNT_SUM_AND_ID_COUNT_COMBINER_ZERO_NOISE,
        testDataExtractors,
        LOCAL_EF,
      )

    val dpAggregates =
      (computationalGraph.aggregate(inputData) as LocalTable<String, DpAggregates>).data.toMap()

    // Assert.
    // The user contributed to 3 partitions but maxPartitionsContributed is set to 2. Hence,
    // contributions to 2 partitions should appear in the result.
    assertThat(dpAggregates.values.map { it.count }).containsExactly(1.0, 1.0)
    assertThat(dpAggregates.values.map { it.sum }).containsExactly(10.0, 10.0)
  }

  @Test
  fun aggregate_addsNoise() {
    val inputData =
      LocalCollection((0..10).map { TestDataRow("PrivacyKey$it", "partition", 1.0) }.asSequence())
    val preAggregationPartitionSelector =
      DpLibPreAggregationPartitionSelector(
        maxPartitionsContributed = 5,
        preThreshold = 1,
        PARTITION_SELECTION_ALLOCATED_BUDGET,
        PreAggregationPartitionSelectionFactory(),
      )
    val computationalGraph =
      PrivatePartitionsComputationalGraph(
        PartitionSampler(
          maxPartitionsContributed = 5,
          LOCAL_EF.strings(),
          LOCAL_EF.strings(),
          LOCAL_EF,
        ),
        preAggregationPartitionSelector,
        PRIVACY_ID_COUNT_COMBINER,
        testDataExtractors,
        LOCAL_EF,
      )

    val dpAggregates =
      (computationalGraph.aggregate(inputData) as LocalTable<String, DpAggregates>).data.toMap()

    // Assert.
    assertThat(dpAggregates.keys).containsExactly("partition")
    assertThat(dpAggregates.get("partition")!!.count).isNotEqualTo(10.0)
  }

  @Test
  fun constructor_failNoPreNorPostAggregationThresholding() {
    assertFailsWith<IllegalArgumentException>("Computational graph must have either") {
      PrivatePartitionsComputationalGraph(
        PartitionSampler(
          maxPartitionsContributed = 5,
          LOCAL_EF.strings(),
          LOCAL_EF.strings(),
          LOCAL_EF,
        ),
        preAggregationPartitionSelector = null,
        PRIVACY_ID_COUNT_COMBINER,
        testDataExtractors,
        LOCAL_EF,
      )
    }
  }

  private companion object {
    val PRIVACY_ID_COUNT_PARAMS =
      AggregationParams(
        metrics = ImmutableList.of(MetricDefinition(PRIVACY_ID_COUNT)),
        noiseKind = GAUSSIAN,
        maxPartitionsContributed = 10,
        maxContributionsPerPartition = 5,
      )
    val COUNT_SUM_AND_ID_COUNT_PARAMS =
      AggregationParams(
        metrics =
          ImmutableList.of(
            MetricDefinition(COUNT),
            MetricDefinition(SUM),
            MetricDefinition(PRIVACY_ID_COUNT),
          ),
        noiseKind = GAUSSIAN,
        maxPartitionsContributed = 100,
        maxContributionsPerPartition = 100,
        minTotalValue = -100.0,
        maxTotalValue = 100.0,
      )
    val METRICS_ALLOCATED_BUDGET = AllocatedBudget().apply { initialize(1.1, 1e-3) }
    // High epsilon/delta for partition selection. Partitions with ~10 privacy unit have ~1
    // probability to be kept.
    val PARTITION_SELECTION_ALLOCATED_BUDGET = AllocatedBudget().apply { initialize(10.0, 1e-1) }
    val THRESHOLDING_BUDGET = AllocatedBudget().apply { initialize(0.0, 1e-1) }

    val LOCAL_EF = LocalEncoderFactory()
    val COUNT_SUM_AND_ID_COUNT_COMBINER_ZERO_NOISE =
      CompoundCombiner(
        listOf(
          CountCombiner(
            COUNT_SUM_AND_ID_COUNT_PARAMS,
            METRICS_ALLOCATED_BUDGET,
            ZeroNoiseFactory(),
          ),
          SumCombiner(COUNT_SUM_AND_ID_COUNT_PARAMS, METRICS_ALLOCATED_BUDGET, ZeroNoiseFactory()),
          PrivacyIdCountCombiner(
            COUNT_SUM_AND_ID_COUNT_PARAMS,
            METRICS_ALLOCATED_BUDGET,
            ZeroNoiseFactory(),
          ),
        )
      )

    val PRIVACY_ID_COUNT_COMBINER =
      CompoundCombiner(
        listOf(
          PrivacyIdCountCombiner(PRIVACY_ID_COUNT_PARAMS, METRICS_ALLOCATED_BUDGET, NoiseFactory())
        )
      )

    val POST_AGGREGATION_THRESHOLDING_COMBINER =
      CompoundCombiner(
        listOf(
          PostAggregationPartitionSelectionCombiner(
            PRIVACY_ID_COUNT_PARAMS,
            METRICS_ALLOCATED_BUDGET,
            THRESHOLDING_BUDGET,
            NoiseFactory(),
          )
        )
      )
  }
}

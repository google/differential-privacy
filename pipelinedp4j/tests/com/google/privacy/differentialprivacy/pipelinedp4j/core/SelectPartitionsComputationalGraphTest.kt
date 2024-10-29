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

import com.google.common.truth.Truth.assertThat
import com.google.privacy.differentialprivacy.pipelinedp4j.core.budget.AllocatedBudget
import com.google.privacy.differentialprivacy.pipelinedp4j.dplibrary.PreAggregationPartitionSelectionFactory
import com.google.privacy.differentialprivacy.pipelinedp4j.local.LocalCollection
import com.google.privacy.differentialprivacy.pipelinedp4j.local.LocalEncoderFactory
import com.google.testing.junit.testparameterinjector.TestParameterInjector
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.kotlin.any
import org.mockito.kotlin.doReturn
import org.mockito.kotlin.mock
import org.mockito.kotlin.times
import org.mockito.kotlin.verify

@RunWith(TestParameterInjector::class)
class SelectPartitionsComputationalGraphTest {
  @Test
  fun selectPartitions_selectorsDropsEverything_emptyResult() {
    val inputData =
      LocalCollection(
        sequenceOf(
          TestDataRow("Alice", "partition1"),
          TestDataRow("Alice", "partition2"),
          TestDataRow("Bob", "partition1"),
        )
      )
    val partitionSelector =
      mock<PreAggregationPartitionSelector>() {
        on { this.shouldKeep(any()) } doReturn false // Drop all partitions
      }
    val computationalGraph =
      SelectPartitionsComputationalGraph(
        PartitionSampler(10, LOCAL_EF.strings(), LOCAL_EF.strings(), LOCAL_EF),
        partitionSelector,
        testDataExtractors,
        LOCAL_EF,
      )

    val dpAggregates =
      (computationalGraph.selectPartitions(inputData) as LocalCollection<String>).data.toList()

    assertThat(dpAggregates).isEmpty()
  }

  @Test
  fun aggregate_withPartitionSampler_appliesPartitionSampling() {
    val inputData =
      LocalCollection(
        sequenceOf(
          TestDataRow("Alice", "red"),
          TestDataRow("Alice", "green"),
          TestDataRow("Alice", "blue"),
        )
      )
    val partitionSelectorMock =
      mock<PreAggregationPartitionSelector>() {
        on { this.shouldKeep(any()) } doReturn true // Keep all partitions
      }
    val computationalGraph =
      SelectPartitionsComputationalGraph(
        PartitionSampler(
          maxPartitionsContributed = 2,
          LOCAL_EF.strings(),
          LOCAL_EF.strings(),
          LOCAL_EF,
        ),
        partitionSelectorMock,
        testDataExtractors,
        LOCAL_EF,
      )

    val dpAggregates =
      (computationalGraph.selectPartitions(inputData) as LocalCollection<String>).data.toList()

    assertThat(dpAggregates).hasSize(2)
    verify(partitionSelectorMock, times(2)).shouldKeep(1)
  }

  @Test
  fun selectPartition_keepsFrequentPartition() {
    // Generate a dataset with 2 partitions. One partition has 100 contributions and another 1. Each
    // user contributes one record.
    val inputData =
      LocalCollection(
        (0..100).map { TestDataRow("PrivacyKey$it", "partition${it/100}") }.asSequence()
      )
    val partitionSelector =
      DpLibPreAggregationPartitionSelector(
        maxPartitionsContributed = 1,
        preThreshold = 1,
        allocatedBudget,
        PreAggregationPartitionSelectionFactory(),
      )
    val computationalGraph =
      SelectPartitionsComputationalGraph(
        PartitionSampler(10, LOCAL_EF.strings(), LOCAL_EF.strings(), LOCAL_EF),
        partitionSelector,
        testDataExtractors,
        LOCAL_EF,
      )

    val dpAggregates =
      (computationalGraph.selectPartitions(inputData) as LocalCollection<String>).data.toList()

    assertThat(dpAggregates).containsExactly("partition0")
  }

  private companion object {
    val allocatedBudget = AllocatedBudget().apply { initialize(epsilon = 1.1, delta = 1e-3) }
    val LOCAL_EF = LocalEncoderFactory()
  }
}

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
import com.google.common.truth.extensions.proto.ProtoTruth.assertThat
import com.google.privacy.differentialprivacy.Noise
import com.google.privacy.differentialprivacy.pipelinedp4j.core.ContributionBoundingLevel.DATASET_LEVEL
import com.google.privacy.differentialprivacy.pipelinedp4j.core.ExecutionMode.FULL_TEST_MODE
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.COUNT
import com.google.privacy.differentialprivacy.pipelinedp4j.core.NoiseKind.GAUSSIAN
import com.google.privacy.differentialprivacy.pipelinedp4j.core.budget.AllocatedBudget
import com.google.privacy.differentialprivacy.pipelinedp4j.dplibrary.NoiseFactory
import com.google.privacy.differentialprivacy.pipelinedp4j.dplibrary.ZeroNoiseFactory
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.PrivacyIdContributionsKt.multiValueContribution
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.countAccumulator
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.privacyIdContributions
import com.google.testing.junit.testparameterinjector.TestParameterInjector
import com.google.testing.junit.testparameterinjector.TestParameters
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.kotlin.mock
import org.mockito.kotlin.verify

@RunWith(TestParameterInjector::class)
class CountCombinerTest {
  private val AGG_PARAMS =
    AggregationParams(
      metrics = ImmutableList.of(MetricDefinition(COUNT)),
      noiseKind = GAUSSIAN,
      maxPartitionsContributed = 3,
      maxContributionsPerPartition = 5,
    )

  private val noiseMock: Noise = mock()
  private val noiseFactoryMock: (NoiseKind) -> Noise = { _ -> noiseMock }
  private val UNUSED_ALLOCATED_BUDGET = AllocatedBudget()

  init {
    UNUSED_ALLOCATED_BUDGET.initialize(1.1, 1e-3)
  }

  @Test
  fun emptyAccumulator_countIsZero() {
    val combiner = CountCombiner(AGG_PARAMS, UNUSED_ALLOCATED_BUDGET, NoiseFactory())

    val accumulator = combiner.emptyAccumulator()

    assertThat(accumulator).isEqualTo(countAccumulator { count = 0 })
  }

  @Test
  fun createAccumulator_singleValueContributions_countsItems() {
    val combiner = CountCombiner(AGG_PARAMS, UNUSED_ALLOCATED_BUDGET, NoiseFactory())

    val accumulator =
      combiner.createAccumulator(
        privacyIdContributions { singleValueContributions += listOf(1.0, 1.0, 1.0) }
      )

    assertThat(accumulator).isEqualTo(countAccumulator { count = 3 })
  }

  @Test
  fun createAccumulator_multiValueContributions_countsItems() {
    val combiner = CountCombiner(AGG_PARAMS, UNUSED_ALLOCATED_BUDGET, NoiseFactory())

    val accumulator =
      combiner.createAccumulator(
        privacyIdContributions {
          multiValueContributions +=
            listOf(
              multiValueContribution { values += listOf(1.0, 1.0, 1.0) },
              multiValueContribution { values += listOf(2.0, 2.0, 2.0) },
              multiValueContribution { values += listOf(3.0, 3.0, 3.0) },
            )
        }
      )

    assertThat(accumulator).isEqualTo(countAccumulator { count = 3 })
  }

  @Test
  fun createAccumulator_privacyLevelWithContributionBounding_clampsCount() {
    val combiner =
      CountCombiner(
        AGG_PARAMS.copy(
          maxContributionsPerPartition = 2,
          contributionBoundingLevel = DATASET_LEVEL,
        ),
        UNUSED_ALLOCATED_BUDGET,
        NoiseFactory(),
      )

    val accumulator =
      combiner.createAccumulator(
        privacyIdContributions { singleValueContributions += listOf(1.0, 1.0, 1.0) }
      )

    assertThat(accumulator).isEqualTo(countAccumulator { count = 2 })
  }

  @Test
  fun createAccumulator_privacyLevelWithoutContributionBounding_doesNotClampCount() {
    val combiner =
      CountCombiner(
        AGG_PARAMS.copy(
          maxContributionsPerPartition = 2,
          contributionBoundingLevel = DATASET_LEVEL,
          executionMode = FULL_TEST_MODE,
        ),
        UNUSED_ALLOCATED_BUDGET,
        NoiseFactory(),
      )

    val accumulator =
      combiner.createAccumulator(
        privacyIdContributions { singleValueContributions += listOf(1.0, 1.0, 1.0) }
      )

    assertThat(accumulator).isEqualTo(countAccumulator { count = 3 })
  }

  @Test
  fun mergeAccumulators_sumsCounts() {
    val combiner = CountCombiner(AGG_PARAMS, UNUSED_ALLOCATED_BUDGET, NoiseFactory())

    val accumulator =
      combiner.mergeAccumulators(countAccumulator { count = 1 }, countAccumulator { count = 2 })

    assertThat(accumulator).isEqualTo(countAccumulator { count = 3 })
  }

  @Test
  @TestParameters("{noiseKind: LAPLACE, delta: 0.0}", "{noiseKind: GAUSSIAN, delta: 0.1}")
  fun computeMetrics_addsNoise(noiseKind: NoiseKind, delta: Double) {
    val paramsWithNoise =
      AggregationParams(
        metrics = ImmutableList.of(MetricDefinition(COUNT)),
        noiseKind = noiseKind,
        maxPartitionsContributed = 30,
        maxContributionsPerPartition = 50,
      )
    val allocatedBudget = AllocatedBudget()
    allocatedBudget.initialize(1.1, delta)
    val combiner = CountCombiner(paramsWithNoise, allocatedBudget, NoiseFactory())

    val result = combiner.computeMetrics(countAccumulator { count = 1 })

    assertThat(result).isNotEqualTo(1)
  }

  @Test
  fun computeMetrics_passesCorrectParametersToNoise() {
    val params =
      AggregationParams(
        metrics = ImmutableList.of(MetricDefinition(COUNT)),
        noiseKind = GAUSSIAN,
        maxPartitionsContributed = 3,
        maxContributionsPerPartition = 5,
      )
    val allocatedBudget = AllocatedBudget()
    allocatedBudget.initialize(1.1, 1e-3)
    val combiner = CountCombiner(params, allocatedBudget, noiseFactoryMock)

    val unused = combiner.computeMetrics(countAccumulator { count = 1 })

    verify(noiseMock)
      .addNoise(
        /* x= */ 1.0,
        /* l0Sensitivity= */ 3,
        /* lInfSensitivity= */ 5.0,
        /* epsilon= */ 1.1,
        /*delta= */ 1e-3,
      )
  }

  @Test
  fun computeMetrics_withoutNoise_withMultipleContributionsIncludingEmptyAccumulator_returnsCorrectResult() {
    val allocatedBudget = AllocatedBudget()
    allocatedBudget.initialize(1.1, 1e-3)
    val combiner = CountCombiner(AGG_PARAMS, allocatedBudget, ZeroNoiseFactory())

    val accumulator0 = combiner.emptyAccumulator()
    val accumulator1 =
      combiner.createAccumulator(
        privacyIdContributions { singleValueContributions += listOf(0.0, 0.0) }
      )
    val accumulator2 =
      combiner.createAccumulator(privacyIdContributions { singleValueContributions += listOf(0.0) })
    val accumulator3 = combiner.mergeAccumulators(accumulator0, accumulator1)
    val finalAccumulator = combiner.mergeAccumulators(accumulator2, accumulator3)
    val result = combiner.computeMetrics(finalAccumulator)

    assertThat(result).isEqualTo(3.0)
  }

  @Test
  fun computeMetrics_withoutNoise_onlyEmptyAccumulator_returnsZeroCount() {
    val allocatedBudget = AllocatedBudget()
    allocatedBudget.initialize(1.1, 1e-3)
    val combiner = CountCombiner(AGG_PARAMS, allocatedBudget, ZeroNoiseFactory())

    val result = combiner.computeMetrics(combiner.emptyAccumulator())

    assertThat(result).isEqualTo(0.0)
  }
}

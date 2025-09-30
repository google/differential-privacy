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
import com.google.privacy.differentialprivacy.pipelinedp4j.core.ExecutionMode.FULL_TEST_MODE
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.SUM
import com.google.privacy.differentialprivacy.pipelinedp4j.core.NoiseKind.GAUSSIAN
import com.google.privacy.differentialprivacy.pipelinedp4j.core.budget.AllocatedBudget
import com.google.privacy.differentialprivacy.pipelinedp4j.dplibrary.NoiseFactory
import com.google.privacy.differentialprivacy.pipelinedp4j.dplibrary.ZeroNoiseFactory
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.privacyIdContributions
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.sumAccumulator
import com.google.testing.junit.testparameterinjector.TestParameterInjector
import com.google.testing.junit.testparameterinjector.TestParameters
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.kotlin.mock
import org.mockito.kotlin.verify

@RunWith(TestParameterInjector::class)
class SumCombinerTest {
  private val SUM_AGG_PARAMS =
    AggregationParams(
      nonFeatureMetrics = ImmutableList.of(),
      features =
        ImmutableList.of(
          ScalarFeatureSpec(
            featureId = "value",
            metrics = ImmutableList.of(MetricDefinition(SUM)),
            minTotalValue = -1.0,
            maxTotalValue = 3.0,
          )
        ),
      noiseKind = GAUSSIAN,
      maxPartitionsContributed = 5,
    )

  private val noiseMock: Noise = mock()
  private val noiseFactoryMock: (NoiseKind) -> Noise = { _ -> noiseMock }
  private val UNUSED_ALLOCATED_BUDGET = AllocatedBudget()

  init {
    UNUSED_ALLOCATED_BUDGET.initialize(1.1, 1e-3)
  }

  @Test
  fun emptyAccumulator_minIsGreaterThanZero_returnsZeroAndIgnoresContributionBounds() {
    val params =
      SUM_AGG_PARAMS.copy(
        features =
          ImmutableList.of(
            ScalarFeatureSpec(
              featureId = "value",
              metrics = ImmutableList.of(MetricDefinition(SUM)),
              minTotalValue = 1.0,
              maxTotalValue = 2.0,
            )
          )
      )
    val combiner =
      SumCombiner(
        params,
        UNUSED_ALLOCATED_BUDGET,
        NoiseFactory(),
        ExecutionMode.PRODUCTION,
        params.features[0] as ScalarFeatureSpec,
      )

    val accumulator = combiner.emptyAccumulator()

    assertThat(accumulator).isEqualTo(sumAccumulator { sum = 0.0 })
  }

  @Test
  fun createAccumulator_sumsItems() {
    val params =
      SUM_AGG_PARAMS.copy(
        features =
          ImmutableList.of(
            ScalarFeatureSpec(
              featureId = "value",
              metrics = ImmutableList.of(MetricDefinition(SUM)),
              minTotalValue = -300.0,
              maxTotalValue = 300.0,
            )
          )
      )
    val combiner =
      SumCombiner(
        params,
        UNUSED_ALLOCATED_BUDGET,
        NoiseFactory(),
        ExecutionMode.TEST_MODE_WITH_CONTRIBUTION_BOUNDING,
        params.features[0] as ScalarFeatureSpec,
      )

    val accumulator =
      combiner.createAccumulator(
        privacyIdContributions { singleValueContributions += listOf(-10.0, 15.0, 0.0) }
      )

    assertThat(accumulator).isEqualTo(sumAccumulator { sum = 5.0 })
  }

  @Test
  fun createAccumulator_privacyLevelWithContributionBounding_clampsOnlyTotalSum() {
    val params =
      SUM_AGG_PARAMS.copy(
        features =
          ImmutableList.of(
            ScalarFeatureSpec(
              featureId = "value",
              metrics = ImmutableList.of(MetricDefinition(SUM)),
              minValue = -1.0,
              maxValue = 4.0,
              minTotalValue = -2.0,
              maxTotalValue = 300.0,
            )
          )
      )
    val combiner =
      SumCombiner(
        params,
        UNUSED_ALLOCATED_BUDGET,
        NoiseFactory(),
        ExecutionMode.TEST_MODE_WITH_CONTRIBUTION_BOUNDING,
        params.features[0] as ScalarFeatureSpec,
      )

    val accumulator =
      combiner.createAccumulator(
        privacyIdContributions { singleValueContributions += listOf(-1000.0, 1000.0, 500.0) }
      )

    assertThat(accumulator).isEqualTo(sumAccumulator { sum = 300.0 })
  }

  @Test
  fun createAccumulator_fullTestMode_doesNotClampTotalSum() {
    val params =
      SUM_AGG_PARAMS.copy(
        features =
          ImmutableList.of(
            ScalarFeatureSpec(
              featureId = "value",
              metrics = ImmutableList.of(MetricDefinition(SUM)),
              minValue = -1.0,
              maxValue = 4.0,
              minTotalValue = -2.0,
              maxTotalValue = 300.0,
            )
          )
      )
    val combiner =
      SumCombiner(
        params,
        UNUSED_ALLOCATED_BUDGET,
        NoiseFactory(),
        FULL_TEST_MODE,
        params.features[0] as ScalarFeatureSpec,
      )

    val accumulator =
      combiner.createAccumulator(
        privacyIdContributions { singleValueContributions += listOf(-1000.0, 1000.0, 500.0) }
      )

    assertThat(accumulator).isEqualTo(sumAccumulator { sum = 500.0 })
  }

  @Test
  fun mergeAccumulators_sumsPartialSums() {
    val combiner =
      SumCombiner(
        SUM_AGG_PARAMS,
        UNUSED_ALLOCATED_BUDGET,
        NoiseFactory(),
        ExecutionMode.TEST_MODE_WITH_CONTRIBUTION_BOUNDING,
        SUM_AGG_PARAMS.features[0] as ScalarFeatureSpec,
      )

    val accumulator =
      combiner.mergeAccumulators(sumAccumulator { sum = 1000.0 }, sumAccumulator { sum = -2000.0 })

    assertThat(accumulator).isEqualTo(sumAccumulator { sum = -1000.0 })
  }

  @Test
  @TestParameters("{noiseKind: LAPLACE, delta: 0.0}", "{noiseKind: GAUSSIAN, delta: 1e-5}")
  fun computeMetrics_addsNoise(noiseKind: NoiseKind, delta: Double) {
    val allocatedBudget = AllocatedBudget()
    allocatedBudget.initialize(1.1, delta)
    val combiner =
      SumCombiner(
        SUM_AGG_PARAMS.copy(noiseKind = noiseKind),
        allocatedBudget,
        NoiseFactory(),
        ExecutionMode.PRODUCTION,
        SUM_AGG_PARAMS.copy(noiseKind = noiseKind).features[0] as ScalarFeatureSpec,
      )

    val result = combiner.computeMetrics(sumAccumulator { sum = 1.0 })

    assertThat(result).isNotEqualTo(1.0)
  }

  @Test
  fun computeMetrics_passesCorrectParametersToNoise() {
    val allocatedBudget = AllocatedBudget()
    allocatedBudget.initialize(1.1, 1e-3)
    val params =
      SUM_AGG_PARAMS.copy(
        features =
          ImmutableList.of(
            ScalarFeatureSpec(
              featureId = "value",
              metrics = ImmutableList.of(MetricDefinition(SUM)),
              minTotalValue = -4.0,
              maxTotalValue = 3.0,
            )
          ),
        noiseKind = GAUSSIAN,
        maxPartitionsContributed = 10,
      )
    val combiner =
      SumCombiner(
        params,
        allocatedBudget,
        noiseFactoryMock,
        ExecutionMode.TEST_MODE_WITH_CONTRIBUTION_BOUNDING,
        params.features[0] as ScalarFeatureSpec,
      )

    val unused = combiner.computeMetrics(sumAccumulator { sum = 1.0 })

    verify(noiseMock)
      .addNoise(
        /* x= */ 1.0,
        /* l0Sensitivity= */ 10,
        /* lInfSensitivity= */ 4.0,
        /* epsilon= */ 1.1,
        /*delta= */ 1e-3,
      )
  }

  @Test
  fun computeMetrics_withoutNoise_withMultipleContributionsIncludingEmptyAccumulator_returnsCorrectResult() {
    val allocatedBudget = AllocatedBudget()
    allocatedBudget.initialize(1.1, 1e-5)
    val params =
      SUM_AGG_PARAMS.copy(
        features =
          ImmutableList.of(
            ScalarFeatureSpec(
              featureId = "value",
              metrics = ImmutableList.of(MetricDefinition(SUM)),
              minTotalValue = -1.0,
              maxTotalValue = 3.0,
            )
          )
      )
    val combiner =
      SumCombiner(
        params,
        allocatedBudget,
        ZeroNoiseFactory(),
        ExecutionMode.TEST_MODE_WITH_CONTRIBUTION_BOUNDING,
        params.features[0] as ScalarFeatureSpec,
      )

    val accumulator0 = combiner.emptyAccumulator()
    val accumulator1 =
      combiner.createAccumulator(
        privacyIdContributions { singleValueContributions += listOf(-2.0, 3.0) }
      )
    val accumulator2 =
      combiner.createAccumulator(
        privacyIdContributions { singleValueContributions += listOf(4.0, -1.0) }
      )
    val accumulator3 = combiner.mergeAccumulators(accumulator0, accumulator1)
    val finalAccumulator = combiner.mergeAccumulators(accumulator2, accumulator3)
    val result = combiner.computeMetrics(finalAccumulator)

    assertThat(result).isEqualTo(4.0)
  }

  @Test
  fun computeMetrics_withoutNoiseAndEmptyAccumulatorThenMerged_returnsZeroSum() {
    val allocatedBudget = AllocatedBudget()
    allocatedBudget.initialize(1.1, 1e-5)
    val params =
      SUM_AGG_PARAMS.copy(
        features =
          ImmutableList.of(
            ScalarFeatureSpec(
              featureId = "value",
              metrics = ImmutableList.of(MetricDefinition(SUM)),
              minTotalValue = -1.0,
              maxTotalValue = 3.0,
            )
          )
      )
    val combiner =
      SumCombiner(
        params,
        allocatedBudget,
        ZeroNoiseFactory(),
        ExecutionMode.TEST_MODE_WITH_CONTRIBUTION_BOUNDING,
        params.features[0] as ScalarFeatureSpec,
      )

    val result = combiner.computeMetrics(combiner.emptyAccumulator())

    assertThat(result).isEqualTo(0.0)
  }
}

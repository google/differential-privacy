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
import com.google.privacy.differentialprivacy.pipelinedp4j.core.ContributionBoundingLevel.PARTITION_LEVEL
import com.google.privacy.differentialprivacy.pipelinedp4j.core.ExecutionMode.FULL_TEST_MODE
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.VECTOR_SUM
import com.google.privacy.differentialprivacy.pipelinedp4j.core.NoiseKind.GAUSSIAN
import com.google.privacy.differentialprivacy.pipelinedp4j.core.budget.AllocatedBudget
import com.google.privacy.differentialprivacy.pipelinedp4j.dplibrary.NoiseFactory
import com.google.privacy.differentialprivacy.pipelinedp4j.dplibrary.ZeroNoiseFactory
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.PrivacyIdContributionsKt.multiValueContribution
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.privacyIdContributions
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.vectorSumAccumulator
import com.google.testing.junit.testparameterinjector.TestParameterInjector
import com.google.testing.junit.testparameterinjector.TestParameters
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.kotlin.mock
import org.mockito.kotlin.verify

@RunWith(TestParameterInjector::class)
class VectorSumCombinerTest {
  private val VECTOR_SUM_AGG_PARAMS =
    AggregationParams(
      metrics = ImmutableList.of(MetricDefinition(VECTOR_SUM)),
      noiseKind = GAUSSIAN,
      vectorNormKind = NormKind.L2,
      vectorMaxTotalNorm = 3.0,
      vectorSize = 3,
      maxPartitionsContributed = 5,
    )

  private val noiseMock: Noise = mock()
  private val noiseFactoryMock: (NoiseKind) -> Noise = { _ -> noiseMock }
  private val UNUSED_ALLOCATED_BUDGET = AllocatedBudget()

  init {
    UNUSED_ALLOCATED_BUDGET.initialize(1.1, 1e-3)
  }

  @Test
  fun emptyAccumulator_returnsZeroVector() {
    val combiner =
      VectorSumCombiner(
        VECTOR_SUM_AGG_PARAMS.copy(vectorSize = 3),
        UNUSED_ALLOCATED_BUDGET,
        NoiseFactory(),
      )

    val accumulator = combiner.emptyAccumulator()

    assertThat(accumulator)
      .isEqualTo(vectorSumAccumulator { sumsPerDimension += listOf(0.0, 0.0, 0.0) })
  }

  @Test
  fun createAccumulator_sumsVectors() {
    val combiner =
      VectorSumCombiner(
        VECTOR_SUM_AGG_PARAMS.copy(
          vectorNormKind = NormKind.L1,
          vectorMaxTotalNorm = 300.0,
          vectorSize = 3,
        ),
        UNUSED_ALLOCATED_BUDGET,
        NoiseFactory(),
      )

    val accumulator =
      combiner.createAccumulator(
        privacyIdContributions {
          multiValueContributions +=
            listOf(
              multiValueContribution { values += listOf(-10.0, 15.0, 0.0) },
              multiValueContribution { values += listOf(10.0, 20.0, -1.0) },
            )
        }
      )

    // The vector sum is [0.0, 35.0, 1.0], which has L1 norm of 36.
    // The max norm is 300 > 36, so the vector is not clipped.
    assertThat(accumulator)
      .isEqualTo(vectorSumAccumulator { sumsPerDimension += listOf(0.0, 35.0, -1.0) })
  }

  @Test
  fun createAccumulator_perPartitionContributionBoundingEnabled_clampsOnlyTotalVectorSum() {
    val combiner =
      VectorSumCombiner(
        VECTOR_SUM_AGG_PARAMS.copy(
          vectorNormKind = NormKind.L_INF,
          vectorMaxTotalNorm = 30.0,
          vectorSize = 3,
          contributionBoundingLevel = PARTITION_LEVEL,
        ),
        UNUSED_ALLOCATED_BUDGET,
        NoiseFactory(),
      )

    val accumulator =
      combiner.createAccumulator(
        privacyIdContributions {
          multiValueContributions +=
            listOf(
              multiValueContribution { values += listOf(-10.0, 15.0, 0.0) },
              multiValueContribution { values += listOf(10.0, 20.0, -1.0) },
            )
        }
      )

    // The vector sum is [0.0, 35.0, 1.0], which has L_INF norm of 35.
    // The max norm is 30 < 35, so the vector is clipped to
    // 30 / 35 * [0.0, 35.0, 1.0] = [0.0, 30.0, 30.0 / 35.0].
    assertThat(accumulator)
      .isEqualTo(vectorSumAccumulator { sumsPerDimension += listOf(0.0, 30.0, 30.0 / 35.0 * -1.0) })
  }

  @Test
  fun createAccumulator_fullTestMode_doesNotClampTotalSum() {
    val combiner =
      VectorSumCombiner(
        VECTOR_SUM_AGG_PARAMS.copy(
          vectorNormKind = NormKind.L_INF,
          vectorMaxTotalNorm = 30.0,
          vectorSize = 3,
          executionMode = FULL_TEST_MODE,
        ),
        UNUSED_ALLOCATED_BUDGET,
        NoiseFactory(),
      )

    val accumulator =
      combiner.createAccumulator(
        privacyIdContributions {
          multiValueContributions +=
            listOf(
              multiValueContribution { values += listOf(-10.0, 15.0, 0.0) },
              multiValueContribution { values += listOf(10.0, 20.0, -1.0) },
            )
        }
      )

    assertThat(accumulator)
      .isEqualTo(vectorSumAccumulator { sumsPerDimension += listOf(0.0, 35.0, -1.0) })
  }

  @Test
  fun mergeAccumulators_sumsPartialVectorSums() {
    val combiner = VectorSumCombiner(VECTOR_SUM_AGG_PARAMS, UNUSED_ALLOCATED_BUDGET, NoiseFactory())

    val accumulator =
      combiner.mergeAccumulators(
        vectorSumAccumulator { sumsPerDimension += listOf(0.0, 35.0, 1.0) },
        vectorSumAccumulator { sumsPerDimension += listOf(-10.0, 0.0, 1.0) },
      )

    assertThat(accumulator)
      .isEqualTo(vectorSumAccumulator { sumsPerDimension += listOf(-10.0, 35.0, 2.0) })
  }

  @Test
  @TestParameters("{noiseKind: LAPLACE, delta: 0.0}", "{noiseKind: GAUSSIAN, delta: 1e-5}")
  fun computeMetrics_addsNoise(noiseKind: NoiseKind, delta: Double) {
    val allocatedBudget = AllocatedBudget()
    allocatedBudget.initialize(1.1, delta)
    val combiner =
      VectorSumCombiner(
        VECTOR_SUM_AGG_PARAMS.copy(noiseKind = noiseKind),
        allocatedBudget,
        NoiseFactory(),
      )

    val result =
      combiner.computeMetrics(vectorSumAccumulator { sumsPerDimension += listOf(1.0, 2.0, 3.0) })

    assertThat(result).isNotEqualTo(listOf(1.0, 2.0, 3.0))
  }

  @Test
  fun computeMetrics_passesCorrectParametersToNoise() {
    val allocatedBudget = AllocatedBudget()
    allocatedBudget.initialize(1.1, 1e-3)
    val combiner =
      VectorSumCombiner(
        VECTOR_SUM_AGG_PARAMS.copy(
          vectorNormKind = NormKind.L2,
          vectorMaxTotalNorm = 30.0,
          vectorSize = 3,
          maxPartitionsContributed = 10,
        ),
        allocatedBudget,
        noiseFactoryMock,
      )

    val unused =
      combiner.computeMetrics(vectorSumAccumulator { sumsPerDimension += listOf(1.0, 2.0, 3.0) })

    verify(noiseMock)
      .addNoise(
        /* x= */ 1.0,
        /* l0Sensitivity= */ 10,
        /* lInfSensitivity= */ 30.0,
        /* epsilon= */ 1.1 / 3,
        /* delta= */ 1e-3 / 3,
      )
    verify(noiseMock)
      .addNoise(
        /* x= */ 2.0,
        /* l0Sensitivity= */ 10,
        /* lInfSensitivity= */ 30.0,
        /* epsilon= */ 1.1 / 3,
        /* delta= */ 1e-3 / 3,
      )
    verify(noiseMock)
      .addNoise(
        /* x= */ 3.0,
        /* l0Sensitivity= */ 10,
        /* lInfSensitivity= */ 30.0,
        /* epsilon= */ 1.1 / 3,
        /* delta= */ 1e-3 / 3,
      )
  }

  @Test
  fun computeMetrics_withoutNoise_withMultipleContributionsIncludingEmptyAccumulator_returnsCorrectResult() {
    val allocatedBudget = AllocatedBudget()
    allocatedBudget.initialize(1.1, 1e-5)
    val combiner =
      VectorSumCombiner(
        VECTOR_SUM_AGG_PARAMS.copy(
          vectorNormKind = NormKind.L2,
          vectorMaxTotalNorm = 30.0,
          vectorSize = 3,
          maxPartitionsContributed = 10,
        ),
        allocatedBudget,
        ZeroNoiseFactory(),
      )

    val accumulator0 = combiner.emptyAccumulator()
    val accumulator1 =
      combiner.createAccumulator(
        privacyIdContributions {
          multiValueContributions +=
            listOf(
              multiValueContribution { values += listOf(-10.0, 15.0, 1.0) },
              multiValueContribution { values += listOf(10.0, 20.0, -1.0) },
            )
        }
      )
    val accumulator2 =
      combiner.createAccumulator(
        privacyIdContributions {
          multiValueContributions +=
            listOf(multiValueContribution { values += listOf(3.0, 0.0, 4.0) })
        }
      )
    val accumulator3 = combiner.mergeAccumulators(accumulator0, accumulator1)
    val finalAccumulator = combiner.mergeAccumulators(accumulator2, accumulator3)
    val result = combiner.computeMetrics(finalAccumulator)

    // Accumulator 0: [0.0, 0.0, 0.0]
    // Accumulator 1: [0.0, 30.0, 0.0] ([0.0, 35.0, 0.0], L2 norm of 35.0, clipped)
    // Accumulator 2: [3.0, 0.0, 4.0] (L2 norm of 5.0, not clipped)
    assertThat(result).isEqualTo(listOf(3.0, 30.0, 4.0))
  }

  @Test
  fun computeMetrics_withoutNoiseAndEmptyAccumulator_returnsZeroVectorSum() {
    val allocatedBudget = AllocatedBudget()
    allocatedBudget.initialize(1.1, 1e-5)
    val combiner =
      VectorSumCombiner(
        VECTOR_SUM_AGG_PARAMS.copy(vectorSize = 3),
        allocatedBudget,
        ZeroNoiseFactory(),
      )

    val result = combiner.computeMetrics(combiner.emptyAccumulator())

    assertThat(result).isEqualTo(listOf(0.0, 0.0, 0.0))
  }
}

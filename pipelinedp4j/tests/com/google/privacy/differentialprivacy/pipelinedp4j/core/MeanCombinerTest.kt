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
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.COUNT
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.MEAN
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.SUM
import com.google.privacy.differentialprivacy.pipelinedp4j.core.budget.AllocatedBudget
import com.google.privacy.differentialprivacy.pipelinedp4j.dplibrary.NoiseFactory
import com.google.privacy.differentialprivacy.pipelinedp4j.dplibrary.ZeroNoiseFactory
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.meanAccumulator
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.privacyIdContributions
import com.google.testing.junit.testparameterinjector.TestParameter
import com.google.testing.junit.testparameterinjector.TestParameterInjector
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.kotlin.mock
import org.mockito.kotlin.verify

@RunWith(TestParameterInjector::class)
class MeanCombinerTest {
  companion object {
    private val AGG_PARAMS =
      AggregationParams(
        metrics = ImmutableList.of(MetricDefinition(MEAN)),
        noiseKind = NoiseKind.GAUSSIAN,
        maxPartitionsContributed = 3,
        maxContributionsPerPartition = 5,
        minValue = -10.0,
        maxValue = 10.0,
      )

    private val noiseMock: Noise = mock()
    private val noiseFactoryMock: (NoiseKind) -> Noise = { _ -> noiseMock }
    private val UNUSED_ALLOCATED_BUDGET = AllocatedBudget()

    @JvmStatic
    @BeforeClass
    fun beforeClass() {
      UNUSED_ALLOCATED_BUDGET.initialize(1.1, 1e-3)
    }
  }

  @Test
  fun emptyAccumulator_countAndSumAreZero() {
    val combiner =
      MeanCombiner(AGG_PARAMS, UNUSED_ALLOCATED_BUDGET, UNUSED_ALLOCATED_BUDGET, NoiseFactory())

    val accumulator = combiner.emptyAccumulator()

    assertThat(accumulator)
      .isEqualTo(
        meanAccumulator {
          count = 0
          normalizedSum = 0.0
        }
      )
  }

  @Test
  fun createAccumulator_doesNotClampContributionsWithinBounds() {
    val combiner =
      MeanCombiner(
        AGG_PARAMS.copy(minValue = -10.0, maxValue = 10.0),
        UNUSED_ALLOCATED_BUDGET,
        UNUSED_ALLOCATED_BUDGET,
        NoiseFactory(),
      )

    val accumulator =
      combiner.createAccumulator(privacyIdContributions { singleValueContributions += listOf(5.5) })

    assertThat(accumulator)
      .isEqualTo(
        meanAccumulator {
          count = 1
          normalizedSum = 5.5
        }
      )
  }

  @Test
  fun createAccumulator_privacyLevelWithContributionBounding_clampsValues() {
    val combiner =
      MeanCombiner(
        AGG_PARAMS.copy(minValue = -10.0, maxValue = 10.0),
        UNUSED_ALLOCATED_BUDGET,
        UNUSED_ALLOCATED_BUDGET,
        NoiseFactory(),
      )

    val accumulator =
      combiner.createAccumulator(
        privacyIdContributions { singleValueContributions += listOf(-20.0, 30.0) }
      )

    assertThat(accumulator)
      .isEqualTo(
        meanAccumulator {
          count = 2
          normalizedSum = 0.0 // = sum of clamped values = -10 + 10
        }
      )
  }

  @Test
  fun createAccumulator_fullTestMode_doesNotClampValues() {
    val combiner =
      MeanCombiner(
        AGG_PARAMS.copy(minValue = -10.0, maxValue = 10.0, executionMode = FULL_TEST_MODE),
        UNUSED_ALLOCATED_BUDGET,
        UNUSED_ALLOCATED_BUDGET,
        NoiseFactory(),
      )

    val accumulator =
      combiner.createAccumulator(
        privacyIdContributions { singleValueContributions += listOf(-20.0, 30.0) }
      )

    assertThat(accumulator)
      .isEqualTo(
        meanAccumulator {
          count = 2
          normalizedSum = 10.0 // = sum of non-clamped values = -20 + 30
        }
      )
  }

  @Test
  fun createAccumulator_normalizesSum() {
    val combiner =
      MeanCombiner(
        AGG_PARAMS.copy(minValue = 5.0),
        UNUSED_ALLOCATED_BUDGET,
        UNUSED_ALLOCATED_BUDGET,
        NoiseFactory(),
      )

    val accumulator =
      combiner.createAccumulator(privacyIdContributions { singleValueContributions += listOf(6.0) })

    assertThat(accumulator)
      .isEqualTo(
        meanAccumulator {
          count = 1
          normalizedSum = -1.5
        }
      )
  }

  @Test
  fun createAccumulator_normalizationAndClamping() {
    val combiner =
      MeanCombiner(
        AGG_PARAMS.copy(minValue = 5.0, maxValue = 10.0),
        UNUSED_ALLOCATED_BUDGET,
        UNUSED_ALLOCATED_BUDGET,
        NoiseFactory(),
      )

    val accumulator =
      combiner.createAccumulator(
        privacyIdContributions { singleValueContributions += listOf(30.0) }
      )

    assertThat(accumulator)
      .isEqualTo(
        meanAccumulator {
          count = 1
          normalizedSum = 2.5
        }
      )
  }

  @Test
  fun createAccumulator_aggregatesMultipleElements() {
    val combiner =
      MeanCombiner(
        AGG_PARAMS.copy(minValue = 4.0),
        UNUSED_ALLOCATED_BUDGET,
        UNUSED_ALLOCATED_BUDGET,
        NoiseFactory(),
      )

    val accumulator =
      combiner.createAccumulator(
        privacyIdContributions { singleValueContributions += listOf(3.0, 5.5, 6.0) }
      )

    assertThat(accumulator)
      .isEqualTo(
        meanAccumulator {
          count = 3
          normalizedSum = -5.5 // = sum of normalized values = -3 - 1.5 - 1
        }
      )
  }

  @Test
  fun mergeAccumulator_sumsValuesInMergedAccumulators() {
    val combiner =
      MeanCombiner(AGG_PARAMS, UNUSED_ALLOCATED_BUDGET, UNUSED_ALLOCATED_BUDGET, NoiseFactory())

    val accumulator =
      combiner.mergeAccumulators(
        meanAccumulator {
          count = 1
          normalizedSum = -5.0
        },
        meanAccumulator {
          count = 10
          normalizedSum = 8.5
        },
      )

    assertThat(accumulator)
      .isEqualTo(
        meanAccumulator {
          count = 11
          normalizedSum = 3.5
        }
      )
  }

  @Test
  fun computeMetrics_passesCorrectParametersToNoise() {
    val countBudget = AllocatedBudget()
    countBudget.initialize(2.0, 1e-5)
    val sumBudget = AllocatedBudget()
    sumBudget.initialize(1.0, 1e-3)
    val combiner =
      MeanCombiner(
        AGG_PARAMS.copy(
          metrics = ImmutableList.of(MetricDefinition(MEAN)),
          maxPartitionsContributed = 5,
          maxContributionsPerPartition = 7,
          minValue = 4.0,
          maxValue = 10.0,
        ),
        countBudget,
        sumBudget,
        noiseFactoryMock,
      )
    val accumulator = meanAccumulator {
      count = 10
      normalizedSum = 120.0
    }

    val unused = combiner.computeMetrics(accumulator)

    // Verify noise is added to count.
    verify(noiseMock)
      .addNoise(
        /* x= */ 10.0,
        /* l0Sensitivity= */ 5,
        /* lInfSensitivity= */ 7.0,
        /* epsilon= */ 2.0,
        /* delta= */ 1e-5,
      )
    // Verify noise is added to sum.
    verify(noiseMock)
      .addNoise(
        /* x= */ 120.0,
        /* l0Sensitivity= */ 5,
        /* lInfSensitivity= */ 21.0,
        /* epsilon= */ 1.0,
        /* delta= */ 1e-3,
      )
  }

  @Test
  fun computeMetrics_returnsMeanCountSum() {
    // Use high budget for low noise.
    val countBudget = AllocatedBudget()
    countBudget.initialize(10000.0, 0.0)
    val sumBudget = AllocatedBudget()
    sumBudget.initialize(10000.0, 0.0)

    val combiner =
      MeanCombiner(
        AGG_PARAMS.copy(
          metrics =
            ImmutableList.of(
              MetricDefinition(MEAN),
              MetricDefinition(SUM),
              MetricDefinition(COUNT),
            ),
          maxPartitionsContributed = 5,
          maxContributionsPerPartition = 7,
          minValue = 4.0,
          maxValue = 12.0,
          noiseKind = NoiseKind.LAPLACE,
        ),
        countBudget,
        sumBudget,
        NoiseFactory(),
      )

    val accumulator = meanAccumulator {
      count = 10
      normalizedSum = 120.0
    }

    val result = combiner.computeMetrics(accumulator)

    assertThat(result.count).isNotEqualTo(10.0)
    assertThat(result.count).isWithin(0.1).of(10.0)

    val approximatedExpectedSum = /* normalizedSum= */ 120.0 + /* dp_count * midValue= */ 10 * 8
    assertThat(result.sum).isNotEqualTo(approximatedExpectedSum)
    assertThat(result.sum).isWithin(1.0).of(approximatedExpectedSum)
    assertThat(result.mean).isWithin(1e-9).of(result.sum!! / result.count!!)
  }

  enum class ReturnedMetricsTestCase(
    val requestedMetrics: ImmutableList<MetricDefinition>,
    val countExpected: Boolean,
    val sumExpected: Boolean,
  ) {
    NO_SUM_NO_COUNT(
      requestedMetrics = ImmutableList.of(MetricDefinition(MEAN)),
      countExpected = false,
      sumExpected = false,
    ),
    ONLY_SUM(
      requestedMetrics = ImmutableList.of(MetricDefinition(MEAN), MetricDefinition(SUM)),
      countExpected = false,
      sumExpected = true,
    ),
    ONLY_COUNT(
      requestedMetrics = ImmutableList.of(MetricDefinition(MEAN), MetricDefinition(COUNT)),
      countExpected = true,
      sumExpected = false,
    ),
    COUNT_AND_SUM(
      requestedMetrics =
        ImmutableList.of(MetricDefinition(MEAN), MetricDefinition(SUM), MetricDefinition(COUNT)),
      countExpected = true,
      sumExpected = true,
    ),
  }

  @Test
  fun aggregate_computeMetrics_checkWhichMetricReturned(
    @TestParameter testCase: ReturnedMetricsTestCase
  ) {
    val combiner =
      MeanCombiner(
        AGG_PARAMS.copy(metrics = testCase.requestedMetrics),
        UNUSED_ALLOCATED_BUDGET,
        UNUSED_ALLOCATED_BUDGET,
        NoiseFactory(),
      )

    val metrics =
      combiner.computeMetrics(
        meanAccumulator {
          count = 10
          normalizedSum = 120.0
        }
      )
    if (testCase.countExpected) {
      assertThat(metrics.count).isNotNull()
    } else {
      assertThat(metrics.count).isNull()
    }

    if (testCase.sumExpected) {
      assertThat(metrics.sum).isNotNull()
    } else {
      assertThat(metrics.sum).isNull()
    }
  }

  @Test
  fun computeMetrics_withoutNoise_withMultipleContributionsIncludingEmptyAccumulator_returnsCorrectResult() {
    val combiner =
      MeanCombiner(
        AGG_PARAMS.copy(minValue = -10.0, maxValue = 10.0),
        UNUSED_ALLOCATED_BUDGET,
        UNUSED_ALLOCATED_BUDGET,
        ZeroNoiseFactory(),
      )

    val accumulator0 = combiner.emptyAccumulator()
    val accumulator1 =
      combiner.createAccumulator(
        privacyIdContributions { singleValueContributions += listOf(10.0, -10.0) }
      )
    val accumulator2 =
      combiner.createAccumulator(privacyIdContributions { singleValueContributions += listOf(9.0) })
    val accumulator3 = combiner.mergeAccumulators(accumulator0, accumulator1)
    val finalAccumulator = combiner.mergeAccumulators(accumulator2, accumulator3)
    val result = combiner.computeMetrics(finalAccumulator)

    assertThat(result.mean).isEqualTo(3.0)
  }

  @Test
  fun computeMetrics_withoutNoise_onlyEmptyAccumulator_returnsZeroCountAndNaNForSumAndMean() {
    val combiner =
      MeanCombiner(
        AGG_PARAMS.copy(
          ImmutableList.of(MetricDefinition(MEAN), MetricDefinition(SUM), MetricDefinition(COUNT)),
          minValue = 4.0,
          maxValue = 10.0,
        ),
        UNUSED_ALLOCATED_BUDGET,
        UNUSED_ALLOCATED_BUDGET,
        ZeroNoiseFactory(),
      )

    val result = combiner.computeMetrics(combiner.emptyAccumulator())

    assertThat(result.count).isEqualTo(0.0)
    // NaN because mean is not defined for count = 0. With noise enabled we will return a very
    // noised mean with added mid value.
    assertThat(result.mean).isNaN()
    // sum is NaN as well because it is computed as count * mean = 0 * NaN = NaN.
    assertThat(result.sum).isNaN()
  }
}

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
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.VARIANCE
import com.google.privacy.differentialprivacy.pipelinedp4j.core.budget.AllocatedBudget
import com.google.privacy.differentialprivacy.pipelinedp4j.dplibrary.NoiseFactory
import com.google.privacy.differentialprivacy.pipelinedp4j.dplibrary.ZeroNoiseFactory
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.privacyIdContributions
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.varianceAccumulator
import com.google.testing.junit.testparameterinjector.TestParameter
import com.google.testing.junit.testparameterinjector.TestParameterInjector
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.kotlin.mock
import org.mockito.kotlin.verify

@RunWith(TestParameterInjector::class)
class VarianceCombinerTest {
  @Test
  fun emptyAccumulator_countAndSumAndSumSquaresAreZero() {
    val combiner =
      VarianceCombiner(
        AGG_PARAMS,
        UNUSED_ALLOCATED_BUDGET,
        UNUSED_ALLOCATED_BUDGET,
        UNUSED_ALLOCATED_BUDGET,
        NoiseFactory(),
      )

    val accumulator = combiner.emptyAccumulator()

    assertThat(accumulator)
      .isEqualTo(
        varianceAccumulator {
          count = 0
          normalizedSum = 0.0
          normalizedSumSquares = 0.0
        }
      )
  }

  @Test
  fun createAccumulator_doesNotClampContributionsWithinBounds() {
    val combiner =
      VarianceCombiner(
        AGG_PARAMS.copy(minValue = -8.0, maxValue = 12.0),
        UNUSED_ALLOCATED_BUDGET,
        UNUSED_ALLOCATED_BUDGET,
        UNUSED_ALLOCATED_BUDGET,
        NoiseFactory(),
      )

    val accumulator =
      combiner.createAccumulator(privacyIdContributions { singleValueContributions += listOf(5.5) })
    // midValue is the midpoint between minValue = -8.0 and maxValue = 12.0 = 2
    assertThat(accumulator)
      .isEqualTo(
        varianceAccumulator {
          count = 1
          normalizedSum = 3.5 // = 5.5 - 2.0 = contribution - midValue
          normalizedSumSquares = 12.25 // (5.5 - 2.0)^2 = (contribution - midValue)^2
        }
      )
  }

  @Test
  fun createAccumulator_privacyLevelWithContributionBounding_clampssingleValueContributions() {
    val combiner =
      VarianceCombiner(
        AGG_PARAMS.copy(minValue = -10.0, maxValue = 10.0),
        UNUSED_ALLOCATED_BUDGET,
        UNUSED_ALLOCATED_BUDGET,
        UNUSED_ALLOCATED_BUDGET,
        NoiseFactory(),
      )

    val accumulator =
      combiner.createAccumulator(
        privacyIdContributions { singleValueContributions += listOf(-20.0, 30.0) }
      )
    // midValue is the midpoint between minValue = -10.0 and maxValue = 10.0 = 0
    assertThat(accumulator)
      .isEqualTo(
        varianceAccumulator {
          count = 2
          normalizedSum = 0.0 // (-10.0 - 0) + (10.0 - 0) = two clamped contributions minus midValue
          normalizedSumSquares =
            200.0 // (-10.0 - 0)^2 + (10.0 - 0)^2 = two clamped contributions minus midValue squared
        }
      )
  }

  @Test
  fun createAccumulator_fullTestMode_doesNotClampSingleValueContributions() {
    val combiner =
      VarianceCombiner(
        AGG_PARAMS.copy(minValue = -10.0, maxValue = 10.0, executionMode = FULL_TEST_MODE),
        UNUSED_ALLOCATED_BUDGET,
        UNUSED_ALLOCATED_BUDGET,
        UNUSED_ALLOCATED_BUDGET,
        NoiseFactory(),
      )

    val accumulator =
      combiner.createAccumulator(
        privacyIdContributions { singleValueContributions += listOf(-20.0, 30.0) }
      )
    // midValue is the midpoint between minValue = -10.0 and maxValue = 10.0 = 0
    assertThat(accumulator)
      .isEqualTo(
        varianceAccumulator {
          count = 2
          normalizedSum = 10.0 // (-20.0 - 0.0) + (30.0 - 0.0) Not clamped
          normalizedSumSquares = 1300.0 // (-20.0 - 0.0)^2 + (30.0 - 0.0)^2 Not clamped
        }
      )
  }

  @Test
  fun createAccumulator_normalizesSumAndSumOfSquares() {
    val combiner =
      VarianceCombiner(
        AGG_PARAMS.copy(minValue = 5.0, maxValue = 10.0),
        UNUSED_ALLOCATED_BUDGET,
        UNUSED_ALLOCATED_BUDGET,
        UNUSED_ALLOCATED_BUDGET,
        NoiseFactory(),
      )

    val accumulator =
      combiner.createAccumulator(privacyIdContributions { singleValueContributions += listOf(6.0) })

    assertThat(accumulator)
      .isEqualTo(
        varianceAccumulator {
          count = 1
          normalizedSum = -1.5
          normalizedSumSquares = (-1.5) * (-1.5)
        }
      )
  }

  @Test
  fun createAccumulator_normalizationAndClamping() {
    val combiner =
      VarianceCombiner(
        AGG_PARAMS.copy(minValue = 5.0, maxValue = 10.0),
        UNUSED_ALLOCATED_BUDGET,
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
        varianceAccumulator {
          count = 1
          normalizedSum = 2.5
          normalizedSumSquares = 2.5 * 2.5
        }
      )
  }

  @Test
  fun createAccumulator_aggregatesMultipleElements() {
    val combiner =
      VarianceCombiner(
        AGG_PARAMS.copy(minValue = 4.0),
        UNUSED_ALLOCATED_BUDGET,
        UNUSED_ALLOCATED_BUDGET,
        UNUSED_ALLOCATED_BUDGET,
        NoiseFactory(),
      )

    // Create list with one value that is clamped to min value.
    val accumulator =
      combiner.createAccumulator(
        privacyIdContributions { singleValueContributions += listOf(3.0, 5.5, 6.0) }
      )

    assertThat(accumulator)
      .isEqualTo(
        varianceAccumulator {
          count = 3
          normalizedSum = -5.5 // = sum of normalized singleValueContributions = -3 - 1.5 - 1
          normalizedSumSquares = 12.25 // sum of each normalized value squared
        }
      )
  }

  @Test
  fun mergeAccumulator_sumssingleValueContributionsInMergedAccumulators() {
    val combiner =
      VarianceCombiner(
        AGG_PARAMS,
        UNUSED_ALLOCATED_BUDGET,
        UNUSED_ALLOCATED_BUDGET,
        UNUSED_ALLOCATED_BUDGET,
        NoiseFactory(),
      )

    val accumulator =
      combiner.mergeAccumulators(
        varianceAccumulator {
          count = 1
          normalizedSum = -5.0
          normalizedSumSquares = 25.0
        },
        varianceAccumulator {
          count = 10
          normalizedSum = 8.5
          normalizedSumSquares = 72.5
        },
      )

    assertThat(accumulator)
      .isEqualTo(
        varianceAccumulator {
          count = 11
          normalizedSum = 3.5
          normalizedSumSquares = 97.5
        }
      )
  }

  @Test
  fun computeMetrics_passesCorrectParametersToNoise() {
    val countBudget = AllocatedBudget()
    countBudget.initialize(2.0, 1e-5)
    val sumBudget = AllocatedBudget()
    sumBudget.initialize(1.0, 1e-3)
    val sumSquaresBudget = AllocatedBudget()
    sumSquaresBudget.initialize(3.0, 1e-2)
    val noise: Noise = mock()
    val noiseFactory: (NoiseKind) -> Noise = { _ -> noise }

    val combiner =
      VarianceCombiner(
        AGG_PARAMS.copy(
          metrics = ImmutableList.of(MetricDefinition(MEAN), MetricDefinition(VARIANCE)),
          maxPartitionsContributed = 5,
          maxContributionsPerPartition = 7,
          minValue = 4.0,
          maxValue = 10.0,
        ),
        countBudget,
        sumBudget,
        sumSquaresBudget,
        noiseFactory,
      )

    val accumulator = varianceAccumulator {
      count = 10
      normalizedSum = 120.0
      normalizedSumSquares = 1500.0
    }

    val unused = combiner.computeMetrics(accumulator)

    // Verify noise is added to count.
    verify(noise)
      .addNoise(
        /* x= */ 10.0,
        /* l0Sensitivity= */ 5,
        /* lInfSensitivity= */ 7.0,
        /* epsilon= */ 2.0,
        /* delta= */ 1e-5,
      )
    // Verify noise is added to sum.
    verify(noise)
      .addNoise(
        /* x= */ 120.0,
        /* l0Sensitivity= */ 5,
        /* lInfSensitivity= */ 21.0, // (maxValue - midValue) * maxContributionsPerPartition
        /* epsilon= */ 1.0,
        /* delta= */ 1e-3,
      )
    // Verify noise is added to normalized sum of squares
    verify(noise)
      .addNoise(
        /* x= */ 1500.0,
        /* l0Sensitivity= */ 5,
        /* lInfSensitivity= */ 63.0, // (maxValue - midValue)^2 * maxContributionsPerPartition
        /* epsilon= */ 3.0,
        /* delta= */ 1e-2,
      )
  }

  @Test
  fun computeMetrics_returnsVarianceMeanCountSum() {
    // Use high budget for low noise.
    val countBudget = AllocatedBudget()
    countBudget.initialize(10000.0, 0.0)
    val sumBudget = AllocatedBudget()
    sumBudget.initialize(10000.0, 0.0)
    val sumSquaresBudget = AllocatedBudget()
    sumSquaresBudget.initialize(10000.0, 0.0)

    val combiner =
      VarianceCombiner(
        AGG_PARAMS.copy(
          metrics =
            ImmutableList.of(
              MetricDefinition(VARIANCE),
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
        sumSquaresBudget,
        NoiseFactory(),
      )

    val accumulator = varianceAccumulator {
      count = 10
      normalizedSum = 120.0
      normalizedSumSquares = 1500.0
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
    val meanExpected: Boolean,
  ) {
    NO_SUM_NO_COUNT_NO_MEAN(
      requestedMetrics = ImmutableList.of(MetricDefinition(VARIANCE)),
      countExpected = false,
      sumExpected = false,
      meanExpected = false,
    ),
    ONLY_SUM(
      requestedMetrics = ImmutableList.of(MetricDefinition(VARIANCE), MetricDefinition(SUM)),
      countExpected = false,
      sumExpected = true,
      meanExpected = false,
    ),
    ONLY_COUNT(
      requestedMetrics = ImmutableList.of(MetricDefinition(VARIANCE), MetricDefinition(COUNT)),
      countExpected = true,
      sumExpected = false,
      meanExpected = false,
    ),
    ONLY_MEAN(
      requestedMetrics = ImmutableList.of(MetricDefinition(VARIANCE), MetricDefinition(MEAN)),
      countExpected = false,
      sumExpected = false,
      meanExpected = true,
    ),
    COUNT_AND_SUM_AND_MEAN(
      requestedMetrics =
        ImmutableList.of(
          MetricDefinition(VARIANCE),
          MetricDefinition(MEAN),
          MetricDefinition(SUM),
          MetricDefinition(COUNT),
        ),
      countExpected = true,
      sumExpected = true,
      meanExpected = true,
    ),
  }

  @Test
  fun aggregate_computeMetrics_checkWhichMetricReturned(
    @TestParameter testCase: ReturnedMetricsTestCase
  ) {
    val combiner =
      VarianceCombiner(
        AGG_PARAMS.copy(metrics = testCase.requestedMetrics),
        UNUSED_ALLOCATED_BUDGET,
        UNUSED_ALLOCATED_BUDGET,
        UNUSED_ALLOCATED_BUDGET,
        NoiseFactory(),
      )

    val metrics =
      combiner.computeMetrics(
        varianceAccumulator {
          count = 10
          normalizedSum = 120.0
          normalizedSumSquares = 1500.0
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

    if (testCase.meanExpected) {
      assertThat(metrics.mean).isNotNull()
    } else {
      assertThat(metrics.mean).isNull()
    }
  }

  @Test
  fun computeMetrics_withoutNoise_withMultipleContributionsIncludingEmptyAccumulator_returnsCorrectResult() {
    val combiner =
      VarianceCombiner(
        AGG_PARAMS.copy(
          ImmutableList.of(MetricDefinition(VARIANCE)),
          minValue = -10.0,
          maxValue = 10.0,
        ),
        UNUSED_ALLOCATED_BUDGET,
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
    val accumulator3 =
      combiner.createAccumulator(privacyIdContributions { singleValueContributions += listOf(0.0) })
    val accumulator01 = combiner.mergeAccumulators(accumulator0, accumulator1)
    val accumulator23 = combiner.mergeAccumulators(accumulator2, accumulator3)
    val finalAccumulator = combiner.mergeAccumulators(accumulator01, accumulator23)
    val result = combiner.computeMetrics(finalAccumulator)

    // (10.0^2 + (-10.0)^2 + 9.0^2 + 0.0^2) / 4 - ((10.0 + -10.0 + 9.0 + 0.0) / 4)^2 = 65.1875
    assertThat(result.variance).isEqualTo(65.1875)
  }

  @Test
  fun computeMetrics_withoutNoise_onlyEmptyAccumulator_returnsZeroCountAndNaNForCountMeanAndVariance() {
    val combiner =
      VarianceCombiner(
        AGG_PARAMS.copy(
          ImmutableList.of(
            MetricDefinition(VARIANCE),
            MetricDefinition(MEAN),
            MetricDefinition(SUM),
            MetricDefinition(COUNT),
          ),
          minValue = 4.0,
          maxValue = 10.0,
        ),
        UNUSED_ALLOCATED_BUDGET,
        UNUSED_ALLOCATED_BUDGET,
        UNUSED_ALLOCATED_BUDGET,
        ZeroNoiseFactory(),
      )

    val result = combiner.computeMetrics(combiner.emptyAccumulator())

    assertThat(result.count).isEqualTo(0.0)
    // NaN because mean is not defined for count = 0. With noise enabled we will return a very
    // noised mean with added mid value.
    assertThat(result.mean).isNaN()
    // NaN because variance is not defined for count = 0. With noise enabled we will return a very
    // noised variance.
    assertThat(result.variance).isNaN()
    // sum is NaN as well because it is computed as count * mean = 0 * NaN = NaN.
    assertThat(result.sum).isNaN()
  }

  companion object {
    private val AGG_PARAMS =
      AggregationParams(
        metrics = ImmutableList.of(MetricDefinition(MEAN), MetricDefinition(VARIANCE)),
        noiseKind = NoiseKind.GAUSSIAN,
        maxPartitionsContributed = 3,
        maxContributionsPerPartition = 5,
        minValue = -10.0,
        maxValue = 10.0,
      )

    private val UNUSED_ALLOCATED_BUDGET = AllocatedBudget()

    @JvmStatic
    @BeforeClass
    fun beforeClass() {
      UNUSED_ALLOCATED_BUDGET.initialize(1.1, 1e-3)
    }
  }
}

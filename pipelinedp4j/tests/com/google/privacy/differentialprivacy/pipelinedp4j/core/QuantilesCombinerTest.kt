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
import com.google.privacy.differentialprivacy.pipelinedp4j.core.NoiseKind.GAUSSIAN
import com.google.privacy.differentialprivacy.pipelinedp4j.core.budget.AllocatedBudget
import com.google.privacy.differentialprivacy.pipelinedp4j.dplibrary.NoiseFactory
import com.google.privacy.differentialprivacy.pipelinedp4j.dplibrary.ZeroNoiseFactory
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.privacyIdContributions
import com.google.testing.junit.testparameterinjector.TestParameterInjector
import com.google.testing.junit.testparameterinjector.TestParameters
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(TestParameterInjector::class)
class QuantilesCombinerTest {
  private fun defaultQuantilesAggParams() =
    AggregationParams(
      metrics = ImmutableList.of(),
      noiseKind = GAUSSIAN,
      maxPartitionsContributed = 1,
      maxContributionsPerPartition = 1,
      minValue = -10000.0,
      maxValue = 10000.0,
    )

  @Test
  fun computeMetrics_noNoise_withEmptyAccumulator_returnsCorrectQuantiles() {
    val allocatedBudget = AllocatedBudget()
    allocatedBudget.initialize(1.1, 1e-5)
    val combiner =
      QuantilesCombiner(
        sortedRanks = listOf(0.25, 0.5, 0.75),
        defaultQuantilesAggParams(),
        allocatedBudget,
        ZeroNoiseFactory(),
      )

    val accumulator0 = combiner.emptyAccumulator()
    val accumulator1 =
      combiner.createAccumulator(
        privacyIdContributions { singleValueContributions += listOf(1.0, 3.0) }
      )
    val accumulator2 =
      combiner.createAccumulator(
        privacyIdContributions { singleValueContributions += listOf(2.0, 4.0) }
      )
    val accumulator3 =
      combiner.createAccumulator(privacyIdContributions { singleValueContributions += listOf(5.0) })
    val accumulator01 = combiner.mergeAccumulators(accumulator0, accumulator1)
    val accumulator012 = combiner.mergeAccumulators(accumulator01, accumulator2)
    val accumulator0123 = combiner.mergeAccumulators(accumulator3, accumulator012)
    val quantiles = combiner.computeMetrics(accumulator0123)

    assertThat(quantiles).hasSize(3)
    assertThat(quantiles.get(0)).isWithin(0.5).of(2.0)
    assertThat(quantiles.get(1)).isWithin(0.5).of(3.0)
    assertThat(quantiles.get(2)).isWithin(0.5).of(4.0)
  }

  @Test
  fun computeMetrics_noNoise_onlyEmptyAccumulator_returnsQuantilesBetweenMinMaxValues() {
    val allocatedBudget = AllocatedBudget()
    allocatedBudget.initialize(1.1, 1e-5)
    val combiner =
      QuantilesCombiner(
        sortedRanks = listOf(0.0, 0.5, 1.0),
        defaultQuantilesAggParams().copy(minValue = -10.0, maxValue = 10.0),
        allocatedBudget,
        ZeroNoiseFactory(),
      )

    val quantiles = combiner.computeMetrics(combiner.emptyAccumulator())

    assertThat(quantiles).hasSize(3)
    assertThat(quantiles.get(0)).isWithin(0.5).of(-10.0)
    assertThat(quantiles.get(1)).isWithin(0.5).of(0.0)
    assertThat(quantiles.get(2)).isWithin(0.5).of(10.0)
  }

  @Test
  @TestParameters("{noiseKind: LAPLACE, delta: 0.0}", "{noiseKind: GAUSSIAN, delta: 0.1}")
  fun computeMetrics_smallNoise_returnsQuantilesCloseToReal(noiseKind: NoiseKind, delta: Double) {
    val allocatedBudget = AllocatedBudget()
    allocatedBudget.initialize(100.0, delta)
    val combiner =
      QuantilesCombiner(
        sortedRanks = listOf(0.0, 0.5, 1.0),
        defaultQuantilesAggParams().copy(minValue = 1.0, maxValue = 1000.0, noiseKind = noiseKind),
        allocatedBudget,
        NoiseFactory(),
      )

    val accumulator =
      combiner.createAccumulator(
        privacyIdContributions {
          singleValueContributions += (1..1000).map { it.toDouble() }.toList()
        }
      )
    val quantiles = combiner.computeMetrics(accumulator)

    assertThat(quantiles).hasSize(3)
    assertThat(quantiles.get(0)).isWithin(10.0).of(1.0)
    assertThat(quantiles.get(1)).isWithin(10.0).of(500.0)
    assertThat(quantiles.get(2)).isWithin(10.0).of(1000.0)
  }
}

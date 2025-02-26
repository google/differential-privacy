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
import com.google.privacy.differentialprivacy.pipelinedp4j.core.NoiseKind.GAUSSIAN
import com.google.privacy.differentialprivacy.pipelinedp4j.core.budget.AllocatedBudget
import com.google.privacy.differentialprivacy.pipelinedp4j.dplibrary.NoiseFactory
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.PrivacyIdContributionsKt.multiValueContribution
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.privacyIdContributions
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.privacyIdCountAccumulator
import com.google.testing.junit.testparameterinjector.TestParameterInjector
import com.google.testing.junit.testparameterinjector.TestParameters
import kotlin.test.assertFailsWith
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.kotlin.mock
import org.mockito.kotlin.verify

@RunWith(TestParameterInjector::class)
class PostAggregationPartitionSelectionCombinerTest {

  @Test
  fun createAccumulator_singleValueContributions_initsAccumulatorWithOne() {
    val combiner =
      PostAggregationPartitionSelectionCombiner(
        AGGREGATION_PARAMS,
        unusedAllocatedBudget,
        unusedAllocatedBudget,
        NoiseFactory(),
      )

    val accumulator =
      combiner.createAccumulator(
        privacyIdContributions { singleValueContributions += listOf(1.0, 1.0, 1.0) }
      )

    assertThat(accumulator).isEqualTo(privacyIdCountAccumulator { count = 1 })
  }

  @Test
  fun createAccumulator_multiValueContributions_initsAccumulatorWithOne() {
    val combiner =
      PostAggregationPartitionSelectionCombiner(
        AGGREGATION_PARAMS,
        unusedAllocatedBudget,
        unusedAllocatedBudget,
        NoiseFactory(),
      )

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

    assertThat(accumulator).isEqualTo(privacyIdCountAccumulator { count = 1 })
  }

  @Test
  fun createAccumulator_noContributions_initsAccumulatorWithZero() {
    val combiner =
      PostAggregationPartitionSelectionCombiner(
        AGGREGATION_PARAMS,
        unusedAllocatedBudget,
        unusedAllocatedBudget,
        NoiseFactory(),
      )

    val e =
      assertFailsWith<IllegalArgumentException> {
        combiner.createAccumulator(privacyIdContributions {})
      }
    assertThat(e).hasMessageThat().contains("There must be contributions")
  }

  @Test
  fun mergeAccumulators_sumsCounts() {
    val combiner =
      PostAggregationPartitionSelectionCombiner(
        AGGREGATION_PARAMS,
        unusedAllocatedBudget,
        unusedAllocatedBudget,
        NoiseFactory(),
      )

    val accumulator =
      combiner.mergeAccumulators(
        privacyIdCountAccumulator { count = 1 },
        privacyIdCountAccumulator { count = 2 },
      )

    assertThat(accumulator).isEqualTo(privacyIdCountAccumulator { count = 3 })
  }

  @TestParameters("{noiseKind: LAPLACE}", "{noiseKind: GAUSSIAN}")
  fun computeMetrics_twoSmallNumberOfPartitions_returnsNull(noiseKind: NoiseKind) {
    val paramsWithNoise =
      AGGREGATION_PARAMS.copy(
        noiseKind = noiseKind,
        maxPartitionsContributed = 30,
        maxContributionsPerPartition = 50,
      )
    val noiseBudget = AllocatedBudget().apply { initialize(0.5, 1e-12) }
    val thresholdingBudget = AllocatedBudget().apply { initialize(0.5, 1e-12) }
    val combiner =
      PostAggregationPartitionSelectionCombiner(
        paramsWithNoise,
        noiseBudget,
        thresholdingBudget,
        NoiseFactory(),
      )

    val result = combiner.computeMetrics(privacyIdCountAccumulator { count = 1 })

    assertThat(result).isNull()
  }

  @Test
  fun computeMetrics_passesCorrectParametersToNoise() {
    val allocatedBudget = AllocatedBudget().apply { initialize(5.5, 1e-3) }
    val thresholdingBudget = AllocatedBudget().apply { initialize(0.0, 1e-8) }
    val combiner =
      PostAggregationPartitionSelectionCombiner(
        AGGREGATION_PARAMS,
        allocatedBudget,
        thresholdingBudget,
        noiseFactoryMock,
      )

    val unused = combiner.computeMetrics(privacyIdCountAccumulator { count = 105 })

    verify(noiseMock)
      .addNoise(
        /* x= */ 105.0,
        /* l0Sensitivity= */ 3,
        /* lInfSensitivity= */ 1.0,
        /* epsilon= */ 5.5,
        /* delta= */ 1e-3,
      )

    verify(noiseMock)
      .computeQuantile(
        /* rank= */ 1e-8 / 3,
        /* x= */ 0.0,
        /* l0Sensitivity= */ 3,
        /* lInfSensitivity= */ 1.0,
        /* epsilon= */ 5.5,
        /* delta= */ 1e-3,
      )
  }

  @Test
  fun computeMetrics_returnsNoiseValue() {
    val allocatedBudget = AllocatedBudget().apply { initialize(5.5, 1e-3) }
    val thresholdingBudget = AllocatedBudget().apply { initialize(0.0, 1e-8) }
    val combiner =
      PostAggregationPartitionSelectionCombiner(
        AGGREGATION_PARAMS,
        allocatedBudget,
        thresholdingBudget,
        NoiseFactory(),
      )

    val noisedValue = combiner.computeMetrics(privacyIdCountAccumulator { count = 200 })
    assertThat(noisedValue).isWithin(10.0).of(200.0)
    assertThat(noisedValue).isNotEqualTo(200.0)
  }

  companion object {
    private val AGGREGATION_PARAMS =
      AggregationParams(
        metrics = ImmutableList.of(MetricDefinition(MetricType.PRIVACY_ID_COUNT)),
        noiseKind = GAUSSIAN,
        maxPartitionsContributed = 3,
        maxContributionsPerPartition = 5,
      )

    private val noiseMock: Noise = mock()
    private val noiseFactoryMock: (NoiseKind) -> Noise = { _ -> noiseMock }
    private val unusedAllocatedBudget = AllocatedBudget()
  }
}

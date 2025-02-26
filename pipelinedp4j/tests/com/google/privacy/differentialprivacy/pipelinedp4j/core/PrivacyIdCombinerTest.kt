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
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.PRIVACY_ID_COUNT
import com.google.privacy.differentialprivacy.pipelinedp4j.core.NoiseKind.GAUSSIAN
import com.google.privacy.differentialprivacy.pipelinedp4j.core.budget.AllocatedBudget
import com.google.privacy.differentialprivacy.pipelinedp4j.dplibrary.NoiseFactory
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.PrivacyIdContributionsKt.multiValueContribution
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.privacyIdContributions
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.privacyIdCountAccumulator
import com.google.testing.junit.testparameterinjector.TestParameterInjector
import com.google.testing.junit.testparameterinjector.TestParameters
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.kotlin.mock
import org.mockito.kotlin.verify

@RunWith(TestParameterInjector::class)
class PrivacyIdCombinerTest {
  companion object {
    private val AGG_PARAMS =
      AggregationParams(
        metrics = ImmutableList.of(MetricDefinition(PRIVACY_ID_COUNT)),
        noiseKind = GAUSSIAN,
        maxPartitionsContributed = 3,
        maxContributionsPerPartition = 5,
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
  fun createAccumulator_singleValueContributions_initsAccumulatorWithOne() {
    val combiner = PrivacyIdCountCombiner(AGG_PARAMS, UNUSED_ALLOCATED_BUDGET, NoiseFactory())

    val accumulator =
      combiner.createAccumulator(
        privacyIdContributions { singleValueContributions += listOf(1.0, 1.0, 1.0) }
      )

    assertThat(accumulator).isEqualTo(privacyIdCountAccumulator { count = 1 })
  }

  @Test
  fun createAccumulator_multiValueContributions_initsAccumulatorWithOne() {
    val combiner = PrivacyIdCountCombiner(AGG_PARAMS, UNUSED_ALLOCATED_BUDGET, NoiseFactory())

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
    val combiner = PrivacyIdCountCombiner(AGG_PARAMS, UNUSED_ALLOCATED_BUDGET, NoiseFactory())

    val accumulator = combiner.createAccumulator(privacyIdContributions {})

    assertThat(accumulator).isEqualTo(privacyIdCountAccumulator { count = 0 })
  }

  @Test
  fun mergeAccumulators_sumsCounts() {
    val combiner = PrivacyIdCountCombiner(AGG_PARAMS, UNUSED_ALLOCATED_BUDGET, NoiseFactory())

    val accumulator =
      combiner.mergeAccumulators(
        privacyIdCountAccumulator { count = 1 },
        privacyIdCountAccumulator { count = 2 },
      )

    assertThat(accumulator).isEqualTo(privacyIdCountAccumulator { count = 3 })
  }

  @TestParameters("{noiseKind: LAPLACE}", "{noiseKind: GAUSSIAN}")
  fun computeMetrics_addsNoise(noiseKind: NoiseKind) {
    val paramsWithNoise =
      AGG_PARAMS.copy(
        noiseKind = noiseKind,
        maxPartitionsContributed = 30,
        maxContributionsPerPartition = 50,
      )
    val allocatedBudget = AllocatedBudget()
    allocatedBudget.initialize(1.1, 1e-3)
    val combiner = PrivacyIdCountCombiner(paramsWithNoise, allocatedBudget, NoiseFactory())

    val result = combiner.computeMetrics(privacyIdCountAccumulator { count = 1 })

    assertThat(result).isNotEqualTo(1)
  }

  @Test
  fun computeMetrics_passesCorrectParametersToNoise() {
    val allocatedBudget = AllocatedBudget()
    allocatedBudget.initialize(1.1, 1e-3)
    val combiner = PrivacyIdCountCombiner(AGG_PARAMS, allocatedBudget, noiseFactoryMock)

    val unused = combiner.computeMetrics(privacyIdCountAccumulator { count = 1 })

    verify(noiseMock)
      .addNoise(
        /* x= */ 1.0,
        /* l0Sensitivity= */ 3,
        /* lInfSensitivity= */ 1.0,
        /* epsilon= */ 1.1,
        /* delta= */ 1e-3,
      )
  }
}

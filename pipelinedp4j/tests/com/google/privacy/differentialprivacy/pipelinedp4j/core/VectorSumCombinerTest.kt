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
import com.google.privacy.differentialprivacy.GaussianNoise
import com.google.privacy.differentialprivacy.LaplaceNoise
import com.google.privacy.differentialprivacy.Noise
import com.google.privacy.differentialprivacy.pipelinedp4j.core.ContributionBoundingLevel.PARTITION_LEVEL
import com.google.privacy.differentialprivacy.pipelinedp4j.core.ExecutionMode.FULL_TEST_MODE
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.VECTOR_SUM
import com.google.privacy.differentialprivacy.pipelinedp4j.core.NoiseKind.GAUSSIAN
import com.google.privacy.differentialprivacy.pipelinedp4j.core.budget.AllocatedBudget
import com.google.privacy.differentialprivacy.pipelinedp4j.dplibrary.NoiseFactory
import com.google.privacy.differentialprivacy.pipelinedp4j.dplibrary.ZeroNoiseFactory
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.PrivacyIdContributionsKt.featureContribution
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.PrivacyIdContributionsKt.multiValueContribution
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.privacyIdContributions
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.vectorSumAccumulator
import com.google.testing.junit.testparameterinjector.TestParameterInjector
import com.google.testing.junit.testparameterinjector.TestParameters
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.kotlin.mock
import org.mockito.kotlin.verify
import org.mockito.kotlin.verifyNoMoreInteractions

@RunWith(TestParameterInjector::class)
class VectorSumCombinerTest {
  private val VECTOR_SUM_AGG_PARAMS =
    AggregationParams(
      nonFeatureMetrics = ImmutableList.of(),
      features =
        ImmutableList.of(
          VectorFeatureSpec(
            featureId = "value",
            metrics = ImmutableList.of(MetricDefinition(VECTOR_SUM)),
            vectorSize = 3,
            normKind = NormKind.L_INF,
            vectorMaxTotalNorm = 3.0,
          )
        ),
      noiseKind = GAUSSIAN,
      maxPartitionsContributed = 5,
    )

  private val UNUSED_ALLOCATED_BUDGET = AllocatedBudget()

  init {
    UNUSED_ALLOCATED_BUDGET.initialize(1.1, 1e-3)
  }

  @Test
  fun emptyAccumulator_returnsZeroVector() {
    val params =
      VECTOR_SUM_AGG_PARAMS.copy(
        features =
          ImmutableList.of(
            VectorFeatureSpec(
              featureId = "value",
              metrics = ImmutableList.of(MetricDefinition(VECTOR_SUM)),
              vectorSize = 3,
              normKind = NormKind.L_INF,
              vectorMaxTotalNorm = 3.0,
            )
          )
      )
    val combiner =
      VectorSumCombiner(
        params,
        UNUSED_ALLOCATED_BUDGET,
        NoiseFactory(),
        ExecutionMode.PRODUCTION,
        params.features[0] as VectorFeatureSpec,
      )

    val accumulator = combiner.emptyAccumulator()

    assertThat(accumulator)
      .isEqualTo(
        vectorSumAccumulator {
          featureId = "value"
          sumsPerDimension += listOf(0.0, 0.0, 0.0)
        }
      )
  }

  @Test
  fun createAccumulator_sumsVectors() {
    val params =
      VECTOR_SUM_AGG_PARAMS.copy(
        features =
          ImmutableList.of(
            VectorFeatureSpec(
              featureId = "value",
              metrics = ImmutableList.of(MetricDefinition(VECTOR_SUM)),
              vectorSize = 3,
              normKind = NormKind.L1,
              vectorMaxTotalNorm = 300.0,
            )
          )
      )
    val combiner =
      VectorSumCombiner(
        params,
        UNUSED_ALLOCATED_BUDGET,
        NoiseFactory(),
        ExecutionMode.PRODUCTION,
        params.features[0] as VectorFeatureSpec,
      )

    val accumulator =
      combiner.createAccumulator(
        privacyIdContributions {
          features += featureContribution {
            featureId = "value"
            multiValueContributions +=
              listOf(
                multiValueContribution { values += listOf(-10.0, 15.0, 0.0) },
                multiValueContribution { values += listOf(10.0, 20.0, -1.0) },
              )
          }
        }
      )

    // The vector sum is [0.0, 35.0, 1.0], which has L1 norm of 36.
    // The max norm is 300 > 36, so the vector is not clipped.
    assertThat(accumulator)
      .isEqualTo(
        vectorSumAccumulator {
          featureId = "value"
          sumsPerDimension += listOf(0.0, 35.0, -1.0)
        }
      )
  }

  @Test
  fun createAccumulator_perPartitionContributionBoundingEnabledLInfNorm_clampsOnlyTotalVectorSum() {
    val params =
      VECTOR_SUM_AGG_PARAMS.copy(
        features =
          ImmutableList.of(
            VectorFeatureSpec(
              featureId = "value",
              metrics = ImmutableList.of(MetricDefinition(VECTOR_SUM)),
              vectorSize = 3,
              normKind = NormKind.L_INF,
              vectorMaxTotalNorm = 30.0,
            )
          ),
        contributionBoundingLevel = PARTITION_LEVEL,
      )
    val combiner =
      VectorSumCombiner(
        params,
        UNUSED_ALLOCATED_BUDGET,
        NoiseFactory(),
        ExecutionMode.PRODUCTION,
        params.features[0] as VectorFeatureSpec,
      )

    val accumulator =
      combiner.createAccumulator(
        privacyIdContributions {
          features += featureContribution {
            featureId = "value"
            multiValueContributions +=
              listOf(
                multiValueContribution { values += listOf(-10.0, 75.0, 0.0) },
                multiValueContribution { values += listOf(10.0, -40.0, -1.0) },
              )
          }
        }
      )

    // The vector sum is [0.0, 35.0, 1.0], which has L_INF norm of 35.
    // Each component is clipped to -30, 30.
    assertThat(accumulator)
      .isEqualTo(
        vectorSumAccumulator {
          featureId = "value"
          sumsPerDimension += listOf(0.0, 30.0, -1.0)
        }
      )
  }

  @Test
  fun createAccumulator_perPartitionContributionBoundingEnabledL1Norm_clampsOnlyTotalVectorSum() {
    val params =
      VECTOR_SUM_AGG_PARAMS.copy(
        features =
          ImmutableList.of(
            VectorFeatureSpec(
              featureId = "value",
              metrics = ImmutableList.of(MetricDefinition(VECTOR_SUM)),
              vectorSize = 2,
              normKind = NormKind.L1,
              vectorMaxTotalNorm = 10.0,
            )
          )
      )
    val combiner =
      VectorSumCombiner(
        params,
        UNUSED_ALLOCATED_BUDGET,
        NoiseFactory(),
        ExecutionMode.PRODUCTION,
        params.features[0] as VectorFeatureSpec,
      )

    val accumulator =
      combiner.createAccumulator(
        privacyIdContributions {
          features += featureContribution {
            featureId = "value"
            multiValueContributions +=
              listOf(
                multiValueContribution { values += listOf(-4.0, 2.0) },
                multiValueContribution { values += listOf(-5.0, 1.0) },
                multiValueContribution { values += listOf(-3.0, 1.0) },
              )
          }
        }
      )

    assertThat(accumulator)
      .isEqualTo(
        vectorSumAccumulator {
          featureId = "value"
          sumsPerDimension += listOf(-7.5, 2.5)
        }
      )
  }

  @Test
  fun createAccumulator_perPartitionContributionBoundingEnabledL2Norm_clampsOnlyTotalVectorSum() {
    val params =
      VECTOR_SUM_AGG_PARAMS.copy(
        features =
          ImmutableList.of(
            VectorFeatureSpec(
              featureId = "value",
              metrics = ImmutableList.of(MetricDefinition(VECTOR_SUM)),
              vectorSize = 2,
              normKind = NormKind.L2,
              vectorMaxTotalNorm = 6.5,
            )
          )
      )
    val combiner =
      VectorSumCombiner(
        params,
        UNUSED_ALLOCATED_BUDGET,
        NoiseFactory(),
        ExecutionMode.PRODUCTION,
        params.features[0] as VectorFeatureSpec,
      )

    val accumulator =
      combiner.createAccumulator(
        privacyIdContributions {
          features += featureContribution {
            featureId = "value"
            multiValueContributions +=
              listOf(
                multiValueContribution { values += listOf(-10.0, 2.0) },
                multiValueContribution { values += listOf(-2.0, 3.0) },
              )
          }
        }
      )

    assertThat(accumulator)
      .isEqualTo(
        vectorSumAccumulator {
          featureId = "value"
          sumsPerDimension += listOf(-6.0, 2.5)
        }
      )
  }

  @Test
  fun createAccumulator_fullTestMode_doesNotClampTotalSum() {
    val params =
      VECTOR_SUM_AGG_PARAMS.copy(
        features =
          ImmutableList.of(
            VectorFeatureSpec(
              featureId = "value",
              metrics = ImmutableList.of(MetricDefinition(VECTOR_SUM)),
              vectorSize = 3,
              normKind = NormKind.L_INF,
              vectorMaxTotalNorm = 30.0,
            )
          )
      )
    val combiner =
      VectorSumCombiner(
        params,
        UNUSED_ALLOCATED_BUDGET,
        NoiseFactory(),
        FULL_TEST_MODE,
        params.features[0] as VectorFeatureSpec,
      )

    val accumulator =
      combiner.createAccumulator(
        privacyIdContributions {
          features += featureContribution {
            featureId = "value"
            multiValueContributions +=
              listOf(
                multiValueContribution { values += listOf(-10.0, 15.0, 0.0) },
                multiValueContribution { values += listOf(10.0, 20.0, -1.0) },
              )
          }
        }
      )

    assertThat(accumulator)
      .isEqualTo(
        vectorSumAccumulator {
          featureId = "value"
          sumsPerDimension += listOf(0.0, 35.0, -1.0)
        }
      )
  }

  @Test
  fun mergeAccumulators_sumsPartialVectorSums() {
    val combiner =
      VectorSumCombiner(
        VECTOR_SUM_AGG_PARAMS,
        UNUSED_ALLOCATED_BUDGET,
        NoiseFactory(),
        ExecutionMode.PRODUCTION,
        VECTOR_SUM_AGG_PARAMS.features[0] as VectorFeatureSpec,
      )

    val accumulator =
      combiner.mergeAccumulators(
        vectorSumAccumulator {
          featureId = "value"
          sumsPerDimension += listOf(0.0, 35.0, 1.0)
        },
        vectorSumAccumulator {
          featureId = "value"
          sumsPerDimension += listOf(-10.0, 0.0, 1.0)
        },
      )

    assertThat(accumulator)
      .isEqualTo(
        vectorSumAccumulator {
          featureId = "value"
          sumsPerDimension += listOf(-10.0, 35.0, 2.0)
        }
      )
  }

  @Test
  @TestParameters("{noiseKind: LAPLACE, delta: 0.0}", "{noiseKind: GAUSSIAN, delta: 1e-5}")
  fun computeMetrics_addsNoise(noiseKind: NoiseKind, delta: Double) {
    val allocatedBudget = AllocatedBudget()
    allocatedBudget.initialize(1.1, delta)
    val params = VECTOR_SUM_AGG_PARAMS.copy(noiseKind = noiseKind)
    val combiner =
      VectorSumCombiner(
        params,
        allocatedBudget,
        NoiseFactory(),
        ExecutionMode.PRODUCTION,
        params.features[0] as VectorFeatureSpec,
      )

    val result =
      combiner.computeMetrics(vectorSumAccumulator { sumsPerDimension += listOf(1.0, 2.0, 3.0) })

    assertThat(result).isNotEqualTo(listOf(1.0, 2.0, 3.0))
  }

  @Test
  @TestParameters(
    // For all test cases vectorMaxTotalNorm = 30.0, vectorSize = 3, maxPartitionsContributed = 10.
    // Then sensitivity with L_INF norm is: 30 * 3 * 10 = 900.
    // with L1 norm is: 30 * 10 = 300.
    "{normKind: L_INF, expectedSensitivity: 900.0}",
    "{normKind: L1, expectedSensitivity: 300.0}",
  )
  fun computeMetrics_laplaceNoise_passesCorrectParametersToNoise(
    normKind: NormKind,
    expectedSensitivity: Double,
  ) {
    val noiseMock: LaplaceNoise = mock()
    val noiseFactoryMock: (NoiseKind) -> Noise = { _ -> noiseMock }
    val allocatedBudget = AllocatedBudget()
    allocatedBudget.initialize(1.1, 1e-3)
    val params =
      VECTOR_SUM_AGG_PARAMS.copy(
        features =
          ImmutableList.of(
            VectorFeatureSpec(
              featureId = "value",
              metrics = ImmutableList.of(MetricDefinition(VECTOR_SUM)),
              vectorSize = 3,
              normKind = normKind,
              vectorMaxTotalNorm = 30.0,
            )
          ),
        maxPartitionsContributed = 10,
      )
    val combiner =
      VectorSumCombiner(
        params,
        allocatedBudget,
        noiseFactoryMock,
        ExecutionMode.PRODUCTION,
        params.features[0] as VectorFeatureSpec,
      )

    val unused =
      combiner.computeMetrics(vectorSumAccumulator { sumsPerDimension += listOf(1.0, 2.0, 3.0) })

    verify(noiseMock)
      .addNoise(
        /* x= */ 1.0,
        /* l1Sensitivity= */ expectedSensitivity,
        /* epsilon= */ 1.1,
        /* delta= */ 1e-3,
      )
    verify(noiseMock)
      .addNoise(
        /* x= */ 2.0,
        /* l1Sensitivity= */ expectedSensitivity,
        /* epsilon= */ 1.1,
        /* delta= */ 1e-3,
      )
    verify(noiseMock)
      .addNoise(
        /* x= */ 3.0,
        /* l1Sensitivity= */ expectedSensitivity,
        /* epsilon= */ 1.1,
        /* delta= */ 1e-3,
      )
    verifyNoMoreInteractions(noiseMock)
  }

  @Test
  @TestParameters(
    // For all test cases vectorMaxTotalNorm = 30.0, vectorSize = 4, maxPartitionsContributed = 100.
    // Then sensitivity with L_INF norm is: 30 * sqrt(4) * sqrt(100) ~= 600.
    // with L2 norm is: 30 * sqrt(100) ~= 300.
    "{normKind: L_INF, expectedSensitivity: 600.0}",
    "{normKind: L2, expectedSensitivity: 300.0}",
  )
  fun computeMetrics_gaussianNoise_passesCorrectParametersToNoise(
    normKind: NormKind,
    expectedSensitivity: Double,
  ) {
    val noiseMock: GaussianNoise = mock()
    val noiseFactoryMock: (NoiseKind) -> Noise = { _ -> noiseMock }
    val allocatedBudget = AllocatedBudget()
    allocatedBudget.initialize(1.1, 1e-3)
    val params =
      VECTOR_SUM_AGG_PARAMS.copy(
        features =
          ImmutableList.of(
            VectorFeatureSpec(
              featureId = "value",
              metrics = ImmutableList.of(MetricDefinition(VECTOR_SUM)),
              vectorSize = 4,
              normKind = normKind,
              vectorMaxTotalNorm = 30.0,
            )
          ),
        maxPartitionsContributed = 100,
      )
    val combiner =
      VectorSumCombiner(
        params,
        allocatedBudget,
        noiseFactoryMock,
        ExecutionMode.PRODUCTION,
        params.features[0] as VectorFeatureSpec,
      )

    val unused =
      combiner.computeMetrics(
        vectorSumAccumulator { sumsPerDimension += listOf(1.0, 2.0, 3.0, 4.0) }
      )

    verify(noiseMock)
      .addNoise(
        /* x= */ 1.0,
        /* l2Sensitivity= */ expectedSensitivity,
        /* epsilon= */ 1.1,
        /* delta= */ 1e-3,
      )
    verify(noiseMock)
      .addNoise(
        /* x= */ 2.0,
        /* l2Sensitivity= */ expectedSensitivity,
        /* epsilon= */ 1.1,
        /* delta= */ 1e-3,
      )
    verify(noiseMock)
      .addNoise(
        /* x= */ 3.0,
        /* l2Sensitivity= */ expectedSensitivity,
        /* epsilon= */ 1.1,
        /* delta= */ 1e-3,
      )
    verify(noiseMock)
      .addNoise(
        /* x= */ 4.0,
        /* l2Sensitivity= */ expectedSensitivity,
        /* epsilon= */ 1.1,
        /* delta= */ 1e-3,
      )
    verifyNoMoreInteractions(noiseMock)
  }

  @Test
  fun computeMetrics_withoutNoise_withMultipleContributionsIncludingEmptyAccumulator_returnsCorrectResult() {
    val allocatedBudget = AllocatedBudget()
    allocatedBudget.initialize(1.1, 1e-5)
    val params =
      VECTOR_SUM_AGG_PARAMS.copy(
        features =
          ImmutableList.of(
            VectorFeatureSpec(
              featureId = "value",
              metrics = ImmutableList.of(MetricDefinition(VECTOR_SUM)),
              vectorSize = 3,
              normKind = NormKind.L2,
              vectorMaxTotalNorm = 30.0,
            )
          ),
        maxPartitionsContributed = 10,
      )
    val combiner =
      VectorSumCombiner(
        params,
        allocatedBudget,
        ZeroNoiseFactory(),
        ExecutionMode.PRODUCTION,
        params.features[0] as VectorFeatureSpec,
      )

    val accumulator0 = combiner.emptyAccumulator()
    val accumulator1 =
      combiner.createAccumulator(
        privacyIdContributions {
          features += featureContribution {
            featureId = "value"
            multiValueContributions +=
              listOf(
                multiValueContribution { values += listOf(-10.0, 15.0, 1.0) },
                multiValueContribution { values += listOf(10.0, 20.0, -1.0) },
              )
          }
        }
      )
    val accumulator2 =
      combiner.createAccumulator(
        privacyIdContributions {
          features += featureContribution {
            featureId = "value"
            multiValueContributions.add(multiValueContribution { values += listOf(3.0, 0.0, 4.0) })
          }
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
    val params =
      VECTOR_SUM_AGG_PARAMS.copy(
        features =
          ImmutableList.of(
            VectorFeatureSpec(
              featureId = "value",
              metrics = ImmutableList.of(MetricDefinition(VECTOR_SUM)),
              vectorSize = 3,
              normKind = NormKind.L_INF,
              vectorMaxTotalNorm = 3.0,
            )
          )
      )
    val combiner =
      VectorSumCombiner(
        params,
        allocatedBudget,
        ZeroNoiseFactory(),
        ExecutionMode.PRODUCTION,
        params.features[0] as VectorFeatureSpec,
      )

    val result = combiner.computeMetrics(combiner.emptyAccumulator())

    assertThat(result).isEqualTo(listOf(0.0, 0.0, 0.0))
  }
}

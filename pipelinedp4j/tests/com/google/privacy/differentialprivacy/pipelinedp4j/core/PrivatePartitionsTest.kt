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
import com.google.privacy.differentialprivacy.GaussianNoise
import com.google.privacy.differentialprivacy.Noise
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.PRIVACY_ID_COUNT
import com.google.privacy.differentialprivacy.pipelinedp4j.core.NoiseKind.GAUSSIAN
import com.google.privacy.differentialprivacy.pipelinedp4j.core.NoiseKind.LAPLACE
import com.google.privacy.differentialprivacy.pipelinedp4j.core.budget.AllocatedBudget
import com.google.privacy.differentialprivacy.pipelinedp4j.dplibrary.NoiseFactory
import com.google.privacy.differentialprivacy.pipelinedp4j.dplibrary.PreAggregationPartitionSelectionFactory
import com.google.testing.junit.testparameterinjector.TestParameter
import com.google.testing.junit.testparameterinjector.TestParameterInjector
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.kotlin.eq
import org.mockito.kotlin.spy
import org.mockito.kotlin.verify

@RunWith(TestParameterInjector::class)
class PrivatePartitionsTest {
  companion object {
    private val AGG_PARAMS =
      AggregationParams(
        metrics = ImmutableList.of(MetricDefinition(PRIVACY_ID_COUNT)),
        noiseKind = GAUSSIAN,
        maxPartitionsContributed = 3,
        maxContributionsPerPartition = 5,
      )
  }

  @Test
  fun shouldKeep_manyContributions_returnsTrue() {
    val allocatedBudget = AllocatedBudget.create()
    allocatedBudget.initialize(epsilon = 1.5, delta = 2e-5)
    val factorySpy: PreAggregationPartitionSelectionFactory = spy()
    val selector =
      DpLibPreAggregationPartitionSelector(
        maxPartitionsContributed = 3,
        preThreshold = 1,
        allocatedBudget,
        factorySpy,
      )

    // Large number of privacy units with probability close to 1 should be kept.
    assertThat(selector.shouldKeep(10000)).isTrue()

    verify(factorySpy).create(1.5, 2e-5, 3, 1)
  }

  @Test
  fun shouldKeep_manyContributionsButWithHighPreThreshold_returnsFalse() {
    val allocatedBudget = AllocatedBudget.create()
    allocatedBudget.initialize(epsilon = 1.5, delta = 2e-5)
    val factorySpy: PreAggregationPartitionSelectionFactory = spy()
    val highPreThresholdValue = 100000
    val selector =
      DpLibPreAggregationPartitionSelector(
        maxPartitionsContributed = 3,
        preThreshold = highPreThresholdValue,
        allocatedBudget,
        factorySpy,
      )

    // PreThreshold is too high, so the partition is dropped.
    assertThat(selector.shouldKeep(10000)).isFalse()

    verify(factorySpy).create(1.5, 2e-5, 3, highPreThresholdValue)
  }

  @Test
  fun shouldKeep_fewContributions_returnsFalse() {
    val allocatedBudget = AllocatedBudget.create()
    allocatedBudget.initialize(epsilon = 1.0, delta = 1e-15)
    val selector =
      DpLibPreAggregationPartitionSelector(
        maxPartitionsContributed = 3,
        preThreshold = 1,
        allocatedBudget,
        PreAggregationPartitionSelectionFactory(),
      )

    // Only 1 privacy unit, with probability < 1e-15 should be dropped.
    assertThat(selector.shouldKeep(1)).isFalse()
  }

  enum class ThresholdTestCase(
    val epsilon: Double,
    val delta: Double,
    val thresholdingDelta: Double,
    val noiseKind: NoiseKind,
    val maxPartitionsContributed: Int,
    val expectedThreshold: Double,
  ) {
    TEST_CASE_GAUSSIAN_1(
      epsilon = 1.0,
      delta = 1e-4,
      thresholdingDelta = 1e-8,
      noiseKind = GAUSSIAN,
      maxPartitionsContributed = 1,
      expectedThreshold = 18.88,
    ),
    TEST_CASE_GAUSSIAN_2(
      epsilon = 0.001,
      delta = 1e-5,
      thresholdingDelta = 1e-9,
      noiseKind = GAUSSIAN,
      maxPartitionsContributed = 2,
      expectedThreshold = 14905.02,
    ),
    TEST_CASE_GAUSSIAN_3(
      epsilon = 10.0,
      delta = 1e-3,
      thresholdingDelta = 2e-6,
      noiseKind = GAUSSIAN,
      maxPartitionsContributed = 3,
      expectedThreshold = 4.4,
    ),
    TEST_CASE_LAPLACE_1(
      epsilon = 1.1,
      delta = 0.0,
      thresholdingDelta = 1e-5,
      noiseKind = LAPLACE,
      maxPartitionsContributed = 1,
      expectedThreshold = 10.84,
    ),
    TEST_CASE_LAPLACE_2(
      epsilon = 0.01,
      delta = 0.0,
      thresholdingDelta = 1e-10,
      noiseKind = LAPLACE,
      maxPartitionsContributed = 2,
      expectedThreshold = 4606.17,
    ),
    TEST_CASE_LAPLACE_3(
      epsilon = 5.0,
      delta = 0.0,
      thresholdingDelta = 1e-2,
      noiseKind = LAPLACE,
      maxPartitionsContributed = 10,
      expectedThreshold = 13.42,
    ),
  }

  @Test
  fun threshold_returnsExpectedThreshold(@TestParameter testCase: ThresholdTestCase) {
    val noiseBudget =
      AllocatedBudget.create().apply {
        initialize(epsilon = testCase.epsilon, delta = testCase.delta)
      }
    val thresholdingBudget =
      AllocatedBudget.create().apply {
        initialize(epsilon = 0.0, delta = testCase.thresholdingDelta)
      }
    val selector =
      PostAggregationPartitionSelectorImpl(
        testCase.maxPartitionsContributed,
        testCase.noiseKind,
        preThreshold = 1,
        noiseBudget,
        thresholdingBudget,
        NoiseFactory(),
      )

    assertThat(selector.threshold).isWithin(1e-2).of(testCase.expectedThreshold)
  }

  @Test
  fun addNoiseIfShouldKeep_keepDropAsExpected() {
    val noiseBudget = AllocatedBudget.create().apply { initialize(epsilon = 1.0, delta = 1e-3) }
    val thresholdingBudget =
      AllocatedBudget.create().apply { initialize(epsilon = 0.0, delta = 1e-10) }
    val gaussianNoiseSpy: GaussianNoise = spy()
    val noiseFactoryMock: (NoiseKind) -> Noise = { _ -> gaussianNoiseSpy }

    val selector =
      PostAggregationPartitionSelectorImpl(
        maxPartitionsContributed = 2,
        GAUSSIAN,
        preThreshold = 1,
        noiseBudget,
        thresholdingBudget,
        noiseFactoryMock,
      )

    assertThat(selector.threshold).isWithin(1e-1).of(24.5)

    // A partition with small number of privacy units is dropped
    assertThat(selector.addNoiseIfShouldKeep(1)).isNull()
    verify(gaussianNoiseSpy).addNoise(eq(1.0), eq(2), eq(1.0), eq(1.0), eq(1e-3))

    // A partition with large number of privacy units is kept
    assertThat(selector.addNoiseIfShouldKeep(200)).isWithin(70.0).of(200.0)
    verify(gaussianNoiseSpy).addNoise(eq(200.0), eq(2), eq(1.0), eq(1.0), eq(1e-3))
  }

  @Test
  fun addNoiseIfShouldKeep_preThresholdGreaterThanOne_keepDropAsExpected() {
    val noiseBudget = AllocatedBudget.create().apply { initialize(epsilon = 1.0, delta = 1e-3) }
    val thresholdingBudget =
      AllocatedBudget.create().apply { initialize(epsilon = 0.0, delta = 1e-10) }
    val gaussianNoiseSpy: GaussianNoise = spy()
    val noiseFactoryMock: (NoiseKind) -> Noise = { _ -> gaussianNoiseSpy }

    val selector =
      PostAggregationPartitionSelectorImpl(
        maxPartitionsContributed = 2,
        GAUSSIAN,
        preThreshold = 100,
        noiseBudget,
        thresholdingBudget,
        noiseFactoryMock,
      )

    // A partition with number of privacy units equal to preThreshold is dropped
    assertThat(selector.addNoiseIfShouldKeep(100)).isNull()

    // A partition with large number of privacy units is kept
    assertThat(selector.addNoiseIfShouldKeep(200)).isWithin(70.0).of(200.0)
  }

  @Test
  fun addNoiseIfShouldKeep_privacyIdCountIsLessThanPreThreshold_dropsImmediatelyWithoutAnyComputations() {
    val selector =
      PostAggregationPartitionSelectorImpl(
        maxPartitionsContributed = 0,
        noiseKind = GAUSSIAN,
        // only preThreshold matters.
        preThreshold = 100,
        noiseBudget = AllocatedBudget.create(),
        thresholdingBudget = AllocatedBudget.create(),
        NoiseFactory(),
      )

    assertThat(selector.addNoiseIfShouldKeep(99)).isNull()
  }

  @Test
  fun shouldKeep_noPrivacy_alwaysReturnsTrue() {
    val selector = NoPrivacyPartitionSelector()

    // Large number of privacy units should be kept.
    assertThat(selector.shouldKeep(10000)).isTrue()
    // No privacy units should also be kept.
    assertThat(selector.shouldKeep(0)).isTrue()
  }
}

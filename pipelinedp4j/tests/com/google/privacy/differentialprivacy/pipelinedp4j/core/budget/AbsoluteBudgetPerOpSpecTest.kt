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

package com.google.privacy.differentialprivacy.pipelinedp4j.core.budget

import com.google.common.truth.Truth.assertThat
import com.google.testing.junit.testparameterinjector.TestParameterInjector
import com.google.testing.junit.testparameterinjector.TestParameters
import kotlin.test.assertFailsWith
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(TestParameterInjector::class)
class AbsoluteBudgetPerOpSpecTest {
  @Test
  @TestParameters(
    "{epsilon: -1.0, delta: 0.5}",
    "{epsilon: 0.0, delta: 0.0}",
    "{epsilon: 0.5, delta: -1.0}",
  )
  fun create_invalidParameters_throws(epsilon: Double, delta: Double) {
    assertFailsWith<IllegalArgumentException> { AbsoluteBudgetPerOpSpec(epsilon, delta) }
  }

  @Test
  @TestParameters(
    "{epsilon: 0.5, delta: 0.5}",
    "{epsilon: 0.0, delta: 0.5}",
    "{epsilon: 0.5, delta: 0.0}",
  )
  fun create_validParameters_createsObjectChecksContents(epsilon: Double, delta: Double) {
    val absoluteBudgetPerOpSpec = AbsoluteBudgetPerOpSpec(epsilon, delta)
    assertThat(absoluteBudgetPerOpSpec).isNotNull()
    assertThat(absoluteBudgetPerOpSpec.epsilon).isEqualTo(epsilon)
    assertThat(absoluteBudgetPerOpSpec.delta).isEqualTo(delta)
  }

  @Test
  @TestParameters(
    "{initialEpsilon: 2.0, delta: 0.5, factor: 0.5, calculatedWeight: 1.0, calculatedDelta: 0.25}",
    "{initialEpsilon: 2.0, delta: 0.5, factor: 1, calculatedWeight: 2.0, calculatedDelta: 0.5}",
  )
  fun times_validInput_hasCorrectCalculation(
    initialEpsilon: Double,
    delta: Double,
    factor: Double,
    calculatedWeight: Double,
    calculatedDelta: Double,
  ) {
    assertThat(AbsoluteBudgetPerOpSpec(initialEpsilon, delta).times(factor))
      .isEqualTo(AbsoluteBudgetPerOpSpec(calculatedWeight, calculatedDelta))
  }

  @Test
  fun times_invalidCalculatedWeight_throws() {
    assertFailsWith<IllegalArgumentException> { AbsoluteBudgetPerOpSpec(1.0, 0.5).times(-1.0) }
  }
}

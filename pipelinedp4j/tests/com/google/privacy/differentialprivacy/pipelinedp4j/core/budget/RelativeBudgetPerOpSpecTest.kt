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
class RelativeBudgetPerOpSpecTest {
  @Test
  @TestParameters("{weight: -1.0}", "{weight: 0.0}")
  fun create_invalidWeight_throws(weight: Double) {
    assertFailsWith<IllegalArgumentException> { RelativeBudgetPerOpSpec(weight) }
  }

  @Test
  fun create_validWeight_createsObjectChecksContents() {
    val relativeBudgetPerOpSpec = RelativeBudgetPerOpSpec(1.0)
    assertThat(relativeBudgetPerOpSpec).isNotNull()
    assertThat(relativeBudgetPerOpSpec.weight).isEqualTo(1.0)
  }

  @Test
  @TestParameters(
    "{initialWeight: 2.0, factor: 0.5, calculatedWeight: 1.0}",
    "{initialWeight: 2.0, factor: 1, calculatedWeight: 2.0}",
  )
  fun times_validInput_hasCorrectCalculation(
    initialWeight: Double,
    factor: Double,
    calculatedWeight: Double,
  ) {
    assertThat(RelativeBudgetPerOpSpec(initialWeight).times(factor))
      .isEqualTo(RelativeBudgetPerOpSpec(calculatedWeight))
  }

  @Test
  @TestParameters("{factor: 0.0}", "{factor: -1.0}")
  fun times_invalidCalculatedWeight_throws(factor: Double) {
    assertFailsWith<IllegalArgumentException> { RelativeBudgetPerOpSpec(1.0).times(factor) }
  }
}

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
class TotalBudgetTest {
  @Test
  @TestParameters(
    "{epsilon: -1.0, delta: 0.5}",
    "{epsilon: 0.0, delta: 0.5}",
    "{epsilon: 0.5, delta: -1.0}",
  )
  fun create_invalidParameters_throws(epsilon: Double, delta: Double) {
    assertFailsWith<IllegalArgumentException> { TotalBudget(epsilon, delta) }
  }

  @Test
  @TestParameters("{epsilon: 0.5, delta: 0.5}", "{epsilon: 0.5, delta: 0.0}")
  fun create_validParameters_createsObjectChecksContents(epsilon: Double, delta: Double) {
    val totalBudget = TotalBudget(epsilon, delta)
    assertThat(totalBudget).isNotNull()
    assertThat(totalBudget.epsilon).isEqualTo(epsilon)
    assertThat(totalBudget.delta).isEqualTo(delta)
  }
}

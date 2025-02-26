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

import com.google.common.truth.Truth.assertThat
import com.google.common.truth.extensions.proto.ProtoTruth.assertThat
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.privacyIdContributions
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.privacyIdCountAccumulator
import com.google.testing.junit.testparameterinjector.TestParameterInjector
import kotlin.test.assertFailsWith
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(TestParameterInjector::class)
class ExactPrivacyIdCountCombinerTest {

  @Test
  fun createAccumulator_initsAccumulatorWithOne() {
    val combiner = ExactPrivacyIdCountCombiner()

    val accumulator =
      combiner.createAccumulator(
        privacyIdContributions { singleValueContributions += listOf(1.0, 1.0, 1.0) }
      )

    assertThat(accumulator).isEqualTo(privacyIdCountAccumulator { count = 1 })
  }

  @Test
  fun mergeAccumulators_sumsCounts() {
    val combiner = ExactPrivacyIdCountCombiner()

    val accumulator =
      combiner.mergeAccumulators(
        privacyIdCountAccumulator { count = 1 },
        privacyIdCountAccumulator { count = 2 },
      )

    assertThat(accumulator).isEqualTo(privacyIdCountAccumulator { count = 3 })
  }

  @Test
  fun computeMetrics_throwsExceptions() {
    val combiner = ExactPrivacyIdCountCombiner()

    val throwable =
      assertFailsWith<UnsupportedOperationException> {
        combiner.computeMetrics(privacyIdCountAccumulator { count = 1 })
      }
    assertThat(throwable)
      .hasMessageThat()
      .contains("ExactPrivacyIdCountCombiner does not support compute_metrics")
  }
}

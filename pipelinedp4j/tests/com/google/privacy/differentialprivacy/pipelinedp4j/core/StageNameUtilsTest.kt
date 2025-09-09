/*
 * Copyright 2025 Google LLC
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
import com.google.privacy.differentialprivacy.pipelinedp4j.core.StageNameUtils.makeStageNameUnique
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4

@RunWith(JUnit4::class)
class StageNameUtilsTest {
  @Test
  fun makeStageNameUnique_handlesCollisions() {
    val stageName = "stage"
    assertThat(stageName.makeStageNameUnique()).isEqualTo("stage")
    assertThat(stageName.makeStageNameUnique()).isEqualTo("stage_1")
    assertThat(stageName.makeStageNameUnique()).isEqualTo("stage_2")
    assertThat(stageName.makeStageNameUnique()).isEqualTo("stage_3")

    val anotherStageName = "anotherstage"
    assertThat(anotherStageName.makeStageNameUnique()).isEqualTo("anotherstage")
    assertThat(anotherStageName.makeStageNameUnique()).isEqualTo("anotherstage_1")
    assertThat(anotherStageName.makeStageNameUnique()).isEqualTo("anotherstage_2")
  }
}

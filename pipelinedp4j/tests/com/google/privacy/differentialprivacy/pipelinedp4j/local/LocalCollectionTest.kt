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

package com.google.privacy.differentialprivacy.pipelinedp4j.local

import com.google.common.truth.Truth.assertThat
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4

@RunWith(JUnit4::class)
class LocalCollectionTest {
  val ENCODER_FACTORY: LocalEncoderFactory = LocalEncoderFactory()

  @Test
  fun distinct_removesDuplicates() {
    val localCollection = LocalCollection(sequenceOf(1, 2, 1))

    val result = localCollection.distinct("stageName")

    assertThat(result.data.asIterable()).containsExactly(1, 2)
  }

  @Test
  fun map_appliesMapFn() {
    val localCollection = LocalCollection(sequenceOf(1))

    val result: LocalCollection<String> =
      localCollection.map("Test", ENCODER_FACTORY.strings(), { v -> v.toString() })
        as LocalCollection<String>

    assertThat(result.data.asIterable()).containsExactly("1")
  }

  @Test
  fun keyBy_keysCollection() {
    val localCollection = LocalCollection(sequenceOf(1))

    val result: LocalTable<String, Int> =
      localCollection.keyBy("Test", ENCODER_FACTORY.strings(), { v -> v.toString() })
        as LocalTable<String, Int>

    assertThat(result.data.asIterable()).containsExactly(Pair("1", 1))
  }

  @Test
  fun mapToTable_appliesMapFn() {
    val localCollection = LocalCollection(sequenceOf(1))

    val result: LocalTable<String, Int> =
      localCollection.mapToTable(
        "Test",
        ENCODER_FACTORY.strings(),
        ENCODER_FACTORY.ints(),
        { v -> Pair(v.toString(), v) },
      ) as LocalTable<String, Int>

    assertThat(result.data.asIterable()).containsExactly(Pair("1", 1))
  }
}

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

package com.google.privacy.differentialprivacy.pipelinedp4j.beam

import com.google.common.truth.Truth.assertThat
import org.apache.beam.sdk.coders.VarIntCoder
import org.apache.beam.sdk.testing.PAssert
import org.apache.beam.sdk.testing.TestPipeline
import org.apache.beam.sdk.transforms.Create
import org.apache.beam.sdk.values.KV
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4

@RunWith(JUnit4::class)
class BeamCollectionTest {
  @get:Rule val testPipeline: TestPipeline = TestPipeline.create()

  @Test
  fun elementsEncoder_returnsCorrectEncoder() {
    val pCollection = testPipeline.apply(Create.of<Int>(listOf()).withCoder(VarIntCoder.of()))
    val beamCollection = BeamCollection(pCollection)

    val result = beamCollection.elementsEncoder

    testPipeline.run().waitUntilFinish()

    assertThat(result).isInstanceOf(BeamEncoder::class.java)
    assertThat(result.coder).isEqualTo(VarIntCoder.of())
  }

  @Test
  fun distinct_removesDuplicates() {
    val pCollection = testPipeline.apply(Create.of(listOf(1, 2, 1)).withCoder(VarIntCoder.of()))
    val beamCollection = BeamCollection(pCollection)

    val result: BeamCollection<Int> = beamCollection.distinct("stageName")

    PAssert.that(result.data).containsInAnyOrder(1, 2)

    testPipeline.run().waitUntilFinish()
  }

  @Test
  fun map_appliesMapFn() {
    val pCollection = testPipeline.apply(Create.of(listOf(1)).withCoder(VarIntCoder.of()))
    val beamCollection = BeamCollection(pCollection)

    val result: BeamCollection<String> =
      beamCollection.map("Test", beamEncoderFactory.strings(), { v -> v.toString() })

    PAssert.that(result.data).containsInAnyOrder("1")

    testPipeline.run().waitUntilFinish()
  }

  @Test
  fun keyBy_keysCollection() {
    val pCollection = testPipeline.apply(Create.of(listOf(1)).withCoder(VarIntCoder.of()))
    val beamCollection = BeamCollection(pCollection)

    val result: BeamTable<String, Int> =
      beamCollection.keyBy("Test", beamEncoderFactory.strings(), { v -> v.toString() })

    PAssert.that(result.data).containsInAnyOrder(KV.of("1", 1))

    testPipeline.run().waitUntilFinish()
  }

  @Test
  fun mapToTable_appliesMapFn() {
    val pCollection = testPipeline.apply(Create.of(listOf(1)).withCoder(VarIntCoder.of()))
    val beamCollection = BeamCollection(pCollection)

    val result: BeamTable<String, Int> =
      beamCollection.mapToTable(
        "Test",
        beamEncoderFactory.strings(),
        beamEncoderFactory.ints(),
        { v -> Pair(v.toString(), v) },
      )

    PAssert.that(result.data).containsInAnyOrder(KV.of("1", 1))

    testPipeline.run().waitUntilFinish()
  }

  companion object {
    private val beamEncoderFactory = BeamEncoderFactory()
  }
}

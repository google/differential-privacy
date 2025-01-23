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
import com.google.privacy.differentialprivacy.pipelinedp4j.local.LocalCollection
import com.google.privacy.differentialprivacy.pipelinedp4j.local.LocalTable
import com.google.testing.junit.testparameterinjector.TestParameter
import com.google.testing.junit.testparameterinjector.TestParameterInjector
import org.apache.beam.sdk.coders.KvCoder
import org.apache.beam.sdk.coders.StringUtf8Coder
import org.apache.beam.sdk.coders.VarIntCoder
import org.apache.beam.sdk.testing.PAssert
import org.apache.beam.sdk.testing.TestPipeline
import org.apache.beam.sdk.transforms.Create
import org.apache.beam.sdk.values.KV
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(TestParameterInjector::class)
class BeamTableTest {
  @get:Rule val testPipeline: TestPipeline = TestPipeline.create()

  @Test
  fun keysEncoder_returnsCorrectEncoder() {
    val pCollection =
      testPipeline.apply(
        Create.of<KV<String, Int>>(listOf())
          .withCoder(KvCoder.of(StringUtf8Coder.of(), VarIntCoder.of()))
      )
    val beamCollection = BeamTable(pCollection)

    val result = beamCollection.keysEncoder

    testPipeline.run().waitUntilFinish()

    assertThat(result).isInstanceOf(BeamEncoder::class.java)
    assertThat(result.coder).isEqualTo(StringUtf8Coder.of())
  }

  @Test
  fun valuesEncoder_returnsCorrectEncoder() {
    val pCollection =
      testPipeline.apply(
        Create.of<KV<String, Int>>(listOf())
          .withCoder(KvCoder.of(StringUtf8Coder.of(), VarIntCoder.of()))
      )
    val beamCollection = BeamTable(pCollection)

    val result = beamCollection.valuesEncoder

    testPipeline.run().waitUntilFinish()

    assertThat(result).isInstanceOf(BeamEncoder::class.java)
    assertThat(result.coder).isEqualTo(VarIntCoder.of())
  }

  @Test
  fun map_appliesMapFn() {
    val pCollection =
      testPipeline.apply(
        Create.of(listOf(KV.of(1, 10))).withCoder(KvCoder.of(VarIntCoder.of(), VarIntCoder.of()))
      )
    val beamCollection = BeamTable(pCollection)
    val mapFn: (Int, Int) -> String = { k, v -> "${k}_$v" }

    val result: BeamCollection<String> =
      beamCollection.map("Test", beamEncoderFactory.strings(), mapFn)

    PAssert.that(result.data).containsInAnyOrder("1_10")

    testPipeline.run().waitUntilFinish()
  }

  @Test
  fun groupAndCombineValues_appliesCombiner() {
    val pCollection =
      testPipeline.apply(
        Create.of(
            listOf(
              KV.of("positive", 1),
              KV.of("positive", 10),
              KV.of("negative", -1),
              KV.of("negative", -10),
            )
          )
          .withCoder(KvCoder.of(StringUtf8Coder.of(), VarIntCoder.of()))
      )
    val beamCollection = BeamTable(pCollection)
    val combineFn: (Int, Int) -> Int = { v1, v2 -> v1 + v2 }

    val result: BeamTable<String, Int> = beamCollection.groupAndCombineValues("Test", combineFn)

    PAssert.that(result.data).containsInAnyOrder(KV.of("positive", 11), KV.of("negative", -11))

    testPipeline.run().waitUntilFinish()
  }

  @Test
  fun groupByKey_groupsValues() {
    val pCollection =
      testPipeline.apply(
        Create.of(listOf(KV.of("positive", 1), KV.of("positive", 10), KV.of("negative", -1)))
          .withCoder(KvCoder.of(StringUtf8Coder.of(), VarIntCoder.of()))
      )
    val beamCollection = BeamTable(pCollection)

    val result: BeamTable<String, Iterable<Int>> = beamCollection.groupByKey("stageName")

    val expected = mapOf("positive" to setOf(1, 10), "negative" to setOf(-1))
    // We can't use PAssert.containsInAnyOrder because order in Iterable<V> is not deterministic.
    PAssert.that(result.data).satisfies { output: Iterable<KV<String, Iterable<Int>>> ->
      val kotlinMap = output.associate { it.key to it.value.toSet() }
      assertThat(expected).isEqualTo(kotlinMap)
      // we need to return something to satisfy the function signature.
      null
    }

    testPipeline.run().waitUntilFinish()
  }

  @Test
  fun keys_returnsKeys() {
    val pCollection =
      testPipeline.apply(
        Create.of(listOf(KV.of("key", "value")))
          .withCoder(KvCoder.of(StringUtf8Coder.of(), StringUtf8Coder.of()))
      )
    val beamCollection = BeamTable(pCollection)

    val result: BeamCollection<String> = beamCollection.keys("stageName")

    PAssert.that(result.data).containsInAnyOrder("key")

    testPipeline.run().waitUntilFinish()
  }

  @Test
  fun values_returnsValues() {
    val pCollection =
      testPipeline.apply(
        Create.of(listOf(KV.of("key", "value")))
          .withCoder(KvCoder.of(StringUtf8Coder.of(), StringUtf8Coder.of()))
      )
    val beamCollection = BeamTable(pCollection)

    val result: BeamCollection<String> = beamCollection.values("stageName")

    PAssert.that(result.data).containsInAnyOrder("value")

    testPipeline.run().waitUntilFinish()
  }

  @Test
  fun mapValues_appliesMapFn() {
    val pCollection =
      testPipeline.apply(
        Create.of(listOf(KV.of("one", 1)))
          .withCoder(KvCoder.of(StringUtf8Coder.of(), VarIntCoder.of()))
      )
    val beamCollection = BeamTable(pCollection)
    val mapFn: (String, Int) -> String = { k, v -> "${k}_$v" }

    val result: BeamTable<String, String> =
      beamCollection.mapValues("stageName", beamEncoderFactory.strings(), mapFn)

    PAssert.that(result.data).containsInAnyOrder(KV.of("one", "one_1"))

    testPipeline.run().waitUntilFinish()
  }

  @Test
  fun mapToTable_appliesMapFn() {
    val pCollection =
      testPipeline.apply(
        Create.of(listOf(KV.of("one", 1)))
          .withCoder(KvCoder.of(StringUtf8Coder.of(), VarIntCoder.of()))
      )
    val beamCollection = BeamTable(pCollection)
    val mapFn: (String, Int) -> Pair<Int, String> = { k, v -> Pair(v, k) }

    val result: BeamTable<Int, String> =
      beamCollection.mapToTable(
        "Test",
        beamEncoderFactory.ints(),
        beamEncoderFactory.strings(),
        mapFn,
      )

    PAssert.that(result.data).containsInAnyOrder(KV.of(1, "one"))

    testPipeline.run().waitUntilFinish()
  }

  @Test
  fun flatMapToTable_appliesMapFn() {
    val pCollection =
      testPipeline.apply(
        Create.of(listOf(KV.of("one", 1)))
          .withCoder(KvCoder.of(StringUtf8Coder.of(), VarIntCoder.of()))
      )
    val beamCollection = BeamTable(pCollection)
    val mapFn: (String, Int) -> Sequence<Pair<Int, String>> = { k, v ->
      sequenceOf(Pair(v, k), Pair(v, k))
    }

    val result: BeamTable<Int, String> =
      beamCollection.flatMapToTable(
        "Test",
        beamEncoderFactory.ints(),
        beamEncoderFactory.strings(),
        mapFn,
      )

    PAssert.that(result.data).containsInAnyOrder(KV.of(1, "one"), KV.of(1, "one"))

    testPipeline.run().waitUntilFinish()
  }

  @Test
  fun filterValues_appliesPredicate() {
    val pCollection =
      testPipeline.apply(
        Create.of(listOf(KV.of("one", 1), KV.of("two", 2)))
          .withCoder(KvCoder.of(StringUtf8Coder.of(), VarIntCoder.of()))
      )
    val beamCollection = BeamTable(pCollection)
    val predicate: (Int) -> Boolean = { v -> v == 1 }

    val result: BeamTable<String, Int> = beamCollection.filterValues("Test", predicate)

    PAssert.that(result.data).containsInAnyOrder(KV.of("one", 1))

    testPipeline.run().waitUntilFinish()
  }

  @Test
  fun filterKeys_appliesPredicate() {
    val pCollection =
      testPipeline.apply(
        Create.of(listOf(KV.of("one", 1), KV.of("two", 2), KV.of("three", 3), KV.of("two", -2)))
          .withCoder(KvCoder.of(StringUtf8Coder.of(), VarIntCoder.of()))
      )
    val beamCollection = BeamTable(pCollection)
    val predicate: (String) -> Boolean = { k -> k == "two" }

    val result: BeamTable<String, Int> = beamCollection.filterKeys("Test", predicate)

    PAssert.that(result.data).containsInAnyOrder(KV.of("two", 2), KV.of("two", -2))

    testPipeline.run().waitUntilFinish()
  }

  @Test
  fun filterKeys_allowedKeysStoredInBeamCollection_keepsOnlyAllowedKeys(
    @TestParameter unbalancedKeys: Boolean
  ) {
    val pCollection =
      testPipeline.apply(
        "CreateInputData",
        Create.of(listOf(KV.of("one", 1), KV.of("two", 2), KV.of("three", 3), KV.of("two", -2)))
          .withCoder(KvCoder.of(StringUtf8Coder.of(), VarIntCoder.of())),
      )
    val beamCollection = BeamTable(pCollection)
    val allowedKeysPCollection =
      testPipeline.apply(
        "CreateAllowedKeys",
        Create.of(listOf("three", "two", "four")).withCoder(StringUtf8Coder.of()),
      )
    val allowedKeysBeamCollection = BeamCollection(allowedKeysPCollection)

    val result: BeamTable<String, Int> =
      beamCollection.filterKeys("stageName", allowedKeysBeamCollection, unbalancedKeys)

    PAssert.that(result.data)
      .containsInAnyOrder(KV.of("two", 2), KV.of("three", 3), KV.of("two", -2))

    testPipeline.run().waitUntilFinish()
  }

  @Test
  fun filterKeys_allowedKeysStoredInLocalCollection_keepsOnlyAllowedKeys() {
    val pCollection =
      testPipeline.apply(
        "CreateInputData",
        Create.of(listOf(KV.of("one", 1), KV.of("two", 2), KV.of("three", 3), KV.of("two", -2)))
          .withCoder(KvCoder.of(StringUtf8Coder.of(), VarIntCoder.of())),
      )
    val beamCollection = BeamTable(pCollection)
    val allowedKeys = sequenceOf("three", "two", "four")
    val allowedKeysLocalCollection = LocalCollection(allowedKeys)

    val result: BeamTable<String, Int> =
      beamCollection.filterKeys("stageName", allowedKeysLocalCollection)

    PAssert.that(result.data)
      .containsInAnyOrder(KV.of("two", 2), KV.of("three", 3), KV.of("two", -2))

    testPipeline.run().waitUntilFinish()
  }

  @Test
  fun flattenWith_beamTable_flattensCollections() {
    val pCollection =
      testPipeline.apply(
        "Create1",
        Create.of(listOf(KV.of("one", 1)))
          .withCoder(KvCoder.of(StringUtf8Coder.of(), VarIntCoder.of())),
      )
    val beamCollection = BeamTable(pCollection)
    val otherPCollection =
      testPipeline.apply(
        "Create2",
        Create.of(listOf(KV.of("two", 2)))
          .withCoder(KvCoder.of(StringUtf8Coder.of(), VarIntCoder.of())),
      )
    val otherBeamCollection = BeamTable(otherPCollection)

    val result: BeamTable<String, Int> =
      beamCollection.flattenWith("stageName", otherBeamCollection)

    PAssert.that(result.data).containsInAnyOrder(KV.of("one", 1), KV.of("two", 2))

    testPipeline.run().waitUntilFinish()
  }

  @Test
  fun flattenWith_localTable_flattensCollections() {
    val pCollection =
      testPipeline.apply(
        "Create1",
        Create.of(listOf(KV.of("one", 1)))
          .withCoder(KvCoder.of(StringUtf8Coder.of(), VarIntCoder.of())),
      )
    val beamCollection = BeamTable(pCollection)
    val otherPCollection = sequenceOf("two" to 2)
    val otherBeamCollection = LocalTable(otherPCollection)

    val result: BeamTable<String, Int> =
      beamCollection.flattenWith("stageName", otherBeamCollection)

    PAssert.that(result.data).containsInAnyOrder(KV.of("one", 1), KV.of("two", 2))

    testPipeline.run().waitUntilFinish()
  }

  @Test
  fun samplePerKey_samplesElements() {
    val inputData =
      listOf(
        KV.of("one", 1),
        KV.of("one", 2),
        KV.of("one", 3),
        KV.of("one", 4),
        KV.of("one", 5),
        KV.of("two", 6),
        KV.of("two", 7),
        KV.of("two", 8),
        KV.of("two", 9),
        KV.of("two", 10),
        KV.of("three", 14),
        KV.of("three", 15),
        KV.of("four", 21),
      )
    val pCollection =
      testPipeline.apply(
        "CreateInputData",
        Create.of(inputData).withCoder(KvCoder.of(StringUtf8Coder.of(), VarIntCoder.of())),
      )
    val beamTable = BeamTable(pCollection)

    val result: BeamTable<String, Iterable<Int>> = beamTable.samplePerKey("Test", 3)

    PAssert.that(result.data).satisfies { output: Iterable<KV<String, Iterable<Int>>> ->
      val kotlinMap = output.associate { it.key to it.value.toList() }
      assertThat(kotlinMap.size).isEqualTo(4)
      assertThat(kotlinMap["one"]!!.size).isEqualTo(3)
      assertThat(kotlinMap["two"]!!.size).isEqualTo(3)
      assertThat(kotlinMap["three"]!!.size).isEqualTo(2)
      assertThat(kotlinMap["four"]!!.size).isEqualTo(1)
      null
    }

    testPipeline.run().waitUntilFinish()
  }

  companion object {
    private val beamEncoderFactory = BeamEncoderFactory()
  }
}

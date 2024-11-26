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
import com.google.testing.junit.testparameterinjector.TestParameter
import com.google.testing.junit.testparameterinjector.TestParameterInjector
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(TestParameterInjector::class)
class LocalTableTest {
  val ENCODER_FACTORY: LocalEncoderFactory = LocalEncoderFactory()

  @Test
  fun map_appliesMapFn() {
    val localTable = LocalTable(sequenceOf(1 to 10))
    val mapFn: (Int, Int) -> String = { k, v -> k.toString() + "_" + v.toString() }

    val result: LocalCollection<String> =
      localTable.map("Test", ENCODER_FACTORY.strings(), mapFn) as LocalCollection<String>

    assertThat(result.data.asIterable()).containsExactly("1_10")
  }

  @Test
  fun groupAndCombineValues_appliesCombiner() {
    val localTable =
      LocalTable(
        sequenceOf(
          Pair("positive", 1),
          Pair("positive", 10),
          Pair("negative", -1),
          Pair("negative", -10),
        )
      )
    val combineFn: (Int, Int) -> Int = { v1, v2 -> v1 + v2 }

    val result = localTable.groupAndCombineValues("Test", combineFn) as LocalTable<String, Int>

    assertThat(result.data.asIterable())
      .containsExactly(Pair("positive", 11), Pair("negative", -11))
  }

  @Test
  fun groupAndCombineValues_executesLazy() {
    var initialized = false
    val localTable =
      LocalTable(
        (1..2).asSequence().map {
          require(initialized) { "Not initialized" }
          Pair("key", it)
        }
      )

    val combineFn: (Int, Int) -> Int = { v1, v2 -> v1 + v2 }

    // Check that groupAndCombineValues() is lazy, i.e. it does not access elements of localTable.
    val result = localTable.groupAndCombineValues("Test", combineFn) as LocalTable<String, Int>

    // Check that when the input collection is initialized, it is safe to access the output
    // elements.
    initialized = true
    assertThat(result.data.toList()).containsExactly("key" to 3)
  }

  @Test
  fun groupByKey_groupsValues() {
    val localTable =
      LocalTable(sequenceOf(Pair("positive", 1), Pair("positive", 10), Pair("negative", -1)))

    val result = localTable.groupByKey("Test")

    assertThat(result.data.toList())
      .containsExactly(Pair("positive", listOf(1, 10)), Pair("negative", listOf(-1)))
  }

  @Test
  fun groupByKey_executesLazy() {
    var initialized = false
    val localTable =
      LocalTable(
        (1..2).asSequence().map {
          require(initialized) { "Not initialized" }
          Pair(it, it)
        }
      )

    // Check that groupByKey() is lazy, i.e. it does not access elements of localTable.
    val result = localTable.groupByKey("Test")

    // Check that when the input collection is initialized, it is safe to access the output
    // elements.
    initialized = true
    assertThat(result.data.toList()).containsExactly(1 to listOf(1), 2 to listOf(2))
  }

  @Test
  fun keys_returnsKeys() {
    val localTable = LocalTable(sequenceOf("key" to "value"))

    val result: LocalCollection<String> = localTable.keys("Test") as LocalCollection<String>

    assertThat(result.data.asIterable()).containsExactly("key")
  }

  @Test
  fun keys_returnsValues() {
    val localTable = LocalTable(sequenceOf("key" to "value"))

    val result: LocalCollection<String> = localTable.values("Test") as LocalCollection<String>

    assertThat(result.data.asIterable()).containsExactly("value")
  }

  @Test
  fun mapValues_appliesMapFn() {
    val localTable = LocalTable(sequenceOf("one" to 1))
    val mapFn: (String, Int) -> String = { k, v -> k + "_" + v.toString() }

    val result: LocalTable<String, String> =
      localTable.mapValues("Test", ENCODER_FACTORY.strings(), mapFn) as LocalTable<String, String>

    assertThat(result.data.asIterable()).containsExactly("one" to "one_1")
  }

  @Test
  fun mapToTable_appliesMapFn() {
    val localTable = LocalTable(sequenceOf("one" to 1))
    val mapFn: (String, Int) -> Pair<Int, String> = { k, v -> Pair(v, k) }

    val result: LocalTable<Int, String> =
      localTable.mapToTable("Test", ENCODER_FACTORY.ints(), ENCODER_FACTORY.strings(), mapFn)
        as LocalTable<Int, String>

    assertThat(result.data.asIterable()).containsExactly(1 to "one")
  }

  @Test
  fun flatMapToTable_appliesMapFn() {
    val localTable = LocalTable(sequenceOf("one" to 1))
    val mapFn: (String, Int) -> Sequence<Pair<Int, String>> = { k, v ->
      sequenceOf(Pair(v, k), Pair(v, k))
    }

    val result: LocalTable<Int, String> =
      localTable.flatMapToTable("Test", ENCODER_FACTORY.ints(), ENCODER_FACTORY.strings(), mapFn)
        as LocalTable<Int, String>

    assertThat(result.data.asIterable()).containsExactly(1 to "one", 1 to "one")
  }

  @Test
  fun filterValues_appliesPredicate() {
    val localTable = LocalTable(sequenceOf("one" to 1, "two" to 2))
    val predicate: (Int) -> Boolean = { v -> v == 1 }

    val result: LocalTable<String, Int> =
      localTable.filterValues("Test", predicate) as LocalTable<String, Int>

    assertThat(result.data.asIterable()).containsExactly("one" to 1)
  }

  @Test
  fun filterKeys_appliesPredicate() {
    val localTable = LocalTable(sequenceOf("one" to 1, "two" to 2, "two" to -2))
    val predicate: (String) -> Boolean = { k -> k == "two" }

    val result = localTable.filterKeys("Test", predicate)

    assertThat(result.data.asIterable()).containsExactly("two" to -2, "two" to 2)
  }

  @Test
  fun filterKeys_keepsAllowedKeys(@TestParameter unbalancedKeys: Boolean) {
    val localTable = LocalTable(sequenceOf("one" to 1, "two" to 2, "three" to 3, "two" to -2))
    val allowedKeys: LocalCollection<String> = LocalCollection(sequenceOf("three", "two", "four"))

    val result: LocalTable<String, Int> =
      localTable.filterKeys("Test", allowedKeys, unbalancedKeys) as LocalTable<String, Int>

    assertThat(result.data.asIterable()).containsExactly("two" to 2, "two" to -2, "three" to 3)
  }

  @Test
  fun flattenWith_flattensCollections() {
    val localTable = LocalTable(sequenceOf("one" to 1))
    val otherLocalTable: LocalTable<String, Int> = LocalTable(sequenceOf("two" to 2))

    val result: LocalTable<String, Int> = localTable.flattenWith("Test", otherLocalTable)

    assertThat(result.data.asIterable()).containsExactly("one" to 1, "two" to 2)
  }

  @Test
  fun samplePerKey_samplesElements() {
    val localTable =
      LocalTable(
        sequenceOf(
          "one" to 1,
          "one" to 2,
          "one" to 3,
          "one" to 4,
          "one" to 5,
          "two" to 6,
          "two" to 7,
          "two" to 8,
          "two" to 9,
          "two" to 10,
          "three" to 11,
          "three" to 12,
        )
      )

    val result: LocalTable<String, Iterable<Int>> = localTable.samplePerKey("Test", 3)

    val resultData = result.data.toList()
    assertThat(resultData.size).isEqualTo(3)
    assertThat(resultData.filter { it.first == "one" }[0].second.toList().size).isEqualTo(3)
    assertThat(resultData.filter { it.first == "two" }[0].second.toList().size).isEqualTo(3)
    assertThat(resultData.filter { it.first == "three" }[0].second.toList().size).isEqualTo(2)
  }
}

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

package com.google.privacy.differentialprivacy.pipelinedp4j.spark

import com.google.common.truth.Truth.assertThat
import com.google.privacy.differentialprivacy.pipelinedp4j.local.LocalCollection
import com.google.privacy.differentialprivacy.pipelinedp4j.local.LocalTable
import com.google.testing.junit.testparameterinjector.TestParameter
import com.google.testing.junit.testparameterinjector.TestParameterInjector
import org.apache.spark.sql.Encoders
import org.junit.ClassRule
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(TestParameterInjector::class)
@Suppress("UNCHECKED_CAST")
class SparkTableTest {
  @Test
  fun keysEncoder_returnsCorrectEncoder() {
    val dataset =
      sparkSession.spark.createDataset(
        listOf(),
        Encoders.kryo(Pair::class.java) as org.apache.spark.sql.Encoder<Pair<String, Int>>,
      )
    val sparkTable = SparkTable(dataset, Encoders.STRING(), Encoders.INT())
    val result = sparkTable.keysEncoder

    assertThat(result).isInstanceOf(SparkEncoder::class.java)
    assertThat(result.encoder).isEqualTo(Encoders.STRING())
  }

  @Test
  fun valuesEncoder_returnsCorrectEncoder() {
    val dataset =
      sparkSession.spark.createDataset(
        listOf(),
        Encoders.kryo(Pair::class.java) as org.apache.spark.sql.Encoder<Pair<String, Int>>,
      )
    val sparkTable = SparkTable(dataset, Encoders.STRING(), Encoders.INT())
    val result = sparkTable.valuesEncoder

    assertThat(result).isInstanceOf(SparkEncoder::class.java)
    assertThat(result.encoder).isEqualTo(Encoders.INT())
  }

  @Test
  fun map_appliesMapFn() {
    val dataset =
      sparkSession.spark.createDataset(
        listOf(Pair(1, 10)),
        Encoders.kryo(Pair::class.java) as org.apache.spark.sql.Encoder<Pair<Int, Int>>,
      )
    val sparkTable = SparkTable(dataset, Encoders.INT(), Encoders.INT())
    val mapFn: (Int, Int) -> String = { k, v -> "${k}_$v" }
    val result = sparkTable.map("Test", sparkEncoderFactory.strings(), mapFn)
    assertThat(result.data.collectAsList()).containsExactly("1_10")
  }

  @Test
  fun groupAndCombineValues_appliesCombiner() {
    val dataset =
      sparkSession.spark.createDataset(
        listOf(
          Pair("positive", 1),
          Pair("positive", 10),
          Pair("negative", -1),
          Pair("negative", -10),
        ),
        Encoders.kryo(Pair::class.java) as org.apache.spark.sql.Encoder<Pair<String, Int>>,
      )
    val sparkTable = SparkTable(dataset, Encoders.STRING(), Encoders.INT())
    val combineFn: (Int, Int) -> Int = { v1, v2 -> v1 + v2 }
    val result = sparkTable.groupAndCombineValues("Test", combineFn)
    assertThat(result.data.collectAsList())
      .containsExactly(Pair("positive", 11), Pair("negative", -11))
  }

  @Test
  fun groupByKey_groupsValues() {
    val dataset =
      sparkSession.spark.createDataset(
        listOf(Pair("positive", 1), Pair("positive", 10), Pair("negative", -1)),
        Encoders.kryo(Pair::class.java) as org.apache.spark.sql.Encoder<Pair<String, Int>>,
      )
    val sparkTable = SparkTable(dataset, Encoders.STRING(), Encoders.INT())
    val result: SparkTable<String, Iterable<Int>> = sparkTable.groupByKey("stageName")
    assertThat(result.data.count()).isEqualTo(2)
    val ans = result.data.collectAsList()
    assertThat(ans).containsExactly(Pair("positive", listOf(1, 10)), Pair("negative", listOf(-1)))
  }

  @Test
  fun keys_returnKeys() {
    val data =
      sparkSession.spark.createDataset(
        listOf(Pair("key", "value")),
        Encoders.kryo(Pair::class.java) as org.apache.spark.sql.Encoder<Pair<String, String>>,
      )
    val sparkTable = SparkTable(data, Encoders.STRING(), Encoders.STRING())
    val result = sparkTable.keys("stageName")
    assertThat(result.data.collectAsList()).containsExactly("key")
  }

  @Test
  fun values_returnsValues() {
    val data =
      sparkSession.spark.createDataset(
        listOf(Pair("key", "value")),
        Encoders.kryo(Pair::class.java) as org.apache.spark.sql.Encoder<Pair<String, String>>,
      )
    val sparkTable = SparkTable(data, Encoders.STRING(), Encoders.STRING())
    val result = sparkTable.values("stageName")
    assertThat(result.data.collectAsList()).containsExactly("value")
  }

  @Test
  fun mapValues_appliesMapFn() {
    val dataset =
      sparkSession.spark.createDataset(
        listOf(Pair("one", 1)),
        Encoders.kryo(Pair::class.java) as org.apache.spark.sql.Encoder<Pair<String, Int>>,
      )
    val sparkTable = SparkTable(dataset, Encoders.STRING(), Encoders.INT())
    val mapFn: (String, Int) -> String = { k, v -> "${k}_$v" }

    val result = sparkTable.mapValues("stageName", sparkEncoderFactory.strings(), mapFn)
    assertThat(result.data.collectAsList()).containsExactly(Pair("one", "one_1"))
  }

  @Test
  fun mapToTable_appliesMapFn() {
    val dataset =
      sparkSession.spark.createDataset(
        listOf(Pair("one", 1)),
        Encoders.kryo(Pair::class.java) as org.apache.spark.sql.Encoder<Pair<String, Int>>,
      )
    val sparkTable = SparkTable(dataset, Encoders.STRING(), Encoders.INT())
    val mapFn: (String, Int) -> Pair<Int, String> = { k, v -> Pair(v, k) }
    val result =
      sparkTable.mapToTable(
        "Test",
        sparkEncoderFactory.ints(),
        sparkEncoderFactory.strings(),
        mapFn,
      )
    assertThat(result.data.collectAsList()).containsExactly(Pair(1, "one"))
  }

  @Test
  fun flatMapToTable_appliesMapFn() {
    val dataset =
      sparkSession.spark.createDataset(
        listOf(Pair("one", 1)),
        Encoders.kryo(Pair::class.java) as org.apache.spark.sql.Encoder<Pair<String, Int>>,
      )
    val sparkTable = SparkTable(dataset, Encoders.STRING(), Encoders.INT())
    val mapFn: (String, Int) -> Sequence<Pair<Int, String>> = { k, v ->
      sequenceOf(Pair(v, k), Pair(v, k))
    }
    val result =
      sparkTable.flatMapToTable(
        "Test",
        sparkEncoderFactory.ints(),
        sparkEncoderFactory.strings(),
        mapFn,
      )
    assertThat(result.data.collectAsList()).containsExactly(Pair(1, "one"), Pair(1, "one"))
  }

  @Test
  fun filterValues_appliesPredicate() {
    val dataset =
      sparkSession.spark.createDataset(
        listOf(Pair("one", 1), Pair("two", 2)),
        Encoders.kryo(Pair::class.java) as org.apache.spark.sql.Encoder<Pair<String, Int>>,
      )
    val sparkTable = SparkTable(dataset, Encoders.STRING(), Encoders.INT())
    val predicate: (Int) -> Boolean = { v -> v == 1 }
    val result = sparkTable.filterValues("Test", predicate)
    assertThat(result.data.collectAsList()).containsExactly(Pair("one", 1))
  }

  @Test
  fun filterKeys_appliesPredicate() {
    val dataset =
      sparkSession.spark.createDataset(
        listOf(Pair("one", 1), Pair("two", 2), Pair("two", -2)),
        Encoders.kryo(Pair::class.java) as org.apache.spark.sql.Encoder<Pair<String, Int>>,
      )
    val sparkTable = SparkTable(dataset, Encoders.STRING(), Encoders.INT())
    val predicate: (String) -> Boolean = { k -> k == "two" }
    val result: SparkTable<String, Int> = sparkTable.filterKeys("Test", predicate)
    assertThat(result.data.collectAsList()).containsExactly(Pair("two", 2), Pair("two", -2))
  }

  @Test
  fun filterKeys_allowedKeysStoredInSparkollection_keepsOnlyAllowedKeys(
    @TestParameter unbalancedKeys: Boolean
  ) {
    val dataset =
      sparkSession.spark.createDataset(
        listOf(Pair("one", 1), Pair("two", 2), Pair("three", 3), Pair("two", -2)),
        Encoders.kryo(Pair::class.java) as org.apache.spark.sql.Encoder<Pair<String, Int>>,
      )
    val sparkTable = SparkTable(dataset, Encoders.STRING(), Encoders.INT())
    val allowedKeysCollection =
      sparkSession.spark.createDataset(listOf("three", "two", "four"), Encoders.STRING())
    val allowedKeysDataset = SparkCollection(allowedKeysCollection)
    val result = sparkTable.filterKeys("stageName", allowedKeysDataset, unbalancedKeys)
    assertThat(result.data.collectAsList())
      .containsExactly(Pair("two", 2), Pair("three", 3), Pair("two", -2))
  }

  @Test
  fun filterKeys_allowedKeysStoredInLocalCollection_keepsOnlyAllowedKeys() {
    val dataset =
      sparkSession.spark.createDataset(
        listOf(Pair("one", 1), Pair("two", 2), Pair("three", 3), Pair("two", -2)),
        Encoders.kryo(Pair::class.java) as org.apache.spark.sql.Encoder<Pair<String, Int>>,
      )
    val sparkTable = SparkTable(dataset, Encoders.STRING(), Encoders.INT())
    val allowedKeys = sequenceOf("three", "two", "four")
    val allowedKeysLocalCollection = LocalCollection(allowedKeys)
    val result = sparkTable.filterKeys("stageName", allowedKeysLocalCollection)
    assertThat(result.data.collectAsList())
      .containsExactly(Pair("two", 2), Pair("three", 3), Pair("two", -2))
  }

  @Test
  fun flattenWith_sparkTable_flattensCollections() {
    val dataset =
      sparkSession.spark.createDataset(
        listOf(Pair("one", 1)),
        Encoders.kryo(Pair::class.java) as org.apache.spark.sql.Encoder<Pair<String, Int>>,
      )
    val sparkTable = SparkTable(dataset, Encoders.STRING(), Encoders.INT())
    val otherSparkDataset =
      sparkSession.spark.createDataset(
        listOf(Pair("two", 2)),
        Encoders.kryo(Pair::class.java) as org.apache.spark.sql.Encoder<Pair<String, Int>>,
      )
    val otherSparkTable = SparkTable(otherSparkDataset, Encoders.STRING(), Encoders.INT())

    val result = sparkTable.flattenWith("stageName", otherSparkTable)

    assertThat(result.data.collectAsList()).containsExactly(Pair("one", 1), Pair("two", 2))
  }

  @Test
  fun flattenWith_localTable_flattensCollections() {
    val dataset =
      sparkSession.spark.createDataset(
        listOf(Pair("one", 1)),
        Encoders.kryo(Pair::class.java) as org.apache.spark.sql.Encoder<Pair<String, Int>>,
      )
    val sparkTable = SparkTable(dataset, Encoders.STRING(), Encoders.INT())
    val otherTable = sequenceOf("two" to 2)
    val otherLocalTable = LocalTable(otherTable)

    val result = sparkTable.flattenWith("stageName", otherLocalTable)

    assertThat(result.data.collectAsList()).containsExactly(Pair("one", 1), Pair("two", 2))
  }

  @Test
  fun samplePerKey_samplesElements() {
    val dataset =
      sparkSession.spark.createDataset(
        listOf(
          Pair("one", 1),
          Pair("one", 2),
          Pair("one", 3),
          Pair("one", 4),
          Pair("one", 5),
          Pair("two", 6),
          Pair("two", 7),
          Pair("two", 8),
          Pair("two", 9),
          Pair("two", 10),
          Pair("three", 11),
          Pair("three", 12),
        ),
        Encoders.kryo(Pair::class.java) as org.apache.spark.sql.Encoder<Pair<String, Int>>,
      )
    val sparkTable = SparkTable(dataset, Encoders.STRING(), Encoders.INT())

    val result: SparkTable<String, Iterable<Int>> = sparkTable.samplePerKey("Test", 3)

    val resultData = result.data.collectAsList()
    assertThat(resultData.size).isEqualTo(3)

    assertThat(resultData.filter { it.first == "one" }[0].second.count()).isEqualTo(3)
    assertThat(resultData.filter { it.first == "two" }[0].second.count()).isEqualTo(3)
    assertThat(resultData.filter { it.first == "three" }[0].second.count()).isEqualTo(2)
  }

  companion object {
    @JvmField @ClassRule val sparkSession = SparkSessionRule()
    private val sparkEncoderFactory = SparkEncoderFactory()
  }
}

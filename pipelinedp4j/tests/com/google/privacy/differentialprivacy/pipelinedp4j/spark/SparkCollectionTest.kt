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
import org.apache.spark.sql.Encoders
import org.junit.ClassRule
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4

@RunWith(JUnit4::class)
class SparkCollectionTest {
  @Test
  fun elementsEncoder_returnsCorrectEncoder() {
    val dataset = sparkSession.spark.createDataset(listOf(), Encoders.INT())
    val sparkCollection = SparkCollection(dataset)
    val result = sparkCollection.elementsEncoder

    assertThat(result).isInstanceOf(SparkEncoder::class.java)
    assertThat(result.encoder).isEqualTo(Encoders.INT())
  }

  @Test
  fun distinct_removesDuplicates() {
    val dataset = sparkSession.spark.createDataset(listOf(1, 2, 1), Encoders.INT())
    val sparkCollection = SparkCollection(dataset)
    val result: SparkCollection<Int> = sparkCollection.distinct("stageName")

    assertThat(result.data.collectAsList()).containsExactly(1, 2)
  }

  @Test
  fun map_appliesMapFn() {
    val dataset = sparkSession.spark.createDataset(listOf(1), Encoders.INT())
    val sparkCollection = SparkCollection(dataset)
    val result: SparkCollection<String> =
      sparkCollection.map("Map Test", sparkEncoderFactory.strings(), { v -> v.toString() })
    assertThat(result.data.collectAsList()).containsExactly("1")
  }

  @Test
  fun keyBy_keysCollection() {
    val dataset = sparkSession.spark.createDataset(listOf(1), Encoders.INT())
    val sparkCollection = SparkCollection(dataset)

    val result: SparkTable<String, Int> =
      sparkCollection.keyBy("Test", sparkEncoderFactory.strings(), { v -> v.toString() })

    assertThat(result.data.collectAsList()).containsExactly(Pair("1", 1))
  }

  @Test
  fun mapToTable_appliesMapFn() {
    val dataset = sparkSession.spark.createDataset(listOf(1), Encoders.INT())
    val sparkCollection = SparkCollection(dataset)

    val result: SparkTable<String, Int> =
      sparkCollection.mapToTable(
        "Test",
        sparkEncoderFactory.strings(),
        sparkEncoderFactory.ints(),
        { v -> Pair(v.toString(), v) },
      )
    assertThat(result.data.collectAsList()).containsExactly(Pair("1", 1))
  }

  companion object {
    @JvmField @ClassRule val sparkSession = SparkSessionRule()
    private val sparkEncoderFactory = SparkEncoderFactory()
  }
}

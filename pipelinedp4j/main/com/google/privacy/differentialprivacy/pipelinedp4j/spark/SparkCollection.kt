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

import com.google.privacy.differentialprivacy.pipelinedp4j.core.Encoder
import com.google.privacy.differentialprivacy.pipelinedp4j.core.FrameworkCollection
import org.apache.spark.api.java.function.MapFunction
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Encoders

/** An implementation of [FrameworkCollection], which runs all operations on Spark. */
class SparkCollection<T>(val data: Dataset<T>) : FrameworkCollection<T> {

  override val elementsEncoder: SparkEncoder<T> = SparkEncoder<T>(data.encoder())

  override fun distinct(stageName: String): SparkCollection<T> {
    return SparkCollection<T>(data.distinct())
  }

  override fun <R> map(
    stageName: String,
    outputType: Encoder<R>,
    mapFn: (T) -> R,
  ): SparkCollection<R> {
    val outputCoder = (outputType as SparkEncoder<R>).encoder
    val transformedData = data.map(MapFunction { mapFn(it) }, outputCoder)
    return SparkCollection(transformedData)
  }

  override fun <K, V> mapToTable(
    stageName: String,
    keyType: Encoder<K>,
    valueType: Encoder<V>,
    mapFn: (T) -> Pair<K, V>,
  ): SparkTable<K, V> {
    val keySparkType = keyType as SparkEncoder<K>
    val valueSparkType = valueType as SparkEncoder<V>
    @Suppress("UNCHECKED_CAST")
    val outputEncoder = Encoders.kryo(Pair::class.java) as org.apache.spark.sql.Encoder<Pair<K, V>>
    val dataset = data.map(MapFunction { mapFn(it) }, outputEncoder)
    return SparkTable(dataset, keySparkType.encoder, valueSparkType.encoder)
  }

  override fun <K> keyBy(
    stageName: String,
    outputType: Encoder<K>,
    keyFn: (T) -> K,
  ): SparkTable<K, T> {
    val valueEncoder = data.encoder()
    val keyEncoder = (outputType as SparkEncoder<K>).encoder
    @Suppress("UNCHECKED_CAST")
    val keyValueEncoder =
      Encoders.kryo(Pair::class.java) as org.apache.spark.sql.Encoder<Pair<K, T>>
    val keyDataset = data.map(MapFunction { t: T -> Pair(keyFn(t), t) }, keyValueEncoder)
    return SparkTable(keyDataset, keyEncoder, valueEncoder)
  }
}

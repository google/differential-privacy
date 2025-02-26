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
import com.google.privacy.differentialprivacy.pipelinedp4j.core.EncoderFactory
import com.google.protobuf.Message
import org.apache.spark.sql.Encoders

/** A serializer and a deserializer for the data types to convert into Spark internal data types. */
class SparkEncoder<T>(val encoder: org.apache.spark.sql.Encoder<T>) : Encoder<T>

class SparkEncoderFactory : EncoderFactory {

  override fun strings(): SparkEncoder<String> {
    return SparkEncoder<String>(Encoders.STRING())
  }

  override fun doubles(): SparkEncoder<Double> {
    return SparkEncoder<Double>(Encoders.DOUBLE())
  }

  override fun ints(): SparkEncoder<Int> {
    return SparkEncoder<Int>(Encoders.INT())
  }

  override fun <T : Any> records(recordClass: Class<T>): SparkEncoder<T> {
    // RecordClass might not be a Java Bean class then kryo encoding is used (it is less optimized).
    val encoder =
      runCatching { Encoders.bean(recordClass) }.getOrElse { Encoders.kryo(recordClass) }
    return SparkEncoder(encoder!!)
  }

  override fun <T : Message> protos(protoClass: Class<T>): SparkEncoder<T> {
    return SparkEncoder<T>(Encoders.kryo(protoClass))
  }

  override fun <T1 : Any, T2 : Any> tuple2sOf(
    first: Encoder<T1>,
    second: Encoder<T2>,
  ): SparkEncoder<Pair<T1, T2>> {
    @Suppress("UNCHECKED_CAST")
    val pairEncoder = Encoders.kryo(Pair::class.java) as org.apache.spark.sql.Encoder<Pair<T1, T2>>
    return SparkEncoder(pairEncoder)
  }

  override fun <T : Any> lists(unusedElementEncoder: Encoder<T>): SparkEncoder<List<T>> {
    @Suppress("UNCHECKED_CAST")
    val listEncoder = Encoders.kryo(List::class.java) as org.apache.spark.sql.Encoder<List<T>>
    return SparkEncoder(listEncoder)
  }
}

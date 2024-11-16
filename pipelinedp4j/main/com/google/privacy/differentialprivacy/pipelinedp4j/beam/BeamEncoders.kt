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

import org.apache.beam.sdk.extensions.protobuf.ProtoCoder
import com.google.privacy.differentialprivacy.pipelinedp4j.core.Encoder
import com.google.privacy.differentialprivacy.pipelinedp4j.core.EncoderFactory
import com.google.protobuf.Message
import java.io.InputStream
import java.io.OutputStream
import kotlin.reflect.KClass
import org.apache.beam.sdk.coders.Coder
import org.apache.beam.sdk.coders.CustomCoder
import org.apache.beam.sdk.coders.DoubleCoder
import org.apache.beam.sdk.coders.StringUtf8Coder
import org.apache.beam.sdk.coders.VarIntCoder
import org.apache.beam.sdk.extensions.avro.coders.AvroCoder

class BeamEncoder<T>(val coder: Coder<T>) : Encoder<T>

class BeamEncoderFactory() : EncoderFactory {
  override fun strings() = BeamEncoder<String>(StringUtf8Coder.of())

  override fun doubles() = BeamEncoder<Double>(DoubleCoder.of())

  override fun ints() = BeamEncoder<Int>(VarIntCoder.of())

  override fun <T : Any> records(recordClass: KClass<T>) =
    BeamEncoder<T>(AvroCoder.of(recordClass.java))

  override fun <T : Message> protos(protoClass: KClass<T>) =
    BeamEncoder<T>(ProtoCoder.of(protoClass.java))

  override fun <T1 : Any, T2 : Any> tuple2sOf(first: Encoder<T1>, second: Encoder<T2>): BeamEncoder<Pair<T1, T2>> =
    BeamEncoder(
      KotlinPairCoder((first as BeamEncoder<T1>).coder, (second as BeamEncoder<T2>).coder)
    )
}

private class KotlinPairCoder<FirstT, SecondT>(
  private val firstCoder: Coder<FirstT>,
  private val secondCoder: Coder<SecondT>,
) : CustomCoder<Pair<FirstT, SecondT>>() {
  override fun encode(value: Pair<FirstT, SecondT>, out: OutputStream) {
    firstCoder.encode(value.first, out)
    secondCoder.encode(value.second, out)
  }

  override fun decode(inStream: InputStream): Pair<FirstT, SecondT> {
    val first = firstCoder.decode(inStream)
    val second = secondCoder.decode(inStream)
    return Pair(first, second)
  }
}

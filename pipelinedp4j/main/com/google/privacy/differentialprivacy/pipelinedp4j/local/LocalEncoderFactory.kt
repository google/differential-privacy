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

import com.google.privacy.differentialprivacy.pipelinedp4j.core.Encoder
import com.google.privacy.differentialprivacy.pipelinedp4j.core.EncoderFactory
import com.google.protobuf.Message

class LocalEncoderFactory() : EncoderFactory {
  // The implementation of local encoders is empty because when the data is being processed
  // locally (in-process), it doesn't need to be serialized.
  override fun strings(): Encoder<String> {
    return object : Encoder<String> {}
  }

  override fun doubles(): Encoder<Double> {
    return object : Encoder<Double> {}
  }

  override fun ints(): Encoder<Int> {
    return object : Encoder<Int> {}
  }

  override fun <T : Any> records(recordClass: Class<T>): Encoder<T> = object : Encoder<T> {}

  override fun <T : Message> protos(protoClass: Class<T>): Encoder<T> = object : Encoder<T> {}

  override fun <T1 : Any, T2 : Any> tuple2sOf(first: Encoder<T1>, second: Encoder<T2>) =
    object : Encoder<Pair<T1, T2>> {}

  override fun <T : Any> lists(elementsEncoder: Encoder<T>) = object : Encoder<List<T>> {}

  fun <T> encoderForArbitraryType(): Encoder<T> = object : Encoder<T> {}
}

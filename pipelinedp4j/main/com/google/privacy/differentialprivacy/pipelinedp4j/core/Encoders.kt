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

package com.google.privacy.differentialprivacy.pipelinedp4j.core

import com.google.protobuf.Message
import kotlin.reflect.KClass

/**
 * A serializer and a deserializer for the data types processed by PipelineDP4j.
 *
 * An [Encoder] converts between regular Kotlin values and encoded byte-string representations
 * stored in a [FrameworkCollection], which are automatically invoked by the rest of the
 * PipelineDP4j system whenever it needs to convert between an in-memory Kotlin object and an
 * externalizable byte-string representation.
 */
interface Encoder<T> {}

/** A factory for [Encoder]s */
interface EncoderFactory {
  /** Returns an [Encoder] for a [String] value, which can be stored in a [FrameworkCollection]. */
  fun strings(): Encoder<String>

  /** Returns an [Encoder] for a double value, which can be stored in a [FrameworkCollection]. */
  fun doubles(): Encoder<Double>

  /** Returns an [Encoder] for an integer value, which can be stored in a [FrameworkCollection]. */
  fun ints(): Encoder<Int>

  /** Encoder for classes. */
  fun <T : Any> records(recordClass: Class<T>): Encoder<T>

  /** Same as [records(Class)] but accepts Kotlin class. */
  fun <T : Any> records(recordClass: KClass<T>) = records(recordClass.java)

  /** Returns an [Encoder] for a protobuf value, which can be stored in a [FrameworkCollection]. */
  fun <T : Message> protos(protoClass: Class<T>): Encoder<T>

  /** Same as [protos(Class)] but accepts Kotlin class. */
  fun <T : Message> protos(protoClass: KClass<T>) = protos(protoClass.java)

  /** Returns an [Encoder] for a pair, which can be stored in a [FrameworkCollection]. */
  fun <T1 : Any, T2 : Any> tuple2sOf(first: Encoder<T1>, second: Encoder<T2>): Encoder<Pair<T1, T2>>

  /** Returns an [Encoder] for a list, which can be stored in a [FrameworkCollection]. */
  fun <T : Any> lists(elementsEncoder: Encoder<T>): Encoder<List<T>>

  /**
   * Returns the most specific [Encoder] for any record type given its [KClass], including primitive
   * types and proto but except pairs.
   *
   * Use it when the record type is not known at compile time and it can be a primitive type or a
   * proto. This method will return the most appropriate (and efficient) [Encoder] for the given
   * type.
   *
   * Note that this method does not work for pairs ([tuple2sOf]) and for any other classes that are
   * parameterized by generic types.
   */
  fun <T : Any> recordsOfUnknownClass(recordClass: Class<T>) =
    when {
      recordClass == String::class.java -> strings()
      recordClass == Double::class.java -> doubles()
      recordClass == Int::class.java -> ints()
      Message::class.java.isAssignableFrom(recordClass) -> {
        @Suppress("UNCHECKED_CAST") protos(recordClass as Class<out Message>)
      }
      else -> records(recordClass)
    }

  /** Same as [recordsOfUnknownClass(Class)] but accepts Kotlin class. */
  fun <T : Any> recordsOfUnknownClass(recordClass: KClass<T>) =
    recordsOfUnknownClass(recordClass.java)
}

/**
 * Inlines the function and the type parameter which allows to use [EncoderFactory.records] without
 * specifying the class.
 */
inline fun <reified T : Any> EncoderFactory.records() = this.records(T::class)

/**
 * Inlines the function and the type parameter which allows to use [EncoderFactory.protos] without
 * specifying the class.
 */
inline fun <reified T : Message> EncoderFactory.protos() = this.protos(T::class)

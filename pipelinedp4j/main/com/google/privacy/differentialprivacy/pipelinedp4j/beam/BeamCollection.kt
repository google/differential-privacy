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

import com.google.privacy.differentialprivacy.pipelinedp4j.core.Encoder
import com.google.privacy.differentialprivacy.pipelinedp4j.core.FrameworkCollection
import com.google.privacy.differentialprivacy.pipelinedp4j.core.StageNameUtils.makeStageNameUnique
import org.apache.beam.sdk.coders.KvCoder
import org.apache.beam.sdk.transforms.Distinct
import org.apache.beam.sdk.transforms.MapElements
import org.apache.beam.sdk.transforms.SerializableFunction
import org.apache.beam.sdk.transforms.WithKeys
import org.apache.beam.sdk.values.KV
import org.apache.beam.sdk.values.PCollection

/** An implementation of [FrameworkCollection], which runs all operations on Beam. */
class BeamCollection<T>(val data: PCollection<T>) : FrameworkCollection<T> {
  override val elementsEncoder: BeamEncoder<T> = BeamEncoder<T>(data.coder)

  /**
   * Removes duplicates from the collection, i.e. makes it a set.
   *
   * @param stageName the name of the Beam stage using this operation.
   * @return a [BeamCollection] with unique elements.
   *
   * THIS is a change
   */
  override fun distinct(stageName: String): BeamCollection<T> =
    BeamCollection<T>(data.apply(stageName.makeStageNameUnique(), Distinct.create()))

  /**
   * Returns a [BeamCollection] consisting of the results of applying the [mapFn] to the elements of
   * this [BeamCollection].
   *
   * @param stageName the name of the Beam stage using this operation.
   * @param outputType the type of the output elements.
   * @param mapFn the function to apply to each element.
   * @return a [BeamCollection] containing the results of applying [mapFn] to the elements of this
   *   collection.
   */
  override fun <R> map(
    stageName: String,
    outputType: Encoder<R>,
    mapFn: (T) -> R,
  ): BeamCollection<R> {
    val outputCoder = (outputType as BeamEncoder<R>).coder
    return BeamCollection<R>(
      data
        .apply(
          stageName.makeStageNameUnique(),
          MapElements.into(outputCoder.encodedTypeDescriptor).via(SerializableFunction(mapFn)),
        )
        .setCoder(outputCoder)
    )
  }

  /**
   * Returns a [BeamTable] consisting of one key-value pair for each element in this
   * [BeamCollection], where the value is the original element from the [BeamCollection], and the
   * key is the result of applying the [keyFn] to that element.
   *
   * @param stageName the name of the Beam stage using this operation.
   * @param outputType the type of the output keys.
   * @param keyFn the function to apply to each element to extract the key.
   * @return a [BeamTable] where each element of this collection is a value, keyed by the result of
   *   [keyFn].
   */
  override fun <K> keyBy(
    stageName: String,
    outputType: Encoder<K>,
    keyFn: (T) -> K,
  ): BeamTable<K, T> {
    val keyCoder = (outputType as BeamEncoder<K>).coder
    return BeamTable<K, T>(
      data
        .apply(
          stageName.makeStageNameUnique(),
          WithKeys.of(SerializableFunction(keyFn)).withKeyType(keyCoder.encodedTypeDescriptor),
        )
        .setCoder(KvCoder.of(keyCoder, data.coder))
    )
  }

  /**
   * Returns a [BeamTable] consisting of the results of applying [mapFn] to the elements of this
   * collection.
   *
   * @param stageName the name of the Beam stage using this operation.
   * @param keyType the type of the output keys.
   * @param valueType the type of the output values.
   * @param mapFn the function to apply to each element to produce key-value pairs.
   * @return a [BeamTable] containing the key-value pairs produced by applying [mapFn] to the
   *   elements of this collection.
   */
  override fun <K, V> mapToTable(
    stageName: String,
    keyType: Encoder<K>,
    valueType: Encoder<V>,
    mapFn: (T) -> Pair<K, V>,
  ): BeamTable<K, V> {
    val keyBeamType = keyType as BeamEncoder<K>
    val valueBeamType = valueType as BeamEncoder<V>
    val outputCoder = KvCoder.of(keyBeamType.coder, valueBeamType.coder)
    val kvMapFn = { x: T -> mapFn(x).toKV() }
    return BeamTable<K, V>(
      data
        .apply(
          stageName.makeStageNameUnique(),
          MapElements.into(outputCoder.encodedTypeDescriptor).via(SerializableFunction(kvMapFn)),
        )
        .setCoder(outputCoder)
    )
  }
}

internal fun <K, V> Pair<K, V>.toKV(): KV<K, V> = KV.of(first, second)

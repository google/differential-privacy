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
import com.google.privacy.differentialprivacy.pipelinedp4j.core.FrameworkCollection
import com.google.privacy.differentialprivacy.pipelinedp4j.core.FrameworkTable

/** An implementation of [FrameworkTable], which runs all operations locally. */
class LocalTable<K, V>(val data: Sequence<Pair<K, V>>) : FrameworkTable<K, V> {
  override val keysEncoder = object : Encoder<K> {}

  override val valuesEncoder = object : Encoder<V> {}

  override fun <R> map(
    stageName: String,
    outputType: Encoder<R>,
    mapFn: (K, V) -> R,
  ): FrameworkCollection<R> = LocalCollection(data.map { mapFn(it.first, it.second) })

  override fun groupAndCombineValues(stageName: String, combFn: (V, V) -> V): FrameworkTable<K, V> {
    return LocalTable(
      groupByKey(stageName).data.map { (key, values) -> key to values.reduce(combFn) }
    )
  }

  override fun groupByKey(stageName: String): LocalTable<K, Iterable<V>> {
    return LocalTable(
      sequence {
        yieldAll(
          data
            .groupBy(keySelector = { kv -> kv.first }, valueTransform = { kv -> kv.second })
            .toList()
        )
      }
    )
  }

  override fun keys(stageName: String): FrameworkCollection<K> =
    LocalCollection(data.map { it.first })

  override fun values(stageName: String): FrameworkCollection<V> =
    LocalCollection(data.map { it.second })

  override fun <VO> mapValues(
    stageName: String,
    outputType: Encoder<VO>,
    mapValuesFn: (K, V) -> VO,
  ): FrameworkTable<K, VO> {
    return LocalTable(data.map { Pair(it.first, mapValuesFn(it.first, it.second)) })
  }

  override fun <KO, VO> mapToTable(
    stageName: String,
    outputKeyType: Encoder<KO>,
    outputValueType: Encoder<VO>,
    mapFn: (K, V) -> Pair<KO, VO>,
  ): FrameworkTable<KO, VO> {
    return LocalTable(data.map { mapFn(it.first, it.second) })
  }

  override fun <KO, VO> flatMapToTable(
    stageName: String,
    keyType: Encoder<KO>,
    valueType: Encoder<VO>,
    mapFn: (K, V) -> Sequence<Pair<KO, VO>>,
  ): FrameworkTable<KO, VO> = LocalTable(data.flatMap { (k, v) -> mapFn(k, v) })

  override fun filterValues(stageName: String, predicate: (V) -> Boolean): FrameworkTable<K, V> =
    LocalTable(data.filter { (_, value) -> predicate(value) })

  override fun filterKeys(stageName: String, predicate: (K) -> Boolean) =
    LocalTable(data.filter { (key, _) -> predicate(key) })

  override fun filterKeys(
    stageName: String,
    allowedKeys: FrameworkCollection<K>,
    unbalancedKeys: Boolean,
  ): FrameworkTable<K, V> {
    val allowedKeysHashSet = (allowedKeys as LocalCollection<K>).data.toCollection(HashSet())
    return filterKeys(stageName) { k -> k in allowedKeysHashSet }
  }

  override fun flattenWith(stageName: String, other: FrameworkTable<K, V>): LocalTable<K, V> {
    val localOther = other as LocalTable<K, V>
    return LocalTable(sequenceOf(data, localOther.data).flatten())
  }

  override fun samplePerKey(stageName: String, count: Int): LocalTable<K, Iterable<V>> {
    return LocalTable(
      groupByKey(stageName).data.map { (k, v) ->
        val elements = v.toList()
        val sampledElements =
          if (elements.size <= count) elements else elements.shuffled().take(count)

        k to sampledElements.toList()
      }
    )
  }
}

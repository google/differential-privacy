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

/** An implementation of [FrameworkCollection], which runs all operations locally. */
class LocalCollection<T>(val data: Sequence<T>) : FrameworkCollection<T> {
  override val elementsEncoder = object : Encoder<T> {}

  override fun distinct(stageName: String) = LocalCollection<T>(data.distinct())

  override fun <R> map(
    stageName: String,
    outputType: Encoder<R>,
    mapFn: (T) -> R,
  ): FrameworkCollection<R> {
    return LocalCollection<R>(data.map(mapFn))
  }

  override fun <K> keyBy(
    stageName: String,
    outputType: Encoder<K>,
    keyFn: (T) -> K,
  ): FrameworkTable<K, T> {
    val tableData: Sequence<Pair<K, T>> = data.map { keyFn(it) to it }
    return LocalTable<K, T>(tableData)
  }

  override fun <K, V> mapToTable(
    stageName: String,
    keyType: Encoder<K>,
    valueType: Encoder<V>,
    mapFn: (T) -> Pair<K, V>,
  ): FrameworkTable<K, V> {
    return LocalTable<K, V>(data.map { mapFn(it) })
  }
}

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

/**
 * An abstraction for a framework-specific collection. The internal PipelineDP4j logic is
 * framework-agnostic and operates on this abstraction.
 */
interface FrameworkCollection<T> {
  /** Encoder of elements type in this collection. */
  val elementsEncoder: Encoder<T>

  /** Removes duplicates from the collection, i.e. makes it a set. */
  fun distinct(stageName: String): FrameworkCollection<T>

  /**
   * Returns a [FrameworkCollection] consisting of the results of applying the [mapFn] to the
   * elements of this [FrameworkCollection].
   */
  fun <R> map(stageName: String, outputType: Encoder<R>, mapFn: (T) -> R): FrameworkCollection<R>

  /**
   * Returns a [FrameworkTable] consisting of one key-value pair for each element in this
   * [FrameworkCollection], where the value is the original element from the [FrameworkCollection],
   * and the key is the result of applying the [keyFn] to that element.
   */
  fun <K> keyBy(stageName: String, outputType: Encoder<K>, keyFn: (T) -> K): FrameworkTable<K, T>

  /**
   * Returns a [FrameworkTable] consisting of the results of applying [mapFn] to the elements of
   * this collection.
   */
  fun <K, V> mapToTable(
    stageName: String,
    keyType: Encoder<K>,
    valueType: Encoder<V>,
    mapFn: (T) -> Pair<K, V>,
  ): FrameworkTable<K, V>
}

// TODO: verify for stage name collisions (especially for Beam) (add tests and ability
// to add suffixes to avoid collisisons).
object StageNameUtils {
  /** Appends name of the next stage to the current stage name. */
  fun String.append(nextStageName: String) = "$this/$nextStageName"
}

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
 * An abstraction for a framework-specific table. The internal PipelineDP4j logic is
 * framework-agnostic and operates on this abstraction.
 */
interface FrameworkTable<K, V> {
  /** Encoder of keys type in this table. */
  val keysEncoder: Encoder<K>

  /** Encoder of values type in this table. */
  val valuesEncoder: Encoder<V>

  /**
   * Returns a [FrameworkCollection] consisting of the results of applying the given [mapFn] to
   * every key-value pair in this table.
   */
  fun <R> map(stageName: String, outputType: Encoder<R>, mapFn: (K, V) -> R): FrameworkCollection<R>

  /**
   * Returns a [FrameworkTable] mapping each distinct key of this table to the result of combining
   * (using the [combFn]) all the values of this table with that key.
   *
   * [combFn] should be a reducer, i.e. it accumulates value starting with the first element and
   * applying operation from left to right to current accumulator value and each element.
   */
  fun groupAndCombineValues(stageName: String, combFn: (V, V) -> V): FrameworkTable<K, V>

  /**
   * Returns a new table mapping each distinct key of this table to a collection of all the values
   * associated with that key in this table.
   */
  fun groupByKey(stageName: String): FrameworkTable<K, Iterable<V>>

  /** Returns a [FrameworkCollection] containing all keys in this table. */
  fun keys(stageName: String): FrameworkCollection<K>

  /** Returns a [FrameworkCollection] containing all values in this table. */
  fun values(stageName: String): FrameworkCollection<V>

  /**
   * Returns a [FrameworkTable] consisting of the results of applying the [mapValuesFn] to every
   * value in this table, leaving the keys unchanged.
   */
  fun <VO> mapValues(
    stageName: String,
    outputType: Encoder<VO>,
    mapValuesFn: (K, V) -> VO,
  ): FrameworkTable<K, VO>

  /**
   * Returns a [FrameworkTable] consisting of the results of each item of the output [Sequence]
   * produced by applying the [mapFn] to each key-value pair of this table.
   */
  fun <KO, VO> flatMapToTable(
    stageName: String,
    keyType: Encoder<KO>,
    valueType: Encoder<VO>,
    mapFn: (K, V) -> Sequence<Pair<KO, VO>>,
  ): FrameworkTable<KO, VO>

  /**
   * Returns a [FrameworkTable] consisting of the results of applying the [mapFn] to every key-value
   * pair in this table.
   */
  fun <KO, VO> mapToTable(
    stageName: String,
    outputKeyType: Encoder<KO>,
    outputValueType: Encoder<VO>,
    mapFn: (K, V) -> Pair<KO, VO>,
  ): FrameworkTable<KO, VO>

  /**
   * Returns a [FrameworkTable] consisting of only the key-value pairs in this table for which the
   * value matches the [predicate].
   */
  fun filterValues(stageName: String, predicate: (V) -> Boolean): FrameworkTable<K, V>

  /**
   * Returns a [FrameworkTable] consisting of only the key-value pairs in this table for which the
   * key matches the predicate.
   */
  fun filterKeys(stageName: String, predicate: (K) -> Boolean): FrameworkTable<K, V>

  /**
   * Returns a [FrameworkTable] consisting of only the key-value pairs in this table for which the
   * keys are in [allowedKeys].
   *
   * @param unbalancedKeys whether the number of values per keys are very different. If true, the
   *   implementation may use a more efficient algorithm.
   */
  fun filterKeys(
    stageName: String,
    allowedKeys: FrameworkCollection<K>,
    unbalancedKeys: Boolean = false,
  ): FrameworkTable<K, V>

  /** Returns a [FrameworkTable] consisting of the key-value pairs in this table and in [other] */
  fun flattenWith(stageName: String, other: FrameworkTable<K, V>): FrameworkTable<K, V>

  /**
   * Samples values without replacement per key. The output table will contain the same keys as this
   * table, each key will appear only once. The number of values per key will be at most [count].
   */
  fun samplePerKey(stageName: String, count: Int): FrameworkTable<K, Iterable<V>>
}

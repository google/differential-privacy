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

import com.google.privacy.differentialprivacy.pipelinedp4j.core.PartitionsBalance.UNBALANCED
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.CompoundAccumulator

/*
 * Filters out all [ContributionWithPrivacyId]s whose [PartitionKey]s are not present in
 * the [publicPartitions] collection.
 */
internal fun <PrivacyIdT : Any, PartitionKeyT : Any> FrameworkCollection<
  ContributionWithPrivacyId<PrivacyIdT, PartitionKeyT>
>
  .dropNonPublicPartitions(
  publicPartitions: FrameworkCollection<PartitionKeyT>,
  partitionKeyEncoder: Encoder<PartitionKeyT>,
  partitionsBalance: PartitionsBalance,
): FrameworkCollection<ContributionWithPrivacyId<PrivacyIdT, PartitionKeyT>> {
  return keyBy("KeyByPartitionKey", partitionKeyEncoder) { it.partitionKey() }
    .filterKeys(
      "DropNonPublicPartition",
      publicPartitions,
      unbalancedKeys = partitionsBalance == UNBALANCED,
    )
    .values("DropPartitionKey")
}

/*
 * Adds all public partitions to the [FrameworkTable] with values equal to
 * [CompoundCombiner#emptyAccumulator].
 *
 * Note that it just extends the table with this data, i.e. even if a public partition is already
 * present in the table it will be added anyway with an empty value. You can think of it as just
 * concatenation of two tables (collections of pairs) where the first collection is the initial data
 * and the second one is a collection of (public_partition, empty_accumulator_value) pairs which
 * includes all public partitions that are passed into this function.
 */
internal fun <PartitionKeyT : Any> FrameworkTable<PartitionKeyT, CompoundAccumulator>
  .insertPublicPartitions(
  publicPartitions: FrameworkCollection<PartitionKeyT>,
  combiner: CompoundCombiner,
  partitionKeyEncoder: Encoder<PartitionKeyT>,
  encoderFactory: EncoderFactory,
): FrameworkTable<PartitionKeyT, CompoundAccumulator> {
  return insertAllKeysWithValues(
    publicPartitions,
    // Some accumulators might require budget to be allocated, therefore we should create empty
    // accumulator only in the function and not outside, because function will be called only when
    // budget is allocated.
    { combiner.emptyAccumulator() },
    partitionKeyEncoder,
    encoderFactory.protos(CompoundAccumulator::class),
  )
}

/*
 * Inserts provided [keys] into the table associated with values produced by
 * [insertionElementProducer].
 */
private fun <K, V> FrameworkTable<K, V>.insertAllKeysWithValues(
  keys: FrameworkCollection<K>,
  insertionElementProducer: (K) -> V,
  keyEncoder: Encoder<K>,
  valueEncoder: Encoder<V>,
): FrameworkTable<K, V> {
  val tableWithInsertedElements =
    keys.mapToTable("AddValuesToKeys", keyEncoder, valueEncoder) {
      it to insertionElementProducer(it)
    }
  return flattenWith("UniteCollectionWithTableWithInsertedElements", tableWithInsertedElements)
}

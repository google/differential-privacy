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
import com.google.privacy.differentialprivacy.pipelinedp4j.core.FrameworkTable
import com.google.privacy.differentialprivacy.pipelinedp4j.core.StageNameUtils.append
import com.google.privacy.differentialprivacy.pipelinedp4j.local.LocalCollection
import com.google.privacy.differentialprivacy.pipelinedp4j.local.LocalTable
import org.apache.beam.sdk.coders.KvCoder
import org.apache.beam.sdk.coders.VoidCoder
import org.apache.beam.sdk.transforms.Combine
import org.apache.beam.sdk.transforms.Create
import org.apache.beam.sdk.transforms.DoFn
import org.apache.beam.sdk.transforms.DoFn.ProcessContext
import org.apache.beam.sdk.transforms.DoFn.ProcessElement
import org.apache.beam.sdk.transforms.Filter
import org.apache.beam.sdk.transforms.Flatten
import org.apache.beam.sdk.transforms.GroupByKey
import org.apache.beam.sdk.transforms.Keys
import org.apache.beam.sdk.transforms.MapElements
import org.apache.beam.sdk.transforms.ParDo
import org.apache.beam.sdk.transforms.Sample
import org.apache.beam.sdk.transforms.SerializableBiFunction
import org.apache.beam.sdk.transforms.SerializableFunction
import org.apache.beam.sdk.transforms.Values
import org.apache.beam.sdk.transforms.join.CoGbkResult
import org.apache.beam.sdk.transforms.join.CoGroupByKey
import org.apache.beam.sdk.transforms.join.KeyedPCollectionTuple
import org.apache.beam.sdk.values.KV
import org.apache.beam.sdk.values.PCollection
import org.apache.beam.sdk.values.PCollectionList

/** An implementation of [FrameworkTable], which runs all operations on Beam. */
class BeamTable<K, V>(val data: PCollection<KV<K, V>>) : FrameworkTable<K, V> {
  private val beamPipeline = data.pipeline
  private val kvCoder = (data.coder as KvCoder<K, V>)

  override val keysEncoder = BeamEncoder<K>(kvCoder.keyCoder)

  override val valuesEncoder = BeamEncoder<V>(kvCoder.valueCoder)

  override fun <R> map(
    stageName: String,
    outputType: Encoder<R>,
    mapFn: (K, V) -> R,
  ): BeamCollection<R> {
    val outputCoder = (outputType as BeamEncoder<R>).coder
    val kvMapFn = { x: KV<K, V> -> mapFn(x.getKey(), x.getValue()) }
    return BeamCollection<R>(
      data
        .apply(
          stageName,
          MapElements.into(outputCoder.encodedTypeDescriptor).via(SerializableFunction(kvMapFn)),
        )
        .setCoder(outputCoder)
    )
  }

  override fun groupAndCombineValues(stageName: String, combFn: (V, V) -> V): BeamTable<K, V> {
    return BeamTable(data.apply(stageName, Combine.perKey(SerializableBiFunction(combFn))))
  }

  override fun groupByKey(stageName: String): BeamTable<K, Iterable<V>> {
    return BeamTable(data.apply(stageName, GroupByKey.create()))
  }

  override fun keys(stageName: String): BeamCollection<K> =
    BeamCollection<K>(data.apply(Keys.create()))

  override fun values(stageName: String): BeamCollection<V> =
    BeamCollection<V>(data.apply(Values.create()))

  override fun <VO> mapValues(
    stageName: String,
    outputType: Encoder<VO>,
    mapValuesFn: (K, V) -> VO,
  ): BeamTable<K, VO> {
    val beamOutputValueType = outputType as BeamEncoder<VO>
    val outputCoder = KvCoder.of(keysEncoder.coder, beamOutputValueType.coder)
    val kvMapFn = { x: KV<K, V> -> KV.of(x.getKey(), mapValuesFn(x.getKey(), x.getValue())) }
    return BeamTable(
      data
        .apply(
          stageName,
          MapElements.into(outputCoder.encodedTypeDescriptor).via(SerializableFunction(kvMapFn)),
        )
        .setCoder(outputCoder)
    )
  }

  override fun <KO, VO> mapToTable(
    stageName: String,
    outputKeyType: Encoder<KO>,
    outputValueType: Encoder<VO>,
    mapFn: (K, V) -> Pair<KO, VO>,
  ): BeamTable<KO, VO> {
    val keyBeamEncoder: BeamEncoder<KO> = outputKeyType as BeamEncoder<KO>
    val valueBeamEncoder: BeamEncoder<VO> = outputValueType as BeamEncoder<VO>
    val outputCoder = KvCoder.of(keyBeamEncoder.coder, valueBeamEncoder.coder)
    val kvMapFn = { x: KV<K, V> -> mapFn(x.getKey(), x.getValue()).toKV() }
    return BeamTable(
      data
        .apply(
          stageName,
          MapElements.into(outputCoder.encodedTypeDescriptor).via(SerializableFunction(kvMapFn)),
        )
        .setCoder(outputCoder)
    )
  }

  override fun <KO, VO> flatMapToTable(
    stageName: String,
    keyType: Encoder<KO>,
    valueType: Encoder<VO>,
    mapFn: (K, V) -> Sequence<Pair<KO, VO>>,
  ): BeamTable<KO, VO> {
    val keyBeamEncoder: BeamEncoder<KO> = keyType as BeamEncoder<KO>
    val valueBeamEncoder: BeamEncoder<VO> = valueType as BeamEncoder<VO>
    return BeamTable(
      data
        .apply(
          stageName,
          ParDo.of(
            object : DoFn<KV<K, V>, KV<KO, VO>>() {
              @ProcessElement
              fun processElement(c: ProcessContext) {
                val kv = c.element()
                val results = mapFn(kv.getKey(), kv.getValue())
                for (result in results) {
                  c.output(KV.of(result.first, result.second))
                }
              }
            }
          ),
        )
        .setCoder(KvCoder.of(keyBeamEncoder.coder, valueBeamEncoder.coder))
    )
  }

  override fun filterValues(stageName: String, predicate: (V) -> Boolean): BeamTable<K, V> {
    val kvPredicate = { x: KV<K, V> -> predicate(x.getValue()) }
    return BeamTable(data.apply(stageName, Filter.by(SerializableFunction(kvPredicate))))
  }

  override fun filterKeys(stageName: String, predicate: (K) -> Boolean): BeamTable<K, V> {
    val kvPredicate = { x: KV<K, V> -> predicate(x.getKey()) }
    return BeamTable(data.apply(stageName, Filter.by(SerializableFunction(kvPredicate))))
  }

  override fun filterKeys(
    stageName: String,
    allowedKeys: FrameworkCollection<K>,
    unbalancedKeys: Boolean,
  ): BeamTable<K, V> {
    return when (allowedKeys) {
      is BeamCollection<K> -> {
        // There is no special optimized implementation for unbalanced keys in Beam.
        filterKeysStoredInBeamCollection(stageName, allowedKeys)
      }
      is LocalCollection<K> -> {
        filterKeysStoredInLocalCollection(stageName, allowedKeys)
      }
      else ->
        throw IllegalArgumentException(
          "Collection is of unsupported backend. Only Beam and local backends are supported, " +
            "the type of the given collection is ${allowedKeys.javaClass}"
        )
    }
  }

  override fun flattenWith(stageName: String, other: FrameworkTable<K, V>): BeamTable<K, V> {
    return when (other) {
      is BeamTable<K, V> -> {
        val collectionsList = PCollectionList.of(this.data).and(other.data)
        BeamTable(collectionsList.apply(stageName, Flatten.pCollections()))
      }
      is LocalTable<K, V> -> {
        flattenWith(stageName, other.toBeamTable(stageName.append("ConvertLocalTableToBeamTable")))
      }
      else ->
        throw IllegalArgumentException(
          "Table is of unsupported backend. Only Beam or local backends are supported, " +
            "the type of the given table is ${other.javaClass}."
        )
    }
  }

  /**
   * Keeps only those table entries whose keys are in [allowedKeys] Beam collection.
   *
   * Filtering is done by joining the table with [allowedKeys] and keeping only those entries that
   * matched with some key in [allowedKeys]. The data that belongs to one key is processed in a
   * single worker, however it does not have to fit in memory. The algorithm does not handle
   * unbalanced keys (i.e. hot partitions) in any specific way.
   */
  private fun filterKeysStoredInBeamCollection(
    stageName: String,
    allowedKeys: BeamCollection<K>,
  ): BeamTable<K, V> {
    val allowedKeysAsTable =
      allowedKeys
        .map(
          stageName.append("ConvertAllowedKeysToTable"),
          BeamEncoder(KvCoder.of(allowedKeys.elementsEncoder.coder, VoidCoder.of())),
          { k -> KV.of(k, null) },
        )
        .data
    val dataTag = "DataTag"
    val allowedKeysTag = "AllowedKeysTag"
    val pCollectionTuple =
      KeyedPCollectionTuple.of(dataTag, data).and(allowedKeysTag, allowedKeysAsTable)
    val joinResult = pCollectionTuple.apply(CoGroupByKey.create())

    val filteredTable =
      joinResult
        .apply(
          stageName.append("FilterForAllowedKeys"),
          ParDo.of(
            object : DoFn<KV<K, CoGbkResult>, KV<K, V>>() {
              @ProcessElement
              fun processElement(c: ProcessContext) {
                val kv = c.element()
                val tableValues = kv.value.getAll<V>(dataTag)
                val keyIsAllowed = kv.value.getAll<Void>(allowedKeysTag).any()
                if (keyIsAllowed) {
                  for (value in tableValues) {
                    c.output(KV.of(kv.key, value))
                  }
                }
              }
            }
          ),
        )
        .setCoder(KvCoder.of(keysEncoder.coder, valuesEncoder.coder))

    return BeamTable(filteredTable)
  }

  /**
   * Keeps only those table entries whose keys are in [allowedKeys] local collection.
   *
   * Filtering is done by converting [allowedKeys] to a HashSet and checking if the key of the entry
   * is in that set.
   *
   * The [HashSet] is sent to workers over the network. It might be a bottleneck in the future if
   * there are clients whose set is large. If such situation occurs, it might be worth thinking how
   * we can send [List] to workers because [List] is smaller in size. However, at the time of
   * writing it seemed not possible to send list and create one hashset per worker. The only
   * solution we found was to send list and in the lambda create hashset from this list but it means
   * that for each key we will create a hashset which is too inefficient and consumes too much
   * memory. There might be a solution with side inputs if we could convert the [allowedKeys]
   * sequence to PCollection, however to do that we need Beam pipeline instance which we don't have
   * access to.
   */
  private fun filterKeysStoredInLocalCollection(
    stageName: String,
    allowedKeys: LocalCollection<K>,
  ): BeamTable<K, V> {
    // TODO: add end2end where public partitions are stored in local collection.
    val allowedKeysHashSet = allowedKeys.data.toHashSet()
    return filterKeys(stageName) { k -> k in allowedKeysHashSet }
  }

  override fun samplePerKey(stageName: String, count: Int): BeamTable<K, Iterable<V>> {
    return BeamTable(data.apply(stageName, Sample.fixedSizePerKey<K, V>(count)))
  }

  private fun LocalTable<K, V>.toBeamTable(stageName: String): BeamTable<K, V> {
    val localInput = data.map { it.toKV() }.toList()
    return BeamTable(beamPipeline.apply(stageName, Create.of(localInput).withCoder(kvCoder)))
  }
}

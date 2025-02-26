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

package com.google.privacy.differentialprivacy.pipelinedp4j.spark

import com.google.privacy.differentialprivacy.pipelinedp4j.core.Encoder
import com.google.privacy.differentialprivacy.pipelinedp4j.core.FrameworkCollection
import com.google.privacy.differentialprivacy.pipelinedp4j.core.FrameworkTable
import com.google.privacy.differentialprivacy.pipelinedp4j.local.LocalCollection
import com.google.privacy.differentialprivacy.pipelinedp4j.local.LocalTable
import kotlin.random.Random
import org.apache.spark.api.java.function.FlatMapFunction
import org.apache.spark.api.java.function.MapFunction
import org.apache.spark.api.java.function.MapGroupsFunction
import org.apache.spark.api.java.function.ReduceFunction
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.row_number
import scala.Tuple2
import scala.Tuple3

/** An implementation of [FrameworkTable], which runs all operations on Spark. */
class SparkTable<K, V>(
  val data: Dataset<Pair<K, V>>,
  val keyEncoder: org.apache.spark.sql.Encoder<K>,
  val valueEncoder: org.apache.spark.sql.Encoder<V>,
) : FrameworkTable<K, V> {
  private val sparkSession = data.sparkSession()
  @Suppress("UNCHECKED_CAST")
  private val keyValueEncoder =
    Encoders.kryo(Pair::class.java) as org.apache.spark.sql.Encoder<Pair<K, V>>

  override val keysEncoder = SparkEncoder<K>(keyEncoder)

  override val valuesEncoder = SparkEncoder<V>(valueEncoder)

  override fun <R> map(
    stageName: String,
    outputType: Encoder<R>,
    mapFn: (K, V) -> R,
  ): SparkCollection<R> {
    val outputEncoder = (outputType as SparkEncoder<R>).encoder
    val transformedData =
      data.map(MapFunction { kv: Pair<K, V> -> mapFn(kv.first, kv.second) }, outputEncoder)
    return SparkCollection(transformedData)
  }

  override fun groupAndCombineValues(stageName: String, combFn: (V, V) -> V): SparkTable<K, V> {
    val dataset =
      data
        .groupByKey(MapFunction { kv: Pair<K, V> -> kv.first }, keyEncoder)
        .reduceGroups(
          ReduceFunction { t1: Pair<K, V>, t2: Pair<K, V> ->
            Pair(t1.first, combFn(t1.second, t2.second))
          }
        )
        .map(MapFunction { it._2 }, keyValueEncoder)
    return SparkTable(dataset, keyEncoder, valueEncoder)
  }

  @Suppress("UNCHECKED_CAST")
  override fun groupByKey(stageName: String): SparkTable<K, Iterable<V>> {
    val itrEncoder =
      Encoders.kryo(Iterable::class.java) as org.apache.spark.sql.Encoder<Iterable<V>>
    val dats =
      data
        .groupByKey(MapFunction { kv: Pair<K, V> -> kv.first }, keyEncoder)
        .mapValues(MapFunction { kv: Pair<K, V> -> kv.second }, valueEncoder)
        .mapGroups(
          MapGroupsFunction { k: K, v: Iterator<V> -> Pair(k, v.asSequence().toList()) },
          Encoders.kryo(Pair::class.java) as org.apache.spark.sql.Encoder<Pair<K, Iterable<V>>>,
        )
    return SparkTable(dats, keyEncoder, itrEncoder)
  }

  override fun keys(stageName: String): SparkCollection<K> {
    return SparkCollection(data.map(MapFunction { kv: Pair<K, V> -> kv.first }, keyEncoder))
  }

  override fun values(stageName: String): SparkCollection<V> {
    return SparkCollection(data.map(MapFunction { kv: Pair<K, V> -> kv.second }, valueEncoder))
  }

  override fun <VO> mapValues(
    stageName: String,
    outputType: Encoder<VO>,
    mapValuesFn: (K, V) -> VO,
  ): SparkTable<K, VO> {
    val valueEncoder = (outputType as SparkEncoder<VO>).encoder
    @Suppress("UNCHECKED_CAST")
    val outputEncoder = Encoders.kryo(Pair::class.java) as org.apache.spark.sql.Encoder<Pair<K, VO>>
    val kvMapFn = { x: Pair<K, V> -> Pair(x.first, mapValuesFn(x.first, x.second)) }
    val transformedData = data.map(MapFunction { kvMapFn(it) }, outputEncoder)
    return SparkTable(transformedData, keyEncoder, valueEncoder)
  }

  override fun <KO, VO> mapToTable(
    stageName: String,
    outputKeyType: Encoder<KO>,
    outputValueType: Encoder<VO>,
    mapFn: (K, V) -> Pair<KO, VO>,
  ): SparkTable<KO, VO> {
    val keySparkEncoder = outputKeyType as SparkEncoder<KO>
    val valueSparkEncoder = outputValueType as SparkEncoder<VO>
    @Suppress("UNCHECKED_CAST")
    val outputEncoder =
      Encoders.kryo(Pair::class.java) as org.apache.spark.sql.Encoder<Pair<KO, VO>>
    val transformedData =
      data.map(MapFunction { kv: Pair<K, V> -> mapFn(kv.first, kv.second) }, outputEncoder)
    return SparkTable(transformedData, keySparkEncoder.encoder, valueSparkEncoder.encoder)
  }

  override fun <KO, VO> flatMapToTable(
    stageName: String,
    keyType: Encoder<KO>,
    valueType: Encoder<VO>,
    mapFn: (K, V) -> Sequence<Pair<KO, VO>>,
  ): SparkTable<KO, VO> {
    val keySparkEncoder = keyType as SparkEncoder<KO>
    val valueSparkEncoder = valueType as SparkEncoder<VO>
    @Suppress("UNCHECKED_CAST")
    val outputEncoder =
      Encoders.kryo(Pair::class.java) as org.apache.spark.sql.Encoder<Pair<KO, VO>>
    val kvMapFn = { x: Pair<K, V> -> mapFn(x.first, x.second) }
    val transformedData = data.flatMap(FlatMapFunction { kvMapFn(it).iterator() }, outputEncoder)
    return SparkTable(transformedData, keySparkEncoder.encoder, valueSparkEncoder.encoder)
  }

  override fun filterValues(stageName: String, predicate: (V) -> Boolean): SparkTable<K, V> {
    val kvPredicate = { x: Pair<K, V> -> predicate(x.second) }
    return SparkTable(data.filter { kv: Pair<K, V> -> kvPredicate(kv) }, keyEncoder, valueEncoder)
  }

  override fun filterKeys(
    stageName: String,
    allowedKeys: FrameworkCollection<K>,
    unbalancedKeys: Boolean,
  ): SparkTable<K, V> {
    return when (allowedKeys) {
      is SparkCollection<K> -> {
        filterKeysStoredInSparkCollection(stageName, allowedKeys)
      }
      is LocalCollection<K> -> {
        filterKeysStoredInLocalCollection(stageName, allowedKeys)
      }
      else ->
        throw IllegalArgumentException(
          "Collection is of unsupported backend. Only Spark and local backends are supported, " +
            "the type of the given collection is ${allowedKeys.javaClass}"
        )
    }
  }

  override fun filterKeys(stageName: String, predicate: (K) -> Boolean): SparkTable<K, V> {
    val kvPredicate = { x: Pair<K, V> -> predicate(x.first) }
    return SparkTable(data.filter { kv: Pair<K, V> -> kvPredicate(kv) }, keyEncoder, valueEncoder)
  }

  override fun flattenWith(stageName: String, other: FrameworkTable<K, V>): SparkTable<K, V> {
    return when (other) {
      is SparkTable<K, V> -> {
        val thisAndOther = data.union(other.data)
        SparkTable(thisAndOther, keyEncoder, valueEncoder)
      }
      is LocalTable<K, V> -> {
        flattenWith(stageName, other.toSparkTable())
      }
      else ->
        throw IllegalArgumentException(
          "Table is of unsupported backend. Only Spark or local backends are supported, " +
            "the type of the given table is ${other.javaClass}."
        )
    }
  }

  /**
   * Keeps only those table entries whose keys are in [allowedKeys] Spark collection.
   *
   * Filtering is done by joining the table with [allowedKeys] and keeping only those entries that
   * matched with some key in [allowedKeys]. The data that belongs to one key is processed in a
   * single worker, however it does not have to fit in memory. The algorithm does not handle
   * unbalanced keys (i.e. hot partitions) in any specific way.
   */
  @Suppress("UNCHECKED_CAST")
  private fun filterKeysStoredInSparkCollection(
    stageName: String,
    allowedKeys: SparkCollection<K>,
  ): SparkTable<K, V> {
    val allowedKeysDataset =
      allowedKeys.data.map(
        MapFunction { k: K -> Pair(k, "") },
        Encoders.kryo(Pair::class.java) as org.apache.spark.sql.Encoder<Pair<K, String>>,
      )

    // Dataset<Pair<>> is converted into Dataset<Tuple2<>> to work with Join in Spark
    // We can check in future if we can work with Pair<> for join
    val tupleDataset =
      data.map(
        MapFunction { kv: Pair<K, V> -> kv.toTuple2() },
        Encoders.tuple(keyEncoder, valueEncoder),
      )
    val allowedKeysTupleDataset =
      allowedKeysDataset.map(
        MapFunction { kv: Pair<K, String> -> kv.toTuple2() },
        Encoders.tuple(keyEncoder, Encoders.STRING()),
      )
    val filteredTupleDataset =
      tupleDataset.joinWith(
        allowedKeysTupleDataset,
        tupleDataset.col("_1").equalTo(allowedKeysTupleDataset.col("_1")),
        "inner",
      )
    val filteredPairDataset =
      filteredTupleDataset.map(
        MapFunction { kv: Tuple2<Tuple2<K, V>, Tuple2<K, String>> -> Pair(kv._1._1, kv._1._2) },
        keyValueEncoder,
      )

    return SparkTable(filteredPairDataset, keyEncoder, valueEncoder)
  }

  /**
   * Keeps only those table entries whose keys are in [allowedKeys] spark collection.
   *
   * Filtering is done by converting [allowedKeys] to a HashSet and checking if the key of the entry
   * is in that set.
   */
  private fun filterKeysStoredInLocalCollection(
    stageName: String,
    allowedKeys: LocalCollection<K>,
  ): SparkTable<K, V> {
    val allowedKeysHashSet = allowedKeys.data.toHashSet()
    return filterKeys(stageName) { k -> k in allowedKeysHashSet }
  }

  /**
   * Randomly samples values per key. The output table will contain same keys as this table, each
   * key will appear only once. The number of values per key will be at most [count]. It uses window
   * partition by function which requires an extra shuffle and sort operation and introduces an
   * extra step to transfer data over network but is a scalable and efficient approach for large
   * dataset.
   */
  @Suppress("UNCHECKED_CAST")
  override fun samplePerKey(stageName: String, count: Int): SparkTable<K, Iterable<V>> {
    val iterableEncoder =
      Encoders.kryo(List::class.java) as org.apache.spark.sql.Encoder<Iterable<V>>
    val outputEncoder =
      Encoders.kryo(Pair::class.java) as org.apache.spark.sql.Encoder<Pair<K, Iterable<V>>>
    val randomValueEncoder = Encoders.tuple(keyEncoder, valueEncoder, Encoders.DOUBLE())
    val rowNumberEncoder = Encoders.tuple(keyEncoder, valueEncoder, Encoders.INT())

    // Generate a random score for each record in dataset
    val randomValueDataset =
      data.map(
        MapFunction { kv: Pair<K, V> -> Tuple3(kv.first, kv.second, Random.nextDouble()) },
        randomValueEncoder,
      )

    // Partition records by key and order them withIn each partition window by the random score and
    // assign a sequential row_number to them
    val windowSpec = Window.partitionBy("_1").orderBy("_3")
    val rowNumberDataset =
      randomValueDataset
        .withColumn("rowNum", row_number().over(windowSpec))
        .select("_1", "_2", "rowNum")
        .`as`(rowNumberEncoder)

    // Filter rows which has row_number <= count
    val sampledDataset =
      rowNumberDataset.filter { withRowNum: Tuple3<K, V, Int> -> withRowNum._3() <= count }

    // group by key and create list of selected values
    val sampledPerKeyData =
      sampledDataset
        .groupByKey(MapFunction { data: Tuple3<K, V, Int> -> data._1() }, keyEncoder)
        .mapValues(MapFunction { kv: Tuple3<K, V, Int> -> kv._2() }, valueEncoder)
        .mapGroups(
          MapGroupsFunction { k: K, v: Iterator<V> ->
            Pair(k, v.asSequence().toList().asIterable())
          },
          outputEncoder,
        )

    return SparkTable(sampledPerKeyData, keyEncoder, iterableEncoder)
  }

  private fun LocalTable<K, V>.toSparkTable(): SparkTable<K, V> {
    val dataset = sparkSession.createDataset(data.toList(), keyValueEncoder)
    return SparkTable(dataset, keyEncoder, valueEncoder)
  }
}

private fun <K, V> Pair<K, V>.toTuple2(): Tuple2<K, V> = Tuple2(first, second)

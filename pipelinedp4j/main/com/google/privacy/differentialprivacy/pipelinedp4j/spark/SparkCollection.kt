package com.google.privacy.differentialprivacy.pipelinedp4j.spark

import com.google.privacy.differentialprivacy.pipelinedp4j.core.Encoder
import com.google.privacy.differentialprivacy.pipelinedp4j.core.FrameworkCollection
import org.apache.spark.api.java.function.MapFunction
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Encoders
import scala.Tuple2

/** An implementation of [FrameworkCollection], which runs all operations on Spark. */
class SparkCollection<T>(val data: Dataset<T>): FrameworkCollection<T>  {

    override val elementsEncoder: SparkEncoder<T> = SparkEncoder<T>(data.encoder())

    override fun distinct(stageName: String): SparkCollection<T> {
        return SparkCollection<T>(data.distinct())
    }

    override fun <R> map(stageName: String, outputType: Encoder<R>, mapFn: (T) -> R): SparkCollection<R> {
        val outputCoder = (outputType as SparkEncoder<R>).encoder
        val transformedData = data.map(MapFunction { mapFn(it) }, outputCoder)
        return SparkCollection(transformedData)
    }

    override fun <K, V> mapToTable(stageName: String, keyType: Encoder<K>, valueType: Encoder<V>, mapFn: (T) -> Pair<K, V>): SparkTable<K, V> {
        val keySparkType = keyType as SparkEncoder<K>
        val valueSparkType = valueType as SparkEncoder<V>
        val outputCoder = Encoders.tuple(keySparkType.encoder, valueSparkType.encoder)
        val kvMapFn = { x: T -> mapFn(x).toTuple2() }
        val dataset = data.map(MapFunction {kvMapFn(it)}, outputCoder)
        return SparkTable(dataset, keySparkType.encoder, valueSparkType.encoder)
    }

    override fun <K> keyBy(stageName: String, outputType: Encoder<K>, keyFn: (T) -> K): SparkTable<K, T> {
        val inputEncoder = data.encoder()
        val outputEncoder = (outputType as SparkEncoder<K>).encoder
        val tupleEncoder = Encoders.tuple(outputEncoder, inputEncoder)
        val keyDataset = data.map(MapFunction { t: T ->  Tuple2(keyFn(t), t)}, tupleEncoder)
        return SparkTable(keyDataset, outputEncoder, inputEncoder)
    }
}

internal fun <K, V> Pair<K, V>.toTuple2(): Tuple2<K, V> = Tuple2(first, second)

package com.google.privacy.differentialprivacy.pipelinedp4j.spark

import com.google.privacy.differentialprivacy.pipelinedp4j.core.Encoder
import com.google.privacy.differentialprivacy.pipelinedp4j.core.FrameworkCollection
import com.google.privacy.differentialprivacy.pipelinedp4j.core.FrameworkTable
import org.apache.spark.api.java.function.MapFunction
import org.apache.spark.sql.Dataset

class SparkCollection<T>(val data: Dataset<T>): FrameworkCollection<T>  {

    override val elementsEncoder: SparkEncoder<T> = SparkEncoder<T>(data.encoder())

    override fun distinct(stageName: String): FrameworkCollection<T> {
        return SparkCollection(data.distinct())
    }

    override fun <K, V> mapToTable(stageName: String, keyType: Encoder<K>, valueType: Encoder<V>, mapFn: (T) -> Pair<K, V>): FrameworkTable<K, V> {
        TODO("Not yet implemented")
    }

    override fun <K> keyBy(stageName: String, outputType: Encoder<K>, keyFn: (T) -> K): FrameworkTable<K, T> {
        TODO("Not yet implemented")
    }

    override fun <R> map(stageName: String, outputType: Encoder<R>, mapFn: (T) -> R): FrameworkCollection<R> {
        val outputCoder = (outputType as SparkEncoder<R>).encoder
        val transformedData = data.map(MapFunction { mapFn(it) }, outputCoder)
        return SparkCollection(transformedData)
    }
}

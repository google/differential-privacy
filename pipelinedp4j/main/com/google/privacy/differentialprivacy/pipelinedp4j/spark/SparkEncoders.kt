package com.google.privacy.differentialprivacy.pipelinedp4j.spark

import com.google.privacy.differentialprivacy.pipelinedp4j.core.Encoder
import com.google.privacy.differentialprivacy.pipelinedp4j.core.EncoderFactory
import com.google.protobuf.Message
import org.apache.spark.sql.Encoders
import kotlin.reflect.KClass


/** A serializer and a deserializer for the data types to convert into Spark internal data types. */
class SparkEncoder<T>(val encoder: org.apache.spark.sql.Encoder<T>) : Encoder<T>

class SparkEncoderFactory: EncoderFactory {

    override fun strings(): SparkEncoder<String> {
        return SparkEncoder<String>(Encoders.STRING())
    }

    override fun doubles(): SparkEncoder<Double> {
        return SparkEncoder<Double>(Encoders.DOUBLE())
    }

    override fun ints(): SparkEncoder<Int> {
        return SparkEncoder<Int>(Encoders.INT())
    }

    override fun <T : Any> records(recordClass: KClass<T>): SparkEncoder<T> {
        return SparkEncoder(Encoders.bean(recordClass.java))
    }

    override fun <T : Message> protos(protoClass: KClass<T>): SparkEncoder<T> {
        return SparkEncoder<T>(Encoders.kryo(protoClass.java))
    }

    override fun <T1 : Any, T2 : Any> tuple2sOf(first: Encoder<T1>, second: Encoder<T2>): SparkEncoder<Pair<T1, T2>> {
        val pairEncoder = Encoders.kryo(Pair::class.java) as org.apache.spark.sql.Encoder<Pair<T1, T2>>
        return SparkEncoder(pairEncoder)
    }
}
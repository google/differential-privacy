package com.google.privacy.differentialprivacy.pipelinedp4j.spark

import com.google.privacy.differentialprivacy.pipelinedp4j.core.Encoder
import com.google.privacy.differentialprivacy.pipelinedp4j.core.EncoderFactory
import com.google.protobuf.Message
import kotlin.reflect.KClass
import org.apache.spark.sql.Encoders


class SparkEncoder<T>(val encoder: org.apache.spark.sql.Encoder<T>) : Encoder<T>

class SparkEncoderFactory(): EncoderFactory {

    override fun strings(): Encoder<String> {
        return SparkEncoder<String>(Encoders.STRING())
    }

    override fun doubles(): Encoder<Double> {
        return SparkEncoder<Double>(Encoders.DOUBLE())
    }

    override fun ints(): Encoder<Int> {
        return SparkEncoder<Int>(Encoders.INT())
    }

    override fun <T : Any> records(recordClass: KClass<T>): Encoder<T> {
        return SparkEncoder(Encoders.bean(recordClass.java))
    }

    override fun <T : Message> protos(protoClass: KClass<T>): Encoder<T> {
        TODO("Not yet implemented")
    }

    override fun <T1 : Any, T2 : Any> tuple2sOf(first: Encoder<T1>, second: Encoder<T2>): Encoder<Pair<T1, T2>> {
        TODO("Not yet implemented")
    }

}


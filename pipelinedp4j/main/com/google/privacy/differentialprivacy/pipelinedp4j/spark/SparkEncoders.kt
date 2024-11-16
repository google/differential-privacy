package com.google.privacy.differentialprivacy.pipelinedp4j.spark

import com.google.privacy.differentialprivacy.pipelinedp4j.core.Encoder
import com.google.privacy.differentialprivacy.pipelinedp4j.core.EncoderFactory
import com.google.protobuf.Message
import org.apache.spark.sql.Encoders
import scala.Tuple2
import kotlin.reflect.KClass

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
//        Pair(first, second).toKV()
//        val firstEncoder = (first as SparkEncoder<T1>).encoder
//        val firstExpression = firstEncoder as ExpressionEncoder<T1>
//        val secondEncoder = (second as SparkEncoder<T2>).encoder
//        val secondExpression = secondEncoder as ExpressionEncoder<T2>
//        val firstObjSerializer = firstExpression.objSerializer()
//        val secondObjSerializer = secondExpression.objSerializer()
//        val firstObjDeserializer = firstExpression.objDeserializer()
//        val secondObjDeserializer = secondExpression.objDeserializer()
//        val tupleEncoder: org.apache.spark.sql.Encoder<Tuple2<T1, T2>> = Encoders.tuple(firstEncoder, secondEncoder)
//
//        val firstClsTag = firstEncoder.clsTag()
//        val secondClsTag = secondEncoder.clsTag()
//        val clsTag = ClassTag.apply<Pair<T1, T2>>(Pair::class.java)
//
//
//        return SparkEncoder<Pair<T1, T2>>(ExpressionEncoder(
//            firstObjSerializer,
//            firstObjDeserializer,
//            clsTag)
//        )
        TODO("Not yet implemented")
    }

//    fun <T1, T2> pairEncoder(
//        encoder1: org.apache.spark.sql.Encoder<T1>,
//        encoder2: org.apache.spark.sql.Encoder<T2>
//    ): org.apache.spark.sql.Encoder<Pair<T1, T2>> {
//        val tupleEncoder = Encoders.tuple(encoder1, encoder2)
//        return Encoders.javaSerialization(Pair::class.java).map(
//            { t: Tuple2<T1, T2> -> Pair(t._1, t._2) },
//            { p: Pair<T1, T2> -> Tuple2(p.first, p.second) },
//            tupleEncoder
//        )
//    }
}

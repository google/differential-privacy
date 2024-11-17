package com.google.privacy.differentialprivacy.pipelinedp4j.spark

import com.google.privacy.differentialprivacy.pipelinedp4j.core.Encoder
import com.google.privacy.differentialprivacy.pipelinedp4j.core.EncoderFactory
import com.google.protobuf.Message
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.types.StructType
import scala.reflect.ClassTag
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

        val sparkEncoder1 = first as SparkEncoder<T1>
        val sparkEncoder2 = second as SparkEncoder<T2>

        return SparkEncoder(SparkPairEncoder(sparkEncoder1.encoder, sparkEncoder2.encoder))

//        val firstEncoder = (first as SparkEncoder<T1>).encoder
//        val secondEncoder = (second as SparkEncoder<T2>).encoder
////        val pair = Pair(firstEncoder, secondEncoder)
////        val encoder = Encoders.bean(pair::class.java)
////        return SparkEncoder(encoder)
////
////        val product = Encoders.product<PairExpressionEncoder<T1, T2>>(PairExpressionEncoder(firstEncoder, secondEncoder))
////
////        Encode
////        val firstEncoderExpression = firstEncoder as ExpressionEncoder<T1>
////        val secondEncoderExpression = secondEncoder as ExpressionEncoder<T2>
////        val serializer = PairExpressionEncoder(firstEncoderExpression, secondEncoderExpression)
////        val tupleEncoder = Encoders.tuple(sparkEncoder1, sparkEncoder2)
////        val tupleExpression = tupleEncoder as ExpressionEncoder<Pair<T1, T2>>
////
//        val cls1 = firstEncoder.clsTag()
//        val cls2 = secondEncoder.clsTag()
//        val pairCls = Pair(cls1, cls2)
//        val pairClassTag = ClassTag.apply<Pair<T1, T2>>(pairCls::class.java)
//
//        val serializer = EncodeUsingSerializer(
//            BoundReference(0, ObjectType(firstEncoder.clsTag().javaClass), true), false)
//        val deserializer =
//            DecodeUsingSerializer<T1>(
//                org.apache.spark.sql.catalyst.expressions.Cast(GetColumnByOrdinal(0, BinaryType), BinaryType),
//                firstEncoder.clsTag(),
//                false)
//        val clsTag = pairClassTag
//        return SparkEncoder(ExpressionEncoder(serializer, deserializer, clsTag))
    }
}

private class SparkPairEncoder<T1, T2>(val encoder1: org.apache.spark.sql.Encoder<T1>, val encoder2: org.apache.spark.sql.Encoder<T2>):
    org.apache.spark.sql.Encoder<Pair<T1, T2>> {

    val tupleEncoder = Encoders.tuple(encoder1, encoder2)
    override fun schema(): StructType {
        return tupleEncoder.schema()
    }

    override fun clsTag(): ClassTag<Pair<T1, T2>> {
        val cls1 = encoder1.clsTag()
        val cls2 = encoder2.clsTag()
        val pair = Pair(cls1, cls2)
        return ClassTag.apply(pair.javaClass)
    }
}


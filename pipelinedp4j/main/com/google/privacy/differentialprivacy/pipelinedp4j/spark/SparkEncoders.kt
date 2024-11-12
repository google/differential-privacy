package com.google.privacy.differentialprivacy.pipelinedp4j.spark

import com.google.privacy.differentialprivacy.pipelinedp4j.core.Encoder
import com.google.privacy.differentialprivacy.pipelinedp4j.core.EncoderFactory
import com.google.protobuf.Message
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.catalyst.expressions.BoundReference
import org.apache.spark.sql.catalyst.expressions.CreateNamedStruct
import org.apache.spark.sql.catalyst.expressions.Expression
import org.apache.spark.sql.catalyst.expressions.Literal
import org.apache.spark.sql.catalyst.expressions.objects.Invoke
import org.apache.spark.sql.catalyst.expressions.objects.NewInstance
import org.apache.spark.sql.types.Metadata
import org.apache.spark.sql.types.ObjectType
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types.StructType
import scala.collection.immutable.Seq
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

    override fun <T1 : Any, T2 : Any> tuple2sOf(first: Encoder<T1>, second: Encoder<T2>): Encoder<Pair<T1, T2>> {
        TODO("Not yet implemented")
    }


}

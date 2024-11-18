package com.google.privacy.differentialprivacy.pipelinedp4j.spark

import com.fasterxml.jackson.databind.ser.SerializerFactory
import com.google.privacy.differentialprivacy.pipelinedp4j.core.Encoder
import com.google.privacy.differentialprivacy.pipelinedp4j.core.EncoderFactory
import com.google.protobuf.Message
import org.apache.spark.serializer.SerializerInstance
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.JavaTypeInference
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.catalyst.expressions.BoundReference
import org.apache.spark.sql.catalyst.expressions.CreateNamedStruct
import org.apache.spark.sql.catalyst.expressions.Expression
import org.apache.spark.sql.catalyst.expressions.JsonTuple
import org.apache.spark.sql.catalyst.expressions.Literal
import org.apache.spark.sql.catalyst.expressions.TransformExpression
import org.apache.spark.sql.catalyst.expressions.codegen.CodegenContext
import org.apache.spark.sql.catalyst.expressions.codegen.ExprCode
import org.apache.spark.sql.catalyst.expressions.objects.NewInstance
import org.apache.spark.sql.types.Metadata
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types.StructType
import scala.Tuple2
import scala.collection.IndexedSeq
import scala.collection.JavaConverters
import scala.reflect.ClassTag
import kotlin.reflect.KClass
import scala.collection.Seq



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
//        val firstEncoder = (first as SparkEncoder<T1>).encoder
//        val secondEncoder = (second as SparkEncoder<T2>).encoder
//        val tupleEncoder = Encoders.tuple(firstEncoder, secondEncoder)
//
//        val cls1 = firstEncoder.clsTag()
//        val cls2 = secondEncoder.clsTag()
//        val pair = Pair(cls1, cls2)
//
//        val exprEncoder = tupleEncoder as ExpressionEncoder<Tuple2<T1, T2>>
//        val serializerExpressions = CreateNamedStruct(exprEncoder.serializer())
//        val deserializerExpressions = exprEncoder.deserializer()
//
//        val expression = ExpressionEncoder(
//            serializerExpressions,
//            deserializerExpressions,
//            ClassTag.apply(pair.javaClass)
//        )
//        return SparkEncoder(expression)
        TODO("Not yet implemented")
    }
}

object PairTupleAdapter {
    // Convert Kotlin Pair to Scala Tuple2
    fun <T1, T2> toScalaTuple(pair: Pair<T1, T2>): Tuple2<T1, T2> {
        return Tuple2(pair.first, pair.second)
    }

    // Convert Scala Tuple2 to Kotlin Pair
    fun <T1, T2> toKotlinPair(tuple: Tuple2<T1, T2>): Pair<T1, T2> {
        return Pair(tuple._1, tuple._2)
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

//object PairEncoder {
//    fun <T1, T2> createPairEncoder(
//        keyEncoder: ExpressionEncoder<T1>,
//        valueEncoder: ExpressionEncoder<T2>
//    ): ExpressionEncoder<Pair<T1, T2>> {
//        // Step 1: Define the schema
//        val schema = StructType(
//            arrayOf(
//                StructField("first", keyEncoder.schema(), false, null),
//                StructField("second", valueEncoder.schema(), false, null)
//            )
//        )
//
//        // Step 2: Define the serializer (Pair<K, V> -> InternalRow)
//        val serializer = CreateNamedStruct(
//            JavaConverters.asScalaBuffer(
//                listOf(keyEncoder.objSerializer(),
//                    valueEncoder.objSerializer())
//                ).toSeq())
//
//        // Step 3: Define the deserializer (InternalRow -> Pair<K, V>)
//        val deserializer = CreateNamedStruct(
//            JavaConverters.asScalaBuffer(
//                listOf(keyEncoder.objDeserializer(),
//                    valueEncoder.objDeserializer())
//            ).toSeq())
//
//
//        val cls1 = keyEncoder.clsTag()
//        val cls2 = valueEncoder.clsTag()
//        val pair = Pair(cls1, cls2)
//        val clsTag = ClassTag.apply<Pair<T1, T2>>(pair.javaClass)
//
//        val objSerializer = JavaTypeInference.serializerFor(pair::class.java)
//        val objDeserializer = JavaTypeInference.deserializerFor(pair::class.java)
////        val d = JavaTypeInference.inferDataType(fir)
//
//        // Step 4: Combine into an ExpressionEncoder
//        return ExpressionEncoder(
//            serializer,
//            PairDeserializer<T1, T2>(keyEncoder.objDeserializer(), valueEncoder.objDeserializer()),
//            clsTag // scala.reflect.classTag<Pair<K, V>>() // Ensures runtime type safety
//        )
//    }
//}
//
//class PairDeserializer<T1, T2>(
//    private val keyDeserializer: Expression,
//    private val valueDeserializer: Expression
//) : Expression() {
//
//    override fun nullable(): Boolean = false
//
//    override fun dataType(): StructType = StructType(
//        listOf(
//            StructField("key", keyDeserializer.dataType(), keyDeserializer.nullable(), Metadata.empty()),
//            StructField("value", valueDeserializer.dataType(), valueDeserializer.nullable(), Metadata.empty())
//        ).toTypedArray()
//    )
//
//    override fun eval(input: InternalRow?): Pair<T1, T2>? {
//        if (input == null) return null
//        val key = keyDeserializer.eval(input) as T1
//        val value = valueDeserializer.eval(input) as T2
//        return Pair(key, value)
//    }
//
//    override fun canEqual(that: Any?): Boolean {
//        TODO("Not yet implemented")
//    }
//
//    override fun productElement(n: Int): Any {
//        TODO("Not yet implemented")
//    }
//
//    override fun productArity(): Int {
//        TODO("Not yet implemented")
//    }
//
//    override fun children(): Seq<Expression> =
//        JavaConverters.asScalaBuffer(listOf(keyDeserializer, valueDeserializer)).toSeq()
//
//    override fun withNewChildrenInternal(newChildren: IndexedSeq<Expression>?): Expression {
//        TODO("Not yet implemented")
//    }
//
//    override fun doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode {
//        throw UnsupportedOperationException("Code generation is not supported for PairDeserializer")
//    }
//}

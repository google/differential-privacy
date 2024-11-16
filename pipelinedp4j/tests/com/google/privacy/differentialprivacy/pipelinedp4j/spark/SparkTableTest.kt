package com.google.privacy.differentialprivacy.pipelinedp4j.spark

import com.google.common.truth.Truth
import com.google.common.truth.Truth.assertThat
import com.google.privacy.differentialprivacy.pipelinedp4j.core.Encoder
import com.google.testing.junit.testparameterinjector.TestParameterInjector
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.SparkSession
import org.junit.AfterClass
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import scala.Tuple2

@RunWith(TestParameterInjector::class)
class SparkTableTest {
    @Test
    fun keysEncoder_returnsCorrectEncoder() {
        val dataset = spark.createDataset(listOf(), Encoders.tuple(Encoders.STRING(), Encoders.INT()))
        val sparkTable = SparkTable(dataset, Encoders.STRING(), Encoders.INT())
        val result = sparkTable.keysEncoder

        assertThat(result).isInstanceOf(SparkEncoder::class.java)
        assertThat(result.encoder).isEqualTo(Encoders.STRING())
    }

    @Test
    fun valuesEncoder_returnsCorrectEncoder() {
        val dataset = spark.createDataset(listOf(), Encoders.tuple(Encoders.STRING(), Encoders.INT()))
        val sparkTable = SparkTable(dataset, Encoders.STRING(), Encoders.INT())
        val result = sparkTable.valuesEncoder

        assertThat(result).isInstanceOf(SparkEncoder::class.java)
        assertThat(result.encoder).isEqualTo(Encoders.INT())
    }

    @Test
    fun map_appliesMapFn() {
        val dataset = spark.createDataset(listOf(Tuple2(1, 10)), Encoders.tuple(Encoders.INT(), Encoders.INT()))
        val sparkTable = SparkTable(dataset, Encoders.INT(), Encoders.INT())
        val mapFn: (Int, Int) -> String = { k, v -> "${k}_$v" }
        val result = sparkTable.map("Test", sparkEncoderFactory.strings(), mapFn)
        assertThat(result.data.collectAsList()).containsExactly("1_10")
    }

    @Test
    fun groupAndCombineValues_appliesCombiner() {
        val dataset = spark.createDataset(listOf(Tuple2("positive", 1),
            Tuple2("positive", 10), Tuple2("negative", -1),
            Tuple2("negative", -10)
        ), Encoders.tuple(Encoders.STRING(), Encoders.INT()))
        val sparkTable = SparkTable(dataset, Encoders.STRING(), Encoders.INT())
        val combineFn: (Int, Int) -> Int = { v1, v2 -> v1 + v2 }
        val result = sparkTable.groupAndCombineValues("Test", combineFn)
        assertThat(result.data.collectAsList()).containsExactly(Tuple2("positive", 11), Tuple2("negative", -11))
    }

    companion object {
        private val sparkEncoderFactory = SparkEncoderFactory()
        private lateinit var spark: SparkSession
        @BeforeClass
        @JvmStatic
        fun setup() {
            try {
                spark = SparkSession.builder()
                    .appName("Kotlin Spark Example")
                    .master("local[*]")
                    .config("spark.driver.bindAddress", "127.0.0.1")
                    .getOrCreate();
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
        @AfterClass
        @JvmStatic
        fun tearDown() {
            // Stop SparkSession after all tests are done
            if (::spark.isInitialized) {
                spark.stop()
            }
        }
    }
}
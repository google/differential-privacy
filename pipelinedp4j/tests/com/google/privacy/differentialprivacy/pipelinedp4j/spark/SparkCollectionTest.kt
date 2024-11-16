package com.google.privacy.differentialprivacy.pipelinedp4j.spark

import com.google.common.truth.Truth
import com.google.common.truth.Truth.assertThat
import org.apache.spark.api.java.function.MapFunction
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.SparkSession
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4
import org.junit.AfterClass
import org.junit.BeforeClass

@RunWith(JUnit4::class)
class SparkCollectionTest {
    @Test
    fun elementsEncoder_returnsCorrectEncoder() {
        val dataset = spark.createDataset(listOf(), Encoders.INT())
        val sparkCollection = SparkCollection(dataset)
        val result = sparkCollection.elementsEncoder

        assertThat(result).isInstanceOf(SparkEncoder::class.java)
        assertThat(result.encoder).isEqualTo(Encoders.INT())
    }

    @Test
    fun distinct_removesDuplicates() {
        val dataset = spark.createDataset(listOf(1, 2, 1), Encoders.INT())
        val sparkCollection = SparkCollection(dataset)
        val result: SparkCollection<Int> = sparkCollection.distinct("stageName")

        assertThat(result.data.collectAsList()).containsExactly(1, 2)
    }

    @Test
    fun map_appliesMapFn() {
        val dataset = spark.createDataset(listOf(1), Encoders.INT())
        val sparkCollection = SparkCollection(dataset)
        val result: SparkCollection<String> = sparkCollection.map("Map Test", sparkEncoderFactory.strings(),
            {v -> v.toString() })
        assertThat(result.data.collectAsList()).containsExactly("1")

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
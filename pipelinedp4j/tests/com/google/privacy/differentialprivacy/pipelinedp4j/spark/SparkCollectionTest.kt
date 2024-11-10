package com.google.privacy.differentialprivacy.pipelinedp4j.spark

import com.google.common.truth.Truth
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.SparkSession
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4
import org.junit.AfterClass
import org.junit.BeforeClass

@RunWith(JUnit4::class)
class SparkCollectionTest {

    companion object {
        private lateinit var spark: SparkSession
        @BeforeClass
        @JvmStatic
        fun setup() {
            try {
                spark = SparkSession.builder()
                        .appName("Kotlin Spark Example")
                        .master("local[*]")
                        .getOrCreate()
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
        @AfterClass
        @JvmStatic
        fun tearDown() {
            // Stop SparkSession after all tests are done
            spark.stop()
        }
    }
    @Test
    fun elementsEncoder_returnsCorrectEncoder() {
        val dataset = spark.createDataset(listOf(1, 2, 3, 4, 5), Encoders.INT())
        val sparkCollection = SparkCollection(dataset)
        val result = sparkCollection.elementsEncoder

        Truth.assertThat(result).isInstanceOf(SparkEncoder::class.java)
        Truth.assertThat(result.encoder).isEqualTo(Encoders.INT())
    }

}
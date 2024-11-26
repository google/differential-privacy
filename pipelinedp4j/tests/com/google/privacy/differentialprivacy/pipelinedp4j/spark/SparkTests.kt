package com.google.privacy.differentialprivacy.pipelinedp4j.spark

import org.apache.spark.sql.SparkSession
import org.junit.rules.ExternalResource
import org.junit.runner.RunWith
import org.junit.runners.Suite

/** Provides a list of JUnit test classes to Bazel. When creating a new test class, add it here. */
@RunWith(Suite::class)
@Suite.SuiteClasses(SparkCollectionTest::class, SparkEncodersTest::class, SparkTableTest::class)
class SparkTests {}

/** Class rule to start and stop spark session once per test class which is equivalent to @BeforeClass and @AfterClass */
class SparkSessionRule : ExternalResource() {
    lateinit var spark: SparkSession

    override fun before() {
        // Create SparkSession once for the entire test class
        spark = SparkSession.builder()
            .appName("Kotlin Spark Example")
            .master("local[*]")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .getOrCreate()
    }

    override fun after() {
        // Stop SparkSession after all tests in the class
        spark.stop()
    }
}

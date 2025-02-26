/*
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.privacy.differentialprivacy.pipelinedp4j.spark

import com.google.common.truth.Truth.assertThat
import com.google.privacy.differentialprivacy.pipelinedp4j.core.ContributionWithPrivacyId
import com.google.privacy.differentialprivacy.pipelinedp4j.core.contributionWithPrivacyId
import com.google.privacy.differentialprivacy.pipelinedp4j.core.encoderOfContributionWithPrivacyId
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.CompoundAccumulator
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.compoundAccumulator
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.meanAccumulator
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.quantilesAccumulator
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.sumAccumulator
import com.google.protobuf.ByteString
import org.apache.spark.sql.Encoders
import org.junit.ClassRule
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4

@RunWith(JUnit4::class)
class SparkEncodersTest {
  @Test
  fun strings_isPossibleToCreateSparkCollectionOfThatType() {
    val input = listOf("a", "b", "c")
    val inputCoder = sparkEncoderFactory.strings().encoder

    val dataset = sparkSession.spark.createDataset(input, inputCoder)

    assertThat(dataset.collectAsList()).containsExactlyElementsIn(input)
  }

  @Test
  fun doubles_isPossibleToCreateSparkCollectionOfThatType() {
    val input = listOf(-1.2, 0.0, 2.1)
    val inputCoder = sparkEncoderFactory.doubles().encoder

    val dataset = sparkSession.spark.createDataset(input, inputCoder)

    assertThat(dataset.collectAsList()).containsExactlyElementsIn(input)
  }

  @Test
  fun ints_isPossibleToCreateSparkCollectionOfThatType() {
    val input = listOf(-1, 0, 1)
    val inputCoder = sparkEncoderFactory.ints().encoder

    val dataset = sparkSession.spark.createDataset(input, inputCoder)

    assertThat(dataset.collectAsList()).containsExactlyElementsIn(input)
  }

  @Test
  fun records_isPossibleToCreateSparkCollectionOfThatType() {
    val input = listOf(TestRecord("privacyId1", 1.0, -1), TestRecord("privacyId2", 2.0, 2))
    val inputCoder =
      (sparkEncoderFactory.records(TestRecord::class) as SparkEncoder<TestRecord>).encoder

    val dataset = sparkSession.spark.createDataset(input, inputCoder)

    assertThat(dataset.collectAsList()).containsExactlyElementsIn(input)
  }

  @Test
  fun protos_isPossibleToSparkCollectionOfThatType() {
    val input =
      listOf(
        compoundAccumulator {
          sumAccumulator = sumAccumulator { sum = -123.0 }
          meanAccumulator = meanAccumulator {
            count = 12
            normalizedSum = -1.543
          }
          quantilesAccumulator = quantilesAccumulator {
            serializedQuantilesSummary =
              ByteString.copyFrom(byteArrayOf(0x48, 0x65, 0x6c, 0x6c, 0x6f))
          }
        },
        compoundAccumulator {},
      )
    val inputCoder =
      (sparkEncoderFactory.protos(CompoundAccumulator::class) as SparkEncoder<CompoundAccumulator>)
        .encoder

    val dataset = sparkSession.spark.createDataset(input, inputCoder)

    assertThat(dataset.collectAsList()).containsExactlyElementsIn(input)
  }

  @Test
  fun tuple2sOf_isPossibleToCreateSparkCollectionOfThatType() {
    val input = listOf("pid1" to 1, "pid1" to 1, "pid1" to -2, "pid2" to 3)
    val inputEncoder =
      sparkEncoderFactory
        .tuple2sOf(sparkEncoderFactory.strings(), sparkEncoderFactory.ints())
        .encoder

    val dataset = sparkSession.spark.createDataset(input, inputEncoder)

    assertThat(dataset.collectAsList()).containsExactlyElementsIn(input)
  }

  @Test
  fun lists_isPossibleToCreateSparkCollectionOfThatType() {
    val input = listOf(listOf("pid1", "pid1"), listOf("pid1", "pid2"))
    val inputEncoder = sparkEncoderFactory.lists(sparkEncoderFactory.strings()).encoder

    val dataset = sparkSession.spark.createDataset(input, inputEncoder)

    assertThat(dataset.collectAsList()).containsExactlyElementsIn(input)
  }

  @Test
  fun tuple2sOf_tuple2sOf_isPossibleToCreateSparkCollectionOfThatType() {
    val input =
      listOf(
        Pair("pid1", 1) to "pid1",
        Pair("pid1", 1) to "pid1",
        Pair("pid1", -2) to "pid1",
        Pair("pid2", 3) to "pid2",
      )
    val inputEncoder =
      sparkEncoderFactory
        .tuple2sOf(
          sparkEncoderFactory.tuple2sOf(sparkEncoderFactory.strings(), sparkEncoderFactory.ints()),
          sparkEncoderFactory.strings(),
        )
        .encoder

    val dataset = sparkSession.spark.createDataset(input, inputEncoder)

    assertThat(dataset.collectAsList()).containsExactlyElementsIn(input)
  }

  @Test
  fun contributionWithPrivacyIdOf_isPossibleToCreateSparkCollectionOfThatType() {
    val input =
      listOf(
        contributionWithPrivacyId("privacyId1", "partitionKey1", -1.0),
        contributionWithPrivacyId("privacyId2", "partitionKey1", 0.0),
        contributionWithPrivacyId("privacyId1", "partitionKey2", 1.0),
        contributionWithPrivacyId("privacyId3", "partitionKey3", 1.2345),
      )
    val inputCoder =
      (encoderOfContributionWithPrivacyId(
          sparkEncoderFactory.strings(),
          sparkEncoderFactory.strings(),
          sparkEncoderFactory,
        )
          as SparkEncoder<ContributionWithPrivacyId<String, String>>)
        .encoder

    val dataset = sparkSession.spark.createDataset(input, inputCoder)

    assertThat(dataset.collectAsList()).containsExactlyElementsIn(input)
  }

  @Test
  fun recordsOfUnknownClass_string_createsEncoderWithSparkStringEncoder() {
    @Suppress("UNCHECKED_CAST")
    val encoder = sparkEncoderFactory.recordsOfUnknownClass(String::class) as SparkEncoder<String>

    assertThat(encoder.encoder).isEqualTo(Encoders.STRING())
  }

  @Test
  fun recordsOfUnknownClass_double_createsEncoderWithSparkDoubleEncoder() {
    @Suppress("UNCHECKED_CAST")
    val encoder = sparkEncoderFactory.recordsOfUnknownClass(Double::class) as SparkEncoder<Double>

    assertThat(encoder.encoder).isEqualTo(Encoders.DOUBLE())
  }

  @Test
  fun recordsOfUnknownClass_int_createsEncoderWithSparkIntEncoder() {
    @Suppress("UNCHECKED_CAST")
    val encoder = sparkEncoderFactory.recordsOfUnknownClass(Int::class) as SparkEncoder<Int>

    assertThat(encoder.encoder).isEqualTo(Encoders.INT())
  }

  @Test
  fun recordsOfUnknownClass_kotlinClass_createsEncoderWithSparkBeanEncoder() {
    @Suppress("UNCHECKED_CAST")
    val encoder =
      sparkEncoderFactory.recordsOfUnknownClass(TestRecord::class) as SparkEncoder<TestRecord>

    assertThat(encoder.encoder).isEqualTo(Encoders.bean(TestRecord::class.java))
  }

  @Test
  fun recordsOfUnknownClass_proto_createsEncoderWithSparkKryoEncoder() {
    @Suppress("UNCHECKED_CAST")
    val encoder =
      sparkEncoderFactory.recordsOfUnknownClass(CompoundAccumulator::class)
        as SparkEncoder<CompoundAccumulator>

    assertThat(encoder.encoder).isEqualTo(Encoders.kryo(CompoundAccumulator::class.java))
  }

  data class TestRecord(var string: String = "", var double: Double = 0.0, var int: Int = 0)

  companion object {
    @JvmField @ClassRule val sparkSession = SparkSessionRule()
    private val sparkEncoderFactory = SparkEncoderFactory()
  }
}

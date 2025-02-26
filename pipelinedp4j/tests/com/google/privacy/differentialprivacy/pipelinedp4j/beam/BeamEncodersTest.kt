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

package com.google.privacy.differentialprivacy.pipelinedp4j.beam

import org.apache.beam.sdk.extensions.protobuf.ProtoCoder
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
import org.apache.beam.sdk.coders.DoubleCoder
import org.apache.beam.sdk.coders.StringUtf8Coder
import org.apache.beam.sdk.coders.VarIntCoder
import org.apache.beam.sdk.extensions.avro.coders.AvroCoder
import org.apache.beam.sdk.testing.PAssert
import org.apache.beam.sdk.testing.TestPipeline
import org.apache.beam.sdk.transforms.Create
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4

@RunWith(JUnit4::class)
class BeamEncodersTest {
  @get:Rule val testPipeline: TestPipeline = TestPipeline.create()

  @Test
  fun strings_isPossibleToCreateBeamPCollectionOfThatType() {
    val input = listOf("a", "b", "c")
    val inputCoder = beamEncoderFactory.strings().coder

    val pCollection = testPipeline.apply(Create.of(input).withCoder(inputCoder))

    PAssert.that(pCollection).containsInAnyOrder(input)

    testPipeline.run().waitUntilFinish()
  }

  @Test
  fun doubles_isPossibleToCreateBeamPCollectionOfThatType() {
    val input = listOf(-1.2, 0.0, 2.1)
    val inputCoder = beamEncoderFactory.doubles().coder

    val pCollection = testPipeline.apply(Create.of(input).withCoder(inputCoder))

    PAssert.that(pCollection).containsInAnyOrder(input)

    testPipeline.run().waitUntilFinish()
  }

  @Test
  fun ints_isPossibleToCreateBeamPCollectionOfThatType() {
    val input = listOf(-1, 0, 1)
    val inputCoder = beamEncoderFactory.ints().coder

    val pCollection = testPipeline.apply(Create.of(input).withCoder(inputCoder))

    PAssert.that(pCollection).containsInAnyOrder(input)

    testPipeline.run().waitUntilFinish()
  }

  @Test
  fun records_isPossibleToCreateBeamPCollectionOfThatType() {
    val input = listOf(TestRecord("privacyId1", 1.0, -1), TestRecord("privacyId2", 2.0, 2))
    val inputCoder =
      (beamEncoderFactory.records(TestRecord::class) as BeamEncoder<TestRecord>).coder

    val pCollection = testPipeline.apply(Create.of(input).withCoder(inputCoder))

    PAssert.that(pCollection).containsInAnyOrder(input)

    testPipeline.run().waitUntilFinish()
  }

  @Test
  fun protos_isPossibleToCreateBeamPCollectionOfThatType() {
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
      (beamEncoderFactory.protos(CompoundAccumulator::class) as BeamEncoder<CompoundAccumulator>)
        .coder

    val pCollection = testPipeline.apply(Create.of(input).withCoder(inputCoder))

    PAssert.that(pCollection).containsInAnyOrder(input)

    testPipeline.run().waitUntilFinish()
  }

  @Test
  fun tuple2sOf_isPossibleToCreateBeamPCollectionOfThatType() {
    val input = listOf("pid1" to 1, "pid1" to 1, "pid1" to -2, "pid2" to 3)
    val inputCoder =
      beamEncoderFactory.tuple2sOf(beamEncoderFactory.strings(), beamEncoderFactory.ints()).coder

    val pCollection = testPipeline.apply(Create.of(input).withCoder(inputCoder))

    PAssert.that(pCollection).containsInAnyOrder(input)

    testPipeline.run().waitUntilFinish()
  }

  @Test
  fun lists_isPossibleToCreateBeamPCollectionOfThatType() {
    val input = listOf(listOf("pid1", "pid1"), listOf("pid1", "pid2"))
    val inputCoder = beamEncoderFactory.lists(beamEncoderFactory.strings()).coder

    val pCollection = testPipeline.apply(Create.of(input).withCoder(inputCoder))

    PAssert.that(pCollection).containsInAnyOrder(input)

    testPipeline.run().waitUntilFinish()
  }

  @Test
  fun contributionWithPrivacyIdOf_isPossibleToCreateBeamPCollectionOfThatType() {
    val input =
      listOf(
        contributionWithPrivacyId("privacyId1", "partitionKey1", -1.0),
        contributionWithPrivacyId("privacyId2", "partitionKey1", 0.0),
        contributionWithPrivacyId("privacyId1", "partitionKey2", 1.0),
        contributionWithPrivacyId("privacyId3", "partitionKey3", 1.2345),
      )
    val inputCoder =
      (encoderOfContributionWithPrivacyId(
          beamEncoderFactory.strings(),
          beamEncoderFactory.strings(),
          beamEncoderFactory,
        )
          as BeamEncoder<ContributionWithPrivacyId<String, String>>)
        .coder

    val pCollection = testPipeline.apply(Create.of(input).withCoder(inputCoder))

    PAssert.that(pCollection).containsInAnyOrder(input)

    testPipeline.run().waitUntilFinish()
  }

  @Test
  fun recordsOfUnknownClass_string_createsEncoderWithStringCoder() {
    @Suppress("UNCHECKED_CAST")
    val encoder = beamEncoderFactory.recordsOfUnknownClass(String::class) as BeamEncoder<String>

    assertThat(encoder.coder).isInstanceOf(StringUtf8Coder::class.java)
  }

  @Test
  fun recordsOfUnknownClass_double_createsEncoderWithDoubleCoder() {
    @Suppress("UNCHECKED_CAST")
    val encoder = beamEncoderFactory.recordsOfUnknownClass(Double::class) as BeamEncoder<Double>

    assertThat(encoder.coder).isInstanceOf(DoubleCoder::class.java)
  }

  @Test
  fun recordsOfUnknownClass_int_createsEncoderWithIntCoder() {
    @Suppress("UNCHECKED_CAST")
    val encoder = beamEncoderFactory.recordsOfUnknownClass(Int::class) as BeamEncoder<Int>

    assertThat(encoder.coder).isInstanceOf(VarIntCoder::class.java)
  }

  @Test
  fun recordsOfUnknownClass_kotlinClass_createsEncoderWithAvroCoder() {
    @Suppress("UNCHECKED_CAST")
    val encoder =
      beamEncoderFactory.recordsOfUnknownClass(TestRecord::class) as BeamEncoder<TestRecord>

    assertThat(encoder.coder).isInstanceOf(AvroCoder::class.java)
  }

  @Test
  fun recordsOfUnknownClass_proto_createsEncoderWithProtoCoder() {
    @Suppress("UNCHECKED_CAST")
    val encoder =
      beamEncoderFactory.recordsOfUnknownClass(CompoundAccumulator::class)
        as BeamEncoder<CompoundAccumulator>

    assertThat(encoder.coder).isInstanceOf(ProtoCoder::class.java)
  }

  private data class TestRecord(val string: String, val double: Double, val int: Int) {
    // Required for Beam serialization.
    private constructor() : this("", 0.0, 0)
  }

  companion object {
    private val beamEncoderFactory = BeamEncoderFactory()
  }
}

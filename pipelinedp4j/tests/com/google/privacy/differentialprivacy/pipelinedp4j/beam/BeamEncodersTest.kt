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

import com.google.privacy.differentialprivacy.pipelinedp4j.core.ContributionWithPrivacyId
import com.google.privacy.differentialprivacy.pipelinedp4j.core.contributionWithPrivacyId
import com.google.privacy.differentialprivacy.pipelinedp4j.core.encoderOfContributionWithPrivacyId
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.CompoundAccumulator
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.compoundAccumulator
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.meanAccumulator
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.quantilesAccumulator
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.sumAccumulator
import com.google.protobuf.ByteString
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
    val inputCoder = beamEncoderFactory.protos(CompoundAccumulator::class).coder

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

  companion object {
    private val beamEncoderFactory = BeamEncoderFactory()
  }
}

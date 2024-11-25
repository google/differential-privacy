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

package com.google.privacy.differentialprivacy.pipelinedp4j.api

import com.google.privacy.differentialprivacy.pipelinedp4j.beam.BeamCollection
import com.google.privacy.differentialprivacy.pipelinedp4j.beam.BeamEncoderFactory
import com.google.privacy.differentialprivacy.pipelinedp4j.core.EncoderFactory
import com.google.privacy.differentialprivacy.pipelinedp4j.core.FrameworkCollection
import com.google.privacy.differentialprivacy.pipelinedp4j.local.LocalCollection
import com.google.privacy.differentialprivacy.pipelinedp4j.local.LocalEncoderFactory
import com.google.privacy.differentialprivacy.pipelinedp4j.spark.SparkCollection
import com.google.privacy.differentialprivacy.pipelinedp4j.spark.SparkEncoderFactory
import org.apache.spark.sql.Dataset
import org.apache.beam.sdk.values.PCollection as BeamPCollection

/**
 * An internal interface to represent an arbitrary collection that is supported by PipelineDP4j.
 *
 * This interface is used to represent collections in a generic way, so that we can use the same
 * code for different collection types. Essentially this is just a helper interface.
 *
 * @param T the type of the elements in the collection.
 */
sealed interface PipelineDpCollection<T> {
  val encoderFactory: EncoderFactory

  fun toFrameworkCollection(): FrameworkCollection<T>
}

/** Beam PCollection. */
internal data class BeamPipelineDpCollection<T>(val data: BeamPCollection<T>) :
  PipelineDpCollection<T> {
  override val encoderFactory = BeamEncoderFactory()

  override fun toFrameworkCollection() = BeamCollection(data)
}

/** Local collection represented as a Kotlin sequence. */
internal data class LocalPipelineDpCollection<T>(val data: Sequence<T>) : PipelineDpCollection<T> {
  override val encoderFactory = LocalEncoderFactory()

  override fun toFrameworkCollection() = LocalCollection<T>(data)
}

/** Spark Collection represented as a Spark Dataset. */
internal data class SparkPipelineDpCollection<T>(val data: Dataset<T>) : PipelineDpCollection<T> {
  override val encoderFactory = SparkEncoderFactory()

  override fun toFrameworkCollection() = SparkCollection<T>(data)
}

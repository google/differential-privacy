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

package com.google.privacy.differentialprivacy.pipelinedp4j.core

import java.io.Serializable

/** An interface of a function that is also serializable. */
fun interface SerializableFunction<S, T> : (S) -> T, Serializable {}

/**
 * An extractor for a feature, which can be a single value (scalar) or a list of values (vector).
 *
 * @param T the type of the input data row.
 * @param featureId the identifier of the feature.
 * @param extractor a function that extracts a [List] of [Double]s from the input.
 */
data class FeatureValuesExtractor<T>(
  val featureId: String,
  private val extractor: SerializableFunction<T, List<Double>>,
) : Serializable {
  /** Extracts the feature value(s) from the input row. */
  fun extract(input: T): List<Double> = extractor(input)
}

/** An extractor of [MultiFeatureContribution] from the row of the input data being anonymized. */
class DataExtractors<T, PrivacyIdT : Any, PartitionKeyT : Any>
@PublishedApi
internal constructor(
  val contributionExtractor:
    SerializableFunction<T, MultiFeatureContribution<PrivacyIdT, PartitionKeyT>>,
  val privacyIdEncoder: Encoder<PrivacyIdT>,
  val partitionKeyEncoder: Encoder<PartitionKeyT>,
  val hasValueExtractor: Boolean,
) {
  companion object {
    /**
     * Constructs a [DataExtractors] that uses the provided functions to extract a
     * [MultiFeatureContribution] from the input data row.
     *
     * This version is useful when the user contribution consists of one or more features, where
     * each feature can be a single value (scalar) or multiple values (vector).
     */
    inline fun <T, PrivacyIdT : Any, PartitionKeyT : Any> from(
      crossinline privacyIdExtractor: (T) -> PrivacyIdT,
      privacyIdEncoder: Encoder<PrivacyIdT>,
      crossinline partitionKeyExtractor: (T) -> PartitionKeyT,
      partitionKeyEncoder: Encoder<PartitionKeyT>,
      valuesExtractors: List<FeatureValuesExtractor<T>>,
    ) =
      DataExtractors<T, PrivacyIdT, PartitionKeyT>(
        {
          multiFeatureContribution(
            privacyId = privacyIdExtractor(it),
            partitionKey = partitionKeyExtractor(it),
            values =
              valuesExtractors.map { extractor ->
                PerFeatureValues(extractor.featureId, extractor.extract(it))
              },
          )
        },
        privacyIdEncoder = privacyIdEncoder,
        partitionKeyEncoder = partitionKeyEncoder,
        hasValueExtractor = true,
      )

    /**
     * Constructs a [DataExtractors] that uses the provided functions to extract a privacy id and a
     * partition key into [MultiFeatureContribution] from the input data row.
     */
    inline fun <T, PrivacyIdT : Any, PartitionKeyT : Any> from(
      crossinline privacyIdExtractor: (T) -> PrivacyIdT,
      privacyIdEncoder: Encoder<PrivacyIdT>,
      crossinline partitionKeyExtractor: (T) -> PartitionKeyT,
      partitionKeyEncoder: Encoder<PartitionKeyT>,
    ) =
      DataExtractors<T, PrivacyIdT, PartitionKeyT>(
        {
          multiFeatureContribution(
            privacyId = privacyIdExtractor(it),
            partitionKey = partitionKeyExtractor(it),
            value = PerFeatureValues(featureId = "", values = listOf(0.0)),
          )
        },
        privacyIdEncoder = privacyIdEncoder,
        partitionKeyEncoder = partitionKeyEncoder,
        hasValueExtractor = false,
      )
  }
}

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

/** An extractor of [ContributionWithPrivacyId] from the row of the input data being anonymized. */
class DataExtractors<T, PrivacyIdT : Any, PartitionKeyT : Any>
@PublishedApi
internal constructor(
  val contributionExtractor:
    SerializableFunction<T, ContributionWithPrivacyId<PrivacyIdT, PartitionKeyT>>,
  val privacyIdEncoder: Encoder<PrivacyIdT>,
  val partitionKeyEncoder: Encoder<PartitionKeyT>,
  val hasValueExtractor: Boolean,
) {
  companion object {
    /**
     * Constructs a [DataExtractors] that uses the provided functions to extract a
     * [ContributionWithPrivacyId] from the input data row.
     */
    inline fun <T, PrivacyIdT : Any, PartitionKeyT : Any> from(
      crossinline privacyIdExtractor: (T) -> PrivacyIdT,
      privacyIdEncoder: Encoder<PrivacyIdT>,
      crossinline partitionKeyExtractor: (T) -> PartitionKeyT,
      partitionKeyEncoder: Encoder<PartitionKeyT>,
      crossinline valueExtractor: (T) -> Double,
    ) =
      DataExtractors<T, PrivacyIdT, PartitionKeyT>(
        {
          contributionWithPrivacyId(
            privacyId = privacyIdExtractor(it),
            partitionKey = partitionKeyExtractor(it),
            value = valueExtractor(it),
          )
        },
        privacyIdEncoder = privacyIdEncoder,
        partitionKeyEncoder = partitionKeyEncoder,
        hasValueExtractor = true,
      )

    /**
     * Constructs a [DataExtractors] that uses the provided functions to extract a
     * [ContributionWithPrivacyId] from the input data row.
     *
     * This version is useful when the user contribution consists of multiple values, e.g. when
     * claculating vector sum.
     */
    inline fun <T, PrivacyIdT : Any, PartitionKeyT : Any> forVectorFrom(
      crossinline privacyIdExtractor: (T) -> PrivacyIdT,
      privacyIdEncoder: Encoder<PrivacyIdT>,
      crossinline partitionKeyExtractor: (T) -> PartitionKeyT,
      partitionKeyEncoder: Encoder<PartitionKeyT>,
      crossinline valuesExtractor: (T) -> List<Double>,
    ) =
      DataExtractors<T, PrivacyIdT, PartitionKeyT>(
        {
          contributionWithPrivacyId(
            privacyId = privacyIdExtractor(it),
            partitionKey = partitionKeyExtractor(it),
            values = valuesExtractor(it),
          )
        },
        privacyIdEncoder = privacyIdEncoder,
        partitionKeyEncoder = partitionKeyEncoder,
        hasValueExtractor = true,
      )

    /**
     * Constructs a [DataExtractors] that uses the provided functions to extract a privacy id and a
     * partition key into [ContributionWithPrivacyId] from the input data row.
     */
    inline fun <T, PrivacyIdT : Any, PartitionKeyT : Any> from(
      crossinline privacyIdExtractor: (T) -> PrivacyIdT,
      privacyIdEncoder: Encoder<PrivacyIdT>,
      crossinline partitionKeyExtractor: (T) -> PartitionKeyT,
      partitionKeyEncoder: Encoder<PartitionKeyT>,
    ) =
      DataExtractors<T, PrivacyIdT, PartitionKeyT>(
        {
          contributionWithPrivacyId(
            privacyId = privacyIdExtractor(it),
            partitionKey = partitionKeyExtractor(it),
            value = .0,
          )
        },
        privacyIdEncoder = privacyIdEncoder,
        partitionKeyEncoder = partitionKeyEncoder,
        hasValueExtractor = false,
      )
  }
}

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

/**
 * Holds the contributing privacy ID of PrivacyIdT type, the contributed partition of PartitionKeyT
 * type and value of Double type.
 */
typealias ContributionWithPrivacyId<PrivacyIdT, PartitionKeyT> =
  Pair<Pair<PrivacyIdT, PartitionKeyT>, List<Double>>

/** Helper function to create a [ContributionWithPrivacyId] from the list of values. */
fun <PrivacyIdT : Any, PartitionKeyT : Any> contributionWithPrivacyId(
  privacyId: PrivacyIdT,
  partitionKey: PartitionKeyT,
  values: List<Double>,
): ContributionWithPrivacyId<PrivacyIdT, PartitionKeyT> =
  Pair(Pair(privacyId, partitionKey), values)

/** Helper function to create a [ContributionWithPrivacyId] from one value. */
fun <PrivacyIdT : Any, PartitionKeyT : Any> contributionWithPrivacyId(
  privacyId: PrivacyIdT,
  partitionKey: PartitionKeyT,
  value: Double,
): ContributionWithPrivacyId<PrivacyIdT, PartitionKeyT> =
  contributionWithPrivacyId(privacyId, partitionKey, listOf(value))

/** Encoder of [ContributionWithPrivacyId]. */
fun <PrivacyIdT : Any, PartitionKeyT : Any> encoderOfContributionWithPrivacyId(
  privacyIdEncoder: Encoder<PrivacyIdT>,
  partitionKeyEncoder: Encoder<PartitionKeyT>,
  encodersFactory: EncoderFactory,
) =
  encodersFactory.tuple2sOf(
    encodersFactory.tuple2sOf(privacyIdEncoder, partitionKeyEncoder),
    encodersFactory.lists(encodersFactory.doubles()),
  )

/** Helper function to get the privacy ID of the given [ContributionWithPrivacyId]. */
fun <PrivacyIdT : Any, PartitionKeyT : Any> ContributionWithPrivacyId<PrivacyIdT, PartitionKeyT>
  .privacyId() = first.first

/** Helper function to get the partition key of the given [ContributionWithPrivacyId]. */
fun <PrivacyIdT : Any, PartitionKeyT : Any> ContributionWithPrivacyId<PrivacyIdT, PartitionKeyT>
  .partitionKey() = first.second

/** Helper function to get the value of the given [ContributionWithPrivacyId]. */
fun <PrivacyIdT : Any, PartitionKeyT : Any> ContributionWithPrivacyId<PrivacyIdT, PartitionKeyT>
  .values() = second

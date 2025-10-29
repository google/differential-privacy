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

import com.google.privacy.differentialprivacy.pipelinedp4j.proto.PrivacyIdContributions
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.PrivacyIdContributionsKt.multiValueContribution
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.privacyIdContributions

/** Bounds contributions to the entire non-aggregated data collection. */
sealed interface ContributionSampler<PrivacyIdT : Any, PartitionKeyT : Any> {
  /**
   * Samples the contributions of the privacy IDs.
   *
   * Returns a [FrameworkTable] where each entry contains a [PartitionKey] and the contributions of
   * a privacy ID to that [PartitionKey] after sampling. For each privacy ID contributing to a given
   * [PartitionKey], all its contributions are grouped inside the same entry.
   */
  fun sampleContributions(
    data: FrameworkCollection<MultiFeatureContribution<PrivacyIdT, PartitionKeyT>>
  ): FrameworkTable<PartitionKeyT, PrivacyIdContributions>
}

/**
 * Samples contributions to [maxPartitionsContributed] partitions among the given [contributions]
 * assuming that they all belong to the same [PrivacyId].
 */
internal fun <PrivacyIdT : Any, PartitionKeyT : Any> samplePartitions(
  contributions: Iterable<MultiFeatureContribution<PrivacyIdT, PartitionKeyT>>,
  maxPartitionsContributed: Int,
): Collection<MultiFeatureContribution<PrivacyIdT, PartitionKeyT>> {
  val allPartitions = contributions.map { it.partitionKey() }.toSet()
  val keptPartitions = sampleNElements(allPartitions, maxPartitionsContributed).toSet()
  return contributions.filter { it.partitionKey() in keptPartitions }
}

/**
 * Samples [maxContributionsPerPartition] contributions among the given [partitionContributions]
 * assuming that they all belong to the same [PrivacyId] and [PartitionKey]. Combines the result
 * into a [PrivacyIdContributions] and returns it.
 */
internal fun <PrivacyIdT : Any, PartitionKeyT : Any> sampleContributionsPerPartition(
  partitionContributions: Iterable<MultiFeatureContribution<PrivacyIdT, PartitionKeyT>>,
  maxContributionsPerPartition: Int,
): PrivacyIdContributions {
  val sampledContributions =
    sampleNElements(partitionContributions.toList(), maxContributionsPerPartition)
  return privacyIdContributions {
    for (contribution in sampledContributions) {
      // TODO: Update to add support for multiple features.
      // We expect that contribution contains only one feature with featureId="",
      // produced by DataExtractors.
      val perFeatureValues = contribution.perFeatureValues().single()
      if (perFeatureValues.values.size == 1) {
        singleValueContributions += perFeatureValues.values
      } else {
        multiValueContributions += multiValueContribution { values += perFeatureValues.values }
      }
    }
  }
}

private fun <T> sampleNElements(elements: Collection<T>, N: Int): Collection<T> {
  if (elements.size <= N) return elements
  return elements.shuffled().take(N)
}

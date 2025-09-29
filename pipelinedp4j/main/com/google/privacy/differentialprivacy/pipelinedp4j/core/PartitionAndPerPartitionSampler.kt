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

/**
 * Samples partitions contributed by each [PrivacyId] and per-partition contributions.
 *
 * Returns a map from [PartitionKey] to [PrivacyIdContributions] where [PrivacyIdContributions] is a
 * representation of all contributions of a [PrivacyId] to the corresponding [PartitionKey].
 *
 * Note: this class does not perform any checks on the consistency of [AggregationParams]. We expect
 * this to be done earlier in the call.
 *
 * This class uses [samplePerKey] for sampling and grouping per key, which is more efficient than
 * [groupByKey] in case one key is hot, and the bound is smaller than INTEGER.MAX_VALUE. If the
 * bound is MAX_VALUE, [samplePerKey] is equivalent to [groupByKey]. Using [samplePerKey] in this
 * way allows us to use this sampler for all other samplers, avoiding code reuse.
 *
 * Note: the bounder assumes that all contributions that belong to the same privacy ID can fit in
 * memory.
 */
class PartitionAndPerPartitionSampler<PrivacyIdT : Any, PartitionKeyT : Any>(
  private val maxPartitionsContributed: Int,
  private val maxContributionsPerPartition: Int,
  private val privacyIdEncoder: Encoder<PrivacyIdT>,
  private val partitionKeyEncoder: Encoder<PartitionKeyT>,
  private val encoderFactory: EncoderFactory,
) : ContributionSampler<PrivacyIdT, PartitionKeyT> {
  override fun sampleContributions(
    data: FrameworkCollection<MultiFeatureContribution<PrivacyIdT, PartitionKeyT>>
  ): FrameworkTable<PartitionKeyT, PrivacyIdContributions> {
    val perPartitionAggregatedData:
      FrameworkTable<PrivacyIdT, Pair<PartitionKeyT, PrivacyIdContributions>> =
      data
        .keyBy(
          "KeyByPrivacyIdAndPartitionKey",
          encoderFactory.tuple2sOf(privacyIdEncoder, partitionKeyEncoder),
        ) {
          it.privacyId() to it.partitionKey()
        }
        .samplePerKey("LInfSampling", maxContributionsPerPartition)
        .mapToTable(
          "MergeContributionsAndReKeyByPrivacyId",
          privacyIdEncoder,
          encoderFactory.tuple2sOf(
            partitionKeyEncoder,
            encoderFactory.protos(PrivacyIdContributions::class),
          ),
        ) { (privacyId, partitionKey), contributions ->
          privacyId to (partitionKey to mergeContributions(contributions))
        }

    // If the bound is MAX_VALUE, no need to sample partitions.
    if (maxPartitionsContributed == Int.MAX_VALUE) {
      return perPartitionAggregatedData.mapToTable(
        "RekeyByPartitionKey",
        partitionKeyEncoder,
        encoderFactory.protos(PrivacyIdContributions::class),
      ) { _, (partitionKey, contribution) ->
        partitionKey to contribution
      }
    } else {
      return perPartitionAggregatedData
        .samplePerKey("L0Sampling", maxPartitionsContributed)
        .flatMapToTable(
          "Flatten",
          partitionKeyEncoder,
          encoderFactory.protos(PrivacyIdContributions::class),
        ) { _, contributions ->
          contributions.asSequence().map { it }
        }
    }
  }
}

/**
 * Merges contributions of the same (privacy ID, partition key) into one [PrivacyIdContributions].
 */
private fun <PrivacyIdT : Any, PartitionKeyT : Any> mergeContributions(
  partitionContributions: Iterable<MultiFeatureContribution<PrivacyIdT, PartitionKeyT>>
): PrivacyIdContributions = privacyIdContributions {
  for (partitionContribution in partitionContributions) {
    // TODO: Update to add support for multiple features.
    // We expect that contribution contains only one feature with featureId="",
    // produced by DataExtractors.
    val perFeatureValues = partitionContribution.perFeatureValues().single()
    if (perFeatureValues.values.size == 1) {
      singleValueContributions += perFeatureValues.values
    } else {
      multiValueContributions += multiValueContribution { values += perFeatureValues.values }
    }
  }
}

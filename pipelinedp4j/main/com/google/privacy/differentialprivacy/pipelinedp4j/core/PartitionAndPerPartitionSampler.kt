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

/**
 * Samples partitions contributed by each [PrivacyId] and per-partition contributions.
 *
 * Returns a map from [PartitionKey] to [PrivacyIdContributions] where [PrivacyIdContributions] is a
 * representation of all contributions of a [PrivacyId] to the corresponding [PartitionKey].
 *
 * Note: this class does not perform any checks on the consistency of [AggregationParams]. We expect
 * this to be done earlier in the call.
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
    data: FrameworkCollection<ContributionWithPrivacyId<PrivacyIdT, PartitionKeyT>>
  ): FrameworkTable<PartitionKeyT, PrivacyIdContributions> {
    val inputByPid:
      FrameworkTable<PrivacyIdT, Iterable<ContributionWithPrivacyId<PrivacyIdT, PartitionKeyT>>> =
      data
        .keyBy("KeyByPrivacyId", privacyIdEncoder) { it.privacyId() }
        .groupByKey("GroupByPrivacyId")
    // TODO: Cover with tests (i.e. test should fail if this is not copied).
    // Necessary for DoFn to be serializable.
    val maxPartitionsContributedCopy = maxPartitionsContributed
    val maxContributionsPerPartitionCopy = maxContributionsPerPartition
    return inputByPid.flatMapToTable(
      "SamplePartitionsAndPerPartitionContributions",
      partitionKeyEncoder,
      encoderFactory.protos(PrivacyIdContributions::class),
    ) { _, contributions ->
      val l0BoundedData = samplePartitions(contributions, maxPartitionsContributedCopy)
      val groupedByPartitionKey:
        Map<PartitionKeyT, List<ContributionWithPrivacyId<PrivacyIdT, PartitionKeyT>>> =
        l0BoundedData.groupBy { it.partitionKey() }
      groupedByPartitionKey
        .mapValues { (_, partitionContributions) ->
          sampleContributionsPerPartition(partitionContributions, maxContributionsPerPartitionCopy)
        }
        .map { it.toPair() }
        .asSequence()
    }
  }
}

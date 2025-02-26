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
 * Samples partitions contributed by each [PrivacyId].
 *
 * Returns a map from [PartitionKey] to [PrivacyIdContributions] where [PrivacyIdContributions] is a
 * representation of all contributions of a [PrivacyId] to the corresponding [PartitionKey].
 *
 * Note: this class does not perform any checks on the consistency of [AggregationParams]. We expect
 * this to be done earlier in the call.
 */
class PartitionSampler<PrivacyIdT : Any, PartitionKeyT : Any>(
  private val maxPartitionsContributed: Int,
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
    val maxPartitionsContributed = maxPartitionsContributed
    return inputByPid.flatMapToTable(
      "SamplePartitions",
      partitionKeyEncoder,
      encoderFactory.protos(PrivacyIdContributions::class),
    ) { _, intialContributions ->
      val l0BoundedData = samplePartitions(intialContributions, maxPartitionsContributed)
      val groupedByPartitionKey:
        Map<PartitionKeyT, List<ContributionWithPrivacyId<PrivacyIdT, PartitionKeyT>>> =
        l0BoundedData.groupBy { it.partitionKey() }
      groupedByPartitionKey
        .mapValues { (_, partitionContributions) ->
          privacyIdContributions {
            for (partitionContribution in partitionContributions) {
              val contributionValues = partitionContribution.values()
              if (contributionValues.size == 1) {
                singleValueContributions += contributionValues.first()
              } else {
                multiValueContributions += multiValueContribution { values += contributionValues }
              }
            }
          }
        }
        .map { it.toPair() }
        .asSequence()
    }
  }
}

/**
 * Samples partitions contributed by each [PrivacyId]. Drops the values of the contributions.
 *
 * Returns a map from [PartitionKey] to [PrivacyIdContributions], where [PrivacyIdContributions] has
 * empty [values]. This class is more scalable than PartitionSampler because:
 * 1. It does not keep values (which are not need for SelectPartitions).
 * 2. It does 2 sampling per key - by (partition, privacy_id) and by partition. As a result records
 *    corresponding to one partitions are processed on different machines.
 *
 * The algorithm which performs one grouping per key (per privacy id) is faster for on average
 * dataset, but it less scalable, when one privacy id has a lot of contributions.
 *
 * Note: this class does not perform any checks on the consistency of [AggregationParams]. We expect
 * this to be done earlier in the call.
 */
class PartitionSamplerWithoutValues<PrivacyIdT : Any, PartitionKeyT : Any>(
  private val maxPartitionsContributed: Int,
  private val privacyIdEncoder: Encoder<PrivacyIdT>,
  private val partitionKeyEncoder: Encoder<PartitionKeyT>,
  private val encoderFactory: EncoderFactory,
) : ContributionSampler<PrivacyIdT, PartitionKeyT> {
  override fun sampleContributions(
    data: FrameworkCollection<ContributionWithPrivacyId<PrivacyIdT, PartitionKeyT>>
  ): FrameworkTable<PartitionKeyT, PrivacyIdContributions> {
    val maxPartitionsContributed = maxPartitionsContributed
    return data
      .keyBy("KeyByPrivacyId", encoderFactory.tuple2sOf(privacyIdEncoder, partitionKeyEncoder)) {
        it.privacyId() to it.partitionKey()
      }
      .samplePerKey("LinfSampling", 1)
      .mapToTable("DropPartitionKeyFromKey", privacyIdEncoder, partitionKeyEncoder) {
        _,
        contributions ->
        // Contribtions is a list of size 1, since we sampled 1 element per key.
        contributions.first().privacyId() to contributions.first().partitionKey()
      }
      .samplePerKey("L0Sampling", maxPartitionsContributed)
      .flatMapToTable(
        "ConvertToPrivacyIdContributions",
        partitionKeyEncoder,
        encoderFactory.protos(PrivacyIdContributions::class),
      ) { _, partitionKeys ->
        partitionKeys.asSequence().map { it to privacyIdContributions {} }
      }
  }
}

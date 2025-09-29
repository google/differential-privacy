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
) :
  ContributionSampler<PrivacyIdT, PartitionKeyT> by PartitionAndPerPartitionSampler(
    maxPartitionsContributed,
    maxContributionsPerPartition = Integer.MAX_VALUE,
    privacyIdEncoder,
    partitionKeyEncoder,
    encoderFactory,
  )

/**
 * Samples partitions contributed by each [PrivacyId]. Drops the values of the contributions.
 *
 * Returns a map from [PartitionKey] to [PrivacyIdContributions], where [PrivacyIdContributions] has
 * empty [values].
 *
 * Note: this class does not perform any checks on the consistency of [AggregationParams]. We expect
 * this to be done earlier in the call.
 */
class PartitionSamplerWithoutValues<PrivacyIdT : Any, PartitionKeyT : Any>(
  private val maxPartitionsContributed: Int,
  private val privacyIdEncoder: Encoder<PrivacyIdT>,
  private val partitionKeyEncoder: Encoder<PartitionKeyT>,
  private val encoderFactory: EncoderFactory,
  // need explicit delegation to add post-processing its [sampleContributions]
  private val sampler: ContributionSampler<PrivacyIdT, PartitionKeyT> =
    PartitionAndPerPartitionSampler(
      maxPartitionsContributed,
      maxContributionsPerPartition = 1,
      privacyIdEncoder,
      partitionKeyEncoder,
      encoderFactory,
    ),
) : ContributionSampler<PrivacyIdT, PartitionKeyT> {
  override fun sampleContributions(
    data: FrameworkCollection<MultiFeatureContribution<PrivacyIdT, PartitionKeyT>>
  ): FrameworkTable<PartitionKeyT, PrivacyIdContributions> =
    sampler
      .sampleContributions(data)
      .mapValues(
        "ConvertToEmptyPrivacyIdContributions",
        encoderFactory.protos(PrivacyIdContributions::class),
        { _, _ -> privacyIdContributions {} },
      )
}

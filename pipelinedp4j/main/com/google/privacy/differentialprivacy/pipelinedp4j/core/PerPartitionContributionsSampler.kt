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
 * Samples per-partition contributions by each [PrivacyId].
 *
 * Returns a map from [PartitionKey] to [PrivacyIdContributions] where [PrivacyIdContributions] is a
 * representation of all contributions of a [PrivacyId] to the corresponding [PartitionKey].
 *
 * Note: this class does not perform any checks on the consistency of [AggregationParams]. We expect
 * this to be done earlier in the call.
 */
class PerPartitionContributionsSampler<PrivacyIdT : Any, PartitionKeyT : Any>(
  maxContributionsPerPartition: Int,
  privacyIdEncoder: Encoder<PrivacyIdT>,
  partitionKeyEncoder: Encoder<PartitionKeyT>,
  encoderFactory: EncoderFactory,
) :
  ContributionSampler<PrivacyIdT, PartitionKeyT> by PartitionAndPerPartitionSampler(
    maxPartitionsContributed = Integer.MAX_VALUE,
    maxContributionsPerPartition,
    privacyIdEncoder,
    partitionKeyEncoder,
    encoderFactory,
  )

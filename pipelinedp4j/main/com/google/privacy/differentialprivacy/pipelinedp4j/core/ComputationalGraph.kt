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

import com.google.privacy.differentialprivacy.pipelinedp4j.proto.CompoundAccumulator
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.DpAggregates
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.PrivacyIdContributions
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.PrivacyIdCountAccumulator

/**
 * An assembly of transformations applied to the input data in order to produce anonymized metrics.
 *
 * [T] is the type of the elements stored in the collection whose data is being anonymized. It must
 * be possible to extract the privacy ID, the partition key and optionally the value being
 * aggregated from an element of type [T].
 */
internal interface ComputationalGraph<T, PrivacyIdT : Any, PartitionKeyT : Any> {
  fun aggregate(collection: FrameworkCollection<T>): FrameworkTable<PartitionKeyT, DpAggregates>
}

/**
 * Creates an instance of [ComputationalGraph]. The class is "open" because we spy on it in the
 * tests in order to check that the arguments passed to [DpEngine.aggregate] are correctly mapped to
 * the computational graph components.
 */
internal open class ComputationalGraphFactory {
  open fun <T, PrivacyIdT : Any, PartitionKeyT : Any> createForPublicPartitions(
    contributionSampler: ContributionSampler<PrivacyIdT, PartitionKeyT>,
    combiner: CompoundCombiner,
    extractors: DataExtractors<T, PrivacyIdT, PartitionKeyT>,
    encodersFactory: EncoderFactory,
    publicPartitions: FrameworkCollection<PartitionKeyT>,
    partitionsBalance: PartitionsBalance,
  ) =
    PublicPartitionsComputationalGraph(
      contributionSampler,
      publicPartitions,
      partitionsBalance,
      combiner,
      extractors,
      encodersFactory,
    )

  open fun <T, PrivacyIdT : Any, PartitionKeyT : Any> createForPrivatePartitions(
    contributionSampler: ContributionSampler<PrivacyIdT, PartitionKeyT>,
    preAggregationPartitionSelector: PreAggregationPartitionSelector?,
    combiner: CompoundCombiner,
    extractors: DataExtractors<T, PrivacyIdT, PartitionKeyT>,
    encodersFactory: EncoderFactory,
  ) =
    PrivatePartitionsComputationalGraph(
      contributionSampler,
      preAggregationPartitionSelector,
      combiner,
      extractors,
      encodersFactory,
    )

  open fun <T, PrivacyIdT : Any, PartitionKeyT : Any> createForSelectPartitions(
    contributionSampler: ContributionSampler<PrivacyIdT, PartitionKeyT>,
    partitionSelector: PreAggregationPartitionSelector,
    extractors: DataExtractors<T, PrivacyIdT, PartitionKeyT>,
    encodersFactory: EncoderFactory,
  ) =
    SelectPartitionsComputationalGraph(
      contributionSampler,
      partitionSelector,
      extractors,
      encodersFactory,
    )
}

/** An implementation of a [ComputationalGraph] that computes metrics with public partitions. */
internal class PublicPartitionsComputationalGraph<T, PrivacyIdT : Any, PartitionKeyT : Any>
internal constructor(
  private val contributionSampler: ContributionSampler<PrivacyIdT, PartitionKeyT>,
  private val publicPartitions: FrameworkCollection<PartitionKeyT>,
  private val partitionsBalance: PartitionsBalance,
  private val combiner: CompoundCombiner,
  private val extractors: DataExtractors<T, PrivacyIdT, PartitionKeyT>,
  private val encodersFactory: EncoderFactory,
) : ComputationalGraph<T, PrivacyIdT, PartitionKeyT> {

  override fun aggregate(
    collection: FrameworkCollection<T>
  ): FrameworkTable<PartitionKeyT, DpAggregates> {
    val extractedCol: FrameworkCollection<ContributionWithPrivacyId<PrivacyIdT, PartitionKeyT>> =
      collection.map(
        "ExtractContributions",
        encoderOfContributionWithPrivacyId(
          extractors.privacyIdEncoder,
          extractors.partitionKeyEncoder,
          encodersFactory,
        ),
        extractors.contributionExtractor,
      )
    val filteredCol =
      extractedCol.dropNonPublicPartitions(
        publicPartitions,
        extractors.partitionKeyEncoder,
        partitionsBalance,
      )
    val boundedCollection: FrameworkTable<PartitionKeyT, PrivacyIdContributions> =
      contributionSampler.sampleContributions(filteredCol)
    // We cannot refer to the combiner of the PublicPartitionsComputationalGraph from the DoFn
    // below because it breaks serialization.
    val combinerCopy = combiner
    val accumulatorsPerPrivacyIdContributions: FrameworkTable<PartitionKeyT, CompoundAccumulator> =
      boundedCollection.mapValues(
        "CreateAccumulatorFromPrivacyIdContributions",
        encodersFactory.protos(CompoundAccumulator::class),
        { _, privacyIdContributions -> combinerCopy.createAccumulator(privacyIdContributions) },
      )
    val collectionWithPublicPartitions: FrameworkTable<PartitionKeyT, CompoundAccumulator> =
      accumulatorsPerPrivacyIdContributions.insertPublicPartitions(
        publicPartitions,
        combinerCopy,
        extractors.partitionKeyEncoder,
        encodersFactory,
      )
    val perPartitionAccumulators: FrameworkTable<PartitionKeyT, CompoundAccumulator> =
      collectionWithPublicPartitions.groupAndCombineValues(
        "CombinePerPartitionKey",
        combinerCopy::mergeAccumulators,
      )
    val dpAggregates: FrameworkTable<PartitionKeyT, DpAggregates> =
      perPartitionAccumulators.mapValues(
        "ComputeDpAggregates",
        encodersFactory.protos(DpAggregates::class),
      ) { _, acc: CompoundAccumulator ->
        combinerCopy.computeMetrics(acc)
      }
    return dpAggregates
  }
}

/**
 * An implementation of a [ComputationalGraph] that computes metrics with private partition
 * selection.
 */
internal class PrivatePartitionsComputationalGraph<T, PrivacyIdT : Any, PartitionKeyT : Any>
internal constructor(
  private val contributionSampler: ContributionSampler<PrivacyIdT, PartitionKeyT>,
  private val preAggregationPartitionSelector: PreAggregationPartitionSelector?,
  private val combiner: CompoundCombiner,
  private val extractors: DataExtractors<T, PrivacyIdT, PartitionKeyT>,
  private val encodersFactory: EncoderFactory,
) : ComputationalGraph<T, PrivacyIdT, PartitionKeyT> {

  init {
    val hasPostAggregationCombiner = combiner.hasPostAggregationCombiner()
    val hasPreAggregationCombiner = preAggregationPartitionSelector != null
    require(hasPreAggregationCombiner != hasPostAggregationCombiner) {
      "Computational graph must have either PreAggregationPartitionSelector or " +
        "PostAggregationPartitionSelectionCombiner."
    }
  }

  override fun aggregate(
    collection: FrameworkCollection<T>
  ): FrameworkTable<PartitionKeyT, DpAggregates> {
    val extractedCol: FrameworkCollection<ContributionWithPrivacyId<PrivacyIdT, PartitionKeyT>> =
      collection.map(
        "ExtractContributions",
        encoderOfContributionWithPrivacyId(
          extractors.privacyIdEncoder,
          extractors.partitionKeyEncoder,
          encodersFactory,
        ),
        extractors.contributionExtractor,
      )
    val boundedCollection: FrameworkTable<PartitionKeyT, PrivacyIdContributions> =
      contributionSampler.sampleContributions(extractedCol)
    // We cannot refer to the combiner of the PrivatePartitionsComputationalGraph from the DoFn
    // below because it breaks serialization.
    val combinerCopy = combiner
    val accumulatorsPerPrivacyIdContributions: FrameworkTable<PartitionKeyT, CompoundAccumulator> =
      boundedCollection.mapValues(
        "CreateAccumulatorFromPrivacyIdContributions",
        encodersFactory.protos(CompoundAccumulator::class),
        { _, privacyIdContributions -> combinerCopy.createAccumulator(privacyIdContributions) },
      )
    var perPartitionAccumulators: FrameworkTable<PartitionKeyT, CompoundAccumulator> =
      accumulatorsPerPrivacyIdContributions.groupAndCombineValues(
        "CombinePerPartitionKey",
        combinerCopy::mergeAccumulators,
      )

    // Fields of this can not be referred from DoFn below because it breaks serialization.
    if (preAggregationPartitionSelector != null) {
      val partitionSelectorCopy = preAggregationPartitionSelector
      perPartitionAccumulators =
        perPartitionAccumulators.filterValues(
          "ApplyPreAggregationPartitionSelector",
          { v -> partitionSelectorCopy.shouldKeep(v.privacyIdCountAccumulator.count) },
        )
    }

    val dpAggregates: FrameworkTable<PartitionKeyT, DpAggregates> =
      perPartitionAccumulators.mapValues(
        "ComputeDpAggregates",
        encodersFactory.protos(DpAggregates::class),
      ) { _, acc: CompoundAccumulator ->
        combinerCopy.computeMetrics(acc)
      }
    if (combiner.hasPostAggregationCombiner()) {
      return dpAggregates.filterValues(
        "PostAggregationPartitionSelection",
        { it.privacyIdCount != 0.0 },
      )
    }
    return dpAggregates
  }
}

/** An implementation of a [ComputationalGraph] for selecting partitions. */
internal class SelectPartitionsComputationalGraph<T, PrivacyIdT : Any, PartitionKeyT : Any>(
  private val contributionSampler: ContributionSampler<PrivacyIdT, PartitionKeyT>,
  private val partitionSelector: PreAggregationPartitionSelector,
  private val extractors: DataExtractors<T, PrivacyIdT, PartitionKeyT>,
  private val encodersFactory: EncoderFactory,
) {

  fun selectPartitions(collection: FrameworkCollection<T>): FrameworkCollection<PartitionKeyT> {
    val extractedCol: FrameworkCollection<ContributionWithPrivacyId<PrivacyIdT, PartitionKeyT>> =
      collection.map(
        "ExtractContributions",
        encoderOfContributionWithPrivacyId(
          extractors.privacyIdEncoder,
          extractors.partitionKeyEncoder,
          encodersFactory,
        ),
        extractors.contributionExtractor,
      )
    val boundedCollection: FrameworkTable<PartitionKeyT, PrivacyIdContributions> =
      contributionSampler.sampleContributions(extractedCol)
    val combiner = ExactPrivacyIdCountCombiner()
    val accumulatorsPerPrivacyIdContributions:
      FrameworkTable<PartitionKeyT, PrivacyIdCountAccumulator> =
      boundedCollection.mapValues(
        "CreateAccumulatorFromPrivacyIdContributions",
        encodersFactory.protos(PrivacyIdCountAccumulator::class),
        { _, privacyIdContributions -> combiner.createAccumulator(privacyIdContributions) },
      )
    val perPartitionAccumulators: FrameworkTable<PartitionKeyT, PrivacyIdCountAccumulator> =
      accumulatorsPerPrivacyIdContributions.groupAndCombineValues(
        "CombinePerPartitionKey",
        combiner::mergeAccumulators,
      )

    // We cannot refer to the this.partitionSelector from the DoFn because it breaks serialization.
    val partitionSelectorCopy = partitionSelector

    return perPartitionAccumulators
      .filterValues(
        "ApplyPreAggregationPartitionSelector",
        { partitionSelectorCopy.shouldKeep(it.count) },
      )
      .keys("KeepPartitionKeys")
  }
}

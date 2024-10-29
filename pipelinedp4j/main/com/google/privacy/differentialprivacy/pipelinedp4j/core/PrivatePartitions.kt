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

import com.google.privacy.differentialprivacy.Noise
import com.google.privacy.differentialprivacy.pipelinedp4j.core.budget.AllocatedBudget
import com.google.privacy.differentialprivacy.pipelinedp4j.dplibrary.PreAggregationPartitionSelectionFactory
import java.io.Serializable

/** Interface for pre-aggregation partition selectors. */
interface PreAggregationPartitionSelector : Serializable {
  fun shouldKeep(privacyIdCount: Long): Boolean
}

/**
 * An Implementation of [PreAggregationPartitionSelector] that uses the DP library.
 *
 * @property maxPartitionsContributed performs contribution bounding.
 * @property prethreshold thresholds contributions before partition selection occurs.
 * @property budget is the amount of privacy budget that can be used by partition selection.
 * @property factory is used to create DP building block library objects. It is also used for
 *   dependency injection in tests.
 */
class DpLibPreAggregationPartitionSelector(
  private val maxPartitionsContributed: Int,
  val preThreshold: Int,
  private val budget: AllocatedBudget,
  private val factory: PreAggregationPartitionSelectionFactory,
) : PreAggregationPartitionSelector, Serializable {

  override fun shouldKeep(privacyIdCount: Long): Boolean {
    val preAggregationPartitionSelection =
      factory.create(
        epsilon = budget.epsilon(),
        delta = budget.delta(),
        maxPartitionsContributed = maxPartitionsContributed,
        preThreshold = preThreshold,
      )
    preAggregationPartitionSelection.incrementBy(privacyIdCount)
    return preAggregationPartitionSelection.shouldKeepPartition()
  }
}

/** Interface for post-aggregation partition selectors. */
interface PostAggregationPartitionSelector : Serializable {
  /**
   * @param privacyIdCount is the true number of privacy units that contributed to this partition.
   * @return the noise value if the partition should be kept or returns null if the partition should
   *   be dropped.
   */
  fun addNoiseIfShouldKeep(privacyIdCount: Long): Double?

  /**
   * The [threshold] for the partition selection. As an esential part of the computational graph,
   * the [threshold] will be reported to clients.
   */
  val threshold: Double
}

/**
 * An Implementation of [PostAggregationPartitionSelector] that uses the DP library.
 *
 * @property maxPartitionsContributed performs contribution bounding.
 * @property prethreshold thresholds contributions before partition selection occurs.
 * @property noiseBudget is the amount of privacy budget that can be used by partition selection.
 * @property thresholdingBudget is the amount of privacy budget that can be used by thresholding.
 * @property noiseFactory is used to generate noise in partition selection.
 */
class PostAggregationPartitionSelectorImpl(
  private val maxPartitionsContributed: Int,
  private val noiseKind: NoiseKind,
  val preThreshold: Int,
  private val noiseBudget: AllocatedBudget,
  private val thresholdingBudget: AllocatedBudget,
  private val noiseFactory: (NoiseKind) -> Noise,
) : PostAggregationPartitionSelector, Serializable {

  /**
   * @param privacyIdCount is the true number of privacy units that contributed to this partition.
   * @return the noise value if the partition should be kept or returns null if the partition should
   *   be dropped.
   */
  override fun addNoiseIfShouldKeep(privacyIdCount: Long): Double? {
    if (privacyIdCount < preThreshold) return null

    val noisePrivacyIdCount =
      noiseFactory(noiseKind)
        .addNoise(
          privacyIdCount.toDouble(),
          maxPartitionsContributed,
          /* lInfSensitivity = */ 1.0,
          noiseBudget.epsilon(),
          noiseBudget.delta(),
        )
    return if (noisePrivacyIdCount >= threshold + (preThreshold - 1)) noisePrivacyIdCount else null
  }

  override val threshold: Double by lazy {
    1 -
      noiseFactory(noiseKind)
        .computeQuantile(
          /* rank= */ thresholdingBudget.delta() / maxPartitionsContributed,
          /* x= */ 0.0,
          maxPartitionsContributed,
          /* lInfSensitivity=*/ 1.0,
          noiseBudget.epsilon(),
          noiseBudget.delta(),
        )
  }
}

class NoPrivacyPartitionSelector : PreAggregationPartitionSelector, Serializable {
  /**
   * @param privacyIdCount is the true number of privacy units that contributed to this partition.
   * @return always returns true because no privacy is applied.
   */
  override fun shouldKeep(privacyIdCount: Long): Boolean = true
}

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
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.COUNT
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.MEAN
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.PRIVACY_ID_COUNT
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.QUANTILES
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.SUM
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.VARIANCE
import com.google.privacy.differentialprivacy.pipelinedp4j.core.MetricType.VECTOR_SUM
import com.google.privacy.differentialprivacy.pipelinedp4j.core.NoiseKind.GAUSSIAN
import com.google.privacy.differentialprivacy.pipelinedp4j.core.NoiseKind.LAPLACE
import com.google.privacy.differentialprivacy.pipelinedp4j.core.budget.AccountedMechanism.GAUSSIAN_NOISE
import com.google.privacy.differentialprivacy.pipelinedp4j.core.budget.AccountedMechanism.LAPLACE_NOISE
import com.google.privacy.differentialprivacy.pipelinedp4j.core.budget.AccountedMechanism.POSTAGGREGATED_PARTITION_SELECTION
import com.google.privacy.differentialprivacy.pipelinedp4j.core.budget.AccountedMechanism.PREAGGREGATED_PARTITION_SELECTION
import com.google.privacy.differentialprivacy.pipelinedp4j.core.budget.AllocatedBudget
import com.google.privacy.differentialprivacy.pipelinedp4j.core.budget.BudgetAccountant
import com.google.privacy.differentialprivacy.pipelinedp4j.core.budget.BudgetAccountantFactory
import com.google.privacy.differentialprivacy.pipelinedp4j.core.budget.BudgetAccountingStrategy
import com.google.privacy.differentialprivacy.pipelinedp4j.core.budget.BudgetAccountingStrategy.NAIVE
import com.google.privacy.differentialprivacy.pipelinedp4j.core.budget.BudgetPerOpSpec
import com.google.privacy.differentialprivacy.pipelinedp4j.core.budget.BudgetRequest
import com.google.privacy.differentialprivacy.pipelinedp4j.core.budget.RelativeBudgetPerOpSpec
import com.google.privacy.differentialprivacy.pipelinedp4j.core.budget.TotalBudget
import com.google.privacy.differentialprivacy.pipelinedp4j.dplibrary.NoiseFactory
import com.google.privacy.differentialprivacy.pipelinedp4j.dplibrary.PreAggregationPartitionSelectionFactory
import com.google.privacy.differentialprivacy.pipelinedp4j.dplibrary.ZeroNoiseFactory
import com.google.privacy.differentialprivacy.pipelinedp4j.proto.DpAggregates

/**
 * An engine that computes and returns the DP metrics.
 *
 * In order to request computation of DP metrics, call [aggregate]. You can call it multiple times
 * on the same or different input data collections. After calling [aggregate] once or multiple
 * times, call [done]. Once [done] has been called, the [DpEngine] instance cannot be used.
 *
 * The result of [aggregate] cannot be used immediately. Internally, [aggregate] builds the
 * computational graph that implements the anonymization logic but does not immediately know the
 * values of the privacy budget parameters (epsilon and delta) that should be used to compute the
 * metrics. The actual budget values get allocated when [done] is called. After calling [done], you
 * should be able to access the results of all [aggregate] calls on a given instance of [DpEngine]
 * unless the computational framework that you are using introduces additional constraints.
 *
 * The "lazy" privacy budget evaluation logic described above introduces limitations on the
 * framework that can be used to perform computations as it should allow for "lazy" evaluation of
 * the content of the [aggregate] result. For example, to address this limitation when running
 * [aggregate] locally, we use Kotlin [Sequence] as its elements don't get computed until they are
 * accessed.
 */
open class DpEngine
internal constructor(
  private val encoderFactory: EncoderFactory,
  private val budgetAccountant: BudgetAccountant,
  private val defaultNoiseFactory: (NoiseKind) -> Noise = NoiseFactory(),
  private val computationalGraphFactory: ComputationalGraphFactory = ComputationalGraphFactory(),
) {
  companion object Factory {
    fun create(encoderFactory: EncoderFactory, budgetSpec: DpEngineBudgetSpec) =
      DpEngine(
        encoderFactory,
        BudgetAccountantFactory.forStrategy(budgetSpec.accountingStrategy, budgetSpec.budget),
        NoiseFactory(),
        ComputationalGraphFactory(),
      )
  }

  private var doneCalled = false

  /**
   * Creates a pipeline that computes the DP metrics defined in [aggregationParams]. The metrics are
   * computed on [collection]. [dataExtractors] define how the privacy ID, partition key and the
   * aggregated value should be extracted from each data element in the [collection]. If
   * [publicPartitions] aren't provided, private partition selection is performed.
   *
   * The actual computation of the DP metrics cannot happen until [done] is called. [done] triggers
   * the privacy budget allocation. Hence, any access to the result of this method before [done] has
   * been called will throw an exception.
   */
  open fun <T, PrivacyIdT : Any, PartitionKeyT : Any> aggregate(
    collection: FrameworkCollection<T>,
    aggregationParams: AggregationParams,
    dataExtractors: DataExtractors<T, PrivacyIdT, PartitionKeyT>,
    publicPartitions: FrameworkCollection<PartitionKeyT>? = null,
  ): FrameworkTable<PartitionKeyT, DpAggregates> {
    throwIfDoneWasCalled()
    // Ensure that public partitions are unique, it is important for correctness.
    val uniquePublicPartitions = publicPartitions?.distinct("MakeSuppliedPartitionsUnique")
    validateAggregationParams(
      aggregationParams,
      uniquePublicPartitions != null,
      dataExtractors.hasValueExtractor,
    )
    // Beyond this point we can assume aggregation parameters are correct.
    val noiseFactory =
      if (aggregationParams.executionMode.appliesNoise) {
        defaultNoiseFactory
      } else {
        ZeroNoiseFactory()
      }
    val compoundCombiner =
      createCompoundCombiner(aggregationParams, uniquePublicPartitions != null, noiseFactory)
    val contributionSampler =
      createContributionsSampler(
        aggregationParams,
        compoundCombiner,
        dataExtractors.privacyIdEncoder,
        dataExtractors.partitionKeyEncoder,
      )

    val graph: ComputationalGraph<T, PrivacyIdT, PartitionKeyT> =
      if (uniquePublicPartitions == null) {
        val partitionSelector = createPartitionSelectorIfPreaggregationIsUsed(aggregationParams)
        computationalGraphFactory.createForPrivatePartitions(
          contributionSampler,
          partitionSelector,
          compoundCombiner,
          dataExtractors,
          encoderFactory,
        )
      } else {
        computationalGraphFactory.createForPublicPartitions(
          contributionSampler,
          compoundCombiner,
          dataExtractors,
          encoderFactory,
          uniquePublicPartitions,
          aggregationParams.partitionsBalance,
        )
      }
    return graph.aggregate(collection)
  }

  /**
   * Creates a pipeline that computes partition keys from [collection] in a differentially-private
   * manner. [dataExtractors] define how the privacy ID and partition key should be extracted from
   * each data element in the [collection].
   *
   * The actual computation of the DP metrics cannot happen until [done] is called. [done] triggers
   * the privacy budget allocation. Hence, any access to the result of this method before [done] has
   * been called will throw an exception.
   */
  open fun <T, PrivacyIdT : Any, PartitionKeyT : Any> selectPartitions(
    collection: FrameworkCollection<T>,
    params: SelectPartitionsParams,
    dataExtractors: DataExtractors<T, PrivacyIdT, PartitionKeyT>,
  ): FrameworkCollection<PartitionKeyT> {
    throwIfDoneWasCalled()
    validateSelectPartitionsParams(params)

    val partitionSelector =
      if (params.executionMode.partitionSelectionIsNonDeterministic) createPartitionSelector(params)
      else NoPrivacyPartitionSelector()
    val contributionSampler =
      if (params.applyPartitionsContributedBounding)
        PartitionSamplerWithoutValues(
          params.maxPartitionsContributed,
          dataExtractors.privacyIdEncoder,
          dataExtractors.partitionKeyEncoder,
          encoderFactory,
        )
      else
        NoPrivacySampler(
          dataExtractors.privacyIdEncoder,
          dataExtractors.partitionKeyEncoder,
          encoderFactory,
        )

    val graph: SelectPartitionsComputationalGraph<T, PrivacyIdT, PartitionKeyT> =
      computationalGraphFactory.createForSelectPartitions(
        contributionSampler,
        partitionSelector,
        dataExtractors,
        encoderFactory,
      )
    return graph.selectPartitions(collection)
  }

  /**
   * Allocates privacy budgets to the metrics whose computation has been requested by calling
   * [aggregate]. This method must be called once per [DpEngine] instance.
   */
  fun done() {
    throwIfDoneWasCalled()
    doneCalled = true
    budgetAccountant.allocateBudgets()
  }

  private fun throwIfDoneWasCalled() {
    if (doneCalled) {
      throw IllegalStateException(
        "done() has already been called on this instance. The instance cannot be used anymore."
      )
    }
  }

  private fun <PrivacyIdT : Any, PartitionKeyT : Any> createContributionsSampler(
    params: AggregationParams,
    combiner: CompoundCombiner,
    privacyIdEncoder: Encoder<PrivacyIdT>,
    partitionKeyEncoder: Encoder<PartitionKeyT>,
  ): ContributionSampler<PrivacyIdT, PartitionKeyT> {
    return if (params.applyPerPartitionBounding && params.applyPartitionsContributedBounding) {
      if (combiner.requiresPerPartitionBoundedInput) {
        PartitionAndPerPartitionSampler(
          params.maxPartitionsContributed!!,
          params.maxContributionsPerPartition!!,
          privacyIdEncoder,
          partitionKeyEncoder,
          encoderFactory,
        )
      } else {
        PartitionSampler(
          params.maxPartitionsContributed!!,
          privacyIdEncoder,
          partitionKeyEncoder,
          encoderFactory,
        )
      }
    } else if (params.applyPartitionsContributedBounding) {
      // && !applyPerPartitionBounding
      PartitionSampler(
        params.maxPartitionsContributed!!,
        privacyIdEncoder,
        partitionKeyEncoder,
        encoderFactory,
      )
    } else if (params.applyPerPartitionBounding) {
      // && !applyPartitionsContributedBounding
      if (combiner.requiresPerPartitionBoundedInput) {
        PerPartitionContributionsSampler(
          params.maxContributionsPerPartition!!,
          privacyIdEncoder,
          partitionKeyEncoder,
          encoderFactory,
        )
      } else {
        NoPrivacySampler(privacyIdEncoder, partitionKeyEncoder, encoderFactory)
      }
    } else {
      // !applyPartitionsContributedBounding && !applyPerPartitionBounding
      NoPrivacySampler(privacyIdEncoder, partitionKeyEncoder, encoderFactory)
    }
  }

  private fun createCompoundCombiner(
    params: AggregationParams,
    usePublicPartitions: Boolean,
    noiseFactory: (NoiseKind) -> Noise,
  ): CompoundCombiner {
    val meanInMetrics = params.metrics.any { it.type == MEAN }
    val varianceInMetrics = params.metrics.any { it.type == VARIANCE }
    val metricCombiners =
      params.metrics
        .mapNotNull { metric ->
          when (metric.type) {
            PRIVACY_ID_COUNT -> {
              if (usePostAggregationPartitionSelection(params, usePublicPartitions)) {
                PostAggregationPartitionSelectionCombiner(
                  params,
                  getBudgetForMetric(metric, params),
                  getBudgetForPostAggregationPartitionSelection(params.partitionSelectionBudget),
                  noiseFactory,
                )
              } else {
                PrivacyIdCountCombiner(params, getBudgetForMetric(metric, params), noiseFactory)
              }
            }
            COUNT -> {
              if (!meanInMetrics && !varianceInMetrics) {
                CountCombiner(params, getBudgetForMetric(metric, params), noiseFactory)
              } else {
                null
              }
            }
            SUM -> {
              if (!meanInMetrics && !varianceInMetrics) {
                SumCombiner(params, getBudgetForMetric(metric, params), noiseFactory)
              } else {
                null
              }
            }
            VECTOR_SUM -> {
              VectorSumCombiner(params, getBudgetForMetric(metric, params), noiseFactory)
            }
            MEAN -> {
              if (!varianceInMetrics) {
                val (countBudget, sumBudget) = calculateCountSumBudgetsForMean(params)
                MeanCombiner(params, countBudget, sumBudget, noiseFactory)
              } else {
                null
              }
            }
            VARIANCE -> {
              val (countBudget, sumBudget, sumSquaresBudget) = calculateBudgetsForVariance(params)
              VarianceCombiner(params, countBudget, sumBudget, sumSquaresBudget, noiseFactory)
            }

            is QUANTILES -> {
              QuantilesCombiner(
                (metric.type as QUANTILES).sortedRanks,
                params,
                getBudgetForMetric(metric, params),
                noiseFactory,
              )
            }
          }
        }
        .toMutableList()
    if (!usePublicPartitions && !params.metrics.any { it.type == PRIVACY_ID_COUNT }) {
      // For private partitions, we need to compute the privacy ID count, even if PRIVACY_ID_COUNT
      // is not requested in metrics.
      metricCombiners.add(ExactPrivacyIdCountCombiner())
    }
    return CompoundCombiner(metricCombiners.toList())
  }

  private fun createPartitionSelectorIfPreaggregationIsUsed(
    params: AggregationParams
  ): PreAggregationPartitionSelector? {
    // If partition selection is deterministic, we create a no-op partition selector.
    if (!params.executionMode.partitionSelectionIsNonDeterministic) {
      return NoPrivacyPartitionSelector()
    }
    if (usePostAggregationPartitionSelection(params, usePublicPartitions = false)) return null
    val budget = getBudgetForPreAggregationPartitionSelection(params.partitionSelectionBudget)
    // If maxPartitionsContributed unset for partition selection, NullPointerException is expected.
    return DpLibPreAggregationPartitionSelector(
      params.maxPartitionsContributed!!,
      params.preThreshold,
      budget,
      PreAggregationPartitionSelectionFactory(),
    )
  }

  private fun createPartitionSelector(
    params: SelectPartitionsParams
  ): PreAggregationPartitionSelector {
    val budget = getBudgetForPreAggregationPartitionSelection(params.budget)
    // If maxPartitionsContributed unset for partition selection, NullPointerException is expected.
    // TODO: Support MaxContributions contribution bounding parameter.
    return DpLibPreAggregationPartitionSelector(
      params.maxPartitionsContributed,
      params.preThreshold,
      budget,
      PreAggregationPartitionSelectionFactory(),
    )
  }

  private fun getBudgetForMetric(
    metric: MetricDefinition,
    params: AggregationParams,
  ): AllocatedBudget {
    val budgetSpec = metric.budgetSpec ?: RelativeBudgetPerOpSpec(weight = 1.0)
    return budgetAccountant.requestBudget(
      BudgetRequest(budgetSpec, getNoiseAccountedMechanism(params.noiseKind))
    )
  }

  private fun getBudgetForPreAggregationPartitionSelection(
    partitionSelectionBudget: BudgetPerOpSpec?
  ): AllocatedBudget {
    val budgetSpec = partitionSelectionBudget ?: RelativeBudgetPerOpSpec(weight = 1.0)
    return budgetAccountant.requestBudget(
      BudgetRequest(budgetSpec, PREAGGREGATED_PARTITION_SELECTION)
    )
  }

  private fun getBudgetForPostAggregationPartitionSelection(
    partitionSelectionBudget: BudgetPerOpSpec?
  ): AllocatedBudget {
    val budgetSpec = partitionSelectionBudget ?: RelativeBudgetPerOpSpec(weight = 1.0)
    return budgetAccountant.requestBudget(
      BudgetRequest(budgetSpec, POSTAGGREGATED_PARTITION_SELECTION)
    )
  }

  private fun calculateCountSumBudgetsForMean(
    params: AggregationParams
  ): Pair<AllocatedBudget, AllocatedBudget> {
    fun getMetricDefinition(metricType: MetricType) = params.metrics.find { it.type == metricType }

    // meanDefinition is not null, because this function is called only when MEAN is in metrics.
    val meanDefinition = getMetricDefinition(MEAN)!!

    // Budget spec for COUNT.
    val countBudgetSpec: BudgetPerOpSpec =
      if (meanDefinition.budgetSpec != null) {
        // It is 50% of MEAN spec, if MEAN spec is provided.
        meanDefinition.budgetSpec!!.times(0.5)
      } else {
        // Or COUNT spec or the default budget spec.
        getMetricDefinition(COUNT)?.budgetSpec ?: RelativeBudgetPerOpSpec(weight = 1.0)
      }

    // Budget spec for SUM.
    val sumBudgetSpec: BudgetPerOpSpec =
      if (meanDefinition.budgetSpec != null) {
        // It is 50% of MEAN spec, if MEAN spec is provided.
        meanDefinition.budgetSpec!!.times(0.5)
      } else {
        // Or SUM spec or the default budget spec.
        getMetricDefinition(SUM)?.budgetSpec ?: RelativeBudgetPerOpSpec(weight = 1.0)
      }

    return budgetAccountant.requestBudget(
      BudgetRequest(countBudgetSpec, getNoiseAccountedMechanism(params.noiseKind))
    ) to
      budgetAccountant.requestBudget(
        BudgetRequest(sumBudgetSpec, getNoiseAccountedMechanism(params.noiseKind))
      )
  }

  private fun calculateBudgetsForVariance(
    params: AggregationParams
  ): Triple<AllocatedBudget, AllocatedBudget, AllocatedBudget> {
    // Variance is not null because this function is called only when it is in metrics.
    val varianceDefinition = params.metrics.find { it.type == VARIANCE }!!
    // Budget is split equally between COUNT, SUM and SUM_SQUARES.
    val budgetSplit = 1.0 / 3.0
    // If varianceDefinition.budgetSpec is null, the default budget spec is used.
    val defaultBudgetSpec = RelativeBudgetPerOpSpec(weight = 1.0)
    val noiseAccountedMechanism = getNoiseAccountedMechanism(params.noiseKind)

    val budgetSpec = varianceDefinition.budgetSpec?.times(budgetSplit) ?: defaultBudgetSpec
    val budgetRequest = BudgetRequest(budgetSpec, noiseAccountedMechanism)

    return Triple(
      budgetAccountant.requestBudget(budgetRequest), // COUNT
      budgetAccountant.requestBudget(budgetRequest), // SUM
      budgetAccountant.requestBudget(budgetRequest), // SUM_SQUARES
    )
  }

  private fun getNoiseAccountedMechanism(noiseKind: NoiseKind) =
    when (noiseKind) {
      LAPLACE -> LAPLACE_NOISE
      GAUSSIAN -> GAUSSIAN_NOISE
    }
}

/**
 * The total amount of budget that can be consumed by the partition selection and aggregations
 * computed by the [DpEngine] instance.
 */
data class DpEngineBudgetSpec(
  val budget: TotalBudget,
  val accountingStrategy: BudgetAccountingStrategy = NAIVE,
)

private fun usePostAggregationPartitionSelection(
  params: AggregationParams,
  usePublicPartitions: Boolean,
): Boolean =
  !usePublicPartitions &&
    params.metrics.any { it.type == PRIVACY_ID_COUNT } &&
    params.executionMode.partitionSelectionIsNonDeterministic

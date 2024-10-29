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

package com.google.privacy.differentialprivacy.pipelinedp4j.core.budget

import com.google.privacy.differentialprivacy.pipelinedp4j.core.budget.BudgetAccountingStrategy.NAIVE
import java.lang.IllegalArgumentException
import java.lang.IllegalStateException

/**
 * An accountant who keeps track of the available amount of budget and allocates budget.
 *
 * Various mechanisms can request budget by calling. As a response, [BudgetAccountant] returns an
 * instance of [AllocatedBudget], which is not initialized initially. In order to initialize the
 * [AllocatedBudget]s, call [allocateBudgets]. The [BudgetAccountant] instance cannot be used once
 * budgets have been allocated.
 */
interface BudgetAccountant {
  /**
   * Records a [BudgetRequest] and returns the corresponding uninitialized [AllocatedBudget]. The
   * actual budget will be allocated when [allocateBudgets] is called.
   *
   * @param budgetRequest the request for a privacy budget.
   * @return an uninitialized [AllocatedBudget] that will be updated later.
   * @throws IllegalStateException if [allocateBudgets] has already been called on this instance.
   */
  fun requestBudget(budgetRequest: BudgetRequest): AllocatedBudget

  /**
   * Allocates budgets to all previously recorded [BudgetRequest]s. This method should only be
   * called once.
   *
   * @throws IllegalStateException if budgets have already been allocated.
   */
  fun allocateBudgets()
}

/**
 * A [BudgetAccountant] who uses naive budget composition to compose budgets (i.e., it sums-up
 * epsilon and delta values). It is initialized with the [TotalBudget] available.
 *
 * A [NaiveBudgetAccountant] accepts budget requests with [AbsoluteBudgetPerOpSpec] and
 * [RelativeBudgetPerOpSpec]. If there is enough budget to serve all requests,
 * [NaiveBudgetAccountant] allocates the exact epsilon and delta values requested by the
 * corresponding [AbsoluteBudgetPerOpSpec]s. Then it allocates the remaining budget according to the
 * [RelativeBudgetPerOpSpec]s. For example, if the total available epsilon is 10.0 and the following
 * requests have been sent:
 * - Request1 with [AbsoluteBudgetPerOpSpec] and epsilon = 4.0,
 * - Request2 with [RelativeBudgetPerOpSpec] and weight = 1.0,
 * - Request3 with [RelativeBudgetPerOpSpec] and weight = 2.0,
 *
 * Then [NaiveBudgetAccountant] will allocate:
 * - epsilon = 4.0 for Request1
 * - epsilon = <remaining_budget> * (<request_weight> / <total_weight>) = = (10.0 - 4.0) * (1.0 /
 *   3.0) = 2.0 for Request2
 * - epsilon = <remaining_budget> * (<request_weight> / <total_weight>) = = (10.0 - 4.0) * (2.0 /
 *   3.0) = 4.0 for Request3
 *
 * If there is not enough budget to serve all requests, [allocateBudgets] throws an exception and
 * none of the [AllocatedBudget]s created by this instance gets updated.
 *
 * @property totalBudget the total budget available for allocation.
 * @constructor Creates a new [NaiveBudgetAccountant] with the specified total budget.
 */
class NaiveBudgetAccountant(private val totalBudget: TotalBudget) : BudgetAccountant {
  private val absoluteBudgets: MutableList<RequestedAndAllocatedBudget> = mutableListOf()
  private val relativeBudgets: MutableList<RequestedAndAllocatedBudget> = mutableListOf()
  private var budgetsAllocated = false

  companion object {
    const val FLOATING_POINT_ARITHMETICS_TOLERANCE = 1e9
  }

  override fun requestBudget(budgetRequest: BudgetRequest): AllocatedBudget {
    if (budgetsAllocated) {
      throw IllegalStateException(
        "Budget cannot be requested because allocateBudgets() has already been called on this instance."
      )
    }
    val allocatedBudget = AllocatedBudget.create()
    when (budgetRequest.budgetSpec) {
      is AbsoluteBudgetPerOpSpec ->
        absoluteBudgets.add(RequestedAndAllocatedBudget(budgetRequest, allocatedBudget))
      is RelativeBudgetPerOpSpec ->
        relativeBudgets.add(RequestedAndAllocatedBudget(budgetRequest, allocatedBudget))
    }
    return allocatedBudget
  }

  override fun allocateBudgets() {
    if (budgetsAllocated) {
      throw IllegalStateException("Budgets have already been allocated.")
    }
    budgetsAllocated = true

    var totalRequestedEpsilon = 0.0
    var totalRequestedDelta = 0.0
    for (requestedAndAllocated in absoluteBudgets) {
      val budgetSpec = requestedAndAllocated.requested.budgetSpec as AbsoluteBudgetPerOpSpec
      totalRequestedEpsilon += budgetSpec.epsilon
      totalRequestedDelta += budgetSpec.delta
    }
    val remainingEpsilon = totalBudget.epsilon - totalRequestedEpsilon
    val remainingDelta = totalBudget.delta - totalRequestedDelta

    checkEnoughAbsoluteBudget(totalRequestedEpsilon, totalRequestedDelta)
    checkEnoughRelativeBudget(remainingEpsilon, remainingDelta)

    allocateAbsoluteBudgets()
    allocateRelativeBudgets(remainingEpsilon, remainingDelta)
  }

  private fun checkEnoughAbsoluteBudget(requestedEpsilon: Double, requestedDelta: Double) {
    if (
      notEnoughBudget(requestedEpsilon, totalBudget.epsilon) ||
        notEnoughBudget(requestedDelta, totalBudget.delta)
    ) {
      throw IllegalArgumentException(
        "Can't allocate absolute budget. The total requested budget is higher " +
          "than the available budget. " +
          "Total requested epsilon = $requestedEpsilon, " +
          "available epsilon = ${totalBudget.epsilon}, " +
          "total requested delta = $requestedDelta, " +
          "available delta = ${totalBudget.delta}."
      )
    }
  }

  fun notEnoughBudget(requested: Double, remaining: Double): Boolean {
    val diff = remaining - requested
    if (diff >= 0.0) {
      return false
    }
    return Math.abs(diff) > remaining / FLOATING_POINT_ARITHMETICS_TOLERANCE
  }

  private fun allocateAbsoluteBudgets() {
    for (requestedAndAllocated in absoluteBudgets) {
      val budgetSpec = requestedAndAllocated.requested.budgetSpec as AbsoluteBudgetPerOpSpec
      requestedAndAllocated.allocated.initialize(
        epsilon = budgetSpec.epsilon,
        delta = budgetSpec.delta,
      )
    }
  }

  private fun allocateRelativeBudgets(remainingEpsilon: Double, remainingDelta: Double) {
    var totalEpsilonWeight = 0.0
    var totalDeltaWeight = 0.0
    for (requestedAndAllocated in relativeBudgets) {
      val budgetSpec = requestedAndAllocated.requested.budgetSpec as RelativeBudgetPerOpSpec
      if (requestedAndAllocated.requested.mechanism.usesEpsilon) {
        totalEpsilonWeight += budgetSpec.weight
      }
      if (requestedAndAllocated.requested.mechanism.usesDelta) {
        totalDeltaWeight += budgetSpec.weight
      }
    }
    for (requestedAndAllocated in relativeBudgets) {
      val budgetSpec = requestedAndAllocated.requested.budgetSpec as RelativeBudgetPerOpSpec
      val allocatedEpsilon =
        if (requestedAndAllocated.requested.mechanism.usesEpsilon) {
          budgetSpec.weight / totalEpsilonWeight * remainingEpsilon
        } else {
          0.0
        }
      val allocatedDelta =
        if (requestedAndAllocated.requested.mechanism.usesDelta) {
          budgetSpec.weight / totalDeltaWeight * remainingDelta
        } else {
          0.0
        }
      requestedAndAllocated.allocated.initialize(allocatedEpsilon, allocatedDelta)
    }
  }

  private fun checkEnoughRelativeBudget(remainingEpsilon: Double, remainingDelta: Double) {
    if (relativeEpsilonRequested() && remainingEpsilon <= 0.0) {
      throw IllegalArgumentException(
        "Can't allocate relative budget. There is no epsilon available after allocation of the absolute budget."
      )
    }
    if (relativeDeltaRequested() && remainingDelta <= 0.0) {
      throw IllegalArgumentException(
        "Can't allocate relative budget. There is no delta available after allocation of the absolute budget."
      )
    }
  }

  private fun relativeEpsilonRequested(): Boolean =
    relativeBudgets.any { it.requested.mechanism.usesEpsilon }

  private fun relativeDeltaRequested(): Boolean =
    relativeBudgets.any { it.requested.mechanism.usesDelta }
}

/**
 * A request to [BudgetAccountant] to allocate the cost to the budget consumed by an operation.
 *
 * The requested consumption can be expressed in terms of relative weights or absolute values. The
 * weights are relative to the other [BudgetRequest]s sent to the same instance of
 * [BudgetAccountant]. See the documentation of a specific [BudgetAccountant] implementation in
 * order to learn how it composes relative and absolute budgets.
 *
 * @property budgetSpec the privacy budget specification for the operation.
 * @property mechanism the type of mechanism for which the budget is requested.
 */
data class BudgetRequest(val budgetSpec: BudgetPerOpSpec, val mechanism: AccountedMechanism)

/**
 * Represents the type of mechanism that consumes the privacy budget.
 *
 * @property usesEpsilon whether the mechanism consumes epsilon.
 * @property usesDelta whether the mechanism consumes delta.
 */
enum class AccountedMechanism(val usesEpsilon: Boolean, val usesDelta: Boolean) {
  GAUSSIAN_NOISE(true, true),
  LAPLACE_NOISE(true, false),
  PREAGGREGATED_PARTITION_SELECTION(true, true),
  POSTAGGREGATED_PARTITION_SELECTION(false, true),
}

/**
 * A pair of [BudgetRequest] and the corresponding [AllocatedBudget].
 *
 * When [BudgetAccountant] receives a [BudgetRequest], it doesn't allocate budgets immediately but
 * returns an [AllocatedBudget] instance, which is initialized later. We use this data structure to
 * remember which [AllocatedBudget] corresponds to which [BudgetRequest].
 */
internal data class RequestedAndAllocatedBudget(
  val requested: BudgetRequest,
  val allocated: AllocatedBudget,
)

enum class BudgetAccountingStrategy {
  NAIVE
}

/** A factory for creating [BudgetAccountant] instances based on a given strategy. */
object BudgetAccountantFactory {
  /**
   * Creates a [BudgetAccountant] instance for the specified [BudgetAccountingStrategy] and
   * [TotalBudget].
   *
   * @param accountingStrategy the budgeting strategy to use.
   * @param totalBudget the total budget available for allocation.
   * @return a new [BudgetAccountant] instance.
   */
  fun forStrategy(accountingStrategy: BudgetAccountingStrategy, totalBudget: TotalBudget) =
    when (accountingStrategy) {
      NAIVE -> NaiveBudgetAccountant(totalBudget)
    }
}

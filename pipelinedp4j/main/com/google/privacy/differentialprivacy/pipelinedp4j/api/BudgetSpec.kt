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

package com.google.privacy.differentialprivacy.pipelinedp4j.api

import com.google.errorprone.annotations.Immutable
import com.google.privacy.differentialprivacy.pipelinedp4j.core.budget.AbsoluteBudgetPerOpSpec as InternalAbsoluteBudgetPerOpSpec
import com.google.privacy.differentialprivacy.pipelinedp4j.core.budget.BudgetPerOpSpec as InternalBudgetPerOpSpec
import com.google.privacy.differentialprivacy.pipelinedp4j.core.budget.RelativeBudgetPerOpSpec as InternalRelativeBudgetPerOpSpec
import com.google.privacy.differentialprivacy.pipelinedp4j.core.budget.TotalBudget as InternalTotalBudget
import java.io.Serializable

/**
 * Represents the budget allocated for anonymizing a metric or group selection.
 *
 * This is a sealed interface with two implementations: [AbsoluteBudgetPerOpSpec] for absolute
 * budget values and [RelativeBudgetPerOpSpec] for relative weights.
 */
@Immutable sealed interface BudgetPerOpSpec

/**
 * Represents an absolute budget (epsilon and delta) for anonymizing a metric or group selection.
 *
 * @property epsilon the epsilon (ε) privacy budget value. It must be zero if your query selects
 *   private groups and also counts distinct privacy units. In all other cases it must be positive.
 * @property delta the delta (δ) privacy budget value. If this is a budget for an aggregation with
 *   Laplace noise, it can be not set (zero). If this is a budget for an aggregation with Gaussian
 *   noise or for a private group selection, it must be positive.
 */
@Immutable
data class AbsoluteBudgetPerOpSpec(val epsilon: Double, val delta: Double) :
  BudgetPerOpSpec, Serializable {
  init {
    BudgetValidationUtils.validateEpsilon(epsilon)
    BudgetValidationUtils.validateDelta(delta)
  }
}

/**
 * @usesMathJax
 *
 * Represents a relative weight for anonymizing a metric or group selection.
 *
 * The weight is relative to the weights of other metrics computed by the same query.
 *
 * The formula to calculate absolute budget for a relative budget is:
 *
 * $$ \text{absolute budget} = (\text{total budget} - \text{sum of absolute budgets}) *
 * \frac{\text{budget weight}}{\text{sum of relative budget weights}} $$
 *
 * For example, if a query has total budget of 2.0 and selects groups privately with absolute budget
 * of 0.5 and then computes two metrics, one with a relative weight of 1.0 and the other with a
 * relative weight of 3.0 then the first metrics will have an effective budget of (2 - 0.5) * 1.0 /
 * 4.0 = 0.375 and the second will have an effective budget of (2 - 0.5) * 3.0 / 4.0 = 1.125.
 *
 * @property weight the relative weight. Must be strictly positive.
 */
@Immutable
data class RelativeBudgetPerOpSpec(val weight: Double) : BudgetPerOpSpec, Serializable {
  init {
    BudgetValidationUtils.validateWeight(weight)
  }
}

/**
 * The total amount of budget allowed to be used in a single query for accounting both relative and
 * absolute operation costs.
 *
 * @property epsilon the total epsilon (ε) privacy budget value. Must be positive.
 * @property delta the total delta (δ) privacy budget value. Can be not set (zero) if Laplace noise
 *   is used and groups are publicly known. In all other cases it must be positive.
 */
@Immutable
data class TotalBudget(val epsilon: Double, val delta: Double = 0.0) : Serializable {
  init {
    BudgetValidationUtils.validateEpsilon(epsilon)
    BudgetValidationUtils.validateDelta(delta)
  }
}

/**
 * Converts the [BudgetPerOpSpec] to the [InternalBudgetPerOpSpec].
 *
 * We delibaretly do not expose the internal classes in the public API to limit the surface of the
 * API. This will give us more flexibility to change the implementation.
 */
internal fun BudgetPerOpSpec.toInternalBudgetPerOpSpec(): InternalBudgetPerOpSpec =
  when (this) {
    is AbsoluteBudgetPerOpSpec -> toInternalAbsoluteBudgetPerOpSpec(this)
    is RelativeBudgetPerOpSpec -> toInternalRelativeBudgetPerOpSpec(this)
  }

private fun toInternalAbsoluteBudgetPerOpSpec(spec: AbsoluteBudgetPerOpSpec) =
  InternalAbsoluteBudgetPerOpSpec(spec.epsilon, spec.delta)

private fun toInternalRelativeBudgetPerOpSpec(spec: RelativeBudgetPerOpSpec) =
  InternalRelativeBudgetPerOpSpec(spec.weight)

/**
 * Converts the [TotalBudget] to the [InternalTotalBudget].
 *
 * We delibaretly do not expose the internal classes in the public API to limit the surface of the
 * API. This will give us more flexibility to change the implementation.
 */
internal fun TotalBudget.toInternalTotalBudget() = InternalTotalBudget(epsilon, delta)

/** Utility object for validating budget parameters. */
private object BudgetValidationUtils {
  /**
   * Validates that epsilon is non-negative.
   *
   * @param epsilon the epsilon value to validate.
   * @throws IllegalArgumentException if epsilon is negative.
   */
  fun validateEpsilon(epsilon: Double) {
    require(epsilon >= 0.0) { "Epsilon must be >= 0.0. Provided epsilon: $epsilon." }
  }

  /**
   * Validates that delta is non-negative.
   *
   * @param delta the delta value to validate.
   * @throws IllegalArgumentException if delta is negative.
   */
  fun validateDelta(delta: Double) {
    require(delta >= 0.0) { "Delta must be >= 0.0. Provided delta: $delta." }
  }

  /**
   * Validates that a weight is strictly positive.
   *
   * @param weight the weight value to validate.
   * @throws IllegalArgumentException if weight is not strictly positive.
   */
  fun validateWeight(weight: Double) {
    require(weight > 0.0) { "Weight must be > 0. Provided weight: $weight." }
  }
}

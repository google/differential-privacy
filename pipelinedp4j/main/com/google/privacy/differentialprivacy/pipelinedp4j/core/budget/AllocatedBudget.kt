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

import java.io.Serializable

/**
 * Epsilon and delta values allocated for a DP mechanism.
 *
 * [AllocatedBudget] is created by a [BudgetAccountant] as a response to a budget consumption
 * request ([BudgetRequest]). The exact epsilon and delta values are not populated when
 * [AllocatedBudget] is created. Instead, they are populated by the [initialize] method called by
 * the [BudgetAccountant] when budgets are getting allocated. Once the epsilon and delta have been
 * populated, the mechanism who requested the budget can use them.
 *
 * @property epsilon the allocated epsilon value (ε). Initially -1.0.
 * @property delta the allocated delta value (δ). Initially -1.0.
 * @property initialized indicates whether the budget has been initialized with valid values.
 */
class AllocatedBudget : Serializable {
  private var epsilon = -1.0
  private var delta = -1.0
  private var initialized = false

  companion object Factory {
    fun create() = AllocatedBudget()
  }

  /**
   * Initializes the allocated budget with the given epsilon and delta values.
   *
   * This method should only be called once to set the values. Attempts to re-initialize will result
   * in an exception.
   *
   * @param epsilon the epsilon (ε) value for the allocated budget.
   * @param delta the delta (δ) value for the allocated budget.
   * @throws IllegalStateException if the budget has already been initialized.
   */
  fun initialize(epsilon: Double, delta: Double) {
    if (initialized) {
      throw IllegalStateException(
        "The budget has already been initialized with epsilon = $epsilon and delta = $delta. It can't be initialized second time."
      )
    }
    this.epsilon = epsilon
    this.delta = delta
    initialized = true
  }

  /**
   * Returns the allocated epsilon (ε) value.
   *
   * @throws IllegalStateException if the budget has not been initialized.
   */
  fun epsilon() =
    if (initialized) epsilon else throw IllegalStateException("The budget hasn't been initialized.")

  /**
   * Returns the allocated delta (δ) value.
   *
   * @throws IllegalStateException if the budget has not been initialized.
   */
  fun delta() =
    if (initialized) delta else throw IllegalStateException("The budget hasn't been initialized.")
}

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

import com.google.common.truth.Truth.assertThat
import com.google.privacy.differentialprivacy.pipelinedp4j.core.budget.AccountedMechanism.GAUSSIAN_NOISE
import com.google.privacy.differentialprivacy.pipelinedp4j.core.budget.AccountedMechanism.LAPLACE_NOISE
import com.google.privacy.differentialprivacy.pipelinedp4j.core.budget.AccountedMechanism.POSTAGGREGATED_PARTITION_SELECTION
import com.google.testing.junit.testparameterinjector.TestParameter
import com.google.testing.junit.testparameterinjector.TestParameterInjector
import kotlin.test.assertFailsWith
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(TestParameterInjector::class)
class NaiveBudgetAccountantTest {

  @Test
  fun accessAllocatedBudget_beforeCallingAllocateBudgets_throws() {
    val accountant = NaiveBudgetAccountant(TotalBudget(epsilon = 2.0, delta = 0.2))

    val allocatedBudget =
      accountant.requestBudget(
        BudgetRequest(AbsoluteBudgetPerOpSpec(epsilon = 1.0, delta = 0.1), GAUSSIAN_NOISE)
      )

    assertFailsWith<IllegalStateException> { allocatedBudget.epsilon() }
    assertFailsWith<IllegalStateException> { allocatedBudget.delta() }
  }

  @Test
  fun requestBudget_calledAfterAllocateBudget_throws() {
    val accountant = NaiveBudgetAccountant(TotalBudget(epsilon = 2.0, delta = 0.2))
    val budgetRequest =
      BudgetRequest(AbsoluteBudgetPerOpSpec(epsilon = 1.0, delta = 0.1), GAUSSIAN_NOISE)
    accountant.allocateBudgets()
    val e = assertFailsWith<IllegalStateException> { accountant.requestBudget(budgetRequest) }
    assertThat(e).hasMessageThat().contains("Budget cannot be requested")
  }

  @Test
  fun allocateBudgets_allocatesAbsoluteBudget() {
    val accountant = NaiveBudgetAccountant(TotalBudget(epsilon = 2.0, delta = 0.2))
    val allocatedBudget =
      accountant.requestBudget(
        BudgetRequest(AbsoluteBudgetPerOpSpec(epsilon = 1.0, delta = 0.1), GAUSSIAN_NOISE)
      )

    accountant.allocateBudgets()

    assertThat(allocatedBudget.epsilon()).isEqualTo(1.0)
    assertThat(allocatedBudget.delta()).isEqualTo(0.1)
  }

  @Test
  fun allocateBudgets_absoluteBudgetRequest_notEnoughEpsilon_throws() {
    val accountant = NaiveBudgetAccountant(TotalBudget(epsilon = 2.0, delta = 0.2))
    val unused1 =
      accountant.requestBudget(
        BudgetRequest(AbsoluteBudgetPerOpSpec(epsilon = 1.0, delta = 0.1), GAUSSIAN_NOISE)
      )
    val unused2 =
      accountant.requestBudget(
        BudgetRequest(AbsoluteBudgetPerOpSpec(epsilon = 1.1, delta = 0.1), GAUSSIAN_NOISE)
      )

    assertFailsWith<IllegalArgumentException>("Can't allocate absolute budget") {
      accountant.allocateBudgets()
    }
  }

  @Test
  fun allocateBudgets_absoluteBudgetRequest_notEnoughDelta_throws() {
    val accountant = NaiveBudgetAccountant(TotalBudget(epsilon = 2.0, delta = 0.2))
    val unused1 =
      accountant.requestBudget(
        BudgetRequest(AbsoluteBudgetPerOpSpec(epsilon = 1.0, delta = 0.1), GAUSSIAN_NOISE)
      )
    val unused2 =
      accountant.requestBudget(
        BudgetRequest(AbsoluteBudgetPerOpSpec(epsilon = 1.0, delta = 0.11), GAUSSIAN_NOISE)
      )

    assertFailsWith<IllegalArgumentException>("Can't allocate absolute budget") {
      accountant.allocateBudgets()
    }
  }

  @Test
  fun allocateBudgets_allocatesRelativeBudget() {
    val accountant = NaiveBudgetAccountant(TotalBudget(epsilon = 60.0, delta = 0.6))
    val allocatedBudgetWeightOne =
      accountant.requestBudget(BudgetRequest(RelativeBudgetPerOpSpec(1.0), GAUSSIAN_NOISE))
    val allocatedBudgetWeightTwo =
      accountant.requestBudget(BudgetRequest(RelativeBudgetPerOpSpec(2.0), GAUSSIAN_NOISE))
    val allocatedBudgetWeightThree =
      accountant.requestBudget(BudgetRequest(RelativeBudgetPerOpSpec(3.0), GAUSSIAN_NOISE))

    accountant.allocateBudgets()

    assertThat(allocatedBudgetWeightOne.epsilon()).isEqualTo(10.0)
    assertThat(allocatedBudgetWeightOne.delta()).isWithin(1e-13).of(0.1)
    assertThat(allocatedBudgetWeightTwo.epsilon()).isEqualTo(20.0)
    assertThat(allocatedBudgetWeightTwo.delta()).isWithin(1e-13).of(0.2)
    assertThat(allocatedBudgetWeightThree.epsilon()).isEqualTo(30.0)
    assertThat(allocatedBudgetWeightThree.delta()).isWithin(1e-13).of(0.3)
  }

  @Test
  fun allocateBudgets_allocatesRelativeDeltaOnlyIfMechanismNeedsIt(
    @TestParameter mechanism: AccountedMechanism
  ) {
    val accountant = NaiveBudgetAccountant(TotalBudget(epsilon = 1.0, delta = 0.1))
    val allocatedBudget =
      accountant.requestBudget(BudgetRequest(RelativeBudgetPerOpSpec(1.0), mechanism))

    accountant.allocateBudgets()

    if (mechanism.usesDelta) {
      assertThat(allocatedBudget.delta()).isEqualTo(0.1)
    } else {
      assertThat(allocatedBudget.delta()).isEqualTo(0.0)
    }
  }

  @Test
  fun allocateBudgets_relativeEpsilonRequest_budgetLessThenRelativeEpsilon() {
    val accountant = NaiveBudgetAccountant(TotalBudget(epsilon = 0.5, delta = 0.1))
    val allocatedBudget =
      accountant.requestBudget(BudgetRequest(RelativeBudgetPerOpSpec(1.0), GAUSSIAN_NOISE))
    accountant.allocateBudgets()

    assertThat(allocatedBudget.epsilon()).isEqualTo(0.5)
    assertThat(allocatedBudget.delta()).isEqualTo(0.1)
  }

  @Test
  fun allocateBudgets_relativeEpsilonRequest_noEpsilonLeft_throws() {
    val accountant = NaiveBudgetAccountant(TotalBudget(epsilon = 1.0, delta = 0.1))
    // Consume all epsilon with an absolute budget request
    val unused1 =
      accountant.requestBudget(
        BudgetRequest(AbsoluteBudgetPerOpSpec(epsilon = 1.0, delta = 0.05), GAUSSIAN_NOISE)
      )
    // All relative budget requests request epsilon to be consumed.
    val unused2 =
      accountant.requestBudget(BudgetRequest(RelativeBudgetPerOpSpec(1.0), GAUSSIAN_NOISE))

    assertFailsWith<IllegalArgumentException>("Can't allocate relative budget") {
      accountant.allocateBudgets()
    }
  }

  @Test
  fun allocateBudgets_relativeDeltaRequest_noDeltaAllocated_throws() {
    val accountant = NaiveBudgetAccountant(TotalBudget(epsilon = 1.0, delta = 0.0))
    // Request budget for Gaussian mechanism, it requires delta.
    val unused1 =
      accountant.requestBudget(BudgetRequest(RelativeBudgetPerOpSpec(1.0), GAUSSIAN_NOISE))

    assertFailsWith<IllegalArgumentException>("Can't allocate relative budget") {
      accountant.allocateBudgets()
    }
  }

  @Test
  fun allocateBudgets_relativeDeltaRequest_noDeltaLeft_throws() {
    val accountant = NaiveBudgetAccountant(TotalBudget(epsilon = 1.0, delta = 0.1))
    // Consume all delta with an absolute budget request
    val unused1 =
      accountant.requestBudget(
        BudgetRequest(AbsoluteBudgetPerOpSpec(epsilon = 0.5, delta = 0.1), GAUSSIAN_NOISE)
      )
    // Request budget for Gaussian mechanism, it requires delta.
    val unused2 =
      accountant.requestBudget(BudgetRequest(RelativeBudgetPerOpSpec(1.0), GAUSSIAN_NOISE))

    assertFailsWith<IllegalArgumentException>("Can't allocate relative budget") {
      accountant.allocateBudgets()
    }
  }

  @Test
  fun allocateBudgets_composesAbsoluteAndRelativeBudgets() {
    val accountant = NaiveBudgetAccountant(TotalBudget(epsilon = 60.0, delta = 0.6))
    val absoluteAllocatedBudget =
      accountant.requestBudget(
        BudgetRequest(AbsoluteBudgetPerOpSpec(epsilon = 30.0, delta = 0.3), GAUSSIAN_NOISE)
      )
    val relativeAllocatedBudget =
      accountant.requestBudget(BudgetRequest(RelativeBudgetPerOpSpec(1.0), GAUSSIAN_NOISE))
    val relativeAllocatedBudgetTwiceMore =
      accountant.requestBudget(BudgetRequest(RelativeBudgetPerOpSpec(2.0), GAUSSIAN_NOISE))

    accountant.allocateBudgets()

    assertThat(absoluteAllocatedBudget.epsilon()).isEqualTo(30.0)
    assertThat(absoluteAllocatedBudget.delta()).isEqualTo(0.3)
    assertThat(relativeAllocatedBudget.epsilon()).isEqualTo(10.0)
    assertThat(relativeAllocatedBudget.delta()).isWithin(1e-13).of(0.1)
    assertThat(relativeAllocatedBudgetTwiceMore.epsilon()).isEqualTo(20.0)
    assertThat(relativeAllocatedBudgetTwiceMore.delta()).isWithin(1e-13).of(0.2)
  }

  @Test
  fun allocateBudgets_accountsForFloatingPointTolerance() {
    val accountant = NaiveBudgetAccountant(TotalBudget(epsilon = 1.0, delta = 0.0))
    val smallerThanFloatingPointError = 1.0 / 1e10
    val allocatedBudget =
      accountant.requestBudget(
        BudgetRequest(
          AbsoluteBudgetPerOpSpec(epsilon = 1.0 + smallerThanFloatingPointError, delta = 0.0),
          LAPLACE_NOISE,
        )
      )

    accountant.allocateBudgets()

    assertThat(allocatedBudget.epsilon()).isEqualTo(1.0 + smallerThanFloatingPointError)
    assertThat(allocatedBudget.delta()).isEqualTo(0.0)
  }

  @Test
  fun allocateBudgets_postAggregationThresholdingAllocatedCorrectly() {
    val accountant = NaiveBudgetAccountant(TotalBudget(epsilon = 10.0, delta = 0.4))
    val absoluteAllocatedBudget =
      accountant.requestBudget(
        BudgetRequest(AbsoluteBudgetPerOpSpec(epsilon = 3.0, delta = 0.1), GAUSSIAN_NOISE)
      )
    val relativeAllocatedBudget =
      accountant.requestBudget(BudgetRequest(RelativeBudgetPerOpSpec(1.0), GAUSSIAN_NOISE))
    val relativeAllocatedBudgetTwiceMore =
      accountant.requestBudget(
        BudgetRequest(RelativeBudgetPerOpSpec(2.0), POSTAGGREGATED_PARTITION_SELECTION)
      )

    accountant.allocateBudgets()

    assertThat(absoluteAllocatedBudget.epsilon()).isEqualTo(3.0)
    assertThat(absoluteAllocatedBudget.delta()).isEqualTo(0.1)
    assertThat(relativeAllocatedBudget.epsilon()).isEqualTo(7.0)
    assertThat(relativeAllocatedBudget.delta()).isWithin(1e-13).of(0.1)
    assertThat(relativeAllocatedBudgetTwiceMore.epsilon()).isEqualTo(0.0)
    assertThat(relativeAllocatedBudgetTwiceMore.delta()).isWithin(1e-13).of(0.2)
  }
}

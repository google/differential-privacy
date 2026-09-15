package com.google.privacy.differentialprivacy.pipelinedp4j.core.budget

/**
 * Sealed class representing the details of a budget allocation for different differential privacy
 * mechanisms.
 *
 * Each subclass contains only the parameters relevant to that mechanism.
 *
 * This class is used to pass budget allocation details from [DpEngine] to API implementations
 * (e.g., BeamApi), but it is not intended for direct use by end-users of the library.
 *
 * Extend this class if you need to propagate more details about the budget allocation from DPEngine
 * to the backend-specific API implementations in the API package (e.g. BeamApi, etc.).
 */
sealed class BudgetAllocationDetails {
  /**
   * Budget allocation details for Gaussian mechanism used for aggregation.
   *
   * Uses both epsilon and delta.
   */
  data class GaussianAggregationAllocation(val epsilon: Double, val delta: Double) :
    BudgetAllocationDetails()

  /**
   * Budget allocation details for Laplace mechanism used for aggregation.
   *
   * Uses only epsilon.
   */
  data class LaplaceAggregationAllocation(val epsilon: Double) : BudgetAllocationDetails()

  /**
   * Budget allocation details for pre-aggregated partition selection.
   *
   * Uses both epsilon and delta.
   */
  data class PreaggregatedPartitionSelectionAllocation(val epsilon: Double, val delta: Double) :
    BudgetAllocationDetails()

  /**
   * Budget allocation details for post-aggregated partition selection.
   *
   * Only uses delta.
   */
  data class PostaggregatedPartitionSelectionAllocation(val delta: Double) :
    BudgetAllocationDetails()
}

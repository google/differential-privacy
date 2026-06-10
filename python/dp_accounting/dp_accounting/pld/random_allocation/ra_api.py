"""Public API surface for random-allocation PLD construction."""

from __future__ import annotations

from functools import partial

from dp_accounting.pld import privacy_loss_distribution

from dp_accounting.pld.random_allocation import ra_core
from dp_accounting.pld.random_allocation import ra_distributions
from dp_accounting.pld.random_allocation import ra_types
from dp_accounting.pld.random_allocation import ra_utils


def gaussian_allocation_pld(
    params: ra_types.PrivacyParams,
    config: ra_types.AllocationSchemeConfig,
    bound_type: ra_types.BoundType = ra_types.BoundType.DOMINATES,
) -> privacy_loss_distribution.PrivacyLossDistribution:
  """Compute upper/lower PLD for random-allocation with the Gaussian mechanism.

  Args:
      params: Privacy parameters describing noise scale, number of steps,
          and optional delta/epsilon query target.
      config: Discretization and convolution configuration.
      bound_type: Whether to compute a dominating or dominated discretized
          bound.

  Returns:
      A ``dp_accounting`` ``PrivacyLossDistribution`` for both privacy
      directions.

  """
  # Input validation
  ra_utils._validate_privacy_params(params)
  ra_utils._validate_allocation_scheme_config(config)
  ra_utils._validate_bound_type(bound_type)

  compute_base_pld_remove = partial(
      ra_core._gaussian_allocation_pld_core,
      direction=ra_types.Direction.REMOVE,
      sigma=params.sigma,
  )
  compute_base_pld_add = partial(
      ra_core._gaussian_allocation_pld_core,
      direction=ra_types.Direction.ADD,
      sigma=params.sigma,
  )
  return ra_core._allocation_full_pld(
      compute_base_pld_remove=compute_base_pld_remove,
      compute_base_pld_add=compute_base_pld_add,
      num_steps=params.num_steps,
      num_selected=params.num_selected,
      num_epochs=params.num_epochs,
      loss_discretization=config.value_discretization_interval,
      tail_truncation=config.tail_truncation,
      bound_type=bound_type,
  )


def general_allocation_pld(
    num_steps: int,
    num_selected: int,
    num_epochs: int,
    remove_realization: ra_distributions.PLDRealization,
    add_realization: ra_distributions.PLDRealization,
    config: ra_types.AllocationSchemeConfig,
    bound_type: ra_types.BoundType = ra_types.BoundType.DOMINATES,
) -> privacy_loss_distribution.PrivacyLossDistribution:
  """Build a random-allocation PLD from explicit PLD realizations.

  Args:
      num_steps: Total number of random-allocation steps.
      num_selected: Number of selections per epoch.
      num_epochs: Number of epochs.
      remove_realization: Explicit remove-direction PLD realization.
      add_realization: Explicit add-direction PLD realization.
      config: Discretization and convolution configuration.
      bound_type: Whether to compute a dominating or dominated discretized
          bound.

  Returns:
      A ``dp_accounting`` ``PrivacyLossDistribution`` for the composed
      realization.

  Notes:
      The delivery package always uses the geometric convolution path.

  """
  # Input validation
  ra_utils._validate_allocation_params(num_steps, num_selected, num_epochs)
  if not isinstance(remove_realization, ra_distributions.PLDRealization):
    raise TypeError(
        "remove_realization must be PLDRealization, got "
        f"{type(remove_realization)}"
    )
  if not isinstance(add_realization, ra_distributions.PLDRealization):
    raise TypeError(
        f"add_realization must be PLDRealization, got {type(add_realization)}"
    )
  ra_utils._validate_allocation_scheme_config(config)
  ra_utils._validate_bound_type(bound_type)

  compute_base_pld_remove = partial(
      ra_core._geometric_allocation_pld_base_remove,
      base_distributions_creation=partial(
          ra_core._realization_remove_base_distributions,
          realization=remove_realization,
      ),
  )
  compute_base_pld_add = partial(
      ra_core._geometric_allocation_pld_base_add,
      base_distributions_creation=partial(
          ra_core._realization_add_base_distribution,
          realization=add_realization,
      ),
  )
  return ra_core._allocation_full_pld(
      compute_base_pld_remove=compute_base_pld_remove,
      compute_base_pld_add=compute_base_pld_add,
      num_steps=num_steps,
      num_selected=num_selected,
      num_epochs=num_epochs,
      loss_discretization=config.value_discretization_interval,
      tail_truncation=config.tail_truncation,
      bound_type=bound_type,
  )
